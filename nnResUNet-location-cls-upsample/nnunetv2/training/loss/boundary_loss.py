"""Boundary Loss — Kervadec et al. 2019 (MIDL).

L_boundary = mean over fg voxels of (softmax_pred * SDF_of_GT)
  SDF: 邊內負、邊外正 →
    - 預測在 GT 外 (positive SDF) → loss 升 → 罰擴張
    - 預測在 GT 內 (negative SDF) → loss 降 → 鼓勵收緊邊界

對小病灶尤其有效：直接優化邊界精度，不被 Dice 的 union size 稀釋。

兩種 SDF 引擎：
  - torch (預設)：iterative max-pool 算近似 L∞ Chebyshev 距離，**全 GPU**，
    每 batch 100ms 量級；精度足以做 boundary regularization。
    （借 Shit et al. 2021 SoftSkeletonize 的 morphology 技巧）
  - scipy：Felzenszwalb-Huttenlocher Euclidean SDF，精度最高，CPU 跑，
    每 batch 1-3s；要 anisotropic spacing 或最高精度時用。

學術上仍稱 Boundary Loss；method section 註明 "GPU-efficient SDF
approximation via iterative morphological max-pooling" 即可。
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Engine 1: torch max-pool (GPU)
# ---------------------------------------------------------------------------
def _approx_sdf_torch(gt_c: torch.Tensor, max_dist: int = 10) -> torch.Tensor:
    """純 GPU SDF 近似 — iterative max-pool morphological dilation。

    Args:
        gt_c: (B, 1, [D,] H, W) binary float (0/1)
        max_dist: 距離截斷上限 (voxel)，預設 10

    Returns:
        sdf: (B, 1, [D,] H, W) float — 邊內負、邊外正、邊上 0
             單位為 voxel (Chebyshev L∞ 距離，不是 Euclidean)
             |sdf| > max_dist 一律截到 ±max_dist
    """
    is_3d = gt_c.ndim == 5
    pool_fn = F.max_pool3d if is_3d else F.max_pool2d

    sdf = torch.zeros_like(gt_c, dtype=torch.float32)

    # ─── 外擴：從 GT 開始 dilate，新接觸到的 outside voxel → +d ───
    cur = gt_c.float()
    for d in range(1, max_dist + 1):
        dil = pool_fn(cur, kernel_size=3, stride=1, padding=1)
        new = (dil > 0.5) & (cur < 0.5)
        sdf = torch.where(new, torch.full_like(sdf, float(d)), sdf)
        cur = dil

    # ─── 內縮：從 ~GT 開始 dilate，新接觸到的 inside voxel → -d ───
    cur = (1.0 - gt_c).float()
    for d in range(1, max_dist + 1):
        dil = pool_fn(cur, kernel_size=3, stride=1, padding=1)
        new = (dil > 0.5) & (cur < 0.5)
        sdf = torch.where(new, torch.full_like(sdf, -float(d)), sdf)
        cur = dil

    return sdf


# ---------------------------------------------------------------------------
# Engine 2: scipy Euclidean SDF (CPU fallback)
# ---------------------------------------------------------------------------
def _exact_sdf_scipy(gt_mask: np.ndarray, spacing=None) -> np.ndarray:
    """scipy Felzenszwalb-Huttenlocher Euclidean SDF — CPU but exact."""
    from scipy.ndimage import distance_transform_edt
    if not gt_mask.any() or gt_mask.all():
        return np.zeros_like(gt_mask, dtype=np.float32)
    pos = distance_transform_edt(~gt_mask, sampling=spacing)
    neg = distance_transform_edt(gt_mask, sampling=spacing)
    return (pos - neg).astype(np.float32)


# ---------------------------------------------------------------------------
# BoundaryLoss module
# ---------------------------------------------------------------------------
class BoundaryLoss(nn.Module):
    """Boundary Loss with switchable SDF engine.

    Args:
        idc:       哪些 class index 算 boundary，預設 [1] (foreground)
        engine:    "torch" (預設, GPU, max-pool Chebyshev) | "scipy" (CPU, Euclidean)
        max_dist:  torch engine 距離截斷 (voxel)，預設 10；scipy 用不到
        normalize: True = 用 patch 內最大 |SDF| 歸一化，loss 大致 [-1, 1] 量級
        spacing:   scipy engine 用的物理距離 (sx, sy, sz)；torch 引擎忽略
    """
    def __init__(self,
                 idc: Optional[List[int]] = None,
                 engine: str = "torch",
                 max_dist: int = 10,
                 normalize: bool = True,
                 spacing: Optional[tuple] = None):
        super().__init__()
        assert engine in ("torch", "scipy"), f"engine must be 'torch' or 'scipy', got {engine!r}"
        self.idc = idc if idc is not None else [1]
        self.engine = engine
        self.max_dist = int(max_dist)
        self.normalize = bool(normalize)
        self.spacing = spacing

    def _make_sdf_batch(self, gt_one: torch.Tensor) -> torch.Tensor:
        """gt_one: (B, C, [D,] H, W) one-hot float → 同 shape SDF tensor"""
        out = torch.zeros_like(gt_one, dtype=torch.float32)
        for c in self.idc:
            if c >= gt_one.shape[1]:
                continue
            gt_c = gt_one[:, c:c + 1]  # (B, 1, ...)

            if self.engine == "torch":
                sdf_c = _approx_sdf_torch(gt_c, max_dist=self.max_dist)
            else:  # scipy
                gt_np = gt_c.detach().cpu().numpy().astype(bool)
                sdf_np = np.zeros_like(gt_np, dtype=np.float32)
                for b in range(gt_np.shape[0]):
                    sdf_np[b, 0] = _exact_sdf_scipy(gt_np[b, 0], spacing=self.spacing)
                sdf_c = torch.from_numpy(sdf_np).to(gt_one.device)

            if self.normalize:
                # 對每個 sample 個別歸一化
                dims = tuple(range(1, sdf_c.ndim))
                max_abs = sdf_c.abs().amax(dim=dims, keepdim=True).clamp(min=1e-6)
                sdf_c = sdf_c / max_abs

            out[:, c:c + 1] = sdf_c
        return out

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        net_output: (B, C, [D,] H, W) raw logits
        target:     (B, 1, [D,] H, W) integer GT labels  OR  (B, C, [D,] H, W) one-hot
        """
        # one-hot target
        if target.shape[1] == 1:
            gt_idx = target.long().squeeze(1)
            gt_one = torch.zeros_like(net_output, dtype=torch.float32)
            gt_one.scatter_(1, gt_idx.unsqueeze(1), 1)
        else:
            gt_one = target.float()

        pc = torch.softmax(net_output, dim=1)
        sdf = self._make_sdf_batch(gt_one).to(net_output.dtype)

        loss_val = pc.new_tensor(0.0)
        n = 0
        for c in self.idc:
            if c >= net_output.shape[1]:
                continue
            loss_val = loss_val + (pc[:, c] * sdf[:, c]).mean()
            n += 1
        if n > 0:
            loss_val = loss_val / n
        # 記錄 inner component（給 Compound_loss 遞迴展開用），key 短名 'Boundary'
        try:
            self.last_components = {'Boundary': float(loss_val.detach())}
        except Exception:
            self.last_components = {'Boundary': float('nan')}
        return loss_val

    def extra_repr(self) -> str:
        return f"idc={self.idc}, engine={self.engine!r}, max_dist={self.max_dist}, normalize={self.normalize}"
