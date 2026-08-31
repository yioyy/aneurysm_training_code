"""帶 AMAP-style case/patch prompt 的 trainer。

既有的 nnUNetTrainer 完全不動 —— 本檔只是子類別，由 recipe 的
`engine.nnunet_trainer_class: nnUNetTrainerPrompt` 選用。

**為什麼不是直接改 train_step 裡的 `self.network(data)`**
父類別的 `train_step` / `validation_step` 各約 60 行（含梯度累積、AMP、多頭分支），
整段複製過來只為了改一行呼叫，日後上游一動就會靜默分歧。改成：
  * 覆寫 train_step / validation_step —— 明確地把當前 batch 的兩個向量交給 wrapper
  * wrapper 的 forward(x) 再轉呼叫 net(x, case_vec, patch_vec)
向量在**同一個 thread 內、同一次呼叫中**設定並立即使用（dataloader worker 不碰網路），
沒有跨 thread 競爭。這是「改 trainer」與「不複製長方法」之間的折衷。
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class PromptNetworkWrapper(nn.Module):
    """把 net(x, case_vec, patch_vec) 包成 net(x)，向量由 trainer 逐 step 指定。"""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self._case_vec = None
        self._patch_vec = None

    def set_prompt(self, case_vec, patch_vec):
        self._case_vec, self._patch_vec = case_vec, patch_vec

    def forward(self, x):
        return self.net(x, self._case_vec, self._patch_vec)

    # ⚠ 讓 wrapper 對 state_dict 完全透明：不加 `net.` 前綴。
    #   load_pretrained_weights 是逐鍵比對 model.state_dict()，若鍵名變成
    #   `net.encoder.stem...`，checkpoint 的 `encoder.stem...` 會全部對不上
    #   → 實測 "Loaded 0 / 492"，四個實驗都變成隨機初始化，_p16_ft 形同虛設。
    def state_dict(self, *a, **kw):
        return self.net.state_dict(*a, **kw)

    def load_state_dict(self, state_dict, strict: bool = True, **kw):
        return self.net.load_state_dict(state_dict, strict=strict, **kw)

    def named_parameters(self, prefix: str = "", recurse: bool = True, **kw):
        # optimizer 仍要看到 prompt 模組的參數 → 用 self.net 之外還要含 wrapper 自身
        return super().named_parameters(prefix=prefix, recurse=recurse, **kw)

    def __getattr__(self, name):
        # decoder / deep_supervision 等屬性要能穿透到內層（nnUNetTrainer 會直接存取）
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["net"], name)


class nnUNetTrainerPrompt(nnUNetTrainer):
    # --- 由 recipe 的 model.extra 覆寫（見 mutp nnunet_backend）---
    CASE_FEATURE_CSVS: list = []
    USE_PATCH_FEATURES: bool = True
    V4_CHANNEL = 1                  # vessel4 在 preprocessed data 的 channel index
    NETWORK_IN_CHANNELS = None      # 不為 None 時只把前 N 個 channel 餵進網路（s1 用 1）
    NEIGHBORHOOD_HALF = 16

    # ⚠️ 必須照抄父類別的完整簽名，不能用 *args/**kwargs ——
    #    父類別 __init__ 內有 `for k in inspect.signature(self.__init__).parameters: locals()[k]`，
    #    子類別若用 *a/**kw，它會去 locals() 找不存在的 'a' → KeyError: 'a'。
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'),
                 initial_lr: float = 1e-4,
                 oversample_foreground_percent: float = 0.5,
                 oversample_foreground_percent_val: float = 0.2,
                 num_iterations_per_epoch: int = 500,
                 num_epochs: int = 1000,
                 optimizer_type: str = 'AdamW',
                 lr_scheduler_type: str = 'CosineAnnealingLR',
                 enable_early_stopping: bool = False,
                 early_stopping_patience: int = 50,
                 early_stopping_min_delta: float = 0.0001):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device,
                         initial_lr, oversample_foreground_percent,
                         oversample_foreground_percent_val, num_iterations_per_epoch,
                         num_epochs, optimizer_type, lr_scheduler_type,
                         enable_early_stopping, early_stopping_patience,
                         early_stopping_min_delta)
        self._case_table = None
        self._load_env_config()

    def _load_env_config(self):
        """從環境變數讀 prompt 設定（由 mutp nnunet_backend 依 engine.prompt 寫入）。

        走 env 而非建構參數：nnU-Net 只吃 trainer 類別名稱，且 DDP 會另起 process。
        """
        import os
        csvs = os.environ.get("MUTP_PROMPT_CASE_CSVS", "")
        if csvs:
            self.CASE_FEATURE_CSVS = [c for c in csvs.split(",") if c]
        self.USE_PATCH_FEATURES = os.environ.get("MUTP_PROMPT_USE_PATCH", "1") == "1"
        self.V4_CHANNEL = int(os.environ.get("MUTP_PROMPT_V4_CHANNEL", "1"))
        self.NEIGHBORHOOD_HALF = int(os.environ.get("MUTP_PROMPT_NEIGHBORHOOD", "16"))
        ic = os.environ.get("MUTP_PROMPT_IN_CHANNELS")
        self.NETWORK_IN_CHANNELS = int(ic) if ic else None
        self.print_to_log_file(
            f"[prompt] CSV {len(self.CASE_FEATURE_CSVS)} 份  use_patch={self.USE_PATCH_FEATURES}"
            f"  v4_ch={self.V4_CHANNEL}  in_ch={self.NETWORK_IN_CHANNELS}")

    # ---------- 特徵表 ----------
    def _ensure_case_table(self):
        if self._case_table is not None or not self.CASE_FEATURE_CSVS:
            return self._case_table
        from mutp.data.prompt_features import CaseFeatureTable
        paths = [pathlib.Path(p) for p in self.CASE_FEATURE_CSVS]
        self._case_table = CaseFeatureTable(*paths)
        self.print_to_log_file(f"[prompt] case 特徵表載入 {len(self._case_table)} 個 case")
        return self._case_table

    # ---------- dataloader ----------
    def get_dataloaders(self):
        from nnunetv2.training.dataloading import data_loader_3d as _m3d
        from nnunetv2.training.dataloading.data_loader_3d_prompt import nnUNetDataLoader3DPrompt

        tab = self._ensure_case_table()
        nnUNetDataLoader3DPrompt.case_feature_table = tab
        nnUNetDataLoader3DPrompt.use_patch_features = self.USE_PATCH_FEATURES
        nnUNetDataLoader3DPrompt.v4_channel = self.V4_CHANNEL
        nnUNetDataLoader3DPrompt.neighborhood_half = self.NEIGHBORHOOD_HALF
        orig = _m3d.nnUNetDataLoader3D
        _m3d.nnUNetDataLoader3D = nnUNetDataLoader3DPrompt
        try:
            return super().get_dataloaders()
        finally:
            _m3d.nnUNetDataLoader3D = orig

    # ---------- 網路 ----------
    def initialize(self):
        super().initialize()
        if not isinstance(self.network, PromptNetworkWrapper):
            self.network = PromptNetworkWrapper(self.network).to(self.device)
            self.print_to_log_file("[prompt] network 已包上 PromptNetworkWrapper")

    # ---------- 每步注入 ----------
    def _apply_prompt(self, batch: dict):
        cv = batch.get("case_feat")
        pv = batch.get("patch_feat")
        to_t = lambda v: (None if v is None else
                          (v if isinstance(v, torch.Tensor) else torch.from_numpy(np.asarray(v)))
                          .to(self.device, non_blocking=True).float())
        self.network.set_prompt(to_t(cv), to_t(pv) if self.USE_PATCH_FEATURES else None)
        if self.NETWORK_IN_CHANNELS is not None:
            d = batch["data"]
            if d.shape[1] > self.NETWORK_IN_CHANNELS:
                batch = dict(batch, data=d[:, :self.NETWORK_IN_CHANNELS])
        return batch

    def train_step(self, batch: dict) -> dict:
        return super().train_step(self._apply_prompt(batch))

    def validation_step(self, batch: dict) -> dict:
        return super().validation_step(self._apply_prompt(batch))

    # ---------- 監控 prompt 有沒有被用到 ----------
    def _install_modulation_probes(self):
        """在 FiLM / cross-attention 上掛 hook，量測**實際調制強度**。

        `|W_film|` 只是權重範數 —— 權重大但 prompt 向量小時，實際 γ/β 仍可能≈0。
        真正該看的是 ‖輸出 − 輸入‖ / ‖輸入‖：這個模組把 feature 改動了多少比例。
        0 = 完全沒作用（等同 identity），0.1 = 改動 10%。
        """
        if getattr(self, "_probes_done", False):
            return
        net = getattr(self.network, "net", self.network)
        self._mod_stats = {}

        def mk(tag):
            def fn(mod, inp, out):
                if not torch.is_tensor(out) or not inp or not torch.is_tensor(inp[0]):
                    return
                with torch.no_grad():
                    d = (out - inp[0]).norm().item()
                    b = inp[0].norm().item()
                    if b > 0:
                        a = self._mod_stats.setdefault(tag, [0.0, 0])
                        a[0] += d / b
                        a[1] += 1
            return fn

        n = 0
        if getattr(net, "bottleneck_attn", None) is not None:
            net.bottleneck_attn.register_forward_hook(mk("attn"))
            n += 1
        for i, f in enumerate(getattr(net, "dec_film", []) or []):
            f.register_forward_hook(mk(f"film{i}"))
            n += 1
        self._probes_done = True
        self.print_to_log_file(f"[prompt] 已掛上 {n} 個調制強度探針")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._install_modulation_probes()
        self._mod_stats = {}

    def on_epoch_end(self):
        super().on_epoch_end()
        net = getattr(self.network, "net", self.network)
        try:
            eta = float(net.bottleneck_attn.eta.detach())
            gam = float(torch.stack([f.to_film.weight.detach().norm()
                                     for f in net.dec_film]).mean())
            st = getattr(self, "_mod_stats", {})
            mod = {k: (v[0] / max(v[1], 1)) for k, v in st.items()}
            attn_r = mod.get("attn", 0.0)
            film_r = (sum(v for k, v in mod.items() if k.startswith("film"))
                      / max(sum(1 for k in mod if k.startswith("film")), 1))
            self.print_to_log_file(
                f"[prompt] eta={eta:+.5f}  mean|W_film|={gam:.5f}  "
                f"調制強度 attn={attn_r:.5f} film={film_r:.5f}"
                f"  (‖Δ‖/‖feat‖；0=無作用)")
        except AttributeError:
            pass
