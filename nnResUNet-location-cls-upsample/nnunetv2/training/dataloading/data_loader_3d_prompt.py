"""帶 AMAP-style prompt 特徵的 3D dataloader。

在既有 batch dict 上多回傳兩個陣列：
    case_feat  [B, 11]  —— 全腦屬性，CSV 查表（同一 case 的多個 crop 共用）
    patch_feat [B, 14]  —— 隨 patch 位置變動，取樣時動態算

實作要點：不複製 `generate_train_batch`（那有 90 行、且會隨上游演進），改成
  1. 暫時包住 `self._data.load_case`，把每個 case 裁切**前**的完整 volume 記下來
  2. 暫時包住 `self.get_bbox`，把每個樣本的 bbox 記下來
  3. 呼叫父類別的原方法，再用記下來的東西補算特徵
這樣沒有重複 I/O、也不動上游程式碼。

⚠️ patch_coord 用**前處理框**自身定義（`(center − vol_center)/vol_half`），
不是 README 的原始 crop bbox 框 —— 兩者不同座標系，混用會讓訓練與推論對不上。
訓練與推論用同一套即可，詳見 docs/plan_feature_conditioning_v4.md §6。
"""
from __future__ import annotations

import numpy as np

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D


class nnUNetDataLoader3DPrompt(nnUNetDataLoader3D):
    #: 由 trainer 設定；None 時該特徵回傳零向量（等同 prompt dropout）
    case_feature_table = None
    #: vessel4 在 preprocessed data 裡的 channel index；None 表示不算 patch 特徵
    v4_channel = 1
    #: 鄰域半徑（README 用 16）
    neighborhood_half = 16
    use_patch_features = True

    def generate_train_batch(self):
        from mutp.data.prompt_features import N_CASE, N_PATCH, patch_features

        cases, bboxes = {}, []
        orig_load, orig_bbox = self._data.load_case, self.get_bbox

        def load_case(key):
            out = orig_load(key)
            cases[key] = out[0]          # data（含所有 channel，裁切前）
            return out

        def get_bbox(*a, **kw):
            lbs, ubs = orig_bbox(*a, **kw)
            bboxes.append((list(lbs), list(ubs)))
            return lbs, ubs

        self._data.load_case, self.get_bbox = load_case, get_bbox
        try:
            batch = super().generate_train_batch()
        finally:
            self._data.load_case, self.get_bbox = orig_load, orig_bbox

        keys = list(batch["keys"])
        b = len(keys)
        cf = np.zeros((b, N_CASE), dtype=np.float32)
        pf = np.zeros((b, N_PATCH), dtype=np.float32)
        for j, k in enumerate(keys):
            if self.case_feature_table is not None:
                cf[j] = self.case_feature_table.get(k)
            if not (self.use_patch_features and self.v4_channel is not None):
                continue
            data = cases.get(k)
            if data is None or data.shape[0] <= self.v4_channel or j >= len(bboxes):
                continue
            lbs, ubs = bboxes[j]
            center = [(lo + hi) // 2 for lo, hi in zip(lbs, ubs)]
            pf[j] = patch_features(np.rint(data[self.v4_channel]).astype(np.int16),
                                   center, vol_shape=data.shape[1:],
                                   half=self.neighborhood_half)
        batch["case_feat"] = cf
        batch["patch_feat"] = pf
        return batch
