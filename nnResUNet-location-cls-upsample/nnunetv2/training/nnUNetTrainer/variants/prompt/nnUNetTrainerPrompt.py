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

import os
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
        self._register_case_id_aliases()
        return self._case_table

    def _register_case_id_aliases(self):
        """把 mutp 改名後的 nnunet_id 接回原始 case_id。

        mutp 前處理會改名（`DeepAneurysm_v4_combineFEMH_010000`），特徵 CSV
        則以原始 case_id 為鍵。少了這一步，dataloader 查表 0/3057 全部落空，
        `get()` 回零向量而不報錯 —— 模型看起來在訓練，實際上條件化完全沒作用。
        """
        import csv as _csv
        raw = os.environ.get("nnUNet_raw")
        if not raw:
            return
        ds = getattr(self.plans_manager, "dataset_name", None) or ""
        mp = pathlib.Path(raw) / ds / "case_id_mapping.csv"
        if not mp.is_file():
            self.print_to_log_file(f"[prompt] 無 case_id_mapping.csv（{mp}），使用原始鍵")
            return
        with open(mp, newline="") as fh:
            rows = list(_csv.DictReader(fh))
        m = {r["nnunet_id"]: r["original_case_id"] for r in rows
             if r.get("nnunet_id") and r.get("original_case_id")}
        self._id_map = m
        ok = self._case_table.add_aliases(m)
        self.print_to_log_file(f"[prompt] case_id_mapping 對照 {ok}/{len(m)} 筆接上原始 case_id")

    # ---------- dataloader ----------
    def get_dataloaders(self):
        from nnunetv2.training.dataloading import data_loader_3d as _m3d
        from nnunetv2.training.dataloading.data_loader_3d_prompt import nnUNetDataLoader3DPrompt

        tab = self._ensure_case_table()
        self._assert_feature_coverage(tab)
        nnUNetDataLoader3DPrompt.case_feature_table = tab
        nnUNetDataLoader3DPrompt.use_patch_features = self.USE_PATCH_FEATURES
        nnUNetDataLoader3DPrompt.v4_channel = self.V4_CHANNEL
        nnUNetDataLoader3DPrompt.neighborhood_half = self.NEIGHBORHOOD_HALF
        # ⚠ 必須改「nnUNetTrainer 模組」的 global，不能只改來源模組。
        #   nnUNetTrainer.py 是 `from ...data_loader_3d import nnUNetDataLoader3D`，
        #   import 當下就把類別綁進自己的命名空間；只改 _m3d 的屬性對它毫無作用，
        #   結果是 prompt dataloader 從未被使用、batch 裡沒有 case_feat/patch_feat，
        #   set_prompt(None, None) → 網路補零 → 條件化靜默失效。
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _mtr
        saved = [(m, getattr(m, "nnUNetDataLoader3D", None)) for m in (_mtr, _m3d)]
        for m, _ in saved:
            m.nnUNetDataLoader3D = nnUNetDataLoader3DPrompt
        try:
            return super().get_dataloaders()
        finally:
            for m, o in saved:
                if o is not None:
                    m.nnUNetDataLoader3D = o

    def _assert_feature_coverage(self, tab, min_cov: float = 0.99):
        """訓練啟動前確認特徵查得到，查不到就中止。

        2026-09-01：四個實驗各跑滿 300 epoch 後才發現查表 0/3057 全落空
        （nnunet_id 對不上以原始 case_id 為鍵的 CSV），等於整輪都餵零向量。
        當時的監控探針量的是「有沒有在調制」，而 encoder 光靠 bias 就會產生
        非零常數，所以看起來一切正常。寧可在這裡炸掉也不要再靜默跑三天。
        """
        if tab is None:
            return
        tr, val = self.do_split()
        cov = tab.coverage(list(tr) + list(val))
        self.print_to_log_file(f"[prompt] 特徵覆蓋率 {cov:.1%}（train {len(tr)} + val {len(val)}）")
        if cov < min_cov:
            raise RuntimeError(
                f"[prompt] 特徵覆蓋率只有 {cov:.1%}（需 ≥{min_cov:.0%}）。"
                f"查表落空會靜默回零向量、條件化完全失效。"
                f"請確認 case_id_mapping.csv 是否存在、CSV 的 case_id 是否為原始命名。")

    # ---------- 網路 ----------
    def initialize(self):
        super().initialize()
        if not isinstance(self.network, PromptNetworkWrapper):
            self.network = PromptNetworkWrapper(self.network).to(self.device)
            self.print_to_log_file("[prompt] network 已包上 PromptNetworkWrapper")

    # ---------- 每步注入 ----------
    def _apply_prompt(self, batch: dict):
        if not getattr(self, "_batch_checked", False):
            self._batch_checked = True
            missing = [k for k in ("case_feat",) if k not in batch]
            if self.USE_PATCH_FEATURES and "patch_feat" not in batch:
                missing.append("patch_feat")
            if missing:
                raise RuntimeError(
                    f"[prompt] batch 缺少 {missing} —— prompt dataloader 沒有生效，"
                    f"特徵不會進入網路（set_prompt 會收到 None，forward 補零向量，"
                    f"訓練照跑但條件化完全無效）。實際拿到的鍵：{sorted(batch.keys())}")
        cv = batch.get("case_feat")
        pv = batch.get("patch_feat")
        to_t = lambda v: (None if v is None else
                          (v if isinstance(v, torch.Tensor) else torch.from_numpy(np.asarray(v)))
                          .to(self.device, non_blocking=True).float())
        self.network.set_prompt(to_t(cv), to_t(pv) if self.USE_PATCH_FEATURES else None)
        self._track_feature_spread(cv, pv)
        self._audit_first_batch(batch)
        if self.NETWORK_IN_CHANNELS is not None:
            d = batch["data"]
            if d.shape[1] > self.NETWORK_IN_CHANNELS:
                batch = dict(batch, data=d[:, :self.NETWORK_IN_CHANNELS])
        return batch

    def _audit_first_batch(self, batch):
        """第一疊代把實際餵進網路的特徵印到 log，可直接與 CSV 逐值核對。

        存在的理由：查表落空、鍵對不上、座標框搞錯這幾種錯都不會拋例外，
        只會安靜地餵零或餵錯。唯一可靠的驗證是把實際跑的那一批印出來對照。
        """
        if getattr(self, "_audited", False):
            return
        self._audited = True
        try:
            import pandas as pd
            from mutp.data.prompt_features import CASE_COLS, PATCH_COLS, strip_crop_suffix
            log = self.print_to_log_file
            # ⚠ 不能寫 `batch.get("keys") or []` —— keys 是 ndarray，
            #   對它取布林值會拋 "truth value of an array is ambiguous"
            _k = batch.get("keys")
            keys = [] if _k is None else list(_k)
            cf = np.asarray(batch.get("case_feat"))
            pf = batch.get("patch_feat")
            pf = None if pf is None else np.asarray(pf)
            ctr = batch.get("patch_center")
            vsh = batch.get("patch_vol_shape")

            # 獨立重讀 CSV（不經 CaseFeatureTable），才有對照價值
            df = pd.concat([pd.read_csv(x) for x in self.CASE_FEATURE_CSVS], ignore_index=True)
            df = df.drop_duplicates("case_id").set_index("case_id")
            idmap = getattr(self, "_id_map", {}) or {}

            COV = PATCH_COLS.index("patch_vessel_coverage")
            log(f"[prompt] ═══ 第一疊代特徵稽核（case {len(CASE_COLS)} 維 / patch {len(PATCH_COLS)} 維）═══")
            for j in range(min(2, len(keys))):
                k = keys[j]
                orig = idmap.get(k, k)
                log(f"[prompt] 樣本{j}: {k} → {orig}")
                # 與正式查表同一套解析：先精確、再剝 crop 尾碼
                ck = orig if orig in df.index else strip_crop_suffix(orig)
                row = df.loc[ck] if ck in df.index else None
                if row is None:
                    log(f"[prompt]   ⚠ CSV 找不到 {orig}（也試過 {strip_crop_suffix(orig)}）"
                        f" —— 查表落空，特徵是零向量！")
                else:
                    csv_v = row[CASE_COLS].to_numpy(dtype=np.float32)
                    d = float(np.abs(cf[j] - csv_v).max())
                    log("[prompt]   pipeline: " + " ".join(f"{c}={v:+.4f}" for c, v in zip(CASE_COLS, cf[j])))
                    log("[prompt]   CSV     : " + " ".join(f"{c}={v:+.4f}" for c, v in zip(CASE_COLS, csv_v)))
                    log(f"[prompt]   最大絕對差={d:.3e}  {'✓ 一致' if d < 1e-4 else '✗ 不一致'}")
                if pf is not None and self.USE_PATCH_FEATURES:
                    c3 = None if ctr is None else np.asarray(ctr)[j]
                    v3 = None if vsh is None else np.asarray(vsh)[j]
                    log(f"[prompt]   patch 中心={c3.tolist() if c3 is not None else '?'} "
                        f"vol_shape={v3.tolist() if v3 is not None else '?'}"
                        f"（patch_coord 的分母就是這個 volume；已驗證與 CSV 的 crop_half 框等價）")
                    log("[prompt]   " + " ".join(f"{c}={v:+.4f}" for c, v in zip(PATCH_COLS[:3], pf[j][:3])))
                    # 索引由欄名推導，維度變動時不會再對不上（14→13 就踩過一次）
                    dm = [i for i, c in enumerate(PATCH_COLS) if c.startswith("patch_dominant")]
                    nb = [i for i, c in enumerate(PATCH_COLS) if c.startswith("patch_neighbor")]
                    log("[prompt]   dominant=" + np.array2string(pf[j][dm], precision=2)
                        + f" ({','.join(PATCH_COLS[i].split('_')[-1] for i in dm)})")
                    log("[prompt]   neighbor=" + np.array2string(pf[j][nb], precision=4)
                        + f" ({','.join(PATCH_COLS[i].split('_')[-1] for i in nb)})"
                        + f"  vessel_coverage={pf[j][COV]:.4f}")
            if pf is not None and self.USE_PATCH_FEATURES:
                log(f"[prompt] 全批 patch: coord 範圍 [{pf[:, :3].min():+.3f}, {pf[:, :3].max():+.3f}]"
                    f"  vessel_coverage 平均 {pf[:, COV].mean():.4f}"
                    f"  取樣到血管的比例 {(pf[:, COV] > 0).mean():.1%}"
                    f"  各維 std 最大/最小 {pf.std(0).max()/max(pf.std(0).min(), 1e-9):.1f}×")
            log(f"[prompt] 全批 case: 跨樣本 std 平均 {cf.std(0).mean():.4f}"
                f"（0=每個樣本特徵相同，條件化無效）")
            log("[prompt] ═══════════════════════════════════")
        except Exception as ex:
            self.print_to_log_file(f"[prompt] 稽核輸出失敗（不影響訓練）：{type(ex).__name__}: {ex}")

    def _track_feature_spread(self, cv, pv):
        """累計「特徵是否隨 case 變化」。

        只量調制強度是不夠的：即使餵進來的是全零，encoder 光靠 bias 也會輸出
        一個非零常數，探針照樣顯示 attn≈0.45 / film≈0.20，看起來完全正常。
        真正的判準是**跨 batch 的離散度** —— 全零或常數輸入時 std=0。
        """
        acc = getattr(self, "_feat_spread", None)
        if acc is None:
            acc = self._feat_spread = {"c_std": 0.0, "c_abs": 0.0, "p_std": 0.0, "n": 0}
        for v, ks in ((cv, ("c_std", "c_abs")), (pv, ("p_std", None))):
            if v is None:
                continue
            t = v if isinstance(v, torch.Tensor) else torch.from_numpy(np.asarray(v))
            t = t.detach().float()
            if t.ndim != 2 or t.shape[0] < 2:
                continue
            acc[ks[0]] += float(t.std(dim=0).mean())
            if ks[1]:
                acc[ks[1]] += float(t.abs().mean())
        acc["n"] += 1

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
            sp = getattr(self, "_feat_spread", None)
            if sp and sp["n"]:
                n = sp["n"]
                c_std, c_abs, p_std = sp["c_std"] / n, sp["c_abs"] / n, sp["p_std"] / n
                warn = "  ⚠ case 特徵無變異＝查表落空或常數，條件化等同無效" if c_std < 1e-4 else ""
                self.print_to_log_file(
                    f"[prompt] 特徵離散度 case_std={c_std:.4f} case_|x|={c_abs:.4f} "
                    f"patch_std={p_std:.4f}（跨 batch；0=每個 case 都一樣）{warn}")
                self._feat_spread = None
        except AttributeError:
            pass
