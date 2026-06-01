# nnResUNet-github 版本盤點清單
# 建立日期：2026-03-23
# 總計：4 個漸進式版本

---

## 版本演進路線

```
nnResUNet (基礎版)
    → nnResUNet-classifier (加入分類頭)
        → nnResUNet-upsample (加入加權取樣)
            → nnResUNet-location-cls-upsample (分類器+上採樣+加權取樣) ← 目前開發版
```

---

## 各版本比較

| 特性 | nnResUNet | nnResUNet-classifier | nnResUNet-upsample | nnResUNet-location-cls-upsample |
|------|-----------|---------------------|--------------------|---------------------------------|
| 狀態 | 已棄用 | 實驗性 | 中間版本 | **目前開發中** ✅ |
| 基礎分割 | ✅ | ✅ | ✅ | ✅ |
| 分類器頭 | - | ✅ | - | ✅ |
| 加權取樣 | - | - | ✅ | ✅ |
| 位置分類 | - | - | - | ✅ |
| 注意力機制 | - | - | - | ✅ (Attention/Guided) |
| AdamW + Cosine | - | - | - | ✅ |
| Early Stopping | - | - | - | ✅ |
| MLflow 追蹤 | - | - | - | ✅ |

---

## 版本 1：nnResUNet/（基礎版）

- 標準 nnU-Net v2 分支
- 基本 PlainConvUNet / ResidualEncoderUNet
- 無特殊客製化
- **建議：不納入 MUTP，已被後續版本完全覆蓋**

---

## 版本 2：nnResUNet-classifier/（分類器版）

- 加入動脈瘤存在與否的分類頭
- 測試用 notebooks：
  - Show_train_input.ipynb — 訓練資料檢視
  - Show_train_input_augmentation.ipynb — 增強效果檢視
  - Show_train_input_FEMH.ipynb — 遠東紀念醫院資料
  - test_dataloader.ipynb — DataLoader 測試
- **建議：分類器概念已整合至版本 4，不單獨納入**

---

## 版本 3：nnResUNet-upsample/（加權取樣版）

- 實作類別加權取樣（4 類血管）
- 兩種權重模式：
  - `multiplier`：直接乘以權重倍率
  - `target_proportion`：目標比例平衡
- CLI 參數：`--sampling_category_weights "2:1:1:1" --sampling_category_weight_mode target_proportion`
- 文件：SAMPLING_CATEGORY_UPSAMPLE.md
- **建議：取樣機制已整合至版本 4，不單獨納入**

---

## 版本 4：nnResUNet-location-cls-upsample/（目前開發版）✅

整合所有前版特性，為最完整版本。

### 模型架構（6 種）
| 架構 | 分類器 | 注意力 | 導引 | 2D 版 |
|------|--------|--------|------|-------|
| ResidualEncoderUNetClassifier | ✅ | - | - | ✅ |
| ResidualEncoderUNetAttentionClassifier | ✅ | ✅ | - | ✅ |
| ResidualEncoderUNetGuidedClassifier | ✅ | - | ✅ | ✅ |

### 訓練特性
- 優化器：AdamW（lr=1e-4, weight_decay=3e-5）
- 排程器：CosineAnnealingLR
- Early stopping：可設定 patience/min_delta
- 加權取樣：4 類血管類別
- 實驗追蹤：MLflow

### 相關文件
- SAMPLING_CATEGORY_UPSAMPLE.md — 取樣機制文件
- PREPROCESSOR_USAGE.md — 前處理用法
- TRAINING_CONFIG_USAGE.md — 訓練設定用法
- AttentionClassifier_architecture.png — 架構圖

### 關鍵檔案
- `nnunetv2/utilities/unet_v2.py` — 自定義架構定義
- `nnunetv2/training/loss/dice.py` — Loss 函數
- `nnunetv2/training/dataloading/utils.py` — 加權取樣工具
- `setup.py` — CLI 入口點定義

---

## 根目錄共用腳本

| 檔案 | 用途 |
|------|------|
| pipeline_aneurysm_torch.py | 完整推論管線（血管分割→動脈瘤偵測）|
| pipline_aneurysm.ipynb | Notebook 版推論管線 |
| gpu_aneurysm.py | GPU 處理工具 |
| make_normalized_img.py | 影像正規化前處理 |

---

## 資料集

| 資料集 | 用途 | 類別數 |
|--------|------|--------|
| Dataset550_MRA_Vessel | 血管分割 | 4 類（背景+3種血管）|
| Dataset127_DeepAneurysm | 動脈瘤偵測 | 2 類（背景+動脈瘤）|

---

## MUTP 納入建議

- **納入版本 4 (nnResUNet-location-cls-upsample) 的所有架構**
- 版本 1-3 不需單獨納入（已被版本 4 覆蓋）
- 加權取樣機制可抽取為 MUTP 通用模組
- MLflow 整合可作為 MUTP 追蹤模組的參考
- 根目錄推論管線可改寫為 MUTP 推論引擎

**→ 請確認是否同意只保留版本 4**
