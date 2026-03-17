# DefaultPreprocessor 使用說明

## 概述

`DefaultPreprocessor` 已經過客製化，支援額外的 `vessel` 和 `dilate` 標註資料。設計上同時支援**訓練**和**推理**兩種使用場景。

## 使用方式

### 1. 推理時使用（Jupyter Notebook / 測試）

推理時**不需要**提供 `vessel_file` 和 `dilate_file` 參數，它們會預設為 `None`：

```python
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

# 初始化
pp = DefaultPreprocessor()
plans_manager = PlansManager(plans_file)

# 推理時調用 - vessel_file 和 dilate_file 可省略
data, seg, properties = pp.run_case(
    image_files=input_images,
    seg_file=None,  # 推理時通常為 None
    plans_manager=plans_manager,
    configuration_manager=plans_manager.get_configuration(configuration),
    dataset_json=dataset_json_file
    # vessel_file 和 dilate_file 預設為 None，不需要指定
)
```

### 2. 訓練時使用（前處理資料集）

訓練時會自動處理 `vesselsTr` 和 `dilationsTr` 資料夾：

```python
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

pp = DefaultPreprocessor(verbose=True)

# run 方法會自動尋找並處理 vessel 和 dilate 檔案
pp.run(
    dataset_name_or_id='Dataset001_MyDataset',
    configuration_name='3d_fullres',
    plans_identifier='nnUNetPlans',
    num_processes=8
)
```

在訓練前處理過程中：
- 自動從 `vesselsTr/` 資料夾讀取 vessel 標註
- 自動從 `dilationsTr/` 資料夾讀取 dilate 標註
- 生成的 `.pkl` 檔案會包含 `vessel_locations` 和 `dilate_locations` 資訊

## 資料夾結構

訓練資料集應包含以下結構：

```
nnUNet_raw/Dataset001_MyDataset/
├── imagesTr/
│   ├── case001_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── case001.nii.gz
│   └── ...
├── vesselsTr/          # 血管標註（可選，訓練時使用）
│   ├── case001.nii.gz
│   └── ...
└── dilationsTr/        # 擴張標註（可選，訓練時使用）
    ├── case001.nii.gz
    └── ...
```

## 向後相容性

✅ **完全相容**：此設計保持與原版 nnUNet 的相容性
- 推理代碼無需修改
- 不使用 vessel/dilate 功能時行為與原版相同
- 訓練時若 `vesselsTr/` 或 `dilationsTr/` 不存在會自動跳過

## 技術細節

### 參數設計

- `run_case()` 方法：
  - `vessel_file` 和 `dilate_file` 是**可選的關鍵字參數**（預設值 `None`）
  - 推理時可省略這些參數

- `run_case_save()` 方法：
  - 內部方法，供 `run()` 批次處理時使用
  - 與 `ptqdm` 平行處理框架配合

### 產生的資料

處理後的資料會包含：
- `class_locations`: 前景類別的採樣位置（原有）
- `vessel_locations`: 血管位置座標（新增，如果提供 vessel 檔案）
- `dilate_locations`: 擴張區域位置（新增，如果提供 dilate 檔案）

## 常見問題

### Q: Jupyter notebook 中出現參數錯誤？
A: 確認調用 `run_case()` 時沒有傳入 `vessel_file` 和 `dilate_file`，或使用關鍵字參數形式。

### Q: 訓練時沒有 vessel 或 dilate 資料怎麼辦？
A: 程式碼已經處理了 `None` 的情況，不會影響正常訓練流程。

### Q: 如何在推理時使用 vessel 資訊？
A: 如果需要，可以明確傳入：
```python
data, seg, properties = pp.run_case(
    ...,
    vessel_file=vessel_path,
    dilate_file=dilate_path
)
```

## 更新日誌

- **2025-11**: 將 `vessel_file` 和 `dilate_file` 改為可選參數，實現向後相容

