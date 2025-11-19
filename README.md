# 腦動脈瘤檢測與分割系統

## 專案簡介
本專案實現了基於 nnU-Net 架構的腦動脈瘤自動檢測與分割系統，整合了血管分割、動脈瘤檢測等多個模組，能夠從 MRA 影像自動識別和定位顱內動脈瘤。

## 環境設置

### 系統需求

**硬體需求：**
- GPU：NVIDIA GPU（建議 V100 或更高等級）
- VRAM：建議 16GB 以上
- CPU：16 核心以上
- RAM：建議 64GB 以上（訓練時 128GB 更佳）
- 儲存空間：建議 500GB 以上

**軟體需求：**
- Python：3.9-3.11
- CUDA：12.4
- cuDNN：對應 CUDA 12.4 的版本

### 套件安裝

本專案提供完整的 `requirements.txt` 檔案，包含所有必要的依賴套件。

#### 快速安裝指南

**步驟 1：建立虛擬環境**
```bash
conda create -n aneurysm python=3.10
conda activate aneurysm
```

**步驟 2：安裝 PyTorch（CUDA 12.4）**
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

**步驟 3：安裝其他套件**
```bash
pip install -r requirements.txt
```

**步驟 4：安裝 nnU-Net**
```bash
cd nnResUNet
pip install -e .
```

**步驟 5：驗證安裝**
```bash
python -c "import torch; import tensorflow as tf; import nnunetv2; print('✅ 安裝成功！')"
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"
```

#### 主要套件版本

| 套件 | 版本 | 用途 |
|------|------|------|
| PyTorch | 2.6.0 | nnU-Net 深度學習框架 |
| TensorFlow | 2.14.0 | 血管分割與前處理模型 |
| NumPy | 1.26.4 | 數值計算 |
| nbclassic | 0.5.5 | 互動式開發環境（支援 Ctrl+/ 註解） |
| nibabel | 5.2.1 | NIfTI 影像處理 |
| SimpleITK | 2.3.1 | 醫學影像處理 |
| scikit-image | 0.22.0 | 影像處理與形態學操作 |
| brainextractor | latest | 腦部組織提取 |

完整的套件列表請參考 `requirements.txt` 檔案。

#### 疑難排解

**問題 1：Jupyter 啟動錯誤（ModuleNotFoundError: jupyter_server.contents）**
```bash
# 錯誤原因：notebook 和 jupyter-server 版本不相容
# 解決方案：使用 nbclassic（Notebook 6.x 的現代實現）
# 
# === 快速修復（推薦）===
pip uninstall -y notebook jupyter-server jupyter-client jupyter-core nbclassic
pip install -r requirements.txt

# === 驗證安裝 ===
jupyter --version
python -c "import nbclassic; print('✅ nbclassic 安裝成功')"

# === 啟動 Jupyter Notebook ===
# 方法 1：使用 nbclassic（推薦）
jupyter nbclassic

# 方法 2：創建別名後使用 notebook（需先設定別名）
# echo "alias notebook='jupyter nbclassic'" >> ~/.bashrc
# source ~/.bashrc
# notebook --ip=10.103.1.188 --no-browser

# 遠端存取範例
jupyter nbclassic --ip=10.103.1.188 --no-browser

# 允許所有 IP 存取
jupyter nbclassic --ip=0.0.0.0 --port=8888 --no-browser

# 注意：
# - nbclassic 是 Notebook 6.x 的現代化實現
# - 完全支援 Ctrl+/ 快速註解/取消註解程式碼
# - 介面和功能與傳統 notebook 完全相同
# - 更穩定，相容性更好
# - 如需使用 'jupyter notebook' 指令，請設定別名（見上方）
```

**如何使用 `jupyter notebook` 指令（可選設定）：**

如果您習慣使用 `jupyter notebook` 而非 `jupyter nbclassic`，可以設定別名：

```bash
# 在 ~/.bashrc 中添加別名
echo "alias notebook='jupyter nbclassic'" >> ~/.bashrc
source ~/.bashrc

# 現在可以使用
notebook
notebook --ip=10.103.1.188 --no-browser
```

**Jupyter Notebook 常用快捷鍵：**
- `Ctrl + /`：註解/取消註解選中的程式碼 ⭐
- `Shift + Enter`：執行當前 cell 並移到下一個
- `Ctrl + Enter`：執行當前 cell 不移動
- `A`：在上方插入新 cell（命令模式）
- `B`：在下方插入新 cell（命令模式）
- `D + D`：刪除當前 cell（命令模式）
- `M`：將 cell 轉為 Markdown（命令模式）
- `Y`：將 cell 轉為 Code（命令模式）

**問題 2：brainextractor 安裝失敗**
```bash
# 嘗試從 GitHub 安裝
pip install git+https://github.com/dylanhsu/BrainExtractor.git
```

**問題 3：CUDA 版本不符**
```bash
# 確認系統 CUDA 版本
nvcc --version

# 安裝對應版本的 PyTorch
# 請至 https://pytorch.org/ 查詢對應版本
```

**問題 4：記憶體不足**
- 減少 batch size
- 使用較小的 patch size
- 啟用混合精度訓練

### TWCC 開發環境

本專案在 TWCC（台灣 AI 雲）上進行開發和訓練：

**容器配置：**
- 容器類型：開發型容器
- 映像檔：`pytorch-24.05-py3:latest`
- GPU：NVIDIA V100
- CPU：16 核心以上
- 記憶體：建議 128 GB 以上

**環境安裝：**
```bash
# 1. 按照上述「套件安裝」步驟安裝所有依賴

# 2. 設置 nnU-Net 環境變數
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# 建議將上述環境變數寫入 ~/.bashrc
echo 'export nnUNet_raw="/path/to/nnUNet_raw"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"' >> ~/.bashrc
echo 'export nnUNet_results="/path/to/nnUNet_results"' >> ~/.bashrc
source ~/.bashrc
```

## 資料準備

### 資料結構
本專案使用標準的 nnU-Net 資料結構：

```
nnUNet_raw/
├── Dataset550_MRA_Vessel/        # 血管分割模型資料集
│   ├── imagesTr/                 # 訓練影像
│   ├── labelsTr/                 # 訓練標註
│   ├── imagesTs/                 # 測試影像（可選）
│   └── dataset.json              # 資料集描述檔
│
└── Dataset127_DeepAneurysm/      # 動脈瘤分割模型資料集
    ├── imagesTr/                 # 訓練影像
    ├── labelsTr/                 # 訓練標註
    ├── vesselsTr/                # 控制訓練時的正常樣本範圍，放Vessel.nii.gz，使訓練時只會隨機到存在vessel的地方
    ├── dilationsTr/              # 訓練時的病灶範圍，只跟採樣有關，放label，即只會隨機採樣有標註的座標點，手動dilate標註可以增加採樣的範圍(可做)      
    ├── imagesTs/                 # 測試影像（可選）
    └── dataset.json              # 資料集描述檔
```

### 資料格式要求
- 影像格式：NIfTI (.nii.gz)
- 檔名格式：`{CASE_ID}_0000.nii.gz` （影像）
- 標註格式：`{CASE_ID}.nii.gz` （標註）
- 影像類型：MRA 影像（TOF-MRA）

## 模型訓練

### 1. 血管分割模型訓練 (Dataset550_MRA_Vessel)

**步驟 1：資料前處理和實驗規劃**
需要先使用make_normalized_img.py將影像正規化或在dataset.json與nnU-Net中自訂正規化的方式

```bash
# 使用 16 個 CPU 核心進行前處理
taskset -c 0-15 nnUNetv2_plan_and_preprocess -d 550 -c 3d_fullres -np 16
```

**步驟 2：修改訓練計劃（如需要）**
```bash
# 可選：使用自定義的 plans 檔案
# 將提供的 nnUNetPlans_5L.json 複製到對應資料夾
cp nnUNetPlans_5L.json nnUNet_preprocessed/Dataset550_MRA_Vessel/
```

**步驟 3：開始訓練**
```bash
# 訓練 fold 0（單一 fold 訓練）
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 nnUNetv2_train 550 3d_fullres 0 \
  -p nnUNetPlans_5L \
  --num_iterations_per_epoch 500 \
  --enable_early_stopping
```

### 2. 動脈瘤分割模型訓練 (Dataset127_DeepAneurysm)

**步驟 1：資料前處理和實驗規劃**
```bash
# 使用 16 個 CPU 核心進行前處理
taskset -c 0-45 nnUNetv2_plan_and_preprocess -d 127 -c 3d_fullres -np 16
```

**步驟 2：修改訓練計劃**
```bash
# 使用自定義的 plans 檔案以優化模型效能
cp nnUNetPlans_5L-b900.json nnUNet_preprocessed/Dataset127_DeepAneurysm/
```

**步驟 3：開始訓練**
```bash
# 訓練 fold 0
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 nnUNetv2_train 127 3d_fullres 0 \
  -p nnUNetPlans_64x-5L-b110 \
  --num_iterations_per_epoch 500 \
  --enable_early_stopping
```

### 訓練參數說明

#### 必要參數
- `dataset_name_or_id` (位置參數 1)：資料集名稱或 ID
  - 範例：`127` 或 `Dataset127_DeepAneurysm`
  - 127 = 動脈瘤分割，550 = 血管分割

- `configuration` (位置參數 2)：訓練配置
  - `3d_fullres`：3D 全解析度訓練（最常用）
  - `3d_lowres`：3D 低解析度訓練（用於級聯架構）
  - `2d`：2D 訓練

- `fold` (位置參數 3)：交叉驗證的 fold 編號
  - `0-4`：單一 fold 訓練
  - `all`：訓練所有 5 個 folds

#### 常用選項參數
- `-tr` / `--trainer`：Trainer 類別名稱
  - 預設：`nnUNetTrainer`
  - 可自定義 trainer 以實現特殊訓練策略

- `-p` / `--plans`：Plans 檔案識別符
  - 預設：`nnUNetPlans`
  - 範例：`nnUNetPlans_64x-5L-b110` 使用自定義的 plans 設定
  - Plans 檔案決定：網路架構、patch size、batch size、資料增強策略等

- `-pretrained_weights`：預訓練權重檔案路徑
  - 用於遷移學習或微調
  - 範例：`/path/to/checkpoint_best.pth`

- `-num_gpus`：使用的 GPU 數量
  - 預設：`1`
  - 多 GPU 訓練會使用 DDP (Distributed Data Parallel)

- `-device`：運算設備
  - 選項：`cuda` (GPU), `cpu` (CPU), `mps` (Apple M1/M2)
  - 預設：`cuda`
  - 注意：使用 `CUDA_VISIBLE_DEVICES` 指定 GPU ID，而非此參數

#### 訓練超參數
- `--initial_lr`：初始學習率
  - 預設：`1e-4` (0.0001)
  - 較大的學習率可能加速收斂但不穩定
  - 較小的學習率更穩定但訓練較慢

- `--num_iterations_per_epoch`：每個 epoch 的迭代次數
  - 預設：`500`
  - 建議範圍：100-500
  - 較少的迭代可加速訓練但可能影響效能

- `--num_epochs`：訓練的總 epoch 數
  - 預設：`1000`
  - 實際訓練通常會因早停機制提前結束

- `--optimizer`：優化器類型
  - 選項：`SGD`, `AdamW`
  - 預設：`AdamW`
  - AdamW 通常收斂更快，SGD 可能獲得更好的泛化

- `--lr_scheduler`：學習率調度器
  - 選項：`PolyLRScheduler`, `CosineAnnealingLR`
  - 預設：`CosineAnnealingLR`
  - Cosine 調度器在訓練後期提供更平滑的學習率衰減

- `--oversample_foreground_percent`：訓練時前景過採樣比例
  - 預設：`0.5` (50%)
  - 範圍：0.0-1.0
  - 較高的值會讓模型更關注前景（病灶）區域

- `--oversample_foreground_percent_val`：驗證時前景過採樣比例
  - 預設：`0.2` (20%)
  - 範圍：0.0-1.0
  - 通常低於訓練值以獲得更真實的驗證結果

#### 早停機制參數
- `--enable_early_stopping`：啟用早停機制
  - 預設：關閉
  - 建議開啟以避免過擬合並節省訓練時間

- `--early_stopping_patience`：早停耐心值（epoch 數）
  - 預設：`200`
  - 在驗證指標無改善的情況下等待的 epoch 數

- `--early_stopping_min_delta`：早停最小改善量
  - 預設：`0.0001`
  - 驗證 Dice 分數的最小改善量才算有效改善

#### 資料與檢查點選項
- `--use_compressed`：使用壓縮資料
  - 預設：關閉
  - 開啟後不解壓訓練資料，節省磁碟空間但增加 CPU/RAM 負擔

- `--npz`：儲存驗證集的 softmax 預測
  - 用於後續的模型 ensemble

- `--c`：繼續訓練
  - 從最新的檢查點繼續訓練
  - 自動尋找 `checkpoint_final.pth` 或 `checkpoint_latest.pth`

- `--val`：僅執行驗證
  - 不進行訓練，僅在驗證集上評估模型
  - 需要 `checkpoint_final.pth` 存在

- `--disable_checkpointing`：停用檢查點儲存
  - 用於測試，不建議正式訓練時使用

#### 系統資源控制
- `CUDA_VISIBLE_DEVICES=X`：指定使用的 GPU
  - 範例：`CUDA_VISIBLE_DEVICES=0` 使用第 0 號 GPU
  - 範例：`CUDA_VISIBLE_DEVICES=0,1` 使用第 0 和 1 號 GPU

- `taskset -c 0-15`：限制使用的 CPU 核心
  - 避免與其他程序資源競爭
  - 範例：`taskset -c 0-15` 使用核心 0-15（共 16 核心）

#### 訓練指令範例

**基本訓練（使用預設參數）：**
需要先使用make_normalized_img.py將影像正規化或在dataset.json與nnU-Net中自訂正規化的方式
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 127 3d_fullres 0
```

**自定義訓練（推薦設定）：**
```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 nnUNetv2_train 127 3d_fullres 0 \
  -p nnUNetPlans_64x-5L-b110 \
  --num_iterations_per_epoch 500 \
  --num_epochs 1000 \
  --initial_lr 1e-4 \
  --optimizer AdamW \
  --lr_scheduler CosineAnnealingLR \
  --enable_early_stopping \
  --early_stopping_patience 200 \
  --oversample_foreground_percent 0.5
```

**使用預訓練權重進行微調：**
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 127 3d_fullres 0 \
  -p nnUNetPlans_64x-5L-b110 \
  -pretrained_weights /path/to/pretrained/checkpoint_best.pth \
  --initial_lr 5e-5 \
  --num_iterations_per_epoch 250 \
  --enable_early_stopping
```

**多 GPU 訓練（DDP）：**
```bash
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 127 3d_fullres 0 \
  -p nnUNetPlans_64x-5L-b110 \
  -num_gpus 2 \
  --enable_early_stopping
```

**繼續中斷的訓練：**
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 127 3d_fullres 0 \
  -p nnUNetPlans_64x-5L-b110 \
  --c
```

## 模型推理

### 方法 1：單一動脈瘤模型推理

使用 `nnResUNet/predict_from_raw_data_jupyter_test_gaussian_1to1_1_aneurysm.ipynb`

**操作步驟：**
1. 需要先使用make_normalized_img.py將影像正規化

2. 開啟 Jupyter Notebook：
   ```bash
   jupyter nbclassic nnResUNet/predict_from_raw_data_jupyter_test_gaussian_1to1_1_aneurysm.ipynb
   ```

3. 執行前面的所有 cell 以載入必要的函數和類別

4. 修改最後一個 cell 的推理參數：
   ```python
   predict_from_raw_data(
       # 輸入資料夾（MRA 正規化後的影像）
       '/path/to/input/images/',
       
       # 血管遮罩資料夾（用於加速推理）
       '/path/to/vessel/masks/',
       
       # 輸出資料夾
       '/path/to/output/',
       
       # 模型路徑
       '/path/to/nnUNet_results/Dataset127_DeepAneurysm/nnUNetTrainer__nnUNetPlans__3d_fullres',
       
       # 使用的 folds（例如：(0,) 或 (0,1,2,3,4) 用於 ensemble）
       (0,),
       
       # 滑動視窗步長（0.25 = 75% 重疊）
       0.25,
       
       # Gaussian 平滑（True = 高品質，False = 高速度）
       use_gaussian=True,
       
       # 測試時資料增強（鏡像）
       use_mirroring=False,
       
       # GPU 運算
       perform_everything_on_gpu=True,
       
       # 詳細輸出
       verbose=True,
       
       # 儲存機率圖
       save_probabilities=False,
       
       # 是否覆蓋現有結果
       overwrite=False,
       
       # 使用的檢查點
       checkpoint_name='checkpoint_best.pth',
       
       # 前處理程序數
       num_processes_preprocessing=2,
       
       # 後處理程序數
       num_processes_segmentation_export=3,
       
       # GPU 編號
       desired_gpu_index=0,
       
       # Batch size（建議：4-112，取決於 GPU 記憶體）
       batch_size=112
   )
   ```

5. 執行該 cell 開始推理

**效能調整建議：**
- **最高品質模式**：`use_gaussian=True, batch_size=4`
- **平衡模式**：`use_gaussian=True, batch_size=8-16`
- **最快速度模式**：`use_gaussian=False, batch_size=32-112`

### 方法 2：完整 Pipeline 推理(目前僅驗證過雙和、亞東資料)

使用 `pipline_aneurysm.ipynb` 進行完整的 MRA 影像處理（包含前處理、血管分割、動脈瘤檢測）

**操作步驟：**

1. 開啟 Jupyter Notebook：
   ```bash
   jupyter nbclassic pipline_aneurysm.ipynb
   ```

2. 準備輸入資料：
   - 將 MRA 影像（NIfTI 格式）放置在指定的輸入資料夾
   - 確保檔名符合命名規範

3. 修改 Notebook 中的配置參數：
   - 輸入資料夾路徑
   - 輸出資料夾路徑
   - 模型權重路徑（血管模型 + 動脈瘤模型）
   - GPU 設定

4. 按順序執行所有 cell：
   - **Cell 1-N**：載入套件和模型
   - **預處理 Cell**：影像標準化、重採樣等
   - **血管分割 Cell**：使用 Dataset550 模型進行血管分割
   - **動脈瘤檢測 Cell**：使用 Dataset127 模型進行動脈瘤檢測
   - **後處理 Cell**：結果整合與輸出

5. 檢查輸出結果：
   - 血管分割結果
   - 動脈瘤分割結果
   - 動脈瘤機率圖

**Pipeline 處理流程：**
```
原始 MRA 影像 
    ↓
前處理（標準化、裁切、重採樣）
    ↓
血管分割（Dataset550 模型）
    ↓
動脈瘤檢測（Dataset127 模型，僅在血管區域）
    ↓
後處理（連通域分析、小區域過濾）
    ↓
輸出結果（分割遮罩 + 機率圖）
```

## 輸出結果說明

推理完成後，輸出資料夾將包含以下檔案：

**完整 Pipeline 輸出（使用 `pipline_aneurysm.ipynb`）：**

```
example_output/
└── {CASE_ID}/                              # 案例資料夾
    ├── Pred_Aneurysm.nii.gz               # 動脈瘤分割結果（二值遮罩）
    ├── Prob_Aneurysm.nii.gz               # 動脈瘤機率圖（0-1 連續值）
    ├── Pred_Aneurysm_Vessel.nii.gz        # 血管結果
    └── Pred_Aneurysm_Vessel16.nii.gz      # 16 個血管區域結果
```

### 範例輸出

參考 `example_output/00165585_20221018_MR_21110170148/` 資料夾查看完整的輸出範例。

## 目錄結構

```
nnResUNet-github/
├── nnResUNet/                                    # nnU-Net 核心程式碼
│   ├── nnunetv2/                                # nnU-Net v2 套件
│   ├── predict_from_raw_data_jupyter_test_*.ipynb  # 推理 notebook
│   └── gpu_nnUNet.py                            # GPU 推理腳本
├── pipeline_aneurysm_torch.py                    # PyTorch Pipeline 腳本
├── pipline_aneurysm.ipynb                       # 完整 Pipeline Notebook
├── gpu_aneurysm.py                              # 動脈瘤推理腳本
├── model_weights/                               # 模型權重資料夾
├── example_input/                               # 範例輸入資料
├── example_output/                              # 範例輸出結果
├── nnUNet_results/                              # 訓練結果（模型權重）
│   ├── Dataset550_MRA_Vessel/
│   └── Dataset127_DeepAneurysm/
└── README.md                                    # 本文件
```

## 引用

如果您使用本專案，請引用以下論文：

```bibtex
@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## 致謝

本專案基於以下開源專案開發：

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - 醫學影像分割框架
- [MONAI](https://github.com/Project-MONAI/MONAI) - 醫學影像 AI 工具包
- [PyTorch](https://pytorch.org/) - 深度學習框架

## 授權

本專案遵循 Apache 2.0 授權條款。

## 聯絡方式

如有任何問題或建議，請透過 GitHub Issues 聯繫。

