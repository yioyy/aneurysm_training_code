# 訓練參數配置使用說明

## 概述

現在可以通過命令行參數配置以下訓練超參數：

1. **initial_lr**: 初始學習率
2. **oversample_foreground_percent**: 訓練時前景過採樣比例
3. **oversample_foreground_percent_val**: 驗證時前景過採樣比例
4. **num_iterations_per_epoch**: 每個 epoch 的迭代次數
5. **num_epochs**: 訓練的總 epoch 數
6. **optimizer**: 優化器類型 (SGD 或 AdamW)
7. **lr_scheduler**: 學習率調度器類型 (PolyLRScheduler 或 CosineAnnealingLR)
8. **enable_early_stopping**: 啟用提前停止機制
9. **early_stopping_patience**: 提前停止的耐心值（連續多少個 epoch 沒有改善）
10. **early_stopping_min_delta**: 定義改善的最小閾值

## 預設值

```python
initial_lr = 1e-4
oversample_foreground_percent = 0.5
oversample_foreground_percent_val = 0.2
num_iterations_per_epoch = 500
num_epochs = 1000
optimizer = 'AdamW'
lr_scheduler = 'CosineAnnealingLR'
enable_early_stopping = False
early_stopping_patience = 50
early_stopping_min_delta = 0.0001
```

## 使用範例

### 1. 使用預設值訓練

```bash
python run_training.py Dataset001_BrainTumor 3d_fullres 0
```

### 2. 自定義學習率和 epoch 數

```bash
python run_training.py Dataset001_BrainTumor 3d_fullres 0 \
    --initial_lr 0.0001 \
    --num_epochs 2000
```

### 3. 使用 SGD 優化器和 PolyLRScheduler

```bash
python run_training.py Dataset001_BrainTumor 3d_fullres 0 \
    --optimizer SGD \
    --lr_scheduler PolyLRScheduler \
    --initial_lr 0.01
```

### 4. 自定義過採樣比例

```bash
python run_training.py Dataset001_BrainTumor 3d_fullres 0 \
    --oversample_foreground_percent 0.33 \
    --oversample_foreground_percent_val 0.1
```

### 5. 自定義每個 epoch 的迭代次數

```bash
python run_training.py Dataset001_BrainTumor 3d_fullres 0 \
    --num_iterations_per_epoch 250
```

### 6. 完整配置範例

```bash
python run_training.py Dataset001_BrainTumor 3d_fullres 0 \
    --initial_lr 0.0001 \
    --oversample_foreground_percent 0.5 \
    --oversample_foreground_percent_val 0.2 \
    --num_iterations_per_epoch 500 \
    --num_epochs 1000 \
    --optimizer AdamW \
    --lr_scheduler CosineAnnealingLR
```

### 7. 多GPU訓練配置

```bash
python run_training.py Dataset001_BrainTumor 3d_fullres 0 \
    -num_gpus 4 \
    --optimizer AdamW \
    --lr_scheduler CosineAnnealingLR \
    --initial_lr 0.0002 \
    --num_epochs 1500
```

### 8. 啟用提前停止機制

```bash
python run_training.py Dataset001_BrainTumor 3d_fullres 0 \
    --enable_early_stopping \
    --early_stopping_patience 30 \
    --early_stopping_min_delta 0.0001
```

### 9. 完整配置（包含提前停止）

```bash
python run_training.py Dataset001_BrainTumor 3d_fullres 0 \
    --initial_lr 0.0001 \
    --num_epochs 1000 \
    --optimizer AdamW \
    --lr_scheduler CosineAnnealingLR \
    --enable_early_stopping \
    --early_stopping_patience 50 \
    --early_stopping_min_delta 0.0001
```

## 優化器選項

### SGD (Stochastic Gradient Descent)
- 適合：需要更穩定訓練的場景
- 參數：momentum=0.99, nesterov=True
- 建議學習率：0.01

```bash
--optimizer SGD --initial_lr 0.01
```

### AdamW
- 適合：快速收斂的場景
- 參數：使用 PyTorch 預設參數
- 建議學習率：1e-4

```bash
--optimizer AdamW --initial_lr 0.0001
```

## 學習率調度器選項

### PolyLRScheduler
- 多項式學習率衰減
- 穩定且平滑的學習率下降
- 適合較長的訓練週期

```bash
--lr_scheduler PolyLRScheduler
```

### CosineAnnealingLR
- 餘弦退火學習率
- 適合需要週期性學習率調整的場景
- T_max 自動設置為 num_epochs

```bash
--lr_scheduler CosineAnnealingLR
```

## 提前停止機制 (Early Stopping)

提前停止機制會監控驗證集的 EMA pseudo Dice 指標，當連續多個 epoch 沒有改善時自動停止訓練。

### 啟用方式
```bash
--enable_early_stopping
```

### 參數說明
- **early_stopping_patience**: 容忍的連續未改善 epoch 數（預設：50）
- **early_stopping_min_delta**: 判定為改善的最小變化量（預設：0.0001）

### 監控指標
- 監控指標：驗證集的 EMA pseudo Dice (ema_fg_dice)
- 改善定義：當前指標 - 最佳指標 > min_delta

### 使用建議
1. **快速實驗**: `--early_stopping_patience 20`
2. **標準訓練**: `--early_stopping_patience 50`
3. **穩定訓練**: `--early_stopping_patience 100`

### 範例
```bash
# 快速實驗（20 個 epoch 未改善就停止）
python run_training.py Dataset001_BrainTumor 3d_fullres 0 \
    --enable_early_stopping \
    --early_stopping_patience 20

# 較嚴格的改善標準
python run_training.py Dataset001_BrainTumor 3d_fullres 0 \
    --enable_early_stopping \
    --early_stopping_patience 50 \
    --early_stopping_min_delta 0.001
```

## 參數說明

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--initial_lr` | float | 1e-4 | 初始學習率 |
| `--oversample_foreground_percent` | float | 0.5 | 訓練時前景過採樣比例 (0.0-1.0) |
| `--oversample_foreground_percent_val` | float | 0.2 | 驗證時前景過採樣比例 (0.0-1.0) |
| `--num_iterations_per_epoch` | int | 500 | 每個 epoch 的迭代次數 |
| `--num_epochs` | int | 1000 | 訓練的總 epoch 數 |
| `--optimizer` | str | AdamW | 優化器類型 (SGD 或 AdamW) |
| `--lr_scheduler` | str | CosineAnnealingLR | 學習率調度器 (PolyLRScheduler 或 CosineAnnealingLR) |
| `--enable_early_stopping` | flag | False | 啟用提前停止機制 |
| `--early_stopping_patience` | int | 50 | 提前停止的耐心值（epoch 數） |
| `--early_stopping_min_delta` | float | 0.0001 | 判定為改善的最小閾值 |
## 建議配置

### 快速實驗
```bash
--num_epochs 100 --num_iterations_per_epoch 100 --optimizer AdamW --lr_scheduler CosineAnnealingLR
```

### 標準訓練
```bash
--num_epochs 1000 --num_iterations_per_epoch 500 --optimizer AdamW --lr_scheduler CosineAnnealingLR
```

### 長時間訓練
```bash
--num_epochs 2000 --num_iterations_per_epoch 500 --optimizer SGD --lr_scheduler PolyLRScheduler --initial_lr 0.01
```

### 小數據集
```bash
--num_epochs 500 --num_iterations_per_epoch 250 --oversample_foreground_percent 0.7 --optimizer AdamW
```

## 注意事項

1. **學習率選擇**：
   - SGD 建議使用較大的學習率 (如 0.01)
   - AdamW 建議使用較小的學習率 (如 1e-4)

2. **過採樣比例**：
   - 前景類別較少時，可以增加 oversample_foreground_percent
   - 驗證時建議使用較低的過採樣比例

3. **迭代次數**：
   - num_iterations_per_epoch 影響每個 epoch 的時間
   - 較少的迭代次數可以加快實驗速度

4. **Epoch 數量**：
   - 需要根據驗證集表現調整
   - 建議啟用 early stopping 機制以節省時間

5. **組合建議**：
   - AdamW + CosineAnnealingLR：適合大多數情況
   - SGD + PolyLRScheduler：適合需要穩定訓練的場景

6. **Early Stopping 使用**：
   - 建議在長時間訓練中啟用（如 num_epochs > 500）
   - patience 值可根據數據集大小調整：
     - 小數據集：20-30 epochs
     - 中等數據集：50 epochs
     - 大數據集：100+ epochs
   - min_delta 建議設為 0.0001 到 0.001 之間
   - 監控的指標是驗證集的 EMA pseudo Dice，較為穩定可靠

