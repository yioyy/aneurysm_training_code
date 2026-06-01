# 2026-05-13 — Training metrics CSV output

## What changed
- 加入 `_write_metrics_csv()` 到 `nnUNetTrainer.on_epoch_end`，每 epoch 直接寫入結構化 CSV
- 取代「事後解析 training_log.txt」的 workflow（過去 `aneurysm_analysis_pipeline/integrated_workflow/parse_training_logs.py` 在做的事）

## File
- `<output_folder>/training_metrics.csv`（與 `training_log.txt` 同目錄）

## Columns
基礎欄位（所有訓練都有）：
- `epoch, lr, train_loss, train_seg_loss, val_loss, val_seg_loss`
- `train_dice, val_dice`（取 dice_per_class_or_region 第一個前景類別 — 向後相容 log parser）
- `ema_val_loss, ema_val_dice, epoch_time_s, new_best_ema`

Per-class dice 欄位（class 數量由 dataset 決定，可變）：
- `train_dice_class_0, train_dice_class_1, ...`
- `val_dice_class_0, val_dice_class_1, ...`
- `ema_val_dice_class_0, ema_val_dice_class_1, ...`

Classifier head 變體額外有：
- `train_cls_loss, train_accuracy, train_sensitivity, train_specificity`
- `val_cls_loss, val_accuracy, val_sensitivity, val_specificity`

## 識別欄位（cohort/model/fold 等）
**不加**。多模型比較時改用 recipe diff（比 fixed columns 更彈性，支援任何消融參數差異）。

## Behavior
- **Resume / continue training**：相同 epoch 的 row 會被取代（讀回 → 過濾掉 epoch == current_epoch → append 新 row）。不會重複，不會錯位。
- **DDP**：只有 local_rank=0 寫入。
- **pandas 缺失**：自動 skip，不影響訓練。
- **non-fatal**：CSV 寫入錯誤只 log warning，不中斷訓練。

## new_best_ema 邏輯
- 在 best checkpoint 邏輯前後比對 `_best_ema` 與 `_best_ema_model_dice`
- 任一變化 → 該值記到 new_best_ema 欄位（後觸發者覆蓋，與 log parser RX_NEW_BEST_EMA 行為一致）
- 沒變化 → NaN

## Impact
- `aneurysm_analysis_pipeline/integrated_workflow/parse_training_logs.py` 不再需要解析 log；可改為直接 concat 多個 `training_metrics.csv`
- MUTP image 自此可直接產出可比較的訓練曲線資料
