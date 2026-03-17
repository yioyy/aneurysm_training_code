# 依類別加權抽樣（4 分類 upsample）說明

此專案支援在 **case 層級** 進行加權抽樣：先為每個訓練 case 指定一個「抽樣類別」（例如血管 4 分類：1～4），再依照類別權重建立 `sampling_probabilities`，交由 dataloader 在每個 iteration 抽取 case。

## splits_final.json 格式（加入 sampling_categories）

要啟用依類別抽樣，需要在預處理資料夾（`nnUNet_preprocessed/<Dataset>/`）的 `splits_final.json` 內提供 `sampling_categories`。支援的新格式如下：

```json
{
  "splits": [
    {"train": ["Case_0001", "Case_0002"], "val": ["Case_0003"]},
    {"train": ["..."], "val": ["..."]}
  ],
  "sampling_categories": {
    "Case_0001": 1,
    "Case_0002": 2,
    "Case_0003": 4
  }
}
```

其中 `sampling_categories` 的 key 為 case id（需與 dataset keys 一致），value 為類別 id（常見為 1～4）。

## 權重的語意（重要）

類別權重由 trainer 的類別變數控制：

- `SAMPLING_CATEGORY_WEIGHTS`: 類別 -> 權重（例如 `{1: 2, 2: 1, 3: 1, 4: 1}`）
- `SAMPLING_CATEGORY_WEIGHT_MODE`: 權重解讀模式（影響最後抽樣比例）

目前提供兩種模式：

### 1) multiplier（每個 case 的乘數權重）

- 每個 case 的權重直接等於其類別權重。
- 因此某類別被抽到的期望比例 \(\propto\) `該類別 case 數量 × 該類別權重`。
- 例：類別 1 有 4 個 case、類別 2 有 2 個 case，權重設 1:1 時，最後仍會是 \(4:2=2:1\)（因為類別 1 本來 case 就多）。

### 2) target_proportion（目標類別抽樣比例）

- 權重被解讀成「**目標類別比例**」，函式會自動用該 fold 的類別 case 數量做校正。
- 直覺上就是：你設定 1:1，就會更接近真的抽到 1:1（前提是該類別在此 fold 內存在且樣本量足夠）。

## 會印出的「校正後期望類別比例」

trainer 在建立 `sampling_probabilities` 後，除了印出你設定的權重比例外，還會額外印出：

- **Corrected expected sampling proportions (by category, from probabilities)**

這一行是把 `sampling_probabilities` 依照 `sampling_categories` 分類後加總得到的結果（也就是每個類別的機率總和）。它代表「**實際抽樣機率所對應的期望類別抽樣比例**」，能直接反映：

- fold 內各類別 case 數量不均時的影響
- `SAMPLING_CATEGORY_WEIGHT_MODE` 不同時的最終差異

## 在 run_training.py 透過參數設定比例

你可以在啟動訓練時，直接用參數覆寫 trainer 的 `SAMPLING_CATEGORY_WEIGHTS`（以及可選的 mode）：

```bash
# 例：設定 4 類別權重為 2:1:1:1（對應類別 1~4）
python nnunetv2/run/run_training.py DatasetXXX_YYY 3d_fullres 0 --sampling_category_weights 2:1:1:1

# 例：顯式指定 mode（可選）
python nnunetv2/run/run_training.py DatasetXXX_YYY 3d_fullres 0 --sampling_category_weights 1:1:1:1 --sampling_category_weight_mode target_proportion
```

