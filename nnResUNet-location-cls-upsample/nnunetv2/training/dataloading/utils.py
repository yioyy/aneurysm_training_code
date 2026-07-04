import multiprocessing
import os
from multiprocessing import Pool
from typing import List, Dict, Optional, Literal

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes


def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False) -> None:
    try:
        a = np.load(npz_file)  # inexpensive, no compression is done here. This just reads metadata
        if overwrite_existing or not isfile(npz_file[:-3] + "npy"):
            np.save(npz_file[:-3] + "npy", a['data'])
        if unpack_segmentation and (overwrite_existing or not isfile(npz_file[:-4] + "_seg.npy")):
            np.save(npz_file[:-4] + "_seg.npy", a['seg'])
    except KeyboardInterrupt:
        if isfile(npz_file[:-3] + "npy"):
            os.remove(npz_file[:-3] + "npy")
        if isfile(npz_file[:-4] + "_seg.npy"):
            os.remove(npz_file[:-4] + "_seg.npy")
        raise KeyboardInterrupt


def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes):
    """
    Unpack all npz files in folder → npy.

    MUTP change: 新 preprocessor 直接寫 .npy 不寫 .npz，此時整個 folder 沒 .npz → early return，
    避免無謂啟動 multiprocessing pool。

    Excludes mutp multi-head inline aux files ({case}_aux_seg.npz): those carry only a 'seg' key
    (not 'data'), so unpacking them as main case files would raise KeyError. They are loaded
    on-demand from inside nnUNetDataset, not via unpack.
    """
    npz_files = subfiles(folder, True, None, ".npz", True)
    npz_files = [f for f in npz_files if not f.endswith("_aux_seg.npz")]
    if not npz_files:
        return  # 新格式：全部 .npy，無 .npz 可 unpack
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files))
                  )


def get_case_identifiers(folder: str) -> List[str]:
    """
    Finds case identifiers in the given folder.

    MUTP change: 掃 .npy (新 preprocessor 產出) 和 .npz (舊 preprocessor 產出) 兩種，去重。
    這樣新舊 preprocess 出的 dataset 都能被讀取。

    Excludes:
      - segFromPrevStage: original nnUNet cascaded-stage aux files
      - _seg.npy:         segmentation label files (not a case)
      - _aux_seg.npy/npz: mutp multi-head inline aux seg files
    """
    seen = set()
    for i in os.listdir(folder):
        if "segFromPrevStage" in i:
            continue
        if i.endswith("_seg.npy") or i.endswith("_seg.npz"):
            continue
        if i.endswith("_aux_seg.npy") or i.endswith("_aux_seg.npz"):
            continue
        if i.endswith(".npy") or i.endswith(".npz"):
            seen.add(i[:-4])
    return list(seen)


def build_sampling_probabilities(
    tr_keys: List[str],
    sampling_categories: Optional[Dict[str, int]] = None,
    category_weights: Optional[Dict[int, float]] = None,
    default_category: int = 1,
    default_weight: float = 1.0,
    mode: Literal["multiplier", "target_proportion"] = "multiplier",
) -> Optional[np.ndarray]:
    """
    依 sampling_categories（case_id -> 類別 1~4）與 category_weights（類別 -> 權重）
    建立每個 training case 的抽樣機率，總和為 1，順序與 tr_keys 一致。
    若 sampling_categories 或 category_weights 為 None，則回傳 None（不加權）。

    Args:
        tr_keys: 本 fold 的訓練 case 名稱列表（與 dataset_tr.keys() 順序一致）
        sampling_categories: case_id -> 類別 (1~4)，來自 splits_final.json 的 "sampling_categories"
        category_weights: 類別 -> 權重。
            - mode="multiplier": 權重會直接套用到該類別的每一個 case（類別內所有 case 權重相同）。
            - mode="target_proportion": 權重會被解讀為「目標類別抽樣比例」，函式會自動除以該 fold
              類別 case 數量，使得期望的類別抽樣比例更接近 category_weights 的比例。
        default_category: 若 case 不在 sampling_categories 中，使用的類別
        default_weight: 若類別不在 category_weights 中，使用的權重
        mode: "multiplier"（預設）或 "target_proportion"

    Returns:
        shape (len(tr_keys),)、sum=1 的機率陣列，或 None
    """
    if not sampling_categories or not category_weights:
        return None
    # map case -> category
    cats = [sampling_categories.get(k, default_category) for k in tr_keys]

    if mode == "multiplier":
        weights = np.array([category_weights.get(c, default_weight) for c in cats], dtype=np.float64)
    elif mode == "target_proportion":
        # category_weights is interpreted as desired category proportions (not per-case multipliers)
        # per-case weight = desired_prop[cat] / count_in_fold[cat]
        from collections import Counter
        counts = Counter(cats)
        weights = np.array([
            (category_weights.get(c, default_weight) / counts[c]) if counts.get(c, 0) > 0 else 0.0
            for c in cats
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'multiplier' or 'target_proportion'.")

    if weights.size == 0:
        return None
    s = weights.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    return weights / s


if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/2d')