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
    all npz files in this folder belong to the dataset, unpack them all
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files))
                  )


def get_case_identifiers(folder: str) -> List[str]:
    """
    finds all npz files in the given folder and reconstructs the training case names from them
    """
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


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