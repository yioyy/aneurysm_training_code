# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:30:35 2024

保存出正規化完成的影像提供給nnUNet使用，順便也把血管mask複製出來

@author: user
"""

import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp


path_main = r'E:\RSNA2025_Aneurysm\MRA_Aneurysm_nnU-Net\All_case_with_psudo_labels_RSNA'
path_img_out = r'C:\Users\user\Desktop\nnUNet\nnResUNet-github\Normalized_Image'
path_vessel_out = r'C:\Users\user\Desktop\nnUNet\nnResUNet-github\Vessel'
path_excel = r'C:\Users\user\Desktop\nnUNet\nnResUNet-github'
json_file = r'C:\Users\user\Desktop\nnUNet\nnResUNet-github\nnUNet_results\Dataset080_DeepAneurysm\nnUNetTrainer__nnUNetPlans__3d_fullres\dataset.json'

if not os.path.exists(path_img_out):
    os.makedirs(path_img_out)
if not os.path.exists(path_vessel_out):
    os.makedirs(path_vessel_out)

#讀取json以獲得資料集參數   
with open(json_file) as f:
    json_data = json.load(f)

def case_json(json_file_path, channel_names, labels, numTraining, file_ending):
    json_dict = OrderedDict()
    #json_dict["name"] = Task #目前的專案是哪一個
    json_dict["channel_names"] = channel_names #影像種類  formerly modalities
    json_dict["labels"] = labels #標註種類 THIS IS DIFFERENT NOW!
    json_dict["numTraining"] = numTraining #訓練影像張數
    json_dict["file_ending"] = file_ending #檔案格式
    
    with open(json_file_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示
        
#會使用到的一些predict技巧 - 優化版本，減少記憶體複製
def data_translate(img, nii):
    # 使用 in-place 操作減少記憶體使用
    img = np.swapaxes(img, 0, 1)
    img = np.flip(img, 0)
    img = np.flip(img, -1)
    header = nii.header  # 不需要複製，只是讀取
    pixdim = header['pixdim']
    if pixdim[0] > 0:
        img = np.flip(img, 1)  
    return img


def data_translate_back(img, nii):
    header = nii.header  # 不需要複製，只是讀取
    pixdim = header['pixdim']
    if pixdim[0] > 0:
        img = np.flip(img, 1)      
    img = np.flip(img, -1)
    img = np.flip(img, 0)
    img = np.swapaxes(img, 1, 0)
    return img

#nii改變影像後儲存，data => nib.load的輸出，new_img => 更改的影像
def nii_img_replace(data, new_img):
    affine = data.affine
    header = data.header.copy()
    new_nii = nib.nifti1.Nifti1Image(new_img, affine, header=header)
    return new_nii


def custom_normalize_1(volume, mask=None, new_min=0., new_max=0.5, min_percentile=0.5, max_percentile=99.5, use_positive_only=True):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
    where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
    where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """
    # 檢查輸入維度
    if volume.ndim != 3:
        raise ValueError(f"Volume must be 3D, got {volume.ndim}D")
    
    if mask is not None and mask.ndim != 3:
        raise ValueError(f"Mask must be 3D, got {mask.ndim}D")
    
    if mask is not None and volume.shape != mask.shape:
        raise ValueError(f"Volume shape {volume.shape} and mask shape {mask.shape} must match")
    
    # select intensities
    new_volume = volume.copy()
    new_volume = new_volume.astype("float32")
    
    if mask is not None and use_positive_only:
        # 使用布林索引，確保結果是1維
        mask_positive = (mask > 0) & (new_volume > 0)
        intensities = new_volume[mask_positive].ravel()
    elif mask is not None:
        # 使用布林索引，確保結果是1維
        intensities = new_volume[mask > 0].ravel()
    elif use_positive_only:
        intensities = new_volume[new_volume > 0].ravel()
    else:
        intensities = new_volume.ravel()

    # 檢查是否有有效的強度值
    if len(intensities) == 0:
        print("Warning: No valid intensities found, returning zeros")
        return np.zeros_like(new_volume)

    # define min and max intensities in original image for normalisation
    robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

#     # trim values outside range
#     new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        new_volume = new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:  # avoid dividing by zero
        new_volume = np.zeros_like(new_volume)

#     # clip normalized values [0:1]
#     new_volume = np.clip(new_volume, 0, 1)
    return new_volume

def process_single_case(args):
    """處理單個案例的函數，用於多進程處理"""
    idx, ID, Rename, path_main, path_img_out, path_vessel_out, TASK = args
    
    try:
        print(f"[{idx}] {ID} Start...")
        
        # 載入原始影像
        img_nii = nib.load(os.path.join(path_main, ID, 'image.nii.gz'))
        img = np.array(img_nii.dataobj, dtype=np.float32)  # 直接指定dtype避免後續轉換
        img = data_translate(img, img_nii)

        # 載入brain mask
        brain_nii = nib.load(os.path.join(path_main, ID, 'brain_mask.nii.gz'))
        brain_array = np.array(brain_nii.dataobj, dtype=bool)  # 直接轉為bool
        brain_array = data_translate(brain_array, brain_nii)
        
        # 正規化
        #norm_img = custom_normalize_1(img, mask=brain_array)
        norm_img = custom_normalize_1(img, mask=brain_array, min_percentile=0.0, max_percentile=99.25)

        # 轉換回原始座標系統並儲存
        new_norm_img = data_translate_back(norm_img, img_nii)
        new_norm_img_nii = nii_img_replace(img_nii, new_norm_img)
        
        # 儲存正規化影像
        output_filename = TASK + '_' + str(Rename).rjust(5,'0') + '_0000.nii.gz'
        nib.save(new_norm_img_nii, os.path.join(path_img_out, output_filename))
        
        # 處理血管mask
        vessel_nii = nib.load(os.path.join(path_main, ID, 'pred_vessel.nii.gz'))
        vessel = np.array(vessel_nii.dataobj)
        
        new_vessel_nii = nii_img_replace(img_nii, vessel)
        nib.save(new_vessel_nii, os.path.join(path_vessel_out, output_filename))
        
        print(f"[{idx}] {ID} Completed!")
        return True, ID
        
    except Exception as e:
        print(f"[{idx}] {ID} Error: {str(e)}")
        return False, ID

#根據json去整理原先的資料集
TASK = json_data["TASK"]
No_num = json_data["No."]
channels = list(json_data["channel_names"].values()) #把所有影像種類展示出來並編號

text = [['ID', 'No.']]

#files = sorted(os.listdir(path_img_in))
#IDs = [y[:-17] for y in files]
#count = 1

#這邊來讀取excel，先讀取excel，取得patientid跟accession_number當成資料夾檔名條件
dtypes1 = {'ID': str, 'PatientID': str, 'StudyDate': str, 'AccessionNumber': str, 'SubjectLabel': str, 'ExperimentLabel': str, 
          'SeriesInstanceUID': str, 'url': str}

excel_name = 'Aneurysm_Pred_list.xlsx'

df = pd.read_excel(os.path.join(path_excel, excel_name), dtype=dtypes1).fillna('')
IDs = list(df['ID'])
Renames = list(df['No.'])


# 設定多線程參數
n_threads = mp.cpu_count() - 1  # 使用CPU核心數-1
print(f"使用 {n_threads} 個線程進行平行處理...")

# 準備參數列表
args_list = []
for idx, (ID, Rename) in enumerate(zip(IDs, Renames)):
    args_list.append((idx, ID, Rename, path_main, path_img_out, path_vessel_out, TASK))

# 記錄開始時間
start_time = time.time()
successful_cases = 0
failed_cases = []

# 使用多線程處理
if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # 提交所有任務
        future_to_args = {executor.submit(process_single_case, args): args for args in args_list}
        
        # 處理完成的任務
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            try:
                success, case_id = future.result()
                if success:
                    successful_cases += 1
                else:
                    failed_cases.append(case_id)
                    
                # 顯示進度
                completed = successful_cases + len(failed_cases)
                progress = (completed / len(args_list)) * 100
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time * len(args_list) / completed if completed > 0 else 0
                remaining_time = estimated_total - elapsed_time
                
                print(f"進度: {completed}/{len(args_list)} ({progress:.1f}%) - "
                      f"成功: {successful_cases}, 失敗: {len(failed_cases)} - "
                      f"預估剩餘時間: {remaining_time/60:.1f} 分鐘")
                      
            except Exception as exc:
                case_id = args[1]  # ID is at index 1
                failed_cases.append(case_id)
                print(f'案例 {case_id} 產生例外: {exc}')

    # 顯示最終結果
    total_time = time.time() - start_time
    print("\n處理完成!")
    print(f"總時間: {total_time/60:.1f} 分鐘")
    print(f"成功處理: {successful_cases} 個案例")
    print(f"失敗案例: {len(failed_cases)} 個")
    if failed_cases:
        print(f"失敗的案例ID: {failed_cases}")
else:
    # 如果不是主程式，使用單線程處理（用於調試）
    print("使用單線程模式...")
    for args in args_list:
        success, case_id = process_single_case(args)
        if success:
            successful_cases += 1
        else:
            failed_cases.append(case_id)

#text_df = pd.DataFrame(text[1:], columns=text[0]) #我们将 data[0] 作为 columns 参数传递给 pd.DataFrame，这样它会将该列表的值作为列名。
#text_df.to_excel(os.path.join(path_excel, 'FEMH.xlsx'), index = False)



