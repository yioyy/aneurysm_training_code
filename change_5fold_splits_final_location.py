# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 14:05:58 2023

依照excel的編號修改nnU-Net的splits_final.json去轉換成設定的5 fold cross val分類

@author: user
"""

import os
import time
import numpy as np
import pydicom
import glob
import shutil
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import sys
import logging
import cv2
import pandas as pd
#要安裝pillow 去開啟
from skimage import measure,color,morphology
import json
from collections import OrderedDict


path_preprocessed = 'C:/Users/user/Desktop/nnUNet/nnUNet_preprocessed/'
path_excel = r'D:\Aneurysm資料集重新整理\林君彥AI預測\第八輪\All_case_for_nnUNet_positive_aneurysm_vessel_cutAneurysm_cropVesselPatch_combineFEMH_vessel4\excel'

Task = 'DeepAneurysm'
No_num = '137'

kfold = 100 #總共做幾fold
json_file = os.path.join(path_preprocessed, 'Dataset' + No_num + '_' + Task, 'splits_final.json')


def case_json(json_file_path, channel_names, labels, numTraining, file_ending):
    json_dict = OrderedDict()
    #json_dict["name"] = Task #目前的專案是哪一個
    json_dict["channel_names"] = channel_names #影像種類  formerly modalities
    json_dict["labels"] = labels #標註種類 THIS IS DIFFERENT NOW!
    json_dict["numTraining"] = numTraining #訓練影像張數
    json_dict["file_ending"] = file_ending #檔案格式
    
    with open(json_file_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示


#讀取json以獲得資料集參數
#with open(json_file) as f:
#    json_data = json.load(f)

#讀取各個excel以做出5fold分類
#這邊fold0 ~ fold5都用同一個資料集
train_name = 'train_list_vessel4.xlsx'
test_name = 'test_list_vessel4.xlsx'
df_train = pd.read_excel(os.path.join(path_excel, train_name))
tr_id_lists = list(df_train['ID'])
tr_num_lists = list(df_train['No.'])

df_test = pd.read_excel(os.path.join(path_excel, test_name))
ts_id_lists = list(df_test['ID'])
ts_num_lists = list(df_test['No.'])

# 從 Excel 的 vessel4 欄位建立 sampling_categories（case_id -> 1~4），供訓練時抽樣比例 1:5:5:4 使用
sampling_categories = OrderedDict()
for idx, tr_num_list in enumerate(tr_num_lists):
    case_id = Task + '_' + str(tr_num_list).rjust(6, '0')
    sampling_categories[case_id] = int(df_train['vessel4'].iloc[idx])
for idx, ts_num_list in enumerate(ts_num_lists):
    case_id = Task + '_' + str(ts_num_list).rjust(6, '0')
    sampling_categories[case_id] = int(df_test['vessel4'].iloc[idx])

split_json = [] #用於做5fold cross val的數據分類

for i in range(kfold):
    #讀取每個fold的excel表，但這裡每個fold弄一樣就好
    #找出位置並予以編號
    kfold_dict = {"train":[],"val":[]}
    for idx, tr_num_list in enumerate(tr_num_lists):
        kfold_dict["train"].append(Task + '_' + str(tr_num_list).rjust(6,'0'))
    
    for idx, ts_num_list in enumerate(ts_num_lists):
        kfold_dict["val"].append(Task + '_' + str(ts_num_list).rjust(6,'0')) 
        
    split_json.append(kfold_dict)

# 輸出格式：包含 splits 與 sampling_categories，方便 Trainer 讀取並做加權抽樣
output = OrderedDict()
output["splits"] = split_json
output["sampling_categories"] = sampling_categories
    
#存檔寫出新的split_json（含 sampling_categories）
with open(json_file, 'w', encoding='utf8') as json_file:
    json.dump(output, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示