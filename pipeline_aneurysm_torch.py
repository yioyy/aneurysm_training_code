
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

動脈瘤檢測 Pipeline - 使用 nnU-Net 模型

@author: chuan
"""

# 導入必要的模組
import os
import time
import logging
import shutil
import json
import argparse
import subprocess
import pathlib
import warnings
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import pynvml  # GPU memory info

# 設置環境變數和警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


#會使用到的一些predict技巧
def data_translate(img, nii):
    img = np.swapaxes(img,0,1)
    img = np.flip(img,0)
    img = np.flip(img, -1)
    header = nii.header.copy() #抓出nii header 去算體積 
    pixdim = header['pixdim']  #可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)  
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img

def data_translate_back(img, nii):
    header = nii.header.copy() #抓出nii header 去算體積 
    pixdim = header['pixdim']  #可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)      
    img = np.flip(img, -1)
    img = np.flip(img,0)
    img = np.swapaxes(img,1,0)
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img

#case_json(json_path_name)     
def case_json(json_file_path, ID):
    json_dict = OrderedDict()
    json_dict["PatientID"] = ID #使用的程式是哪一支python api  

    with open(json_file_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示

def pipeline_aneurysm(ID, 
                      MRA_BRAIN_file,  
                      path_output,
                      path_code = '/mnt/e/pipeline/chuan/code/', 
                      path_gpu_aneurysm = '/data/4TB1/pipeline/chuan/code/gpu_aneurysm.py',
                      path_nnunet_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset080_DeepAneurysm/nnUNetTrainer__nnUNetPlans__3d_fullres',
                      path_processModel = '/mnt/e/pipeline/chuan/process/Deep_Aneurysm/', 
                      path_log = '/mnt/e/pipeline/chuan/log/', 
                      gpu_n = 0
                      ):

    #當使用gpu有錯時才確認
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    #以log紀錄資訊，先建置log
    localt = time.localtime(time.time()) # 取得 struct_time 格式的時間
    #以下加上時間註記，建立唯一性
    time_str_short = str(localt.tm_year) + str(localt.tm_mon).rjust(2,'0') + str(localt.tm_mday).rjust(2,'0')
    log_file = os.path.join(path_log, time_str_short + '.log')
    if not os.path.isfile(log_file):  #如果log檔不存在
        f = open(log_file, "a+") #a+	可讀可寫	建立，不覆蓋
        f.write("")        #寫入檔案，設定為空
        f.close()      #執行完結束

    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT)

    logging.info('!!! Pre_Aneurysm call.')

    path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)
    if not os.path.isdir(path_processID):  #如果資料夾不存在就建立
        os.mkdir(path_processID) #製作nii資料夾

    print(ID, ' Start...')
    logging.info(ID + ' Start...')

    #依照不同情境拆分try需要小心的事項 <= 重要
    try:
        # %% Deep learning相關
        pynvml.nvmlInit()  # 初始化
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)  # 获取GPU i的handle，后续通过handle来处理
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 通过handle获取GPU i的信息
        gpumRate = memoryInfo.used / memoryInfo.total
        # print('gpumRate:', gpumRate) #先設定gpu使用率小於0.2才跑predict code

        if gpumRate < 0.6:
            # 配置 GPU
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type='GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_n], True)

            #先判斷有無影像，複製過去
            #print('ADC_file:', ADC_file, ' copy:', os.path.join(path_nii, ID + '_ADC.nii.gz'))
            if not os.path.isfile(os.path.join(path_processID, 'MRA_BRAIN.nii.gz')):
                shutil.copy(MRA_BRAIN_file, os.path.join(path_processID, 'MRA_BRAIN.nii.gz'))
            
            #因為松諭會用排程，但因為ai跟mip都要用到gpu，所以還是要管gpu ram，#multiprocessing沒辦法釋放gpu，要改用subprocess.run()
            #model_predict_aneurysm(path_code, path_processID, ID, path_log, gpu_n)
            print("Running stage 1: Aneurysm inference!!!")
            logging.info("Running stage 1: Aneurysm inference!!!")

            # 定義要傳入的參數，建立指令
            cmd = [
                   "python", path_gpu_aneurysm,
                   "--path_code", path_code,
                   "--path_process", path_processID,
                   "--path_nnunet_model", path_nnunet_model,
                   "--case", ID,
                   "--path_log", path_log,
                   "--gpu_n", str(gpu_n)  # 注意要轉成字串
                  ]

            #result = subprocess.run(cmd, capture_output=True, text=True) 這會讓 subprocess.run() 自動幫你捕捉 stdout 和 stderr 的輸出，不然預設是印在 terminal 上，不會儲存。
            # 執行 subprocess
            start = time.time()
            subprocess.run(cmd)
            print(f"[Done AI Inference... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done AI Inference... ] spend {time.time() - start:.0f} sec")

            # 將結果輸出到 output 資料夾
            output_path = os.path.join(path_output, ID)
            os.makedirs(output_path, exist_ok=True)
            shutil.copy(os.path.join(path_processID, 'Pred.nii.gz'), os.path.join(output_path, 'Pred_Aneurysm.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Prob.nii.gz'), os.path.join(output_path, 'Prob_Aneurysm.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(output_path, 'Pred_Aneurysm_Vessel.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel_16.nii.gz'), os.path.join(output_path, 'Pred_Aneurysm_Vessel16.nii.gz'))

            print(f"[Done All Pipeline!!! ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done All Pipeline!!! ] spend {time.time() - start:.0f} sec")
            logging.info('!!! ' + ID +  ' post_aneurysm finish.')
        
        else:
            logging.error('!!! ' + str(ID) + ' Insufficient GPU Memory.')

    except Exception:
        logging.error('!!! ' + str(ID) + ' gpu have error code.')
        logging.error("Catch an exception.", exc_info=True)
   
    print('end!!!')
    return

#其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, default='17390820_20250604_MR_21406040004', help='目前執行的case的patient_id or study id')
    parser.add_argument('--Inputs', type=str, nargs='+', default=['example_input/17390820_20250604_MR_21406040004/MRA_BRAIN.nii.gz'], help='用於輸入的檔案（相對或絕對路徑）')
    parser.add_argument('--Output_folder', type=str, default='example_output/', help='用於輸出結果的資料夾（相對或絕對路徑）')    
    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    path_output = str(args.Output_folder)

    MRA_BRAIN_file = Inputs[0]
    
    # 設定相對路徑（相對於腳本所在目錄）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_code = script_dir
    path_gpu_aneurysm = os.path.join(script_dir, 'gpu_aneurysm.py')
    path_nnunet_model = os.path.join(script_dir, 'nnUNet_results', 'Dataset080_DeepAneurysm', 'nnUNetTrainer__nnUNetPlans__3d_fullres')
    
    # 建立 process、json、log 資料夾於腳本目錄
    path_process = os.path.join(script_dir, 'process')
    path_processModel = os.path.join(path_process, 'Deep_Aneurysm')
    path_log = os.path.join(script_dir, 'log')
    
    # GPU 設定
    gpu_n = 0

    # 建置必要資料夾
    os.makedirs(path_processModel, exist_ok=True)
    os.makedirs(path_log, exist_ok=True)
    os.makedirs(path_output, exist_ok=True)

    # 執行 pipeline（不需要 DICOM 相關參數）
    pipeline_aneurysm(
        ID=ID, 
        MRA_BRAIN_file=MRA_BRAIN_file, 
        path_output=path_output, 
        path_code=path_code,
        path_gpu_aneurysm=path_gpu_aneurysm,
        path_nnunet_model=path_nnunet_model, 
        path_processModel=path_processModel,
        path_log=path_log, 
        gpu_n=gpu_n
    )
    

    # #最後再讀取json檔結果
    # with open(json_path_name) as f:
    #     data = json.load(f)

    # logging.info('Json!!! ' + str(data))
