# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:44:17 2023

nnU-Net簡易版inference pipeline

@author: user
"""

import inspect
import multiprocessing
import os
import shutil
import traceback
from asyncio import sleep
from copy import deepcopy
from typing import Tuple, Union, List

import nnunetv2
import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json, save_pickle
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_prediction_from_softmax
#from nnunetv2.inference.sliding_window_prediction import predict_sliding_window_return_logits, compute_gaussian
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.file_path_utilities import get_output_folder, should_i_save_to_file, check_workers_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels, convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder

import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu, gaussian, threshold_otsu, frangi
from skimage.measure import label, regionprops, regionprops_table
import time

import warnings
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from scipy.ndimage import gaussian_filter
from torch import nn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
import torch.nn.functional as F

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import nibabel as nib

from copy import deepcopy
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from multiprocessing import Pool

#這邊是定義DataLoader
class PreprocessAdapter(DataLoader):
    def __init__(self, list_of_lists: List[List[str]], list_of_segs_from_prev_stage_files: Union[List[None], List[str]],
                 preprocessor: DefaultPreprocessor, output_filenames_truncated: List[str],
                 plans_manager: PlansManager, dataset_json: dict, configuration_manager: ConfigurationManager,
                 num_threads_in_multithreaded: int = 1):
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json = \
            preprocessor, plans_manager, configuration_manager, dataset_json

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        super().__init__(list(zip(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated)),
                         1, num_threads_in_multithreaded,
                         seed_for_shuffle=1, return_incomplete=True,
                         shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_lists)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        files = self._data[idx][0]
        seg_prev_stage = self._data[idx][1]
        ofile = self._data[idx][2]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg, data_properites = self.preprocessor.run_case(files, seg_prev_stage, self.plans_manager,
                                                                self.configuration_manager,
                                                                self.dataset_json)
        #if seg_prev_stage is not None:
        #    seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
        #    data = np.vstack((data, seg_onehot))

        if np.prod(data.shape) > (2e9 / 4 * 0.85):
            # we need to temporarily save the preprocessed image due to process-process communication restrictions
            np.save(ofile + '.npy', data)
            data = ofile + '.npy'

        return {'data': data, 'seg': seg, 'data_properites': data_properites, 'ofile': ofile}

#讀取需要的資訊
def load_what_we_need(model_training_output_dir, use_folds, checkpoint_name, plans_json_name='nnUNetPlans_5L-b900.json'):
    # we could also load plans and dataset_json from the init arguments in the checkpoint. Not quite sure what is the
    # best method so we leave things as they are for the moment.
    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    plans = load_json(join(model_training_output_dir, plans_json_name))
    #plans = load_json(join(model_training_output_dir, 'nnUNetPlans_classifier_64x-5L-b110.json'))
    
    plans_manager = PlansManager(plans)

    if isinstance(use_folds, str):
        use_folds = [use_folds]

    parameters = []
    for i, f in enumerate(use_folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                map_location=torch.device('cpu'), weights_only=False)
        if i == 0:
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
            inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])

    configuration_manager = plans_manager.get_configuration(configuration_name)
    # restore network
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                       num_input_channels, enable_deep_supervision=False)
    return parameters, configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json, network, trainer_name

#讀資料夾順序
def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
    print('use_folds is None, attempting to auto detect available folds')
    fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
    fold_folders = [i for i in fold_folders if i != 'fold_all']
    fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
    use_folds = [int(i.split('_')[-1]) for i in fold_folders]
    print(f'found the following folds: {use_folds}')
    return use_folds

#計算高斯map
def compute_gaussian(tile_size: Tuple[int, ...], sigma_scale: float = 1. / 8, dtype=np.float16) \
        -> np.ndarray:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

#計算 patch box
def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps

#做sliding_window生成器
def get_sliding_window_generator(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float,
                                 verbose: bool = False):
    if len(tile_size) < len(image_size):
        assert len(tile_size) == len(image_size) - 1, 'if tile_size has less entries than image_size, len(tile_size) ' \
                                                      'must be one shorter than len(image_size) (only dimension ' \
                                                      'discrepancy of 1 allowed).'
        steps = compute_steps_for_sliding_window(image_size[1:], tile_size, tile_step_size)
        if verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicer = tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), tile_size)]])
                    yield slicer
    else:
        steps = compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)
        if verbose: print(f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicer = tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), tile_size)]])
                    yield slicer

#是否要更複雜的inference(可選)
def maybe_mirror_and_predict(network: nn.Module, x: torch.Tensor, mirror_axes: Tuple[int, ...] = None, 
                            has_classifier_output: bool = False) \
        -> torch.Tensor:
    if has_classifier_output:
        prediction, cls = network(x)
    else:
        prediction = network(x)

    if mirror_axes is not None:
        # check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

        num_predictons = 2 ** len(mirror_axes)
        if 0 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2,))), (2,))
        if 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3,))), (3,))
        if 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (4,))), (4,))
        if 0 in mirror_axes and 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3))), (2, 3))
        if 0 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 4))), (2, 4))
        if 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3, 4))), (3, 4))
        if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
        prediction /= num_predictons
    return prediction

#sliding_window的pipeline，最需要改的地方
import torch.nn.functional as F

def predict_sliding_window_return_logits(network: nn.Module,
                                         input_image: Union[np.ndarray, torch.Tensor],
                                         vessel_image: Union[np.ndarray, torch.Tensor],
                                         num_segmentation_heads: int,
                                         tile_size: Tuple[int, ...],
                                         mirror_axes: Tuple[int, ...] = None,
                                         tile_step_size: float = 0.5,
                                         use_gaussian: bool = True,
                                         precomputed_gaussian: torch.Tensor = None,
                                         perform_everything_on_gpu: bool = True,
                                         verbose: bool = True,
                                         device: torch.device = torch.device('cuda'),
                                         batch_size: int = 1,
                                         has_classifier_output: bool = False) -> Union[np.ndarray, torch.Tensor]:
    if perform_everything_on_gpu:
        assert device.type == 'cuda', 'Can use perform_everything_on_gpu=True only when device="cuda"'

    network = network.to(device)
    network.eval()

    empty_cache(device)
    
    with torch.no_grad():
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
            assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if not torch.cuda.is_available():
                if perform_everything_on_gpu:
                    print('WARNING! "perform_everything_on_gpu" was True but cuda is not available! Set it to False...')
                perform_everything_on_gpu = False

            results_device = device if perform_everything_on_gpu else torch.device('cpu')

            if verbose: print("step_size:", tile_step_size)
            if verbose: print("mirror_axes:", mirror_axes)

            if not isinstance(input_image, torch.Tensor):
                # pytorch will warn about the numpy array not being writable. This doesnt matter though because we
                # just want to read it. Suppress the warning in order to not confuse users...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    input_image = torch.from_numpy(input_image)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, tile_size, 'constant', {'value': 0}, True, None)
            #data_vessel, slicer_revert_padding = pad_nd_image(vessel_image, data.shape, 'constant', {'value': 0}, True, None)
            
            # 計算需要補齊的大小:tensorA = torch.randn(1, 127, 512, 512)  # 假設是 tensorA
            pad_height = (data.shape[2] - vessel_image.shape[2])  # 需要補齊的高度 (上和下)
            pad_width = (data.shape[3] - vessel_image.shape[3])  # 需要補齊的寬度 (左和右)
            pad_depth = (data.shape[1] - vessel_image.shape[1])  # 需要補齊的深度 (前和後)

            # 計算每一維的補齊值
            # pad順序為 (左, 右, 上, 下, 前, 後)
            padding = (pad_width // 2, pad_width - pad_width // 2,  # 深度（前後）
                       pad_height // 2, pad_height - pad_height // 2,  # 高度（上下）
                       pad_depth, 0)  # 寬度（左右）
            
            # 使用 F.pad 進行補齊
            data_vessel = F.pad(vessel_image, padding, mode='constant', value=0)
                        
            print("step_size:", tile_step_size) #0.5 => 步距重疊率
            print("mirror_axes:", mirror_axes) #還是0
            print('data pad後的大小:', data.shape) #這邊還是512
            print('data_vessel pad後的大小:', data_vessel.shape) #這邊還是512

            if use_gaussian:
                gaussian = torch.from_numpy(
                    compute_gaussian(tile_size, sigma_scale=1. / 8)) if precomputed_gaussian is None else precomputed_gaussian
                gaussian = gaussian.half()
                # make sure nothing is rounded to zero or we get division by zero :-(
                mn = gaussian.min()
                if mn == 0:
                    gaussian.clip_(min=mn)
            else:
                # 不使用 gaussian 時，設置為 None 以節省記憶體
                gaussian = None
                    
            slicers = get_sliding_window_generator(data.shape[1:], tile_size, tile_step_size, verbose=verbose)

            # preallocate results and num_predictions. Move everything to the correct device
            try:
                predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                               device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                            device=results_device)
                if use_gaussian and gaussian is not None:
                    gaussian = gaussian.to(results_device)
            except RuntimeError:
                # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                results_device = torch.device('cpu')
                predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                               device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                            device=results_device)
                if use_gaussian and gaussian is not None:
                    gaussian = gaussian.to(results_device)
            finally:
                empty_cache(device)

            if use_gaussian:
                # 使用 gaussian 權重的情況：需要處理所有 patches
                # 收集所有需要處理的 patches 和對應的 slicers
                patches_to_process = []
                slicers_to_process = []
                
                for sl in slicers:
                    #只預測有血管的地方
                    if torch.sum(data_vessel[sl]) > 0:
                        workon = data[sl][None]
                        patches_to_process.append(workon)
                        slicers_to_process.append(sl)
                    else:
                        # 對於沒有血管的地方，直接設置為零
                        prediction = torch.zeros((num_segmentation_heads, *tile_size)).to(results_device)
                        predicted_logits[sl] += prediction * gaussian
                        n_predictions[sl[1:]] += gaussian
                
                # 批次處理有血管的 patches
                if len(patches_to_process) > 0:
                    if verbose:
                        print(f"[Gaussian模式] 處理 {len(patches_to_process)} 個有血管的 patches，使用 batch_size={batch_size}")
                    
                    for i in range(0, len(patches_to_process), batch_size):
                        batch_end = min(i + batch_size, len(patches_to_process))
                        batch_patches = patches_to_process[i:batch_end]
                        batch_slicers = slicers_to_process[i:batch_end]
                        
                        # 將 batch 中的 patches 組合成一個 tensor
                        batch_tensor = torch.cat(batch_patches, dim=0).to(device, non_blocking=False)
                        
                        # 批次預測
                        #start_time_batch = time.time()
                        batch_predictions = maybe_mirror_and_predict(network, batch_tensor, mirror_axes, has_classifier_output).to(results_device)
                        #print(f"[Done] maybe_mirror_and_predict no. {i} spend {time.time() - start_time_batch:.3f} sec")
                        
                        
                        # 處理每個預測結果
                        for j, (prediction, sl) in enumerate(zip(batch_predictions, batch_slicers)):
                            # prediction 這邊直接套用 softmax 去正規化輸出
                            prediction = torch.softmax(prediction, 0)
                            
                            # 使用高斯權重
                            predicted_logits[sl] += prediction * gaussian
                            n_predictions[sl[1:]] += gaussian
            else:
                # 不使用 gaussian 權重的情況：可以完全跳過沒有血管的區域，大幅加速
                patches_to_process = []
                slicers_to_process = []
                all_slicers = list(slicers)  # 先轉換成 list 以便重複使用
                
                # 只收集有血管的 patches，完全跳過空白區域
                for sl in all_slicers:
                    if torch.sum(data_vessel[sl]) > 0:
                        workon = data[sl][None]
                        patches_to_process.append(workon)
                        slicers_to_process.append(sl)
                
                if len(patches_to_process) > 0:
                    if verbose:
                        total_patches = len(all_slicers)
                        print(f"[非Gaussian模式] 跳過 {total_patches - len(patches_to_process)} 個空白 patches，只處理 {len(patches_to_process)} 個有血管的 patches，使用 batch_size={batch_size}")
                    
                    for i in range(0, len(patches_to_process), batch_size):
                        batch_end = min(i + batch_size, len(patches_to_process))
                        batch_patches = patches_to_process[i:batch_end]
                        batch_slicers = slicers_to_process[i:batch_end]
                        
                        # 將 batch 中的 patches 組合成一個 tensor
                        batch_tensor = torch.cat(batch_patches, dim=0).to(device, non_blocking=False)
                        
                        # 批次預測
                        start_time_batch = time.time()
                        batch_predictions = maybe_mirror_and_predict(network, batch_tensor, mirror_axes, has_classifier_output).to(results_device)
                        print(f"[Done] maybe_mirror_and_predict no. {i} spend {time.time() - start_time_batch:.3f} sec")
                        
                        # 處理每個預測結果
                        for j, (prediction, sl) in enumerate(zip(batch_predictions, batch_slicers)):
                            # prediction 這邊直接套用 softmax 去正規化輸出
                            prediction = torch.softmax(prediction, 0)
                            
                            # 不使用高斯權重，直接累加
                            predicted_logits[sl] += prediction
                            n_predictions[sl[1:]] += 1
                
                # 對於完全沒有血管的區域，設置一個預設的背景預測
                # 這樣可以避免這些區域保持未初始化狀態
                if len(patches_to_process) < len(all_slicers):
                    # 創建背景預測：[1.0, 0.0] 表示背景類別的機率為 1
                    background_prediction = torch.zeros((num_segmentation_heads, *tile_size), device=results_device)
                    background_prediction[0] = 1.0  # 背景類別設為 1
                    
                    for sl in all_slicers:
                        if torch.sum(data_vessel[sl]) == 0:  # 沒有血管的區域
                            predicted_logits[sl] += background_prediction
                            n_predictions[sl[1:]] += 1

            # 安全除法，避免除以零產生 NaN
            # 對於 n_predictions 為 0 的位置，保持 predicted_logits 為 0
            mask = n_predictions > 0
            predicted_logits = torch.where(mask.unsqueeze(0), 
                                         predicted_logits / n_predictions.unsqueeze(0), 
                                         predicted_logits)
            
            if verbose:
                zero_predictions = torch.sum(n_predictions == 0).item()
                total_voxels = torch.numel(n_predictions)
                if zero_predictions > 0:
                    print(f"警告：有 {zero_predictions}/{total_voxels} 個體素沒有被任何 patch 覆蓋到")
                    print(f"這些位置將保持為零值（通常是影像邊緣或完全沒有血管的區域）")
            #print('predicted_logits.shape:', predicted_logits.shape, ' data_vessel.shape:', data_vessel.shape)
            #predicted_logits.shape: torch.Size([2, 127, 512, 512])  data_vessel.shape: torch.Size([1, 127, 512, 512])
            
            #只與vessel相乘
            #print('predicted_logits.shape:', predicted_logits.shape, ' data_vessel.shape:', data_vessel.shape)
            #predicted_logits = predicted_logits[1, :, :, :] * data_vessel.to(results_device)
            #print('predicted_logits.shape:', predicted_logits.shape)
            # 使用 unsqueeze 增加一個維度，放在最前面
            #predicted_logits = predicted_logits[0, :, :, :].unsqueeze(0)
            #predicted_logits = data_vessel.to(results_device)
                        
            #與vessel相乘
            repeat_vessel = data_vessel.repeat(2, 1, 1, 1)
            predicted_logits = predicted_logits * repeat_vessel.to(results_device)
            #predicted_logits = repeat_vessel.to(results_device)            

    empty_cache(device)
    return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]

#存出probabilities map
def write_probabilities(seg, output_fname, img_nii):
    # revert transpose
    seg = seg.transpose((2, 1, 0)).astype(np.float32)
    
    affine = img_nii.affine
    header = img_nii.header.copy()
    new_nii = nib.nifti1.Nifti1Image(seg, affine, header=header)
    
    nib.save(new_nii, output_fname) 
    #nibabel.save(seg_nib, output_fname)

#輸出結果，最需要改的地方2
def export_prediction_probabilities(predicted_array_or_file: Union[np.ndarray, str], properties_dict: dict,
                                    vessel_image, img_nii,
                                    configuration_manager: ConfigurationManager,
                                    plans_manager: PlansManager,
                                    dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                    save_probabilities: bool = False):
    
    if isinstance(predicted_array_or_file, str):
        tmp = deepcopy(predicted_array_or_file)
        if predicted_array_or_file.endswith('.npy'):
            predicted_array_or_file = np.load(predicted_array_or_file)
        elif predicted_array_or_file.endswith('.npz'):
            predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
        os.remove(tmp)

    predicted_array_or_file = predicted_array_or_file.astype(np.float32)
    print('before')
    print('predicted_array_or_file.shape:', predicted_array_or_file.shape)
    print('np.max(predicted_array_or_file):', np.max(predicted_array_or_file))
    print('np.median(predicted_array_or_file):', np.median(predicted_array_or_file))

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    
    print('properties_dict[shape_after_cropping_and_before_resampling]:', properties_dict['shape_after_cropping_and_before_resampling'])
    print('current_spacing:', current_spacing)
    print('properties_dict[spacing]:', properties_dict['spacing'])
    
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted_array_or_file,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            properties_dict['spacing'])
    
    print('after')
    print('predicted_array_or_file.shape:', predicted_array_or_file.shape)
    print('np.max(predicted_array_or_file):', np.max(predicted_array_or_file))
    print('np.median(predicted_array_or_file):', np.median(predicted_array_or_file))    
    
    
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    
    """
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)

    # put result in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'], dtype=np.uint8)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation
    print('segmentation_reverted_cropping.shape:', segmentation_reverted_cropping.shape)

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    """
    
    # save
    # probabilities are already resampled

    # apply nonlinearity
#     predicted_array_or_file = label_manager.apply_inference_nonlin(predicted_array_or_file)

#     print('apply nonlinearity')
#     print('predicted_array_or_file.shape:', predicted_array_or_file.shape)
#     print('np.max(predicted_array_or_file):', np.max(predicted_array_or_file))
#     print('np.median(predicted_array_or_file):', np.median(predicted_array_or_file))
    
    # revert cropping
    probs_reverted_cropping = label_manager.revert_cropping(predicted_array_or_file,
                                                            properties_dict['bbox_used_for_cropping'],
                                                            properties_dict['shape_before_cropping'])
    
    print('revert cropping')
    print('probs_reverted_cropping.shape:', probs_reverted_cropping.shape)
    print('np.max(probs_reverted_cropping):', np.max(probs_reverted_cropping))
    print('np.median(probs_reverted_cropping):', np.median(probs_reverted_cropping))
    
    probs_reverted_cropping = np.expand_dims(probs_reverted_cropping[1,:,:,:], axis=0)
    
    if probs_reverted_cropping is None:
        raise ValueError("Reverting cropping failed, 'probs_reverted_cropping' is None.")
        
    # $revert transpose
    probs_reverted_cropping = probs_reverted_cropping.transpose([0] + [i + 1 for i in
                                                                plans_manager.transpose_backward])
    
#     print('revert transpose')
#     print('probs_reverted_cropping.shape:', probs_reverted_cropping.shape)
#     print('np.max(probs_reverted_cropping):', np.max(probs_reverted_cropping))
#     print('np.median(probs_reverted_cropping):', np.median(probs_reverted_cropping))
    #np.savez_compressed(output_file_truncated + '.npz', probabilities=probs_reverted_cropping)
    #save_pickle(properties_dict, output_file_truncated + '.pkl')
    #del probs_reverted_cropping
    #del predicted_array_or_file

#     rw = plans_manager.image_reader_writer_class()
#     rw.write_seg(probs_reverted_cropping[0,:,:,:], output_file_truncated + dataset_json_dict_or_file['file_ending'],
#                            properties_dict)
    
    #print('properties_dict:', properties_dict)
    #這邊用額外的自寫輸出成nifti方式好惹
    write_probabilities(probs_reverted_cropping[0,:,:,:], output_file_truncated + dataset_json_dict_or_file['file_ending'], img_nii)

#從raw data開始處理的pipeline
def predict_from_raw_data(list_of_lists_or_source_folder: Union[str, List[List[str]]],
                          Mask_list_of_lists_or_Mask_folder: Union[str, List[List[str]]],
                          output_folder: str,
                          model_training_output_dir: str,
                          use_folds: Union[Tuple[int, ...], str] = None,
                          tile_step_size: float = 0.5,
                          use_gaussian: bool = True,
                          use_mirroring: bool = True,
                          perform_everything_on_gpu: bool = True,
                          verbose: bool = True,
                          save_probabilities: bool = False,
                          overwrite: bool = True,
                          checkpoint_name: str = 'checkpoint_final.pth',
                          plans_json_name: str = 'nnUNetPlans_5L-b900.json',
                          has_classifier_output: bool = False,
                          num_processes_preprocessing: int = default_num_processes,
                          num_processes_segmentation_export: int = default_num_processes,
                          folder_with_segs_from_prev_stage: str = None,
                          num_parts: int = 1,
                          part_id: int = 0,
                          desired_gpu_index : int = 0,
                          device: torch.device = torch.device('cuda'),
                          batch_size: int = 1):
    print("\n#######################################################################\nPlease cite the following paper "
          "when using nnU-Net:\n"
          "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
          "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
          "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    # 假設你想要在某個特定 GPU 上執行（例如GPU 1，編號從0開始）
    #desired_gpu_index = 0  # 修改此處來指定你希望使用的 GPU 編號

    # 檢查是否為 CUDA 設備，並指定 GPU 編號
    if device.type == 'cuda':
        device = torch.device(type='cuda', index=desired_gpu_index)  # 根據 desired_gpu_index 設定具體的 GPU

    if device.type != 'cuda':
        perform_everything_on_gpu = False

    # let's store the input arguments so that its clear what was used to generate the prediction
    my_init_kwargs = {}
    for k in inspect.signature(predict_from_raw_data).parameters.keys():
        my_init_kwargs[k] = locals()[k]
    my_init_kwargs = deepcopy(my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
    # safety precaution.
    recursive_fix_for_json_export(my_init_kwargs)
    maybe_mkdir_p(output_folder)
    save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

    if use_folds is None:
        use_folds = auto_detect_available_folds(model_training_output_dir, checkpoint_name)

    # load all the stuff we need from the model_training_output_dir
    # 這邊獲得都是模型的參數
    parameters, configuration_manager, inference_allowed_mirroring_axes, \
    plans_manager, dataset_json, network, trainer_name = \
        load_what_we_need(model_training_output_dir, use_folds, checkpoint_name, plans_json_name)
    
    print('總共有幾個網路parameters(同時拿幾個網路預測):', len(parameters)) #用來得知網路參數有幾個
    
    #這邊先不用到
    """
    # check if we need a prediction from the previous stage
    if configuration_manager.previous_stage_name is not None:
        if folder_with_segs_from_prev_stage is None:
            print(f'WARNING: The requested configuration is a cascaded model and requires predctions from the '
                  f'previous stage! folder_with_segs_from_prev_stage was not provided. Trying to run the '
                  f'inference of the previous stage...')
            folder_with_segs_from_prev_stage = join(output_folder,
                                                    f'prediction_{configuration_manager.previous_stage_name}')
            predict_from_raw_data(list_of_lists_or_source_folder,
                                  folder_with_segs_from_prev_stage,
                                  get_output_folder(plans_manager.dataset_name,
                                                    trainer_name,
                                                    plans_manager.plans_name,
                                                    configuration_manager.previous_stage_name),
                                  use_folds, tile_step_size, use_gaussian, use_mirroring, perform_everything_on_gpu,
                                  verbose, False, overwrite, checkpoint_name,
                                  num_processes_preprocessing, num_processes_segmentation_export, None,
                                  num_parts=num_parts, part_id=part_id, device=device)
    """

    # sort out input and output filenames
    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                   dataset_json['file_ending'])
        Mask_list_of_lists_or_Mask_folder = create_lists_from_splitted_dataset_folder(Mask_list_of_lists_or_Mask_folder,
                                                                                   dataset_json['file_ending'])
    print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
    list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
    caseids = [os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for i in list_of_lists_or_source_folder]
    print(f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
    print(f'There are {len(caseids)} cases that I would like to predict')
    print('list_of_lists_or_source_folder example:', list_of_lists_or_source_folder[0])
    print('Mask_list_of_lists_or_Mask_folder:', Mask_list_of_lists_or_Mask_folder[0])

    output_filename_truncated = [join(output_folder, i) for i in caseids]
    seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + dataset_json['file_ending']) if
                                 folder_with_segs_from_prev_stage is not None else None for i in caseids]
    # remove already predicted files form the lists
    if not overwrite:
        tmp = [isfile(i + dataset_json['file_ending']) for i in output_filename_truncated]
        not_existing_indices = [i for i, j in enumerate(tmp) if not j]

        output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
        list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
        seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
        print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
              f'That\'s {len(not_existing_indices)} cases.')
        # caseids = [caseids[i] for i in not_existing_indices]

    # placing this into a separate function doesnt make sense because it needs so many input variables...
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    # hijack batchgenerators, yo
    # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
    # way we don't have to reinvent the wheel here.
    num_processes = max(1, min(num_processes_preprocessing, len(list_of_lists_or_source_folder)))
    #print('seg_from_prev_stage_files:', seg_from_prev_stage_files) #這邊原本都是None
    
    ppa = PreprocessAdapter(list_of_lists_or_source_folder, Mask_list_of_lists_or_Mask_folder, preprocessor,
                            output_filename_truncated, plans_manager, dataset_json,
                            configuration_manager, num_processes)
    mta = MultiThreadedAugmenter(ppa, NumpyToTensor(), num_processes, 1, None, pin_memory=device.type == 'cuda')
    
    # precompute gaussian
    inference_gaussian = torch.from_numpy(
        compute_gaussian(configuration_manager.patch_size)).half()
    if perform_everything_on_gpu:
        inference_gaussian = inference_gaussian.to(device)
    print('inference_gaussian.shape:', inference_gaussian.shape)

    # num seg heads is needed because we need to preallocate the results in predict_sliding_window_return_logits
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_seg_heads = label_manager.num_segmentation_heads
    #num_seg_heads 這邊為 0背景 1.動脈瘤，所以為2
    #print('num_seg_heads:', num_seg_heads)

    # go go go
    # spawn allows the use of GPU in the background process in case somebody wants to do this. Not recommended. Trust me.
    # export_pool = multiprocessing.get_context('spawn').Pool(num_processes_segmentation_export)
    # export_pool = multiprocessing.Pool(num_processes_segmentation_export)
    with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
        network = network.to(device)

        r = []
        with torch.no_grad():
            for preprocessed, nii_path in zip(mta, list_of_lists_or_source_folder):
                start_time = time.time()
                data = preprocessed['data']
                data_vessel = preprocessed['seg']
                #print('data:', data.shape, 'data_vessel:', data_vessel.shape)
                #讀取nifti只是為了affine
                img_nii = nib.load(str(nii_path[0]))
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)
                
                if isinstance(data_vessel, str):
                    data_vessel = torch.from_numpy(np.load(data_vessel))

                ofile = preprocessed['ofile']
                print(f'\nPredicting {os.path.basename(ofile)}:')
                print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')
                print('configuration_manager.patch_size:', configuration_manager.patch_size)
                
                properties = preprocessed['data_properites'] #組回nifti的參數

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_busy(export_pool, r, allowed_num_queued=len(export_pool._pool))
                while not proceed:
                    sleep(1)
                    proceed = not check_workers_busy(export_pool, r, allowed_num_queued=len(export_pool._pool))

                # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
                # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
                # things a lot faster for some datasets.
                prediction = None
                overwrite_perform_everything_on_gpu = perform_everything_on_gpu
                #目前是走perform_everything_on_gpu = 1
                if perform_everything_on_gpu:
                    try:
                        for params in parameters:
                            network.load_state_dict(params)
                            if prediction is None:
                                prediction = predict_sliding_window_return_logits(
                            network, data, data_vessel, num_seg_heads,
                            configuration_manager.patch_size,
                            mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                            tile_step_size=tile_step_size,
                            use_gaussian=use_gaussian,
                            precomputed_gaussian=inference_gaussian,
                            perform_everything_on_gpu=perform_everything_on_gpu,
                            verbose=verbose,
                            device=device,
                            batch_size=batch_size,
                            has_classifier_output=has_classifier_output)
                            else:
                                prediction += predict_sliding_window_return_logits(
                                    network, data, data_vessel, num_seg_heads,
                                    configuration_manager.patch_size,
                                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                    tile_step_size=tile_step_size,
                                    use_gaussian=use_gaussian,
                                    precomputed_gaussian=inference_gaussian,
                                    perform_everything_on_gpu=perform_everything_on_gpu,
                                    verbose=verbose,
                                    device=device,
                                    batch_size=batch_size,
                                    has_classifier_output=has_classifier_output)
                            if len(parameters) > 1:
                                prediction /= len(parameters)

                    except RuntimeError:
                        print('Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                              'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
                        print('Error:')
                        traceback.print_exc()
                        prediction = None
                        overwrite_perform_everything_on_gpu = False

                #如果gpu失敗，走以下
                if prediction is None:
                    for params in parameters:
                        network.load_state_dict(params)
                        if prediction is None:
                            prediction = predict_sliding_window_return_logits(
                                network, data, data_vessel, num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=perform_everything_on_gpu,
                                verbose=verbose,
                                device=device,
                                batch_size=batch_size,
                                has_classifier_output=has_classifier_output)
                        else:
                            prediction += predict_sliding_window_return_logits(
                                network, data, data_vessel, num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=perform_everything_on_gpu,
                                verbose=verbose,
                                device=device,
                                batch_size=batch_size,
                                has_classifier_output=has_classifier_output)
                        if len(parameters) > 1:
                            prediction /= len(parameters)

                print('Prediction done, transferring to CPU if needed')
                prediction = prediction.to('cpu').numpy()
                
                #print('final prediction.shape:', prediction.shape)
                if should_i_save_to_file(prediction, r, export_pool):
                    print(
                        'output is either too large for python process-process communication or all export workers are '
                        'busy. Saving temporarily to file...')
                    np.save(ofile + '.npy', prediction)
                    prediction = ofile + '.npy'

                """
                # this needs to go into background processes
                # export_prediction(prediction, properties, configuration_name, plans, dataset_json, ofile,
                #                   save_probabilities)
                print('sending off prediction to background worker for resampling and export')
                r.append(
                    export_pool.starmap_async(
                        export_prediction_probabilities, ((prediction, properties, configuration_manager, plans_manager,
                                                          dataset_json, ofile, save_probabilities),)
                    )
                )
                print(f'done with {os.path.basename(ofile)}')
                """
                print(f"[Done] spend {time.time() - start_time:.2f} sec")
                export_prediction_probabilities(prediction, properties, data_vessel, img_nii, configuration_manager, plans_manager,
                                                dataset_json, ofile, save_probabilities)
                
                print(f"[Done] spend {time.time() - start_time:.2f} sec")
        #[i.get() for i in r]


#主程式
if __name__ == "__main__":
    from multiprocessing import Pool
    predict_from_raw_data('/data/chuan/nnUNet/nnUNet_raw/Dataset063_DeepAneurysm/Normalized_Image_External_Test1/',
                          '/data/chuan/nnUNet/nnUNet_raw/Dataset063_DeepAneurysm/Vessel_External_Test1/',
                          '/data/chuan/nnUNet/nnUNet_inference/Dataset077_DeepAneurysm/3d_fullres/nnResUNet-long-BigBatch-cosine-AneDilate-4L-ft-9to1/NewTest_gaussian1',
                          '/data/chuan/nnUNet/nnUNet_results/Dataset077_DeepAneurysm/nnUNetTrainer__nnUNetPlans__3d_fullres',
                          (15,),
                          0.25,
                          use_gaussian=True,  # 關閉 gaussian 以獲得最大速度
                          use_mirroring=False,
                          perform_everything_on_gpu=True,
                          verbose=True,
                          save_probabilities=False,
                          overwrite=False,
                          checkpoint_name='checkpoint_best.pth',
                          plans_json_name='nnUNetPlans_5L-b900.json',  # 可以根據需要修改json檔名
                          has_classifier_output=False,  # 如果模型有classifier輸出則設為True
                          num_processes_preprocessing=2,
                          num_processes_segmentation_export=3,
                          desired_gpu_index = 0,
                          batch_size=112  # 可以使用更大的 batch_size，因為跳過了很多空白區域
                          )