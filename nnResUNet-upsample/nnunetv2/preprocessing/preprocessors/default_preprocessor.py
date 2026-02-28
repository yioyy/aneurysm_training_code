#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import shutil
from typing import Union, Tuple

import nnunetv2
import numpy as np
from acvl_utils.miscellaneous.ptqdm import ptqdm
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    create_lists_from_splitted_dataset_folder
import random

class DefaultPreprocessor(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str],
                 vessel_file: Union[str, None] = None,
                 dilate_file: Union[str, None] = None):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        vessel用跟seg一樣的處理，只差最後儲存出所有座標放到properites裡

        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properites = rw.read_images(image_files)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        # if possible, load vessel
        if vessel_file is not None:
            vessel, _ = rw.read_seg(vessel_file)
        else:
            vessel = None

        # if possible, load vessel
        if dilate_file is not None:
            dilate, _ = rw.read_seg(dilate_file)
        else:
            dilate = None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if vessel is not None:
            vessel = vessel.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if dilate is not None:
            dilate = dilate.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])

        original_spacing = [data_properites['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        data_properites['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg) #是基於data切，所以不會錯
        data1, vessel, bbox1 = crop_to_nonzero(data, vessel)
        data2, dilate, bbox2 = crop_to_nonzero(data, dilate)

        data_properites['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        data_properites['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 3d we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        vessel = configuration_manager.resampling_fn_seg(vessel, new_shape, original_spacing, target_spacing)
        dilate = configuration_manager.resampling_fn_seg(dilate, new_shape, original_spacing, target_spacing)
       
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if seg_file is not None:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            data_properites['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)

        #最後，用一個欄位紀錄出vessel位置
        if vessel_file is not None:
            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            seed = random.randint(0, 2**32 - 1)  # 生成一個隨機的 seed
            data_properites['vessel_locations'] = self._sample_locations(vessel, seed = seed, verbose=self.verbose)

        #最後，用一個欄位紀錄出動脈瘤dilate位置
        if dilate_file is not None:
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            data_properites['dilate_locations'] = self._sample_foreground_locations(dilate, collect_for_this,
                                                                                   verbose=self.verbose)

        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)

        return data, seg, data_properites

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      vessel_file: str, dilate_file: str, plans_manager: PlansManager, 
                      configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        data, seg, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, 
                                               dataset_json, vessel_file=vessel_file, dilate_file=dilate_file)
        # print('dtypes', data.dtype, seg.dtype)
        np.savez_compressed(output_filename_truncated + '.npz',  data=data.astype('float16'), seg=seg) #以float16保存
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        #num_samples = 10000
        #min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            #target_num_samples = min(num_samples, len(all_locs))
            #target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            #動脈瘤太小拉，可以全採
            selected = all_locs[rndst.choice(len(all_locs),  len(all_locs), replace=False)]
            class_locs[k] = selected
            #if verbose:
            #    print(c, target_num_samples)
        return class_locs
    
    @staticmethod
    def _sample_locations(seg: np.ndarray, seed: int = 1234, verbose: bool = False):
        """
        vessel 取樣座標。

        - **二元/單類**（所有 >0 都視為同一類）: 回傳所有前景 voxel 座標（隨機打散）
        - **多類別**（值=1..n）: 針對每個 label 分別取出座標，並把每類補齊到「最大類」的 voxel 數，
          使後續隨機抽樣時各類別機率趨近 1:1:...（允許重複座標）
        """

        def _repeat_to_target(locs: np.ndarray, target: int, rnd: np.random.RandomState) -> np.ndarray:
            n = int(len(locs))
            if n == 0 or target <= 0:
                return np.empty((0, locs.shape[1]), dtype=locs.dtype)

            full_repeats = target // n
            remainder = target % n
            chunks = []

            # 完整取多輪（每輪都是不重複的 permutation）
            for _ in range(full_repeats):
                chunks.append(locs[rnd.permutation(n)])

            # 最後一輪截斷補齊差額
            if remainder > 0:
                perm = rnd.permutation(n)
                chunks.append(locs[perm[:remainder]])

            return np.concatenate(chunks, axis=0) if len(chunks) > 0 else np.empty((0, locs.shape[1]), dtype=locs.dtype)

        rndst = np.random.RandomState(seed)

        if seg is None:
            return np.empty((0, 0), dtype=np.int64)

        seg_arr = np.asarray(seg)
        fg_mask = seg_arr > 0
        if not np.any(fg_mask):
            return np.empty((0, seg_arr.ndim), dtype=np.int64)

        # 只看前景 label（忽略 0）
        labels = np.unique(seg_arr[fg_mask])
        # 安全起見轉成 int，避免 float label（resample 後理論上是 int）
        labels = np.array([int(i) for i in labels.tolist()], dtype=np.int64)
        labels = labels[labels > 0]

        # 若只有一個 label，行為等同「取全部前景」但仍保持隨機打散
        if len(labels) <= 1:
            all_locs = np.argwhere(fg_mask)
            return all_locs[rndst.permutation(len(all_locs))]

        # 多類別：每類補齊到最大類 voxel 數
        per_label_locs = []
        counts = []
        for lb in labels:
            locs = np.argwhere(seg_arr == lb)
            per_label_locs.append(locs)
            counts.append(int(len(locs)))

        max_count = int(max(counts)) if len(counts) > 0 else 0
        if max_count == 0:
            return np.empty((0, seg_arr.ndim), dtype=np.int64)

        if verbose:
            msg = ", ".join([f"{int(lb)}:{c}" for lb, c in zip(labels.tolist(), counts)])
            print(f"[vessel sampling] labels/counts = {msg}; target_per_label = {max_count}")

        balanced = [_repeat_to_target(locs, max_count, rndst) for locs in per_label_locs]
        selected = np.concatenate(balanced, axis=0)

        # 打散各類別的座標，避免順序偏差（即使後續通常是隨機抽樣）
        if len(selected) > 1:
            selected = selected[rndst.permutation(len(selected))]
        return selected

    def _normalize(self, data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'nnunetv2.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError('Unable to locate class \'%s\' for normalization' % scheme)
            normalizer = normalizer_class(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                                          intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], seg[0])
        return data

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        """
        data identifier = configuration name in plans. EZ.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
        if self.verbose:
            print(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        identifiers = get_identifiers_from_splitted_dataset_folder(join(nnUNet_raw, dataset_name, 'imagesTr'),
                                                               dataset_json['file_ending'])
        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        output_filenames_truncated = [join(output_directory, i) for i in identifiers]

        file_ending = dataset_json['file_ending']
        # list of lists with image filenames
        image_fnames = create_lists_from_splitted_dataset_folder(join(nnUNet_raw, dataset_name, 'imagesTr'), file_ending,
                                                                 identifiers)
        # list of segmentation filenames
        seg_fnames = [join(nnUNet_raw, dataset_name, 'labelsTr', i + file_ending) for i in identifiers]
        vessel_fnames = [join(nnUNet_raw, dataset_name, 'vesselsTr', i + file_ending) for i in identifiers]
        dilation_fnames = [join(nnUNet_raw, dataset_name, 'dilationsTr', i + file_ending) for i in identifiers]

        #print('go go go dilation !!!')
        #print('vessel_fnames:', vessel_fnames)

        _ = ptqdm(self.run_case_save, (output_filenames_truncated, image_fnames, seg_fnames, vessel_fnames, dilation_fnames),
                  processes=num_processes, zipped=True, plans_manager=plans_manager,
                  configuration_manager=configuration_manager,
                  dataset_json=dataset_json, disable=self.verbose)

    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        # this function will be called at the end of self.run_case. Can be used to change the segmentation
        # after resampling. Useful for experimenting with sparse annotations: I can introduce sparsity after resampling
        # and don't have to create a new dataset each time I modify my experiments
        return seg


def example_test_case_preprocessing():
    # (paths to files may need adaptations)
    plans_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/nnUNetPlans.json'
    dataset_json_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/dataset.json'
    input_images = ['/home/isensee/drives/e132-rohdaten/nnUNetv2/Dataset219_AMOS2022_postChallenge_task2/imagesTr/amos_0600_0000.nii.gz', ]  # if you only have one channel, you still need a list: ['case000_0000.nii.gz']

    configuration = '3d_fullres'
    pp = DefaultPreprocessor()

    # _ because this position would be the segmentation if seg_file was not None (training case)
    # even if you have the segmentation, don't put the file there! You should always evaluate in the original
    # resolution. What comes out of the preprocessor might have been resampled to some other image resolution (as
    # specified by plans)
    plans_manager = PlansManager(plans_file)
    data, _, properties = pp.run_case(input_images, seg_file=None, plans_manager=plans_manager,
                                      configuration_manager=plans_manager.get_configuration(configuration),
                                      dataset_json=dataset_json_file)

    # voila. Now plug data into your prediction function of choice. We of course recommend nnU-Net's default (TODO)
    return data


if __name__ == '__main__':
    example_test_case_preprocessing()
    # pp = DefaultPreprocessor()
    # pp.run(2, '2d', 'nnUNetPlans', 8)

    ###########################################################################################################
    # how to process a test cases? This is an example:
    # example_test_case_preprocessing()
