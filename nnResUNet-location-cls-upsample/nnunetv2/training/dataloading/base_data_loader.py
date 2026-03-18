from typing import Union, Tuple

from batchgenerators.dataloading.data_loader import DataLoader
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class nnUNetDataLoaderBase(DataLoader):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 sampling_categories: dict = None,
                 vessel_class_weights: dict = None,
                 compute_positives: bool = False,
                 cls_foreground_labels: list = None):
        super().__init__(data, batch_size, 1, None, True, False, True, sampling_probabilities)
        assert isinstance(data, nnUNetDataset), 'nnUNetDataLoaderBase only supports dictionaries as data'
        self.indices = list(data.keys())
        self.sampling_categories = sampling_categories
        # vessel_class_weights: {label: weight} for upsample mode
        # e.g. {1: 1, 2: 1, 3: 1, 4: 1} → 等比例取樣
        # None → 非 upsample 模式，所有座標合併後隨機取樣
        self.vessel_class_weights = vessel_class_weights
        # 是否計算 classification label（positives）；僅 classifier 架構需要
        self.compute_positives = compute_positives
        # cls_foreground_labels: 分類頭判斷 positive 時只看哪些 label
        # None → 所有 > 0 的 label 都算前景（原始行為）
        # [1] → 只有 label==1 算前景（例如只看動脈瘤）
        # [1, 2] → label==1 或 label==2 算前景
        self.cls_foreground_labels = cls_foreground_labels
        # cache for _sample_vessel_voxel concatenation（避免每次重新 concatenate）
        self._vessel_concat_cache_id = None
        self._vessel_concat_cache_arr = None

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple(label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        # load one case
        data, seg, properties = self._data.load_case(self.indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None], vessel_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        #碩造box的上下限然後取樣
        lbs = [- need_to_pad[i] // 2 for i in range(dim)] #下限座標
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)] #上限座標

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            #根據血管mask去篩選，至少包含一個血管
            selected_voxel = self._sample_vessel_voxel(vessel_locations)
            if selected_voxel is not None:
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # 沒有 vessel_locations，回退到隨機採樣
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I do not want a random location')
        else:
            if not force_fg and self.has_ignore:
                #這裡先不管
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    print('Warning! No annotated pixels in image!')
                    selected_class = None
                # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key}')
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to vessel or random
                #沒有任何前景，我會取血管
                selected_voxel = self._sample_vessel_voxel(vessel_locations)
                if selected_voxel is not None:
                    bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
                else:
                    # 沒有 vessel_locations，回退到隨機採樣
                    bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        # 防止 bbox_lbs 超出上限（vessel/foreground 取樣可能選到邊緣 voxel）
        bbox_lbs = [min(bbox_lbs[i], ubs[i]) for i in range(dim)]
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

    def _sample_vessel_voxel(self, vessel_locations):
        """
        從 vessel_locations 中取樣一個 voxel。

        支援兩種格式（自動偵測）：
        - 舊格式：numpy.ndarray, shape (N, 4)，直接隨機取樣
        - 新格式：dict {label: ndarray}，支援 upsample 模式

        回傳 None 則由呼叫端改用隨機採樣。
        """
        if vessel_locations is None:
            return None

        # 舊格式：直接是 numpy array (N, 4)
        if isinstance(vessel_locations, np.ndarray):
            if len(vessel_locations) == 0:
                return None
            return vessel_locations[np.random.choice(len(vessel_locations))]

        # 新格式：dict {label: ndarray}
        if len(vessel_locations) == 0:
            return None

        if self.vessel_class_weights is None:
            # 非 upsample 模式：所有座標合併（使用 cache 避免每次重新 concatenate）
            vid = id(vessel_locations)
            if self._vessel_concat_cache_id != vid:
                all_locs = np.concatenate([locs for locs in vessel_locations.values() if len(locs) > 0], axis=0)
                self._vessel_concat_cache_id = vid
                self._vessel_concat_cache_arr = all_locs
            all_locs = self._vessel_concat_cache_arr
            if len(all_locs) == 0:
                return None
            return all_locs[np.random.choice(len(all_locs))]
        else:
            # upsample 模式：按 vessel_class_weights 比例選類別
            available_labels = [lb for lb in vessel_locations.keys() if len(vessel_locations[lb]) > 0]
            if len(available_labels) == 0:
                return None

            weights = np.array([self.vessel_class_weights.get(lb, 1.0) for lb in available_labels])
            weights = weights / weights.sum()
            selected_label = available_labels[np.random.choice(len(available_labels), p=weights)]
            locs = vessel_locations[selected_label]
            return locs[np.random.choice(len(locs))]
