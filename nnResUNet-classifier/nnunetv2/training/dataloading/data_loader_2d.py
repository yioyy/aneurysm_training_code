import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
import torch


class nnUNetDataLoader2D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        positives = torch.zeros((self.seg_shape[0], self.seg_shape[1]), dtype=torch.int64)

        for j, current_key in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)
            data, seg, properties = self._data.load_case(current_key)

            # select a class/region first, then a slice where this class is present, then crop to that area
            if not force_fg:
                if self.has_ignore:
                    selected_class_or_region = self.annotated_classes_key
                else:
                    selected_class_or_region = None
            else:
                # filter out all classes that are not present here
                eligible_classes_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                selected_class_or_region = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                    len(eligible_classes_or_regions) > 0 else None
            if selected_class_or_region is not None:
                selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
            else:
                selected_slice = np.random.choice(len(data[0]))

            data = data[:, selected_slice]
            seg = seg[:, selected_slice]
            lesion_all = np.sum(seg)

            # the line of death lol
            # this needs to be a separate variable because we could otherwise permanently overwrite
            # properties['class_locations']
            # selected_class_or_region is:
            # - None if we do not have an ignore label and force_fg is False OR if force_fg is True but there is no foreground in the image
            # - A tuple of all (non-ignore) labels if there is an ignore label and force_fg is False
            # - a class or region if force_fg is True
            class_locations = {
                selected_class_or_region: properties['class_locations'][selected_class_or_region][properties['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
            } if (selected_class_or_region is not None) else None

            # print(properties)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
                                               properties['dilate_locations'], properties['vessel_locations'], overwrite_class=selected_class_or_region)

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            # 修正：根據目標 patch 尺寸計算 padding，而非 bbox 尺寸
            # 目標尺寸從 data_all 的 shape 取得（去掉 batch 和 channel 維度）
            target_shape = self.data_shape[2:]
            cropped_shape = data.shape[1:]  # 裁剪後的實際尺寸（去掉 channel 維度）
            
            padding = []
            for i in range(dim):
                # 總共需要補齊的大小
                total_pad = target_shape[i] - cropped_shape[i]
                # 左側 padding：bbox 超出左邊界的部分
                left_pad = valid_bbox_lbs[i] - bbox_lbs[i]
                # 右側 padding：剩餘的部分
                right_pad = total_pad - left_pad
                padding.append((left_pad, right_pad))
            
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

            lesion_patch = np.sum(seg_all[j]) #data: (10, 1, 16, 32, 32)
            c_i, y_i, x_i = np.where(seg_all[j] > 0)
            if len(y_i) > 0:
                y_long = np.max(y_i) - np.min(y_i) + 1 #種樹要加1
                x_long = np.max(x_i) - np.min(x_i) + 1 #種樹要加1

                #正樣本條件判斷
                if lesion_patch >= lesion_all:
                    positives[j, 0] = 1 #先假定只有一類
                elif y_long >= self.data_shape[2]/2:
                    positives[j, 0] = 1 #先假定只有一類
                elif x_long >= self.data_shape[3]/2:
                    positives[j, 0] = 1 #先假定只有一類
                else:
                    positives[j, 0] = 0 #先假定只有一類
            else:
                positives[j, 0] = 0 #先假定只有一類

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'positives': positives, 'keys': selected_keys}            
        #return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset004_Hippocampus/2d'
    ds = nnUNetDataset(folder, None, 1000)  # this should not load the properties!
    dl = nnUNetDataLoader2D(ds, 366, (65, 65), (56, 40), 0.33, None, None)
    a = next(dl)
