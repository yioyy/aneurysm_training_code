import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        # 用 numpy 避免在 background thread 裡呼叫 torch（會搶 GIL 導致多線程效能下降）
        positives = np.zeros((self.seg_shape[0], self.seg_shape[1]), dtype=np.int64)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg,
                                               properties.get('dilate_locations', properties.get('class_locations')),
                                               properties.get('vessel_locations', None))

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

            # 修正：根據目標 patch 尺寸計算 padding，而非 bbox 尺寸（與 data_loader_2d 同步）
            target_shape = self.data_shape[2:]
            cropped_shape = data.shape[1:]  # 裁剪後的實際尺寸（去掉 channel 維度）

            padding = []
            for d in range(dim):
                # 總共需要補齊的大小
                total_pad = target_shape[d] - cropped_shape[d]
                # 左側 padding：bbox 超出左邊界的部分
                left_pad = valid_bbox_lbs[d] - bbox_lbs[d]
                # 右側 padding：剩餘的部分
                right_pad = total_pad - left_pad
                padding.append((left_pad, right_pad))

            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

            # classification label（僅 classifier 架構需要，純 U-Net 跳過以加速）
            if self.compute_positives:
                lesion_all = np.sum(seg)
                lesion_patch = np.sum(seg_all[j])
                c_i, z_i, y_i, x_i = np.where(seg_all[j] > 0)
                if len(z_i) > 0:
                    z_long = np.max(z_i) - np.min(z_i) + 1
                    y_long = np.max(y_i) - np.min(y_i) + 1
                    x_long = np.max(x_i) - np.min(x_i) + 1

                    is_positive = (lesion_patch >= lesion_all or
                                   z_long >= self.data_shape[2] / 2 or
                                   y_long >= self.data_shape[3] / 2 or
                                   x_long >= self.data_shape[4] / 2)

                    if is_positive and self.sampling_categories is not None:
                        positives[j, 0] = self.sampling_categories.get(i, 0)
                    elif is_positive:
                        positives[j, 0] = 1

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'positives': positives, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
