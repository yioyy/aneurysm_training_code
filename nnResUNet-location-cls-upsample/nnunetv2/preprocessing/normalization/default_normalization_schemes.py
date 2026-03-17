from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number


class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict)
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass


class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (max(std, 1e-8))
        return image


class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        image = image.astype(self.target_dtype)
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']
        image = np.clip(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        return image


class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype)


class RescaleTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype)
        image = image - image.min()
        image = image / np.clip(image.max(), a_min=1e-8, a_max=None)
        return image


class RGBTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert image.min() >= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
        assert image.max() <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                   ". Your images do not seem to be RGB images"
        image = image.astype(self.target_dtype)
        image = image / 255.
        return image


#以下皆為自訂義
class ADCNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype)
        image = image / 600.
        return image
    

class ZScoreBrainNormalization_ERROR(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        只針對腦內區域做正規化
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        mask = image > 0
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        return image
    
class ZScoreBrainNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        只針對腦內區域做正規化
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        mask = image > 0
        mean = image[mask].mean()
        std = image[mask].std()
        image = (image - mean) / (max(std, 1e-8))
        return image

class ZScoreImageNormalization(ImageNormalization):
    #使用整張影像來ZScoreNormalization
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        使用整張影像來ZScoreNormalization
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (max(std, 1e-8))
        return image

def custom_normalize_1(volume, mask=None, new_min=0., new_max=0.5, 
                    min_percentile=0.5, max_percentile=99.5, use_positive_only=True, use_min_max=False):
    """
    自定義正規化函數
    
    Args:
        volume: 輸入體積資料
        mask: 遮罩
        new_min, new_max: 新的最小最大值
        min_percentile, max_percentile: 百分位數
        use_positive_only: 是否只使用正值
        use_min_max: 是否使用最大最小值正規化（True）或百分位數正規化（False）
        
    Returns:
        正規化後的體積
    """
    new_volume = volume.copy()
    new_volume = new_volume.astype("float32")
    
    if (mask is not None) and (use_positive_only):
        intensities = new_volume[mask].ravel()
        intensities = intensities[intensities > 0]
    elif mask is not None:
        intensities = new_volume[mask].ravel()
    elif use_positive_only:
        intensities = new_volume[new_volume > 0].ravel()
    else:
        intensities = new_volume.ravel()

    if use_min_max:
        # 使用最大最小值正規化
        robust_min = np.min(intensities)
        robust_max = np.max(intensities)
    else:
        # 使用百分位數正規化
        robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
        robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

    if robust_min != robust_max:
        new_volume = new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:
        new_volume = np.zeros_like(new_volume)
        
    return new_volume


class CTANormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTANormalization requires intensity properties"
        image = image.astype(self.target_dtype)
        #對影像取cut off 
        image = np.clip(image, -250, 550)

        image_norm = custom_normalize_1(image, new_min=0., new_max=1, use_positive_only=False, use_min_max=True)
        image_norm[image_norm<0] = 0

        return image_norm

class MRI_custom_normalize_1(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "MRI_custom_normalize_1 requires intensity properties"
        image = image.astype(self.target_dtype)

        image_norm = custom_normalize_1(image, use_min_max=False)

        return image_norm

class Max995_Min005_Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "MRI_custom_normalize_1 requires intensity properties"
        image = image.astype(self.target_dtype)

        image_norm = custom_normalize_1(image, new_min=0., new_max=1, 
                    min_percentile=0.5, max_percentile=99.5, use_min_max=False)

        return image_norm