#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

predict model，非常重要:使用這個程式，安裝的版本都需要一致，才能運行
conda create -n orthanc-stroke tensorflow-gpu=2.3.0 anaconda python=3.7
這邊弄2個版本，一個是nnU-Net的版本，另一個是君彥的版本

@author: chuan
"""
import warnings
warnings.filterwarnings("ignore") # 忽略警告输出


import glob, re, os, sys, time, itertools
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import glob
import cv2 
import time
import pydicom
import nibabel as nib
#import gdcm
import numpy as np
import keras
import pandas as pd
# from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
import skimage 
import skimage.feature
import skimage.measure
from skimage import measure,color,morphology
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from scipy import signal
from scipy import ndimage as ndi
from skimage.morphology import disk, ball, binary_dilation, binary_closing, remove_small_objects, binary_erosion, binary_opening, skeletonize
from skimage.filters import threshold_multiotsu, gaussian, threshold_otsu, frangi
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import watershed, expand_labels
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt as dist_field
from IPython.display import display, HTML
from brainextractor import BrainExtractor
import random
from random import uniform
import argparse
import logging
import json
import psutil
import pynvml #导包
from collections import OrderedDict
from nii_transforms import nii_img_replace
autotune = tf.data.experimental.AUTOTUNE
from nnResUNet.gpu_nnUNet import predict_from_raw_data

# histogram
def get_histogram_xy(arr, mask=None):
    if mask is None:
        mask = np.ones_like(arr, dtype=bool)
    hist, bin_edges = np.histogram(arr[mask].ravel(), bins=100)
    bins_mean = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(100)]
    return [bins_mean, hist]

# keep largest (cpu)
def binary_keep_largest(mask, k=1):
    label_arr = label(mask, connectivity=1)
    props = regionprops_table(label_arr, properties=['label','area'])
    sort_idx = np.argsort(props['area'])[::-1][:k]
    largest_lb = props['label'][sort_idx]
    mask = np.isin(label_arr, largest_lb)
    del label_arr, props, sort_idx, largest_lb
    return mask

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
    # select intensities
    new_volume = volume.copy()
    new_volume = new_volume.astype("float32")
    if (mask is not None)and(use_positive_only):
        intensities = new_volume[mask].ravel()
        intensities = intensities[intensities > 0]
    elif mask is not None:
        intensities = new_volume[mask].ravel()
    elif use_positive_only:
        intensities = new_volume[new_volume > 0].ravel()
    else:
        intensities = new_volume.ravel()

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

def get_struc(diameter, spacing):  # diameter in mm
    structure_size = np.round(diameter / np.array(spacing)).astype('int32')
    structure_size = np.maximum([1,1,1], structure_size)
    structure = resize_volume(ball(diameter*5), target_size=structure_size, dtype='bool')
    center_point = np.array(structure.shape) // 2
    structure[center_point] = True
    return structure

# resampling
def resize_volume(arr, spacing=None, target_spacing=None, target_size=None, order=3, dtype='float32'):
    if ("int8" in dtype)or(dtype == "bool"):
        order = 0

    if (spacing is not None)and(target_spacing is not None):
        # from spacing to target_spacing
        scale = np.array(spacing) / np.array(target_spacing)
        out_vol = ndi.zoom(arr, zoom=scale, order=order,
                           mode='grid-mirror', prefilter=True, grid_mode=False)
    elif target_size is not None:
        # to target_size
        scale = np.array(target_size) / np.array(arr.shape)
        out_vol = ndi.zoom(arr, zoom=scale, output=np.zeros(target_size, dtype=dtype), order=order,
                           mode='grid-mirror', prefilter=True, grid_mode=False)

    if 'int' in dtype:  # clip values
        dtype_info = np.iinfo(dtype)
        out_vol = np.clip(out_vol, dtype_info.min, dtype_info.max)
    return out_vol.astype(dtype)

# mask interpolation
def mask_interpolation(mask, factor, **kwargs):
    """ Resizing the mask through interpolation by building the distance field.
        Randomly sampling factor and multi-categories are considered.
        Implemented by Kuan (Kevin) Zhang, Ph.D., Radiology Informatics Laboratory, Mayo Clinic.

        This implementation follows:
        How to properly interpolate masks for deep learning in medical imaging?
        Args:
            mask (np.array): The initial mask matrix in the shape: (slices, x, y)
            factor (tuple): The sampling factor of resizing in the shape: (fx, fy, fz)
        Output:
            The interpolated mask matrix to return.
    """

    # Check the number of mask types contained in the input.
    mask_types = int(mask.max())

    if mask_types > 1:
        # Creat a list for the multi-category lables.
        mask_mul = []
        # Conver the mask values into binary for each type.
        for i in range(mask_types):
            mask_mul.append(np.where(mask == i+1,1,0))
        mask_dist = mask_mul.copy()

        for i in range(mask_types):
            for z in range(mask.shape[-1]):
                # The inner distance field to the edge:
                dist_field_inner = ndi.distance_transform_edt(mask_mul[i][:,:,z])
                # The outer distance field to the edge:
                dist_field_outer = ndi.distance_transform_edt(np.logical_not(mask_mul[i][:,:,z]))
                mask_dist[i][:,:,z] = dist_field_inner - dist_field_outer
            mask_dist[i] = ndi.zoom(mask_dist[i], factor, order=1,
                                    mode='grid-mirror', prefilter=True, grid_mode=False)

        # Apply the threshold on the interpolated mask values.
        # The threshold is set as 0.0.
        mask_new = np.zeros(mask_dist[0].shape)

        for i in range(mask_types):
            mask_dist[i] = np.where(mask_dist[i] > 0., 1, 0)
        m_index = np.any(np.array(mask_dist)!=0, axis=0)
        mask_new[m_index] = np.argmax(np.array(mask_dist)[:,m_index],axis=0) + 1
        return mask_new.astype(mask.dtype)

    else:  # For the binary mask type.
        mask_dist = np.zeros(mask.shape)

        for z in range(mask.shape[-1]):
            # The inner distance field to the edge:
            dist_field_inner = ndi.distance_transform_edt(mask[:,:,z])
            # The outer distance field to the edge:
            dist_field_outer = ndi.distance_transform_edt(np.logical_not(mask[:,:,z]))
            mask_dist[:,:,z] = dist_field_inner - dist_field_outer
        mask_dist = ndi.zoom(mask_dist, factor, order=1,
                             mode='grid-mirror', prefilter=True, grid_mode=False)
        mask_new = np.where(mask_dist > 0., 1,0)
        return mask_new.astype(mask.dtype)

#@title Load image
def load_volume(path_volume, im_only=False, squeeze=True, dtype=None, LPS_coor=True):
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with an identity affine matrix and blank header.
    :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
    :param squeeze: (optional) whether to squeeze the volume when loading.
    :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
    The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
    :return: the volume, with corresponding affine matrix and header if im_only is False.
    """
    path_volume = str(path_volume)
    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        x = nib.as_closest_canonical(x)  # to RAS space
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
        spacing = list(x.header.get_zooms())
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
        spacing = [1., 1., 1.]
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)
    if LPS_coor:
        volume = volume[::-1,::-1,:]

    if im_only:
        return volume
    else:
        return volume, spacing, aff

#@title SynthSeg (unet2 only) (w/ isotropic-1mm vol)
def get_brain_seg(image_arr, spacing, model0):

    def center_crop(volume, centroid=None, size=192):
        """ center crop a cube with desired size,
        if the target area is smaller than the desired size, it will be padded with the minimum
        input:: volume is a 3d-array
        input:: centroid is center coordinates (x, y, z)
        output:: cropped cube with desired size ex.(28, 28, 28)
        """
        w, h, d = volume.shape
        r = int(size // 2)
        if centroid is None:
            centroid = (w//2, h//2, d//2)
        x0 = int(centroid[0])
        y0 = int(centroid[1])
        z0 = int(centroid[2])
        if (r < x0 < w-r-1)and(r < y0 < h-r-1)and(r < z0 < d-r-1):
            return volume[x0-r:x0-r+size, y0-r:y0-r+size, z0-r:z0-r+size]
        else:
            volume = np.pad(volume, ((r,r),(r,r),(r,r)), 'minimum')
            return volume[x0:x0+size, y0:y0+size, z0:z0+size]

    def rescale_volume(volume, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5, use_positive_only=False):
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
        # select only positive intensities
        new_volume = volume.copy()
        intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

        # define min and max intensities in original image for normalisation
        robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
        robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

        # trim values outside range
        new_volume = np.clip(new_volume, robust_min, robust_max)

        # rescale image
        if robust_min != robust_max:
            return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
        else:  # avoid dividing by zero
            return np.zeros_like(new_volume)

    def center_pad(crop_arr, size):
        w, h, d = crop_arr.shape
        x0 = max(0, (size[0] - w) // 2)
        x1 = max(0, size[0] - w - x0)
        y0 = max(0, (size[1] - h) // 2)
        y1 = max(0, size[1] - h - y0)
        z0 = max(0, size[2] - d)
        return np.pad(crop_arr, ((x0,x1),(y0,y1),(z0,0)), 'constant')

    # image preprocess
    target_spacing = np.array([1, 1, 1], dtype=np.float32)
    image_iso = resize_volume(image_arr, spacing, target_spacing=target_spacing, order=0, dtype='float32')
    z_pad = 192 - image_iso.shape[2]
    #print("z_pad =", z_pad)
    if z_pad >= 0:  # image_iso slices <= 192
        image_iso_pad = np.pad(image_iso, ((0,0),(0,0),(z_pad,0)), 'minimum')  # pad z
    else:  # image_iso slices > 192
        image_iso_pad = image_iso[:, :, -192:]  # crop z
    image_iso_crop = center_crop(image_iso_pad)
    del image_iso_pad

    # model prediction
    seg_iso_crop = model0.serve(rescale_volume(image_iso_crop)[np.newaxis, ::-1, ::-1, :, np.newaxis])
    synthseg33_array = np.argmax(seg_iso_crop[0, ::-1, ::-1, :, :], axis=3).astype('uint8')
    # plot_3view_label(image_iso_crop, synthseg33_array, target_spacing)
    del image_iso_crop

    # seg postprocess
    if z_pad >= 0:  # image_iso slices <= 192
        seg_iso = center_pad(synthseg33_array, size=image_iso.shape)[:, :, z_pad:]  # unpad z
    else:  # image_iso slices > 192
        seg_iso = center_pad(synthseg33_array, size=image_iso.shape)  # uncrop z
    #plot_3view_label(image_iso, seg_iso, target_spacing)
    brain_seg = resize_volume(seg_iso, target_size=image_arr.shape, dtype='uint8')
    brain_seg = binary_keep_largest(brain_seg > 0)
    return brain_seg

#@title BET (w/ isotropic(spacing-z, spacing-z, spacing-z) vol)
def get_BET_brain_mask(image_arr, spacing, bet_iter=1000, pad=32):
    # resize to isotropic
    target_spacing = np.array([spacing[2], spacing[2], spacing[2]], dtype=np.float32)
    print("BET target_spacing =", target_spacing)
    image_iso = resize_volume(image_arr, spacing, target_spacing=target_spacing, order=0, dtype='int16')

    pv = np.percentile(image_iso.ravel(), 15)
    print("pad value =", pv)
    image_iso = np.pad(image_iso, ((pad,pad),(pad,pad),(pad,pad)), 'constant', constant_values=pv)  # minimum, mean

    # nibabel obj
    affine = np.array([ [target_spacing[0], 0, 0, 0],
                        [0, target_spacing[1], 0, 0],
                        [0, 0, target_spacing[2], 0],
                        [0, 0, 0, 1]], dtype=np.float32)
    nib_image = nib.Nifti1Image(image_iso[::-1, ::-1, :], affine=affine)  # LPS to RAS orientation

    # BET process
    bet = BrainExtractor(img=nib_image)
    bet.run(iterations=bet_iter)
    brain_mask_iso = bet.compute_mask()[::-1, ::-1, :] > 0  # RAS to LPS orientation
    brain_mask_iso = brain_mask_iso[pad:-pad, pad:-pad, pad:-pad]

    # back to original spacing
    brain_mask = resize_volume(brain_mask_iso, target_size=image_arr.shape, dtype='bool')
    return brain_mask

#@title modify brain mask
def modify_brain_mask(brain_mask, spacing, verbose=False):
    # merge and modify brain mask
    if verbose: print(">>> modify_brain_mask ", end='')
    brain_mask = binary_dilation(brain_mask, get_struc(8, spacing))
    if verbose: print(".", end='')
    brain_mask = binary_fill_holes(brain_mask)
    if verbose: print(".", end='')
    brain_mask = binary_erosion(brain_mask, get_struc(14, spacing).sum(axis=2, keepdims=True).astype(bool))  # 12
    if verbose: print(".", end='')
    brain_mask = binary_opening(brain_mask, get_struc(14, spacing).sum(axis=2, keepdims=True).astype(bool))
    if verbose: print(">>>", brain_mask.shape)
    return brain_mask

#@title threshold segmentation algorithm
class VesselSegmenter(object):
    """ ref paper: 2015 Threshold segmentation algorithm for automatic extraction of cerebral vessels from brain magnetic resonance angiography images
    https://www.sciencedirect.com/science/article/pii/S0165027014004166?via%3Dihub
    """
    def keep_brain_img(self, img, mask, spacing, radius_mm=40):
        r = np.int32(radius_mm // spacing[0])
        x0, y0 = np.array(mask.shape[:2])//2 - np.array([r, r])
        seed_area = np.zeros(mask.shape[:2], dtype=bool)
        disk_mask = disk(r)
        seed_area[x0:x0+disk_mask.shape[0], y0:y0+disk_mask.shape[1]] = disk_mask
        seed_area = np.repeat(seed_area[..., np.newaxis], mask.shape[2], axis=-1)
        mask = mask | seed_area
        return img * mask

    def otsu_threshold(self, img:np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        threshold = threshold_otsu(img)
        return img[img > threshold]

    def distribution(self, x:np.ndarray, mu:float, sigma:float,
                     stats:str, coefficient:float=1) -> np.ndarray:
        """ The probability density function (pdf) of distribution is,
        "gumbel": pdf(x; mu, sigma) = exp(-(x - mu) / sigma - exp(-(x - mu) / sigma)) / sigma
        "normal": pdf(x; mu, sigma) = exp(-(x - mu)**2 / (2 * sigma**2)) / ((2 * pi)**0.5) / sigma
        """
        if stats == "gumbel":
            return np.exp(-(x - mu) / sigma - np.exp(-(x - mu) / sigma)) / sigma
        elif stats == "normal":
            return coefficient * np.exp(-((x - mu)**2) / (2 * sigma**2)) / ((2 * np.pi)**0.5) / sigma

    def threshold_segmentation(self, img:np.ndarray, mask, spacing) -> np.ndarray:
        img = self.keep_brain_img(img, mask, spacing)
        foreground = self.otsu_threshold(img).astype(np.uint16)

        mu, sigma = np.mean(foreground), np.std(foreground)
        x = np.linspace(0, foreground.max(), foreground.max())
        p_gumbel = self.distribution(x, mu, sigma, stats="gumbel")
        p_normal = self.distribution(x, mu, sigma, stats="normal", coefficient=95)  # W 25
        threshold = np.nonzero(np.greater_equal(p_gumbel, p_normal))[0].min()
        return np.uint16(img > threshold), threshold
    
#@title get major vessel seeds for selection
def get_brain_radius_area(brain_mask, spacing, radius_mm=40):
    r = np.int32(radius_mm // spacing[0])
    x0, y0 = np.array(brain_mask.shape[:2])//2 - np.array([r, r])
    area = np.zeros(brain_mask.shape[:2], dtype=bool)
    disk_mask = disk(r) > 0
    area[x0:x0+disk_mask.shape[0], y0:y0+disk_mask.shape[1]] = disk_mask
    area = np.repeat(area[..., np.newaxis], brain_mask.shape[2], axis=-1)
    return brain_mask & area

def get_vessel_seed(candidates, mask=None, spacing=(1,1,1)):
    # Remove the region below brain
    candidates_mask = candidates > 0
    if mask is not None:  # keep brain region
        candidates_mask = candidates > 0
        brain_bottom_idx = np.where(np.any(mask > 0, axis=(0,1)))[0][0]
        candidates_mask[:,:,:brain_bottom_idx] = False

    # make major seeds
    major_seed_mask = binary_erosion(candidates_mask.copy(), get_struc(2.5, spacing))  # 2.5, 3.0
    area_mask = get_brain_radius_area(mask, spacing, radius_mm=45)
    major_seed_mask = major_seed_mask & area_mask

    # make center seeds
    center_seed_mask = candidates_mask.copy()
    if mask is not None:  # keep brain region
        upper1of3_brain_height = (candidates_mask.shape[2] - brain_bottom_idx) // 3
        center_seed_mask[:, :, :brain_bottom_idx + upper1of3_brain_height] = False
        center_seed_mask[:, :, -upper1of3_brain_height:] = False
    center_seed_mask = binary_erosion(center_seed_mask, get_struc(1.5, spacing))
    area_mask = get_brain_radius_area(mask, spacing, radius_mm=20)
    center_seed_mask = center_seed_mask & area_mask

    # make upper seeds
    upper_seed_mask = candidates_mask.copy()
    if mask is not None:  # keep brain region
        upper1of3_brain_height = (candidates_mask.shape[2] - brain_bottom_idx) // 3
        upper_seed_mask[:, :, :brain_bottom_idx + upper1of3_brain_height*2] = False
    upper_seed_mask = binary_erosion(upper_seed_mask, get_struc(1.5, spacing))
    area_mask = get_brain_radius_area(mask, spacing, radius_mm=40)
    upper_seed_mask = upper_seed_mask & area_mask
    return major_seed_mask | center_seed_mask | upper_seed_mask

#@title Sliding_window_inference help functions
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/Segmentation/nnUNet/models/sliding_window.py

def get_window_slices(image_size, roi_size, overlap, strategy):
    dim_starts = []
    for image_x, roi_x in zip(image_size, roi_size):
        interval = roi_x if roi_x == image_x else int(roi_x * (1 - overlap))
        starts = list(range(0, image_x - roi_x + 1, interval))
        if strategy == "overlap_inside" and starts[-1] + roi_x < image_x:
            starts.append(image_x - roi_x)
        dim_starts.append(starts)
    slices = [(starts + (0,), roi_size + (-1,)) for starts in itertools.product(*dim_starts)]
    batched_window_slices = [((0,) + start, (1,) + roi_size) for start, roi_size in slices]
    return batched_window_slices

@tf.function
def gaussian_kernel(roi_size, sigma):
    gauss = signal.windows.gaussian(roi_size[0], std=sigma * roi_size[0])
    for s in roi_size[1:]:
        gauss = np.outer(gauss, signal.windows.gaussian(s, std=sigma * s))
    gauss = np.reshape(gauss, roi_size)
    gauss = np.power(gauss, 1 / len(roi_size))
    gauss /= gauss.max()
    return tf.convert_to_tensor(gauss, dtype=tf.float32)

def get_importance_kernel(roi_size, blend_mode, sigma):
    if blend_mode == "constant":
        return tf.ones(roi_size, dtype=tf.float32)
    elif blend_mode == "gaussian":
        return gaussian_kernel(roi_size, sigma)
    else:
        raise ValueError(f'Invalid blend mode: {blend_mode}. Use either "constant" or "gaussian".')
    
@tf.function
def run_model1(x, model1, importance_map, **kwargs):
    pred1 = model1.serve(x)  # segment conf.
    return pred1 * importance_map

#@title model predict vessel
def predict_vessel(image_arr, brain_mask,
                   model1, patch_size=(64, 64, 32), sigma=0.125, n_class=1, overlap=0.5, conf_th=0.1,
                   verbose=False):

    def sliding_window_inference1(inputs, roi_size, model, overlap, n_class, importance_map, strategy="overlap_inside", mask=None, **kwargs):
        image_size = tuple(inputs.shape[1:-1])
        roi_size = tuple(roi_size)
        # Padding to make sure that the image size is at least roi size
        padded_image_size = tuple(max(image_size[i], roi_size[i]) for i in range(3))
        padding_size = [image_x - input_x for image_x, input_x in zip(image_size, padded_image_size)]
        paddings = [[0, 0]] + [[x // 2, x - x // 2] for x in padding_size] + [[0, 0]]
        input_padded = tf.pad(inputs, paddings)
        if mask is not None:
            mask_padded = tf.pad(tf.convert_to_tensor(mask, dtype=tf.bool), paddings)

        output_shape = (1, *padded_image_size, n_class)
        output_sum = tf.zeros(output_shape, dtype=tf.float32)
        output_weight_sum = tf.zeros(output_shape, dtype=tf.float32)
        window_slices = get_window_slices(padded_image_size, roi_size, overlap, strategy)

        for i, window_slice in enumerate(window_slices):
            if (mask is None) or ((mask is not None)and(tf.math.reduce_any(tf.slice(mask_padded, begin=window_slice[0], size=window_slice[1])))):
                window = tf.slice(input_padded, begin=window_slice[0], size=window_slice[1])
                pred = run_model1(window, model, importance_map, **kwargs)
                padding = [
                    [start, output_size - (start + size)] for start, size, output_size in zip(*window_slice, output_shape)
                ]
                padding = padding[:-1] + [[0, 0]]
                output_sum = output_sum + tf.pad(pred, padding)
                output_weight_sum = output_weight_sum + tf.pad(importance_map, padding)
            if verbose:
                if i % 100 == 0: print('.', end='')

        output = output_sum / tf.clip_by_value(output_weight_sum, 1, 256)
        crop_slice = [slice(pad[0], pad[0] + input_x) for pad, input_x in zip(paddings, inputs.shape[:-1])]
        return output[crop_slice]

    # prepare importance_map
    if verbose: print(">>> predict_vessel ", end='')
    image_arr = custom_normalize_1(image_arr)

    importance_kernel = get_importance_kernel(patch_size, blend_mode="gaussian", sigma=sigma)
    importance_map = tf.tile(tf.reshape(importance_kernel, shape=[1, *patch_size, 1]), multiples=[1, 1, 1, 1, n_class],)
    # sliding_window_inference
    vessel_conf = sliding_window_inference1(inputs=image_arr[np.newaxis,:,:,:,np.newaxis],
                                            roi_size=patch_size, model=model1, overlap=overlap,
                                            n_class=n_class, importance_map=importance_map,
                                            mask=brain_mask[np.newaxis,:,:,:,np.newaxis])
    vessel_conf = vessel_conf[0,:,:,:,0].numpy()

    if verbose: print(">>>", vessel_conf.shape)
    return vessel_conf > conf_th

#@title combine two vessel results
def combine_two_vessels(vessel_threshold, vessel_frangi, seed_mask=None, brain_mask=None):
    if brain_mask is not None:  # keep brain region
        brain_bottom_idx = np.where(np.any(brain_mask > 0, axis=(0,1)))[0][0]
        vessel_threshold[:,:,:brain_bottom_idx] = False  # cut off below brain area
        vessel_frangi[~brain_mask] = 0  # remove out-brain frangi

    vessel_mask = (vessel_threshold > 0) | (vessel_frangi > 0)
    label_map = label(vessel_mask)
    if seed_mask is None:
        vessel_mask = label_map > 0
    else:  # select vessels via seeds
        lbs = np.unique(label_map[seed_mask])
        vessel_mask = np.isin(label_map, lbs)

    vessel_mask = remove_small_objects(vessel_mask, min_size=500)
    return vessel_mask

#@title get_vessel_skeleton_labels
def get_vessel_skeleton_labels(vessel_mask, spacing):
    target_thickness = spacing[0]
    Z_FAC = spacing[2] / target_thickness  # Sampling factor in Z direction
    vessel_sr = mask_interpolation(vessel_mask, factor=(1,1,Z_FAC)).astype(bool)
    # get skeleton diameter
    distance_sr = ndi.distance_transform_edt(vessel_sr)  # px
    skeleton_sr = skeletonize(vessel_sr > 0) > 0
    # plot pseudo_vessel
    pseudo_vessel_sr = np.zeros_like(skeleton_sr, dtype=bool)
    for center in np.stack(np.where(skeleton_sr), axis=-1):
        r = distance_sr[center[0], center[1], center[2]] + 1  # float px + 1
        d = ball(r).shape[0]
        x0 = int(center[0] - (d // 2))
        y0 = int(center[1] - (d // 2))
        z0 = int(center[2] - (d // 2))
        try:
            pseudo_vessel_sr[x0:x0 + d, y0:y0 + d, z0:z0 + d] = pseudo_vessel_sr[x0:x0 + d, y0:y0 + d, z0:z0 + d] | ball(r)
        except:
            pass
    # diff vessel
    diff_vessel_sr = (vessel_sr.astype('int8') - pseudo_vessel_sr.astype('int8')) > 0
    # back to original space
    pseudo_vessel = resize_volume(pseudo_vessel_sr, target_size=vessel_mask.shape, dtype='uint8')
    diameter = resize_volume(distance_sr, target_size=vessel_mask.shape, dtype='float32') * target_thickness * 2
    skeleton = skeletonize(pseudo_vessel > 0) > 0
    diff_vessel = resize_volume(diff_vessel_sr, target_size=vessel_mask.shape, dtype='uint8')
    # return diameter * skeleton.astype('float32'), diff_vessel

    # merge labels
    skeleton_labels = np.clip(expand_labels(skeleton.astype('uint8'), 1) + (diff_vessel * 2), 0, 2)
    return skeleton_labels

#@title predict vessel_16labels (w/ isotropic 0.8mm vol)
def predict_vessel_16labels(vessel_mask, model3, spacing, post_proc=True, min_size=3, verbose=False):

    def find_end_points(skeleton_mask):
        ends_map = np.zeros(skeleton_mask.shape, dtype='uint8')
        centers = np.stack(np.where(skeleton_mask), axis=-1)
        for center in centers:
            patch = skeleton_mask[center[0]-1:center[0]+2, center[1]-1:center[1]+2, center[2]-1:center[2]+2]
            if np.sum(patch) == 3:  # line
                ends_map[center[0], center[1], center[2]] = 1
            elif np.sum(patch) == 2:  # end point
                ends_map[center[0], center[1], center[2]] = 2
            elif np.sum(patch) > 3:  # branch point
                ends_map[center[0], center[1], center[2]] = 6
        return ends_map

    def crop_resample(arr, spacing, target_shape=(160, 160, 160)):
        arr = resize_volume(arr, spacing=spacing, target_spacing=(0.8, 0.8, 0.8), dtype='uint8')
        zd = np.ceil(max(0, target_shape[2] - arr.shape[2])).astype('int32')
        arr = np.pad(arr, ((0,0),(0,0),(zd,0)), 'constant')
        # xy center crop
        x0, y0 = np.floor(np.array(arr.shape[:2])/2).astype('int32') - np.array(target_shape[:2])//2
        arr = arr[x0:x0+target_shape[0], y0:y0+target_shape[1], -target_shape[2]:]
        return arr

    def back_to_origin(arr, spacing, target_shape=vessel_mask.shape):
        arr = resize_volume(arr, spacing=(0.8, 0.8, 0.8), target_spacing=spacing, dtype='uint8')
        p0 = int((target_shape[0] - arr.shape[0]) // 2)
        p1 = int((target_shape[1] - arr.shape[1]) // 2)
        p2 = max(0, int(target_shape[2] - arr.shape[2]))
        arr = np.pad(arr, ((p0,p0+1),(p1,p1+1),(p2,0)), 'constant')
        arr = arr[:target_shape[0], :target_shape[1], -target_shape[2]:]
        return arr

    # run
    if verbose: print(">>> predict_vessel_16labels ", end='')
    vessel_mask_crop = crop_resample(vessel_mask, spacing).astype("float32")
    vessel_16labels_crop = model3.serve(vessel_mask_crop[np.newaxis,:,:,:,np.newaxis]).numpy()[0]
    vessel_16labels_crop[:, :, 0] = -1  # ignore background 0
    vessel_16labels_crop = np.argmax(vessel_16labels_crop, axis=-1).astype('uint8')
    vessel_16labels = back_to_origin(vessel_16labels_crop, spacing)

    # post-process
    if post_proc:  # post-proc2
        # reduce noise of preds
        preds = np.zeros_like(vessel_16labels)
        for lb in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 18]:  # ignore 17(branch)
            mask = ndi.binary_erosion(vessel_16labels==lb)
            mask = ndi.binary_fill_holes(mask)
            preds[mask&(preds==0)] = lb
            if verbose: print(".", end='')

        # filter with skeleton fragments
        ends_map = find_end_points(skeletonize(vessel_mask) > 0)
        fragments_markers = label(ends_map==1, connectivity=3)
        # watershed fragments
        distance = ndi.distance_transform_edt(vessel_mask, sampling=spacing)
        fragments = watershed(-distance, fragments_markers, mask=vessel_mask)

        markers = np.zeros_like(fragments_markers, dtype="uint8")
        for idx in range(1, fragments_markers.max()):
            masked_pred = preds[fragments==idx]
            unique, counts = np.unique(masked_pred[masked_pred > 0], return_counts=True)
            filter_idx = np.where(counts > min_size)[0]
            unique, counts = unique[filter_idx], counts[filter_idx]
            if len(unique) > 0:
                markers[fragments_markers==idx] = unique[np.argmax(counts)]
            if verbose:
                if idx % 50 == 0: print(".", end='')
        # watershed vessels
        vessel_16labels = watershed(-distance, markers, mask=vessel_mask)
        vessel_16labels[vessel_16labels==18] = 0  # exclude
    else:
        markers = vessel_16labels
        # watershed vessels
        distance = ndi.distance_transform_edt(vessel_mask, sampling=spacing)
        vessel_16labels = watershed(-distance, markers, mask=vessel_mask)

    # add one pixel annotation
    for lb in range(1, 16+1):
        vessel_16labels[lb, lb, -lb] = lb
        vessel_16labels[-lb-1, -lb-1, -lb-1] = lb

    if verbose: print(">>>", vessel_16labels.shape)
    return vessel_16labels

#新增加修正vessel_16labels
def modify_vessel_16labels(vessel_16labels, spacing):
    new_vessel_mask = vessel_16labels > 0
    # regenrate vessel_seed
    new_vessel_seed = binary_erosion(new_vessel_mask, get_struc(2.5, spacing))  # 2.5, 3.0

    # select connected vessels via seeds
    label_map = label(new_vessel_mask)
    lbs = np.unique(label_map[new_vessel_seed])
    lbs = [lb for lb in lbs if lb != 0]  # drop zero
    new_vessel_mask = np.isin(label_map, lbs)
    new_vessel_mask = remove_small_objects(new_vessel_mask, min_size=500)

    return new_vessel_mask.astype('uint8') * vessel_16labels, new_vessel_mask

@tf.function
def run_model2(x, model2, importance_map, **kwargs):
    x = tf.reshape(x, (-1, 32, 32, 16, 1))
    pred1 = model2.serve(x)  # segment conf.
    return pred1 * importance_map

@tf.function
def run_model_batch(x, model, importance_map):
    """批量推論並乘以重要性權重"""
    pred = model.serve(x)  # (B, D, H, W, n_class)
    return pred * importance_map  # 自動廣播到 batch 維

#@title model predict aneurysm
def predict_aneurysm(image_arr, brain_mask, vessel_16labels, spacing,
                     model2, patch_size=(32, 32, 16), sigma=0.125, n_class=1, overlap=0.75,
                     conf_th=0.1, min_diameter=1, top_k=4, obj_th=0.67, verbose=False):

    # resampling
    target_spacing = (spacing[0], spacing[1], spacing[0])  # isotropic
    image_iso = resize_volume(image_arr, spacing=spacing, target_spacing=target_spacing, dtype='float32')
    vessel_iso = resize_volume(vessel_16labels, target_size=image_iso.shape, dtype='bool')

    def sliding_window_inference2(inputs, roi_size, model, overlap, n_class, importance_map, strategy="overlap_inside", mask=None, **kwargs):
        image_size = tuple(inputs.shape[1:-1])
        roi_size = tuple(roi_size)
        # Padding to make sure that the image size is at least roi size
        padded_image_size = tuple(max(image_size[i], roi_size[i]) for i in range(3))
        padding_size = [image_x - input_x for image_x, input_x in zip(image_size, padded_image_size)]
        paddings = [[0, 0]] + [[x // 2, x - x // 2] for x in padding_size] + [[0, 0]]
        input_padded = tf.pad(inputs, paddings)
        if mask is not None:
            mask_padded = tf.pad(tf.convert_to_tensor(mask, dtype=tf.bool), paddings)

        output_shape = (1, *padded_image_size, n_class)
        output_sum = tf.zeros(output_shape, dtype=tf.float32)
        output_weight_sum = tf.ones(output_shape, dtype=tf.float32)
        window_slices = get_window_slices(padded_image_size, roi_size, overlap, strategy)

        for i, window_slice in enumerate(window_slices):
            if (mask is None) or ((mask is not None)and(tf.math.reduce_any(tf.slice(mask_padded, begin=window_slice[0], size=window_slice[1])))):
                window = tf.slice(input_padded, begin=window_slice[0], size=window_slice[1])
                pred = run_model2(window, model, importance_map, **kwargs)
                padding = [
                    [start, output_size - (start + size)] for start, size, output_size in zip(*window_slice, output_shape)
                ]
                padding = padding[:-1] + [[0, 0]]
                output_sum = output_sum + tf.pad(pred, padding)
                output_weight_sum = output_weight_sum + tf.pad(importance_map, padding)
            if verbose:
                if i % 2000 == 0: print('.', end='')

        # output = output_sum / tf.clip_by_value(output_weight_sum, 1, 256)
        output = output_sum / output_weight_sum
        crop_slice = [slice(pad[0], pad[0] + input_x) for pad, input_x in zip(paddings, inputs.shape[:-1])]
        return output[crop_slice]

    # prepare importance_map
    if verbose: print(">>> predict_aneurysm ", end='')
    importance_kernel = get_importance_kernel(patch_size, blend_mode="gaussian", sigma=sigma)
    importance_map = tf.tile(tf.reshape(importance_kernel, shape=[1, *patch_size, 1]), multiples=[1, 1, 1, 1, n_class],)
    # sliding_window_inference
    pred_prob_map = sliding_window_inference2(inputs=custom_normalize_1(image_iso)[np.newaxis,:,:,:,np.newaxis],
                                              roi_size=patch_size, model=model2, overlap=overlap,
                                              n_class=n_class, importance_map=importance_map,
                                              mask=vessel_iso[np.newaxis,:,:,:,np.newaxis])
    pred_prob_map = pred_prob_map[0,:,:,:,0].numpy() * vessel_iso

    # back to original spacing
    pred_prob_map = resize_volume(pred_prob_map, target_size=image_arr.shape, dtype='float32')

    # object_analysis
    pred_label = label(pred_prob_map > conf_th)
    # pred_label measurement
    props = regionprops_table(pred_label, pred_prob_map, properties=('label', 'bbox', 'intensity_max', 'intensity_mean'))
    df_pred = pd.DataFrame({'ori_Pred_label':props['label'],
                            'Pred_diameter':((props['bbox-3']-props['bbox-0'])*spacing[0] + (props['bbox-4']-props['bbox-1'])*spacing[1]) / 2,
                            'Pred_max':props['intensity_max'],
                            'Pred_mean':props['intensity_mean'],})
    # sort and false-positive filtering
    df_pred = df_pred.sort_values(by='Pred_max', ascending=False)
    df_pred = df_pred[df_pred['Pred_diameter'] >= min_diameter]  # object size filter
    if top_k > 0: df_pred = df_pred[:top_k]  # top_k filter
    df_pred = df_pred[df_pred['Pred_max'] >= obj_th]  # object filter
    # remap pred_label array
    df_pred['Pred_label'] = np.arange(1, df_pred.shape[0]+1)
    new_pred_label = np.zeros_like(pred_label)
    for k, v in zip(df_pred['ori_Pred_label'].values, df_pred['Pred_label'].values):
        new_pred_label[pred_label==k] = v

    # get object location
#     if vessel_16labels.max() > 7:  # QC of vessel_16labels
    for idx in df_pred.index:
        masked_veseel = vessel_16labels[new_pred_label == df_pred.loc[idx, 'Pred_label']]
        unique, counts = np.unique(masked_veseel[masked_veseel > 0], return_counts=True)
        df_pred.loc[idx, 'vessel_16label'] = unique[np.argmax(counts)]

    if verbose: print(">>>", new_pred_label.shape)
    return pred_prob_map, df_pred, new_pred_label


def predict_aneurysm_best(image_arr, brain_mask, vessel_16labels, spacing,
                          model2, patch_size=(32,32,16), sigma=0.125, n_class=1,
                          overlap=0.75, conf_th=0.1, min_diameter=1, top_k=4,
                          obj_th=0.67, batch_size=300, verbose=False):
    """GPU 高效 + 安全 padding 版本"""

    # 1. === 預處理 ===
    target_spacing = (spacing[0], spacing[1], spacing[0])
    image_iso = resize_volume(image_arr, spacing=spacing,
                              target_spacing=target_spacing, dtype='float32')
    vessel_iso = resize_volume(vessel_16labels, target_size=image_iso.shape, dtype='bool')

    if verbose:
        print(f">>> image_iso: {image_iso.shape}, vessel_iso: {vessel_iso.shape}")

    # 2. === sliding window inference 函數 ===
    def sliding_window_inference_safe(inputs, roi_size, model, overlap, n_class,
                                      importance_map, mask=None, batch_size=300):
        image_size = tuple(inputs.shape[1:-1])  # (D,H,W)
        roi_size = tuple(roi_size)

        # === Padding ===
        padded_image_size = tuple(max(image_size[i], roi_size[i]) for i in range(3))
        padding_size = [padded_image_size[i] - image_size[i] for i in range(3)]
        paddings = [[0,0]] + [[x//2, x-x//2] for x in padding_size] + [[0,0]]
        input_padded = tf.pad(inputs, paddings)

        if mask is not None:
            mask_padded = tf.pad(tf.convert_to_tensor(mask, dtype=tf.bool), paddings)
        else:
            mask_padded = tf.ones_like(input_padded, dtype=tf.bool)

        output_shape = tf.concat([[1], tf.constant(padded_image_size, tf.int32), [n_class]], axis=0)
        output_sum = tf.zeros(output_shape, dtype=tf.float32)
        #output_weight_sum = tf.zeros(output_shape, dtype=tf.float32)
        output_weight_sum = tf.ones(output_shape, dtype=tf.float32) #上面才是正確，但君彥版本用這個

        window_slices = get_window_slices(padded_image_size, roi_size, overlap, strategy="overlap_inside")

        # 過濾掉沒有mask覆蓋的patch
        valid_slices = [(s, sz) for s, sz in window_slices
                        if tf.reduce_any(tf.slice(mask_padded, begin=s, size=sz))]

        batch_inputs = []
        batch_indices = []

        for i, (start, size) in enumerate(valid_slices):
            patch = tf.slice(input_padded, begin=start, size=size)[0]
            batch_inputs.append(patch)
            batch_indices.append((start, size))

            # 滿批或最後一次 -> 推論
            if len(batch_inputs) == batch_size or i == len(valid_slices)-1:
                batch_tensor = tf.stack(batch_inputs, axis=0)
                preds = model.serve(batch_tensor) * importance_map  # (B,D,H,W,n_class)

                # === 向量化累加 ===
                for j, (s, sz) in enumerate(batch_indices):
                    pred = preds[j]  # (D,H,W,n_class)

                    # 安全 padding 計算
                    padding = []
                    for dim in range(5):  # (B,D,H,W,C)
                        if dim == 0 or dim == 4:
                            padding.append([0,0])
                        else:
                            pad_before = s[dim]
                            pad_after = int(output_shape[dim]) - (s[dim] + sz[dim])
                            pad_after = max(pad_after, 0)  # 防止負數
                            padding.append([pad_before, pad_after])

                    pred_padded = tf.pad(pred[tf.newaxis], padding)
                    weight_padded = tf.pad(importance_map, padding)

                    output_sum += pred_padded
                    output_weight_sum += weight_padded

                batch_inputs = []
                batch_indices = []

        # 安全除零
        output = output_sum / tf.maximum(output_weight_sum, 1e-8)

        # Crop 回原圖大小
        crop_slice = [slice(pad[0], pad[0]+image_size[i])
                      for i, pad in enumerate(paddings[1:-1])]
        return output[(slice(0,1), *crop_slice, slice(0,n_class))]

    # 3. === importance map ===
    importance_kernel = get_importance_kernel(patch_size, blend_mode="gaussian", sigma=sigma)
    importance_map = tf.tile(tf.reshape(importance_kernel, shape=[1,*patch_size,1]),
                             [1,1,1,1,n_class])

    # 4. === sliding window inference ===
    pred_prob_map = sliding_window_inference_safe(
        inputs=custom_normalize_1(image_iso)[np.newaxis,...,np.newaxis],
        roi_size=patch_size,
        model=model2,
        overlap=overlap,
        n_class=n_class,
        importance_map=importance_map,
        mask=vessel_iso[np.newaxis,...,np.newaxis],
        batch_size=batch_size
    )

    pred_prob_map = pred_prob_map[0,...,0].numpy() * vessel_iso
    pred_prob_map = resize_volume(pred_prob_map, target_size=image_arr.shape, dtype='float32')

    # 5. === Post-processing ===
    pred_label = label(pred_prob_map > conf_th)
    props = regionprops_table(pred_label, pred_prob_map,
                              properties=('label', 'bbox', 'intensity_max', 'intensity_mean'))
    df_pred = pd.DataFrame({
        'ori_Pred_label': props['label'],
        'Pred_diameter': ((props['bbox-3'] - props['bbox-0']) * spacing[0] +
                          (props['bbox-4'] - props['bbox-1']) * spacing[1]) / 2,
        'Pred_max': props['intensity_max'],
        'Pred_mean': props['intensity_mean'],
    })

    df_pred = df_pred[df_pred['Pred_diameter'] >= min_diameter]
    df_pred = df_pred.sort_values(by='Pred_max', ascending=False)
    if top_k > 0:
        df_pred = df_pred[:top_k]
    df_pred = df_pred[df_pred['Pred_max'] >= obj_th]

    df_pred['Pred_label'] = np.arange(1, df_pred.shape[0]+1)
    new_pred_label = np.zeros_like(pred_label)
    for k, v in zip(df_pred['ori_Pred_label'].values, df_pred['Pred_label'].values):
        new_pred_label[pred_label == k] = v

    for idx in df_pred.index:
        masked_vessel = vessel_16labels[new_pred_label == df_pred.loc[idx, 'Pred_label']]
        unique, counts = np.unique(masked_vessel[masked_vessel > 0], return_counts=True)
        df_pred.loc[idx, 'vessel_16label'] = unique[np.argmax(counts)] if len(unique)>0 else -1

    if verbose:
        print(">>> Done.", new_pred_label.shape)

    return pred_prob_map, df_pred, new_pred_label


#@title save nifti，這邊是用我的
def save_nii(path_process, brain_mask, pred_vessel, vessel_16labels, pred_label, pred_prob_map, out_dir='test_label_out', verbose=False):
    ## load original image
    img = nib.load(os.path.join(path_process, 'MRA_BRAIN.nii.gz'))
    img = nib.as_closest_canonical(img)  # to RAS space
    aff = img.affine
    hdr = img.header
    spacing = tuple(img.header.get_zooms())
    shape = tuple(img.header.get_data_shape())
    original_size = tuple(img.header.get_data_shape())
    #nib.save(img, f"{out_dir}/image.nii.gz")
    #print("[save] --->", f"{out_dir}/image.nii.gz")
    if verbose:
        image_arr = img.get_fdata().astype('int16')
        print(f"shape={shape} spacing={spacing}  {image_arr.dtype}:{image_arr.min()}-{image_arr.max()}")
        print("affine matrix =\n", aff)

    # Make sure the dimensions are the same as the original image
    if brain_mask.shape != original_size:
        brain_mask = resize_volume(brain_mask, target_size=original_size, dtype='uint8')
    if pred_vessel.shape != original_size:
        pred_vessel = resize_volume(pred_vessel, target_size=original_size, dtype='uint8')        
    if vessel_16labels.shape != original_size:
        vessel_16labels = resize_volume(vessel_16labels, target_size=original_size, dtype='uint8')
    if pred_label is not None:
        if pred_label.shape != original_size:
            pred_label = resize_volume(pred_label, target_size=original_size, dtype='uint8')
    if pred_prob_map is not None:
        if pred_prob_map.shape != original_size:
            pred_prob_map = resize_volume(pred_prob_map, target_size=original_size, dtype='uint8')

    ## brain_mask
    brain_mask = brain_mask[::-1, ::-1, :].astype('uint8')  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(brain_mask, affine=aff)
    nib.save(x, f"{out_dir}/brain_mask.nii.gz")
    print("[save] --->", f"{out_dir}/brain_mask.nii.gz")

    ## pred_vessel
    # pred_vessel = pred_vessel[::-1, ::-1, :].astype('uint8')  # LPS to RAS orientation
    # x = nib.nifti1.Nifti1Image(pred_vessel, affine=aff)
    # nib.save(x, f"{out_dir}/pred_vessel_mask.nii.gz")
    # print("[save] --->", f"{out_dir}/pred_vessel_mask.nii.gz")

    ## pred_vessel
    # build NifTi1 image
    pred_vessel = pred_vessel[::-1, ::-1, :].astype('uint8')  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(pred_vessel, affine=aff)
    nib.save(x, f"{out_dir}/Vessel.nii.gz")
    print("[save] --->", f"{out_dir}/Vessel.nii.gz")

    # pred_vessel_16labels
    vessel_16labels = vessel_16labels[::-1, ::-1, :].astype('uint8')  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(vessel_16labels, affine=aff)
    nib.save(x, f"{out_dir}/Vessel_16.nii.gz")
    print("[save] --->", f"{out_dir}/Vessel_16.nii.gz")

    ## pred_label
    pred_label = pred_label[::-1, ::-1, :].astype('uint8')  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(pred_label, affine=aff)
    nib.save(x, f"{out_dir}/Pred.nii.gz")
    print("[save] --->", f"{out_dir}/Pred.nii.gz")
    
    #存出predict map
    pred_prob_map = pred_prob_map[::-1, ::-1, :]  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(pred_prob_map, affine=aff)
    nib.save(x, f"{out_dir}/Prob.nii.gz")
    print("[save] --->", f"{out_dir}/Prob.nii.gz")
    return 

#@title save nifti，這邊是用我的
def save_nii_preprocess(path_process, image_arr, pred_vessel, vessel_16labels, out_dir='test_label_out', verbose=False):
    ## load original image
    #建立分別放正規化影像與vessel的資料夾
    if not os.path.isdir(os.path.join(out_dir, 'Normalized_Image')):
        os.mkdir(os.path.join(out_dir, 'Normalized_Image'))
    if not os.path.isdir(os.path.join(out_dir, 'Vessel')):
        os.mkdir(os.path.join(out_dir, 'Vessel'))

    img = nib.load(os.path.join(path_process, 'MRA_BRAIN.nii.gz'))
    img = nib.as_closest_canonical(img)  # to RAS space
    aff = img.affine
    hdr = img.header
    spacing = tuple(img.header.get_zooms())
    shape = tuple(img.header.get_data_shape())
    original_size = tuple(img.header.get_data_shape())
    #nib.save(img, f"{out_dir}/image.nii.gz")
    #print("[save] --->", f"{out_dir}/image.nii.gz")

    # Make sure the dimensions are the same as the original image
    if image_arr.shape != original_size:
        image_arr = resize_volume(image_arr, target_size=original_size, dtype='uint8')  
    if pred_vessel.shape != original_size:
        pred_vessel = resize_volume(pred_vessel, target_size=original_size, dtype='uint8')  
    if vessel_16labels.shape != original_size:
        vessel_16labels = resize_volume(vessel_16labels, target_size=original_size, dtype='uint8')

    #做正規化
    image_arr = custom_normalize_1(image_arr)  

    # build NifTi1 image
    image_arr = image_arr[::-1, ::-1, :]  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(image_arr, affine=aff, header=hdr)
    nib.save(x, os.path.join(out_dir, 'Normalized_Image', 'DeepAneurysm_00001_0000.nii.gz'))
    print("[save] --->", os.path.join(out_dir, 'Normalized_Image', 'DeepAneurysm_00001_0000.nii.gz'))    

    ## pred_vessel
    # build NifTi1 image
    pred_vessel = pred_vessel[::-1, ::-1, :].astype('uint8')  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(pred_vessel, affine=aff, header=hdr)
    nib.save(x, os.path.join(out_dir, 'Vessel', 'DeepAneurysm_00001_0000.nii.gz'))
    print("[save] --->", os.path.join(out_dir, 'Vessel', 'DeepAneurysm_00001_0000.nii.gz'))
    nib.save(x, f"{out_dir}/Vessel.nii.gz")
    print("[save] --->", f"{out_dir}/Vessel.nii.gz")

    # pred_vessel_16labels
    vessel_16labels = vessel_16labels[::-1, ::-1, :].astype('uint8')  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(vessel_16labels, affine=aff, header=hdr)
    nib.save(x, f"{out_dir}/Vessel_16.nii.gz")
    print("[save] --->", f"{out_dir}/Vessel_16.nii.gz")
    return 

#篩選掉threshold不到的動脈瘤
def filter_aneurysm(pred_prob_map, spacing, conf_th=0.1, min_diameter=2, top_k=4, obj_th=0.67):
    # object_analysis
    pred_label = label(pred_prob_map > conf_th)
    # pred_label measurement，取得pred的各種屬性數值，例如intensity_max, intensity_max
    props = regionprops_table(pred_label, pred_prob_map, properties=('label', 'bbox', 'intensity_max', 'intensity_mean'))
    df_pred = pd.DataFrame({'ori_Pred_label':props['label'],
                            'Pred_diameter':((props['bbox-3']-props['bbox-0'])*spacing[0] + (props['bbox-4']-props['bbox-1'])*spacing[1]) / 2,
                            'Pred_max':props['intensity_max'],
                            'Pred_mean':props['intensity_mean'],})
    
    df_pred = df_pred[(df_pred['Pred_diameter'] >= min_diameter)]  # filter too small object
    df_pred = df_pred.sort_values(by='Pred_max', ascending=False) #用最大強度來排序
    df_pred['Pred_label'] = np.arange(1, df_pred.shape[0]+1)
    if top_k > 0:
        df_pred = df_pred[:top_k]  # top_k filter
    df_pred = df_pred[df_pred['Pred_max'] >= obj_th]  # object filter
    # remap pred_label array
    new_pred_label = np.zeros_like(pred_label)
    for k, v in zip(df_pred['ori_Pred_label'].values, df_pred['Pred_label'].values):
        new_pred_label[pred_label==k] = v
        
    return pred_prob_map, df_pred, new_pred_label.astype(int)

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

#nii改變影像後儲存，data => nib.load的輸出，new_img => 更改的影像
def nii_img_replace(data, new_img):
    affine = data.affine
    header = data.header.copy()
    new_nii = nib.nifti1.Nifti1Image(new_img, affine, header=header)
    return new_nii


#"主程式"
#model_predict_aneurysm(path_code, path_process, case_name, path_log, gpu_n)
def model_predict_aneurysm(path_code, path_process, path_nnunet_model, case_name, path_log, gpu_n):
    path_model = os.path.join(path_code, 'model_weights')

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

    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT)

    def case_json(json_file_path, ret, code, msg):
        json_dict = OrderedDict()
        json_dict["ret"] = ret #使用的程式是哪一支python api
        json_dict["code"] = code #目前程式執行的結果狀態 0: 成功 1: 失敗
        json_dict["msg"] = msg #描述狀態訊息
       
        with open(json_file_path, 'w', encoding='utf8') as json_file:
            json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示

    try:
        logging.info('!!! ' + case_name + ' gpu_aneurysm call.')

        #把一些json要記錄的資訊弄出空集合或欲填入的值，這樣才能留空資訊
        ret = "gpu_aneurysm.py " #使用的程式是哪一支python api
        code_pass = 0 #確定是否成功
        msg = "ok" #描述狀態訊息
        
        #%% Deep learning相關
        pynvml.nvmlInit() #初始化
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)#获取GPU i的handle，后续通过handle来处理
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息
        gpumRate = memoryInfo.used/memoryInfo.total
        #print('gpumRate:', gpumRate) #先設定gpu使用率小於0.2才跑predict code
    
        if gpumRate < 0.6 :
            #plt.ion()    # 開啟互動模式，畫圖都是一閃就過
            #一些記憶體的配置
            autotune = tf.data.experimental.AUTOTUNE
            #print(keras.__version__)
            #print(tf.__version__)
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
            tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type='GPU')
            #print(gpus, cpus)
            tf.config.experimental.set_memory_growth(gpus[0],True)

            #%%
            #底下正式開始predict任務
            verbose = True
            MRA_BRAIN_file = os.path.join(path_process, 'MRA_BRAIN.nii.gz')

            # 1 load image
            image_arr, spacing, _ = load_volume(MRA_BRAIN_file, dtype='int16')

            # 1.5 load synthseg model
            #@title load SynthSeg base model
            #model_root = '/content/gdrive/Shareddrives/2024_雙和_Aneurysm/code/SynthSeg_inference'  # google drive
            model_root = os.path.join(path_model, 'SynthSeg_inference')
            # model_root = 'SynthSeg_inference'  # server
            # load model0 (unet2)
            model0 = tf.saved_model.load(f"{model_root}/saved_model/net_unet2")
            model0.trainable = False
            model0.jit_compile = True

            #@title load vessel seg model
            #model_root = '/content/gdrive/Shareddrives/2024_雙和_Aneurysm/model/dataV2-brain_vessel-PosSample-ResUnet'  # google drive
            model_root = os.path.join(path_model, 'dataV2-brain_vessel-PosSample-ResUnet')
            # model_root = '../model/dataV2-brain_vessel-PosSample-ResUnet'  # server
            model_name = "MP-CustomNorm1-vessel_sampling_pos.9_64x64x32_b1x48-ResUnet_F32L4BN_mish_1ch-BCE_dice_ema" #我有改短名稱
            model1 = tf.saved_model.load(f"{model_root}/{model_name}/best_saved_model")
            model1.trainable = False
            model1.jit_compile = True

            #@title load vessel 16labels model
            #model_root = '/content/gdrive/Shareddrives/2024_雙和_Aneurysm/model/dataV2-Vessel_16label-ResUnet'  # google drive
            model_root = os.path.join(path_model, 'dataV2-Vessel_16label-ResUnet')
            # model_root = '../model/dataV2-Vessel_16label-ResUnet'  # server
            model_name = "MP-Vessel_16label+branch_PostProc2AIAA_Aug2of4_160_b4-ResUnet_8f_L5_BN_mish_19ch_ema_cw" #我有改短名稱
            model3 = tf.saved_model.load(f"{model_root}/{model_name}/best_saved_model")
            model3.trainable = False
            model3.jit_compile = True

            #@title load aneurysm seg model
            #model_root = '/content/gdrive/Shareddrives/2024_雙和_Aneurysm/model/aneurysm_seg_model_v3-restart'  # google drive
            model_root = os.path.join(path_model, 'aneurysm_seg_model_v3-restart')
            # model_root = "../model/aneurysm_seg_model_v3-restart"  # server
            model_name = "MP-FEMHv3_SHHv3_P3_iso-b8-FCresunet32-px_size_sw-Adam1E4_ema"  # iso
            model2 = tf.saved_model.load(f"{model_root}/{model_name}/best_saved_model")
            model2.trainable = False
            model2.jit_compile = True

            # 2 Get brain mask，先全照君彥pipeline，來不及拉!!!
            brain_seg = get_brain_seg(image_arr, spacing, model0)
            brain_bottom_idx = np.where(np.any(brain_seg > 0, axis=(0,1)))[0][0]
            bet_brain_mask = np.zeros_like(image_arr, dtype=bool)
            bet_brain_mask[:,:,brain_bottom_idx:] = get_BET_brain_mask(image_arr[:,:,brain_bottom_idx:], spacing, bet_iter=1000)
            brain_mask = modify_brain_mask((brain_seg > 0)|(bet_brain_mask), spacing, verbose=verbose)
            del brain_seg, bet_brain_mask
                
            # 3 Get vessel mask
            vessel_threshold, _ = VesselSegmenter().threshold_segmentation(image_arr, brain_mask, spacing)
            vessel_seed = get_vessel_seed(vessel_threshold, mask=brain_mask, spacing=spacing)
            pred_vessel_mask = predict_vessel(image_arr, brain_mask, model1, verbose=verbose)
            vessel_mask = combine_two_vessels(vessel_threshold, pred_vessel_mask, seed_mask=vessel_seed, brain_mask=brain_mask)
            del vessel_threshold, vessel_seed, pred_vessel_mask, brain_bottom_idx

            # 3.5 Get vessel skeleton
            #skeleton_labels = get_vessel_skeleton_labels(vessel_mask, spacing)

            # 4 get vessel 16labels
            vessel_16labels = predict_vessel_16labels(vessel_mask, model3, spacing, verbose=verbose)
            vessel_16labels, vessel_mask = modify_vessel_16labels(vessel_16labels, spacing)  # 20250716 add

            # 5 Pred aneurysm，從這一步開始，底下置換成nnU-Net的model，先把正規化的image跟vessel mask存出，準備放入nnU-Net中
            # 5.1 先存出正規化的影像跟血管
            save_nii_preprocess(path_process, image_arr, vessel_mask, vessel_16labels, out_dir=path_process) 
            path_normimg = os.path.join(path_process, 'Normalized_Image')
            path_vessel = os.path.join(path_process, 'Vessel')

            predict_from_raw_data(path_normimg,
                                  path_vessel,
                                  path_process,
                                  path_nnunet_model,
                                  (13,),
                                  0.25,
                                  use_gaussian=True,
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
                                  batch_size=112
                                 )
            
            #複製inference result
            shutil.copy(os.path.join(path_process, 'DeepAneurysm_00001.nii.gz'), os.path.join(path_process, 'Prob.nii.gz')) #取threshold跟cluster放到後面做
            # shutil.copy(os.path.join(path_process, 'DeepAneurysm_00001.nii.gz'), os.path.join(path_nnunetlow, 'Prob.nii.gz')) #取threshold跟cluster放到後面做

            #用threshold修改輸出
            prob_nii = nib.load(os.path.join(path_process, 'Prob.nii.gz'))
            prob = np.array(prob_nii.dataobj) #讀出label的array矩陣      #256*256*22   
            prob = data_translate(prob, prob_nii)
        
            #改讀取mifti的pixel_size資訊
            header_true = prob_nii.header.copy() #抓出nii header 去算體積 
            pixdim = header_true['pixdim']  #可以借此從nii的header抓出voxel size
        
            spacing_nn = [pixdim[1], pixdim[2]]
        
            pred_prob_map, df_pred, new_pred_label = filter_aneurysm(prob, spacing_nn, conf_th=0.1, min_diameter=2, top_k=4, obj_th=0.65)
        
            #最後存出新mask，存出nifti
            new_pred_label = data_translate_back(new_pred_label, prob_nii).astype(int)
            new_pred_label_nii = nii_img_replace(prob_nii, new_pred_label)
            nib.save(new_pred_label_nii, os.path.join(path_process, 'Pred.nii.gz'))  

            #舊版君彥inference model
            start_tensorflow = time.time()
            #pred_prob_map, df_pred, pred_label = predict_aneurysm_best(image_arr, brain_mask, vessel_mask, spacing, model2, verbose=verbose)
            print(f"[Done AI Inference... ] spend {time.time() - start_tensorflow:.0f} sec")
            logging.info(f"[Done AI Inference...  ] spend {time.time() - start_tensorflow:.0f} sec")
            #以json做輸出
            time.sleep(1)
            logging.info('!!! ' + str(case_name) +  ' gpu_aneurysm finish.')

        else:
            logging.error('!!! ' + str(case_name) + ' Insufficient GPU Memory.')
            
            # #刪除資料夾
            # if os.path.isdir(path_process):  #如果資料夾存在
            #     shutil.rmtree(path_process) #清掉整個資料夾 

    except:
        logging.error('!!! ' + str(case_name) + ' gpu have error code.')
        logging.error("Catch an exception.", exc_info=True)

        #刪除資料夾
        # if os.path.isdir(path_process):  #如果資料夾存在
        #     shutil.rmtree(path_process) #清掉整個資料夾
 
    return

#其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    #model_predict_aneurysm(path_code, path_processID, ID, path_log, gpu_n)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_code', type=str, help='目前執行的code')
    parser.add_argument('--path_process', type=str, help='目前執行的資料夾')
    parser.add_argument('--path_nnunet_model', type=str, help='nnU-Net model路徑')
    parser.add_argument('--case', type=str, help='目前執行的case的ID')
    parser.add_argument('--path_log', type=str, help='log資料夾')
    parser.add_argument('--gpu_n', type=int, help='第幾顆gpu')
    args = parser.parse_args()

    path_code = str(args.path_code)
    path_process = str(args.path_process)
    path_nnunet_model = str(args.path_nnunet_model)
    case_name = str(args.case)
    path_log = str(args.path_log)
    gpu_n = args.gpu_n

    model_predict_aneurysm(path_code, path_process, path_nnunet_model, case_name, path_log, gpu_n)
