# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:02:06 2019

@author: chuan
"""

#處理nii資料的整合式function檔

import os
import numpy as np
import nibabel as nib
import glob
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 

#nii改變影像後儲存，data => nib.load的輸出，new_img => 更改的影像
def nii_img_replace(data, new_img):
    affine = data.affine
    header = data.header.copy()
    new_nii = nib.nifti1.Nifti1Image(new_img, affine, header=header)
    return new_nii

