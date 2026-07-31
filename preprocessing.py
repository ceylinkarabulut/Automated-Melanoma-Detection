"""Yeniden boyutlandırma ve patch çıkarma."""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def rescale_image(img, long_side, is_mask=False):
    H, W = img.shape[:2]
    if H>W:
        scale= long_side/H
    else:
        scale= long_side/W

    new_H = int(round(scale * H))
    new_W = int(round(scale * W))
    if is_mask:
     interpolation= cv2.INTER_NEAREST
    else:
     interpolation= cv2.INTER_LINEAR


    resized=  cv2.resize(img,( new_W, new_H), interpolation= interpolation)
    return resized



def extract_patches(img, mask, n, stride, threshold):
    """Kayan pencere + maske filtresi -> [patch, ...]  her biri (n,n,3)"""
    windows= sliding_window_view(mask, (n,n))
    sampled = windows[::stride, ::stride]
    img_windows= sliding_window_view(img, (n,n), axis=(0,1))
    img_sampled= img_windows[::stride, ::stride]
    img_sampled = np.moveaxis(img_sampled, 2, -1)
    ratios = np.mean(sampled == 255, axis=(2, 3))

    keep = ratios >= threshold


    selected = img_sampled[keep]




    patches = [p.astype(np.uint8) for p in selected]
    return patches

