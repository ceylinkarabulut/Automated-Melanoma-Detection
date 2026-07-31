"""Yeniden boyutlandırma ve patch çıkarma."""

import cv2
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view


def rescale_image(img, long_side, is_mask=False):
    H, W = img.shape[:2]
    if H > W:
        scale = long_side / H
    else:
        scale = long_side / W

    new_H = int(round(scale * H))
    new_W = int(round(scale * W))
    if is_mask:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR

    resized = cv2.resize(img, (new_W, new_H), interpolation=interpolation)
    return resized


def extract_patches(img, mask, n, stride, threshold):
    """Kayan pencere + maske filtresi -> [patch, ...]  her biri (n,n,3)"""
    windows = sliding_window_view(mask, (n, n))
    sampled = windows[::stride, ::stride]
    img_windows = sliding_window_view(img, (n, n), axis=(0, 1))
    img_sampled = img_windows[::stride, ::stride]
    img_sampled = np.moveaxis(img_sampled, 2, -1)
    ratios = np.mean(sampled == 255, axis=(2, 3))

    keep = ratios >= threshold

    selected = img_sampled[keep]

    patches = [p.astype(np.uint8) for p in selected]
    return patches


def compute_mh_long_side(samples):
    """
    ISIC 2017 için MH çözünürlüğü: (ortalama uzun kenar + max uzun kenar) / 2
    """
    long_sides = []

    for sample in samples:
        with Image.open(sample.img_path) as im:
            width, height = im.size

        if width > height:
            long_side_val = width
        else:
            long_side_val = height

        long_sides.append(long_side_val)

    mean_val = sum(long_sides) / len(long_sides)
    max_val = max(long_sides)

    mh = (mean_val + max_val) / 2
    return int(round(mh))
