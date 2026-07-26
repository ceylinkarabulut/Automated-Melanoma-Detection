"""Yeniden boyutlandırma ve patch çıkarma."""

import cv2
import numpy as np


def rescale_image(img, long_side, is_mask=False):
    """Uzun kenarı long_side yap, en-boy oranını koru.
    is_mask=True -> INTER_NEAREST"""
    pass


def extract_patches(img, mask, n, stride, threshold):
    """Kayan pencere + maske filtresi -> [patch, ...]  her biri (n,n,3)"""
    pass