"""PCA / BRI / SAT ağırlık haritaları."""

import cv2
import numpy as np


def compute_pca_map(patch):
    """-> (n,n,3) float [0,1], 1'den çıkarılmış"""
    pass


def compute_bri_map(patch):
    """-> (n,n,3) float [0,1], ters çevrilmez"""
    pass


def compute_sat_map(patch):
    """-> (n,n,3) float [0,1], 1'den çıkarılmış"""
    pass


def enhance_patch(patch, maps):
    """Haritaları eleman bazlı çarp. maps=() -> patch değişmeden döner.
    -> (n,n,3) uint8"""
    pass