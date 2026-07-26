"""Veri seti listesi ve etiket okuma."""

import os
import csv


def load_split(csv_path, image_dir, mask_dir):
    """CSV oku -> [(img_path, mask_path, label), ...]   label: 0=benign, 1=malignant"""
    pass


def class_counts(samples):
    """-> {0: n_benign, 1: n_malignant}   sağlık kontrolü"""
    pass