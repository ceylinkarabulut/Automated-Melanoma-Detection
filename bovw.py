"""Weighted Bag of Visual Words: sözlük ve histogram."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def train_dictionary(features_list, labels, k):
    """Benign ve malignant AYRI kümelenir, her biri k//2 merkez.
    MiniBatchKMeans. SADECE TRAIN.
    -> (kmeans_benign, kmeans_malignant)"""
    pass


def build_histogram(features, km_b, km_m, w_mal, w_ben):
    """Global en yakın merkez (transform + hstack + argmin),
    ağırlıklı sayım, l2 normalize -> (K,) array"""
    pass