"""Weighted Bag of Visual Words: sözlük ve histogram."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist


def train_dictionary(features_list, labels, k):
    """Benign ve malignant AYRI kümelenir, her biri k//2 merkez.
    MiniBatchKMeans. SADECE TRAIN.
    -> (1000, 2048) tek array: satır 0-499 benign, 500-999 malignant"""
    pass


def build_histogram(features, centers, w_mal, w_ben):
    """Tüm 1000 merkezde global en yakın arama (cdist + argmin),
    index < 500 -> w_ben, index >= 500 -> w_mal,
    l2 normalize -> (K,) array"""

    def build_histogram(features, centers, w_mal, w_ben):
        """Global en yakın merkez (kendi mesafe hesabımız + argmin),
        ağırlıklı sayım, l2 normalize -> (K,) array
        """
        k_total = centers.shape[0]
        half = k_total // 2

        histogram = np.zeros(k_total, dtype=np.float64)

        if features is None or len(features) == 0:
            return histogram

        distances = cdist(features, centers, metric="euclidean")
        nearest_idx = distances.argmin(axis=1)

        for idx in nearest_idx:
            weight = w_mal if idx >= half else w_ben
            histogram[idx] += weight

        norm = np.linalg.norm(histogram)
        if norm > 1e-8:
            histogram = histogram / norm

        return histogram

    pass