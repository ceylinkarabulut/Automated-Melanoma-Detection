"""Weighted Bag of Visual Words: sözlük ve histogram."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist


def train_dictionary(features_list, labels, k):
    benign_features = features_list[labels == 0]
    malignant_features = features_list[labels == 1]
    kmeans_benign = MiniBatchKMeans(n_clusters=k // 2)
    kmeans_benign.fit(benign_features)
    kmeans_malignant = MiniBatchKMeans(n_clusters=k // 2)
    kmeans_malignant.fit(malignant_features)
    centers = np.vstack([kmeans_benign.cluster_centers_, kmeans_malignant.cluster_centers_])
    return centers


def build_histogram(features, centers, w_mal, w_ben):
    """Tüm 1000 merkezde global en yakın arama (cdist + argmin),
    index < 500 -> w_ben, index >= 500 -> w_mal,
    l2 normalize -> (K,) array"""
    pass