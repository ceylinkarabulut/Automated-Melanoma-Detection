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
    """Global en yakın merkez (kendi mesafe hesabımız + argmin),
    ağırlıklı sayım, l2 normalize -> (K,) array

    features: (N, 2048) -- tek bir görüntünün patch feature'ları
    centers:  (K, 2048) -- train_dictionary çıktısı (önce benign, sonra malignant)
    w_mal, w_ben: malignant / benign merkeze düşen patch'lerin histogram ağırlığı
    """
    k_total = centers.shape[0]
    half = k_total // 2  # bu indeksten itibaren malignant merkezler başlar

    histogram = np.zeros(k_total, dtype=np.float64)

    if features is None or len(features) == 0:
        return histogram

    # KMeans objesi olmadığı için .transform() yok -- mesafeyi kendimiz hesaplıyoruz
    distances = cdist(features, centers, metric="euclidean")  # (N, K)
    nearest_idx = distances.argmin(axis=1)  # (N,) -- her patch'in en yakın merkezi

    for idx in nearest_idx:
        weight = w_mal if idx >= half else w_ben
        histogram[idx] += weight

    norm = np.linalg.norm(histogram)
    if norm > 1e-8:
        histogram = histogram / norm

    return histogram