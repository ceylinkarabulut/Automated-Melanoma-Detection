"""Pipeline akışı.

Planlanan sıra:
    1. load_split ile train ve test yüklenir
    2. balance_train_set SADECE train'e uygulanır
    3. Her görüntü: imread -> rescale -> extract_patches -> enhance_patch -> extract_features
    4. train_dictionary SADECE train feature'larıyla eğitilir
    5. build_histogram ile her görüntü K boyutlu vektöre indirgenir
    6. train_svm SADECE train ile eğitilir
    7. compute_metrics SADECE test üzerinde çalıştırılır
"""

import os

import cv2
import numpy as np

import config
from augmentation import balance_train_set
from dataset import load_split
from preprocessing import rescale_image, extract_patches
from patch_enhancement import (
    compute_pca_map,
    compute_bri_map,
    compute_sat_map,
    enhance_patch,
)
from feature_extraction import build_model, build_preprocess, extract_features
from bovw import train_dictionary, build_histogram
from classifier import train_svm, predict
from evaluation import compute_metrics

# config.ENHANCEMENT_MAPS içindeki isimleri gerçek harita fonksiyonlarına bağlar
MAP_FUNCTIONS = {
    "pca": compute_pca_map,
    "bri": compute_bri_map,
    "sat": compute_sat_map,
}


def compute_maps_for_patch(patch, map_names):
    """map_names (config.ENHANCEMENT_MAPS) -> [harita_array, ...] (enhance_patch'in beklediği format)"""
    return [MAP_FUNCTIONS[name](patch) for name in map_names]


def load_image_and_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Maske okunamadı: {mask_path}")
    return image, mask


def images_to_features(triples, model, preprocess):
    image_features = []
    image_labels = []

    for image_path, mask_path, label in triples:
        image, mask = load_image_and_mask(image_path, mask_path)

        image = rescale_image(image, config.MH_LONG_SIDE, is_mask=False)
        mask = rescale_image(mask, config.MH_LONG_SIDE, is_mask=True)

        patches = extract_patches(image, mask, config.PATCH_SIZE, config.STRIDE, config.LESION_THRESHOLD)
        enhanced_patches = [
            enhance_patch(p, compute_maps_for_patch(p, config.ENHANCEMENT_MAPS))
            for p in patches
        ]
        features = extract_features(enhanced_patches, model, preprocess, config.BATCH_SIZE)

        image_features.append(features)
        image_labels.append(label)

    return image_features, image_labels

def flatten_patch_features(image_features, image_labels):
    all_features = np.concatenate(image_features, axis=0)
    all_patch_labels = np.concatenate([
        np.full(len(feats), label) for feats, label in zip(image_features, image_labels)
    ])
    return all_features, all_patch_labels


def build_histograms(image_features, centers):
    return np.array([
        build_histogram(f, centers, w_mal=config.MALIGNANT_WEIGHT, w_ben=config.BENIGN_WEIGHT)
        for f in image_features
    ])


def main():
    train_data = load_split(config.TRAIN_CSV, config.IMAGE_DIR, config.MASK_DIR)
    test_data = load_split(config.TEST_CSV, config.IMAGE_DIR, config.MASK_DIR)

    os.makedirs(config.AUG_DIR, exist_ok=True)
    train_data = balance_train_set(train_data, config.AUG_DIR, config.SEED)

    model = build_model()
    preprocess = build_preprocess()

    train_features, train_labels = images_to_features(train_data, model, preprocess)
    test_features, test_labels = images_to_features(test_data, model, preprocess)

    # train_dictionary artık tüm train patch'lerini + patch-bazlı etiketleri
    # TEK array olarak bekliyor (ayrımı fonksiyonun içinde kendisi yapıyor!! :3 )
    all_train_features, all_train_patch_labels = flatten_patch_features(train_features, train_labels)
    centers = train_dictionary(all_train_features, all_train_patch_labels, k=config.K_CLUSTERS)

    X_train = build_histograms(train_features, centers)
    y_train = np.array(train_labels)

    X_test = build_histograms(test_features, centers)
    y_test = np.array(test_labels)

    svm_model = train_svm(X_train, y_train, config.SVM_C, config.SVM_GAMMA)

    y_pred, y_scores = predict(svm_model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_scores)

    print(metrics)


if __name__ == "__main__":
    main()