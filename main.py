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

import json
import os
from datetime import datetime

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
from feature_extraction import build_model, build_preprocess, load_or_extract_features
from bovw import train_dictionary, build_histogram
from classifier import train_svm, tune_svm, predict
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
        filename = os.path.basename(image_path)
        image_id = os.path.splitext(filename)[0]

        def enhanced_patches(image_path=image_path, mask_path=mask_path):
            # yalnızca cache'te yoksa çağrılır: okuma, rescale ve harita hesabı run'ın en pahalı kısmı
            image, mask = load_image_and_mask(image_path, mask_path)
            image = rescale_image(image, config.MH_LONG_SIDE, is_mask=False)
            mask = rescale_image(mask, config.MH_LONG_SIDE, is_mask=True)
            patches = extract_patches(image, mask, config.PATCH_SIZE, config.STRIDE, config.LESION_THRESHOLD)
            return [enhance_patch(p, compute_maps_for_patch(p, config.ENHANCEMENT_MAPS)) for p in patches]

        features = load_or_extract_features(
            image_id, enhanced_patches, model, preprocess, config.BATCH_SIZE, config.CACHE_DIR,
            config.ENHANCEMENT_MAPS, config.FEATURE_LAYER
        )

        image_features.append(features)
        image_labels.append(label)

    return image_features, image_labels

def flatten_patch_features(image_features, image_labels):
    all_features = np.concatenate(image_features, axis=0)
    all_patch_labels = np.concatenate([
        np.full(len(feats), label) for feats, label in zip(image_features, image_labels)
    ])
    return all_features, all_patch_labels


def source_image_ids(triples):
    """Augment edilmiş kopyayı kaynak görüntüsünün id'sine eşler (CV grupları için).
    ISIC_0000001_augmented_2.jpg -> ISIC_0000001"""
    ids = []
    for image_path, _, _ in triples:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        ids.append(image_id.split("_augmented_")[0])
    return np.array(ids)


def save_run(metrics, svm_params, cv_score):
    """Run'ın ayarlarını ve metriklerini results/ altına json olarak yazar,
    böylece ablation run'ları yan yana karşılaştırılabilir."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    run = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "feature_layer": config.FEATURE_LAYER,
        "enhancement_maps": list(config.ENHANCEMENT_MAPS),
        "k_clusters": config.K_CLUSTERS,
        "malignant_weight": config.MALIGNANT_WEIGHT,
        "svm_tuned": config.SVM_TUNE,
        "svm_params": svm_params,
        "svm_cv_scoring": config.SVM_CV_SCORING if config.SVM_TUNE else None,
        "svm_cv_score": cv_score,
        "metrics": metrics,
    }
    maps_str = "_".join(config.ENHANCEMENT_MAPS) if config.ENHANCEMENT_MAPS else "nomap"
    filename = f"{datetime.now():%Y%m%d_%H%M%S}__{config.FEATURE_LAYER}__{maps_str}.json"
    path = os.path.join(config.RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(run, f, indent=2, default=str)
    return path


def build_histograms(image_features, centers):
    return np.array([
        build_histogram(f, centers, w_mal=config.MALIGNANT_WEIGHT, w_ben=config.BENIGN_WEIGHT)
        for f in image_features
    ])


def main():
    train_data = load_split(config.TRAIN_CSV, config.TRAIN_IMAGE_DIR, config.TRAIN_MASK_DIR)
    test_data = load_split(config.TEST_CSV, config.TEST_IMAGE_DIR, config.TEST_MASK_DIR)

    os.makedirs(config.AUG_DIR, exist_ok=True)
    train_data = balance_train_set(train_data, config.AUG_DIR, config.SEED)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
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

    if config.SVM_TUNE:
        groups = source_image_ids(train_data)
        svm_model, svm_params, cv_score = tune_svm(
            X_train, y_train, groups, config.SVM_C_GRID, config.SVM_GAMMA_GRID,
            config.SVM_CV_FOLDS, config.SVM_CV_SCORING, config.SEED
        )
        print(f"En iyi SVM: {svm_params} (CV {config.SVM_CV_SCORING} = {cv_score:.4f})")
    else:
        svm_params = {"C": config.SVM_C, "gamma": config.SVM_GAMMA}
        svm_model = train_svm(X_train, y_train, config.SVM_C, config.SVM_GAMMA, config.SEED)
        cv_score = None

    y_pred, y_scores = predict(svm_model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_scores)

    print(metrics)
    print(f"Sonuçlar kaydedildi: {save_run(metrics, svm_params, cv_score)}")


if __name__ == "__main__":
    main()