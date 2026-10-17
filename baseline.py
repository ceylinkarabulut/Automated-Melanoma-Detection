"""BoVW'yi kontrol etmek için iki basit karşılaştırma (aynı train/test, aynı SVM CV ayarı).

    python baseline.py global   -> lezyonun bounding box'ı kırpılır, tüm görüntü ResNet'ten tek vektör olarak geçer
    python baseline.py meanpool -> main.py'nin cache'indeki patch feature'larının ortalaması (BoVW yerine)

meanpool için önce aynı FEATURE_LAYER / ENHANCEMENT_MAPS ile main.py çalışmış olmalı.
Sonuçlar results/ altına main.py ile aynı formatta yazılır.
"""

import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

import config
from augmentation import balance_train_set
from classifier import tune_svm, predict
from dataset import load_split
from evaluation import compute_metrics
from feature_extraction import build_model, build_preprocess, extract_features
from main import load_image_and_mask, source_image_ids

CROP_MARGIN = 0.1   # bounding box'ın her yönüne eklenen pay (kenar uzunluğuna oranla)


def crop_lesion(image, mask, size):
    """Maskenin bounding box'ını (pay ile) kırpar, kare olacak şekilde genişletir, size x size'a indirger."""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        y0, y1, x0, x1 = 0, image.shape[0], 0, image.shape[1]
    else:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * CROP_MARGIN))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h, w = image.shape[:2]
    top, left = max(cy - side // 2, 0), max(cx - side // 2, 0)
    bottom, right = min(top + side, h), min(left + side, w)
    crop = image[top:bottom, left:right]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def global_features(triples, model, preprocess):
    cache_dir = os.path.join(config.CACHE_DIR, "global")
    os.makedirs(cache_dir, exist_ok=True)
    features = []
    for image_path, mask_path, _ in triples:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        cache_path = os.path.join(cache_dir, f"{image_id}__{config.FEATURE_LAYER}.npy")
        if os.path.exists(cache_path):
            f = np.load(cache_path)
        else:
            image, mask = load_image_and_mask(image_path, mask_path)
            crop = crop_lesion(image, mask, config.PATCH_SIZE)
            f = extract_features([crop], model, preprocess, 1)[0]
            np.save(cache_path, f)
        features.append(f)
    return np.array(features)


def meanpool_features(triples):
    maps_str = "_".join(config.ENHANCEMENT_MAPS) if config.ENHANCEMENT_MAPS else "nomap"
    features = []
    for image_path, _, _ in triples:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        cache_path = os.path.join(config.CACHE_DIR, f"{image_id}__{maps_str}__{config.FEATURE_LAYER}.npy")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"{cache_path} yok: önce aynı ayarlarla main.py çalıştırılmalı")
        f = np.load(cache_path)
        features.append(f.mean(axis=0) if len(f) else np.zeros(config.FEATURE_DIM, dtype=np.float32))
    X = np.array(features)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def main(mode):
    train_data = load_split(config.TRAIN_CSV, config.TRAIN_IMAGE_DIR, config.TRAIN_MASK_DIR)
    test_data = load_split(config.TEST_CSV, config.TEST_IMAGE_DIR, config.TEST_MASK_DIR)
    os.makedirs(config.AUG_DIR, exist_ok=True)
    train_data = balance_train_set(train_data, config.AUG_DIR, config.SEED)

    if mode == "global":
        model, preprocess = build_model(), build_preprocess()
        X_train = global_features(train_data, model, preprocess)
        X_test = global_features(test_data, model, preprocess)
        X_train /= np.maximum(np.linalg.norm(X_train, axis=1, keepdims=True), 1e-8)
        X_test /= np.maximum(np.linalg.norm(X_test, axis=1, keepdims=True), 1e-8)
    elif mode == "meanpool":
        X_train, X_test = meanpool_features(train_data), meanpool_features(test_data)
    else:
        raise ValueError("mode: global | meanpool")

    y_train = np.array([label for _, _, label in train_data])
    y_test = np.array([label for _, _, label in test_data])

    svm_model, svm_params, cv_score = tune_svm(
        X_train, y_train, source_image_ids(train_data), config.SVM_C_GRID, config.SVM_GAMMA_GRID,
        config.SVM_CV_FOLDS, config.SVM_CV_SCORING, config.SEED
    )
    print(f"En iyi SVM: {svm_params} (CV {config.SVM_CV_SCORING} = {cv_score:.4f})")
    y_pred, y_scores = predict(svm_model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_scores)
    print(metrics)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    maps_str = "_".join(config.ENHANCEMENT_MAPS) if config.ENHANCEMENT_MAPS else "nomap"
    tag = maps_str if mode == "meanpool" else "crop"
    path = os.path.join(config.RESULTS_DIR, f"{datetime.now():%Y%m%d_%H%M%S}__baseline_{mode}__{config.FEATURE_LAYER}__{tag}.json")
    with open(path, "w") as f:
        json.dump({
            "time": datetime.now().isoformat(timespec="seconds"),
            "baseline": mode,
            "feature_layer": config.FEATURE_LAYER,
            "enhancement_maps": list(config.ENHANCEMENT_MAPS) if mode == "meanpool" else [],
            "svm_params": svm_params,
            "svm_cv_scoring": config.SVM_CV_SCORING,
            "svm_cv_score": cv_score,
            "metrics": metrics,
        }, f, indent=2, default=str)
    print(f"Sonuçlar kaydedildi: {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "global")
