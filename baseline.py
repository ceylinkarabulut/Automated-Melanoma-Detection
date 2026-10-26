"""BoVW'yi kontrol etmek için iki basit karşılaştırma (aynı train/test, aynı SVM CV ayarı).

    python baseline.py global   -> lezyonun bounding box'ı kırpılır, tüm görüntü ResNet'ten tek vektör olarak geçer
    python baseline.py meanpool -> main.py'nin cache'indeki patch feature'larının ortalaması (BoVW yerine)
    python baseline.py fusion   -> global vektör + BoVW histogramı yan yana (ikisi de l2 normalize)

global/fusion: backbone config.GLOBAL_BACKBONE, TTA config.GLOBAL_TTA.
config.TUNE_THRESHOLD: karar eşiği train'in out-of-fold skorlarıyla seçilir, iki sonuç da kaydedilir.

meanpool ve fusion için önce aynı FEATURE_LAYER / ENHANCEMENT_MAPS ile main.py çalışmış olmalı.
Sonuçlar results/ altına main.py ile aynı formatta yazılır.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.models as models

import config
from augmentation import balance_train_set
from classifier import tune_svm, tune_threshold, predict
from dataset import load_split
from evaluation import compute_metrics
from feature_extraction import build_model, build_preprocess, extract_features
from bovw import train_dictionary
from main import load_image_and_mask, source_image_ids, flatten_patch_features, build_histograms

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


def build_global_model(name):
    """torchvision backbone, sınıflandırma katmanı Identity -> (model, feature_dim)
    resnet101 main.py ile aynı modeli (FEATURE_LAYER'a göre) kullanır."""
    if name == "resnet101":
        return build_model(), config.FEATURE_DIM
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc, dim = torch.nn.Identity(), 2048
    elif name == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        model.classifier, dim = torch.nn.Identity(), 1920
    elif name == "efficientnet_v2_m":
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        model.classifier, dim = torch.nn.Identity(), 1280
    elif name == "convnext_base":
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        model.classifier[2], dim = torch.nn.Identity(), 1024
    else:
        raise ValueError(f"Bilinmeyen GLOBAL_BACKBONE: {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    return model, dim


def dihedral_variants(img):
    """4 döndürme x yansıtma = 8 varyant (dermoskopide lezyonun yönü anlamsız)."""
    variants = []
    for flip in (img, cv2.flip(img, 1)):
        for k in range(4):
            variants.append(np.ascontiguousarray(np.rot90(flip, k)))
    return variants


def global_cache_name(image_id):
    # resnet101 tek görüntü, ilk global run'ın cache adını korur
    if config.GLOBAL_BACKBONE == "resnet101" and not config.GLOBAL_TTA:
        return f"{image_id}__{config.FEATURE_LAYER}.npy"
    layer = config.FEATURE_LAYER if config.GLOBAL_BACKBONE == "resnet101" else "pool"
    return f"{image_id}__{config.GLOBAL_BACKBONE}__{layer}__{'tta' if config.GLOBAL_TTA else 'single'}.npy"


def lesion_crop(image_path, mask_path):
    """Kırpılmış lezyon (PATCH_SIZE x PATCH_SIZE) diskte cache'lenir: tam çözünürlüklü JPEG'i açmak
    run'ın en pahalı CPU işi, kırpma backbone ve TTA'dan bağımsız olduğu için bir kez yapılması yeter."""
    crop_dir = os.path.join(config.CACHE_DIR, "crops")
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    crop_path = os.path.join(crop_dir, f"{image_id}__{config.PATCH_SIZE}.png")
    crop = cv2.imread(crop_path) if os.path.exists(crop_path) else None
    if crop is None:
        image, mask = load_image_and_mask(image_path, mask_path)
        crop = crop_lesion(image, mask, config.PATCH_SIZE)
        cv2.imwrite(crop_path, crop)
    return crop


def global_features(triples, model, preprocess, feature_dim, chunk=32, workers=8):
    cache_dir = os.path.join(config.CACHE_DIR, "global")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(config.CACHE_DIR, "crops"), exist_ok=True)
    cache_paths = [os.path.join(cache_dir, global_cache_name(os.path.splitext(os.path.basename(p))[0]))
                   for p, _, _ in triples]
    missing = [i for i, cp in enumerate(cache_paths) if not os.path.exists(cp)]

    # eksikler chunk chunk: kırpmalar thread havuzunda paralel hazırlanır (cv2 GIL'i bırakır),
    # sonra chunk'ın bütün kırpmaları tek seferde GPU'ya gider
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(missing), chunk):
            idx = missing[start:start + chunk]
            crops = list(pool.map(lambda i: lesion_crop(triples[i][0], triples[i][1]), idx))
            n_var = 8 if config.GLOBAL_TTA else 1
            batch = [v for c in crops for v in (dihedral_variants(c) if config.GLOBAL_TTA else [c])]
            feats = extract_features(batch, model, preprocess, config.BATCH_SIZE, feature_dim)
            feats = feats.reshape(len(idx), n_var, -1).mean(axis=1)
            for i, f in zip(idx, feats):
                np.save(cache_paths[i], f)

    features = []
    for cache_path in cache_paths:
        f = np.load(cache_path)
        features.append(f)
    return np.array(features)


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def cached_patch_features(triples):
    """main.py'nin cache'inden her görüntünün (patch_sayısı, dim) feature'ları."""
    maps_str = "_".join(config.ENHANCEMENT_MAPS) if config.ENHANCEMENT_MAPS else "nomap"
    features = []
    for image_path, _, _ in triples:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        cache_path = os.path.join(config.CACHE_DIR, f"{image_id}__{maps_str}__{config.FEATURE_LAYER}.npy")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"{cache_path} yok: önce aynı ayarlarla main.py çalıştırılmalı")
        features.append(np.load(cache_path))
    return features


def meanpool_features(triples):
    X = np.array([f.mean(axis=0) if len(f) else np.zeros(config.FEATURE_DIM, dtype=np.float32)
                  for f in cached_patch_features(triples)])
    return l2(X)


def main(mode):
    train_data = load_split(config.TRAIN_CSV, config.TRAIN_IMAGE_DIR, config.TRAIN_MASK_DIR)
    test_data = load_split(config.TEST_CSV, config.TEST_IMAGE_DIR, config.TEST_MASK_DIR)
    os.makedirs(config.AUG_DIR, exist_ok=True)
    train_data = balance_train_set(train_data, config.AUG_DIR, config.SEED)

    y_train = np.array([label for _, _, label in train_data])

    if mode in ("global", "fusion"):
        (model, feature_dim), preprocess = build_global_model(config.GLOBAL_BACKBONE), build_preprocess()
        X_train = l2(global_features(train_data, model, preprocess, feature_dim))
        X_test = l2(global_features(test_data, model, preprocess, feature_dim))
        if mode == "fusion":
            # BoVW sözlüğü main.py'deki gibi SADECE train patch'leriyle kurulur
            train_patches, test_patches = cached_patch_features(train_data), cached_patch_features(test_data)
            all_feats, all_labels = flatten_patch_features(train_patches, list(y_train))
            centers = train_dictionary(all_feats, all_labels, k=config.K_CLUSTERS)
            X_train = np.hstack([X_train, build_histograms(train_patches, centers)])
            X_test = np.hstack([X_test, build_histograms(test_patches, centers)])
    elif mode == "meanpool":
        X_train, X_test = meanpool_features(train_data), meanpool_features(test_data)
    else:
        raise ValueError("mode: global | meanpool | fusion")

    y_test = np.array([label for _, _, label in test_data])

    groups = source_image_ids(train_data)
    svm_model, svm_params, cv_score = tune_svm(
        X_train, y_train, groups, config.SVM_C_GRID, config.SVM_GAMMA_GRID,
        config.SVM_CV_FOLDS, config.SVM_CV_SCORING, config.SEED
    )
    print(f"En iyi SVM: {svm_params} (CV {config.SVM_CV_SCORING} = {cv_score:.4f})")
    y_pred, y_scores = predict(svm_model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_scores)
    print(metrics)

    threshold = threshold_metrics = None
    if config.TUNE_THRESHOLD:
        threshold, oof_score = tune_threshold(X_train, y_train, groups, svm_params["C"], svm_params["gamma"],
                                              config.SVM_CV_FOLDS, config.SEED)
        y_pred_t = (svm_model.decision_function(X_test) >= threshold).astype(int)
        threshold_metrics = compute_metrics(y_test, y_pred_t, y_scores)
        print(f"Eşik {threshold:.4f} (OOF balanced accuracy = {oof_score:.4f}): {threshold_metrics}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    maps_str = "_".join(config.ENHANCEMENT_MAPS) if config.ENHANCEMENT_MAPS else "nomap"
    crop_tag = "crop" if config.GLOBAL_BACKBONE == "resnet101" else f"crop-{config.GLOBAL_BACKBONE}"
    crop_tag += "-tta" if config.GLOBAL_TTA else ""
    tag = {"meanpool": maps_str, "global": crop_tag, "fusion": crop_tag + "+" + maps_str}[mode]
    path = os.path.join(config.RESULTS_DIR, f"{datetime.now():%Y%m%d_%H%M%S}__baseline_{mode}__{config.FEATURE_LAYER}__{tag}.json")
    with open(path, "w") as f:
        json.dump({
            "time": datetime.now().isoformat(timespec="seconds"),
            "baseline": mode,
            "feature_layer": config.FEATURE_LAYER,
            "backbone": config.GLOBAL_BACKBONE if mode != "meanpool" else "resnet101",
            "tta": config.GLOBAL_TTA if mode != "meanpool" else False,
            "enhancement_maps": list(config.ENHANCEMENT_MAPS) if mode != "global" else [],
            "svm_params": svm_params,
            "svm_cv_scoring": config.SVM_CV_SCORING,
            "svm_cv_score": cv_score,
            "metrics": metrics,
            "threshold": threshold,
            "metrics_tuned_threshold": threshold_metrics,
        }, f, indent=2, default=str)
    print(f"Sonuçlar kaydedildi: {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "global")
