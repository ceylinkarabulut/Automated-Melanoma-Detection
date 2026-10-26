"""Tüm sabitler burada. Hiçbir modül kendi içine sayı gömmez."""

import os

# ---------- Yollar ----------
DATA_DIR        = "data"
TRAIN_DIR       = os.path.join(DATA_DIR, "train")
TEST_DIR        = os.path.join(DATA_DIR, "test")
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, "images")
TRAIN_MASK_DIR  = os.path.join(TRAIN_DIR, "masks")
TEST_IMAGE_DIR  = os.path.join(TEST_DIR, "images")
TEST_MASK_DIR   = os.path.join(TEST_DIR, "masks")
AUG_DIR         = os.path.join(TRAIN_DIR, "augmented")   # augment edilmiş kopyalar yalnızca train tarafında
TRAIN_CSV       = os.path.join(DATA_DIR, "train_ground_truth.csv")
TEST_CSV        = os.path.join(DATA_DIR, "test_ground_truth.csv")
CACHE_DIR       = "cache"
RESULTS_DIR     = "results"                  # her run'ın config + metrikleri buraya json olarak yazılır

# ---------- Ön işleme ----------
MH_LONG_SIDE     = 1430                      # uzun kenar, en-boy oranı korunur
PATCH_SIZE       = 224                       # n
PATCH_OVERLAP    = 100                       # m
STRIDE           = PATCH_SIZE - PATCH_OVERLAP  # 124
LESION_THRESHOLD = 0.5

# ---------- Augmentation (yalnızca train) ----------
AUG_SHIFT    = 0.10
AUG_ZOOM     = 0.20
AUG_ROTATION = 270

# ---------- Patch enhancement (ablation buradan yönetilir) ----------
ENHANCEMENT_MAPS = ("pca", "bri", "sat")

# ---------- Feature extraction ----------
# "avgpool": fc öncesi 2048 boyutlu çıktı (makaledeki flattening katmanı)
# "logits" : son FC çıktısı, softmax öncesi 1000 boyutlu vektör (hocanın önerisi)
FEATURE_LAYER = "avgpool"
FEATURE_DIMS  = {"avgpool": 2048, "logits": 1000}
FEATURE_DIM   = FEATURE_DIMS[FEATURE_LAYER]
BATCH_SIZE    = 64

# ---------- Tüm lezyon (baseline.py global / fusion) ----------
# resnet101 | resnet50 | densenet201 | efficientnet_v2_m | convnext_base
GLOBAL_BACKBONE = "resnet101"
GLOBAL_TTA      = False     # True: 8 döndürme/yansıtma varyantının feature ortalaması
TUNE_THRESHOLD  = True      # karar eşiğini train'in out-of-fold skorlarıyla seç (balanced accuracy)

# ---------- BoVW ----------
K_CLUSTERS        = 1000                     # sınıf başına 500
MALIGNANT_WEIGHT  = 10
BENIGN_WEIGHT     = 1
KMEANS_BATCH_SIZE = 1024

# ---------- SVM ----------
# SVM_TUNE=True: C ve gamma, SADECE train üzerinde stratified + gruplu k-fold CV ile seçilir
# (augment edilmiş kopyalar kaynak görüntüleriyle aynı fold'da kalır). False: aşağıdaki sabit değerler.
SVM_TUNE       = True
SVM_CV_FOLDS   = 10
SVM_CV_SCORING = "balanced_accuracy"
SVM_C_GRID     = (0.1, 1, 10, 100, 1000, 10000, 100000)
SVM_GAMMA_GRID = ("scale", 0.0001, 0.001, 0.01, 0.1, 1)
SVM_C          = 1.0
SVM_GAMMA      = "scale"

SEED = 42
