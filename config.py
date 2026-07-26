"""Tüm sabitler burada. Hiçbir modül kendi içine sayı gömmez."""

import os

# ---------- Yollar ----------
DATA_DIR  = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR  = os.path.join(DATA_DIR, "masks")
AUG_DIR   = os.path.join(DATA_DIR, "augmented")
TRAIN_CSV = os.path.join(DATA_DIR, "train_ground_truth.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_ground_truth.csv")
CACHE_DIR = "cache"

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
FEATURE_DIM = 2048
BATCH_SIZE  = 64

# ---------- BoVW ----------
K_CLUSTERS        = 1000                     # sınıf başına 500
MALIGNANT_WEIGHT  = 10
BENIGN_WEIGHT     = 1
KMEANS_BATCH_SIZE = 1024

# ---------- SVM ----------
SVM_C     = 1.0
SVM_GAMMA = "scale"

SEED = 42