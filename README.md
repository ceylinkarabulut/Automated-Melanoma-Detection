# Automated Melanoma Detection

A classical computer-vision pipeline for classifying skin lesion images as **benign** or **malignant (melanoma)**, built around patch-level deep features and a weighted Bag-of-Visual-Words (BoVW) representation. Built as a team project (see [Contributors](#contributors)).

## Pipeline

1. **Load split** — read image/mask pairs and labels from a train/test CSV (`dataset.py`).
2. **Balance training set** — augment the minority class (shift, zoom, rotation) so the classes are balanced; augmentation is applied to the training split only (`augmentation.py`).
3. **Preprocess** — rescale each image/mask to a common long-side, then slide a window over the lesion mask to extract only the patches that sufficiently overlap the lesion (`preprocessing.py`).
4. **Enhance patches** — compute PCA, brightness, and saturation maps for each patch and fuse them in (`patch_enhancement.py`).
5. **Extract deep features** — run each enhanced patch through a pretrained **ResNet-101** (ImageNet weights, final FC layer removed) to get a 2048-d feature vector per patch, with disk caching keyed by image + enhancement combo (`feature_extraction.py`).
6. **Build a weighted visual dictionary** — cluster benign and malignant training patches *separately* with `MiniBatchKMeans` (K/2 clusters each), then build a per-image histogram over the combined dictionary, weighting malignant-cluster assignments more heavily to counter class imbalance (`bovw.py`).
7. **Classify** — train an RBF-kernel SVM on the per-image histograms (`classifier.py`).
8. **Evaluate** — compute confusion-matrix metrics (accuracy, sensitivity, specificity, PPV, NPV, Dice) plus ROC-AUC and PR-AUC on the held-out test split (`evaluation.py`).

Run the full pipeline with:

```bash
python main.py
```

## Data

The pipeline expects a `data/` directory (gitignored, not included in this repo) with:

```
data/
  images/                       # <image_id>.jpg
  masks/                        # <image_id>_segmentation.png
  train_ground_truth.csv        # columns: image_id, melanoma
  test_ground_truth.csv
```

This matches the layout of the ISIC melanoma classification datasets. All tunable paths and hyperparameters (patch size/stride, cluster count, class weights, SVM params, etc.) live in `config.py`.

## Setup

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is used automatically if available (falls back to CPU otherwise).

## Contributors

Team project for an image processing course — feature extraction/pipeline/evaluation, preprocessing, patch enhancement, and dataset handling were split across the team (see `notes.txt`) with the BoVW stage built together.
