# Automated Melanoma Detection

A classical computer-vision pipeline for classifying skin lesion images as **benign** or **malignant (melanoma)**, built around patch-level deep features and a weighted Bag-of-Visual-Words (BoVW) representation. Built as a team project (see [Contributors](#contributors)).

For the full project history, experiments, results and decisions (in Turkish), see [docs/SUREC.md](docs/SUREC.md). The original paper-reproduction version is preserved on `main` and tagged `v1.0-bovw`.

## Pipeline

1. **Load split** — read image/mask pairs and labels from a train/test CSV (`dataset.py`).
2. **Balance training set** — augment the minority class (shift, zoom, rotation) so the classes are balanced; augmentation is applied to the training split only (`augmentation.py`).
3. **Preprocess** — rescale each image/mask to a common long-side, then slide a window over the lesion mask to extract only the patches that sufficiently overlap the lesion (`preprocessing.py`).
4. **Enhance patches** — compute PCA, brightness, and saturation maps for each patch and fuse them in (`patch_enhancement.py`).
5. **Extract deep features** — run each enhanced patch through a pretrained **ResNet-101** (ImageNet weights). `FEATURE_LAYER` in `config.py` picks the output: `"avgpool"` (final FC removed, 2048-d) or `"logits"` (pre-softmax FC output, 1000-d). Features are cached on disk, keyed by image + enhancement combo + feature layer (`feature_extraction.py`).
6. **Build a weighted visual dictionary** — cluster benign and malignant training patches *separately* with `MiniBatchKMeans` (K/2 clusters each), then build a per-image histogram over the combined dictionary, weighting malignant-cluster assignments more heavily to counter class imbalance (`bovw.py`).
7. **Classify** — train an RBF-kernel SVM on the per-image histograms. With `SVM_TUNE = True`, `C` and `gamma` are chosen by grouped, stratified k-fold cross-validation on the training set only (augmented copies stay in the same fold as their source image) (`classifier.py`).
8. **Evaluate** — compute confusion-matrix metrics (accuracy, sensitivity, specificity, PPV, NPV, Dice) plus ROC-AUC and PR-AUC on the held-out test split (`evaluation.py`). Each run's settings and metrics are written to `results/` as JSON so ablation runs can be compared side by side.

Run the full pipeline with:

```bash
python main.py
```

## Data

The pipeline expects a `data/` directory (gitignored, not included in this repo) with:

```
data/
  train/
    images/                     # <image_id>.jpg
    masks/                      # <image_id>_segmentation.png
    augmented/                  # created at runtime, training split only
  test/
    images/
    masks/
  train_ground_truth.csv        # columns: image_id, melanoma
  test_ground_truth.csv
```

Train and test images are kept in physically separate folders. After changing `FEATURE_LAYER` or the enhancement maps no cache clearing is needed, since both are part of the cache key.

This matches the layout of the ISIC melanoma classification datasets. All tunable paths and hyperparameters (patch size/stride, cluster count, class weights, SVM params, etc.) live in `config.py`.

## Setup

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is used automatically if available (falls back to CPU otherwise).

## Contributors

Team project for an image processing course — feature extraction/pipeline/evaluation, preprocessing, patch enhancement, and dataset handling were split across the team (see `notes.txt`) with the BoVW stage built together.
