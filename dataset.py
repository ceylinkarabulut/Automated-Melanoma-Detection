"""Veri seti listesi ve etiket okuma."""

import os
import csv


def _read_labels(csv_path: str) -> dict[str, int]:
    labels = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            melanoma_raw = row["melanoma"]
            label = int(float(melanoma_raw))
            labels[image_id] = label
    return labels


def _build_paths(image_id: str, image_dir: str, mask_dir: str) -> tuple[str, str]:
    image_path = os.path.join(image_dir, image_id + ".jpg")
    mask_path = os.path.join(mask_dir, image_id + "_segmentation.png")
    return image_path, mask_path


def _find_missing_files(image_path: str, mask_path: str) -> list[str]:
    missing = []
    if not os.path.exists(image_path):
        missing.append(image_path)
    if not os.path.exists(mask_path):
        missing.append(mask_path)
    return missing


def load_split(csv_path: str, image_dir: str, mask_dir: str) -> list[tuple[str, str, int]]:
    labels = _read_labels(csv_path)
    results = []
    missing = []

    for image_id, label in labels.items():
        image_path, mask_path = _build_paths(image_id, image_dir, mask_dir)
        missing_here = _find_missing_files(image_path, mask_path)
        if missing_here:
            missing.extend(missing_here)
        else:
            results.append((image_path, mask_path, label))

    if missing:
        raise ValueError(f"Eksik dosyalar var: {missing}")
    return results


def class_counts(samples: list[tuple[str, str, int]]) -> dict[int, int]:
    counts = {}
    for image_path, mask_path, label in samples:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    return counts
