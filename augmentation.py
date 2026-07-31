"""Train setindeki melanom sayısını benign sayısına eşitleme."""
import os
import cv2
import numpy as np
import config
from dataset import class_counts


def augment_image(img: np.ndarray, mask: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-config.AUG_ROTATION, config.AUG_ROTATION)
    scale = rng.uniform(1 - config.AUG_ZOOM, 1 + config.AUG_ZOOM)
    shift_x = rng.uniform(-config.AUG_SHIFT, config.AUG_SHIFT)
    shift_y = rng.uniform(-config.AUG_SHIFT, config.AUG_SHIFT)
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += shift_x * width
    M[1, 2] += shift_y * height
    aug_img = cv2.warpAffine(img, M, (width, height))
    aug_mask = cv2.warpAffine(mask, M, (width, height))
    return aug_img, aug_mask


def balance_train_set(samples: list[tuple[str, str, int]], out_dir: str, seed: int) -> list[tuple[str, str, int]]:
    counts = class_counts(samples)
    n_needed = counts[0] - counts[1]
    melanoma_samples = []
    for sample in samples:
        if sample[2] == 1:
            melanoma_samples.append(sample)
    results = []
    for i in range(n_needed):
        source = melanoma_samples[i % len(melanoma_samples)]
        image_path, mask_path, label = source
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        aug_img, aug_mask = augment_image(img, mask, seed + i)
        copy_number = i // len(melanoma_samples)
        filename = os.path.basename(image_path)
        image_id = os.path.splitext(filename)[0]
        new_image_name = image_id + "_augmented_" + str(copy_number) + ".jpg"
        new_mask_name = image_id + "_augmented_" + str(copy_number) + "_segmentation.png"
        new_image_path = os.path.join(out_dir, new_image_name)
        new_mask_path = os.path.join(out_dir, new_mask_name)
        cv2.imwrite(new_image_path, aug_img)
        cv2.imwrite(new_mask_path, aug_mask)
        results.append((new_image_path, new_mask_path, 1))
    return samples + results
