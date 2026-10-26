"""ResNet-101 ile deep feature çıkarma."""

import cv2
import numpy as np
import torch
import torchvision.models as models
import os
from torchvision import transforms

import config


def build_model(feature_layer=config.FEATURE_LAYER):
    """ResNet-101 pretrained, eval(), GPU varsa GPU'da -> model
    feature_layer="avgpool": fc=Identity(), çıktı 2048 boyutlu
    feature_layer="logits":  fc olduğu gibi kalır, çıktı softmax öncesi 1000 boyutlu"""
    if feature_layer not in config.FEATURE_DIMS:
        raise ValueError(f"Bilinmeyen FEATURE_LAYER: {feature_layer}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    if feature_layer == "avgpool":
        model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def build_preprocess():
    """transforms.Compose -> preprocess"""
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess


def extract_features(patches, model, preprocess, batch_size=32, feature_dim=config.FEATURE_DIM):
    """Batch'li forward pass -> (len(patches), feature_dim) numpy array"""
    device = next(model.parameters()).device

    if len(patches) == 0:
        return np.empty((0, feature_dim), dtype=np.float32)

    all_features = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            chunk = patches[i:i + batch_size]

            tensors = []
            for patch in chunk:
                if patch.ndim == 3 and patch.shape[2] == 3:
                    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                else:
                    rgb = patch
                tensors.append(preprocess(rgb))

            batch = torch.stack(tensors).to(device)
            features = model(batch)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)

def load_or_extract_features(image_id, get_patches, model, preprocess, batch_size, cache_dir, enhancement_maps,
                             feature_layer=config.FEATURE_LAYER):
    """Cache key = image_id + enhancement kombinasyonu + feature katmanı.
    Varsa diskten oku, yoksa get_patches() ile patch'leri üret, extract_features çağır, diske yaz.
    get_patches bir fonksiyon: cache varsa patch çıkarma ve harita hesabı hiç yapılmaz.
    Katman key'de olduğu için 2048 ve 1000 boyutlu cache'ler birbirine karışmaz.
    -> (patch_sayısı, feature_dim) array"""
    maps_str = "_".join(enhancement_maps) if enhancement_maps else "nomap"
    filename = image_id + "__" + maps_str + "__" + feature_layer + ".npy"
    cache_path = os.path.join(cache_dir, filename)
    feature_dim = config.FEATURE_DIMS[feature_layer]
    if os.path.exists(cache_path):
        features = np.load(cache_path)
    else:
        features = extract_features(get_patches(), model, preprocess, batch_size, feature_dim)
        np.save(cache_path, features)

    return features