"""ResNet-101 ile deep feature çıkarma."""

import cv2
import numpy as np
import torch
import torchvision.models as models
import os
from torchvision import transforms


def build_model():
    """ResNet-101 pretrained, fc=Identity(), eval(), GPU varsa GPU'da -> model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
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


def extract_features(patches, model, preprocess, batch_size=32):
    """Batch'li forward pass -> (len(patches), 2048) numpy array"""
    device = next(model.parameters()).device

    if len(patches) == 0:
        return np.empty((0, 2048), dtype=np.float32)

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

def load_or_extract_features(image_id, patches, model, preprocess, batch_size, cache_dir, enhancement_maps):
    """Cache key = image_id + enhancement kombinasyonu.
    Varsa diskten oku, yoksa extract_features çağır, diske yaz.
    -> (patch_sayısı, 2048) array"""
    maps_str = "_".join(enhancement_maps)
    filename = image_id + "__" + maps_str + ".npy"
    cache_path = os.path.join(cache_dir, filename)
    if os.path.exists(cache_path):
        features = np.load(cache_path)
    else:
        features = extract_features(patches, model, preprocess, batch_size)
        np.save(cache_path, features)

    return features