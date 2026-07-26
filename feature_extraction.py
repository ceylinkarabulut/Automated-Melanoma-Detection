"""ResNet-101 ile deep feature çıkarma."""

import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms


def build_model():
    """ResNet-101 pretrained, fc=Identity(), eval() -> model"""
    pass


def build_preprocess():
    """transforms.Compose -> preprocess"""
    pass


def extract_features(patches, model, preprocess, batch_size):
    """Batch'li forward pass -> (len(patches), 2048) numpy array"""
    pass