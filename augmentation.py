"""Train setindeki melanom sayısını benign sayısına eşitleme."""

import cv2
import numpy as np


def augment_image(img, mask, seed):
    """Rastgele kaydırma/zoom/rotasyon. Görüntü ve maskeye AYNI dönüşüm.
    -> (aug_img, aug_mask)"""
    pass


def balance_train_set(samples, out_dir, seed):
    """Melanom sayısını benign'e eşitle. Augment kopyalar diske yazılır.
    SADECE TRAIN'E UYGULANIR.
    -> genişletilmiş samples listesi"""
    pass