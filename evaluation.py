"""Değerlendirme metrikleri."""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


def compute_metrics(y_true, y_pred, y_scores):
    """-> {TP,TN,FP,FN, ACC,SENS,SPEC,PPV,NPV,DC, ROC_AUC,PR_AUC}"""
    pass