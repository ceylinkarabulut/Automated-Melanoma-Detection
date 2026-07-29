"""Değerlendirme metrikleri."""


""" Bu kısım görev scope'umda olmamasına rağmen erken bitirdiğim için ben yazıp commit attım, 
çünkü arkadaşlarımın görevleri bana verilen göreve kıyasla daha vakit alan şeyler. GO TEAM GITGUD!!!"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


def compute_metrics(y_true, y_pred, y_scores):
    """-> {TP,TN,FP,FN, ACC,SENS,SPEC,PPV,NPV,DC, ROC_AUC,PR_AUC}"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    dc = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_scores)
    except ValueError:
        pr_auc = float("nan")

    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "ACC": acc, "SENS": sens, "SPEC": spec,
        "PPV": ppv, "NPV": npv, "DC": dc,
        "ROC_AUC": roc_auc, "PR_AUC": pr_auc,
    }