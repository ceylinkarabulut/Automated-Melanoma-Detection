"""SVM sınıflandırıcı."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.svm import SVC


def train_svm(X, y, C, gamma, random_state):
    """SVC(kernel='rbf', probability=True) -> model
    random_state, predict_proba için sklearn'ün yaptığı iç CV'yi sabitler."""
    model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=random_state)
    model.fit(X, y)
    return model


def tune_svm(X, y, groups, c_grid, gamma_grid, n_folds, scoring, random_state):
    """C ve gamma'yı SADECE train üzerinde gruplu stratified k-fold CV ile seçer,
    sonra en iyi değerlerle tüm train'e train_svm uygular.
    groups: her örneğin kaynak görüntü id'si -- augment edilmiş kopyalar kaynaklarıyla
    aynı fold'a düşer, yoksa CV skoru şişer.
    -> (model, {"C":..., "gamma":...}, en iyi CV skoru)"""
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        SVC(kernel='rbf'),
        param_grid={"C": list(c_grid), "gamma": list(gamma_grid)},
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=False,
    )
    search.fit(X, y, groups=groups)
    best = search.best_params_
    model = train_svm(X, y, best["C"], best["gamma"], random_state)
    return model, best, search.best_score_


def tune_threshold(X, y, groups, C, gamma, n_folds, random_state):
    """SVM karar fonksiyonu için eşik: train üzerinde out-of-fold decision_function
    skorlarından balanced accuracy'yi en yükseğe çıkaran değer. Test seti hiç kullanılmaz.
    -> (eşik, o eşikteki OOF balanced accuracy)"""
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = cross_val_predict(SVC(kernel='rbf', C=C, gamma=gamma), X, y, groups=groups, cv=cv,
                            method="decision_function", n_jobs=-1)
    candidates = np.unique(oof)
    scores = [balanced_accuracy_score(y, (oof >= t).astype(int)) for t in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def predict(model, X):
    """-> (tahminler, malignant olasılıkları)"""
    predictions = model.predict(X)
    probs = model.predict_proba(X)
    malignant_probs = probs[:, 1]
    return predictions, malignant_probs
