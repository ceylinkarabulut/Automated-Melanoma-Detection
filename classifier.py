"""SVM sınıflandırıcı."""

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
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


def predict(model, X):
    """-> (tahminler, malignant olasılıkları)"""
    predictions = model.predict(X)
    probs = model.predict_proba(X)
    malignant_probs = probs[:, 1]
    return predictions, malignant_probs
