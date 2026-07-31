"""SVM sınıflandırıcı."""

from sklearn.svm import SVC


def train_svm(X, y, C, gamma):
    """SVC(kernel='rbf', probability=True) -> model"""
    model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True)
    model.fit(X, y)
    return model

def predict(model, X):
    """-> (tahminler, malignant olasılıkları)"""
    predictions = model.predict(X)
    probs = model.predict_proba(X)
    malignant_probs = probs[:, 1]
    return predictions, malignant_probs
