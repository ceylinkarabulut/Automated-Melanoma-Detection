"""Pipeline akışı.

Planlanan sıra:
    1. load_split ile train ve test yüklenir
    2. balance_train_set SADECE train'e uygulanır
    3. Her görüntü: rescale -> extract_patches -> enhance_patch -> extract_features
    4. train_dictionary SADECE train feature'larıyla eğitilir
    5. build_histogram ile her görüntü K boyutlu vektöre indirgenir
    6. train_svm SADECE train ile eğitilir
    7. compute_metrics SADECE test üzerinde çalıştırılır
"""

import numpy as np

import config
from dataset import load_split, balance_train_set
from preprocessing import rescale, extract_patches
from patch_enhancement import enhance_patch
from feature_extraction import build_model, build_preprocess, extract_features
from bovw import train_dictionary, build_histogram
from classifier import train_svm
from evaluation import compute_metrics


def images_to_features(triples, model, preprocess):
    image_features = []
    image_labels = []

    for image, mask, label in triples:
        image = rescale(image)
        patches = extract_patches(image, mask=mask)
        enhanced_patches = [enhance_patch(p) for p in patches]
        features = extract_features(enhanced_patches, model, preprocess, config.FEATURE_BATCH_SIZE)

        image_features.append(features)
        image_labels.append(label)

    return image_features, image_labels


def main():
    train_data = load_split("train")
    test_data = load_split("test")

    train_data = balance_train_set(train_data)

    model = build_model()
    preprocess = build_preprocess()

    train_features, train_labels = images_to_features(train_data, model, preprocess)
    test_features, test_labels = images_to_features(test_data, model, preprocess)

    benign_pool = [f for f, l in zip(train_features, train_labels) if l == config.LABEL_BENIGN]
    malignant_pool = [f for f, l in zip(train_features, train_labels) if l == config.LABEL_MALIGNANT]

    benign_features = np.concatenate(benign_pool, axis=0)
    malignant_features = np.concatenate(malignant_pool, axis=0)

    dictionary = train_dictionary(benign_features, malignant_features, k=config.K)

    X_train = np.array([build_histogram(f, dictionary) for f in train_features])
    y_train = np.array(train_labels)

    X_test = np.array([build_histogram(f, dictionary) for f in test_features])
    y_test = np.array(test_labels)

    svm_model = train_svm(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    print(metrics)


if __name__ == "__main__":
    main()