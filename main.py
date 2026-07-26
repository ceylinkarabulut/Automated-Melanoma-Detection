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

import config


def main():
    pass


if __name__ == "__main__":
    main()