"""PCA / BRI / SAT ağırlık haritaları."""

import cv2
import numpy as np


def compute_pca_map(patch):
    """-> (n,n,3) float [0,1], 1'den çıkarılmış"""
    h, w, c = patch.shape
    # Renk kanallarını vektörlere dönüştürerek matris sütunları halinde istifle[cite: 1]
    pixels = patch.reshape(-1, c).astype(np.float32)

    # PCA gözlem skorlarını hesapla[cite: 1]
    mean, eigenvectors = cv2.PCACompute(pixels, mean=None)
    scores = np.dot(pixels - mean, eigenvectors.T)

    # Her bir skor vektörünü [0, 1] aralığına doğrusal olarak normalize et[cite: 1]
    scores_min = np.min(scores, axis=0)
    scores_max = np.max(scores, axis=0)

    diff = scores_max - scores_min
    diff[diff == 0] = 1.0  # Sıfıra bölünmeyi engelle
    normalized_scores = (scores - scores_min) / diff

    # Daha az baskın piksellere daha büyük ağırlık vermek için değerleri 1'den çıkar[cite: 1]
    pca_map = 1.0 - normalized_scores
    return pca_map.reshape(h, w, c)


def compute_bri_map(patch):
    """-> (n,n,3) float [0,1], ters çevrilmez"""
    patch_float = patch.astype(np.float32)

    # Konumdaki renk kanallarının ortalamasını al[cite: 1]
    mean_channels = np.mean(patch_float, axis=2, keepdims=True)

    # Pikselin parlaklığını, ortalamadan olan mutlak fark (absolute disparity) olarak belirle[cite: 1]
    bri_map = np.abs(patch_float - mean_channels)

    # Her bir renk kanalı için haritayı [0, 1] aralığına doğrusal olarak normalize et[cite: 1]
    for i in range(3):
        c_min = np.min(bri_map[:, :, i])
        c_max = np.max(bri_map[:, :, i])
        if c_max > c_min:
            bri_map[:, :, i] = (bri_map[:, :, i] - c_min) / (c_max - c_min)
        else:
            bri_map[:, :, i] = 0.0

    return bri_map


def compute_sat_map(patch):
    """-> (n,n,3) float [0,1], 1'den çıkarılmış"""
    # HSV renk uzayının S (Doygunluk) kanalını kullan
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1].astype(np.float32) / 255.0

    # Değerleri 1'den çıkar
    sat_map = 1.0 - s_channel

    # Aynı ağırlığı her bir renk kanalına (RGB) uygula
    return np.stack([sat_map, sat_map, sat_map], axis=-1)


def enhance_patch(patch, maps):
    """Haritaları eleman bazlı çarp. maps=() -> patch değişmeden döner.
    -> (n,n,3) uint8"""
    # Eğer maps tuple'ı boşsa (maps=()) görüntüyü hiç değiştirmeden döndür
    if not maps:
        return patch.copy()

    # Verilen haritaları kendi aralarında çarp (Örn: PCA x BRI x SAT)
    combined_weight = np.ones(patch.shape, dtype=np.float32)
    for m in maps:
        combined_weight *= m

    # Orijinal görüntüyü birleştirilmiş ağırlık haritası ile eleman bazlı çarp
    enhanced_patch = patch.astype(np.float32) * combined_weight

    # Taşmayı (overflow) engellemek için 0-255 arasına kırp ve uint8 formatında döndür
    return np.clip(enhanced_patch, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    # 1. Veri setinden rastgele bir görüntüyü oku
    # (Dosya adının klasöründeki bir görselle eşleştiğinden emin ol)
    sample_image = cv2.imread("data/images/ISIC_0000000.jpg")

    if sample_image is not None:
        # 2. Test için görüntüyü 224x224 boyutuna getir (patch simülasyonu)
        test_patch = cv2.resize(sample_image, (224, 224))

        # 3. Haritaları oluştur
        pca_map = compute_pca_map(test_patch)
        bri_map = compute_bri_map(test_patch)
        sat_map = compute_sat_map(test_patch)

        # 4. Görüntüyü iyileştir (Makaledeki gibi üç haritayı da kullanarak)
        enhanced_result = enhance_patch(test_patch, maps=(pca_map, bri_map, sat_map))

        # 5. Görsel olarak karşılaştırmak için sonuçları diske kaydet
        cv2.imwrite("test_before.jpg", test_patch)
        cv2.imwrite("test_after.jpg", enhanced_result)

        print("Test başarılı! Lütfen 'test_before.jpg' ve 'test_after.jpg' dosyalarını açıp gözle kontrol et.")
    else:
        print("Hata: Görüntü bulunamadı. Dosya yolunu kontrol et.")