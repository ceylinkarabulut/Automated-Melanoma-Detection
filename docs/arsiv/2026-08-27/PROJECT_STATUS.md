# Automated Melanoma Detection — Proje Durumu

> Bu dosya önceki çalışma notlarının (Ağustos bug-tur özeti + BoVW modülü
> sözleşmesi) ve 2026-08-26 tarihinde gerçek repo (`C:\Users\omen\PycharmProjects\Automated-Melanoma-Detection`)
> üzerinde yapılan doğrulamanın birleşimidir. Repo, önceki sohbet özetinden
> daha ileride; bazı "kalan bug"lar zaten kapatılmış, yeni bir iş (hocanın
> fc katmanı talebi) yarım kalmış ve pipeline'ın gerçekten çalışması önünde
> 3 net blokaj tespit edilmiş.

## Ne yapıyor

Okur & Turkan, 2024 ("Weighted Bag of Visual Words with enhanced deep
features for melanoma detection", Expert Systems With Applications)
makalesinin yeniden üretimi. ISIC 2017 veri seti.

Pipeline: dermoskopi görüntüsü → lezyon maskesiyle patch'lere bölünür → her
patch PCA/parlaklık/doygunluk haritalarıyla ağırlıklandırılır → ResNet-101
ile derin özellik çıkarılır → sınıf-bazlı (benign/malignant ayrı) K-means ile
BoVW sözlüğü kurulur → Weighted BoVW histogramı (malignant küme ağırlığı 10,
benign 1) → RBF-kernel SVM ile sınıflandırma.

Makale hedefi (ISIC 2017): ACC=0.962, SENS=0.998, SPEC=0.925, DC=0.963,
PPV=0.931, NPV=0.998.

## Ekip

Branch stratejisi: `feature/*` → `main`'e PR ile birleşiyor.

- **Ceylin** — `feature_extraction.py`, `main.py`, kısmen `evaluation.py`, `bovw.py`'de `build_histogram`
- **idil** — `preprocessing.py`
- **Metin** — `patch_enhancement.py`
- **Kutay** — `dataset.py`, `bovw.py`'de `train_dictionary`

## 🔴 Şu an açık olan iş: hocanın fc-katmanı talebi (YARIM, commit edilmemiş)

`feature_extraction.py` içindeki güncel yorum (build_model docstring'i):

> "fc katmanı KORUNUYOR: çıktı 1000 boyutlu, softmax'a girecek ham skor
> (logit) vektörü. **(Hocanın istediği versiyon** — önceki denemede
> fc=Identity() ile 2048 boyutlu avgpool çıktısı kullanılıyordu.)"

Yani hoca, ResNet-101'in `fc` katmanının **silinmemesini** (`fc=Identity()`
kullanılmamasını), orijinal MATLAB/makale baseline'ına daha yakın olan
**1000 boyutlu (softmax öncesi logit) çıktının** kullanılmasını istemiş.
Önceki versiyon `fc=Identity()` ile 2048 boyutlu avgpool çıktısı
üretiyordu.

Bu değişiklik kodda **yapılmış ama commit edilmemiş** (working tree'de
staged olmayan değişiklik olarak duruyor):

- `feature_extraction.py`: `model.fc = torch.nn.Identity()` satırı
  kaldırılmış → artık tam `resnet101` (1000-d çıktı) kullanılıyor
- `config.py`: `FEATURE_DIM = 2048` → `1000` yapılmış
- `config.py`: `IMAGE_DIR/MASK_DIR` → ayrı `TRAIN_IMAGE_DIR/TRAIN_MASK_DIR/TEST_IMAGE_DIR/TEST_MASK_DIR`'a bölünmüş (train/test fiziksel klasör ayrımı)
- `main.py`: yukarıdaki yeni config isimlerini kullanacak şekilde güncellenmiş
- Yeni dosya (untracked): `diagnose_cache.py` — augmentation mask bug'ının
  sıfır-patch görüntülere etkisini cache üzerinden analiz eden yardımcı script

**Yapılması gereken:** bu değişiklikleri gözden geçirip commit etmek, SONRA
aşağıdaki 3 blokajı çözüp gerçek test sonuçlarını almak.

## 🔴 Test sonucu almadan önce 3 net blokaj

1. **Bozuk/eski cache — en kritik.** `cache/` klasöründe 1984 adet
   `*.npy` dosyası var, hepsi **2048 boyutlu** (eski `fc=Identity()`
   sürümünden kalma). `load_or_extract_features`'ın cache key'i
   (`image_id + "__" + maps_str + ".npy"`) **model versiyonunu/boyutunu
   içermiyor** — yani şu an `main.py` çalıştırılırsa, yeni 1000-d modele
   rağmen sessizce eski 2048-d cache'i okuyup kullanacak. Hocanın istediği
   değişiklik pipeline'a hiç yansımamış olacak ve kimse bunu fark etmeyecek
   (crash olmaz, sadece yanlış/eski sonuç üretir). **Cache'i tamamen
   silmek ya da cache key'ine feature boyutunu/model imzasını eklemek
   şart.**
2. **Test görüntüleri config'in beklediği yerde değil.** `config.py`
   `TEST_IMAGE_DIR = data/test/images` bekliyor ama gerçek dosyalar
   `data/test/images/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data/*.jpg`
   içinde — 2 katman daha derinde (zip'ten çıkarılırken oluşan iç içe klasör).
3. **Segmentasyon maskeleri hiç yok — hem train hem test.** `data/train/masks/`
   ve `data/test/masks/` klasörlerinde sadece ground-truth **etiket** CSV'leri
   var (`train_ground_truth.csv`, `test_ground_truth.csv`) — gerçek lezyon
   sınır maskesi PNG'leri (`ISIC_XXXXXXX_segmentation.png`) **repo'da da,
   diskte başka hiçbir yerde de bulunamadı**. `extract_patches` maskeye göre
   patch seçtiği için maskeler olmadan pipeline çalışamaz. Bunlar ISIC 2017
   arşivinin ayrı bir indirme parçası (`*_Part1_GroundTruth` / segmentation
   klasörü) — muhtemelen henüz indirilmemiş/yerleştirilmemiş.

## Config sabit isimleri (2026-08-26 itibarıyla doğrulanmış, gerçek dosyadan)

`TRAIN_IMAGE_DIR`, `TRAIN_MASK_DIR`, `TEST_IMAGE_DIR`, `TEST_MASK_DIR`,
`AUG_DIR`, `TRAIN_CSV`, `TEST_CSV`, `CACHE_DIR`, `MH_LONG_SIDE`,
`PATCH_SIZE`, `PATCH_OVERLAP`, `STRIDE`, `LESION_THRESHOLD`, `AUG_SHIFT`,
`AUG_ZOOM`, `AUG_ROTATION`, `ENHANCEMENT_MAPS`, `FEATURE_DIM`, `BATCH_SIZE`,
`K_CLUSTERS`, `MALIGNANT_WEIGHT`, `BENIGN_WEIGHT`, `KMEANS_BATCH_SIZE`,
`SVM_C`, `SVM_GAMMA`, `SEED`.

(Not: Ağustos özetindeki `config.RANDOM_SEED`/`config.FEATURE_BATCH_SIZE`/
`config.K` gibi yer tutucu isimler yanlıştı, gerçek isimler yukarıdaki gibi —
zaten düzeltilmişler.)

## Ağustos turundan bu yana kapanan işler (git log'da doğrulandı)

- `bovw.py`: `train_dictionary`'ye `random_state=config.SEED` eklendi (fakat
  bkz. aşağıdaki "hâlâ eksik" — `main.py`'nin çağrısında `batch_size`
  geçilmiyor, `MiniBatchKMeans` kendi varsayılanını kullanıyor)
  — **DÜZELTME:** `k//2` kümeleri hâlâ `MiniBatchKMeans(n_clusters=k//2,
  random_state=config.SEED)` ile deterministik kuruluyor, sorun yok.
- `feature_extraction.py`: `IMAGENET1K_V2` → `IMAGENET1K_V1`'e geçildi
  (commit `d50cd4b`, "closer to original MATLAB ResNet-101 baseline")
- `feature_extraction.py`: disk cache eklendi (`load_or_extract_features`,
  commit `a38a833`) — ama yukarıda anlatıldığı gibi versiyon-bilinçli değil
- `augmentation.py`: config sabitlerini kullanıyor (`AUG_ROTATION/AUG_ZOOM/
  AUG_SHIFT` artık gerçekten okunuyor), mask `IMREAD_GRAYSCALE` ile okunuyor
  (commit `c5b357d`)
- `preprocessing.py`: eski `compute_mh_long_side` (sample.img_path
  AttributeError'lı, ölü kod) dosyadan tamamen kaldırılmış — artık sorun yok

## Hâlâ açık olan, dokunulmamış bug'lar

1. **`augmentation.py` — mask interpolation.** `cv2.warpAffine(mask, M, ...)`
   interpolation belirtmiyor (varsayılan `INTER_LINEAR`) → binary maske
   kenarlarda bulanıklaşıyor, `INTER_NEAREST` olmalı (görüntü tarafı zaten
   `preprocessing.rescale_image`'daki `is_mask` mantığıyla doğru
   yapılıyor, augmentation'da unutulmuş). `diagnose_cache.py`'nin
   docstring'i bunu "artık düzeltilmiş" diye anıyor ama kodda düzeltme
   **yok** — muhtemelen düzeltme planlanmış ama unutulmuş, ya da yorum
   yanlış yazılmış. Netleştirilmeli.
2. **`classifier.py` — `train_svm`'e `random_state` yok.**
   `SVC(..., probability=True)` sklearn içinde ayrı bir iç çapraz-doğrulama
   kullanıyor (Platt scaling), o da seed almadan çalışıyor → run'dan run'a
   SVM olasılık tahminlerinde küçük farklar olabilir. `random_state=config.SEED`
   eklenmeli.

## Karar bekleyen (bug değil, tercih)

- `config.CACHE_DIR` kullanılıyor artık (yukarıda çözüldü) ama cache key
  model versiyonunu içermiyor — bu hem blokaj hem de tasarım kararı.

## ✅ 2026-08-27 gece: ilk uçtan uca test sonucu alındı

Tüm blokajlar çözüldü ve pipeline gerçekten çalıştırıldı:

- Segmentasyon maskeleri (Downloads'taki ISIC-2017 Part1 GroundTruth zip'leri)
  `data/train/masks/` (2000) ve `data/test/masks/` (600) altına yerleştirildi.
- `data/test/images/` içindeki fazladan iç içe klasör düzleştirildi, test CSV'si
  `config.TEST_CSV`'nin beklediği `data/test_ground_truth.csv` konumuna kondu.
- Eski (2048-d) cache tamamen silindi — hocanın fc-korumalı (1000-d) modeli
  gerçekten devreye girdi.
- `augmentation.py` mask interpolation (`INTER_NEAREST`) ve `classifier.py`
  `train_svm` `random_state` bug'ları da bu turda kapatıldı.
- GPU/PyTorch uyumsuzluğu çıktı: kurulu `torch==2.13.0+cu126`, RTX 5070
  Laptop GPU'nun (CC 12.0 / sm_120) çekirdeklerini içermiyordu
  (`CUDA error: no kernel image is available for execution on the device`).
  `torch==2.13.0+cu130` / `torchvision==0.28.0+cu130` (cu130 index) ile
  değiştirildi, aynı sürüm numaraları korunarak API uyumluluğu riski
  minimize edildi.

**Test seti sonucu (600 görüntü, 117 malign / 483 benign):**

| ACC | SENS | SPEC | PPV | NPV | DC | ROC_AUC | PR_AUC |
|---|---|---|---|---|---|---|---|
| 0.632 | 0.667 | 0.623 | 0.300 | 0.885 | 0.414 | 0.698 | 0.350 |

TP=78, TN=301, FP=182, FN=39.

Makale hedefinin (ACC=0.962 vb.) çok altında. Düşük PPV (0.30) + yüksek FP
(182) modelin gereğinden fazla "malignant" tahmin ettiğini gösteriyor —
`config.MALIGNANT_WEIGHT=10` ağırlığı, sözlük/özellik kalitesi ya da SVM
hiperparametreleri (`SVM_C`, `SVM_GAMMA`) şüpheli. Bu gecenin kapsamı
"pipeline'ı çalışır hale getirip ilk sonucu almak"tı — performans debug'ı
sonraki tur.

**Commit edilmemiş değişiklikler hâlâ bekliyor:** `config.py`,
`feature_extraction.py`, `main.py` (hoca'nın fc değişikliği) +
`augmentation.py`, `classifier.py` (bu geceki 2 bug fix) + yeni
`PROJECT_STATUS.md`, `diagnose_cache.py`. Henüz commit edilmedi.

## Sonraki adımlar (öneri sırası)

1. `feature_extraction.py`/`config.py`/`main.py`'deki mevcut (commit
   edilmemiş) değişiklikleri gözden geçirip commit et.
2. `cache/` klasörünü temizle (ya da cache dosya adına feature boyutu/model
   imzası ekle) — yoksa yeni fc-korumalı model hiç devreye girmeden eski
   sonuçlar rapor edilir.
3. Test görüntülerini `config.TEST_IMAGE_DIR`'ın beklediği düz yapıya taşı
   (ya da config'i gerçek iç içe yapıya göre güncelle).
4. Segmentasyon maskelerini (ISIC 2017 Part 1 Ground Truth / lezyon sınır
   PNG'leri) indirip `data/train/masks/` ve `data/test/masks/` içine koy.
5. `main.py`'yi çalıştırıp gerçek test metriklerini al (`compute_metrics`
   ACC/SENS/SPEC/DC/PPV/NPV/ROC_AUC/PR_AUC döndürüyor).
6. İsteğe bağlı: `augmentation.py` mask interpolation ve `classifier.py`
   `random_state` bug'larını da bu turda kapat.

GitHub: https://github.com/ceylinkarabulut/Automated-Melanoma-Detection
