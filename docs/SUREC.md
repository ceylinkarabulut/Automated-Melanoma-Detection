# Otomatik Melanom Tespiti: Süreç Dokümanı

Son güncelleme: 23 Ekim 2026

Bu doküman projenin baştan bugüne nasıl ilerlediğini, kodun ne yaptığını, hangi deneyleri yaptığımızı, hangi sonuçları aldığımızı ve hangi kararları neden verdiğimizi tek yerde topluyor.

**Kısa sonuç:** En iyi yöntemimiz, lezyonu kırpıp ResNet-101'den tek bir özellik vektörü alan ve bunu SVM ile sınıflandıran "tüm lezyon" yöntemi. Gerçek ISIC 2017 test setinde **ROC-AUC 0.815, doğruluk 0.80, duyarlılık 0.66, özgüllük 0.83**. Makalenin yöntemi (Weighted BoVW) aynı test setinde AUC 0.65-0.68'de kaldı.

## 1. Proje ve hedef

- **Makale:** Okur & Turkan (2024), *Weighted Bag of Visual Words with enhanced deep features for melanoma detection*, Expert Systems With Applications.
- **Veri seti:** ISIC 2017. Train 2000 görüntü (374 melanom, 1626 benign), test 600 görüntü (117 melanom, 483 benign).
- **İlk hedef:** Makaledeki yöntemi yeniden üretmek.
- **Güncel hedef (Ekim 2026, hocamızın isteği):** Makaledeki yüzdeleri birebir tutturmaya çalışmak yerine yöntemi geliştirip daha iyi sonuç almak.

### Ekip

| Kişi | Modül |
|---|---|
| Ceylin | `feature_extraction.py`, `main.py`, `evaluation.py` (kısmen), `bovw.py` içindeki `build_histogram` |
| İdil | `preprocessing.py` |
| Metin | `patch_enhancement.py` |
| Kutay | `dataset.py`, `bovw.py` içindeki `train_dictionary` |

## 2. Versiyonlar ve branch'ler

| Ne | Nerede | İçerik |
|---|---|---|
| Eski versiyon (v1) | `main` branch'i ve `v1.0-bovw` etiketi | Makalenin yönteminin ilk çalışan hali: tek görüntü klasörü, sabit SVM parametreleri, 2048 boyutlu özellik. |
| Güncel versiyon | `claude/project-thread-u8itds` branch'i | `improve-results` (PR #6) üzerindeki tüm geliştirmeler + `results/` altındaki deney sonuçları + bu doküman. |
| Eski çalışma dalları | `feature/*` | Ekibin modül bazlı çalıştığı dallar; hepsi `main`'e birleştirilmiş durumda. |

`main` bilerek değiştirilmedi. Eski versiyona dönmek için `git checkout v1.0-bovw` yeterli.

## 3. Kod ne yapıyor

Kodda iki yol var: makaleyi uygulayan `main.py` (BoVW) ve onu geçen `baseline.py` (tüm lezyon). İkisi de aynı veri hazırlığını, aynı SVM ayarını ve aynı test setini kullanıyor, yani sonuçlar doğrudan karşılaştırılabilir. Bütün sabitler `config.py` içinde.

### Ortak adımlar

1. **Veri (`dataset.py`, `config.py`):** Train ve test fiziksel olarak ayrı klasörlerde (`data/train`, `data/test`). Test hiçbir aşamada eğitimde kullanılmıyor.
2. **Dengeleme (`augmentation.py`):** Sadece train'de melanomlar döndürme, yakınlaştırma ve kaydırma ile çoğaltılıp benign sayısına (1626) eşitleniyor. Toplam 3252 train görüntüsü. Maskeler en yakın komşu interpolasyonla döndürülüyor, böylece ikili (binary) kalıyor.
3. **SVM (`classifier.py`):** RBF çekirdekli SVM. `C` ve `gamma` sadece train üzerinde 10 katlı, gruplu ve tabakalı çapraz doğrulama ile seçiliyor. Bir görüntünün augment kopyaları kaynağıyla aynı katta tutuluyor, böylece CV skoru şişmiyor.
4. **Karar eşiği:** `TUNE_THRESHOLD = True` ise eşik, train'in out-of-fold skorlarıyla dengeli doğruluğu en iyileyecek şekilde seçiliyor.
5. **Değerlendirme (`evaluation.py`):** ACC, SENS, SPEC, PPV, NPV, Dice, ROC-AUC ve PR-AUC gerçek (dengesiz) test setinde hesaplanıyor. Her çalıştırmanın ayarları ve sonuçları `results/` altına JSON olarak yazılıyor.

### Yol A: Makalenin yöntemi, Weighted BoVW (`main.py`)

1. Görüntü uzun kenarı 1430 piksel olacak şekilde ölçekleniyor; lezyon maskesinin içinden 224×224 boyutunda, 100 piksel üst üste binen patch'ler kesiliyor (`preprocessing.py`).
2. Her patch PCA, parlaklık (BRI) ve doygunluk (SAT) haritalarıyla ağırlıklandırılıyor (`patch_enhancement.py`). Hangi haritaların kullanılacağı `ENHANCEMENT_MAPS` ile seçiliyor.
3. Her patch ImageNet ağırlıklı ResNet-101'den geçiyor. `FEATURE_LAYER = "avgpool"` 2048 boyutlu, `"logits"` 1000 boyutlu (hocamızın istediği) vektör veriyor (`feature_extraction.py`). Özellikler diske önbellekleniyor.
4. Train patch'leri sınıf başına ayrı kümeleniyor: 500 benign + 500 melanom, toplam 1000 "görsel kelime" (`bovw.py`).
5. Her görüntü ağırlıklı bir histograma dönüşüyor (melanom kelimeleri 10 kat ağırlıklı) ve SVM'e veriliyor.

### Yol B: Bizim geliştirmemiz, tüm lezyon (`baseline.py global`)

1. Maskeden lezyonun sınırlayıcı kutusu bulunuyor, her yöne %10 pay eklenip kare olarak kırpılıyor ve 224×224'e indiriliyor. Kırpmalar `cache/crops/` altına kaydediliyor.
2. Bu tek görüntü ImageNet ağırlıklı omurga ağdan (varsayılan ResNet-101) geçiyor, görüntü başına tek bir 2048 boyutlu vektör çıkıyor.
3. Vektör doğrudan SVM'e veriliyor. Ağ yeniden eğitilmiyor; makaledeki gibi sadece özellik çıkarıcı.

`baseline.py` ayrıca şu modları ve seçenekleri destekliyor:

- `meanpool`: BoVW yerine patch özelliklerinin ortalaması.
- `fusion`: tüm lezyon vektörü + BoVW histogramı yan yana.
- `GLOBAL_BACKBONE`: `resnet101`, `resnet50`, `densenet201`, `efficientnet_v2_m`, `convnext_base`.
- `GLOBAL_TTA`: 8 döndürme/yansıtma varyantının özellik ortalaması.

### Nasıl çalıştırılır

```bash
pip install -r requirements.txt
python main.py               # Yol A: BoVW
python baseline.py global    # Yol B: tüm lezyon (en iyi sonuç)
python baseline.py meanpool  # karşılaştırma; önce main.py çalışmış olmalı
python baseline.py fusion    # karşılaştırma; önce main.py çalışmış olmalı
```

Veri klasör yapısı için `README.md` dosyasına bakın. `data/` ve `cache/` repoda yok.

## 4. Süreç: neler yaptık

### Ağustos 2026: Pipeline'ı çalışır hale getirme

1. **`main.py` düzeltildi.** Diğer modüllerin fonksiyon imzalarıyla uyuşmuyordu; 10 ayrı yerde düzeltildi (görüntü/maske okuma, patch parametreleri, harita seçimi, config isim hataları).
2. **GPU hatası giderildi.** ResNet hiç GPU'ya taşınmıyordu, her şey CPU'da çalışıyordu.
3. **Tekrarlanabilirlik sağlandı.** K-means'e ve SVM'e sabit `random_state` verildi; sonuçlar her çalıştırmada aynı.
4. **ImageNet ağırlıkları `IMAGENET1K_V1` yapıldı** (makaledeki MATLAB ResNet-101'e daha yakın).
5. **Özellik önbelleği eklendi** (`load_or_extract_features`).
6. O dönemin çalışma notları `docs/arsiv/2026-08-27/` altında duruyor.

Bu noktadaki kod `main` branch'inde ve `v1.0-bovw` etiketinde.

### Ekim 2026: Sonuçları geliştirme (PR #6)

1. **Train ve test fiziksel olarak ayrıldı** (`data/train`, `data/test`). Augment edilmiş görüntüler yalnızca train tarafında üretiliyor.
2. **1000 boyutlu logit seçeneği eklendi**, 2048 boyutlu avgpool korundu. Önbellek anahtarı katmanı da içeriyor, karışmıyor.
3. **Augmentation maske hataları düzeltildi.** Maskeler döndürülürken bulanıklaşıyordu ve gri tonlamalı okunmuyordu.
4. **SVM ayarı otomatikleştirildi.** `C` ve `gamma` train üzerinde gruplu 10 katlı CV ile seçiliyor.
5. **Makalenin yöntemi gerçek test setinde ölçüldü.** Altı BoVW varyantı (2048/1000 boyut, üç harita, iki harita, haritasız) AUC 0.65-0.68'de kaldı.
6. **Hızlandırma.** İlk çalıştırma yaklaşık 40 dakika sürüyordu; darboğaz CPU'daki harita hesabıydı. Önbellekle 11 dakikaya indi. Lezyon kırpmaları da önbelleğe alındı, 8 iş parçacığıyla paralel hazırlanıp GPU'ya toplu gönderiliyor.
7. **Kontrol deneyleri eklendi.** Patch özelliklerini kümelemek yerine ortalamak AUC 0.76 verdi; lezyonu kırpıp tek vektör almak AUC 0.815 verdi.
8. **Birleştirme denendi.** Tüm lezyon vektörü + BoVW histogramı AUC'yi 0.78'e düşürdü.
9. **Tüm lezyon yöntemi iyileştirildi.** CV ile ayarlı eşik duyarlılığı 0.62'den 0.66'ya çıkardı. 8 yönlü TTA fayda etmedi.

## 5. Sonuçlar

Tüm sayılar gerçek ISIC 2017 test setinde (117 melanom / 483 benign), 17 Ekim 2026 çalıştırmaları. Ham sonuçlar `results/*.json` dosyalarında.

| Yöntem | ACC | SENS | SPEC | PPV | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Tüm lezyon, ResNet-101, ayarlı eşik** | 0.800 | **0.658** | 0.834 | 0.490 | **0.815** | 0.526 |
| Tüm lezyon, ResNet-101, varsayılan eşik | 0.815 | 0.624 | 0.861 | 0.521 | 0.815 | 0.526 |
| Tüm lezyon, ResNet-101 + TTA (8 yön) | 0.828 | 0.504 | 0.907 | 0.567 | 0.811 | 0.536 |
| Birleşim (tüm lezyon + BoVW histogramı) | 0.743 | 0.624 | 0.772 | 0.399 | 0.780 | 0.504 |
| Patch özelliklerinin ortalaması | 0.730 | 0.632 | 0.754 | 0.383 | 0.763 | 0.441 |
| BoVW, logit (1000), PCA×BRI×SAT | 0.602 | 0.684 | 0.582 | 0.284 | 0.679 | 0.327 |
| BoVW, avgpool (2048), haritasız | 0.623 | 0.590 | 0.631 | 0.279 | 0.659 | 0.334 |
| BoVW, avgpool (2048), PCA×BRI×SAT | 0.625 | 0.564 | 0.640 | 0.275 | 0.657 | 0.333 |
| BoVW, avgpool (2048), PCA×BRI | 0.667 | 0.530 | 0.700 | 0.300 | 0.649 | 0.287 |

### Çıkarımlar

- BoVW'nin kümeleme adımı bilgi kaybettiriyor: aynı patch özelliklerini sadece ortalamak bile AUC'yi 0.66'dan 0.76'ya çıkarıyor.
- Lezyonu bir bütün olarak ağa vermek en iyisi (AUC 0.815). Üstüne BoVW eklemek sonucu kötüleştiriyor.
- PCA/BRI/SAT haritaları bizim testimizde belirgin fark yaratmadı (haritasız 0.659, üç harita 0.657 AUC). Makalede en büyük katkıyı bu haritalar sağlıyordu.
- TTA AUC'yi değiştirmedi, duyarlılığı 0.62'den 0.50'ye düşürdü.

### Makale ile karşılaştırma

- Makale (Tablo 2) PCA×BRI×SAT için ACC 0.962, SENS 0.998, SPEC 0.925, PPV 0.931 bildiriyor.
- Bu sayılar dengesiz test setiyle tutarlı görünmüyor: 117/483 dağılımında SENS 0.998 ve SPEC 0.925 ile PPV en fazla yaklaşık 0.76 olabilir, 0.93 olamaz. Makale muhtemelen dengelenmiş bir sette ölçmüş.
- Makalenin Tablo 3'ündeki ISIC 2017 ilk 10 takımı: ACC 0.827-0.872, SENS 0.171-0.735. Bizim sonucumuz (ACC 0.80, SENS 0.66) bu sıralamaya yakın ve duyarlılıkta çoğundan iyi.

## 6. Alınan kararlar

| Karar | Gerekçe |
|---|---|
| Train ve test fiziksel olarak ayrıldı | Hocamızın isteği; veri sızıntısı ihtimalini kaldırıyor. |
| 1000 boyutlu logit seçeneği eklendi, 2048 de korundu | Hocamız 1000 istedi, makale 2048 kullanıyor. BoVW'de logit biraz daha iyi (AUC 0.68'e karşı 0.66). |
| SVM parametreleri train'de CV ile seçiliyor | Makale de otomatik ayarlı SVM kullanıyor; sabit C/gamma adil bir karşılaştırma değil. |
| Ana yöntem: tüm lezyon + derin ağ + SVM | En yüksek AUC (0.815). |
| BoVW raporda karşılaştırma olarak kalıyor | Makalenin yöntemini uyguladığımızı ve neden geçtiğimizi gösteriyor. |
| TTA kullanılmıyor | AUC değişmedi, duyarlılık düştü. Melanomda duyarlılık daha önemli. |
| Karar eşiği CV ile ayarlanıyor | Duyarlılık 0.62'den 0.66'ya çıkıyor, özgüllük 0.86'dan 0.83'e iniyor. |
| `main` korunuyor, güncel hal ayrı branch'te | Eski versiyon kaybolmasın, istenirse karşılaştırılabilsin. |

## 7. Sonraki adımlar

- [ ] Daha güçlü omurga ağlar: ConvNeXt-Base, EfficientNetV2-M, DenseNet-201 (kod hazır, sadece `GLOBAL_BACKBONE` değiştirilip çalıştırılması gerekiyor).
- [ ] En iyi iki ağın vektörlerini birleştirmek.
- [ ] Sonuçları raporda BoVW ile yan yana sunmak (ablation tablosu).
- [ ] Güncel branch'i inceleyip `main`'e birleştirmeye karar vermek.

### Hocaya sorular

- Ana yöntemi "tüm lezyon + derin ağ + SVM" olarak değiştirmemiz uygun mu, yoksa BoVW'nin ana yöntem olarak kalması mı bekleniyor?
- Ağı ince ayarla (fine-tuning) eğitmemiz serbest mi, yoksa makaledeki gibi sadece özellik çıkarıcı olarak mı kalmalı?
- Başarı ölçütü olarak hangi metriği öne çıkarmalıyız (AUC, duyarlılık, dengeli doğruluk)?

## 8. Dosya haritası

| Dosya / klasör | Ne içeriyor |
|---|---|
| `config.py` | Tüm yollar ve hiperparametreler |
| `dataset.py` | CSV'den görüntü/maske yolları ve etiketler |
| `augmentation.py` | Sadece train için melanom çoğaltma |
| `preprocessing.py` | Ölçekleme ve patch kesme |
| `patch_enhancement.py` | PCA, BRI, SAT haritaları |
| `feature_extraction.py` | ResNet-101 özellik çıkarımı ve önbellek |
| `bovw.py` | Sınıf bazlı sözlük ve ağırlıklı histogram |
| `classifier.py` | SVM, CV ile parametre ve eşik seçimi |
| `evaluation.py` | Metrikler |
| `main.py` | Yol A (BoVW) |
| `baseline.py` | Yol B (tüm lezyon) ve karşılaştırma modları |
| `results/` | Her çalıştırmanın ayarları ve metrikleri (JSON) |
| `docs/SUREC.md` | Bu doküman |
| `docs/arsiv/2026-08-27/` | Ağustos dönemi çalışma notları |
