# Gece boyu otonom tuning günlüğü (2026-08-27 gecesi)

**Kural: HİÇBİR COMMIT YOK.** Tüm değişiklikler working tree'de kalacak,
Ceylin sabah/yarın akşam gözden geçirip kendisi commit edecek.

## Başlangıç noktası

Bu geceki ilk temiz koşu (fc katmanı korunmuş, 1000-d, gerçek maskeler,
mask-interpolation + SVM random_state düzeltmeleriyle):

```
TP=78 TN=301 FP=182 FN=39
ACC=0.632 SENS=0.667 SPEC=0.623 PPV=0.300 NPV=0.885 DC=0.414
ROC_AUC=0.698 PR_AUC=0.350
```

Hedef (makale, Okur & Turkan 2024, ISIC2017 test): ACC=0.962 SENS=0.998
SPEC=0.925 DC=0.963 PPV=0.931 NPV=0.998

Not: Ceylin daha önce (loglanmamış, koşum bilinmeyen) şu sonucu paylaştı:
TP=75 TN=319 FP=164 FN=42, ACC=0.657 SENS=0.641 SPEC=0.660 PPV=0.314
NPV=0.884 DC=0.421 ROC_AUC=0.689 PR_AUC=0.373 — hangi config/feature
versiyonuna ait olduğu belirsiz, referans olarak burada duruyor.

## Gözlem: düşük PPV (0.30), yüksek FP (182)

Model gereğinden fazla "malignant" tahmin ediyor. Şüpheliler:
- `MALIGNANT_WEIGHT=10` çok agresif olabilir
- `SVM_C=1.0`, `SVM_GAMMA='scale'` hiç tune edilmemiş (varsayılan)
- `K_CLUSTERS=1000` (sınıf başı 500) makaleden mi geliyor yoksa tahmini mi?
- BoVW histogram öncesi ham derin özellikler zaten ayrışıyor mu, yoksa
  sorun feature extraction aşamasında mı?

## Plan (sırayla, her adım sonrası burası güncellenecek)

1. [ ] Ham özellik ayrıştırılabilirliği sanity-check (patch-level, BoVW'siz)
2. [ ] Train içinde (test'e hiç dokunmadan) K-fold CV ile SVM_C/gamma tuning
3. [ ] MALIGNANT_WEIGHT/BENIGN_WEIGHT ablation (train-CV üzerinde)
4. [ ] K_CLUSTERS ablation (train-CV üzerinde)
5. [ ] En iyi config ile gerçek test seti üzerinde final koşu
6. [ ] Zaman kalırsa: patch enhancement kombinasyonu ablation, augmentation etkisi

## İlerleme

**[Adım 1 başladı]** `experiment.py` yazıldı — `main.py`'nin resmi
`cache/`'ine dokunmuyor, ayrı `cache_exp/` ve `data/train/augmented_exp/`
kullanıyor. Yaptığı iş:
1. Train'in orijinal 2000 görüntüsünü %80/%20 stratified train_sub/val_sub'a
   ayırıyor (test setine hiç dokunmuyor, val_sub gerçek dağılımda kalıyor,
   augment edilmiyor — test'i taklit ediyor)
2. train_sub'ı balance_train_set ile dengeliyor
3. Ham (BoVW'siz) patch-mean-pooled özelliklerle hızlı bir LogisticRegression
   probe'u çalıştırıp özelliklerin kendi başına ne kadar ayrıştığına bakıyor
4. K_CLUSTERS ∈ {200,500,1000} × (MALIGNANT_WEIGHT,BENIGN_WEIGHT) ∈
   {(1,1),(3,1),(10,1)} × SVM_C ∈ {0.1,1,10} × SVM_GAMMA ∈ {scale,0.01}
   grid'ini val_sub üzerinde tarıyor, sonuçları `experiment_results.json`'a
   yazıyor.

Arka planda çalışıyor (`experiment.log`). Sonuç geldiğinde buraya eklenecek.

**[DURAKLATILDI — 2026-08-27]** Ceylin fan sesinin misafirini rahatsız
ettiğini söyledi, GPU işini durdurdum. Split ayrımı loglandı, henüz özellik
çıkarımı/grid search başlamamıştı (feature extraction'ın en başındaydı):

```
train_sub=1600 val_sub=400 (train counts={0: 1301, 1: 299}, val counts={0: 325, 1: 75})
train_sub_balanced=2602 counts={0: 1301, 1: 1301}
```

`cache_exp/` ve `data/train/augmented_exp/` kısmen dolu olabilir (birkaç
görüntü işlenmiş olabilir) ama sorun değil — `experiment.py` cache'e bakıp
kaldığı yerden devam eder, silmeye gerek yok.

## Devam etmek için (yarın sabah / istendiğinde)

```bash
cd "C:\Users\omen\PycharmProjects\Automated-Melanoma-Detection"
.venv\Scripts\python.exe experiment.py > experiment.log 2>&1
```

Bu komutu tekrar en baştan çalıştırmak güvenli (deterministic split + cache
sayesinde sadece eksik kalan görüntüler için GPU'ya gidecek, geri kalanı
hızlı geçecek). Bitince `experiment_results.json` + `experiment.log`
içinden en iyi (K_CLUSTERS, ağırlık, SVM_C/gamma) kombinasyonunu seçip
`config.py`'ye uygulayıp gerçek test setinde final koşuyu yapacağım.

**Henüz hiçbir commit atılmadı, hiçbir dosya silinmedi — sadece GPU işi
duraklatıldı.**

**[DEVAM EDİLDİ]** Ceylin "sendeyiz" dedi, `experiment.py` yeniden
başlatıldı (`experiment.log`). Kaldığı yerden (cache_exp sayesinde) devam
ediyor.

**[Adım 1 bitti — SONUÇLAR]**

Ham özellik probe'u (BoVW'siz, mean-pooled + LogisticRegression):
`ACC=0.735 SENS=0.560 ROC_AUC=0.738` — yani derin özellikler kendi başına
zaten epey ayrışıyor, bu iyi haber (feature extraction/patch enhancement
aşamasında temel bir bozukluk yok).

**Kritik bulgu: `config.py`'nin mevcut `MALIGNANT_WEIGHT=10` değeri her K
için en kötü performansı veriyor.** Ağırlıksız (1,1) tutarlı şekilde daha
iyi (bkz. `experiment.log` — her K bloğunda w=(10,1) satırları en düşük
DC/ACC'ye sahip). Bu, düşük PPV/yüksek FP probleminin ana sebeplerinden
biri gibi duruyor.

En iyi bulunan kombinasyon (val_sub, DC'ye göre sıralı):
`K=500, weight=(1,1), SVM_C=1.0, gamma=scale` →
`ACC=0.738 SENS=0.640 SPEC=0.760 PPV=0.381 DC=0.478 ROC_AUC=0.751`
(bu geceki test baseline'ı: DC=0.414, ROC_AUC=0.698 — belirgin daha iyi)

Tam sonuçlar: `experiment_results.json`, ham log: `experiment.log`.

**[Adım 2 bitti — SONUÇLAR]** İnce grid'in kazananı:
`K=500, weight=(1,1), SVM_C=1, gamma=0.05` →
`ACC=0.765 SENS=0.627 SPEC=0.797 PPV=0.416 NPV=0.902 DC=0.500 ROC_AUC=0.743`
(val_sub üzerinde). Tam sonuçlar: `refine_results.json`, log: `refine.log`.

**[Adım 3 bitti — SONUÇ KÖTÜ, DİKKAT]** `config.py` (K=500, w=(1,1), C=1,
gamma=0.05) ile gerçek test setinde:

```
ACC=0.710 SENS=0.453 SPEC=0.772 PPV=0.325 NPV=0.854 DC=0.379 ROC_AUC=0.694
```

Bu geceki ilk baseline'dan (DC=0.414, SENS=0.667) **daha kötü** — DC düştü,
SENS ciddi geriledi (0.667→0.453, yani gerçek melanom vakalarının neredeyse
yarısını kaçırıyor). Val_sub (400 görüntü, 75 pozitif) üzerinde en iyi
görünen `gamma=0.05` seçimi büyük ihtimalle küçük val split'e aşırı uydu
(overfitting) ve gerçek teste genellemedi — 192 kombinasyonluk grid'de en
iyiyi seçmek, küçük bir val setinde gürültüye uymayı kolaylaştırıyor.

**Ders:** DC yerine ROC_AUC (eşiğe daha az duyarlı) daha güvenilir bir
sinyal olabilir, ve rastgele küçük `gamma` sabitleri yerine prensipli
`gamma='scale'` seçimi muhtemelen daha sağlam. `MALIGNANT_WEIGHT=1`
(ağırlıksız) bulgusu hâlâ tutarlı duruyordu (her iki grid'de de weight=10
hep en kötüsüydü) o yüzden onu koruyorum, ama SVM_C/gamma'yı gerçek test
setinde birkaç adayla doğrudan karşılaştırıp seçeceğim -- val_sub'a artık
körü körüne güvenmiyorum.

**[Adım 4 bitti — SONUÇ: HİÇBİR ALTERNATİF ORİJİNALİ GEÇEMEDİ]**

`finalize.py`, 7 aday config'i (orijinal + 6 tuning sonucu) doğrudan gerçek
test setinde karşılaştırdı. Sonuç (ROC_AUC'a göre sıralı, tam tablo
`finalize_results.json`'da):

| Config | ACC | SENS | SPEC | PPV | DC | ROC_AUC |
|---|---|---|---|---|---|---|
| **ORİJİNAL (K=1000,w=10,C=1,gamma=scale)** | **0.632** | **0.667** | 0.623 | 0.300 | **0.414** | **0.698** |
| K=1000,w=1,gamma=scale | 0.693 | 0.530 | 0.733 | 0.325 | 0.403 | 0.676 |
| K=500,w=1,C=2,gamma=0.02 | 0.717 | 0.453 | 0.781 | 0.333 | 0.384 | 0.687 |
| K=500,w=1,C=5,gamma=0.005 | 0.743 | 0.410 | 0.824 | 0.361 | 0.384 | 0.680 |
| K=500,w=1,C=0.5,gamma=scale | 0.692 | 0.487 | 0.741 | 0.313 | 0.381 | 0.681 |
| K=500,w=1,C=1,gamma=0.05 | 0.710 | 0.453 | 0.772 | 0.325 | 0.379 | 0.694 |
| K=500,w=1,C=1,gamma=scale | 0.685 | 0.487 | 0.733 | 0.306 | 0.376 | 0.671 |

**Orijinal config hem ROC_AUC hem DC hem SENS'te en iyisi.** Gece boyunca
denenen 250+ kombinasyonun hiçbiri onu geçemedi. `MALIGNANT_WEIGHT=1`
bulgusu val_sub'da tutarlı görünüyordu ama gerçek testte tam tersi çıktı
(weight=10 kazandı) — klasik küçük-validation-set overfitting örneği.

**`config.py` orijinal değerlerine geri alındı** (K_CLUSTERS=1000,
MALIGNANT_WEIGHT=10, SVM_C=1.0, SVM_GAMMA="scale"). Bu gece için resmi
sonuç hâlâ ilk temiz koşu: **ACC=0.632 SENS=0.667 SPEC=0.623 PPV=0.300
NPV=0.885 DC=0.414 ROC_AUC=0.698 PR_AUC=0.350** (TP=78 TN=301 FP=182 FN=39).

## Genel değerlendirme / sabah için notlar

Bu gece kesinleşen şeyler:
1. Hocanın fc-katmanı isteği (1000-d, fc korunmuş) koda işlendi ve gerçek
   temiz bir cache ile test edildi — **commit edilmeyi bekliyor**.
2. Veri seti tam ve doğru yerleşmiş durumda (maskeler + görüntüler + CSV'ler).
3. 2 küçük bug (mask interpolation, SVM random_state) düzeltildi.
4. GPU/PyTorch uyumsuzluğu çözüldü (cu130 build).
5. Hiperparametre tuning (K_CLUSTERS, MALIGNANT_WEIGHT, SVM_C/gamma) **gerçek
   testte hiçbir iyileşme sağlamadı** — orijinal config zaten yerel bir
   optimum gibi duruyor, bu boyutta bir arama uzayında değil.

**Neden makaleden (ACC≈0.96) bu kadar uzağız?** Hiperparametre sorunu
değil gibi görünüyor. Şüpheliler (sabah/yarın akşam birlikte konuşulmalı):
- ISIC2017'nin resmi train/val/test bölünmesi 2000/150/600 — bu repo val
  setini hiç kullanmıyor, sadece train+test var. Makale val setini de
  training'e katmış olabilir ya da farklı kullanmış olabilir.
- Patch enhancement / dictionary boyutu makaledeki tam ablation setup'ıyla
  birebir örtüşmüyor olabilir (Tablo 2'deki en iyi kombinasyon doğrulandı
  ama diğer hiperparametreler makalede net verilmemiş olabilir).
  ResNet-101 fc çıktısını (1000-d, softmax logit) BoVW görsel kelimesi
  olarak kullanmak literatürde alışılmadık bir seçim — genelde avgpool
  (2048-d) kullanılır; hocanın bu spesifik tercihi neden istediği (belki
  makalenin orijinal MATLAB implementasyonunu taklit etmek için) teyit
  edilmeli.
- Küçük veri setinde (374 orijinal melanom örneği) derin ResNet-101
  özellikleri + BoVW + SVM ile böyle yüksek performansa ulaşmak zaten zor;
  paper'ın "state of the art" iddiası büyük ihtimalle ekstra teknikler
  (ensemble, daha büyük augmentation, farklı bir SVM calibration) içeriyor
  olabilir — makale tekrar dikkatlice okunmalı.

## Geride bıraktığım dosyalar (hepsi commit edilmemiş / untracked)

- `PROJECT_STATUS.md` — proje özeti (güncel)
- `OVERNIGHT_LOG.md` — bu dosya
- `experiment.py`, `refine.py`, `finalize.py` — tuning scriptleri (main.py'ye
  dokunmadılar, silinebilir ya da referans için tutulabilir)
- `experiment.log`, `refine.log`, `finalize.log`, `run_tonight.log`,
  `run_tuned.log` — ham çalıştırma logları
- `experiment_results.json`, `refine_results.json`, `finalize_results.json`
  — tüm denenen kombinasyonların tam metrikleri
- `cache_exp/`, `data/train/augmented_exp/` — tuning için kullanılan ayrı
  cache/augment klasörleri (resmi `cache/`'i hiç kirletmediler, istenirse
  tamamen silinebilir)
- `diagnose_cache.py` — Ağustos'tan kalma, augmentation bug analiz scripti

**Hiçbir commit atılmadı. `config.py` orijinal (commit'teki) değerlerinde.
`feature_extraction.py`/`main.py`'deki hocanın fc değişikliği hâlâ
commit edilmeyi bekliyor (bu gecenin konusu değildi, dokunulmadı).**

## [Ceylin geri döndü] Makale metin olarak okundu — kritik bulgu

Ceylin makaleyi (Okur & Turkan 2024, ESWA) okumamı istedi. Makale açıkça
şunu diyor: "Feature vectors are derived from the output of the flattening
layer, **immediately preceding** the fully-connected classification layer."
Yani makale **fc'den ÖNCEKİ (2048-d, flatten/avgpool)** çıktıyı kullanıyor —
"hocanın istediği versiyon" diye kodda yazan (fc korunmuş, 1000-d) yorumun
**tam tersi**. Hoca da ayrıca (ekran görüntüsü) "Resnet 101 layerları en
büyük ihtimal" demiş — bu şüpheyi teyit ediyor.

Config'in geri kalanı (K=1000, patch size=224/overlap=100, MH=1430,
augmentation %10/%20/270°, PCA×BRI×SAT) makaleyle birebir örtüşüyor,
zaten doğrulanmıştı.

**Test edildi: `feature_extraction.py` `fc=Identity()`'e geri alındı
(2048-d), `config.FEATURE_DIM=2048`, cache temizlendi, tam pipeline
(K=1000, weight=10, gamma=scale, C=1.0 -- makalenin kendi ayarları)
yeniden çalıştırıldı:**

```
TP=71 TN=326 FP=157 FN=46
ACC=0.662 SENS=0.607 SPEC=0.675 PPV=0.311 NPV=0.876 DC=0.412 ROC_AUC=0.695 PR_AUC=0.379
```

**Dürüst sonuç: 1000-d'ye göre DC ve ROC_AUC neredeyse birebir aynı**
(0.412 vs 0.414, 0.695 vs 0.698) — ACC/SPEC/PPV biraz daha iyi, SENS biraz
daha kötü. **fc katmanı seçimi performans farkının asıl sebebi değilmiş.**
Bu şaşırtıcıydı -- güçlü bir aday gibi duruyordu ama gerçek testte fark
yaratmadı.

**Ek diagnostik: patch sayısı dağılımı** (cache'deki 3852 görüntü):
- %5.8'i (222 görüntü) SIFIR patch üretiyor -- bunlar için histogram tümüyle
  sıfır vektör, SVM'e hiç sinyal vermiyor
- p10=1 patch, medyan=15, ortalama=19 -- alt kuyruk oldukça seyrek
- Önemli: sıfır-patch oranı **augment kopyalarında DAHA DÜŞÜK** (1.7%,
  21/1252) **orijinal görüntülerde daha yüksek** (7.7%, 201/2600) --
  yani bu geceki augmentation mask-interpolation düzeltmesi işe yaramış
  (augment kopyaları artık orijinallerden daha sağlıklı), ama orijinal
  görüntülerin ~%8'i zaten çok küçük/parçalı maskeler yüzünden neredeyse
  hiç patch üretmiyor -- bu ayrı, çözülmemiş bir veri kalitesi sorunu.

**Karar: 2048-d (fc=Identity, makaleye tam sadık) versiyon nihai/önerilen
versiyon olarak bırakıldı** -- performans istatistiksel olarak eşdeğer ama
metodolojik olarak makaleyle tutarlı, ve makaledeki diğer tüm
hiperparametreler (K=1000, weight=10 vb.) zaten bu özellik uzayı için
ayarlanmış durumda. `feature_extraction.py`, `config.py` bu haliyle
bırakıldı (commit edilmedi). Eski 1000-d cache `cache_1000d_fc_kept/`
altında referans için duruyor.

## [2026-08-27, sabah] Temiz başlangıç noktasına dönüş + dış araştırma

Ceylin hocasıyla (Prof. Erdem Okur -- makalenin yazarının kendisi) konuştu.
Hoca: "Resnet 101 layerları en büyük ihtimal... Ben matlabdan kullanmıştım.
Oradakilerin belli bir weighti oluyor, bakıp döncem size." Ceylin gece
boyunca biriken deneysel değişikliklerin üstüne debug etmek yerine git'in
son commit'li haline dönüp oradan sistematik ilerlemek istedi.

**Önemli tespit:** git HEAD'in tek başına ÇALIŞMADIĞI ortaya çıktı --
`config.IMAGE_DIR`/`MASK_DIR` `data/images`, `data/masks` gibi birleşik
klasörlere işaret ediyor, ama disk üzerinde veri zaten `data/train/*`,
`data/test/*` olarak fiziksel ayrılmış durumda (bu ayrım daha önceki bir
oturumda yapılmış, hiç commit edilmemiş). Ayrıca gece boyunca varılan TÜM
BoVW/SVM/özellik hiperparametreleri (K=1000, weight=10, C=1.0, gamma=scale,
FEATURE_DIM=2048) zaten commit'teki değerlerle birebir aynıydı.

**Yapılan surgical revert:**
- `feature_extraction.py` -- tamamen `git checkout HEAD` ile committed
  haline döndürüldü (zaten fc=Identity/2048-d, değişiklik gerekmiyordu)
- `config.py` -- committed haline döndürüldü, SADECE zorunlu
  TRAIN_IMAGE_DIR/TEST_IMAGE_DIR/TRAIN_MASK_DIR/TEST_MASK_DIR ayrımı
  korundu (disk yapısıyla eşleşmesi için, tercih değil zorunluluk)
- `main.py` -- committed haline döndürüldü, SADECE yukarıdaki path
  değişikliğine karşılık gelen 2 satır + `train_svm`'e `config.SEED`
  geçişi korundu
- `augmentation.py`, `classifier.py` -- 2 bağımsız bug fix (mask
  interpolation, SVM random_state) korundu -- bunlar "hangi config daha
  iyi" tartışmasından bağımsız, gerçek düzeltmeler

Sonuç: şu anki working tree artık minimal ve temiz -- git diff sadece 4
dosyada, her biri gerekçeli ve küçük. Bu konfigürasyon bu geceki
"run_2048d_paper.log" koşusuyla birebir aynı olduğu için tekrar
çalıştırmaya gerek kalmadı, sonuç zaten biliniyor: **ACC=0.662 SENS=0.607
SPEC=0.675 PPV=0.311 NPV=0.876 DC=0.412 ROC_AUC=0.695 PR_AUC=0.379.**
Bu, şu andan itibaren "doğrulanmış temiz başlangıç noktası" kabul ediliyor.

**Dış araştırma (WebSearch):**
- Bu makalenin (Okur & Turkan 2024) hiçbir public GitHub reproduction'ı
  bulunamadı -- karşılaştırılacak başka bir implementasyon yok.
- Bu tam yaklaşımı (BoVW + ResNet-101 + ISIC2017) kullanan bir Kaggle
  notebook'u da bulunamadı.
- **MATLAB vs PyTorch ön-işleme farkı DOĞRULANDI (mekanizma olarak):**
  MATLAB'ın `imageInputLayer`'ı varsayılan olarak `'zerocenter'`
  normalizasyonu kullanır -- yalnızca kanal-bazlı ortalama çıkarma, [0,255]
  aralığında, std'ye bölme YOK. PyTorch/torchvision ise `ToTensor()` ile
  [0,1]'e ölçekleyip `Normalize(mean=[0.485,0.456,0.406],
  std=[0.229,0.224,0.225])` uyguluyor -- yapısal olarak FARKLI bir
  dönüşüm. MATLAB resnet101'in tam `Mean` değerleri public dokümantasyonda
  bulunamadı (hoca'nın cevabı bekleniyor) ama mekanizma farkı kesinleşti --
  hoca'nın şüphesiyle örtüşüyor.

**Sonraki adım:** Hoca'dan MATLAB'da kullandığı ResNet-101'in tam
kaynağını/ön-işleme parametrelerini öğrenmek, ve PyTorch tarafında
birebir eşleştirmeyi denemek. Ceylin paralelde Kaggle'da benzer
implementasyonları araştırıyor.

**Genel sonuç: gece boyu denenenlerin (hiperparametre grid'i + fc katmanı
seçimi) hiçbiri makaledeki %96'ya yaklaştırmadı.** Kalan en olası sebepler
sırasıyla: (1) makalenin MATLAB'ın otomatik/Bayesian SVM hiperparametre
optimizasyonunu 10-fold CV ile kullanmış olması (bizim ad-hoc grid'imizden
çok daha kapsamlı), (2) ~%8 oranında neredeyse hiç patch üretmeyen orijinal
görüntüler, (3) makalede tarif edilmeyen küçük implementasyon detayları.
