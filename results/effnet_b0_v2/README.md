# effnet_b0_v2 — dondurulmuş sonuçlar (POSTER RAKAMLARI)

- Tarih: 16 Kasım 2026
- Model: EfficientNet-B0 (timm, ImageNet ön eğitimli), 224 px — ayarlar v1 ile aynı
- Veri: Kaggle "Solar Panel Images Clean and Faulty Images" (pythonafroz), temizlik sonrası 790 görüntü
- Bölme: `splits_v2.csv` — **yakın kopya grupları korunarak** stratified 563 / 114 / 113, seed 42
  - Yakın kopya: dHash Hamming ≤ 12 veya ImageNet öznitelik kosinüs benzerliği ≥ 0.88 (86 grup)
  - Bağımsız kontrol: bölmeler arasında dHash ≤ 30 ve benzerlik ≥ 0.88 olan çift sayısı 0
- Komut: `python src/prepare_data.py` ardından `python src/train.py --run effnet_b0_v2`
- En iyi epoch: 11 / 18 (erken durdurma)

| Ölçüt | Değer |
|---|---|
| Doğrulama macro-F1 | 0.893 |
| **Test macro-F1** | **0.888** |
| Test accuracy | 0.894 (101 / 113) |

| Sınıf | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Temiz | 0.806 | 0.962 | 0.877 | 26 |
| Tozlu | 0.960 | 0.923 | 0.941 | 26 |
| Kuş pisliği | 0.944 | 0.739 | 0.829 | 23 |
| Kar | 0.941 | 1.000 | 0.970 | 16 |
| Fiziksel hasar | 0.875 | 0.778 | 0.824 | 9 |
| Elektriksel hasar | 0.857 | 0.923 | 0.889 | 13 |

Servis uyarısı (fiziksel + elektriksel hasar birlikte): 22 hasarlı panelin 19'u yakalandı; 91 hasarsız panelde 3 yanlış alarm.
Kaçan 3 hasarın 2'si "temiz" tahmin edildi (Electrical (75), Physical (53)).

Not: Test kümesi küçüktür (113 görüntü, %95 güven aralığı kabaca ±6 puan). Sonuçlar gösterge niteliğindedir.

Model ağırlıkları (`best.pt`, ~16 MB) depoda değil: `outputs/effnet_b0_v2/best.pt`
SHA-256 (ilk 16 karakter): `3aec827a7b2e1b0f`

## Dış test (veri seti dışı görüntüler)

Kaynak: Kaggle "PV Panel Defect Dataset" (Lenarczyk, CC BY-NC-SA 4.0). 1574 dosyanın 871'i benzersiz; bunların
805'i bizim veri setimizle aynı/yakın kopya (MD5, dHash ayna/döndürme varyantları, ImageNet benzerliği ≥ 0.88).
Kalan **66 yeni görüntü** dış test olarak kullanıldı (`data/external_test.csv`, `python src/build_external_test.py`).
Model bu görüntüleri hiç görmedi ve bunlar başka bir kişi tarafından farklı kaynaklardan toplandı.

| Sınıf | n | Recall |
|---|---|---|
| Kar | 44 | **1.00** (44/44) |
| Elektriksel hasar | 5 | 0.80 (4/5) |
| Fiziksel hasar | 17 | **0.12** (2/17) |

Servis uyarısı: 9/22 hasarlı panel. Hiçbir hasarlı panel "temiz" tahmin edilmedi.

Yorum: Eğitim verisindeki fiziksel hasar örnekleri neredeyse tamamen **yakın çekim kırık cam** fotoğrafları.
Dış kümedeki fiziksel hasarlar ise fırtına/dolu hasarı, devrilmiş diziler, geniş açılı saha fotoğrafları.
Model "fiziksel hasar" kavramını dar öğrenmiş; bu bir **dağılım kayması (distribution shift)** örneği.
Kar sınıfı ise farklı kaynaklardan gelen görüntülerde de genelleşiyor (Grad-CAM çoğunlukla kar üzerinde).
Not: Dış kümede yalnızca 3 sınıf var; temiz/tozlu/kuş pisliği dışarıda test edilemedi.
Yeniden üretmek için: Lenarczyk setini Kaggle'dan indirip `data/lenarczyk/` içine açın, sonra
`python src/build_external_test.py` ve `python src/evaluate.py` çalıştırın (görüntüler depoda tutulmuyor).
