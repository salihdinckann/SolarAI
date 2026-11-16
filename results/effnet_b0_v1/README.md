# effnet_b0_v1 — dondurulmuş sonuçlar (poster rakamları)

- Tarih: 16 Kasım 2026
- Model: EfficientNet-B0 (timm, ImageNet ön eğitimli), 224 px
- Veri: Kaggle "Solar Panel Images Clean and Faulty Images" (pythonafroz), kopyalar temizlenmiş 792 görüntü
- Bölme: `data/splits.csv` (stratified 554 / 119 / 119, seed 42)
- Komut: `python src/train.py --run effnet_b0_v1`
- En iyi epoch: 10 / 17 (erken durdurma)

| Ölçüt | Değer |
|---|---|
| Doğrulama macro-F1 | 0.856 |
| **Test macro-F1** | **0.886** |
| Test accuracy | 0.874 (104 / 119) |

Hasar sınıfları (fiziksel + elektriksel): 23 hasarlı panelin 22'si servis uyarısı tetikleyen bir sınıfa atandı.

Not: Test kümesi küçüktür (119 görüntü). Sonuçlar gösterge niteliğindedir.

Model ağırlıkları (`best.pt`, ~16 MB) depoda değil, yerelde `outputs/effnet_b0_v1/best.pt` dosyasında duruyor.
SHA-256 (ilk 16 karakter): `1e6d01776c1a41b7`
