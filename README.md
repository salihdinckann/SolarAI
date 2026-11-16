# SolarAI

**Güneş Panellerinde Kirlenmenin Yapay Zekâ ile Tespiti ve Ekonomik Temizlik Planlaması**

> Panelin ne kadar kirli olduğunu değil, ne kadar kaybettirdiğini söyler.

**Canlı demo:** https://solarai-gaziantep.streamlit.app

SolarAI bir güneş paneli fotoğrafından panelin durumunu (temiz, tozlu, kuş pisliği, kar, fiziksel hasar,
elektriksel hasar) tahmin eder, bu durumun Gaziantep koşullarında kaç kWh ve kaç TL kaybettirdiğini hesaplar
ve panelin ne zaman temizlenmesinin ekonomik olduğunu planlar.

HASAT-13 Proje Pazarı (Hasan Kalyoncu Üniversitesi, 25–26 Kasım 2026) için geliştirilmiştir.
Temalar: Sürdürülebilir Enerji · Yapay Zekâ ve Sürdürülebilirlik.

## Nasıl çalışır?

1. **Sınıflandırma:** ImageNet'te ön eğitimli EfficientNet-B0 (timm), 6 sınıf üzerinde ince ayarla eğitildi.
2. **Açıklanabilirlik:** Grad-CAM, modelin kararını fotoğrafın hangi bölgesine dayandırdığını gösterir.
3. **Kayıp hesabı:** `kayıp = Σ p(sınıf) · kayıp(sınıf)`. Sınıf kayıp oranları literatürden alındı.
   Günlük üretim için Gaziantep PVGIS verisi kullanıldı. Hasar sınıflarında TL tahmini yerine servis uyarısı verilir.
4. **Temizlik planı:** Aylık en uygun aralık `T* = √(2C / (r·E·p))`. Yıllık takvim, dinamik programlamayla
   toplam maliyeti (temizlik + kayıp) en aza indirerek hesaplanır.

## Sonuçlar (model `effnet_b0_v2`)

| Ölçüt | Değer |
|---|---|
| Test macro-F1 | **0.888** |
| Test accuracy | 0.894 (101/113) |
| Dış test, kar (44 yeni görüntü) | 44/44 |
| Dış test, fiziksel hasar (17 yeni görüntü) | 2/17 |

Veri setindeki 91 birebir kopya ve 86 yakın kopya grubu bulunup bölme öncesinde ayıklandı
(eğitim–test sızıntısı yok). Ayrıntılar: [`results/effnet_b0_v2/README.md`](results/effnet_b0_v2/README.md).
Sınırlamalar: küçük test kümesi (±6 puan), fiziksel hasar sınıfı yalnızca yakın çekim kırık cam
fotoğraflarıyla genelleşiyor, kirlenme oranı r ve temizlik ücreti varsayımdır.

## Kurulum

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU yoksa: pip install torch torchvision
pip install -r requirements.txt
```

Veri: Kaggle'dan [Solar Panel Images Clean and Faulty Images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images)
setini indirip `data/raw/` içine açın (`data/raw/Faulty_solar_panel/<sınıf>/...`).

## Kullanım

```bash
python src/prepare_data.py                 # temizlik + yakın kopya gruplama + bölme -> data/splits.csv
python src/train.py --run effnet_b0_v2     # eğitim -> outputs/effnet_b0_v2/
python src/gradcam_examples.py             # Grad-CAM örnekleri
python tests/test_loss_engine.py           # kayıp motoru testleri
streamlit run app/app.py                   # demo uygulaması
```

## Klasör yapısı

```
app/            Streamlit uygulaması
data/           PVGIS Gaziantep verisi, bölme listeleri (görüntüler depoda değil)
docs/           Literatür notları ve kaynaklar
results/        Dondurulmuş sonuçlar (metrikler, karışıklık matrisi, Grad-CAM)
src/            Veri hazırlama, eğitim, Grad-CAM, kayıp ve temizlik hesap motoru
tests/          Birim testleri
```

## Kaynaklar

- Veri: Afroz, *Solar Panel Images Clean and Faulty Images*, Kaggle. Dış test: Lenarczyk, *PV Panel Defect Dataset*, Kaggle (CC BY-NC-SA 4.0).
- Üretim verisi: PVGIS 5.3, European Commission JRC.
- Grad-CAM: Selvaraju vd., 2017, ICCV.
- EfficientNet: Tan & Le, 2019, ICML.
- Kayıp oranları ve kirlenme hızı: [`docs/literatur_notlari.md`](docs/literatur_notlari.md)
