# Literatür notları — kayıp oranları ve kirlenme hızı

Bu notlar kayıp motorundaki (`src/loss_engine.py`) varsayılan değerlerin dayanağıdır.
Kaynaklar 16 Kasım 2026'da web aramasıyla bulundu; "doğrulanmadı" işaretli olanların orijinal metni okunmadı,
posterde kullanmadan önce DOI üzerinden kontrol edilmeli.

## 1. Sınıf başına anlık üretim kaybı

| Sınıf | Literatür aralığı | Önerilen varsayılan | Dayanak |
|---|---|---|---|
| Temiz | 0 | **%0** | — |
| Tozlu | %5–17 (Bursa, ölçüm); %18 (Şanlıurfa, simülasyon); kurak bölgelerde %10–40+ | **%15** | [1], [2], [3] |
| Kuş pisliği | ~%24–35 (yerel gölgelenme, bypass diyot etkisi; kaplama alanına çok bağlı) | **%25** | [4], [5] |
| Kar | Kısmi örtüde onlarca %, tam örtüde ~%100 | **%60** | [6], [7] |
| Fiziksel hasar | Çok değişken (çatlak tipine göre %0–50+) | TL hesabı **yapılmaz** → servis uyarısı | — |
| Elektriksel hasar | Çok değişken (sıcak nokta, diyot arızası; yangın riski) | TL hesabı **yapılmaz** → servis uyarısı | — |

Not: Bu değerler tek bir fotoğrafın "tipik" durumu içindir; gerçek kayıp kaplama oranına, panel bağlantısına,
ışınıma ve sıcaklığa bağlıdır. Uygulamada kullanıcı tarafından değiştirilebilir olacak ve
"gösterim amaçlıdır" notuyla sunulacak.

## 2. Günlük kirlenme oranı r (temizlik planlayıcısı)

| Kaynak | Değer |
|---|---|
| Kimber vd. 2006 (ABD güneybatı, kurak dönem) | %0,1–0,3 / gün; çöl %0,3 / gün [8] |
| pvlib varsayılanı (Kimber modeli) | %0,15 / gün [9] |
| Katar QEERI (ölçüm, sabit eğim) | %0,49 / gün [10] |
| Sandia/Luminate 2025, ölçüm yoksa muhafazakâr | %0,04 / gün [11] |

Gaziantep yarı kurak; Haziran–Eylül neredeyse yağışsız, kışın yağmur paneli doğal olarak temizler.
Öneri: **kurak aylarda (Haz–Eyl) r = %0,20/gün, geçiş aylarında (Nis–May, Eki–Kas) %0,10/gün,
yağışlı aylarda (Ara–Mar) %0,05/gün.** Bu bir VARSAYIMDIR (Gaziantep'e özgü ölçüm bulunamadı); arayüzde ayarlanabilir.

## 3. En uygun temizlik aralığı

Doğrusal kirlenme varsayımıyla (kayıp oranı r·t), T günde bir temizlikte günlük ortalama maliyet:

    maliyet(T) = C / T  +  r · E · p · T / 2

Türev sıfıra eşitlenince:  **T\* = √( 2C / (r · E · p) )**

- C: bir temizliğin maliyeti (TL), E: sistemin günlük üretimi (kWh/gün = E_d × kWp),
  p: elektrik fiyatı (TL/kWh), r: günlük kirlenme oranı (1/gün)
- Aynı mantık (temizlik maliyeti ↔ kaybedilen gelir dengesi) Jones vd. 2016'da genel ifade olarak türetilmiştir [12].

## 4. Ekonomik parametreler (varsayılanlar, arayüzde değiştirilebilir)

- Elektrik fiyatı p: Nisan 2026 tarife artışı sonrası mesken için 100 kWh ≈ 323,8 TL → **~3,24 TL/kWh** (vergiler dahil, haber kaynağı) [13]
- Temizlik ücreti C: Türkiye için güvenilir bir fiyat listesi bulunamadı → **varsayım**, kullanıcı girer.

## Kaynaklar

1. Uykan, Çelik, Birgül (2025). Quantifying the impact of soiling and thermal stress on rooftop PV performance: seasonal analysis from an industrial urban region in Türkiye. *Sustainability* 17(17):8038. https://doi.org/10.3390/su17178038
2. Güneş pili veriminin farklı koşullar altında incelenmesi (İnönü Üniv.), Şanlıurfa simülasyonu, %18 kirlenme kaybı. https://abakus.inonu.edu.tr/items/e513541b-d56d-4d52-96c3-b9d2d0fc3f0b (doğrulanmadı)
3. Sarver, Al-Qaraghuli, Kazmerski (2013). A comprehensive review of the impact of dust on the use of solar energy. *Renewable and Sustainable Energy Reviews* 22:698–733. (doğrulanmadı)
4. Sukkur IBA Journal of Emerging Technologies (2024): kuş pisliği, panelin ~%30'u kaplı, 25 gün → %35 güç kaybı. https://doaj.org/article/71c698b467fa4f199b38b54de3ce5255 (doğrulanmadı)
5. Shaik vd. (2023). *Case Studies in Thermal Engineering* (toz ve kuş pisliği deneyi). https://doaj.org/article/76b52fb683934b5cb2cf43f53e5e2d95 (doğrulanmadı)
6. Marion, Schaefer, Caine, Sanchez (2013). Measured and modeled photovoltaic system energy losses from snow for Colorado and Wisconsin locations. *Solar Energy* 97:112–121.
7. Øgaard vd. (2026). *EPJ Photovoltaics* 17:9 — en ağır durumlarda aylık %100, yıllık %30 kar kaybı. https://doi.org/10.1051/epjpv/2026004
8. Kimber, Mitchell, Nogradi, Wenger (2006). The effect of soiling on large grid-connected PV systems in California and the Southwest region of the United States. IEEE WCPEC-4. (ikincil kaynaklardan: https://pvpmc.sandia.gov/?p=5702)
9. pvlib-python, `pvlib.soiling.kimber`. https://pvlib-python.readthedocs.io/en/v0.9.3/reference/generated/pvlib.soiling.kimber.html
10. PV Tech, Challenges of PV soiling in desert climates (QEERI, Katar). https://www.pv-tech.org/challenges-of-pv-soiling-in-desert-climates
11. Sandia PVPMC 2025, Luminate soiling sunumu. https://www.sandia.gov/app/uploads/sites/243/dlm_uploads/2025/05/16_Luminate_PVPMC2025_soiling.pdf
12. Jones vd. (2016). Optimized cleaning cost and schedule based on observed soiling conditions for photovoltaic plants in Central Saudi Arabia. *IEEE Journal of Photovoltaics* 6(3):730–738.
13. Milliyet Uzmanpara, Elektrik zammı 2026 Nisan. https://uzmanpara.milliyet.com.tr/uzmanpara/elektrik-zammi-2026-nisanda-ne-kadar-oldu-elektrige-ne-kadar-zam-geldi-iste-yeni-tarife-7565631
14. PVGIS 5.3, JRC Avrupa Komisyonu (Gaziantep aylık üretim, SARAH3 2005–2023). https://re.jrc.ec.europa.eu/pvg_tools/
