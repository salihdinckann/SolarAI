"""SolarAI ekonomik karar katmanı: kayıp hesabı ve temizlik planlaması.

İki iş yapar:
  1. photo_loss(): Modelin sınıf olasılıklarından beklenen üretim kaybını (kWh, TL) hesaplar.
         kayıp_oranı = Σ p(sınıf) · kayıp(sınıf)     (hasar sınıfları hariç → servis uyarısı)
  2. cleaning_plan(): Gaziantep'in aylık PVGIS üretimiyle, her ay için en uygun temizlik aralığını
         T* = √( 2C / (r · E · p) )
     hesaplar (sezgi/yorum için). Yıllık takvim ise dinamik programlamayla KESİN optimumdan gelir:
     aylar arasında r ve E değiştiği ve kir doyduğu için "her ay T*" kuralı yalnızca yaklaşıktır.

Varsayılan değerlerin dayanağı: docs/literatur_notlari.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import ROOT

# ----------------------------------------------------------------- varsayılanlar
# Bir fotoğraftaki tipik anlık üretim kaybı (oran). Hasar sınıfları TL hesabına girmez.
LOSS_RATE = {
    "clean": 0.00,
    "dusty": 0.15,
    "bird_drop": 0.25,
    "snow_covered": 0.60,
}
DAMAGE_CLASSES = ("physical_damage", "electrical_damage")
# Hasar olasılıkları toplamı bu eşiği geçerse, en olası sınıf "temiz" olsa bile servis uyarısı verilir
DAMAGE_ALERT_THRESHOLD = 0.20

# Günlük kirlenme oranı r (1/gün), Gaziantep iklimine göre aylık VARSAYIM
MONTHLY_SOILING_RATE = {
    1: 0.0005, 2: 0.0005, 3: 0.0005,          # yağışlı: yağmur doğal temizlik yapar
    4: 0.0010, 5: 0.0010,                      # geçiş
    6: 0.0020, 7: 0.0020, 8: 0.0020, 9: 0.0020,  # kurak yaz
    10: 0.0010, 11: 0.0010,                    # geçiş
    12: 0.0005,
}
MAX_SOILING_LOSS = 0.30   # doğrusal birikim sınırsız büyümesin diye üst sınır (doygunluk)

DEFAULT_PRICE_TL_PER_KWH = 3.24   # mesken, Nisan 2026 sonrası, vergiler dahil (yaklaşık)
DEFAULT_CLEANING_COST_TL = 1500.0  # VARSAYIM: Türkiye için güvenilir fiyat bulunamadı
DEFAULT_KWP = 5.0


def load_pvgis() -> pd.DataFrame:
    """Gaziantep aylık üretimi: month, E_d (kWh/kWp/gün), E_m (kWh/kWp/ay)."""
    df = pd.read_csv(ROOT / "data" / "pvgis_gaziantep.csv")
    return df.rename(columns={"E_d_kwh_per_kwp": "E_d", "E_m_kwh_per_kwp": "E_m"})[["month", "E_d", "E_m"]]


# ----------------------------------------------------------------- 1) fotoğraf kaybı
@dataclass
class PhotoLoss:
    loss_fraction: float          # beklenen anlık üretim kaybı (0-1)
    daily_kwh: float
    daily_tl: float
    monthly_kwh: float
    monthly_tl: float
    damage_prob: float            # hasar sınıflarının toplam olasılığı
    service_alert: bool           # "Teknik servis çağırın"
    contributions: dict = field(default_factory=dict)  # sınıf → kayba katkısı (oran)


def photo_loss(probs: dict[str, float], month: int, kwp: float = DEFAULT_KWP,
               price: float = DEFAULT_PRICE_TL_PER_KWH, loss_rate: dict[str, float] | None = None,
               pvgis: pd.DataFrame | None = None) -> PhotoLoss:
    """Model olasılıklarından beklenen kaybı hesaplar.

    Kirlenme sınıflarında: kayıp = Σ p(sınıf) · kayıp(sınıf).
    Hasar sınıflarının olasılık payı kayba eklenmez (kaybı belirsiz); bunun yerine
    toplam hasar olasılığı eşiği aşarsa servis uyarısı verilir.
    """
    loss_rate = loss_rate or LOSS_RATE
    pvgis = pvgis if pvgis is not None else load_pvgis()
    row = pvgis.loc[pvgis["month"] == month].iloc[0]

    contributions = {c: probs.get(c, 0.0) * loss_rate[c] for c in loss_rate}
    frac = sum(contributions.values())
    damage_prob = sum(probs.get(c, 0.0) for c in DAMAGE_CLASSES)
    top = max(probs, key=probs.get)

    daily_kwh = frac * row["E_d"] * kwp
    monthly_kwh = frac * row["E_m"] * kwp
    return PhotoLoss(
        loss_fraction=frac,
        daily_kwh=daily_kwh, daily_tl=daily_kwh * price,
        monthly_kwh=monthly_kwh, monthly_tl=monthly_kwh * price,
        damage_prob=damage_prob,
        service_alert=(top in DAMAGE_CLASSES) or damage_prob >= DAMAGE_ALERT_THRESHOLD,
        contributions=contributions,
    )


# ----------------------------------------------------------------- 2) temizlik planı
def optimal_interval(cost: float, r: float, energy_per_day: float, price: float) -> float:
    """T* = √(2C / (r·E·p)) gün.

    Türetme: T günde bir temizlikte günlük ortalama maliyet
        f(T) = C/T + r·E·p·T/2   →   f'(T) = 0   →   T* = √(2C/(r·E·p))
    """
    if min(cost, r, energy_per_day, price) <= 0:
        raise ValueError("C, r, E ve p pozitif olmalı")
    return math.sqrt(2 * cost / (r * energy_per_day * price))


def monthly_intervals(kwp: float = DEFAULT_KWP, cost: float = DEFAULT_CLEANING_COST_TL,
                      price: float = DEFAULT_PRICE_TL_PER_KWH, rates: dict[int, float] | None = None,
                      pvgis: pd.DataFrame | None = None) -> pd.DataFrame:
    """Her ay için T* tablosu."""
    rates = rates or MONTHLY_SOILING_RATE
    pvgis = pvgis if pvgis is not None else load_pvgis()
    rows = []
    for _, m in pvgis.iterrows():
        month = int(m["month"])
        E = m["E_d"] * kwp
        rows.append({"month": month, "E_kwh_per_day": E, "r_per_day": rates[month],
                     "T_star_days": optimal_interval(cost, rates[month], E, price)})
    return pd.DataFrame(rows)


@dataclass
class SimResult:
    name: str
    cleaning_days: list[pd.Timestamp]
    lost_kwh: float
    lost_tl: float
    cleaning_tl: float

    @property
    def total_tl(self) -> float:
        return self.lost_tl + self.cleaning_tl


def _daily_arrays(year: int, kwp: float, rates: dict[int, float], pvgis: pd.DataFrame):
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    E_d = dict(zip(pvgis["month"], pvgis["E_d"]))
    r = np.array([rates[d.month] for d in days])
    E = np.array([E_d[d.month] * kwp for d in days])
    return days, r, E


def _segment_loss_kwh(r: np.ndarray, E: np.ndarray, start: int) -> np.ndarray:
    """start gününde temizlendiyse, start..k günleri boyunca kümülatif kayıp (kWh), k = start..n-1."""
    soil = np.minimum(np.cumsum(r[start:]), MAX_SOILING_LOSS)
    return np.cumsum(soil * E[start:])


def _optimal_cleaning_days(r: np.ndarray, E: np.ndarray, cost: float, price: float) -> list[int]:
    """Dinamik programlama: toplam (temizlik + kayıp) maliyetini en aza indiren temizlik günleri.

    f[a] = a gününde temiz başlayan panel için yıl sonuna kadarki en düşük maliyet
         = min( a'dan yıl sonuna temizlemeden kayıp,
                min_{b>a} [ a..b-1 kaybı + C + f[b] ] )
    Gün 0'da panel temiz kabul edilir (temizlik ücreti yok). Karmaşıklık O(n²), n=365.
    """
    n = len(r)
    f = np.zeros(n + 1)
    nxt = np.full(n, -1)
    for a in range(n - 1, -1, -1):
        seg = _segment_loss_kwh(r, E, a) * price          # seg[j] = a..a+j günlerinin kaybı (TL)
        best, best_b = seg[-1], -1                         # hiç temizlemeden yıl sonuna
        if a + 1 < n:
            b = np.arange(a + 1, n)
            cand = seg[b - a - 1] + cost + f[b]            # a..b-1 kaybı + b'de temizlik + sonrası
            i = int(np.argmin(cand))
            if cand[i] < best:
                best, best_b = cand[i], int(b[i])
        f[a], nxt[a] = best, best_b
    out, a = [], 0
    while nxt[a] != -1:
        a = nxt[a]
        out.append(a)
    return out


def simulate(policy: str, year: int = 2027, kwp: float = DEFAULT_KWP, cost: float = DEFAULT_CLEANING_COST_TL,
             price: float = DEFAULT_PRICE_TL_PER_KWH, rates: dict[int, float] | None = None,
             fixed_days: int = 30, pvgis: pd.DataFrame | None = None) -> SimResult:
    """Yılı gün gün simüle eder. Panel 1 Ocak'ta temiz başlar; kir MAX_SOILING_LOSS'ta doyar.

    policy:
      "optimal" → dinamik programlamayla bulunan en düşük maliyetli temizlik günleri
      "fixed"   → her `fixed_days` günde bir temizle
      "never"   → hiç temizleme
    """
    rates = rates or MONTHLY_SOILING_RATE
    pvgis = pvgis if pvgis is not None else load_pvgis()
    days, r, E = _daily_arrays(year, kwp, rates, pvgis)

    if policy == "optimal":
        idx = _optimal_cleaning_days(r, E, cost, price)
    elif policy == "fixed":
        idx = list(range(fixed_days, len(days), fixed_days))
    elif policy == "never":
        idx = []
    else:
        raise ValueError(policy)

    bounds = [0, *idx, len(days)]
    lost_kwh = sum(_segment_loss_kwh(r, E, a)[b - a - 1] for a, b in zip(bounds[:-1], bounds[1:]))
    name = {"optimal": "SolarAI (en uygun)", "fixed": f"Her {fixed_days} günde bir", "never": "Hiç temizlememe"}[policy]
    return SimResult(name, [days[i] for i in idx], float(lost_kwh), float(lost_kwh) * price, len(idx) * cost)


def cleaning_plan(kwp: float = DEFAULT_KWP, cost: float = DEFAULT_CLEANING_COST_TL,
                  price: float = DEFAULT_PRICE_TL_PER_KWH, year: int = 2027,
                  rates: dict[int, float] | None = None) -> dict:
    """Uygulamanın 3. sekmesi için tek çağrı: aylık T*, takvim ve stratejilerin karşılaştırması."""
    pvgis = load_pvgis()
    kw = dict(kwp=kwp, cost=cost, price=price, rates=rates, pvgis=pvgis)
    opt = simulate("optimal", year, **kw)
    others = [simulate("fixed", year, fixed_days=30, **kw), simulate("fixed", year, fixed_days=90, **kw),
              simulate("never", year, **kw)]
    comparison = pd.DataFrame([{"strateji": s.name, "temizlik_sayisi": len(s.cleaning_days),
                                "kayip_kwh": round(s.lost_kwh), "kayip_tl": round(s.lost_tl),
                                "temizlik_tl": round(s.cleaning_tl), "toplam_tl": round(s.total_tl)}
                               for s in [opt, *others]])
    return {
        "intervals": monthly_intervals(**kw),
        "cleaning_days": opt.cleaning_days,
        "comparison": comparison,
        "saving_vs_monthly_tl": others[0].total_tl - opt.total_tl,
    }


def month_name_tr(m: int) -> str:
    return ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos",
            "Eylül", "Ekim", "Kasım", "Aralık"][m - 1]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print("== Örnek fotoğraf: %70 tozlu, %20 temiz, %10 kuş pisliği — Temmuz, 5 kWp")
    pl = photo_loss({"dusty": 0.7, "clean": 0.2, "bird_drop": 0.1}, month=7)
    print(f"  kayıp oranı %{pl.loss_fraction * 100:.1f} | günlük {pl.daily_kwh:.2f} kWh = {pl.daily_tl:.2f} TL"
          f" | aylık {pl.monthly_kwh:.1f} kWh = {pl.monthly_tl:.0f} TL | servis: {pl.service_alert}")

    for kwp, cost in [(5, 1500), (100, 10000)]:
        plan = cleaning_plan(kwp=kwp, cost=cost)
        print(f"\n== {kwp} kWp, temizlik {cost:.0f} TL, elektrik {DEFAULT_PRICE_TL_PER_KWH} TL/kWh")
        iv = plan["intervals"]
        print("  T* (gün): " + ", ".join(f"{month_name_tr(int(m))[:3]} {t:.0f}" for m, t in zip(iv.month, iv.T_star_days)))
        print("  Temizlik günleri: " + ", ".join(d.strftime("%d %b") for d in plan["cleaning_days"]))
        print(plan["comparison"].to_string(index=False))
        print(f"  Ayda bir temizliğe göre yıllık kazanç: {plan['saving_vs_monthly_tl']:.0f} TL")
