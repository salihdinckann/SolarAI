"""SolarAI — Streamlit demo uygulaması.

Çalıştırma (proje kökünden):
    streamlit run app/app.py

Sekmeler:
  1. Fotoğraf Analizi    : sınıf olasılıkları, Grad-CAM, kWh/TL kaybı, servis uyarısı
  2. Toz Simülatörü      : temiz fotoğrafa sentetik toz; tahmin ve kayıp canlı değişir
  3. Temizlik Planlayıcı : Gaziantep için aylık T*, yıllık takvim ve kazanç
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import loss_engine as le  # noqa: E402
from common import CLASS_TR, CLASSES  # noqa: E402
from dust import add_dust  # noqa: E402
from inference import Predictor, overlay  # noqa: E402

CKPT_CANDIDATES = [ROOT / "models" / "solarai_effnet_b0_v2.pt", ROOT / "outputs" / "effnet_b0_v2" / "best.pt"]
EXAMPLES_DIR = ROOT / "app" / "examples"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

ORANGE, AMBER, NAVY, SLATE = "#F59E0B", "#FBBF24", "#1E3A8A", "#64748B"
CLASS_COLOR = {"clean": "#10B981", "dusty": "#D97706", "bird_drop": "#8B5CF6", "snow_covered": "#38BDF8",
               "physical_damage": "#EF4444", "electrical_damage": "#B91C1C"}

st.set_page_config(page_title="SolarAI", page_icon="☀️", layout="wide", initial_sidebar_state="expanded")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: 'Inter', system-ui, sans-serif !important;
}}
#MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1250px; }}

.hero {{
  background: linear-gradient(120deg, {NAVY} 0%, #1E40AF 55%, #2563EB 100%);
  border-radius: 18px; padding: 26px 30px; color: #fff; margin-bottom: 18px;
  position: relative; overflow: hidden;
}}
.hero:after {{
  content: ""; position: absolute; right: -60px; top: -60px; width: 240px; height: 240px;
  border-radius: 50%; background: radial-gradient(circle, {AMBER} 0%, {ORANGE} 45%, rgba(245,158,11,0) 70%);
  opacity: .85;
}}
.hero .logo {{ font-size: 2.6rem; font-weight: 800; letter-spacing: -1px; line-height: 1; }}
.hero .logo .solar {{ color: {AMBER}; }}
.hero .logo .ai {{ color: #fff; }}
.hero .title {{ font-size: 1.02rem; font-weight: 500; opacity: .92; margin-top: 8px; max-width: 760px; }}
.hero .slogan {{ font-size: 1.05rem; font-weight: 600; margin-top: 10px; color: #FDE68A; }}
.hero .badges {{ margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; position: relative; z-index: 1; }}
.badge {{ background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); border-radius: 999px;
          padding: 4px 12px; font-size: .78rem; font-weight: 500; }}

.card {{ background: #fff; border: 1px solid #E2E8F0; border-radius: 14px; padding: 16px 18px;
         box-shadow: 0 1px 2px rgba(15,23,42,.04); height: 100%; }}
.kpi-label {{ color: {SLATE}; font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
.kpi-value {{ color: #0F172A; font-size: 1.75rem; font-weight: 800; margin-top: 4px; line-height: 1.1; }}
.kpi-sub {{ color: {SLATE}; font-size: .85rem; margin-top: 4px; }}
.kpi-accent {{ border-top: 4px solid {ORANGE}; }}
.kpi-navy {{ border-top: 4px solid {NAVY}; }}
.kpi-green {{ border-top: 4px solid #10B981; }}

.pred-box {{ border-radius: 14px; padding: 16px 18px; color: #fff; margin-bottom: 10px; }}
.pred-box .lbl {{ font-size: .78rem; opacity: .9; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }}
.pred-box .cls {{ font-size: 1.6rem; font-weight: 800; }}
.pred-box .conf {{ font-size: .95rem; opacity: .95; }}

.alert {{ border-radius: 12px; padding: 14px 16px; margin: 6px 0 14px 0; font-size: .95rem; }}
.alert-danger {{ background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; }}
.alert-warn {{ background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; }}
.alert-info {{ background: #EFF6FF; border: 1px solid #BFDBFE; color: #1E3A8A; }}

.section-title {{ font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 18px 0 8px 0; }}
.muted {{ color: {SLATE}; font-size: .85rem; }}
.img-cap {{ color: {SLATE}; font-size: .8rem; text-align: center; margin-top: -6px; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #E2E8F0; }}
.stTabs [data-baseweb="tab"] {{ height: 46px; padding: 0 18px; border-radius: 10px 10px 0 0;
                                font-weight: 600; color: {SLATE}; }}
.stTabs [aria-selected="true"] {{ color: {NAVY} !important; background: #F8FAFC; }}
[data-testid="stSidebar"] {{ background: #F8FAFC; border-right: 1px solid #E2E8F0; }}
[data-testid="stSidebar"] .side-logo {{ font-size: 1.5rem; font-weight: 800; }}
.footer {{ margin-top: 36px; padding-top: 14px; border-top: 1px solid #E2E8F0; color: {SLATE}; font-size: .8rem; }}
</style>
"""


# ------------------------------------------------------------------ yardımcılar
@st.cache_resource(show_spinner="Model yükleniyor…")
def get_predictor() -> Predictor:
    for path in CKPT_CANDIDATES:
        if path.exists():
            return Predictor(path)
    st.error("Model dosyası bulunamadı: " + " veya ".join(str(p.relative_to(ROOT)) for p in CKPT_CANDIDATES))
    st.stop()


@st.cache_data
def get_pvgis() -> pd.DataFrame:
    return le.load_pvgis()


def to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def kpi(label: str, value: str, sub: str = "", tone: str = "accent") -> str:
    return (f"<div class='card kpi-{tone}'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>")


def alert(text: str, kind: str = "info") -> None:
    st.markdown(f"<div class='alert alert-{kind}'>{text}</div>", unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def fmt_tl(x: float) -> str:
    return f"{x:,.0f} TL".replace(",", ".")


def list_examples() -> list[Path]:
    return sorted(p for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in IMG_EXTS) if EXAMPLES_DIR.exists() else []


def prob_chart(probs: dict[str, float]) -> alt.Chart:
    df = pd.DataFrame({"Sınıf": [CLASS_TR[c] for c in CLASSES], "Olasılık": [probs[c] * 100 for c in CLASSES],
                       "Renk": [CLASS_COLOR[c] for c in CLASSES], "sıra": range(len(CLASSES))})
    base = alt.Chart(df).encode(y=alt.Y("Sınıf:N", sort=alt.SortField("sıra"), title=None,
                                        axis=alt.Axis(labelFontSize=12, labelFont="Inter")))
    bars = base.mark_bar(cornerRadiusEnd=6, height=18).encode(
        x=alt.X("Olasılık:Q", scale=alt.Scale(domain=[0, 100]), title="Olasılık (%)",
                axis=alt.Axis(grid=False, labelFont="Inter")),
        color=alt.Color("Renk:N", scale=None), tooltip=["Sınıf", alt.Tooltip("Olasılık:Q", format=".1f")])
    text = base.mark_text(align="left", dx=4, font="Inter", fontSize=11, color="#334155").encode(
        x="Olasılık:Q", text=alt.Text("Olasılık:Q", format=".0f"))
    return (bars + text).properties(height=230).configure_view(strokeWidth=0)


def prediction_box(probs: dict[str, float]) -> None:
    top = max(probs, key=probs.get)
    st.markdown(f"<div class='pred-box' style='background:{CLASS_COLOR[top]}'>"
                f"<div class='lbl'>Model tahmini</div><div class='cls'>{CLASS_TR[top]}</div>"
                f"<div class='conf'>güven %{probs[top] * 100:.0f}</div></div>", unsafe_allow_html=True)


def loss_cards(loss: le.PhotoLoss, cfg: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Anlık üretim kaybı", f"%{loss.loss_fraction * 100:.1f}", "kirlenme kaynaklı"), unsafe_allow_html=True)
    c2.markdown(kpi("Günlük kayıp", f"{loss.daily_tl:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", "."),
                    f"{loss.daily_kwh:.2f} kWh · {le.month_name_tr(cfg['month'])}", "navy"), unsafe_allow_html=True)
    c3.markdown(kpi("Aylık kayıp", fmt_tl(loss.monthly_tl), f"{loss.monthly_kwh:.0f} kWh · {cfg['kwp']:g} kWp", "navy"),
                unsafe_allow_html=True)


# ------------------------------------------------------------------ düzen
def hero() -> None:
    st.markdown(
        "<div class='hero'><div class='logo'><span class='solar'>Solar</span><span class='ai'>AI</span></div>"
        "<div class='title'>Güneş Panellerinde Kirlenmenin Yapay Zekâ ile Tespiti ve Ekonomik Temizlik Planlaması</div>"
        "<div class='slogan'>Panelin ne kadar kirli olduğunu değil, ne kadar kaybettirdiğini söyler.</div>"
        "<div class='badges'><span class='badge'>EfficientNet-B0 · macro-F1 0.888</span>"
        "<span class='badge'>Grad-CAM açıklanabilirlik</span><span class='badge'>PVGIS · Gaziantep</span>"
        "<span class='badge'>HASAT-13 Proje Pazarı</span></div></div>",
        unsafe_allow_html=True,
    )


def sidebar() -> dict:
    sb = st.sidebar
    sb.markdown(f"<div class='side-logo'><span style='color:{ORANGE}'>Solar</span>"
                f"<span style='color:{NAVY}'>AI</span></div><div class='muted'>Sistem ayarları</div>",
                unsafe_allow_html=True)
    sb.write("")
    kwp = sb.number_input("Sistem gücü (kWp)", 0.5, 1000.0, le.DEFAULT_KWP, 0.5,
                          help="Ev çatısı tipik 3–10 kWp, işletme 50–500 kWp")
    price = sb.number_input("Elektrik fiyatı (TL/kWh)", 0.5, 20.0, le.DEFAULT_PRICE_TL_PER_KWH, 0.1,
                            help="Varsayılan: mesken, Nisan 2026 sonrası, vergiler dahil (yaklaşık)")
    month = sb.selectbox("Ay", range(1, 13), index=6, format_func=le.month_name_tr,
                         help="Kayıp, o ayın Gaziantep üretimine göre hesaplanır")
    with sb.expander("Sınıf başına kayıp oranları"):
        rates = {c: st.slider(CLASS_TR[c], 0, 100, int(le.LOSS_RATE[c] * 100), format="%d%%", key=f"rate_{c}") / 100
                 for c in le.LOSS_RATE}
        st.caption("Varsayılanlar literatürden (docs/literatur_notlari.md). Hasar sınıfları TL hesabına girmez.")
    sb.markdown("<div class='alert alert-info' style='font-size:.82rem'>📍 Konum: <b>Gaziantep</b><br>"
                "PVGIS 5.3 · 2005–2023 ortalaması<br>Yıllık üretim: <b>1580 kWh/kWp</b></div>", unsafe_allow_html=True)
    return {"kwp": kwp, "price": price, "month": month, "rates": rates}


def image_picker(key: str, only_clean: bool = False) -> tuple[Image.Image | None, str]:
    examples = [p for p in list_examples() if not only_clean or "temiz" in p.stem]
    options = (["Örnek fotoğraf", "Fotoğraf yükle"] if examples else ["Fotoğraf yükle"])
    source = st.radio("Kaynak", options, horizontal=True, key=f"{key}_source", label_visibility="collapsed")
    if source == "Örnek fotoğraf":
        name = st.selectbox("Örnek", [p.name for p in examples], key=f"{key}_example",
                            format_func=lambda n: Path(n).stem.split("_", 1)[-1].replace("_", " ").capitalize())
        return Image.open(EXAMPLES_DIR / name).convert("RGB"), name
    file = st.file_uploader("Panel fotoğrafı (JPG/PNG)", type=["jpg", "jpeg", "png", "webp"], key=f"{key}_upload")
    return (Image.open(file).convert("RGB"), file.name) if file else (None, "")


# ------------------------------------------------------------------ 1) fotoğraf analizi
def tab_photo(cfg: dict) -> None:
    img, name = image_picker("photo")
    if img is None:
        alert("📷 Bir güneş paneli fotoğrafı yükleyin. En iyi sonuç: panele <b>yakından</b> ve "
              "<b>düz açıyla</b> çekilmiş fotoğraf.", "info")
        return

    pred = get_predictor()
    probs = pred.predict(img)
    top = max(probs, key=probs.get)
    view = pred.preprocess_view(img)
    heat = pred.gradcam(img, top)
    loss = le.photo_loss(probs, cfg["month"], cfg["kwp"], cfg["price"], cfg["rates"], get_pvgis())

    if loss.service_alert:
        alert(f"⚠️ <b>Teknik servis çağırın.</b> Hasar olasılığı %{loss.damage_prob * 100:.0f}. "
              "Fiziksel/elektriksel hasarın üretim kaybı ve güvenlik riski fotoğraftan güvenilir biçimde "
              "tahmin edilemez; panel bir uzman tarafından kontrol edilmelidir.", "danger")

    c1, c2, c3 = st.columns([1, 1, 1.15], gap="medium")
    with c1:
        st.image(view, width="stretch")
        st.markdown("<div class='img-cap'>Yüklenen fotoğraf</div>", unsafe_allow_html=True)
    with c2:
        st.image(overlay(view, heat), width="stretch")
        st.markdown(f"<div class='img-cap'>Grad-CAM · model “{CLASS_TR[top]}” derken buraya baktı</div>",
                    unsafe_allow_html=True)
    with c3:
        prediction_box(probs)
        st.altair_chart(prob_chart(probs), width="stretch")

    section(f"Tahmini kayıp · {le.month_name_tr(cfg['month'])}, {cfg['kwp']:g} kWp")
    if loss.service_alert:
        alert("Aşağıdaki rakamlar yalnızca <b>kirlenme</b> kaynaklı kaybı gösterir. Hasarın yol açtığı kayıp "
              "bu hesaba <b>dahil değildir</b> ve daha büyük olabilir.", "warn")
    loss_cards(loss, cfg)

    with st.expander("🧮 Hesap nasıl yapıldı?"):
        rows = [{"Sınıf": CLASS_TR[c], "Olasılık": f"%{probs[c] * 100:.1f}",
                 "Kayıp oranı": f"%{cfg['rates'][c] * 100:.0f}", "Katkı": f"%{loss.contributions[c] * 100:.2f}"}
                for c in le.LOSS_RATE]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        e_d = get_pvgis().set_index("month").loc[cfg["month"], "E_d"]
        st.markdown(
            f"**kayıp = Σ p(sınıf) · kayıp(sınıf) = %{loss.loss_fraction * 100:.2f}**  \n"
            f"günlük kayıp = %{loss.loss_fraction * 100:.2f} × {e_d:.2f} kWh/kWp (PVGIS) × {cfg['kwp']:g} kWp "
            f"= {loss.daily_kwh:.2f} kWh × {cfg['price']:.2f} TL/kWh = **{loss.daily_tl:.2f} TL**  \n"
            f"Hasar sınıflarının olasılığı kayba eklenmez; toplam hasar olasılığı ≥ "
            f"%{le.DAMAGE_ALERT_THRESHOLD * 100:.0f} ise servis uyarısı verilir.")
    st.caption("Kayıp rakamları gösterim amaçlıdır. Model yakın çekim panel fotoğraflarıyla eğitildi; "
               "geniş açılı saha fotoğraflarında (ör. fırtına hasarı) daha az güvenilirdir.")


# ------------------------------------------------------------------ 2) toz simülatörü
@st.cache_data(show_spinner=False)
def dust_curve(img_bytes: bytes, seed: int, month: int, kwp: float, price: float,
               rates: tuple) -> pd.DataFrame:
    pred, img, rows = get_predictor(), from_bytes(img_bytes), []
    for level in np.linspace(0, 1, 11):
        probs = pred.predict(add_dust(img, float(level), seed))
        loss = le.photo_loss(probs, month, kwp, price, dict(rates), get_pvgis())
        rows.append({"Toz (%)": round(level * 100), "Günlük kayıp (TL)": loss.daily_tl,
                     "p(tozlu) (%)": probs["dusty"] * 100})
    return pd.DataFrame(rows)


def tab_dust(cfg: dict) -> None:
    alert("🌫️ Temiz bir panel fotoğrafına <b>sentetik toz</b> ekleyin. Kaydırdıkça modelin tahmini, "
          "Grad-CAM ve kayıp canlı olarak güncellenir.", "info")
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        img, _ = image_picker("dust", only_clean=True)
        if img is None:
            return
        level = st.slider("Toz yoğunluğu", 0, 100, 40, 5, format="%d%%", key="dust_level")
        if "dust_seed" not in st.session_state:
            st.session_state.dust_seed = 1
        if st.button("🎲 Yeni toz deseni", width="stretch"):
            st.session_state.dust_seed += 1
        seed = st.session_state.dust_seed

    pred = get_predictor()
    dusty = add_dust(img, level / 100, seed)
    probs = pred.predict(dusty)
    top = max(probs, key=probs.get)
    view = pred.preprocess_view(dusty)
    loss = le.photo_loss(probs, cfg["month"], cfg["kwp"], cfg["price"], cfg["rates"], get_pvgis())

    with right:
        a, b, c = st.columns([1, 1, 1.1], gap="medium")
        a.image(view, width="stretch")
        a.markdown(f"<div class='img-cap'>Toz %{level}</div>", unsafe_allow_html=True)
        b.image(overlay(view, pred.gradcam(dusty, top)), width="stretch")
        b.markdown("<div class='img-cap'>Grad-CAM</div>", unsafe_allow_html=True)
        with c:
            prediction_box(probs)
            st.markdown(kpi("Günlük kayıp", f"{loss.daily_tl:.2f} TL",
                            f"%{loss.loss_fraction * 100:.1f} · {loss.daily_kwh:.2f} kWh", "navy"),
                        unsafe_allow_html=True)

    section("Toz arttıkça kayıp nasıl değişiyor?")
    curve = dust_curve(to_bytes(img), seed, cfg["month"], cfg["kwp"], cfg["price"],
                       tuple(sorted(cfg["rates"].items())))
    now = pd.DataFrame({"Toz (%)": [level], "Günlük kayıp (TL)": [loss.daily_tl]})
    line = alt.Chart(curve).mark_line(color=ORANGE, strokeWidth=3, point=alt.OverlayMarkDef(color=ORANGE, size=40)).encode(
        x=alt.X("Toz (%):Q", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Günlük kayıp (TL):Q"),
        tooltip=[alt.Tooltip("Toz (%):Q"), alt.Tooltip("Günlük kayıp (TL):Q", format=".2f"),
                 alt.Tooltip("p(tozlu) (%):Q", format=".0f")])
    dot = alt.Chart(now).mark_point(color=NAVY, size=220, filled=True).encode(x="Toz (%):Q", y="Günlük kayıp (TL):Q")
    rule = alt.Chart(now).mark_rule(color=NAVY, strokeDash=[4, 4]).encode(x="Toz (%):Q")
    st.altair_chart((line + rule + dot).properties(height=280).configure_axis(labelFont="Inter", titleFont="Inter")
                    .configure_view(strokeWidth=0), width="stretch")
    st.caption("Kayıp rakamları gösterim amaçlıdır. Sentetik toz gerçek toz fiziğinin bir modeli değildir; "
               "amacı modelin kirliliğe tepkisini görselleştirmektir. Eğri 11 toz seviyesinde modelin "
               "tahminiyle hesaplanır.")


# ------------------------------------------------------------------ 3) temizlik planlayıcısı
@st.cache_data(show_spinner="Takvim hesaplanıyor…")
def get_plan(kwp: float, cost: float, price: float, rates: tuple) -> dict:
    return le.cleaning_plan(kwp=kwp, cost=cost, price=price, rates=dict(rates))


def tab_plan(cfg: dict) -> None:
    alert("📅 Gaziantep'in aylık güneş üretimine göre panelin <b>ne zaman</b> temizlenmesinin ekonomik olduğunu "
          "hesaplar. Temizlik ücreti ↔ kirin kaybettirdiği elektrik dengesi.", "info")
    c1, c2 = st.columns([1, 2.2], gap="large")
    with c1:
        cost = st.number_input("Bir temizliğin maliyeti (TL)", 50.0, 500000.0, le.DEFAULT_CLEANING_COST_TL, 100.0,
                               help="VARSAYIM: Türkiye için güvenilir bir fiyat listesi bulunamadı; kendi teklifinizi girin.")
        st.markdown(f"<div class='muted'>Sistem: <b>{cfg['kwp']:g} kWp</b> · Elektrik: <b>{cfg['price']:.2f} TL/kWh</b> "
                    "(sol menüden değiştirilebilir)</div>", unsafe_allow_html=True)
        st.write("")
        with st.expander("Günlük kirlenme oranı r (%/gün)"):
            r_df = pd.DataFrame({"Ay": [le.month_name_tr(m) for m in range(1, 13)],
                                 "r (%/gün)": [le.MONTHLY_SOILING_RATE[m] * 100 for m in range(1, 13)]})
            edited = st.data_editor(r_df, hide_index=True, disabled=["Ay"], width="stretch", key="r_editor",
                                    column_config={"r (%/gün)": st.column_config.NumberColumn(min_value=0.001,
                                                                                               max_value=2.0,
                                                                                               step=0.01, format="%.2f")})
            st.caption("Varsayım: kurak yaz %0,20 · geçiş %0,10 · yağışlı kış %0,05 (Kimber vd. 2006 aralığı içinde).")
        rates = {m: float(v) / 100 for m, v in zip(range(1, 13), edited["r (%/gün)"])}

    plan = get_plan(cfg["kwp"], cost, cfg["price"], tuple(sorted(rates.items())))
    comp = plan["comparison"]
    opt = comp.iloc[0]

    with c2:
        k1, k2, k3 = st.columns(3)
        days = ", ".join(f"{d.day} {le.month_name_tr(d.month)}" for d in plan["cleaning_days"]) or "Gerekmiyor"
        k1.markdown(kpi("Önerilen temizlik", f"{len(plan['cleaning_days'])} kez/yıl", days), unsafe_allow_html=True)
        k2.markdown(kpi("Ayda bir temizliğe göre", fmt_tl(plan["saving_vs_monthly_tl"]), "yıllık tasarruf", "green"),
                    unsafe_allow_html=True)
        k3.markdown(kpi("Hiç temizlememeye göre", fmt_tl(plan["saving_vs_never_tl"]), "yıllık tasarruf", "green"),
                    unsafe_allow_html=True)

        section("Yıllık takvim · günlük kirlenme kaybı")
        d = plan["daily"].assign(Ay=lambda x: x["month"].map(le.month_name_tr), Kayıp=lambda x: x["soiling"] * 100)
        months = [le.month_name_tr(m) for m in range(1, 13)]
        heat = alt.Chart(d).mark_rect(cornerRadius=2).encode(
            x=alt.X("day:O", title="Gün", axis=alt.Axis(labelAngle=0, values=[1, 5, 10, 15, 20, 25, 30])),
            y=alt.Y("Ay:N", sort=months, title=None),
            color=alt.Color("Kayıp:Q", title="Kayıp (%)", scale=alt.Scale(scheme="orangered")),
            tooltip=[alt.Tooltip("date:T", title="Tarih", format="%d.%m"), alt.Tooltip("Kayıp:Q", format=".1f")])
        marks = alt.Chart(d[d["cleaning"]]).mark_text(text="🧽", fontSize=15).encode(
            x="day:O", y=alt.Y("Ay:N", sort=months))
        st.altair_chart((heat + marks).properties(height=330).configure_axis(labelFont="Inter", titleFont="Inter")
                        .configure_view(strokeWidth=0), width="stretch")
        st.caption("🧽 = önerilen temizlik günü. Renk, o gün panelde birikmiş kirin yol açtığı üretim kaybıdır.")

    section("Stratejilerin karşılaştırması (1 yıl)")
    show = comp.rename(columns={"strateji": "Strateji", "temizlik_sayisi": "Temizlik", "kayip_kwh": "Kayıp (kWh)",
                                "kayip_tl": "Kayıp (TL)", "temizlik_tl": "Temizlik (TL)", "toplam_tl": "Toplam (TL)"})
    bars = alt.Chart(show).mark_bar(cornerRadiusEnd=6, height=26).encode(
        y=alt.Y("Strateji:N", sort=None, title=None, axis=alt.Axis(labelLimit=220, labelFontSize=12)),
        x=alt.X("Toplam (TL):Q", title="Yıllık toplam maliyet (TL)"),
        color=alt.condition(alt.datum.Strateji == opt["strateji"], alt.value(ORANGE), alt.value("#CBD5E1")),
        tooltip=list(show.columns))
    lc, rc = st.columns([1.3, 1], gap="large")
    lc.altair_chart(bars.properties(height=200).configure_axis(labelFont="Inter", titleFont="Inter")
                    .configure_view(strokeWidth=0), width="stretch")
    rc.dataframe(show, hide_index=True, width="stretch")

    with st.expander("🧮 Formül: en uygun temizlik aralığı T*"):
        iv = plan["intervals"].assign(Ay=lambda x: x["month"].map(le.month_name_tr))
        st.markdown(
            "T günde bir temizlikte günlük ortalama maliyet: **f(T) = C/T + r·E·p·T/2**  \n"
            "İlk terim (temizlik ücreti) T büyüdükçe azalır, ikincisi (kirin kaybettirdiği elektrik) artar. "
            "Türevi sıfıra eşitleyince: **T\\* = √(2C / (r·E·p))**  \n"
            "Aylar arasında r ve E değiştiği, kir de %30'da doyduğu için yıllık takvim **dinamik programlamayla** "
            "kesin optimum olarak hesaplanır; aşağıdaki aylık T* değerleri sezgi içindir.")
        tchart = alt.Chart(iv).mark_bar(color=NAVY, cornerRadiusEnd=4).encode(
            x=alt.X("Ay:N", sort=months, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("T_star_days:Q", title="T* (gün)"),
            tooltip=[alt.Tooltip("T_star_days:Q", title="T* (gün)", format=".0f"),
                     alt.Tooltip("E_kwh_per_day:Q", title="E (kWh/gün)", format=".1f")])
        st.altair_chart(tchart.properties(height=220).configure_view(strokeWidth=0), width="stretch")
    st.caption("Model varsayımları: kir doğrusal birikir ve %30'da doyar; panel 1 Ocak'ta temiz başlar; "
               "r ve temizlik ücreti varsayımdır. Üretim: PVGIS 5.3 Gaziantep (optimum eğim 31°).")


# ------------------------------------------------------------------ ana
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    hero()
    cfg = sidebar()
    t1, t2, t3 = st.tabs(["📷  Fotoğraf Analizi", "🌫️  Toz Simülatörü", "📅  Temizlik Planlayıcısı"])
    with t1:
        tab_photo(cfg)
    with t2:
        tab_dust(cfg)
    with t3:
        tab_plan(cfg)
    st.markdown("<div class='footer'>SolarAI · HASAT-13 Proje Pazarı · Hasan Kalyoncu Üniversitesi, Gaziantep · "
                "Veri: Kaggle (Afroz) · Üretim: PVGIS 5.3 (JRC) · "
                "<a href='https://github.com/salihdinckann/SolarAI' target='_blank'>GitHub</a></div>",
                unsafe_allow_html=True)


main()
