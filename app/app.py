"""SolarAI — Streamlit demo uygulaması.

Çalıştırma (proje kökünden):
    streamlit run app/app.py

Sekmeler:
  1. Fotoğraf Analizi   : sınıf olasılıkları, Grad-CAM, kWh/TL kaybı, servis uyarısı
  2. Toz Simülatörü     : (23–24 Kasım)
  3. Temizlik Planlayıcı: (23–24 Kasım)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import loss_engine as le  # noqa: E402
from common import CLASS_TR, CLASSES  # noqa: E402
from inference import Predictor, overlay  # noqa: E402

# Model dosyası: önce depoya konmuş kopya (Streamlit Cloud için), yoksa yerel eğitim çıktısı
CKPT_CANDIDATES = [ROOT / "models" / "solarai_effnet_b0_v2.pt", ROOT / "outputs" / "effnet_b0_v2" / "best.pt"]
EXAMPLES_DIR = ROOT / "app" / "examples"

st.set_page_config(page_title="SolarAI", page_icon="☀️", layout="wide")


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


def header() -> None:
    st.markdown(
        "<h1 style='margin-bottom:0'><span style='color:#F59E0B'>Solar</span>"
        "<span style='color:#1E3A8A'>AI</span></h1>",
        unsafe_allow_html=True,
    )
    st.caption("Panelin ne kadar kirli olduğunu değil, ne kadar kaybettirdiğini söyler.")


def sidebar() -> dict:
    st.sidebar.header("Sistem bilgileri")
    kwp = st.sidebar.number_input("Sistem gücü (kWp)", 0.5, 1000.0, le.DEFAULT_KWP, 0.5)
    price = st.sidebar.number_input("Elektrik fiyatı (TL/kWh)", 0.5, 20.0, le.DEFAULT_PRICE_TL_PER_KWH, 0.1)
    month = st.sidebar.selectbox("Ay", range(1, 13), index=6, format_func=le.month_name_tr)
    with st.sidebar.expander("Sınıf başına kayıp oranları (%)"):
        rates = {c: st.slider(CLASS_TR[c], 0, 100, int(le.LOSS_RATE[c] * 100), key=f"rate_{c}") / 100
                 for c in le.LOSS_RATE}
        st.caption("Varsayılanlar literatürden: docs/literatur_notlari.md")
    st.sidebar.info("Konum: Gaziantep (PVGIS 5.3, 2005–2023 ortalaması)")
    return {"kwp": kwp, "price": price, "month": month, "rates": rates}


def load_input_image() -> tuple[Image.Image | None, str]:
    examples = sorted(p for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
        if EXAMPLES_DIR.exists() else []
    source = st.radio("Fotoğraf kaynağı", ["Yükle", "Örnek seç"] if examples else ["Yükle"], horizontal=True, key="source")
    if source == "Örnek seç":
        name = st.selectbox("Örnek", [p.name for p in examples], key="example")
        return Image.open(EXAMPLES_DIR / name).convert("RGB"), name
    file = st.file_uploader("Panel fotoğrafı", type=["jpg", "jpeg", "png", "webp"])
    if file is None:
        return None, ""
    return Image.open(file).convert("RGB"), file.name


def tab_photo(cfg: dict) -> None:
    img, name = load_input_image()
    if img is None:
        st.info("Bir güneş paneli fotoğrafı yükleyin. En iyi sonuç: panele yakından, düz açıyla çekilmiş fotoğraf.")
        return

    pred = get_predictor()
    probs = pred.predict(img)
    top = max(probs, key=probs.get)
    heat = pred.gradcam(img, top)
    view = pred.preprocess_view(img)
    loss = le.photo_loss(probs, cfg["month"], cfg["kwp"], cfg["price"], cfg["rates"], get_pvgis())

    if loss.service_alert:
        st.error(f"⚠️ **Teknik servis çağırın.** Hasar olasılığı %{loss.damage_prob * 100:.0f}. "
                 "Fiziksel/elektriksel hasarın üretim kaybı ve güvenlik riski fotoğraftan güvenilir "
                 "biçimde tahmin edilemez; panel bir uzman tarafından kontrol edilmelidir.")

    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        st.image(view, caption=name, width="stretch")
    with c2:
        st.image(overlay(view, heat), caption=f"Grad-CAM: model '{CLASS_TR[top]}' derken buraya baktı",
                 width="stretch")
    with c3:
        st.subheader(f"Tahmin: {CLASS_TR[top]} (%{probs[top] * 100:.0f})")
        prob_df = pd.DataFrame({"Sınıf": [CLASS_TR[c] for c in CLASSES],
                                "Olasılık (%)": [round(probs[c] * 100, 1) for c in CLASSES]})
        st.bar_chart(prob_df, x="Sınıf", y="Olasılık (%)", horizontal=True, height=260, sort=False,
                     color="#F59E0B")

    st.divider()
    st.subheader(f"Tahmini kayıp — {le.month_name_tr(cfg['month'])}, {cfg['kwp']:g} kWp")
    if loss.service_alert:
        st.warning("Aşağıdaki rakamlar yalnızca **kirlenme** kaynaklı kaybı gösterir. "
                   "Hasarın yol açtığı kayıp bu hesaba **dahil değildir** ve daha büyük olabilir.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Anlık üretim kaybı", f"%{loss.loss_fraction * 100:.1f}")
    m2.metric("Günlük kayıp", f"{loss.daily_kwh:.2f} kWh", f"{loss.daily_tl:.2f} TL", delta_color="off")
    m3.metric("Aylık kayıp", f"{loss.monthly_kwh:.1f} kWh", f"{loss.monthly_tl:.0f} TL", delta_color="off")

    with st.expander("Hesap nasıl yapıldı?"):
        rows = [{"Sınıf": CLASS_TR[c], "Olasılık": f"%{probs[c] * 100:.1f}",
                 "Kayıp oranı": f"%{cfg['rates'][c] * 100:.0f}", "Katkı": f"%{loss.contributions[c] * 100:.2f}"}
                for c in le.LOSS_RATE]
        st.table(pd.DataFrame(rows))
        e_d = get_pvgis().set_index("month").loc[cfg["month"], "E_d"]
        st.markdown(
            f"**kayıp = Σ p(sınıf) · kayıp(sınıf) = %{loss.loss_fraction * 100:.2f}**  \n"
            f"günlük kayıp = %{loss.loss_fraction * 100:.2f} × {e_d:.2f} kWh/kWp (PVGIS) × {cfg['kwp']:g} kWp "
            f"= {loss.daily_kwh:.2f} kWh × {cfg['price']:.2f} TL/kWh = **{loss.daily_tl:.2f} TL**  \n"
            "Hasar sınıflarının olasılığı kayba eklenmez; bunun yerine servis uyarısı verilir "
            f"(toplam hasar olasılığı ≥ %{le.DAMAGE_ALERT_THRESHOLD * 100:.0f} ise)."
        )
    st.caption("Kayıp rakamları gösterim amaçlıdır. Model yakın çekim panel fotoğraflarıyla eğitildi; "
               "geniş açılı saha fotoğraflarında (ör. fırtına hasarı) daha az güvenilirdir.")


def main() -> None:
    header()
    cfg = sidebar()
    t1, t2, t3 = st.tabs(["📷 Fotoğraf Analizi", "🌫️ Toz Simülatörü", "📅 Gaziantep Temizlik Planlayıcısı"])
    with t1:
        tab_photo(cfg)
    with t2:
        st.info("Yakında: temiz bir fotoğrafa sentetik toz ekleyip tahminin ve kaybın canlı değişimini izleyin.")
    with t3:
        st.info("Yakında: Gaziantep için aylık en uygun temizlik aralığı, yıllık takvim ve kazanç.")


main()
