"""SolarAI — Streamlit demo uygulaması.

Çalıştırma (proje kökünden):
    streamlit run app/app.py

Sekmeler:
  1. Fotoğraf Analizi    : sınıf olasılıkları, Grad-CAM, kWh/TL kaybı, servis uyarısı
  2. Toz Simülatörü      : temiz fotoğrafa sentetik toz; tahmin ve kayıp canlı değişir
  3. Temizlik Planlayıcı : Gaziantep için aylık T*, yıllık takvim ve kazanç
  4. Proje Hakkında      : problem, yöntem, sonuçlar, sınırlamalar, kaynaklar
"""

from __future__ import annotations

import base64
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
from brand import logo_svg, mark_svg  # noqa: E402
from common import CLASS_TR, CLASSES  # noqa: E402
from dust import add_dust  # noqa: E402
from inference import Predictor, overlay  # noqa: E402

CKPT_CANDIDATES = [ROOT / "models" / "solarai_effnet_b0_v2.pt", ROOT / "outputs" / "effnet_b0_v2" / "best.pt"]
EXAMPLES_DIR = ROOT / "app" / "examples"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

ORANGE, AMBER, NAVY = "#F59E0B", "#FBBF24", "#1E3A8A"
BG, TXT, MUTED, LINE = "#081129", "#E2E8F0", "#94A3B8", "rgba(148,163,184,.16)"
CLASS_COLOR = {"clean": "#34D399", "dusty": "#F59E0B", "bird_drop": "#A78BFA", "snow_covered": "#38BDF8",
               "physical_damage": "#F87171", "electrical_damage": "#EF4444"}

_FAVICON = ROOT / "app" / "assets" / "favicon.png"
st.set_page_config(page_title="SolarAI · Güneş paneli kayıp analizi",
                   page_icon=Image.open(_FAVICON) if _FAVICON.exists() else "☀️",
                   layout="wide", initial_sidebar_state="collapsed")

# Lucide tarzı çizgi ikonlar (emoji yerine)
_ICON = {
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
             '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "zap": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="M22 4 12 14.01l-3-3"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41'
           'M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor") -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" style="flex:none">{_ICON[name]}</svg>')


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {{ --glass: rgba(17,29,68,.55); --glass-2: rgba(255,255,255,.04); --line: {LINE}; --muted: {MUTED}; }}
html, body, .stApp, [class*="css"], p, li, label, input, select, textarea, button {{ font-family: 'Inter', system-ui, sans-serif !important; }}
h1, h2, h3, h4, .g, .kpi-value, .pred-cls, .sec, .hero-title, .stat-num, .step-t {{ font-family: 'Space Grotesk', 'Inter', sans-serif !important; }}
.stApp {{ background:
    radial-gradient(1100px 520px at 85% -8%, rgba(245,158,11,.16), transparent 60%),
    radial-gradient(900px 600px at -10% 10%, rgba(37,99,235,.20), transparent 55%), {BG}; }}
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stStatusWidget"], [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1200px; }}

/* hero */
.hero {{ position: relative; border-radius: 24px; padding: 28px 30px; overflow: hidden;
        background: linear-gradient(140deg, rgba(30,58,138,.55), rgba(8,17,41,.35)); border: 1px solid var(--line); }}
.hero:before {{ content: ""; position: absolute; inset: 0; opacity: .08;
  background-image: linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px);
  background-size: 32px 32px; mask-image: linear-gradient(90deg, transparent, #000 60%); }}
.hero-grid {{ position: relative; display: grid; grid-template-columns: minmax(0,1fr) 300px; gap: 30px; align-items: center; }}
@media (max-width: 1000px) {{ .hero-grid {{ grid-template-columns: 1fr; }} }}
.eyebrow {{ display: inline-flex; gap: 8px; align-items: center; font-size: .75rem; font-weight: 600; letter-spacing: .08em;
           text-transform: uppercase; color: {AMBER}; background: rgba(245,158,11,.10); border: 1px solid rgba(245,158,11,.25);
           padding: 5px 12px; border-radius: 999px; margin-top: 14px; }}
.hero-title {{ font-size: clamp(1.6rem, 3vw, 2.35rem); font-weight: 700; line-height: 1.12; letter-spacing: -.02em;
              color: #fff; margin: 14px 0 0; max-width: 680px; }}
.hero-title span {{ color: {AMBER}; }}
.hero-sub {{ color: #CBD5E1; font-size: 1rem; line-height: 1.6; margin-top: 12px; max-width: 620px; }}
.stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.stat {{ background: var(--glass-2); border: 1px solid var(--line); border-radius: 16px; padding: 14px; backdrop-filter: blur(8px); }}
.stat-num {{ font-size: 1.7rem; font-weight: 700; color: #fff; line-height: 1; }}
.stat-num small {{ color: {AMBER}; font-size: .9rem; margin-left: 2px; }}
.stat-lbl {{ color: var(--muted); font-size: .76rem; margin-top: 6px; }}

/* steps */
.steps {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin: 14px 0 6px; }}
@media (max-width: 900px) {{ .steps {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
.step {{ display: flex; gap: 10px; align-items: flex-start; padding: 12px 14px; border-radius: 16px;
        background: var(--glass-2); border: 1px solid var(--line); }}
.step-n {{ flex: none; width: 26px; height: 26px; border-radius: 9px; display: grid; place-items: center; font-weight: 700;
          font-size: .8rem; color: #0B1530; background: {AMBER}; }}
.step-t {{ font-weight: 600; color: #fff; font-size: .95rem; }}
.step-d {{ color: var(--muted); font-size: .78rem; line-height: 1.45; margin-top: 2px; }}

/* cards & glass containers */
.card {{ background: var(--glass); border: 1px solid var(--line); border-radius: 18px; padding: 16px 18px; backdrop-filter: blur(10px); height: 100%; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{ border-radius: 18px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--line) !important; }}
.kpi-label {{ color: var(--muted); font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; }}
.kpi-value {{ color: #fff; font-size: 1.8rem; font-weight: 700; margin-top: 6px; line-height: 1.05; letter-spacing: -.02em; }}
.kpi-sub {{ color: var(--muted); font-size: .8rem; margin-top: 6px; }}
.kpi-accent {{ box-shadow: inset 0 3px 0 {ORANGE}; }} .kpi-blue {{ box-shadow: inset 0 3px 0 #3B82F6; }} .kpi-green {{ box-shadow: inset 0 3px 0 #34D399; }}

.pred {{ border-radius: 18px; padding: 16px 18px; border: 1px solid var(--line); background: var(--glass); }}
.pred-top {{ display: flex; justify-content: space-between; align-items: center; }}
.pred-lbl {{ color: var(--muted); font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }}
.pred-chip {{ font-size: .75rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; }}
.pred-cls {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -.02em; margin: 6px 0 2px; }}
.bars {{ margin-top: 12px; display: grid; gap: 9px; }}
.bar-row {{ display: grid; grid-template-columns: 112px 1fr 44px; gap: 10px; align-items: center; font-size: .82rem; color: #CBD5E1; }}
.bar-track {{ height: 8px; border-radius: 99px; background: rgba(148,163,184,.14); overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 99px; }}
.bar-val {{ text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }}

.note {{ display: flex; gap: 10px; align-items: flex-start; border-radius: 14px; padding: 12px 14px; margin: 6px 0 12px; font-size: .9rem; line-height: 1.5; }}
.note-info {{ background: rgba(59,130,246,.10); border: 1px solid rgba(59,130,246,.30); color: #BFDBFE; }}
.note-warn {{ background: rgba(245,158,11,.10); border: 1px solid rgba(245,158,11,.35); color: #FDE68A; }}
.note-danger {{ background: rgba(239,68,68,.12); border: 1px solid rgba(239,68,68,.40); color: #FECACA; }}
.note b {{ color: #fff; }}

.sec {{ font-size: 1.2rem; font-weight: 700; color: #fff; margin: 26px 0 10px; display: flex; align-items: center; gap: 10px; }}
.sec:before {{ content: ""; width: 6px; height: 20px; border-radius: 3px; background: linear-gradient({AMBER}, {ORANGE}); }}
.muted {{ color: var(--muted); font-size: .85rem; }}
.cap {{ color: var(--muted); font-size: .78rem; text-align: center; margin-top: 4px; }}
[data-testid="stImage"] img {{ border-radius: 16px; border: 1px solid var(--line); }}
.thumbs {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 2px 0 6px; }}
.thumbs img {{ width: 56px; height: 56px; object-fit: cover; border-radius: 12px; border: 2px solid transparent; opacity: .55; }}
.thumbs img.on {{ border-color: {ORANGE}; opacity: 1; }}

/* tabs → segmentli düğmeler */
.stTabs [role="tablist"] {{ gap: 6px; background: rgba(255,255,255,.04); border: 1px solid var(--line) !important; padding: 6px;
                            border-radius: 16px; box-shadow: none !important; flex-wrap: wrap; margin-top: 14px; }}
.stTabs [role="tablist"]:after, .stTabs [role="tablist"]:before {{ display: none !important; }}
.stTabs [role="tab"] {{ height: 42px; padding: 0 16px; border-radius: 11px; border: none !important; color: var(--muted); }}
.stTabs [role="tab"] p {{ font-family: 'Space Grotesk', sans-serif !important; font-weight: 600; font-size: .95rem; color: inherit; }}
.stTabs [role="tab"]:hover {{ color: #fff; background: rgba(255,255,255,.05); }}
.stTabs [role="tab"][aria-selected="true"] {{ background: {ORANGE} !important; color: #0B1530 !important; }}
.stTabs [role="tab"][aria-selected="true"] p, .stTabs [role="tab"][aria-selected="true"] span {{ color: #0B1530 !important; }}
.stTabs [role="tab"]:after, .stTabs [role="tab"]:before {{ display: none !important; }}

/* widgets */
.stButton > button, [data-testid="stPopover"] button {{ border-radius: 12px; border: 1px solid var(--line); background: rgba(255,255,255,.04); font-weight: 600; }}
.stButton > button:hover {{ border-color: {ORANGE}; color: {AMBER}; }}
[data-testid="stNumberInput"] input, [data-baseweb="select"] > div {{ border-radius: 12px !important; }}
[data-testid="stFileUploaderDropzone"] {{ border-radius: 16px; border: 1.5px dashed rgba(245,158,11,.45); background: rgba(245,158,11,.05); }}
[data-testid="stExpander"] details {{ border-radius: 16px; border: 1px solid var(--line); background: rgba(255,255,255,.02); }}
[data-testid="stWidgetLabel"] p {{ color: #CBD5E1; font-weight: 500; }}

/* about */
.grid2 {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }}
@media (max-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
.ac h4 {{ margin: 0 0 6px; font-size: 1.02rem; color: #fff; }}
.ac p, .ac li {{ font-size: .88rem; color: #CBD5E1; line-height: 1.6; }}
.ac ul {{ padding-left: 18px; margin: 6px 0 0; }}
.ac b {{ color: #fff; }}

.footer {{ margin-top: 40px; padding: 18px 0 6px; border-top: 1px solid var(--line); color: var(--muted); font-size: .8rem;
          display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
.footer a {{ color: {AMBER}; text-decoration: none; font-weight: 600; }}
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


def to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


@st.cache_data
def thumb_b64(path: str) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((160, 160))
    return base64.b64encode(to_bytes(im, "JPEG")).decode()


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "accent") -> str:
    return (f"<div class='card kpi-{tone}'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>")


def note(text: str, kind: str = "info") -> None:
    ic = {"info": "info", "warn": "alert", "danger": "alert"}[kind]
    html(f"<div class='note note-{kind}'>{icon(ic)}<div>{text}</div></div>")


def section(title: str) -> None:
    html(f"<div class='sec'>{title}</div>")


def tl(x: float, digits: int = 0) -> str:
    s = f"{x:,.{digits}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") + " TL"


def list_examples(only_clean: bool = False) -> list[Path]:
    if not EXAMPLES_DIR.exists():
        return []
    ex = sorted(p for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in IMG_EXTS)
    return [p for p in ex if "temiz" in p.stem] if only_clean else ex


def example_label(p: Path) -> str:
    stem = p.stem.split("_", 1)[-1]
    return {"temiz": "Temiz", "temiz_2": "Temiz 2", "temiz_3": "Temiz 3", "tozlu": "Tozlu", "kus_pisligi": "Kuş pisliği",
            "kar": "Kar", "fiziksel_hasar": "Kırık cam", "elektriksel_hasar": "Elektriksel"}.get(stem, stem)


def dark_chart(ch: alt.Chart, height: int) -> alt.Chart:
    return (ch.properties(height=height, background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=LINE, domainColor=LINE, tickColor=LINE,
                            labelFont="Inter", titleFont="Inter", labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFont="Inter", titleFont="Inter"))


# ------------------------------------------------------------------ bileşenler
def prediction_card(probs: dict[str, float]) -> None:
    top = max(probs, key=probs.get)
    col = CLASS_COLOR[top]
    rows = "".join(
        f"<div class='bar-row'><span>{CLASS_TR[c]}</span><div class='bar-track'>"
        f"<div class='bar-fill' style='width:{probs[c] * 100:.1f}%;background:{CLASS_COLOR[c]};"
        f"opacity:{1 if c == top else .55}'></div></div><span class='bar-val'>%{probs[c] * 100:.0f}</span></div>"
        for c in CLASSES)
    html(f"<div class='pred'><div class='pred-top'><span class='pred-lbl'>Model tahmini</span>"
         f"<span class='pred-chip' style='background:{col}22;color:{col};border:1px solid {col}55'>"
         f"güven %{probs[top] * 100:.0f}</span></div>"
         f"<div class='pred-cls' style='color:{col}'>{CLASS_TR[top]}</div>"
         f"<div class='bars'>{rows}</div></div>")


def loss_kpis(loss: le.PhotoLoss, cfg: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Üretim kaybı", f"%{loss.loss_fraction * 100:.1f}", "kirlenme kaynaklı"), unsafe_allow_html=True)
    c2.markdown(kpi("Günlük", tl(loss.daily_tl, 2), f"{loss.daily_kwh:.2f} kWh · {le.month_name_tr(cfg['month'])}", "blue"),
                unsafe_allow_html=True)
    c3.markdown(kpi("Aylık", tl(loss.monthly_tl), f"{loss.monthly_kwh:.0f} kWh · {cfg['kwp']:g} kWp", "blue"),
                unsafe_allow_html=True)


def image_viewer(view: Image.Image, heat_img: Image.Image, key: str, caption: str) -> None:
    mode = st.segmented_control("Görünüm", ["Fotoğraf", "Grad-CAM", "Yan yana"], default="Yan yana",
                                key=f"{key}_view", label_visibility="collapsed") or "Yan yana"
    if mode == "Yan yana":
        a, b = st.columns(2, gap="small")
        a.image(view, width="stretch")
        b.image(heat_img, width="stretch")
        html(f"<div class='cap'>Fotoğraf ve Grad-CAM · {caption}</div>")
    else:
        st.image(view if mode == "Fotoğraf" else heat_img, width="stretch")
        html(f"<div class='cap'>{'Model girdisi' if mode == 'Fotoğraf' else 'Grad-CAM: ' + caption}</div>")


def image_picker(key: str, only_clean: bool = False) -> tuple[Image.Image | None, str]:
    """Tam genişlikte: solda örnek seçimi (hap düğmeler + küçük resimler), sağda yükleme alanı."""
    examples = list_examples(only_clean)
    with st.container(border=True):
        a, b = st.columns([1.6, 1], gap="large")
        with b:
            upload = st.file_uploader("Kendi fotoğrafınızı yükleyin", type=["jpg", "jpeg", "png", "webp"],
                                      key=f"{key}_upload")
        with a:
            if upload is not None:
                html("<div class='muted' style='padding-top:28px'>Yüklediğiniz fotoğraf analiz ediliyor. "
                     "Örneklere dönmek için yüklenen dosyayı kaldırın (×).</div>")
                return Image.open(upload).convert("RGB"), upload.name
            if not examples:
                return None, ""
            labels = [example_label(p) for p in examples]
            choice = st.pills("Bir örnek seçin", labels, default=labels[0], key=f"{key}_example") or labels[0]
            sel = examples[labels.index(choice)]
            html("<div class='thumbs'>" + "".join(
                f"<img class='{'on' if p == sel else ''}' src='data:image/jpeg;base64,{thumb_b64(str(p))}' "
                f"alt='{example_label(p)}' title='{example_label(p)}'>" for p in examples) + "</div>")
    return Image.open(sel).convert("RGB"), sel.name


# ------------------------------------------------------------------ düzen
STEPS = [("Fotoğraf", "Panelin yakın çekim fotoğrafını yükleyin ya da örnek seçin."),
         ("Yapay zekâ", "EfficientNet-B0 durumu 6 sınıfta tahmin eder, Grad-CAM nereye baktığını gösterir."),
         ("Kayıp", "Olasılıklar Gaziantep üretimiyle kWh ve TL kaybına çevrilir."),
         ("Plan", "Temizlik ücreti ile kayıp dengelenir, en ekonomik takvim çıkar.")]


def hero() -> None:
    stats = [("0.888", "", "test macro-F1"), ("790", "", "temizlenmiş görüntü"),
             ("1580", "kWh", "kWp başına yıllık · Gaziantep"), ("%96", "", "doğru tahminde güven (kar)")]
    html("<div class='hero'><div class='hero-grid'><div>"
         f"{logo_svg(dark=True, height=52)}"
         f"<div class='eyebrow'>{icon('sun', 14)} HASAT-13 · Sürdürülebilir enerji · Yapay zekâ</div>"
         "<div class='hero-title'>Panelin ne kadar kirli olduğunu değil, <span>ne kadar kaybettirdiğini</span> söyler.</div>"
         "<div class='hero-sub'>Güneş panellerinde kirlenmenin yapay zekâ ile tespiti ve Gaziantep verisiyle "
         "ekonomik temizlik planlaması.</div></div><div class='stats'>"
         + "".join(f"<div class='stat'><div class='stat-num'>{n}<small>{u}</small></div><div class='stat-lbl'>{lbl}</div></div>"
                   for n, u, lbl in stats[:3])
         + "<div class='stat'><div class='stat-num'>6</div><div class='stat-lbl'>panel durumu sınıfı</div></div>"
         "</div></div></div>")
    html("<div class='steps'>" + "".join(
        f"<div class='step'><div class='step-n'>{i}</div><div><div class='step-t'>{t}</div><div class='step-d'>{d}</div></div></div>"
        for i, (t, d) in enumerate(STEPS, 1)) + "</div>")


def settings_bar() -> dict:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.1], vertical_alignment="bottom")
        kwp = c1.number_input("Sistem gücü (kWp)", 0.5, 1000.0, le.DEFAULT_KWP, 0.5,
                              help="Ev çatısı tipik 3–10 kWp, işletme 50–500 kWp")
        price = c2.number_input("Elektrik fiyatı (TL/kWh)", 0.5, 20.0, le.DEFAULT_PRICE_TL_PER_KWH, 0.1,
                                help="Mesken, Nisan 2026 sonrası, vergiler dahil (yaklaşık)")
        month = c3.selectbox("Ay", range(1, 13), index=6, format_func=le.month_name_tr,
                             help="Kayıp, o ayın Gaziantep üretimine göre hesaplanır (PVGIS 5.3)")
        with c4.popover("Kayıp oranları", icon=":material/tune:", use_container_width=True):
            st.caption("Bir fotoğraftaki tipik üretim kaybı. Varsayılanlar literatürden; hasar sınıfları TL hesabına girmez.")
            rates = {c: st.slider(CLASS_TR[c], 0, 100, int(le.LOSS_RATE[c] * 100), format="%d%%", key=f"rate_{c}") / 100
                     for c in le.LOSS_RATE}
    return {"kwp": kwp, "price": price, "month": month, "rates": rates}


# ------------------------------------------------------------------ 1) fotoğraf analizi
def tab_photo(cfg: dict) -> None:
    img, name = image_picker("photo")
    left, right = st.columns([1.35, 1], gap="large")
    if img is None:
        note("Bir güneş paneli fotoğrafı yükleyin. En iyi sonuç: panele <b>yakından</b> ve <b>düz açıyla</b> çekilmiş fotoğraf.")
        return

    pred = get_predictor()
    probs = pred.predict(img)
    top = max(probs, key=probs.get)
    view = pred.preprocess_view(img)
    loss = le.photo_loss(probs, cfg["month"], cfg["kwp"], cfg["price"], cfg["rates"], get_pvgis())

    with left:
        image_viewer(view, overlay(view, pred.gradcam(img, top)), "photo", f"model “{CLASS_TR[top]}” derken buraya baktı")
    with right:
        if loss.service_alert:
            note(f"<b>Teknik servis çağırın.</b> Hasar olasılığı %{loss.damage_prob * 100:.0f}. Hasarın kaybı ve "
                 "güvenlik riski fotoğraftan güvenilir biçimde tahmin edilemez.", "danger")
        prediction_card(probs)

    section(f"Tahmini kayıp · {le.month_name_tr(cfg['month'])}, {cfg['kwp']:g} kWp")
    if loss.service_alert:
        note("Rakamlar yalnızca <b>kirlenme</b> kaybını gösterir; hasarın yol açtığı kayıp <b>dahil değildir</b>.", "warn")
    loss_kpis(loss, cfg)

    with st.expander("Hesap nasıl yapıldı?", icon=":material/calculate:"):
        rows = [{"Sınıf": CLASS_TR[c], "Olasılık": f"%{probs[c] * 100:.1f}",
                 "Kayıp oranı": f"%{cfg['rates'][c] * 100:.0f}", "Katkı": f"%{loss.contributions[c] * 100:.2f}"}
                for c in le.LOSS_RATE]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        e_d = get_pvgis().set_index("month").loc[cfg["month"], "E_d"]
        st.markdown(
            f"**kayıp = Σ p(sınıf) · kayıp(sınıf) = %{loss.loss_fraction * 100:.2f}**  \n"
            f"günlük kayıp = %{loss.loss_fraction * 100:.2f} × {e_d:.2f} kWh/kWp (PVGIS) × {cfg['kwp']:g} kWp "
            f"= {loss.daily_kwh:.2f} kWh × {cfg['price']:.2f} TL/kWh = **{loss.daily_tl:.2f} TL**  \n"
            f"Toplam hasar olasılığı ≥ %{le.DAMAGE_ALERT_THRESHOLD * 100:.0f} ise servis uyarısı verilir.")
    html("<div class='muted' style='margin-top:10px'>Kayıp rakamları gösterim amaçlıdır. Model yakın çekim panel "
         "fotoğraflarıyla eğitildi; geniş açılı saha fotoğraflarında daha az güvenilirdir.</div>")


# ------------------------------------------------------------------ 2) toz simülatörü
@st.cache_data(show_spinner=False)
def dust_curve(img_bytes: bytes, seed: int, month: int, kwp: float, price: float, rates: tuple) -> pd.DataFrame:
    pred, img, rows = get_predictor(), from_bytes(img_bytes), []
    for level in np.linspace(0, 1, 11):
        probs = pred.predict(add_dust(img, float(level), seed))
        loss = le.photo_loss(probs, month, kwp, price, dict(rates), get_pvgis())
        rows.append({"Toz (%)": round(level * 100), "Günlük kayıp (TL)": loss.daily_tl, "p(tozlu) (%)": probs["dusty"] * 100})
    return pd.DataFrame(rows)


def tab_dust(cfg: dict) -> None:
    note("Temiz bir panel fotoğrafına <b>sentetik toz</b> ekleyin. Kaydırdıkça tahmin, Grad-CAM ve kayıp canlı güncellenir.")
    img, _ = image_picker("dust", only_clean=True)
    if img is None:
        return
    left, right = st.columns([1.35, 1], gap="large")
    with right:
        with st.container(border=True):
            level = st.slider("Toz yoğunluğu", 0, 100, 40, 5, format="%d%%", key="dust_level")
            if "dust_seed" not in st.session_state:
                st.session_state.dust_seed = 1
            if st.button("Yeni toz deseni", icon=":material/casino:", width="stretch"):
                st.session_state.dust_seed += 1
        seed = st.session_state.dust_seed

    pred = get_predictor()
    dusty = add_dust(img, level / 100, seed)
    probs = pred.predict(dusty)
    top = max(probs, key=probs.get)
    view = pred.preprocess_view(dusty)
    loss = le.photo_loss(probs, cfg["month"], cfg["kwp"], cfg["price"], cfg["rates"], get_pvgis())
    with left:
        image_viewer(view, overlay(view, pred.gradcam(dusty, top)), "dust", f"toz %{level}")
    with right:
        prediction_card(probs)
        st.write("")
        html(kpi("Günlük kayıp", tl(loss.daily_tl, 2), f"%{loss.loss_fraction * 100:.1f} · {loss.daily_kwh:.2f} kWh", "blue"))

    section("Toz arttıkça kayıp nasıl değişiyor?")
    curve = dust_curve(to_bytes(img), seed, cfg["month"], cfg["kwp"], cfg["price"], tuple(sorted(cfg["rates"].items())))
    now = pd.DataFrame({"Toz (%)": [level], "Günlük kayıp (TL)": [loss.daily_tl]})
    area = alt.Chart(curve).mark_area(color=ORANGE, opacity=.12, interpolate="monotone").encode(
        x="Toz (%):Q", y="Günlük kayıp (TL):Q")
    line = alt.Chart(curve).mark_line(color=ORANGE, strokeWidth=3, interpolate="monotone",
                                      point=alt.OverlayMarkDef(color=ORANGE, size=36)).encode(
        x=alt.X("Toz (%):Q", scale=alt.Scale(domain=[0, 100])), y=alt.Y("Günlük kayıp (TL):Q"),
        tooltip=[alt.Tooltip("Toz (%):Q"), alt.Tooltip("Günlük kayıp (TL):Q", format=".2f"),
                 alt.Tooltip("p(tozlu) (%):Q", format=".0f")])
    rule = alt.Chart(now).mark_rule(color="#fff", strokeDash=[4, 4], opacity=.6).encode(x="Toz (%):Q")
    dot = alt.Chart(now).mark_point(color="#fff", size=200, filled=True).encode(x="Toz (%):Q", y="Günlük kayıp (TL):Q")
    st.altair_chart(dark_chart(area + line + rule + dot, 280), width="stretch")
    html("<div class='muted'>Kayıp rakamları gösterim amaçlıdır. Sentetik toz gerçek toz fiziğinin modeli değildir; "
         "amacı modelin kirliliğe tepkisini görselleştirmektir.</div>")


# ------------------------------------------------------------------ 3) temizlik planlayıcısı
@st.cache_data(show_spinner="Takvim hesaplanıyor…")
def get_plan(kwp: float, cost: float, price: float, rates: tuple) -> dict:
    return le.cleaning_plan(kwp=kwp, cost=cost, price=price, rates=dict(rates))


def tab_plan(cfg: dict) -> None:
    note("Gaziantep'in aylık güneş üretimine göre panelin <b>ne zaman</b> temizlenmesinin ekonomik olduğunu hesaplar: "
         "temizlik ücreti ile kirin kaybettirdiği elektrik dengelenir.")
    c1, c2 = st.columns([1, 2.3], gap="large")
    with c1:
        with st.container(border=True):
            cost = st.number_input("Bir temizliğin maliyeti (TL)", 50.0, 500000.0, le.DEFAULT_CLEANING_COST_TL, 100.0,
                                   help="Varsayım: Türkiye için güvenilir fiyat listesi bulunamadı; kendi teklifinizi girin.")
            html(f"<div class='muted'>Sistem <b style='color:#fff'>{cfg['kwp']:g} kWp</b> · Elektrik "
                 f"<b style='color:#fff'>{cfg['price']:.2f} TL/kWh</b></div>")
            with st.popover("Kirlenme oranı r", icon=":material/water_drop:", use_container_width=True):
                r_df = pd.DataFrame({"Ay": [le.month_name_tr(m) for m in range(1, 13)],
                                     "r (%/gün)": [le.MONTHLY_SOILING_RATE[m] * 100 for m in range(1, 13)]})
                edited = st.data_editor(r_df, hide_index=True, disabled=["Ay"], key="r_editor",
                                        column_config={"r (%/gün)": st.column_config.NumberColumn(
                                            min_value=0.001, max_value=2.0, step=0.01, format="%.2f")})
                st.caption("Varsayım: kurak yaz %0,20 · geçiş %0,10 · yağışlı kış %0,05 (Kimber vd. 2006 aralığı).")
        rates = {m: float(v) / 100 for m, v in zip(range(1, 13), edited["r (%/gün)"])}
    plan = get_plan(cfg["kwp"], cost, cfg["price"], tuple(sorted(rates.items())))
    comp = plan["comparison"]
    days = ", ".join(f"{d.day} {le.month_name_tr(d.month)}" for d in plan["cleaning_days"]) or "Gerekmiyor"
    with c2:
        k1, k2, k3 = st.columns(3)
        k1.markdown(kpi("Önerilen temizlik", f"{len(plan['cleaning_days'])} kez/yıl", days), unsafe_allow_html=True)
        k2.markdown(kpi("Ayda bire göre", tl(plan["saving_vs_monthly_tl"]), "yıllık tasarruf", "green"), unsafe_allow_html=True)
        k3.markdown(kpi("Hiç temizlememeye göre", tl(plan["saving_vs_never_tl"]), "yıllık tasarruf", "green"),
                    unsafe_allow_html=True)

    section("Yıllık takvim · günlük kirlenme kaybı")
    d = plan["daily"].assign(Ay=lambda x: x["month"].map(le.month_name_tr), Kayıp=lambda x: x["soiling"] * 100)
    months = [le.month_name_tr(m) for m in range(1, 13)]
    heat = alt.Chart(d).mark_rect(cornerRadius=3, stroke=BG, strokeWidth=1.5).encode(
        x=alt.X("day:O", title="Gün", axis=alt.Axis(labelAngle=0, values=[1, 5, 10, 15, 20, 25, 30])),
        y=alt.Y("Ay:N", sort=months, title=None),
        color=alt.Color("Kayıp:Q", title="Kayıp (%)", scale=alt.Scale(range=["#1E293B", "#7C2D12", ORANGE, AMBER])),
        tooltip=[alt.Tooltip("date:T", title="Tarih", format="%d.%m"), alt.Tooltip("Kayıp:Q", format=".1f")])
    marks = alt.Chart(d[d["cleaning"]]).mark_point(shape="circle", size=140, filled=True, color="#34D399",
                                                   stroke="#fff", strokeWidth=2).encode(x="day:O", y=alt.Y("Ay:N", sort=months))
    st.altair_chart(dark_chart(heat + marks, 340), width="stretch")
    html("<div class='muted'><span style='color:#34D399'>●</span> önerilen temizlik günü · renk, o gün birikmiş kirin "
         "yol açtığı üretim kaybıdır.</div>")

    section("Stratejilerin karşılaştırması (1 yıl)")
    show = comp.rename(columns={"strateji": "Strateji", "temizlik_sayisi": "Temizlik", "kayip_kwh": "Kayıp (kWh)",
                                "kayip_tl": "Kayıp (TL)", "temizlik_tl": "Temizlik (TL)", "toplam_tl": "Toplam (TL)"})
    bars = alt.Chart(show).mark_bar(cornerRadiusEnd=8, height=26).encode(
        y=alt.Y("Strateji:N", sort=None, title=None, axis=alt.Axis(labelLimit=220, labelFontSize=12, labelColor=TXT)),
        x=alt.X("Toplam (TL):Q", title="Yıllık toplam maliyet (TL)"),
        color=alt.condition(alt.datum.Strateji == comp.iloc[0]["strateji"], alt.value(ORANGE), alt.value("#334155")),
        tooltip=list(show.columns))
    lc, rc = st.columns([1.2, 1], gap="large")
    lc.altair_chart(dark_chart(bars, 200), width="stretch")
    rc.dataframe(show, hide_index=True, width="stretch")

    with st.expander("Formül: en uygun temizlik aralığı T*", icon=":material/functions:"):
        iv = plan["intervals"].assign(Ay=lambda x: x["month"].map(le.month_name_tr))
        st.markdown(
            "T günde bir temizlikte günlük ortalama maliyet: **f(T) = C/T + r·E·p·T/2**. İlk terim (temizlik ücreti) "
            "T büyüdükçe azalır, ikincisi (kirin kaybettirdiği elektrik) artar; türevi sıfıra eşitleyince "
            "**T\\* = √(2C / (r·E·p))**. Aylar arasında r ve E değiştiği ve kir %30'da doyduğu için yıllık takvim "
            "**dinamik programlamayla** kesin optimum olarak hesaplanır; aylık T* değerleri sezgi içindir.")
        tchart = alt.Chart(iv).mark_bar(color="#3B82F6", cornerRadiusEnd=6).encode(
            x=alt.X("Ay:N", sort=months, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("T_star_days:Q", title="T* (gün)"),
            tooltip=[alt.Tooltip("T_star_days:Q", title="T* (gün)", format=".0f"),
                     alt.Tooltip("E_kwh_per_day:Q", title="E (kWh/gün)", format=".1f")])
        st.altair_chart(dark_chart(tchart, 220), width="stretch")
    html("<div class='muted' style='margin-top:8px'>Varsayımlar: kir doğrusal birikir ve %30'da doyar; panel 1 Ocak'ta "
         "temiz başlar; r ve temizlik ücreti varsayımdır. Üretim: PVGIS 5.3 Gaziantep (optimum eğim 31°).</div>")


# ------------------------------------------------------------------ 4) proje hakkında
def ac(title: str, body: str) -> str:
    return f"<div class='card ac'><h4>{title}</h4>{body}</div>"


def tab_about() -> None:
    section("Problem")
    html("<div class='grid2'>"
         + ac("Kirli panel, sessiz kayıp",
              "<p>Toz, kuş pisliği ve kar panelin üretimini düşürür; kurak bölgelerde kayıp %10–40'a ulaşabilir. Asıl soru "
              "<b>ne kadar kirli</b> değil, <b>ne kadar kaybettirdiği</b> ve <b>temizliğe değip değmediğidir</b>.</p>")
         + ac("Sezgiyle değil, hesapla",
              "<p>Temizlik çoğunlukla 'ayda bir' gibi alışkanlıklarla yapılır. Gaziantep'te 5 kWp'lik bir ev sistemi için "
              "bu, en uygun plana göre yılda <b>14.000 TL'den fazla</b> fazladan maliyet demek.</p>")
         + "</div>")
    section("Yöntem")
    html("<div class='grid2'>"
         + ac("1 · Görüntü sınıflandırma", "<ul><li>ImageNet ön eğitimli <b>EfficientNet-B0</b> (timm), 6 sınıfta ince ayar</li>"
              "<li>Veri artırma, sınıf ağırlıklı kayıp, erken durdurma</li><li><b>Grad-CAM</b> ile açıklanabilirlik</li></ul>")
         + ac("2 · Veri temizliği", "<ul><li>MD5 ile 91 birebir kopya ve 2 çelişkili etiket çıkarıldı</li>"
              "<li>dHash + ImageNet benzerliğiyle 86 <b>yakın kopya grubu</b></li>"
              "<li>Gruplar bölünmeden %70/15/15 bölme → <b>sızıntı yok</b></li></ul>")
         + ac("3 · Kayıp hesabı", "<p><b>kayıp = Σ p(sınıf) · kayıp(sınıf)</b>; oranlar literatürden (toz %15, kuş pisliği %25, "
              "kar %60). Üretim: PVGIS 5.3 Gaziantep. Hasarda TL yerine <b>servis uyarısı</b>.</p>")
         + ac("4 · Temizlik planı", "<p><b>T* = √(2C / (r·E·p))</b>; yıllık takvim 365 gün üzerinde <b>dinamik programlama</b> "
              "ile kesin optimum, kaba kuvvet aramasıyla doğrulandı.</p>")
         + "</div>")
    section("Sonuçlar")
    cols = st.columns(4)
    for col, (lbl, val, sub, tone) in zip(cols, [("Test macro-F1", "0.888", "113 görüntü, sızıntısız", "accent"),
                                                ("Doğruluk", "%89.4", "101 / 113", "blue"),
                                                ("Hasar yakalama", "19 / 22", "servis uyarısı", "blue"),
                                                ("Dış test · kar", "44 / 44", "başka kaynaktan", "green")]):
        col.markdown(kpi(lbl, val, sub, tone), unsafe_allow_html=True)
    st.write("")
    section("Gelecek hedefleri")
    html(ac("Sırada ne var?", "<ul><li>Gaziantep'te sahadan toplanan etiketli fotoğraflar</li>"
            "<li>Referans hücreyle yerel kirlenme ölçümü ve r kalibrasyonu</li>"
            "<li>Drone görüntüleriyle saha ölçeğinde tarama</li><li>Hasar için termal kamera verisi</li></ul>"))
    section("Kaynaklar")
    html("<div class='card ac'><ul>"
         "<li>Afroz, <i>Solar Panel Images Clean and Faulty Images</i>, Kaggle · Lenarczyk, <i>PV Panel Defect Dataset</i>, Kaggle</li>"
         "<li>PVGIS 5.3, Avrupa Komisyonu Ortak Araştırma Merkezi (JRC)</li>"
         "<li>Tan &amp; Le (2019), EfficientNet, ICML · Selvaraju vd. (2017), Grad-CAM, ICCV</li>"
         "<li>Kimber vd. (2006), IEEE WCPEC · Jones vd. (2016), IEEE J. Photovoltaics · Uykan vd. (2025), <i>Sustainability</i></li>"
         "<li>Tam liste: <a style='color:#FBBF24' href='https://github.com/salihdinckann/SolarAI/blob/main/docs/literatur_notlari.md' "
         "target='_blank'>docs/literatur_notlari.md</a></li></ul></div>")


# ------------------------------------------------------------------ ana
def main() -> None:
    html(CSS)
    hero()
    cfg = settings_bar()
    t1, t2, t3, t4 = st.tabs([":material/photo_camera: Fotoğraf analizi", ":material/blur_on: Toz simülatörü",
                              ":material/calendar_month: Temizlik planı", ":material/info: Proje hakkında"])
    with t1:
        tab_photo(cfg)
    with t2:
        tab_dust(cfg)
    with t3:
        tab_plan(cfg)
    with t4:
        tab_about()
    html(f"<div class='footer'>{mark_svg(dark=True, size=26)}<span><b style='color:#fff'>SolarAI</b> · HASAT-13 Proje Pazarı · "
         "Hasan Kalyoncu Üniversitesi, Gaziantep · Veri: Kaggle (Afroz) · Üretim: PVGIS 5.3 (JRC)</span>"
         "<a href='https://github.com/salihdinckann/SolarAI' target='_blank'>Kaynak kod · GitHub</a></div>")


main()
