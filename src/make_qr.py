"""Poster için marka uyumlu QR kartı üretir (koyu + açık sürüm, 300 dpi PNG).

  pip install "qrcode[pil]"
  python src/make_qr.py https://solarai-gaziantep.streamlit.app

Tasarım: yuvarlatılmış modüller, lacivert çerçeveli turuncu "gözler", ortada B1 işareti.
Hata düzeltme seviyesi H (%30): ortadaki logo ve olası kir/katlanmaya rağmen okunur.
Çıktılar docs/poster/ altına yazılır ve OpenCV ile çeşitli boyutlarda okunarak doğrulanır.
"""

from __future__ import annotations

import argparse

import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageFont

from brand import AMBER, NAVY, ORANGE, mark_png
from common import ROOT

BG_DARK, CARD_LINE = (8, 17, 41), (40, 56, 100)
FONT = "bahnschrift.ttf"  # Windows'ta yüklü geometrik yazı tipi (Space Grotesk'e yakın)


def font(size: int, weight: str = "Bold") -> ImageFont.FreeTypeFont:
    try:
        f = ImageFont.truetype(FONT, size)
        f.set_variation_by_name(weight)
        return f
    except OSError:
        return ImageFont.truetype("arialbd.ttf" if weight == "Bold" else "arial.ttf", size)


def qr_matrix(url: str) -> np.ndarray:
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, border=0)
    qr.add_data(url)
    qr.make(fit=True)
    return np.array(qr.get_matrix(), dtype=bool)


def draw_qr(m: np.ndarray, px: int, logo_frac: float = 0.22) -> Image.Image:
    """Beyaz zeminli, yuvarlatılmış modüllü QR (sessiz bölge dahil)."""
    n = m.shape[0]
    quiet = 4
    cell = px // (n + 2 * quiet)
    size = cell * (n + 2 * quiet)
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    off = quiet * cell

    def in_finder(r, c):
        return (r < 7 and c < 7) or (r < 7 and c >= n - 7) or (r >= n - 7 and c < 7)

    # logo için boşaltılacak merkez (modül cinsinden, tek sayıya yuvarlanır)
    lm = int(n * logo_frac) | 1
    l0 = (n - lm) // 2
    for r in range(n):
        for c in range(n):
            if not m[r, c] or in_finder(r, c) or (l0 <= r < l0 + lm and l0 <= c < l0 + lm):
                continue
            x, y = off + c * cell, off + r * cell
            # Hafif yuvarlak köşeli kare modül. (Nokta modüller de okunuyor ama telefonlarda kare daha güvenli.)
            d.rounded_rectangle([x, y, x + cell, y + cell], radius=cell * 0.25, fill=NAVY)

    # Konum "gözleri": yalnızca HAFİF yuvarlatma. Testte köşe yarıçapı ~2 modül olunca QR hiç okunmadı;
    # 0.6 modülde her boyutta okunuyor. İç kare koyu kalmalı (turuncu, eşiklemede beyaz sayılabilir).
    for r0, c0 in ((0, 0), (0, n - 7), (n - 7, 0)):
        x, y = off + c0 * cell, off + r0 * cell
        d.rounded_rectangle([x, y, x + 7 * cell, y + 7 * cell], radius=cell * 0.6, fill=NAVY)
        d.rounded_rectangle([x + cell, y + cell, x + 6 * cell, y + 6 * cell], radius=cell * 0.42, fill="white")
        d.rounded_rectangle([x + 2 * cell, y + 2 * cell, x + 5 * cell, y + 5 * cell], radius=cell * 0.3, fill=NAVY)

    # merkez logo: beyaz yuvarlatılmış kare + B1 işareti
    lx = off + l0 * cell
    lw = lm * cell
    d.rounded_rectangle([lx, lx, lx + lw, lx + lw], radius=lw * 0.22, fill="white")
    mk = mark_png(int(lw * 0.86), dark=False)
    img.paste(mk, (lx + (lw - mk.width) // 2, lx + (lw - mk.height) // 2), mk)
    return img


def radial_glow(w: int, h: int) -> Image.Image:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), np.float32) + BG_DARK
    for (cx, cy, rad, col, a) in ((0.92 * w, -0.05 * h, 0.75 * w, (245, 158, 11), 0.22),
                                  (0.0, 0.25 * h, 0.8 * w, (37, 99, 235), 0.22)):
        t = np.clip(1 - np.hypot(x - cx, y - cy) / rad, 0, 1) ** 2 * a
        base = base * (1 - t[..., None]) + np.array(col, np.float32) * t[..., None]
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def text_center(d: ImageDraw.ImageDraw, y: int, w: int, parts: list[tuple[str, ImageFont.FreeTypeFont, str]]) -> None:
    total = sum(d.textlength(t, font=f) for t, f, _ in parts)
    x = (w - total) / 2
    for t, f, col in parts:
        d.text((x, y), t, font=f, fill=col)
        x += d.textlength(t, font=f)


def card(url: str, dark: bool, W: int = 1600, H: int = 2150) -> Image.Image:
    bg = radial_glow(W, H) if dark else Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(bg)
    fg, muted = ("white", (148, 163, 184)) if dark else (NAVY, (100, 116, 139))
    if not dark:
        d.rounded_rectangle([20, 20, W - 20, H - 20], radius=90, outline=(226, 232, 240), width=6)

    # üst: logo
    mk = mark_png(190, dark=dark)
    f_logo = font(150)
    solar_w, ai_w = d.textlength("Solar", font=f_logo), d.textlength("AI", font=f_logo)
    total = mk.width + 40 + solar_w + ai_w
    x0 = int((W - total) / 2)
    bg.paste(mk, (x0, 120), mk)
    tx = x0 + mk.width + 40
    d.text((tx, 140), "Solar", font=f_logo, fill=AMBER if dark else ORANGE)
    d.text((tx + solar_w, 140), "AI", font=f_logo, fill=fg)
    text_center(d, 345, W, [("Panelin ne kadar kaybettirdiğini öğrenin", font(52, "SemiBold"), muted)])

    # orta: QR (beyaz yuvarlatılmış kutuda)
    qr = draw_qr(qr_matrix(url), 1160)
    box = 1240
    bx, by = (W - box) // 2, 470
    d.rounded_rectangle([bx, by, bx + box, by + box], radius=80, fill="white",
                        outline=CARD_LINE if dark else (226, 232, 240), width=4)
    bg.paste(qr, (bx + (box - qr.width) // 2, by + (box - qr.height) // 2))

    # alt: çağrı + adres
    y = by + box + 80
    text_center(d, y, W, [("Telefonunuzla okutun, ", font(78), fg), ("deneyin", font(78), AMBER if dark else ORANGE)])
    text_center(d, y + 120, W, [(url.replace("https://", ""), font(50, "Regular"), muted)])
    text_center(d, y + 200, W, [("HASAT-13 · Hasan Kalyoncu Üniversitesi", font(44, "SemiBold"), muted)])
    return bg


def verify(path, url: str) -> list[tuple[int, bool]]:
    import cv2

    img = cv2.imread(str(path))
    det, out = cv2.QRCodeDetector(), []
    for w in (img.shape[1], 900, 600, 400, 300):
        im = cv2.resize(img, (w, int(w * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        txt, _, _ = det.detectAndDecode(im)
        out.append((w, txt == url))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="docs/poster/qr_solarai")
    args = ap.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for dark, suffix in ((True, "_koyu"), (False, "_acik")):
        path = f"{out}{suffix}.png"
        card(args.url, dark).save(path, dpi=(300, 300))
        draw_qr(qr_matrix(args.url), 1200).save(f"{out}{suffix.replace('_koyu', '_yalniz').replace('_acik', '_yalniz')}.png",
                                                dpi=(300, 300))
        print(f"{path}: okuma testi (genişlik px, okundu mu) -> {verify(path, args.url)}")


if __name__ == "__main__":
    main()
