"""Poster için uygulama QR kodu üretir (PNG + baskı için vektörel SVG).

  pip install "qrcode[pil]"
  python src/make_qr.py https://solarai-gaziantep.streamlit.app

Hata düzeltme seviyesi H (%30): QR'ın bir kısmı kirlense/katlansa da okunur.
"""

from __future__ import annotations

import argparse

import qrcode
import qrcode.image.svg
from PIL import Image, ImageDraw, ImageFont

from common import ROOT

NAVY, ORANGE = "#1E3A8A", "#F59E0B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="docs/poster/qr_solarai")
    args = ap.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=40, border=4)
    qr.add_data(args.url)
    qr.make(fit=True)

    # Vektörel SVG (posterde her boyutta keskin)
    qr.make_image(image_factory=qrcode.image.svg.SvgPathFillImage).save(f"{out}.svg")

    # PNG: lacivert QR + altında etiket
    img = qr.make_image(fill_color=NAVY, back_color="white").convert("RGB")
    w, h = img.size
    canvas = Image.new("RGB", (w, h + 260), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        big = ImageFont.truetype("arialbd.ttf", 110)
        small = ImageFont.truetype("arial.ttf", 52)
    except OSError:
        big = small = ImageFont.load_default()
    label_solar, label_ai = "Solar", "AI"
    sw = draw.textlength(label_solar, font=big)
    aw = draw.textlength(label_ai, font=big)
    x0 = (w - sw - aw) / 2
    draw.text((x0, h - 20), label_solar, fill=ORANGE, font=big)
    draw.text((x0 + sw, h - 20), label_ai, fill=NAVY, font=big)
    sub = "Telefonunuzla okutun ve deneyin"
    draw.text(((w - draw.textlength(sub, font=small)) / 2, h + 120), sub, fill="#334155", font=small)
    canvas.save(f"{out}.png", dpi=(300, 300))
    print(f"QR versiyonu {qr.version}, {qr.modules_count}x{qr.modules_count} modül")
    print(f"Yazıldı: {out.relative_to(ROOT)}.png, {out.relative_to(ROOT)}.svg")


if __name__ == "__main__":
    main()
