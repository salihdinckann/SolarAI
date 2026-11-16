"""SolarAI marka öğeleri: B1 logosu (güneş + panel + tarama çerçevesi).

İşaret: yuvarlatılmış tarama köşeleri içinde güneş; güneşin alt yarısı panel hücrelerine dönüşür,
ortadan bir tarama çizgisi geçer. Yazı: Space Grotesk 700, "Solar" turuncu + "AI" lacivert.

  python src/brand.py      -> docs/brand/ altına SVG dosyaları ve app/assets/favicon.png
"""

from __future__ import annotations

import itertools

NAVY, NAVY_DEEP, ORANGE, AMBER, WHITE = "#1E3A8A", "#0F2563", "#F59E0B", "#FBBF24", "#FFFFFF"
FONT = "'Space Grotesk', 'Inter', sans-serif"
_ids = itertools.count()

BRACKETS = ("M8 26 V14 a6 6 0 0 1 6 -6 H26", "M64 8 H76 a6 6 0 0 1 6 6 V26",
            "M82 64 V76 a6 6 0 0 1 -6 6 H64", "M26 82 H14 a6 6 0 0 1 -6 -6 V64")


def _mark(dark: bool, stroke_w: float = 5) -> str:
    """90x90 koordinatlarında işaret (SVG grubu)."""
    cid = f"solarai-sun-{next(_ids)}"
    frame = WHITE if dark else NAVY
    panel = NAVY_DEEP if dark else NAVY
    brackets = "".join(f'<path d="{d}"/>' for d in BRACKETS)
    return (
        f'<defs><clipPath id="{cid}"><circle cx="45" cy="45" r="21"/></clipPath></defs>'
        f'<g fill="none" stroke="{frame}" stroke-width="{stroke_w}" stroke-linecap="round" '
        f'stroke-linejoin="round">{brackets}</g>'
        f'<circle cx="45" cy="45" r="21" fill="{ORANGE}"/>'
        f'<g clip-path="url(#{cid})"><rect x="20" y="47" width="50" height="22" fill="{panel}"/>'
        f'<g stroke="{ORANGE}" stroke-width="1.6"><line x1="20" y1="55" x2="70" y2="55"/>'
        f'<line x1="20" y1="62" x2="70" y2="62"/><line x1="35" y1="47" x2="35" y2="70"/>'
        f'<line x1="45" y1="47" x2="45" y2="70"/><line x1="55" y1="47" x2="55" y2="70"/></g></g>'
        f'<line x1="16" y1="45" x2="74" y2="45" stroke="{AMBER}" stroke-width="2.5" stroke-linecap="round"/>'
    )


def logo_svg(dark: bool = False, height: int = 56, tagline: str | None = None) -> str:
    """Yatay logo: işaret + "SolarAI" yazısı. dark=True koyu zemin içindir."""
    solar, ai = (AMBER, WHITE) if dark else (ORANGE, NAVY)
    w = 280 if not tagline else 330
    h = 90 if not tagline else 100
    tag = (f'<text x="100" y="86" font-family="{FONT}" font-weight="500" font-size="12.5" '
           f'letter-spacing="1.2" fill="{"#CBD5E1" if dark else "#64748B"}">{tagline}</text>') if tagline else ""
    y_text = 58 if not tagline else 56
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" height="{height}" '
        f'role="img" aria-label="SolarAI">{_mark(dark)}'
        f'<text x="98" y="{y_text}" font-family="{FONT}" font-weight="700" font-size="40" '
        f'letter-spacing="-1" fill="{solar}">Solar<tspan fill="{ai}">AI</tspan></text>{tag}</svg>'
    )


def mark_svg(dark: bool = False, size: int = 64) -> str:
    """Yalnızca işaret (kare)."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 90 90" width="{size}" height="{size}" '
            f'role="img" aria-label="SolarAI">{_mark(dark, stroke_w=6)}</svg>')


def mark_png(size: int = 512, dark: bool = False):
    """B1 işaretinin yüksek çözünürlüklü PNG'si (panel çizgileri dahil, şeffaf zemin)."""
    from PIL import Image, ImageDraw

    ss = 4
    S = size * ss
    k = S / 90
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    p = lambda *xy: [v * k for v in xy]  # noqa: E731
    frame = WHITE if dark else NAVY
    w = round(5.5 * k)

    def rounded_corner(x0, y0, dx, dy):
        """(x0,y0) köşesinden dx/dy yönlerine uzanan yuvarlatılmış L."""
        r = 6
        d.line(p(x0, y0 + dy * 18, x0, y0 + dy * r), fill=frame, width=w)
        d.line(p(x0 + dx * r, y0, x0 + dx * 18, y0), fill=frame, width=w)
        box = [min(x0, x0 + 2 * dx * r), min(y0, y0 + 2 * dy * r), max(x0, x0 + 2 * dx * r), max(y0, y0 + 2 * dy * r)]
        start = {(1, 1): 180, (-1, 1): 270, (-1, -1): 0, (1, -1): 90}[(dx, dy)]
        d.arc(p(*box), start, start + 90, fill=frame, width=w)
        for x, y in ((x0, y0 + dy * 18), (x0 + dx * 18, y0)):
            d.ellipse(p(x - 2.75, y - 2.75, x + 2.75, y + 2.75), fill=frame)

    for x0, y0, dx, dy in ((8, 8, 1, 1), (82, 8, -1, 1), (82, 82, -1, -1), (8, 82, 1, -1)):
        rounded_corner(x0, y0, dx, dy)

    sun = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sun)
    sd.ellipse(p(24, 24, 66, 66), fill=ORANGE)
    sd.rectangle(p(20, 47, 70, 70), fill=NAVY_DEEP if dark else NAVY)
    for y in (55, 62):
        sd.line(p(20, y, 70, y), fill=ORANGE, width=round(1.6 * k))
    for x in (35, 45, 55):
        sd.line(p(x, 47, x, 70), fill=ORANGE, width=round(1.6 * k))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse(p(24, 24, 66, 66), fill=255)
    img.paste(sun, (0, 0), mask)
    d.line(p(16, 45, 74, 45), fill=AMBER, width=round(2.5 * k))
    for x in (16, 74):
        d.ellipse(p(x - 1.25, 43.75, x + 1.25, 46.25), fill=AMBER)
    return img.resize((size, size), Image.LANCZOS)


def favicon_png(size: int = 128):
    """Tarayıcı sekmesi ikonu (PIL ile, metin yok). 4 kat büyük çizip küçültür (kenar yumuşatma)."""
    from PIL import Image, ImageDraw

    s = 4 * size / 90
    img = Image.new("RGBA", (4 * size, 4 * size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    p = lambda *xy: [v * s for v in xy]  # noqa: E731
    w = int(8 * s)

    def seg(x1, y1, x2, y2):
        d.line(p(x1, y1, x2, y2), fill=NAVY, width=w)
        for x, y in ((x1, y1), (x2, y2)):
            d.ellipse(p(x - 4, y - 4, x + 4, y + 4), fill=NAVY)

    for (ax, ay), (bx, by), (cx, cy) in [((8, 28), (8, 8), (28, 8)), ((62, 8), (82, 8), (82, 28)),
                                         ((82, 62), (82, 82), (62, 82)), ((28, 82), (8, 82), (8, 62))]:
        seg(ax, ay, bx, by)
        seg(bx, by, cx, cy)
    d.ellipse(p(21, 21, 69, 69), fill=ORANGE)
    half = Image.new("L", img.size, 0)
    ImageDraw.Draw(half).ellipse(p(21, 21, 69, 69), fill=255)
    ImageDraw.Draw(half).rectangle(p(0, 0, 90, 47), fill=0)
    img.paste(Image.new("RGBA", img.size, NAVY), (0, 0), half)
    d.line(p(16, 45, 74, 45), fill=AMBER, width=int(3 * s))
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    from common import ROOT

    out = ROOT / "docs" / "brand"
    out.mkdir(parents=True, exist_ok=True)
    (out / "solarai_logo.svg").write_text(logo_svg(False, 180), encoding="utf-8")
    (out / "solarai_logo_koyu.svg").write_text(logo_svg(True, 180), encoding="utf-8")
    (out / "solarai_isaret.svg").write_text(mark_svg(False, 512), encoding="utf-8")
    assets = ROOT / "app" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    favicon_png(128).save(assets / "favicon.png")
    print("Yazıldı: docs/brand/*.svg, app/assets/favicon.png")
