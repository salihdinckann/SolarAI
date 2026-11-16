"""Temiz bir panel fotoğrafına sentetik toz tabakası bindirir (Toz Simülatörü için).

Toz; farklı ölçeklerde düzgünleştirilmiş gürültüden oluşan düzensiz bir örtü, ince tanecik
benekleri ve hafif kontrast/doygunluk kaybıyla taklit edilir. Aynı `seed` her zaman aynı dokuyu verir.
Gerçek toz fiziğinin modeli değildir; amacı modelin tepkisini görselleştirmektir.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

DUST_COLOR = np.array([188, 172, 146], dtype=np.float32)  # açık kahve-gri


def _smooth_noise(h: int, w: int, cell: int, rng: np.random.Generator) -> np.ndarray:
    """cell piksel ölçeğinde düzgün gürültü (0-1)."""
    small = rng.random((max(2, h // cell + 2), max(2, w // cell + 2))).astype(np.float32)
    im = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255


def dust_mask(h: int, w: int, seed: int = 0) -> np.ndarray:
    """0-1 arası toz yoğunluk haritası (düzensiz örtü + tanecikler)."""
    rng = np.random.default_rng(seed)
    s = max(h, w)
    cover = (0.5 * _smooth_noise(h, w, s // 4, rng) + 0.3 * _smooth_noise(h, w, s // 16, rng)
             + 0.2 * _smooth_noise(h, w, s // 64, rng))
    cover = (cover - cover.min()) / (np.ptp(cover) + 1e-6)
    grains = (rng.random((h, w)) > 0.985).astype(np.float32)
    grains = np.asarray(Image.fromarray((grains * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)),
                        dtype=np.float32) / 255
    return np.clip(0.75 * cover + 0.25 + 0.6 * grains, 0, 1)


def add_dust(image: Image.Image, level: float, seed: int = 0) -> Image.Image:
    """level: 0 (temiz) – 1 (çok tozlu)."""
    if level <= 0:
        return image.convert("RGB")
    img = image.convert("RGB")
    # Toz önce renkleri soldurur ve kontrastı düşürür
    img = ImageEnhance.Color(img).enhance(1 - 0.6 * level)
    img = ImageEnhance.Contrast(img).enhance(1 - 0.45 * level)
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    alpha = (0.85 * level * dust_mask(h, w, seed))[..., None]
    out = (1 - alpha) * arr + alpha * DUST_COLOR
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
