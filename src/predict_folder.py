"""Herhangi bir klasördeki fotoğraflar için tahmin + Grad-CAM ızgarası üretir.

Veri setinde OLMAYAN fotoğraflarla (internetten, sahadan, telefondan) modeli denemek için:
  1. Fotoğrafları data/external/ klasörüne koyun (jpg/png).
  2. python src/predict_folder.py data/external
  3. Sonuç: outputs/external_gradcam.png ve outputs/external_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLASS_TR, CLASSES, ROOT, load_rgb
from inference import Predictor, overlay

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--ckpt", default="outputs/effnet_b0_v2/best.pt")
    ap.add_argument("--out", default="outputs/external")
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = ROOT / folder
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(f"{folder} içinde görüntü yok.")

    pred = Predictor(ROOT / args.ckpt)
    rows = []
    n_rows = int(np.ceil(len(paths) / args.cols))
    fig, axes = plt.subplots(2 * n_rows, args.cols, figsize=(3 * args.cols, 6.4 * n_rows))
    axes = np.atleast_2d(axes)
    for k, path in enumerate(paths):
        img = load_rgb(path)
        view = pred.preprocess_view(img)
        probs = pred.predict(img)
        top = max(probs, key=probs.get)
        second = sorted(probs, key=probs.get)[-2]
        rows.append({"file": path.name, "pred": top, **{f"p_{c}": round(probs[c], 4) for c in CLASSES}})

        i, j = divmod(k, args.cols)
        axes[2 * i, j].imshow(view)
        axes[2 * i, j].set_title(path.name[:28], fontsize=8)
        axes[2 * i + 1, j].imshow(overlay(view, pred.gradcam(img, top)))
        axes[2 * i + 1, j].set_title(
            f"{CLASS_TR[top]} %{probs[top] * 100:.0f}\n(2.: {CLASS_TR[second]} %{probs[second] * 100:.0f})", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}_gradcam.png", dpi=90)
    pd.DataFrame(rows).to_csv(f"{out}_predictions.csv", index=False)
    print(pd.DataFrame(rows)[["file", "pred"]].to_string(index=False))
    print(f"\nYazıldı: {args.out}_gradcam.png, {args.out}_predictions.csv")


if __name__ == "__main__":
    main()
