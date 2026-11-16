"""Test kümesinden her sınıf için örnek seçip Grad-CAM ızgarası çizer (poster görseli).

  python src/gradcam_examples.py --ckpt outputs/effnet_b0_v2/best.pt --out results/effnet_b0_v2/gradcam_examples.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import CLASS_TR, CLASSES, ROOT, load_rgb
from inference import Predictor, overlay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/effnet_b0_v2/best.pt")
    ap.add_argument("--preds", default="results/effnet_b0_v2/test_predictions.csv")
    ap.add_argument("--out", default="results/effnet_b0_v2/gradcam_examples.png")
    args = ap.parse_args()

    pred = Predictor(ROOT / args.ckpt)
    df = pd.read_csv(ROOT / args.preds)

    # Her sınıftan, modelin DOĞRU ve en emin olduğu test görüntüsü
    picks = []
    for c in CLASSES:
        ok = df[(df["label"] == c) & (df["pred"] == c)]
        picks.append(ok.loc[ok[f"p_{c}"].idxmax()] if len(ok) else df[df["label"] == c].iloc[0])

    fig, axes = plt.subplots(2, len(CLASSES), figsize=(3 * len(CLASSES), 7.2),
                             gridspec_kw={"hspace": 0.25})
    for j, row in enumerate(picks):
        img = load_rgb(ROOT / row["path"])
        view = pred.preprocess_view(img)
        heat = pred.gradcam(img, row["pred"])
        p = pred.predict(img)[row["pred"]]
        axes[0, j].imshow(view)
        axes[0, j].set_title(f"Gerçek: {CLASS_TR[row['label']]}", fontsize=11)
        axes[1, j].imshow(overlay(view, heat))
        axes[1, j].set_title(f"Tahmin: {CLASS_TR[row['pred']]} (%{p * 100:.0f})", fontsize=11)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle("Grad-CAM: model kararını görüntünün neresine dayandırıyor?", fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.05, hspace=0.25)
    out = ROOT / args.out
    fig.savefig(out, dpi=200)
    print(f"Yazıldı: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
