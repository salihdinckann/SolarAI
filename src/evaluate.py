"""Eğitilmiş bir modeli herhangi bir etiketli CSV (path, label) üzerinde değerlendirir.

  python src/evaluate.py --csv data/external_test.csv --ckpt outputs/effnet_b0_v2/best.pt --out results/effnet_b0_v2/external
Çıktı: <out>_metrics.json, <out>_predictions.csv, <out>_gradcam.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import CLASS_TR, CLASSES, ROOT, load_rgb
from inference import Predictor, overlay

DAMAGE = {"physical_damage", "electrical_damage"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/external_test.csv")
    ap.add_argument("--ckpt", default="outputs/effnet_b0_v2/best.pt")
    ap.add_argument("--out", default="results/effnet_b0_v2/external")
    ap.add_argument("--cols", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.csv)
    pred = Predictor(ROOT / args.ckpt)
    probs = [pred.predict(load_rgb(ROOT / p)) for p in df["path"]]
    df["pred"] = [max(p, key=p.get) for p in probs]
    for c in CLASSES:
        df[f"p_{c}"] = [round(p[c], 4) for p in probs]

    # Sınıf bazında recall (dış kümede her sınıf bulunmayabilir; precision bu yüzden yanıltıcı olur)
    per_class = {}
    for c in CLASSES:
        d = df[df["label"] == c]
        if len(d):
            per_class[c] = {"n": int(len(d)), "recall": round(float((d["pred"] == c).mean()), 3),
                            "tahminler": d["pred"].value_counts().to_dict()}
    dmg = df[df["label"].isin(DAMAGE)]
    metrics = {
        "n": int(len(df)),
        "accuracy": round(float((df["label"] == df["pred"]).mean()), 3),
        "per_class": per_class,
        "hasar_servis_uyarisi": f"{int(dmg['pred'].isin(DAMAGE).sum())}/{len(dmg)}",
        "hasar_temiz_denen": int((dmg["pred"] == "clean").sum()),
        "confusion_matrix": {"labels": CLASSES,
                             "matrix": confusion_matrix(df["label"], df["pred"], labels=CLASSES).tolist()},
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / f"{out.name}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(out.parent / f"{out.name}_predictions.csv", index=False)

    # Grad-CAM ızgarası (yanlışlar kırmızı başlıklı)
    n_rows = int(np.ceil(len(df) / args.cols))
    fig, axes = plt.subplots(n_rows, args.cols, figsize=(2.6 * args.cols, 3.0 * n_rows))
    for ax, (_, r) in zip(np.atleast_2d(axes).flat, df.iterrows()):
        img = load_rgb(ROOT / r["path"])
        ax.imshow(overlay(pred.preprocess_view(img), pred.gradcam(img, r["pred"])))
        ok = r["label"] == r["pred"]
        ax.set_title(f"{CLASS_TR[r['label']]}\n→ {CLASS_TR[r['pred']]} %{r[f'p_{r.pred}'] * 100:.0f}",
                     fontsize=8, color="green" if ok else "red")
    for ax in np.atleast_2d(axes).flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out.parent / f"{out.name}_gradcam.png", dpi=70)

    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
