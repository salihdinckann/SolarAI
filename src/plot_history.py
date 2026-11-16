"""history.csv'den eğitim eğrilerini (kayıp ve macro-F1) çizer.

  python src/plot_history.py results/effnet_b0_v1
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

run_dir = Path(sys.argv[1])
h = pd.read_csv(run_dir / "history.csv")
best = h.loc[h["val_macro_f1"].idxmax(), "epoch"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, key, title in [(axes[0], "loss", "Kayıp (loss)"), (axes[1], "macro_f1", "Macro-F1")]:
    ax.plot(h["epoch"], h[f"train_{key}"], marker="o", ms=3, label="Eğitim")
    ax.plot(h["epoch"], h[f"val_{key}"], marker="o", ms=3, label="Doğrulama")
    ax.axvline(best, color="gray", ls="--", lw=1, label=f"En iyi epoch ({best})")
    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(run_dir / "training_curves.png", dpi=200)
print(f"Yazıldı: {run_dir / 'training_curves.png'}")
