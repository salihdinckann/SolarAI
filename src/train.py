"""EfficientNet (timm) ile 6 sınıflı panel durumu sınıflandırıcısını eğitir.

Girdi : data/splits.csv  (önce `python src/prepare_data.py` çalıştırılmalı)
Çıktı : outputs/<run>/
          best.pt                 -> en iyi doğrulama macro-F1'ine sahip ağırlıklar
          history.csv             -> her epoch'un kayıp/doğruluk/macro-F1 değerleri
          test_metrics.json       -> test kümesinde macro-F1, accuracy, sınıf bazında P/R/F1
          classification_report.txt
          confusion_matrix.csv / confusion_matrix.png
          test_predictions.csv    -> her test görüntüsü için tahmin ve olasılıklar
          config.json             -> bu koşunun tüm ayarları (tekrarlanabilirlik)

Örnek:
  python src/train.py                               # varsayılan: efficientnet_b0, 25 epoch
  python src/train.py --epochs 1 --limit 64 --run smoke   # hızlı duman testi
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ekran açmadan PNG kaydet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import CLASS_TO_IDX, CLASS_TR, CLASSES, ROOT, build_model, eval_transform, load_rgb, train_transform


# ---------------------------------------------------------------- veri
class PanelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = [ROOT / p for p in df["path"]]
        self.labels = [CLASS_TO_IDX[label] for label in df["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return self.transform(load_rgb(self.paths[i])), self.labels[i]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- eğitim / değerlendirme
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """optimizer verilirse eğitir, verilmezse sadece değerlendirir."""
    training = optimizer is not None
    model.train(training)
    total_loss, y_true, y_prob = 0.0, [], []

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            y_true.append(y.cpu())
            y_prob.append(torch.softmax(logits.float(), dim=1).detach().cpu())

    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()
    y_pred = y_prob.argmax(1)
    return {
        "loss": total_loss / len(y_true),
        "acc": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(len(CLASSES)), zero_division=0)),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def class_weights(labels: list[int], device) -> torch.Tensor:
    # Az örnekli sınıfların hatası daha pahalı olsun: ağırlık ~ 1 / sınıf_sıklığı
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


# ---------------------------------------------------------------- çıktılar
def save_confusion_matrix(cm: np.ndarray, path_png: Path, path_csv: Path) -> None:
    names = [CLASS_TR[c] for c in CLASSES]
    pd.DataFrame(cm, index=names, columns=names).to_csv(path_csv)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Karışıklık matrisi (test)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=7, help="val macro-F1 bu kadar epoch iyileşmezse dur")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="test amaçlı: her bölmeden en fazla N görüntü")
    ap.add_argument("--run", default=None, help="çıktı klasörü adı (varsayılan: model_tarih)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run or f"{args.model}_{time.strftime('%Y%m%d-%H%M')}"
    out_dir = ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cihaz: {device} | Çıktı: {out_dir.relative_to(ROOT)}")

    # --- veri
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    parts = {s: splits[splits["split"] == s] for s in ("train", "val", "test")}
    if args.limit:
        parts = {s: d.sample(min(args.limit, len(d)), random_state=args.seed) for s, d in parts.items()}
    for s, d in parts.items():
        print(f"  {s:5s}: {len(d)} görüntü")

    datasets = {
        "train": PanelDataset(parts["train"], train_transform(args.img_size)),
        "val": PanelDataset(parts["val"], eval_transform(args.img_size)),
        "test": PanelDataset(parts["test"], eval_transform(args.img_size)),
    }
    loaders = {
        s: DataLoader(ds, batch_size=args.batch_size, shuffle=(s == "train"),
                      num_workers=args.num_workers, pin_memory=device.type == "cuda",
                      persistent_workers=args.num_workers > 0)
        for s, ds in datasets.items()
    }

    # --- model, kayıp, optimizer
    model = build_model(args.model, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(datasets["train"].labels, device), label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    config = {**vars(args), "run": run_name, "device": str(device), "classes": CLASSES,
              "n_train": len(datasets["train"]), "n_val": len(datasets["val"]), "n_test": len(datasets["test"])}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- eğitim döngüsü (en iyi val macro-F1'i sakla, iyileşme durursa erken dur)
    history, best_f1, best_epoch, wait = [], -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        va = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"], "train_macro_f1": tr["macro_f1"],
                        "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"],
                        "lr": optimizer.param_groups[0]["lr"]})
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        improved = va["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, wait = va["macro_f1"], epoch, 0
            torch.save({"model": args.model, "img_size": args.img_size, "classes": CLASSES,
                        "state_dict": model.state_dict(), "epoch": epoch, "val_macro_f1": best_f1},
                       out_dir / "best.pt")
        else:
            wait += 1
        print(f"[{epoch:02d}/{args.epochs}] train loss {tr['loss']:.3f} F1 {tr['macro_f1']:.3f} | "
              f"val loss {va['loss']:.3f} acc {va['acc']:.3f} F1 {va['macro_f1']:.3f}"
              f"{'  *' if improved else ''}  ({time.time() - t0:.0f} sn)")
        if wait >= args.patience:
            print(f"Erken durdurma: {args.patience} epoch boyunca iyileşme yok.")
            break

    # --- test: sadece EN İYİ model ile ve sadece BİR kez
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    te = run_epoch(model, loaders["test"], criterion, device)
    y_true, y_pred = te["y_true"], te["y_prob"].argmax(1)

    names = [CLASS_TR[c] for c in CLASSES]
    labels = list(range(len(CLASSES)))
    report_txt = classification_report(y_true, y_pred, labels=labels, target_names=names, digits=3, zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES, output_dict=True, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    metrics = {
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "test_macro_f1": te["macro_f1"],
        "test_accuracy": te["acc"],
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in CLASSES},
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    save_confusion_matrix(confusion_matrix(y_true, y_pred, labels=labels),
                          out_dir / "confusion_matrix.png", out_dir / "confusion_matrix.csv")

    preds = parts["test"][["path", "label"]].reset_index(drop=True).copy()
    preds["pred"] = [CLASSES[i] for i in y_pred]
    for i, c in enumerate(CLASSES):
        preds[f"p_{c}"] = te["y_prob"][:, i].round(4)
    preds.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"\nEn iyi epoch: {best_epoch} (val macro-F1 {best_f1:.3f})")
    print(f"TEST  macro-F1: {te['macro_f1']:.3f} | accuracy: {te['acc']:.3f}\n")
    print(report_txt)
    print(f"Çıktılar: {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
