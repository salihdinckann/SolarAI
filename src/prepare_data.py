"""Ham Kaggle görüntülerini temizler ve stratified train/val/test bölmesi üretir.

Adımlar:
  1. data/raw/Faulty_solar_panel altındaki görüntüleri bulur (desktop.ini vb. atlanır).
  2. Bozuk/açılamayan görüntüleri eler.
  3. MD5 özetiyle birebir kopyaları bulur; her içerikten yalnızca bir tane tutar.
  4. Aynı içerik farklı sınıflarda geçiyorsa (çelişkili etiket) o görüntüyü tamamen çıkarır.
  5. Algısal özet (dHash) ile YAKIN kopyaları bulur (yeniden boyutlanmış / yeniden sıkıştırılmış
     aynı fotoğraf) ve bunları bir GRUP yapar. Yakın kopyası farklı sınıfta olan grup çıkarılır.
  6. Grupları bölmez: bir gruptaki tüm görüntüler aynı bölmeye düşer (eğitim-test sızıntısı olmaz).
     Sınıf oranları korunarak ~%70 / %15 / %15 bölünür (seed sabit, tekrarlanabilir).

Çıktılar:
  data/splits.csv          -> path, label, md5, group, split
  data/data_report.csv     -> çıkarılan dosyalar ve nedenleri

Çalıştırma (proje kök klasöründen):
  python src/prepare_data.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "Faulty_solar_panel"
OUT_SPLITS = ROOT / "data" / "splits.csv"
OUT_REPORT = ROOT / "data" / "data_report.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Kaggle klasör adı -> projede kullandığımız sınıf adı
CLASS_MAP = {
    "Clean": "clean",
    "Dusty": "dusty",
    "Bird-drop": "bird_drop",
    "Snow-Covered": "snow_covered",
    "Physical-Damage": "physical_damage",
    "Electrical-damage": "electrical_damage",
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def scan(raw_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows, dropped = [], []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        label = CLASS_MAP.get(class_dir.name)
        if label is None:
            raise ValueError(f"Bilinmeyen sınıf klasörü: {class_dir.name}")
        # rglob: Bird-drop/New gibi alt klasörleri de kapsar
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix.lower() not in IMAGE_EXTS:
                dropped.append({"path": rel, "label": label, "reason": "goruntu_degil"})
                continue
            if not is_readable_image(path):
                dropped.append({"path": rel, "label": label, "reason": "bozuk_goruntu"})
                continue
            rows.append({"path": rel, "label": label, "md5": md5_of(path)})
    return pd.DataFrame(rows), dropped


def deduplicate(df: pd.DataFrame, dropped: list[dict]) -> pd.DataFrame:
    # Aynı içerik birden fazla sınıfta mı? -> çelişkili etiket, hepsini çıkar
    labels_per_hash = df.groupby("md5")["label"].nunique()
    conflict_hashes = set(labels_per_hash[labels_per_hash > 1].index)
    for r in df[df["md5"].isin(conflict_hashes)].itertuples():
        dropped.append({"path": r.path, "label": r.label, "reason": "celiskili_etiket"})
    df = df[~df["md5"].isin(conflict_hashes)]

    # Aynı sınıf içindeki kopyalar -> ilkini tut
    dup_mask = df.duplicated("md5", keep="first")
    for r in df[dup_mask].itertuples():
        dropped.append({"path": r.path, "label": r.label, "reason": "kopya"})
    return df[~dup_mask].reset_index(drop=True)


def dhash(path: Path, size: int = 16) -> np.ndarray:
    """Fark özeti (difference hash): görüntü küçültülüp gri yapılır, komşu pikseller
    karşılaştırılır. Boyut/sıkıştırma değişse de aynı fotoğraf ~aynı bitleri verir."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def imagenet_embeddings(paths: list[str], hflip: bool = False) -> np.ndarray:
    """Bizim veriyle HİÇ eğitilmemiş, ImageNet ağırlıklı EfficientNet'ten 1280 boyutlu,
    birim uzunluklu öznitelik vektörü. Kırpma/boyut farkına dHash'ten çok daha dayanıklı."""
    import timm
    import torch

    from common import eval_transform, load_rgb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).to(device).eval()
    tf = eval_transform(224)
    out = []
    with torch.no_grad():
        for k in range(0, len(paths), 64):
            x = torch.stack([tf(load_rgb(ROOT / p)) for p in paths[k:k + 64]]).to(device)
            if hflip:
                x = torch.flip(x, dims=[3])
            out.append(torch.nn.functional.normalize(model(x), dim=1).cpu())
    return torch.cat(out).numpy()


def near_duplicate_groups(df: pd.DataFrame, max_dist: int, min_sim: float) -> np.ndarray:
    """İki görüntü şu durumda aynı gruba girer (birleşim-bul / union-find ile, zincirleme):
       - dHash Hamming mesafesi <= max_dist   (yeniden boyutlanmış / sıkıştırılmış kopya)
       - VEYA ImageNet öznitelik kosinüs benzerliği >= min_sim (kırpılmış kopya, aynı çatının
         art arda çekilmiş kareleri)"""
    hashes = np.stack([dhash(ROOT / p) for p in df["path"]])
    dist = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
    emb = imagenet_embeddings(list(df["path"]))
    sim = emb @ emb.T
    linked = (dist <= max_dist) | (sim >= min_sim)
    parent = list(range(len(df)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(linked, 1))):
        parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(df))]
    return pd.factorize(pd.Series(roots))[0]


def drop_conflicting_groups(df: pd.DataFrame, dropped: list[dict]) -> pd.DataFrame:
    labels_per_group = df.groupby("group")["label"].nunique()
    bad = set(labels_per_group[labels_per_group > 1].index)
    for r in df[df["group"].isin(bad)].itertuples():
        dropped.append({"path": r.path, "label": r.label, "reason": "yakin_kopya_celiskili_etiket"})
    return df[~df["group"].isin(bad)].reset_index(drop=True)


def split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    """Grup korumalı, sınıf oranlı bölme: önce test, sonra kalan içinden val ayrılır."""
    def carve(d: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        sgkf = StratifiedGroupKFold(n_splits=round(1 / frac), shuffle=True, random_state=seed)
        rest_idx, part_idx = next(sgkf.split(d, d["label"], d["group"]))
        return d.iloc[rest_idx], d.iloc[part_idx]

    rest, test_df = carve(df, test_frac)
    train_df, val_df = carve(rest, val_frac / (1 - test_frac))
    return pd.concat(
        [train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")]
    ).sort_values(["split", "label", "path"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near-dup-dist", type=int, default=12,
                    help="dHash Hamming mesafesi (0-256) bu değere eşit/küçükse yakın kopya sayılır")
    ap.add_argument("--near-dup-sim", type=float, default=0.88,
                    help="ImageNet öznitelik kosinüs benzerliği bu değere eşit/büyükse yakın kopya sayılır")
    args = ap.parse_args()

    if not RAW_DIR.exists():
        raise SystemExit(f"{RAW_DIR} bulunamadı. Kaggle verisini data/raw/ içine açın (README'ye bakın).")

    df, dropped = scan(RAW_DIR)
    print(f"Okunan görüntü: {len(df)}")
    df = deduplicate(df, dropped)
    df["group"] = near_duplicate_groups(df, args.near_dup_dist, args.near_dup_sim)
    df = drop_conflicting_groups(df, dropped)
    sizes = df["group"].value_counts()
    print(f"Yakın kopya grubu (2+ görüntü): {(sizes > 1).sum()}, toplam grup: {len(sizes)}, "
          f"en büyük grup: {sizes.max()} görüntü")
    out = split(df, args.val, args.test, args.seed)
    assert out.groupby("group")["split"].nunique().max() == 1, "Bir grup birden fazla bölmeye düştü!"

    OUT_SPLITS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_SPLITS, index=False)
    pd.DataFrame(dropped, columns=["path", "label", "reason"]).to_csv(OUT_REPORT, index=False)

    print(f"Çıkarılan dosya: {len(dropped)}")
    print(pd.Series([d["reason"] for d in dropped]).value_counts().to_string())
    print(f"\nTemiz görüntü: {len(out)}")
    print(pd.crosstab(out["label"], out["split"], margins=True, margins_name="toplam").to_string())
    print(f"\nYazıldı: {OUT_SPLITS.relative_to(ROOT)}, {OUT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
