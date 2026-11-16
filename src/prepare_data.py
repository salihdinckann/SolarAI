"""Ham Kaggle görüntülerini temizler ve stratified train/val/test bölmesi üretir.

Adımlar:
  1. data/raw/Faulty_solar_panel altındaki görüntüleri bulur (desktop.ini vb. atlanır).
  2. Bozuk/açılamayan görüntüleri eler.
  3. MD5 özetiyle birebir kopyaları bulur; her içerikten yalnızca bir tane tutar.
  4. Aynı içerik farklı sınıflarda geçiyorsa (çelişkili etiket) o görüntüyü tamamen çıkarır.
  5. Sınıf oranlarını koruyarak %70 / %15 / %15 böler (seed sabit, tekrarlanabilir).

Çıktılar:
  data/splits.csv          -> path, label, md5, split
  data/data_report.csv     -> çıkarılan dosyalar ve nedenleri

Çalıştırma (proje kök klasöründen):
  python src/prepare_data.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

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


def split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        df, test_size=val_frac + test_frac, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_frac / (val_frac + test_frac),
        stratify=temp_df["label"],
        random_state=seed,
    )
    return pd.concat(
        [train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")]
    ).sort_values(["split", "label", "path"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not RAW_DIR.exists():
        raise SystemExit(f"{RAW_DIR} bulunamadı. Kaggle verisini data/raw/ içine açın (README'ye bakın).")

    df, dropped = scan(RAW_DIR)
    print(f"Okunan görüntü: {len(df)}")
    df = deduplicate(df, dropped)
    out = split(df, args.val, args.test, args.seed)

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
