"""Lenarczyk "PV Panel Defect Dataset" içinden, bizim veri setimizde (Afroz) HİÇBİR benzeri
olmayan görüntüleri seçip bir DIŞ TEST kümesi oluşturur.

Bir Lenarczyk görüntüsü şu durumlardan birinde ELENİR (bizim 885 ham görüntünün hepsine karşı):
  - MD5 aynı (birebir kopya)
  - dHash Hamming mesafesi <= --max-dist  (bizim görüntünün 8 ayna/döndürme varyantından herhangi biriyle)
  - ImageNet öznitelik kosinüs benzerliği >= --min-sim (bizim görüntünün kendisi veya yatay aynası ile)
Kalan yeni görüntüler kendi aralarında da gruplanır; her gruptan yalnızca bir tane tutulur
(aynı fotoğrafın çoğaltılmış kopyaları sonucu şişirmesin).

Kullanım:
  python src/build_external_test.py
Çıktı:
  data/external_test.csv   -> path, label  (src/evaluate.py ile değerlendirilir)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from prepare_data import CLASS_MAP, IMAGE_EXTS, ROOT, imagenet_embeddings, is_readable_image, md5_of

LEN_DIR = ROOT / "data" / "lenarczyk"
OUR_DIR = ROOT / "data" / "raw" / "Faulty_solar_panel"
OUT = ROOT / "data" / "external_test.csv"


def dhash_img(im: Image.Image, size: int = 16) -> np.ndarray:
    a = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def dihedral(im: Image.Image) -> list[Image.Image]:
    """Görüntünün 8 ayna/döndürme varyantı."""
    out = []
    for base in (im, ImageOps.mirror(im)):
        for k in range(4):
            out.append(base.rotate(90 * k, expand=True))
    return out


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_of(path: Path, root: Path) -> str:
    for part in path.relative_to(root).parts:
        if part in CLASS_MAP:
            return CLASS_MAP[part]
    raise ValueError(f"Sınıf bulunamadı: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist", type=int, default=30)
    ap.add_argument("--min-sim", type=float, default=0.88)
    args = ap.parse_args()

    ours = [p for p in list_images(OUR_DIR) if is_readable_image(p)]
    theirs = [p for p in list_images(LEN_DIR) if is_readable_image(p)]
    print(f"Bizim ham görüntü: {len(ours)} | Lenarczyk: {len(theirs)}")

    # 1) MD5
    our_md5 = {md5_of(p) for p in ours}
    df = pd.DataFrame({"path": [p.relative_to(ROOT).as_posix() for p in theirs],
                       "label": [label_of(p, LEN_DIR) for p in theirs],
                       "md5": [md5_of(p) for p in theirs]})
    df["neden"] = np.where(df["md5"].isin(our_md5), "md5", "")
    df = df.drop_duplicates("md5").reset_index(drop=True)  # Lenarczyk içi birebir kopyalar

    # 2) dHash (bizimkilerin 8 varyantına karşı)
    our_h = []
    for p in ours:
        with Image.open(p) as im:
            our_h += [dhash_img(v) for v in dihedral(im.convert("RGB"))]
    our_h = np.stack(our_h)
    their_h = np.stack([dhash_img(Image.open(ROOT / p)) for p in df["path"]])
    min_dist = np.array([(our_h != h).sum(1).min() for h in their_h])
    df.loc[(df["neden"] == "") & (min_dist <= args.max_dist), "neden"] = "dhash"

    # 3) ImageNet öznitelik benzerliği (bizimkiler + yatay aynaları)
    our_paths = [p.relative_to(ROOT).as_posix() for p in ours]
    our_e = imagenet_embeddings(our_paths)
    our_e_flip = imagenet_embeddings(our_paths, hflip=True)
    their_e = imagenet_embeddings(list(df["path"]))
    max_sim = np.maximum(their_e @ our_e.T, their_e @ our_e_flip.T).max(1)
    df.loc[(df["neden"] == "") & (max_sim >= args.min_sim), "neden"] = "benzerlik"
    df["min_dhash"], df["max_sim"] = min_dist, max_sim.round(3)

    print("\nLenarczyk görüntülerinin bizim veriyle ilişkisi:")
    print(df["neden"].replace("", "YENİ").value_counts().to_string())

    # 4) Yeni olanları kendi içinde grupla, her gruptan birini tut
    new = df[df["neden"] == ""].reset_index(drop=True)
    e = their_e[df.index[df["neden"] == ""]]
    h = their_h[df.index[df["neden"] == ""]]
    linked = ((h[:, None] != h[None]).sum(-1) <= args.max_dist) | (e @ e.T >= args.min_sim)
    parent = list(range(len(new)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(linked, 1))):
        parent[find(i)] = find(j)
    new["group"] = [find(i) for i in range(len(new))]
    conflict = new.groupby("group")["label"].nunique()
    new = new[new["group"].isin(conflict[conflict == 1].index)]
    ext = new.drop_duplicates("group").sort_values(["label", "path"])

    ext[["path", "label"]].to_csv(OUT, index=False)
    print(f"\nDış test kümesi: {len(ext)} görüntü (yeni {len(new)} görüntüden, gruplar tekilleştirildi)")
    print(ext["label"].value_counts().to_string())
    print(f"Yazıldı: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
