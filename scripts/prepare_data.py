#!/usr/bin/env python3
"""Arrange the downloaded dataset zips into the data/ layout this repository reads.

Download solar_coronal_hole_seg.zip, solar_sunspot_seg.zip and solar_prominence_seg.zip (see README 2.4) into one
folder, then:

    python scripts/prepare_data.py --src <folder with the zips> --out data

Result:  data/images/<split>/<stem>.jpg|.xml   and   data/masks/<class>/<split>/<stem>.png   (per-image masks)
Zips that are missing are skipped, so a single class can be prepared on its own.
"""
import argparse, zipfile
from pathlib import Path

CLASSES = ("coronal_hole", "sunspot", "prominence")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default="."); ap.add_argument("--out", default="data"); a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    for cls in CLASSES:
        root = f"solar_{cls}_seg"; z = src / f"{root}.zip"
        if not z.is_file(): print(f"skip {cls}: {z} not found"); continue
        n = dict(images=0, boxes=0, masks=0)
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                p = Path(name).parts
                if len(p) != 4 or p[0] != root: continue                      # <root>/<kind>/<split>/<file>
                kind, split, fn = p[1:]
                if kind in ("images", "boxes"): dst = out / "images" / split / fn
                elif kind == "masks_full": dst = out / "masks" / cls / split / fn; kind = "masks"
                else: continue                                               # per-box masks and labels.csv stay in the zip
                dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(zf.read(name)); n[kind] += 1
        print(f"{cls}: {n['images']} images, {n['boxes']} box files, {n['masks']} masks -> {out}")


if __name__ == "__main__":
    main()
