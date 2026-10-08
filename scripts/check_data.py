#!/usr/bin/env python3
"""Validate the data layout and print per-class statistics.

    python scripts/check_data.py --data-root data
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solarseg.config import CLASSES
from solarseg.data import list_frames, mask_path, read_voc_boxes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", default="data"); a = ap.parse_args()
    ok = True
    print(f"{'class':13s} {'split':5s} {'frames':>6s} {'masks':>6s} {'boxes':>6s} {'fg %':>7s} {'empty':>6s}")
    for cls in CLASSES:
        for split in ("train", "test"):
            fr = list_frames(a.data_root, cls, split)
            n_mask = boxes = empty = 0; fg = []
            for p in fr:
                boxes += len(read_voc_boxes(p.with_suffix(".xml"), cls))
                mp = mask_path(a.data_root, cls, split, p.stem)
                if mp.exists():
                    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    if m is None or m.shape != (512, 512):
                        print(f"  bad mask: {mp}"); ok = False; continue
                    n_mask += 1; f = (m > 127).mean(); fg.append(f); empty += f == 0
            print(f"{cls:13s} {split:5s} {len(fr):6d} {n_mask:6d} {boxes:6d} {np.mean(fg)*100 if fg else 0:7.3f} {empty:6d}")
            if not fr:
                ok = False
    print("OK" if ok else "problems found")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
