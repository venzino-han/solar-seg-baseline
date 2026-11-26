#!/usr/bin/env python3
"""Export the human-edited coronal-hole masks into a training-ready GT set.

Source of truth
---------------
  eval_web/manifest.json        72 samples (sid, stem, bi, box, w, h)
  eval_web/mask/<sid>.png       model mask on the 460-high edit canvas
  eval_web/edited/<dir>/<sid>.png   labeler-edited mask, same canvas
  eval_web/evaluations.db       who evaluated what, and whether they edited

The edit canvas is a padded crop of the 512x512 frame, upscaled so its height is
460 (see gen_edit_assets.py):

    cx1, cy1 = max(0, x1-PAD), max(0, y1-PAD)
    cx2, cy2 = min(512, x2+PAD), min(512, y2+PAD)
    sc       = 460 / (cy2-cy1)
    canvas   = (int((cx2-cx1)*sc), 460)

This script inverts that mapping (NEAREST) to put every mask back in 512x512
frame coordinates, and writes:

    gold/masks_512/<sid>.png   512x512 uint8 {0,255} GT
    gold/crops/<sid>_img.png   padded crop of the original frame (512-scale)
    gold/crops/<sid>_gt.png    matching GT crop
    gold/gold.jsonl            one record per sample

Usage:
    .venv_train/bin/python export_gold.py --evaluator labeller_8
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
WEB = BASE / "eval_web"
DATASET = BASE.parent / "dataset"
PAD = 36
CANVAS_H = 460
FRAME = 512

# `labeller_7` is a 1-character typo of `labeller_8` made by the same labeler: the 7/28
# session covered sid 000-003, the 8/3 session re-did all 72. Both wrote into
# eval_web/edited/___/ because eval_app.safe() collapses Hangul to underscores.
ALIASES = {"labeller_8": ["labeller_8", "labeller_7"]}


def crop_geometry(box):
    """(cx1, cy1, cx2, cy2, canvas_w) for a 512-frame box."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cx1, cy1 = max(0, x1 - PAD), max(0, y1 - PAD)
    cx2, cy2 = min(FRAME, x2 + PAD), min(FRAME, y2 + PAD)
    sc = CANVAS_H / max(1, cy2 - cy1)
    return cx1, cy1, cx2, cy2, int((cx2 - cx1) * sc)


def canvas_to_frame(canvas_mask, box):
    """Invert the crop+upscale: canvas mask -> full 512x512 boolean frame."""
    cx1, cy1, cx2, cy2, cw = crop_geometry(box)
    ch = cy2 - cy1
    small = cv2.resize(canvas_mask.astype(np.uint8), (cx2 - cx1, ch),
                       interpolation=cv2.INTER_NEAREST) > 0
    out = np.zeros((FRAME, FRAME), bool)
    out[cy1:cy2, cx1:cx2] = small
    return out


def find_image(stem):
    for split in ("train", "test"):
        p = DATASET / split / f"{stem}.jpg"
        if p.exists():
            return p, split
    raise FileNotFoundError(stem)


def load_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator", default="labeller_8")
    ap.add_argument("--out", default="gold")
    args = ap.parse_args()

    names = ALIASES.get(args.evaluator, [args.evaluator])
    out = BASE / args.out
    (out / "masks_512").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    manifest = {m["sid"]: m for m in json.loads((WEB / "manifest.json").read_text())}

    # newest DB row per sid across the labeler's aliases
    con = sqlite3.connect(WEB / "evaluations.db")
    con.row_factory = sqlite3.Row
    q = ",".join("?" * len(names))
    rows = {}
    for r in con.execute(
            f"SELECT sid,evaluator,verdict,note,edited,ts FROM evals "
            f"WHERE evaluator IN ({q}) ORDER BY ts", names):
        rows[r["sid"]] = dict(r)
    con.close()
    print(f"DB rows for {names}: {len(rows)} sids")

    # edited PNGs live in the collapsed directory; take whatever is on disk
    edit_dirs = [d for d in (WEB / "edited").iterdir() if d.is_dir()]

    recs, skipped = [], []
    for sid, man in sorted(manifest.items()):
        row = rows.get(sid)
        if row is None:
            skipped.append((sid, "no DB row"))
            continue

        model_mask = load_mask(WEB / "mask" / f"{sid}.png")
        edited_path = next((d / f"{sid}.png" for d in edit_dirs
                            if (d / f"{sid}.png").exists()), None)

        if edited_path is not None:
            canvas = load_mask(edited_path)
            source = "edited"
            mtime = edited_path.stat().st_mtime
        else:
            # verdict was "적절함" with no brush edit -> labeler approved the model mask
            canvas = model_mask
            source = "approved"
            mtime = row["ts"]

        if canvas.shape != (man["h"], man["w"]):
            skipped.append((sid, f"canvas {canvas.shape} != manifest {(man['h'], man['w'])}"))
            continue

        gt = canvas_to_frame(canvas, man["box"])
        orig = canvas_to_frame(model_mask, man["box"])

        img_path, split = find_image(man["stem"])
        bgr = cv2.imread(str(img_path))

        cx1, cy1, cx2, cy2, _ = crop_geometry(man["box"])
        cv2.imwrite(str(out / "masks_512" / f"{sid}.png"), gt.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "crops" / f"{sid}_img.png"), bgr[cy1:cy2, cx1:cx2])
        cv2.imwrite(str(out / "crops" / f"{sid}_gt.png"),
                    gt[cy1:cy2, cx1:cx2].astype(np.uint8) * 255)

        inter = int((gt & orig).sum())
        union = int((gt | orig).sum())
        x1, y1, x2, y2 = man["box"]
        recs.append({
            "sid": sid,
            "stem": man["stem"],
            "bi": man["bi"],
            "date": man["date"],
            "day": man["stem"][:8],
            "orig_split": split,
            "image": str(img_path.relative_to(BASE.parent)),
            "box": man["box"],
            "crop": [cx1, cy1, cx2, cy2],
            "source": source,
            "weight": 1.0 if source == "edited" else 0.5,
            "verdict": row["verdict"],
            "evaluator": row["evaluator"],
            "labeled_at": mtime,
            "gt_area_px": int(gt.sum()),
            "model_area_px": int(orig.sum()),
            "gt_frac_of_frame": round(float(gt.sum()) / FRAME ** 2, 5),
            "gt_frac_of_box": round(float(gt[y1:y2, x1:x2].sum()) / max(1, (x2 - x1) * (y2 - y1)), 4),
            "gt_px_outside_box": int(gt.sum() - gt[y1:y2, x1:x2].sum()),
            "iou_model_vs_gt": round(inter / union, 4) if union else 1.0,
            "delta_area": round((orig.sum() - gt.sum()) / max(1, gt.sum()), 4),
            "agree_sam2_sam3": man["agree"],
            "coverage_model": man["coverage"],
        })

    (out / "gold.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")

    n_ed = sum(1 for r in recs if r["source"] == "edited")
    ious = np.array([r["iou_model_vs_gt"] for r in recs])
    print(f"\nexported {len(recs)} samples  (edited {n_ed} / approved {len(recs)-n_ed})")
    print(f"  unique images {len({r['stem'] for r in recs})}  unique days {len({r['day'] for r in recs})}")
    print(f"  GT px total {sum(r['gt_area_px'] for r in recs)}  mean {np.mean([r['gt_area_px'] for r in recs]):.0f}")
    print(f"  GT px painted outside the prompt box: {sum(r['gt_px_outside_box'] for r in recs)}")
    print(f"  model-vs-GT IoU  mean {ious.mean():.4f}  median {np.median(ious):.4f}")
    print(f"  -> {out}/gold.jsonl")
    if skipped:
        print(f"\nSKIPPED {len(skipped)}:")
        for sid, why in skipped:
            print(f"  {sid}: {why}")


if __name__ == "__main__":
    main()
