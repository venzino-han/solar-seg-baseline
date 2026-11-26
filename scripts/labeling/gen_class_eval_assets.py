#!/usr/bin/env python3
"""Build editable eval assets for sunspot / prominence from the SAM 3 label pack.

Same on-disk layout eval_app.py expects, in an isolated directory per class so
the existing coronal-hole sets (eval_web/, eval_web_ft/) and their DBs are never
touched:

    eval_web_<cls>/base/<sid>.jpg    crop around the VOC box, upscaled, box drawn
    eval_web_<cls>/mask/<sid>.png    the SAM 3 mask for that box, same canvas
    eval_web_<cls>/manifest.json     sid, stem, bi, box, w, h, coverage, agree, date
    eval_web_<cls>/config.json       title / brand / overlay colour for the web UI

Samples come from the TRAIN split only (the user asked for train-set labels to be
corrected — the test split stays untouched so it remains a clean held-out set).

Selection: confirmed boxes, at most one per image, round-robin over years so a
single active period cannot dominate, deterministic under --seed.

Usage:
  .venv_web/bin/python gen_class_eval_assets.py --cls sunspot --n 100
  .venv_web/bin/python gen_class_eval_assets.py --cls prominence --n 100
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "dataset"
FRAME = 512
UP = 460          # default canvas height, identical to the coronal-hole tool

CFG = {
    "sunspot": dict(
        pack="label_pack_sunspot", ko="흑점", channel="HMI Continuum",
        # Spots are ~20-60 px; a fixed 36 px pad would bury them in context, so the
        # pad scales with the box and the crop is upscaled hard.
        pad_frac=0.6, pad_min=12, pad_max=44, up=UP,
        mask_rgb=[0, 200, 255], min_area=6, allow_outside=False,
    ),
    "prominence": dict(
        pack="label_pack_prominence", ko="홍염", channel="AIA 304Å",
        # A prominence is an off-limb structure that routinely extends past the
        # detection box, and the SAM 3 label is hard-clipped at the box edge
        # (label_other_classes.py: "never leak outside the label box"). The
        # labeler has to be able to carry it outwards, so the view is widened
        # (box goes from ~51% to ~35% of the frame) on a taller canvas to keep
        # detail, and out-of-box paint is kept on export.
        pad_frac=0.9, pad_min=28, pad_max=120, up=520,
        mask_rgb=[150, 255, 60], min_area=20, allow_outside=True,
    ),
}


def crop_up(img, box, pad, interp=cv2.INTER_CUBIC, up=UP):
    x1, y1, x2, y2 = box
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(FRAME, x2 + pad), min(FRAME, y2 + pad)
    c = img[cy1:cy2, cx1:cx2]
    # Fit the LONGER side to `up`, not the height: a crop clipped by the frame
    # edge can be much wider than tall, and scaling by height alone pushed the
    # canvas past 1000 px so the two panes wrapped onto separate rows.
    sc = up / max(1, c.shape[0], c.shape[1])
    c = cv2.resize(c, (max(1, round(c.shape[1] * sc)), max(1, round(c.shape[0] * sc))),
                   interpolation=interp)
    return c, (cx1, cy1, cx2, cy2, sc)


def pick(rows, n, seed):
    """One box per image, round-robin across years, deterministic."""
    best = {}
    for r in rows:                       # per image keep the largest confirmed box
        k = r["stem"]
        if k not in best or int(r["area_px"]) > int(best[k]["area_px"]):
            best[k] = r
    by_year = {}
    for r in best.values():
        by_year.setdefault(r["stem"][:4], []).append(r)
    rng = np.random.RandomState(seed)
    for v in by_year.values():
        v.sort(key=lambda r: r["stem"])  # stable order before the shuffle
        rng.shuffle(v)
    yrs, picks, k = sorted(by_year), [], 0
    while len(picks) < n and any(by_year.values()):
        y = yrs[k % len(yrs)]
        if by_year[y]:
            picks.append(by_year[y].pop())
        k += 1
    picks.sort(key=lambda r: r["stem"])
    return picks[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, choices=sorted(CFG))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--pack", default=None, help="레이블 팩 디렉터리 (기본: CFG 의 pack)")
    ap.add_argument("--exclude-manifest", default=None,
                    help="이 manifest.json 에 이미 있는 프레임은 제외 (다른 서비스와 샘플이 겹치지 않게)")
    ap.add_argument("--label-source", default=None, help="화면 제목에 표시할 레이블 출처")
    ap.add_argument("--all", action="store_true",
                    help="전수 검사: --split(쉼표 구분)의 모든 박스를 날짜순으로 (게이트 미통과 포함, 이미지당 1개 제한 없음)")
    args = ap.parse_args()

    cfg = CFG[args.cls]
    pack = BASE / (args.pack or cfg["pack"])
    out = BASE / (args.out or f"eval_web_{args.cls}")
    for d in ("base", "mask", "edited"):
        (out / d).mkdir(parents=True, exist_ok=True)

    splits = [x.strip() for x in args.split.split(",") if x.strip()]
    if args.all:
        rows = [r for r in csv.DictReader((pack / "labels.csv").open())
                if r["split"] in splits and r["mask_path"]]
    else:
        rows = [r for r in csv.DictReader((pack / "labels.csv").open())
                if r["split"] == args.split and r["confirmed"] == "1"
                and int(r["area_px"]) >= cfg["min_area"]]
    if args.exclude_manifest:
        seen = {m["stem"] for m in json.loads((BASE / args.exclude_manifest).read_text())}
        rows = [r for r in rows if r["stem"] not in seen]
        print(f"  excluded frames already in {args.exclude_manifest}: {len(seen)}")
    if args.all:
        recs = sorted(rows, key=lambda r: (r["stem"], int(r["bi"])))
    else:
        recs = pick(rows, args.n, args.seed)
    sw = max(3, len(str(len(recs) - 1)))
    print(f"{args.cls}: {len(rows)} confirmed {args.split} boxes -> picked {len(recs)} "
          f"from {len({r['stem'][:4] for r in recs})} years")

    manifest = []
    for i, r in enumerate(recs):
        stem = r["stem"]
        img_p = DATA / r["split"] / f"{stem}.jpg"
        bgr = cv2.imread(str(img_p))
        m_p = pack / r["mask_path"]
        mask = cv2.imread(str(m_p), 0)
        if bgr is None or mask is None:
            print(f"  skip {stem}: missing {'image' if bgr is None else 'mask'}")
            continue
        mask = mask > 127
        box = [int(v) for v in r["box"].split()]
        x1, y1, x2, y2 = box
        pad = int(np.clip(cfg["pad_frac"] * max(x2 - x1, y2 - y1),
                          cfg["pad_min"], cfg["pad_max"]))

        base, (cx1, cy1, cx2, cy2, sc) = crop_up(bgr, box, pad, up=cfg["up"])
        bq = [int((x1 - cx1) * sc), int((y1 - cy1) * sc),
              int((x2 - cx1) * sc), int((y2 - cy1) * sc)]
        # A thinner box when out-of-box painting is the point: it is a reference
        # marker there, not a boundary the labeler must stay inside.
        cv2.rectangle(base, (bq[0], bq[1]), (bq[2], bq[3]), (0, 255, 255),
                      1 if cfg["allow_outside"] else 2)
        msub = mask[cy1:cy2, cx1:cx2].astype(np.uint8) * 255
        # NEAREST_EXACT: legacy INTER_NEAREST shifts the mask toward the
        # bottom-right, and at this 4-7x zoom that is several visible pixels.
        msub = cv2.resize(msub, (base.shape[1], base.shape[0]),
                          interpolation=cv2.INTER_NEAREST_EXACT)

        sid = f"{i:0{sw}d}"
        cv2.imwrite(str(out / "base" / f"{sid}.jpg"), base, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(out / "mask" / f"{sid}.png"), msub)
        manifest.append({
            "sid": sid, "stem": stem, "bi": int(r["bi"]), "box": box,
            "w": int(base.shape[1]), "h": int(base.shape[0]),
            # crop rect in 512-space; export_class_edits.py needs it to map an
            # edited canvas mask back onto the original frame.
            "crop": [cx1, cy1, cx2, cy2],
            "boxq": bq,          # the box in canvas coords, for the out-of-box counter
            "mask_path": r["mask_path"],
            "coverage": round(float(r["box_coverage"]), 3),
            # UI field named "agree": here it is SAM 3 vs the CV prior's IoU, the
            # only per-box confidence the label pack carries.
            "agree": round(float(r["sam_cv_iou"]), 3),
            "area_px": int(r["area_px"]),
            "date": f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}",
            "split": r["split"], "confirmed": int(r["confirmed"]),
            "method": "sam3_labelpack", "pack": pack.name,
        })
        if (i + 1) % (500 if args.all else 25) == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "config.json").write_text(json.dumps({
        "cls": args.cls, "ko": cfg["ko"], "channel": cfg["channel"],
        "title": f"{cfg['ko']} 세그멘테이션 평가 · 편집",
        "brand": f"{cfg['ko']} 평가 · 편집  ({cfg['channel']} · {args.split} set"
                 + (f" · {args.label_source})" if args.label_source else ")"),
        "mask_rgb": cfg["mask_rgb"],
        "agree_label": "SAM·CV 일치",
        "allow_outside": cfg["allow_outside"],
        "hint": ("노란 박스는 참고용 표시입니다. 홍염이 박스 밖으로 이어지면 "
                 "박스 바깥 영역까지 칠하세요 — 그대로 저장됩니다."
                 if cfg["allow_outside"] else ""),
    }, ensure_ascii=False, indent=1))

    cov = np.mean([m["coverage"] for m in manifest]) if manifest else 0
    ar = np.median([m["area_px"] for m in manifest]) if manifest else 0
    print(f"\nwrote {len(manifest)} samples -> {out}  (mean box cov {cov:.3f}, median area {ar:.0f}px)")


if __name__ == "__main__":
    main()
