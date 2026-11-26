#!/usr/bin/env python3
"""Pull the reviewed/edited masks out of a class eval instance back into 512-space.

The web editor works on an upscaled crop, so an edited canvas mask has to be
resized down and pasted into the original frame at the crop offset (manifest
["crop"]) before it can be used as a training label.

Writes, for the chosen evaluator (or the most recent edit per sid if --any):

    <out>/masks/<stem>__b<bi>.png     corrected mask, 512x512, white = label
    <out>/overlays/<stem>__b<bi>.jpg  the crop with the corrected mask drawn
    <out>/labels.csv                  stem, bi, box, verdict, note, edited, area
    <out>/SUMMARY.md                  counts by verdict, edited vs untouched

Samples the evaluator judged 적절함 and did NOT edit are exported unchanged, so
the output is a complete set of human-confirmed labels, not just the fixed ones.

Usage:
  .venv_train/bin/python export_class_edits.py --cls sunspot --evaluator labeller_8
  .venv_train/bin/python export_class_edits.py --cls prominence --any
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "dataset"
FRAME = 512


# Mirrors eval_app.safe()/legacy_safe(). Copied rather than imported because
# eval_app pulls in flask, which the CV/torch venv used for exporting lacks.
def safe(name):
    raw = (name or "").strip()
    slug = re.sub(r"[^0-9A-Za-z가-힣._-]", "", raw)[:24] or "u"
    return f"{slug}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}"


def legacy_safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", (name or "").strip())[:40]


def edited_png(web: Path, evaluator: str, sid: str) -> Path | None:
    for d in (safe(evaluator), legacy_safe(evaluator)):
        p = web / "edited" / d / f"{sid}.png"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, choices=("sunspot", "prominence", "coronal_hole"))
    ap.add_argument("--web", default=None, help="default eval_web_<cls>")
    ap.add_argument("--evaluator", default=None)
    ap.add_argument("--any", action="store_true", help="take the latest eval per sid, any evaluator")
    ap.add_argument("--exclude", default="", help="쉼표로 구분한 제외 평가자 (--any 와 함께)")
    ap.add_argument("--out", default=None, help="default <web>/export")
    ap.add_argument("--clip-box", action="store_true",
                    help="박스 밖으로 나간 편집분을 잘라냄 (기본: 사람이 칠한 그대로 보존)")
    args = ap.parse_args()
    if not args.evaluator and not args.any:
        ap.error("--evaluator NAME 또는 --any 중 하나가 필요합니다")

    web = BASE / (args.web or f"eval_web_{args.cls}")
    man = {m["sid"]: m for m in json.loads((web / "manifest.json").read_text())}
    cfg = json.loads((web / "config.json").read_text()) if (web / "config.json").exists() else {}
    r, g, b = cfg.get("mask_rgb", [60, 220, 130])
    col = (b, g, r)

    con = sqlite3.connect(web / "evaluations.db"); con.row_factory = sqlite3.Row
    if args.any:
        rows = con.execute("SELECT * FROM evals ORDER BY ts").fetchall()
        # Drop excluded evaluators BEFORE the dedup, not after: a sid that an
        # excluded evaluator touched last would otherwise shadow a good earlier
        # edit, and a sid only they touched would survive as their work.
        drop = {s.strip() for s in args.exclude.split(",") if s.strip()}
        if drop:
            n0 = len(rows)
            rows = [r for r in rows if r["evaluator"] not in drop]
            print(f"제외 {sorted(drop)}: {n0 - len(rows)}건 제거")
        rows = list({r["sid"]: r for r in rows}.values())      # last write per sid wins
    else:
        rows = con.execute("SELECT * FROM evals WHERE evaluator=? ORDER BY sid",
                           (args.evaluator,)).fetchall()
    if not rows:
        print("평가 기록이 없습니다 — 웹에서 저장한 뒤 다시 실행하세요.")
        return

    out = BASE / (args.out or (web / "export").relative_to(BASE))
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    recs, verdicts, n_edit, n_outside = [], Counter(), 0, 0
    for row in rows:
        m = man.get(row["sid"])
        if m is None:
            continue
        cx1, cy1, cx2, cy2 = m["crop"]
        src = edited_png(web, row["evaluator"], m["sid"]) if row["edited"] else None
        canvas = cv2.imread(str(src or (web / "mask" / f"{m['sid']}.png")), 0)
        if canvas is None:
            print(f"  skip {m['sid']}: 마스크 읽기 실패")
            continue
        # NEAREST_EXACT keeps the human's edit where they painted it; legacy
        # INTER_NEAREST would walk it toward the bottom-right on the way down.
        sub = cv2.resize((canvas > 127).astype(np.uint8) * 255, (cx2 - cx1, cy2 - cy1),
                         interpolation=cv2.INTER_NEAREST_EXACT)
        full = np.zeros((FRAME, FRAME), np.uint8)
        full[cy1:cy2, cx1:cx2] = sub

        # The editor canvas is padded beyond the VOC box, so a labeler can paint
        # outside it. That is kept by default (it may be a deliberate correction)
        # and only reported; --clip-box restores the label-pack convention.
        bx1, by1, bx2, by2 = m["box"]
        out_px = int((full > 127).sum() - (full[by1:by2, bx1:bx2] > 127).sum())
        if out_px:
            n_outside += 1
        if args.clip_box and out_px:
            keep = np.zeros_like(full); keep[by1:by2, bx1:bx2] = full[by1:by2, bx1:bx2]
            full, sub = keep, keep[cy1:cy2, cx1:cx2]

        name = f"{m['stem']}__b{m['bi']}"
        cv2.imwrite(str(out / "masks" / f"{name}.png"), full)

        img = cv2.imread(str(DATA / m.get("split", "train") / f"{m['stem']}.jpg"))
        if img is not None:
            crop = img[cy1:cy2, cx1:cx2].astype(np.float32)
            mk = sub > 127
            for c in range(3):
                crop[..., c][mk] = 0.45 * crop[..., c][mk] + 0.55 * col[c]
            crop = cv2.resize(crop.astype(np.uint8), (m["w"], m["h"]), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(out / "overlays" / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])

        x1, y1, x2, y2 = m["box"]
        area = int((full > 127).sum())
        verdicts[row["verdict"]] += 1
        n_edit += int(row["edited"] or 0)
        recs.append({"split": m.get("split", "train"), "stem": m["stem"], "bi": m["bi"],
                     "box": " ".join(map(str, m["box"])), "area_px": area,
                     "box_coverage": round(area / max(1, (x2 - x1) * (y2 - y1)), 4),
                     "outside_box_px": 0 if args.clip_box else out_px,
                     "sam3_area_px": m.get("area_px", ""), "verdict": row["verdict"],
                     "note": row["note"] or "", "edited": int(row["edited"] or 0),
                     "evaluator": row["evaluator"],
                     "mask_path": f"masks/{name}.png", "overlay_path": f"overlays/{name}.jpg"})

    with (out / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader(); w.writerows(recs)

    ok = verdicts.get("적절함", 0)
    (out / "SUMMARY.md").write_text(
        f"# {cfg.get('ko', args.cls)} 사람 검수 레이블\n\n"
        f"- 원본: `{web.name}` (SAM 3 자동 레이블, "
        f"{'/'.join(sorted({r['split'] for r in recs}))} split {len(man)}건 중 검수 {len(recs)}건)\n"
        f"- 편집으로 수정된 마스크: **{n_edit}건** / 무수정 승인 {len(recs) - n_edit}건\n"
        f"- 판정: " + ", ".join(f"{k} {v}" for k, v in verdicts.most_common()) + "\n"
        f"- '적절함' 비율: **{ok}/{len(recs)} ({ok/len(recs)*100:.1f}%)**\n"
        + (f"- 박스 밖으로 나간 마스크: {n_outside}건 — `--clip-box` 로 잘라낼 수 있습니다\n"
           if not args.clip_box else "- 박스 밖 마스크는 `--clip-box` 로 잘라냈습니다\n")
        + "\n마스크는 512x512, 흰색=레이블. 편집본은 웹 캔버스에서 512 좌표로 역변환한 결과입니다.\n")

    print(f"{len(recs)}건 내보냄 -> {out}")
    print(f"  편집 {n_edit} / 무수정 {len(recs)-n_edit}")
    for k, v in verdicts.most_common():
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
