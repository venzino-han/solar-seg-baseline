#!/usr/bin/env python3
"""Auto-label every coronal hole with the label-trained SAM 3, and package the
confirmed labels + image overlays.

For each coronal-hole box in the whole dataset (train+test, 9,984 boxes) the
fine-tuned model predicts a mask. A box counts as "labeled / 확인됨" only if the
mask passes a validity check:
    area >= MIN_AREA px, box-coverage in [COV_LO, COV_HI], confidence >= MIN_SCORE.
Confirmed items are written as:
    label_pack/masks/<split>/<stem>__b<bi>.png     per-box binary label (512x512)
    label_pack/masks_full/<split>/<stem>.png       per-image union label (512x512)
    label_pack/overlays/<split>/<stem>.jpg         real AIA image with holes mapped on
Plus label_pack/labels.csv (every box, confirmed flag + metadata).

Shardable across GPUs:  --shard 0/2  and  --shard 1/2.

Usage:
  CUDA_VISIBLE_DEVICES=0 python build_label_pack.py --shard 0/2
  CUDA_VISIBLE_DEVICES=1 python build_label_pack.py --shard 1/2
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
import segmenters as S
import sam3_common as C

BASE = Path(__file__).resolve().parent
OUT = BASE / "label_pack"
FRAME = 512

# validity ("labeled?") criteria — same spirit as the pseudo-label filter
MIN_AREA = 60
COV_LO, COV_HI = 0.05, 0.90
MIN_SCORE = 0.30

HOLE_COLOR = (200, 90, 220)   # BGR magenta fill/contour


def box_inputs(proc, box, device):
    inp = proc(images=None, text=C.TEXT_PROMPT, input_boxes=[[[int(v) for v in box]]],
               original_sizes=torch.tensor([[FRAME, FRAME]]), return_tensors="pt")
    return {k: v.to(device) for k, v in inp.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="gold/runs/deploy/sam3_coronal_hole_heads.pt")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))

    for sub in ("masks", "masks_full", "overlays"):
        for sp in ("train", "test"):
            (OUT / sub / sp).mkdir(parents=True, exist_ok=True)

    model, proc = C.load_model(device=args.device)
    ck = torch.load(BASE / args.ckpt, map_location="cpu", weights_only=False)
    with torch.no_grad():
        own = dict(model.named_parameters())
        for kk, v in ck["state_dict"].items():
            own[kk].data.copy_(v.to(own[kk].device, own[kk].dtype))
    thr = ck.get("thr", 0.0)

    gold_boxes = {(r["stem"], tuple(r["box"])) for r in C.load_gold()}
    idx = json.loads((BASE / "index_coronal_hole.json").read_text())
    idx = [im for ii, im in enumerate(idx) if ii % n == k]
    print(f"shard {k}/{n}: {len(idx)} images / {sum(i['n_boxes'] for i in idx)} boxes  thr {thr:+.1f}",
          flush=True)

    rows = []
    n_conf = 0
    t0 = time.time()
    for ii, im in enumerate(idx):
        stem, split = im["stem"], im["split"]
        bgr = cv2.imread(str(BASE.parent / "dataset" / split / f"{stem}.jpg"))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        disk = S.solar_disk_mask(gray)
        w = bgr.copy(); w[~disk] = (255, 255, 255)
        rgb = cv2.cvtColor(w, cv2.COLOR_BGR2RGB)

        pix = proc(images=rgb, text=C.TEXT_PROMPT,
                   input_boxes=[[[int(v) for v in im["boxes"][0][:4]]]],
                   return_tensors="pt").to(args.device)
        with torch.no_grad():
            ve = model.vision_encoder(pix["pixel_values"])
            te = model.get_text_features(input_ids=pix["input_ids"],
                                         attention_mask=pix["attention_mask"], return_dict=True)
        attn = pix["attention_mask"]

        union = np.zeros((FRAME, FRAME), bool)
        confirmed_here = []
        for bi, b in enumerate(im["boxes"]):
            box = [int(v) for v in b[:4]]
            x1, y1, x2, y2 = box
            bm = np.zeros((FRAME, FRAME), bool); bm[y1:y2, x1:x2] = True
            binp = box_inputs(proc, box, args.device)
            with torch.no_grad():
                o = model(vision_embeds=ve, text_embeds=te, attention_mask=attn,
                          input_boxes=binp["input_boxes"], input_boxes_labels=binp["input_boxes_labels"])
            qi, qs, qb = C.select_query(o, box)
            score = float(qs[qi] * qb[qi])
            mask = (C.mask_logits_512(o.pred_masks[0, qi]).cpu().numpy() > thr) & disk & bm
            area = int(mask.sum())
            barea = (x2 - x1) * (y2 - y1)
            cov = area / max(1, barea)
            confirmed = (area >= MIN_AREA and COV_LO <= cov <= COV_HI and score >= MIN_SCORE)

            if confirmed:
                cv2.imwrite(str(OUT / "masks" / split / f"{stem}__b{bi}.png"),
                            mask.astype(np.uint8) * 255)
                union |= mask
                confirmed_here.append(box)
                n_conf += 1

            rows.append({"split": split, "stem": stem, "bi": bi,
                         "box": f"{x1} {y1} {x2} {y2}", "area_px": area,
                         "box_coverage": round(cov, 4), "confidence": round(score, 4),
                         "is_gold": int((stem, tuple(box)) in gold_boxes),
                         "confirmed": int(confirmed),
                         "mask_path": f"masks/{split}/{stem}__b{bi}.png" if confirmed else "",
                         "overlay_path": f"overlays/{split}/{stem}.jpg" if confirmed else ""})

        if confirmed_here:
            cv2.imwrite(str(OUT / "masks_full" / split / f"{stem}.png"),
                        union.astype(np.uint8) * 255)
            ov = bgr.astype(np.float32)
            for c in range(3):
                ov[..., c][union] = 0.45 * ov[..., c][union] + 0.55 * HOLE_COLOR[c]
            ov = ov.astype(np.uint8)
            cnts, _ = cv2.findContours(union.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnts, -1, HOLE_COLOR, 1)
            for box in confirmed_here:
                cv2.rectangle(ov, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)
            cv2.putText(ov, f"{stem}  holes:{len(confirmed_here)}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(OUT / "overlays" / split / f"{stem}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 86])

        if (ii + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {ii+1}/{len(idx)} img · {len(rows)} box · conf {n_conf} · {el:.0f}s · "
                  f"eta {el/(ii+1)*(len(idx)-ii-1):.0f}s", flush=True)

    frag = OUT / f"_index.shard{k}of{n}.csv"
    with frag.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"shard {k}/{n} done: {len(rows)} boxes, {n_conf} confirmed, {time.time()-t0:.0f}s -> {frag}",
          flush=True)


if __name__ == "__main__":
    main()
