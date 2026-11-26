#!/usr/bin/env python3
"""Coronal hole: retry the single unlabelled box with the fine-tuned SAM 3 (text+box prompt) under mirrored input and
slightly resized prompt boxes; same validity gate (area>=60, coverage 0.05-0.90, confidence>=0.30)."""
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
B = Path(__file__).resolve().parent; sys.path.insert(0, str(B))
import segmenters as S, sam3_common as C, build_label_pack as P
dev = "cuda:0"; stem, split, box = "20130207_120018_SDO_AIA_193_512", "train", [351, 108, 429, 181]; O = B / "retry_unlabeled/coronal_hole"
model, proc = C.load_model(device=dev); ck = torch.load(B / "gold/runs/deploy/sam3_coronal_hole_heads.pt", map_location="cpu", weights_only=False)
with torch.no_grad():
    own = dict(model.named_parameters())
    for k, v in ck["state_dict"].items(): own[k].data.copy_(v.to(own[k].device, own[k].dtype))
thr = ck.get("thr", 0.0); bgr0 = cv2.imread(str(B.parent / "dataset" / split / f"{stem}.jpg"))
def run(flip, grow):
    bgr = bgr0 if flip is None else cv2.flip(bgr0, flip); x1, y1, x2, y2 = box
    if flip == 1: x1, x2 = 512 - x2, 512 - x1
    if flip == 0: y1, y2 = 512 - y2, 512 - y1
    gx, gy = int(grow * (x2 - x1) / 2), int(grow * (y2 - y1) / 2); pb = [max(0, x1 - gx), max(0, y1 - gy), min(512, x2 + gx), min(512, y2 + gy)]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); disk = S.solar_disk_mask(gray); w = bgr.copy(); w[~disk] = (255, 255, 255)
    pix = proc(images=cv2.cvtColor(w, cv2.COLOR_BGR2RGB), text=C.TEXT_PROMPT, input_boxes=[[pb]], return_tensors="pt").to(dev)
    with torch.no_grad():
        ve = model.vision_encoder(pix["pixel_values"]); te = model.get_text_features(input_ids=pix["input_ids"], attention_mask=pix["attention_mask"], return_dict=True)
        b = P.box_inputs(proc, pb, dev); o = model(vision_embeds=ve, text_embeds=te, attention_mask=pix["attention_mask"], input_boxes=b["input_boxes"], input_boxes_labels=b["input_boxes_labels"])
    qi, qs, qb = C.select_query(o, pb); score = float(qs[qi] * qb[qi]); bm = np.zeros((512, 512), bool); bm[y1:y2, x1:x2] = True
    m = (C.mask_logits_512(o.pred_masks[0, qi]).cpu().numpy() > thr) & disk & bm
    if flip is not None: m = cv2.flip(m.astype(np.uint8), flip) > 0
    area = int(m.sum()); cov = area / ((x2 - x1) * (y2 - y1))
    return m, dict(flip={None: "none", 1: "horizontal", 0: "vertical"}[flip], box_grow=grow, confidence=round(score, 4), area=area, cov=round(cov, 4), ok=int(area >= 60 and 0.05 <= cov <= 0.9 and score >= 0.30))
res, ms = [], []
for fl in (None, 1, 0):
    for g in (0.0, 0.1, -0.1, 0.2):
        m, d = run(fl, g); res.append(d); ms.append(m)
base = ms[0]
for d, m in zip(res, ms): d["iou_vs_first"] = round(float((m & base).sum() / max(1, (m | base).sum())), 4); print(d)
np.savez_compressed(O / "masks.npz", m=np.packbits(np.stack(ms))); json.dump(res, open(O / "variants.json", "w"), indent=1)
