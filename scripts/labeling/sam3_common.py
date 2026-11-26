#!/usr/bin/env python3
"""Shared plumbing for SAM 3 box-prompted coronal-hole segmentation.

Conventions (identical in eval and training, so the numbers are comparable):
  * input   : 512x512 AIA 193 frame, off-disk background whitened to 255
  * prompt  : the VOC bounding box of one coronal hole
  * target  : the labeler's mask, in 512-frame coordinates (gold/masks_512)
  * scoring : restricted to the CROP region the labeler actually saw
              (box + PAD 36, clipped to the frame) -- pixels they never looked
              at must not count as true negatives.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
FRAME = 512
MASK_RES = 288          # SAM3 pred_masks / semantic_seg resolution
TEXT_PROMPT = "dark coronal hole"

_DISK_CACHE: dict[str, np.ndarray] = {}


# --------------------------------------------------------------------------- data
def load_gold(path="gold/gold.jsonl"):
    return [json.loads(l) for l in (BASE / path).read_text().splitlines() if l.strip()]


def disk_mask(gray, stem):
    if stem not in _DISK_CACHE:
        import segmenters as S
        _DISK_CACHE[stem] = S.solar_disk_mask(gray)
    return _DISK_CACHE[stem]


def load_sample(rec):
    """-> rgb uint8 (512,512,3) whitened off-disk, gt bool, disk bool."""
    bgr = cv2.imread(str(BASE.parent / rec["image"]))
    if bgr is None:
        raise FileNotFoundError(rec["image"])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    disk = disk_mask(gray, rec["stem"])
    w = bgr.copy()
    w[~disk] = (255, 255, 255)
    rgb = cv2.cvtColor(w, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(str(BASE / "gold" / "masks_512" / f"{rec['sid']}.png"),
                    cv2.IMREAD_GRAYSCALE) > 127
    return rgb, gt, disk


def crop_region(rec):
    cx1, cy1, cx2, cy2 = rec["crop"]
    m = np.zeros((FRAME, FRAME), bool)
    m[cy1:cy2, cx1:cx2] = True
    return m


def flip_variant(rgb, gt, disk, box, mode):
    """mode in {0: none, 1: horizontal, 2: vertical, 3: both}."""
    x1, y1, x2, y2 = box
    if mode in (1, 3):
        rgb, gt, disk = rgb[:, ::-1], gt[:, ::-1], disk[:, ::-1]
        x1, x2 = FRAME - x2, FRAME - x1
    if mode in (2, 3):
        rgb, gt, disk = rgb[::-1], gt[::-1], disk[::-1]
        y1, y2 = FRAME - y2, FRAME - y1
    return (np.ascontiguousarray(rgb), np.ascontiguousarray(gt),
            np.ascontiguousarray(disk), [x1, y1, x2, y2])


# --------------------------------------------------------------------------- model
def load_model(src="sam3_model", device="cuda", dtype=torch.float32):
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained(str(BASE / src))
    model = Sam3Model.from_pretrained(str(BASE / src), dtype=dtype).to(device).eval()
    return model, proc


def encode(model, proc, rgb, box, device="cuda", text=TEXT_PROMPT):
    """Run the frozen encoders. Returns (vision_embeds, text_embeds, inputs)."""
    inp = proc(images=rgb, text=text, input_boxes=[[[int(v) for v in box]]],
               return_tensors="pt").to(device)
    with torch.no_grad():
        ve = model.vision_encoder(inp["pixel_values"])
        te = model.get_text_features(input_ids=inp["input_ids"],
                                     attention_mask=inp["attention_mask"],
                                     return_dict=True)
    return ve, te, inp


def heads(model, ve, te, inp):
    """Prompt-conditioned heads only; gradients stop at the frozen encoders."""
    return model(vision_embeds=ve, text_embeds=te,
                 attention_mask=inp["attention_mask"],
                 input_boxes=inp["input_boxes"],
                 input_boxes_labels=inp["input_boxes_labels"])


def box_iou_to_prompt(pred_boxes, box):
    """pred_boxes (Q,4) normalised xyxy vs one pixel-space prompt box -> (Q,)."""
    b = torch.tensor([box[0] / FRAME, box[1] / FRAME, box[2] / FRAME, box[3] / FRAME],
                     device=pred_boxes.device, dtype=pred_boxes.dtype)
    x1 = torch.maximum(pred_boxes[:, 0], b[0]); y1 = torch.maximum(pred_boxes[:, 1], b[1])
    x2 = torch.minimum(pred_boxes[:, 2], b[2]); y2 = torch.minimum(pred_boxes[:, 3], b[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (a + ab - inter + 1e-6)


def select_query(out, box):
    """Deterministic query pick, used at train AND eval time.

    score = P(concept) * P(presence) * IoU(pred_box, prompt_box)
    """
    q = out.pred_logits[0].sigmoid()
    if out.presence_logits is not None:
        q = q * out.presence_logits[0].sigmoid()
    iou = box_iou_to_prompt(out.pred_boxes[0], box)
    return int((q * iou).argmax()), q, iou


def mask_logits_512(logits_288):
    """(H,W) or (1,1,H,W) mask logits -> (512,512) logits, differentiable."""
    x = logits_288
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None]
    return F.interpolate(x, size=(FRAME, FRAME), mode="bilinear", align_corners=False)[0, 0]


# --------------------------------------------------------------------------- metrics
def metrics(pred, gt, region, box):
    """pred/gt/region bool (512,512). All pixel metrics are region-restricted."""
    p = pred & region
    g = gt & region
    inter = float((p & g).sum()); union = float((p | g).sum())
    tp = inter; fp = float((p & ~g).sum()); fn = float((g & ~p).sum())
    x1, y1, x2, y2 = box
    return {
        "iou": inter / union if union else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
        "precision": tp / (tp + fp) if (tp + fp) else 1.0,
        "recall": tp / (tp + fn) if (tp + fn) else 1.0,
        "pred_area": float(p.sum()),
        "gt_area": float(g.sum()),
        "delta_area": (float(p.sum()) - float(g.sum())) / max(1.0, float(g.sum())),
        "cov_box": float(p[y1:y2, x1:x2].sum()) / max(1, (x2 - x1) * (y2 - y1)),
    }


def boundary_f1(pred, gt, region, tol=2):
    """Boundary F1 at `tol` px, region-restricted."""
    def edge(m):
        u = m.astype(np.uint8)
        return (cv2.morphologyEx(u, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0) & region
    ep, eg = edge(pred & region), edge(gt & region)
    if not ep.any() and not eg.any():
        return 1.0
    if not ep.any() or not eg.any():
        return 0.0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    dp = cv2.dilate(ep.astype(np.uint8), k) > 0
    dg = cv2.dilate(eg.astype(np.uint8), k) > 0
    prec = (ep & dg).sum() / ep.sum()
    rec = (eg & dp).sum() / eg.sum()
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def aggregate(rows, key="iou"):
    v = np.array([r[key] for r in rows], float)
    return dict(mean=float(v.mean()), median=float(np.median(v)),
                std=float(v.std()), min=float(v.min()), max=float(v.max()))


# --------------------------------------------------------------------------- splits
def day_folds(recs, k=5, seed=0):
    """Group-k-fold by observation day: no day appears in two folds.

    72 boxes / 65 days, so this also keeps sibling boxes of one image together.
    """
    days = sorted({r["day"] for r in recs})
    rng = np.random.RandomState(seed)
    rng.shuffle(days)
    assign = {d: i % k for i, d in enumerate(days)}
    folds = [[] for _ in range(k)]
    for i, r in enumerate(recs):
        folds[assign[r["day"]]].append(i)
    return folds
