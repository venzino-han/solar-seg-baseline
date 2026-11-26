#!/usr/bin/env python3
"""Coronal hole: majority vote over the retry variants (retry_unlabeled_ch.py) that pass the confidence gate."""
import json, numpy as np, cv2
import xml.etree.ElementTree as ET
from pathlib import Path
B = Path(__file__).resolve().parent; O = B / "retry_unlabeled/coronal_hole"; V = json.load(open(O / "variants.json"))
M = np.unpackbits(np.load(O / "masks.npz")["m"]).reshape(-1, 512, 512) > 0; ok = np.array([v["ok"] for v in V]) > 0
maj = M[ok].mean(0) >= 0.5
stem = "20130207_120018_SDO_AIA_193_512"; bgr = cv2.imread(str(B.parent / f"dataset/train/{stem}.jpg")); g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
bx = list(ET.parse(B.parent / f"dataset/train/{stem}.xml").getroot().iter("bndbox"))[2]; x1, y1, x2, y2 = [int(float(bx.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")]
S = dict(stem=stem, box_index=2, box=[x1, y1, x2, y2], n_variants=len(V), n_pass=int(ok.sum()), area=int(maj.sum()), cov=round(float(maj.sum() / ((x2 - x1) * (y2 - y1))), 3),
         iou_majority_vs_passing=[round(float((maj & m).sum() / (maj | m).sum()), 3) for m in M[ok]], iou_majority_vs_first_attempt=round(float((maj & M[0]).sum() / (maj | M[0]).sum()), 3),
         rel_intensity=round(float(np.median(g[maj]) / np.median(g[y1:y2, x1:x2][~maj[y1:y2, x1:x2]])), 3))
json.dump(S, open(O / "summary.json", "w"), indent=1); print(S); cv2.imwrite(str(O / "majority_mask.png"), maj.astype(np.uint8) * 255)
p = 40; a, b, c, d = max(0, x1 - p), max(0, y1 - p), min(512, x2 + p), min(512, y2 + p); s = 4
def t(m, col):
    im = cv2.resize(bgr[b:d, a:c], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC); cv2.rectangle(im, ((x1 - a) * s, (y1 - b) * s), ((x2 - a) * s, (y2 - b) * s), (0, 255, 255), 1)
    if m is not None:
        cn, _ = cv2.findContours(cv2.resize(m[b:d, a:c].astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cn, -1, col, 2)
    return im
cv2.imwrite(str(O / "sheet.jpg"), np.hstack([t(None, 0), t(M[0], (0, 0, 255)), t(maj, (0, 255, 0))]))
