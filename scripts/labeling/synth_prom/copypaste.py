#!/usr/bin/env python3
"""HUMAN-ANCHORED copy-paste synthesis: real prominence pixels + the human mask, moved and deformed.

For each human label (33 = 32 earlier, with the targets re-labelled by labeller_1 replaced by those labels):
  1. unroll the SOURCE frame around the label's limb angle into (v = height above limb, u = along limb);
     both the brightness (relative to the source's own limb band) and the human mask are unrolled;
  2. expand: random mirror, scale (0.7-1.4 along u, 0.7-1.5 in height, root kept on the limb), shear,
     smooth elastic warp, brightness x0.75-1.3, optional partial transparency; 35 % of samples paste TWO
     different human prominences side by side;
  3. paste onto a TRAIN-split host frame at a free limb angle: the source brightness (relative to the
     source limb) is re-expressed relative to the HOST limb and drawn through the host's own AIA-304 colour
     map, blended with a feathered version of the human mask (never darkening the host);
  4. label = the same transform applied to the human mask (as painted: roots inside the limb kept).
--keep restricts the sources (cross-validation: only the training fold's labels).

    .venv_train/bin/python synth_prom/copypaste.py --n 200 --out synth_prom/dataset_copypaste/p0 --seed 1100
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import cv2, numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
B = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(B)); sys.path.insert(0, str(B / "synth_prom"))
import label_other_classes as L
import synth as S
DATA = B.parent / "dataset"
VMIN, VMAX, UHALF = S.VMIN, S.VMAX, S.UHALF
H_, W_ = VMAX - VMIN, 2 * UHALF


def human_sources():
    """33 effective human labels: labeller_5 / labeller_6, with labeller_1 replacing earlier labels of the same target."""
    g = [r for r in csv.DictReader(open(B / "gold_prominence/labels.csv")) if r["evaluator"] in ("labeller_5", "labeller_6")]
    k = list(csv.DictReader(open(B / "gold_prominence_r2/labels.csv")))
    kk = {(r["stem"], r["bi"]) for r in k}
    out = [dict(r, root="gold_prominence") for r in g if (r["stem"], r["bi"]) not in kk] + [dict(r, root="gold_prominence_r2") for r in k]
    for r in out: r["key"] = f"{r['stem']}__b{r['bi']}"
    return out


def canvas_coords(cx, cy, R, th0):
    vv, uu = np.mgrid[VMIN:VMAX, -UHALF:UHALF].astype(np.float32)
    th = th0 + uu / R
    return cy + (R + vv) * np.sin(th), cx + (R + vv) * np.cos(th)


class Source:
    def __init__(self, r):
        bgr = cv2.imread(str(DATA / r["split"] / f"{r['stem']}.jpg")); m = cv2.imread(str(B / r["root"] / r["mask_path"]), 0) > 127
        cx, cy, R, rmap = L.solar_region_colour(bgr)
        ys, xs = np.nonzero(m); th0 = float(np.angle(np.exp(1j * np.arctan2(ys - cy, xs - cx)).mean()))
        y, x = canvas_coords(cx, cy, R, th0)
        lev = bgr.astype(np.float32).sum(2)
        ang = np.arctan2(*(np.mgrid[0:512, 0:512] - np.array([[[cy]], [[cx]]])))
        band = (rmap > R - 8) & (rmap < R - 2) & (np.abs(np.angle(np.exp(1j * (ang - th0)))) * R < 60)
        self.band = float(np.median(lev[band])) if band.sum() > 10 else float(np.median(lev[rmap < R]))
        self.rel = map_coordinates(lev, [y, x], order=1, mode="constant") / max(1.0, self.band)    # brightness / source limb
        self.mask = map_coordinates(m.astype(np.float32), [y, x], order=0, mode="constant") > 0.5
        self.key, self.lab = r["key"], r["evaluator"]
        cols = np.nonzero(self.mask.any(0))[0]; self.width = (cols.max() - cols.min()) if len(cols) else 0


def warp(rng, src: Source, shift_u=0.0):
    """Expanded copy on the canvas: (relative brightness, mask)."""
    vv, uu = np.mgrid[VMIN:VMAX, -UHALF:UHALF].astype(np.float32)
    su, sv = rng.uniform(0.7, 1.4), rng.uniform(0.7, 1.5); mir = -1.0 if rng.rand() < 0.5 else 1.0
    shear = rng.uniform(-0.35, 0.35)
    du = gaussian_filter(rng.standard_normal((H_, W_)), 12) ; du = du / (np.abs(du).max() + 1e-6) * rng.uniform(1, 5)
    dv = gaussian_filter(rng.standard_normal((H_, W_)), 12) ; dv = dv / (np.abs(dv).max() + 1e-6) * rng.uniform(0.5, 3)
    lift = np.clip(vv / 10.0, 0, 1)                                   # roots stay glued to the limb
    v_src = np.where(vv > 0, vv / sv, vv) + dv * lift
    u_src = mir * ((uu - shift_u) / su - shear * np.clip(vv, 0, None)) + du * lift
    r_i, c_i = v_src - VMIN, u_src + UHALF
    rel = map_coordinates(src.rel, [r_i, c_i], order=1, mode="constant")
    msk = map_coordinates(src.mask.astype(np.float32), [r_i, c_i], order=0, mode="constant") > 0.5
    return rel, msk


def copypaste_one(rng, sources, host, xml, margins):
    cx, cy, R, rmap = L.solar_region_colour(host)
    avoid = S.real_box_angles(xml, cx, cy)
    hsv = cv2.cvtColor(host, cv2.COLOR_BGR2HSV)
    text = cv2.dilate(((hsv[..., 1] < 70) & (hsv[..., 2] > 170) & (rmap > R + 3)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    lev_host = host.astype(np.float32).sum(2)
    for _ in range(40):
        two = rng.rand() < 0.35 and len(sources) > 1
        picks = list(rng.choice(len(sources), 2 if two else 1, replace=False))
        rels, msks, used = [], [], []
        off = 0.0
        for j, i in enumerate(picks):
            s = sources[i]
            if j == 1: off = (sources[picks[0]].width / 2 + s.width / 2) * rng.uniform(0.55, 0.95) * (1 if rng.rand() < .5 else -1)
            rel, msk = warp(rng, s, off); rels.append(rel); msks.append(msk); used.append(s.key)
        union = np.any(msks, 0)
        if union.sum() < 40: continue
        th0 = rng.uniform(-np.pi, np.pi); cols = np.nonzero(union.any(0))[0]
        half = (max(abs(cols.min() - UHALF), abs(cols.max() - UHALF)) + 6) / R
        if any(abs(np.angle(np.exp(1j * (th0 - c)))) < half + w + 0.08 for c, w in avoid): continue
        rr, cc = S.roll_maps(cx, cy, R, th0)
        valid = (rr >= 0) & (rr < H_) & (cc >= 0) & (cc < W_)
        mask = (map_coordinates(union.astype(np.float32), [rr, cc], order=0) > 0.5) & valid
        if mask.sum() < 40 or (rmap[mask] > R).mean() < 0.4: continue
        ys, xs = np.nonzero(mask)
        if xs.min() < 3 or ys.min() < 3 or xs.max() > 508 or ys.max() > 508: continue
        if (cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)[text].any(): continue
        ang = np.arctan2(*(np.mgrid[0:512, 0:512] - np.array([[[cy]], [[cx]]])))
        band = (rmap > R - 8) & (rmap < R - 2) & (np.abs(np.angle(np.exp(1j * (ang - th0)))) * R < 60)
        band_h = float(np.median(lev_host[band])) if band.sum() > 10 else 300.0
        gain = rng.uniform(0.85, 1.35); opac = 1.0 if rng.rand() < 0.7 else rng.uniform(0.6, 0.9)
        # paste each source in turn (canvas -> image), feathered by its own mask
        lut = S.colour_lut(host); img_lev = lev_host.copy()
        for rel, msk in zip(rels, msks):
            a_c = gaussian_filter(cv2.dilate(msk.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), 1.2)
            a = np.clip(map_coordinates(a_c, [rr, cc], order=1) * valid, 0, 1) * opac
            src_lev = map_coordinates(rel, [rr, cc], order=1) * band_h * gain
            src_lev = np.where(rmap < R - 8, np.minimum(src_lev, lev_host * 1.15), src_lev)   # no glowing patches on the disk
            img_lev = np.maximum(img_lev, img_lev * (1 - a) + src_lev * a)
        # add only the colour CHANGE implied by the brightness lift (keeps host texture & hue, no limb artefacts)
        dcol = lut[np.clip(img_lev, 0, 765).astype(int)] - lut[np.clip(lev_host, 0, 765).astype(int)]
        out = host.astype(np.float32) + np.where((img_lev > lev_host + 0.5)[..., None], dcol, 0)
        ok, enc = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92]); img = cv2.imdecode(enc, 1)
        offl = mask & (rmap > R + 1)
        if offl.sum() < 60 or ((img.astype(np.float32).sum(2) - lev_host)[offl] > 25).mean() < 0.6: continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1; w, h = x2 - x1, y2 - y1
        mL, mR, mT, mB = margins[rng.randint(len(margins))]
        box = [int(np.clip(x1 - mL * w, 0, 511)), int(np.clip(y1 - mT * h, 0, 511)), int(np.clip(x2 + mR * w, 1, 512)), int(np.clip(y2 + mB * h, 1, 512))]
        return img, mask, box, dict(sources=used, n_sources=len(used), gain=round(gain, 3), opacity=round(opac, 3), area=int(mask.sum()), th0=float(th0))
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--keep", default=None, help="쉼표로 구분한 사람 레이블 key (교차검증용)")
    a = ap.parse_args(); rng = np.random.RandomState(a.seed); out = B / a.out
    for d in ("images", "masks"): (out / d).mkdir(parents=True, exist_ok=True)
    srcs = human_sources()
    if a.keep: keep = set(a.keep.split(",")); srcs = [r for r in srcs if r["key"] in keep]
    sources = [Source(r) for r in srcs]
    mg = S.box_margins(); H = S.hosts("train"); rows = []
    while len(rows) < a.n:
        xml, jpg = H[rng.randint(len(H))]
        r = copypaste_one(rng, sources, cv2.imread(jpg), xml, mg)
        if r is None: continue
        img, mask, box, meta = r; sid = f"cp{a.seed}_{len(rows):04d}"
        cv2.imwrite(str(out / "images" / f"{sid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "masks" / f"{sid}.png"), mask.astype(np.uint8) * 255)
        rows.append(dict(sid=sid, host=Path(jpg).stem, box=box, **meta))
        if len(rows) % 50 == 0: print(" ", len(rows), flush=True)
    json.dump(dict(sources=[s.key for s in sources], rows=rows), open(out / "index_meta.json", "w"), ensure_ascii=False, indent=1)
    json.dump(rows, open(out / "index.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
