#!/usr/bin/env python3
"""HARD case: several overlapping ARC layers in front of a soft, diffuse GRADIENT (haze).

The haze is rendered into the image but is NOT part of the label -- the label is the exact union of the
arc footprints only. The model therefore has to separate thin bright arcs from a smooth glow behind them.

Haze kinds (one per sample, drawn BEFORE the arcs):
  glow  : heavily blurred envelope around the arcs (broad halo)
  band  : limb-hugging band that fades outward (exp fall-off in height) over the arcs' angular span
  cloud : 1-3 offset gaussian clouds partly overlapping the arcs
Haze brightness 35-75 % of the arc level, opacity 45-85 %, visible lift >= 25 (sum of BGR). Arcs (3-8, thickness ~1.5-6 px, own brightness and
transparency) are rendered on top. A sample is kept only if the arcs are clearly brighter than the haze-only
image (contrast gate), and the usual text / visibility / placement gates of strands.py apply.

    .venv_train/bin/python synth_prom/haze_arcs.py --n 150 --out synth_prom/dataset_hazearcs/p0 --seed 700
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
B = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(B)); sys.path.insert(0, str(B / "synth_prom"))
import label_other_classes as L
import synth as S, strands as ST, fine_arcs as FA
from geo_shapes import blank, stamp_path

HAZE = ("glow", "band", "cloud")
HAZE_KO = {"glow": "빛무리", "band": "림브 띠", "cloud": "확산 구름"}


def make_arcs(rng):
    fam = rng.choice(list(FA.FAM_W), p=list(FA.FAM_W.values())); want = ST.target_area(rng) * rng.uniform(0.8, 1.5); s = 1.0
    seed = rng.randint(1 << 30)
    for _ in range(5):
        r2 = np.random.RandomState(seed); arcs = FA.FAMS[fam](r2, s); fps = []
        for j, pts in enumerate(arcs):
            r0 = r2.uniform(0.75, 3.0)                                   # thickness ~1.5-6 px
            c = blank(); stamp_path(c, pts, r0, r0 * r2.uniform(0.6, 1.0), np.random.RandomState(seed + j), 0.15); fps.append(c > 127)
        area = np.any(fps, 0).sum()
        if area == 0: return None
        k = np.sqrt(want / area)
        if abs(k - 1) < 0.1: break
        s = float(np.clip(s * np.clip(k, 0.6, 1.6), 0.9, 2.2))
    union = np.any(fps, 0).sum(); total = sum(f.sum() for f in fps)
    if total == 0 or 1 - union / total > 0.35: return None
    return fps, fam, want


def haze_field(rng, union, kind):
    """Soft 0..1 field on the limb-unrolled canvas (rows = height above limb)."""
    H, W = union.shape
    if kind == "glow":
        f = gaussian_filter(cv2.dilate(union.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32), rng.uniform(7, 16))
    elif kind == "band":
        cols = np.nonzero(union.any(0))[0]; c0, c1 = cols.min() - rng.uniform(5, 30), cols.max() + rng.uniform(5, 30)
        v = np.arange(H)[:, None] + S.VMIN; u = np.arange(W)[None, :]
        prof_u = 1 / (1 + np.exp(-(u - c0) / 6)) * 1 / (1 + np.exp((u - c1) / 6))
        f = np.exp(-np.clip(v, 0, None) / rng.uniform(18, 45)) * (v > -4) * prof_u
    else:
        ys, xs = np.nonzero(union); f = np.zeros((H, W), np.float32)
        for _ in range(rng.randint(1, 4)):
            i = rng.randint(len(ys)); cy, cx = ys[i] + rng.uniform(-15, 15), xs[i] + rng.uniform(-25, 25)
            sy, sx = rng.uniform(8, 22), rng.uniform(12, 35)
            yy, xx = np.mgrid[0:H, 0:W]; f = np.maximum(f, np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2)))
    f = f / (f.max() + 1e-6)
    return gaussian_filter(f, 2.0).astype(np.float32)


def haze_arcs_one(rng, host, xml, margins):
    cx, cy, R, rmap = L.solar_region_colour(host)
    avoid = S.real_box_angles(xml, cx, cy)
    hsv = cv2.cvtColor(host, cv2.COLOR_BGR2HSV)
    text = cv2.dilate(((hsv[..., 1] < 70) & (hsv[..., 2] > 170) & (rmap > R + 3)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    lev_host = host.astype(np.float32).sum(2)
    for _ in range(40):
        r = make_arcs(rng)
        if r is None: continue
        fps, fam, want = r; union = np.any(fps, 0)
        kind = HAZE[rng.randint(3)]; hz = haze_field(rng, union, kind)
        th0 = rng.uniform(-np.pi, np.pi); cols = np.nonzero((union | (hz > 0.05)).any(0))[0]
        half = (max(abs(cols.min() - S.UHALF), abs(cols.max() - S.UHALF)) + 6) / R
        if any(abs(np.angle(np.exp(1j * (th0 - c)))) < half + w + 0.08 for c, w in avoid): continue
        rr, cc = S.roll_maps(cx, cy, R, th0)
        valid = (rr >= 0) & (rr < S.VMAX - S.VMIN) & (cc >= 0) & (cc < 2 * S.UHALF)
        umask = (map_coordinates(union.astype(np.float32), [rr, cc], order=0) > 0.5) & valid
        if umask.sum() < 60 or (rmap[umask] > R).mean() < 0.5: continue
        ys, xs = np.nonzero(umask)
        if xs.min() < 3 or ys.min() < 3 or xs.max() > 508 or ys.max() > 508: continue
        hzi = map_coordinates(hz, [rr, cc], order=1) * valid
        if (cv2.dilate(((hzi > 0.15) | umask).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)[text].any(): continue
        ang = np.arctan2(*(np.mgrid[0:512, 0:512] - np.array([[[cy]], [[cx]]])))
        band = (rmap > R - 8) & (rmap < R - 2) & (np.abs(np.angle(np.exp(1j * (ang - th0)))) * R < 60)
        band_level = float(np.median(lev_host[band])) if band.sum() > 10 else 300.0
        tpl_lr = [0.44, 0.46, 0.5, 0.51, 0.52, 0.56, 0.61, 0.67, 0.68, 0.74, 0.75, 0.78]
        arc_lr = float(np.clip(rng.choice(tpl_lr) * 1.65 * rng.uniform(0.85, 1.25), 0.4, 1.4))
        # ---- haze first (not labelled)
        hz_lr = arc_lr * rng.uniform(0.35, 0.75); hz_tr = rng.uniform(0.15, 0.55)
        tex_h = map_coordinates(gaussian_filter(S.canvas_texture(rng), 3), [rr, cc], order=1, mode="nearest")
        img, _ = S.render(rng, host.copy(), hzi, rmap - R, hz_lr, band_level, tex_h, hz_tr, None, (90, 170))
        lev_haze = img.astype(np.float32).sum(2)
        # ---- arcs on top (labelled)
        mask = np.zeros((512, 512), bool); meta = []
        for fp in fps:
            m_j = (map_coordinates(fp.astype(np.float32), [rr, cc], order=0) > 0.5) & valid
            fpi = map_coordinates(gaussian_filter(fp.astype(np.float32), 0.6), [rr, cc], order=1) * valid
            lr = float(np.clip(arc_lr * rng.uniform(0.8, 1.2), 0.35, 1.5)); tr = float(rng.uniform(0.0, 0.35))
            tex = map_coordinates(S.canvas_texture(rng), [rr, cc], order=1, mode="nearest")
            img, _ = S.render(rng, img, fpi, rmap - R, lr, band_level, tex, tr, None, (90, 170))
            mask |= m_j; meta.append(dict(transparency=round(tr, 3), area=int(m_j.sum())))
        if mask[text].any(): continue
        lev = img.astype(np.float32).sum(2); offl = mask & (rmap > R + 1)
        if offl.sum() < 40: continue
        if ((lev - lev_haze)[offl] > 20).mean() < 0.6: continue          # arcs must stand out above the haze
        hz_only = (hzi > 0.2) & ~cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (rmap > R + 1)
        haze_gain = float(np.median((lev_haze - lev_host)[hz_only])) if hz_only.sum() > 30 else 0.0
        if haze_gain < 25: continue                                      # haze must actually be visible
        ys, xs = np.nonzero(mask); x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1; w, h = x2 - x1, y2 - y1
        mL, mR, mT, mB = margins[rng.randint(len(margins))]
        box = [int(np.clip(x1 - mL * w, 0, 511)), int(np.clip(y1 - mT * h, 0, 511)), int(np.clip(x2 + mR * w, 1, 512)), int(np.clip(y2 + mB * h, 1, 512))]
        return img, mask, box, (hzi > 0.2), dict(family=fam, haze=kind, haze_level=round(hz_lr / arc_lr, 3), haze_transparency=round(hz_tr, 3),
                                                  haze_gain=round(haze_gain, 1), haze_area=int((hzi > 0.2).sum()), n_strands=len(fps), strands=meta,
                                                  area=int(mask.sum()), th0=float(th0))
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--split", default="train", help="호스트 프레임 분할 (평가용은 test)")
    a = ap.parse_args(); rng = np.random.RandomState(a.seed); out = B / a.out
    for d in ("images", "masks", "haze"): (out / d).mkdir(parents=True, exist_ok=True)
    mg = S.box_margins(); H = S.hosts(a.split); rows = []
    while len(rows) < a.n:
        xml, jpg = H[rng.randint(len(H))]
        r = haze_arcs_one(rng, cv2.imread(jpg), xml, mg)
        if r is None: continue
        img, mask, box, hz, meta = r; sid = f"ha{a.seed}_{len(rows):04d}"
        cv2.imwrite(str(out / "images" / f"{sid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "masks" / f"{sid}.png"), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "haze" / f"{sid}.png"), hz.astype(np.uint8) * 255)       # for inspection only, not a label
        rows.append(dict(sid=sid, host=Path(jpg).stem, host_split=a.split, box=box, **meta))
        if len(rows) % 50 == 0: print(" ", len(rows), flush=True)
    json.dump(rows, open(out / "index.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
