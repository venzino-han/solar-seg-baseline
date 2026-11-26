#!/usr/bin/env python3
"""LAYERED synthetic prominences: 2-4 shapes stacked at one limb location, each with its own
source (human-shape brush remix / straight geo / arc geo), limb offset, transparency, blur or
diffusion effect and brightness. Layers are composited back-to-front on the host frame, so later
layers show semi-transparently over earlier ones. Label = union of every layer's (diffused) label.

    .venv_train/bin/python synth_prom/composite.py --n 500 --out synth_prom/dataset_comp/p0 --seed 100
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

B = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(B / "synth_prom"))
import label_other_classes as L
import synth as S
import geo_shapes as G

SRC_P = {"human": 0.4, "straight": 0.3, "arc": 0.3}
TR_PATTERNS = ("independent", "increasing", "decreasing", "alternating")


def transparencies(rng, k):
    pat = TR_PATTERNS[rng.randint(len(TR_PATTERNS))]
    if pat == "independent":
        t = rng.uniform(0.0, 0.6, k)
    elif pat == "increasing":                       # back layer solid, front layers see-through
        t = np.linspace(rng.uniform(0, 0.15), rng.uniform(0.4, 0.6), k)
    elif pat == "decreasing":                       # faint veil behind, solid structure in front
        t = np.linspace(rng.uniform(0.4, 0.6), rng.uniform(0, 0.15), k)
    else:
        lo, hi = rng.uniform(0, 0.15), rng.uniform(0.35, 0.6)
        t = np.array([lo if i % 2 == 0 else hi for i in range(k)])
    return pat, [float(x) for x in t]


def layer_shape(rng, tpls):
    src = rng.choice(list(SRC_P), p=list(SRC_P.values()))
    if src == "human":
        fp, t, keys = S.make_shape(rng, tpls)
        return fp, t, f"human:{t.rec['lab']}"
    fp, g, kind = G.make_geo(rng, src)
    return fp, tpls[rng.randint(len(tpls))], f"geo:{g}:{kind}"


def composite_one(rng, tpls, host, xml, margins, kmin=2, kmax=4, fx_prob=0.35):
    cx, cy, R, rmap = L.solar_region_colour(host)
    avoid = S.real_box_angles(xml, cx, cy)
    hsv = cv2.cvtColor(host, cv2.COLOR_BGR2HSV)
    text = cv2.dilate(((hsv[..., 1] < 70) & (hsv[..., 2] > 170) & (rmap > R + 3)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    lev_host = host.astype(np.float32).sum(2)
    k = int(rng.choice(np.arange(kmin, kmax + 1), p=[0.45, 0.35, 0.2][: kmax - kmin + 1] / np.sum([0.45, 0.35, 0.2][: kmax - kmin + 1])))
    for _ in range(40):
        # ---- shapes on the canvas, each nudged along the limb
        layers = []
        for j in range(k):
            fp, t, src = layer_shape(rng, tpls)
            shift = 0 if j == 0 else int(rng.uniform(-35, 35))
            if shift:
                fp = np.roll(fp, shift, 1)
                if shift > 0: fp[:, :shift] = False
                else: fp[:, shift:] = False
            if fp.any():
                layers.append((fp, t, src))
        if len(layers) < 2:
            continue
        union = np.any([l[0] for l in layers], 0)
        th0 = rng.uniform(-np.pi, np.pi)
        cols = np.nonzero(union.any(0))[0]
        half = (max(abs(cols.min() - S.UHALF), abs(cols.max() - S.UHALF)) + 6) / R
        if any(abs(np.angle(np.exp(1j * (th0 - c)))) < half + w + 0.08 for c, w in avoid):
            continue
        rr, cc = S.roll_maps(cx, cy, R, th0)
        valid = (rr >= 0) & (rr < S.VMAX - S.VMIN) & (cc >= 0) & (cc < 2 * S.UHALF)
        umask = (map_coordinates(union.astype(np.float32), [rr, cc], order=0) > 0.5) & valid
        if umask.sum() < 60 or (rmap[umask] > R).mean() < 0.4:
            continue
        ys, xs = np.nonzero(umask)
        if xs.min() < 3 or ys.min() < 3 or xs.max() > 508 or ys.max() > 508:
            continue
        if (cv2.dilate(umask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)[text].any():
            continue
        # ---- render layers back-to-front
        ang = np.arctan2(*(np.mgrid[0:512, 0:512] - np.array([[[cy]], [[cx]]])))
        band = (rmap > R - 8) & (rmap < R - 2) & (np.abs(np.angle(np.exp(1j * (ang - th0)))) * R < 60)
        band_level = float(np.median(lev_host[band])) if band.sum() > 10 else 300.0
        pat, trs = transparencies(rng, len(layers))
        img = host.copy(); mask = np.zeros((512, 512), bool); meta_layers = []; fail = False; lmasks = []
        for (fp, t, src), tr in zip(layers, trs):
            m_j = (map_coordinates(fp.astype(np.float32), [rr, cc], order=0) > 0.5) & valid
            fpi = map_coordinates(gaussian_filter(fp.astype(np.float32), 0.9), [rr, cc], order=1) * valid
            lr = float(np.clip((t.rec["level_ratio"] or 0.35) * 1.65 * rng.uniform(0.8, 1.2), 0.3, 1.4))
            tex = map_coordinates(S.canvas_texture(rng), [rr, cc], order=1, mode="nearest")
            kind = S.EFFECTS[rng.randint(3)] if rng.rand() < fx_prob else None
            stg = float(rng.uniform(0.2, 1.0)) if kind else None
            eff, eprm = S.make_effect(kind, stg, (cx, cy, R, th0, rr, cc, valid))
            geo = src.startswith("geo")
            img, lab_field = S.render(rng, img, fpi, rmap - R, lr, band_level, tex, tr, eff if kind else None,
                                      (90, 170) if geo else (45, 95))
            if kind:
                m_j = m_j | ((lab_field > S.LABEL_TAU) & valid)
            # each layer must add something visible where it is not hidden behind an earlier layer
            own = m_j & ~mask & (rmap > R + 1)
            if own.sum() > 30:
                gain = img.astype(np.float32).sum(2) - lev_host
                if (gain[own] > 20).mean() < 0.4:
                    fail = True; break
            mask |= m_j; lmasks.append(m_j)
            meta_layers.append(dict(src=src, transparency=round(tr, 3), effect=kind, strength=stg, area=int(m_j.sum())))
        if fail or mask[text].any():
            continue
        gain = img.astype(np.float32).sum(2) - lev_host
        offl = mask & (rmap > R + 1)
        if offl.sum() < 40 or (gain[offl] > 20).mean() < 0.6:
            continue
        ys, xs = np.nonzero(mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        w, h = x2 - x1, y2 - y1
        mL, mR, mT, mB = margins[rng.randint(len(margins))]
        box = [int(np.clip(x1 - mL * w, 0, 511)), int(np.clip(y1 - mT * h, 0, 511)),
               int(np.clip(x2 + mR * w, 1, 512)), int(np.clip(y2 + mB * h, 1, 512))]
        return img, mask, box, dict(n_layers=len(meta_layers), tr_pattern=pat, layers=meta_layers, th0=float(th0),
                                   area=int(mask.sum()), ondisk=float((rmap[mask] < R).mean()), _layer_masks=lmasks)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = np.random.RandomState(a.seed); out = B / a.out
    for s in ("images", "masks"): (out / s).mkdir(parents=True, exist_ok=True)
    tpls = S.load_templates(); mg = S.box_margins(); H = S.hosts("train"); rows = []
    while len(rows) < a.n:
        xml, jpg = H[rng.randint(len(H))]
        r = composite_one(rng, tpls, cv2.imread(jpg), xml, mg)
        if r is None: continue
        img, mask, box, meta = r; sid = f"cmp{a.seed}_{len(rows):04d}"
        cv2.imwrite(str(out / "images" / f"{sid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "masks" / f"{sid}.png"), mask.astype(np.uint8) * 255)
        meta.pop("_layer_masks", None)
        rows.append(dict(sid=sid, host=Path(jpg).stem, box=box, **meta))
        if len(rows) % 50 == 0: print(" ", len(rows), flush=True)
    json.dump(rows, open(out / "index.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
