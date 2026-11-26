#!/usr/bin/env python3
"""MULTI-STRAND synthetic prominences, sized after the new labeller_1 labels.

Target morphology (shape_stats.py on the 15 new labels, median [p10-p90]):
    area 1036 px [511-1505], stroke thickness 6.3 px [4.0-12.5], skeleton/thickness 22 [-37],
    60 % with enclosed holes (1-3 holes, total 10 [-66] px), ~6 skeleton end points.

Generator (limb-unrolled canvas of synth.py, root at the limb):
  - 2-6 strands, each a smooth random curve 30-120 px long, own brush radius (thickness ~2.5-11 px),
    tapered toward the tip, mild knots;
  - families: fan of threads / braid (two strands crossing repeatedly -> small holes) /
    arch with split legs / hedgerow of parallel threads / branching tree;
  - gaps between strands are kept OUT of the label, so the model must separate strands;
  - each strand is rendered as its own layer (own brightness and transparency 0-40 %), no blur, so
    the gaps stay dark;
  - total area is rescaled toward a target drawn from the new labels' area distribution.
Label = exact union of strand footprints (no margin, no diffusion).

    .venv_train/bin/python synth_prom/strands.py --n 200 --out synth_prom/dataset_strands/p0 --seed 300
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
from geo_shapes import P, blank, bezier, stamp_path, SS

R2_AREA = None


def target_area(rng):
    global R2_AREA
    if R2_AREA is None:
        import csv
        R2_AREA = [int(r["area_px"]) for r in csv.DictReader(open(B / "gold_prominence_r2/labels.csv"))]
    return float(rng.choice(R2_AREA)) * rng.uniform(0.85, 1.55)   # canvas->frame mapping + limb clipping shrink it ~20 %


def curve(rng, p0, length, heading, wiggle):
    """Smooth random curve from p0 (u, v): heading in radians from +v (radial), curvature random-walk."""
    n = max(8, int(length / 2)); step = length / n
    h = heading; pts = [p0]
    dh = gaussian_filter(rng.standard_normal(n), 2.0) * wiggle
    for k in range(n):
        h += dh[k]
        u, v = pts[-1]; pts.append((u + step * np.sin(h), v + step * np.cos(h)))
    return np.array(pts)


def strand(can, rng, pts, r0, taper=True):
    r1 = r0 * (rng.uniform(0.35, 0.8) if taper else rng.uniform(0.9, 1.1))
    stamp_path(can, pts, r0, r1, rng, rng.uniform(0.0, 0.2))


def fam_fan(rng, s):
    k = rng.randint(2, 6); base = rng.uniform(-4, 4); out = []
    for i in range(k):
        h = np.deg2rad(rng.uniform(-45, 45)); L_ = rng.uniform(35, 120) * s
        out.append((curve(rng, (base + rng.uniform(-6, 6), rng.uniform(-3, 0)), L_, h, rng.uniform(0.03, 0.12)), rng.uniform(1.2, 4.5) * s))
    return out


def fam_braid(rng, s):
    L_ = rng.uniform(50, 120) * s; n = int(L_ / 2); v = np.linspace(rng.uniform(-3, 0), L_, n)
    amp = rng.uniform(4.5, 10) * s; per = rng.uniform(22, 50) * s; tilt = rng.uniform(-0.5, 0.5); out = []
    for ph in (0, np.pi):
        u = tilt * v + amp * np.sin(2 * np.pi * v / per + ph)
        out.append((np.c_[u, v], rng.uniform(1.2, 3.2) * s))
    if rng.rand() < 0.4:
        out.append((curve(rng, (rng.uniform(-6, 6), rng.uniform(-3, 0)), rng.uniform(30, 80) * s, rng.uniform(-0.7, 0.7), 0.08), rng.uniform(1.0, 2.5) * s))
    return out


def fam_arch_legs(rng, s):
    w = rng.uniform(25, 80) * s; h = rng.uniform(25, 75) * s; sk = rng.uniform(-0.4, 0.4) * w; out = []
    top = bezier((-w / 2, 0), (-w / 2 + sk, h * 1.3), (w / 2 + sk, h * 1.3 * rng.uniform(0.7, 1.1)), (w / 2, 0))
    out.append((top, rng.uniform(1.5, 4.0) * s))
    for side in (-1, 1):                                  # split feet -> enclosed gaps near the limb
        if rng.rand() < 0.7:
            u0 = side * w / 2 + side * rng.uniform(4, 10) * s
            out.append((curve(rng, (u0, rng.uniform(-3, 0)), rng.uniform(15, 40) * s, -side * rng.uniform(0.1, 0.5), 0.05), rng.uniform(1.0, 2.5) * s))
    return out


def fam_hedgerow(rng, s):
    k = rng.randint(3, 7); gap = rng.uniform(7, 15) * s; tilt = rng.uniform(-0.4, 0.4); out = []
    for i in range(k):
        out.append((curve(rng, ((i - (k - 1) / 2) * gap, rng.uniform(-3, 0)), rng.uniform(25, 90) * s, tilt + rng.uniform(-0.15, 0.15), rng.uniform(0.02, 0.07)), rng.uniform(1.0, 3.0) * s))
    if rng.rand() < 0.7:                                  # canopy linking the tops -> holes between threads
        top = max(p[:, 1].max() for p, _ in out) * rng.uniform(0.6, 0.95)
        out.append((np.array([[-(k - 1) / 2 * gap - 4, top], [(k - 1) / 2 * gap + 4, top * rng.uniform(0.85, 1.15)]]), rng.uniform(1.0, 2.5) * s))
    return out


def fam_tree(rng, s):
    trunk = curve(rng, (0, rng.uniform(-3, 0)), rng.uniform(30, 70) * s, rng.uniform(-0.4, 0.4), 0.05)
    out = [(trunk, rng.uniform(2.5, 5.0) * s)]
    for _ in range(rng.randint(2, 5)):
        j = rng.randint(len(trunk) // 3, len(trunk)); h = rng.uniform(-1.2, 1.2)
        out.append((curve(rng, tuple(trunk[j]), rng.uniform(20, 60) * s, h, 0.1), rng.uniform(1.0, 2.5) * s))
    return out


def fam_split(rng, s):
    """One strand that splits into two and re-joins (1-3 times) -> small lens-shaped holes like the labels."""
    L_ = rng.uniform(45, 110) * s; n = int(L_ / 2); v = np.linspace(rng.uniform(-3, 0), L_, n)
    tilt = rng.uniform(-0.6, 0.6); base_u = tilt * v + gaussian_filter(rng.standard_normal(n), 6) * 4 * s
    sep = np.zeros(n); k = rng.randint(1, 4)
    for c in rng.uniform(0.15, 0.9, k) * n:
        w = rng.uniform(6, 14) * s / 2; sep += rng.uniform(2.5, 6) * s * np.exp(-0.5 * ((np.arange(n) - c) / w) ** 2)
    r = rng.uniform(1.5, 3.2) * s; out = [(np.c_[base_u + sep, v], r), (np.c_[base_u - sep, v], r * rng.uniform(0.7, 1.0))]
    if rng.rand() < 0.4: out += fam_fan(rng, s * 0.7)[:2]
    return out


FAMS = {"split": fam_split, "fan": fam_fan, "braid": fam_braid, "arch_legs": fam_arch_legs, "hedgerow": fam_hedgerow, "tree": fam_tree}
FAM_W = {"split": 0.3, "fan": 0.12, "braid": 0.2, "arch_legs": 0.13, "hedgerow": 0.15, "tree": 0.1}
FAM_KO = {"split": "갈라졌다 합쳐지는 줄기", "fan": "부채꼴 실가닥", "braid": "꼬인 두 줄기", "arch_legs": "아치 + 갈라진 다리", "hedgerow": "평행 실가닥", "tree": "가지 뻗은 나무형"}


def make_strands(rng):
    """-> list of per-strand canvas footprints (bool), family name, total-area target."""
    fam = rng.choice(list(FAM_W), p=list(FAM_W.values())); want = target_area(rng); s = 1.0
    seed = rng.randint(1 << 30)
    for _ in range(5):
        parts = FAMS[fam](np.random.RandomState(seed), s)
        fps = []
        for pts, r in parts:
            c = blank(); strand(c, np.random.RandomState(seed + len(fps)), pts, r * 1.35); fps.append(c > 127)
        area = np.any(fps, 0).sum() * 1.0 if fps else 0
        if area == 0: return None
        k = np.sqrt(want / area)
        if abs(k - 1) < 0.08: break
        s *= float(np.clip(k, 0.6, 1.6))
    return fps, fam, want


def strands_one(rng, host, xml, margins, min_union=60, min_off=40):
    cx, cy, R, rmap = L.solar_region_colour(host)
    avoid = S.real_box_angles(xml, cx, cy)
    hsv = cv2.cvtColor(host, cv2.COLOR_BGR2HSV)
    text = cv2.dilate(((hsv[..., 1] < 70) & (hsv[..., 2] > 170) & (rmap > R + 3)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    lev_host = host.astype(np.float32).sum(2)
    for _ in range(40):
        r = make_strands(rng)
        if r is None: continue
        fps, fam, want = r
        union = np.any(fps, 0)
        th0 = rng.uniform(-np.pi, np.pi); cols = np.nonzero(union.any(0))[0]
        half = (max(abs(cols.min() - S.UHALF), abs(cols.max() - S.UHALF)) + 6) / R
        if any(abs(np.angle(np.exp(1j * (th0 - c)))) < half + w + 0.08 for c, w in avoid): continue
        rr, cc = S.roll_maps(cx, cy, R, th0)
        valid = (rr >= 0) & (rr < S.VMAX - S.VMIN) & (cc >= 0) & (cc < 2 * S.UHALF)
        umask = (map_coordinates(union.astype(np.float32), [rr, cc], order=0) > 0.5) & valid
        if umask.sum() < min_union or (rmap[umask] > R).mean() < 0.5: continue
        ys, xs = np.nonzero(umask)
        if xs.min() < 3 or ys.min() < 3 or xs.max() > 508 or ys.max() > 508: continue
        if (cv2.dilate(umask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)[text].any(): continue
        ang = np.arctan2(*(np.mgrid[0:512, 0:512] - np.array([[[cy]], [[cx]]])))
        band = (rmap > R - 8) & (rmap < R - 2) & (np.abs(np.angle(np.exp(1j * (ang - th0)))) * R < 60)
        band_level = float(np.median(lev_host[band])) if band.sum() > 10 else 300.0
        tpl_lr = [0.44, 0.46, 0.5, 0.51, 0.52, 0.56, 0.61, 0.67, 0.68, 0.74, 0.75, 0.78]
        img = host.copy(); mask = np.zeros((512, 512), bool); meta = []
        for fp in fps:
            m_j = (map_coordinates(fp.astype(np.float32), [rr, cc], order=0) > 0.5) & valid
            fpi = map_coordinates(gaussian_filter(fp.astype(np.float32), 0.6), [rr, cc], order=1) * valid
            lr = float(np.clip(rng.choice(tpl_lr) * 1.65 * rng.uniform(0.75, 1.25), 0.3, 1.4))
            tex = map_coordinates(S.canvas_texture(rng), [rr, cc], order=1, mode="nearest")
            tr = float(rng.uniform(0.0, 0.4))
            img, _ = S.render(rng, img, fpi, rmap - R, lr, band_level, tex, tr, None, (90, 170))
            mask |= m_j; meta.append(dict(transparency=round(tr, 3), area=int(m_j.sum())))
        if mask[text].any(): continue
        gain = img.astype(np.float32).sum(2) - lev_host; offl = mask & (rmap > R + 1)
        if offl.sum() < min_off or (gain[offl] > 20).mean() < 0.6: continue
        ys, xs = np.nonzero(mask); x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1; w, h = x2 - x1, y2 - y1
        mL, mR, mT, mB = margins[rng.randint(len(margins))]
        box = [int(np.clip(x1 - mL * w, 0, 511)), int(np.clip(y1 - mT * h, 0, 511)), int(np.clip(x2 + mR * w, 1, 512)), int(np.clip(y2 + mB * h, 1, 512))]
        return img, mask, box, dict(family=fam, n_strands=len(fps), strands=meta, target_area=round(want), area=int(mask.sum()), th0=float(th0))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = np.random.RandomState(a.seed); out = B / a.out
    for s in ("images", "masks"): (out / s).mkdir(parents=True, exist_ok=True)
    mg = S.box_margins(); H = S.hosts("train"); rows = []
    while len(rows) < a.n:
        xml, jpg = H[rng.randint(len(H))]
        r = strands_one(rng, cv2.imread(jpg), xml, mg)
        if r is None: continue
        img, mask, box, meta = r; sid = f"str{a.seed}_{len(rows):04d}"
        cv2.imwrite(str(out / "images" / f"{sid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "masks" / f"{sid}.png"), mask.astype(np.uint8) * 255)
        rows.append(dict(sid=sid, host=Path(jpg).stem, box=box, **meta))
        if len(rows) % 50 == 0: print(" ", len(rows), flush=True)
    json.dump(rows, open(out / "index.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
