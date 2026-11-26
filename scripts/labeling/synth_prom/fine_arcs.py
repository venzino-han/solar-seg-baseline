#!/usr/bin/env python3
"""FINE OVERLAPPING ARCS: 3-8 thin arcs (brush thickness ~1.5-4 px) stacked/crossing at one limb location.
Families: nested bundle (shared foot region, different heights) / crossing loops (different sizes & tilts) /
fine arcade (overlapping arches along the limb) / tangled (random bezier loops through a common region).
Each arc is its own render layer (own brightness, transparency 0-35 %), gaps between arcs stay OUT of the
label (exact union of footprints, no margin). Placement / rendering / gates shared with strands.py.

    .venv_train/bin/python synth_prom/fine_arcs.py --n 150 --out synth_prom/dataset_finearcs/p0 --seed 500
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
B = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(B)); sys.path.insert(0, str(B / "synth_prom"))
import synth as S, strands as ST
from geo_shapes import blank, bezier, stamp_path


def arc_pts(rng, u0, u1, h, tilt=0.0, n=120):
    """Arch from foot u0 to foot u1 (both at the limb), apex height h, sideways skew tilt."""
    w = u1 - u0; sk = tilt * w
    return bezier((u0, rng.uniform(-2.5, 0)), (u0 + sk, h * 1.33), (u1 + sk, h * 1.33 * rng.uniform(0.8, 1.15)), (u1, rng.uniform(-2.5, 0)), n)


def fam_bundle(rng, s):
    c = rng.uniform(-5, 5); k = rng.randint(3, 7); out = []
    for i in range(k):
        w = rng.uniform(25, 70) * s * (0.7 + 0.12 * i); h = rng.uniform(18, 60) * s * (0.6 + 0.15 * i)
        out.append(arc_pts(rng, c - w / 2 + rng.uniform(-4, 4), c + w / 2 + rng.uniform(-4, 4), h, rng.uniform(-0.25, 0.25)))
    return out


def fam_crossing(rng, s):
    k = rng.randint(3, 6); out = []
    for _ in range(k):
        c = rng.uniform(-20, 20) * s; w = rng.uniform(20, 75) * s; h = rng.uniform(15, 70) * s
        out.append(arc_pts(rng, c - w / 2, c + w / 2, h, rng.uniform(-0.5, 0.5)))
    return out


def fam_arcade(rng, s):
    k = rng.randint(4, 9); span = rng.uniform(60, 130) * s; out = []
    for i in range(k):
        c = -span / 2 + span * i / max(1, k - 1) + rng.uniform(-5, 5); w = rng.uniform(18, 40) * s; h = rng.uniform(12, 40) * s
        out.append(arc_pts(rng, c - w / 2, c + w / 2, h, rng.uniform(-0.3, 0.3)))
    return out


def fam_tangle(rng, s):
    k = rng.randint(3, 7); out = []
    for _ in range(k):
        a = (rng.uniform(-25, 25) * s, rng.uniform(-2, 0)); d = (rng.uniform(-25, 25) * s, rng.uniform(-2, 30) * s)
        b = (rng.uniform(-45, 45) * s, rng.uniform(20, 80) * s); c = (rng.uniform(-45, 45) * s, rng.uniform(20, 80) * s)
        out.append(bezier(a, b, c, d, 120))
    return out


FAMS = {"bundle": fam_bundle, "crossing": fam_crossing, "arcade": fam_arcade, "tangle": fam_tangle}
FAM_W = {"bundle": 0.3, "crossing": 0.3, "arcade": 0.2, "tangle": 0.2}
FAM_KO = {"bundle": "겹친 아치 다발", "crossing": "교차하는 고리", "arcade": "가는 아케이드", "tangle": "얽힌 가는 고리"}


def make_fine(rng):
    fam = rng.choice(list(FAM_W), p=list(FAM_W.values())); want = ST.target_area(rng) * rng.uniform(0.7, 1.3); s = 1.0
    seed = rng.randint(1 << 30)
    for _ in range(5):
        r2 = np.random.RandomState(seed); arcs = FAMS[fam](r2, s); fps = []
        for j, pts in enumerate(arcs):
            r0 = r2.uniform(0.75, 2.0)                        # brush radius -> thickness ~1.5-4 px
            c = blank(); stamp_path(c, pts, r0, r0 * r2.uniform(0.6, 1.0), np.random.RandomState(seed + j), 0.15); fps.append(c > 127)
        area = np.any(fps, 0).sum()
        if area == 0: return None
        k = np.sqrt(want / area)
        if abs(k - 1) < 0.1: break
        s = float(np.clip(s * np.clip(k, 0.6, 1.6), 0.9, 2.2))  # rescale geometry (never below 0.9), NOT brush width
    union = np.any(fps, 0).sum(); total = sum(f.sum() for f in fps)
    if total == 0 or 1 - union / total > 0.35: return None  # arcs piled on top of each other -> unreadable blob
    return fps, fam, want


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = np.random.RandomState(a.seed); out = B / a.out
    for d in ("images", "masks"): (out / d).mkdir(parents=True, exist_ok=True)
    ST.make_strands = make_fine                               # reuse placement/rendering/gates of strands_one
    mg = S.box_margins(); H = S.hosts("train"); rows = []
    while len(rows) < a.n:
        xml, jpg = H[rng.randint(len(H))]
        r = ST.strands_one(rng, cv2.imread(jpg), xml, mg)
        if r is None: continue
        img, mask, box, meta = r; sid = f"fa{a.seed}_{len(rows):04d}"
        cv2.imwrite(str(out / "images" / f"{sid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "masks" / f"{sid}.png"), mask.astype(np.uint8) * 255)
        rows.append(dict(sid=sid, host=Path(jpg).stem, box=box, **meta))
        if len(rows) % 50 == 0: print(" ", len(rows), flush=True)
    json.dump(rows, open(out / "index.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
