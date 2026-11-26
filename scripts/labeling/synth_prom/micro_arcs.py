#!/usr/bin/env python3
"""MICRO OVERLAPPING ARCS: the fine-arc idea at ~1/5 of the existing synthetic label AREA (finearcs median ~1.5K px
-> ~300 px), with thinner strokes (label width ~1.5-3 px vs ~4.5 px in finearcs) and 3-10 overlapping arcs per sample.
Each arc draws its OWN height, aspect ratio (apex height / foot span, log-uniform 0.08-0.9 since v2 = low arcs;
v1 was 0.2-3.5), sideways lean, apex asymmetry and foot offset, so one sample mixes very different arc shapes.
Families: nested (shared centre, growing) / crossing / arcade (chain along the limb) / taller-flat mix /
fan (loops sharing one foot) / tangle (free bezier loops). Gaps between arcs stay OUT of the label (exact
union of footprints, no margin). Placement / rendering / gates shared with strands.py (size gates lowered).

    .venv_train/bin/python synth_prom/micro_arcs.py --n 150 --out synth_prom/dataset_microarcs/p0 --seed 700
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
B = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(B)); sys.path.insert(0, str(B / "synth_prom"))
import synth as S, strands as ST
from geo_shapes import bezier, VMIN, UHALF, H, W


def thin_path(pts, w0, w1, K=4):
    """Stroke of width w0 -> w1 px drawn on a Kx supersampled canvas, area-downsampled and thresholded:
    gives true 1-2.5 px footprints (geo_shapes.stamp_path bottoms out at ~3 px because of AA + radius floor)."""
    big = np.zeros((H * K, W * K), np.uint8); pts = np.asarray(pts, float); n = len(pts)
    P = lambda q: (int(round((q[0] + UHALF) * K)), int(round((q[1] - VMIN) * K)))
    for i in range(n - 1):
        w = w0 + (w1 - w0) * i / max(1, n - 2)
        cv2.line(big, P(pts[i]), P(pts[i + 1]), 255, max(1, int(round(w * K))), cv2.LINE_8)
    return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA) > 127

AREA_SCALE = 0.2                     # 1/5 of the usual synthetic target area


def aspect(rng): return float(np.exp(rng.uniform(np.log(0.08), np.log(0.9))))   # v2: low arcs (v1 was 0.2-3.5)


def arc(rng, c, w, h=None, lean=None):
    """Arch centred at c with foot span w and apex height h (default: from a random aspect ratio)."""
    h = w * aspect(rng) if h is None else h
    lean = rng.uniform(-0.6, 0.6) if lean is None else lean; sk = lean * w
    asym = rng.uniform(0.7, 1.3)                                  # one shoulder higher than the other
    u0, u1 = c - w / 2 + rng.uniform(-0.1, 0.1) * w, c + w / 2 + rng.uniform(-0.1, 0.1) * w
    return bezier((u0, rng.uniform(-2, 0)), (u0 + sk, h * 1.33 * asym), (u1 + sk, h * 1.33 / asym), (u1, rng.uniform(-2, 0)), 120)


def fam_nested(rng, s):
    c = rng.uniform(-3, 3); k = rng.randint(3, 8); w0 = rng.uniform(8, 20) * s
    return [arc(rng, c + rng.uniform(-3, 3) * s, w0 * (1 + 0.35 * i) * rng.uniform(0.85, 1.15)) for i in range(k)]


def fam_crossing(rng, s):
    k = rng.randint(3, 8)
    return [arc(rng, rng.uniform(-12, 12) * s, rng.uniform(8, 35) * s) for _ in range(k)]


def fam_arcade(rng, s):
    k = rng.randint(4, 11); span = rng.uniform(30, 70) * s
    return [arc(rng, -span / 2 + span * i / (k - 1) + rng.uniform(-3, 3) * s, rng.uniform(8, 22) * s) for i in range(k)]


def fam_tallflat(rng, s):
    k = rng.randint(4, 10); out = []
    for i in range(k):
        w = rng.uniform(8, 30) * s
        h = w * (rng.uniform(0.5, 0.9) if i % 2 == 0 else rng.uniform(0.06, 0.2))   # alternate (moderately) taller loops / flat ribbons
        out.append(arc(rng, rng.uniform(-10, 10) * s, w, h, rng.uniform(-0.4, 0.4)))
    return out


def fam_fan(rng, s):
    k = rng.randint(3, 8); foot = rng.uniform(-4, 4) * s; side = rng.choice([-1, 1]); out = []
    for _ in range(k):
        w = rng.uniform(6, 35) * s; c = foot + side * w / 2              # every loop starts at the same foot
        out.append(arc(rng, c, w, lean=rng.uniform(-0.5, 0.5)))
    return out


def fam_tangle(rng, s):
    k = rng.randint(3, 8); out = []
    for _ in range(k):
        a = (rng.uniform(-12, 12) * s, rng.uniform(-2, 0)); d = (rng.uniform(-12, 12) * s, rng.uniform(-2, 5) * s)
        b = (rng.uniform(-22, 22) * s, rng.uniform(3, 16) * s); c = (rng.uniform(-22, 22) * s, rng.uniform(3, 16) * s)
        out.append(bezier(a, b, c, d, 120))
    return out


FAMS = {"nested": fam_nested, "crossing": fam_crossing, "arcade": fam_arcade, "tallflat": fam_tallflat, "fan": fam_fan, "tangle": fam_tangle}
FAM_W = {"nested": 0.18, "crossing": 0.2, "arcade": 0.17, "tallflat": 0.2, "fan": 0.15, "tangle": 0.1}
FAM_KO = {"nested": "겹친 아치 다발", "crossing": "교차하는 고리", "arcade": "가는 아케이드", "tallflat": "높은 고리+납작한 띠",
          "fan": "한 발에서 퍼지는 고리", "tangle": "얽힌 가는 고리"}


def make_micro(rng):
    fam = rng.choice(list(FAM_W), p=list(FAM_W.values())); want = ST.target_area(rng) * AREA_SCALE * rng.uniform(0.6, 1.4)
    s = 1.0; seed = rng.randint(1 << 30)
    for _ in range(6):
        r2 = np.random.RandomState(seed); arcs = FAMS[fam](r2, s); fps = []
        for j, pts in enumerate(arcs):
            w0 = r2.uniform(1.2, 2.6)                             # label width ~1.5-3 px (finearcs ~4.5 px)
            fps.append(thin_path(pts, w0, w0 * r2.uniform(0.6, 1.0)))
        area = np.any(fps, 0).sum()
        if area == 0: return None
        k = np.sqrt(want / area)
        if abs(k - 1) < 0.12: break
        s = float(np.clip(s * np.clip(k, 0.6, 1.6), 0.5, 2.5))  # rescale geometry, NOT brush width
    union = np.any(fps, 0).sum(); total = sum(f.sum() for f in fps)
    if total == 0 or 1 - union / total > 0.3: return None  # arcs piled on top of each other -> unreadable blob
    return fps, fam, want


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = np.random.RandomState(a.seed); out = B / a.out
    for d in ("images", "masks"): (out / d).mkdir(parents=True, exist_ok=True)
    ST.make_strands = make_micro                              # reuse placement/rendering/gates of strands_one
    mg = S.box_margins(); H = S.hosts("train"); rows = []
    while len(rows) < a.n:
        xml, jpg = H[rng.randint(len(H))]
        r = ST.strands_one(rng, cv2.imread(jpg), xml, mg, min_union=20, min_off=12)
        if r is None: continue
        img, mask, box, meta = r; sid = f"ma{a.seed}_{len(rows):04d}"
        cv2.imwrite(str(out / "images" / f"{sid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "masks" / f"{sid}.png"), mask.astype(np.uint8) * 255)
        rows.append(dict(sid=sid, host=Path(jpg).stem, box=box, **meta))
        if len(rows) % 50 == 0: print(" ", len(rows), flush=True)
    json.dump(rows, open(out / "index.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
