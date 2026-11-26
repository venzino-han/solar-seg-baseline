#!/usr/bin/env python3
"""Primitive CV: find the solar disk by fitting a CIRCLE that best separates
the black space from the orange/yellow Sun, judged purely by COLOUR RATIOS.

Idea
----
The Sun is a circle. So we do not detect edges or fit contours; we simply score
every candidate circle (cx, cy, R) by how well it partitions the frame by colour:

    p_in  = (# Sun-coloured pixels inside)  / (# pixels inside)
    p_out = (# space-coloured pixels outside) / (# pixels outside)
    score = harmonic mean(p_in, p_out)          <- both must be high

A circle that is too small leaves Sun outside (p_out drops); too large swallows
space (p_in drops). The maximum sits on the limb.

Pixel colour classes (no learning, two rules):
    Sun   : bright AND warm      V >= t_otsu  and  R - B >= WARM
    space : dark                 V <  max(20, 0.45 * t_otsu)
`t_otsu` is Otsu's threshold on V, so nothing is hard-coded per channel.

Search is exhaustive but cheap: for a fixed centre, a radial histogram gives the
score for ALL radii at once via cumulative sums, so we only loop over centres.

Usage:
  python solar_disk_cv.py --n 40                 # evaluate all 3 channels
  python solar_disk_cv.py --n 12 --panels out/   # + write visual panels
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "dataset"
FRAME = 512
WARM = 25          # R - B needed to call a pixel "orange/yellow"
CHANNELS = {"AIA_193": "코로나홀 (193Å)", "AIA_304": "홍염 (304Å)", "HMI_Ic": "흑점 (연속광)"}


# --------------------------------------------------------------------- colour
def classify(bgr, rule="core"):
    """-> (sun, space, threshold) boolean masks from brightness + warmth only.

    rule="otsu"  : Otsu on V. Fails on AIA 193, where the off-limb corona is both
                   bright and warm, so it is labelled "Sun" and the fitted circle
                   swells past the limb (R 249 +/- 18 px in our test).
    rule="core"  : threshold relative to the disk's own brightness
                   (v_hi = 90th percentile of V over warm pixels). The corona sits
                   well below 0.75*v_hi, so it falls out of the Sun class and the
                   optimum snaps back onto the limb.
    """
    b, r = bgr[..., 0].astype(np.int16), bgr[..., 2].astype(np.int16)
    v = bgr.max(axis=2)                                   # HSV Value
    warm = (r - b) >= WARM
    if rule == "otsu":
        t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sun = (v >= t) & warm
        space = v < max(20.0, 0.45 * t)
        return sun, space, float(t)
    vw = v[warm]
    v_hi = float(np.percentile(vw, 90)) if vw.size > 500 else float(v.max())
    t = 0.75 * v_hi
    sun = (v >= t) & warm
    space = v < 0.30 * v_hi
    return sun, space, t


# --------------------------------------------------------------------- fitting
def score_all_radii(sun, space, cx, cy, rmax=None):
    """Radial cumulative counts -> score for every integer radius at once."""
    h, w = sun.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.int32)
    rmax = rmax or int(r.max())
    nbin = rmax + 1
    rf = r.ravel()
    area = np.bincount(rf, minlength=nbin)[:nbin]
    s_in = np.bincount(rf, weights=sun.ravel(), minlength=nbin)[:nbin]
    s_sp = np.bincount(rf, weights=space.ravel(), minlength=nbin)[:nbin]

    c_area = np.cumsum(area)                 # pixels with radius <= R
    c_sun = np.cumsum(s_in)
    c_space = np.cumsum(s_sp)
    tot_area, tot_space = c_area[-1], c_space[-1]

    inside_n = np.maximum(c_area, 1)
    outside_n = np.maximum(tot_area - c_area, 1)
    p_in = c_sun / inside_n                                  # sun purity inside
    p_out = (tot_space - c_space) / outside_n                # space purity outside
    f1 = np.where((p_in + p_out) > 0, 2 * p_in * p_out / np.maximum(p_in + p_out, 1e-9), 0.0)
    return f1, p_in, p_out


def ring_profile(sun, cx, cy, rmax):
    """Fraction of Sun-coloured pixels in each 1-px annulus."""
    h, w = sun.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.int32)
    nb = rmax + 1
    rf = r.ravel()
    area = np.bincount(rf, minlength=nb)[:nb]
    s = np.bincount(rf, weights=sun.ravel(), minlength=nb)[:nb]
    return s / np.maximum(area, 1)


def fit_disk(bgr, refine=True, rule="otsu", criterion="ring", ring_thresh=0.5):
    """Circle fit driven purely by colour ratios.

    criterion="f1"   : maximise harmonic mean of (Sun purity inside, space purity
                       outside). Exact where the background is truly black, but on
                       AIA 193 the bright warm corona flattens the optimum and the
                       radius jitters (+/-17 px across frames).
    criterion="ring" : walk outwards and take the first annulus that stops being
                       mostly Sun-coloured (ring fraction < 0.5). Still a pure
                       colour-ratio test, but LOCAL, so a gradual halo cannot drag
                       it outwards -> 2x more stable on 193 (+/-7.7 px).
    The centre is chosen as the one giving the SHARPEST colour transition.
    """
    sun, space, t = classify(bgr, rule)
    h, w = sun.shape
    if sun.sum() < 500:                                      # degenerate frame
        return dict(cx=w / 2, cy=h / 2, R=0.475 * min(h, w), score=0.0,
                    p_in=0.0, p_out=0.0, thr=t, criterion=criterion, fallback=True)

    ys, xs = np.nonzero(sun)
    cx0, cy0 = float(xs.mean()), float(ys.mean())
    rmax, lo = int(0.72 * min(h, w)), int(0.25 * min(h, w))

    def eval_centre(cx, cy):
        if criterion == "f1":
            f1, p_in, p_out = score_all_radii(sun, space, cx, cy, rmax=rmax)
            k = lo + int(np.argmax(f1[lo:]))
            return k, float(f1[k])
        rp = ring_profile(sun, cx, cy, rmax)
        idx = np.where(rp[lo:] < ring_thresh)[0]
        k = lo + int(idx[0]) if len(idx) else rmax
        # sharpness of the limb transition = quality of this centre
        a, b = max(lo, k - 6), min(rmax, k + 7)
        sharp = float(rp[a:k].mean() - rp[k:b].mean()) if b > k and k > a else 0.0
        return k, sharp

    if criterion == "ring":
        # Measured: the plain centroid of the Sun-coloured mask beats every centre
        # search we tried (AIA193 R +/-10.7 px vs +/-14.9 for an 8-sector
        # consistency search; identical on HMI). Searching only adds jitter, so the
        # most primitive option wins and we keep it.
        k, _ = eval_centre(cx0, cy0)
        best = {"cx": cx0, "cy": cy0, "R": float(k), "_q": 0.0}
    else:
        best = None
        for step, span in ((4, 24), (1, 5)) if refine else ((2, 12),):
            for cx in np.arange(cx0 - span, cx0 + span + 1, step):
                for cy in np.arange(cy0 - span, cy0 + span + 1, step):
                    k, q = eval_centre(float(cx), float(cy))
                    if best is None or q > best["_q"]:
                        best = {"cx": float(cx), "cy": float(cy), "R": float(k), "_q": q}
            cx0, cy0 = best["cx"], best["cy"]

    # report colour purities at the chosen circle, whichever criterion was used
    f1, p_in, p_out = score_all_radii(sun, space, best["cx"], best["cy"], rmax=rmax)
    k = int(min(best["R"], len(f1) - 1))
    best.update(score=float(f1[k]), p_in=float(p_in[k]), p_out=float(p_out[k]),
                thr=t, criterion=criterion, fallback=False)
    best.pop("_q")
    return best


def circle_mask(shape, cx, cy, R):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= R * R


# --------------------------------------------------------------------- report
def panel(bgr, fit, sun, space, title):
    m = circle_mask(bgr.shape[:2], fit["cx"], fit["cy"], fit["R"])
    a = bgr.copy()
    cv2.circle(a, (int(round(fit["cx"])), int(round(fit["cy"]))), int(round(fit["R"])), (0, 255, 0), 2)
    cv2.drawMarker(a, (int(round(fit["cx"])), int(round(fit["cy"]))), (0, 255, 255),
                   cv2.MARKER_CROSS, 14, 2)
    cls = np.zeros_like(bgr)
    cls[sun] = (60, 200, 250)          # sun-coloured -> amber
    cls[space] = (70, 70, 70)          # space -> grey
    seg = bgr.copy()
    ov = seg.astype(np.float32)
    ov[..., 1][m] = np.minimum(255, ov[..., 1][m] * 0.55 + 120)   # tint disk green
    seg = ov.astype(np.uint8)
    cells = []
    for im, lab in ((a, "fitted circle"), (cls, "colour classes"), (seg, "disk mask")):
        s = np.full((22, im.shape[1], 3), 255, np.uint8)
        cv2.putText(s, lab, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        cells.append(np.vstack([s, im]))
    gap = np.full((cells[0].shape[0], 5, 3), 255, np.uint8)
    row = np.hstack(sum([[c, gap] for c in cells], [])[:-1])
    head = np.full((24, row.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([head, row])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="images per channel")
    ap.add_argument("--panels", default="", help="dir for visual panels")
    ap.add_argument("--rule", default="otsu", choices=["core","otsu"])
    ap.add_argument("--criterion", default="ring", choices=["ring","f1","both"])
    ap.add_argument("--out", default="disk_cv_results.json")
    args = ap.parse_args()

    pdir = Path(args.panels) if args.panels else None
    if pdir: pdir.mkdir(parents=True, exist_ok=True)

    allres = {}
    crits = ["ring","f1"] if args.criterion=="both" else [args.criterion]
    rule = args.rule
    for criterion in crits:
      print(f"=== criterion={criterion} (colour rule={rule}) ===")
      for chan, ko in CHANNELS.items():
          files = [p for p in sorted(glob.glob(str(DATA / "train" / "*.jpg")))
                   if chan in os.path.basename(p)]
          rng = np.random.RandomState(0); rng.shuffle(files)
          files = files[:args.n]
          rows, t0 = [], time.time()
          for i, p in enumerate(files):
              bgr = cv2.imread(p)
              if bgr is None: continue
              fit = fit_disk(bgr, rule=rule, criterion=criterion)
              rows.append({"file": os.path.basename(p), **fit})
              if pdir and i < 3:
                  sun, space, _ = classify(bgr, rule)
                  cv2.imwrite(str(pdir / f"{criterion}_{chan}_{i}.jpg"),
                              panel(bgr, fit, sun, space,
                                    f"{chan}  R={fit['R']:.0f} c=({fit['cx']:.0f},{fit['cy']:.0f})  "
                                    f"score={fit['score']:.3f} (in {fit['p_in']:.2f} / out {fit['p_out']:.2f})"),
                              [cv2.IMWRITE_JPEG_QUALITY, 90])
          R = np.array([r["R"] for r in rows]); sc = np.array([r["score"] for r in rows])
          pin = np.array([r["p_in"] for r in rows]); pout = np.array([r["p_out"] for r in rows])
          cx = np.array([r["cx"] for r in rows]); cy = np.array([r["cy"] for r in rows])
          allres[f"{criterion}/{chan}"] = {"n": len(rows), "secs_per_img": (time.time() - t0) / max(1, len(rows)),
                          "R_mean": float(R.mean()), "R_std": float(R.std()),
                          "cx_mean": float(cx.mean()), "cy_mean": float(cy.mean()),
                          "cx_std": float(cx.std()), "cy_std": float(cy.std()),
                          "score_mean": float(sc.mean()), "score_min": float(sc.min()),
                          "p_in_mean": float(pin.mean()), "p_out_mean": float(pout.mean()),
                          "rows": rows}
          a = allres[f"{criterion}/{chan}"]
          print(f"{chan:9s} n={a['n']:3d}  R {a['R_mean']:6.1f}±{a['R_std']:4.1f}  "
                f"centre ({a['cx_mean']:.1f},{a['cy_mean']:.1f})±({a['cx_std']:.2f},{a['cy_std']:.2f})  "
                f"score {a['score_mean']:.3f} (min {a['score_min']:.3f})  "
                f"in {a['p_in_mean']:.3f} out {a['p_out_mean']:.3f}  {a['secs_per_img']:.2f}s/img", flush=True)

    (BASE / args.out).write_text(json.dumps(allres, indent=1))
    print(f"-> {BASE / args.out}")


if __name__ == "__main__":
    main()
