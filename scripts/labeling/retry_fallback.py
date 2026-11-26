#!/usr/bin/env python3
"""Last-resort labels for the boxes that retry_select.py left unlabelled, so that every box has a label.

  tier 2  sam3_retry : a gate-passing mask of ANY retry variant (also the ones below the validated IoU band),
                       best-ranked first; sunspots must still pass the limb-darkening check (<= 0.95)
  tier 3  rule       : no model mask passes -> brightness rule inside the box
                       sunspot    = pixels at least 12 % darker than the pixels at the same radius around the box (groups of >= 3 px)
                       prominence = off-limb pixels brighter than the sky in the box (top 20 %)
                       if that gives fewer than MIN pixels, the MIN most extreme pixels are taken
Outputs retry_unlabeled/<cls>/masks_fallback/<stem>__b<bi>.png and fallback.csv (tier, variant, area).
"""
import csv, sys
from pathlib import Path
import cv2, numpy as np
B = Path(__file__).resolve().parent; sys.path.insert(0, str(B)); DATA = B.parent / "dataset"
import label_other_classes as L, retry_select as RS, retry_unlabeled_wide as RW
MIN = dict(sunspot=4, prominence=8)


def rule_mask(cls, bgr, box):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); x1, y1, x2, y2 = box; bm = np.zeros((512, 512), bool); bm[y1:y2, x1:x2] = True
    if cls == "sunspot":
        # compare every pixel with the pixels at the SAME radius around the box (local limb-darkening profile, up to the visible limb)
        vis = gray > 0.25 * np.median(gray[200:312, 200:312]); ys, xs = np.where(vis); cx, cy = xs.mean(), ys.mean()
        yy, xx = np.mgrid[:512, :512]; rr = np.hypot(xx - cx, yy - cy); ri = np.round(rr).astype(int)
        nb = np.zeros((512, 512), bool); nb[max(0, y1 - 25):y2 + 25, max(0, x1 - 25):x2 + 25] = True; nb &= vis
        loc = np.ones((512, 512), np.float32)
        for q in np.unique(ri[bm & vis]):
            sel = nb & (ri == q); loc[sel & bm] = gray[sel & bm] / max(1.0, float(np.percentile(gray[sel], 75)))
        a = bm & vis; val = loc; m = a & (loc < 0.88)
        n, lab, stt, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)       # drop isolated pixels
        m = np.isin(lab, [k for k in range(1, n) if stt[k, cv2.CC_STAT_AREA] >= 3])
    else:
        _, _, Rl, rmap = L.solar_region_colour(bgr); a = bm & (rmap > Rl + L.LIMB_MARGIN); val = -gray.astype(np.float32)
        m = a & (gray >= np.percentile(gray[a], 80)) & (gray > 1.02 * np.median(gray[a])) if a.sum() >= 10 else np.zeros_like(bm)
    if a.sum() < MIN[cls]: a = bm                                   # box has (almost) no pixel in the allowed region
    if m.sum() < MIN[cls]:
        ys, xs = np.where(a); o = np.argsort(val[ys, xs])[:MIN[cls]]; m = np.zeros_like(bm); m[ys[o], xs[o]] = True
    return m


def main(cls):
    O = B / "retry_unlabeled" / cls; R, Z = RS.load(cls); st = RS.stats(R); order = sorted(st, key=lambda n: -st[n]["gold"])
    done = {p.stem for p in (O / "masks").glob("*.png")}; out = O / "masks_fallback"; out.mkdir(exist_ok=True); [p.unlink() for p in out.glob("*.png")]; log = []
    for r in R:
        key = f"{r['stem']}__b{r['bi']}"
        if r["kind"] != "fail" or key in done: continue
        bgr = cv2.imread(str(DATA / r["split"] / f"{r['stem']}.jpg")); gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); m, tier, var = None, "", ""
        flat = RW.wide_disk(gray)[1] if cls == "sunspot" else None
        for n in order:
            if not r["v"].get(n, {}).get("ok") or f"{key}@{n}" not in Z: continue
            c = np.unpackbits(Z[f"{key}@{n}"])[:512 * 512].reshape(512, 512) > 0
            if flat is not None and np.median(flat[c]) > 0.95: continue
            m, tier, var = c, "sam3_retry", n; break
        if m is None: m, tier = rule_mask(cls, bgr, r["box"]), "rule"
        cv2.imwrite(str(out / f"{key}.png"), m.astype(np.uint8) * 255); log.append(dict(split=r["split"], stem=r["stem"], bi=r["bi"], tier=tier, variant=var, area=int(m.sum())))
    with open(O / "fallback.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    from collections import Counter
    print(cls, len(log), Counter(d["tier"] for d in log), "area median", {t: int(np.median([d["area"] for d in log if d["tier"] == t])) for t in {d["tier"] for d in log}}, "min area", min(d["area"] for d in log))


if __name__ == "__main__":
    for c in sys.argv[1:] or ["sunspot", "prominence"]: main(c)
