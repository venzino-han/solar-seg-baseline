#!/usr/bin/env python3
"""Generate pixel labels for SUNSPOT and PROMINENCE with VANILLA SAM 3 (no training).

Prompt design is driven by published physics of each feature (see README/report):

  SUNSPOT — SDO/HMI continuum.
    A spot is DARKER than the surrounding photosphere: umbra ~20% and penumbra
    ~75% of the quiet-Sun intensity (Solanki 2003; HMI detection literature).
    => crop+upscale the box (spots are only ~10-40 px wide in a 512 frame, and
       SAM 3 decodes masks at 288x288, so a full-frame call just fills the box),
       put a POSITIVE POINT on the darkest pixel (the umbra), and pick among the
       multimask candidates the one that best matches the CV dark region
       (I < 0.90 x local quiet-Sun). Masks are clipped to the solar disk.

  PROMINENCE — SDO/AIA 304 (He II).
    Off-limb prominence plasma is BRIGHT against the dark sky (optically thick
    chromospheric emission); the on-disk counterpart (filament) is dark. The VOC
    boxes here straddle the limb, so the mask must NOT be clipped to the disk.
    => crop+upscale, POSITIVE POINT on the brightest pixel OUTSIDE the fitted
       solar body, select the candidate best matching the CV bright region
       (I > max(1.6 x sky, p75 of box)). The body constrains the SEED only; it
       is no longer subtracted from the finished mask (see CFG note).

Both use `Sam3TrackerModel` (the geometry-promptable member of the SAM 3 family)
with pretrained weights only — nothing here is fine-tuned.

Outputs (label_pack_<cls>/):
  masks/<split>/<stem>__b<bi>.png    per-box binary label (512x512)
  masks_full/<split>/<stem>.png      per-image union label
  overlays/<split>/<stem>.jpg        real frame with the labels mapped on
  _index.shard<k>of<n>.csv           per-box metadata + confirmed flag

Usage:
  CUDA_VISIBLE_DEVICES=0 python label_other_classes.py --cls sunspot --shard 0/2
  CUDA_VISIBLE_DEVICES=0 python label_other_classes.py --cls prominence --validate 20
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
import segmenters as S

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "dataset"
FRAME = 512

CFG = {
    # min_area is class-specific: in a 512-px frame the solar disk is ~460 px, so
    # 1 px ~ 3,000 km. A small sunspot umbra (~30 millionths of a hemisphere) is
    # only ~10 px, and 80% of real spots here land under 25 px -- a generic
    # threshold would silently reject most of the class.
    "sunspot": dict(channel="HMI_Ic", pad=0.8, up=512, want="dark",
                    clip_disk=True, clip_box=True, color=(60, 90, 235),   # red-ish
                    min_area=4, cov_lo=0.0002,
                    cv_rule="I < 0.90 x local quiet-Sun"),
    # `limb_body` and `clip_off_limb` are deliberately SEPARATE knobs.
    #   limb_body     -- compute the solar body and use it to place the point
    #                    prompt. Required: in AIA 304 the chromospheric disk is
    #                    brighter than the off-limb prominence, so an
    #                    unrestricted "brightest pixel" seed lands on the disk
    #                    and SAM then segments a disk wedge instead.
    #   clip_off_limb -- also SUBTRACT the body from the finished mask.
    # These were one flag until the human labels arrived. The labelers follow a
    # prominence down to its root, so 12.0% of what they painted sat inside the
    # fitted limb and the subtraction threw it away. Seeding still needs the
    # body; the output no longer does.
    # clip_box=False: the eval tool deliberately lets labelers paint past the VOC
    # box (prominences routinely overflow it), and 5.3% of what they painted sat
    # outside. Clipping there discarded evidence we had asked them to give us.
    # The mask stays bounded regardless: paste_back can only write inside the
    # padded crop, which is ~4.4x the box area and contains 100% of their paint.
    "prominence": dict(channel="AIA_304", pad=0.5, up=512, want="bright",
                       clip_disk=False, limb_body=True, clip_off_limb=False,
                       clip_box=False,
                       color=(235, 120, 40),                   # BLUE (BGR)
                       min_area=8, cov_lo=0.0005,
                       cv_rule="I > max(1.6 x sky, p75 of box)"),
}
# cov_lo used to be a single 0.002 for both classes, which silently discarded
# genuine detections: a 6-px spot inside a 5,000-px box scores 0.0012 and was
# dropped (867 sunspot boxes), and prominences shrink once the solar body is cut
# away (639 boxes had a correct, clearly-bright mask of ~15 px but lost to
# min_area=25). Absolute min_area is the meaningful floor; cov_lo now only guards
# against degenerate specks.
COV_HI = 0.90
# px added to the fitted solar radius before cutting. NEGATIVE = shrink the Sun
# circle inward so the prominence base near the limb survives.
# Measured trade-off (22 frames, contamination = mask px inside the independent
# gradient limb):
#   +2 -> 54.3% confirmed, contamination 0.0 px      (too aggressive, loses base)
#   -3 -> 76.5% confirmed, contamination 3.4 px      <-- knee, chosen
#   -4 -> 82.7% confirmed, contamination 10.2 px
#   -5 -> 90.1% confirmed, contamination 24.7 px     (40% of masks carry body)
#   -6 -> 87.7% confirmed, contamination 52.6 px
LIMB_MARGIN = -3
# plausible solar radius band for AIA 304 (px). Colour-ratio fit measured
# 210.8 +/- 3.9; anything outside this band is a bad fit -> use the median.
R_BAND, R_MEDIAN = (190.0, 222.0), 206.0
# Ring-crossing level that defines "the ring stopped being the Sun".
# 0.5 (the default for standalone disk finding) sits ~6 px OUTSIDE the true limb
# and swallowed the prominence base, starving the seed in 56% of boxes.
# Calibrated against the independent brightness-gradient limb (204.9 +/- 3.6 px):
#   <0.5 -> 210.4 (+5.6)   <0.7 -> 207.1 (+2.2)   <0.8 -> 205.6 (+0.7)   <0.9 -> unstable
RING_THRESH = 0.8


def solar_region_colour(bgr):
    """Solar body via the COLOUR-RATIO circle fit (solar_disk_cv).

    The Sun is assumed circular and located by maximising the colour split
    (Sun-coloured inside / space-coloured outside) -- no edge detection.
    Measured on 30 AIA-304 frames against the brightness-gradient limb:
      colour-ratio R = 210.8 +/- 3.9 px   vs   gradient R = 204.7 +/- 3.3 px
      difference median +7 px (colour excludes MORE of the body, which is the
      safe direction here), but 2/30 frames came out SMALLER by up to 11 px,
      which would leave a ring of solar body inside the prominence label.
    So the fitted radius is clamped to a plausible band; outside it we fall back
    to the dataset-median radius rather than trusting a bad fit.
    """
    import solar_disk_cv as DCV
    fit = DCV.fit_disk(bgr, rule="otsu", criterion="ring",
                       ring_thresh=RING_THRESH)
    R = fit["R"]
    if not (R_BAND[0] <= R <= R_BAND[1]) or fit.get("fallback"):
        R = R_MEDIAN
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rmap = np.sqrt((xx - fit["cx"]) ** 2 + (yy - fit["cy"]) ** 2)
    return fit["cx"], fit["cy"], float(R), rmap


def limb_circle(gray):
    """Centre and radius of the visible solar limb (steepest radial fall-off).

    NOTE: segmenters.solar_disk_mask() deliberately SHRINKS the radius (x0.92) to
    drop the bright rim, and cv2.minEnclosingCircle on the thresholded body
    OVERSHOOTS (~237 px) because AIA 304 has bright off-limb emission. Neither is
    the true limb. The azimuthal intensity profile's steepest gradient gives a
    stable limb radius (206.4 +/- 1.1 px over sampled frames), which is what we
    need to exclude the solar body and keep only what protrudes beyond it.
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    m = g > max(28.0, float(np.percentile(g, 45)) * 0.4)
    if m.sum() < 1000:
        cx, cy = w / 2.0, h / 2.0
    else:
        ys, xs = np.nonzero(m); cx, cy = float(xs.mean()), float(ys.mean())
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ri = r.astype(np.int32)
    cnt = np.bincount(ri.ravel(), minlength=400)
    tot = np.bincount(ri.ravel(), weights=g.ravel(), minlength=400)
    prof = tot / np.maximum(cnt, 1)
    lo, hi = 120, min(300, len(prof))
    R = lo + int(np.argmin(np.gradient(prof[lo:hi])))
    return cx, cy, float(R), r


def boxes_of(xml, cls):
    r = ET.parse(xml).getroot()
    return [[int(float(o.find("bndbox").findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")]
            for o in r.findall("object") if o.findtext("name") == cls]


def crop_ctx(rgb, box, pad_frac, out):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(pad_frac * bw) + 4, int(pad_frac * bh) + 4
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(FRAME, x2 + px), min(FRAME, y2 + py)
    crop = rgb[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    up = cv2.resize(crop, (out, out), interpolation=cv2.INTER_CUBIC)
    sx, sy = out / cw, out / ch
    b = [max(0, min(out - 1, int((x1 - cx1) * sx))), max(0, min(out - 1, int((y1 - cy1) * sy))),
         max(0, min(out - 1, int((x2 - cx1) * sx))), max(0, min(out - 1, int((y2 - cy1) * sy)))]
    return up, b, (cx1, cy1, cx2, cy2, cw, ch)


def paste_back(mask_up, info):
    cx1, cy1, cx2, cy2, cw, ch = info
    # INTER_NEAREST_EXACT, not INTER_NEAREST: OpenCV's legacy nearest maps a
    # destination pixel to floor(dst*scale) instead of the pixel centre, which
    # walks the mask toward the bottom-right. On a 512->70->512 round trip that
    # is +4.1 px; through this crop/paste path it was +0.5 px per label, and the
    # eval tool then magnifies it 4-7x. Measured, not theoretical -- see
    # check_offset.py --synth.
    small = cv2.resize(mask_up.astype(np.uint8), (cw, ch),
                       interpolation=cv2.INTER_NEAREST_EXACT) > 0
    full = np.zeros((FRAME, FRAME), bool)
    full[cy1:cy2, cx1:cx2] = small
    return full


def cv_reference(gup, bq, want, disk_up=None):
    """Physics-based reference region inside the crop (used to place the point
    prompt and to choose among SAM's multimask candidates).

    For prominences this MUST be restricted to off-disk pixels: in AIA 304 the
    chromospheric disk is brighter than the off-limb prominence, so an
    unrestricted "brightest pixel" seed lands on the disk and SAM then segments
    the disk wedge inside the box instead of the prominence.
    """
    x1, y1, x2, y2 = bq
    sub = gup[y1:y2, x1:x2]
    if sub.size == 0:
        return None, None
    ring = np.concatenate([gup[:y1].ravel(), gup[y2:].ravel()])
    rmed = float(np.median(ring)) if ring.size else float(np.median(gup))
    ref = np.zeros_like(gup, bool)
    boxm = np.zeros_like(gup, bool); boxm[y1:y2, x1:x2] = True

    if want == "dark":
        ref[y1:y2, x1:x2] = sub < 0.90 * rmed
    else:
        off = boxm & (~disk_up if disk_up is not None else np.ones_like(boxm))
        if off.sum() < 10:
            return None, rmed
        vals = gup[off]
        sky = float(np.median(vals))
        # NOTE: a hard floor (was 12.0) silently killed 56% of boxes once the
        # colour-ratio solar region (which sits ~8 px outside the gradient limb)
        # removed the bright prominence base -- the faint remainder never cleared
        # it. Scale the floor to the frame's own sky level instead.
        thr = max(sky * 1.5, float(np.percentile(vals, 75)), sky + 4.0, 3.0)
        ref = off & (gup > thr)                      # bright plasma OUTSIDE the limb
    if ref.sum() < 5:
        return None, rmed
    return ref, rmed


def segment_box(model, proc, rgb, box, disk, cfg, device, body=None):
    """-> (mask_512, meta) using vanilla SAM 3 Tracker with a physics-placed point.

    `body` (prominence only) is the solar-body mask defined by the true limb; the
    prominence label must contain ONLY what protrudes beyond it.
    """
    up, bq, info = crop_ctx(rgb, box, cfg["pad"], cfg["up"])
    gup = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # the region that counts as "inside the Sun" when placing the seed. This is
    # prompt placement only -- whether it is also subtracted from the output is
    # cfg["clip_off_limb"], a separate decision.
    inside = body if body is not None else disk
    disk_up = crop_ctx(np.repeat(inside[:, :, None].astype(np.uint8) * 255, 3, axis=2),
                       box, cfg["pad"], cfg["up"])[0][:, :, 0] > 127
    ref, rmed = cv_reference(gup, bq, cfg["want"], disk_up)
    if ref is None:
        return np.zeros((FRAME, FRAME), bool), {"reason": "no_cv_seed", "score": 0.0}

    ys, xs = np.where(ref)
    vals = gup[ys, xs]
    j = int(np.argmin(vals)) if cfg["want"] == "dark" else int(np.argmax(vals))
    pt = [float(xs[j]), float(ys[j])]

    inp = proc(images=up, input_points=[[[pt]]], input_labels=[[[1]]],
               return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp, multimask_output=True)
    lg = out.pred_masks[0].float()
    lg = lg.flatten(0, 1) if lg.dim() > 3 else lg

    ref_full = paste_back(ref, info)
    bx1, by1, bx2, by2 = box
    cx1, cy1, cx2, cy2 = info[:4]
    denom = ((bx2 - bx1) * (by2 - by1) if cfg.get("clip_box", True)
             else (cx2 - cx1) * (cy2 - cy1))
    best, best_iou = None, -1.0
    for k in range(lg.shape[0]):
        mk = cv2.resize((lg[k] > 0).cpu().numpy().astype(np.uint8), (FRAME, FRAME),
                        interpolation=cv2.INTER_NEAREST_EXACT) > 0
        full = paste_back(mk, info)
        if cfg["clip_disk"]:
            full &= disk
        if cfg.get("clip_off_limb") and body is not None:
            full &= ~body                    # solar body excluded: protrusion only
        x1, y1, x2, y2 = box
        if cfg.get("clip_box", True):
            bm = np.zeros((FRAME, FRAME), bool); bm[y1:y2, x1:x2] = True
            full &= bm                               # never leak outside the label box
        # Coverage must be measured against the region the mask is ALLOWED to
        # occupy, else a legitimately-overflowing prominence reads as cov > 0.9
        # and every candidate gets rejected.
        cov = full.sum() / max(1, denom)
        if not (cfg["cov_lo"] <= cov <= COV_HI):
            continue
        u = (full | ref_full).sum()
        iou = (full & ref_full).sum() / u if u else 0.0
        if iou > best_iou:
            best, best_iou = full, iou
    if best is None:
        return np.zeros((FRAME, FRAME), bool), {"reason": "no_valid_candidate",
                                                "score": 0.0, "cov_denom": denom}
    return best, {"reason": "", "score": float(best_iou), "cov_denom": denom}


def rel_intensity(gray, mask, box, disk=None, want="dark"):
    """Median intensity of the mask relative to its correct physical background.

    sunspot   : compare against the rest of the box (photosphere) -> expect < 1.
    prominence: compare against the OFF-DISK sky in the box, not the whole box —
                the box also contains the bright 304 disk, which would swamp the
                comparison and make a correct off-limb mask look "not bright".
    """
    x1, y1, x2, y2 = box
    bm = np.zeros((FRAME, FRAME), bool); bm[y1:y2, x1:x2] = True
    m = mask & bm
    ring = bm & ~mask
    if want == "bright" and disk is not None:
        ring = ring & ~disk                      # sky background only
    if m.sum() < 3 or ring.sum() < 5:            # 3 px: smallest spot we still measure
        return None
    return float(np.median(gray[m])) / max(1.0, float(np.median(gray[ring])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, choices=list(CFG))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--validate", type=int, default=0, help="render N panels and exit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default=None,
                    help="fine-tuned head weights (train_sam3_sunspot.py --final)")
    ap.add_argument("--out", default=None, help="output dir (default label_pack_<cls>)")
    ap.add_argument("--split", default="train,test", help="처리할 split (쉼표 구분)")
    ap.add_argument("--save-all", action="store_true", help="게이트 미통과(confirmed=0) 박스의 마스크도 저장 (전수 검사용)")
    args = ap.parse_args()
    cls = args.cls; cfg = CFG[cls]
    k, n = (int(x) for x in args.shard.split("/"))

    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    model = Sam3TrackerModel.from_pretrained(str(BASE / "sam3_model")).to(args.device).eval()
    proc = Sam3TrackerProcessor.from_pretrained(str(BASE / "sam3_model"))
    if args.ckpt:
        # Only the trained heads are stored; the frozen vision encoder stays as
        # pretrained, so a partial load is expected and strict=False is wrong --
        # verify every saved key actually lands.
        ck = torch.load(BASE / args.ckpt, map_location="cpu", weights_only=False)
        own = model.state_dict()
        miss = [q for q in ck["state_dict"] if q not in own]
        if miss:
            raise SystemExit(f"체크포인트 키가 모델에 없음: {miss[:5]}")
        own.update({q: v for q, v in ck["state_dict"].items()})
        model.load_state_dict(own)
        print(f"[{cls}] fine-tuned heads loaded: {args.ckpt} "
              f"({len(ck['state_dict'])} tensors, trained={ck.get('trained')}, n={ck.get('n')})",
              flush=True)      # unflushed, this sat in the buffer for the whole run

    splits = tuple(s.strip() for s in args.split.split(",") if s.strip())
    files = []
    for sp in splits:
        files += [(sp, p) for p in sorted(glob.glob(str(DATA / sp / "*.xml")))
                  if cfg["channel"] in os.path.basename(p)]
    files = [f for i, f in enumerate(files) if i % n == k]

    OUT = BASE / (args.out or f"label_pack_{cls}")
    if args.validate:
        vdir = OUT / "validate"; vdir.mkdir(parents=True, exist_ok=True)
    else:
        for sub in ("masks", "masks_full", "overlays"):
            for sp in splits:
                (OUT / sub / sp).mkdir(parents=True, exist_ok=True)

    rows, n_conf, t0, rels = [], 0, time.time(), []
    for fi, (sp, xml) in enumerate(files):
        bs = boxes_of(xml, cls)
        if not bs:
            continue
        stem = Path(xml).stem
        bgr = cv2.imread(str(DATA / sp / f"{stem}.jpg"))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        disk = S.solar_disk_mask(gray)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        body = None
        if cfg.get("limb_body") or cfg.get("clip_off_limb"):
            # solar region from the COLOUR-RATIO circle fit (원형 가정 + 색 구성비)
            _, _, Rl, rmap = solar_region_colour(bgr)
            body = rmap <= (Rl + LIMB_MARGIN)          # the Sun itself

        union = np.zeros((FRAME, FRAME), bool); kept = []
        for bi, box in enumerate(bs):
            mask, meta = segment_box(model, proc, rgb, box, disk, cfg, args.device, body)
            x1, y1, x2, y2 = box
            area = int(mask.sum())
            # same denominator segment_box used, or the confirm gate below would
            # contradict the candidate gate above.
            cov = area / max(1, meta.get("cov_denom") or (x2 - x1) * (y2 - y1))
            rel = rel_intensity(gray, mask, box, body if body is not None else disk, cfg["want"])
            ok = (area >= cfg["min_area"] and cfg["cov_lo"] <= cov <= COV_HI and rel is not None
                  and ((rel < 0.98) if cfg["want"] == "dark" else (rel > 1.02)))
            if ok:
                n_conf += 1; union |= mask; kept.append(box); rels.append(rel)
            if (ok or args.save_all) and not args.validate:
                cv2.imwrite(str(OUT / "masks" / sp / f"{stem}__b{bi}.png"),
                            mask.astype(np.uint8) * 255)
            rows.append({"split": sp, "stem": stem, "bi": bi,
                         "box": f"{x1} {y1} {x2} {y2}", "area_px": area,
                         "box_coverage": round(cov, 4),
                         "rel_intensity": round(rel, 4) if rel is not None else "",
                         "sam_cv_iou": round(meta["score"], 4),
                         "confirmed": int(ok), "note": meta["reason"],
                         "mask_path": f"masks/{sp}/{stem}__b{bi}.png" if (ok or args.save_all) and not args.validate else "",
                         "overlay_path": f"overlays/{sp}/{stem}.jpg" if kept and not args.validate else ""})

        if kept:
            ov = bgr.astype(np.float32)
            col = cfg["color"]
            for c in range(3):
                ov[..., c][union] = 0.45 * ov[..., c][union] + 0.55 * col[c]
            ov = ov.astype(np.uint8)
            cnt, _ = cv2.findContours(union.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnt, -1, col, 1)
            for b in kept:
                cv2.rectangle(ov, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 1)
            cv2.putText(ov, f"{cls}  {stem}  n={len(kept)}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if args.validate:
                cv2.imwrite(str(vdir / f"{stem}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                cv2.imwrite(str(OUT / "masks_full" / sp / f"{stem}.png"),
                            union.astype(np.uint8) * 255)
                cv2.imwrite(str(OUT / "overlays" / sp / f"{stem}.jpg"), ov,
                            [cv2.IMWRITE_JPEG_QUALITY, 86])

        if args.validate and len(list((OUT / "validate").glob("*.jpg"))) >= args.validate:
            break
        if not args.validate and (fi + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {fi+1}/{len(files)} img · {len(rows)} box · conf {n_conf} · "
                  f"{el:.0f}s · eta {el/(fi+1)*(len(files)-fi-1):.0f}s", flush=True)

    if args.validate:
        print(f"[{cls}] validation panels -> {OUT/'validate'}  "
              f"({n_conf}/{len(rows)} confirmed, median rel_I "
              f"{np.median(rels) if rels else float('nan'):.3f})")
        return

    frag = OUT / f"_index.shard{k}of{n}.csv"
    with frag.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[{cls}] shard {k}/{n} done: {len(rows)} boxes, {n_conf} confirmed, "
          f"median rel_I {np.median(rels):.3f}, {time.time()-t0:.0f}s -> {frag}", flush=True)


if __name__ == "__main__":
    main()
