#!/usr/bin/env python3
"""Coronal-hole segmentation tools + solar-disk masking + comparison helpers.

Coronal holes are dark regions ON the solar disk in AIA 193A. Off-disk space is
also dark, so every method operates inside a detected solar-disk mask.

Tools implemented (each returns a binary HxW mask of candidate coronal holes):
  seg_otsu        : Otsu global threshold on disk pixels (classic baseline)
  seg_adaptive    : percentile threshold + morphology (domain-tuned)
  seg_slic        : SLIC superpixels, select dark superpixels
  (SAM2 is run separately in run_segmentation.py via ultralytics; box-prompted)
"""
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.segmentation import slic
from skimage.measure import label as cc_label, regionprops
from scipy import ndimage as ndi

IMG = 512


# ---------------------------------------------------------------------------
# Solar disk detection
# ---------------------------------------------------------------------------
def solar_disk_mask(gray, limb_frac=0.97):
    """Delegate to the robust circular-prior detector (solar_disk.py).

    Uses the full photospheric disk (only the thin bright limb trimmed) so coronal
    holes reaching toward the limb are captured — the goal is the true hole extent,
    not a shrunken region. Falls back to the legacy fit below if the module is absent.
    """
    try:
        import solar_disk as _sd
        return _sd.disk_mask(gray, limb_frac=limb_frac)
    except Exception:
        return _legacy_solar_disk_mask(gray, limb_frac=min(limb_frac, 0.92))


def _legacy_solar_disk_mask(gray, limb_frac=0.90):
    """Return a boolean solar-disk mask (True = on-disk), just inside the limb.

    The photospheric disk in these 512px SDO frames is a stable circle
    (radius ~240px, centered ~(256,256)). Internal coronal holes are dark, so an
    area-based radius is unreliable; instead fit the *minimum enclosing circle*
    of the bright disk body (robust to interior dark holes), then shrink by
    limb_frac to drop the bright limb ring and any off-limb coronal glow.
    Falls back to a fixed centered circle if the fit is degenerate.
    """
    h, w = gray.shape
    g = gray.astype(np.float32)
    # standardized SDO 512 browse frames: disk is a near-fixed circle (R~243, centered).
    pcx, pcy, pR = w / 2.0, h / 2.0, 0.475 * min(h, w)   # strong prior
    cx, cy, R = pcx, pcy, pR
    try:
        # separate disk body (incl. dark holes) from near-black space with a LOW
        # absolute threshold; robust even on dim, low-contrast frames.
        thr = max(28.0, np.percentile(g, 45) * 0.4)
        b = (g > thr).astype(np.uint8)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        lbl = cc_label(b)
        if lbl.max() >= 1:
            ar = np.bincount(lbl.ravel()); ar[0] = 0
            body = (lbl == ar.argmax()).astype(np.uint8)
            body = ndi.binary_fill_holes(body).astype(np.uint8)
            cnts, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            (fx, fy), fR = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
            # accept fit only if close to the standardized prior
            if abs(fR - pR) < 0.06 * min(h, w) and abs(fx - pcx) < 45 and abs(fy - pcy) < 45:
                cx, cy, R = fx, fy, fR
    except Exception:
        pass
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= (limb_frac * R) ** 2
    return disk


# ---------------------------------------------------------------------------
# Post-processing shared by the classical detectors
# ---------------------------------------------------------------------------
def _clean(mask, disk, min_area=200, close=3, open_=2):
    m = mask & disk
    if close:
        m = ndi.binary_closing(m, iterations=close)
    if open_:
        m = ndi.binary_opening(m, iterations=open_)
    m = m & disk
    # remove tiny specks
    lbl = cc_label(m)
    out = np.zeros_like(m)
    for r in regionprops(lbl):
        if r.area >= min_area:
            out[lbl == r.label] = True
    return out


# ---------------------------------------------------------------------------
# Tool 1: Otsu
# ---------------------------------------------------------------------------
def seg_otsu(gray, disk):
    vals = gray[disk]
    if vals.size < 100:
        return np.zeros_like(disk)
    try:
        t = threshold_otsu(vals)
    except Exception:
        t = np.percentile(vals, 30)
    dark = (gray < t) & disk
    return _clean(dark, disk)


# ---------------------------------------------------------------------------
# Tool 2: adaptive percentile + morphology (domain-tuned)
# ---------------------------------------------------------------------------
def seg_adaptive(gray, disk, pct=12, rel=0.85):
    # Coronal holes are the darkest tail of the disk (~<0.85x median, ~darkest 8%).
    vals = gray[disk].astype(np.float32)
    if vals.size < 100:
        return np.zeros_like(disk)
    med = np.median(vals)
    t = min(np.percentile(vals, pct), rel * med)  # dark relative to disk median
    dark = (gray < t) & disk
    return _clean(dark, disk, min_area=250, close=4, open_=3)


# ---------------------------------------------------------------------------
# Tool 3: SLIC superpixels, select dark ones
# ---------------------------------------------------------------------------
def seg_slic(gray, disk, n_segments=400, compactness=10.0, dark_rel=0.72):
    g = gray.astype(np.float32) / 255.0
    rgb = np.dstack([g, g, g])
    mask_for_slic = disk
    try:
        segs = slic(rgb, n_segments=n_segments, compactness=compactness,
                    start_label=1, mask=mask_for_slic, channel_axis=-1)
    except TypeError:
        segs = slic(rgb, n_segments=n_segments, compactness=compactness,
                    start_label=1)
        segs = segs * disk
    disk_med = np.median(gray[disk])
    out = np.zeros_like(disk)
    for sp in np.unique(segs):
        if sp == 0:
            continue
        m = segs == sp
        if not m.any():
            continue
        if np.median(gray[m]) < dark_rel * disk_med:
            out |= m
    return _clean(out, disk, min_area=250, close=3, open_=2)


# ---------------------------------------------------------------------------
# Mask -> detections (connected-component bounding boxes)
# ---------------------------------------------------------------------------
def mask_to_boxes(mask, min_area=200):
    lbl = cc_label(mask)
    boxes = []
    for r in regionprops(lbl):
        if r.area < min_area:
            continue
        y1, x1, y2, x2 = r.bbox  # (min_row, min_col, max_row, max_col)
        boxes.append([x1, y1, x2, y2, int(r.area)])
    return boxes


def iou_xyxy(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0
