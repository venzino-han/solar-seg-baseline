"""Procedural STRAIGHT and ARC prominence shapes drawn with geometric tools (lines, polygons,
ellipse arcs, Bezier curves, spirals, tapered strokes) on the limb-unrolled canvas of synth.py:
rows = v (height above limb, VMIN..VMAX), cols = u (arc length along limb, -UHALF..UHALF).
Every shape is rooted at the limb (v ~ 0). Returns a boolean footprint."""
from __future__ import annotations
import cv2, numpy as np
from synth import VMIN, VMAX, UHALF

H, W = VMAX - VMIN, 2 * UHALF
SS = 4                                      # sub-pixel supersampling for cv2 (shift=2)


def P(u, v):                                # canvas pixel (x=col, y=row) in fixed point
    return (int(round((u + UHALF) * SS)), int(round((v - VMIN) * SS)))


def blank():
    return np.zeros((H, W), np.uint8)


def stamp_path(can, pts, r0, r1, rng=None, knots=0.0):
    """Tapered stroke along a polyline: radius goes r0 -> r1; optional knotty radius noise."""
    pts = np.asarray(pts, float)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    n = max(2, int(seg[-1] * 1.5))
    s = np.linspace(0, seg[-1], n)
    u = np.interp(s, seg, pts[:, 0]); v = np.interp(s, seg, pts[:, 1])
    r = np.linspace(r0, r1, n)
    if rng is not None and knots > 0:
        r = r * (1 + knots * np.convolve(rng.standard_normal(n + 8), np.ones(9) / 3, "valid")[:n])
    for ui, vi, ri in zip(u, v, np.maximum(r, 0.6)):
        cv2.circle(can, P(ui, vi), int(round(ri * SS)), 255, -1, cv2.LINE_AA, 2)
    return can


def bezier(p0, p1, p2, p3=None, n=80):
    t = np.linspace(0, 1, n)[:, None]; p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    if p3 is None:
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    p3 = np.asarray(p3)
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def root(rng):
    return rng.uniform(-3.0, 0.0)           # foot slightly below the fitted limb, as humans paint it


# ------------------------------------------------------------------ STRAIGHT
def jet(rng):
    L = rng.uniform(25, 95); a = np.deg2rad(rng.uniform(-40, 40)); v0 = root(rng)
    end = (L * np.sin(a), v0 + L * np.cos(a))
    return stamp_path(blank(), [(0, v0), end], rng.uniform(3, 9), rng.uniform(0.8, 3), rng, rng.uniform(0, .25))


def column(rng):
    w0 = rng.uniform(10, 30); w1 = w0 * rng.uniform(0.5, 1.3); h = rng.uniform(20, 70)
    tilt = rng.uniform(-0.5, 0.5) * h; v0 = root(rng)
    poly = np.array([P(-w0 / 2, v0), P(w0 / 2, v0), P(tilt + w1 / 2, v0 + h), P(tilt - w1 / 2, v0 + h)], np.int32)
    can = blank(); cv2.fillPoly(can, [poly], 255, cv2.LINE_AA, 2)
    return can


def fan(rng):
    can = blank(); k = rng.randint(3, 8); spread = np.deg2rad(rng.uniform(20, 75)); c = np.deg2rad(rng.uniform(-25, 25)); v0 = root(rng)
    for a in c + np.linspace(-spread / 2, spread / 2, k) + rng.normal(0, 0.05, k):
        L = rng.uniform(20, 80)
        stamp_path(can, [(0, v0), (L * np.sin(a), v0 + L * np.cos(a))], rng.uniform(1.5, 4), rng.uniform(0.6, 1.8))
    cv2.circle(can, P(0, v0 + 2), int(rng.uniform(4, 9) * SS), 255, -1, cv2.LINE_AA, 2)   # common bright base
    return can


def surge(rng):
    can = blank(); k = rng.randint(2, 4); a = np.deg2rad(rng.uniform(-35, 35)); gap = rng.uniform(5, 14)
    for i in range(k):
        du = (i - (k - 1) / 2) * gap; L = rng.uniform(25, 85); v0 = root(rng)
        stamp_path(can, [(du, v0), (du + L * np.sin(a), v0 + L * np.cos(a))], rng.uniform(2, 5), rng.uniform(0.8, 2.5), rng, .2)
    return can


def curtain(rng):
    w = rng.uniform(30, 110); h = rng.uniform(12, 45); sh = rng.uniform(-0.8, 0.8) * h; v0 = root(rng)
    top = np.linspace(-w / 2, w / 2, 12); tv = v0 + h * (1 + 0.25 * rng.standard_normal(12))
    poly = [P(-w / 2, v0), P(w / 2, v0)] + [P(u + sh, v) for u, v in zip(top[::-1], tv[::-1])]
    can = blank(); cv2.fillPoly(can, [np.array(poly, np.int32)], 255, cv2.LINE_AA, 2)
    return can


# ------------------------------------------------------------------ ARC
def loop(rng):
    a = rng.uniform(12, 60); b = rng.uniform(12, 75); th = int(round(rng.uniform(2.5, 8) * SS)); v0 = root(rng)
    can = blank()
    cv2.ellipse(can, P(0, v0), (int(a * SS), int(b * SS)), 0, 0, 180, 255, th, cv2.LINE_AA, 2)   # upper half (v up)
    return can


def arcade(rng):
    can = blank(); k = rng.randint(2, 5); nested = rng.rand() < 0.5; base = rng.uniform(10, 30)
    for i in range(k):
        if nested:
            a = base * (1 + 0.45 * i); b = a * rng.uniform(0.7, 1.4); cu = 0
        else:
            a = rng.uniform(8, 22); b = a * rng.uniform(0.7, 1.6); cu = (i - (k - 1) / 2) * a * 1.6
        cv2.ellipse(can, P(cu, root(rng)), (int(a * SS), int(b * SS)), 0, 0, 180, 255,
                    int(round(rng.uniform(1.5, 4.5) * SS)), cv2.LINE_AA, 2)
    return can


def bezier_loop(rng):
    w = rng.uniform(25, 110); h = rng.uniform(20, 85); sk = rng.uniform(-0.6, 0.6) * w
    pts = bezier((-w / 2, root(rng)), (-w / 2 + sk, h * 1.3), (w / 2 + sk, h * 1.3 * rng.uniform(0.6, 1.2)), (w / 2, root(rng)))
    return stamp_path(blank(), pts, rng.uniform(2.5, 7), rng.uniform(2, 6), rng, rng.uniform(0.1, 0.4))


def bubble(rng):
    """Eruptive prominence: an open ring lifted off the limb, tied down by two legs."""
    can = blank(); r = rng.uniform(12, 32); hc = rng.uniform(r + 10, 85 - r); cu = rng.uniform(-15, 15)
    gap = rng.uniform(50, 110); start = 270 + gap / 2; end = 270 - gap / 2 + 360   # opening points DOWN (toward limb)
    th = int(round(min(rng.uniform(2.5, 6), 0.35 * r) * SS))      # a ring, not a filled disk
    cv2.ellipse(can, P(cu, hc), (int(r * SS), int(r * rng.uniform(0.8, 1.25) * SS)), rng.uniform(-20, 20), start, end, 255, th, cv2.LINE_AA, 2)
    for sgn in (-1, 1):                                                             # legs from ring gap to limb
        ang = np.deg2rad(270 + sgn * gap / 2)
        stamp_path(can, [(cu + r * np.cos(ang), hc + r * np.sin(ang)), (cu + sgn * rng.uniform(4, 18), root(rng))],
                   rng.uniform(1.5, 3.5), rng.uniform(2.5, 5))
    return can


def hook(rng):
    L = rng.uniform(30, 90); side = 1 if rng.rand() < .5 else -1; v0 = root(rng)
    pts = bezier((0, v0), (side * rng.uniform(-10, 15), L * rng.uniform(0.7, 1.1)), (side * rng.uniform(20, 60), L * rng.uniform(0.5, 1.0)))
    return stamp_path(blank(), pts, rng.uniform(3, 8), rng.uniform(0.8, 3), rng, rng.uniform(0, .3))


def spiral(rng):
    turns = rng.uniform(0.6, 1.4); r1 = rng.uniform(12, 35); hc = rng.uniform(r1 + 5, 80 - r1); side = 1 if rng.rand() < .5 else -1
    t = np.linspace(0, turns * 2 * np.pi, 160); r = r1 * (1 - t / (turns * 2 * np.pi) * 0.85)
    pts = np.c_[side * r * np.cos(t), hc + r * np.sin(t)]
    pts = np.r_[[(side * rng.uniform(-8, 8), root(rng))], [(pts[0, 0], hc * 0.5)], pts]          # stalk from limb
    return stamp_path(blank(), pts, rng.uniform(3, 6), rng.uniform(1, 2.5), rng, .2)


FAMILIES = {"straight": {"jet": jet, "column": column, "fan": fan, "surge": surge, "curtain": curtain},
            "arc": {"loop": loop, "arcade": arcade, "bezier_loop": bezier_loop, "bubble": bubble, "hook": hook, "spiral": spiral}}
KO = {"jet": "직선 제트", "column": "기둥", "fan": "부채꼴 분사", "surge": "평행 서지", "curtain": "기울어진 커튼",
      "loop": "반타원 루프", "arcade": "루프 아케이드", "bezier_loop": "비대칭 베지어 루프", "bubble": "분출 버블",
      "hook": "휘어진 갈고리 제트", "spiral": "나선형 컬"}


def make_geo(rng, group="both", kind=None):
    if kind is None:
        g = group if group != "both" else ("straight" if rng.rand() < 0.5 else "arc")
        kind = list(FAMILIES[g])[rng.randint(len(FAMILIES[g]))]
    g = "straight" if kind in FAMILIES["straight"] else "arc"
    can = FAMILIES[g][kind](rng)
    if rng.rand() < 0.2:                                  # occasional compound: add a second shape nearby
        k2 = list(FAMILIES[g])[rng.randint(len(FAMILIES[g]))]
        can2 = FAMILIES[g][k2](rng); shift = int(rng.uniform(15, 45)) * (1 if rng.rand() < .5 else -1)
        can = np.maximum(can, np.roll(can2, shift, 1)); kind = f"{kind}+{k2}"
    fp = can > 127
    return fp, g, kind
