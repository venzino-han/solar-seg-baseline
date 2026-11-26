#!/usr/bin/env python3
"""Synthetic AIA-304 prominences painted with a 'brush' that remixes HUMAN label shapes.

Pipeline per sample
  1. template  : one human prominence mask, unrolled to limb coordinates
                 (u = arc length along the limb, v = height above the fitted limb).
  2. brush     : medial-axis skeleton of that template + local thickness. New shape =
                 skeleton pushed through a random stretch / mirror / smooth warp, then
                 re-stamped with round soft brushes (+ ragged 'bristle' dabs). The root
                 (v<=0) stays anchored to the limb, so each labeler's convention for the
                 lower edge is inherited from its template.
  3. appearance: semi-transparent emission layer (streak texture x height fall-off),
                 blended in the frame's own intensity->colour LUT (AIA 304 colormap), so
                 the added region is orange exactly the way the host frame is orange.
  4. placement : random limb angle on a host frame away from its real prominence boxes.
  5. label     : warped brush footprint = mask; VOC box = mask bbox + margins sampled from
                 how the real boxes sit around the human masks.

    .venv_train/bin/python synth_prom/synth.py --n 200 --out synth_prom/samples
"""
from __future__ import annotations

import argparse, csv, json, glob, os, sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.morphology import medial_axis

B = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(B))
import label_other_classes as L

DATA = B.parent / "dataset"
VMIN, VMAX, UHALF = -30, 110, 160            # unrolled canvas: v in [VMIN,VMAX), u in [-UHALF,UHALF)
LABS = ("labeller_5", "labeller_6")


# ------------------------------------------------------------------ geometry
def unroll(mask, cx, cy, R, th0):
    vv, uu = np.mgrid[VMIN:VMAX, -UHALF:UHALF].astype(np.float32)
    th = th0 + uu / R
    x = cx + (R + vv) * np.cos(th); y = cy + (R + vv) * np.sin(th)
    return map_coordinates(mask.astype(np.float32), [y, x], order=0, mode="constant") > 0.5


def roll_maps(cx, cy, R, th0, shape=(512, 512)):
    """image pixel -> canvas (row, col) for a prominence centred at limb angle th0."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    th = np.arctan2(yy - cy, xx - cx)
    u = np.angle(np.exp(1j * (th - th0))) * R
    v = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - R
    return v - VMIN, u + UHALF


# ------------------------------------------------------------------ templates
class Template:
    def __init__(self, rec, U):
        self.rec, self.U = rec, U
        skel, dist = medial_axis(U, return_distance=True)
        ys, xs = np.nonzero(skel)
        self.pts = np.stack([ys + VMIN, xs - UHALF, dist[ys, xs]], 1).astype(np.float32)  # v, u, r


def load_templates(keep=None):
    rows = [r for r in csv.DictReader(open(B / "gold_prominence/labels.csv")) if r["evaluator"] in LABS]
    out = []
    for r in rows:
        key = f"{r['stem']}__b{r['bi']}"
        if keep is not None and key not in keep:
            continue
        bgr = cv2.imread(str(DATA / r["split"] / f"{r['stem']}.jpg"))
        g = cv2.imread(str(B / "gold_prominence" / r["mask_path"]), 0) > 127
        if g.sum() < 30:
            continue
        cx, cy, R, rmap = L.solar_region_colour(bgr)
        ys, xs = np.nonzero(g)
        th0 = float(np.angle(np.exp(1j * np.arctan2(ys - cy, xs - cx)).mean()))
        U = unroll(g, cx, cy, R, th0)
        # appearance statistics for this human label
        lev = bgr.astype(np.float32).sum(2)
        off = g & (rmap > R + 2)
        band = (rmap > R - 8) & (rmap < R - 2) & (np.abs(np.angle(np.exp(1j * (np.arctan2(*np.mgrid[0:512, 0:512] - np.array([[[cy]], [[cx]]])) - th0)))) * R < 60)
        rat = float(np.median(lev[off]) / max(1.0, np.median(lev[band]))) if off.sum() > 10 and band.sum() > 10 else None
        rec = dict(key=key, lab=r["evaluator"], stem=r["stem"], level_ratio=rat)
        out.append(Template(rec, U))
    return out


# ------------------------------------------------------------------ brush
def smooth_field(rng, shape, sigma, amp):
    f = gaussian_filter(rng.standard_normal(shape), sigma)
    return f / (np.abs(f).max() + 1e-6) * amp


def brush_params(rng, tpl: Template):
    H, W = VMAX - VMIN, 2 * UHALF
    return dict(mirror=rng.rand() < 0.5, su=rng.uniform(0.8, 1.25), sv=rng.uniform(0.8, 1.3),
                sr=rng.uniform(0.85, 1.1), rj=rng.standard_normal(len(tpl.pts)),
                du=smooth_field(rng, (H, W), 14, rng.uniform(2, 7)), dv=smooth_field(rng, (H, W), 14, rng.uniform(1, 5)),
                pb=rng.uniform(0.05, 0.25), bristle_seed=rng.randint(1 << 30))


def brush_paint(tpl: Template, P, k=1.0):
    """Re-stamp a template skeleton with a randomised brush (params P, overall size factor k)."""
    v, u, r = tpl.pts[:, 0].copy(), tpl.pts[:, 1].copy(), tpl.pts[:, 2].copy()
    if P["mirror"]:
        u = -u
    u -= np.median(u)
    lift = np.clip(v / 12.0, 0, 1)                    # root stays glued to the limb
    u = u * P["su"] * k
    v = np.where(v > 0, v * P["sv"] * k, v * min(1.0, k * P["sv"]))
    H, W = VMAX - VMIN, 2 * UHALF
    ri = np.clip((v - VMIN).astype(int), 0, H - 1); ci = np.clip((u + UHALF).astype(int), 0, W - 1)
    u = u + P["du"][ri, ci] * lift; v = v + P["dv"][ri, ci] * lift
    # medial-axis distance overshoots the true half-width by ~0.5 px
    r = np.maximum(0.7, (r - 0.5) * P["sr"] * k * (1 + 0.12 * P["rj"]))
    can = np.zeros((H, W), np.uint8)
    for vi, ui, rr in zip(v, u, r):
        cv2.circle(can, (int(round((ui + UHALF) * 4)), int(round((vi - VMIN) * 4))), int(round(rr * 4)), 255, -1, cv2.LINE_AA, 2)
    # bristles: small dabs just outside the stroke edge -> ragged, hand-painted rim
    br = np.random.RandomState(P["bristle_seed"])
    kk = br.rand(len(r)) < P["pb"]
    for vi, ui, rr in zip(v[kk], u[kk], r[kk]):
        a = br.uniform(0, 2 * np.pi); d = rr * br.uniform(0.7, 1.1)
        cv2.circle(can, (int(round((ui + d * np.cos(a) + UHALF) * 4)), int(round((vi + d * np.sin(a) - VMIN) * 4))),
                   int(round(max(0.6, rr * br.uniform(0.25, 0.5)) * 4)), 255, -1, cv2.LINE_AA, 2)
    fp = can > 127
    return cv2.morphologyEx(fp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0


def sized_paint(rng, tpl: Template, frac=1.0):
    """Brush remix whose area is the template's own area x U(0.7, 1.4) x frac."""
    P = brush_params(rng, tpl)
    fp = brush_paint(tpl, P)
    want = tpl.U.sum() * rng.uniform(0.7, 1.4) * frac
    for _ in range(2):
        k = float(np.clip(np.sqrt(want / max(1, fp.sum())), 0.5, 1.6))
        if abs(k - 1) < 0.05:
            break
        P["su"] *= k; P["sv"] *= k; P["sr"] *= k
        fp = brush_paint(tpl, P)
    return fp


def make_shape(rng, tpls):
    t = tpls[rng.randint(len(tpls))]
    hedge = rng.rand() < 0.15                          # hedgerow: two neighbouring structures
    fp = sized_paint(rng, t, 0.6 if hedge else 1.0)
    src = [t.rec["key"]]
    if hedge:
        t2 = tpls[rng.randint(len(tpls))]
        fp2 = sized_paint(rng, t2, 0.6)
        c1 = np.nonzero(fp.any(0))[0]; c2 = np.nonzero(fp2.any(0))[0]
        if len(c1) and len(c2):
            shift = int((c1.max() - c1.min() + c2.max() - c2.min()) / 2 * rng.uniform(0.4, 0.9)) * (1 if rng.rand() < .5 else -1)
            fp = fp | np.roll(fp2, shift, 1)
            src.append(t2.rec["key"])
    return fp, t, src


# ------------------------------------------------------------------ appearance
def canvas_texture(rng):
    """Log-normal thread texture in limb coordinates: threads run radially (long in v),
    plus knots and blotches. Built on the canvas so the anisotropy follows the limb."""
    Hc, Wc = VMAX - VMIN, 2 * UHALF
    n = lambda sv, su: gaussian_filter(rng.standard_normal((Hc, Wc)), (sv, su))
    th = n(rng.uniform(2.5, 5.0), rng.uniform(0.5, 0.9))
    if rng.rand() < 0.4:                     # some prominences are tangential sheets / arches
        th = 0.6 * th / (th.std() + 1e-6) + 0.6 * (lambda t: t / (t.std() + 1e-6))(n(0.6, 3.0))
    z = (0.6 * th / (th.std() + 1e-6) + 0.35 * (lambda t: t / (t.std() + 1e-6))(n(0.8, 0.8))
         + 0.35 * (lambda t: t / (t.std() + 1e-6))(n(6, 6)))
    z = z / (z.std() + 1e-6)
    return np.exp(rng.uniform(0.3, 0.45) * z - 0.08).astype(np.float32)


def colour_lut(bgr):
    """AIA-304 colormap as seen in this frame: summed level (0..765) -> median BGR."""
    lev = bgr.astype(np.int32).sum(2).ravel()
    px = bgr.reshape(-1, 3).astype(np.float32)
    lut = np.full((766, 3), np.nan, np.float32)
    order = np.argsort(lev); lev_s = lev[order]; px_s = px[order]
    edges = np.searchsorted(lev_s, np.arange(767))
    for s in range(766):
        a, b = edges[s], edges[s + 1]
        if b - a >= 3:
            lut[s] = np.median(px_s[a:b], 0)
    for c in range(3):
        ok = ~np.isnan(lut[:, c])
        lut[:, c] = np.interp(np.arange(766), np.nonzero(ok)[0], lut[ok, c])
        lut[:, c] = np.maximum.accumulate(lut[:, c])       # the 304 map is monotone per channel
    return lut


EFFECTS = ("gauss", "glow", "radial")
LABEL_TAU = 0.35          # diffused brush coverage above this counts as prominence in the label


def to_canvas(field, cx, cy, R, th0):
    vv, uu = np.mgrid[VMIN:VMAX, -UHALF:UHALF].astype(np.float32)
    th = th0 + uu / R
    x = cx + (R + vv) * np.cos(th); y = cy + (R + vv) * np.sin(th)
    return map_coordinates(field, [y, x], order=1, mode="constant")


def make_effect(kind, strength, geo):
    """Linear spatial operator applied identically to the brush opacity, the emission and the label.
    strength in [0,1]. geo = (cx, cy, R, th0, rr, cc, valid) for the radial smear."""
    if kind is None:
        return (lambda x: x), {}
    if kind == "gauss":
        sig = 1.0 + 3.0 * strength
        return (lambda x: gaussian_filter(x, sig)), dict(sigma=round(sig, 2))
    if kind == "glow":
        sig = 3.0 + 6.0 * strength; k = 0.9 + 0.9 * strength
        # core untouched, halo only where the spread envelope exceeds it (no brightening of the core)
        return (lambda x: np.maximum(x, k * gaussian_filter(x, sig))), dict(sigma=round(sig, 2), gain=round(k, 2))
    if kind == "radial":
        Lr = 2.0 + 7.0 * strength
        cx, cy, R, th0, rr, cc, valid = geo
        d = np.arange(int(4 * Lr) + 1); w = np.exp(-d / Lr); w /= w.sum()
        def f(x):
            can = to_canvas(x, cx, cy, R, th0)
            sm = np.zeros_like(can)
            for di, wi in zip(d, w):                   # one-sided: material is pushed OUTWARD (larger v)
                sm[di:] += wi * can[:can.shape[0] - di]
            sm = 0.35 * can + 0.65 * sm
            return map_coordinates(sm, [rr, cc], order=1, mode="constant") * valid
        return f, dict(length=round(Lr, 2))
    raise ValueError(kind)


def render(rng, bgr, fp_img, vmap, level_ratio, band_level, tex, transparency=None, eff=None, fall_range=(45, 95)):
    """Blend a semi-transparent emission layer into bgr where fp_img (float 0..1 soft footprint)."""
    lev = bgr.astype(np.float32).sum(2)
    H = rng.uniform(*fall_range)
    fall = np.exp(-np.clip(vmap, 0, None) / H)
    target = band_level * level_ratio * tex * fall
    op = rng.uniform(0.85, 1.0)                     # default: transparency 0-15 %
    if transparency is not None:
        op = 1.0 - transparency                      # brush opacity = 1 - transparency
    alpha = np.clip(fp_img * 1.15, 0, 1) * op
    alpha = np.where(vmap < 0, alpha * 0.3, alpha)               # over the bright rim it barely shows
    emis = target * alpha
    label_field = np.clip(fp_img * 1.15, 0, 1)
    if eff is not None:                                          # blur / diffusion: same operator on all three
        alpha = np.clip(eff(alpha), 0, 1); emis = eff(emis); label_field = eff(label_field)
    new = np.maximum(lev, lev * (1 - alpha) + emis)
    lut = colour_lut(bgr)
    col = lut[np.clip(new, 0, 765).astype(int)]
    out = bgr.astype(np.float32)
    m = alpha > 0.01
    out[m] = col[m]
    out = np.where(m[..., None], cv2.GaussianBlur(out, (0, 0), 0.6), out)
    ok, enc = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return cv2.imdecode(enc, 1), label_field


# ------------------------------------------------------------------ placement
def real_box_angles(xml, cx, cy):
    out = []
    for b in L.boxes_of(xml, "prominence"):
        x1, y1, x2, y2 = b
        corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], float)
        a = np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx)
        c = np.angle(np.exp(1j * a).mean())
        out.append((c, np.abs(np.angle(np.exp(1j * (a - c)))).max()))
    return out


def synth_one(rng, tpls, host_bgr, host_xml, margins, transparency=None, effect=None, strength=None, geo=None, geo_kind=None):
    cx, cy, R, rmap = L.solar_region_colour(host_bgr)
    avoid = real_box_angles(host_xml, cx, cy)
    # burnt-in timestamp / instrument text: white, unsaturated, outside the Sun
    hsv = cv2.cvtColor(host_bgr, cv2.COLOR_BGR2HSV)
    text = (hsv[..., 1] < 70) & (hsv[..., 2] > 170) & (rmap > R + 3)
    text = cv2.dilate(text.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    for _ in range(40):
        if geo:                                        # procedural straight / arc shape (geo_shapes.py)
            import geo_shapes as G
            fp, ggroup, gkind = G.make_geo(rng, geo, geo_kind)
            if not fp.any():
                continue
            t = tpls[rng.randint(len(tpls))]           # borrow a human label's brightness only
            src = [f"geo:{ggroup}:{gkind}"]
        else:
            fp, t, src = make_shape(rng, tpls)
        th0 = rng.uniform(-np.pi, np.pi)
        cols = np.nonzero(fp.any(0))[0]
        half = (max(abs(cols.min() - UHALF), abs(cols.max() - UHALF)) + 6) / R
        if any(abs(np.angle(np.exp(1j * (th0 - c)))) < half + w + 0.08 for c, w in avoid):
            continue
        rr, cc = roll_maps(cx, cy, R, th0)
        valid = (rr >= 0) & (rr < VMAX - VMIN) & (cc >= 0) & (cc < 2 * UHALF)
        mask = (map_coordinates(fp.astype(np.float32), [rr, cc], order=0, mode="constant") > 0.5) & valid
        if mask.sum() < 40 or (rmap[mask] > R).mean() < 0.4 or (mask & (rmap > R)).sum() < 60:
            continue                                   # mostly on-disk -> invisible, not a usable sample
        ys, xs = np.nonzero(mask)
        if xs.min() < 3 or ys.min() < 3 or xs.max() > 508 or ys.max() > 508:
            continue
        if (cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)[text].any():
            continue                                   # would paint over the timestamp text
        soft = gaussian_filter(fp.astype(np.float32), 0.9)
        fpi = map_coordinates(soft, [rr, cc], order=1, mode="constant") * valid
        ang = np.arctan2(*(np.mgrid[0:512, 0:512] - np.array([[[cy]], [[cx]]])))
        band = (rmap > R - 8) & (rmap < R - 2) & (np.abs(np.angle(np.exp(1j * (ang - th0)))) * R < 60)
        band_level = float(np.median(host_bgr.astype(np.float32).sum(2)[band])) if band.sum() > 10 else 300.0
        lr = t.rec["level_ratio"] or 0.35
        lr = float(np.clip(lr * 1.65 * rng.uniform(0.85, 1.15), 0.3, 1.4))   # template median is height-averaged; base is brighter
        tex = map_coordinates(canvas_texture(rng), [rr, cc], order=1, mode="nearest")
        kind = EFFECTS[rng.randint(len(EFFECTS))] if effect == "random" else effect
        stg = float(rng.uniform(0.2, 1.0)) if (kind and strength is None) else strength
        eff, eprm = make_effect(kind, stg, (cx, cy, R, th0, rr, cc, valid))
        img, lab_field = render(rng, host_bgr, fpi, rmap - R, lr, band_level, tex, transparency,
                                eff if kind else None, (90, 170) if geo else (45, 95))
        # visibility gate: the label must be something one can actually see
        gain = img.astype(np.float32).sum(2) - host_bgr.astype(np.float32).sum(2)
        offl = mask & (rmap > R + 1)
        if offl.sum() < 30 or (gain[offl] > 25).mean() < 0.6:
            continue
        mask0 = mask
        if kind:
            mask = mask | ((lab_field > LABEL_TAU) & valid)     # the label diffuses with the image
            if mask[text].any():
                continue
            ys, xs = np.nonzero(mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        w, h = x2 - x1, y2 - y1
        mL, mR, mT, mB = margins[rng.randint(len(margins))]
        box = [int(np.clip(x1 - mL * w, 0, 511)), int(np.clip(y1 - mT * h, 0, 511)),
               int(np.clip(x2 + mR * w, 1, 512)), int(np.clip(y2 + mB * h, 1, 512))]
        return img, mask, box, dict(src=src, lab=(src[0] if geo else t.rec["lab"]), th0=float(th0), level_ratio=lr, transparency=transparency,
                                   effect=kind, strength=stg, effect_params=eprm, area_before_effect=int(mask0.sum()),
                                   area=int(mask.sum()), ondisk=float((rmap[mask] < R).mean()))
    return None


def box_margins():
    g = json.load(open(B / "synth_prom/gold_shapes.json"))
    m = []
    for o in g:
        b, k = o["box"], o["mbox"]; w, h = k[2] - k[0], k[3] - k[1]
        m.append(tuple(np.clip([(k[0] - b[0]) / w, (b[2] - k[2]) / w, (k[1] - b[1]) / h, (b[3] - k[3]) / h], -0.1, 0.5)))
    return m


def hosts(split="train"):
    xs = sorted(p for p in glob.glob(str(DATA / split / "*.xml")) if "AIA_304" in os.path.basename(p))
    return [(p, p[:-4] + ".jpg") for p in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="synth_prom/samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train", help="host frames")
    ap.add_argument("--transparency", default=None,
                    help="브러시 투명도. 단일값(0.3) 또는 범위(0.1,0.5)에서 균등 추출. 미지정 시 기본 0~15%%")
    ap.add_argument("--effect", default=None, choices=[None, "gauss", "glow", "radial", "random"],
                    help="합성 영역 블러/확산 효과 (레이블도 같이 확산). random = 세 가지 중 무작위")
    ap.add_argument("--geo", default=None, choices=[None, "straight", "arc", "both"],
                    help="사람 형상 대신 도형 도구로 만든 직선형/아크형 분출 형상 사용")
    ap.add_argument("--fx-prob", type=float, default=1.0, help="--effect random 일 때 효과를 거는 비율")
    ap.add_argument("--tr-prob", type=float, default=1.0, help="--transparency 를 적용하는 비율")
    ap.add_argument("--strength", default=None, help="효과 강도 0~1. 단일값 또는 범위(0.2,1.0). 미지정 시 0.2~1.0")
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    out = B / a.out
    for s in ("images", "masks"):
        (out / s).mkdir(parents=True, exist_ok=True)
    tpls = load_templates(); mg = box_margins(); H = hosts(a.split)
    print(f"templates {len(tpls)}, host frames {len(H)}")
    rows = []
    while len(rows) < a.n:
        xml, jpg = H[rng.randint(len(H))]
        bgr = cv2.imread(jpg)
        tr = None
        if a.transparency and rng.rand() < a.tr_prob:
            lo, hi = (float(x) for x in (a.transparency.split(",") * 2)[:2])
            tr = float(rng.uniform(lo, hi))
        sg = None
        if a.strength:
            lo, hi = (float(x) for x in (a.strength.split(",") * 2)[:2])
            sg = float(rng.uniform(lo, hi))
        fx = a.effect if (a.effect != "random" or rng.rand() < a.fx_prob) else None
        r = synth_one(rng, tpls, bgr, xml, mg, tr, fx, sg, a.geo)
        if r is None:
            continue
        img, mask, box, meta = r
        sid = f"syn{len(rows):04d}"
        cv2.imwrite(str(out / "images" / f"{sid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "masks" / f"{sid}.png"), mask.astype(np.uint8) * 255)
        rows.append(dict(sid=sid, host=Path(jpg).stem, box=box, **meta))
        if len(rows) % 50 == 0:
            print(" ", len(rows), flush=True)
    json.dump(rows, open(out / "index.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
