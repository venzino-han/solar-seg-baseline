#!/usr/bin/env python3
"""Pick the retried label per unlabelled box and write the evaluation data.

Policy: variants are ranked by mean IoU against human labels; only variants within 0.05 of the shipped prompting are
eligible; each unlabelled box takes the best-ranked eligible variant whose mask passes the shipped quality gate.
Outputs retry_unlabeled/<cls>/{selected.csv, masks/<stem>__b<bi>.png, summary.json, sheet_*.jpg}
"""
import csv, json, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np
B = Path(__file__).resolve().parent; sys.path.insert(0, str(B)); DATA = B.parent / "dataset"
import label_other_classes as L
BASE = dict(sunspot="cv0|pt|pad0.8|thr0", prominence="cv-3|pt|pad0.5|thr0"); TOL = 0.05


def contrast(gray, disk, box, want):
    """How visible is anything in the box: extreme 5 % of the box vs the box median (sunspot: <1 dark, prominence: >1 bright)."""
    x1, y1, x2, y2 = box; v = gray[y1:y2, x1:x2][disk[y1:y2, x1:x2] if want == "dark" else ~disk[y1:y2, x1:x2]].astype(float)
    if v.size < 10: return None
    return float(np.percentile(v, 5 if want == "dark" else 95) / max(1.0, np.median(v)))


def load(cls):
    """-> records with merged variants, {mask key: packed mask}. Wide-disk sunspot variants appear twice:
    'W..' judged by the shipped gate and 'W..|dg' judged by the same gate with a disk-only background."""
    O = B / "retry_unlabeled" / cls; R = json.load(open(O / "variants.json")); Z = dict(np.load(O / "fail_masks.npz"))
    if (O / "variants_wide.json").exists():
        W = {(r["stem"], r["bi"]): r for r in json.load(open(O / "variants_wide.json"))}; ZW = np.load(O / "fail_masks_wide.npz")
        for r in R:
            for n, g in W[(r["stem"], r["bi"])]["v"].items():
                r["v"][n] = g; r["v"][n + "|dg"] = dict(g, ok=g["ok_disk"])
                k = f"{r['stem']}__b{r['bi']}@{n}"
                if k in ZW.files: Z[k] = Z[k + "|dg"] = ZW[k]
    return R, Z


def stats(R):
    names = sorted({n for r in R for n in r["v"]}); st = {}
    for n in names:
        g = [r["v"].get(n, {}).get("iou", 0.0) for r in R if r["kind"] == "gold"]; o = [r["v"].get(n, {}).get("iou", 0.0) for r in R if r["kind"] == "ok"]
        st[n] = dict(gold=float(np.mean(g)), ok=float(np.mean(o)), gold_pass=float(np.mean([r["v"].get(n, {}).get("ok", 0) for r in R if r["kind"] == "gold"])),
                     ok_pass=float(np.mean([r["v"].get(n, {}).get("ok", 0) for r in R if r["kind"] == "ok"])), fail_pass=sum(r["v"].get(n, {}).get("ok", 0) for r in R if r["kind"] == "fail"))
    return st


def main(cls):
    O = B / "retry_unlabeled" / cls; R, Z = load(cls); st = stats(R); json.dump(st, open(O / "variant_stats.json", "w"), indent=1)
    unpack = lambda k: np.unpackbits(Z[k])[:512 * 512].reshape(512, 512) > 0
    pack = {(r["stem"], int(r["bi"])): r for r in csv.DictReader(open(B / dict(sunspot="label_pack_sunspot_ft", prominence="label_pack_prominence_hs133")[cls] / "labels.csv"))}
    elig = sorted([n for n in st if st[n]["gold"] >= st[BASE[cls]]["gold"] - TOL and n != BASE[cls]], key=lambda n: -st[n]["gold"])
    # extra check for retried sunspot masks only: after dividing out limb darkening the mask must be as dark as human-labelled
    # spots are: at least 5 % darker than the photosphere at the same radius. It removes limb slivers that only look dark because the limb is dark.
    FLAT = None
    if cls == "sunspot":
        import retry_unlabeled_wide as RW
        rel = lambda gray, m: (lambda f: float(np.median(f[m])) if m.sum() else 9.0)(RW.wide_disk(gray)[1])
        from build_release import human
        hv = [rel(cv2.imread(str(next(DATA.glob(f"*/{k[0]}.jpg"))), 0), cv2.imread(str(v[0]), 0) > 127) for k, v in human(cls).items()]
        FLAT = 0.95; print("flat-intensity limit", FLAT, "| human masks: median", round(float(np.median(hv)), 3), "p95", round(float(np.percentile(hv, 95)), 3), "share under limit", round(float(np.mean(np.array(hv) <= FLAT)), 3))
        HUMAN_FLAT = dict(median=round(float(np.median(hv)), 3), p95=round(float(np.percentile(hv, 95)), 3), share_under_limit=round(float(np.mean(np.array(hv) <= FLAT)), 3))
    import shutil; shutil.rmtree(O / "masks", ignore_errors=True); (O / "masks").mkdir(); sel, want = [], L.CFG[cls]["want"]
    for r in R:
        stem, bi, box = r["stem"], r["bi"], r["box"]; gray = cv2.imread(str(DATA / r["split"] / f"{stem}.jpg"), 0); disk = L.S.solar_disk_mask(gray)
        ys, xs = np.where(disk); cx, cy, rad = xs.mean(), ys.mean(), (xs.max() - xs.min() + ys.max() - ys.min()) / 4
        d = dict(kind=r["kind"], split=r["split"], stem=stem, bi=bi, box=" ".join(map(str, box)), year=stem[:4], box_area=(box[2] - box[0]) * (box[3] - box[1]),
                 r_over_R=round(float(np.hypot((box[0] + box[2]) / 2 - cx, (box[1] + box[3]) / 2 - cy) / rad), 3), contrast=contrast(gray, disk, box, want))
        if r["kind"] == "fail":
            d["orig_reason"] = pack[(stem, bi)]["note"] or "gate"; d["n_variants_pass"] = sum(g["ok"] for g in r["v"].values()); d["n_eligible_pass"] = sum(r["v"].get(n, {}).get("ok", 0) for n in elig)
            key = lambda n: f"{stem}__b{bi}@{n}"; cand = [n for n in elig if r["v"].get(n, {}).get("ok")]
            if FLAT is not None:
                fr = {n: rel(gray, unpack(key(n))) for n in cand}; d["n_rejected_flat"] = sum(v > FLAT for v in fr.values()); cand = [n for n in cand if fr[n] <= FLAT]
            pick = cand[0] if cand else None; d["variant"] = pick or ""
            if pick:
                m = unpack(key(pick)); g = r["v"][pick]; d.update(area=g["area"], rel=g["rel"], cov=g["cov"])
                if FLAT is not None: d["flat_rel"] = round(fr[pick], 3)
                # stability: IoU of the picked mask with the other eligible variants that also pass
                oth = [unpack(key(n)) for n in cand if n != pick and n.split("|dg")[0] != pick.split("|dg")[0]]
                d["stability"] = round(float(np.mean([(m & o).sum() / max(1, (m | o).sum()) for o in oth])), 3) if oth else ""
                cv2.imwrite(str(O / "masks" / f"{stem}__b{bi}.png"), m.astype(np.uint8) * 255)
        else:
            g = r["v"].get(BASE[cls], {}); d.update(area=g.get("area"), rel=g.get("rel"))
        sel.append(d)
    keys = sorted({k for d in sel for k in d}, key=lambda k: list(sel[0]).index(k) if k in sel[0] else 99)
    with open(O / "selected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(sel)
    F = [d for d in sel if d["kind"] == "fail"]; rec = [d for d in F if d["variant"]]; rem = [d for d in F if not d["variant"]]; ok = [d for d in sel if d["kind"] == "ok"]
    med = lambda xs, k: (lambda v: round(float(np.median(v)), 3) if v else None)([x[k] for x in xs if x.get(k) not in (None, "")])
    S = dict(flat_limit=FLAT, human_flat=HUMAN_FLAT if FLAT else None, eligible=[dict(variant=n, iou_human=round(st[n]["gold"], 3), reproduce_shipped=round(st[n]["ok"], 3), unlabeled_pass=st[n]["fail_pass"]) for n in elig],
             base=dict(variant=BASE[cls], iou_human=round(st[BASE[cls]]["gold"], 3)), n_fail=len(F), n_recovered=len(rec), n_remaining=len(rem),
             recovered_by_variant=Counter(d["variant"] for d in rec), recovered_by_reason=Counter(d["orig_reason"] for d in rec), remaining_by_reason=Counter(d["orig_reason"] for d in rem),
             any_variant_pass_but_not_eligible=sum(d["n_variants_pass"] > 0 for d in rem),
             recovered_by_split=Counter(d["split"] for d in rec), remaining_by_split=Counter(d["split"] for d in rem),
             median={g: {k: med(x, k) for k in ("area", "rel", "flat_rel", "contrast", "r_over_R", "box_area", "stability")} for g, x in (("recovered", rec), ("remaining", rem), ("confirmed_sample", ok))},
             stability_ge_0_5=sum(1 for d in rec if d["stability"] != "" and d["stability"] >= 0.5), stability_na=sum(1 for d in rec if d["stability"] == ""),
             remaining_r_gt_0_9=sum(d["r_over_R"] > 0.9 for d in rem), confirmed_r_gt_0_9=round(float(np.mean([d["r_over_R"] > 0.9 for d in ok])), 3))
    json.dump(S, open(O / "summary.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(S, indent=1, ensure_ascii=False))
    # contact sheets: crop x4 with box, recovered mask outline
    import random; random.seed(1)
    def tile(d, with_mask):
        bgr = cv2.imread(str(DATA / d["split"] / f"{d['stem']}.jpg")); x1, y1, x2, y2 = map(int, d["box"].split()); p = max(12, int(0.6 * max(x2 - x1, y2 - y1)))
        a, b, c, e = max(0, x1 - p), max(0, y1 - p), min(512, x2 + p), min(512, y2 + p); s = 220 / max(c - a, e - b)
        im = cv2.resize(bgr[b:e, a:c], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC); im2 = im.copy()
        for q in (im, im2): cv2.rectangle(q, (int((x1 - a) * s), int((y1 - b) * s)), (int((x2 - a) * s), int((y2 - b) * s)), (0, 255, 255), 1)
        if with_mask:
            m = cv2.imread(str(O / "masks" / f"{d['stem']}__b{d['bi']}.png"), 0)[b:e, a:c]; m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
            cn, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im2, cn, -1, (0, 255, 0), 1)
        t = np.zeros((240, 448, 3), np.uint8); t[:im.shape[0], :im.shape[1]] = im; t[:im2.shape[0], 224:224 + im2.shape[1]] = im2
        cv2.putText(t, f"{d['stem'][:13]} b{d['bi']} {d.get('variant','')[:22]}", (2, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA); return t
    for name, xs, wm in (("recovered", rec, True), ("remaining", rem, False), ("recovered_faint", [d for d in rec if d.get("flat_rel") and d["flat_rel"] > 0.88], True)):
        xs = random.sample(xs, min(32, len(xs)))
        if xs:
            ts = [tile(d, wm) for d in xs] + [np.zeros((240, 448, 3), np.uint8)] * (-len(xs) % 4)
            cv2.imwrite(str(O / f"sheet_{name}.jpg"), np.vstack([np.hstack(ts[i:i + 4]) for i in range(0, len(ts), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main(sys.argv[1])
