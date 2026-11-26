#!/usr/bin/env python3
"""Retry the boxes that have no label in the release (quality gate failed) with the SAME fine-tuned SAM 3 heads,
varying only how the model is prompted, and measure every prompting variant on boxes whose answer is known.

Sets per class:  fail = boxes without a label,  gold = human-labelled boxes (IoU vs human),
                 ok = random confirmed boxes (IoU vs the shipped mask: does the variant reproduce the normal label?)
Variants = seed (cv | ext[, body margin]) x prompt (pt | box | boxpt) x crop pad x logit threshold.
The quality gate is the shipped one, unchanged.

    .venv_train/bin/python retry_unlabeled.py --cls sunspot --device cuda:0
"""
import argparse, csv, json, random, sys, time
from pathlib import Path
import cv2, numpy as np, torch
B = Path(__file__).resolve().parent; sys.path.insert(0, str(B))
import label_other_classes as L
from build_release import human, CLS
DATA = B.parent / "dataset"; F = 512
CK = dict(sunspot="gold_sunspot/runs/sam3_sunspot_deploy.pt", prominence="synth_prom/final_model/sam3_prom_human133_r215_hs100_strands_hazearcs_cp_microarcs.pt")
PADS = dict(sunspot=[0.8, 0.3, 1.5], prominence=[0.5, 0.2, 1.0]); THR = [0.0, -1.0, -2.0]
SEEDS = dict(sunspot=[("cv", 0), ("ext", 0)], prominence=[("cv", -3), ("ext", -3), ("ext", -10), ("ext", -20)])


def ref_of(gup, bq, want, allow_up, mode):
    if mode == "cv": return L.cv_reference(gup, bq, want, ~allow_up if want == "bright" else allow_up)[0]
    x1, y1, x2, y2 = bq; a = np.zeros_like(allow_up); a[y1:y2, x1:x2] = allow_up[y1:y2, x1:x2]
    if a.sum() < 10: return None
    v = gup[a]; r = a & ((gup <= np.percentile(v, 10)) if want == "dark" else (gup >= np.percentile(v, 90)))
    return r if r.sum() >= 3 else None


def variants(model, proc, cfg, cls, rgb, gray, box, disk, bodies, dev):
    """-> {name: (mask512, gate dict)} for every variant that produced a candidate."""
    out = {}; x1, y1, x2, y2 = box; bm = np.zeros((F, F), bool); bm[y1:y2, x1:x2] = True
    for pad in PADS[cls]:
        up, bq, info = L.crop_ctx(rgb, box, pad, cfg["up"]); gup = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY).astype(np.float32)
        emb = None; cx1, cy1, cx2, cy2 = info[:4]
        denom = (x2 - x1) * (y2 - y1) if cfg.get("clip_box", True) else (cx2 - cx1) * (cy2 - cy1)
        for mode, mg in SEEDS[cls]:
            inside = bodies[mg] if cls == "prominence" else disk
            ins_up = L.crop_ctx(np.repeat(inside[:, :, None].astype(np.uint8) * 255, 3, axis=2), box, pad, cfg["up"])[0][:, :, 0] > 127
            allow = ~ins_up if cls == "prominence" else ins_up
            ref = ref_of(gup, bq, cfg["want"], allow, mode)
            if ref is None: continue
            ys, xs = np.where(ref); v = gup[ys, xs]; j = int(np.argmin(v)) if cfg["want"] == "dark" else int(np.argmax(v))
            pt = [float(xs[j]), float(ys[j])]; ref_full = L.paste_back(ref, info)
            for pr in ("pt", "box", "boxpt"):
                kw = {}
                if pr != "box": kw.update(input_points=[[[pt]]], input_labels=[[[1]]])
                if pr != "pt": kw.update(input_boxes=[[[float(q) for q in bq]]])
                inp = proc(images=up, return_tensors="pt", **kw).to(dev)
                with torch.no_grad():
                    if emb is None: emb = model.get_image_embeddings(inp["pixel_values"])
                    o = model(image_embeddings=emb, multimask_output=True, **{k: inp[k] for k in ("input_points", "input_labels", "input_boxes") if k in inp})
                lg = o.pred_masks[0].float(); lg = (lg.flatten(0, 1) if lg.dim() > 3 else lg).cpu().numpy()
                for th in THR:
                    best, bi_ = None, -1.0
                    for k in range(lg.shape[0]):
                        full = L.paste_back(cv2.resize((lg[k] > th).astype(np.uint8), (F, F), interpolation=cv2.INTER_NEAREST_EXACT) > 0, info)
                        if cfg["clip_disk"]: full &= disk
                        if cfg.get("clip_box", True): full &= bm
                        cov = full.sum() / max(1, denom)
                        if not (cfg["cov_lo"] <= cov <= L.COV_HI): continue
                        u = (full | ref_full).sum(); iou = (full & ref_full).sum() / u if u else 0.0
                        if iou > bi_: best, bi_ = full, iou
                    if best is None: continue
                    area = int(best.sum()); cov = area / max(1, denom)
                    rel = L.rel_intensity(gray, best, box, inside if cls == "prominence" else disk, cfg["want"])
                    ok = area >= cfg["min_area"] and cfg["cov_lo"] <= cov <= L.COV_HI and rel is not None and ((rel < 0.98) if cfg["want"] == "dark" else (rel > 1.02))
                    out[f"{mode}{mg}|{pr}|pad{pad}|thr{th:g}"] = (best, dict(area=area, cov=round(cov, 4), rel=None if rel is None else round(rel, 4), score=round(float(bi_), 4), ok=int(ok)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cls", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n-ok", type=int, default=400)
    a = ap.parse_args(); cls = a.cls; cfg = L.CFG[cls]; dev = a.device; O = B / "retry_unlabeled" / cls; O.mkdir(parents=True, exist_ok=True)
    R = B / "release_20261122" / f"solar_{cls}_seg"; rows = list(csv.DictReader(open(R / "labels.csv", encoding="utf-8")))
    H = human(cls); random.seed(0)
    okr = random.sample([r for r in rows if r["label_source"] == "sam3"], a.n_ok)
    items = [("fail", r) for r in rows if r["label_source"] == "none"] + [("gold", r) for r in rows if r["label_source"] == "human"] + [("ok", r) for r in okr]
    items.sort(key=lambda t: (t[1]["split"], t[1]["stem"]))
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained(str(B / "sam3_model")); model = Sam3TrackerModel.from_pretrained(str(B / "sam3_model"))
    sd = model.state_dict(); ck = torch.load(B / CK[cls], map_location="cpu", weights_only=False)["state_dict"]; assert all(k in sd for k in ck); sd.update(ck); model.load_state_dict(sd)
    model = model.to(dev).eval(); res, masks, cache, t0 = [], {}, (None,), time.time()
    for i, (kind, r) in enumerate(items):
        stem, split, bi = r["stem"], r["split"], int(r["box_index"]); box = [int(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
        if cache[0] != stem:
            bgr = cv2.imread(str(DATA / split / f"{stem}.jpg")); gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); disk = L.S.solar_disk_mask(gray); bodies = {}
            if cls == "prominence":
                _, _, Rl, rmap = L.solar_region_colour(bgr); bodies = {mg: rmap <= (Rl + mg) for _, mg in SEEDS[cls]}
            cache = (stem, bgr, gray, disk, bodies)
        _, bgr, gray, disk, bodies = cache
        V = variants(model, proc, cfg, cls, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), gray, box, disk, bodies, dev)
        tgt = (cv2.imread(str(R / r["mask_path"]), 0) > 127) if kind != "fail" else None
        rec = dict(kind=kind, split=split, stem=stem, bi=bi, box=box, v={})
        for name, (m, g) in V.items():
            if tgt is not None: u = (m | tgt).sum(); g["iou"] = round(float((m & tgt).sum() / u), 4) if u else 1.0
            rec["v"][name] = g
            if kind == "fail" and g["ok"]: masks[f"{stem}__b{bi}@{name}"] = np.packbits(m)
        res.append(rec)
        if (i + 1) % 100 == 0: print(f"{i+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(O / "variants.json", "w")); np.savez_compressed(O / "fail_masks.npz", **masks); print("done", len(res), len(masks))


if __name__ == "__main__":
    main()
