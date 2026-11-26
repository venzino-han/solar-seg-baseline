#!/usr/bin/env python3
"""Second retry for sunspots: the unlabelled boxes sit at the limb, OUTSIDE the disk mask the pipeline uses
(radius ~225 px, the visible limb is ~240 px), so the seed has nowhere to land and the mask is clipped away.
Here the disk is the visible limb (threshold fit, 2 px inside), and one more seed removes limb darkening first:

  seed  cv   = shipped rule (I < 0.90 x local quiet Sun)      ext = darkest 10 % of the box
        flat = darkest pixels after dividing by the radial median profile (limb darkening removed)
Same fine-tuned heads, same quality gate (measured against the wide disk). Sets fail / gold / ok as in retry_unlabeled.py.

    .venv_train/bin/python retry_unlabeled_wide.py --device cuda:0
"""
import argparse, csv, json, random, sys, time
from pathlib import Path
import cv2, numpy as np, torch
B = Path(__file__).resolve().parent; sys.path.insert(0, str(B))
import label_other_classes as L
DATA = B.parent / "dataset"; F = 512; PADS = [0.8, 0.3]; THR = [0.0, -1.0]; cfg = L.CFG["sunspot"]


def wide_disk(gray):
    """-> (disk mask 2 px inside the visible limb, image with limb darkening divided out)"""
    t = (gray > 0.25 * np.median(gray[200:312, 200:312])).astype(np.uint8); n, lab, st, cen = cv2.connectedComponentsWithStats(t)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); cx, cy = cen[k]; R = float(np.sqrt(st[k, cv2.CC_STAT_AREA] / np.pi))
    yy, xx = np.mgrid[:F, :F]; rr = np.hypot(xx - cx, yy - cy); ri = np.clip(rr.astype(int), 0, F)
    prof = np.ones(F + 1, np.float32)
    for r in range(int(R) + 1):
        v = gray[ri == r]; prof[r] = max(1.0, float(np.median(v))) if v.size else prof[r - 1]
    return rr <= R - 2, gray.astype(np.float32) / prof[ri], R


def up_of(a, box, pad):                      # same crop/upsample as the model input, for a float map in [0, 2)
    return L.crop_ctx(np.repeat(np.clip(a * 127, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2), box, pad, cfg["up"])[0][:, :, 0].astype(np.float32) / 127


def variants(model, proc, rgb, gray, box, disk, flat, dev):
    out = {}; x1, y1, x2, y2 = box; bm = np.zeros((F, F), bool); bm[y1:y2, x1:x2] = True; denom = (x2 - x1) * (y2 - y1)
    for pad in PADS:
        up, bq, info = L.crop_ctx(rgb, box, pad, cfg["up"]); gup = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY).astype(np.float32)
        ins = up_of(disk.astype(np.float32), box, pad) > 0.5; fup = up_of(flat, box, pad); a = np.zeros_like(ins); a[bq[1]:bq[3], bq[0]:bq[2]] = ins[bq[1]:bq[3], bq[0]:bq[2]]
        if a.sum() < 10: continue
        emb = None
        for mode in ("cv", "ext", "flat"):
            if mode == "cv": ref = L.cv_reference(gup, bq, "dark", ins)[0]; ref = None if ref is None else ref & a; val = gup
            elif mode == "ext": ref = a & (gup <= np.percentile(gup[a], 10)); val = gup
            else: ref = a & (fup < 0.90); val = fup
            if ref is None or ref.sum() < 3: continue
            ys, xs = np.where(ref); j = int(np.argmin(val[ys, xs])); pt = [float(xs[j]), float(ys[j])]; ref_full = L.paste_back(ref, info)
            inp = proc(images=up, return_tensors="pt", input_points=[[[pt]]], input_labels=[[[1]]]).to(dev)
            with torch.no_grad():
                if emb is None: emb = model.get_image_embeddings(inp["pixel_values"])
                o = model(image_embeddings=emb, multimask_output=True, input_points=inp["input_points"], input_labels=inp["input_labels"])
            lg = o.pred_masks[0].float(); lg = (lg.flatten(0, 1) if lg.dim() > 3 else lg).cpu().numpy()
            for th in THR:
                best, bi_ = None, -1.0
                for k in range(lg.shape[0]):
                    full = L.paste_back(cv2.resize((lg[k] > th).astype(np.uint8), (F, F), interpolation=cv2.INTER_NEAREST_EXACT) > 0, info) & disk & bm
                    cov = full.sum() / max(1, denom)
                    if not (cfg["cov_lo"] <= cov <= L.COV_HI): continue
                    u = (full | ref_full).sum(); iou = (full & ref_full).sum() / u if u else 0.0
                    if iou > bi_: best, bi_ = full, iou
                if best is None: continue
                area = int(best.sum()); cov = area / max(1, denom); rel = L.rel_intensity(gray, best, box, disk, "dark")
                ok = area >= cfg["min_area"] and cfg["cov_lo"] <= cov <= L.COV_HI and rel is not None and rel < 0.98      # shipped gate
                # the shipped gate compares the mask with the REST OF THE BOX, which at the limb is mostly black sky;
                # ok_disk is the same gate with the background restricted to the disk (reported separately)
                ring = bm & disk & ~best; rd = float(np.median(gray[best])) / max(1.0, float(np.median(gray[ring]))) if area >= 3 and ring.sum() >= 5 else None
                okd = area >= cfg["min_area"] and cfg["cov_lo"] <= cov <= L.COV_HI and rd is not None and rd < 0.98
                out[f"W{mode}|pt|pad{pad}|thr{th:g}"] = (best, dict(area=area, cov=round(cov, 4), rel=None if rel is None else round(rel, 4), rel_disk=None if rd is None else round(rd, 4),
                                                              flat_rel=round(float(np.median(flat[best])), 4), score=round(float(bi_), 4), ok=int(ok), ok_disk=int(okd)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args(); dev = a.device; O = B / "retry_unlabeled/sunspot"
    R = B / "release_20261122/solar_sunspot_seg"; rows = {(r["stem"], int(r["box_index"])): r for r in csv.DictReader(open(R / "labels.csv", encoding="utf-8"))}
    items = json.load(open(O / "variants.json"))                       # the same fail / gold / ok boxes as the first retry
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained(str(B / "sam3_model")); model = Sam3TrackerModel.from_pretrained(str(B / "sam3_model"))
    sd = model.state_dict(); sd.update(torch.load(B / "gold_sunspot/runs/sam3_sunspot_deploy.pt", map_location="cpu", weights_only=False)["state_dict"]); model.load_state_dict(sd)
    model = model.to(dev).eval(); res, masks, cache, t0 = [], {}, (None,), time.time()
    for i, it in enumerate(items):
        stem, split, bi, box, kind = it["stem"], it["split"], it["bi"], it["box"], it["kind"]
        if cache[0] != stem:
            bgr = cv2.imread(str(DATA / split / f"{stem}.jpg")); gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); cache = (stem, bgr, gray) + wide_disk(gray)
        _, bgr, gray, disk, flat, Rl = cache
        V = variants(model, proc, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), gray, box, disk, flat, dev)
        tgt = (cv2.imread(str(R / rows[(stem, bi)]["mask_path"]), 0) > 127) if kind != "fail" else None
        rec = dict(kind=kind, split=split, stem=stem, bi=bi, box=box, limb_R=round(Rl, 1), v={})
        for name, (m, g) in V.items():
            if tgt is not None: u = (m | tgt).sum(); g["iou"] = round(float((m & tgt).sum() / u), 4) if u else 1.0
            rec["v"][name] = g
            if kind == "fail" and (g["ok"] or g["ok_disk"]): masks[f"{stem}__b{bi}@{name}"] = np.packbits(m)
        res.append(rec)
        if (i + 1) % 100 == 0: print(f"{i+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(O / "variants_wide.json", "w")); np.savez_compressed(O / "fail_masks_wide.npz", **masks); print("done", len(res), len(masks))


if __name__ == "__main__":
    main()
