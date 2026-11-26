#!/usr/bin/env python3
"""[train_final_microarcs.py + 58 new human labels from the 2230 service (gold_prominence_hs)]  Retrain from BASE SAM 3: human labels where the 10 labeller_1 labels (15, up to 2026-10-16) REPLACE earlier labels of the
same targets (32 - 9 + 10 = 33) + layered synthetic 2K with a 4 px inward label margin.
Train SAM 3 heads on the LAYERED synthetic set (dataset_comp, ~2K) + the 29 human labels
(labeller_5/labeller_6; mirrored, x5), then infer on TEST-split frames that contain no human label.

Embeddings (10.6 MB/sample) go to a disk memmap: 2K samples do not fit next to everything else in RAM.

    .venv_train/bin/python synth_prom/train_comp_sam3.py --device cuda:0
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F

B = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(B)); sys.path.insert(0, str(B / "synth_prom"))
import label_other_classes as L
import train_sam3_prominence as T

O = B / "synth_prom/final_model"; DATA = B.parent / "dataset"
CFG = L.CFG["prominence"]; FRAME = 512
SHAPES = [(32, 288, 288), (64, 144, 144), (256, 72, 72)]


def make_rec(args):
    from exp_sam3 import synth_record
    from margin import shrink
    d, r, mg = args
    img = cv2.imread(str(d / "images" / f"{r['sid']}.jpg")); m = shrink(cv2.imread(str(d / "masks" / f"{r['sid']}.png"), 0) > 0, mg)
    rec = synth_record(img, r["box"], m)
    if rec is None: return None
    for k in ("body", "gold"): rec.pop(k, None)
    rec["sid"] = r["sid"]
    for k in ("ref", "tgt", "reg", "reach"): rec[k] = np.packbits(rec[k])        # RAM: 1 MB -> 128 KB per record
    rec["up"] = cv2.imencode(".png", rec["up"])[1]                                   # lossless, decoded once at encode time
    return rec


def unpack(v):
    return np.unpackbits(v)[: 512 * 512].reshape(512, 512).astype(bool) if v.ndim == 1 else v


def up_img(v):
    return cv2.imdecode(v, cv2.IMREAD_UNCHANGED) if v.ndim == 1 else v


def train(model, recs, emb_of, epochs, dev, lr=1e-4, pos_weight=3.0, alpha=0.3, seed=0):
    for p in model.vision_encoder.parameters(): p.requires_grad_(False)
    params = [p for m in (model.mask_decoder, model.prompt_encoder) for p in m.parameters()]
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(recs))
    rng = np.random.RandomState(seed); model.train(); model.vision_encoder.eval(); hist = []; t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for i in rng.permutation(len(recs)):
            r = recs[i]; emb, pts, lbs = emb_of(i)
            ref_t, tgt_t, reg_t, reach_t = (torch.from_numpy(unpack(r[k])).float().to(dev) for k in ("ref", "tgt", "reg", "reach"))
            lg = T.decode(model, emb, pts, lbs)
            lg_up = F.interpolate(lg[None], size=(512, 512), mode="bilinear", align_corners=False)[0]
            sel = lg_up[T.pick(lg_up, ref_t, reach_t)]
            loss = T.masked_bce(sel, tgt_t, reg_t, pos_weight) + T.soft_tversky(sel, tgt_t, reg_t, alpha, 1 - alpha)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sched.step(); tot += float(loss)
        hist.append(tot / len(recs)); print(f"  epoch {ep+1}/{epochs} loss {hist[-1]:.4f}  {time.time()-t0:.0f}s", flush=True)
    model.eval(); return hist


def heads(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(("mask_decoder", "prompt_encoder"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--human-repeat", type=int, default=5); ap.add_argument("--margin", type=int, default=4)
    a = ap.parse_args(); dev = a.device; O.mkdir(parents=True, exist_ok=True)
    # ---- records (CPU pool)
    jobs = []
    for d in sorted((B / "synth_prom/dataset_comp").glob("p*")):
        jobs += [(d, r, a.margin) for r in json.load(open(d / "index.json"))]
    for d in sorted((B / "synth_prom/dataset_strands").glob("p*")):
        jobs += [(d, r, 0) for r in json.load(open(d / "index.json"))]
    for d in sorted((B / "synth_prom/dataset_hazearcs").glob("p*")):
        jobs += [(d, r, 0) for r in json.load(open(d / "index.json"))]
    for d in sorted((B / "synth_prom/dataset_copypaste").glob("p*")):
        jobs += [(d, r, 0) for r in json.load(open(d / "index.json"))]      # human-anchored copy-paste, labels as painted
    for d in sorted((B / "synth_prom/dataset_microarcs").glob("p*")):
        jobs += [(d, r, 0) for r in json.load(open(d / "index.json"))]      # micro overlapping arcs (~1/5 area, 1.5-3 px strokes)
    t0 = time.time()
    with Pool(14) as pool:
        syn = [r for r in pool.map(make_rec, jobs, chunksize=8) if r is not None]
    print(f"synthetic records {len(syn)}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained(str(B / "sam3_model"))
    model = Sam3TrackerModel.from_pretrained(str(B / "sam3_model")).to(dev).eval()
    base = heads(model)
    real = [r for r in T.load_records(L.LIMB_MARGIN) if r["evaluator"] in ("labeller_5", "labeller_6")]
    # new labels (labeller_1, 2026-10-14) replace any earlier label of the SAME target (stem, box index)
    T.GOLD = B / "gold_prominence_r2"
    r2 = [r for r in T.load_records(L.LIMB_MARGIN) if r["evaluator"] == "labeller_1"]
    T.GOLD = B / "gold_prominence"
    newk = {(r["stem"], r["bi"]) for r in r2}
    dropped = [f"{r['evaluator']}:{r['stem']}__b{r['bi']}" for r in real if (r["stem"], r["bi"]) in newk]
    real = [r for r in real if (r["stem"], r["bi"]) not in newk] + r2
    # 2026-11-04~05: 58 new human labels from the 2230 service (labeller_6, corrections of the _cp model), different frames
    T.GOLD = B / "gold_prominence_hs"
    hs = T.load_records(L.LIMB_MARGIN)
    T.GOLD = B / "gold_prominence"
    print(f"new 2230 labels: {len(hs)}", flush=True)
    real = real + hs
    print(f"human labels: {len(real)} (dropped {len(dropped)} older duplicates: {dropped})", flush=True)
    human = []
    for r in real:
        for v in (r, T.flip_rec(r)):
            v = dict(v); v.pop("body", None); v.pop("gold", None); human.append(v)
    uniq = syn + human                                   # unique samples to encode
    N = len(uniq)
    mm = [np.lib.format.open_memmap(O / f"emb_hs{j}.npy", mode="w+", dtype=np.float16, shape=(N,) + s) for j, s in enumerate(SHAPES)]
    pts_all, lbs_all = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(uniq):
            inp = proc(images=up_img(r["up"]), input_points=[[[r["pt"]]]], input_labels=[[[1]]], return_tensors="pt").to(dev)
            emb = model.get_image_embeddings(inp["pixel_values"])
            for j, e in enumerate(emb): mm[j][i] = e[0].half().cpu().numpy()
            pts_all.append(inp["input_points"].cpu()); lbs_all.append(inp["input_labels"].cpu()); r.pop("up")
            if (i + 1) % 250 == 0: print(f"  encode {i+1}/{N}  {time.time()-t0:.0f}s", flush=True)
    for m in mm: m.flush()
    mm = [np.load(O / f"emb_hs{j}.npy", mmap_mode="r") for j in range(3)]
    # training index: every synthetic once, every human view x repeat
    order = list(range(len(syn))) + [len(syn) + h for h in range(len(human)) for _ in range(a.human_repeat)]
    recs = [uniq[i] for i in order]

    def emb_of(k):
        i = order[k]
        return ([torch.from_numpy(np.ascontiguousarray(m[i]))[None].to(dev, non_blocking=True) for m in mm],
                pts_all[i].to(dev), lbs_all[i].to(dev))

    print(f"training samples: layered synthetic {len(syn)} + human {len(real)} x2 mirror x{a.human_repeat} = {len(recs)}", flush=True)
    hist = train(model, recs, emb_of, a.epochs, dev)
    sd = heads(model)
    gold_stems = sorted({r["stem"] for r in real})
    torch.save({"state_dict": {k: v.cpu() for k, v in sd.items()}, "n_synth": len(syn), "n_human": len(real),
                "human_repeat": a.human_repeat, "epochs": a.epochs, "gold_stems": gold_stems, "loss": hist,
                "synth_label_margin_px": a.margin, "synth": "layered(m4) + strands + hazearcs + copypaste + microarcs (m0)", "replaced_duplicates": dropped, "human_sources": ["gold_prominence (labeller_5/labeller_6)", "gold_prominence_r2", f"gold_prominence_hs (2230, {len(hs)})"], "human_labels": len(real), "trained": ["mask_decoder", "prompt_encoder"]}, O / f"sam3_prom_human{len(real)}_r215_hs{len(hs)}_strands_hazearcs_cp_microarcs.pt")
    del mm
    for j in range(3): os.remove(O / f"emb_hs{j}.npy")                    # 22 GB scratch

    print("done")


if __name__ == "__main__":
    main()
