#!/usr/bin/env python3
"""Does brush-synthesised prominence data help SAM 3?  Day-grouped 5-fold over the 29 human
labels. Per fold, synthetic samples are generated ONLY from the training fold's templates, on
train-split host frames (gold frames are all test-split, so no host overlap).

Conditions: vanilla | real (human train fold) | synth (synthetic only) | real+synth
    .venv_train/bin/python synth_prom/exp_sam3.py --seed 0 --device cuda:0
"""
import argparse, json, sys, time
from argparse import Namespace
from pathlib import Path
import cv2, numpy as np, torch

B = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(B)); sys.path.insert(0, str(B / "synth_prom"))
import label_other_classes as L
import train_sam3_prominence as T
import synth as S

CFG = L.CFG["prominence"]; UP = CFG["up"]
HP = dict(lr=1e-4, pos_weight=3.0, tversky=1.0, alpha=0.3)


def synth_record(bgr, box, mask):
    """Same fields as train_sam3_prominence.load_records; loss region = whole crop."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _, _, R, rmap = L.solar_region_colour(bgr)
    body = rmap <= (R + L.LIMB_MARGIN)
    up, bq, info = L.crop_ctx(rgb, box, CFG["pad"], UP)
    gup = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY).astype(np.float32)
    body_up = L.crop_ctx(np.repeat(body[:, :, None].astype(np.uint8) * 255, 3, axis=2), box, CFG["pad"], UP)[0][:, :, 0] > 127
    ref, _ = L.cv_reference(gup, bq, CFG["want"], body_up)
    if ref is None:
        return None
    ys, xs = np.where(ref); j = int(np.argmax(gup[ys, xs]))
    cx1, cy1, cx2, cy2, cw, ch = info
    tgt = cv2.resize(mask.astype(np.uint8)[cy1:cy2, cx1:cx2], (UP, UP), interpolation=cv2.INTER_NEAREST_EXACT) > 0
    return dict(stem="syn", bi=0, box=box, day="syn", split="synth", edited=1, verdict="", evaluator="synth",
                up=up, pt=[float(xs[j]), float(ys[j])], ref=ref, tgt=tgt, reg=np.ones((UP, UP), bool),
                reach=np.ones((UP, UP), bool), info=info, body=body, gold=mask, sam3_area=0)


@torch.no_grad()
def encode_cpu(model, proc, recs, dev):
    out = []
    for r in recs:
        row = []
        for v in (r, T.flip_rec(r)):
            inp = proc(images=v["up"], input_points=[[[v["pt"]]]], input_labels=[[[1]]], return_tensors="pt").to(dev)
            emb = model.get_image_embeddings(inp["pixel_values"])
            row.append((tuple(e.half().cpu() for e in emb), inp["input_points"].cpu(), inp["input_labels"].cpu()))
        out.append(row)
    return out


class CpuCache:
    """Embeddings kept on CPU, moved to the GPU per step (synthetic sets are too big for VRAM)."""
    def __init__(self, rows, dev): self.rows, self.dev = rows, dev
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        return [([e.to(self.dev, non_blocking=True) for e in emb], p.to(self.dev), l.to(self.dev)) for emb, p, l in self.rows[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--nsyn", type=int, default=120)
    ap.add_argument("--ep-real", type=int, default=40)
    ap.add_argument("--ep-syn", type=int, default=15)
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(); dev = a.device
    (B / "synth_prom/pred").mkdir(exist_ok=True)
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained(str(B / "sam3_model"))
    model = Sam3TrackerModel.from_pretrained(str(B / "sam3_model")).to(dev).eval()
    base = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(("mask_decoder", "prompt_encoder"))}

    real = [r for r in T.load_records(L.LIMB_MARGIN) if r["evaluator"] in S.LABS]
    rc = encode_cpu(model, proc, real, dev)
    days = sorted({r["day"] for r in real}); dfold = {d: k % 5 for k, d in enumerate(days)}
    all_tpl = {t.rec["key"]: t for t in S.load_templates()}
    hosts = S.hosts("train"); mg = S.box_margins()

    def predict(recs, cache, i):
        emb, p, l = cache[i][0]
        return T.infer_full(model, proc, recs[i], dev, use_cache=([e.to(dev) for e in emb], p.to(dev), l.to(dev)))

    res = {c: {} for c in ("vanilla", "real", "synth", "real+synth")}
    rng = np.random.RandomState(1000 + a.seed)
    for f in range(5):
        te = [i for i, r in enumerate(real) if dfold[r["day"]] == f]
        tr = [i for i, r in enumerate(real) if dfold[r["day"]] != f]
        keys = {f"{real[i]['stem']}__b{real[i]['bi']}" for i in tr}
        tpls = [t for k, t in all_tpl.items() if k in keys]
        t0 = time.time(); syn = []
        while len(syn) < a.nsyn:
            xml, jpg = hosts[rng.randint(len(hosts))]
            out = S.synth_one(rng, tpls, cv2.imread(jpg), xml, mg)
            if out is None: continue
            rec = synth_record(out[0], out[2], out[1])
            if rec is not None: syn.append(rec)
        sc = encode_cpu(model, proc, syn, dev)
        print(f"fold {f}: train {len(tr)} / test {len(te)} · templates {len(tpls)} · synth {len(syn)} ({time.time()-t0:.0f}s)", flush=True)
        recs = real + syn; cache = CpuCache(rc + sc, dev)
        sidx = list(range(len(real), len(recs)))
        model.load_state_dict(base, strict=False)
        for i in te:
            pm = predict(recs, rc + sc, i); res["vanilla"][i] = T.iou_of(pm, real[i]["gold"])
            if a.save_masks:
                np.save(B / f"synth_prom/pred/{real[i]['stem']}__b{real[i]['bi']}__vanilla.npy", np.packbits(pm))
        for cond, idx, ep in (("real", tr, a.ep_real), ("synth", sidx, a.ep_syn), ("real+synth", tr + sidx, a.ep_syn)):
            model.load_state_dict(base, strict=False)
            T.train(model, recs, cache, idx, Namespace(seed=a.seed, epochs=ep, **HP), dev)
            for i in te:
                pm = predict(recs, rc + sc, i); res[cond][i] = T.iou_of(pm, real[i]["gold"])
                if a.save_masks:
                    np.save(B / f"synth_prom/pred/{real[i]['stem']}__b{real[i]['bi']}__{cond.replace('+', '_')}.npy", np.packbits(pm))
            print(f"   {cond:10s} fold IoU {np.mean([res[cond][i] for i in te]):.3f}", flush=True)
    summ = {}
    for c, d in res.items():
        summ[c] = {"all": float(np.mean(list(d.values())))}
        for l in S.LABS:
            summ[c][l] = float(np.mean([v for i, v in d.items() if real[i]["evaluator"] == l]))
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    json.dump({"summary": summ, "per": {c: {f"{real[i]['stem']}__b{real[i]['bi']}": v for i, v in d.items()} for c, d in res.items()},
               "labeler": {f"{r['stem']}__b{r['bi']}": r["evaluator"] for r in real}, "args": vars(a)},
              open(B / f"synth_prom/exp_seed{a.seed}{a.tag}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
