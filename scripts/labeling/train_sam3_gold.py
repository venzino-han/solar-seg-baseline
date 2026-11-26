#!/usr/bin/env python3
"""Fine-tune SAM 3 on the 72 human-labeled coronal-hole boxes.

Why this shape of training
--------------------------
72 labeled boxes is far too few for a held-out split that means anything, and
too few to touch the 454M vision encoder. So:

  * the vision and text encoders stay frozen and run under no_grad;
    only the prompt-conditioned heads are trained (2.3M-32.6M params),
  * every sample is used, via group-k-fold CV grouped by observation DAY --
    each box is held out exactly once, giving out-of-fold predictions for all
    72, and is trained on in the other k-1 folds,
  * the loss and every metric are restricted to the crop region the labeler
    actually saw (box + 36px pad); pixels she never looked at are ignored
    rather than counted as background,
  * `approved` samples (she accepted the pipeline mask without editing) get
    half weight -- accepting is weaker evidence than drawing.

After CV, `--final` retrains on all 72 and saves deployable head weights.

Usage:
  .venv_train/bin/python train_sam3_gold.py --fold 0 --epochs 30
  .venv_train/bin/python train_sam3_gold.py --all-folds --tag mgeom
  .venv_train/bin/python train_sam3_gold.py --final --tag deploy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sam3_common as C

BASE = Path(__file__).resolve().parent
TRAIN_SETS = {
    "mask": ["mask_decoder"],
    "mask+geom": ["mask_decoder", "geometry_encoder"],
    "all": ["mask_decoder", "geometry_encoder", "detr_decoder", "detr_encoder",
            "dot_product_scoring"],
}


# ------------------------------------------------------------------ losses
def masked_bce(logits, target, region, pos_weight):
    l = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
        pos_weight=torch.as_tensor(pos_weight, device=logits.device))
    return (l * region).sum() / region.sum().clamp(min=1)


def soft_tversky(logits, target, region, alpha=0.7, beta=0.3, eps=1.0):
    """alpha > beta penalises false positives harder -- the labeler shrank the
    pipeline mask in 35 of 45 edits, so over-prediction is the failure to fight."""
    p = torch.sigmoid(logits) * region
    g = target * region
    tp = (p * g).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()
    return 1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


# ------------------------------------------------------------------ data
class Samples:
    """Loads and caches the CPU-side arrays once; augmentation is applied per step."""

    def __init__(self, recs):
        self.recs = recs
        self.cache = {}

    def get(self, i, flip=0, gamma=1.0):
        r = self.recs[i]
        if i not in self.cache:
            self.cache[i] = C.load_sample(r)
        rgb, gt, disk = self.cache[i]
        rgb, gt, disk, box = C.flip_variant(rgb, gt, disk, r["box"], flip)
        if gamma != 1.0:
            lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
            rgb = lut[rgb]
        cx1, cy1, cx2, cy2 = r["crop"]
        region = np.zeros((C.FRAME, C.FRAME), bool)
        if flip in (1, 3):
            cx1, cx2 = C.FRAME - cx2, C.FRAME - cx1
        if flip in (2, 3):
            cy1, cy2 = C.FRAME - cy2, C.FRAME - cy1
        region[cy1:cy2, cx1:cx2] = True
        return rgb, gt, disk, box, region


# ------------------------------------------------------------------ step
def forward_losses(model, proc, s, dev, cfg, train=True):
    rgb, gt, disk, box, region = s
    ve, te, inp = C.encode(model, proc, rgb, box, device=dev)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        out = C.heads(model, ve, te, inp)
        with torch.no_grad():
            qi, qscore, qiou = C.select_query(out, box)
        inst = C.mask_logits_512(out.pred_masks[0, qi])
        sem = C.mask_logits_512(out.semantic_seg[0, 0])

        t = torch.from_numpy(gt.astype(np.float32)).to(dev)
        rg = torch.from_numpy(region.astype(np.float32)).to(dev)
        pw = cfg["pos_weight"]
        li = masked_bce(inst, t, rg, pw) + cfg["w_tversky"] * soft_tversky(
            inst, t, rg, cfg["alpha"], cfg["beta"])
        ls = masked_bce(sem, t, rg, pw) + cfg["w_tversky"] * soft_tversky(
            sem, t, rg, cfg["alpha"], cfg["beta"])
        loss = li + cfg["w_sem"] * ls
    return loss, inst.detach(), sem.detach(), qi, float(qscore[qi]), float(qiou[qi])


@torch.no_grad()
def evaluate(model, proc, data, idxs, dev, cfg, thresholds=(0.0,)):
    """Region-restricted metrics on `idxs`, for each logit threshold."""
    per_thr = {t: [] for t in thresholds}
    sem_thr = {t: [] for t in thresholds}
    logits_store = []
    for i in idxs:
        s = data.get(i, flip=0)
        rgb, gt, disk, box, region = s
        _, inst, sem, qi, qs, qb = forward_losses(model, proc, s, dev, cfg, train=False)
        inst_np, sem_np = inst.cpu().numpy(), sem.cpu().numpy()
        x1, y1, x2, y2 = box
        boxm = np.zeros((C.FRAME, C.FRAME), bool); boxm[y1:y2, x1:x2] = True
        clip = disk & boxm
        for t in thresholds:
            per_thr[t].append(C.metrics((inst_np > t) & clip, gt, region, box))
            sem_thr[t].append(C.metrics((sem_np > t) & clip, gt, region, box))
        logits_store.append({"sid": data.recs[i]["sid"], "query": qi,
                             "q_score": qs, "q_boxiou": qb})
    return per_thr, sem_thr, logits_store


def run_fold(model, proc, data, tr_idx, va_idx, cfg, dev, tag, log):
    for n, p in model.named_parameters():
        p.requires_grad_(any(n.startswith(m) for m in TRAIN_SETS[cfg["train_set"]]))
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    init_state = {n: p.detach().clone() for n, p in model.named_parameters()
                  if p.requires_grad}

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    steps_per_epoch = max(1, len(tr_idx) // cfg["accum"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch,
        pct_start=0.2)
    rng = np.random.RandomState(cfg["seed"])

    w = {i: data.recs[i]["weight"] for i in tr_idx}
    best = {"iou": -1.0, "state": {}}
    hist = []

    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(tr_idx)
        tot, nsel = 0.0, 0
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        for k, i in enumerate(order):
            flip = int(rng.randint(4)) if cfg["aug_flip"] else 0
            gamma = float(rng.uniform(*cfg["gamma_range"])) if cfg["aug_gamma"] else 1.0
            s = data.get(int(i), flip=flip, gamma=gamma)
            loss, _, _, qi, _, _ = forward_losses(model, proc, s, dev, cfg, train=True)
            (loss * w[int(i)] / cfg["accum"]).backward()
            tot += float(loss); nsel += (qi == 0)
            if (k + 1) % cfg["accum"] == 0 or k == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
                opt.step(); opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()

        model.eval()
        thr = tuple(cfg["thresholds"])
        va_inst, va_sem, _ = evaluate(model, proc, data, va_idx, dev, cfg, thr)
        tr_iou = None
        if cfg["train_eval_every"] and (ep + 1) % cfg["train_eval_every"] == 0:
            tr_inst, _, _ = evaluate(model, proc, data, tr_idx, dev, cfg, thr)
            tr_iou = float(np.mean([m["iou"] for m in tr_inst[0.0]]))
        va_iou = float(np.mean([m["iou"] for m in va_inst[0.0]]))
        row = {"epoch": ep, "loss": tot / len(order), "lr": sched.get_last_lr()[0],
               "train_iou@0": tr_iou, "val_iou@0": va_iou,
               "val_sem_iou@0": float(np.mean([m["iou"] for m in va_sem[0.0]])),
               "val_prec": float(np.mean([m["precision"] for m in va_inst[0.0]])),
               "val_rec": float(np.mean([m["recall"] for m in va_inst[0.0]])),
               "val_dmed": float(np.median([m["delta_area"] for m in va_inst[0.0]])),
               "val_best_thr_iou": max(float(np.mean([m["iou"] for m in va_inst[t]])) for t in thr),
               "q0_frac": nsel / len(order), "secs": time.time() - t0}
        hist.append(row)
        log(f"    ep{ep:02d} loss {row['loss']:.4f} val@0 {va_iou:.4f} "
            f"(P {row['val_prec']:.3f} R {row['val_rec']:.3f} Δ {row['val_dmed']:+.3f})"
            + (f" train@0 {tr_iou:.4f}" if tr_iou is not None else "")
            + f" {row['secs']:.0f}s")
        # `select=val` peeks at the held-out fold -- only for pilot runs.
        if cfg["select"] == "val" and va_iou > best["iou"]:
            best = {"iou": va_iou, "epoch": ep,
                    "state": {n: p.detach().clone() for n, p in model.named_parameters()
                              if p.requires_grad}, "row": row}

    if cfg["select"] == "last" or best["iou"] < 0:
        best = {"iou": hist[-1]["val_iou@0"], "epoch": cfg["epochs"] - 1,
                "state": {n: p.detach().clone() for n, p in model.named_parameters()
                          if p.requires_grad}, "row": hist[-1]}

    for n, p in model.named_parameters():
        if n in best["state"]:
            p.data.copy_(best["state"][n])
    model.eval()

    # threshold tuned on TRAIN only, then applied to the held-out fold
    tr_inst, _, _ = evaluate(model, proc, data, tr_idx, dev, cfg, tuple(cfg["thresholds"]))
    best["thr"] = max(cfg["thresholds"],
                      key=lambda t: np.mean([m["iou"] for m in tr_inst[t]]))
    best["train_iou_at_thr"] = float(np.mean([m["iou"] for m in tr_inst[best["thr"]]]))
    va_inst, va_sem, info = evaluate(model, proc, data, va_idx, dev, cfg,
                                     tuple(cfg["thresholds"]))
    oof = []
    for j, i in enumerate(va_idx):
        r = data.recs[i]
        oof.append({"sid": r["sid"], "day": r["day"], "source": r["source"],
                    "iou": va_inst[best["thr"]][j]["iou"],
                    "dice": va_inst[best["thr"]][j]["dice"],
                    "precision": va_inst[best["thr"]][j]["precision"],
                    "recall": va_inst[best["thr"]][j]["recall"],
                    "delta_area": va_inst[best["thr"]][j]["delta_area"],
                    "sem_iou": va_sem[best["thr"]][j]["iou"],
                    "query": info[j]["query"], "thr": best["thr"]})

    state = best.pop("state")
    # restore the pretrained weights so the next fold starts clean
    for n, p in model.named_parameters():
        if n in init_state:
            p.data.copy_(init_state[n])
    return {"n_trainable": n_tr, "best": best, "history": hist, "oof": oof}, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--train-set", default="mask+geom", choices=list(TRAIN_SETS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=None, help="run a single fold")
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="train on all 72 and save deployable weights")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--w-tversky", type=float, default=1.0)
    ap.add_argument("--w-sem", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--no-aug-flip", action="store_true")
    ap.add_argument("--no-aug-gamma", action="store_true")
    ap.add_argument("--select", default="last", choices=["last", "val"],
                    help="'last' = fixed epoch budget (honest held-out number); "
                         "'val' = early-stop on the held-out fold (pilot runs only)")
    ap.add_argument("--train-eval-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="gold/runs")
    args = ap.parse_args()

    cfg = {"train_set": args.train_set, "epochs": args.epochs, "lr": args.lr,
           "wd": args.wd, "accum": args.accum, "clip": args.clip,
           "pos_weight": args.pos_weight, "w_tversky": args.w_tversky,
           "w_sem": args.w_sem, "alpha": args.alpha, "beta": args.beta,
           "aug_flip": not args.no_aug_flip, "aug_gamma": not args.no_aug_gamma,
           "gamma_range": (0.8, 1.25), "seed": args.seed,
           "select": args.select, "train_eval_every": args.train_eval_every,
           "thresholds": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]}

    outdir = BASE / args.out / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    logf = (outdir / "train.log").open("a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"\n=== {args.tag} | {json.dumps(cfg)} ===")
    recs = C.load_gold()
    data = Samples(recs)
    model, proc = C.load_model(device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)

    folds = C.day_folds(recs, k=args.folds, seed=args.seed)
    log(f"{len(recs)} samples, {len({r['day'] for r in recs})} days, "
        f"fold sizes {[len(f) for f in folds]}")

    results = {"cfg": cfg, "folds": {}}

    if args.final:
        all_idx = list(range(len(recs)))
        res, state = run_fold(model, proc, data, all_idx, all_idx, cfg,
                              args.device, args.tag, log)
        torch.save({"cfg": cfg, "train_set": args.train_set, "state_dict": state},
                   outdir / "sam3_coronal_hole_heads.pt")
        results["final"] = res
        log(f"  final: train-fit IoU {res['best']['iou']:.4f} "
            f"thr {res['best']['thr']:+.1f} -> {outdir/'sam3_coronal_hole_heads.pt'}")
    else:
        todo = range(args.folds) if (args.all_folds or args.fold is None) else [args.fold]
        for f in todo:
            va = folds[f]
            tr = [i for j, g in enumerate(folds) if j != f for i in g]
            log(f"  fold {f}: train {len(tr)} / val {len(va)}")
            res, state = run_fold(model, proc, data, tr, va, cfg, args.device,
                                  f"{args.tag}_f{f}", log)
            torch.save({"cfg": cfg, "fold": f, "state_dict": state},
                       outdir / f"fold{f}.pt")
            results["folds"][str(f)] = res
            log(f"  fold {f} best: ep{res['best']['epoch']} thr {res['best']['thr']:+.1f} "
                f"val IoU {res['best']['iou']:.4f}")

        oof = [o for f in results["folds"].values() for o in f["oof"]]
        if oof:
            results["oof"] = oof
            ed = [o for o in oof if o["source"] == "edited"]
            log(f"\n  OOF n={len(oof)}  IoU mean {np.mean([o['iou'] for o in oof]):.4f} "
                f"median {np.median([o['iou'] for o in oof]):.4f}")
            log(f"  OOF edited-only n={len(ed)} IoU {np.mean([o['iou'] for o in ed]):.4f}")
            log(f"  OOF precision {np.mean([o['precision'] for o in oof]):.4f} "
                f"recall {np.mean([o['recall'] for o in oof]):.4f} "
                f"medΔarea {np.median([o['delta_area'] for o in oof]):+.4f}")

    (outdir / "results.json").write_text(json.dumps(results, indent=1))
    log(f"-> {outdir/'results.json'}")


if __name__ == "__main__":
    main()
