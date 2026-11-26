#!/usr/bin/env python3
"""Fine-tune SAM 3 (Tracker) on labeller_8's 45 human-corrected sunspot labels.

Shape of the training, and why
------------------------------
* The sunspot pipeline uses `Sam3TrackerModel` with a physics-placed POINT
  prompt, not the box-prompted PCS model the coronal-hole run fine-tuned. So
  this trains the tracker's `mask_decoder` + `prompt_encoder` (4.23M) and keeps
  the 454M vision encoder frozen -- 45 samples cannot move an encoder.

* The frozen encoder's output is cached (`get_image_embeddings`), which was
  verified to reproduce the full forward exactly. One encoder pass is 408 ms,
  so caching turns a multi-hour CV sweep into minutes.

* Tversky is inverted relative to the coronal-hole run. There the labeler
  SHRANK 35 of 45 masks, so false positives were penalised (alpha=0.7). Here she
  ENLARGED 35 of 41 (median area +23%, human 95px vs SAM 3 68px), so the failure
  to fight is under-segmentation: alpha=0.3, beta=0.7.

* Loss and metrics are restricted to the crop the labeler actually saw in the
  web tool. Pixels outside it were never inspected and are not evidence.

* Candidate selection matches inference exactly (best IoU against the CV dark
  reference among the 3 multimask outputs), so training optimises the decision
  path that actually ships.

* 45 samples span 39 distinct days -> 5-fold CV grouped by DAY, so near-duplicate
  frames from one day cannot sit on both sides of a split.

Usage:
  .venv_train/bin/python train_sam3_sunspot.py --all-folds --epochs 40
  .venv_train/bin/python train_sam3_sunspot.py --final --epochs 40 --tag deploy
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import segmenters as S
import label_other_classes as L

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "dataset"
FRAME = 512
GOLD = BASE / "gold_sunspot"
OUT = BASE / "gold_sunspot" / "runs"
CFG = L.CFG["sunspot"]


# ------------------------------------------------------------------ data
def load_records():
    """Gold boxes + the exact inference-side crop, prompt and CV reference."""
    seen = {}
    man = json.loads((BASE / "eval_web_sunspot" / "manifest.json").read_text())
    for m in man:
        seen[(m["stem"], m["bi"])] = m["crop"]

    recs = []
    for r in csv.DictReader((GOLD / "labels.csv").open()):
        stem, bi = r["stem"], int(r["bi"])
        bgr = cv2.imread(str(DATA / r["split"] / f"{stem}.jpg"))
        gold = cv2.imread(str(GOLD / r["mask_path"]), 0)
        if bgr is None or gold is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        disk = S.solar_disk_mask(gray)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        box = [int(v) for v in r["box"].split()]

        up, bq, info = L.crop_ctx(rgb, box, CFG["pad"], CFG["up"])
        gup = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY).astype(np.float32)
        disk_up = L.crop_ctx(np.repeat(disk[:, :, None].astype(np.uint8) * 255, 3, axis=2),
                             box, CFG["pad"], CFG["up"])[0][:, :, 0] > 127
        ref, _ = L.cv_reference(gup, bq, CFG["want"], disk_up)
        if ref is None:
            continue
        ys, xs = np.where(ref)
        j = int(np.argmin(gup[ys, xs]))
        pt = [float(xs[j]), float(ys[j])]

        cx1, cy1, cx2, cy2, cw, ch = info
        sx, sy = CFG["up"] / cw, CFG["up"] / ch
        tgt = cv2.resize((gold > 127).astype(np.uint8)[cy1:cy2, cx1:cx2],
                         (CFG["up"], CFG["up"]), interpolation=cv2.INTER_NEAREST_EXACT) > 0

        # region the labeler actually saw, in up-space
        reg = np.zeros((CFG["up"], CFG["up"]), bool)
        if (stem, bi) in seen:
            ex1, ey1, ex2, ey2 = seen[(stem, bi)]
            rx1 = int(np.clip((ex1 - cx1) * sx, 0, CFG["up"]))
            ry1 = int(np.clip((ey1 - cy1) * sy, 0, CFG["up"]))
            rx2 = int(np.clip((ex2 - cx1) * sx, 0, CFG["up"]))
            ry2 = int(np.clip((ey2 - cy1) * sy, 0, CFG["up"]))
            reg[ry1:ry2, rx1:rx2] = True
        if reg.sum() < 100:
            reg[:] = True

        recs.append(dict(stem=stem, bi=bi, box=box, day=stem[:8], split=r["split"],
                         edited=int(r["edited"]), up=up, pt=pt, ref=ref, tgt=tgt,
                         reg=reg, info=info, disk=disk, gold=gold > 127,
                         gray=gray))
    return recs


def flip_rec(r):
    """Horizontal mirror of one sample (image, prompt, target, region, reference)."""
    W = CFG["up"]
    return dict(r, up=r["up"][:, ::-1].copy(), pt=[W - 1 - r["pt"][0], r["pt"][1]],
                ref=r["ref"][:, ::-1].copy(), tgt=r["tgt"][:, ::-1].copy(),
                reg=r["reg"][:, ::-1].copy())


@torch.no_grad()
def cache_embeddings(model, proc, recs, device):
    """Encode every sample (and its mirror) once; the encoder never trains."""
    cache = []
    t0 = time.time()
    for i, r in enumerate(recs):
        row = []
        for v in (r, flip_rec(r)):
            inp = proc(images=v["up"], input_points=[[[v["pt"]]]], input_labels=[[[1]]],
                       return_tensors="pt").to(device)
            emb = model.get_image_embeddings(inp["pixel_values"])
            row.append(([e.half() for e in emb], inp["input_points"], inp["input_labels"]))
        cache.append(row)
        if (i + 1) % 15 == 0:
            print(f"    encode {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)
    return cache


# ------------------------------------------------------------------ loss
def masked_bce(logits, target, region, pos_weight):
    l = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
        pos_weight=torch.as_tensor(pos_weight, device=logits.device))
    return (l * region).sum() / region.sum().clamp(min=1)


def soft_tversky(logits, target, region, alpha=0.3, beta=0.7, eps=1.0):
    """beta > alpha penalises FALSE NEGATIVES harder -- the inverse of the
    coronal-hole run, because here the labeler enlarged 35 of 41 masks."""
    p = torch.sigmoid(logits) * region
    g = target * region
    tp = (p * g).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()
    return 1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def decode(model, emb, points, labels):
    o = model(input_points=points, input_labels=labels,
              image_embeddings=[e.float() for e in emb], multimask_output=True)
    lg = o.pred_masks[0].float()
    return lg.flatten(0, 1) if lg.dim() > 3 else lg          # (3, 288, 288)


def pick(lg512, ref_t):
    """Same rule as inference: the candidate whose binary mask best matches the
    CV dark reference. Selection is non-differentiable; the loss then flows
    through the selected candidate's logits."""
    with torch.no_grad():
        b = (lg512 > 0).float()
        inter = (b * ref_t).sum((1, 2))
        union = ((b + ref_t) > 0).float().sum((1, 2)).clamp(min=1)
        return int((inter / union).argmax())


# ------------------------------------------------------------------ eval
@torch.no_grad()
def infer_full(model, proc, r, device, use_cache=None):
    """Run the deployed pipeline for one record -> 512-frame mask (as shipped)."""
    if use_cache is not None:
        emb, pts, lbs = use_cache
        lg = decode(model, emb, pts, lbs)
    else:
        inp = proc(images=r["up"], input_points=[[[r["pt"]]]], input_labels=[[[1]]],
                   return_tensors="pt").to(device)
        lg = decode(model, model.get_image_embeddings(inp["pixel_values"]),
                    inp["input_points"], inp["input_labels"])
    up512 = F.interpolate(lg[None], size=(FRAME, FRAME), mode="bilinear",
                          align_corners=False)[0]
    ref_full = L.paste_back(r["ref"], r["info"])
    x1, y1, x2, y2 = r["box"]
    bm = np.zeros((FRAME, FRAME), bool); bm[y1:y2, x1:x2] = True
    best, best_iou = None, -1.0
    for k in range(up512.shape[0]):
        full = L.paste_back((up512[k] > 0).cpu().numpy(), r["info"])
        full &= r["disk"] & bm
        cov = full.sum() / max(1, (x2 - x1) * (y2 - y1))
        if not (CFG["cov_lo"] <= cov <= L.COV_HI):
            continue
        u = (full | ref_full).sum()
        iou = (full & ref_full).sum() / u if u else 0.0
        if iou > best_iou:
            best, best_iou = full, iou
    return best if best is not None else np.zeros((FRAME, FRAME), bool)


def iou_of(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else 1.0


# ------------------------------------------------------------------ train
def train(model, recs, cache, idx, args, device):
    for p in model.vision_encoder.parameters():
        p.requires_grad_(False)
    train_mods = [model.mask_decoder, model.prompt_encoder]
    params = [p for m in train_mods for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.RandomState(args.seed)

    dev_ref = {}
    for i in idx:
        for f in (0, 1):
            r = recs[i] if f == 0 else flip_rec(recs[i])
            dev_ref[(i, f)] = (
                torch.from_numpy(r["ref"]).float().to(device),
                torch.from_numpy(r["tgt"]).float().to(device),
                torch.from_numpy(r["reg"]).float().to(device))

    model.train()
    model.vision_encoder.eval()
    hist = []
    for ep in range(args.epochs):
        order = rng.permutation(len(idx))
        tot, n = 0.0, 0
        for oi in order:
            i = idx[oi]
            f = int(rng.rand() < 0.5)
            emb, pts, lbs = cache[i][f]
            ref_t, tgt_t, reg_t = dev_ref[(i, f)]
            lg = decode(model, emb, pts, lbs)
            lg512 = F.interpolate(lg[None], size=(CFG["up"], CFG["up"]),
                                  mode="bilinear", align_corners=False)[0]
            k = pick(lg512, ref_t)
            sel = lg512[k]
            w = 1.0 if recs[i]["edited"] else 0.5      # accepting is weaker evidence
            loss = w * (masked_bce(sel, tgt_t, reg_t, args.pos_weight)
                        + args.tversky * soft_tversky(sel, tgt_t, reg_t,
                                                      args.alpha, 1 - args.alpha))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += float(loss); n += 1
        sched.step()
        hist.append(tot / max(1, n))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    ep {ep+1:3d}/{args.epochs}  loss {hist[-1]:.4f}", flush=True)
    model.eval()
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--tversky", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="cv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    print("기록 로딩...", flush=True)
    recs = load_records()
    print(f"  gold {len(recs)}건 · 날짜 {len({r['day'] for r in recs})}개 · "
          f"편집 {sum(r['edited'] for r in recs)}건")

    proc = Sam3TrackerProcessor.from_pretrained(str(BASE / "sam3_model"))
    model = Sam3TrackerModel.from_pretrained(str(BASE / "sam3_model")).to(args.device).eval()
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    print("인코더 임베딩 캐시 (동결)...", flush=True)
    cache = cache_embeddings(model, proc, recs, args.device)

    # vanilla baseline on every gold box
    print("바닐라 SAM3 기준선...", flush=True)
    base_iou = []
    for i, r in enumerate(recs):
        m = infer_full(model, proc, r, args.device, use_cache=cache[i][0])
        base_iou.append(iou_of(m, r["gold"]))
    print(f"  vanilla IoU 평균 {np.mean(base_iou):.4f} 중앙값 {np.median(base_iou):.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    days = sorted({r["day"] for r in recs})
    dfold = {d: k % args.folds for k, d in enumerate(days)}

    if args.final:
        print(f"\n=== FINAL: 전체 {len(recs)}건 학습 ===", flush=True)
        model.load_state_dict(base_sd)
        hist = train(model, recs, cache, list(range(len(recs))), args, args.device)
        sd = {k: v.cpu() for k, v in model.state_dict().items()
              if k.startswith(("mask_decoder", "prompt_encoder"))}
        p = OUT / f"sam3_sunspot_{args.tag}.pt"
        torch.save({"state_dict": sd, "cfg": vars(args), "n": len(recs),
                    "trained": ["mask_decoder", "prompt_encoder"]}, p)
        ft = [iou_of(infer_full(model, proc, r, args.device, use_cache=cache[i][0]), r["gold"])
              for i, r in enumerate(recs)]
        print(f"  (in-sample) vanilla {np.mean(base_iou):.4f} → finetuned {np.mean(ft):.4f}")
        print(f"  저장 -> {p}")
        (OUT / f"final_{args.tag}.json").write_text(json.dumps(
            {"loss": hist, "in_sample_iou": float(np.mean(ft)),
             "vanilla_iou": float(np.mean(base_iou))}, indent=1))
        return

    folds = range(args.folds) if args.all_folds else [args.fold]
    oof = {}
    for f in folds:
        te = [i for i, r in enumerate(recs) if dfold[r["day"]] == f]
        tr = [i for i, r in enumerate(recs) if dfold[r["day"]] != f]
        print(f"\n=== fold {f}: train {len(tr)} / test {len(te)} ===", flush=True)
        model.load_state_dict(base_sd)
        train(model, recs, cache, tr, args, args.device)
        for i in te:
            m = infer_full(model, proc, recs[i], args.device, use_cache=cache[i][0])
            oof[i] = iou_of(m, recs[i]["gold"])
        print(f"  fold IoU {np.mean([oof[i] for i in te]):.4f} "
              f"(vanilla {np.mean([base_iou[i] for i in te]):.4f})")

    ks = sorted(oof)
    ft = np.array([oof[i] for i in ks]); bs = np.array([base_iou[i] for i in ks])
    print(f"\n=== 교차검증 결과 (n={len(ks)}) ===")
    print(f"  vanilla    IoU {bs.mean():.4f}  중앙값 {np.median(bs):.4f}")
    print(f"  fine-tuned IoU {ft.mean():.4f}  중앙값 {np.median(ft):.4f}")
    print(f"  개선 {ft.mean()-bs.mean():+.4f} · 향상 {(ft>bs).sum()}/{len(ks)}건")
    (OUT / f"cv_{args.tag}.json").write_text(json.dumps({
        "vanilla": {"mean": float(bs.mean()), "median": float(np.median(bs)),
                    "per": [float(x) for x in bs]},
        "finetuned": {"mean": float(ft.mean()), "median": float(np.median(ft)),
                      "per": [float(x) for x in ft]},
        "stems": [f"{recs[i]['stem']}__b{recs[i]['bi']}" for i in ks],
        "cfg": vars(args)}, indent=1))


if __name__ == "__main__":
    main()
