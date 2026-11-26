#!/usr/bin/env python3
"""Fine-tune SAM 3 (Tracker) on the 29 human-corrected PROMINENCE labels
collected from labeller_5 (13) and labeller_6 (16). labeller_4's 5 entries are excluded.

Shape of the training, and why
------------------------------
* Same family as the sunspot run: `Sam3TrackerModel` with a physics-placed POINT
  prompt, so the trainable part is `mask_decoder` + `prompt_encoder` (4.23M) and
  the 454M vision encoder stays frozen. 29 samples cannot move an encoder.

* Prompt polarity is inverted from sunspot. In AIA 304 the off-limb prominence is
  BRIGHT against the sky, so the seed is the brightest pixel of the CV reference,
  and the reference itself is restricted to off-limb pixels -- the 304 disk is
  brighter still, and an unrestricted seed lands on the disk.

* The solar body constrains the SEED but is no longer subtracted from the mask.
  Seeding still needs it -- the 304 disk outshines the prominence, so an
  unrestricted "brightest pixel" prompt lands on the chromosphere. Subtracting it
  from the output, however, threw away 12.0% of what the labelers painted: they
  follow a prominence down to its root, and p5 of their label depth sits 7.5 px
  INSIDE the fitted limb. `CFG["prominence"]["clip_off_limb"]` is therefore False
  and the only remaining clip is the VOC box itself.

* Tversky is asymmetric toward false negatives (alpha=0.3, beta=0.7), matching
  the observed edit direction: the labelers ENLARGED 20 of 29 masks (median area
  ratio 1.39, human 962px vs SAM 3 703px) and recorded
  "영역이 너무 적게 잡힘" on 17 of 29.

* The human label is trained on AS PAINTED. Loss is restricted only to the web
  crop the labeler actually saw -- neither the solar body nor the VOC box is
  masked out any more. Both clips together were discarding 17.2% of what the
  labelers drew (12.0% inside the fitted limb, 5.3% past the box), and both are
  now off in `CFG["prominence"]`. The mask stays bounded by the padded crop that
  `paste_back` writes into, which contains 100% of the human paint.

* 29 samples span 22 distinct days -> day-grouped k-fold CV, so near-duplicate
  frames from one day cannot sit on both sides of a split.

Usage:
  .venv_train/bin/python train_sam3_prominence.py --sweep-margin
  .venv_train/bin/python train_sam3_prominence.py --all-folds --epochs 40
  .venv_train/bin/python train_sam3_prominence.py --final --epochs 40 --tag deploy
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

import label_other_classes as L

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "dataset"
FRAME = 512
GOLD = BASE / "gold_prominence"
OUT = GOLD / "runs"
WEB = BASE / "eval_web_prominence"
CFG = L.CFG["prominence"]
UP = CFG["up"]


# ------------------------------------------------------------------ data
def body_mask(bgr, margin):
    """The Sun itself, per the deployed colour-ratio circle fit."""
    _, _, R, rmap = L.solar_region_colour(bgr)
    return rmap <= (R + margin)


def load_records(margin):
    """Gold masks + the exact inference-side crop, prompt, reference and region."""
    seen = {}
    for m in json.loads((WEB / "manifest.json").read_text()):
        seen[(m["stem"], m["bi"])] = m["crop"]

    recs = []
    for r in csv.DictReader((GOLD / "labels.csv").open()):
        stem, bi = r["stem"], int(r["bi"])
        bgr = cv2.imread(str(DATA / r["split"] / f"{stem}.jpg"))
        gold = cv2.imread(str(GOLD / r["mask_path"]), 0)
        if bgr is None or gold is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        box = [int(v) for v in r["box"].split()]
        body = body_mask(bgr, margin)

        up, bq, info = L.crop_ctx(rgb, box, CFG["pad"], UP)
        gup = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY).astype(np.float32)
        body_up = L.crop_ctx(np.repeat(body[:, :, None].astype(np.uint8) * 255, 3, axis=2),
                             box, CFG["pad"], UP)[0][:, :, 0] > 127
        ref, _ = L.cv_reference(gup, bq, CFG["want"], body_up)
        if ref is None:
            print(f"  skip {stem}__b{bi}: CV 시드 없음")
            continue
        ys, xs = np.where(ref)
        j = int(np.argmax(gup[ys, xs]))          # BRIGHTEST -- prominence plasma
        pt = [float(xs[j]), float(ys[j])]

        cx1, cy1, cx2, cy2, cw, ch = info
        sx, sy = UP / cw, UP / ch
        tgt = cv2.resize((gold > 127).astype(np.uint8)[cy1:cy2, cx1:cx2], (UP, UP),
                         interpolation=cv2.INTER_NEAREST_EXACT) > 0

        # What the deployed pipeline can ever output, in up-space. Neither the
        # body nor the box is excluded any more -- the whole crop is reachable,
        # which is exactly what paste_back can write. So the human label is used
        # as painted, with nothing masked away.
        reach = np.ones((UP, UP), bool)

        # region the labeler actually saw, intersected with what is reachable
        reg = np.zeros((UP, UP), bool)
        if (stem, bi) in seen:
            ex1, ey1, ex2, ey2 = seen[(stem, bi)]
            rx1 = int(np.clip((ex1 - cx1) * sx, 0, UP)); ry1 = int(np.clip((ey1 - cy1) * sy, 0, UP))
            rx2 = int(np.clip((ex2 - cx1) * sx, 0, UP)); ry2 = int(np.clip((ey2 - cy1) * sy, 0, UP))
            reg[ry1:ry2, rx1:rx2] = True
        if reg.sum() < 100:
            reg[:] = True
        reg &= reach

        recs.append(dict(stem=stem, bi=bi, box=box, day=stem[:8], split=r["split"],
                         edited=int(r["edited"]), verdict=r["verdict"],
                         evaluator=r["evaluator"], up=up, pt=pt, ref=ref, tgt=tgt,
                         reg=reg, reach=reach, info=info, body=body,
                         gold=gold > 127, sam3_area=int(r["sam3_area_px"] or 0)))
    return recs


def flip_rec(r):
    """Horizontal mirror of one sample (image, prompt, target, region, reference)."""
    return dict(r, up=r["up"][:, ::-1].copy(), pt=[UP - 1 - r["pt"][0], r["pt"][1]],
                ref=r["ref"][:, ::-1].copy(), tgt=r["tgt"][:, ::-1].copy(),
                reg=r["reg"][:, ::-1].copy(), reach=r["reach"][:, ::-1].copy())


@torch.no_grad()
def cache_embeddings(model, proc, recs, device):
    """Encode every sample (and its mirror) once; the encoder never trains."""
    cache, t0 = [], time.time()
    for i, r in enumerate(recs):
        row = []
        for v in (r, flip_rec(r)):
            inp = proc(images=v["up"], input_points=[[[v["pt"]]]], input_labels=[[[1]]],
                       return_tensors="pt").to(device)
            emb = model.get_image_embeddings(inp["pixel_values"])
            row.append(([e.half() for e in emb], inp["input_points"], inp["input_labels"]))
        cache.append(row)
        if (i + 1) % 10 == 0:
            print(f"    encode {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)
    return cache


# ------------------------------------------------------------------ loss
def masked_bce(logits, target, region, pos_weight):
    l = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
        pos_weight=torch.as_tensor(pos_weight, device=logits.device))
    return (l * region).sum() / region.sum().clamp(min=1)


def soft_tversky(logits, target, region, alpha=0.3, beta=0.7, eps=1.0):
    """beta > alpha penalises FALSE NEGATIVES harder -- the labelers enlarged
    20 of 29 masks, so under-segmentation is the failure to fight."""
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


def pick(lg_up, ref_t, reach_t):
    """Same rule as inference: of the 3 multimask candidates, the one whose
    clipped binary mask best matches the CV bright reference."""
    with torch.no_grad():
        b = (lg_up > 0).float() * reach_t
        inter = (b * ref_t).sum((1, 2))
        union = ((b + ref_t) > 0).float().sum((1, 2)).clamp(min=1)
        return int((inter / union).argmax())


# ------------------------------------------------------------------ eval
@torch.no_grad()
def infer_full(model, proc, r, device, use_cache=None):
    """Mirror label_other_classes.segment_box exactly -> 512-frame mask."""
    if use_cache is not None:
        emb, pts, lbs = use_cache
        lg = decode(model, emb, pts, lbs)
    else:
        inp = proc(images=r["up"], input_points=[[[r["pt"]]]], input_labels=[[[1]]],
                   return_tensors="pt").to(device)
        lg = decode(model, model.get_image_embeddings(inp["pixel_values"]),
                    inp["input_points"], inp["input_labels"])

    ref_full = L.paste_back(r["ref"], r["info"])
    cx1, cy1, cx2, cy2 = r["info"][:4]
    denom = (cx2 - cx1) * (cy2 - cy1)      # no box clip -> coverage vs the crop
    best, best_iou = None, -1.0
    for k in range(lg.shape[0]):
        mk = cv2.resize((lg[k] > 0).cpu().numpy().astype(np.uint8), (UP, UP),
                        interpolation=cv2.INTER_NEAREST_EXACT) > 0
        full = L.paste_back(mk, r["info"])   # crop bound is the only remaining limit
        cov = full.sum() / max(1, denom)
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


def scored(model, proc, recs, device, cache=None):
    """IoU of the shipped pipeline against gold, per record."""
    return [iou_of(infer_full(model, proc, r, device,
                              use_cache=cache[i][0] if cache else None), r["gold"])
            for i, r in enumerate(recs)]


# ------------------------------------------------------------------ train
def train(model, recs, cache, idx, args, device):
    for p in model.vision_encoder.parameters():
        p.requires_grad_(False)
    params = [p for m in (model.mask_decoder, model.prompt_encoder) for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.RandomState(args.seed)

    dev = {}
    for i in idx:
        for f in (0, 1):
            r = recs[i] if f == 0 else flip_rec(recs[i])
            dev[(i, f)] = tuple(torch.from_numpy(r[k]).float().to(device)
                                for k in ("ref", "tgt", "reg", "reach"))

    model.train()
    model.vision_encoder.eval()
    hist = []
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for oi in rng.permutation(len(idx)):
            i = idx[oi]
            f = int(rng.rand() < 0.5)
            emb, pts, lbs = cache[i][f]
            ref_t, tgt_t, reg_t, reach_t = dev[(i, f)]
            lg = decode(model, emb, pts, lbs)
            lg_up = F.interpolate(lg[None], size=(UP, UP), mode="bilinear",
                                  align_corners=False)[0]
            sel = lg_up[pick(lg_up, ref_t, reach_t)]
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
    ap.add_argument("--sweep-margin", action="store_true",
                    help="vanilla SAM3 를 여러 LIMB_MARGIN 에서 채점하고 종료")
    ap.add_argument("--margin", type=int, default=None,
                    help="시드 배치용 LIMB_MARGIN (기본: label_other_classes.LIMB_MARGIN)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--tversky", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="cv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    margin = L.LIMB_MARGIN if args.margin is None else args.margin

    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained(str(BASE / "sam3_model"))
    model = Sam3TrackerModel.from_pretrained(str(BASE / "sam3_model")).to(args.device).eval()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.sweep_margin:
        print("=== LIMB_MARGIN 스윕 (바닐라 SAM3, gold 29건) ===", flush=True)
        res = {}
        for mg in (-3, -5, -7, -9, -11):
            recs = load_records(mg)
            s = scored(model, proc, recs, args.device)
            res[mg] = {"mean": float(np.mean(s)), "median": float(np.median(s)), "n": len(s)}
            print(f"  margin {mg:+3d}  n={len(s):2d}  IoU 평균 {np.mean(s):.4f}  "
                  f"중앙값 {np.median(s):.4f}", flush=True)
        best = max(res, key=lambda m: res[m]["mean"])
        print(f"\n최적 margin = {best:+d}")
        (OUT / "margin_sweep.json").write_text(json.dumps(
            {"result": res, "best": best}, indent=1))
        return

    print(f"기록 로딩 (LIMB_MARGIN {margin:+d})...", flush=True)
    recs = load_records(margin)
    print(f"  gold {len(recs)}건 · 날짜 {len({r['day'] for r in recs})}개 · "
          f"편집 {sum(r['edited'] for r in recs)}건")

    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    print("인코더 임베딩 캐시 (동결)...", flush=True)
    cache = cache_embeddings(model, proc, recs, args.device)

    print("바닐라 SAM3 기준선...", flush=True)
    base_iou = scored(model, proc, recs, args.device, cache)
    print(f"  vanilla IoU 평균 {np.mean(base_iou):.4f} 중앙값 {np.median(base_iou):.4f}")

    days = sorted({r["day"] for r in recs})
    dfold = {d: k % args.folds for k, d in enumerate(days)}

    if args.final:
        print(f"\n=== FINAL: 전체 {len(recs)}건 학습 ===", flush=True)
        model.load_state_dict(base_sd)
        hist = train(model, recs, cache, list(range(len(recs))), args, args.device)
        sd = {k: v.cpu() for k, v in model.state_dict().items()
              if k.startswith(("mask_decoder", "prompt_encoder"))}
        p = OUT / f"sam3_prominence_{args.tag}.pt"
        torch.save({"state_dict": sd, "cfg": vars(args), "n": len(recs),
                    "limb_margin": margin,
                    "trained": ["mask_decoder", "prompt_encoder"]}, p)
        ft = scored(model, proc, recs, args.device, cache)
        print(f"  (in-sample) vanilla {np.mean(base_iou):.4f} → finetuned {np.mean(ft):.4f}")
        print(f"  저장 -> {p}")
        (OUT / f"final_{args.tag}.json").write_text(json.dumps(
            {"loss": hist, "in_sample_iou": float(np.mean(ft)),
             "vanilla_iou": float(np.mean(base_iou)), "limb_margin": margin}, indent=1))
        return

    oof = {}
    for f in (range(args.folds) if args.all_folds else [args.fold]):
        te = [i for i, r in enumerate(recs) if dfold[r["day"]] == f]
        tr = [i for i, r in enumerate(recs) if dfold[r["day"]] != f]
        print(f"\n=== fold {f}: train {len(tr)} / test {len(te)} ===", flush=True)
        model.load_state_dict(base_sd)
        train(model, recs, cache, tr, args, args.device)
        for i in te:
            oof[i] = iou_of(infer_full(model, proc, recs[i], args.device,
                                       use_cache=cache[i][0]), recs[i]["gold"])
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
        "limb_margin": margin, "cfg": vars(args)}, indent=1))


if __name__ == "__main__":
    main()
