#!/usr/bin/env python3
"""Train a segmentation baseline for one solar feature.

    python train.py --cls coronal_hole --arch unet
    python train.py --cls sunspot      --arch segformer
    python train.py --cls prominence   --arch deeplabv3 --epochs 40

The test split doubles as the validation set (the best epoch by val IoU is kept).
Outputs -> runs/<cls>/<arch>/
    best.pt           checkpoint of the best epoch (model state + config + val metrics)
    config.json       full run configuration
    history.json      per-epoch loss and validation metrics
    val_final.json    metrics of the reloaded best checkpoint
    samples/*.jpg     input | label | prediction panels from the validation set
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from solarseg.config import ARCH_DEFAULTS, ARCHS, CLASSES
from solarseg.data import SolarSegDataset
from solarseg.losses import seg_loss
from solarseg.metrics import evaluate
from solarseg.models import build_model, count_params
from solarseg.viz import triptych


def save_samples(model, ds, device, out, amp_dtype, n=8):
    out.mkdir(parents=True, exist_ok=True)
    idx = [i for i in range(len(ds)) if ds.msks[i].sum() > 0] or list(range(len(ds)))
    idx = [idx[k] for k in np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
    model.eval()
    for k, i in enumerate(idx):
        x, m = ds[i]
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
            pr = torch.sigmoid(model(x[None].to(device)).float())[0, 0].cpu().numpy() > 0.5
        cv2.imwrite(str(out / f"sample_{k:02d}_{ds.stems[i]}.jpg"),
                    triptych(ds.imgs[i], m[0].numpy() > 0.5, pr), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cls", required=True, choices=list(CLASSES))
    ap.add_argument("--arch", default="unet", choices=ARCHS)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--epochs", type=int, default=None, help="default: per-arch (40)")
    ap.add_argument("--bs", type=int, default=None, help="default: per-arch (16)")
    ap.add_argument("--lr", type=float, default=None, help="default: per-arch (U-Net/SegFormer 1e-3, DeepLabV3 3e-4)")
    ap.add_argument("--pos-weight", type=float, default=None, help="default: per-class")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=None, help="default: runs/<cls>/<arch>")
    ap.add_argument("--require-mask", action="store_true", help="skip frames that have no mask file")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N frames per split (smoke test)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    d = ARCH_DEFAULTS[a.arch]
    epochs, bs, lr = a.epochs or d["epochs"], a.bs or d["bs"], a.lr or d["lr"]
    pos_weight = a.pos_weight if a.pos_weight is not None else CLASSES[a.cls]["pos_weight"]
    if a.arch == "deeplabv3" and bs < 2:
        ap.error("DeepLabV3 needs --bs >= 2 (its ASPP image-pooling branch uses BatchNorm on a 1x1 map)")
    assert torch.cuda.is_available(), "a CUDA GPU is required"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    out = Path(a.run_dir or f"runs/{a.cls}/{a.arch}"); out.mkdir(parents=True, exist_ok=True)

    model, name, in_ch = build_model(a.arch)
    model = model.to(dev)
    t = time.time()
    tr = SolarSegDataset(a.data_root, a.cls, "train", in_ch, train=True, require_mask=a.require_mask, limit=a.limit)
    va = SolarSegDataset(a.data_root, a.cls, "test", in_ch, train=False, require_mask=a.require_mask, limit=a.limit)
    fg = tr.foreground_fraction()
    print(f"[{a.cls}/{a.arch}] {name} {count_params(model)/1e6:.2f}M params | train {len(tr)} / val {len(va)} frames "
          f"| foreground {fg*100:.3f}% | pos_weight {pos_weight:g} | loaded in {time.time()-t:.0f}s", flush=True)
    if tr.missing_masks or va.missing_masks:
        print(f"  note: frames without a mask file (treated as empty): train {tr.missing_masks}, val {va.missing_masks}")

    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=a.workers, pin_memory=True,
                    drop_last=len(tr) > bs, persistent_workers=a.workers > 0)
    vl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=a.workers, pin_memory=True,
                    persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # bf16 keeps fp32-range exponents; with fp16 a large pos_weight (sunspot: 40) can overflow to NaN.
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))
    pw = torch.tensor(pos_weight, device=dev)

    cfg = dict(cls=a.cls, channel=CLASSES[a.cls]["channel"], arch=a.arch, model=name,
               params_M=round(count_params(model) / 1e6, 2), in_channels=in_ch, pretrained=False,
               input="512x512 grayscale, x/255-0.5", train_frames=len(tr), val_frames=len(va),
               foreground_frac=round(fg, 6), epochs=epochs, batch_size=bs, lr=lr, weight_decay=1e-4,
               pos_weight=pos_weight, loss="BCE(pos_weight) + Dice", optimizer="AdamW + CosineAnnealing",
               amp=str(amp).replace("torch.", ""), augmentation="random h/v flip", seed=a.seed,
               data_root=str(a.data_root))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    history, best, t0 = [], {"iou": -1.0}, time.time()
    for ep in range(epochs):
        model.train(); te = time.time(); run = 0.0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp):
                logit = model(x)
            loss = seg_loss(logit.float(), y, pw)                 # loss in fp32
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            run += loss.item()
        sched.step()
        val = evaluate(model, vl, dev, amp)
        row = dict(epoch=ep, loss=run / max(1, len(tl)), lr=sched.get_last_lr()[0], secs=time.time() - te,
                   **{f"val_{k}": v for k, v in val.items()})
        history.append(row); (out / "history.json").write_text(json.dumps(history, indent=1))
        flag = ""
        if val["iou"] > best["iou"]:
            best = dict(epoch=ep, **val)
            torch.save({"model": model.state_dict(), "cfg": cfg, "val": val, "epoch": ep}, out / "best.pt")
            flag = "  *best"
        print(f"[{a.cls}/{a.arch}] ep {ep:02d} loss {row['loss']:.4f} | val IoU {val['iou']:.4f} Dice {val['dice']:.4f} "
              f"P {val['precision']:.3f} R {val['recall']:.3f} | {row['secs']:.0f}s{flag}", flush=True)

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
    final = evaluate(model, vl, dev, amp)
    save_samples(model, va, dev, out / "samples", amp)
    (out / "val_final.json").write_text(json.dumps(dict(best_epoch=best["epoch"], metrics=final,
                                                        train_minutes=round((time.time() - t0) / 60, 1),
                                                        params_M=cfg["params_M"]), indent=1))
    print(f"[{a.cls}/{a.arch}] done: best val IoU {best['iou']:.4f} @ epoch {best['epoch']} "
          f"({(time.time()-t0)/60:.1f} min) -> {out}")


if __name__ == "__main__":
    main()
