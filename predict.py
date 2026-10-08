#!/usr/bin/env python3
"""Run a trained model on images and write binary masks (+ optional overlays).

    python predict.py --ckpt runs/prominence/unet/best.pt --input path/to/frames --out preds/
    python predict.py --ckpt runs/prominence/unet/best.pt --input one_frame.jpg --out preds/ --overlay

Input frames must be 512x512 SDO images of the channel the model was trained on.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from solarseg.config import CLASSES
from solarseg.data import load_image
from solarseg.models import build_model
from solarseg.viz import PRED_COLOR, overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="an image file or a directory of .jpg/.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--overlay", action="store_true", help="also save <stem>_overlay.jpg")
    ap.add_argument("--all-channels", action="store_true",
                    help="do not skip files whose name lacks the model's channel token")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model, _, in_ch = build_model(cfg["arch"])
    model.load_state_dict(ck["model"]); model.to(a.device).eval()
    src = Path(a.input)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    ch = CLASSES[cfg["cls"]]["channel"]
    if not a.all_channels:
        files = [p for p in files if f"_{ch}_" in p.name] or files
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for p in files:
        x, g = load_image(p, in_ch)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            prob = torch.sigmoid(model(x.to(a.device)).float())[0, 0].cpu().numpy()
        m = prob > a.thr
        cv2.imwrite(str(out / f"{p.stem}.png"), (m * 255).astype(np.uint8))
        if a.overlay:
            cv2.imwrite(str(out / f"{p.stem}_overlay.jpg"), overlay(g, m, PRED_COLOR))
    print(f"{cfg['cls']}/{cfg['arch']}: {len(files)} frames -> {out}")


if __name__ == "__main__":
    main()
