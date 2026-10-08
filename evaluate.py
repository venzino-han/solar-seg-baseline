#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a split.

    python evaluate.py --ckpt runs/sunspot/unet/best.pt --split test
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from solarseg.data import SolarSegDataset
from solarseg.metrics import evaluate
from solarseg.models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model, name, in_ch = build_model(cfg["arch"])
    model.load_state_dict(ck["model"]); model.to(a.device)
    ds = SolarSegDataset(a.data_root, cfg["cls"], a.split, in_ch, limit=a.limit)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    m = evaluate(model, DataLoader(ds, batch_size=a.bs), a.device, amp, a.thr)
    print(json.dumps(dict(cls=cfg["cls"], model=name, split=a.split, thr=a.thr, **m), indent=1))


if __name__ == "__main__":
    main()
