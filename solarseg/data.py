"""Dataset: SDO 512x512 full-disk frames + per-image binary masks.

Expected layout (see README):

    <data_root>/images/<split>/<stem>.jpg      input frame (3-channel JPEG, used as grayscale)
    <data_root>/images/<split>/<stem>.xml      Pascal-VOC bounding boxes (optional, not used for training)
    <data_root>/masks/<class>/<split>/<stem>.png   binary mask, 255 = feature, 0 = background

Only frames of the class's SDO channel are used (the channel token is part of <stem>).
A frame without a mask file is treated as containing no feature (all-zero mask) unless
--require-mask is given.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CLASSES


def list_frames(data_root, cls, split):
    ch = CLASSES[cls]["channel"]
    d = Path(data_root) / "images" / split
    return sorted(p for p in d.glob("*.jpg") if f"_{ch}_" in p.name)


def mask_path(data_root, cls, split, stem):
    return Path(data_root) / "masks" / cls / split / f"{stem}.png"


def read_voc_boxes(xml_path, cls=None):
    """-> list of [xmin, ymin, xmax, ymax] for objects of `cls` (all objects if None)."""
    if not Path(xml_path).exists():
        return []
    root = ET.parse(xml_path).getroot()
    out = []
    for o in root.findall("object"):
        if cls is None or o.findtext("name") == cls:
            b = o.find("bndbox")
            out.append([int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")])
    return out


class SolarSegDataset(Dataset):
    """Frames are preloaded into RAM as uint8 (a few hundred MB at most)."""

    def __init__(self, data_root, cls, split, in_ch=1, train=False, require_mask=False, limit=None):
        self.cls, self.split, self.in_ch, self.train = cls, split, in_ch, train
        frames = list_frames(data_root, cls, split)
        self.stems, self.imgs, self.msks, missing = [], [], [], 0
        for p in frames:
            mp = mask_path(data_root, cls, split, p.stem)
            if not mp.exists():
                missing += 1
                if require_mask:
                    continue
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            m = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) if mp.exists() \
                else np.zeros_like(g, np.uint8)
            self.stems.append(p.stem); self.imgs.append(g); self.msks.append(m)
            if limit and len(self.stems) >= limit:
                break
        self.missing_masks = missing
        if not self.stems:
            raise RuntimeError(f"no {cls} frames found under {data_root}/images/{split}")

    def __len__(self):
        return len(self.stems)

    def foreground_fraction(self):
        return float(np.mean([m.mean() for m in self.msks]))

    def __getitem__(self, i):
        g, m = self.imgs[i], self.msks[i]
        if self.train:                                   # flips are physically harmless for a full disk
            if np.random.rand() < 0.5: g, m = g[:, ::-1], m[:, ::-1]
            if np.random.rand() < 0.5: g, m = g[::-1], m[::-1]
        x = torch.from_numpy(np.ascontiguousarray(g, np.float32) / 255.0 - 0.5)[None]
        if self.in_ch == 3:
            x = x.repeat(3, 1, 1)
        return x, torch.from_numpy(np.ascontiguousarray(m, np.float32))[None]


def load_image(path, in_ch):
    """Single frame -> (tensor[1,C,H,W], grayscale uint8) using the training normalisation."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    x = torch.from_numpy(g.astype(np.float32) / 255.0 - 0.5)[None, None]
    if in_ch == 3:
        x = x.repeat(1, 3, 1, 1)
    return x, g
