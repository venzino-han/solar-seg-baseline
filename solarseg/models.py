"""Three representative segmentation architectures, all trained FROM SCRATCH (no pretrained weights).

    unet       U-Net (Ronneberger et al. 2015), base 64 channels      31.0M params, 1-ch input
    deeplabv3  DeepLabV3 with ResNet-50 backbone (Chen et al. 2017)    39.6M params, 3-ch input
    segformer  SegFormer MiT-B0 (Xie et al. 2021)                       3.7M params, 3-ch input

Every model outputs one logit map of the input size (binary segmentation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Standard 4-level U-Net with transposed-conv upsampling. Output: logits (B,1,H,W)."""

    def __init__(self, in_ch=1, base=64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.d1, self.d2 = DoubleConv(in_ch, c[0]), DoubleConv(c[0], c[1])
        self.d3, self.d4 = DoubleConv(c[1], c[2]), DoubleConv(c[2], c[3])
        self.bott = DoubleConv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.u4, self.c4 = nn.ConvTranspose2d(c[4], c[3], 2, 2), DoubleConv(c[4], c[3])
        self.u3, self.c3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), DoubleConv(c[3], c[2])
        self.u2, self.c2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), DoubleConv(c[2], c[1])
        self.u1, self.c1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), DoubleConv(c[1], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        s1 = self.d1(x); s2 = self.d2(self.pool(s1)); s3 = self.d3(self.pool(s2)); s4 = self.d4(self.pool(s3))
        b = self.bott(self.pool(s4))
        x = self.c4(torch.cat([self.u4(b), s4], 1))
        x = self.c3(torch.cat([self.u3(x), s3], 1))
        x = self.c2(torch.cat([self.u2(x), s2], 1))
        x = self.c1(torch.cat([self.u1(x), s1], 1))
        return self.out(x)


class DeepLabV3(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50
        self.m = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1)

    def forward(self, x):
        return self.m(x)["out"]


class SegFormer(nn.Module):
    """MiT-B0 encoder + all-MLP decoder; logits come out at 1/4 resolution and are upsampled."""

    def __init__(self):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        self.m = SegformerForSemanticSegmentation(SegformerConfig(num_labels=1, num_channels=3))

    def forward(self, x):
        lo = self.m(pixel_values=x).logits
        return F.interpolate(lo, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(arch):
    """-> (model, display name, input channels)."""
    if arch == "unet":
        return UNet(in_ch=1, base=64), "U-Net", 1
    if arch == "deeplabv3":
        return DeepLabV3(), "DeepLabV3-ResNet50", 3
    if arch == "segformer":
        return SegFormer(), "SegFormer-MiT-B0", 3
    raise ValueError(f"unknown arch {arch!r}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
