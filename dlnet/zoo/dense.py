"""Encoder-decoder composition for dense prediction (segmentation, reconstruction), torch only.

Decoders: unet (Ronneberger et al., arXiv:1505.04597, from memory), fcn (Long et al.,
arXiv:1411.4038, from memory), deconv (rasbt/deeplearning-models ConvolutionalAutoencoder
decoder), fpn (Lin et al., arXiv:1612.03144, from memory; pytorch/vision feature_pyramid_network),
aspp (DeepLabV3 head, pytorch/vision segmentation.deeplabv3.ASPP, with rates scaled to the map).
"""

from __future__ import annotations

import torch
from torch import nn

from .ops_basic import ConvBNAct

DECODERS = ("unet", "fcn", "deconv", "fpn", "aspp")


def resize(x, size):
    return (
        x
        if tuple(x.shape[-2:]) == tuple(size)
        else nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
    )


def cbr(i, o, k=3, d=1):
    return ConvBNAct(2, i, o, k, 1, 1, d, "relu", "batch")


class EncDec(nn.Module):
    def __init__(self, stages, feats, dec, width, in_dims):
        super().__init__()
        if dec not in DECODERS:
            raise ValueError(f"unknown decoder {dec!r}; use one of {DECODERS}")
        self.stages, self.dec, self.n = nn.ModuleList(stages), dec, len(stages)
        ch = [f[0] for f in feats]
        if dec == "unet":
            self.up = nn.ModuleList(
                nn.Sequential(cbr(ch[i + 1] + ch[i], ch[i]), cbr(ch[i], ch[i]))
                for i in range(self.n - 1)
            )
            self.out_ch = ch[0]
        elif dec == "fcn":
            self.score = nn.ModuleList(nn.Conv2d(c, width, 1) for c in ch)
            self.out_ch = width
        elif dec == "deconv":
            self.up = nn.ModuleList(
                nn.Sequential(
                    nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2),
                    nn.BatchNorm2d(ch[i]),
                    nn.LeakyReLU(),
                )
                for i in range(self.n - 1)
            )
            self.out_ch = ch[0]
        elif dec == "fpn":
            self.lat = nn.ModuleList(nn.Conv2d(c, width, 1) for c in ch)
            self.smooth = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in ch)
            self.out_ch = width
        else:  # aspp on the deepest feature
            h, w = feats[-1][1:]
            rates = sorted({r for r in (1, 2, 4) if r < min(h, w)} or {1})
            self.branches = nn.ModuleList(
                [cbr(ch[-1], width, 1)] + [cbr(ch[-1], width, 3, r) for r in rates]
            )
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch[-1], width, 1), nn.ReLU()
            )
            self.project = cbr(width * (len(rates) + 2), width, 1)
            self.out_ch = width

    def forward(self, x):
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        size = feats[0].shape[-2:]
        if self.dec == "unet":
            y = feats[-1]
            for i in range(self.n - 2, -1, -1):
                y = self.up[i](torch.cat([resize(y, feats[i].shape[-2:]), feats[i]], 1))
            return y
        if self.dec == "fcn":
            y = self.score[-1](feats[-1])
            for i in range(self.n - 2, -1, -1):
                y = resize(y, feats[i].shape[-2:]) + self.score[i](feats[i])
            return y
        if self.dec == "deconv":
            y = feats[-1]
            for i in range(self.n - 2, -1, -1):
                y = resize(self.up[i](y), feats[i].shape[-2:])
            return y
        if self.dec == "fpn":
            y, outs = self.lat[-1](feats[-1]), []
            outs.append(self.smooth[-1](y))
            for i in range(self.n - 2, -1, -1):
                y = resize(y, feats[i].shape[-2:]) + self.lat[i](feats[i])
                outs.append(self.smooth[i](y))
            return sum(resize(o, size) for o in outs)
        z = feats[-1]
        parts = [b(z) for b in self.branches]
        parts.append(resize(self.pool(z), z.shape[-2:]))
        return self.project(torch.cat(parts, 1))
