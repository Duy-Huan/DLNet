"""HSI model library (torch only). Copied into every exported HSI package as model.py.

All models take a patch batch [n, C, p, p] (C = spectral components after the stem) and return logits.

Families:
  cnn1d        spectral-only 1D CNN on the centre pixel
  cnn2d        2D CNN over the patch
  tucker_cnn2d 2D CNN whose 3x3 convs are Tucker-2 factorised (1x1 -> 3x3 -> 1x1), see ref/04_tucker-*
  cnn3d        3D CNN over (spectral, height, width)
  hybridsn     HybridSN-style: 3D convs -> reshape -> 2D conv -> FC
               (re-implemented from memory of Roy et al. 2019; not verified against the original code)
"""

from __future__ import annotations

import torch
from torch import nn

FAMILIES = ("cnn1d", "cnn2d", "tucker_cnn2d", "cnn3d", "hybridsn")


def _cbr2d(i, o, k=3):
    return [
        nn.Conv2d(i, o, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    ]


class CNN1D(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, patch):
        super().__init__()
        self.r = patch // 2
        layers, c = [], 1
        for i in range(depth):
            w = width * min(2**i, 4)
            layers += [
                nn.Conv1d(c, w, 5, padding=2, bias=False),
                nn.BatchNorm1d(w),
                nn.ReLU(inplace=True),
            ]
            if i < depth - 1:
                layers.append(nn.MaxPool1d(2, ceil_mode=True))
            c = w
        self.body = nn.Sequential(
            *layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(c, out_dim)

    def forward(self, x):
        return self.head(self.body(x[:, :, self.r, self.r].unsqueeze(1)))


class CNN2D(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, patch):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(depth):
            layers += _cbr2d(c, width)
            c = width
        self.body = nn.Sequential(
            *layers, nn.AdaptiveAvgPool2d(3), nn.Flatten(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(c * 9, out_dim)

    def forward(self, x):
        return self.head(self.body(x))


class TuckerConv(nn.Module):
    """3x3 conv with Tucker-2 kernel: 1x1 (in->r1), 3x3 (r1->r2), 1x1 (r2->out)."""

    def __init__(self, c_in, c_out, ratio):
        super().__init__()
        r1, r2 = max(4, int(c_in * ratio)), max(4, int(c_out * ratio))
        self.net = nn.Sequential(
            nn.Conv2d(c_in, r1, 1, bias=False),
            nn.Conv2d(r1, r2, 3, padding=1, bias=False),
            nn.Conv2d(r2, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TuckerCNN2D(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, patch, ratio):
        super().__init__()
        layers = [*_cbr2d(in_ch, width)]
        layers += [TuckerConv(width, width, ratio) for _ in range(depth - 1)]
        self.body = nn.Sequential(
            *layers, nn.AdaptiveAvgPool2d(3), nn.Flatten(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(width * 9, out_dim)

    def forward(self, x):
        return self.head(self.body(x))


class CNN3D(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, patch):
        super().__init__()
        layers, c = [], 1
        for i in range(depth):
            layers += [
                nn.Conv3d(c, width, 3, padding=1, bias=False),
                nn.BatchNorm3d(width),
                nn.ReLU(inplace=True),
            ]
            if i < depth - 1:
                layers.append(nn.MaxPool3d((2, 1, 1), ceil_mode=True))
            c = width
        self.body = nn.Sequential(
            *layers, nn.AdaptiveAvgPool3d((2, 3, 3)), nn.Flatten(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(width * 18, out_dim)

    def forward(self, x):
        return self.head(self.body(x.unsqueeze(1)))


class HybridSN(nn.Module):
    def __init__(self, in_ch, out_dim, width, dropout, patch):
        super().__init__()
        c1, c2, c3 = width, 2 * width, 4 * width
        self.c3d = nn.Sequential(
            nn.Conv3d(1, c1, (7, 3, 3)), nn.BatchNorm3d(c1), nn.ReLU(inplace=True),
            nn.Conv3d(c1, c2, (5, 3, 3)), nn.BatchNorm3d(c2), nn.ReLU(inplace=True),
            nn.Conv3d(c2, c3, (3, 3, 3)), nn.BatchNorm3d(c3), nn.ReLU(inplace=True),
        )  # fmt: skip
        with torch.no_grad():
            z = self.c3d(
                torch.zeros(2, 1, in_ch, patch, patch)
            )  # batch of 2: BatchNorm needs >1 value
        _, c, k, h, w = z.shape
        self.c2d = nn.Sequential(
            nn.Conv2d(c * k, 64, 3, padding=1 if min(h, w) < 3 else 0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat = self.c2d(z.reshape(2, c * k, h, w)).numel() // 2
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, out_dim),
        )  # fmt: skip

    def forward(self, x):
        z = self.c3d(x.unsqueeze(1))
        n, c, k, h, w = z.shape
        return self.fc(self.c2d(z.reshape(n, c * k, h, w)))


def build_model(cfg: dict, in_dim: int, out_dim: int) -> nn.Module:
    f, w, d, p, patch = (
        cfg["family"],
        cfg["width"],
        cfg.get("depth", 3),
        cfg["dropout"],
        cfg["patch"],
    )
    if f == "cnn1d":
        return CNN1D(in_dim, out_dim, w, d, p, patch)
    if f == "cnn2d":
        return CNN2D(in_dim, out_dim, w, d, p, patch)
    if f == "tucker_cnn2d":
        return TuckerCNN2D(in_dim, out_dim, w, d, p, patch, cfg["rank_ratio"])
    if f == "cnn3d":
        return CNN3D(in_dim, out_dim, w, d, p, patch)
    if f == "hybridsn":
        return HybridSN(in_dim, out_dim, w, p, patch)
    raise ValueError(f"unknown family {f}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
