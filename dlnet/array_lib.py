"""Model library for images and time series (torch only). Copied into packages as model.py.

Image families (input N x C x H x W):
  cnn             conv-BN-ReLU stages with max pooling, global average pooling
  resnet          small residual network in the CIFAR style of He et al. (arXiv:1512.03385):
                  3x3 basic blocks, stride-2 stages, 1x1 projection shortcut
  vit             small Vision Transformer (Dosovitskiy et al., arXiv:2010.11929): patch
                  embedding, class token, learned positions, pre-norm encoder
Time-series families (input N x C x T):
  tcn             residual blocks of dilated 1-D convolutions, dilation 2^i (Bai et al.,
                  arXiv:1803.01271). Deviation: "same" (non-causal) padding and BatchNorm instead
                  of causal padding and weight norm, because the task is whole-sequence prediction
  gru             bidirectional GRU, mean over time
  ts_transformer  per-step linear embedding, sinusoidal positions, encoder, mean over time
All blocks are written from memory: not verified against the authors' code. There is no
state-space (Mamba) family yet.
"""

from __future__ import annotations

import math

import torch
from torch import nn

IMAGE_FAMILIES = ("cnn", "resnet", "vit")
TS_FAMILIES = ("tcn", "gru", "ts_transformer")
FAMILIES = IMAGE_FAMILIES + TS_FAMILIES


def _conv_bn(i, o, k=3, stride=1, dim=2):
    conv, bn = (nn.Conv2d, nn.BatchNorm2d) if dim == 2 else (nn.Conv1d, nn.BatchNorm1d)
    return [conv(i, o, k, stride, k // 2, bias=False), bn(o), nn.ReLU()]


class CNN(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, size):
        super().__init__()
        layers, c, w = [], in_ch, width
        for s in range(depth):
            layers += _conv_bn(c, w) + _conv_bn(w, w)
            c = w
            if size >= 8 and s < depth - 1:  # halve while the map is still large
                layers.append(nn.MaxPool2d(2))
                size //= 2
                w *= 2
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, out_dim))

    def forward(self, x):
        return self.head(self.body(x).mean((2, 3)))


class BasicBlock(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.f = nn.Sequential(
            *_conv_bn(i, o, 3, stride), nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o)
        )
        self.short = None
        if stride != 1 or i != o:
            self.short = nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.f(x) + (x if self.short is None else self.short(x)))


class ResNet(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, size):
        super().__init__()
        layers, c = _conv_bn(in_ch, width), width
        for stage in range(3):
            o = width * 2**stage
            down = stage > 0 and size >= 8
            for b in range(depth):
                layers.append(BasicBlock(c, o, 2 if (down and b == 0) else 1))
                c = o
            size = size // 2 if down else size
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, out_dim))

    def forward(self, x):
        return self.head(self.body(x).mean((2, 3)))


def _encoder(width, depth, dropout):
    layer = nn.TransformerEncoderLayer(
        width,
        nhead=4,
        dim_feedforward=width * 2,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class ViT(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, hw, patch):
        super().__init__()
        h, w = hw
        if h % patch or w % patch:
            raise ValueError(f"patch {patch} does not divide the image {h}x{w}")
        self.embed = nn.Conv2d(in_ch, width, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.randn(1, 1 + (h // patch) * (w // patch), width) * 0.02)
        self.enc = _encoder(width, depth, dropout)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        t = self.embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1) + self.pos
        return self.head(self.norm(self.enc(t)[:, 0]))


class TCNBlock(nn.Module):
    def __init__(self, i, o, k, dilation, dropout):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.f = nn.Sequential(
            nn.Conv1d(i, o, k, padding=pad, dilation=dilation, bias=False), nn.BatchNorm1d(o),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(o, o, k, padding=pad, dilation=dilation, bias=False), nn.BatchNorm1d(o),
        )  # fmt: skip
        self.short = None if i == o else nn.Conv1d(i, o, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.f(x) + (x if self.short is None else self.short(x)))


class TCN(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, kernel=3):
        super().__init__()
        blocks, c = [], in_ch
        for i in range(depth):
            blocks.append(TCNBlock(c, width, kernel, 2**i, dropout))
            c = width
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.head(self.body(x).mean(2))


class GRUNet(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout):
        super().__init__()
        self.rnn = nn.GRU(
            in_ch, width, depth, batch_first=True, bidirectional=True,
            dropout=dropout if depth > 1 else 0.0,
        )  # fmt: skip
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * width, out_dim))

    def forward(self, x):
        return self.head(self.rnn(x.transpose(1, 2))[0].mean(1))


def sinusoid(T: int, width: int) -> torch.Tensor:
    """Vaswani et al. (arXiv:1706.03762) sec. 3.5: PE(pos, 2i) = sin(pos / 10000^(2i/d))."""
    pos = torch.arange(T)[:, None]
    div = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
    pe = torch.zeros(T, width)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)[:, : width // 2]
    return pe


class TSTransformer(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, length):
        super().__init__()
        self.embed = nn.Linear(in_ch, width)
        self.register_buffer("pos", sinusoid(length, width))
        self.enc = _encoder(width, depth, dropout)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        t = self.embed(x.transpose(1, 2)) + self.pos[: x.shape[2]]
        return self.head(self.norm(self.enc(t).mean(1)))


def build_model(cfg: dict, in_dim: int, out_dim: int) -> nn.Module:
    """cfg['shape'] is the per-sample input shape: (C, H, W) for images, (C, T) for series."""
    f, w, d, p, shape = cfg["family"], cfg["width"], cfg["depth"], cfg["dropout"], cfg["shape"]
    if f in IMAGE_FAMILIES and len(shape) != 3 or f in TS_FAMILIES and len(shape) != 2:
        raise ValueError(f"family {f} does not accept inputs of shape {shape}")
    if f == "cnn":
        return CNN(in_dim, out_dim, w, d, p, min(shape[1:]))
    if f == "resnet":
        return ResNet(in_dim, out_dim, w, d, p, min(shape[1:]))
    if f == "vit":
        return ViT(in_dim, out_dim, w, d, p, shape[1:], cfg["patch"])
    if f == "tcn":
        return TCN(in_dim, out_dim, w, d, p, cfg.get("kernel", 3))
    if f == "gru":
        return GRUNet(in_dim, out_dim, w, d, p)
    if f == "ts_transformer":
        return TSTransformer(in_dim, out_dim, w, d, p, shape[1])
    raise ValueError(f"unknown family {f}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
