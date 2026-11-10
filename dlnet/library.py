"""Module library (torch only). Copied into every exported package as model.py.

Families:
  mlp            plain MLP
  resmlp         pre-norm residual MLP
  tt_resmlp      residual MLP whose hidden linears are Tensor-Train factorised
  lowrank_mlp    MLP whose hidden linears are low-rank factorised (U V)
  ft_transformer feature-tokenizer transformer (one token per feature)
"""

from __future__ import annotations

import math

import torch
from torch import nn

FAMILIES = ("mlp", "resmlp", "tt_resmlp", "lowrank_mlp", "ft_transformer")


def _factor(n: int) -> tuple[int, int]:
    best = (1, n)
    for a in range(1, math.isqrt(n) + 1):
        if n % a == 0:
            best = (a, n // a)
    return best


class TTLinear(nn.Module):
    """Linear layer with weight in 2-core Tensor-Train format.

    W[(i1,i2),(o1,o2)] = sum_r G1[i1,o1,r] * G2[r,i2,o2]
    Parameters: r*(i1*o1 + i2*o2) instead of in*out.
    """

    def __init__(self, in_f: int, out_f: int, rank: int):
        super().__init__()
        self.i1, self.i2 = _factor(in_f)
        self.o1, self.o2 = _factor(out_f)
        std = (1.0 / (in_f * rank)) ** 0.25
        self.g1 = nn.Parameter(torch.randn(self.i1, self.o1, rank) * std)
        self.g2 = nn.Parameter(torch.randn(rank, self.i2, self.o2) * std)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        b = x.shape[0]
        t = torch.einsum("bij,ior->bjor", x.reshape(b, self.i1, self.i2), self.g1)
        y = torch.einsum("bjor,rjp->bop", t, self.g2)
        return y.reshape(b, -1) + self.bias


class LowRankLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, rank: int):
        super().__init__()
        self.u = nn.Linear(in_f, rank, bias=False)
        self.v = nn.Linear(rank, out_f)

    def forward(self, x):
        return self.v(self.u(x))


def _linear(kind: str, i: int, o: int, rank: int) -> nn.Module:
    if kind == "tt":
        return TTLinear(i, o, rank)
    if kind == "lowrank":
        return LowRankLinear(i, o, rank)
    return nn.Linear(i, o)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, dropout, kind="dense", rank=8):
        super().__init__()
        layers, d = [], in_dim
        for k in range(depth):
            lin = nn.Linear(d, width) if k == 0 else _linear(kind, width, width, rank)
            layers += [lin, nn.GELU(), nn.Dropout(dropout)]
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.head(self.body(x))


class ResBlock(nn.Module):
    def __init__(self, width, dropout, kind, rank):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.l1 = _linear(kind, width, width, rank)
        self.l2 = _linear(kind, width, width, rank)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.l2(self.drop(torch.nn.functional.gelu(self.l1(self.norm(x)))))


class ResMLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, dropout, kind="dense", rank=8):
        super().__init__()
        self.stem = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width, dropout, kind, rank) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.head(self.norm(self.blocks(self.stem(x))))


class FTTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, dropout):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_dim, width) * 0.02)
        self.b = nn.Parameter(torch.zeros(in_dim, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=width * 2, dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        t = x.unsqueeze(-1) * self.w + self.b
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        return self.head(self.norm(self.enc(t)[:, 0]))


def build_model(cfg: dict, in_dim: int, out_dim: int) -> nn.Module:
    f, w, d, p, r = cfg["family"], cfg["width"], cfg["depth"], cfg["dropout"], cfg.get("rank", 8)
    if f == "mlp":
        return MLP(in_dim, out_dim, w, d, p)
    if f == "lowrank_mlp":
        return MLP(in_dim, out_dim, w, d, p, kind="lowrank", rank=r)
    if f == "resmlp":
        return ResMLP(in_dim, out_dim, w, d, p)
    if f == "tt_resmlp":
        return ResMLP(in_dim, out_dim, w, d, p, kind="tt", rank=r)
    if f == "ft_transformer":
        return FTTransformer(in_dim, out_dim, w, d, p)
    raise ValueError(f"unknown family {f}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
