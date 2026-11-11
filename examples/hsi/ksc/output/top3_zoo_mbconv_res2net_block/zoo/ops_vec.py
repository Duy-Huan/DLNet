"""Vector-valued blocks: tensor-factorised linear layers, feature interaction, cross network.

Sources: this repository (TT / TR / low-rank linear, moved here from library.py), ylongqi/openrec
(SecondOrderFeatureInteraction, DLRM), Tensor-Train layer (Novikov et al., arXiv:1509.06569),
Tensor Ring (Zhao et al., arXiv:1606.05535), DCN (Wang et al., arXiv:1708.05123, from memory).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .core import op


def factor(n: int) -> tuple[int, int]:
    """n = a * b with a <= b and a as large as possible (a = 1 for a prime)."""
    best = (1, n)
    for a in range(1, math.isqrt(n) + 1):
        if n % a == 0:
            best = (a, n // a)
    return best


class TTLinear(nn.Module):
    """Linear layer with weight in 2-core Tensor-Train format: W[(i1,i2),(o1,o2)] = sum_r G1[i1,o1,r] G2[r,i2,o2]."""

    def __init__(self, in_f: int, out_f: int, rank: int):
        super().__init__()
        self.i1, self.i2 = factor(in_f)
        self.o1, self.o2 = factor(out_f)
        std = (1.0 / (in_f * rank)) ** 0.25
        self.g1 = nn.Parameter(torch.randn(self.i1, self.o1, rank) * std)
        self.g2 = nn.Parameter(torch.randn(rank, self.i2, self.o2) * std)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        lead = x.shape[:-1]
        x = x.reshape(-1, self.i1, self.i2)
        t = torch.einsum("bij,ior->bjor", x, self.g1)
        y = torch.einsum("bjor,rjp->bop", t, self.g2)
        return y.reshape(*lead, -1) + self.bias


class TRLinear(nn.Module):
    """Linear layer with weight in 2-core Tensor-Ring format: W[(i1,i2),(o1,o2)] = Tr(G1[i1,o1] G2[i2,o2])."""

    def __init__(self, in_f: int, out_f: int, rank: int):
        super().__init__()
        self.i1, self.i2 = factor(in_f)
        self.o1, self.o2 = factor(out_f)
        std = (1.0 / (in_f * rank * rank)) ** 0.25
        self.g1 = nn.Parameter(torch.randn(rank, self.i1, self.o1, rank) * std)
        self.g2 = nn.Parameter(torch.randn(rank, self.i2, self.o2, rank) * std)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        lead = x.shape[:-1]
        x = x.reshape(-1, self.i1, self.i2)
        t = torch.einsum("bij,aior->bjoar", x, self.g1)
        y = torch.einsum("bjoar,rjpa->bop", t, self.g2)
        return y.reshape(*lead, -1) + self.bias


class LowRankLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, rank: int):
        super().__init__()
        self.u, self.v = nn.Linear(in_f, rank, bias=False), nn.Linear(rank, out_f)

    def forward(self, x):
        return self.v(self.u(x))


@op("tt_linear", ("vec", "seq"), space={"out": [64, 128, 256], "rank": [4, 8, 16]}, tags={"block", "tensor"},
    source="Tensorizing Neural Networks (arXiv:1509.06569); this repository")  # fmt: skip
def _tt(dims, kind, out, rank):
    return TTLinear(dims[-1], out, rank)


@op("tr_linear", ("vec", "seq"), space={"out": [64, 128, 256], "rank": [2, 4, 8]}, tags={"block", "tensor"},
    source="Tensor Ring (arXiv:1606.05535, eq. 1); this repository")  # fmt: skip
def _tr(dims, kind, out, rank):
    return TRLinear(dims[-1], out, rank)


@op("lowrank_linear", ("vec", "seq"), space={"out": [64, 128, 256], "rank": [8, 16, 32]}, tags={"block", "tensor"},
    source="this repository (U V factorisation)")  # fmt: skip
def _lowrank(dims, kind, out, rank):
    return LowRankLinear(dims[-1], out, rank)


class FMInteraction(nn.Module):
    """x concatenated with the pairwise dot products of x_i v_i (openrec SecondOrderFeatureInteraction)."""

    def __init__(self, d, rank, self_interaction):
        super().__init__()
        self.v = nn.Parameter(torch.randn(d, rank) * 0.1)
        self.i, self.j = torch.triu_indices(d, d, offset=0 if self_interaction else 1)

    def forward(self, x):
        f = x.unsqueeze(-1) * self.v  # (B, D, rank)
        g = f @ f.transpose(1, 2)
        return torch.cat([x, g[:, self.i, self.j]], 1)


@op("fm_interaction", "vec", space={"rank": [4, 8, 16], "self_interaction": [0, 1]}, tags={"block", "interaction"},
    source="ylongqi/openrec SecondOrderFeatureInteraction (tf2) ported to PyTorch")  # fmt: skip
def _fm(dims, kind, rank, self_interaction):
    if dims[0] > 128:
        raise ValueError("pairwise interaction is quadratic in the feature count")
    return FMInteraction(dims[0], rank, bool(self_interaction))


class CrossNet(nn.Module):
    """x_{l+1} = x0 * (w_l . x_l) + b_l + x_l with x0 the input (Deep & Cross Network v1)."""

    def __init__(self, d, layers):
        super().__init__()
        self.w = nn.ParameterList(nn.Parameter(torch.randn(d) * d**-0.5) for _ in range(layers))
        self.b = nn.ParameterList(nn.Parameter(torch.zeros(d)) for _ in range(layers))

    def forward(self, x0):
        x = x0
        for w, b in zip(self.w, self.b, strict=True):
            x = x0 * (x @ w).unsqueeze(1) + b + x
        return x


@op(
    "cross_net",
    "vec",
    space={"layers": [1, 2, 3]},
    tags={"block", "interaction"},
    source="DCN (arXiv:1708.05123), from memory",
)
def _cross(dims, kind, layers):
    return CrossNet(dims[0], layers)
