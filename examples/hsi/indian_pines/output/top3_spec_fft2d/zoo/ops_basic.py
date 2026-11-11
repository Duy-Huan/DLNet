"""Primitives and adapters between tensor kinds (torch only).

Sources: rasbt/deeplearning-models (FilterResponseNormalization, MLP, NiN), prescient-design/
cortex (SinePosEncoder, MLP, MeanPooling, WeightedMeanPooling), ylongqi/openrec (MLP),
yamsgithub/modular_deep_learning (escort activation), pytorch/vision (Conv2dNormActivation).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .core import ND, op

CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
BN = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
GAP = {1: nn.AdaptiveAvgPool1d, 2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}
MAXP = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
AVGP = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}
SP = ("ser", "map", "vol")  # kinds with a convolutional structure
ALL = ("vec", "seq", "ser", "map", "vol")


class Escort(nn.Module):
    """x^2 / sum(x^2) over features (modular_deep_learning `escort`: a softmax alternative).

    The source divides by the exact sum; an epsilon is added here so a zero vector does not give NaN.
    """

    def forward(self, x):
        n = x.square()
        return n / (n.sum(1, keepdim=True) + 1e-8)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


ACTS = {
    "relu": nn.ReLU, "gelu": nn.GELU, "swish": nn.SiLU, "tanh": nn.Tanh, "leaky": lambda: nn.LeakyReLU(0.1),
    "relu6": nn.ReLU6, "hswish": nn.Hardswish, "sigmoid": nn.Sigmoid, "escort": Escort, "identity": nn.Identity,
}  # fmt: skip


def act(name: str) -> nn.Module:
    if name not in ACTS:
        raise ValueError(f"unknown activation {name!r}")
    return ACTS[name]()


def channels(dims, kind) -> int:
    return dims[1] if kind == "seq" else dims[0]


class FRN(nn.Module):
    """Filter Response Normalization + TLU (Singh & Krishnan, arXiv:1911.09737).

    Port of rasbt/deeplearning-models `FilterResponseNormalization`: x / sqrt(mean(x^2) + |eps|),
    then max(gamma x + beta, tau) with all parameters per channel.
    """

    def __init__(self, c, nd, eps=1e-6):
        super().__init__()
        shape = (1, c, *[1] * nd)
        self.gamma, self.beta = nn.Parameter(torch.ones(shape)), nn.Parameter(torch.zeros(shape))
        self.tau, self.eps = nn.Parameter(torch.zeros(shape)), nn.Parameter(torch.full((1,), eps))
        self.dims = tuple(range(2, 2 + nd))

    def forward(self, x):
        x = x * torch.rsqrt(x.square().mean(self.dims, keepdim=True) + self.eps.abs())
        return torch.max(self.gamma * x + self.beta, self.tau)


class ChannelLast(nn.Module):
    """Apply `f` (acts on the last axis) to a channels-first tensor."""

    def __init__(self, f, nd):
        super().__init__()
        self.f, self.p, self.q = f, (0, *range(2, 2 + nd), 1), (0, nd + 1, *range(1, nd + 1))

    def forward(self, x):
        return self.f(x.permute(*self.p)).permute(*self.q)


def norm_layer(kind, c, name):
    if kind in ("vec", "seq"):
        if name in ("layer", "group", "frn"):
            return nn.LayerNorm(c)
        if kind == "vec":
            return nn.BatchNorm1d(c)
        return _SeqBN(c)
    nd = ND[kind]
    if name == "batch":
        return BN[nd](c)
    if name == "layer":
        return nn.GroupNorm(1, c)
    if name == "group":
        g = next(g for g in (8, 4, 2, 1) if c % g == 0)
        return nn.GroupNorm(g, c)
    if name == "frn":
        return FRN(c, nd)
    raise ValueError(f"unknown norm {name!r}")


class _SeqBN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


@op("norm", ALL, space={"norm": ["batch", "layer", "group", "frn"]}, tags={"norm"},
    source="torch.nn; rasbt/deeplearning-models (FRN)")  # fmt: skip
def _norm(dims, kind, norm):
    """Normalisation over channels / features (batch, layer, group, filter-response)."""
    return norm_layer(kind, channels(dims, kind), norm)


@op(
    "act",
    ALL,
    space={"act": list(ACTS)},
    tags={"act"},
    source="torch.nn; modular_deep_learning (escort)",
)
def _act(dims, kind, act):
    """Pointwise activation (escort only makes sense for vectors)."""
    if act == "escort" and kind != "vec":
        raise ValueError("escort works on vectors")
    return ACTS[act]()


@op("dropout", ALL, space={"p": [0.0, 0.1, 0.2, 0.3, 0.5]}, tags={"reg"}, source="torch.nn")
def _dropout(dims, kind, p):
    return nn.Dropout(p)


def _same_pad(k, dilation):
    """'same' padding for an int or per-axis kernel size."""
    if isinstance(k, (list, tuple)):
        return tuple(dilation * (x - 1) // 2 for x in k)
    return dilation * (k - 1) // 2


class ConvBNAct(nn.Sequential):
    def __init__(self, nd, i, o, k, s, groups, dilation, a, norm, pad="same"):
        pad = 0 if pad == "valid" else _same_pad(k, dilation)
        k, s = (tuple(k) if isinstance(k, list) else k), (tuple(s) if isinstance(s, list) else s)
        layers = [CONV[nd](i, o, k, s, pad, dilation=dilation, groups=groups, bias=norm == "none")]
        if norm != "none":
            layers.append(norm_layer({1: "ser", 2: "map", 3: "vol"}[nd], o, norm))
        if a != "identity":
            layers.append(ACTS[a]())
        super().__init__(*layers)


@op("conv_bn_act", SP, space={"out": [16, 32, 64, 128], "k": [3, 1, 5, 7], "s": [1, 2],
    "groups": [1, 2, 4, 8, 16], "dilation": [1, 2, 4], "act": ["relu", "gelu", "swish", "leaky", "identity"],
    "norm": ["batch", "group", "layer", "frn", "none"], "pad": ["same", "valid"]}, tags={"stem", "block", "down"},
    source="pytorch/vision Conv2dNormActivation; rasbt/deeplearning-models (conv layers)")  # fmt: skip
def _conv_bn_act(dims, kind, out, k, s, groups, dilation, act, norm, pad):
    """Convolution (any of 1/2/3-D), normalisation, activation; grouped and dilated variants.

    k and s may be a list with one entry per spatial axis (e.g. [7, 3, 3] for a 3-D convolution
    over (bands, height, width)); pad "valid" uses no padding.
    """
    c = dims[0]
    if c % groups or out % groups:
        raise ValueError("groups must divide the channels")
    return ConvBNAct(ND[kind], c, out, k, s, groups, dilation, act, norm, pad)


@op("dw_sep", SP, space={"out": [16, 32, 64, 128], "k": [3, 5], "s": [1, 2], "act": ["relu", "relu6", "swish"]},
    tags={"block", "down"}, source="MobileNet (Howard et al., arXiv:1704.04861), from memory")  # fmt: skip
def _dw_sep(dims, kind, out, k, s, act):
    """Depthwise k x k convolution then 1 x 1 pointwise convolution (each with BN + activation)."""
    nd, c = ND[kind], dims[0]
    return nn.Sequential(
        ConvBNAct(nd, c, c, k, s, c, 1, act, "batch"),
        ConvBNAct(nd, c, out, 1, 1, 1, 1, act, "batch"),
    )


@op(
    "pool",
    SP,
    space={"mode": ["max", "avg"], "k": [2, 3], "s": [2, 1]},
    tags={"down"},
    source="torch.nn",
)
def _pool(dims, kind, mode, k, s):
    """Max or average pooling; k and s may be per-axis lists, odd sizes round up (ceil mode)."""
    nd = ND[kind]
    k, s = (tuple(k) if isinstance(k, list) else k), (tuple(s) if isinstance(s, list) else s)
    pad = k // 2 if (s == 1 and isinstance(k, int)) else 0
    return (MAXP if mode == "max" else AVGP)[nd](k, s, pad, ceil_mode=pad == 0)


# ---------------------------------------------------------------------- adapters


@op("gap", SP, "vec", space={"mode": ["avg", "max"]}, tags={"adapt"}, source="torch.nn")
def _gap(dims, kind, mode):
    """Global pooling over the spatial axes: (C, ...) -> (C)."""
    nd = ND[kind]
    return nn.Sequential(
        (GAP if mode == "avg" else {1: nn.AdaptiveMaxPool1d, 2: nn.AdaptiveMaxPool2d, 3: nn.AdaptiveMaxPool3d})[nd](1),
        nn.Flatten(),
    )  # fmt: skip


@op(
    "flatten",
    ("seq", "ser", "map", "vol", "graph"),
    "vec",
    space={},
    tags={"adapt"},
    source="torch.nn",
)
def _flatten(dims, kind):
    return nn.Flatten()


class PoolFlatten(nn.Module):
    """Adaptive average pooling to a fixed spatial size, then flatten (what the old CNN families end with)."""

    def __init__(self, nd, size):
        super().__init__()
        self.f = nn.Sequential(
            GAP[nd](tuple(size) if isinstance(size, list) else size), nn.Flatten()
        )

    def forward(self, x):
        return self.f(x)


@op(
    "pool_flatten",
    SP,
    "vec",
    space={"size": [3, 2, 1]},
    tags={"adapt"},
    source="torch.nn (CNN heads of the earlier families)",
)
def _pool_flatten(dims, kind, size):
    """(C, *spatial) -> (C * prod(size)): adaptive average pooling to `size` (int or per-axis list), flattened."""
    return PoolFlatten(ND[kind], size)


class CenterSpectrum(nn.Module):
    def forward(self, x):
        h, w = x.shape[2] // 2, x.shape[3] // 2
        return x[:, :, h, w].unsqueeze(1)


@op(
    "center_spectrum",
    "map",
    "ser",
    space={},
    tags={"adapt", "stem"},
    source="DLNet HSI: the spectrum of the centre pixel as a series (1, bands)",
)
def _center_spectrum(dims, kind):
    return CenterSpectrum()


class ToVol(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)


@op(
    "to_vol",
    "map",
    "vol",
    space={},
    tags={"adapt", "stem"},
    source="DLNet HSI: (bands, H, W) -> (1, bands, H, W), the spectrum becomes the depth axis",
)
def _to_vol(dims, kind):
    return ToVol()


class VolToMap(nn.Module):
    def forward(self, x):
        n, c, d, h, w = x.shape
        return x.reshape(n, c * d, h, w)


@op(
    "vol_to_map",
    "vol",
    "map",
    space={},
    tags={"adapt"},
    source="HybridSN (Roy et al., arXiv:1902.06701): fold depth into channels",
)
def _vol_to_map(dims, kind):
    return VolToMap()


class Tokens(nn.Module):
    def forward(self, x):
        return x.flatten(2).transpose(1, 2)


@op("tokens", SP, "seq", space={}, tags={"adapt"}, source="DLNet (spatial positions become tokens)")
def _tokens(dims, kind):
    """(C, *spatial) -> (T, C): every spatial position is a token whose features are the channels."""
    return Tokens()


@op("patchify", ("map", "ser"), "seq", space={"dim": [32, 64, 128], "patch": [1, 2, 4, 8]}, tags={"adapt", "stem"},
    source="ViT (arXiv:2010.11929) patch embedding, from memory")  # fmt: skip
def _patchify(dims, kind, dim, patch):
    """Non-overlapping patch embedding by a strided convolution, then tokens."""
    nd = ND[kind]
    if any(s % patch for s in dims[1:]):
        raise ValueError(f"patch {patch} does not divide {dims[1:]}")
    return nn.Sequential(CONV[nd](dims[0], dim, patch, patch), Tokens())


class SeqToSer(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


@op("seq_to_ser", "seq", "ser", space={}, tags={"adapt"}, source="DLNet")
def _seq_to_ser(dims, kind):
    return SeqToSer()


class SeqPool(nn.Module):
    def __init__(self, mode, d):
        super().__init__()
        self.mode = mode
        if mode == "attn":  # WeightedMeanPooling (cortex): softmax over positions of a linear score
            self.score = nn.Linear(d, d)

    def forward(self, x):
        if self.mode == "mean":
            return x.mean(1)
        if self.mode == "max":
            return x.max(1).values
        if self.mode == "cls":
            return x[:, 0]
        return (x * torch.softmax(self.score(x), 1)).sum(1)


@op("seq_pool", "seq", "vec", space={"mode": ["mean", "max", "cls", "attn"]}, tags={"adapt"},
    source="prescient-design/cortex MeanPooling / WeightedMeanPooling (padding mask dropped)")  # fmt: skip
def _seq_pool(dims, kind, mode):
    return SeqPool(mode, dims[1])


class FeatureTokens(nn.Module):
    def __init__(self, d, dim):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.randn(d, dim) * 0.02), nn.Parameter(torch.zeros(d, dim))

    def forward(self, x):
        return x.unsqueeze(-1) * self.w + self.b


@op("feature_tokens", "vec", "seq", space={"dim": [16, 32, 64]}, tags={"adapt", "stem"},
    source="FT-Transformer (Gorishniy et al., arXiv:2106.11959): one token per feature, from memory")  # fmt: skip
def _feature_tokens(dims, kind, dim):
    if dims[0] > 128:
        raise ValueError("too many features for one token each")
    return FeatureTokens(dims[0], dim)


class AddCls(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x):
        return torch.cat([self.cls.expand(len(x), -1, -1), x], 1)


@op("cls_token", "seq", space={}, tags={"embed"}, source="ViT / BERT class token")
def _cls_token(dims, kind):
    return AddCls(dims[1])


class SinePos(nn.Module):
    """Sinusoidal positions (Vaswani et al., arXiv:1706.03762 sec 3.5); port of cortex SinePosEncoder."""

    def __init__(self, t, d):
        super().__init__()
        pos = torch.arange(t).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(1, t, d)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)[:, : d // 2]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe


class LearnedPos(nn.Module):
    def __init__(self, t, d):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, t, d) * 0.02)

    def forward(self, x):
        return x + self.pe


@op(
    "pos",
    "seq",
    space={"mode": ["sine", "learned"]},
    tags={"embed"},
    source="prescient-design/cortex SinePosEncoder; ViT",
)
def _pos(dims, kind, mode):
    return SinePos(*dims) if mode == "sine" else LearnedPos(*dims)


# -------------------------------------------------------------------- vector ops


class MLPBlock(nn.Sequential):
    def __init__(self, i, width, depth, a, dropout, out=None):
        layers, d = [], i
        for _ in range(depth):
            layers += [nn.Linear(d, width), ACTS[a](), nn.Dropout(dropout)]
            d = width
        if out:
            layers.append(nn.Linear(d, out))
        super().__init__(*layers)


@op("linear_act", ("vec", "seq"), space={"out": [32, 64, 128, 256], "act": ["gelu", "relu", "swish", "tanh", "identity"]},
    tags={"block", "stem"}, source="torch.nn")  # fmt: skip
def _linear_act(dims, kind, out, act):
    """Linear layer (on the last axis) and activation."""
    return nn.Sequential(nn.Linear(dims[-1], out), ACTS[act]())


@op("mlp", ("vec", "seq"), space={"width": [64, 128, 256], "depth": [1, 2, 3], "act": ["gelu", "relu", "swish"],
    "dropout": [0.0, 0.1, 0.2]}, tags={"block"},
    source="rasbt/deeplearning-models MLP; ylongqi/openrec MLP; cortex MLP")  # fmt: skip
def _mlp(dims, kind, width, depth, act, dropout):
    return MLPBlock(dims[-1], width, depth, act, dropout)


class PreNormMLP(nn.Module):
    def __init__(self, d, ratio, a, dropout):
        super().__init__()
        self.f = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * ratio),
            ACTS[a](),
            nn.Dropout(dropout),
            nn.Linear(d * ratio, d),
        )

    def forward(self, x):
        return self.f(x)


@op("ffn", ("vec", "seq"), space={"ratio": [1, 2, 4], "act": ["gelu", "relu", "swish"], "dropout": [0.0, 0.1]},
    tags={"block"}, source="Transformer feed-forward (cortex MLP uses ratio 4); wrap in res for a residual")  # fmt: skip
def _ffn(dims, kind, ratio, act, dropout):
    """Pre-norm two-layer feed-forward, shape preserving. Use inside {"op": "res"}."""
    return PreNormMLP(dims[-1], ratio, act, dropout)


class GLU(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.f = nn.Linear(i, 2 * o)

    def forward(self, x):
        a, b = self.f(x).chunk(2, -1)
        return a * torch.sigmoid(b)


@op("glu", ("vec", "seq"), space={"out": [32, 64, 128, 256]}, tags={"block"},
    source="Gated linear unit (Dauphin et al., arXiv:1612.08083), from memory")  # fmt: skip
def _glu(dims, kind, out):
    return GLU(dims[-1], out)
