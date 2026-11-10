"""Convolutional blocks for 1-D (ser), 2-D (map) and 3-D (vol) inputs (torch only).

Ported to be shape-generic and parameterised. Sources (see THIRD_PARTY.md):
  rasbt/deeplearning-models   BasicBlock, Bottleneck, DenseNet layer/block/transition, NiN, LeNet-5,
                              VGG stage, ResidualBlock
  pytorch/vision              ResNet/ResNeXt/Wide bottleneck (groups, base width), MBConv and
                              FusedMBConv (EfficientNet), MobileNetV3 block, ConvNeXt block,
                              RegNet block, ShuffleNetV2 unit, SqueezeNet Fire, PatchMerging (Swin),
                              SqueezeExcitation
  papers, from memory         CBAM (Woo et al., arXiv:1807.06521), ECA (Wang et al., arXiv:1910.03151),
                              Inception module (Szegedy et al., arXiv:1409.4842), window attention
                              (Swin, arXiv:2103.14030; no relative position bias)
"""

from __future__ import annotations

import torch
from torch import nn

from .core import ND, op
from .ops_basic import ACTS, AVGP, BN, CONV, MAXP, SP, ChannelLast, ConvBNAct, norm_layer

KIND = {1: "ser", 2: "map", 3: "vol"}
CT = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}


def spatial(x):
    return tuple(range(2, x.dim()))


class SE(nn.Module):
    """Squeeze-and-excitation (Hu et al., arXiv:1709.01507): x * gate(FC(act(FC(mean over space))))."""

    def __init__(self, c, nd, hidden, a="relu", gate="sigmoid"):
        super().__init__()
        h = max(int(hidden), 4)
        self.f = nn.Sequential(
            CONV[nd](c, h, 1),
            ACTS[a](),
            CONV[nd](h, c, 1),
            nn.Hardsigmoid() if gate == "hard" else nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f(x.mean(spatial(x), keepdim=True))


@op(
    "se",
    SP,
    space={"ratio": [0.25, 0.5, 0.125]},
    tags={"attn"},
    source="pytorch/vision SqueezeExcitation; Hu et al.",
)
def _se(dims, kind, ratio):
    return SE(dims[0], ND[kind], dims[0] * ratio)


class ECA(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        s = x.mean(spatial(x)).unsqueeze(1)  # (B, 1, C)
        w = torch.sigmoid(self.conv(s)).squeeze(1)
        return x * w.view(*w.shape, *[1] * (x.dim() - 2))


@op(
    "eca",
    SP,
    space={"k": [3, 5, 7]},
    tags={"attn"},
    source="ECA-Net (arXiv:1910.03151), from memory",
)
def _eca(dims, kind, k):
    return ECA(k)


class CBAM(nn.Module):
    def __init__(self, c, ratio):
        super().__init__()
        h = max(c // ratio, 4)
        self.mlp = nn.Sequential(nn.Conv2d(c, h, 1), nn.ReLU(), nn.Conv2d(h, c, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        a = self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True))
        x = x * torch.sigmoid(a)
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.spatial(s))


@op(
    "cbam",
    "map",
    space={"ratio": [4, 8, 16]},
    tags={"attn"},
    source="CBAM (arXiv:1807.06521), from memory",
)
def _cbam(dims, kind, ratio):
    return CBAM(dims[0], ratio)


class BasicBlock(nn.Module):
    """ResNet basic block (rasbt BasicBlock / torchvision): two 3-wide convolutions + shortcut."""

    def __init__(self, nd, i, o, s, a, norm):
        super().__init__()
        self.f = nn.Sequential(
            ConvBNAct(nd, i, o, 3, s, 1, 1, a, norm),
            ConvBNAct(nd, o, o, 3, 1, 1, 1, "identity", norm),
        )
        self.short = (
            None if s == 1 and i == o else ConvBNAct(nd, i, o, 1, s, 1, 1, "identity", norm)
        )
        self.act = ACTS[a]()

    def forward(self, x):
        return self.act(self.f(x) + (x if self.short is None else self.short(x)))


@op("basic_block", SP, space={"out": [16, 32, 64, 128], "s": [1, 2], "act": ["relu", "gelu", "swish"],
    "norm": ["batch", "group", "frn"]}, tags={"block", "down"}, source="rasbt/deeplearning-models BasicBlock; pytorch/vision resnet")  # fmt: skip
def _basic(dims, kind, out, s, act, norm):
    return BasicBlock(ND[kind], dims[0], out, s, act, norm)


class Bottleneck(nn.Module):
    """ResNet / ResNeXt / Wide bottleneck (torchvision): width = out / ratio * base_width / 64 * groups."""

    def __init__(self, nd, i, o, s, ratio, groups, base_width, a, norm):
        super().__init__()
        w = int(o / ratio * (base_width / 64.0)) * groups
        if w < 1 or w % groups:
            raise ValueError("bottleneck width must be a positive multiple of groups")
        self.f = nn.Sequential(
            ConvBNAct(nd, i, w, 1, 1, 1, 1, a, norm), ConvBNAct(nd, w, w, 3, s, groups, 1, a, norm),
            ConvBNAct(nd, w, o, 1, 1, 1, 1, "identity", norm),
        )  # fmt: skip
        self.short = (
            None if s == 1 and i == o else ConvBNAct(nd, i, o, 1, s, 1, 1, "identity", norm)
        )
        self.act = ACTS[a]()

    def forward(self, x):
        return self.act(self.f(x) + (x if self.short is None else self.short(x)))


@op("bottleneck", SP, space={"out": [32, 64, 128, 256], "s": [1, 2], "ratio": [4, 2], "groups": [1, 4, 8, 32],
    "base_width": [64, 128, 4], "act": ["relu", "gelu", "swish"], "norm": ["batch", "group"]},
    tags={"block", "down"}, source="rasbt/deeplearning-models Bottleneck; pytorch/vision resnet (ResNeXt: groups, Wide: base_width)")  # fmt: skip
def _bottleneck(dims, kind, out, s, ratio, groups, base_width, act, norm):
    return Bottleneck(ND[kind], dims[0], out, s, ratio, groups, base_width, act, norm)


class PreActBlock(nn.Module):
    """ResNet v2 block (He et al., arXiv:1603.05027): norm-act-conv twice, identity stays clean."""

    def __init__(self, nd, i, o, s, a, norm):
        super().__init__()
        n = lambda c: norm_layer(KIND[nd], c, norm)
        self.pre = nn.Sequential(n(i), ACTS[a]())
        self.c1 = CONV[nd](i, o, 3, s, 1, bias=False)
        self.c2 = nn.Sequential(n(o), ACTS[a](), CONV[nd](o, o, 3, 1, 1, bias=False))
        self.short = None if s == 1 and i == o else CONV[nd](i, o, 1, s, bias=False)

    def forward(self, x):
        z = self.pre(x)
        return self.c2(self.c1(z)) + (x if self.short is None else self.short(z))


@op("preact_block", SP, space={"out": [16, 32, 64, 128], "s": [1, 2], "act": ["relu", "gelu", "swish"],
    "norm": ["batch", "group"]}, tags={"block", "down"}, source="ResNet v2 (arXiv:1603.05027), from memory")  # fmt: skip
def _preact(dims, kind, out, s, act, norm):
    return PreActBlock(ND[kind], dims[0], out, s, act, norm)


class DenseLayer(nn.Module):
    """DenseNet-BC layer (rasbt _DenseLayer): BN-ReLU-1x1 (bn_size * growth) then BN-ReLU-3x3 (growth); concat."""

    def __init__(self, nd, i, growth, bn_size, dropout):
        super().__init__()
        k = KIND[nd]
        self.f = nn.Sequential(
            norm_layer(k, i, "batch"), nn.ReLU(), CONV[nd](i, bn_size * growth, 1, bias=False),
            norm_layer(k, bn_size * growth, "batch"), nn.ReLU(), CONV[nd](bn_size * growth, growth, 3, padding=1, bias=False),
            nn.Dropout(dropout),
        )  # fmt: skip

    def forward(self, x):
        return torch.cat([x, self.f(x)], 1)


@op("dense_block", SP, space={"layers": [2, 3, 4, 6], "growth": [8, 12, 16, 32], "bn_size": [2, 4], "dropout": [0.0, 0.1]},
    tags={"block"}, source="rasbt/deeplearning-models _DenseBlock/_DenseLayer; Huang et al., arXiv:1608.06993")  # fmt: skip
def _dense_block(dims, kind, layers, growth, bn_size, dropout):
    nd = ND[kind]
    return nn.Sequential(
        *[DenseLayer(nd, dims[0] + k * growth, growth, bn_size, dropout) for k in range(layers)]
    )


@op("transition", SP, space={"compression": [0.5, 1.0], "pool": [2, 1]}, tags={"down"},
    source="rasbt/deeplearning-models _Transition")  # fmt: skip
def _transition(dims, kind, compression, pool):
    nd, k = ND[kind], KIND[ND[kind]]
    o = max(int(dims[0] * compression), 4)
    layers = [norm_layer(k, dims[0], "batch"), nn.ReLU(), CONV[nd](dims[0], o, 1, bias=False)]
    return nn.Sequential(*layers, AVGP[nd](pool, pool) if pool > 1 else nn.Identity())


@op("nin_block", SP, space={"out": [32, 64, 128], "k": [3, 5], "dropout": [0.0, 0.3, 0.5]}, tags={"block"},
    source="rasbt/deeplearning-models NiN (Lin et al., arXiv:1312.4400): conv, then two 1x1 convs")  # fmt: skip
def _nin(dims, kind, out, k, dropout):
    nd = ND[kind]
    c = CONV[nd]
    return nn.Sequential(
        c(dims[0], out, k, padding=k // 2),
        nn.ReLU(),
        c(out, out, 1),
        nn.ReLU(),
        c(out, out, 1),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


@op("lenet_stage", SP, space={"out": [6, 16, 32, 64], "k": [5, 3]}, tags={"block", "down"},
    source="rasbt/deeplearning-models LeNet5 feature stage: conv, tanh, 2x pooling")  # fmt: skip
def _lenet(dims, kind, out, k):
    nd = ND[kind]
    return nn.Sequential(
        CONV[nd](dims[0], out, k, padding=k // 2), nn.Tanh(), AVGP[nd](2, 2, ceil_mode=True)
    )


@op("vgg_stage", SP, space={"out": [16, 32, 64, 128], "n": [1, 2, 3], "norm": ["batch", "none"], "pool": [2, 1]},
    tags={"block", "down"}, source="rasbt/deeplearning-models VGG16 / torchvision vgg: n convs then max pooling")  # fmt: skip
def _vgg(dims, kind, out, n, norm, pool):
    nd, layers, c = ND[kind], [], dims[0]
    for _ in range(n):
        layers.append(ConvBNAct(nd, c, out, 3, 1, 1, 1, "relu", norm))
        c = out
    return nn.Sequential(
        *layers, MAXP[nd](pool, pool, ceil_mode=True) if pool > 1 else nn.Identity()
    )


class MBConv(nn.Module):
    """Mobile inverted bottleneck (torchvision efficientnet.MBConv / mobilenetv3): expand, depthwise, SE, project."""

    def __init__(self, nd, i, o, expand, k, s, se, a):
        super().__init__()
        mid = i * expand
        layers = [ConvBNAct(nd, i, mid, 1, 1, 1, 1, a, "batch")] if expand != 1 else []
        layers.append(ConvBNAct(nd, mid, mid, k, s, mid, 1, a, "batch"))
        if (
            se and a == "hswish"
        ):  # MobileNetV3: SE on the expanded channels (// 4, multiple of 8), ReLU + hard sigmoid
            layers.append(SE(mid, nd, max(round(mid // 4 / 8) * 8, 8), "relu", "hard"))
        elif se:  # EfficientNet: squeeze to input channels // 4
            layers.append(SE(mid, nd, max(i // 4, 4), "swish" if a != "relu" else "relu"))
        layers.append(ConvBNAct(nd, mid, o, 1, 1, 1, 1, "identity", "batch"))
        self.f, self.res = nn.Sequential(*layers), s == 1 and i == o

    def forward(self, x):
        return x + self.f(x) if self.res else self.f(x)


@op("mbconv", SP, space={"out": [16, 24, 32, 64, 96], "expand": [4, 6, 1], "k": [3, 5], "s": [1, 2], "se": [0, 1],
    "act": ["swish", "hswish", "relu6"]}, tags={"block", "down"},
    source="pytorch/vision efficientnet.MBConv, mobilenetv3.InvertedResidual")  # fmt: skip
def _mbconv(dims, kind, out, expand, k, s, se, act):
    return MBConv(ND[kind], dims[0], out, expand, k, s, se, act)


class FusedMBConv(nn.Module):
    def __init__(self, nd, i, o, expand, k, s, se, a):
        super().__init__()
        mid = i * expand
        layers = [ConvBNAct(nd, i, mid, k, s, 1, 1, a, "batch")]
        if se:
            layers.append(SE(mid, nd, max(i // 4, 4), "swish"))
        if expand != 1:
            layers.append(ConvBNAct(nd, mid, o, 1, 1, 1, 1, "identity", "batch"))
        self.f, self.res = nn.Sequential(*layers), s == 1 and i == o and expand != 1

    def forward(self, x):
        return x + self.f(x) if self.res else self.f(x)


@op("fused_mbconv", SP, space={"out": [16, 24, 32, 64], "expand": [4, 2, 1], "k": [3, 5], "s": [1, 2], "se": [0, 1],
    "act": ["swish", "relu"]}, tags={"block", "down"}, source="pytorch/vision efficientnet.FusedMBConv")  # fmt: skip
def _fused(dims, kind, out, expand, k, s, se, act):
    return FusedMBConv(ND[kind], dims[0], out, expand, k, s, se, act)


class CNBlock(nn.Module):
    """ConvNeXt block (torchvision convnext.CNBlock): depthwise k conv, LayerNorm, 1x1 expand, GELU, 1x1, layer scale."""

    def __init__(self, nd, c, k, ratio, ls):
        super().__init__()
        self.dw = CONV[nd](c, c, k, padding=k // 2, groups=c)
        self.ln, self.p1, self.p2 = (
            nn.LayerNorm(c, eps=1e-6),
            nn.Linear(c, ratio * c),
            nn.Linear(ratio * c, c),
        )
        self.gamma = nn.Parameter(torch.full((c,), ls))

    def forward(self, x):
        y = self.dw(x).movedim(1, -1)
        y = self.p2(nn.functional.gelu(self.p1(self.ln(y)))) * self.gamma
        return x + y.movedim(-1, 1)


@op("convnext_block", SP, space={"k": [7, 5, 3], "ratio": [4, 2], "layer_scale": [1e-6, 1e-2]}, tags={"block"},
    source="pytorch/vision convnext.CNBlock (shape preserving, residual inside)")  # fmt: skip
def _convnext(dims, kind, k, ratio, layer_scale):
    return CNBlock(ND[kind], dims[0], k, ratio, layer_scale)


class RegNetBlock(nn.Module):
    """RegNet X/Y bottleneck block with grouped 3x3 (torchvision regnet.ResBottleneckBlock)."""

    def __init__(self, nd, i, o, s, gw, ratio, se):
        super().__init__()
        wb = max(round(o * ratio), gw)
        if wb % gw:
            raise ValueError("bottleneck width must be a multiple of the group width")
        layers = [
            ConvBNAct(nd, i, wb, 1, 1, 1, 1, "relu", "batch"),
            ConvBNAct(nd, wb, wb, 3, s, wb // gw, 1, "relu", "batch"),
        ]
        if se:
            layers.append(SE(wb, nd, max(i * se, 4)))
        layers.append(ConvBNAct(nd, wb, o, 1, 1, 1, 1, "identity", "batch"))
        self.f = nn.Sequential(*layers)
        self.short = (
            None if s == 1 and i == o else ConvBNAct(nd, i, o, 1, s, 1, 1, "identity", "batch")
        )

    def forward(self, x):
        return torch.relu(self.f(x) + (x if self.short is None else self.short(x)))


@op("regnet_block", SP, space={"out": [32, 64, 128], "s": [1, 2], "group_width": [8, 16], "ratio": [1.0, 0.5], "se": [0, 0.25]},
    tags={"block", "down"}, source="pytorch/vision regnet.ResBottleneckBlock")  # fmt: skip
def _regnet(dims, kind, out, s, group_width, ratio, se):
    return RegNetBlock(ND[kind], dims[0], out, s, group_width, ratio, se)


def channel_shuffle(x, groups):
    b, c = x.shape[:2]
    return x.view(b, groups, c // groups, *x.shape[2:]).transpose(1, 2).reshape(x.shape)


class ShuffleUnit(nn.Module):
    """ShuffleNetV2 unit (torchvision shufflenetv2.InvertedResidual), 2-D only."""

    def __init__(self, i, o, s):
        super().__init__()
        self.s = s
        bf = o // 2
        dw = lambda c, st: nn.Sequential(
            nn.Conv2d(c, c, 3, st, 1, groups=c, bias=False), nn.BatchNorm2d(c)
        )
        pw = lambda a, b: nn.Sequential(
            nn.Conv2d(a, b, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU()
        )
        self.b1 = nn.Sequential(dw(i, s), pw(i, bf)) if s > 1 else nn.Identity()
        self.b2 = nn.Sequential(pw(i if s > 1 else bf, bf), dw(bf, s), pw(bf, bf))

    def forward(self, x):
        if self.s == 1:
            a, b = x.chunk(2, 1)
            y = torch.cat([a, self.b2(b)], 1)
        else:
            y = torch.cat([self.b1(x), self.b2(x)], 1)
        return channel_shuffle(y, 2)


@op("shuffle_unit", "map", space={"out": [32, 64, 128], "s": [1, 2]}, tags={"block", "down"},
    source="pytorch/vision shufflenetv2.InvertedResidual (out is ignored when s=1)")  # fmt: skip
def _shuffle(dims, kind, out, s):
    c = dims[0]
    if s == 1:
        if c % 2:
            raise ValueError("stride-1 shuffle unit needs an even channel count")
        out = c
    elif out % 2:
        raise ValueError("out must be even")
    return ShuffleUnit(c, out, s)


class Fire(nn.Module):
    """SqueezeNet Fire module (torchvision squeezenet.Fire): 1x1 squeeze, then parallel 1x1 and 3x3 expand."""

    def __init__(self, nd, i, sq, e1, e3):
        super().__init__()
        self.sq = nn.Sequential(CONV[nd](i, sq, 1), nn.ReLU())
        self.e1, self.e3 = (
            nn.Sequential(CONV[nd](sq, e1, 1), nn.ReLU()),
            nn.Sequential(CONV[nd](sq, e3, 3, padding=1), nn.ReLU()),
        )

    def forward(self, x):
        x = self.sq(x)
        return torch.cat([self.e1(x), self.e3(x)], 1)


@op(
    "fire",
    SP,
    space={"out": [32, 64, 128], "squeeze_ratio": [8, 4]},
    tags={"block"},
    source="pytorch/vision squeezenet.Fire",
)
def _fire(dims, kind, out, squeeze_ratio):
    return Fire(ND[kind], dims[0], max(out // squeeze_ratio, 4), out // 2, out - out // 2)


class Inception(nn.Module):
    """Inception module with 1x1, 3x3, 5x5 and pooling branches (GoogLeNet, from memory)."""

    def __init__(self, nd, i, o):
        super().__init__()
        q, c = o // 4, CONV[nd]
        cbr = lambda a, b, k: ConvBNAct(nd, a, b, k, 1, 1, 1, "relu", "batch")
        self.b1 = cbr(i, q, 1)
        self.b2 = nn.Sequential(cbr(i, max(q // 2, 4), 1), cbr(max(q // 2, 4), q, 3))
        self.b3 = nn.Sequential(cbr(i, max(q // 2, 4), 1), cbr(max(q // 2, 4), q, 5))
        self.b4 = nn.Sequential(MAXP[nd](3, 1, 1), cbr(i, o - 3 * q, 1))
        del c

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


@op(
    "inception",
    SP,
    space={"out": [32, 64, 128]},
    tags={"block"},
    source="GoogLeNet inception module (arXiv:1409.4842), from memory",
)
def _inception(dims, kind, out):
    return Inception(ND[kind], dims[0], out)


class PatchMerging(nn.Module):
    """Swin PatchMerging (torchvision swin_transformer): 2x2 neighbours -> concat -> LayerNorm -> Linear(4C, 2C)."""

    def __init__(self, c):
        super().__init__()
        self.norm, self.red = nn.LayerNorm(4 * c), nn.Linear(4 * c, 2 * c, bias=False)

    def forward(self, x):
        x = torch.cat(
            [x[..., 0::2, 0::2], x[..., 1::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 1::2]], 1
        )
        return self.red(self.norm(x.movedim(1, -1))).movedim(-1, 1)


@op(
    "patch_merge",
    "map",
    space={},
    tags={"down"},
    source="pytorch/vision swin_transformer.PatchMerging",
)
def _patch_merge(dims, kind):
    if dims[1] % 2 or dims[2] % 2:
        raise ValueError("patch merging needs even height and width")
    return PatchMerging(dims[0])


class SelfAttn(nn.Module):
    """Global multi-head self-attention over all spatial positions, pre-norm, residual inside."""

    def __init__(self, c, heads):
        super().__init__()
        if c % heads:
            raise ValueError("heads must divide channels")
        self.ln, self.attn = nn.LayerNorm(c), nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        t = self.ln(x.flatten(2).transpose(1, 2))
        return x + self.attn(t, t, t, need_weights=False)[0].transpose(1, 2).reshape(x.shape)


@op(
    "self_attn",
    SP,
    space={"heads": [1, 2, 4, 8]},
    tags={"attn"},
    source="torch.nn.MultiheadAttention over spatial tokens",
)
def _self_attn(dims, kind, heads):
    return SelfAttn(dims[0], heads)


class WindowAttn(nn.Module):
    """Window multi-head self-attention with optional cyclic shift (Swin, arXiv:2103.14030), no relative bias."""

    def __init__(self, c, heads, w, shift):
        super().__init__()
        if c % heads:
            raise ValueError("heads must divide channels")
        self.w, self.shift = w, shift
        self.ln, self.attn = nn.LayerNorm(c), nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        z = torch.roll(x, (-self.shift, -self.shift), (2, 3)) if self.shift else x
        z = (
            z.view(b, c, h // self.w, self.w, w // self.w, self.w)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, self.w * self.w, c)
        )
        z = self.ln(z)
        z = self.attn(z, z, z, need_weights=False)[0]
        z = (
            z.view(b, h // self.w, w // self.w, self.w, self.w, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, c, h, w)
        )
        return x + (torch.roll(z, (self.shift, self.shift), (2, 3)) if self.shift else z)


@op("window_attn", "map", space={"heads": [1, 2, 4], "window": [2, 4, 7, 8], "shift": [0, 1]}, tags={"attn", "block"},
    source="Swin window attention (arXiv:2103.14030), from memory; torchvision swin_transformer for the layout")  # fmt: skip
def _window_attn(dims, kind, heads, window, shift):
    if dims[1] % window or dims[2] % window:
        raise ValueError("window must divide height and width")
    return WindowAttn(dims[0], heads, window, window // 2 if shift else 0)


@op("up_bilinear", SP, space={"scale": [2]}, tags={"up"}, source="torch.nn.Upsample")
def _up(dims, kind, scale):
    return nn.Upsample(
        scale_factor=scale, mode={"ser": "linear", "map": "bilinear", "vol": "trilinear"}[kind]
    )


@op(
    "up_deconv",
    SP,
    space={"out": [16, 32, 64]},
    tags={"up"},
    source="rasbt/deeplearning-models ConvolutionalAutoencoder decoder",
)
def _up_deconv(dims, kind, out):
    nd = ND[kind]
    return nn.Sequential(CT[nd](dims[0], out, 2, 2), BN[nd](out), nn.LeakyReLU())


_ = (ChannelLast,)
