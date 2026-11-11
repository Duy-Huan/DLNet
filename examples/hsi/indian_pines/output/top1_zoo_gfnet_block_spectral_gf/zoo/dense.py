"""Encoder-decoder composition for dense prediction (segmentation, reconstruction), torch only.

Decoders ported from qubvel-org/segmentation_models_pytorch 0.6.0 (`decoders/*/decoder.py`):
unet, unetpp (UNet++), fpn, deeplabv3, deeplabv3plus, pspnet, linknet, upernet, segformer.
The ports keep the layer structure, layer order and initial values of smp; the only change is
that every fixed "upsample x2" resizes to the size of the feature it is fused with, so an
encoder may have any number of stages and any input size. With the standard layout (stride 2
between features) the ports are numerically equal to smp (scripts/verify_numeric.py, THIRD_PARTY.md).

Feature convention (as in smp): features = [input, f0, f1, ...], one entry per encoder stage.
Kept from earlier DLNet versions: fcn (FCN-8s style score fusion, Long et al.,
arXiv:1411.4038, written from memory) and deconv (rasbt/deeplearning-models
ConvolutionalAutoencoder decoder, used by the autoencoders).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

DECODERS = (
    "unet", "unetpp", "fpn", "deeplabv3", "deeplabv3plus", "pspnet", "linknet", "upernet",
    "segformer", "fcn", "deconv",
)  # fmt: skip
ALIASES = {"aspp": "deeplabv3"}  # name used by DLNet <= 0.12
ATTENTIONS = ("none", "scse")


def resize(x, size, mode="bilinear", align_corners=False):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    if mode == "nearest":
        return F.interpolate(x, size=tuple(size), mode="nearest")
    return F.interpolate(x, size=tuple(size), mode=mode, align_corners=align_corners)


# --------------------------------------------------------------------- smp base modules


class Conv2dReLU(nn.Sequential):
    """smp base.modules.Conv2dReLU: conv (bias only without norm), BatchNorm, ReLU."""

    def __init__(self, i, o, k, padding=0, stride=1, use_norm=True):
        norm = nn.BatchNorm2d(o) if use_norm else nn.Identity()
        conv = nn.Conv2d(i, o, k, stride=stride, padding=padding, bias=not use_norm)
        super().__init__(conv, norm, nn.ReLU())


class SCSE(nn.Module):
    """smp base.modules.SCSEModule (Roy et al., arXiv:1803.02579): channel SE + spatial SE, summed."""

    def __init__(self, c, reduction=16):
        super().__init__()
        r = max(c // reduction, 1)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, r, 1), nn.ReLU(), nn.Conv2d(r, c, 1), nn.Sigmoid()
        )
        self.sSE = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


def _attn(name, c):
    return SCSE(c) if name == "scse" else nn.Identity()


# ----------------------------------------------------------------------------- U-Net


class UnetBlock(nn.Module):
    def __init__(self, i, skip, o, attention="none"):
        super().__init__()
        self.conv1 = Conv2dReLU(i + skip, o, 3, padding=1)
        self.attention1 = _attn(attention, i + skip)
        self.conv2 = Conv2dReLU(o, o, 3, padding=1)
        self.attention2 = _attn(attention, o)

    def forward(self, x, size, skip=None):
        x = resize(x, size, "nearest")
        if skip is not None:
            x = self.attention1(torch.cat([x, skip], 1))
        return self.attention2(self.conv2(self.conv1(x)))


class UnetDecoder(nn.Module):
    """smp UnetDecoder (Ronneberger et al., arXiv:1505.04597), add_center_block=False."""

    def __init__(self, enc_ch, dec_ch, attention="none"):
        super().__init__()
        enc = list(enc_ch[1:])[::-1]
        ins = [enc[0], *dec_ch[:-1]]
        skips = [*enc[1:], 0]
        self.blocks = nn.ModuleList(
            UnetBlock(i, s, o, attention) for i, s, o in zip(ins, skips, dec_ch, strict=True)
        )
        self.out_channels = dec_ch[-1]

    def forward(self, features):
        shapes = [f.shape[2:] for f in features][::-1]
        feats = features[1:][::-1]
        x, skips = feats[0], feats[1:]
        for i, b in enumerate(self.blocks):
            x = b(x, shapes[i + 1], skips[i] if i < len(skips) else None)
        return x


class UnetPPDecoder(nn.Module):
    """smp UnetPlusPlusDecoder (Zhou et al., arXiv:1807.10165), center=False; nested dense skips."""

    def __init__(self, enc_ch, dec_ch, attention="none"):
        super().__init__()
        enc = list(enc_ch[1:])[::-1]
        self.in_channels = [enc[0], *dec_ch[:-1]]
        self.skip_channels = [*enc[1:], 0]
        self.out_channels_ = list(dec_ch)
        blocks = {}
        for layer in range(len(self.in_channels) - 1):
            for depth in range(layer + 1):
                if depth == 0:
                    i = self.in_channels[layer]
                    s = self.skip_channels[layer] * (layer + 1)
                    o = self.out_channels_[layer]
                else:
                    o = self.skip_channels[layer]
                    s = self.skip_channels[layer] * (layer + 1 - depth)
                    i = self.skip_channels[layer - 1]
                blocks[f"x_{depth}_{layer}"] = UnetBlock(i, s, o, attention)
        n = len(self.in_channels) - 1
        blocks[f"x_0_{n}"] = UnetBlock(self.in_channels[-1], 0, self.out_channels_[-1], attention)
        self.blocks = nn.ModuleDict(blocks)
        self.depth = n
        self.out_channels = dec_ch[-1]

    def forward(self, features):
        size_in = features[0].shape[2:]
        feats = features[1:][::-1]
        dense = {}
        for layer in range(len(self.in_channels) - 1):
            for depth in range(self.depth - layer):
                if layer == 0:
                    dense[f"x_{depth}_{depth}"] = self.blocks[f"x_{depth}_{depth}"](
                        feats[depth], feats[depth + 1].shape[2:], feats[depth + 1]
                    )
                else:
                    li = depth + layer
                    cat = [dense[f"x_{k}_{li}"] for k in range(depth + 1, li + 1)]
                    cat = torch.cat([*cat, feats[li + 1]], 1)
                    dense[f"x_{depth}_{li}"] = self.blocks[f"x_{depth}_{li}"](
                        dense[f"x_{depth}_{li - 1}"], feats[li + 1].shape[2:], cat
                    )
        return self.blocks[f"x_0_{self.depth}"](dense[f"x_0_{self.depth - 1}"], size_in)


# -------------------------------------------------------------------------------- FPN


class GNReLU(nn.Module):
    """smp fpn.Conv3x3GNReLU; GroupNorm(32) becomes GroupNorm(gcd(32, channels)) for narrow maps."""

    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.GroupNorm(math.gcd(32, o), o), nn.ReLU()
        )

    def forward(self, x, size=None):
        x = self.block(x)
        return x if size is None else resize(x, size, "bilinear", True)


class FPNDecoder(nn.Module):
    """smp FPNDecoder (Kirillov et al., panoptic FPN, arXiv:1901.02446) on the last min(4, n) features."""

    def __init__(self, enc_ch, pyramid=256, seg=128, dropout=0.2, merge="add"):
        super().__init__()
        enc = list(enc_ch[1:])[::-1][:4]
        self.L = len(enc)
        self.p_top = nn.Conv2d(enc[0], pyramid, 1)
        self.skip_convs = nn.ModuleList(nn.Conv2d(c, pyramid, 1) for c in enc[1:])
        self.seg_blocks = nn.ModuleList(
            nn.ModuleList(
                [GNReLU(pyramid, seg)] + [GNReLU(seg, seg) for _ in range(1, self.L - 1 - j)]
            )
            for j in range(self.L)
        )
        self.merge, self.dropout = merge, nn.Dropout2d(dropout)
        self.out_channels = seg if merge == "add" else seg * self.L

    def forward(self, features):
        cs = features[1:][::-1][: self.L]
        ps = [self.p_top(cs[0])]
        for conv, c in zip(self.skip_convs, cs[1:], strict=True):
            ps.append(resize(ps[-1], c.shape[2:], "nearest") + conv(c))
        sizes = [p.shape[2:] for p in ps]
        outs = []
        for j, (blocks, p) in enumerate(zip(self.seg_blocks, ps, strict=True)):
            n_up = self.L - 1 - j
            x = p
            for u, b in enumerate(blocks):
                x = b(x, sizes[j + u + 1] if n_up else None)
            outs.append(x)
        x = torch.stack(outs).sum(0) if self.merge == "add" else torch.cat(outs, 1)
        return self.dropout(x)


# -------------------------------------------------------------------------- DeepLab


class SeparableConv2d(nn.Sequential):
    def __init__(self, i, o, k, stride=1, padding=0, dilation=1, bias=True):
        super().__init__(
            nn.Conv2d(i, i, k, stride=stride, padding=padding, dilation=dilation, groups=i, bias=False),
            nn.Conv2d(i, o, 1, bias=bias),
        )  # fmt: skip


class ASPPPooling(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for m in self:
            x = m(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """smp deeplabv3.ASPP (= torchvision segmentation.deeplabv3.ASPP plus a separable option)."""

    def __init__(self, i, o, rates, separable=False, dropout=0.5):
        super().__init__()

        def branch(r):
            conv = (
                SeparableConv2d(i, o, 3, padding=r, dilation=r, bias=False)
                if separable
                else nn.Conv2d(i, o, 3, padding=r, dilation=r, bias=False)
            )
            return nn.Sequential(conv, nn.BatchNorm2d(o), nn.ReLU())

        mods = [nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())]
        mods += [branch(r) for r in rates] + [ASPPPooling(i, o)]
        self.convs = nn.ModuleList(mods)
        self.project = nn.Sequential(
            nn.Conv2d((len(rates) + 2) * o, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Dropout(dropout),
        )  # fmt: skip

    def forward(self, x):
        return self.project(torch.cat([c(x) for c in self.convs], 1))


def atrous_rates(min_side: int) -> tuple[int, int, int]:
    """smp default (12, 24, 36) for maps of 48+ pixels; else 1/4, 1/2, 3/4 of the side (DLNet)."""
    if min_side >= 48:
        return (12, 24, 36)
    return tuple(max(1, int(min_side * f)) for f in (0.25, 0.5, 0.75))


class DeepLabV3Decoder(nn.Module):
    """smp DeepLabV3Decoder (Chen et al., arXiv:1706.05587): ASPP, then 3x3 conv-BN-ReLU."""

    def __init__(self, enc_ch, o, rates, separable=False, dropout=0.5):
        super().__init__()
        self.aspp = ASPP(enc_ch[-1], o, rates, separable, dropout)
        self.conv = nn.Conv2d(o, o, 3, padding=1, bias=False)
        self.bn, self.relu = nn.BatchNorm2d(o), nn.ReLU()
        self.out_channels = o

    def forward(self, features):
        return self.relu(self.bn(self.conv(self.aspp(features[-1]))))


class DeepLabV3PlusDecoder(nn.Module):
    """smp DeepLabV3PlusDecoder (Chen et al., arXiv:1802.02611): ASPP + 48-channel high-res skip.

    smp takes the skip from features[2] (stride 4); here the skip is features[skip_index]."""

    def __init__(self, enc_ch, o, rates, skip_index=2, separable=True, dropout=0.5):
        super().__init__()
        self.skip_index = skip_index
        self.aspp = nn.Sequential(
            ASPP(enc_ch[-1], o, rates, separable, dropout),
            SeparableConv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
        )  # fmt: skip
        hi = 48
        self.block1 = nn.Sequential(
            nn.Conv2d(enc_ch[skip_index], hi, 1, bias=False), nn.BatchNorm2d(hi), nn.ReLU()
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(hi + o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU()
        )
        self.out_channels = o

    def forward(self, features):
        hr = features[self.skip_index]
        a = resize(self.aspp(features[-1]), hr.shape[2:], "bilinear", True)
        return self.block2(torch.cat([a, self.block1(hr)], 1))


# ---------------------------------------------------------------------------- PSPNet


class PSPBlock(nn.Module):
    def __init__(self, i, o, pool_size):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
            Conv2dReLU(i, o, 1, use_norm=pool_size != 1),  # no BatchNorm on a 1x1 map (smp)
        )

    def forward(self, x):
        return resize(self.pool(x), x.shape[2:], "bilinear", True)


class PSPDecoder(nn.Module):
    """smp PSPDecoder (Zhao et al., arXiv:1612.01105): pyramid pooling (1, 2, 3, 6) on the deepest map."""

    def __init__(self, enc_ch, o=512, dropout=0.2, sizes=(1, 2, 3, 6)):
        super().__init__()
        c = enc_ch[-1]
        if c < len(sizes):
            raise ValueError("pspnet needs at least 4 channels in the deepest feature")
        self.blocks = nn.ModuleList(PSPBlock(c, c // len(sizes), s) for s in sizes)
        self.conv = Conv2dReLU(c + len(sizes) * (c // len(sizes)), o, 1)
        self.dropout = nn.Dropout2d(dropout)
        self.out_channels = o

    def forward(self, features):
        x = features[-1]
        x = torch.cat([b(x) for b in self.blocks] + [x], 1)
        return self.dropout(self.conv(x))


# --------------------------------------------------------------------------- LinkNet


class LinkBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        if i < 4:
            raise ValueError("linknet blocks need at least 4 input channels")
        q = i // 4
        self.block = nn.Sequential(
            Conv2dReLU(i, q, 1),
            nn.Sequential(nn.ConvTranspose2d(q, q, 4, 2, 1), nn.BatchNorm2d(q), nn.ReLU()),
            Conv2dReLU(q, o, 1),
        )

    def forward(self, x, size, skip=None):
        x = resize(self.block(x), size)
        return x if skip is None else x + skip


class LinknetDecoder(nn.Module):
    """smp LinknetDecoder (Chaurasia & Culurciello, arXiv:1707.03718): additive skips."""

    def __init__(self, enc_ch, prefinal=32):
        super().__init__()
        ch = [*list(enc_ch[1:])[::-1], prefinal]
        self.blocks = nn.ModuleList(LinkBlock(ch[i], ch[i + 1]) for i in range(len(ch) - 1))
        self.out_channels = prefinal

    def forward(self, features):
        size_in = features[0].shape[2:]
        feats = features[1:][::-1]
        x, skips = feats[0], feats[1:]
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, size_in if skip is None else skip.shape[2:], skip)
        return x


# --------------------------------------------------------------------------- UPerNet


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class UPerPSP(nn.Module):
    def __init__(self, i, o, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(s), Conv2dReLU(i, o, 1)) for s in sizes
        )
        self.out_conv = Conv2dReLU(i + len(sizes) * o, o, 3, padding=1)

    def forward(self, x):
        parts = [x] + [resize(b(x), x.shape[2:]) for b in self.blocks]
        return self.out_conv(torch.cat(parts, 1))


class UPerNetDecoder(nn.Module):
    """smp UPerNetDecoder (Xiao et al., arXiv:1807.10221): PPM on the deepest map + FPN fusion.

    smp drops the first two features (input and stride 2); with fewer than 3 encoder stages only
    the input is dropped."""

    def __init__(self, enc_ch, o=256):
        super().__init__()
        self.start = 2 if len(enc_ch) >= 4 else 1
        enc = list(enc_ch[self.start :])
        self.feature_norms = nn.ModuleList(LayerNorm2d(c, eps=1e-6) for c in enc)
        self.psp = UPerPSP(enc[-1], o)
        self.fpn_lateral_blocks = nn.ModuleList(Conv2dReLU(c, o, 1) for c in enc[:-1][::-1])
        self.fpn_conv_blocks = nn.ModuleList(Conv2dReLU(o, o, 3, padding=1) for _ in enc[:-1])
        self.fusion_block = Conv2dReLU(len(enc) * o, o, 3, padding=1)
        self.out_channels = o

    def forward(self, features):
        feats = [n(f) for n, f in zip(self.feature_norms, features[self.start :], strict=True)]
        fpn = [self.psp(feats[-1])]
        for lat, f in zip(self.fpn_lateral_blocks, feats[:-1][::-1], strict=True):
            f = lat(f)
            fpn.append(resize(fpn[-1], f.shape[2:]) + f)
        for i, conv in enumerate(self.fpn_conv_blocks, start=1):
            fpn[i] = conv(fpn[i])
        size = fpn[-1].shape[2:]
        return self.fusion_block(torch.cat([resize(f, size) for f in fpn][::-1], 1))


# ------------------------------------------------------------------------- SegFormer


class TokenMLP(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.linear = nn.Linear(i, o)

    def forward(self, x):
        b, _, h, w = x.shape
        return self.linear(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, -1, h, w)


class SegformerDecoder(nn.Module):
    """smp SegformerDecoder (Xie et al., arXiv:2105.15203): per-level linear, resize to 1/4, fuse."""

    def __init__(self, enc_ch, o=256):
        super().__init__()
        enc = list(enc_ch)[::-1]
        self.mlp_stage = nn.ModuleList(TokenMLP(c, o) for c in enc[:-1])
        self.fuse_stage = Conv2dReLU((len(enc) - 1) * o, o, 1)
        self.out_channels = o

    def forward(self, features):
        size = [max(1, d // 4) for d in features[0].shape[2:]]
        feats = features[1:][::-1]
        parts = [resize(m(f), size) for m, f in zip(self.mlp_stage, feats, strict=True)]
        return self.fuse_stage(torch.cat(parts, 1))


# ----------------------------------------------------------------- kept DLNet decoders


def cbr(i, o, k=3, d=1):
    from .ops_basic import ConvBNAct

    return ConvBNAct(2, i, o, k, 1, 1, d, "relu", "batch")


class FCNDecoder(nn.Module):
    """FCN-8s style: 1x1 score maps of every level, upsampled and added (from memory)."""

    def __init__(self, enc_ch, width):
        super().__init__()
        self.score = nn.ModuleList(nn.Conv2d(c, width, 1) for c in enc_ch[1:])
        self.out_channels = width

    def forward(self, features):
        feats = features[1:]
        y = self.score[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            y = resize(y, feats[i].shape[-2:]) + self.score[i](feats[i])
        return y


class DeconvDecoder(nn.Module):
    """rasbt ConvolutionalAutoencoder decoder: transposed convolutions, no skips."""

    def __init__(self, enc_ch):
        super().__init__()
        ch = list(enc_ch[1:])
        self.up = nn.ModuleList(
            nn.Sequential(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2), nn.BatchNorm2d(ch[i]), nn.LeakyReLU())
            for i in range(len(ch) - 1)
        )  # fmt: skip
        self.out_channels = ch[0]

    def forward(self, features):
        feats = features[1:]
        y = feats[-1]
        for i in range(len(feats) - 2, -1, -1):
            y = resize(self.up[i](y), feats[i].shape[-2:])
        return y


def build_decoder(dec, enc_ch, sizes, width, attention="none"):
    """Decoder `dec` for encoder channels [input, f0, ...] and spatial sizes [(h, w), ...]."""
    n = len(enc_ch) - 1
    # as smp: decoder block i has the channels of the skip it fuses (one level above the input)
    dec_ch = [max(int(width * 2.0 ** (n - 2 - i)), 4) for i in range(n)]
    if dec == "unet":
        return UnetDecoder(enc_ch, dec_ch, attention)
    if dec == "unetpp":
        return UnetPPDecoder(enc_ch, dec_ch, attention)
    if dec == "fpn":
        return FPNDecoder(enc_ch, pyramid=2 * width, seg=width)
    rates = atrous_rates(min(sizes[-1]))
    if dec == "deeplabv3":
        return DeepLabV3Decoder(enc_ch, width, rates)
    if dec == "deeplabv3plus":
        # the feature closest to stride 4 (smp: features[2]), never the deepest one
        target = max(1, min(sizes[0]) // 4)
        idx = min(range(1, max(n, 2)), key=lambda i: abs(min(sizes[i]) - target))
        return DeepLabV3PlusDecoder(enc_ch, width, rates, skip_index=idx)
    if dec == "pspnet":
        return PSPDecoder(enc_ch, width)
    if dec == "linknet":
        return LinknetDecoder(enc_ch, prefinal=width)
    if dec == "upernet":
        return UPerNetDecoder(enc_ch, width)
    if dec == "segformer":
        return SegformerDecoder(enc_ch, width)
    if dec == "fcn":
        return FCNDecoder(enc_ch, width)
    return DeconvDecoder(enc_ch)


class EncDec(nn.Module):
    """Encoder stages + one decoder; the output is resized to the input resolution."""

    def __init__(self, stages, feats, dec, width, in_dims, attention="none"):
        super().__init__()
        dec = ALIASES.get(dec, dec)
        if dec not in DECODERS:
            raise ValueError(f"unknown decoder {dec!r}; use one of {DECODERS}")
        if attention not in ATTENTIONS:
            raise ValueError(f"unknown decoder attention {attention!r}; use one of {ATTENTIONS}")
        if dec in ("unetpp", "deeplabv3plus") and len(stages) < 2:
            raise ValueError(f"{dec} needs at least 2 encoder stages")
        self.stages = nn.ModuleList(stages)
        enc_ch = [in_dims[0]] + [f[0] for f in feats]
        sizes = [tuple(in_dims[1:])] + [tuple(f[1:]) for f in feats]
        self.decoder = build_decoder(dec, enc_ch, sizes, width, attention)
        self.out_ch = self.decoder.out_channels

    def forward(self, x):
        feats = [x]
        for s in self.stages:
            feats.append(s(feats[-1]))
        return resize(self.decoder(feats), x.shape[-2:])
