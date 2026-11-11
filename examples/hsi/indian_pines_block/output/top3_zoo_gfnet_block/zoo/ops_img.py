"""Image blocks ported from the reference vision libraries (torch only, no runtime dependency).

Sources (versions and classes in THIRD_PARTY.md; numerical equivalence with weights copied
across in scripts/verify_numeric.py):
  huggingface/pytorch-image-models (timm 1.0.31)
      layers.SelectiveKernel (SKNet), layers.SplitAttn (ResNeSt), layers.GlobalContext (GCNet),
      layers.CoordAttn, layers.GlobalResponseNorm (ConvNeXt-V2), models.ghostnet.GhostBottleneck,
      models.res2net.Bottle2neck, models.convmixer block, models.byobnet.RepVggBlock,
      models.metaformer.MetaFormerBlock with Pooling (PoolFormer)
  qubvel-org/segmentation_models_pytorch (0.6.0)   base.modules.SCSEModule
  Project-MONAI/MONAI (1.6.1)   networks.blocks.segresnet_block.ResBlock,
                                networks.blocks.dynunet_block.UnetResBlock
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .core import ND, op
from .ops_basic import BN, CONV, SP


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """timm.layers.helpers.make_divisible."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of an NCHW map (timm.layers.LayerNorm2d)."""

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


def conv_bn(nd, i, o, k, s=1, d=1, groups=1, act=True):
    """timm ConvNormAct / byobnet conv_norm_act: conv (no bias, 'same' padding), BN, optional ReLU."""
    layers = [
        CONV[nd](i, o, k, s, d * (k - 1) // 2, dilation=d, groups=groups, bias=False),
        BN[nd](o),
    ]
    return nn.Sequential(*layers, *([nn.ReLU()] if act else []))


# ----------------------------------------------------------------------- attention


from .dense import SCSE  # smp module, also used by the U-Net decoders


@op("scse", "map", space={"reduction": [16, 8, 4]}, tags={"attn"},
    source="qubvel-org/segmentation_models_pytorch base.modules.SCSEModule (Roy et al., arXiv:1803.02579)")  # fmt: skip
def _scse(dims, kind, reduction):
    return SCSE(dims[0], reduction)


class CoordAttn(nn.Module):
    """timm layers.CoordAttn (Hou et al., arXiv:2103.02907): pooled along H and W separately."""

    def __init__(self, c, rd_ratio=1 / 16, se_factor=2 / 3):
        super().__init__()
        rd = make_divisible(c * rd_ratio * se_factor, 8, round_limit=0.0)
        self.conv1 = nn.Conv2d(c, rd, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(rd)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(rd, c, 1, bias=False)
        self.conv_w = nn.Conv2d(rd, c, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[2:]
        y = torch.cat([x.mean(3, keepdim=True), x.mean(2, keepdim=True).transpose(-1, -2)], 2)
        y = self.act(self.bn1(self.conv1(y)))
        a_h, a_w = torch.split(y, [h, w], 2)
        return (
            x * torch.sigmoid(self.conv_w(a_w.transpose(-1, -2))) * torch.sigmoid(self.conv_h(a_h))
        )


@op("coord_attn", "map", space={"rd_ratio": [0.0625, 0.125, 0.25]}, tags={"attn"},
    source="huggingface/pytorch-image-models layers.CoordAttn")  # fmt: skip
def _coord(dims, kind, rd_ratio):
    return CoordAttn(dims[0], rd_ratio)


class ConvMlp(nn.Module):
    """timm layers.ConvMlp: 1x1 conv, norm, act, 1x1 conv."""

    def __init__(self, i, h, norm=None):
        super().__init__()
        self.fc1 = nn.Conv2d(i, h, 1)
        self.norm = norm(h) if norm else nn.Identity()
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.0)
        self.fc2 = nn.Conv2d(h, i, 1)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.norm(self.fc1(x)))))


class GlobalContext(nn.Module):
    """timm layers.GlobalContext (Cao et al., GCNet, arXiv:1904.11492), fuse_scale=True, fuse_add=False."""

    def __init__(self, c, rd_ratio=1 / 8):
        super().__init__()
        self.conv_attn = nn.Conv2d(c, 1, 1, bias=True)
        self.mlp_scale = ConvMlp(c, make_divisible(c * rd_ratio, 1, round_limit=0.0), LayerNorm2d)
        nn.init.kaiming_normal_(self.conv_attn.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        b, c, h, w = x.shape
        attn = F.softmax(self.conv_attn(x).reshape(b, 1, h * w), -1).unsqueeze(3)
        ctx = (x.reshape(b, c, h * w).unsqueeze(1) @ attn).view(b, c, 1, 1)
        return x * torch.sigmoid(self.mlp_scale(ctx))


@op("gc_block", "map", space={"rd_ratio": [0.125, 0.25, 0.0625]}, tags={"attn"},
    source="huggingface/pytorch-image-models layers.GlobalContext (GCNet)")  # fmt: skip
def _gc(dims, kind, rd_ratio):
    return GlobalContext(dims[0], rd_ratio)


class GRN(nn.Module):
    """timm layers.GlobalResponseNorm (ConvNeXt-V2, Woo et al., arXiv:2301.00808), channels first."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        sp = tuple(range(2, x.dim()))
        g = x.norm(p=2, dim=sp, keepdim=True)
        n = g / (g.mean(dim=1, keepdim=True) + self.eps)
        shape = (1, -1) + (1,) * len(sp)
        return x + torch.addcmul(self.bias.view(shape), self.weight.view(shape), x * n)


@op("grn", SP, space={}, tags={"attn"},
    source="huggingface/pytorch-image-models layers.GlobalResponseNorm (channels-first form)")  # fmt: skip
def _grn(dims, kind):
    return GRN(dims[0])


# --------------------------------------------------------------------------- blocks


class SKAttn(nn.Module):
    def __init__(self, c, paths, attn_ch):
        super().__init__()
        self.paths = paths
        self.fc_reduce = nn.Conv2d(c, attn_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(attn_ch)
        self.act = nn.ReLU()
        self.fc_select = nn.Conv2d(attn_ch, c * paths, 1, bias=False)

    def forward(self, x):
        x = self.act(self.bn(self.fc_reduce(x.sum(1).mean((2, 3), keepdim=True))))
        x = self.fc_select(x)
        b, c, h, w = x.shape
        return torch.softmax(x.view(b, self.paths, c // self.paths, h, w), 1)


class SelectiveKernel(nn.Module):
    """timm layers.SelectiveKernel (Li et al., SKNet, arXiv:1903.06586): 3x3 paths with dilation 1 and 2
    (keep_3x3), soft attention over paths, split_input=True."""

    def __init__(self, i, o, s=1, kernels=(3, 5), rd_ratio=1 / 16):
        super().__init__()
        dil = [(k - 1) // 2 for k in kernels]
        self.n = len(kernels)
        if i % self.n:
            raise ValueError("selective kernel splits the input channels between the paths")
        self.in_ch = i
        self.paths = nn.ModuleList(conv_bn(2, i // self.n, o, 3, s, d) for d in dil)
        self.attn = SKAttn(o, self.n, make_divisible(o * rd_ratio, 8))

    def forward(self, x):
        xs = torch.split(x, self.in_ch // self.n, 1)
        x = torch.stack([p(xs[i]) for i, p in enumerate(self.paths)], 1)
        return (x * self.attn(x)).sum(1)


@op("sk_conv", "map", space={"out": [32, 64, 128, 16], "s": [1, 2]}, tags={"block", "down"},
    source="huggingface/pytorch-image-models layers.SelectiveKernel (SKNet)")  # fmt: skip
def _sk(dims, kind, out, s):
    return SelectiveKernel(dims[0], out, s)


class SplitAttn(nn.Module):
    """timm layers.SplitAttn (Zhang et al., ResNeSt, arXiv:2004.08955) with BatchNorm, groups = 1."""

    def __init__(self, i, o, s=1, radix=2, rd_ratio=0.25):
        super().__init__()
        self.radix = radix
        mid = o * radix
        attn = make_divisible(i * radix * rd_ratio, min_value=32, divisor=8)
        self.conv = nn.Conv2d(i, mid, 3, s, 1, groups=radix, bias=False)
        self.bn0 = nn.BatchNorm2d(mid)
        self.act0 = nn.ReLU()
        self.fc1 = nn.Conv2d(o, attn, 1)
        self.bn1 = nn.BatchNorm2d(attn)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Conv2d(attn, mid, 1)

    def forward(self, x):
        x = self.act0(self.bn0(self.conv(x)))
        b, rc, h, w = x.shape
        if self.radix > 1:
            x = x.reshape(b, self.radix, rc // self.radix, h, w)
            gap = x.sum(1)
        else:
            gap = x
        a = self.fc2(self.act1(self.bn1(self.fc1(gap.mean((2, 3), keepdim=True)))))
        if self.radix > 1:
            a = F.softmax(a.view(b, 1, self.radix, -1).transpose(1, 2), 1).reshape(b, -1)
            return (x * a.reshape(b, self.radix, rc // self.radix, 1, 1)).sum(1)
        return x * torch.sigmoid(a).view(b, -1, 1, 1)


@op("split_attn", "map", space={"out": [32, 64, 128, 16], "s": [1, 2], "radix": [2, 4, 1]},
    tags={"block", "down"}, source="huggingface/pytorch-image-models layers.SplitAttn (ResNeSt)")  # fmt: skip
def _split(dims, kind, out, s, radix):
    return SplitAttn(dims[0], out, s, radix)


class GhostModule(nn.Module):
    def __init__(self, nd, i, o, act=True, ratio=2, dw=3):
        super().__init__()
        self.o = o
        init = math.ceil(o / ratio)
        new = init * (ratio - 1)
        a = nn.ReLU if act else nn.Identity
        self.primary_conv = nn.Sequential(CONV[nd](i, init, 1, 1, 0, bias=False), BN[nd](init), a())
        self.cheap_operation = nn.Sequential(
            CONV[nd](init, new, dw, 1, dw // 2, groups=init, bias=False), BN[nd](new), a()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        return torch.cat([x1, self.cheap_operation(x1)], 1)[:, : self.o]


class GhostBottleneck(nn.Module):
    """timm models.ghostnet.GhostBottleneck (Han et al., GhostNet, arXiv:1911.11907), no SE."""

    def __init__(self, nd, i, mid, o, k=3, s=1):
        super().__init__()
        self.ghost1 = GhostModule(nd, i, mid, True)
        if s > 1:
            self.conv_dw = CONV[nd](mid, mid, k, s, (k - 1) // 2, groups=mid, bias=False)
            self.bn_dw = BN[nd](mid)
        else:
            self.conv_dw = self.bn_dw = None
        self.ghost2 = GhostModule(nd, mid, o, False)
        if i == o and s == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                CONV[nd](i, i, k, s, (k - 1) // 2, groups=i, bias=False), BN[nd](i),
                CONV[nd](i, o, 1, bias=False), BN[nd](o),
            )  # fmt: skip

    def forward(self, x):
        y = self.ghost1(x)
        if self.conv_dw is not None:
            y = self.bn_dw(self.conv_dw(y))
        return self.ghost2(y) + self.shortcut(x)


@op("ghost_bottleneck", SP, space={"out": [16, 32, 64, 128], "expand": [2, 3, 6], "s": [1, 2], "k": [3, 5]},
    tags={"block", "down"}, source="huggingface/pytorch-image-models models.ghostnet.GhostBottleneck")  # fmt: skip
def _ghost(dims, kind, out, expand, s, k):
    return GhostBottleneck(ND[kind], dims[0], dims[0] * expand, out, k, s)


class Bottle2neck(nn.Module):
    """timm models.res2net.Bottle2neck (Gao et al., Res2Net, arXiv:1904.01169) with the ResNet
    projection shortcut (1x1 conv + BN) when the shape changes; output = planes * 4."""

    def __init__(self, i, planes, s=1, base_width=26, scale=4):
        super().__init__()
        self.scale = scale
        self.is_first = s > 1 or i != planes * 4
        self.num_scales = max(1, scale - 1)
        width = math.floor(planes * (base_width / 64.0))
        self.width = width
        o = planes * 4
        self.conv1 = nn.Conv2d(i, width * scale, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.convs = nn.ModuleList(
            nn.Conv2d(width, width, 3, s, 1, bias=False) for _ in range(self.num_scales)
        )
        self.bns = nn.ModuleList(nn.BatchNorm2d(width) for _ in range(self.num_scales))
        self.pool = nn.AvgPool2d(3, s, 1) if self.is_first else None
        self.conv3 = nn.Conv2d(width * scale, o, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(o)
        self.relu = nn.ReLU()
        self.downsample = (
            nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o))
            if self.is_first
            else None
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        spo, sp = [], spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns, strict=True)):
            sp = spx[i] if i == 0 or self.is_first else sp + spx[i]
            sp = self.relu(bn(conv(sp)))
            spo.append(sp)
        if self.scale > 1:
            spo.append(self.pool(spx[-1]) if self.pool is not None else spx[-1])
        out = self.bn3(self.conv3(torch.cat(spo, 1)))
        short = x if self.downsample is None else self.downsample(x)
        return self.relu(out + short)


@op("res2net_block", "map", space={"out": [64, 128, 256, 32], "s": [1, 2], "scale": [4, 2, 6], "base_width": [26, 14, 48]},
    tags={"block", "down"}, source="huggingface/pytorch-image-models models.res2net.Bottle2neck")  # fmt: skip
def _res2net(dims, kind, out, s, scale, base_width):
    if out % 4 or int(out // 4 * base_width / 64) < 1:
        raise ValueError("res2net needs out divisible by 4 and a positive split width")
    return Bottle2neck(dims[0], out // 4, s, base_width, scale)


class ConvMixerBlock(nn.Sequential):
    """timm models.convmixer block (Trockman & Kolter, arXiv:2201.09792): residual depthwise conv,
    GELU, BN; then pointwise conv, GELU, BN. Shape preserving."""

    class _Res(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x):
            return self.fn(x) + x

    def __init__(self, nd, c, k):
        super().__init__(
            self._Res(nn.Sequential(CONV[nd](c, c, k, groups=c, padding="same"), nn.GELU(), BN[nd](c))),
            CONV[nd](c, c, 1), nn.GELU(), BN[nd](c),
        )  # fmt: skip


@op("convmixer_block", SP, space={"k": [9, 7, 5, 3]}, tags={"block"},
    source="huggingface/pytorch-image-models models.convmixer (ConvMixer block)")  # fmt: skip
def _convmixer(dims, kind, k):
    return ConvMixerBlock(ND[kind], dims[0], k)


class RepVGGBlock(nn.Module):
    """timm models.byobnet.RepVggBlock (Ding et al., RepVGG, arXiv:2101.03697), groups = 1.

    Training form: BN identity (when shapes allow) + 3x3 conv-BN + 1x1 conv-BN, then ReLU.
    `fuse()` re-parameterises the three branches into one 3x3 convolution (eval mode, same output).
    """

    def __init__(self, nd, i, o, s=1):
        super().__init__()
        self.nd = nd
        self.identity = BN[nd](o) if i == o and s == 1 else None
        self.conv_kxk = conv_bn(nd, i, o, 3, s, act=False)
        self.conv_1x1 = conv_bn(nd, i, o, 1, s, act=False)
        self.act = nn.ReLU()
        self.fused = None

    def forward(self, x):
        if self.fused is not None:
            return self.act(self.fused(x))
        y = self.conv_1x1(x) + self.conv_kxk(x)
        return self.act(y if self.identity is None else y + self.identity(x))

    @torch.no_grad()
    def fuse(self):
        def fold(w, bn):
            std = (bn.running_var + bn.eps).sqrt()
            t = (bn.weight / std).reshape(-1, *[1] * (w.dim() - 1))
            return w * t, bn.bias - bn.running_mean * bn.weight / std

        conv = self.conv_kxk[0]
        w3, b3 = fold(conv.weight, self.conv_kxk[1])
        w1, b1 = fold(self.conv_1x1[0].weight, self.conv_1x1[1])
        w = w3 + F.pad(w1, [1, 1] * self.nd)
        b = b3 + b1
        if self.identity is not None:
            eye = torch.zeros_like(w3)
            for c in range(eye.shape[0]):
                eye[(c, c) + (1,) * self.nd] = 1.0
            wi, bi = fold(eye, self.identity)
            w, b = w + wi, b + bi
        self.fused = CONV[self.nd](conv.in_channels, conv.out_channels, 3, conv.stride, 1)
        self.fused.weight.copy_(w)
        self.fused.bias.copy_(b)
        return self


@op("repvgg_block", SP, space={"out": [16, 32, 64, 128], "s": [1, 2]}, tags={"block", "down"},
    source="huggingface/pytorch-image-models models.byobnet.RepVggBlock")  # fmt: skip
def _repvgg(dims, kind, out, s):
    return RepVGGBlock(ND[kind], dims[0], out, s)


class Scale(nn.Module):
    def __init__(self, c, init=1e-5):
        super().__init__()
        self.scale = nn.Parameter(init * torch.ones(c))

    def forward(self, x):
        return x * self.scale.view(1, -1, *[1] * (x.dim() - 2))


class PoolFormerBlock(nn.Module):
    """timm models.metaformer.MetaFormerBlock as in poolformer_v1 (Yu et al., MetaFormer,
    arXiv:2111.11418): GroupNorm(1), average-pool token mixer (pool - x), 1x1-conv MLP with GELU,
    layer scale 1e-5 on both branches."""

    def __init__(self, nd, c, pool=3, ratio=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, c)
        self.pool = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}[nd](
            pool, 1, pool // 2, count_include_pad=False
        )
        self.layer_scale1 = Scale(c)
        self.norm2 = nn.GroupNorm(1, c)
        self.fc1, self.act, self.fc2 = (
            CONV[nd](c, c * ratio, 1),
            nn.GELU(),
            CONV[nd](c * ratio, c, 1),
        )
        self.layer_scale2 = Scale(c)

    def forward(self, x):
        z = self.norm1(x)
        x = x + self.layer_scale1(self.pool(z) - z)
        return x + self.layer_scale2(self.fc2(self.act(self.fc1(self.norm2(x)))))


@op("poolformer_block", SP, space={"pool": [3, 5], "ratio": [4, 2]}, tags={"block"},
    source="huggingface/pytorch-image-models models.metaformer (PoolFormer v1 block)")  # fmt: skip
def _poolformer(dims, kind, pool, ratio):
    return PoolFormerBlock(ND[kind], dims[0], pool, ratio)


# ---------------------------------------------------------------------- MONAI blocks


class SegResBlock(nn.Module):
    """MONAI networks.blocks.segresnet_block.ResBlock (Myronenko, arXiv:1810.11654):
    GN-ReLU-conv-GN-ReLU-conv + identity; GroupNorm(8) becomes GroupNorm(gcd(8, channels))."""

    def __init__(self, nd, c, k=3):
        super().__init__()
        g = math.gcd(8, c)
        self.norm1, self.norm2 = nn.GroupNorm(g, c), nn.GroupNorm(g, c)
        self.act = nn.ReLU()
        self.conv1 = CONV[nd](c, c, k, 1, k // 2, bias=False)
        self.conv2 = CONV[nd](c, c, k, 1, k // 2, bias=False)

    def forward(self, x):
        y = self.conv1(self.act(self.norm1(x)))
        return self.conv2(self.act(self.norm2(y))) + x


@op("segres_block", SP, space={"k": [3, 5]}, tags={"block"},
    source="Project-MONAI/MONAI networks.blocks.segresnet_block.ResBlock")  # fmt: skip
def _segres(dims, kind, k):
    return SegResBlock(ND[kind], dims[0], k)


INORM = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}


class UnetResBlock(nn.Module):
    """MONAI networks.blocks.dynunet_block.UnetResBlock (nnU-Net residual block, Isensee et al.,
    arXiv:1809.10486): conv-IN-LeakyReLU-conv-IN + (1x1 conv-IN) shortcut, LeakyReLU(0.01);
    InstanceNorm without affine parameters, as MONAI."""

    def __init__(self, nd, i, o, k=3, s=1):
        super().__init__()
        self.conv1 = CONV[nd](i, o, k, s, k // 2, bias=False)
        self.conv2 = CONV[nd](o, o, k, 1, k // 2, bias=False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.norm1, self.norm2 = INORM[nd](o), INORM[nd](o)  # MONAI: affine=False
        self.down = i != o or s != 1
        if self.down:
            self.conv3 = CONV[nd](i, o, 1, s, bias=False)
            self.norm3 = INORM[nd](o)

    def forward(self, x):
        y = self.norm2(self.conv2(self.lrelu(self.norm1(self.conv1(x)))))
        r = self.norm3(self.conv3(x)) if self.down else x
        return self.lrelu(y + r)


@op("unet_res_block", SP, space={"out": [16, 32, 64, 128], "k": [3, 5], "s": [1, 2]}, tags={"block", "down"},
    source="Project-MONAI/MONAI networks.blocks.dynunet_block.UnetResBlock")  # fmt: skip
def _unet_res(dims, kind, out, k, s):
    return UnetResBlock(ND[kind], dims[0], out, k, s)
