"""v1.8 blocks for the search: efficient hybrid CNN / attention blocks of 2022-2024, ported from
timm (Apache-2.0, huggingface/pytorch-image-models v1.0.31) line by line, verified there:
scripts/verify_more.py compares each with the timm module (weights copied, outputs equal).

- pconv_block      FasterNet (Chen et al., CVPR 2023, arXiv:2303.03667): a 3x3 conv on 1/n of the
                   channels (partial convolution), then a 1x1 MLP; timm fasternet.MLPBlock.
- gated_cnn_block  MambaOut (Yu & Wang, CVPR 2025, arXiv:2405.07992): the Gated CNN block, a
                   depthwise conv inside a GLU-style gated MLP; timm mambaout.GatedConvBlock.
- mobilevit_block  MobileViT (Mehta & Rastegari, ICLR 2022, arXiv:2110.02178): local conv, a
                   transformer over unfolded patches, fold, fuse; timm mobilevit.MobileVitBlock.
- mobilevitv2_block  MobileViTv2 (Mehta & Rastegari, 2022, arXiv:2206.02680): separable
                   (linear-complexity) self-attention; timm mobilevit.MobileVitV2Block.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .core import op


class ConvNormAct(nn.Module):
    """timm ConvNormAct: conv without bias, BatchNorm, activation (SiLU / ReLU / none)."""

    def __init__(self, cin, cout, k=1, s=1, groups=1, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = {"silu": nn.SiLU(), "relu": nn.ReLU(), "none": nn.Identity()}[act]

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ----------------------------------------------------------------------------- FasterNet


class PConvBlock(nn.Module):
    """fasternet.MLPBlock with Partial_conv3 (split_cat): x + ls * MLP(PConv(x))."""

    def __init__(self, dim, n_div=4, mlp_ratio=2.0, layer_scale=0.0):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Conv2d(dim, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(),
                                 nn.Conv2d(hidden, dim, 1, bias=False))  # fmt: skip
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.layer_scale = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        y = torch.cat((self.partial_conv3(x1), x2), 1)
        y = self.mlp(y)
        if self.layer_scale is not None:
            y = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * y
        return x + y


@op("pconv_block", "map", space={"n_div": [4, 2], "mlp_ratio": [2.0, 1.0], "layer_scale": [0.0, 1e-5]},
    tags={"block", "conv", "efficient"}, source="FasterNet (arXiv:2303.03667); timm fasternet.MLPBlock (Apache-2.0)")  # fmt: skip
def _pconv_block(dims, kind, n_div, mlp_ratio, layer_scale):
    """FasterNet block: a 3x3 conv on 1/n_div of the channels only, then a 1x1 MLP, residual."""
    if dims[0] < n_div:
        raise ValueError(f"pconv_block: {dims[0]} channels < n_div {n_div}")
    return PConvBlock(dims[0], int(n_div), float(mlp_ratio), float(layer_scale))


# ----------------------------------------------------------------------------- MambaOut


class GatedCNNBlock(nn.Module):
    """mambaout.GatedConvBlock on a (C, H, W) map (the timm block works channels-last)."""

    def __init__(self, dim, expansion=8 / 3, kernel=7, conv_ratio=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(expansion * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.GELU()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(
            conv_channels, conv_channels, kernel, padding=kernel // 2, groups=conv_channels
        )
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C], as the timm stage holds it
        shortcut = x
        x = self.fc1(self.norm(x))
        g, i, c = torch.split(x, self.split_indices, dim=-1)
        c = self.conv(c.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        return (x + shortcut).permute(0, 3, 1, 2)


@op("gated_cnn_block", "map", space={"expansion": [8 / 3, 2.0], "kernel": [7, 5, 3], "conv_ratio": [1.0, 0.5]},
    tags={"block", "conv", "gated"}, source="MambaOut (arXiv:2405.07992); timm mambaout.GatedConvBlock (Apache-2.0)")  # fmt: skip
def _gated_cnn_block(dims, kind, expansion, kernel, conv_ratio):
    """MambaOut's Gated CNN block: LayerNorm, linear to 2 x hidden, a depthwise conv on part of it,
    gate (GELU), linear back, residual."""
    hidden = int(float(expansion) * dims[0])
    if int(float(conv_ratio) * dims[0]) > hidden:
        raise ValueError("gated_cnn_block: conv channels exceed the hidden width")
    return GatedCNNBlock(dims[0], float(expansion), int(kernel), float(conv_ratio))


# ----------------------------------------------------------------------------- MobileViT


class ViTBlock(nn.Module):
    """timm vision_transformer.Block (pre-norm attention + MLP), qkv bias, no layer scale."""

    def __init__(self, dim, heads, mlp_ratio=2.0, act=nn.SiLU):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), act(), nn.Linear(hidden, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .reshape(b, n, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(a)
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class MobileViTBlock(nn.Module):
    """mobilevit.MobileVitBlock (v1, stride 1, with fusion)."""

    def __init__(self, cin, tdim, depth=2, patch=2, heads=4, mlp_ratio=2.0, kernel=3):
        super().__init__()
        self.conv_kxk = ConvNormAct(cin, cin, kernel)
        self.conv_1x1 = nn.Conv2d(cin, tdim, 1, bias=False)
        self.transformer = nn.Sequential(*[ViTBlock(tdim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(tdim)
        self.conv_proj = ConvNormAct(tdim, cin, 1)
        self.conv_fusion = ConvNormAct(cin + cin, cin, kernel)
        self.patch = (patch, patch)
        self.patch_area = patch * patch

    def forward(self, x):
        shortcut = x
        x = self.conv_1x1(self.conv_kxk(x))
        ph, pw = self.patch
        b, c, h, w = x.shape
        nh_, nw_ = math.ceil(h / ph) * ph, math.ceil(w / pw) * pw
        nph, npw = nh_ // ph, nw_ // pw
        n = nph * npw
        interp = nh_ != h or nw_ != w
        if interp:
            x = F.interpolate(x, size=(nh_, nw_), mode="bilinear", align_corners=False)
        x = x.reshape(b * c * nph, ph, npw, pw).transpose(1, 2)
        x = x.reshape(b, c, n, self.patch_area).transpose(1, 3).reshape(b * self.patch_area, n, -1)
        x = self.norm(self.transformer(x))
        x = x.contiguous().view(b, self.patch_area, n, -1)
        x = x.transpose(1, 3).reshape(b * c * nph, npw, ph, pw)
        x = x.transpose(1, 2).reshape(b, c, nph * ph, npw * pw)
        if interp:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = self.conv_proj(x)
        return self.conv_fusion(torch.cat((shortcut, x), dim=1))


@op("mobilevit_block", "map", space={"tdim": [0, 64, 96], "depth": [2, 1, 3], "patch": [2], "heads": [4, 2],
                                     "mlp_ratio": [2.0]},
    tags={"block", "attn", "hybrid"}, source="MobileViT (arXiv:2110.02178); timm mobilevit.MobileVitBlock (Apache-2.0)")  # fmt: skip
def _mobilevit_block(dims, kind, tdim, depth, patch, heads, mlp_ratio):
    """MobileViT block: local 3x3 conv, a transformer over every patch position (global context),
    fold back, fuse with the input. tdim 0 = the input width."""
    t = int(tdim) or dims[0]
    if t % int(heads):
        raise ValueError(f"mobilevit_block: width {t} not divisible by {heads} heads")
    return MobileViTBlock(dims[0], t, int(depth), int(patch), int(heads), float(mlp_ratio))


class LinearSelfAttention(nn.Module):
    """mobilevit.LinearSelfAttention (self-attention, linear in the number of patches)."""

    def __init__(self, dim):
        super().__init__()
        self.embed_dim = dim
        self.qkv_proj = nn.Conv2d(dim, 1 + 2 * dim, 1, bias=True)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x):
        q, k, v = self.qkv_proj(x).split([1, self.embed_dim, self.embed_dim], dim=1)
        ctx = (k * F.softmax(q, dim=-1)).sum(dim=-1, keepdim=True)
        return self.out_proj(F.relu(v) * ctx.expand_as(v))


class LinearTransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = LinearSelfAttention(dim)
        self.norm2 = nn.GroupNorm(1, dim)
        hidden = int(dim * mlp_ratio)
        self.fc1, self.act, self.fc2 = (
            nn.Conv2d(dim, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class MobileViTv2Block(nn.Module):
    """mobilevit.MobileVitV2Block (depthwise local conv, separable attention, no fusion)."""

    def __init__(self, cin, tdim, depth=2, patch=2, mlp_ratio=2.0, kernel=3):
        super().__init__()
        self.conv_kxk = ConvNormAct(cin, cin, kernel, groups=cin)
        self.conv_1x1 = nn.Conv2d(cin, tdim, 1, bias=False)
        self.transformer = nn.Sequential(
            *[LinearTransformerBlock(tdim, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.GroupNorm(1, tdim)
        self.conv_proj = ConvNormAct(tdim, cin, 1, act="none")
        self.patch = (patch, patch)

    def forward(self, x):
        b, c, h, w = x.shape
        ph, pw = self.patch
        nh_, nw_ = math.ceil(h / ph) * ph, math.ceil(w / pw) * pw
        nph, npw = nh_ // ph, nw_ // pw
        n = nph * npw
        if nh_ != h or nw_ != w:
            x = F.interpolate(x, size=(nh_, nw_), mode="bilinear", align_corners=True)
        x = self.conv_1x1(self.conv_kxk(x))
        c = x.shape[1]
        x = x.reshape(b, c, nph, ph, npw, pw).permute(0, 1, 3, 5, 2, 4).reshape(b, c, -1, n)
        x = self.norm(self.transformer(x))
        x = (
            x.reshape(b, c, ph, pw, nph, npw)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(b, c, nph * ph, npw * pw)
        )
        return self.conv_proj(x)


@op("mobilevitv2_block", "map", space={"tdim": [0, 64], "depth": [2, 1, 3], "patch": [2], "mlp_ratio": [2.0]},
    tags={"block", "attn", "hybrid", "efficient"},
    source="MobileViTv2 (arXiv:2206.02680); timm mobilevit.MobileVitV2Block (Apache-2.0)")  # fmt: skip
def _mobilevitv2_block(dims, kind, tdim, depth, patch, mlp_ratio):
    """MobileViTv2 block: depthwise local conv, separable self-attention (linear cost) over patch
    positions, fold, 1x1 projection. tdim 0 = the input width. Output size = input size rounded up to
    the patch (as timm)."""
    return MobileViTv2Block(dims[0], int(tdim) or dims[0], int(depth), int(patch), float(mlp_ratio))


# ----------------------------------------------------------------------------- EdgeNeXt


class FourierPosEnc(nn.Module):
    """edgenext.PositionalEncodingFourier: sine / cosine of the normalised (y, x) position, 1x1 conv."""

    def __init__(self, dim, hidden_dim=32, temperature=10000.0):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale, self.temperature, self.hidden_dim = 2 * math.pi, temperature, hidden_dim

    def forward(self, b, h, w):
        dev = self.token_projection.weight.device
        inv = ~torch.zeros((b, h, w), device=dev, dtype=torch.bool)
        y = inv.cumsum(1, dtype=torch.float32)
        x = inv.cumsum(2, dtype=torch.float32)
        y = y / (y[:, -1:, :] + 1e-6) * self.scale
        x = x / (x[:, :, -1:] + 1e-6) * self.scale
        t = torch.arange(self.hidden_dim, dtype=torch.int64, device=dev).to(torch.float32)
        t = self.temperature ** (2 * torch.div(t, 2, rounding_mode="floor") / self.hidden_dim)
        px, py = x[:, :, :, None] / t, y[:, :, :, None] / t
        px = torch.stack((px[:, :, :, 0::2].sin(), px[:, :, :, 1::2].cos()), dim=4).flatten(3)
        py = torch.stack((py[:, :, :, 0::2].sin(), py[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((py, px), dim=3).permute(0, 3, 1, 2)
        return self.token_projection(pos.to(self.token_projection.weight.dtype))


class XCA(nn.Module):
    """edgenext.CrossCovarianceAttn: attention between channels (C x C), linear in the pixels."""

    def __init__(self, dim, heads=8, qkv_bias=True):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, -1).permute(2, 0, 3, 4, 1).unbind(0)
        attn = (
            F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        ) * self.temperature
        x = (attn.softmax(dim=-1) @ v).permute(0, 3, 1, 2).reshape(b, n, c)
        return self.proj(x)


class SDTABlock(nn.Module):
    """edgenext.SplitTransposeBlock: multi-scale depthwise convs on channel splits, Fourier position
    encoding, cross-covariance attention, inverted-bottleneck MLP (channels-last), layer scales."""

    def __init__(self, dim, scales=2, heads=8, expand=4.0, pos_emb=True, ls=1e-6):
        super().__init__()
        width = max(math.ceil(dim / scales), dim // scales)
        self.width, self.num_scales = width, max(1, scales - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(width, width, 3, padding=1, groups=width) for _ in range(self.num_scales)]
        )
        self.pos_embd = FourierPosEnc(dim) if pos_emb else None
        self.norm_xca = nn.LayerNorm(dim, eps=1e-6)
        self.gamma_xca = nn.Parameter(ls * torch.ones(dim))
        self.xca = XCA(dim, heads)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(expand * dim)
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(ls * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        spx = x.chunk(len(self.convs) + 1, dim=1)
        spo, sp = [], spx[0]
        for i, conv in enumerate(self.convs):
            if i > 0:
                sp = sp + spx[i]
            sp = conv(sp)
            spo.append(sp)
        spo.append(spx[-1])
        x = torch.cat(spo, 1)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        if self.pos_embd is not None:
            x = x + self.pos_embd(b, h, w).reshape(b, -1, x.shape[1]).permute(0, 2, 1)
        x = x + self.gamma_xca * self.xca(self.norm_xca(x))
        x = x.reshape(b, h, w, c)
        x = self.gamma * self.fc2(self.act(self.fc1(self.norm(x))))
        return shortcut + x.permute(0, 3, 1, 2)


@op("sdta_block", "map", space={"scales": [2, 3, 4], "heads": [4, 8], "expand": [4.0, 2.0], "pos_emb": [True, False],
                                "ls": [1e-6]},
    tags={"block", "attn", "hybrid", "efficient"},
    source="EdgeNeXt (arXiv:2206.10589); timm edgenext.SplitTransposeBlock (Apache-2.0)")  # fmt: skip
def _sdta_block(dims, kind, scales, heads, expand, pos_emb, ls):
    """EdgeNeXt SDTA block: depthwise convs over channel splits (growing receptive field), then
    transposed (channel) attention, then an MLP. Cost linear in the number of pixels."""
    c = dims[0]
    if c % int(heads) or c < int(scales):
        raise ValueError(f"sdta_block: {c} channels, {heads} heads, {scales} scales")
    return SDTABlock(c, int(scales), int(heads), float(expand), bool(pos_emb), float(ls))
