"""Frequency-domain modules: FFT -> learned operation on the spectrum -> inverse FFT (torch.fft).

Written from the OFFICIAL code (pinned commits in THIRD_PARTY.md; numerical check with weights
copied across: scripts/verify_freq.py, run on GitHub Actions by .github/workflows/verify-modern.yml):

- GFNet global filter (Rao et al., NeurIPS 2021, arXiv:2107.00645; raoyongming/GFNet gfnet.py
  GlobalFilter, Block): X = rFFT(x); X * K (a learned complex filter, one per frequency and channel);
  iFFT. A depthwise global circular convolution with a kernel as large as the input, O(N log N).
- AFNO (Guibas et al., ICLR 2022, arXiv:2111.13587; NVlabs/AFNO-transformer afno/afno2d.py AFNO2D,
  classification/afnonet.py Block): a block-diagonal two-layer complex MLP shared by all
  frequencies, soft-shrinkage for sparsity, then iFFT.
- FNO spectral convolution (Li et al., ICLR 2021, arXiv:2010.08895, eq. 4-5): the lowest `modes`
  frequencies are multiplied by a learned complex matrix (in -> out channels), the others set to 0.
  The original fourier_2d.py of zongyi-li/fourier_neural_operator is no longer in that repository
  (it became neuraloperator, whose SpectralConv uses tensorly): written from the paper and the
  well-known original file, CHECKED by closed-form tests only (tests/test_freq.py).
- FFC, Fast Fourier Convolution (Chi et al., NeurIPS 2020; pkumivision/FFC model_zoo/ffc.py,
  ffc_resnet.py BasicBlock): a local branch (3x3 conv) and a global branch (real and imaginary
  parts stacked as channels -> 1x1 conv, BN, ReLU -> iFFT) that exchange information.
  torch.rfft(normalized=True) of the original (removed from torch) = torch.fft.rfft2(norm="ortho").
- FcaNet multi-spectral channel attention (Qin et al., ICCV 2021, arXiv:2012.11879;
  cfzd/FcaNet model/layer.py): squeeze-and-excitation whose pooling is a set of 2-D DCT
  coefficients (global average pooling is the lowest DCT frequency).
- FNet Fourier mixing (Lee-Thorp et al., NAACL 2022, arXiv:2105.03824): Re(FFT over the hidden
  axis and the sequence axis) replaces self-attention; post-LN encoder block. The official code is
  JAX (google-research/f_net): CHECKED by a closed-form test against numpy.fft.
- For hyperspectral cubes (DLNet): `spectral_gf` is the GFNet filter applied along the SPECTRAL
  axis (each pixel's spectrum is filtered in the frequency domain), with a residual connection;
  `spectral_rfft` turns each pixel's spectrum into its Fourier coefficients (real and imaginary
  parts as channels, the FFC layout). These two are DLNet's adaptation of the above, not a port
  of a published HSI network. Note: the spectral axis must be the physical band order, i.e. use
  `--pca 0`; along PCA components a Fourier transform has no physical meaning.

FFTs run in float32 (as GFNet and AFNO do) and cast back, so they are safe under autocast.
FLOPs of the FFT itself are not counted by dlnet/zoo/cost.py (only the learned products are).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import op

FFT_KINDS = ("ser", "map", "vol", "seq")


def _to_last(x, kind):
    """(B, C, *S) -> (B, *S, C) for the convolutional kinds; seq (B, T, D) already is."""
    return x if kind == "seq" else x.movedim(1, -1)


def _from_last(x, kind):
    return x if kind == "seq" else x.movedim(-1, 1)


def _space_channels(dims, kind):
    """(spatial sizes, channels) of a kind."""
    if kind == "seq":
        return tuple(dims[:1]), dims[1]
    return tuple(dims[1:]), dims[0]


# ------------------------------------------------------------------------------ GFNet


class GlobalFilter(nn.Module):
    """GFNet GlobalFilter for any number of spatial axes; input channels-last (B, *S, C)."""

    def __init__(self, dim, sizes):
        super().__init__()
        self.sizes = tuple(sizes)
        shape = (*self.sizes[:-1], self.sizes[-1] // 2 + 1, dim, 2)
        self.complex_weight = nn.Parameter(torch.randn(*shape, dtype=torch.float32) * 0.02)

    def forward(self, x):
        dtype, axes = x.dtype, tuple(range(1, 1 + len(self.sizes)))
        x = torch.fft.rfftn(x.float(), dim=axes, norm="ortho")
        x = x * torch.view_as_complex(self.complex_weight)
        x = torch.fft.irfftn(x, s=self.sizes, dim=axes, norm="ortho")
        return x.to(dtype)


class Mlp(nn.Module):
    """timm-style Mlp of GFNet / AFNO: fc1, GELU, dropout, fc2, dropout."""

    def __init__(self, d, hidden, drop=0.0):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class GFNetBlock(nn.Module):
    """GFNet Block: x + MLP(LN(GlobalFilter(LN(x)))) (one residual, as in the official code)."""

    def __init__(self, dim, sizes, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim, sizes)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        return x + self.mlp(self.norm2(self.filter(self.norm1(x))))


class _ChannelsLast(nn.Module):
    def __init__(self, body, kind):
        super().__init__()
        self.body, self.kind = body, kind

    def forward(self, x):
        return _from_last(self.body(_to_last(x, self.kind)), self.kind)


@op("global_filter", FFT_KINDS, space={}, tags={"attn"},
    source="raoyongming/GFNet gfnet.py GlobalFilter (Rao et al., arXiv:2107.00645), n-D")  # fmt: skip
def _global_filter(dims, kind):
    sizes, c = _space_channels(dims, kind)
    return _ChannelsLast(GlobalFilter(c, sizes), kind)


@op("gfnet_block", FFT_KINDS, space={"mlp_ratio": [4, 2], "dropout": [0.0, 0.1]}, tags={"block"},
    source="raoyongming/GFNet gfnet.py Block (LayerNorm, GlobalFilter, LayerNorm, Mlp, residual)")  # fmt: skip
def _gfnet_block(dims, kind, mlp_ratio, dropout):
    sizes, c = _space_channels(dims, kind)
    return _ChannelsLast(GFNetBlock(c, sizes, mlp_ratio, dropout), kind)


# ------------------------------------------------------------------------------- AFNO


class AFNO2D(nn.Module):
    """NVlabs AFNO2D on a channels-last map (B, H, W, C); returns x + filtered x (as the original).

    The number of kept frequencies follows the original exactly: int((H*W // 2 + 1) * fraction)
    along the last (rfft) axis; with fraction 1 every frequency is kept."""

    def __init__(self, hidden, num_blocks=8, sparsity=0.01, fraction=1.0, factor=1):
        super().__init__()
        if hidden % num_blocks:
            raise ValueError(f"AFNO: channels {hidden} not divisible by num_blocks {num_blocks}")
        self.hidden, self.num_blocks, self.bs = hidden, num_blocks, hidden // num_blocks
        self.sparsity, self.fraction, self.factor = sparsity, fraction, factor
        s, nb, bs = 0.02, num_blocks, self.bs
        self.w1 = nn.Parameter(s * torch.randn(2, nb, bs, bs * factor))
        self.b1 = nn.Parameter(s * torch.randn(2, nb, bs * factor))
        self.w2 = nn.Parameter(s * torch.randn(2, nb, bs * factor, bs))
        self.b2 = nn.Parameter(s * torch.randn(2, nb, bs))

    def forward(self, x):
        bias, dtype = x, x.dtype
        x = x.float()
        B, H, W, C = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.bs)
        kept = int((H * W // 2 + 1) * self.fraction)
        xr, xi = x[:, :, :kept].real, x[:, :, :kept].imag
        e = lambda a, w: torch.einsum("...bi,bio->...bo", a, w)
        o1r = F.relu(e(xr, self.w1[0]) - e(xi, self.w1[1]) + self.b1[0])
        o1i = F.relu(e(xi, self.w1[0]) + e(xr, self.w1[1]) + self.b1[1])
        o2r = e(o1r, self.w2[0]) - e(o1i, self.w2[1]) + self.b2[0]
        o2i = e(o1i, self.w2[0]) + e(o1r, self.w2[1]) + self.b2[1]
        o = torch.zeros(*x.shape, 2, device=x.device)
        o[:, :, :kept] = torch.stack([o2r, o2i], dim=-1)
        o = torch.view_as_complex(F.softshrink(o, lambd=self.sparsity).contiguous())
        o = torch.fft.irfft2(o.reshape(B, H, W // 2 + 1, C), s=(H, W), dim=(1, 2), norm="ortho")
        return o.to(dtype) + bias


class AFNOBlock(nn.Module):
    """afnonet.Block with mixing_type "afno": LN, AFNO2D, [double skip], LN, Mlp, residual."""

    def __init__(self, dim, num_blocks, mlp_ratio=4.0, drop=0.0, double_skip=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = AFNO2D(dim, num_blocks)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.filter(self.norm1(x))
        if self.double_skip:
            x = x + residual
            residual = x
        return self.mlp(self.norm2(x)) + residual


@op("afno", "map", space={"num_blocks": [1, 4, 8]}, tags={"attn"},
    source="NVlabs/AFNO-transformer afno/afno2d.py AFNO2D (Guibas et al., arXiv:2111.13587)")  # fmt: skip
def _afno(dims, kind, num_blocks):
    return _ChannelsLast(AFNO2D(dims[0], num_blocks), kind)


@op("afno_block", "map",
    space={"num_blocks": [1, 4, 8], "mlp_ratio": [4, 2], "double_skip": [0, 1], "dropout": [0.0, 0.1]},
    tags={"block"}, source="NVlabs/AFNO-transformer classification/afnonet.py Block (mixing afno)")  # fmt: skip
def _afno_block(dims, kind, num_blocks, mlp_ratio, double_skip, dropout):
    return _ChannelsLast(
        AFNOBlock(dims[0], num_blocks, mlp_ratio, dropout, bool(double_skip)), kind
    )


# -------------------------------------------------------------------------------- FNO


class SpectralConv(nn.Module):
    """FNO SpectralConv1d/2d/3d (fourier_1d/2d/3d.py): channels-first (B, C, *S).

    Keeps the `modes` lowest frequencies of every axis (both signs on all but the last, rfft,
    axis: 2^(n-1) weight tensors in the original order weights1, weights2, ...). `modes` is
    clipped to what the input size allows (S//2, last axis S//2+1)."""

    def __init__(self, c_in, c_out, sizes, modes):
        super().__init__()
        self.sizes, self.c_out = tuple(sizes), c_out
        n = len(sizes)
        self.modes = [max(1, min(modes, s // 2)) for s in sizes[:-1]]
        self.modes.append(max(1, min(modes, sizes[-1] // 2 + 1)))
        scale = 1 / (c_in * c_out)
        self.weights = nn.ParameterList(
            nn.Parameter(scale * torch.rand(c_in, c_out, *self.modes, 2))
            for _ in range(2 ** (n - 1))
        )
        # corner k: axis i (i < n-1) takes the high (negative) frequencies when bit i of k is set
        self.corners = [
            tuple(slice(-m, None) if (k >> i) & 1 else slice(0, m) for i, m in enumerate(self.modes[:-1]))
            + (slice(0, self.modes[-1]),)
            for k in range(2 ** (n - 1))
        ]  # fmt: skip

    def forward(self, x):
        dtype, n = x.dtype, len(self.sizes)
        axes = tuple(range(-n, 0))
        xf = torch.fft.rfftn(x.float(), dim=axes)
        out = torch.zeros(
            x.shape[0], self.c_out, *self.sizes[:-1], self.sizes[-1] // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )  # fmt: skip
        for w, c in zip(self.weights, self.corners, strict=True):
            idx = (slice(None), slice(None), *c)
            out[idx] = torch.einsum("bi...,io...->bo...", xf[idx], torch.view_as_complex(w))
        return torch.fft.irfftn(out, s=self.sizes, dim=axes).to(dtype)


class FNOBlock(nn.Module):
    """One FNO layer (FNO2d.forward): act(SpectralConv(x) + W x), W a 1x1 convolution."""

    def __init__(self, c, sizes, modes, act=True):
        super().__init__()
        self.conv = SpectralConv(c, c, sizes, modes)
        self.w = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[len(sizes) - 1](c, c, 1)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x) + self.w(x))


@op("spectral_conv", ("ser", "map", "vol"), space={"modes": [4, 2, 8, 12], "out": [0]}, tags={"attn"},
    source="Li et al. FNO arXiv:2010.08895 eq. 4-5, original zongyi-li fourier_{1,2,3}d.py SpectralConv")  # fmt: skip
def _spectral_conv(dims, kind, modes, out):
    return SpectralConv(dims[0], out or dims[0], dims[1:], modes)


@op("fno_block", ("ser", "map", "vol"), space={"modes": [4, 2, 8, 12]}, tags={"block"},
    source="Li et al. FNO arXiv:2010.08895, original FNO2d layer: gelu(SpectralConv2d(x) + Conv1x1(x))")  # fmt: skip
def _fno_block(dims, kind, modes):
    return FNOBlock(dims[0], dims[1:], modes)


# -------------------------------------------------------------------------------- FFC


class FourierUnit(nn.Module):
    """ffc.FourierUnit: rFFT2 -> (real, imag) as channels -> 1x1 conv, BN, ReLU -> iFFT2."""

    def __init__(self, c_in, c_out, groups=1):
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(c_in * 2, c_out * 2, 1, 1, 0, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, _, h, w = x.shape
        dtype = x.dtype
        f = torch.view_as_real(torch.fft.rfft2(x.float(), norm="ortho"))  # (b, c, h, w/2+1, 2)
        f = f.permute(0, 1, 4, 2, 3).reshape(b, -1, h, w // 2 + 1).to(dtype)
        f = self.relu(self.bn(self.conv_layer(f)))
        f = f.float().reshape(b, -1, 2, h, w // 2 + 1).permute(0, 1, 3, 4, 2).contiguous()
        return torch.fft.irfft2(torch.view_as_complex(f), s=(h, w), norm="ortho").to(dtype)


class SpectralTransform(nn.Module):
    """ffc.SpectralTransform (stride 1): 1x1 conv, FourierUnit, optional local FU, 1x1 conv."""

    def __init__(self, c_in, c_out, groups=1, enable_lfu=True):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.downsample = nn.Identity()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_out // 2, 1, groups=groups, bias=False),
            nn.BatchNorm2d(c_out // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(c_out // 2, c_out // 2, groups)
        if enable_lfu:
            self.lfu = FourierUnit(c_out // 2, c_out // 2, groups)
        self.conv2 = nn.Conv2d(c_out // 2, c_out, 1, groups=groups, bias=False)

    def forward(self, x):
        x = self.conv1(self.downsample(x))
        out = self.fu(x)
        xs = 0
        if self.enable_lfu:
            _, c, h, w = x.shape
            xs = torch.cat(torch.split(x[:, : c // 4], h // 2, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, w // 2, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs).repeat(1, 1, 2, 2).contiguous()
        return self.conv2(x + out + xs)


class FFC(nn.Module):
    """ffc.FFC: local->local, local->global, global->local convs and the global SpectralTransform."""

    def __init__(self, c_in, c_out, k, ratio_gin, ratio_gout, padding=0, enable_lfu=True):
        super().__init__()
        in_cg = int(c_in * ratio_gin)
        in_cl = c_in - in_cg
        out_cg = int(c_out * ratio_gout)
        out_cl = c_out - out_cg
        self.ratio_gin, self.ratio_gout = ratio_gin, ratio_gout
        conv = lambda a, b: (
            nn.Identity() if a == 0 or b == 0 else nn.Conv2d(a, b, k, 1, padding, bias=False)
        )
        self.convl2l = conv(in_cl, out_cl)
        self.convl2g = conv(in_cl, out_cg)
        self.convg2l = conv(in_cg, out_cl)
        self.convg2g = (
            nn.Identity() if in_cg == 0 or out_cg == 0 else SpectralTransform(in_cg, out_cg, 1, enable_lfu)
        )  # fmt: skip

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_l, out_g = 0, 0
        if self.ratio_gout != 1:
            out_l = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_g = self.convl2g(x_l) + self.convg2g(x_g)
        return out_l, out_g


class FFC_BN_ACT(nn.Module):
    def __init__(
        self, c_in, c_out, k, ratio_gin, ratio_gout, padding=0, act=False, enable_lfu=True
    ):
        super().__init__()
        self.ffc = FFC(c_in, c_out, k, ratio_gin, ratio_gout, padding, enable_lfu)
        self.bn_l = (
            nn.Identity() if ratio_gout == 1 else nn.BatchNorm2d(int(c_out * (1 - ratio_gout)))
        )
        self.bn_g = nn.Identity() if ratio_gout == 0 else nn.BatchNorm2d(int(c_out * ratio_gout))
        a = (lambda: nn.ReLU(inplace=True)) if act else nn.Identity
        self.act_l = nn.Identity() if ratio_gout == 1 else a()
        self.act_g = nn.Identity() if ratio_gout == 0 else a()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        return self.act_l(self.bn_l(x_l)), self.act_g(self.bn_g(x_g))


class FFCBasicBlock(nn.Module):
    """ffc_resnet.BasicBlock (stride 1, no downsample, no SE): two FFC_BN_ACT, residual, ReLU.

    Takes and returns ONE map: the last int(C * ratio) channels are the global (spectral) part."""

    def __init__(self, c, ratio=0.5, lfu=True):
        super().__init__()
        self.cg = int(c * ratio)
        self.conv1 = FFC_BN_ACT(c, c, 3, ratio, ratio, 1, act=True, enable_lfu=lfu)
        self.conv2 = FFC_BN_ACT(c, c, 3, ratio, ratio, 1, act=False, enable_lfu=lfu)
        self.relu_l = nn.Identity() if ratio == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio == 0 else nn.ReLU(inplace=True)

    def forward(self, x):
        c = x.shape[1]
        id_l, id_g = x[:, : c - self.cg], x[:, c - self.cg :]
        x_l, x_g = self.conv2(self.conv1((id_l, id_g)))
        return torch.cat([self.relu_l(x_l + id_l), self.relu_g(x_g + id_g)], 1)


@op("ffc_block", "map", space={"ratio": [0.5, 0.25, 0.75], "lfu": [1, 0]}, tags={"block"},
    source="pkumivision/FFC model_zoo/ffc_resnet.py BasicBlock + ffc.py (Chi et al., NeurIPS 2020)")  # fmt: skip
def _ffc_block(dims, kind, ratio, lfu):
    c, h, w = dims
    cg = int(c * ratio)
    if int(c * (1 - ratio)) != c - cg:  # the official BN sizes would not match the convs
        raise ValueError(f"FFC: {c} channels do not split evenly at ratio {ratio}")
    if cg < 2 or c - cg < 1:
        raise ValueError(f"FFC: {c} channels leave no local or global part at ratio {ratio}")
    if lfu and (h % 2 or w % 2 or (cg // 2) % 4):
        raise ValueError(
            "FFC local Fourier unit needs even H, W and global channels / 2 divisible by 4"
        )
    return FFCBasicBlock(c, ratio, bool(lfu))


# ----------------------------------------------------------------------------- FcaNet

FCA_INDICES = {  # cfzd/FcaNet model/layer.py get_freq_indices (7x7 frequency space)
    "top": ([0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1],
            [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]),
    "low": ([0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
            [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]),
    "bot": ([6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6],
            [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]),
}  # fmt: skip


def dct_filters(h, w, mapper_x, mapper_y, channel):
    """MultiSpectralDCTLayer.get_dct_filter: channel groups, each one 2-D DCT-II basis function."""

    def f(pos, freq, n):
        r = math.cos(math.pi * freq * (pos + 0.5) / n) / math.sqrt(n)
        return r if freq == 0 else r * math.sqrt(2)

    out = torch.zeros(channel, h, w)
    part = channel // len(mapper_x)
    for i, (u, v) in enumerate(zip(mapper_x, mapper_y, strict=True)):
        for tx in range(h):
            for ty in range(w):
                out[i * part : (i + 1) * part, tx, ty] = f(tx, u, h) * f(ty, v, w)
    return out


class MultiSpectralAttention(nn.Module):
    """FcaNet MultiSpectralAttentionLayer: DCT pooling -> FC, ReLU, FC, sigmoid -> channel gates."""

    def __init__(self, channel, dct_h=7, dct_w=7, reduction=16, method="top16"):
        super().__init__()
        n = int(method[3:])
        xs, ys = (v[:n] for v in FCA_INDICES[method[:3]])
        xs, ys = [u * (dct_h // 7) for u in xs], [v * (dct_w // 7) for v in ys]
        if channel % n:
            raise ValueError(f"FcaNet: channels {channel} not divisible by {n} frequencies")
        self.dct_h, self.dct_w = dct_h, dct_w
        self.register_buffer("weight", dct_filters(dct_h, dct_w, xs, ys, channel))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        n, c, h, w = x.shape
        xp = (
            x
            if (h, w) == (self.dct_h, self.dct_w)
            else F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        )
        y = (xp * self.weight).sum(dim=(2, 3))
        return x * self.fc(y).view(n, c, 1, 1)


@op("fca", "map", space={"reduction": [16, 8, 4], "method": ["top16", "low16", "top4", "top1"]},
    tags={"attn"}, source="cfzd/FcaNet model/layer.py MultiSpectralAttentionLayer (Qin et al., arXiv:2012.11879)")  # fmt: skip
def _fca(dims, kind, reduction, method):
    if dims[0] // reduction < 1:
        raise ValueError(f"FcaNet: {dims[0]} channels with reduction {reduction}")
    return MultiSpectralAttention(dims[0], 7, 7, reduction, method)


# ------------------------------------------------------------------------------- FNet


class FNetBlock(nn.Module):
    """FNet encoder block: LN(x + Re(FFT_seq(FFT_hidden(x)))), then LN(x + FFN(x)) (post-LN)."""

    def __init__(self, d, ratio=4, dropout=0.0):
        super().__init__()
        self.mixing_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * ratio), nn.GELU(), nn.Linear(d * ratio, d), nn.Dropout(dropout)
        )
        self.output_norm = nn.LayerNorm(d)

    def forward(self, x):
        dtype = x.dtype
        mixed = torch.fft.fftn(x.float(), dim=(-2, -1)).real.to(dtype)
        x = self.mixing_norm(x + mixed)
        return self.output_norm(x + self.ff(x))


@op("fnet_block", "seq", space={"ratio": [4, 2], "dropout": [0.0, 0.1]}, tags={"block"},
    source="Lee-Thorp et al. FNet arXiv:2105.03824 (google-research/f_net, JAX): Fourier mixing")  # fmt: skip
def _fnet(dims, kind, ratio, dropout):
    return FNetBlock(dims[1], ratio, dropout)


# --------------------------------------------------------------- hyperspectral (DLNet)


class SpectralGlobalFilter(nn.Module):
    """x + iFFT_band(K * FFT_band(x)) along the channel (spectral) axis, K shared by all pixels."""

    def __init__(self, bands):
        super().__init__()
        self.bands = bands
        self.complex_weight = nn.Parameter(torch.randn(bands // 2 + 1, 1, 1, 2) * 0.02)

    def forward(self, x):
        dtype = x.dtype
        f = torch.fft.rfft(x.float(), dim=1, norm="ortho") * torch.view_as_complex(
            self.complex_weight
        )
        return x + torch.fft.irfft(f, n=self.bands, dim=1, norm="ortho").to(dtype)


class SpectralRFFT(nn.Module):
    """Each pixel's spectrum -> its rFFT coefficients, (real, imag) interleaved as channels."""

    def forward(self, x):
        b, _, h, w = x.shape
        f = torch.view_as_real(torch.fft.rfft(x.float(), dim=1, norm="ortho"))  # (b, F, h, w, 2)
        return f.permute(0, 1, 4, 2, 3).reshape(b, -1, h, w).to(x.dtype)


@op("spectral_gf", "map", space={}, tags={"attn"},
    source="DLNet: GFNet GlobalFilter (arXiv:2107.00645) along the spectral axis, residual")  # fmt: skip
def _spectral_gf(dims, kind):
    return SpectralGlobalFilter(dims[0])


@op("spectral_rfft", "map", space={}, tags={"stem"},
    source="DLNet: per-pixel spectrum -> Fourier coefficients (FFC real/imag channel layout)")  # fmt: skip
def _spectral_rfft(dims, kind):
    return SpectralRFFT()


__all__ = [
    "AFNO2D", "AFNOBlock", "FFCBasicBlock", "FNOBlock", "FNetBlock", "GFNetBlock", "GlobalFilter",
    "MultiSpectralAttention", "SpectralConv", "SpectralGlobalFilter", "SpectralRFFT",
]  # fmt: skip
