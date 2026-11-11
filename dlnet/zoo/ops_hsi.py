"""Hyperspectral networks of 2022-2025 written from their OFFICIAL code (v1.0; THIRD_PARTY.md).

Inputs are HSI patches, kind `map` (bands or PCA components, H, W). Mamba is DLNet's pure-PyTorch
mixer (ops_seq.Mamba, equal to state-spaces/mamba's reference path: scripts/verify_numeric.py),
used where the originals import `mamba_ssm.Mamba` (CUDA). Numerical check of the surrounding code
with weights copied across: scripts/verify_hsi.py (job `verify-hsi`, mamba_ssm replaced by that
same Mamba).

- MambaHSI (Li et al., IEEE TGRS 2024, arXiv:2501.04944; li-yapeng/MambaHSI model/MambaHSI.py):
  spatial Mamba over the pixels + spectral Mamba over groups of bands, fused by learned weights.
  The original runs on the whole scene and flattens ALL pixels of the batch into one sequence
  (`view(1, -1, C)`); on patches DLNet scans the pixels of each patch (`view(B, -1, C)`).
- MambaHSI+ (TGRS 2025; RockAilab/MambaHSI_Plus model/MambaHSI_Plus.py, Utils.py): row and
  column scans, each forward + backward, and a Mamba channel attention (ECA layer with Mamba).
- SSFTT (Sun et al., IEEE TGRS 2022; zgr6010/HSI_SSFTT cls_SSFTT_IP/SSFTTnet.py): 3-D conv, 2-D
  conv, Gaussian-weighted tokenizer, one transformer encoder; the original scales attention by
  dim**-0.5 (not the head width), kept.
- FAHM (Zhang et al. 2025; zhangxc0105/FAHM model/fahm.py, rope_spectral.py): three stages of a
  frequency-domain embedding (rFFT2 -> block-diagonal complex layer, GELU, soft-shrink -> iFFT2),
  rotary position on the tokens, Mamba, and a weighted fusion of the stages.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import op
from .ops_seq import Mamba

# --------------------------------------------------------------------------- MambaHSI


class SpeMamba(nn.Module):
    """MambaHSI SpeMamba: each pixel's channels split into `token_num` tokens, Mamba over them."""

    def __init__(self, channels, token_num=4, use_residual=True, group_num=4):
        super().__init__()
        self.token_num, self.use_residual = token_num, use_residual
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = token_num * self.group_channel_num
        self.mamba = Mamba(self.group_channel_num, 16, 4, 2)
        self.proj = nn.Sequential(nn.GroupNorm(group_num, self.channel_num), nn.SiLU())

    def pad(self, x):
        b, c, h, w = x.shape
        return (
            x
            if c >= self.channel_num
            else torch.cat([x, x.new_zeros(b, self.channel_num - c, h, w)], 1)
        )

    def forward(self, x):
        xp = self.pad(x).permute(0, 2, 3, 1).contiguous()
        b, h, w, cp = xp.shape
        y = self.mamba(xp.view(b * h * w, self.token_num, self.group_channel_num))
        y = self.proj(y.view(b, h, w, cp).permute(0, 3, 1, 2).contiguous())
        return x + y if self.use_residual else y


class SpaMamba(nn.Module):
    """MambaHSI SpaMamba: Mamba over the pixels of each patch (row-major), GroupNorm, SiLU."""

    def __init__(self, channels, use_residual=True, group_num=4):
        super().__init__()
        self.use_residual = use_residual
        self.mamba = Mamba(channels, 16, 4, 2)
        self.proj = nn.Sequential(nn.GroupNorm(group_num, channels), nn.SiLU())

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.mamba(x.permute(0, 2, 3, 1).reshape(b, h * w, c))
        y = self.proj(y.view(b, h, w, c).permute(0, 3, 1, 2).contiguous())
        return y + x if self.use_residual else y


class BothMamba(nn.Module):
    """MambaHSI BothMamba: softmax-weighted sum of the spatial and spectral branches, residual."""

    def __init__(self, channels, token_num=4, use_residual=True, group_num=4, use_att=True):
        super().__init__()
        self.use_att, self.use_residual = use_att, use_residual
        if use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
        self.spa_mamba = SpaMamba(channels, use_residual, group_num)
        self.spe_mamba = SpeMamba(channels, token_num, use_residual, group_num)

    def forward(self, x):
        a, s = self.spa_mamba(x), self.spe_mamba(x)
        if self.use_att:
            w = torch.softmax(self.weights, 0)
            f = a * w[0] + s * w[1]
        else:
            f = a + s
        return f + x if self.use_residual else f


def _gn_ok(c, token_num, groups):
    spe = token_num * math.ceil(c / token_num)
    return c % groups == 0 and spe % groups == 0


@op("mambahsi_block", "map", space={"token_num": [4, 8, 2], "group_num": [4, 2]}, tags={"block"},
    source="li-yapeng/MambaHSI model/MambaHSI.py BothMamba (Li et al., TGRS 2024), per-patch scan")  # fmt: skip
def _mambahsi(dims, kind, token_num, group_num):
    if not _gn_ok(dims[0], token_num, group_num):
        raise ValueError(f"MambaHSI: {dims[0]} channels do not split into {group_num} groups")
    return BothMamba(dims[0], token_num, True, group_num, True)


# -------------------------------------------------------------------------- MambaHSI+


class ECALayer(nn.Module):
    """MambaHSI+ Utils.ECALayer: global average pool, Mamba across the channel sequence, sigmoid."""

    def __init__(self, channel):
        super().__init__()
        self.mamba = Mamba(channel, 16, 4, 2)

    def forward(self, x):
        y = x.mean((2, 3)).unsqueeze(1)  # (B, 1, C)
        return x * torch.sigmoid(self.mamba(y).permute(0, 2, 1).unsqueeze(-1))


class SpaMambaProcessor(nn.Module):
    """Mamba over the flattened map and over its reverse, summed."""

    def __init__(self, channels):
        super().__init__()
        self.mamba = Mamba(channels, 16, 4, 2)

    def forward(self, x):  # (B, H, W, C)
        b = x.shape[0]
        f = x.reshape(b, -1, x.shape[-1])
        return (self.mamba(f) + torch.flip(self.mamba(torch.flip(f, dims=[1])), dims=[1])).view(
            *x.shape
        )


class SpeMambaProcessor(nn.Module):
    def __init__(self, group_channel_num):
        super().__init__()
        self.mamba, self.g = Mamba(group_channel_num, 16, 4, 2), group_channel_num

    def forward(self, x):  # (B, H, W, C)
        b, h, w, _ = x.shape
        f = x.reshape(b * h * w, -1, self.g)
        return (self.mamba(f) + torch.flip(self.mamba(torch.flip(f, dims=[1])), dims=[1])).view(
            *x.shape
        )


class _RowCol(nn.Module):
    """Row scan (B, H, W, C) and column scan (B, W, H, C), weighted, GroupNorm + SiLU, ECA."""

    def __init__(self, row, col, c, group_num, use_att=True):
        super().__init__()
        self.use_att = use_att
        if use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
        self.mamba_col, self.mamba_row = col, row
        self.eca = ECALayer(c)
        self.proj = nn.Sequential(nn.GroupNorm(group_num, c), nn.SiLU())

    def mix(self, x):
        xr = self.mamba_row(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        xc = self.mamba_col(x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1)
        if self.use_att:
            w = torch.softmax(self.weights, 0)
            r = xr * w[0] + xc * w[1]
        else:
            r = xr + xc
        return self.eca(self.proj(r))


class SpaMambaPlus(_RowCol):
    def __init__(self, channels, group_num=2):
        super().__init__(
            SpaMambaProcessor(channels), SpaMambaProcessor(channels), channels, group_num
        )

    def forward(self, x):
        return self.mix(x) + x


class SpeMambaPlus(_RowCol):
    def __init__(self, channels, token_num=4, group_num=2):
        g = math.ceil(channels / token_num)
        self.channel_num = token_num * g
        super().__init__(SpeMambaProcessor(g), SpeMambaProcessor(g), self.channel_num, group_num)

    def forward(self, x):
        b, c, h, w = x.shape
        xp = (
            x
            if c >= self.channel_num
            else torch.cat([x, x.new_zeros(b, self.channel_num - c, h, w)], 1)
        )
        return self.mix(xp) + xp


class BothMambaPlus(nn.Module):
    """MambaHSI_Plus BothMamba: spatial + spectral (row/column, bidirectional, ECA), residual."""

    def __init__(self, channels, token_num=4, group_num=2):
        super().__init__()
        self.spa_mamba = SpaMambaPlus(channels, group_num)
        self.spe_mamba = SpeMambaPlus(channels, token_num, group_num)

    def forward(self, x):
        return self.spa_mamba(x) + self.spe_mamba(x) + x


@op("mambahsi_plus_block", "map", space={"token_num": [4, 8, 2], "group_num": [2, 4]}, tags={"block"},
    source="RockAilab/MambaHSI_Plus model/MambaHSI_Plus.py BothMamba + Utils (TGRS 2025)")  # fmt: skip
def _mambahsi_plus(dims, kind, token_num, group_num):
    if dims[0] % token_num or not _gn_ok(dims[0], token_num, group_num):
        raise ValueError(
            f"MambaHSI+: {dims[0]} channels must be divisible by {token_num} tokens and {group_num} groups"
        )
    return BothMambaPlus(dims[0], token_num, group_num)


# ------------------------------------------------------------------------------- SSFTT


class _Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class _LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm, self.fn = nn.LayerNorm(dim), fn

    def forward(self, x):
        return self.fn(self.norm(x))


class _MLPBlock(nn.Module):
    def __init__(self, dim, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _SSFTTAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads, self.scale = heads, dim**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.nn1, self.do1 = nn.Linear(dim, dim), nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = (
            t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in self.to_qkv(x).chunk(3, dim=-1)
        )
        attn = (torch.einsum("bhid,bhjd->bhij", q, k) * self.scale).softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).reshape(b, n, d)
        return self.do1(self.nn1(out))


class _SSFTTTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleList([
                _Residual(_LayerNormalize(dim, _SSFTTAttention(dim, heads, dropout))),
                _Residual(_LayerNormalize(dim, _MLPBlock(dim, mlp_dim, dropout))),
            ])
            for _ in range(depth)
        )  # fmt: skip

    def forward(self, x):
        for attention, mlp in self.layers:
            x = mlp(attention(x))
        return x


class SSFTT(nn.Module):
    """SSFTTnet without its final linear layer: (B, C, H, W) -> class-token features (B, dim)."""

    def __init__(
        self, bands, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1
    ):
        super().__init__()
        self.L, self.cT = num_tokens, dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, 8, (3, 3, 3)), nn.BatchNorm3d(8), nn.ReLU()
        )
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(8 * (bands - 2), 64, (3, 3)), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.token_wA = nn.Parameter(torch.empty(1, num_tokens, 64))
        self.token_wV = nn.Parameter(torch.empty(1, 64, dim))
        nn.init.xavier_normal_(self.token_wA)
        nn.init.xavier_normal_(self.token_wV)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _SSFTTTransformer(dim, depth, heads, mlp_dim, dropout)

    def forward(self, x):
        x = self.conv3d_features(x.unsqueeze(1))  # (B, 8, C-2, H-2, W-2)
        b, c, d, h, w = x.shape
        x = self.conv2d_features(x.reshape(b, c * d, h, w))
        x = x.flatten(2).transpose(1, 2)  # (B, HW, 64)
        a = (
            torch.einsum("bij,bjk->bik", x, self.token_wA.transpose(1, 2))
            .transpose(1, 2)
            .softmax(dim=-1)
        )
        t = torch.einsum("bij,bjk->bik", a, torch.einsum("bij,bjk->bik", x, self.token_wV))
        x = torch.cat((self.cls_token.expand(b, -1, -1), t), dim=1) + self.pos_embedding
        return self.transformer(self.dropout(x))[:, 0]


@op("ssftt", "map", "vec", space={"tokens": [4, 8], "dim": [64, 32], "heads": [8, 4], "depth": [1, 2]}, tags={"stem"},
    source="zgr6010/HSI_SSFTT cls_SSFTT_IP/SSFTTnet.py (Sun et al., TGRS 2022), without the last linear layer")  # fmt: skip
def _ssftt(dims, kind, tokens, dim, heads, depth):
    c, h, w = dims
    if c < 3 or h < 5 or w < 5:
        raise ValueError(
            "SSFTT needs >= 3 spectral components and patches >= 5x5 (two valid 3x3 convolutions)"
        )
    if dim % heads:
        raise ValueError(f"SSFTT: dim {dim} not divisible by {heads} heads")
    return SSFTT(c, tokens, dim, depth, heads)


# -------------------------------------------------------------------------------- FAHM


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """rope_spectral.RotaryEmbedding ('lang' frequencies) used as rotate_queries_or_keys on (B, N, D)."""

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.register_buffer(
            "freqs", 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        )

    def rotate_queries_or_keys(self, t):
        n = t.shape[-2]
        f = torch.arange(n, device=t.device, dtype=t.dtype)[:, None] * self.freqs.to(t.dtype)
        f = f.repeat_interleave(2, dim=-1).to(t)
        r = f.shape[-1]
        left, right = t[..., :r], t[..., r:]
        return torch.cat((left * f.cos() + rotate_half(left) * f.sin(), right), dim=-1)


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x * torch.sigmoid(self.pa_conv(x))


class FGEM(nn.Module):
    """FAHM frequency-domain embedding: rFFT2, block-diagonal complex layer + GELU + residual,
    soft-shrink, + spectrum, iFFT2, 1x1 conv, BN, GELU -> tokens (B, HW, embed_dim)."""

    def __init__(self, in_chans, embed_dim, n_groups=2, sparsity_threshold=0.01):
        super().__init__()
        if in_chans % n_groups:
            raise ValueError(f"FGEM: {in_chans} channels not divisible by {n_groups} groups")
        self.n_groups, self.block_size, self.sparsity_threshold = (
            n_groups,
            in_chans // n_groups,
            sparsity_threshold,
        )
        self.w1 = nn.Parameter(0.02 * torch.randn(2, n_groups, self.block_size, self.block_size))
        self.b1 = nn.Parameter(0.02 * torch.randn(2, n_groups, self.block_size))
        self.act = nn.GELU()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        b, c, h, w = x.shape
        f0 = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        f = f0.reshape(b, self.n_groups, self.block_size, f0.shape[2], f0.shape[3])
        e = lambda a, m: torch.einsum("bkihw,kio->bkohw", a, m)
        re = (
            self.act(e(f.real, self.w1[0]) - e(f.imag, self.w1[1]) + self.b1[0, :, :, None, None])
            + f.real
        )
        im = (
            self.act(e(f.imag, self.w1[0]) + e(f.real, self.w1[1]) + self.b1[1, :, :, None, None])
            + f.imag
        )
        z = torch.view_as_complex(
            F.softshrink(torch.stack([re, im], dim=-1), lambd=self.sparsity_threshold).contiguous()
        )
        z = z.reshape(b, c, f0.shape[2], f0.shape[3]) + f0
        x = torch.fft.irfft2(z, s=(h, w), dim=(2, 3), norm="ortho").type(dtype)
        return self.act(self.batch_norm(self.proj(x))).flatten(2).transpose(1, 2)


class Channelaffinity(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc1, self.fc2 = (
            nn.Linear(dim, max(1, dim // reduction)),
            nn.Linear(max(1, dim // reduction), dim),
        )

    def forward(self, x):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(dim=1))))).unsqueeze(1) * x


class PreConv(nn.Module):
    """fahm.pre_conv: (1x1 conv, BN, GELU, 3x3 conv, BN, GELU) + GELU(1x1x1 conv3d(x))."""

    def __init__(self, c):
        super().__init__()
        self.act = nn.GELU()
        self.conv1, self.bn1 = nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c)
        self.conv2, self.bn2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.proj = nn.Conv3d(
            1, 1, kernel_size=1, bias=False
        )  # the original's bn4 is never used: left out

    def forward(self, x):
        y = self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))))
        return y + self.act(self.proj(x.unsqueeze(1))).squeeze(1)


class ChannelMerge(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.channel_conv = nn.Conv1d(cin, cout, kernel_size=1)

    def forward(self, x):
        return self.channel_conv(x.transpose(1, 2)).transpose(1, 2)


class FAHM(nn.Module):
    """fahm.FAHM (3 stages) without its head: (B, C, s, s) -> mean token (B, embed_dims[-1])."""

    def __init__(
        self, in_chans, patch, n_groups=(2, 2, 2), embed_dims=(256, 128, 64), depths=(1, 1, 1)
    ):
        super().__init__()
        self.patch, self.depths = patch, depths
        self.new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]
        self.channel_fusion = Channelaffinity(embed_dims[-1])
        self.postion = PA(in_chans)
        self.w = nn.Parameter(torch.ones(3))
        self.merge = nn.ModuleList()
        for i in range(3):
            setattr(
                self,
                f"patch_embed{i + 1}",
                FGEM(self.new_bands if i == 0 else embed_dims[i - 1], embed_dims[i], n_groups[i]),
            )
            setattr(self, f"norm{i + 1}", nn.LayerNorm(embed_dims[i]))
            setattr(self, f"rope{i + 1}", RotaryEmbedding(embed_dims[i] // 2))
            self.merge.append(ChannelMerge(embed_dims[i], embed_dims[-1]))
        self.mamba = nn.ModuleList(
            Mamba(embed_dims[i], 16, 4, 2) for i in range(3) for _ in range(depths[i])
        )
        self.pre_conv = PreConv(in_chans)

    def forward(self, x):
        b, c = x.shape[:2]
        x = self.postion(self.pre_conv(x))
        if self.new_bands > c:  # ReplicationPad3d on the channel axis
            x = torch.cat([x, x[:, -1:].expand(-1, self.new_bands - c, -1, -1)], 1)
        z, li = [], 0
        for i in range(3):
            x = getattr(self, f"patch_embed{i + 1}")(x)
            x = getattr(self, f"rope{i + 1}").rotate_queries_or_keys(x)
            for _ in range(self.depths[i]):
                x = self.mamba[li](x)
                li += 1
            x = getattr(self, f"norm{i + 1}")(x)
            z.append(self.channel_fusion(self.merge[i](x)))
            if i < 2:
                x = x.reshape(b, self.patch, self.patch, -1).permute(0, 3, 1, 2).contiguous()
        return (z[0] * self.w[0] + z[1] * self.w[1] + z[2] * self.w[2]).mean(dim=1)


FAHM_SIZES = {"base": (256, 128, 64), "small": (128, 64, 32), "tiny": (64, 32, 16)}


@op("fahm", "map", "vec", space={"size": list(FAHM_SIZES), "groups": [2, 4]}, tags={"stem"},
    source="zhangxc0105/FAHM model/fahm.py FAHM + rope_spectral.RotaryEmbedding (Frequency-Aware Hierarchical Mamba, 2025), without head")  # fmt: skip
def _fahm(dims, kind, size, groups):
    c, h, w = dims
    if h != w:
        raise ValueError("FAHM takes square patches")
    e = FAHM_SIZES[size]
    if any(d % groups for d in e):
        raise ValueError(f"FAHM: widths {e} not divisible by {groups} groups")
    return FAHM(c, h, (groups,) * 3, e)
