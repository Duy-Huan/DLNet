"""Drop-in operators of 2023-2025 written from their OFFICIAL code (v1.0; THIRD_PARTY.md).

Numerical check with weights copied across: scripts/verify_next.py (GitHub Actions job
`verify-next`). Each class keeps the parameter names of the original so that a state_dict copies.

Attention / normalisation (kind `seq`, or `map` flattened to tokens):
- Gated Attention (Qiu et al., NeurIPS 2025, arXiv:2505.06708; qiuzh20/gated_attention
  modeling_qwen3.py Qwen3Attention): sigmoid gate on the SDPA output, per head or per element.
- Differential attention (Ye et al., ICLR 2025, arXiv:2410.05258; microsoft/unilm
  Diff-Transformer/multihead_diffattn.py MultiheadDiffAttn): softmax(Q1K1) - lambda softmax(Q2K2).
- DyT (Zhu, Chen, He, LeCun, Liu, CVPR 2025, arXiv:2503.10622; jiachenzhu/DyT dynamic_tanh.py):
  normalisation layer replaced by weight * tanh(alpha x) + bias.
- 2-D RoPE (Heo et al., ECCV 2024, arXiv:2403.13298; naver-ai/rope-vit models/vit_rope.py):
  axial or mixed (learned) rotary position embedding inside a ViT block on a map.
Convolutional blocks (kind `map`):
- WTConv (Finder et al., ECCV 2024, arXiv:2407.05848; BGU-CS-VIL/WTConv wtconv2d.py, wtconvnext.py):
  depthwise convolutions on Haar wavelet sub-bands, large receptive field with few parameters.
- MobileNetV4 UIB and Mobile MQA (Qin et al., ECCV 2024, arXiv:2404.10518; timm 1.0.31
  _efficientnet_blocks.UniversalInvertedResidual, MobileAttention, layers MultiQueryAttention2d).
- LSNet LS convolution (Wang et al., CVPR 2025, arXiv:2503.23135; jameslahm/lsnet model/lsnet.py
  LKP, LSConv, Block; the Triton SKA kernel is rewritten with unfold).
- StarNet block (Ma et al., CVPR 2024, arXiv:2403.19967; ma-xu/Rewrite-the-Stars imagenet/starnet.py).
- EfficientViT block with ReLU linear attention (Cai et al., ICCV 2023, arXiv:2205.14756;
  mit-han-lab/efficientvit models/nn/ops.py LiteMLA, MBConv, EfficientViTBlock).
- SegNeXt MSCA (Guo et al., NeurIPS 2022, arXiv:2209.08575; Visual-Attention-Network/SegNeXt
  mmseg/models/backbones/mscan.py AttentionModule, SpatialAttention, Block; SyncBN -> BatchNorm).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import ND, op

# ------------------------------------------------------------------------------- DyT


class DynamicTanh(nn.Module):
    """dynamic_tanh.DynamicTanh: weight * tanh(alpha * x) + bias (channels last or first, any nd)."""

    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5, nd=2):
        super().__init__()
        self.channels_last, self.nd = channels_last, nd
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            return x * self.weight + self.bias
        shape = (-1,) + (1,) * self.nd
        return x * self.weight.view(shape) + self.bias.view(shape)


@op("dyt", ("vec", "seq", "ser", "map", "vol"), space={"alpha": [0.5, 1.0, 0.1]}, tags={"norm"},
    source="jiachenzhu/DyT dynamic_tanh.py DynamicTanh (Zhu et al., arXiv:2503.10622)")  # fmt: skip
def _dyt(dims, kind, alpha):
    last = kind in ("vec", "seq")
    return DynamicTanh(dims[-1] if last else dims[0], last, alpha, ND.get(kind, 0))


class RMSNorm(nn.Module):
    """Diff-Transformer rms_norm.RMSNorm / Qwen3RMSNorm (float32 statistics)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight


class SwiGLU(nn.Module):
    """Qwen3MLP / LLaMA FFN: down(silu(gate(x)) * up(x))."""

    def __init__(self, d, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------- gated attention


class GatedAttention(nn.Module):
    """Qwen3Attention with `headwise_attn_output_gate` / `elementwise_attn_output_gate`, MHA
    (num_key_value_heads = num_heads), without the rotary embedding (DLNet adds positions with
    its own `pos` op), optional q/k RMSNorm, optional causal mask. Input/output (B, T, D)."""

    def __init__(self, d, heads, gate="headwise", qk_norm=True, causal=False):
        super().__init__()
        if d % heads:
            raise ValueError(f"width {d} not divisible by {heads} heads")
        self.h, self.hd, self.gate, self.causal = heads, d // heads, gate, causal
        extra = {"headwise": heads, "elementwise": d, "none": 0}[gate]
        self.q_proj = nn.Linear(d, d + extra, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.use_qk_norm = qk_norm
        if qk_norm:
            self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x)
        if self.gate == "headwise":
            q = q.view(b, t, self.h, -1)
            q, g = torch.split(q, [self.hd, 1], dim=-1)
        elif self.gate == "elementwise":
            q = q.view(b, t, self.h, -1)
            q, g = torch.split(q, [self.hd, self.hd], dim=-1)
        q = q.reshape(b, t, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.h, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.h, self.hd).transpose(1, 2)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        a = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.hd)
        if self.causal:
            a = a.masked_fill(
                torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1), float("-inf")
            )
        a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(a, v).transpose(1, 2)  # (b, t, h, hd)
        if self.gate != "none":
            o = o * torch.sigmoid(g)
        return self.o_proj(o.reshape(b, t, -1))


class GatedBlock(nn.Module):
    """Qwen3DecoderLayer: x + attn(RMSNorm(x)); x + SwiGLU(RMSNorm(x))."""

    def __init__(self, d, heads, ratio, gate, causal):
        super().__init__()
        self.input_layernorm = RMSNorm(d)
        self.self_attn = GatedAttention(d, heads, gate, True, causal)
        self.post_attention_layernorm = RMSNorm(d)
        self.mlp = SwiGLU(d, d * ratio)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


@op("gated_attn_block", "seq", space={"heads": [4, 2, 8], "ratio": [4, 2], "gate": ["headwise", "elementwise", "none"],
    "causal": [0, 1]}, tags={"block"},
    source="qiuzh20/gated_attention modeling_qwen3.py Qwen3Attention/DecoderLayer (Qiu et al., NeurIPS 2025)")  # fmt: skip
def _gated_block(dims, kind, heads, ratio, gate, causal):
    return GatedBlock(dims[1], heads, ratio, gate, bool(causal))


# ---------------------------------------------------------------- differential attention


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    """multihead_diffattn.MultiheadDiffAttn (MHA, no rotary, optional causal mask)."""

    def __init__(self, embed_dim, depth, num_heads, causal=False):
        super().__init__()
        if embed_dim % (2 * num_heads):
            raise ValueError(f"width {embed_dim} not divisible by 2 x {num_heads} heads")
        self.num_heads, self.causal = num_heads, causal
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lambda_init = lambda_init_fn(depth)
        p = lambda: nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2 = p(), p(), p(), p()
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

    def forward(self, x):
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, 2 * self.head_dim).transpose(1, 2)
        w = torch.matmul(q * self.scaling, k.transpose(-1, -2))
        if self.causal:
            w = w.masked_fill(
                torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1), float("-inf")
            )
        w = F.softmax(torch.nan_to_num(w), dim=-1, dtype=torch.float32).type_as(w)
        l1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        l2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lam = l1 - l2 + self.lambda_init
        w = w.view(b, self.num_heads, 2, t, t)
        w = w[:, :, 0] - lam * w[:, :, 1]
        o = self.subln(torch.matmul(w, v)) * (1 - self.lambda_init)
        return self.out_proj(o.transpose(1, 2).reshape(b, t, d))


class DiffBlock(nn.Module):
    """Diff-Transformer layer: x + DiffAttn(RMSNorm(x)); x + SwiGLU(RMSNorm(x))."""

    def __init__(self, d, heads, ratio, depth, causal):
        super().__init__()
        self.norm1, self.attn = RMSNorm(d, 1e-5), MultiheadDiffAttn(d, depth, heads, causal)
        self.norm2, self.ffn = RMSNorm(d, 1e-5), SwiGLU(d, d * ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


@op("diff_attn_block", "seq", space={"heads": [2, 4, 1], "ratio": [4, 2], "depth": [1, 2, 4, 8], "causal": [0, 1]},
    tags={"block"}, source="microsoft/unilm Diff-Transformer/multihead_diffattn.py MultiheadDiffAttn (Ye et al., ICLR 2025)")  # fmt: skip
def _diff_block(dims, kind, heads, ratio, depth, causal):
    return DiffBlock(dims[1], heads, ratio, depth, bool(causal))


# ------------------------------------------------------------------------- 2-D RoPE


def init_random_2d_freqs(dim, num_heads, theta=10.0, rotate=True):
    """vit_rope.init_random_2d_freqs -> (2, heads, dim // 2)."""
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    fx, fy = [], []
    for _ in range(num_heads):
        a = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx.append(torch.cat([mag * torch.cos(a), mag * torch.cos(torch.pi / 2 + a)], dim=-1))
        fy.append(torch.cat([mag * torch.sin(a), mag * torch.sin(torch.pi / 2 + a)], dim=-1))
    return torch.stack([torch.stack(fx), torch.stack(fy)])


def init_t_xy(end_x, end_y):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    return (t % end_x).float(), torch.div(t, end_x, rounding_mode="floor").float()


def compute_axial_cis(dim, end_x, end_y, theta=100.0):
    f = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x, t_y = init_t_xy(end_x, end_y)
    fx, fy = torch.outer(t_x, f), torch.outer(t_y, f)
    return torch.cat(
        [torch.polar(torch.ones_like(fx), fx), torch.polar(torch.ones_like(fy), fy)], dim=-1
    )


def apply_rotary_emb(xq, xk, freqs_cis):
    """vit_rope.apply_rotary_emb; freqs_cis (N, hd/2) or (heads, N, hd/2)."""
    q_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    f = freqs_cis.view((1,) * (q_.ndim - freqs_cis.ndim) + tuple(freqs_cis.shape))
    return torch.view_as_real(q_ * f).flatten(3).type_as(xq), torch.view_as_real(k_ * f).flatten(
        3
    ).type_as(xk)


class RoPEAttention(nn.Module):
    """vit_rope.RoPEAttention (DeiT Attention with rotary q, k; no class token here)."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k = apply_rotary_emb(qkv[0], qkv[1], freqs_cis)
        a = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((a @ qkv[2]).transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, d, hidden, drop=0.0):
        super().__init__()
        self.fc1, self.act, self.fc2, self.drop = (
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class RoPEBlock(nn.Module):
    """RoPE_Layer_scale_init_Block on a map (C, H, W): tokens = pixels, rotary from (x, y)."""

    def __init__(self, dim, heads, h, w, mode="mixed", mlp_ratio=4, init_values=1e-4, theta=None):
        super().__init__()
        hd = dim // heads
        self.mode, self.heads, self.hw = mode, heads, (h, w)
        self.norm1, self.attn = nn.LayerNorm(dim), RoPEAttention(dim, heads)
        self.norm2, self.mlp = nn.LayerNorm(dim), Mlp(dim, int(dim * mlp_ratio))
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        t_x, t_y = init_t_xy(w, h)
        self.register_buffer("t_x", t_x, persistent=False)
        self.register_buffer("t_y", t_y, persistent=False)
        if mode == "mixed":
            self.freqs = nn.Parameter(init_random_2d_freqs(hd, heads, theta or 10.0))
        else:
            self.register_buffer(
                "cis", compute_axial_cis(hd, w, h, theta or 100.0), persistent=False
            )

    def freqs_cis(self):
        if self.mode != "mixed":
            return self.cis
        fx = self.t_x.unsqueeze(-1).unsqueeze(0) * self.freqs[0].unsqueeze(1)  # (heads, N, hd/2)
        fy = self.t_y.unsqueeze(-1).unsqueeze(0) * self.freqs[1].unsqueeze(1)
        ang = (fx + fy).float()
        return torch.polar(torch.ones_like(ang), ang)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        cis = self.freqs_cis()
        x = x + self.gamma_1 * self.attn(self.norm1(x), cis)
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x.transpose(1, 2).reshape(b, c, h, w)


@op("rope_vit_block", "map", space={"heads": [2, 4, 8], "mode": ["mixed", "axial"], "mlp_ratio": [4, 2]}, tags={"block"},
    source="naver-ai/rope-vit models/vit_rope.py RoPEAttention + RoPE-Mixed / axial (Heo et al., ECCV 2024)")  # fmt: skip
def _rope_block(dims, kind, heads, mode, mlp_ratio):
    c, h, w = dims
    if c % heads or (c // heads) % 4:
        raise ValueError(f"RoPE: head width {c}/{heads} must be a multiple of 4")
    return RoPEBlock(c, heads, h, w, mode, mlp_ratio)


# ---------------------------------------------------------------------------- WTConv

_S = 1 / math.sqrt(2)
HAAR = {
    "dec_lo": [_S, _S],
    "dec_hi": [-_S, _S],
    "rec_lo": [_S, _S],
    "rec_hi": [_S, -_S],
}  # pywt db1


def haar_2d_filters(c):
    """wavelet.create_2d_wavelet_filter('db1', c, c) without pywt."""
    dl, dh = torch.tensor(HAAR["dec_lo"][::-1]), torch.tensor(HAAR["dec_hi"][::-1])
    rl, rh = torch.tensor(HAAR["rec_lo"]), torch.tensor(HAAR["rec_hi"])
    dec = torch.stack(
        [
            dl[None] * dl[:, None],
            dl[None] * dh[:, None],
            dh[None] * dl[:, None],
            dh[None] * dh[:, None],
        ]
    )
    rec = torch.stack(
        [
            rl[None] * rl[:, None],
            rl[None] * rh[:, None],
            rh[None] * rl[:, None],
            rh[None] * rh[:, None],
        ]
    )
    return dec[:, None].repeat(c, 1, 1, 1), rec[:, None].repeat(c, 1, 1, 1)


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return self.weight * x


class WTConv2d(nn.Module):
    """wtconv2d.WTConv2d (stride 1, Haar): depthwise conv + depthwise convs on wavelet levels."""

    def __init__(self, c, kernel_size=5, bias=True, wt_levels=1):
        super().__init__()
        self.wt_levels = wt_levels
        dec, rec = haar_2d_filters(c)
        self.register_buffer("wt_filter", dec)
        self.register_buffer("iwt_filter", rec)
        self.base_conv = nn.Conv2d(c, c, kernel_size, padding="same", groups=c, bias=bias)
        self.base_scale = _ScaleModule([1, c, 1, 1])
        self.wavelet_convs = nn.ModuleList(
            nn.Conv2d(c * 4, c * 4, kernel_size, padding="same", groups=c * 4, bias=False)
            for _ in range(wt_levels)
        )
        self.wavelet_scale = nn.ModuleList(
            _ScaleModule([1, c * 4, 1, 1], 0.1) for _ in range(wt_levels)
        )

    def _wt(self, x):
        b, c, h, w = x.shape
        return F.conv2d(x, self.wt_filter, stride=2, groups=c).reshape(b, c, 4, h // 2, w // 2)

    def _iwt(self, x):
        b, c, _, h, w = x.shape
        return F.conv_transpose2d(x.reshape(b, c * 4, h, w), self.iwt_filter, stride=2, groups=c)

    def forward(self, x):
        lls, hs, shapes = [], [], []
        cur = x
        for i in range(self.wt_levels):
            shapes.append(cur.shape)
            if cur.shape[2] % 2 or cur.shape[3] % 2:
                cur = F.pad(cur, (0, cur.shape[3] % 2, 0, cur.shape[2] % 2))
            t = self._wt(cur)
            cur = t[:, :, 0]
            s = t.shape
            tag = self.wavelet_scale[i](
                self.wavelet_convs[i](t.reshape(s[0], s[1] * 4, s[3], s[4]))
            ).reshape(s)
            lls.append(tag[:, :, 0])
            hs.append(tag[:, :, 1:4])
        nxt = 0
        for _ in range(self.wt_levels):
            ll, hh, sh = lls.pop(), hs.pop(), shapes.pop()
            nxt = self._iwt(torch.cat([(ll + nxt).unsqueeze(2), hh], dim=2))[:, :, : sh[2], : sh[3]]
        return self.base_scale(self.base_conv(x)) + nxt


class WTConvNeXtBlock(nn.Module):
    """wtconvnext.WTConvNeXtBlock (conv_mlp False): WTConv, LayerNorm, Mlp, layer scale, residual."""

    def __init__(self, c, kernel_size=5, wt_levels=3, mlp_ratio=4, ls_init_value=1e-6):
        super().__init__()
        self.conv_dw = WTConv2d(c, kernel_size, True, wt_levels)
        self.norm = nn.LayerNorm(c, eps=1e-6)
        self.mlp = Mlp(c, int(mlp_ratio * c))
        self.gamma = nn.Parameter(ls_init_value * torch.ones(c))

    def forward(self, x):
        y = self.mlp(self.norm(self.conv_dw(x).permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x + y * self.gamma.reshape(1, -1, 1, 1)


def _wt_levels_ok(h, w, levels):
    for _ in range(levels):
        if min(h, w) < 2:
            return False
        h, w = (h + 1) // 2, (w + 1) // 2
    return True


@op("wtconv", "map", space={"k": [5, 3], "levels": [1, 2, 3]}, tags={"attn"},
    source="BGU-CS-VIL/WTConv wtconv/wtconv2d.py WTConv2d, db1 (Finder et al., ECCV 2024)")  # fmt: skip
def _wtconv(dims, kind, k, levels):
    if not _wt_levels_ok(dims[1], dims[2], levels):
        raise ValueError(f"WTConv: {levels} wavelet levels need a larger map than {dims[1:]}")
    return WTConv2d(dims[0], k, True, levels)


@op("wtconvnext_block", "map", space={"k": [5, 3], "levels": [2, 1, 3], "mlp_ratio": [4, 2]}, tags={"block"},
    source="BGU-CS-VIL/WTConv wtconvnext/wtconvnext.py WTConvNeXtBlock")  # fmt: skip
def _wtconvnext(dims, kind, k, levels, mlp_ratio):
    if not _wt_levels_ok(dims[1], dims[2], levels):
        raise ValueError(f"WTConv: {levels} wavelet levels need a larger map than {dims[1:]}")
    return WTConvNeXtBlock(dims[0], k, levels, mlp_ratio)


# ------------------------------------------------------------------- MobileNetV4 (timm)


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class ConvNormAct(nn.Module):
    """timm ConvNormAct (conv, BatchNormAct2d) with the same parameter names."""

    def __init__(self, cin, cout, k, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, 1, (k - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UIB(nn.Module):
    """timm UniversalInvertedResidual (stride 1, depthwise groups, ReLU, layer scale 1e-5)."""

    def __init__(self, c, start=0, mid=3, end=0, exp_ratio=4.0, layer_scale_init_value=1e-5):
        super().__init__()
        mid_c = make_divisible(c * exp_ratio)
        self.dw_start = ConvNormAct(c, c, start, c, act=False) if start else nn.Identity()
        self.pw_exp = ConvNormAct(c, mid_c, 1)
        self.dw_mid = ConvNormAct(mid_c, mid_c, mid, mid_c) if mid else nn.Identity()
        self.pw_proj = ConvNormAct(mid_c, c, 1, act=False)
        self.dw_end = ConvNormAct(c, c, end, c, act=False) if end else nn.Identity()
        self.layer_scale = LayerScale2d(c, layer_scale_init_value)

    def forward(self, x):
        y = self.dw_end(self.pw_proj(self.dw_mid(self.pw_exp(self.dw_start(x)))))
        return x + self.layer_scale(y)


class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma.view(1, -1, 1, 1)


UIB_VARIANTS = {"extra_dw": (3, 3, 0), "ib": (0, 3, 0), "convnext": (3, 0, 0), "ffn": (0, 0, 0)}


@op("uib", "map", space={"variant": list(UIB_VARIANTS), "exp_ratio": [4.0, 2.0, 6.0], "k": [3, 5]}, tags={"block"},
    source="timm 1.0.31 _efficientnet_blocks.UniversalInvertedResidual (MobileNetV4, Qin et al., ECCV 2024)")  # fmt: skip
def _uib(dims, kind, variant, exp_ratio, k):
    s, m, e = (k if v else 0 for v in UIB_VARIANTS[variant])
    return UIB(dims[0], s, m, e, exp_ratio)


class MultiQueryAttention2d(nn.Module):
    """timm layers.MultiQueryAttention2d (query stride 1): one shared key/value head."""

    def __init__(self, dim, num_heads, key_dim, value_dim, kv_stride=1, dw_kernel_size=3):
        super().__init__()
        self.num_heads, self.key_dim, self.value_dim = num_heads, key_dim, value_dim
        self.scale = key_dim**-0.5
        self.query = nn.Sequential()
        self.query.add_module("proj", nn.Conv2d(dim, num_heads * key_dim, 1, bias=False))
        self.key, self.value = nn.Sequential(), nn.Sequential()
        for seq, out in ((self.key, key_dim), (self.value, value_dim)):
            if kv_stride > 1:
                seq.add_module(
                    "down_conv",
                    nn.Conv2d(
                        dim,
                        dim,
                        dw_kernel_size,
                        kv_stride,
                        (dw_kernel_size - 1) // 2,
                        groups=dim,
                        bias=False,
                    ),
                )
                seq.add_module("norm", nn.BatchNorm2d(dim))
            seq.add_module("proj", nn.Conv2d(dim, out, 1, bias=False))
        self.output = nn.Sequential()
        self.output.add_module("proj", nn.Conv2d(value_dim * num_heads, dim, 1, bias=False))
        for m in (self.query.proj, self.key.proj, self.value.proj, self.output.proj):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        b, _, h, w = x.shape
        q = self.query(x).reshape(b, self.num_heads, self.key_dim, -1).transpose(-1, -2)  # b h n k
        k = self.key(x).reshape(b, self.key_dim, -1).transpose(1, 2).unsqueeze(1)  # b 1 p k
        v = self.value(x).reshape(b, self.value_dim, -1).transpose(1, 2).unsqueeze(1)
        a = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        o = (a @ v).transpose(1, 2).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return self.output(o)


class MobileMQA(nn.Module):
    """timm MobileAttention(use_multi_query=True): BN, MQA, layer scale, residual."""

    def __init__(self, c, heads, key_dim, kv_stride):
        super().__init__()
        self.norm = nn.BatchNorm2d(c)
        self.attn = MultiQueryAttention2d(c, heads, key_dim, key_dim, kv_stride)
        self.layer_scale = LayerScale2d(c, 1e-5)

    def forward(self, x):
        return x + self.layer_scale(self.attn(self.norm(x)))


@op("mobile_mqa", "map", space={"heads": [4, 2, 8], "key_dim": [16, 32, 64], "kv_stride": [1, 2]}, tags={"attn"},
    source="timm 1.0.31 MobileAttention + MultiQueryAttention2d (MobileNetV4 Mobile MQA)")  # fmt: skip
def _mqa(dims, kind, heads, key_dim, kv_stride):
    if kv_stride > 1 and min(dims[1:]) < 2:
        raise ValueError("MQA: kv_stride 2 needs a map of at least 2x2")
    return MobileMQA(dims[0], heads, key_dim, kv_stride)


# ------------------------------------------------------------------------------ LSNet


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module("c", nn.Conv2d(a, b, ks, stride, pad, 1, groups, bias=False))
        self.add_module("bn", nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


def ska(x, w):
    """LSNet SKA (Triton in the original): out[n,c,i,j] = sum_k x[n,c,i+di,j+dj] * w[n, c % G, k, i, j],
    zero padding, k over the ks x ks window; w (n, G, ks*ks, h, w)."""
    n, c, h, wd = x.shape
    g, kk = w.shape[1], w.shape[2]
    ks = math.isqrt(kk)
    cols = F.unfold(x, ks, padding=(ks - 1) // 2).view(n, c, kk, h, wd)
    return (cols * w.repeat(1, c // g, 1, 1, 1)).sum(2)


class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks**2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks**2 * dim // groups)
        self.sks, self.groups, self.dim = sks, groups, dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        return w.view(b, self.dim // self.groups, self.sks**2, h, width)


class LSConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(ska(x, self.lkp(x))) + x


class SqueezeExcite(nn.Module):
    """timm SEModule (rd_ratio 0.25, divisor 8, ReLU, sigmoid)."""

    def __init__(self, c, rd_ratio=0.25):
        super().__init__()
        rd = make_divisible(c * rd_ratio, 8, round_limit=0.0)
        self.fc1, self.act, self.fc2 = (
            nn.Conv2d(c, rd, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(rd, c, 1),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True)))))


class RepVGGDW(nn.Module):
    def __init__(self, ed):
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x


class _Res(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return x + self.m(x)


class LSFFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1, self.act, self.pw2 = (
            Conv2d_BN(ed, h),
            nn.ReLU(),
            Conv2d_BN(h, ed, bn_weight_init=0),
        )

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class LSBlock(nn.Module):
    """lsnet.Block (stage < 3): even depth = RepVGG-DW + SE, odd depth = LS convolution; FFN."""

    def __init__(self, ed, mixer="ls"):
        super().__init__()
        if mixer == "repvgg":
            self.mixer, self.se = RepVGGDW(ed), SqueezeExcite(ed, 0.25)
        else:
            self.mixer, self.se = LSConv(ed), nn.Identity()
        self.ffn = _Res(LSFFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))


@op("lsnet_block", "map", space={"mixer": ["ls", "repvgg"]}, tags={"block"},
    source="jameslahm/lsnet model/lsnet.py Block, LSConv, LKP, SKA (Wang et al., CVPR 2025)")  # fmt: skip
def _ls(dims, kind, mixer):
    if mixer == "ls" and (dims[0] % 8):
        raise ValueError(
            "LS conv needs channels divisible by 8 (its kernels are shared by groups of 8)"
        )
    return LSBlock(dims[0], mixer)


# ---------------------------------------------------------------------------- StarNet


class ConvBN(nn.Sequential):
    def __init__(self, a, b, k=1, s=1, p=0, groups=1, with_bn=True):
        super().__init__()
        self.add_module("conv", nn.Conv2d(a, b, k, s, p, 1, groups))
        if with_bn:
            self.add_module("bn", nn.BatchNorm2d(b))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class StarBlock(nn.Module):
    """starnet.Block: dw7, two 1x1 branches, ReLU6(x1) * x2 (the star), 1x1, dw7, residual."""

    def __init__(self, dim, mlp_ratio=3):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        self.act = nn.ReLU6()

    def forward(self, x):
        y = self.dwconv(x)
        return x + self.dwconv2(self.g(self.act(self.f1(y)) * self.f2(y)))


@op("star_block", "map", space={"mlp_ratio": [3, 4, 2]}, tags={"block"},
    source="ma-xu/Rewrite-the-Stars imagenet/starnet.py Block (Ma et al., CVPR 2024)")  # fmt: skip
def _star(dims, kind, mlp_ratio):
    return StarBlock(dims[0], mlp_ratio)


# ------------------------------------------------------------------------ EfficientViT


class EVConv(nn.Module):
    """efficientvit ConvLayer (conv, optional BN, optional act) with the same names."""

    def __init__(self, cin, cout, k=1, groups=1, use_bias=False, norm=True, act=None):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, 1, k // 2, groups=groups, bias=use_bias)
        self.norm = nn.BatchNorm2d(cout) if norm else None
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x) if self.act is not None else x


class LiteMLA(nn.Module):
    """efficientvit LiteMLA: multi-scale q/k/v (aggregation by 5x5 dw conv), ReLU linear attention."""

    def __init__(self, c, dim=8, scales=(5,), eps=1.0e-15):
        super().__init__()
        heads = c // dim
        if heads < 1:
            raise ValueError(f"LiteMLA: {c} channels < head dim {dim}")
        total = heads * dim
        self.eps, self.dim = eps, dim
        self.qkv = EVConv(c, 3 * total, 1, norm=False)
        self.aggreg = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(3 * total, 3 * total, s, padding=s // 2, groups=3 * total, bias=False),
                nn.Conv2d(3 * total, 3 * total, 1, groups=3 * heads, bias=False),
            )
            for s in scales
        )
        self.proj = EVConv(total * (1 + len(scales)), c, 1, norm=True)

    def _linear(self, qkv):
        b, _, h, w = qkv.shape
        qkv = qkv.float().reshape(b, -1, 3 * self.dim, h * w)
        q, k, v = (
            F.relu(qkv[:, :, : self.dim]),
            F.relu(qkv[:, :, self.dim : 2 * self.dim]),
            qkv[:, :, 2 * self.dim :],
        )
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        out = torch.matmul(torch.matmul(v, k.transpose(-1, -2)), q)
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return out.reshape(b, -1, h, w)

    def _quadratic(self, qkv):
        b, _, h, w = qkv.shape
        qkv = qkv.reshape(b, -1, 3 * self.dim, h * w)
        q, k, v = (
            F.relu(qkv[:, :, : self.dim]),
            F.relu(qkv[:, :, self.dim : 2 * self.dim]),
            qkv[:, :, 2 * self.dim :],
        )
        m = torch.matmul(k.transpose(-1, -2), q)
        m = (m.float() / (torch.sum(m.float(), dim=2, keepdim=True) + self.eps)).to(m.dtype)
        return torch.matmul(v, m).reshape(b, -1, h, w)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv = torch.cat([qkv] + [op(qkv) for op in self.aggreg], dim=1)
        h, w = qkv.shape[-2:]
        out = self._linear(qkv).to(qkv.dtype) if h * w > self.dim else self._quadratic(qkv)
        return self.proj(out)


class EVMBConv(nn.Module):
    """efficientvit MBConv with use_bias (True, True, False), norm (None, None, bn), hswish."""

    def __init__(self, c, expand_ratio=4):
        super().__init__()
        mid = round(c * expand_ratio)
        self.inverted_conv = EVConv(c, mid, 1, use_bias=True, norm=False, act=nn.Hardswish())
        self.depth_conv = EVConv(
            mid, mid, 3, groups=mid, use_bias=True, norm=False, act=nn.Hardswish()
        )
        self.point_conv = EVConv(mid, c, 1, use_bias=False, norm=True)

    def forward(self, x):
        return self.point_conv(self.depth_conv(self.inverted_conv(x)))


class _ResMain(nn.Module):  # efficientvit ResidualBlock(main, IdentityLayer())
    def __init__(self, main):
        super().__init__()
        self.main = main

    def forward(self, x):
        return self.main(x) + x


class EfficientViTBlock(nn.Module):
    def __init__(self, c, dim=8, expand_ratio=4):
        super().__init__()
        self.context_module = _ResMain(LiteMLA(c, dim))
        self.local_module = _ResMain(EVMBConv(c, expand_ratio))

    def forward(self, x):
        return self.local_module(self.context_module(x))


@op("efficientvit_block", "map", space={"dim": [8, 16, 32], "expand_ratio": [4, 2]}, tags={"block"},
    source="mit-han-lab/efficientvit models/nn/ops.py EfficientViTBlock + LiteMLA (Cai et al., ICCV 2023)")  # fmt: skip
def _evit(dims, kind, dim, expand_ratio):
    return EfficientViTBlock(dims[0], dim, expand_ratio)


# ---------------------------------------------------------------------------- SegNeXt


class AttentionModule(nn.Module):
    """mscan.AttentionModule: 5x5 dw, strip convs 1x7/7x1, 1x11/11x1, 1x21/21x1, 1x1, multiply."""

    def __init__(self, dim):
        super().__init__()
        dw = lambda k: nn.Conv2d(dim, dim, k, padding=tuple((s - 1) // 2 for s in k), groups=dim)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1, self.conv0_2 = dw((1, 7)), dw((7, 1))
        self.conv1_1, self.conv1_2 = dw((1, 11)), dw((11, 1))
        self.conv2_1, self.conv2_2 = dw((1, 21)), dw((21, 1))
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        a = self.conv0(x)
        a = (
            a
            + self.conv0_2(self.conv0_1(a))
            + self.conv1_2(self.conv1_1(a))
            + self.conv2_2(self.conv2_1(a))
        )
        return self.conv3(a) * x


class SpatialAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj_1, self.activation = nn.Conv2d(d, d, 1), nn.GELU()
        self.spatial_gating_unit, self.proj_2 = AttentionModule(d), nn.Conv2d(d, d, 1)

    def forward(self, x):
        return self.proj_2(self.spatial_gating_unit(self.activation(self.proj_1(x)))) + x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class MSCAMlp(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1, self.dwconv, self.act, self.fc2 = (
            nn.Conv2d(d, hidden, 1),
            DWConv(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, d, 1),
        )

    def forward(self, x):
        return self.fc2(self.act(self.dwconv(self.fc1(x))))


class MSCABlock(nn.Module):
    """mscan.Block on a map: BN, SpatialAttention (MSCA), layer scale; BN, Mlp with dw conv."""

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.attn = nn.BatchNorm2d(dim), SpatialAttention(dim)
        self.norm2, self.mlp = nn.BatchNorm2d(dim), MSCAMlp(dim, int(dim * mlp_ratio))
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x):
        x = x + self.layer_scale_1.view(1, -1, 1, 1) * self.attn(self.norm1(x))
        return x + self.layer_scale_2.view(1, -1, 1, 1) * self.mlp(self.norm2(x))


@op("msca", "map", space={}, tags={"attn"},
    source="Visual-Attention-Network/SegNeXt mmseg/models/backbones/mscan.py SpatialAttention (Guo et al., NeurIPS 2022)")  # fmt: skip
def _msca(dims, kind):
    return SpatialAttention(dims[0])


@op("msca_block", "map", space={"mlp_ratio": [4, 8, 2]}, tags={"block"},
    source="Visual-Attention-Network/SegNeXt mmseg/models/backbones/mscan.py Block (MSCAN)")  # fmt: skip
def _msca_block(dims, kind, mlp_ratio):
    return MSCABlock(dims[0], mlp_ratio)
