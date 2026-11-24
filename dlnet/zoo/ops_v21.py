"""Sequence mixers of 2024-2026 written from their reference code (v2.1; THIRD_PARTY.md, "v2.1").

All are pure PyTorch: the CUDA / Triton kernels of the originals are replaced by the recurrent
references the same repositories ship to test those kernels (scripts/verify_v21.py compares each
layer with the original layer whose kernel is swapped for that reference, job `verify-v21`).

- RWKV-7 "Goose" (Peng et al. 2025, arXiv:2503.14456): time mixing with the generalised delta rule
  (vector decay, in-context learning rate), from fla-org/flash-linear-attention 0.6.0
  fla/layers/rwkv7.py RWKV7Attention (recurrence: fla/ops/generalized_delta_rule/dplr/naive.py),
  channel mixing from fla/models/rwkv7 RWKV7FeedForward. Each DLNet block is its own first layer:
  no value residual (v_first) across blocks.
- Kimi Delta Attention (Kimi Team 2025, "Kimi Linear", arXiv:2510.26692): gated delta rule with one
  forget gate per key channel; fla/layers/kda.py KimiDeltaAttention (fla/ops/kda/naive.py).
- Forgetting Transformer (Lin et al., ICLR 2025, arXiv:2503.02130): softmax attention with a
  data-dependent forget gate; fla/layers/forgetting_attn.py (fla/ops/forgetting_attn/naive.py).
- Multi-head Latent Attention (DeepSeek-V2, arXiv:2405.04434; DeepSeek-V3): keys and values from a
  low-rank latent, decoupled rotary key; fla/layers/mla.py (flash-attn replaced by exact causal
  softmax attention).
- Mamba-3 (state-spaces, 2025): trapezoidal discretisation, data-dependent rotary
  state (complex SSM), B / C RMSNorm and bias, no short convolution; state-spaces/mamba
  mamba_ssm/modules/mamba3.py (SISO), recurrence tests/ops/triton/test_mamba3_siso.py
  mamba3_siso_step_ref.

`scan2d_block` (DLNet composition) runs one of these mixers over the pixels of a feature map in
both row-major directions and averages the two (bidirectional scan as in Vision Mamba, Zhu et al.
ICML 2024, and Vision-LSTM, Alkin et al. ICLR 2025), so image and HSI models can use them.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import op
from .ops_next import RMSNorm, SwiGLU
from .ops_seq2 import ShortConvolution, l2_norm

DECAY_FLOOR = -0.6065306597126334  # RWKV-7: log decay in [-exp(-0.5), 0]


def _check_heads(name, d, heads):
    if heads < 1 or d % heads:
        raise ValueError(f"{name}: width {d} not divisible by {heads} heads")
    return d // heads


class GatedRMSNorm(nn.Module):
    """fla FusedRMSNormGated: RMSNorm(x) * weight * act(g), act = sigmoid (KDA) or swish."""

    def __init__(self, d, activation="sigmoid", eps=1e-5):
        super().__init__()
        self.eps, self.activation = eps, activation
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, g):
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        gate = torch.sigmoid(g.float()) if self.activation == "sigmoid" else F.silu(g.float())
        return (y * self.weight * gate).to(x.dtype)


# -------------------------------------------------------------------------- RWKV-7


def dplr_recurrence(q, k, v, alpha, beta, gk, scale=1.0):
    """fla dplr_recurrence: S <- S diag(exp(gk)) + k v^T + beta (alpha^T S); o = (scale q) S.
    All (B, H, L, D); returns (B, H, L, Dv)."""
    q, k, v, alpha, beta, gk = (t.float() for t in (q, k, v, alpha, beta, gk))
    s = q.new_zeros(*q.shape[:2], q.shape[-1], v.shape[-1])
    out = []
    for i in range(q.shape[2]):
        kv = k[:, :, i, :, None] * v[:, :, i, None, :]
        kv = kv + (s * alpha[:, :, i, :, None]).sum(-2, keepdim=True) * beta[:, :, i, :, None]
        s = s * gk[:, :, i].exp()[..., None] + kv
        out.append(torch.einsum("bhd,bhdm->bhm", q[:, :, i] * scale, s))
    return torch.stack(out, 2)


class LoRA(nn.Module):
    """fla LoRA: Linear(d, r, no bias) -> activation -> Linear(r, out)."""

    def __init__(self, d, out, rank, activation=None, bias=True):
        super().__init__()
        act = {"tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), None: nn.Identity()}[activation]
        self.lora = nn.Sequential(
            nn.Linear(d, rank, bias=False), act, nn.Linear(rank, out, bias=bias)
        )
        nn.init.zeros_(self.lora[0].weight)
        gain = (
            math.sqrt(rank / out) if rank > out else 1.0
        )  # fla: weight (out, rank), sqrt(rank / out) if wider
        nn.init.orthogonal_(self.lora[2].weight, gain=0.1 * gain)
        if bias:
            nn.init.zeros_(self.lora[2].bias)

    def forward(self, x):
        return self.lora(x)


def _rank(d, scale, factor=1.0):
    return max(32, int(round(scale * d**0.5 * factor / 32) * 32))


class RWKV7Attention(nn.Module):
    """fla RWKV7Attention of the first layer (layer_idx 0 of 2 for the initialisation): token shift,
    six interpolations, low-rank decay / in-context rate / gate, generalised delta rule."""

    def __init__(self, d, heads):
        super().__init__()
        hd = _check_heads("RWKV-7", d, heads)
        self.h, self.hd = heads, hd
        factor = hd / 64
        for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, n, nn.Parameter(torch.zeros(1, 1, d)))
        self.k_k, self.k_a = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.r_k = nn.Parameter(torch.zeros(heads, hd))
        self.r_proj, self.k_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.v_proj, self.o_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.w_lora = LoRA(d, d, _rank(d, 2.5, factor), "tanh")
        self.a_lora = LoRA(d, d, _rank(d, 2.5, factor))
        self.g_lora = LoRA(d, d, _rank(d, 5.0), "sigmoid", bias=False)
        self.g_norm = nn.GroupNorm(heads, d, eps=hd * 1e-5)
        with (
            torch.no_grad()
        ):  # fla _initialize_weights, layer 0 of 2 (ratio_0_to_1 = 0, ratio_1_to_0 = 1)
            n = torch.arange(d, dtype=torch.float32)
            ddd = (n / d).view(1, 1, d)
            linear = n / max(d - 1, 1) - 0.5
            zig = ((n % hd) - (hd - 1) / 2) / max((hd - 1) / 2, 1e-6)
            zig = zig * zig.abs()
            www = -6 + 6 * (n / max(d - 1, 1)) ** 1.0
            for name, p in (
                ("x_r", 0.2),
                ("x_w", 0.9),
                ("x_k", 0.7),
                ("x_v", 0.7),
                ("x_a", 0.9),
                ("x_g", 0.2),
            ):
                getattr(self, name).copy_(1.0 - ddd**p)
            self.k_a.fill_(1.02)
            self.r_k.fill_(-0.04)
            self.k_k.copy_(0.71 - linear * 0.1)
            self.w_lora.lora[2].bias.copy_(www + 0.5 + zig * 2.5)
            self.a_lora.lora[2].bias.copy_(-0.19 + zig * 0.3 + linear * 0.4)
            self.g_norm.weight.fill_(0.5**0.7)
            nn.init.orthogonal_(self.r_proj.weight)
            nn.init.orthogonal_(self.k_proj.weight, gain=0.1)
            nn.init.orthogonal_(self.v_proj.weight)
            self.o_proj.weight.zero_()

    def forward(self, x):
        b, n, d = x.shape
        delta = F.pad(x, (0, 0, 1, -1)) - x  # fla token_shift_ref: previous token - current
        xr, xw, xk, xv, xa, xg = (
            torch.addcmul(x, delta, getattr(self, "x_" + c)) for c in "rwkvag"
        )
        r, k, v = self.r_proj(xr), self.k_proj(xk), self.v_proj(xv)
        w = DECAY_FLOOR * self.w_lora(xw).sigmoid()
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)
        heads = lambda t: t.view(b, n, self.h, self.hd)
        kk = F.normalize(heads(k * self.k_k), dim=-1, p=2.0)
        k = k.addcmul(k * (a - 1), self.k_a)  # fla k_update_ref
        r, w, k, a, vh = heads(r), heads(w), heads(k), heads(a), heads(v)
        tr = lambda t: t.transpose(1, 2)
        o = (
            dplr_recurrence(tr(r), tr(k), tr(vh), tr(-kk), tr(kk * a), tr(w))
            .transpose(1, 2)
            .to(x.dtype)
        )
        o = self.g_norm(o.reshape(b * n, d)).view(b, n, d)
        o = (
            o + ((r * k * self.r_k).sum(-1, keepdim=True) * vh).view(b, n, d)
        ) * g  # gate_output_correction_ref
        return self.o_proj(o)


class RWKV7FeedForward(nn.Module):
    """fla RWKV7FeedForward: value(relu(key(x + shift * x_k))^2), hidden 4 d rounded up to 32."""

    def __init__(self, d, ratio=4):
        super().__init__()
        hidden = 32 * ((d * ratio + 31) // 32)
        self.x_k = nn.Parameter(1.0 - (torch.arange(d, dtype=torch.float32) / d) ** 1.0)
        self.key, self.value = nn.Linear(d, hidden, bias=False), nn.Linear(hidden, d, bias=False)
        nn.init.orthogonal_(self.key.weight)
        nn.init.zeros_(self.value.weight)

    def forward(self, x):
        delta = F.pad(x, (0, 0, 1, -1)) - x
        return self.value(F.relu(self.key(x.addcmul(delta, self.x_k))) ** 2)


class RWKV7Block(nn.Module):
    """fla RWKV7Block: x + attn(LN x); x + ffn(LN x)."""

    def __init__(self, d, heads, ratio=4):
        super().__init__()
        self.attn_norm, self.attn = nn.LayerNorm(d, eps=1e-5), RWKV7Attention(d, heads)
        self.ffn_norm, self.ffn = nn.LayerNorm(d, eps=1e-5), RWKV7FeedForward(d, ratio)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


# --------------------------------------------------------------------- Kimi Delta Attention


def recurrent_kda(q, k, v, g, beta):
    """fla naive_recurrent_kda: S <- S exp(g) (per key channel); S += beta k (v - k^T S)^T;
    o = (q / sqrt(d_k)) S. q, k, g (B, L, H, K); v (B, L, H, V); beta (B, L, H)."""
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    q = q * q.shape[-1] ** -0.5
    s = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1])
    out = []
    for i in range(q.shape[1]):
        s = s * g[:, i, ..., None].exp()
        upd = v[:, i] - (k[:, i, ..., None] * s).sum(-2)
        s = s + torch.einsum("bhk,bhv->bhkv", beta[:, i, :, None] * k[:, i], upd)
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, i], s))
    return torch.stack(out, 1)


class KimiDeltaAttention(nn.Module):
    """fla KimiDeltaAttention (short convolutions, l2-normalised q / k, sigmoid beta, gate
    -exp(A_log) softplus(f + dt_bias), sigmoid-gated RMSNorm on the output, no GVA)."""

    def __init__(self, d, heads, expand_v=1.0, conv_size=4):
        super().__init__()
        hd = _check_heads("KDA", d, heads)
        self.h, self.hd, self.hv = heads, hd, int(hd * expand_v)
        key_dim, value_dim = heads * hd, heads * self.hv
        self.q_proj, self.k_proj = (
            nn.Linear(d, key_dim, bias=False),
            nn.Linear(d, key_dim, bias=False),
        )
        self.v_proj = nn.Linear(d, value_dim, bias=False)
        self.q_conv1d, self.k_conv1d = (
            ShortConvolution(key_dim, conv_size),
            ShortConvolution(key_dim, conv_size),
        )
        self.v_conv1d = ShortConvolution(value_dim, conv_size)
        self.f_proj = nn.Sequential(
            nn.Linear(d, self.hv, bias=False), nn.Linear(self.hv, key_dim, bias=False)
        )
        self.b_proj = nn.Linear(d, heads, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(1, 16)))
        dt = torch.exp(
            torch.rand(key_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.g_proj = nn.Sequential(
            nn.Linear(d, self.hv, bias=False), nn.Linear(self.hv, value_dim, bias=True)
        )
        self.o_norm = GatedRMSNorm(self.hv, "sigmoid")
        self.o_proj = nn.Linear(value_dim, d, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        q, k = self.q_conv1d(self.q_proj(x)), self.k_conv1d(self.k_proj(x))
        v = self.v_conv1d(self.v_proj(x))
        sh = lambda t, dd: t.view(b, n, self.h, dd)
        g = sh(self.f_proj(x), self.hd).float() + self.dt_bias.view(self.h, self.hd)
        g = -self.A_log.view(self.h, 1).float().exp() * F.softplus(g)  # fla naive_kda_gate
        beta = self.b_proj(x).float().sigmoid()
        o = recurrent_kda(
            l2_norm(sh(q, self.hd)), l2_norm(sh(k, self.hd)), sh(v, self.hv), g, beta
        ).to(x.dtype)
        o = self.o_norm(o, sh(self.g_proj(x), self.hv))
        return self.o_proj(o.reshape(b, n, -1))


# ------------------------------------------------------------------------ attention


def _causal_softmax(q, k, v, scale, bias=None):
    """Exact causal softmax attention, (B, L, H, D) in and out, optional additive (B, H, L, L) bias."""
    s = torch.einsum("bqhd,bkhd->bhqk", q.float() * scale, k.float())
    if bias is not None:
        s = s + bias
    n = q.shape[1]
    s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v.float()).to(v.dtype)


class ForgettingAttention(nn.Module):
    """fla ForgettingAttention: softmax(q k^T / sqrt(d) + F_i - F_j) v, F = cumsum logsigmoid(f_proj x);
    optional sigmoid output gate (FoX "Pro" component). No qk norm, no window."""

    def __init__(self, d, heads, output_gate=False):
        super().__init__()
        self.h, self.hd = heads, _check_heads("FoX", d, heads)
        self.q_proj, self.k_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.f_proj = nn.Linear(d, heads, bias=True)
        self.g_proj = nn.Linear(d, d, bias=False) if output_gate else None
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, n, d = x.shape
        sh = lambda t: t.view(b, n, self.h, self.hd)
        gc = F.logsigmoid(self.f_proj(x).float()).cumsum(1).transpose(1, 2)  # (b, h, n)
        o = _causal_softmax(sh(self.q_proj(x)), sh(self.k_proj(x)), sh(self.v_proj(x)), self.hd**-0.5,
                            gc[..., :, None] - gc[..., None, :]).reshape(b, n, d)  # fmt: skip
        if self.g_proj is not None:
            o = self.g_proj(x).sigmoid() * o
        return self.o_proj(o)


def _rotary(x, base=10000.0):
    """fla RotaryEmbedding (not interleaved) at positions 0..L-1 on the whole last axis of (B, L, H, D)."""
    dim, n = x.shape[-1], x.shape[1]
    inv = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim)
    f = torch.outer(torch.arange(n, dtype=torch.float32, device=x.device), inv)
    cos, sin = (
        torch.cat([f.cos()] * 2, -1)[None, :, None],
        torch.cat([f.sin()] * 2, -1)[None, :, None],
    )
    x1, x2 = x.float().chunk(2, -1)
    return (x.float() * cos + torch.cat((-x2, x1), -1) * sin).to(x.dtype)


class MultiheadLatentAttention(nn.Module):
    """fla MultiheadLatentAttention (DeepSeek-V2): q (optionally through a low-rank latent), keys and
    values up-projected from a kv latent (RMSNorm), one shared rotary key, causal softmax."""

    def __init__(self, d, heads, kv_rank, q_rank=0, rope_dim=0):
        super().__init__()
        self.h = heads
        hd = _check_heads("MLA", d, heads)
        self.nope, self.rope, self.vd = hd, rope_dim or max(2, hd // 2 // 2 * 2), hd
        if self.rope % 2:
            raise ValueError(f"MLA: rotary dimension {self.rope} must be even")
        self.qk = self.nope + self.rope
        self.q_proj = (nn.Sequential(nn.Linear(d, q_rank, bias=False), RMSNorm(q_rank, 1e-5),
                                     nn.Linear(q_rank, heads * self.qk, bias=False))
                       if q_rank else nn.Linear(d, heads * self.qk, bias=False))  # fmt: skip
        self.k_rope = nn.Linear(d, self.rope, bias=False)
        self.kv_proj = nn.Sequential(nn.Linear(d, kv_rank, bias=False), RMSNorm(kv_rank, 1e-5),
                                     nn.Linear(kv_rank, heads * (self.nope + self.vd), bias=False))  # fmt: skip
        self.o_proj = nn.Linear(heads * self.vd, d, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.h, self.qk)
        q_pass, q_rot = q.split([self.nope, self.rope], -1)
        kv = self.kv_proj(x).view(b, n, self.h, self.nope + self.vd)
        k_pass, v = kv.split([self.nope, self.vd], -1)
        k_rot = _rotary(self.k_rope(x).view(b, n, 1, self.rope)).expand(-1, -1, self.h, -1)
        q = torch.cat((q_pass, _rotary(q_rot)), -1)
        k = torch.cat((k_pass, k_rot), -1)
        o = _causal_softmax(q, k, v, self.qk**-0.5)
        return self.o_proj(o.reshape(b, n, -1))


class AttnBlock(nn.Module):
    """LLaMA-style block around a token mixer: x + mixer(RMSNorm x); x + SwiGLU(RMSNorm x)."""

    def __init__(self, d, mixer, ratio):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(d, 1e-5), mixer
        self.mlp_norm, self.mlp = RMSNorm(d, 1e-5), SwiGLU(d, d * ratio)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


# --------------------------------------------------------------------------- Mamba-3


def heavy_tail_activation(x):
    """mamba_ssm Mamba-3: 1 + x for x >= 0, 1 / (1 - x) for x < 0."""
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


def _pair_rotate(t, cos, sin):
    """Rotate consecutive pairs (t0, t1) of the first 2 * angles channels; the rest unchanged."""
    pad = t.shape[-1] // 2 - cos.shape[-1]
    cos, sin = F.pad(cos, (0, pad), value=1.0), F.pad(sin, (0, pad), value=0.0)
    p = t.view(*t.shape[:-1], -1, 2)
    t0, t1 = p[..., 0], p[..., 1]
    return torch.stack([t0 * cos - t1 * sin, t0 * sin + t1 * cos], -1).view_as(t)


def mamba3_recurrence(q, k, v, adt, dt, trap, q_bias, k_bias, angles, d_skip, z):
    """mamba3_siso_step_ref: trapezoidal update of S (headdim x state) with the previous and the
    current rotated key / value. q, k (B, L, H, N); v, z (B, L, H, P); adt, dt, trap (B, H, L);
    angles (B, L, H, A); biases (H, N); D (H,)."""
    two_pi = 2 * math.pi
    angles = torch.tanh(angles) * math.pi
    b, n, h, p = v.shape
    s = v.new_zeros(b, h, p, q.shape[-1], dtype=torch.float32)
    ang = v.new_zeros(b, h, angles.shape[-1], dtype=torch.float32)
    k_prev, v_prev = torch.zeros_like(k[:, 0]), torch.zeros_like(v[:, 0])
    out = []
    for i in range(n):
        ang = ang + angles[:, i] * dt[:, :, i, None]
        ang = ang - two_pi * torch.floor(ang / two_pi)
        cos, sin = ang.cos(), ang.sin()
        qr, kr = _pair_rotate(q[:, i] + q_bias, cos, sin), _pair_rotate(k[:, i] + k_bias, cos, sin)
        tr = torch.sigmoid(trap[:, :, i])
        alpha = adt[:, :, i].exp()
        beta, gamma = (1 - tr) * dt[:, :, i] * alpha, tr * dt[:, :, i]
        s = alpha[..., None, None] * s + beta[..., None, None] * (
            k_prev[:, :, None, :] * v_prev[..., None]
        )
        s = s + gamma[..., None, None] * (kr[:, :, None, :] * v[:, i, ..., None])
        y = torch.einsum("bhpn,bhn->bhp", s, qr.float()) + d_skip[None, :, None] * v[:, i]
        out.append(y * z[:, i] * torch.sigmoid(z[:, i]))
        k_prev, v_prev = kr, v[:, i]
    return torch.stack(out, 1)


class Mamba3(nn.Module):
    """mamba_ssm Mamba3, SISO, one B / C group, no output norm (the default), pure PyTorch."""

    def __init__(self, d, d_state=16, expand=2, headdim=16, rope_fraction=0.5, A_floor=1e-4):
        super().__init__()
        self.di = int(expand * d)
        if self.di % headdim:
            raise ValueError(
                f"Mamba-3: expand * width {self.di} not divisible by headdim {headdim}"
            )
        self.nh, self.p, self.ds, self.A_floor = self.di // headdim, headdim, d_state, A_floor
        split = int(d_state * rope_fraction) // 2 * 2
        self.na = split // 2
        if self.na < 1:
            raise ValueError("Mamba-3: d_state * rope_fraction gives no rotary angle")
        self.in_proj = nn.Linear(d, 2 * self.di + 2 * d_state + 3 * self.nh + self.na, bias=False)
        dt = torch.exp(
            torch.rand(self.nh) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.B_bias = nn.Parameter(torch.ones(self.nh, 1, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nh, 1, d_state))
        self.B_norm, self.C_norm = RMSNorm(d_state, 1e-5), RMSNorm(d_state, 1e-5)
        self.D = nn.Parameter(torch.ones(self.nh))
        self.out_proj = nn.Linear(self.di, d, bias=False)

    def forward(self, u):
        b, n, _ = u.shape
        z, x, B, C, dd_dt, dd_A, trap, angles = self.in_proj(u).split(
            [self.di, self.di, self.ds, self.ds, self.nh, self.nh, self.nh, self.na], -1)  # fmt: skip
        A = (-heavy_tail_activation(dd_A.float())).clamp(max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        heads = lambda t: t.view(b, n, 1, self.ds).expand(-1, -1, self.nh, -1)
        y = mamba3_recurrence(
            heads(self.C_norm(C)), heads(self.B_norm(B)), x.view(b, n, self.nh, self.p),
            (A * DT).transpose(1, 2), DT.transpose(1, 2), trap.transpose(1, 2),
            self.C_bias.squeeze(1), self.B_bias.squeeze(1),
            angles.unsqueeze(-2).expand(-1, -1, self.nh, -1).float(), self.D, z.view(b, n, self.nh, self.p),
        )  # fmt: skip
        return self.out_proj(y.reshape(b, n, self.di).to(u.dtype))


class Mamba3Block(nn.Module):
    """mamba_ssm Block without MLP (as Mamba-2): x + Mamba3(RMSNorm x)."""

    def __init__(self, d, d_state, headdim):
        super().__init__()
        self.norm, self.mixer = RMSNorm(d, 1e-5), Mamba3(d, d_state, 2, headdim)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


# ---------------------------------------------------------------------------- ops

MIXERS = ("mamba3", "kda", "rwkv7", "fox", "mla")


def make_block(mixer, d, heads=2, ratio=2):
    """One residual block of a mixer on width d (used by the seq ops and by scan2d_block)."""
    if mixer == "rwkv7":
        return RWKV7Block(d, heads, 2 * ratio)
    if mixer == "kda":
        return AttnBlock(d, KimiDeltaAttention(d, heads), ratio)
    if mixer == "fox":
        return AttnBlock(d, ForgettingAttention(d, heads, output_gate=True), ratio)
    if mixer == "mla":
        return AttnBlock(d, MultiheadLatentAttention(d, heads, kv_rank=max(d // 2, 2)), ratio)
    if mixer == "mamba3":
        hd = _check_heads("Mamba-3", 2 * d, 2 * heads)  # expand 2: 2 heads per requested head
        return Mamba3Block(d, 16, hd)
    raise ValueError(f"unknown mixer {mixer!r}: one of {MIXERS}")


_SEQ = {"heads": [2, 4, 1], "ratio": [2, 4]}


@op("rwkv7_block", "seq", space={"heads": [2, 4, 1], "ratio": [4, 2]}, tags={"block"},
    source="fla-org/flash-linear-attention 0.6.0 fla/layers/rwkv7.py RWKV7Attention + RWKV7FeedForward (RWKV-7, 2025)")  # fmt: skip
def _rwkv7(dims, kind, heads, ratio):
    """RWKV-7 block: LayerNorm, time mixing (generalised delta rule), LayerNorm, squared-ReLU channel mixing."""
    _check_heads("RWKV-7", dims[1], heads)
    return RWKV7Block(dims[1], heads, ratio)


@op("kda_block", "seq", space=_SEQ, tags={"block"},
    source="fla-org/flash-linear-attention 0.6.0 fla/layers/kda.py KimiDeltaAttention (Kimi Linear, 2025)")  # fmt: skip
def _kda(dims, kind, heads, ratio):
    """Kimi Delta Attention (channel-wise gated delta rule) + SwiGLU, pre-RMSNorm."""
    return AttnBlock(dims[1], KimiDeltaAttention(dims[1], heads), ratio)


@op("fox_block", "seq", space={**_SEQ, "output_gate": [1, 0]}, tags={"block", "attn"},
    source="fla-org/flash-linear-attention 0.6.0 fla/layers/forgetting_attn.py (Forgetting Transformer, ICLR 2025)")  # fmt: skip
def _fox(dims, kind, heads, ratio, output_gate):
    """Forgetting Transformer block: causal softmax attention with a forget gate + SwiGLU."""
    return AttnBlock(dims[1], ForgettingAttention(dims[1], heads, bool(output_gate)), ratio)


@op("mla_block", "seq", space={**_SEQ, "kv_rank": [0, 8, 16], "q_rank": [0, 16]}, tags={"block", "attn"},
    source="fla-org/flash-linear-attention 0.6.0 fla/layers/mla.py MultiheadLatentAttention (DeepSeek-V2/V3)")  # fmt: skip
def _mla(dims, kind, heads, ratio, kv_rank, q_rank):
    """Multi-head latent attention (kv_rank 0 = width / 2) + SwiGLU, pre-RMSNorm."""
    d = dims[1]
    return AttnBlock(
        d, MultiheadLatentAttention(d, heads, int(kv_rank) or max(d // 2, 2), int(q_rank)), ratio
    )


@op("mamba3_block", "seq", space={"d_state": [16, 32, 8], "headdim": [16, 8, 32]}, tags={"block"},
    source="state-spaces/mamba e9594ce mamba_ssm/modules/mamba3.py Mamba3, SISO (Mamba-3, 2025)")  # fmt: skip
def _mamba3(dims, kind, d_state, headdim):
    """Mamba-3 block: x + Mamba3(RMSNorm x) (trapezoidal, complex / rotary state, no convolution)."""
    return Mamba3Block(dims[1], int(d_state), int(headdim))


class Scan2D(nn.Module):
    """Pixels of a (C, H, W) map in row-major order, a mixer block forward and one on the reversed
    order, averaged; back to (C, H, W)."""

    def __init__(self, fwd, bwd):
        super().__init__()
        self.fwd, self.bwd = fwd, bwd

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        y = 0.5 * (self.fwd(t) + self.bwd(t.flip(1)).flip(1))
        return y.transpose(1, 2).reshape(b, c, h, w)


@op("scan2d_block", "map", space={"mixer": list(MIXERS), "heads": [2, 4, 1], "ratio": [2, 4]}, tags={"block"},
    source="DLNet: bidirectional scan of a v2.1 mixer over pixels (as Vision Mamba, ICML 2024; Vision-LSTM, ICLR 2025)")  # fmt: skip
def _scan2d(dims, kind, mixer, heads, ratio):
    """A v2.1 sequence block run over the pixels in both row-major directions (two blocks), averaged."""
    return Scan2D(
        make_block(mixer, dims[0], heads, ratio), make_block(mixer, dims[0], heads, ratio)
    )
