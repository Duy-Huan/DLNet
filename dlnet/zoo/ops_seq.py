"""Sequence and time-series blocks (torch only).

Sources (see THIRD_PARTY.md):
  prescient-design/cortex   TransformerBlock (pre-norm attention + MLP with residual projection),
                            Bidirectional / Causal self-attention, Conv1dResidBlock
  rasbt/deeplearning-models RNN (LSTM/GRU/vanilla), TextCNN
  state-spaces/mamba        Mamba block (mamba_simple.py) with the pure-PyTorch selective scan of
                            selective_scan_ref; sequential scan, so lengths are capped
  papers, from memory       MLP-Mixer block (Tolstikhin et al., arXiv:2105.01601)
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .core import op
from .ops_basic import CONV

MAMBA_MAX_LEN = 256


class SelfAttention(nn.Module):
    """Fused-QKV multi-head self-attention (cortex BidirectionalSelfAttention / CausalSelfAttention)."""

    def __init__(self, d, heads, dropout, causal, bias=False):
        super().__init__()
        if d % heads:
            raise ValueError("heads must divide the feature size")
        self.qkv, self.proj, self.drop = (
            nn.Linear(d, 3 * d, bias=bias),
            nn.Linear(d, d, bias=bias),
            nn.Dropout(dropout),
        )
        self.h, self.p, self.causal = heads, dropout, causal

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = (
            z.view(b, t, self.h, d // self.h).transpose(1, 2) for z in self.qkv(x).chunk(3, -1)
        )
        y = nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p if self.training else 0.0, is_causal=self.causal
        )
        return self.drop(self.proj(y.transpose(1, 2).reshape(b, t, d)))


class TransformerBlock(nn.Module):
    """cortex TransformerBlock: x + attn(LN x); then proj(x) + MLP(LN x) (MLP maps D -> `out`)."""

    def __init__(self, d, out, heads, ratio, dropout, causal, bias=False, cortex_skip=False):
        super().__init__()
        self.cortex_skip = cortex_skip
        out = out or d
        self.ln1, self.ln2 = nn.LayerNorm(d, bias=bias), nn.LayerNorm(d, bias=bias)
        self.attn = SelfAttention(d, heads, dropout, causal, bias)
        self.mlp = nn.Sequential(
            nn.Linear(d, ratio * d, bias=bias),
            nn.GELU(),
            nn.Linear(ratio * d, out, bias=bias),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(d, out, bias=bias) if out != d else None

    def forward(self, x0):
        x = x0 + self.attn(self.ln1(x0))
        base = (
            x0 if self.cortex_skip else x
        )  # cortex adds the MLP branch to the block INPUT, not to x + attn
        return (base if self.proj is None else self.proj(base)) + self.mlp(self.ln2(x))


@op("transformer_block", "seq", space={"heads": [2, 4, 8], "ratio": [2, 4], "out": [0, 64, 128], "dropout": [0.0, 0.1],
    "causal": [0, 1], "bias": [0, 1], "cortex_skip": [0, 1]}, tags={"block"},
    source="prescient-design/cortex TransformerBlock (out = 0 keeps the width; bias = 0 as in cortex; cortex_skip = 1 reproduces its skip path)")  # fmt: skip
def _transformer(dims, kind, heads, ratio, out, dropout, causal, bias, cortex_skip):
    return TransformerBlock(
        dims[1], out, heads, ratio, dropout, bool(causal), bool(bias), bool(cortex_skip)
    )


class AttnRes(nn.Module):
    def __init__(self, d, heads, dropout, causal):
        super().__init__()
        self.ln, self.attn = nn.LayerNorm(d), SelfAttention(d, heads, dropout, causal)

    def forward(self, x):
        return x + self.attn(self.ln(x))


@op("attn_seq", "seq", space={"heads": [1, 2, 4, 8], "dropout": [0.0, 0.1], "causal": [0, 1]}, tags={"attn"},
    source="prescient-design/cortex self-attention with pre-norm and residual inside")  # fmt: skip
def _attn_seq(dims, kind, heads, dropout, causal):
    return AttnRes(dims[1], heads, dropout, bool(causal))


class RNN(nn.Module):
    def __init__(self, d, cell, hidden, layers, bidir, dropout):
        super().__init__()
        self.r = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[cell](
            d,
            hidden,
            layers,
            batch_first=True,
            bidirectional=bool(bidir),
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, x):
        return self.r(x)[0]


@op("rnn", "seq", space={"cell": ["gru", "lstm", "rnn"], "hidden": [32, 64, 128], "layers": [1, 2], "bidir": [1, 0],
    "dropout": [0.0, 0.1]}, tags={"block"}, source="rasbt/deeplearning-models RNN (LSTM); output width = hidden x directions")  # fmt: skip
def _rnn(dims, kind, cell, hidden, layers, bidir, dropout):
    return RNN(dims[1], cell, hidden, layers, bidir, dropout)


class MixerBlock(nn.Module):
    """MLP-Mixer block: token-mixing MLP across positions, then channel-mixing MLP; residual inside."""

    def __init__(self, t, d, tok_hidden, ch_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(
            d, eps=1e-6
        )  # timm: eps 1e-6, definition order norm1, tokens, norm2, channels
        self.tok = nn.Sequential(nn.Linear(t, tok_hidden), nn.GELU(), nn.Linear(tok_hidden, t))
        self.ln2 = nn.LayerNorm(d, eps=1e-6)
        self.ch = nn.Sequential(nn.Linear(d, ch_hidden), nn.GELU(), nn.Linear(ch_hidden, d))

    def forward(self, x):
        x = x + self.tok(self.ln1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.ch(self.ln2(x))


@op("mixer_block", "seq", space={"token_ratio": [0.5, 1.0, 2.0], "channel_ratio": [2, 4]}, tags={"block"},
    source="huggingface/pytorch-image-models mlp_mixer.MixerBlock (hidden sizes = ratio x width, as timm)")  # fmt: skip
def _mixer(dims, kind, token_ratio, channel_ratio):
    t, d = dims
    return MixerBlock(t, d, max(int(d * token_ratio), 4), channel_ratio * d)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


def selective_scan(u, delta, A, B, C, D):
    """Sequential selective scan (mamba selective_scan_ref, input-dependent B and C, no gate here).

    u, delta: (b, d, l); A: (d, n); B, C: (b, n, l); D: (d). h_t = exp(delta A) h_{t-1} + delta B u,
    y_t = C h_t; output y + D u.
    """
    dA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    dBu = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    x = u.new_zeros(u.shape[0], A.shape[0], A.shape[1])
    ys = []
    # unbind, not dA[:, :, i]: the backward of each index op allocates a zero tensor as large as dA,
    # O(L^2) memory traffic (FAHM training step 13.8 s -> see tests/test_regression.py)
    for a_i, bu_i, c_i in zip(dA.unbind(2), dBu.unbind(2), C.unbind(2)):
        x = a_i * x + bu_i
        ys.append(torch.einsum("bdn,bn->bd", x, c_i))
    return torch.stack(ys, 2) + u * D[None, :, None]


class Mamba(nn.Module):
    """Mamba mixer (state-spaces/mamba mamba_simple.Mamba, slow path), pure PyTorch."""

    def __init__(self, d, d_state=16, d_conv=4, expand=2, dt_min=0.001, dt_max=0.1, dt_floor=1e-4):
        super().__init__()
        self.d_inner, self.d_state, self.dt_rank, self.d_conv = (
            expand * d,
            d_state,
            math.ceil(d / 16),
            d_conv,
        )
        self.in_proj = nn.Linear(d, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1))
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)

    def forward(self, h):  # (b, l, d)
        l = h.shape[1]
        x, z = self.in_proj(h).transpose(1, 2).chunk(2, 1)  # (b, d_inner, l)
        x = nn.functional.silu(self.conv1d(x)[..., :l])
        dt, B, C = self.x_proj(x.transpose(1, 2)).split(
            [self.dt_rank, self.d_state, self.d_state], -1
        )
        delta = nn.functional.softplus(self.dt_proj(dt).transpose(1, 2))  # dt_proj bias + softplus
        y = selective_scan(
            x, delta, -torch.exp(self.A_log), B.transpose(1, 2), C.transpose(1, 2), self.D
        )
        return self.out_proj((y * nn.functional.silu(z)).transpose(1, 2))


class MambaBlock(nn.Module):
    def __init__(self, d, d_state, d_conv, expand):
        super().__init__()
        self.norm, self.mixer = RMSNorm(d), Mamba(d, d_state, d_conv, expand)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


@op("mamba_block", "seq", space={"d_state": [8, 16], "d_conv": [4, 3], "expand": [2, 1]}, tags={"block"},
    source="state-spaces/mamba mamba_simple.Mamba + selective_scan_ref (pre-RMSNorm, residual inside)")  # fmt: skip
def _mamba(dims, kind, d_state, d_conv, expand):
    if dims[0] > MAMBA_MAX_LEN:
        raise ValueError(
            f"the reference selective scan is sequential: length {dims[0]} > {MAMBA_MAX_LEN}"
        )
    return MambaBlock(dims[1], d_state, d_conv, expand)


def segsum(x):
    """Segment sums of the last axis: out[..., i, j] = sum_{j < k <= i} x[..., k] for j <= i, -inf above the diagonal."""
    t = x.shape[-1]
    x = x.unsqueeze(-1).expand(*x.shape, t)
    x = x.masked_fill(~torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), -1), 0)
    out = torch.cumsum(x, dim=-2)
    return out.masked_fill(
        ~torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), 0), -torch.inf
    )


def ssd_minimal(X, A, B, C, block_len):
    """Structured state-space duality, chunked (mamba_ssm ssd_minimal.ssd_minimal_discrete, Listing 1 of Mamba-2).

    X (b, l, h, p), A (b, l, h), B and C (b, l, h, n); l must be a multiple of block_len. Returns Y (b, l, h, p).
    """
    b, ln, h, p = X.shape
    c = ln // block_len
    X, B, C = (x.reshape(b, c, block_len, *x.shape[2:]) for x in (X, B, C))
    A = A.reshape(b, c, block_len, h).permute(0, 3, 1, 2)  # b h c l
    A_cumsum = torch.cumsum(A, dim=-1)
    L = torch.exp(segsum(A))
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)  # inside each chunk
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum(
        "bclhn,bhcl,bclhp->bchpn", B, decay_states, X
    )  # state at the end of each chunk
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(segsum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[
        :, :-1
    ]  # recurrence across chunks
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, torch.exp(A_cumsum))
    return (y_diag + y_off).reshape(b, ln, h, p)


class Mamba2(nn.Module):
    """Mamba-2 mixer (state-spaces/mamba mamba2_simple.Mamba2Simple, non-fused path) with the pure-PyTorch SSD."""

    def __init__(
        self,
        d,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=32,
        chunk=64,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_floor=1e-4,
    ):
        super().__init__()
        self.d_inner, self.d_state, self.headdim, self.chunk, self.ng = (
            expand * d,
            d_state,
            headdim,
            chunk,
            ngroups,
        )
        if self.d_inner % headdim:
            raise ValueError("headdim must divide expand * width")
        self.nheads = self.d_inner // headdim
        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.in_proj = nn.Linear(
            d, 2 * self.d_inner + 2 * ngroups * d_state + self.nheads, bias=False
        )
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNorm(self.d_inner, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)

    def forward(self, u):  # (b, l, d)
        b, ln, _ = u.shape
        z, xbc, dt = self.in_proj(u).split(
            [self.d_inner, self.d_inner + 2 * self.ng * self.d_state, self.nheads], -1
        )
        dt = nn.functional.softplus(dt + self.dt_bias)
        xbc = nn.functional.silu(self.conv1d(xbc.transpose(1, 2)).transpose(1, 2)[:, :ln])
        x, B, C = xbc.split([self.d_inner, self.ng * self.d_state, self.ng * self.d_state], -1)
        x = x.reshape(b, ln, self.nheads, self.headdim)
        rep = self.nheads // self.ng
        B, C = (m.reshape(b, ln, self.ng, self.d_state).repeat_interleave(rep, 2) for m in (B, C))
        A = -torch.exp(self.A_log)
        pad = (
            -ln
        ) % self.chunk  # zero padding after the end does not change earlier (causal) outputs
        xs, As, Bp, Cp = x * dt.unsqueeze(-1), A * dt, B, C
        if pad:
            xs, As, Bp, Cp = (
                nn.functional.pad(t, (0, 0) * (t.dim() - 2) + (0, pad)) for t in (xs, As, Bp, Cp)
            )
        y = ssd_minimal(xs, As, Bp, Cp, self.chunk)[:, :ln] + x * self.D.unsqueeze(-1)
        y = self.norm(
            y.reshape(b, ln, self.d_inner) * nn.functional.silu(z)
        )  # RMSNormGated, norm_before_gate = False
        return self.out_proj(y)


class Mamba2Block(nn.Module):
    def __init__(self, d, **kw):
        super().__init__()
        self.norm, self.mixer = RMSNorm(d), Mamba2(d, **kw)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


@op("mamba2_block", "seq", space={"d_state": [16, 64], "d_conv": [4], "expand": [2, 1], "headdim": [16, 32, 64], "chunk": [16, 64]},
    tags={"block"}, source="state-spaces/mamba mamba2_simple.Mamba2Simple + ssd_minimal (pre-RMSNorm, residual inside; no Triton/CUDA kernels)")  # fmt: skip
def _mamba2(dims, kind, d_state, d_conv, expand, headdim, chunk):
    return Mamba2Block(
        dims[1], d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, chunk=chunk
    )


@op("conv1d_resid", "ser", space={"out": [0, 32, 64, 128], "k": [3, 5, 7], "dilation": [1, 2, 4, 8], "norm": [1, 0],
    "act": ["swish", "relu"], "dropout": [0.0, 0.1]}, tags={"block"},
    source="prescient-design/cortex Conv1dResidBlock: norm-act-conv-norm-act-conv + residual (out = 0 keeps channels)")  # fmt: skip
def _conv1d_resid(dims, kind, out, k, dilation, norm, act, dropout):
    return Conv1dResid(dims[0], out or dims[0], k, dilation, bool(norm), act, dropout)


class ChanLN(nn.Module):
    """cortex MaskLayerNorm1d over the channel axis of (B, C, T): per-position statistics,
    var = E[x^2] - E[x]^2 clamped at eps (not var + eps as in nn.LayerNorm); affine per channel."""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.w, self.b, self.eps = (
            nn.Parameter(torch.ones(c, 1)),
            nn.Parameter(torch.zeros(c, 1)),
            eps,
        )

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = x.square().mean(1, keepdim=True) - mean.square()
        return (x - mean) * (self.w / var.clamp(self.eps) ** 0.5) + self.b


class Conv1dResid(nn.Module):
    def __init__(self, i, o, k, dilation, norm, act, dropout):
        super().__init__()
        pad = dilation * (k - 1) // 2
        ln = ChanLN if norm else (lambda c: nn.Identity())
        a = nn.SiLU if act == "swish" else nn.ReLU
        self.f = nn.Sequential(ln(i), a(), CONV[1](i, o, k, padding=pad, dilation=dilation, bias=False),
                               ln(o), a(), CONV[1](o, o, k, padding=pad, dilation=dilation, bias=False))  # fmt: skip
        self.proj = CONV[1](i, o, 1) if i != o else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.f(x) + (x if self.proj is None else self.proj(x)))


class TextCNN(nn.Module):
    def __init__(self, c, filters, ks, dropout):
        super().__init__()
        self.convs = nn.ModuleList(CONV[1](c, filters, k) for k in ks)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(torch.cat([torch.relu(cv(x)).amax(2) for cv in self.convs], 1))


@op("textcnn", "ser", "vec", space={"filters": [16, 32, 64], "ks": [[3, 4, 5], [2, 3, 4], [3, 5, 7]], "dropout": [0.0, 0.3, 0.5]},
    tags={"adapt", "block"}, source="rasbt/deeplearning-models TextCNN (Kim 2014): parallel convolutions + max over time")  # fmt: skip
def _textcnn(dims, kind, filters, ks, dropout):
    return TextCNN(dims[0], filters, ks, dropout)
