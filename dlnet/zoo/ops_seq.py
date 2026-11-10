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
from .ops_basic import CONV, ChannelLast

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

    def __init__(self, d, out, heads, ratio, dropout, causal):
        super().__init__()
        out = out or d
        self.ln1, self.attn, self.ln2 = (
            nn.LayerNorm(d),
            SelfAttention(d, heads, dropout, causal),
            nn.LayerNorm(d),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d, ratio * d), nn.GELU(), nn.Linear(ratio * d, out), nn.Dropout(dropout)
        )
        self.proj = nn.Linear(d, out, bias=False) if out != d else None

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return (x if self.proj is None else self.proj(x)) + self.mlp(self.ln2(x))


@op("transformer_block", "seq", space={"heads": [2, 4, 8], "ratio": [2, 4], "out": [0, 64, 128], "dropout": [0.0, 0.1],
    "causal": [0, 1]}, tags={"block"}, source="prescient-design/cortex TransformerBlock (out = 0 keeps the width)")  # fmt: skip
def _transformer(dims, kind, heads, ratio, out, dropout, causal):
    return TransformerBlock(dims[1], out, heads, ratio, dropout, bool(causal))


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
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.tok = nn.Sequential(nn.Linear(t, tok_hidden), nn.GELU(), nn.Linear(tok_hidden, t))
        self.ch = nn.Sequential(nn.Linear(d, ch_hidden), nn.GELU(), nn.Linear(ch_hidden, d))

    def forward(self, x):
        x = x + self.tok(self.ln1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.ch(self.ln2(x))


@op("mixer_block", "seq", space={"token_ratio": [0.5, 1.0, 2.0], "channel_ratio": [2, 4]}, tags={"block"},
    source="MLP-Mixer (arXiv:2105.01601), from memory")  # fmt: skip
def _mixer(dims, kind, token_ratio, channel_ratio):
    t, d = dims
    return MixerBlock(t, d, max(int(t * token_ratio), 4), channel_ratio * d)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.w


def selective_scan(u, delta, A, B, C, D):
    """Sequential selective scan (mamba selective_scan_ref, input-dependent B and C, no gate here).

    u, delta: (b, d, l); A: (d, n); B, C: (b, n, l); D: (d). h_t = exp(delta A) h_{t-1} + delta B u,
    y_t = C h_t; output y + D u.
    """
    dA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    dBu = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    x = u.new_zeros(u.shape[0], A.shape[0], A.shape[1])
    ys = []
    for i in range(u.shape[2]):
        x = dA[:, :, i] * x + dBu[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
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


@op("conv1d_resid", "ser", space={"out": [0, 32, 64, 128], "k": [3, 5, 7], "dilation": [1, 2, 4, 8], "norm": [1, 0],
    "act": ["swish", "relu"], "dropout": [0.0, 0.1]}, tags={"block"},
    source="prescient-design/cortex Conv1dResidBlock: norm-act-conv-norm-act-conv + residual (out = 0 keeps channels)")  # fmt: skip
def _conv1d_resid(dims, kind, out, k, dilation, norm, act, dropout):
    return Conv1dResid(dims[0], out or dims[0], k, dilation, bool(norm), act, dropout)


class Conv1dResid(nn.Module):
    def __init__(self, i, o, k, dilation, norm, act, dropout):
        super().__init__()
        pad = dilation * (k - 1) // 2
        ln = (lambda c: ChannelLast(nn.LayerNorm(c), 1)) if norm else (lambda c: nn.Identity())
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
