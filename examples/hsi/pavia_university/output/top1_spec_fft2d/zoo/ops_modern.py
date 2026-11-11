"""Strong recent families written from their OFFICIAL code (verified: scripts/verify_modern.py).

- TabM (Gorishniy et al., ICLR 2025, arXiv:2410.24210): k BatchEnsemble MLPs that share their
  weights, each member with its own output layer and its own loss (head "ensemble" in core.py);
  official package `tabm` 0.0.3.
- PLR(lite) numerical embeddings (Gorishniy et al., NeurIPS 2022, arXiv:2203.05556; the "lite"
  variant shares the linear layer, TabR arXiv:2307.14338); official `rtdl_num_embeddings` 0.0.12.
- PatchTST (Nie et al., ICLR 2023, arXiv:2211.14730): patches of each channel are tokens of a
  channel-independent Transformer encoder with RevIN; official repo yuqinie98/PatchTST.
- iTransformer (Liu et al., ICLR 2024, arXiv:2310.06625): each variate (whole series) is one
  token; official repo thuml/iTransformer.

Parameters are declared in the same order as the official modules so that weights can be copied
across one by one in the verification.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import op

# ------------------------------------------------------------------------------- TabM


class LinearBatchEnsemble(nn.Module):
    """tabm.LinearBatchEnsemble: x (B, K, D) -> ((x * r) @ W^T) * s + bias (BatchEnsemble eq. 5).

    TabM initialisation: W, bias ~ U(+-1/sqrt(D)) with the SAME bias for all k members; r of the
    first layer random signs (or normal), r of later layers and s ones."""

    def __init__(self, d_in, d_out, k, first_init="ones"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.r = nn.Parameter(torch.empty(k, d_in))
        self.s = nn.Parameter(torch.empty(k, d_out))
        self.bias = nn.Parameter(torch.empty(k, d_out))
        b = d_in**-0.5
        with torch.no_grad():
            nn.init.uniform_(self.weight, -b, b)
            if first_init == "random-signs":
                self.r.bernoulli_(0.5).mul_(2).add_(-1)
            elif first_init == "normal":
                nn.init.normal_(self.r)
            else:
                nn.init.ones_(self.r)
            nn.init.ones_(self.s)
            self.bias.copy_(torch.empty(d_out).uniform_(-b, b).expand(k, d_out))

    def forward(self, x):
        return ((x * self.r) @ self.weight.T) * self.s + self.bias


class TabMBackbone(nn.Module):
    """tabm.MLPBackboneBatchEnsemble (arch_type "tabm"): k members, (B, D) -> (B, K, width)."""

    def __init__(self, d_in, k, width, depth, dropout, first_init="random-signs"):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList(
            nn.Sequential(
                LinearBatchEnsemble(
                    d_in if i == 0 else width, width, k, first_init if i == 0 else "ones"
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for i in range(depth)
        )

    def forward(self, x):
        x = x.unsqueeze(-2).expand(-1, self.k, -1)  # tabm.EnsembleView: every member sees x
        for b in self.blocks:
            x = b(x)
        return x


@op("tabm", "vec", "seq", space={"k": [8, 16, 32], "width": [128, 256, 512], "depth": [2, 3],
                                 "dropout": [0.0, 0.1], "init": ["random-signs", "normal"]}, tags={"ensemble"},
    source="TabM (Gorishniy et al. ICLR 2025, arXiv:2410.24210); official tabm 0.0.3, verified")  # fmt: skip
def _tabm(dims, kind, k=8, width=256, depth=3, dropout=0.1, init="random-signs"):
    """k weight-sharing MLP members (BatchEnsemble); output (k, width), read by head 'ensemble'."""
    return TabMBackbone(dims[0], k, width, depth, dropout, init)


class PeriodicLite(nn.Module):
    """rtdl_num_embeddings.PeriodicEmbeddings(lite=True): x_j -> ReLU(Linear([cos, sin](2 pi c_j x_j)))
    with frequencies c ~ N(0, sigma^2) (truncated at 3 sigma) and one linear layer for all
    features; flattened to (n_features * d)."""

    def __init__(self, n, d=24, freq=48, sigma=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, freq))
        nn.init.trunc_normal_(self.weight, 0.0, sigma, a=-3 * sigma, b=3 * sigma)
        self.linear = nn.Linear(2 * freq, d)

    def forward(self, x):
        z = 2 * math.pi * self.weight * x[..., None]
        return F.relu(self.linear(torch.cat([torch.cos(z), torch.sin(z)], -1))).flatten(1)


@op("plr_embed", "vec", "vec", space={"d": [16, 24, 32], "freq": [16, 48], "sigma": [0.01, 0.1]},
    tags={"embedding"},
    source="PLR(lite) embeddings (Gorishniy et al. NeurIPS 2022, arXiv:2203.05556); official rtdl_num_embeddings "
           "0.0.12, verified")  # fmt: skip
def _plr(dims, kind, d=24, freq=48, sigma=0.01):
    """Periodic embedding of every numerical feature (better than raw inputs for MLPs on tables)."""
    return PeriodicLite(dims[0], d, freq, sigma)


# ---------------------------------------------------------------------------- PatchTST


class RevIN(nn.Module):
    """PatchTST layers/RevIN.py 'norm' mode: per sample and channel, standardise over time, then a
    learnable affine map per channel. Input (B, T, C)."""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(c))
        self.affine_bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mean = x.mean(1, keepdim=True).detach()
        std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - mean) / std * self.affine_weight + self.affine_bias


class _BN(nn.Module):  # nn.Sequential(Transpose(1, 2), BatchNorm1d, Transpose(1, 2))
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class PatchTSTLayer(nn.Module):
    """TSTEncoderLayer (post-norm, BatchNorm, residual attention: scores of the previous layer are
    added to this layer's scores, RealFormer)."""

    def __init__(self, d, heads, d_ff, dropout):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.W_Q, self.W_K, self.W_V = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.scale = nn.Parameter(torch.tensor(self.dk**-0.5), requires_grad=False)
        self.to_out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.norm_attn = _BN(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d)
        )
        self.norm_ffn = _BN(d)

    def forward(self, x, prev=None):
        b, n, _ = x.shape
        q = self.W_Q(x).view(b, n, self.h, self.dk).transpose(1, 2)
        k = self.W_K(x).view(b, n, self.h, self.dk).permute(0, 2, 3, 1)
        v = self.W_V(x).view(b, n, self.h, self.dk).transpose(1, 2)
        scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            scores = scores + prev
        out = torch.matmul(self.drop(F.softmax(scores, -1)), v).transpose(1, 2).reshape(b, n, -1)
        x = self.norm_attn(x + self.drop(self.drop(self.to_out(out))))
        return self.norm_ffn(x + self.drop(self.ff(x))), scores


class PatchTST(nn.Module):
    """RevIN -> patches (length p, stride s) of every channel -> W_P + learned positions ->
    channel-independent encoder. (C, T) -> tokens (C * n_patches, d)."""

    def __init__(self, c, t, d=64, layers=2, heads=4, patch=8, stride=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.revin = RevIN(c)
        self.patch, self.stride = patch, stride
        n = (t - patch) // stride + 1
        self.W_P = nn.Linear(patch, d)
        self.W_pos = nn.Parameter(torch.empty(n, d).uniform_(-0.02, 0.02))  # pe="zeros", learned
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(PatchTSTLayer(d, heads, d_ff, dropout) for _ in range(layers))

    def forward(self, x):  # x (B, C, T)
        x = self.revin(x.transpose(1, 2)).transpose(1, 2)
        z = x.unfold(-1, self.patch, self.stride)  # (B, C, n, p)
        b, c = z.shape[:2]
        u = self.drop(self.W_P(z).reshape(b * c, z.shape[2], -1) + self.W_pos)
        prev = None
        for layer in self.layers:
            u, prev = layer(u, prev)
        return u.reshape(b, c * u.shape[1], u.shape[2])


@op("patchtst", "ser", "seq", space={"d": [32, 64, 128], "layers": [1, 2, 3], "heads": [4, 8],
                                     "patch": [8, 16], "dropout": [0.0, 0.1]}, tags={"stem", "attn"},
    source="PatchTST (Nie et al. ICLR 2023, arXiv:2211.14730); official yuqinie98/PatchTST, verified")  # fmt: skip
def _patchtst(dims, kind, d=64, layers=2, heads=4, patch=8, dropout=0.1):
    c, t = dims
    patch = min(patch, t)
    return PatchTST(c, t, d, layers, heads, patch, max(1, patch // 2), 2 * d, dropout)


# ------------------------------------------------------------------------- iTransformer


class ITransformerLayer(nn.Module):
    """iTransformer EncoderLayer: full attention, post-LayerNorm, 1x1-convolution feed-forward."""

    def __init__(self, d, heads, d_ff, dropout):
        super().__init__()
        self.h = heads
        self.query_projection, self.key_projection = nn.Linear(d, d), nn.Linear(d, d)
        self.value_projection, self.out_projection = nn.Linear(d, d), nn.Linear(d, d)
        self.conv1, self.conv2 = nn.Conv1d(d, d_ff, 1), nn.Conv1d(d_ff, d, 1)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        q = self.query_projection(x).view(b, n, self.h, -1)
        k = self.key_projection(x).view(b, n, self.h, -1)
        v = self.value_projection(x).view(b, n, self.h, -1)
        a = self.drop(
            torch.softmax(torch.einsum("blhe,bshe->bhls", q, k) / math.sqrt(q.shape[-1]), -1)
        )
        out = self.out_projection(torch.einsum("bhls,bshd->blhd", a, v).reshape(b, n, d))
        x = y = self.norm1(x + self.drop(out))
        y = self.drop(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.drop(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class ITransformer(nn.Module):
    """(C, T): optional per-series normalisation, each variate embedded by Linear(T, d) as ONE
    token, encoder over variates, final LayerNorm -> tokens (C, d)."""

    def __init__(self, t, d=64, layers=2, heads=4, d_ff=128, dropout=0.1, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.value_embedding = nn.Linear(t, d)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            ITransformerLayer(d, heads, d_ff, dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x):  # x (B, C, T)
        if self.use_norm:
            x = x - x.mean(-1, keepdim=True).detach()
            x = x / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5)
        x = self.drop(self.value_embedding(x))
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


@op("itransformer", "ser", "seq", space={"d": [32, 64, 128], "layers": [1, 2, 3], "heads": [4, 8],
                                         "dropout": [0.0, 0.1]}, tags={"stem", "attn"},
    source="iTransformer (Liu et al. ICLR 2024, arXiv:2310.06625); official thuml/iTransformer, verified")  # fmt: skip
def _itransformer(dims, kind, d=64, layers=2, heads=4, dropout=0.1):
    return ITransformer(dims[1], d, layers, heads, 2 * d, dropout)
