"""Domain networks of 2024-2025 written from their OFFICIAL code (v1.0; THIRD_PARTY.md).

Numerical check with weights copied across: scripts/verify_domain.py (job `verify-domain`).

Tabular (kind `vec`):
- RealMLP (Holzmueller, Grinsztajn, Steinwart, NeurIPS 2024, arXiv:2407.04491;
  dholzmueller/pytabkit models/nn_models: ScaleLayer, PLREmbeddingsLayerCosBias ("pbld"),
  WeightLayer with weight_param "ntk", BiasLayer, ParametricActivationLayer; RealMLP-TD defaults).
  Not ported: the per-parameter learning-rate factors (scale x6, bias/act/emb x0.1), the
  data-dependent initialisations ("std" weights, "he+5" biases: here normal / zeros as in
  RealMLP-TD-S), the dropout and weight-decay schedules, and pytabkit's own preprocessing (DLNet
  standardises; the robust scaling + smooth clipping x / sqrt(1 + (x/3)^2) is kept as an option).
Time series (kind `ser`, channels x time; output a feature vector for DLNet's head):
- TimeMixer (Wang et al., ICLR 2024, arXiv:2405.14616; kwuking/TimeMixer models/TimeMixer.py,
  classification path, channel_independence 0, moving-average decomposition, avg down-sampling).
- TimeXer (Wang et al., NeurIPS 2024, arXiv:2402.19072; thuml/TimeXer models/TimeXer.py
  forecast_multi path: every variable is endogenous (patch tokens + a global token) and the
  whole series of each variable is an exogenous token for cross-attention); the forecasting
  FlattenHead is replaced by DLNet's head on the flattened encoder output.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import op

# ----------------------------------------------------------------------------- RealMLP


class PBLDEmbedding(nn.Module):
    """pytabkit PLREmbeddingsLayerCosBias, act "linear", densenet: per feature
    cos(2 pi w1 x + b1) -> linear to hidden_2 - 1, then the raw features appended."""

    def __init__(self, n, sigma=0.1, hidden_1=16, hidden_2=4):
        super().__init__()
        h2 = hidden_2 - 1
        self.weight_1 = nn.Parameter(sigma * torch.randn(n, 1, hidden_1))
        self.bias_1 = nn.Parameter(math.pi * (-1 + 2 * torch.rand(n, 1, hidden_1)))
        self.weight_2 = nn.Parameter((-1 + 2 * torch.rand(n, hidden_1, h2)) / math.sqrt(hidden_1))
        self.bias_2 = nn.Parameter((-1 + 2 * torch.rand(n, 1, h2)) / math.sqrt(hidden_1))

    def forward(self, x):
        x0 = x
        x = x.transpose(-1, -2).unsqueeze(-1)  # (n, B, 1)
        x = torch.cos(2 * torch.pi * x.matmul(self.weight_1) + self.bias_1)
        x = x.matmul(self.weight_2) + self.bias_2  # (n, B, h2)
        x = x.transpose(-2, -3)
        return torch.cat([x.reshape(*x.shape[:-2], -1), x0], dim=-1)


class NTKLinear(nn.Module):
    """WeightLayer (weight_param "ntk": x @ W / sqrt(in), W ~ N(0, 1), shape in x out) + BiasLayer."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.factor = 1.0 / math.sqrt(d_in)

    def forward(self, x):
        return self.factor * x.matmul(self.weight) + self.bias


class ParametricAct(nn.Module):
    """ParametricActivationLayer: x + (f(x) - x) * w, w initialised to 1 per feature."""

    def __init__(self, d, f):
        super().__init__()
        self.f = f
        self.weight = nn.Parameter(torch.ones(1, d))

    def forward(self, x):
        return x + (self.f(x) - x) * self.weight


REALMLP_ACTS = {"selu": F.selu, "mish": F.mish, "relu": F.relu, "silu": F.silu}


class RealMLP(nn.Module):
    """RealMLP-TD body: [smooth clip], PBLD embedding, front scale, blocks w-b-a-d. -> (B, width)."""

    def __init__(
        self, d, width=256, depth=3, act="selu", p_drop=0.15, emb="pbld", smooth_clip=False
    ):
        super().__init__()
        self.smooth_clip = smooth_clip
        self.emb = PBLDEmbedding(d) if emb == "pbld" else nn.Identity()
        d_in = 4 * d if emb == "pbld" else d
        self.scale = nn.Parameter(torch.ones(1, d_in))
        layers = []
        for i in range(depth):
            layers += [
                NTKLinear(d_in if i == 0 else width, width),
                ParametricAct(width, REALMLP_ACTS[act]),
                nn.Dropout(p_drop),
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.smooth_clip:
            x = x / torch.sqrt(1 + (x / 3) ** 2)
        return self.layers(self.emb(x) * self.scale)


@op("realmlp", "vec", space={"width": [256, 128, 512], "depth": [3, 2, 4], "act": ["selu", "mish", "relu"],
    "p_drop": [0.15, 0.0, 0.3], "emb": ["pbld", "none"], "smooth_clip": [0, 1]}, tags={"block"},
    source="dholzmueller/pytabkit RealMLP-TD (Holzmueller et al., NeurIPS 2024): PBLD embedding, front scale, NTK linear, parametric act")  # fmt: skip
def _realmlp(dims, kind, width, depth, act, p_drop, emb, smooth_clip):
    return RealMLP(dims[0], width, depth, act, p_drop, emb, bool(smooth_clip))


# ---------------------------------------------------------------------------- TimeMixer


class MovingAvg(nn.Module):
    """Autoformer moving_avg: replicate-pad both ends, average pool, (B, T, C)."""

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.avg = nn.AvgPool1d(kernel_size=k, stride=1, padding=0)

    def forward(self, x):
        p = (self.k - 1) // 2
        x = torch.cat([x[:, :1].repeat(1, p, 1), x, x[:, -1:].repeat(1, p, 1)], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.moving_avg = MovingAvg(k)

    def forward(self, x):
        trend = self.moving_avg(x)
        return x - trend, trend


def _mlp2(a, b):
    return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Linear(b, b))


class PastDecomposableMixing(nn.Module):
    """TimeMixer PastDecomposableMixing, channel_independence 0: decompose each scale, mix the
    seasonal parts bottom-up and the trends top-down with MLPs over time, then add."""

    def __init__(self, seq_len, d_model, d_ff, layers, window, moving_avg, last=False):
        super().__init__()
        lens = [seq_len // window**i for i in range(layers + 1)]
        # in the LAST block only scale 0 reaches the classifier and the bottom-up season mixing
        # feeds the coarser scales only: its weights never change the output, so they are left out
        self.last = last
        self.decompsition = SeriesDecomp(moving_avg)
        self.cross_layer = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.mixing_multi_scale_season = nn.Module()
        self.mixing_multi_scale_season.down_sampling_layers = nn.ModuleList(
            _mlp2(lens[i], lens[i + 1]) for i in range(0 if last else layers)
        )
        self.mixing_multi_scale_trend = nn.Module()
        self.mixing_multi_scale_trend.up_sampling_layers = nn.ModuleList(
            _mlp2(lens[i + 1], lens[i]) for i in reversed(range(layers))
        )
        # out_cross_layer of the original is only used with channel_independence 1: left out

    def forward(self, x_list):
        seasons, trends = [], []
        for x in x_list:
            s, t = self.decompsition(x)
            seasons.append(self.cross_layer(s).permute(0, 2, 1))
            trends.append(self.cross_layer(t).permute(0, 2, 1))
        # bottom-up season mixing
        down = self.mixing_multi_scale_season.down_sampling_layers
        high, low = seasons[0], seasons[1]
        out_s = [high.permute(0, 2, 1)]
        for i in range(0 if self.last else len(seasons) - 1):
            low = low + down[i](high)
            high = low
            if i + 2 <= len(seasons) - 1:
                low = seasons[i + 2]
            out_s.append(high.permute(0, 2, 1))
        # top-down trend mixing
        up = self.mixing_multi_scale_trend.up_sampling_layers
        rev = trends[::-1]
        low, high = rev[0], rev[1]
        out_t = [low.permute(0, 2, 1)]
        for i in range(len(rev) - 1):
            high = high + up[i](low)
            low = high
            if i + 2 <= len(rev) - 1:
                high = rev[i + 2]
            out_t.append(low.permute(0, 2, 1))
        out_t.reverse()
        if self.last:
            return [(out_s[0] + out_t[0])[:, : x_list[0].shape[1], :]]
        return [(s + t)[:, : x.shape[1], :] for x, s, t in zip(x_list, out_s, out_t, strict=True)]


class TokenEmbedding(nn.Module):
    """Time-Series-Library TokenEmbedding: circular conv1d k=3, no bias."""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class TimeMixer(nn.Module):
    """TimeMixer Model.classification without the projection: (B, C, T) -> gelu(enc).flatten (B, T*d)."""

    def __init__(
        self,
        c,
        seq_len,
        d_model=16,
        d_ff=32,
        e_layers=2,
        down_layers=1,
        window=2,
        moving_avg=25,
        dropout=0.1,
    ):
        super().__init__()
        self.down_layers, self.window = down_layers, window
        self.pdm_blocks = nn.ModuleList(
            PastDecomposableMixing(
                seq_len, d_model, d_ff, down_layers, window, moving_avg, i == e_layers - 1
            )
            for i in range(e_layers)
        )
        self.enc_embedding = nn.Module()
        self.enc_embedding.value_embedding = TokenEmbedding(c, d_model)
        self.drop_emb, self.dropout = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C)
        xs, cur = [x], x.permute(0, 2, 1)
        for _ in range(self.down_layers):
            cur = F.avg_pool1d(cur, self.window)
            xs.append(cur.permute(0, 2, 1))
        out = [self.drop_emb(self.enc_embedding.value_embedding(t)) for t in xs]
        for blk in self.pdm_blocks:
            out = blk(out)
        return self.dropout(F.gelu(out[0])).reshape(x.shape[0], -1)


@op("timemixer", "ser", "vec", space={"d_model": [16, 32, 64], "e_layers": [2, 1, 3], "down_layers": [1, 2, 3],
    "moving_avg": [25, 5, 13], "dropout": [0.1, 0.0]}, tags={"stem"},
    source="kwuking/TimeMixer models/TimeMixer.py (Wang et al., ICLR 2024), classification path without projection")  # fmt: skip
def _timemixer(dims, kind, d_model, e_layers, down_layers, moving_avg, dropout):
    c, t = dims
    if t // 2**down_layers < 2:
        raise ValueError(
            f"TimeMixer: series of {t} steps too short for {down_layers} down-sampling layers"
        )
    return TimeMixer(c, t, d_model, 2 * d_model, e_layers, down_layers, 2, moving_avg, dropout)


# ----------------------------------------------------------------------------- TimeXer


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class AttentionLayer(nn.Module):
    """Time-Series-Library AttentionLayer + FullAttention (no mask)."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        e = d_model // n_heads
        self.query_projection = nn.Linear(d_model, e * n_heads)
        self.key_projection = nn.Linear(d_model, e * n_heads)
        self.value_projection = nn.Linear(d_model, e * n_heads)
        self.out_projection = nn.Linear(e * n_heads, d_model)
        self.n_heads, self.drop = n_heads, nn.Dropout(dropout)

    def forward(self, q, k, v):
        b, l, _ = q.shape
        s, h = k.shape[1], self.n_heads
        q = self.query_projection(q).view(b, l, h, -1)
        k = self.key_projection(k).view(b, s, h, -1)
        v = self.value_projection(v).view(b, s, h, -1)
        a = self.drop(
            torch.softmax(torch.einsum("blhe,bshe->bhls", q, k) / math.sqrt(q.shape[-1]), dim=-1)
        )
        return self.out_projection(torch.einsum("bhls,bshd->blhd", a, v).reshape(b, l, -1))


class TimeXerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attention = AttentionLayer(d_model, n_heads, dropout)
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1, self.norm2, self.norm3 = (
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross):
        b, _, d = cross.shape
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x)))
        g0 = x[:, -1, :].unsqueeze(1)
        g = self.dropout(self.cross_attention(g0.reshape(b, -1, d), cross, cross))
        g = self.norm2(g0 + g.reshape(-1, d).unsqueeze(1))
        x = torch.cat([x[:, :-1, :], g], dim=1)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class TimeXer(nn.Module):
    """TimeXer forecast_multi encoder: (B, C, T) -> flattened (B, C * d * (T // patch + 1))."""

    def __init__(
        self,
        c,
        seq_len,
        patch_len=8,
        d_model=64,
        n_heads=4,
        d_ff=128,
        e_layers=2,
        dropout=0.1,
        use_norm=True,
    ):
        super().__init__()
        self.patch_len, self.use_norm = patch_len, use_norm
        self.en_embedding = nn.Module()
        self.en_embedding.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.en_embedding.glb_token = nn.Parameter(torch.randn(1, c, 1, d_model))
        self.en_embedding.position_embedding = PositionalEmbedding(d_model)
        self.ex_embedding = nn.Module()
        self.ex_embedding.value_embedding = nn.Linear(seq_len, d_model)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            TimeXerLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)
        )
        self.encoder.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C)
        if self.use_norm:
            x = x - x.mean(1, keepdim=True).detach()
            x = x / torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        b, _, n = x.shape
        e = self.en_embedding
        z = x.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_len)  # (B, C, P, patch)
        z = z.reshape(b * n, z.shape[2], z.shape[3])
        z = e.value_embedding(z) + e.position_embedding(z)
        z = torch.cat(
            [z.reshape(b, n, z.shape[-2], z.shape[-1]), e.glb_token.repeat(b, 1, 1, 1)], dim=2
        )
        z = self.drop(z.reshape(b * n, z.shape[2], z.shape[3]))
        ex = self.drop(self.ex_embedding.value_embedding(x.permute(0, 2, 1)))
        for layer in self.encoder.layers:
            z = layer(z, ex)
        z = self.encoder.norm(z)
        return z.reshape(b, n, z.shape[-2], z.shape[-1]).permute(0, 1, 3, 2).reshape(b, -1)


@op("timexer", "ser", "vec", space={"patch_len": [8, 4, 16], "d_model": [64, 32, 128], "heads": [4, 2, 8],
    "e_layers": [2, 1, 3], "dropout": [0.1, 0.0]}, tags={"stem"},
    source="thuml/TimeXer models/TimeXer.py (Wang et al., NeurIPS 2024), forecast_multi encoder without the forecasting head")  # fmt: skip
def _timexer(dims, kind, patch_len, d_model, heads, e_layers, dropout):
    c, t = dims
    if t < patch_len:
        raise ValueError(f"TimeXer: series of {t} steps shorter than one patch of {patch_len}")
    if d_model % heads:
        raise ValueError(f"TimeXer: d_model {d_model} not divisible by {heads} heads")
    return TimeXer(c, t, patch_len, d_model, heads, 2 * d_model, e_layers, dropout)
