"""Modules that the zoo lacked before v1.4, written from their OFFICIAL code (THIRD_PARTY.md).

Numerical check with weights copied across: scripts/verify_more.py (job `verify-more`).

Time series (kind `ser`, channels x time):
- DLinear (Zeng et al., AAAI 2023, arXiv:2205.13504; thuml/Time-Series-Library models/DLinear.py,
  `encoder`: moving-average decomposition, one linear map per component over the time axis).
- TSMixer (Chen et al., TMLR 2023, arXiv:2303.06053; Time-Series-Library models/TSMixer.py ResBlock).
- TiDE residual block (Das et al., TMLR 2023, arXiv:2304.08424; Time-Series-Library models/TiDE.py
  ResBlock), on the last axis of a vector or a token sequence.
Graphs (kind `graph`, dense (N, F + N); the PyG layers rewritten on a dense adjacency):
- GATv2 (Brody et al., ICLR 2022, arXiv:2105.14491; torch_geometric GATv2Conv),
  GCNII (Chen et al., ICML 2020, arXiv:2007.02133; GCN2Conv with shared weights),
  APPNP (Gasteiger et al., ICLR 2019, arXiv:1810.05997; APPNP),
  SGC (Wu et al., ICML 2019, arXiv:1902.07153; SGConv).
Field embeddings (kind `seq`, fields x embedding; reczoo/FuxiCTR fuxictr/pytorch/layers):
- CIN of xDeepFM (Lian et al., KDD 2018; CompressedInteractionNet), the interacting layer of AutoInt
  (Song et al., CIKM 2019; MultiHeadSelfAttention), and FiBiNET's SENET + bilinear interaction
  (Huang et al., RecSys 2019; SqueezeExcitation, BilinearInteractionV2).
Sets and token pooling (kind `seq`):
- Set Transformer ISAB / PMA (Lee et al., ICML 2019; juho-lee/set_transformer modules.py),
  attention pooling with a learned latent query (timm.layers.AttentionPoolLatent, the MAP head of
  SigLIP / "Better plain ViT baselines").
Images (kind `map`):
- InceptionNeXt block (Yu et al., CVPR 2024; timm.models.inception_next MetaNeXtBlock),
  FocalNet block (Yang et al., NeurIPS 2022; timm.models.focalnet FocalNetBlock),
  non-local block (Wang et al., CVPR 2018; timm.layers.NonLocalAttn),
  PVTv2 block with spatial-reduction attention (Wang et al., CVMJ 2022; timm.models.pvt_v2 Block),
  anti-aliased down-sampling (Zhang, ICML 2019; timm.layers.BlurPool2d),
  SimAM (Yang et al., ICML 2021; ZjjConan/SimAM simam_module), GeM pooling (Radenovic et al.,
  TPAMI 2018; filipradenovic/cnnimageretrieval-pytorch GeM).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import op
from .ops_basic import act
from .ops_domain import SeriesDecomp
from .ops_graph import GraphLayer, _feat, renormalize
from .ops_img import LayerNorm2d, make_divisible
from .ops_next import LayerScale2d

# ----------------------------------------------------------------------------- time series


class DLinear(nn.Module):
    """TSLib DLinear.encoder on (B, C, T): seasonal and trend parts each go through a linear map
    over time (shared by the channels, or one per channel with `individual`)."""

    def __init__(self, c, t, moving_avg=25, individual=False):
        super().__init__()
        self.individual = individual
        self.decompsition = SeriesDecomp(moving_avg)

        def lin():
            m = nn.Linear(t, t)
            m.weight = nn.Parameter(torch.ones(t, t) / t)
            return m

        if individual:
            self.Linear_Seasonal = nn.ModuleList(lin() for _ in range(c))
            self.Linear_Trend = nn.ModuleList(lin() for _ in range(c))
        else:
            self.Linear_Seasonal, self.Linear_Trend = lin(), lin()

    def forward(self, x):
        s, t = self.decompsition(x.transpose(1, 2))
        s, t = s.transpose(1, 2), t.transpose(1, 2)
        if self.individual:
            s = torch.stack([m(s[:, i]) for i, m in enumerate(self.Linear_Seasonal)], 1)
            t = torch.stack([m(t[:, i]) for i, m in enumerate(self.Linear_Trend)], 1)
            return s + t
        return self.Linear_Seasonal(s) + self.Linear_Trend(t)


@op("dlinear", "ser", space={"moving_avg": [25, 5, 13], "individual": [0, 1]}, tags={"block"},
    source="thuml/Time-Series-Library models/DLinear.py encoder (Zeng et al., AAAI 2023)")  # fmt: skip
def _dlinear(dims, kind, moving_avg, individual):
    return DLinear(dims[0], dims[1], moving_avg, bool(individual))


class TSMixerBlock(nn.Module):
    """TSLib TSMixer ResBlock on (B, C, T): an MLP over time, then an MLP over the channels."""

    def __init__(self, c, t, d_model=64, dropout=0.1):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(t, d_model), nn.ReLU(), nn.Linear(d_model, t), nn.Dropout(dropout)
        )
        self.channel = nn.Sequential(
            nn.Linear(c, d_model), nn.ReLU(), nn.Linear(d_model, c), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.temporal(x)
        return x + self.channel(x.transpose(1, 2)).transpose(1, 2)


@op("tsmixer_block", "ser", space={"d_model": [64, 32, 128], "dropout": [0.1, 0.0]}, tags={"block"},
    source="thuml/Time-Series-Library models/TSMixer.py ResBlock (Chen et al., TMLR 2023)")  # fmt: skip
def _tsmixer_block(dims, kind, d_model, dropout):
    return TSMixerBlock(dims[0], dims[1], d_model, dropout)


class TiDEBlock(nn.Module):
    """TSLib TiDE ResBlock: LayerNorm(fc2(relu(fc1 x)) + fc3 x) on the last axis."""

    def __init__(self, i, hidden, o, dropout=0.1):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(i, hidden), nn.Linear(hidden, o), nn.Linear(i, o)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(o)

    def forward(self, x):
        return self.ln(self.dropout(self.fc2(torch.relu(self.fc1(x)))) + self.fc3(x))


@op("tide_block", ("vec", "seq"), space={"hidden": [128, 64, 256], "out": [64, 32, 128], "dropout": [0.1, 0.0]},
    tags={"block"}, source="thuml/Time-Series-Library models/TiDE.py ResBlock (Das et al., TMLR 2023)")  # fmt: skip
def _tide_block(dims, kind, hidden, out, dropout):
    return TiDEBlock(dims[-1], hidden, out, dropout)


# ----------------------------------------------------------------------------- graphs


class GATv2(GraphLayer):
    """PyG GATv2Conv (concat heads, self loops) on a dense adjacency:
    e_ij = att . LeakyReLU(lin_r x_i + lin_l x_j), softmax over the neighbours j of i and i itself."""

    def __init__(self, n, i, o, heads=1, nonlinearity="identity"):
        super().__init__(n)
        if o % heads:
            raise ValueError("gatv2: heads must divide out")
        self.h, self.d = heads, o // heads
        self.act = act(nonlinearity)
        self.lin_l, self.lin_r = nn.Linear(i, o), nn.Linear(i, o)
        self.att = nn.Parameter(torch.empty(1, heads, self.d))
        self.bias = nn.Parameter(torch.zeros(o))
        nn.init.xavier_uniform_(self.att)

    def nodes(self, h, a):
        b, n, _ = h.shape
        xl = self.lin_l(h).view(b, n, self.h, self.d)
        xr = self.lin_r(h).view(b, n, self.h, self.d)
        e = (F.leaky_relu(xr.unsqueeze(2) + xl.unsqueeze(1), 0.2) * self.att).sum(-1)  # b i j h
        eye = torch.eye(n, device=a.device, dtype=torch.bool)
        mask = ((a > 0) | eye).unsqueeze(-1)
        alpha = torch.softmax(e.masked_fill(~mask, float("-inf")), dim=2)
        return self.act(torch.einsum("bijh,bjhd->bihd", alpha, xl).reshape(b, n, -1) + self.bias)


@op("gatv2", "graph", space={"out": [32, 64, 16], "heads": [1, 2, 4], "act": ["elu", "relu", "identity"]}, tags={"block"},
    source="torch_geometric GATv2Conv 2.6.1 (Brody et al., ICLR 2022), dense adjacency")  # fmt: skip
def _gatv2(dims, kind, out, heads, act):
    n, f = _feat(dims)
    return GATv2(n, f, out, heads, act)


class GCN2Layer(nn.Module):
    """PyG GCN2Conv with shared weights: h = (1 - alpha) A_hat x + alpha x0,
    out = (1 - beta) h + beta h W, beta = log(theta / layer + 1)."""

    def __init__(self, c, alpha, theta, layer):
        super().__init__()
        self.alpha, self.beta = alpha, math.log(theta / layer + 1)
        self.weight1 = nn.Parameter(torch.empty(c, c))
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, x, x0, a_hat):
        h = (1 - self.alpha) * (a_hat @ x) + self.alpha * x0
        return (1 - self.beta) * h + self.beta * (h @ self.weight1)


class GCNII(GraphLayer):
    """The deep stack of PyG's examples/gcn2_cora.py without its classifier:
    x0 = relu(lin x); x = relu(conv_l(dropout x, x0)) for every layer."""

    def __init__(self, n, i, hidden=64, layers=8, alpha=0.1, theta=0.5, dropout=0.0):
        super().__init__(n)
        self.lin = nn.Linear(i, hidden)
        self.convs = nn.ModuleList(GCN2Layer(hidden, alpha, theta, k + 1) for k in range(layers))
        self.dropout = nn.Dropout(dropout)

    def nodes(self, h, a):
        a_hat = renormalize(a)
        x = x0 = torch.relu(self.lin(self.dropout(h)))
        for conv in self.convs:
            x = torch.relu(conv(self.dropout(x), x0, a_hat))
        return x


@op("gcnii", "graph", space={"hidden": [64, 32, 128], "layers": [8, 4, 16], "alpha": [0.1, 0.2, 0.5],
    "theta": [0.5, 1.0, 1.5], "dropout": [0.0, 0.2, 0.5]}, tags={"block"},
    source="torch_geometric GCN2Conv 2.6.1 + examples/gcn2_cora.py (Chen et al., ICML 2020), dense adjacency")  # fmt: skip
def _gcnii(dims, kind, hidden, layers, alpha, theta, dropout):
    n, f = _feat(dims)
    return GCNII(n, f, hidden, layers, alpha, theta, dropout)


class APPNP(GraphLayer):
    """Predict then propagate (PyG APPNP after the MLP of its examples/citation code):
    h = lin2(relu(lin1 x)), then K steps x <- (1 - alpha) A_hat x + alpha h."""

    def __init__(self, n, i, hidden, o, k=10, alpha=0.1, dropout=0.0):
        super().__init__(n)
        self.lin1, self.lin2 = nn.Linear(i, hidden), nn.Linear(hidden, o)
        self.dropout = nn.Dropout(dropout)
        self.k, self.alpha = k, alpha

    def propagate(self, h, a):
        a_hat, x = renormalize(a), h
        for _ in range(self.k):
            x = (1 - self.alpha) * (a_hat @ x) + self.alpha * h
        return x

    def nodes(self, h, a):
        h = self.lin2(self.dropout(torch.relu(self.lin1(self.dropout(h)))))
        return self.propagate(h, a)


@op("appnp", "graph", space={"hidden": [64, 32, 128], "out": [32, 64, 16], "k": [10, 5, 20], "alpha": [0.1, 0.2, 0.05],
    "dropout": [0.0, 0.2, 0.5]}, tags={"block"},
    source="torch_geometric APPNP 2.6.1 (Gasteiger et al., ICLR 2019) after a two-layer MLP, dense adjacency")  # fmt: skip
def _appnp(dims, kind, hidden, out, k, alpha, dropout):
    n, f = _feat(dims)
    return APPNP(n, f, hidden, out, k, alpha, dropout)


class SGC(GraphLayer):
    """Simplified graph convolution: lin(A_hat^K x) (PyG SGConv)."""

    def __init__(self, n, i, o, k=2):
        super().__init__(n)
        self.lin = nn.Linear(i, o)
        self.k = k

    def nodes(self, h, a):
        a_hat = renormalize(a)
        for _ in range(self.k):
            h = a_hat @ h
        return self.lin(h)


@op("sgc", "graph", space={"out": [32, 64, 16], "k": [2, 1, 3]}, tags={"block"},
    source="torch_geometric SGConv 2.6.1 (Wu et al., ICML 2019), dense adjacency")  # fmt: skip
def _sgc(dims, kind, out, k):
    n, f = _feat(dims)
    return SGC(n, f, out, k)


# ----------------------------------------------------------------------------- field embeddings


class CIN(nn.Module):
    """FuxiCTR CompressedInteractionNet without its final linear layer: the sum-pooled feature
    maps of every layer, concatenated (B, sum(units))."""

    def __init__(self, fields, units=(16, 16)):
        super().__init__()
        self.units = list(units)
        self.cin_layer = nn.ModuleDict()
        for i, u in enumerate(self.units):
            cin = fields * self.units[i - 1] if i > 0 else fields**2
            self.cin_layer[f"layer_{i + 1}"] = nn.Conv1d(cin, u, kernel_size=1)

    def forward(self, e):
        b, _, d = e.shape
        pooled, x = [], e
        for i in range(len(self.units)):
            h = torch.einsum("bhd,bmd->bhmd", e, x).reshape(b, -1, d)
            x = self.cin_layer[f"layer_{i + 1}"](h)
            pooled.append(x.sum(-1))
        return torch.cat(pooled, -1)


@op("cin", "seq", "vec", space={"units": [16, 32, 64], "layers": [2, 1, 3]}, tags={"block", "interaction"},
    source="reczoo/FuxiCTR layers/interactions CompressedInteractionNet (xDeepFM, Lian et al., KDD 2018), without the output layer")  # fmt: skip
def _cin(dims, kind, units, layers):
    return CIN(dims[0], [units] * layers)


class AutoIntAttention(nn.Module):
    """FuxiCTR MultiHeadSelfAttention as AutoInt uses it (no bias, residual, ReLU, optional scale)."""

    def __init__(self, i, o=None, heads=1, dropout=0.0, use_scale=False, layer_norm=False):
        super().__init__()
        o = o or i
        if o % heads:
            raise ValueError("autoint_attn: heads must divide out")
        self.heads, self.head_dim = heads, o // heads
        self.scale = self.head_dim**0.5 if use_scale else None
        self.W_q, self.W_k, self.W_v = (nn.Linear(i, o, bias=False) for _ in range(3))
        self.W_res = nn.Linear(i, o, bias=False) if i != o else None
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(o) if layer_norm else None

    def forward(self, x):
        b = x.shape[0]
        q, k, v = (
            w(x).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
            for w in (self.W_q, self.W_k, self.W_v)
        )
        s = q @ k.transpose(-1, -2)
        if self.scale:
            s = s / self.scale
        out = (
            (self.dropout(s.softmax(-1)) @ v)
            .transpose(1, 2)
            .reshape(b, -1, self.heads * self.head_dim)
        )
        out = out + (self.W_res(x) if self.W_res is not None else x)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out.relu()


@op("autoint_attn", "seq", space={"out": [32, 16, 64], "heads": [2, 1, 4], "dropout": [0.0, 0.1],
    "use_scale": [0, 1], "layer_norm": [0, 1]}, tags={"attn", "interaction"},
    source="reczoo/FuxiCTR model_zoo/AutoInt MultiHeadSelfAttention (Song et al., CIKM 2019)")  # fmt: skip
def _autoint_attn(dims, kind, out, heads, dropout, use_scale, layer_norm):
    return AutoIntAttention(dims[-1], out, heads, dropout, bool(use_scale), bool(layer_norm))


class FieldSE(nn.Module):
    """FuxiCTR SqueezeExcitation (FiBiNET SENET): one weight per field from the mean of its embedding."""

    def __init__(self, fields, reduction=3, act="relu"):
        super().__init__()
        r = max(1, int(fields / reduction))
        self.excitation = nn.Sequential(
            nn.Linear(fields, r, bias=False), nn.ReLU(), nn.Linear(r, fields, bias=False),
            nn.ReLU() if act == "relu" else nn.Sigmoid(),
        )  # fmt: skip

    def forward(self, e):
        return e * self.excitation(e.mean(-1)).unsqueeze(-1)


@op("senet_fields", "seq", space={"reduction": [3, 2, 4], "act": ["relu", "sigmoid"]}, tags={"attn"},
    source="reczoo/FuxiCTR layers/attentions SqueezeExcitation (FiBiNET, Huang et al., RecSys 2019)")  # fmt: skip
def _senet_fields(dims, kind, reduction, act):
    return FieldSE(dims[0], reduction, act)


class BilinearInteraction(nn.Module):
    """FuxiCTR BilinearInteractionV2: (v_i W) * v_j for every pair i < j; W shared (`field_all`),
    one per left field (`field_each`) or one per pair (`field_interaction`)."""

    def __init__(self, fields, emb, bilinear_type="field_interaction"):
        super().__init__()
        self.bilinear_type = bilinear_type
        shape = {
            "field_all": (emb, emb),
            "field_each": (fields, emb, emb),
            "field_interaction": (fields * (fields - 1) // 2, emb, emb),
        }[bilinear_type]
        self.bilinear_W = nn.Parameter(torch.empty(shape))
        nn.init.xavier_normal_(self.bilinear_W)
        self.register_buffer(
            "triu_index", torch.triu_indices(fields, fields, offset=1), persistent=False
        )

    def forward(self, e):
        i, j = self.triu_index[0], self.triu_index[1]
        if self.bilinear_type == "field_interaction":
            left = (e.index_select(1, i).unsqueeze(2) @ self.bilinear_W).squeeze(2)
        elif self.bilinear_type == "field_all":
            left = (e @ self.bilinear_W).index_select(1, i)
        else:
            left = (e.unsqueeze(2) @ self.bilinear_W).squeeze(2).index_select(1, i)
        return left * e.index_select(1, j)


@op("bilinear_fields", "seq", space={"bilinear_type": ["field_interaction", "field_all", "field_each"]},
    tags={"block", "interaction"},
    source="reczoo/FuxiCTR layers/interactions BilinearInteractionV2 (FiBiNET, Huang et al., RecSys 2019)")  # fmt: skip
def _bilinear_fields(dims, kind, bilinear_type):
    if dims[0] < 2:
        raise ValueError("bilinear_fields needs at least two fields")
    return BilinearInteraction(dims[0], dims[1], bilinear_type)


# ----------------------------------------------------------------------------- sets, token pooling


class MAB(nn.Module):
    """Set Transformer multihead attention block (the official one: scores scaled by sqrt(dim_V),
    the projected query added back, then a ReLU feed-forward residual)."""

    def __init__(self, dq, dk, dv, heads, ln=False):
        super().__init__()
        if dv % heads:
            raise ValueError("set transformer: heads must divide the width")
        self.dim_V, self.num_heads = dv, heads
        self.fc_q, self.fc_k, self.fc_v = nn.Linear(dq, dv), nn.Linear(dk, dv), nn.Linear(dk, dv)
        self.ln0 = nn.LayerNorm(dv) if ln else None
        self.ln1 = nn.LayerNorm(dv) if ln else None
        self.fc_o = nn.Linear(dv, dv)

    def forward(self, q, k):
        q = self.fc_q(q)
        k, v = self.fc_k(k), self.fc_v(k)
        s = self.dim_V // self.num_heads
        q_, k_, v_ = (torch.cat(t.split(s, 2), 0) for t in (q, k, v))
        a = torch.softmax(q_.bmm(k_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        o = torch.cat((q_ + a.bmm(v_)).split(q.size(0), 0), 2)
        o = o if self.ln0 is None else self.ln0(o)
        o = o + F.relu(self.fc_o(o))
        return o if self.ln1 is None else self.ln1(o)


class ISAB(nn.Module):
    """Induced set attention: the set attends to `num_inds` learned points and back (O(N m))."""

    def __init__(self, i, o, heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, num_inds, o))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(o, i, o, heads, ln)
        self.mab1 = MAB(i, o, o, heads, ln)

    def forward(self, x):
        return self.mab1(x, self.mab0(self.I.repeat(x.size(0), 1, 1), x))


@op("isab", "seq", space={"out": [64, 32, 128], "heads": [4, 2, 1], "num_inds": [16, 8, 32], "ln": [0, 1]},
    tags={"attn", "block"}, source="juho-lee/set_transformer modules.py ISAB (Lee et al., ICML 2019)")  # fmt: skip
def _isab(dims, kind, out, heads, num_inds, ln):
    return ISAB(dims[-1], out, heads, num_inds, bool(ln))


class PMA(nn.Module):
    """Pooling by multihead attention with `num_seeds` learned seeds; the seeds are flattened."""

    def __init__(self, d, heads, num_seeds=1, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, num_seeds, d))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(d, d, d, heads, ln)

    def forward(self, x):
        return self.mab(self.S.repeat(x.size(0), 1, 1), x).flatten(1)


@op("pma", "seq", "vec", space={"heads": [4, 2, 1], "num_seeds": [1, 2, 4], "ln": [0, 1]}, tags={"adapt"},
    source="juho-lee/set_transformer modules.py PMA (Lee et al., ICML 2019), seeds flattened")  # fmt: skip
def _pma(dims, kind, heads, num_seeds, ln):
    return PMA(dims[-1], heads, num_seeds, bool(ln))


class AttentionPoolLatent(nn.Module):
    """timm AttentionPoolLatent (pool 'token', one latent query): attention of a learned query over
    the tokens, projection, then x + MLP(norm(x))."""

    def __init__(self, d, heads=8, mlp_ratio=4.0, norm=True):
        super().__init__()
        if d % heads:
            raise ValueError("attn_pool: heads must divide the width")
        self.num_heads, self.head_dim = heads, d // heads
        self.latent = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.latent, std=d**-0.5)
        self.q, self.kv = nn.Linear(d, d), nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d) if norm else nn.Identity()
        self.mlp = nn.Module()
        self.mlp.fc1, self.mlp.fc2 = (
            nn.Linear(d, int(d * mlp_ratio)),
            nn.Linear(int(d * mlp_ratio), d),
        )

    def forward(self, x):
        b, n, c = x.shape
        q = (
            self.q(self.latent.expand(b, -1, -1))
            .reshape(b, 1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k, v = (
            self.kv(x)
            .reshape(b, n, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, 1, c)
        x = self.proj(x)
        x = x + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm(x))))
        return x[:, 0]


@op("attn_pool", "seq", "vec", space={"heads": [8, 4, 2], "mlp_ratio": [4.0, 2.0], "norm": [1, 0]}, tags={"adapt"},
    source="timm.layers.AttentionPoolLatent 1.0.31 (MAP head of SigLIP / Beyer et al., arXiv:2205.01580)")  # fmt: skip
def _attn_pool(dims, kind, heads, mlp_ratio, norm):
    return AttentionPoolLatent(dims[-1], heads, mlp_ratio, bool(norm))


# ----------------------------------------------------------------------------- images


class InceptionDWConv2d(nn.Module):
    """timm InceptionDWConv2d: channels split into identity / k x k / 1 x band / band x 1 depthwise."""

    def __init__(self, c, square=3, band=11, branch_ratio=0.125):
        super().__init__()
        gc = int(c * branch_ratio)
        if gc < 1:
            raise ValueError(
                f"inception_next_block: {c} channels x branch_ratio {branch_ratio} leaves no branch"
            )
        self.dwconv_hw = nn.Conv2d(gc, gc, square, padding=square // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, (1, band), padding=(0, band // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, (band, 1), padding=(band // 2, 0), groups=gc)
        self.split_indexes = (c - 3 * gc, gc, gc, gc)

    def forward(self, x):
        i, hw, w, h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((i, self.dwconv_hw(hw), self.dwconv_w(w), self.dwconv_h(h)), dim=1)


class _ConvMlp(nn.Module):
    def __init__(self, c, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Conv2d(c, hidden, 1), nn.Conv2d(hidden, c, 1)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MetaNeXtBlock(nn.Module):
    """timm MetaNeXtBlock: Inception depthwise mixer -> BatchNorm -> 1x1 MLP, layer scale, residual."""

    def __init__(self, c, band=11, branch_ratio=0.125, mlp_ratio=4.0, ls_init=1e-6):
        super().__init__()
        self.token_mixer = InceptionDWConv2d(c, 3, band, branch_ratio)
        self.norm = nn.BatchNorm2d(c)
        self.mlp = _ConvMlp(c, int(mlp_ratio * c))
        self.gamma = nn.Parameter(ls_init * torch.ones(c))

    def forward(self, x):
        return self.mlp(self.norm(self.token_mixer(x))) * self.gamma.reshape(1, -1, 1, 1) + x


@op("inception_next_block", "map", space={"band": [11, 7, 9], "branch_ratio": [0.125, 0.25], "mlp_ratio": [4.0, 3.0, 2.0]},
    tags={"block"}, source="timm.models.inception_next MetaNeXtBlock 1.0.31 (InceptionNeXt, Yu et al., CVPR 2024)")  # fmt: skip
def _inception_next_block(dims, kind, band, branch_ratio, mlp_ratio):
    return MetaNeXtBlock(dims[0], band, branch_ratio, mlp_ratio)


class FocalModulation(nn.Module):
    """timm FocalModulation: hierarchical depthwise contexts, gated per level plus a global one,
    modulate the query (no attention matrix)."""

    def __init__(self, c, focal_window=3, focal_level=2, focal_factor=2):
        super().__init__()
        self.focal_level = focal_level
        self.input_split = [c, c, focal_level + 1]
        self.f = nn.Conv2d(c, 2 * c + focal_level + 1, 1)
        self.h = nn.Conv2d(c, c, 1)
        self.proj = nn.Conv2d(c, c, 1)
        self.focal_layers = nn.ModuleList()
        for k in range(focal_level):
            ks = focal_factor * k + focal_window
            self.focal_layers.append(
                nn.Sequential(nn.Conv2d(c, c, ks, groups=c, padding=ks // 2, bias=False), nn.GELU())
            )

    def forward(self, x):
        q, ctx, gates = torch.split(self.f(x), self.input_split, 1)
        ctx_all = 0
        for i, layer in enumerate(self.focal_layers):
            ctx = layer(ctx)
            ctx_all = ctx_all + ctx * gates[:, i : i + 1]
        ctx_all = ctx_all + F.gelu(ctx.mean((2, 3), keepdim=True)) * gates[:, self.focal_level :]
        return self.proj(q * self.h(ctx_all))


class FocalNetBlock(nn.Module):
    """timm FocalNetBlock (pre-norm, layer scale): x + ls(modulation(norm x)); x + ls(mlp(norm x))."""

    def __init__(self, c, focal_window=3, focal_level=2, mlp_ratio=4.0, ls_init=1e-4):
        super().__init__()
        self.norm1 = LayerNorm2d(c, eps=1e-6)
        self.modulation = FocalModulation(c, focal_window, focal_level)
        self.ls1 = LayerScale2d(c, ls_init)
        self.norm2 = LayerNorm2d(c, eps=1e-6)
        self.mlp = _ConvMlp(c, int(c * mlp_ratio))
        self.ls2 = LayerScale2d(c, ls_init)

    def forward(self, x):
        x = x + self.ls1(self.modulation(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


@op("focal_block", "map", space={"focal_window": [3, 5], "focal_level": [2, 1, 3], "mlp_ratio": [4.0, 2.0]},
    tags={"block"}, source="timm.models.focalnet FocalNetBlock 1.0.31 (FocalNet, Yang et al., NeurIPS 2022)")  # fmt: skip
def _focal_block(dims, kind, focal_window, focal_level, mlp_ratio):
    return FocalNetBlock(dims[0], focal_window, focal_level, mlp_ratio)


class NonLocalAttn(nn.Module):
    """timm NonLocalAttn: embedded dot-product non-local block; BatchNorm starts at zero so the
    block starts as the identity."""

    def __init__(self, c, use_scale=True, rd_ratio=1 / 8):
        super().__init__()
        rd = make_divisible(c * rd_ratio, 8)
        self.scale = c**-0.5 if use_scale else 1.0
        self.t, self.p, self.g = nn.Conv2d(c, rd, 1), nn.Conv2d(c, rd, 1), nn.Conv2d(c, rd, 1)
        self.z = nn.Conv2d(rd, c, 1)
        self.norm = nn.BatchNorm2d(c)
        for m in (self.t, self.p, self.g, self.z):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.norm.weight)

    def forward(self, x):
        b, _, h, w = x.shape
        t, p, g = (
            self.t(x).flatten(2).transpose(1, 2),
            self.p(x).flatten(2),
            self.g(x).flatten(2).transpose(1, 2),
        )
        att = F.softmax(torch.bmm(t, p) * self.scale, dim=2)
        y = torch.bmm(att, g).transpose(1, 2).reshape(b, -1, h, w)
        return self.norm(self.z(y)) + x


@op("non_local", "map", space={"rd_ratio": [0.125, 0.25, 0.5], "use_scale": [1, 0]}, tags={"attn"},
    source="timm.layers.NonLocalAttn 1.0.31 (Non-local Neural Networks, Wang et al., CVPR 2018)")  # fmt: skip
def _non_local(dims, kind, rd_ratio, use_scale):
    return NonLocalAttn(dims[0], bool(use_scale), rd_ratio)


class SRAttention(nn.Module):
    """timm pvt_v2 Attention: keys and values come from the map reduced by a strided convolution."""

    def __init__(self, d, heads=8, sr_ratio=1, qkv_bias=True):
        super().__init__()
        self.num_heads, self.head_dim = heads, d // heads
        self.q, self.kv = nn.Linear(d, d, bias=qkv_bias), nn.Linear(d, 2 * d, bias=qkv_bias)
        self.proj = nn.Linear(d, d)
        self.sr = nn.Conv2d(d, d, sr_ratio, stride=sr_ratio) if sr_ratio > 1 else None
        self.norm = nn.LayerNorm(d) if sr_ratio > 1 else None

    def forward(self, x, hw):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        if self.sr is not None:
            x = self.norm(
                self.sr(x.permute(0, 2, 1).reshape(b, c, *hw)).reshape(b, c, -1).permute(0, 2, 1)
            )
        k, v = (
            self.kv(x)
            .reshape(b, -1, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))


class _DWMlp(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, hidden), nn.Linear(hidden, d)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)

    def forward(self, x, hw):
        x = self.fc1(x)
        b, _, c = x.shape
        x = self.dwconv(x.transpose(1, 2).reshape(b, c, *hw)).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(x))


class PVTv2Block(nn.Module):
    """timm pvt_v2 Block on an NCHW map (tokens inside): spatial-reduction attention, then an MLP
    with a 3x3 depthwise convolution; LayerNorm eps 1e-6."""

    def __init__(self, c, heads=2, sr_ratio=2, mlp_ratio=4.0):
        super().__init__()
        if c % heads:
            raise ValueError("pvt_block: heads must divide the channels")
        self.norm1 = nn.LayerNorm(c, eps=1e-6)
        self.attn = SRAttention(c, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(c, eps=1e-6)
        self.mlp = _DWMlp(c, int(c * mlp_ratio))

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = t + self.attn(self.norm1(t), (h, w))
        t = t + self.mlp(self.norm2(t), (h, w))
        return t.transpose(1, 2).reshape(b, c, h, w)


@op("pvt_block", "map", space={"heads": [2, 1, 4], "sr_ratio": [2, 1, 4], "mlp_ratio": [4.0, 8.0, 2.0]},
    tags={"attn", "block"}, source="timm.models.pvt_v2 Block 1.0.31 (PVT v2, Wang et al., CVMJ 2022)")  # fmt: skip
def _pvt_block(dims, kind, heads, sr_ratio, mlp_ratio):
    if min(dims[1:]) < sr_ratio:
        raise ValueError(f"pvt_block: a {dims[1]}x{dims[2]} map cannot be reduced by {sr_ratio}")
    return PVTv2Block(dims[0], heads, sr_ratio, mlp_ratio)


class BlurPool2d(nn.Module):
    """timm BlurPool2d: reflect-pad, binomial low-pass filter, stride (anti-aliased down-sampling)."""

    def __init__(self, c, filt_size=3, stride=2):
        super().__init__()
        self.stride, self.c = stride, c
        self.padding = [((stride - 1) + (filt_size - 1)) // 2] * 4
        k = torch.tensor(
            [math.comb(filt_size - 1, i) for i in range(filt_size)], dtype=torch.float32
        )
        k = k / 2 ** (filt_size - 1)
        self.register_buffer(
            "filt", (k[:, None] * k[None, :])[None, None].repeat(c, 1, 1, 1), persistent=False
        )

    def forward(self, x):
        return F.conv2d(
            F.pad(x, self.padding, mode="reflect"), self.filt, stride=self.stride, groups=self.c
        )


@op("blur_pool", "map", space={"filt_size": [3, 5], "stride": [2, 1]}, tags={"down"},
    source="timm.layers.BlurPool2d 1.0.31 (Making Convolutional Networks Shift-Invariant Again, Zhang, ICML 2019)")  # fmt: skip
def _blur_pool(dims, kind, filt_size, stride):
    if min(dims[1:]) <= filt_size // 2:
        raise ValueError("blur_pool: the map is smaller than the reflect padding")
    return BlurPool2d(dims[0], filt_size, stride)


class SimAM(nn.Module):
    """SimAM: parameter-free 3-D attention from the energy of every neuron within its channel."""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = d / (4 * (d.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)


@op("simam", "map", space={"e_lambda": [1e-4, 1e-3, 1e-5]}, tags={"attn"},
    source="ZjjConan/SimAM networks/attentions/simam_module.py (Yang et al., ICML 2021)")  # fmt: skip
def _simam(dims, kind, e_lambda):
    if dims[1] * dims[2] < 2:
        raise ValueError("simam needs more than one position")
    return SimAM(e_lambda)


class GeM(nn.Module):
    """Generalised-mean pooling with a learned exponent: (mean x^p)^(1/p) over the positions."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return (
            F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1)))
            .pow(1.0 / self.p)
            .flatten(1)
        )


@op("gem", "map", "vec", space={"p": [3.0, 2.0, 4.0]}, tags={"adapt"},
    source="filipradenovic/cnnimageretrieval-pytorch cirtorch/layers/pooling.py GeM (Radenovic et al., TPAMI 2018)")  # fmt: skip
def _gem(dims, kind, p):
    return GeM(p)
