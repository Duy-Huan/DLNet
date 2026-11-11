"""Graph transformers and CTR interaction networks written from their OFFICIAL code (v1.0).

Graphs use DLNet's dense kind `graph` (B, N, F + N): node features followed by the adjacency row;
"j -> i is an edge" iff A[i, j] > 0 (symmetric for undirected graphs). The PyG layers the originals
use (GCNConv, GINConv, GATConv) are written densely with the same parameters and semantics.
Numerical check: scripts/verify_domain.py (job `verify-domain`, with torch_geometric installed).

- GraphGPS (Rampasek et al., NeurIPS 2022, arXiv:2205.12454; rampasek/GraphGPS
  graphgps/layer/gps_layer.py GPSLayer: local GIN + global multi-head attention, BatchNorm, FFN)
  and its random-walk structural encoding RWSE (graphgps/transform/posenc_stats.py
  get_rw_landing_probs: diag((D^-1 A)^k), k = 1..K, appended to the node features).
- Polynormer (Deng, Yue, Zhang, ICLR 2024, arXiv:2403.01232; cornell-zhang/Polynormer model.py:
  local GAT attention with polynomial gating, then linear global attention; the two training
  stages of the original become one module with the global part on).
- SGFormer (Wu et al., NeurIPS 2023, arXiv:2306.10759; qitianwu/SGFormer medium/ours.py TransConv
  with the simple global attention, + a GCN branch, graph_weight 0.8). The original attends over
  one large graph; DLNet applies it to each graph of the batch.
Kind `seq` of field embeddings (F, D), from `embed_fields` on categorical ids:
- FinalMLP (Mao et al., AAAI 2023, arXiv:2304.00902; reczoo/FuxiCTR model_zoo/FinalMLP):
  two MLP streams with feature-selection gates, bilinear interaction aggregation.
- DCNv3 / FCN (Li et al., arXiv:2407.13349; salmon1802/FCN src/FCNv1.py: exponential + linear
  cross networks); the per-network logits and the Tri-BCE loss are left to DLNet's head.
- WuKong (Zhang et al., ICML 2024, arXiv:2403.02545; FuxiCTR model_zoo/WuKong: stacked
  factorization-machine blocks + linear compression blocks).
"""

from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.nn import functional as F

from .core import op
from .ops_graph import GraphLayer, renormalize

# --------------------------------------------------------------------------- PyG layers


class DenseGINConv(nn.Module):
    """torch_geometric GINConv (eps 0, sum aggregation): nn((1 + eps) x_i + sum_j x_j)."""

    def __init__(self, mlp, eps=0.0):
        super().__init__()
        self.nn, self.eps = mlp, eps

    def forward(self, h, a):
        return self.nn((1 + self.eps) * h + torch.matmul((a > 0).to(h.dtype), h))


class DenseGCNConv(nn.Module):
    """torch_geometric GCNConv: D^-1/2 (A + I) D^-1/2 X W + b (weight in .lin, bias separate)."""

    def __init__(self, i, o):
        super().__init__()
        self.lin = nn.Linear(i, o, bias=False)
        self.bias = nn.Parameter(torch.zeros(o))

    def forward(self, h, a):
        return renormalize((a > 0).to(h.dtype)) @ self.lin(h) + self.bias


class DenseGATConv(nn.Module):
    """torch_geometric GATConv(concat=True, add_self_loops=False, bias=False), dense: a node with no
    neighbour gets 0, as in PyG."""

    def __init__(self, i, o, heads):
        super().__init__()
        self.h, self.o = heads, o
        self.lin = nn.Linear(i, heads * o, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, o))  # PyG layout (1, heads, out)
        self.att_dst = nn.Parameter(torch.empty(1, heads, o))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, a):
        b, n, _ = x.shape
        z = self.lin(x).view(b, n, self.h, self.o)
        s = (z * self.att_src.unsqueeze(0)).sum(-1)  # (b, n, h) source scores
        d = (z * self.att_dst.unsqueeze(0)).sum(-1)
        e = F.leaky_relu(d.unsqueeze(2) + s.unsqueeze(1), 0.2)  # (b, i, j, h)
        e = e.masked_fill(~(a > 0).unsqueeze(-1), float("-inf"))
        w = torch.nan_to_num(torch.softmax(e, dim=2), nan=0.0)
        return torch.einsum("bijh,bjhc->bihc", w, z).reshape(b, n, self.h * self.o)


# ------------------------------------------------------------------------------ RWSE


def rw_landing_probs(a, ksteps):
    """GraphGPS get_rw_landing_probs: diag(P^k), P = D^-1 A (rows of isolated nodes are 0)."""
    deg = a.sum(-1, keepdim=True)
    p = torch.where(deg > 0, a / deg.clamp_min(1e-12), torch.zeros_like(a))
    out, pk = [], torch.eye(a.shape[-1], dtype=a.dtype, device=a.device).expand_as(a)
    for k in range(1, max(ksteps) + 1):
        pk = pk @ p
        if k in ksteps:
            out.append(torch.diagonal(pk, dim1=-2, dim2=-1))
    return torch.stack(out, -1)


class RWSE(GraphLayer):
    def __init__(self, n, k):
        super().__init__(n)
        self.ksteps = list(range(1, k + 1))

    def nodes(self, h, a):
        return torch.cat([h, rw_landing_probs((a > 0).to(h.dtype), self.ksteps)], -1)


@op("rwse", "graph", space={"k": [8, 16, 4]}, tags={"stem"},
    source="rampasek/GraphGPS posenc_stats.get_rw_landing_probs (random-walk structural encoding)")  # fmt: skip
def _rwse(dims, kind, k):
    return RWSE(dims[0], k)


# -------------------------------------------------------------------------- GraphGPS


class GPSLayer(GraphLayer):
    """GPSLayer(local GIN, global Transformer, batch_norm): local + global, then FFN, BatchNorm."""

    def __init__(self, n, dim, heads=4, dropout=0.0, attn_dropout=0.0):
        super().__init__(n)
        self.local_model = DenseGINConv(
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        )
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.norm1_local, self.norm1_attn = nn.BatchNorm1d(dim), nn.BatchNorm1d(dim)
        self.ff_linear1, self.ff_linear2 = nn.Linear(dim, dim * 2), nn.Linear(dim * 2, dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _bn(bn, x):
        return bn(x.reshape(-1, x.shape[-1])).view_as(x)

    def nodes(self, h, a):
        loc = self._bn(self.norm1_local, h + self.drop(self.local_model(h, a)))
        att = self._bn(
            self.norm1_attn, h + self.drop(self.self_attn(h, h, h, need_weights=False)[0])
        )
        x = loc + att
        x = x + self.drop(self.ff_linear2(self.drop(F.relu(self.ff_linear1(x)))))
        return self._bn(self.norm2, x)


class GPSIn(GraphLayer):
    """Input projection to the GPS width (the GraphGPS feature encoder, a linear layer)."""

    def __init__(self, n, i, o):
        super().__init__(n)
        self.lin = nn.Linear(i, o)

    def nodes(self, h, a):
        return self.lin(h)


@op("gps_layer", "graph", space={"heads": [4, 2, 1], "dropout": [0.0, 0.1]}, tags={"block"},
    source="rampasek/GraphGPS graphgps/layer/gps_layer.py GPSLayer (local GIN + global attention, BatchNorm)")  # fmt: skip
def _gps(dims, kind, heads, dropout):
    n, f = dims[0], dims[1] - dims[0]
    if f % heads:
        raise ValueError(f"GPS: {f} node features not divisible by {heads} heads")
    return GPSLayer(n, f, heads, dropout)


@op("graph_lin", "graph", space={"out": [32, 64, 16]}, tags={"stem"},
    source="node-wise linear projection (GraphGPS input encoder)")  # fmt: skip
def _graph_lin(dims, kind, out):
    return GPSIn(dims[0], dims[1] - dims[0], out)


# -------------------------------------------------------------------------- Polynormer


class GlobalAttn(nn.Module):
    """Polynormer GlobalAttn: kernelised linear attention with sigmoid keys (qk shared)."""

    def __init__(self, hidden, heads, layers, beta=-1.0, dropout=0.5):
        super().__init__()
        self.hidden, self.heads, self.beta, self.dropout = hidden, heads, beta, dropout
        d = heads * hidden
        self.betas = nn.Parameter(
            torch.zeros(layers, d) if beta < 0 else torch.ones(layers, d) * beta
        )
        self.h_lins = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.k_lins = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.v_lins = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.lns = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.lin_out = nn.Linear(d, d)
        if beta < 0:
            nn.init.xavier_normal_(self.betas)

    def forward(self, x):  # (B, N, d)
        b, n, _ = x.shape
        for i in range(len(self.h_lins)):
            h = self.h_lins[i](x)
            k = torch.sigmoid(self.k_lins[i](x)).view(b, n, self.hidden, self.heads)
            v = self.v_lins[i](x).view(b, n, self.hidden, self.heads)
            kv = torch.einsum("bndh,bnmh->bdmh", k, v)
            num = torch.einsum("bndh,bdmh->bnmh", k, kv)
            den = torch.einsum("bndh,bdh->bnh", k, k.sum(1)).unsqueeze(2)
            beta = torch.sigmoid(self.betas[i]) if self.beta < 0 else self.betas[i]
            x = self.lns[i]((num / den).reshape(b, n, -1)) * (h + beta)
            x = F.dropout(F.relu(self.lin_out(x)), p=self.dropout, training=self.training)
        return x


class Polynormer(GraphLayer):
    """Polynormer forward with _global = True, without pred_global: node features (B, N, heads*hidden)."""

    def __init__(
        self,
        n,
        i,
        hidden,
        heads=1,
        local_layers=3,
        global_layers=2,
        in_dropout=0.15,
        dropout=0.5,
        beta=-1.0,
    ):
        super().__init__(n)
        d = heads * hidden
        self.in_drop, self.dropout, self.beta = in_dropout, dropout, beta
        self.betas = nn.Parameter(
            torch.zeros(local_layers, d) if beta < 0 else torch.ones(local_layers, d) * beta
        )
        self.h_lins = nn.ModuleList(nn.Linear(d, d) for _ in range(local_layers))
        self.local_convs = nn.ModuleList(
            DenseGATConv(d, hidden, heads) for _ in range(local_layers)
        )
        self.lins = nn.ModuleList(nn.Linear(d, d) for _ in range(local_layers))
        self.lns = nn.ModuleList(nn.LayerNorm(d) for _ in range(local_layers))
        self.lin_in = nn.Linear(i, d)
        self.ln = nn.LayerNorm(d)
        self.global_attn = GlobalAttn(hidden, heads, global_layers, beta, dropout)
        if beta < 0:
            nn.init.xavier_normal_(self.betas)

    def nodes(self, x, a):
        x = F.dropout(x, p=self.in_drop, training=self.training)
        x = F.dropout(self.lin_in(x), p=self.dropout, training=self.training)
        x_local = 0
        for i, conv in enumerate(self.local_convs):
            h = F.relu(self.h_lins[i](x))
            x = F.relu(conv(x, a) + self.lins[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            beta = torch.sigmoid(self.betas[i]) if self.beta < 0 else self.betas[i]
            x = (1 - beta) * self.lns[i](h * x) + beta * x
            x_local = x_local + x
        return self.global_attn(self.ln(x_local))


@op("polynormer", "graph", space={"hidden": [32, 64, 16], "heads": [1, 2], "local_layers": [3, 2], "global_layers": [2, 1],
    "dropout": [0.5, 0.2, 0.0]}, tags={"block"},
    source="cornell-zhang/Polynormer model.py (Deng et al., ICLR 2024), global stage on, without the prediction layer")  # fmt: skip
def _polynormer(dims, kind, hidden, heads, local_layers, global_layers, dropout):
    n = dims[0]
    return Polynormer(
        n, dims[1] - n, hidden, heads, local_layers, global_layers, min(0.15, dropout), dropout
    )


# ---------------------------------------------------------------------------- SGFormer


def full_attention_conv(qs, ks, vs):
    """SGFormer full_attention_conv for one graph per batch item: q, k normalised by their norm over
    the graph, (1 + q k^T / N) style linear attention with the +N v and +N terms."""
    qs = qs / torch.linalg.vector_norm(qs, dim=(1, 2, 3), keepdim=True)
    ks = ks / torch.linalg.vector_norm(ks, dim=(1, 2, 3), keepdim=True)
    n = qs.shape[1]
    kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
    num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs) + n * vs
    den = torch.einsum("bnhm,bhm->bnh", qs, ks.sum(1)).unsqueeze(-1) + n
    return num / den


class TransConvLayer(nn.Module):
    def __init__(self, i, o, heads):
        super().__init__()
        self.Wk, self.Wq, self.Wv = (
            nn.Linear(i, o * heads),
            nn.Linear(i, o * heads),
            nn.Linear(i, o * heads),
        )
        self.o, self.heads = o, heads

    def forward(self, x):
        b, n, _ = x.shape
        q = self.Wq(x).reshape(b, n, self.heads, self.o)
        k = self.Wk(x).reshape(b, n, self.heads, self.o)
        v = self.Wv(x).reshape(b, n, self.heads, self.o)
        return full_attention_conv(q, k, v).mean(dim=2)


class SGFormer(GraphLayer):
    """SGFormer (TransConv + GCN branch, aggregate "add", graph_weight 0.8) without the final fc."""

    def __init__(
        self,
        n,
        i,
        hidden,
        layers=1,
        heads=1,
        alpha=0.5,
        dropout=0.5,
        gcn_layers=2,
        graph_weight=0.8,
    ):
        super().__init__(n)
        self.alpha, self.dropout, self.graph_weight = alpha, dropout, graph_weight
        self.trans_conv = nn.Module()
        self.trans_conv.fcs = nn.ModuleList([nn.Linear(i, hidden)])
        self.trans_conv.bns = nn.ModuleList(
            [nn.LayerNorm(hidden)] + [nn.LayerNorm(hidden) for _ in range(layers)]
        )
        self.trans_conv.convs = nn.ModuleList(
            TransConvLayer(hidden, hidden, heads) for _ in range(layers)
        )
        self.gnn = nn.Module()
        self.gnn.convs = nn.ModuleList(
            [DenseGCNConv(i, hidden)]
            + [DenseGCNConv(hidden, hidden) for _ in range(gcn_layers - 1)]
        )
        self.gnn.bns = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(gcn_layers - 1))

    def nodes(self, h, a):
        t = self.trans_conv
        x = F.dropout(F.relu(t.bns[0](t.fcs[0](h))), p=self.dropout, training=self.training)
        layers = [x]
        for i, conv in enumerate(t.convs):
            x = self.alpha * conv(x) + (1 - self.alpha) * layers[i]
            x = F.dropout(t.bns[i + 1](x), p=self.dropout, training=self.training)
            layers.append(x)
        g = h
        for i, conv in enumerate(self.gnn.convs[:-1]):
            g = conv(g, a)
            g = self.gnn.bns[i](g.reshape(-1, g.shape[-1])).view_as(g)
            g = F.dropout(F.relu(g), p=self.dropout, training=self.training)
        g = self.gnn.convs[-1](g, a)
        return self.graph_weight * g + (1 - self.graph_weight) * x


@op("sgformer", "graph", space={"hidden": [32, 64, 16], "layers": [1, 2], "heads": [1, 2], "dropout": [0.5, 0.2, 0.0]},
    tags={"block"}, source="qitianwu/SGFormer medium/ours.py SGFormer + models.GCN (Wu et al., NeurIPS 2023), without fc")  # fmt: skip
def _sgformer(dims, kind, hidden, layers, heads, dropout):
    n = dims[0]
    return SGFormer(n, dims[1] - n, hidden, layers, heads, 0.5, dropout)


# ---------------------------------------------------------------------------- FuxiCTR


def mlp_block(i, hidden, out=None, out_act=None, bn=False, dropout=0.0):
    """FuxiCTR MLP_Block layers: Linear, [BN], ReLU, [Dropout] per hidden layer, [Linear, act]."""
    layers, dims = [], [i, *hidden]
    for a, b in itertools.pairwise(dims):
        layers.append(nn.Linear(a, b))
        if bn:
            layers.append(nn.BatchNorm1d(b))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    if out is not None:
        layers.append(nn.Linear(dims[-1], out))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class _MLPBlock(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.mlp = mlp_block(*a, **k)  # attribute `mlp` as in FuxiCTR

    def forward(self, x):
        return self.mlp(x)


class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super().__init__()
        self.num_heads, self.output_dim = num_heads, output_dim
        self.head_x_dim, self.head_y_dim = x_dim // num_heads, y_dim // num_heads
        self.w_x, self.w_y = nn.Linear(x_dim, output_dim), nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(
            torch.empty(num_heads * self.head_x_dim * self.head_y_dim, output_dim)
        )
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        out = self.w_x(x) + self.w_y(y)
        hx = x.view(-1, self.num_heads, self.head_x_dim)
        hy = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(hx.unsqueeze(2), self.w_xy.view(self.num_heads, self.head_x_dim, -1))
                          .view(-1, self.num_heads, self.output_dim, self.head_y_dim), hy.unsqueeze(-1)).squeeze(-1)  # fmt: skip
        return out + xy.sum(dim=1)


class FinalMLP(nn.Module):
    """FinalMLP on field embeddings (B, F, D): gates (no context), two MLPs, aggregation to out_dim."""

    def __init__(
        self, fields, emb, hidden=(64, 64, 64), fs_hidden=(64,), heads=1, out_dim=16, dropout=0.0
    ):
        super().__init__()
        fd = fields * emb
        self.mlp1 = _MLPBlock(fd, list(hidden), dropout=dropout)
        self.mlp2 = _MLPBlock(fd, list(hidden), dropout=dropout)
        self.fs_module = nn.Module()
        self.fs_module.fs1_ctx_bias = nn.Parameter(torch.zeros(1, emb))
        self.fs_module.fs2_ctx_bias = nn.Parameter(torch.zeros(1, emb))
        self.fs_module.fs1_gate = _MLPBlock(emb, list(fs_hidden), out=fd, out_act=nn.Sigmoid())
        self.fs_module.fs2_gate = _MLPBlock(emb, list(fs_hidden), out=fd, out_act=nn.Sigmoid())
        self.fusion_module = InteractionAggregation(hidden[-1], hidden[-1], out_dim, heads)

    def forward(self, e):
        flat = e.flatten(1)
        fs = self.fs_module
        g1 = fs.fs1_gate(fs.fs1_ctx_bias.repeat(flat.size(0), 1)) * 2
        g2 = fs.fs2_gate(fs.fs2_ctx_bias.repeat(flat.size(0), 1)) * 2
        return self.fusion_module(self.mlp1(flat * g1), self.mlp2(flat * g2))


@op("finalmlp", "seq", "vec", space={"hidden": [64, 128, 256], "depth": [3, 2], "heads": [1, 2, 4], "out": [16, 32, 1],
    "dropout": [0.0, 0.1]}, tags={"block"},
    source="reczoo/FuxiCTR model_zoo/FinalMLP (Mao et al., AAAI 2023), aggregation to `out` features")  # fmt: skip
def _finalmlp(dims, kind, hidden, depth, heads, out, dropout):
    if hidden % heads:
        raise ValueError(f"FinalMLP: {hidden} hidden units not divisible by {heads} heads")
    return FinalMLP(dims[0], dims[1], (hidden,) * depth, (64,), heads, out, dropout)


class CrossNetwork(nn.Module):
    """FCN ExponentialCrossNetwork (anchor = current x) / LinearCrossNetwork (anchor = x0), no BN,
    no layer norm, without the logit layer."""

    def __init__(self, d, layers=3, exponential=True, dropout=0.0):
        super().__init__()
        self.exponential = exponential
        self.w = nn.ModuleList(nn.Linear(d, d // 2, bias=False) for _ in range(layers))
        self.gamma = nn.ParameterList(nn.Parameter(torch.ones(d // 2)) for _ in range(layers))
        self.beta = nn.ParameterList(nn.Parameter(torch.zeros(d // 2)) for _ in range(layers))
        self.b = nn.ParameterList(nn.Parameter(torch.rand(d)) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x0 = x
        for i in range(len(self.w)):
            h1 = self.w[i](x)
            h = torch.cat([h1, self.gamma[i] * h1 + self.beta[i]], dim=-1)
            x = self.dropout((x if self.exponential else x0) * (h + self.b[i]) + x)
        return x


def fcn_embedding(e):
    """FCN MultiHeadFeatureEmbedding with one head: first halves of all fields, then second halves."""
    d = e.shape[-1]
    return torch.cat([e[..., : d // 2].flatten(1), e[..., d // 2 :].flatten(1)], -1)


class DCNv3(nn.Module):
    def __init__(self, fields, emb, exp_layers=3, lin_layers=3, dropout=0.0):
        super().__init__()
        d = fields * emb
        self.ECN = CrossNetwork(d, exp_layers, True, dropout)
        self.LCN = CrossNetwork(d, lin_layers, False, dropout)

    def forward(self, e):
        x = fcn_embedding(e)
        return torch.cat([self.ECN(x), self.LCN(x)], -1)


@op("dcnv3", "seq", "vec", space={"exp_layers": [3, 2, 4], "lin_layers": [3, 2, 4], "dropout": [0.0, 0.1]},
    tags={"block"}, source="salmon1802/FCN src/FCNv1.py ExponentialCrossNetwork + LinearCrossNetwork (DCNv3, arXiv:2407.13349)")  # fmt: skip
def _dcnv3(dims, kind, exp_layers, lin_layers, dropout):
    if dims[1] % 2:
        raise ValueError("DCNv3: the embedding width must be even (it is split in two halves)")
    return DCNv3(dims[0], dims[1], exp_layers, lin_layers, dropout)


class FactorizationMachineBlock(nn.Module):
    def __init__(self, i, o, emb, rank_k=8, hidden=(32, 32), dropout=0.0):
        super().__init__()
        self.emb, self.o = emb, o
        self.proj_Y = nn.Parameter(torch.randn(i, rank_k))
        self.layer_norm = nn.LayerNorm(i * rank_k)
        self.mlp = _MLPBlock(
            i * rank_k, list(hidden), out=o * emb, out_act=nn.ReLU(), dropout=dropout
        )

    def forward(self, x):
        fm = torch.bmm(x, x.transpose(1, 2) @ self.proj_Y).flatten(1)
        return self.mlp(self.layer_norm(fm)).view(-1, self.o, self.emb)


class WuKongLayer(nn.Module):
    def __init__(self, i, lcb, fmb, emb, rank_k=8, hidden=(32, 32), dropout=0.0):
        super().__init__()
        self.fmb = FactorizationMachineBlock(i, fmb, emb, rank_k, hidden, dropout)
        self.lcb = nn.Module()
        self.lcb.linear = nn.Linear(i, lcb, bias=False)
        self.layer_norm = nn.LayerNorm(emb)
        if i != lcb + fmb:
            self.residual_proj = nn.Linear(i, lcb + fmb)

    def forward(self, x):
        out = torch.cat([self.fmb(x), self.lcb.linear(x.transpose(1, 2)).transpose(1, 2)], dim=1)
        res = (
            self.residual_proj(x.transpose(1, 2)).transpose(1, 2)
            if hasattr(self, "residual_proj")
            else x
        )
        return self.layer_norm(out + res)


@op("wukong", "seq", space={"layers": [3, 2, 4], "lcb": [16, 40, 8], "fmb": [16, 40, 8], "rank_k": [8, 4, 16],
    "dropout": [0.0, 0.1]}, tags={"block"},
    source="reczoo/FuxiCTR model_zoo/WuKong (Zhang et al., ICML 2024) wukong_stack, without the output MLP")  # fmt: skip
def _wukong(dims, kind, layers, lcb, fmb, rank_k, dropout):
    f, e = dims
    return nn.Sequential(
        *[
            WuKongLayer(f if i == 0 else lcb + fmb, lcb, fmb, e, rank_k, (32, 32), dropout)
            for i in range(layers)
        ]
    )
