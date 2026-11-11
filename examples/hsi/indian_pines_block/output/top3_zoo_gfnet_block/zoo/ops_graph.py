"""Graph and categorical-id blocks (torch only).

Kind `graph`: a sample is a dense graph (N, F + N): per node F features followed by that node's row of
the N x N adjacency; every graph op returns the new node features with the adjacency appended again, so
ops compose. Small graphs only (dense N x N). Kind `ids`: F categorical fields; dims are the field
cardinalities and the tensor is (B, F) integers.

Sources (see THIRD_PARTY.md):
  rasbt/deeplearning-models   GraphNet on a pixel-grid graph: Gaussian-kernel adjacency with symmetric
                              normalisation, neighbour averaging, spectral filters from the Laplacian
                              eigenvectors (gnn-basic-1, gnn-basic-graph-spectral-1)
  ylongqi/openrec             LatentFactor, MLP, SecondOrderFeatureInteraction, GMF
  papers, from memory         GCN (Kipf & Welling, arXiv:1609.02907), GAT (Velickovic et al., 1710.10903),
                              GraphSAGE mean aggregator (Hamilton et al., 1706.02216), GIN (Xu et al.,
                              1810.00826), NCF / GMF (He et al., 1708.05031)
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .core import op

# --------------------------------------------------------------------------- graph


def split_graph(x, n):
    """(B, N, F + N) -> features (B, N, F), adjacency (B, N, N)."""
    return x[..., : x.shape[-1] - n], x[..., x.shape[-1] - n :]


def renormalize(a):
    """D^-1/2 (A + I) D^-1/2 (the GCN renormalisation trick)."""
    a = a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    d = a.sum(-1).clamp_min(1e-12).pow(-0.5)
    return d.unsqueeze(-1) * a * d.unsqueeze(-2)


class GraphLayer(nn.Module):
    """Base: subclasses implement `nodes(h, a) -> (B, N, F')`; the adjacency is carried through."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        h, a = split_graph(x, self.n)
        return torch.cat([self.nodes(h, a), a], -1)


class GCN(GraphLayer):
    """H' = act(norm(A + I) H W + b) (Kipf & Welling)."""

    def __init__(self, n, i, o, act):
        super().__init__(n)
        self.lin, self.act = nn.Linear(i, o), nn.ReLU() if act == "relu" else nn.Identity()

    def nodes(self, h, a):
        return self.act(renormalize(a) @ self.lin(h))


class GAT(GraphLayer):
    """Multi-head graph attention: e_ij = LeakyReLU(a_src . Wh_i + a_dst . Wh_j), softmax over neighbours and self."""

    def __init__(self, n, i, o, heads):
        super().__init__(n)
        if o % heads:
            raise ValueError("heads must divide out")
        self.h, self.d = heads, o // heads
        self.lin = nn.Linear(i, o, bias=False)
        self.src, self.dst = (
            nn.Parameter(torch.randn(heads, self.d) * 0.1),
            nn.Parameter(torch.randn(heads, self.d) * 0.1),
        )

    def nodes(self, h, a):
        b, n, _ = h.shape
        z = self.lin(h).view(b, n, self.h, self.d)
        e = nn.functional.leaky_relu(
            (z * self.src).sum(-1).unsqueeze(2) + (z * self.dst).sum(-1).unsqueeze(1), 0.2
        )  # b n n h
        mask = (a + torch.eye(n, device=a.device) > 0).unsqueeze(-1)
        att = torch.softmax(e.masked_fill(~mask, -1e9), dim=2)
        return nn.functional.elu(torch.einsum("bijh,bjhd->bihd", att, z).reshape(b, n, -1))


class SAGE(GraphLayer):
    """GraphSAGE with the mean aggregator: W [h || mean of neighbours]."""

    def __init__(self, n, i, o):
        super().__init__(n)
        self.lin = nn.Linear(2 * i, o)

    def nodes(self, h, a):
        mean = (a @ h) / a.sum(-1, keepdim=True).clamp_min(1.0)
        return torch.relu(self.lin(torch.cat([h, mean], -1)))


class GIN(GraphLayer):
    """Graph isomorphism layer: MLP((1 + eps) h + sum of neighbours), eps learned."""

    def __init__(self, n, i, o):
        super().__init__(n)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))

    def nodes(self, h, a):
        return torch.relu(self.mlp((1 + self.eps) * h + a @ h))


def _feat(dims):
    n = dims[0]
    f = dims[1] - n
    if f < 1:
        raise ValueError("a graph sample is (N, F + N) with F >= 1")
    return n, f


@op(
    "gcn",
    "graph",
    space={"out": [16, 32, 64], "act": ["relu", "identity"]},
    tags={"block"},
    source="GCN (Kipf & Welling, arXiv:1609.02907), from memory",
)
def _gcn(dims, kind, out, act):
    n, f = _feat(dims)
    return GCN(n, f, out, act)


@op(
    "gat",
    "graph",
    space={"out": [16, 32, 64], "heads": [1, 2, 4]},
    tags={"block"},
    source="GAT (Velickovic et al., arXiv:1710.10903), from memory; dense attention",
)
def _gat(dims, kind, out, heads):
    n, f = _feat(dims)
    return GAT(n, f, out, heads)


@op(
    "sage",
    "graph",
    space={"out": [16, 32, 64]},
    tags={"block"},
    source="GraphSAGE mean aggregator (arXiv:1706.02216), from memory",
)
def _sage(dims, kind, out):
    n, f = _feat(dims)
    return SAGE(n, f, out)


@op(
    "gin",
    "graph",
    space={"out": [16, 32, 64]},
    tags={"block"},
    source="GIN (Xu et al., arXiv:1810.00826), from memory",
)
def _gin(dims, kind, out):
    n, f = _feat(dims)
    return GIN(n, f, out)


class Readout(nn.Module):
    def __init__(self, n, mode):
        super().__init__()
        self.n, self.mode = n, mode

    def forward(self, x):
        h, _ = split_graph(x, self.n)
        return {"mean": h.mean(1), "max": h.amax(1), "sum": h.sum(1)}[self.mode]


@op(
    "graph_readout",
    "graph",
    "vec",
    space={"mode": ["mean", "max", "sum"]},
    tags={"adapt"},
    source="graph pooling over nodes",
)
def _readout(dims, kind, mode):
    _feat(dims)
    return Readout(dims[0], mode)


# ------------------------------------------------- rasbt grid graphs on image maps


def grid_adjacency(h, w, normalize=True):
    """rasbt `precompute_adjacency_matrix`: Gaussian kernel of pixel distance, cut at 0.01, optionally D^-1/2 A D^-1/2."""
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coord = torch.stack([xs, ys], -1).reshape(-1, 2).double() / max(h, w)
    sigma = 0.05 * math.pi
    a = torch.exp(-torch.cdist(coord, coord) / sigma**2)
    a = a.masked_fill(a < 0.01, 0.0)
    if normalize:
        d = (a.sum(1) + 1e-5) ** -0.5
        a = d[:, None] * a * d[None, :]
    return a.float()


class GridAvg(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.register_buffer("a", grid_adjacency(h, w))

    def forward(self, x):
        b, c, h, w = x.shape
        return (x.reshape(b, c, -1) @ self.a.t()).reshape(b, c, h, w)


@op(
    "grid_graph_avg",
    "map",
    space={},
    tags={"block"},
    source="rasbt/deeplearning-models gnn-basic-1: neighbour averaging with the normalised Gaussian grid adjacency",
)
def _grid_avg(dims, kind):
    if dims[1] * dims[2] > 1024:
        raise ValueError("grid graph is dense: at most 1024 pixels")
    return GridAvg(dims[1], dims[2])


class GridSpectral(nn.Module):
    """Spectral graph filters on the pixel grid (rasbt gnn-basic-graph-spectral-1).

    V = the k eigenvectors of the normalised Laplacian with the smallest eigenvalues; each channel c and
    filter f gives V ((V^T x_c) * (V^T W_f)). The notebook keeps W as a fixed random buffer; here W is trained.
    """

    def __init__(self, c, h, w, k, filters):
        super().__init__()
        a = grid_adjacency(h, w, normalize=False).double()
        d = (a.sum(1) + 1e-5) ** -0.5
        lap = torch.eye(h * w, dtype=torch.double) - d[:, None] * a * d[None, :]
        _, vecs = torch.linalg.eigh(lap)
        self.register_buffer("v", vecs[:, : min(k, h * w)].float())
        self.w = nn.Parameter(torch.empty(h * w, filters))
        nn.init.kaiming_uniform_(self.w)

    def forward(self, x):
        b, c, h, w = x.shape
        xh = torch.einsum("nk,bcn->bck", self.v, x.reshape(b, c, -1))
        wh = self.v.t() @ self.w  # (k, filters)
        z = xh.unsqueeze(2) * wh.t()  # (b, c, filters, k)
        return torch.einsum("nk,bcfk->bcfn", self.v, z).reshape(b, -1, h, w)


@op("grid_spectral_conv", "map", space={"k": [10, 20], "filters": [2, 4]}, tags={"block"},
    source="rasbt/deeplearning-models gnn-basic-graph-spectral-1 (eigh of the dense Laplacian instead of scipy eigsh; trained filters)")  # fmt: skip
def _grid_spectral(dims, kind, k, filters):
    if dims[1] * dims[2] > 1024:
        raise ValueError("grid graph is dense: at most 1024 pixels")
    return GridSpectral(dims[0], dims[1], dims[2], k, filters)


# ----------------------------------------------------------------------------- ids


class EmbedFields(nn.Module):
    """One embedding table per field (openrec LatentFactor: normal init, std 0.01): (B, F) -> (B, F, dim)."""

    def __init__(self, cards, dim):
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(int(c), dim) for c in cards)
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.01)

    def forward(self, ids):
        return torch.stack([t(ids[:, i]) for i, t in enumerate(self.tables)], 1)


@op(
    "embed_fields",
    "ids",
    "seq",
    space={"dim": [8, 16, 32, 64]},
    tags={"stem"},
    source="ylongqi/openrec LatentFactor (tf2), one table per id field",
)
def _embed(dims, kind, dim):
    return EmbedFields(dims, dim)


class GMF(nn.Module):
    def forward(self, x):
        return x[:, 0] * x[:, 1]


@op(
    "gmf",
    "seq",
    "vec",
    space={},
    tags={"adapt"},
    source="Generalised matrix factorisation (He et al., arXiv:1708.05031; ylongqi/openrec gmf.py): product of the first two field embeddings",
)
def _gmf(dims, kind):
    if dims[0] != 2:
        raise ValueError("gmf takes exactly two fields (user, item)")
    return GMF()


class PairDots(nn.Module):
    """openrec SecondOrderFeatureInteraction: dot products of every pair of field embeddings: (B, F, D) -> (B, F (F-1) / 2)."""

    def __init__(self, f, self_interaction):
        super().__init__()
        self.i, self.j = torch.triu_indices(f, f, offset=0 if self_interaction else 1)

    def forward(self, x):
        g = x @ x.transpose(1, 2)
        return g[:, self.i, self.j]


@op(
    "pair_dots",
    "seq",
    "vec",
    space={"self_interaction": [0, 1]},
    tags={"adapt", "interaction"},
    source="ylongqi/openrec SecondOrderFeatureInteraction (tf2) ported to PyTorch",
)
def _pair_dots(dims, kind, self_interaction):
    if dims[0] < 2:
        raise ValueError("pair_dots needs at least two fields")
    return PairDots(dims[0], bool(self_interaction))
