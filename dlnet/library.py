"""Module library (torch only). Copied into every exported package as model.py.

Families:
  mlp            plain MLP
  resmlp         pre-norm residual MLP
  tt_resmlp      residual MLP whose hidden linears are Tensor-Train factorised
  tr_resmlp      residual MLP whose hidden linears are Tensor-Ring factorised
  lowrank_mlp    MLP whose hidden linears are low-rank factorised (U V)
  ft_transformer feature-tokenizer transformer (one token per feature)
  moe_mlp        mixture of experts: small MLP experts combined by a learned gate (soft or top-k
                 routing; blocks.MixtureOfExperts, method of yamsgithub/modular_deep_learning)
  fm_mlp         factorisation-machine style: a vector per feature, all pairwise dot products
                 (blocks.SecondOrderInteraction, ported from ylongqi/openrec) joined with the
                 raw features and fed to an MLP
"""

from __future__ import annotations

import torch
from torch import nn

try:  # inside the dlnet package
    from . import blocks
    from .zoo.ops_vec import LowRankLinear, TRLinear, TTLinear
except ImportError:  # inside an exported package (blocks.py and zoo/ sit next to model.py)
    import blocks
    from zoo.ops_vec import LowRankLinear, TRLinear, TTLinear

FAMILIES = (
    "mlp", "resmlp", "tt_resmlp", "tr_resmlp", "lowrank_mlp", "ft_transformer", "moe_mlp", "fm_mlp",
)  # fmt: skip


def _linear(kind: str, i: int, o: int, rank: int) -> nn.Module:
    if kind == "tt":
        return TTLinear(i, o, rank)
    if kind == "tr":
        return TRLinear(i, o, rank)
    if kind == "lowrank":
        return LowRankLinear(i, o, rank)
    return nn.Linear(i, o)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, dropout, kind="dense", rank=8):
        super().__init__()
        layers, d = [], in_dim
        for k in range(depth):
            lin = nn.Linear(d, width) if k == 0 else _linear(kind, width, width, rank)
            layers += [lin, nn.GELU(), nn.Dropout(dropout)]
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.head(self.body(x))


class ResBlock(nn.Module):
    def __init__(self, width, dropout, kind, rank):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.l1 = _linear(kind, width, width, rank)
        self.l2 = _linear(kind, width, width, rank)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.l2(self.drop(torch.nn.functional.gelu(self.l1(self.norm(x)))))


class ResMLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, dropout, kind="dense", rank=8):
        super().__init__()
        self.stem = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width, dropout, kind, rank) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.head(self.norm(self.blocks(self.stem(x))))


class FTTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, dropout):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_dim, width) * 0.02)
        self.b = nn.Parameter(torch.zeros(in_dim, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        t = x.unsqueeze(-1) * self.w + self.b
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        return self.head(self.norm(self.enc(t)[:, 0]))


class FMMLP(nn.Module):
    """x_i * v_i per feature -> pairwise dot products -> concat with x -> MLP."""

    def __init__(self, in_dim, out_dim, width, depth, dropout, rank):
        super().__init__()
        self.v = nn.Parameter(torch.randn(in_dim, rank) * 0.1)
        self.inter = blocks.SecondOrderInteraction()
        self.mlp = MLP(in_dim + in_dim * (in_dim - 1) // 2, out_dim, width, depth, dropout)

    def forward(self, x):
        return self.mlp(torch.cat([x, self.inter(x.unsqueeze(-1) * self.v)], 1))


def moe_mlp(in_dim, out_dim, width, depth, dropout, experts, routing):
    """`experts` MLPs of width `width // 2`, a linear gate; routing 'soft' or 'topk' (k = 2)."""
    exp = [MLP(in_dim, out_dim, max(width // 2, 8), depth, dropout) for _ in range(experts)]
    return blocks.MixtureOfExperts(exp, nn.Linear(in_dim, experts), mode=routing, k=2)


def build_model(cfg: dict, in_dim: int, out_dim: int) -> nn.Module:
    if cfg.get("spec"):  # an architecture of the module zoo (dlnet/zoo)
        try:
            from . import zoo
        except ImportError:  # exported package: zoo/ sits next to model.py
            import zoo
        return zoo.build_model(cfg, in_dim, out_dim)
    f, w, d, p, r = cfg["family"], cfg["width"], cfg["depth"], cfg["dropout"], cfg.get("rank", 8)
    if f == "mlp":
        return MLP(in_dim, out_dim, w, d, p)
    if f == "lowrank_mlp":
        return MLP(in_dim, out_dim, w, d, p, kind="lowrank", rank=r)
    if f == "resmlp":
        return ResMLP(in_dim, out_dim, w, d, p)
    if f == "tt_resmlp":
        return ResMLP(in_dim, out_dim, w, d, p, kind="tt", rank=r)
    if f == "tr_resmlp":
        return ResMLP(in_dim, out_dim, w, d, p, kind="tr", rank=r)
    if f == "moe_mlp":
        return moe_mlp(in_dim, out_dim, w, d, p, cfg.get("experts", 4), cfg.get("routing", "soft"))
    if f == "fm_mlp":
        return FMMLP(in_dim, out_dim, w, d, p, r)
    if f == "ft_transformer":
        return FTTransformer(in_dim, out_dim, w, d, p)
    raise ValueError(f"unknown family {f}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
