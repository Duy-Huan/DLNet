"""Tensor-LoRA: a frozen linear layer plus a trainable delta weight in a tensor format (torch only).

    y = W0 x + b + (alpha / rank) * dW x,      W0, b frozen, dW built from small trainable cores.

Formats of dW (out x in), each cites the equation it follows:
  * lora       dW = B A, A in R^{r x in} Gaussian, B in R^{out x r} zero
               (Hu et al., arXiv:2106.09685, eq. 3 and the initialisation below it; checked).
  * tt         matrix Tensor-Train (Novikov et al., arXiv:1509.06569, sec. 3-4): with
               in = prod m_k and out = prod n_k,
               dW[(o_1..o_d), (i_1..i_d)] = G_1[i_1, o_1] G_2[i_2, o_2] ... G_d[i_d, o_d],
               G_k[i_k, o_k] an r_{k-1} x r_k matrix, r_0 = r_d = 1.
               Parameters: sum_k r_{k-1} m_k n_k r_k.
  * tr         Tensor Ring (Zhao et al., arXiv:1606.05535, eq. 1): the same cores with
               r_0 = r_d = r and a trace over the product. Using TR for a weight MATRIX with
               paired (i_k, o_k) indices mirrors the TT-matrix format; that combination is not
               taken from a paper.
  * kronecker  dW = sum_{i=1..n} A_i (x) (s_i t_i^T), A_i in R^{n x n}, s_i in R^{out/n x r},
               t_i in R^{in/n x r} (Karimi Mahabadi et al., Compacter, arXiv:2106.04647, eq. 5;
               checked). Compacter shares A_i between layers; here every layer owns its A_i
               unless `shared_a` is passed.
Every format starts at dW = 0 (one factor is zero), so the wrapped model first equals the frozen
one. The initialisation scales of the non-zero factors are not taken from the papers.

`tt_svd` (Oseledets, "Tensor-Train Decomposition", SIAM J. Sci. Comput. 2011, Algorithm 1,
written from memory: not verified against the paper) factorises an existing matrix into TT cores.
"""

from __future__ import annotations

import math

import torch
from torch import nn

FORMATS = ("lora", "tt", "tr", "kronecker")


def factor_pair(n: int) -> tuple[int, int]:
    """n = a * b with a <= b as close to each other as possible (a = 1 for a prime)."""
    best = (1, n)
    for a in range(1, math.isqrt(n) + 1):
        if n % a == 0:
            best = (a, n // a)
    return best


def cores_to_matrix(cores: list[torch.Tensor], ring: bool = False) -> torch.Tensor:
    """Contract cores G_k of shape (r_{k-1}, m_k, n_k, r_k) into the (prod n_k) x (prod m_k) matrix.

    ring=False needs r_0 = r_d = 1 (Tensor-Train); ring=True takes the trace over r_0 = r_d.
    """
    t = cores[0]  # (r0, m1, n1, r1)
    for g in cores[1:]:
        t = torch.einsum("a...r,rmns->a...mns", t, g)
    t = torch.einsum("a...a->...", t) if ring else t.squeeze(0).squeeze(-1)
    d = len(cores)  # t has shape (m1, n1, m2, n2, ..., md, nd)
    t = t.permute(*range(1, 2 * d, 2), *range(0, 2 * d, 2))
    n_out = math.prod(g.shape[2] for g in cores)
    return t.reshape(n_out, -1)


def tt_svd(W: torch.Tensor, in_factors, out_factors, rank: int) -> list[torch.Tensor]:
    """TT-SVD of a matrix W (out x in) into cores (r_{k-1}, m_k, n_k, r_k), every rank <= `rank`.

    Exact when `rank` is at least the TT-rank of W. Otherwise the Frobenius error is at most
    sqrt(sum of the squared singular values dropped over all steps) (Oseledets 2011, Thm. 2.2).
    """
    m, n = list(in_factors), list(out_factors)
    d = len(m)
    if W.shape != (math.prod(n), math.prod(m)) or len(n) != d:
        raise ValueError(f"W {tuple(W.shape)} does not match factors in={m} out={n}")
    t = W.reshape(*n, *m).permute(*[i for k in range(d) for i in (d + k, k)])  # (m1,n1,m2,n2,..)
    cores, r_prev = [], 1
    c = t.reshape(r_prev * m[0] * n[0], -1)
    for k in range(d - 1):
        u, s, vh = torch.linalg.svd(c, full_matrices=False)
        r = max(1, min(rank, len(s)))
        cores.append(u[:, :r].reshape(r_prev, m[k], n[k], r))
        c = (s[:r, None] * vh[:r]).reshape(r * m[k + 1] * n[k + 1], -1)
        r_prev = r
    cores.append(c.reshape(r_prev, m[-1], n[-1], 1))
    return cores


class TensorLoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable delta weight in one of FORMATS."""

    def __init__(
        self,
        base: nn.Linear,
        fmt: str = "lora",
        rank: int = 4,
        alpha: float | None = None,
        in_factors=None,
        out_factors=None,
        n: int = 2,
        shared_a: nn.Parameter | None = None,
    ):
        super().__init__()
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r}; use one of {FORMATS}")
        self.base, self.fmt, self.rank = base, fmt, rank
        for p in base.parameters():
            p.requires_grad_(False)
        out_f, in_f = base.weight.shape
        self.scale = (alpha if alpha is not None else rank) / rank
        std = 1.0 / math.sqrt(in_f)
        if fmt == "lora":
            self.A = nn.Parameter(torch.randn(rank, in_f) * std)
            self.B = nn.Parameter(torch.zeros(out_f, rank))
        elif fmt in ("tt", "tr"):
            m = list(in_factors or factor_pair(in_f))
            nn_ = list(out_factors or factor_pair(out_f))
            if math.prod(m) != in_f or math.prod(nn_) != out_f or len(m) != len(nn_):
                raise ValueError(f"factors {m} / {nn_} do not match a {out_f} x {in_f} weight")
            d = len(m)
            edge = rank if fmt == "tr" else 1
            ranks = [edge] + [rank] * (d - 1) + [edge]
            cores = []
            for k in range(d):
                shape = (ranks[k], m[k], nn_[k], ranks[k + 1])
                last = k == d - 1  # the last core is zero, so dW = 0 at the start
                cores.append(torch.zeros(shape) if last else torch.randn(shape) * std ** (1 / d))
            self.cores = nn.ParameterList(nn.Parameter(c) for c in cores)
        else:  # kronecker
            if in_f % n or out_f % n:
                raise ValueError(f"kronecker needs n={n} to divide both {in_f} and {out_f}")
            self.n = n
            self.A_k = (
                shared_a if shared_a is not None else nn.Parameter(torch.randn(n, n, n) * 0.5)
            )
            self.s = nn.Parameter(torch.zeros(n, out_f // n, rank))
            self.t = nn.Parameter(torch.randn(n, in_f // n, rank) * std)

    def delta(self) -> torch.Tensor:
        """The delta weight dW (out x in), without the alpha / rank scale."""
        if self.fmt == "lora":
            return self.B @ self.A
        if self.fmt in ("tt", "tr"):
            return cores_to_matrix(list(self.cores), ring=self.fmt == "tr")
        b = torch.einsum("ior,ijr->ioj", self.s, self.t)  # B_i = s_i t_i^T
        w = torch.einsum("iab,ioj->aobj", self.A_k, b)  # sum_i A_i (x) B_i
        return w.reshape(self.base.weight.shape)

    def forward(self, x):
        return self.base(x) + self.scale * nn.functional.linear(x, self.delta())

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """A plain nn.Linear with W0 + scale * dW (no extra cost at inference)."""
        lin = nn.Linear(self.base.in_features, self.base.out_features, self.base.bias is not None)
        lin.weight.copy_(self.base.weight + self.scale * self.delta())
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin


def expected_params(fmt: str, in_f: int, out_f: int, rank: int, n: int = 2) -> int:
    """Closed-form number of trainable parameters with the default 2-factor shapes."""
    if fmt == "lora":
        return rank * (in_f + out_f)
    if fmt == "kronecker":
        return n**3 + n * rank * (in_f // n + out_f // n)
    (m1, m2), (n1, n2) = factor_pair(in_f), factor_pair(out_f)
    edge = rank if fmt == "tr" else 1
    return edge * m1 * n1 * rank + rank * m2 * n2 * edge


def apply_tensor_lora(model: nn.Module, fmt="lora", rank=4, alpha=None, skip=(), **kw) -> nn.Module:
    """Freeze `model` and wrap every nn.Linear (except module names in `skip`) with a delta.

    Returns the same model object. Parameter-efficient transfer: only the cores train. Layers
    listed in `skip` (for example a new output head) stay fully trainable.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if full in skip:
                for p in child.parameters():
                    p.requires_grad_(True)
            else:
                setattr(mod, child_name, TensorLoRALinear(child, fmt, rank, alpha, **kw))
    return model


def merge_tensor_lora(model: nn.Module) -> nn.Module:
    """Fold every delta back into a plain nn.Linear (in place)."""
    for mod in list(model.modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, TensorLoRALinear):
                setattr(mod, child_name, child.merge())
    return model
