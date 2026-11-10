"""Deep unfolding for sparse linear inverse problems y = A x + noise (torch only).

Copied into exported packages as model.py. Used when the forward model A is known
(agent.md sec. 5): each iteration of a classical solver becomes one layer.

  * ista    x^{k+1} = eta_{lam/L}(x^k + (1/L) A^T (y - A x^k)), L = largest eigenvalue of A^T A
            (as stated in Chen et al., arXiv:1808.10038, eq. 2-3; checked against the PDF).
  * fista   Beck & Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm", SIAM J.
            Imaging Sci. 2009: momentum t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2. Written from
            memory: not verified against the paper (it is not in ref/).
  * LISTA   the weight-coupled form LISTA-CP of Chen et al. (arXiv:1808.10038, eq. 10; checked):
                x^{k+1} = eta_{theta_k}(x^k + W_k^T (y - A x^k)),   k = 0..K-1,  x^0 = 0,
            with W_k in R^{m x n} and theta_k trainable per layer. Initialised at W_k = A / L and
            theta_k = lam / L, so the untrained network IS K iterations of ISTA.
            Not implemented: support selection (LISTA-CPSS, eq. 13) and the paper's layer-wise
            training schedule; training here is plain end-to-end Adam on the MSE.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn


def soft(x: torch.Tensor, theta) -> torch.Tensor:
    """Soft-thresholding eta_theta(x) = sign(x) * max(|x| - theta, 0)."""
    return torch.sign(x) * torch.clamp(x.abs() - theta, min=0.0)


def lipschitz(A: torch.Tensor) -> float:
    """L = largest eigenvalue of A^T A = squared spectral norm of A."""
    return float(torch.linalg.matrix_norm(A, ord=2) ** 2)


def ista(A: torch.Tensor, Y: torch.Tensor, lam: float, K: int) -> torch.Tensor:
    """K ISTA iterations for min 0.5 ||y - A x||^2 + lam ||x||_1, rows of Y are observations."""
    L = lipschitz(A)
    X = Y.new_zeros(len(Y), A.shape[1])
    for _ in range(K):
        X = soft(X + (Y - X @ A.t()) @ A / L, lam / L)
    return X


def fista(A: torch.Tensor, Y: torch.Tensor, lam: float, K: int) -> torch.Tensor:
    L = lipschitz(A)
    X = Y.new_zeros(len(Y), A.shape[1])
    Z, t = X, 1.0
    for _ in range(K):
        X_new = soft(Z + (Y - Z @ A.t()) @ A / L, lam / L)
        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        Z = X_new + ((t - 1.0) / t_new) * (X_new - X)
        X, t = X_new, t_new
    return X


class LISTA(nn.Module):
    """LISTA-CP: K unfolded layers with trainable W_k (m x n) and thresholds theta_k."""

    def __init__(self, A: torch.Tensor, K: int, lam: float = 0.1, shared: bool = False):
        super().__init__()
        self.register_buffer("A", A.clone())
        self.K, self.shared = K, shared
        L = lipschitz(A)
        n_w = 1 if shared else K
        self.W = nn.ParameterList(nn.Parameter(A.clone() / L) for _ in range(n_w))
        self.theta = nn.Parameter(torch.full((K,), lam / L))

    def forward(self, Y: torch.Tensor, K: int | None = None) -> torch.Tensor:
        X = Y.new_zeros(len(Y), self.A.shape[1])
        for k in range(self.K if K is None else K):
            W = self.W[0 if self.shared else k]
            X = soft(X + (Y - X @ self.A.t()) @ W, self.theta[k])
        return X


def nmse_db(X_hat: torch.Tensor, X: torch.Tensor) -> float:
    """10 log10(sum ||x_hat - x||^2 / sum ||x||^2) over the batch (lower is better)."""
    return float(10.0 * torch.log10(((X_hat - X) ** 2).sum() / (X**2).sum().clamp_min(1e-30)))


def tune_lambda(solver, A, Y, X, K: int, grid=None) -> tuple[float, float]:
    """Best lam for a classical solver at exactly K iterations, by NMSE on (Y, X)."""
    if grid is None:
        top = float((Y @ A).abs().max())  # above this lam the solution is 0
        grid = [top * 10 ** (-e / 4) for e in range(2, 21)]
    scored = [(nmse_db(solver(A, Y, lam, K), X), lam) for lam in grid]
    best = min(scored)
    return best[1], best[0]


def train_lista(model: LISTA, Ytr, Xtr, Yva, Xva, epochs=100, lr=1e-3, batch=128, seed=0) -> float:
    """End-to-end Adam on the MSE; keeps the best-validation weights. Returns the best val NMSE."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    with torch.no_grad():
        best = nmse_db(model(Yva), Xva)
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        perm = torch.randperm(len(Ytr), generator=gen)
        for i in range(0, len(Ytr), batch):
            b = perm[i : i + batch]
            loss = ((model(Ytr[b]) - Xtr[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                model.theta.clamp_(min=0.0)  # a negative threshold is not a shrinkage
        sched.step()
        with torch.no_grad():
            v = nmse_db(model(Yva), Xva)
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best


def load_problem(path: str):
    """.npz with A (m x n), Y (N x m) observations and X (N x n) ground-truth signals."""
    import numpy as np

    z = np.load(path)
    missing = [k for k in ("A", "Y", "X") if k not in z]
    if missing:
        raise SystemExit(f"{path}: missing arrays {missing}; need A (m x n), Y (N x m), X (N x n)")
    A, Y, X = (torch.from_numpy(np.asarray(z[k], dtype=np.float32)) for k in ("A", "Y", "X"))
    if A.ndim != 2 or Y.shape != (len(X), A.shape[0]) or X.shape[1] != A.shape[1]:
        raise SystemExit(
            f"{path}: shapes A{tuple(A.shape)} Y{tuple(Y.shape)} X{tuple(X.shape)} do not fit Y = X A^T"
        )
    return A, Y, X


def split_indices(n: int, seed: int, frac=(0.6, 0.2, 0.2)):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return perm[:a], perm[a:b], perm[b:]
