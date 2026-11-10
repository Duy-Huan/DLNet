"""Tensor decompositions of DATA (numpy only): truncated HOSVD / Tucker.

De Lathauwer, De Moor & Vandewalle, "A Multilinear Singular Value Decomposition", SIAM J. Matrix
Anal. Appl. 2000: T ~ S x_1 U_1 x_2 U_2 ... x_N U_N, U_n = leading left singular vectors of the
mode-n unfolding, S = T x_1 U_1^T ... x_N U_N^T. Truncation error (their Property 10):
    ||T - T_hat||_F^2 <= sum_n sum_{i > R_n} sigma_i^(n)^2.
Written from memory: not verified against the paper (it is not in ref/); the tests check exact
recovery of a tensor of known multilinear rank and the error bound.
"""

from __future__ import annotations

import numpy as np


def unfold(T: np.ndarray, mode: int) -> np.ndarray:
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def mode_product(T: np.ndarray, M: np.ndarray, mode: int) -> np.ndarray:
    """T x_mode M: multiply every mode-`mode` fibre of T by the matrix M."""
    return np.moveaxis(np.tensordot(M, T, axes=(1, mode)), 0, mode)


def mode_singular_values(T: np.ndarray, mode: int) -> np.ndarray:
    M = unfold(T, mode)
    G = M @ M.T if M.shape[0] <= M.shape[1] else M.T @ M
    return np.sqrt(np.linalg.eigvalsh(G)[::-1].clip(0))


def multilinear_rank(T: np.ndarray, energy: float = 0.99) -> list[int]:
    """Per mode, the number of singular values holding `energy` of the squared norm."""
    out = []
    for mode in range(T.ndim):
        w = mode_singular_values(T, mode) ** 2
        out.append(int(np.searchsorted(np.cumsum(w) / max(w.sum(), 1e-300), energy) + 1))
    return out


def hosvd(T: np.ndarray, ranks) -> tuple[np.ndarray, list[np.ndarray]]:
    """Truncated HOSVD. Returns (core, factors) with factors[n] of shape (T.shape[n], ranks[n])."""
    if len(ranks) != T.ndim:
        raise ValueError(f"need one rank per mode: {T.ndim} modes, got {list(ranks)}")
    factors = []
    for mode, r in enumerate(ranks):
        u, _, _ = np.linalg.svd(unfold(T, mode), full_matrices=False)
        factors.append(u[:, : max(1, min(int(r), u.shape[1]))])
    core = T
    for mode, u in enumerate(factors):
        core = mode_product(core, u.T, mode)
    return core, factors


def tucker_to_tensor(core: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    T = core
    for mode, u in enumerate(factors):
        T = mode_product(T, u, mode)
    return T


def hosvd_error_bound(T: np.ndarray, ranks) -> float:
    """Upper bound of ||T - T_hat||_F for the truncated HOSVD with these ranks."""
    return float(
        np.sqrt(sum(np.sum(mode_singular_values(T, m)[r:] ** 2) for m, r in enumerate(ranks)))
    )


def compress_report(T: np.ndarray, energy: float = 0.99) -> dict:
    """Tucker compression of a data tensor at the rank that keeps `energy` per mode."""
    ranks = multilinear_rank(T, energy)
    core, factors = hosvd(T, ranks)
    err = float(
        np.linalg.norm(T - tucker_to_tensor(core, factors)) / max(np.linalg.norm(T), 1e-300)
    )
    stored = core.size + sum(f.size for f in factors)
    return {
        "ranks": ranks,
        "relative_error": err,
        "stored_fraction": stored / T.size,
        "error_bound": hosvd_error_bound(T, ranks) / max(float(np.linalg.norm(T)), 1e-300),
    }
