"""Data profiling and capacity estimation."""

from __future__ import annotations

import numpy as np


def twonn_intrinsic_dim(X: np.ndarray, max_points: int = 2000, seed: int = 0) -> float:
    """TwoNN estimator (Facco et al. 2017): d = N / sum(log(r2/r1))."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    if len(X) > max_points:
        X = X[rng.choice(len(X), max_points, replace=False)]
    dist, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]
    keep = r1 > 0
    mu = r2[keep] / r1[keep]
    return float(len(mu) / np.sum(np.log(mu)))


def profile(data: dict) -> dict:
    Xtr = data["Xnp"]["tr"]
    n, d = Xtr.shape
    d_int = twonn_intrinsic_dim(Xtr)
    s = np.linalg.svd(Xtr - Xtr.mean(0), compute_uv=False)
    energy = np.cumsum(s**2) / np.sum(s**2)
    p = {
        "task": data["task"],
        "n_train": int(n),
        "n_features": int(d),
        "n_classes": len(data["classes"]) if data["classes"] else None,
        "primary_metric": data["primary"],
        "intrinsic_dim": round(d_int, 2),
        "scaling_exponent_hat": round(4.0 / max(d_int, 1e-6), 3),  # alpha ~ 4/d (Sharma & Kaplan)
        "pca_dims_for_95pct_energy": int(np.searchsorted(energy, 0.95) + 1),
    }
    # Heuristic parameter budget (NOT derived from the paper): ~30 params per training sample.
    p["max_params"] = int(np.clip(30 * n, 2e4, 2e6))
    return p
