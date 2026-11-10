"""Data profiling and capacity estimation."""

from __future__ import annotations

import numpy as np


def twonn_intrinsic_dim(X: np.ndarray, max_points: int = 2000, seed: int = 0) -> float:
    """TwoNN estimator (Facco et al. 2017): d = N / sum(log(r2/r1)).

    This is the maximum-likelihood form of TwoNN (mu = r2/r1 is Pareto(d)). The paper fits a
    line to the empirical CDF after discarding the largest 10% of mu; that variant is not
    implemented. Written from memory: not verified against the paper (not in ref/).
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    if len(X) > max_points:
        X = X[rng.choice(len(X), max_points, replace=False)]
    dist, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]
    keep = r1 > 0
    mu = r2[keep] / r1[keep]
    return float(len(mu) / np.sum(np.log(mu)))


def mle_intrinsic_dim(X: np.ndarray, k: int = 20, max_points: int = 2000, seed: int = 0) -> float:
    """Maximum-likelihood intrinsic dimension (Levina & Bickel, NIPS 2004).

    Per point, with T_j = distance to the j-th neighbour:
        1 / m_k(x) = (1 / (k - 1)) * sum_{j=1}^{k-1} log(T_k(x) / T_j(x)).
    The per-point values are combined by averaging the INVERSES, d = 1 / mean_x(1 / m_k(x))
    (correction of MacKay & Ghahramani, 2005). Formula and correction are written from memory:
    not verified against the papers (they are not in ref/). The closed-form test checks it on
    Gaussian data of known dimension.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    if len(X) > max_points:
        X = X[rng.choice(len(X), max_points, replace=False)]
    k = min(k, len(X) - 1)
    if k < 3:
        return float("nan")
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    T = dist[:, 1:]
    keep = T[:, 0] > 0  # duplicated points have no defined ratio
    inv = np.mean(np.log(T[keep, -1:] / T[keep, :-1]), axis=1)
    return float(1.0 / np.mean(inv)) if keep.any() and np.mean(inv) > 0 else float("nan")


ESTIMATORS = {"twonn": twonn_intrinsic_dim, "mle": mle_intrinsic_dim}


def intrinsic_dim_report(X: np.ndarray, n_boot: int = 5, frac: float = 0.8, seed: int = 0) -> dict:
    """Both estimators on `n_boot` random subsamples (`frac` of the points, no replacement).

    Returns per estimator the mean and the std over subsamples, plus
      * `value`: the mean of the two estimator means (what the rest of DLNet uses),
      * `rel_disagreement`: |twonn - mle| / value,
      * `stable`: every std <= 15% of its mean AND the two estimators within 30% of each other.
    The thresholds are a heuristic. Subsampling makes neighbours farther apart, so the std
    measures sensitivity to the sample, it is not a confidence interval.
    """
    rng = np.random.default_rng(seed)
    if len(X) > 2000:
        X = X[rng.choice(len(X), 2000, replace=False)]
    m = max(int(frac * len(X)), min(len(X), 8))
    out = {}
    for name, fn in ESTIMATORS.items():
        vals = np.array([fn(X[rng.choice(len(X), m, replace=False)]) for _ in range(n_boot)])
        vals = vals[np.isfinite(vals)]
        out[name] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std()) if len(vals) else float("nan"),
        }
    means = [v["mean"] for v in out.values()]
    ok = all(np.isfinite(means)) and min(means) > 0
    value = float(np.mean(means)) if ok else float(np.nanmean(means))
    gap = abs(means[0] - means[1]) / value if ok else float("nan")
    out["value"] = value
    out["rel_disagreement"] = gap
    out["stable"] = bool(
        ok and gap <= 0.30 and all(out[k]["std"] <= 0.15 * out[k]["mean"] for k in ESTIMATORS)
    )
    return out


def capacity_hint(d_int: float, n_train: int, n_features: int) -> dict:
    """Capacity suggestion from the intrinsic dimension. HEURISTIC, information only.

    * `min_width`: smallest power of two >= 2 * d_int (a hidden layer narrower than the data
      manifold is a bottleneck), capped by the number of features rounded up to a power of two.
    * `scaling_exponent`: alpha ~ 4 / d (Sharma & Kaplan, arXiv:2004.10802): test loss is
      expected to fall like params^(-alpha), so a high d_int means more parameters buy little.
    Neither number prunes candidates: there is no measurement yet that they predict the best
    width on DLNet data (agent.md 16.1 item 5).
    """
    d = max(float(d_int), 1.0) if np.isfinite(d_int) else 1.0
    cap = 2 ** int(np.ceil(np.log2(max(n_features, 2))))
    return {
        "min_width": int(min(2 ** int(np.ceil(np.log2(2 * d))), max(cap, 8))),
        "scaling_exponent": round(4.0 / d, 3),
        "used_for_pruning": False,
    }


def rounded(rep: dict) -> dict:
    """JSON-friendly copy of an intrinsic_dim_report (3 decimals)."""
    r3 = lambda v: round(v, 3) if isinstance(v, float) else v
    return {
        k: {a: r3(b) for a, b in v.items()} if isinstance(v, dict) else r3(v)
        for k, v in rep.items()
    }


def profile(data: dict) -> dict:
    Xtr = data["Xnp"]["tr"]
    n, d = Xtr.shape
    rep = intrinsic_dim_report(Xtr, seed=data.get("seed", 0))
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
        "intrinsic_dim_estimators": rounded(rep),
        "capacity_hint": capacity_hint(rep["value"], n, d),
    }
    # Heuristic parameter budget (NOT derived from the paper): ~30 params per training sample.
    p["max_params"] = int(np.clip(30 * n, 2e4, 2e6))
    return p
