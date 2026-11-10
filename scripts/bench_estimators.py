"""Measure the intrinsic-dimension estimators on data of known dimension and on real tables.

    python scripts/bench_estimators.py

Synthetic sets have a known answer; the scikit-learn sets (bundled, no download) show whether
the two estimators agree and how much they move between subsamples. Results: agent.md 16.4.
"""

from __future__ import annotations

import numpy as np

from dlnet.profile import intrinsic_dim_report


def synthetic():
    rng = np.random.default_rng(0)
    for d in (5, 10, 20):  # d-dimensional Gaussian in a random linear subspace of R^50
        yield f"linear d={d} in R^50", d, rng.normal(size=(1000, d)) @ rng.normal(size=(d, 50))
    z = rng.normal(size=(1000, 6))
    z /= np.linalg.norm(z, axis=1, keepdims=True)  # unit sphere S^5: a curved 5-d manifold
    yield "sphere S^5 in R^20", 5, np.tanh(z @ rng.normal(size=(6, 20)))
    z = rng.normal(size=(1000, 10)) @ rng.normal(size=(10, 50))
    yield "linear d=10 + noise 0.05", 10, z + 0.05 * z.std() * rng.normal(size=z.shape)


def real():
    from sklearn import datasets

    for name in ("iris", "wine", "breast_cancer", "digits", "diabetes"):
        X = getattr(datasets, f"load_{name}")().data.astype(float)
        sd = X.std(0)
        yield f"sklearn {name} {X.shape}", None, (X - X.mean(0))[:, sd > 0] / sd[sd > 0]


def main():
    print("| dataset | true d | TwoNN | MLE (k=20) | disagreement | stable |")
    print("|---|---|---|---|---|---|")
    for name, d, X in [*synthetic(), *real()]:
        r = intrinsic_dim_report(X)
        print(
            f"| {name} | {d or '?'} | {r['twonn']['mean']:.1f} ± {r['twonn']['std']:.1f} | "
            f"{r['mle']['mean']:.1f} ± {r['mle']['std']:.1f} | {r['rel_disagreement']:.0%} | "
            f"{'yes' if r['stable'] else 'no'} |"
        )


if __name__ == "__main__":
    main()
