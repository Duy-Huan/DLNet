"""Measure deep unfolding against the iterative solvers it unfolds (agent.md sec. 5, M8).

    python scripts/bench_unfold.py [--m 64 --n 128 --samples 5000 --snr 30]

Builds a synthetic sparse recovery problem y = A x + noise (Gaussian A with unit-norm columns,
Bernoulli-Gaussian x, as in Chen et al. arXiv:1808.10038 sec. 4 but smaller) and runs
`dlnet unfold` on it: LISTA versus ISTA / FISTA at the same number of iterations.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from dlnet import cli


def make_problem(m=64, n=128, samples=5000, sparsity=0.1, snr_db=30.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    X = rng.normal(size=(samples, n)) * (rng.random((samples, n)) < sparsity)
    Y0 = X @ A.T
    noise = rng.normal(size=Y0.shape)
    noise *= np.sqrt((Y0**2).mean() / 10 ** (snr_db / 10)) / np.sqrt((noise**2).mean())
    return A.astype(np.float32), (Y0 + noise).astype(np.float32), X.astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--layers", default="4,8,16")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        A, Y, X = make_problem(a.m, a.n, a.samples, snr_db=a.snr)
        np.savez(Path(tmp) / "p.npz", A=A, Y=Y, X=X)
        cli.main(["unfold", "--data", str(Path(tmp) / "p.npz"), "--layers", a.layers, "--epochs",
                  str(a.epochs), "--out", str(Path(tmp) / "out")])  # fmt: skip
        print((Path(tmp) / "out" / "leaderboard.md").read_text())
