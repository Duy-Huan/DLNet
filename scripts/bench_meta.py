"""Measure the meta-learning recommender with leave-one-dataset-out (agent.md M9).

    python scripts/bench_meta.py [--budget 30s] [--store meta_store.jsonl]

Runs `dlnet design` on 12 small tabular datasets (scikit-learn bundled sets and synthetic ones),
storing every run in the meta store, then asks for each dataset: is its truly best family (best
rung-1 validation score) inside the top-3 families recommended from the OTHER datasets?
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from dlnet import cli, meta


def datasets():
    from sklearn import datasets as D

    for name in ("iris", "wine", "breast_cancer", "digits", "diabetes"):
        b = getattr(D, f"load_{name}")()
        yield name, b.data, b.target
    for i, (n, d, inf, k) in enumerate([(400, 10, 5, 2), (1500, 30, 10, 4), (800, 60, 6, 3)]):
        X, y = D.make_classification(n, d, n_informative=inf, n_classes=k, random_state=i)
        yield f"make_classification_{n}x{d}", X, y
    X, y = D.make_moons(600, noise=0.25, random_state=0)
    yield "moons", X, y
    X, y = D.make_friedman1(800, 10, noise=1.0, random_state=0)
    yield "friedman1", X, y
    X, y = D.make_regression(500, 20, n_informative=5, noise=10.0, random_state=0)
    yield "linear_regression", X, y
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 8))
    yield "xor_like", X, ((X[:, 0] * X[:, 1] > 0) ^ (X[:, 2] > 0)).astype(int)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="30s")
    ap.add_argument("--store", default="")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(a.store or Path(tmp) / "store.jsonl")
        if not store.exists():
            for name, X, y in datasets():
                df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
                df["y"] = y
                csv = Path(tmp) / f"{name}.csv"
                df.to_csv(csv, index=False)
                cli.main(["design", "--data", str(csv), "--target", "y", "--budget", a.budget,
                          "--topk", "1", "--per-family", "2", "--no-proxies", "--no-cache",
                          "--meta-store", str(store), "--out", str(Path(tmp) / f"o_{name}")])  # fmt: skip
        recs = meta.load(store)
        print("\n| dataset | best family | families scored |")
        print("|---|---|---|")
        for r in recs:
            fs = r["family_scores"]
            print(f"| {r['name']} | {max(fs, key=fs.get)} | {len(fs)} |")
        for topk in (1, 3):
            print(meta.leave_one_out(recs, "tabular", topk=topk))
