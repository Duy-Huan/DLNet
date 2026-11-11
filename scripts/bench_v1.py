"""v1.0 families vs the earlier ones, same split / epochs / seeds per dataset (agent.md 9.9).

    python scripts/bench_v1.py --suite tabular --seeds 0 1 2 --out tab.json
    python scripts/bench_v1.py --suite image   --seeds 0 1 2 --out img.json
    python scripts/bench_v1.py --suite series  --seeds 0 1 2 --out ts.json

Data (no download): sklearn wine, breast_cancer, diabetes (tabular), sklearn digits 8x8 (image),
two synthetic series tasks (frequency of a noisy sine; which of 3 channels carries a delayed
copy of the others: needs cross-channel mixing). 60/20/20 split per seed (core.prepare), AdamW
lr 1e-3, wd 1e-4, batch 64, core.train_full (warm-up + cosine, best-validation checkpoint).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from dlnet import core
from dlnet.zoo import build_model, count_params, families

SUITES = {
    "tabular": {"families": ("mlp", "resmlp", "tabm", "realmlp"), "epochs": 60, "width": 128, "depth": 3},
    "image": {"families": ("cnn", "resnet", "vit", "mambavision", "kat"), "epochs": 30, "width": 32, "depth": 2},
    "series": {"families": ("tcn", "ts_transformer", "patchtst", "itransformer", "timemixer", "timexer", "xlstm",
                            "gated_deltanet"), "epochs": 30, "width": 32, "depth": 2},
}  # fmt: skip


def datasets(suite):
    from sklearn import datasets as ds

    if suite == "tabular":
        for name, f, task in (("wine", ds.load_wine, "classification"), ("breast_cancer", ds.load_breast_cancer, "classification"),
                              ("diabetes", ds.load_diabetes, "regression")):  # fmt: skip
            X, y = f(return_X_y=True)
            yield name, X, y, task, "vec", (X.shape[1],)
    elif suite == "image":
        X, y = ds.load_digits(return_X_y=True)
        yield "digits", X, y, "classification", "map", (1, 8, 8)
    else:
        rng = np.random.default_rng(0)
        n, t = 600, 64
        y = rng.integers(0, 3, n)
        tt = np.linspace(0, 1, t)
        X = np.sin(2 * np.pi * (3 + 3 * y)[:, None] * tt + rng.uniform(0, 6.3, (n, 1)))[
            :, None
        ] + 0.5 * rng.normal(size=(n, 1, t))
        yield "ts_freq", X.reshape(n, -1), y, "classification", "ser", (1, t)
        base = rng.normal(size=(n, 3, t))
        y2 = rng.integers(0, 3, n)
        for i in range(n):  # channel y2 is a delayed copy of the mean of the other two
            others = [c for c in range(3) if c != y2[i]]
            base[i, y2[i]] = np.roll(base[i, others].mean(0), 4) + 0.3 * rng.normal(size=t)
        yield "ts_crosschannel", base.reshape(n, -1), y2, "classification", "ser", (3, t)


def run(name, X, y, task, kind, dims, fam, seed, cfg):
    data = core.prepare(X, y, task, seed)
    for k in ("tr", "va", "te"):
        data["X" + k] = data["X" + k].reshape(-1, *dims)
    builder = {"vec": families.tabular, "map": families.image, "ser": families.timeseries}[kind]
    c = {"family": fam, "width": cfg["width"], "depth": cfg["depth"], "dropout": 0.1, "patch": 2}
    if fam == "realmlp":
        c.update(width=256, depth=3, dropout=0.15)
    if fam == "tabm":
        c.update(k=8)
    spec = builder(c, dims)
    torch.manual_seed(seed)
    head = "ensemble" if fam in families.ENSEMBLE_FAMILIES else "cls"
    model = build_model(
        {
            "spec": spec,
            "input": {"kind": kind, "dims": list(dims)},
            "head": head,
            "out_dim": data["out_dim"],
        }
    )
    t0 = time.time()
    tr, _ = core.train_full(model, data, 1e-3, 1e-4, 64, cfg["epochs"], seed=seed, recipe="plain")
    m = tr.evaluate("te")
    return {"dataset": name, "family": fam, "seed": seed, "params": count_params(model),
            "seconds": round(time.time() - t0, 1), "score": round(float(m["score"]), 4)}  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--families", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = SUITES[a.suite]
    rows = []
    for name, X, y, task, kind, dims in datasets(a.suite):
        for seed in a.seeds:
            for fam in a.families or cfg["families"]:
                r = run(name, X, y, task, kind, dims, fam, seed, cfg)
                rows.append(r)
                print(json.dumps(r), flush=True)
    summary = {}
    for r in rows:
        summary.setdefault(r["dataset"], {}).setdefault(r["family"], []).append(r["score"])
    summary = {
        d: {f: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] for f, v in fs.items()}
        for d, fs in summary.items()
    }
    with open(a.out, "w") as fh:
        json.dump({"rows": rows, "summary": summary}, fh, indent=1)
    for d, fs in summary.items():
        print(d)
        for f, (m, s) in sorted(fs.items(), key=lambda kv: -kv[1][0]):
            print(f"  {f:<16} {m:.4f} +- {s:.4f}")
