"""Measure what the search saves (M5): successive halving vs training every candidate fully,
and the rank correlation of the zero-cost proxies with the fully trained validation score.

    python scripts/bench_search.py [--kind tabular|hsi] [--per-family 3] [--epochs 30]

Synthetic data only, so the numbers describe the mechanism, not real datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from dlnet import core, hsi_data, proxies, search
from dlnet.plugins.hsi import HSIPlugin
from dlnet.plugins.tabular import TabularPlugin


def make(kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "tabular":
        X = rng.normal(size=(1500, 12)).astype("float32")
        y = (np.sin(X[:, 0] * 2) + X[:, 1] * X[:, 2] + 0.3 * rng.normal(size=1500) > 0).astype(int)
        return TabularPlugin(), core.prepare(X, y, "classification", seed=seed)
    h = w = 40
    gt = (np.arange(h)[:, None] // 10 % 2 * 2 + np.arange(w)[None, :] // 10 % 2 + 1).astype(int)
    base = np.cumsum(rng.normal(size=(4, 40)), axis=1) * 0.5
    cube = (base[gt - 1] + rng.normal(scale=2.0, size=(h, w, 40))).astype("float32")
    d = hsi_data.build(cube, gt, split="pixel", seed=seed, patch=5, pca=12, frac=(0.1, 0.1, 0.8))
    d["info"]["multilinear_rank_99"], d["info"]["cube_shape"] = [0, 0, 0], list(cube.shape)
    return HSIPlugin(), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="tabular", choices=["tabular", "hsi"])
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    plugin, data = make(a.kind, a.seed)
    cands = plugin.generate(plugin.profile(data), a.per_family, a.seed)
    far, quiet = time.time() + 1e9, lambda *_: None
    rungs = (max(a.epochs // 8, 1), max(a.epochs * 2 // 5, 2), a.epochs)

    zc = []
    for c in cands:
        torch.manual_seed(a.seed)
        zc.append(proxies.score(plugin.build_model(c, data["in_dim"], data["out_dim"]), data))

    t = time.time()
    full = []
    for c in cands:  # the expensive reference: every candidate, full epochs, seed 0
        tr = search._new_trainer(c, data, 0, plugin.build_model)
        tr.fit(a.epochs)
        full.append(tr.evaluate("va")["score"])
    t_full = time.time() - t

    t = time.time()
    res = search.run_search(
        data, cands, far, 1, plugin.build_model, rungs=rungs, seeds=1, log=quiet
    )
    t_sh = time.time() - t
    best_full = cands[int(np.argmax(full))]
    rep = proxies.correlation_report(zc, full, a.epochs)
    out = {
        "kind": a.kind,
        "candidates": len(cands),
        "rungs": rungs,
        "seconds_full": round(t_full, 1),
        "seconds_successive_halving": round(t_sh, 1),
        "time_saved_pct": round(100 * (1 - t_sh / t_full), 1),
        "same_top1": res[0]["cfg"] is best_full,
        "top1_val_full": round(max(full), 4),
        "top1_val_sh": round(res[0]["val_mean"], 4),
        "proxy_spearman_with_full_training": {k: v["spearman"] for k, v in rep["proxies"].items()},
        "proxy_n": {k: v["n"] for k, v in rep["proxies"].items()},
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
