"""Frequency-domain HSI families vs the spatial ones, same split / epochs / seeds (agent.md 9.8).

    python scripts/bench_freq_hsi.py --data Indian_pines_corrected.mat --gt Indian_pines_gt.mat \
        --pca 30 --seeds 0 1 2 --epochs 40 --out ip_pca30.json

Every family: width 64, depth 2, dropout 0.2, AdamW lr 1e-3, wd 1e-4, batch 64, the full recipe of
core.train_full (warm-up + cosine, best-validation checkpoint), the same split per seed (pixel, or spatial blocks with --split block).
Reports test OA (accuracy), AA (balanced accuracy), kappa, parameters and seconds.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from dlnet import core, hsi_data
from dlnet.zoo import build_model, count_params, families

BASE = ("cnn2d", "resnet2d", "convnext2d", "hybridsn")
FREQ = ("gfnet2d", "fno2d", "afno2d", "ffc2d", "fcanet2d", "spec_fft2d", "fourier_ss")


def run(data, family, seed, epochs):
    cfg = {"family": family, "width": 64, "depth": 2, "dropout": 0.2}
    if family == "hybridsn":
        cfg["width"] = 8
    dims = (data["in_dim"], data["patch"], data["patch"])
    spec = families.hsi(cfg, dims)
    torch.manual_seed(seed)
    model = build_model({"spec": spec, "input": {"kind": "map", "dims": list(dims)}, "head": "cls",
                         "out_dim": data["out_dim"], "dropout": 0.2})  # fmt: skip
    t = time.time()
    tr, _ = core.train_full(model, data, 1e-3, 1e-4, 64, epochs, seed=seed, recipe="plain")
    m = tr.evaluate("te")
    return {"family": family, "seed": seed, "params": count_params(model), "seconds": round(time.time() - t, 1),
            **{k: round(float(m[k]), 4) for k in ("accuracy", "balanced_accuracy", "kappa")}}  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pca", type=int, default=30)
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--split", default="pixel", choices=["pixel", "block"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--families", nargs="+", default=[*BASE, *FREQ])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cube, gt = hsi_data.load(a.data, a.gt)
    rows = []
    for seed in a.seeds:
        data = hsi_data.build(cube, gt, a.split, seed, a.patch, a.pca)
        for f in a.families:
            r = run(data, f, seed, a.epochs)
            rows.append(r)
            print(json.dumps(r), flush=True)
    summary = {}
    for f in a.families:
        acc = [r["accuracy"] for r in rows if r["family"] == f]
        summary[f] = {"oa_mean": round(float(np.mean(acc)), 4), "oa_std": round(float(np.std(acc)), 4),
                      "aa_mean": round(float(np.mean([r["balanced_accuracy"] for r in rows if r["family"] == f])), 4),
                      "params": next(r["params"] for r in rows if r["family"] == f)}  # fmt: skip
    with open(a.out, "w") as fh:
        json.dump({"args": vars(a), "rows": rows, "summary": summary}, fh, indent=1)
    for f, s in sorted(summary.items(), key=lambda kv: -kv[1]["oa_mean"]):
        print(
            f"{f:<12} OA {s['oa_mean']:.4f} ± {s['oa_std']:.4f}  AA {s['aa_mean']:.4f}  params {s['params']:,}"
        )
