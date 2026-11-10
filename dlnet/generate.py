"""Candidate generation: sample the library, drop candidates that violate the cost budget."""

from __future__ import annotations

import numpy as np

from .library import FAMILIES, build_model, count_params

SPACE = {
    "mlp": {"width": [64, 128, 256], "depth": [2, 3, 4], "dropout": [0.0, 0.1, 0.2]},
    "resmlp": {"width": [64, 128, 256], "depth": [1, 2, 3], "dropout": [0.0, 0.1, 0.2]},
    "tt_resmlp": {
        "width": [64, 128, 256],
        "depth": [1, 2, 3],
        "dropout": [0.0, 0.1],
        "rank": [4, 8, 16],
    },
    "tr_resmlp": {
        "width": [64, 128, 256],
        "depth": [1, 2, 3],
        "dropout": [0.0, 0.1],
        "rank": [2, 4, 8],
    },
    "lowrank_mlp": {
        "width": [128, 256],
        "depth": [2, 3, 4],
        "dropout": [0.0, 0.1],
        "rank": [8, 16, 32],
    },
    "ft_transformer": {"width": [32, 64], "depth": [1, 2], "dropout": [0.0, 0.1]},
    "moe_mlp": {
        "width": [64, 128, 256],
        "depth": [1, 2],
        "dropout": [0.0, 0.1],
        "experts": [2, 4, 8],
        "routing": ["soft", "topk"],
    },
    "fm_mlp": {"width": [64, 128], "depth": [2, 3], "dropout": [0.0, 0.1], "rank": [4, 8, 16]},
}
TRAIN = {"lr": [1e-3, 3e-3], "wd": [1e-4, 1e-2], "batch": [64, 128]}


def generate(profile: dict, n_per_family: int = 8, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    in_dim = profile["n_features"]
    out_dim = profile["n_classes"] or 1
    cands, seen = [], set()
    for fam in FAMILIES:
        for _ in range(n_per_family * 4):
            if sum(c["family"] == fam for c in cands) >= n_per_family:
                break
            cfg = {"family": fam}
            for k, v in {**SPACE[fam], **TRAIN}.items():
                cfg[k] = v[int(rng.integers(len(v)))]
            key = tuple(sorted(cfg.items()))
            if key in seen:
                continue
            seen.add(key)
            # Cost model (v0): parameter count from the instantiated model.
            if fam in ("ft_transformer", "fm_mlp") and in_dim > 128:
                continue  # quadratic in #features
            cfg["params"] = count_params(build_model(cfg, in_dim, out_dim))
            if cfg["params"] <= profile["max_params"]:
                cands.append(cfg)
    return cands
