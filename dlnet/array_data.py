"""Array datasets: images and time series stored as .npz with X and y (numpy + torch).

Copied verbatim into every exported image / timeseries package so that train.py can rebuild the
exact same data.

    image       X (N, H, W) or (N, H, W, C)   -> tensors (N, C, H, W)
    timeseries  X (N, T)    or (N, T, C)      -> tensors (N, C, T)
y has one label (classification) or one number (regression) per sample.
"""

from __future__ import annotations

import numpy as np
import torch

try:  # inside the dlnet package
    from . import core
except ImportError:  # inside an exported package (core.py sits next to this file)
    import core

KINDS = ("image", "timeseries")


def load(path: str, x_key: str | None = None, y_key: str | None = None):
    z = np.load(path)
    if not hasattr(z, "files"):
        raise SystemExit(f"{path}: need a .npz file with the arrays X and y")
    xk, yk = x_key or "X", y_key or "y"
    missing = [k for k in (xk, yk) if k not in z.files]
    if missing:
        raise SystemExit(f"{path}: arrays {missing} not found (has {z.files}); use --x-key/--y-key")
    X, y = np.asarray(z[xk]), np.asarray(z[yk])
    if len(X) != len(y) or y.ndim != 1:
        raise SystemExit(f"{path}: X {X.shape} and y {y.shape} must have one row per sample")
    return X, y


def kind_of(X: np.ndarray) -> str | None:
    """Guess the kind from the shape. (N, S, S) is read as square grey images."""
    if X.ndim == 4:
        return "image"
    if X.ndim == 3:
        return "image" if X.shape[1] == X.shape[2] else "timeseries"
    return None


def to_channels_first(X: np.ndarray, kind: str) -> np.ndarray:
    X = X.astype(np.float32)
    if kind == "image":
        if X.ndim == 3:
            X = X[..., None]
        if X.ndim != 4:
            raise SystemExit(f"image data must be (N, H, W) or (N, H, W, C), got {X.shape}")
        return X.transpose(0, 3, 1, 2)
    if X.ndim == 2:
        X = X[..., None]
    if X.ndim != 3:
        raise SystemExit(f"time series must be (N, T) or (N, T, C), got {X.shape}")
    return X.transpose(0, 2, 1)


def build(X: np.ndarray, y: np.ndarray, kind: str, task: str, seed: int = 0) -> dict:
    """Split 60/20/20 (stratified), standardise per channel with TRAIN statistics, return tensors."""
    X = to_channels_first(X, kind)
    if not np.isfinite(X).all():
        raise SystemExit("X contains NaN or infinite values; clean the data first")
    y, classes = core.encode_target(y, task)
    tr, va, te = core.stratified_split(y, task, seed)
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"too few samples ({len(y)}) for a 60/20/20 split")
    axes = (0, *range(2, X.ndim))
    mu = X[tr].mean(axis=axes, keepdims=True)
    sd = X[tr].std(axis=axes, keepdims=True) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    d = {
        "task": task, "classes": classes, "in_dim": int(X.shape[1]), "seed": seed, "kind": kind,
        "shape": [int(s) for s in X.shape[1:]], "y_raw": {}, "Xnp": {},
    }  # fmt: skip
    if task == "regression":
        d["y_mean"], d["y_std"] = float(y[tr].mean()), float(y[tr].std() + 1e-6)
        yt = ((y - d["y_mean"]) / d["y_std"]).astype(np.float32)
        d["out_dim"], d["primary"] = 1, "neg_rmse"
    else:
        yt = y
        d["out_dim"] = len(classes)
        counts = np.bincount(y[tr])
        d["primary"] = "accuracy" if counts.min() / counts.max() > 0.3 else "balanced_accuracy"
    for k, i in zip(("tr", "va", "te"), (tr, va, te), strict=True):
        d["X" + k] = torch.from_numpy(Xs[i])
        d["y" + k] = torch.from_numpy(yt[i])
        d["y_raw"][k] = y[i]
        d["Xnp"][k] = Xs[i].reshape(len(i), -1)  # flattened view for the classical baselines
    return d
