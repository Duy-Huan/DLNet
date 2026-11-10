"""Array datasets: images and time series stored as .npz with X and y (numpy + torch).

Copied verbatim into every exported image / timeseries package so that train.py can rebuild the
exact same data.

    image       X (N, H, W) or (N, H, W, C)   -> tensors (N, C, H, W)
    timeseries  X (N, T)    or (N, T, C)      -> tensors (N, C, T)
y has one label (classification) or one number (regression) per sample.
    segmentation  X as for image, y (N, H, W) integer class of every pixel
                  (pixels with a negative label are ignored in the loss and in the metrics)
"""

from __future__ import annotations

import numpy as np
import torch

try:  # inside the dlnet package
    from . import core
except ImportError:  # inside an exported package (core.py sits next to this file)
    import core

KINDS = ("image", "timeseries", "segmentation", "graph", "ids", "reconstruction")
IGNORE = -100  # label of pixels that do not count (torch CrossEntropyLoss default)


def load(path: str, x_key: str | None = None, y_key: str | None = None):
    z = np.load(path)
    if not hasattr(z, "files"):
        raise SystemExit(f"{path}: need a .npz file with the arrays X and y")
    xk, yk = x_key or "X", y_key or "y"
    missing = [k for k in (xk, yk) if k not in z.files]
    if missing:
        raise SystemExit(f"{path}: arrays {missing} not found (has {z.files}); use --x-key/--y-key")
    X, y = np.asarray(z[xk]), np.asarray(z[yk])
    if "A" in z.files and X.ndim == 3 and z["A"].shape == (X.shape[0], X.shape[1], X.shape[1]):
        X = np.concatenate(
            [X, np.asarray(z["A"])], -1
        )  # graphs: features then adjacency, (G, N, F + N)
    if len(X) != len(y) or y.ndim not in (1, 3):
        raise SystemExit(f"{path}: X {X.shape} and y {y.shape} must have one row per sample")
    return X, y


def kind_of(X: np.ndarray, y: np.ndarray | None = None) -> str | None:
    """Guess the kind from the shapes. (N, S, S) is read as square grey images."""
    if y is not None and y.ndim == 3:
        return "segmentation" if X.ndim in (3, 4) and y.shape == X.shape[:3] else None
    if X.ndim == 4:
        return "image"
    if X.ndim == 3:
        return "image" if X.shape[1] == X.shape[2] else "timeseries"
    return None


def to_channels_first(X: np.ndarray, kind: str) -> np.ndarray:
    """(N, H, W[, C]) -> (N, C, H, W) or (N, T[, C]) -> (N, C, T), C-CONTIGUOUS float32.

    Contiguity matters: a transposed view kept channels-last strides through fancy indexing, and
    torch 2.15 then crashed (double free / segfault) in multi-threaded convolution backward on
    3-channel images. See tests/test_regression.py.
    """
    X = X.astype(np.float32)
    if kind == "image":
        if X.ndim == 3:
            X = X[..., None]
        if X.ndim != 4:
            raise SystemExit(f"image data must be (N, H, W) or (N, H, W, C), got {X.shape}")
        return np.ascontiguousarray(X.transpose(0, 3, 1, 2))
    if X.ndim == 2:
        X = X[..., None]
    if X.ndim != 3:
        raise SystemExit(f"time series must be (N, T) or (N, T, C), got {X.shape}")
    return np.ascontiguousarray(X.transpose(0, 2, 1))


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


def build_segmentation(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Images + per-pixel masks. Whole IMAGES are split 60/20/20, so no pixel of a validation or
    test image is seen in training. Standardised per channel with TRAIN statistics."""
    X = to_channels_first(X, "image")
    if y.shape != (X.shape[0], *X.shape[2:]):
        raise SystemExit(f"masks {y.shape} must be (N, H, W) matching the images {X.shape}")
    if not np.isfinite(X).all():
        raise SystemExit("X contains NaN or infinite values; clean the data first")
    y = y.astype(np.int64)
    classes = np.unique(y[y >= 0])
    lut = np.full(int(classes.max()) + 1, IGNORE, dtype=np.int64)
    lut[classes] = np.arange(len(classes))
    yi = np.where(y >= 0, lut[np.clip(y, 0, None)], IGNORE)
    perm = np.random.default_rng(seed).permutation(len(X))
    a, b = int(0.6 * len(X)), int(0.8 * len(X))
    tr, va, te = perm[:a], perm[a:b], perm[b:]
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"too few images ({len(X)}) for a 60/20/20 split")
    mu = X[tr].mean(axis=(0, 2, 3), keepdims=True)
    sd = X[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    d = {
        "task": "classification", "classes": classes.tolist(), "in_dim": int(X.shape[1]),
        "out_dim": len(classes), "seed": seed, "kind": "segmentation", "primary": "miou",
        "shape": [int(s) for s in X.shape[1:]], "y_raw": {}, "Xnp": {},
    }  # fmt: skip
    for k, i in zip(("tr", "va", "te"), (tr, va, te), strict=True):
        d["X" + k] = torch.from_numpy(Xs[i])
        d["y" + k] = torch.from_numpy(yi[i])
        d["y_raw"][k] = yi[i]
        d["Xnp"][k] = Xs[i].reshape(len(i), -1)
    return d


def build_any(X, y, kind: str, task: str, seed: int = 0) -> dict:
    """Dispatch on the kind stored in config.yaml (used by the exported train.py)."""
    if kind == "segmentation":
        return build_segmentation(X, y, seed)
    if kind == "graph":
        return build_graph(X, y, task, seed)
    if kind == "ids":
        return build_ids(X, y, task, seed)
    if kind == "reconstruction":
        return build_reconstruction(X, seed)
    return build(X, y, kind, task, seed)


# ---------------------------------------------------------------- graphs and categorical ids


def graph_stats(Xp: np.ndarray) -> np.ndarray:
    """Hand-made features of packed graphs (G, N, F + N) for the classical baselines (no learning).

    Per graph: density, degree mean / std / max, triangles per node, global clustering, the five
    largest eigenvalues of the normalised Laplacian, and per-feature mean and std over nodes.
    """
    g, n, w = Xp.shape
    f = w - n
    out = []
    for gi in range(g):
        a, x = Xp[gi, :, f:], Xp[gi, :, :f]
        deg = a.sum(1)
        tri = float(np.trace(np.linalg.matrix_power(a, 3))) / 6.0
        triples = float((deg * (deg - 1)).sum()) / 2.0
        d = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
        lap = np.eye(n) - d[:, None] * a * d[None, :]
        ev = np.sort(np.linalg.eigvalsh(lap))[::-1][:5]
        ev = np.pad(ev, (0, max(0, 5 - len(ev))))
        out.append(
            [
                a.sum() / max(n * (n - 1), 1),
                deg.mean(),
                deg.std(),
                deg.max(),
                tri / n,
                3 * tri / max(triples, 1e-9),
                *ev,
                *x.mean(0),
                *x.std(0),
            ]
        )
    return np.asarray(out, dtype=np.float32)


def build_graph(Xp: np.ndarray, y: np.ndarray, task: str, seed: int = 0) -> dict:
    """Graph-level prediction. Xp (G, N, F + N): features then adjacency. Whole graphs are split 60/20/20."""
    if Xp.ndim != 3 or Xp.shape[2] <= Xp.shape[1]:
        raise SystemExit(f"packed graphs must be (G, N, F + N) with F >= 1, got {Xp.shape}")
    n = Xp.shape[1]
    f = Xp.shape[2] - n
    Xp = Xp.astype(np.float32)
    if not np.isfinite(Xp).all():
        raise SystemExit("graphs contain NaN or infinite values")
    yt, classes = core.encode_target(y, task)
    tr, va, te = core.stratified_split(yt, task, seed)
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"too few graphs ({len(y)}) for a 60/20/20 split")
    mu, sd = (
        Xp[tr][..., :f].mean((0, 1)),
        Xp[tr][..., :f].std((0, 1)) + 1e-6,
    )  # node features only; adjacency untouched
    Xs = Xp.copy()
    Xs[..., :f] = (Xs[..., :f] - mu) / sd
    stats = graph_stats(Xs)
    sm, ss = stats[tr].mean(0), stats[tr].std(0) + 1e-6
    return _finish(
        {"kind": "graph", "shape": [n, f + n], "in_dim": f},
        Xs,
        (stats - sm) / ss,
        yt,
        classes,
        y,
        task,
        (tr, va, te),
    )


def build_ids(ids: np.ndarray, y: np.ndarray, task: str, seed: int = 0) -> dict:
    """Interactions of categorical fields (user, item, ...) -> a label or a number. ids (N, F) integers."""
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.dtype.kind not in "iu" or ids.min() < 0:
        raise SystemExit("ids must be a (N, F) array of non-negative integers")
    cards = [
        int(ids[:, j].max()) + 1 for j in range(ids.shape[1])
    ]  # all rows: unseen-in-train ids keep their initial embedding
    yt, classes = core.encode_target(y, task)
    tr, va, te = core.stratified_split(yt, task, seed)
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"too few interactions ({len(y)}) for a 60/20/20 split")
    if sum(cards) > 5000:
        raise SystemExit(f"{sum(cards)} ids in total: the one-hot reference baselines need <= 5000")
    onehot = np.zeros((len(ids), sum(cards)), dtype=np.float32)
    off = np.cumsum([0, *cards[:-1]])
    for j in range(ids.shape[1]):
        onehot[np.arange(len(ids)), off[j] + ids[:, j]] = 1.0
    return _finish(
        {"kind": "ids", "shape": cards, "in_dim": len(cards), "cards": cards},
        ids.astype(np.int64),
        onehot,
        yt,
        classes,
        y,
        task,
        (tr, va, te),
    )


def _finish(d, X, flat, yt, classes, y, task, splits):
    """Common tail: tensors per split, classical-baseline view `Xnp`, target scaling, primary metric."""
    tr, va, te = splits
    d.update(task=task, classes=classes, y_raw={}, Xnp={}, keys={})
    if task == "regression":
        d["y_mean"], d["y_std"] = float(yt[tr].mean()), float(yt[tr].std() + 1e-6)
        tt = ((yt - d["y_mean"]) / d["y_std"]).astype(np.float32)
        d["out_dim"], d["primary"] = 1, "neg_rmse"
    else:
        tt = yt
        d["out_dim"] = len(classes)
        counts = np.bincount(yt[tr])
        d["primary"] = "accuracy" if counts.min() / counts.max() > 0.3 else "balanced_accuracy"
    for k, i in zip(("tr", "va", "te"), (tr, va, te), strict=True):
        d["X" + k], d["y" + k] = torch.from_numpy(X[i]), torch.from_numpy(tt[i])
        d["y_raw"][k], d["Xnp"][k] = yt[i], flat[i]
    return d


# ------------------------------------------------------------------------- reconstruction


def load_x(path: str, x_key: str | None = None):
    """Images only (no labels): .npz with X (N, H, W) or (N, H, W, C)."""
    z = np.load(path)
    xk = x_key or "X"
    if not hasattr(z, "files") or xk not in z.files:
        raise SystemExit(f"{path}: need a .npz file with the array {xk!r}")
    return np.asarray(z[xk])


def build_reconstruction(X: np.ndarray, seed: int = 0) -> dict:
    """Unsupervised: the target of every image is the image itself. Split 60/20/20, standardise per channel."""
    X = to_channels_first(X, "image")
    if not np.isfinite(X).all():
        raise SystemExit("X contains NaN or infinite values")
    perm = np.random.default_rng(seed).permutation(len(X))
    a, b = int(0.6 * len(X)), int(0.8 * len(X))
    tr, va, te = perm[:a], perm[a:b], perm[b:]
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"too few images ({len(X)}) for a 60/20/20 split")
    mu = X[tr].mean(axis=(0, 2, 3), keepdims=True)
    sd = X[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    d = {
        "task": "reconstruction", "classes": None, "in_dim": int(X.shape[1]), "out_dim": int(X.shape[1]),
        "kind": "reconstruction", "primary": "neg_mse", "shape": [int(s) for s in X.shape[1:]], "keys": {},
        "y_raw": {}, "Xnp": {}, "seed": seed,
    }  # fmt: skip
    for k, i in zip(("tr", "va", "te"), (tr, va, te), strict=True):
        d["X" + k] = d["y" + k] = torch.from_numpy(
            Xs[i]
        )  # the same tensor: the target is the input
        d["y_raw"][k] = Xs[i]
        d["Xnp"][k] = Xs[i].reshape(len(i), -1)
    return d
