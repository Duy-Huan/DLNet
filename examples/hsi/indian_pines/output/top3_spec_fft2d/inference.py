"""Use a trained baseline package: load it, turn a new data file into model input, predict.

This file is copied into every exported package (next to model.py, core.py, train.py) and needs
only torch and numpy (pandas for tables, scipy for .mat cubes). `python train.py` writes model.pt
and preprocess.json; `python predict.py --data new_file` uses them through the functions below.

    import inference
    model, meta, cfg = inference.load(".")
    labels = inference.predict_file(model, meta, cfg, "new.npz")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load(pkg_dir=".", device="cpu"):
    """(model in eval mode, preprocess.json, config.yaml) of a package trained with train.py."""
    pkg = Path(pkg_dir)
    if not (pkg / "model.pt").exists():
        raise SystemExit(f"{pkg / 'model.pt'} not found: run `python train.py --data ...` first")
    from model import build_model  # the package's own model.py

    cfg = json.loads((pkg / "config.yaml").read_text())
    meta = json.loads((pkg / "preprocess.json").read_text())
    model = build_model(cfg["arch"], cfg["in_dim"], cfg["out_dim"])
    model.load_state_dict(torch.load(pkg / "model.pt", map_location=device, weights_only=True))
    return model.to(device).eval(), meta, cfg


# ----------------------------------------------------------------------------- input


def _npz(path, key):
    z = np.load(path, allow_pickle=False)
    if key not in z.files:
        raise SystemExit(f"{path}: array {key!r} not found (has {z.files})")
    return np.asarray(z[key])


def prepare_input(meta, cfg, path):
    """New data file -> model input, with the TRAINING statistics of preprocess.json."""
    pre, keys = meta["preprocess"], cfg.get("data", {})
    t = pre["type"]
    if t == "table":
        import core

        X = core.load_features(path, pre["columns"], pre.get("target"))
        med, mu, sd = (np.asarray(pre[k], dtype=np.float32) for k in ("median", "mean", "std"))
        bad = np.isnan(X)
        X[bad] = np.take(med, np.nonzero(bad)[1])
        return torch.from_numpy(((X - mu) / sd).astype(np.float32))
    if t == "channels":
        X = _npz(path, keys.get("x_key") or "X").astype(np.float32)
        if pre["layout"] == "image":
            X = X[..., None] if X.ndim == 3 else X
            X = X.transpose(0, 3, 1, 2)
        else:  # series (N, T[, C]) -> (N, C, T)
            X = X[..., None] if X.ndim == 2 else X
            X = X.transpose(0, 2, 1)
        shape = [1, -1] + [1] * (X.ndim - 2)
        mu, sd = (np.asarray(pre[k], dtype=np.float32).reshape(shape) for k in ("mean", "std"))
        return torch.from_numpy(np.ascontiguousarray((X - mu) / sd, dtype=np.float32))
    if t == "graph":
        Xn, A = (
            _npz(path, keys.get("x_key") or "X").astype(np.float32),
            _npz(path, "A").astype(np.float32),
        )
        f = pre["features"]
        Xn = (Xn - np.asarray(pre["mean"], dtype=np.float32)) / np.asarray(
            pre["std"], dtype=np.float32
        )
        if Xn.shape[2] != f:
            raise SystemExit(f"node features have {Xn.shape[2]} columns, the model expects {f}")
        return torch.from_numpy(np.concatenate([Xn, A], axis=2).astype(np.float32))
    if t == "ids":
        ids = _npz(path, keys.get("x_key") or "ids").astype(np.int64)
        cards = pre["cards"]
        if ids.ndim != 2 or ids.shape[1] != len(cards) or (ids < 0).any():
            raise SystemExit(f"ids must be (N, {len(cards)}) non-negative integers")
        unseen = ids >= np.asarray(cards)
        if unseen.any():
            raise SystemExit(f"{int(unseen.sum())} id(s) above the training range {cards}: the model has no "
                             "embedding for them")  # fmt: skip
        return torch.from_numpy(ids)
    if t == "hsi":
        return _hsi_patches(meta, path, keys)
    raise SystemExit(f"unknown preprocessing type {t!r}")


def _hsi_patches(meta, path, keys):
    """Every pixel of a new cube as a patch (the model labels the centre pixel); returns a store."""
    import hsi_data

    pre = meta["preprocess"]
    cube = hsi_data.load_cube(path, keys.get("cube_key"))
    mean, std, V, scale = (
        np.asarray(pre[k], dtype=np.float64) for k in ("mean", "std", "components", "scale")
    )
    if cube.shape[2] != len(mean):
        raise SystemExit(f"cube has {cube.shape[2]} bands, the model was trained on {len(mean)}")
    t = (((cube - mean) / std) @ V / scale).astype(np.float32)
    p = pre["patch"] // 2
    pad = torch.from_numpy(np.pad(t, ((p, p), (p, p), (0, 0)), mode="reflect"))
    h, w = cube.shape[:2]
    rows, cols = np.divmod(np.arange(h * w), w)
    store = hsi_data.PatchStore(pad, rows, cols, pre["patch"])
    store.map_shape = (h, w)
    return store


# ---------------------------------------------------------------------------- output


@torch.no_grad()
def predict(model, meta, X, batch=1024, device="cpu"):
    """Model input -> predictions in the units of the training labels.

    classification: original class labels; regression: original target units; segmentation: a mask
    of original class values; reconstruction: images in the original intensity scale (N, C, H, W);
    hsi: a class map (H, W)."""
    outs = []
    for i in range(0, len(X), batch):
        idx = torch.arange(i, min(i + batch, len(X)))
        xb = X[idx] if hasattr(X, "map_shape") else X[i : i + batch]
        outs.append(model(xb.to(device)).float().cpu())
    out = torch.cat(outs)
    task = meta["task"]
    if task == "regression":
        return out.squeeze(-1).numpy() * meta["y_std"] + meta["y_mean"]
    if task == "reconstruction":
        pre = meta["preprocess"]
        shape = [1, -1] + [1] * (out.dim() - 2)
        mu, sd = (torch.tensor(pre[k]).reshape(shape) for k in ("mean", "std"))
        return (out * sd + mu).numpy()
    classes = np.asarray(meta["classes"])
    pred = classes[out.argmax(1).numpy()]
    return pred.reshape(X.map_shape) if hasattr(X, "map_shape") else pred


def predict_file(model, meta, cfg, path, device="cpu"):
    return predict(model, meta, prepare_input(meta, cfg, path), device=device)


# ---------------------------------------------------------------------------- export


def export(model, sample, path):
    """Save a deployable graph of the model: .pt2 (torch.export, no extra package) or .onnx
    (needs `pip install onnx onnxscript`). `sample` = a model input batch (e.g. 2 rows)."""
    path = Path(path)
    if path.suffix == ".onnx":
        try:
            torch.onnx.export(model, (sample,), str(path), dynamo=True)
        except ImportError as e:
            raise SystemExit(f"ONNX export needs `pip install onnx onnxscript` ({e})") from None
        return path
    prog = torch.export.export(model, (sample,))
    torch.export.save(prog, str(path))
    return path
