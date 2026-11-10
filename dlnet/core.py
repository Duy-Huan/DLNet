"""Shared data and training utilities (numpy + torch + pandas only).

This file is copied verbatim into every exported baseline package.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- data


def load_csv(path: str, target: str):
    import pandas as pd

    df = pd.read_csv(path)
    df = df.dropna(subset=[target])  # a row without a label cannot be used
    y = df[target].to_numpy()
    X = pd.get_dummies(df.drop(columns=[target]))
    X = X.apply(pd.to_numeric, errors="coerce").astype("float32")
    # Missing features stay NaN here; prepare() imputes with TRAIN medians (no test leakage).
    return X.to_numpy(dtype=np.float32), y, list(X.columns)


def detect_task(y: np.ndarray) -> str:
    if y.dtype.kind in "OSUb":
        return "classification"
    integer_like = y.dtype.kind in "iu" or np.allclose(y, np.round(y))
    return "classification" if integer_like and len(np.unique(y)) <= 20 else "regression"


def encode_target(y: np.ndarray, task: str):
    if task == "classification":
        classes, yi = np.unique(y, return_inverse=True)
        return yi.astype(np.int64), classes.tolist()
    return y.astype(np.float32), None


def stratified_split(y: np.ndarray, task: str, seed: int, frac=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    groups = [idx[y == c] for c in np.unique(y)] if task == "classification" else [idx]
    tr, va, te = [], [], []
    for g in groups:
        g = rng.permutation(g)
        a = int(len(g) * frac[0])
        b = a + int(len(g) * frac[1])
        if len(g) >= 3:  # every group with >= 3 members is present in every split
            a = max(a, 1)
            b = min(max(b, a + 1), len(g) - 1)
        tr += list(g[:a])
        va += list(g[a:b])
        te += list(g[b:])
    return np.array(tr), np.array(va), np.array(te)


def prepare(X, y, task: str, seed: int = 0) -> dict:
    """Split 60/20/20, standardise with train statistics, return tensors."""
    y, classes = encode_target(y, task)
    tr, va, te = stratified_split(y, task, seed)
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(
            f"too few samples ({len(y)}) for a 60/20/20 split: "
            f"train={len(tr)} val={len(va)} test={len(te)}"
        )
    X = X.astype(np.float32, copy=True)
    med = np.nanmedian(X[tr], axis=0)  # impute with TRAIN medians only
    med = np.where(np.isnan(med), 0.0, med)
    bad = np.isnan(X)
    X[bad] = np.take(med, np.nonzero(bad)[1])
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    d = {"task": task, "classes": classes, "in_dim": X.shape[1], "seed": seed}
    if task == "regression":
        d["y_mean"], d["y_std"] = float(y[tr].mean()), float(y[tr].std() + 1e-6)
        d["y_raw"] = {k: y[i] for k, i in zip(("tr", "va", "te"), (tr, va, te))}
        yt = ((y - d["y_mean"]) / d["y_std"]).astype(np.float32)
        d["out_dim"] = 1
    else:
        yt = y
        d["out_dim"] = len(classes)
        d["y_raw"] = {k: y[i] for k, i in zip(("tr", "va", "te"), (tr, va, te))}
    for k, i in zip(("tr", "va", "te"), (tr, va, te)):
        d["X" + k] = torch.from_numpy(Xs[i])
        d["y" + k] = torch.from_numpy(yt[i])
    d["Xnp"] = {k: Xs[i] for k, i in zip(("tr", "va", "te"), (tr, va, te))}
    balance = (
        np.bincount(y[tr]).min() / np.bincount(y[tr]).max() if task == "classification" else 1.0
    )
    d["primary"] = (
        "neg_rmse"
        if task == "regression"
        else ("accuracy" if balance > 0.3 else "balanced_accuracy")
    )
    return d


def compute_metrics(y_true, y_pred, task: str, primary: str) -> dict:
    if task == "regression":
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
        m = {"rmse": rmse, "r2": r2}
        m["score"] = -rmse
        return m
    acc = float(np.mean(y_true == y_pred))
    recalls = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    pe = sum(
        np.mean(y_true == c) * np.mean(y_pred == c)
        for c in np.unique(np.concatenate([y_true, y_pred]))
    )
    m = {
        "accuracy": acc,
        "balanced_accuracy": float(np.mean(recalls)),
        "kappa": float((acc - pe) / (1 - pe + 1e-12)),
    }
    m["score"] = m[primary]
    return m


# ------------------------------------------------------------------------ training


class Trainer:
    """Resumable trainer (needed for successive halving)."""

    def __init__(self, model: nn.Module, data: dict, lr=1e-3, wd=1e-4, batch=128, seed=0):
        torch.manual_seed(seed)
        self.model, self.data, self.batch = model, data, batch
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.gen = torch.Generator().manual_seed(seed)
        self.epochs_done = 0
        self.seconds = 0.0
        self.loss_fn = nn.MSELoss() if data["task"] == "regression" else nn.CrossEntropyLoss()

    def fit(self, epochs: int, scheduler=None):
        t0 = time.time()
        X, y = self.data["Xtr"], self.data["ytr"]
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X), generator=self.gen)
            for i in range(0, len(X), self.batch):
                b = perm[i : i + self.batch]
                if len(b) < 2:
                    continue
                out = self.model(X[b])
                out = out.squeeze(-1) if self.data["task"] == "regression" else out
                loss = self.loss_fn(out, y[b])
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()
            if scheduler is not None:
                scheduler.step()
            self.epochs_done += 1
        self.seconds += time.time() - t0

    @torch.no_grad()
    def evaluate(self, split: str) -> dict:
        self.model.eval()
        X = self.data["X" + split]
        out = torch.cat(
            [self.model(X[torch.arange(i, min(i + 1024, len(X)))]) for i in range(0, len(X), 1024)]
        )
        d = self.data
        if d["task"] == "regression":
            pred = out.squeeze(-1).numpy() * d["y_std"] + d["y_mean"]
        else:
            pred = out.argmax(1).numpy()
        return compute_metrics(d["y_raw"][split], pred, d["task"], d["primary"])


def train_full(model, data, lr, wd, batch, epochs, seed=0):
    """Full recipe: cosine LR, keep the best-validation checkpoint."""
    tr = Trainer(model, data, lr=lr, wd=wd, batch=batch, seed=seed)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(tr.opt, T_max=epochs)
    best, best_state = -math.inf, None
    for _ in range(epochs):
        tr.fit(1, scheduler=sched)
        v = tr.evaluate("va")["score"]
        if v > best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return tr, best
