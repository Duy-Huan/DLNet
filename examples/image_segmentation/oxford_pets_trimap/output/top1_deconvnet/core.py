"""Shared data and training utilities (numpy + torch + pandas only).

This file is copied verbatim into every exported baseline package.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time

import numpy as np
import torch
from torch import nn

try:  # inside the dlnet package: epochs count against a work budget (dlnet/clock.py)
    from .clock import spend as _spend
except ImportError:  # inside an exported package there is no budget

    def _spend(_epochs):
        return None


# --------------------------------------------------------------------------- data


def load_csv(path: str, target: str, warnings: list | None = None):
    """Features (float32, NaN kept for TRAIN-median imputation), labels, column names.

    A text column whose values are almost all numbers (>= 90 %) is read as numbers, the rest
    becoming missing: one-hot encoding it would turn every distinct number into a column (a
    50-row file gave 133 columns and test accuracy 0.54 instead of 0.95, audit S6). Warnings are
    printed and appended to `warnings`."""
    import pandas as pd

    def warn(msg):
        print(f"[load] {msg}")
        if warnings is not None:
            warnings.append(msg)

    df = pd.read_csv(path)
    if target not in df.columns:
        import difflib

        cols = [str(c) for c in df.columns]
        close = [c for c in cols if c.lower() == str(target).lower()] or difflib.get_close_matches(
            str(target), cols, n=3
        )
        hint = f"; did you mean {close}?" if close else ""
        raise SystemExit(
            f"--target {target!r} is not a column of {path}{hint} Columns: {list(df.columns)}"
        )
    df = df.dropna(subset=[target])  # a row without a label cannot be used
    y = df[target].to_numpy()
    feats = df.drop(columns=[target])
    text_cols = [c for c in feats.columns if not pd.api.types.is_numeric_dtype(feats[c])
                 and not pd.api.types.is_bool_dtype(feats[c])]  # fmt: skip
    for c in text_cols:
        num = pd.to_numeric(feats[c], errors="coerce")
        n, ok = int(feats[c].notna().sum()), int(num.notna().sum())
        if n and ok / n >= 0.9:
            warn(
                f"column {c!r} is numeric with {n - ok} non-numeric value(s); those are treated as missing"
            )
            feats[c] = num
        elif feats[c].nunique() > 50:
            warn(f"column {c!r} has {feats[c].nunique()} distinct text values: one-hot encoding adds as many "
                 "columns; drop or encode it yourself if it is an identifier or free text")  # fmt: skip
    X = pd.get_dummies(feats)
    X = X.apply(pd.to_numeric, errors="coerce").astype("float32")
    # Missing features stay NaN here; prepare() imputes with TRAIN medians (no test leakage).
    return X.to_numpy(dtype=np.float32), y, list(X.columns)


def load_features(path: str, columns: list, target: str | None = None) -> np.ndarray:
    """Features of a NEW csv in the column layout of training (predict.py): the same numeric
    reading as load_csv, one-hot columns unseen in training dropped, missing ones set to 0, and
    numeric columns absent from the file left missing (imputed with the TRAIN median)."""
    import pandas as pd

    df = pd.read_csv(path)
    feats = df.drop(columns=[target]) if target and target in df.columns else df
    for c in feats.columns:
        if not pd.api.types.is_numeric_dtype(feats[c]) and not pd.api.types.is_bool_dtype(feats[c]):
            num = pd.to_numeric(feats[c], errors="coerce")
            n = int(feats[c].notna().sum())
            if n and int(num.notna().sum()) / n >= 0.9:
                feats[c] = num
    X = pd.get_dummies(feats)
    X = X.apply(pd.to_numeric, errors="coerce").astype("float32")
    missing = [c for c in columns if c not in X.columns]
    one_hot = {c for c in missing if any(c.startswith(f"{b}_") for b in feats.columns)}
    X = X.reindex(columns=columns)
    X[sorted(one_hot)] = X[sorted(one_hot)].fillna(0.0)  # a category absent from this file
    return np.array(X.to_numpy(dtype=np.float32), copy=True)  # writable (pandas may share memory)


def detect_task(y: np.ndarray, labels: bool = False) -> str:
    """classification | regression, guessed from the targets.

    Strings and booleans are classes; whole numbers with at most 20 distinct values are classes.
    `labels=True` (image, series, graph, ids files, whose `y` is normally a label array): an
    INTEGER-typed y holding the consecutive codes 0..K-1 or 1..K, every one at least twice, is a
    label set of any size. Without it the 37 breeds of Oxford-IIIT Pet were silently fitted as a
    regression (agent.md 9.10). Still ambiguous cases get a warning from `task_note`."""
    if y.dtype.kind in "OSUb":
        return "classification"
    integer_like = y.dtype.kind in "iu" or np.allclose(y, np.round(y))
    values, counts = np.unique(y, return_counts=True)
    if integer_like and len(values) <= 20:
        return "classification"
    consecutive = values[0] in (0, 1) and values[-1] - values[0] + 1 == len(values)
    if labels and y.dtype.kind in "iu" and counts.min() >= 2 and consecutive:
        return "classification"
    return "regression"


def task_note(y: np.ndarray, task: str) -> str | None:
    """Warning for a guessed task that could as well be the other one (whole numbers, many values)."""
    if y.dtype.kind in "OSUb" or not (y.dtype.kind in "iu" or np.allclose(y, np.round(y))):
        return None
    k = len(np.unique(y))
    if k <= 20 or k > len(y) // 2:
        return None
    other = "regression" if task == "classification" else "classification"
    return (f"the target holds {k} distinct whole numbers and was read as {task}; "
            f"if it is a {other} target, run again with --task {other}")  # fmt: skip


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
    d = {"task": task, "classes": classes, "in_dim": X.shape[1], "seed": seed,
         "preprocess": {"type": "table", "median": med, "mean": mu, "std": sd}}  # fmt: skip
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
    if task == "reconstruction":  # y_true is the input itself; lower MSE is better
        mse = float(np.mean((y_true - y_pred) ** 2))
        rng = float(y_true.max() - y_true.min()) or 1.0
        return {"mse": mse, "psnr": float(10 * np.log10(rng**2 / max(mse, 1e-12))), "score": -mse}
    if y_true.ndim > 1:  # dense prediction: one label per pixel, negative labels are ignored
        keep = y_true >= 0
        y_true, y_pred = y_true[keep], y_pred[keep]
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
    if primary == "miou":  # mean over the classes present in y_true of |A and B| / |A or B|
        m["miou"] = float(
            np.mean(
                [
                    np.sum((y_true == c) & (y_pred == c)) / np.sum((y_true == c) | (y_pred == c))
                    for c in np.unique(y_true)
                ]
            )
        )
    m["score"] = m[primary]
    return m


# ------------------------------------------------------------------------ device

DEVICE, AMP = "auto", "auto"  # defaults of every Trainer; `dlnet design --device/--amp` sets them
# Training recipe (`dlnet design --recipe`): "modern" = augmentation for images (random shifts of up
# to 1/8 of the side; horizontal flips for colour images and for segmentation, applied to image and
# mask together) and label smoothing 0.1 for classification; "plain" = neither (DLNet <= 0.21).
# Measured by the audit on a standard ResNet: removing both cost 2.3 test points on Fashion-MNIST
# and 17.3 on CIFAR-10 (qa/reports/v0.19.0). train_full adds a linear warm-up to its cosine.
RECIPE = "modern"
RECIPES = {
    "modern": {"augment": True, "label_smoothing": 0.1},
    "plain": {"augment": False, "label_smoothing": 0.0},
}


def pick_device(name="auto") -> torch.device:
    """auto = the first CUDA GPU, else Apple MPS, else the CPU. An explicit device must exist."""
    mps = getattr(torch.backends, "mps", None)
    has_mps = mps is not None and mps.is_available()
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"device {name!r}: torch sees no CUDA GPU (torch.cuda.is_available() is False)"
        )
    if dev.type == "mps" and not has_mps:
        raise ValueError(f"device {name!r}: Apple MPS is not available")
    return dev


def amp_dtype(device: torch.device, amp="auto"):
    """Autocast dtype, or None for full precision.

    auto = bfloat16 on a CUDA GPU that supports it, else float16 (with loss scaling); full
    precision on the CPU and on MPS. bf16 / fp16 force a dtype (also on the CPU, for testing).
    """
    if amp in (None, "off"):
        return None
    if amp == "auto":
        if device.type != "cuda":
            return None
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]


# ------------------------------------------------------------------------ training


def members(model, out, y):
    """(output, target) for the loss. An ensemble model (TabM) returns (B, k, out) in training:
    every member gets its own loss, averaged (Gorishniy et al. 2025, sec. 3), not the loss of the
    averaged prediction."""
    k = getattr(model, "ensemble_k", 0)
    if k and out.dim() == y.dim() + 2:
        return out.flatten(0, 1), y.repeat_interleave(k, dim=0)
    return out, y


class Trainer:
    """Resumable trainer (needed for successive halving)."""

    def __init__(
        self,
        model: nn.Module,
        data: dict,
        lr=1e-3,
        wd=1e-4,
        batch=128,
        seed=0,
        device=None,
        amp=None,
        recipe=None,
    ):
        torch.manual_seed(seed)
        self.recipe = RECIPES[recipe or RECIPE]
        self.device = pick_device(DEVICE if device is None else device)
        self.amp = amp_dtype(self.device, AMP if amp is None else amp)
        # NHWC is faster for convolutions on GPUs (tensor cores); on the CPU it is left as is
        self.channels_last = self.device.type == "cuda" and data["Xtr"][[0]].dim() == 4
        model.to(self.device)
        if self.channels_last:
            model.to(memory_format=torch.channels_last)
        self.model, self.data, self.batch = model, data, batch
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        # float16 needs loss scaling against gradient underflow; bfloat16 and float32 do not
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp is torch.float16)
        self.gen = torch.Generator().manual_seed(seed)
        self.epochs_done = 0
        self.seconds = 0.0
        self.loss_fn = (
            nn.MSELoss()
            if data["task"] in ("regression", "reconstruction")
            else nn.CrossEntropyLoss(label_smoothing=self.recipe["label_smoothing"])
        )
        kind = data.get("kind")
        self.augment = self.recipe["augment"] and kind in ("image", "segmentation")
        if self.augment:
            c, h, w = data["shape"][:3]
            self.shift = 0 if kind == "segmentation" else max(1, min(h, w) // 8)
            self.flip = kind == "segmentation" or c == 3  # digits and letters must not be mirrored

    def _augment(self, xb, yb):
        """Random shift (zero padding) and horizontal flip per sample, on the training device.

        One affine resampling at integer pixel offsets: exact pixel values, no host round trip."""
        n = xb.shape[0]
        g = self.gen
        sign = torch.ones(n)
        if self.flip:
            sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
        theta = torch.zeros(n, 2, 3)
        theta[:, 0, 0], theta[:, 1, 1] = sign, 1.0
        if self.shift:
            h, w = xb.shape[-2:]
            d = torch.randint(-self.shift, self.shift + 1, (n, 2), generator=g).float()
            theta[:, 0, 2], theta[:, 1, 2] = 2 * d[:, 0] / w, 2 * d[:, 1] / h
        theta = theta.to(xb.device, xb.dtype)
        grid = nn.functional.affine_grid(theta, list(xb.shape), align_corners=False)
        xb = nn.functional.grid_sample(
            xb, grid, mode="nearest", padding_mode="zeros", align_corners=False
        )
        if yb.dim() == 3:  # segmentation mask: the same flip (no shift for masks)
            yb = torch.where((sign < 0).to(yb.device)[:, None, None], yb.flip(-1), yb)
        return xb.contiguous(memory_format=torch.channels_last) if self.channels_last else xb, yb

    def split(self, key: str):
        """data[key] on the training device: copied once per data dict and device, unless it
        would take more than half of the free GPU memory (then batches are copied one by one)."""
        t = self.data[key]
        if self.device.type == "cpu" or t.device == self.device:
            return t
        cache = self.data.setdefault("_on_device", {})
        if (key, str(self.device)) not in cache:
            free = (
                torch.cuda.mem_get_info(self.device)[0] if self.device.type == "cuda" else math.inf
            )
            cache[key, str(self.device)] = t.to(self.device) if t.nbytes < free / 2 else t
        return cache[key, str(self.device)]

    def batch_of(self, t, idx):
        """Rows `idx` of `t` (a tensor or a lazy store) on the training device."""
        out = t[idx.to(t.device)].to(self.device, non_blocking=True)
        if self.channels_last and out.dim() == 4:
            out = out.contiguous(memory_format=torch.channels_last)
        return out

    def _autocast(self):
        if self.amp is None:  # full precision: no autocast context (some device types have none)
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.amp)

    def probe_epoch_seconds(self) -> float:
        """Seconds one training epoch should take, from one timed forward + backward of a batch.

        Prediction for an untrained candidate on the time budget: FLOPs miss what Python loops
        cost (MambaVision, KAT, xLSTM ran 10-100x slower than their FLOPs said and overran a 90 s
        budget by 10+ minutes; agent.md 9.10). Weights, buffers, gradients and the RNG are left
        as they were, so training afterwards is unchanged."""
        X, y = self.split("Xtr"), self.split("ytr")
        n = min(self.batch, len(X))
        if n < 2:
            return 0.0
        idx = torch.arange(n)
        rng = torch.get_rng_state()
        buffers = {
            k: v.clone()
            for k, v in self.model.state_dict().items()
            if k not in dict(self.model.named_parameters())
        }
        self.model.train()
        t0 = time.time()
        with self._autocast():
            out, yl = members(self.model, self.model(self.batch_of(X, idx)), self.batch_of(y, idx))
            out = out.squeeze(-1) if self.data["task"] == "regression" else out
            loss = self.loss_fn(out.float(), yl)
        loss.backward()
        sec = (time.time() - t0) * math.ceil(len(X) / self.batch)
        self.opt.zero_grad(set_to_none=True)
        self.model.load_state_dict(buffers, strict=False)
        torch.set_rng_state(rng)
        return sec

    def fit(self, epochs: int, scheduler=None):
        t0 = time.time()
        X, y = self.split("Xtr"), self.split("ytr")
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X), generator=self.gen)
            steps, bad = 0, torch.zeros((), device=self.device)
            total = torch.zeros((), device=self.device)
            for i in range(0, len(X), self.batch):
                b = perm[i : i + self.batch]
                if len(b) < 2:
                    continue
                xb, yb = self.batch_of(X, b), self.batch_of(y, b)
                if self.augment:
                    xb, yb = self._augment(xb, yb)
                with self._autocast():
                    out, yl = members(self.model, self.model(xb), yb)
                    out = out.squeeze(-1) if self.data["task"] == "regression" else out
                    loss = self.loss_fn(out.float(), yl)
                    kl = getattr(self.model, "kl_loss", None)  # variational bottleneck (VAE)
                    if kl is not None and (k := kl()) is not None:
                        loss = loss + self.model.kl_weight * k
                steps += 1
                bad += ~torch.isfinite(loss.detach())  # counted on the device: no sync per batch
                total += loss.detach().float()
                self.opt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            # one host sync per epoch; an epoch without one finite loss means the model diverged
            if steps and self.device.type != "meta" and int(bad) == steps:
                raise FloatingPointError(
                    f"training diverged: every loss of epoch {self.epochs_done + 1} is NaN/inf"
                )
            meta = self.device.type == "meta"  # stands in for a GPU in tests: no values
            self.last_train_loss = float(total / steps) if steps and not meta else math.nan
            if scheduler is not None:
                scheduler.step()
            self.epochs_done += 1
            _spend(1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)  # kernels run asynchronously: time what ran
        self.seconds += time.time() - t0

    @torch.no_grad()
    def evaluate(self, split: str) -> dict:
        self.model.eval()
        X = self.split("X" + split)
        outs = []
        for i in range(0, len(X), 1024):
            with self._autocast():
                outs.append(self.model(self.batch_of(X, torch.arange(i, min(i + 1024, len(X))))))
        out = torch.cat([o.float().cpu() for o in outs])
        d = self.data
        if d["task"] == "reconstruction":
            pred = out.numpy()
        elif d["task"] == "regression":
            pred = out.squeeze(-1).numpy() * d["y_std"] + d["y_mean"]
        else:
            pred = out.argmax(1).numpy()
        return compute_metrics(d["y_raw"][split], pred, d["task"], d["primary"])


def train_full(model, data, lr, wd, batch, epochs, seed=0, device=None, amp=None, recipe=None):
    """Full recipe: linear warm-up (5 % of the epochs, at least 1) then cosine decay, keep the
    best-validation checkpoint."""
    tr = Trainer(
        model, data, lr=lr, wd=wd, batch=batch, seed=seed, device=device, amp=amp, recipe=recipe
    )
    warm = max(1, epochs // 20)

    def factor(e):  # e = epochs finished
        if e < warm:
            return (e + 1) / (warm + 1)
        return 0.5 * (1 + math.cos(math.pi * (e - warm) / max(epochs - warm, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(tr.opt, factor)
    best, best_state = -math.inf, None
    tr.history = []  # one row per epoch: training loss, validation score, learning rate
    for e in range(epochs):
        lr = tr.opt.param_groups[0]["lr"]
        tr.fit(1, scheduler=sched)
        v = tr.evaluate("va")["score"]
        tr.history.append({"epoch": e + 1, "train_loss": tr.last_train_loss, "val": v, "lr": lr})
        if v > best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return tr, best


# ------------------------------------------------------------------------ artifacts


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def save_artifacts(model, data, out_dir=".") -> None:
    """Write model.pt (weights) and preprocess.json (everything predict.py needs to turn a new
    file into model input and the output back into labels: TRAIN statistics, columns, classes)."""
    import json
    from pathlib import Path

    out = Path(out_dir)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model.pt")
    meta = {"task": data["task"], "kind": data.get("kind"), "classes": data.get("classes"),
            "y_mean": data.get("y_mean"), "y_std": data.get("y_std"),
            "shape": data.get("shape"), "preprocess": data.get("preprocess")}  # fmt: skip
    (out / "preprocess.json").write_text(json.dumps(_plain(meta), indent=1))
