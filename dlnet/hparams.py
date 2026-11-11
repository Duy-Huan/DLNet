"""Hyperparameter estimation for the finalists (standard algorithms, see ref/optimazeDL/).

Implemented (each cites the paper it follows; "not verified" marks parts taken from memory):
  * LR range test            Smith, "Cyclical Learning Rates" (arXiv:1506.01186), sec. 3.3: run with the LR
                             increasing, take the largest LR that still converges; the optimum is usually
                             within a factor of two of it, and base_lr = max_lr / 3 or / 4.
                             Exponential sweep + EMA-smoothed loss + stop on divergence follow the common
                             fastai implementation (not verified against the paper).
  * Gradient noise scale     McCandlish et al., "An Empirical Model of Large-Batch Training"
                             (arXiv:1812.06162), eq. 2.9 and appendix A.1: B_simple = tr(Sigma)/|G|^2 from the
                             unbiased two-batch-size estimators of |G|^2 and S = tr(Sigma).
  * Normalized weight decay  Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
                             (arXiv:1711.05101), appendix B.1: lambda = lambda_norm * sqrt(b / (B * T)).
An estimate is only a proposal: `refine` re-trains each finalist with the proposal and adopts it
only when it is clearly better on validation than the searched setting.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from . import clock
from .core import Trainer

MIN_BATCH, MAX_BATCH = 16, 512


def _loss_fn(data):
    return nn.MSELoss() if data["task"] == "regression" else nn.CrossEntropyLoss()


def _loss(model, X, y, data, loss_fn):
    out = model(X)
    out = out.squeeze(-1) if data["task"] == "regression" else out
    return loss_fn(out, y)


# ----------------------------------------------------------------- gradient noise scale


def noise_scale_from_norms(g_small2: float, b_small: int, g_big2: float, b_big: int):
    """McCandlish et al. appendix A.1: unbiased |G|^2 and S = tr(Sigma), and B_simple = S/|G|^2.

    Inputs are E|G_B|^2 for the two batch sizes (E|G_B|^2 = |G|^2 + tr(Sigma)/B).
    Returns (G2, S, B_simple); B_simple is None when the estimated |G|^2 is not positive.
    """
    g2 = (b_big * g_big2 - b_small * g_small2) / (b_big - b_small)
    s = (g_small2 - g_big2) / (1.0 / b_small - 1.0 / b_big)
    return g2, s, (s / g2 if g2 > 0 and s > 0 else None)


def _grad_sq(model, X, y, data, loss_fn) -> float:
    model.zero_grad()
    _loss(model, X, y, data, loss_fn).backward()
    return float(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None))


def gradient_noise_scale(model, data, b_small=16, b_big=256, n_meas=16, seed=0) -> dict | None:
    """Estimate B_simple of `model` at its current weights (works on a copy; BN stats untouched)."""
    X, y = data["Xtr"], data["ytr"]
    b_big = min(b_big, len(X))
    if b_big < 2 * b_small:
        return None
    m = copy.deepcopy(model).train()
    loss_fn = _loss_fn(data)
    gen = torch.Generator().manual_seed(seed)
    small, big = [], []
    for _ in range(n_meas):
        idx = torch.randperm(len(X), generator=gen)[:b_big]
        big.append(_grad_sq(m, X[idx], y[idx], data, loss_fn))
        chunks = idx.split(b_small)
        small.append(
            float(np.mean([_grad_sq(m, X[c], y[c], data, loss_fn) for c in chunks if len(c) > 1]))
        )
    g2, s, bs = noise_scale_from_norms(float(np.mean(small)), b_small, float(np.mean(big)), b_big)
    return {"b_small": b_small, "b_big": b_big, "grad_sq": g2, "trace_sigma": s, "b_simple": bs}


def suggest_batch(b_simple: float | None, n_train: int, current: int) -> int:
    """Largest power of two not above B_simple (the compute-efficient side of B_crit), clipped."""
    if not b_simple or not math.isfinite(b_simple):
        return current
    cap = max(min(MAX_BATCH, n_train // 4), MIN_BATCH)
    b = 2 ** math.floor(math.log2(max(min(b_simple, cap), 1.0)))
    return int(max(b, MIN_BATCH))


def normalized_wd(wd: float, batch_old: int, batch_new: int) -> float:
    """Keep lambda_norm constant when the batch size changes: lambda ~ sqrt(b)."""
    return float(wd * math.sqrt(batch_new / batch_old))


# ------------------------------------------------------------------------ LR range test


def lr_range_test(
    build_fn, cfg, data, batch, seed=0, n_iter=80, lr_lo=1e-5, lr_hi=1.0, beta=0.9, diverge=4.0
) -> dict:
    """Smith's LR range test with an exponential sweep. Returns lr_max and the proposed lr.

    lr_max = LR at the minimum of the smoothed loss (the largest LR that still converges);
    proposal = lr_max / 2 ("within a factor of two of the largest one that converges"), and
    [lr_max / 4, lr_max] is the base/max range Smith suggests for cyclical schedules.
    """
    torch.manual_seed(seed)
    model = build_fn(cfg, data["in_dim"], data["out_dim"])
    tr = Trainer(model, data, lr=lr_lo, wd=cfg["wd"], batch=batch, seed=seed)
    X, y = data["Xtr"], data["ytr"]
    gamma = (lr_hi / lr_lo) ** (1.0 / max(n_iter - 1, 1))
    model.train()
    lrs, smooth, avg, best = [], [], 0.0, math.inf
    perm, pos = torch.randperm(len(X), generator=tr.gen), 0
    for it in range(n_iter):
        lr = lr_lo * gamma**it
        for g in tr.opt.param_groups:
            g["lr"] = lr
        if pos + 2 > len(perm):
            perm, pos = torch.randperm(len(X), generator=tr.gen), 0
        b = perm[pos : pos + batch]
        pos += batch
        if len(b) < 2:
            continue
        loss = _loss(model, X[b], y[b], data, tr.loss_fn)
        if not torch.isfinite(loss):
            break
        tr.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        tr.opt.step()
        avg = beta * avg + (1 - beta) * float(loss.detach())
        s = avg / (1 - beta ** (len(lrs) + 1))  # bias-corrected EMA
        lrs.append(lr)
        smooth.append(s)
        best = min(best, s)
        if s > diverge * best:
            break
    if not lrs:
        return {"lr_max": lr_lo, "lr": lr_lo, "diverged": True, "n_iter": 0}
    k = int(np.argmin(smooth))
    lr_max = lrs[k]
    return {
        "lr_max": float(lr_max),
        "lr": float(lr_max / 2),
        "lr_base": float(lr_max / 4),
        "diverged": bool(len(lrs) < n_iter and k < len(lrs) - 1),
        "n_iter": len(lrs),
        "batch": int(batch),
    }


# ------------------------------------------------------------------- estimate + refine


def estimate(r: dict, data: dict, build_fn, epochs_recipe: int = 100, seed: int = 0) -> dict:
    """Propose lr / batch / weight decay for one finalist `r` (needs r['trainer'], the trained seed-0)."""
    cfg = r["cfg"]
    n = len(data["ytr"])
    gns = gradient_noise_scale(r["trainer"].model, data, seed=seed)
    batch = suggest_batch(gns["b_simple"] if gns else None, n, cfg["batch"])
    lrt = lr_range_test(build_fn, cfg, data, batch, seed=seed)
    wd = normalized_wd(cfg["wd"], cfg["batch"], batch)
    return {
        "lr": lrt["lr"],
        "batch": batch,
        "wd": wd,
        "lr_range": {k: lrt[k] for k in ("lr_base", "lr_max") if k in lrt},
        "lr_range_diverged": lrt["diverged"],
        "gradient_noise_scale": gns,
        "wd_normalized": float(wd * math.sqrt(n * epochs_recipe / batch)),
    }


def _refine_cost(r, data) -> tuple[float, float]:
    """Budget-clock cost of refining finalist `r`: (estimate, re-training all its runs).

    The estimate is the gradient-noise scale (16 x (256 + 16 x 16) samples) plus the LR range test
    (80 batches), scaled by r's measured seconds per epoch; it trains nothing, so it is free in
    work mode, where the re-training costs its epochs."""
    rerun_epochs = r["epochs"] * r["n_seeds"]
    if clock.mode() == "work":
        return 0.0, float(rerun_epochs)
    tr = r.get("trainer")
    per_epoch = tr.seconds / tr.epochs_done if tr is not None and tr.epochs_done else 0.0
    est = per_epoch * (16 * 512 + 80 * r["cfg"]["batch"]) / max(len(data["ytr"]), 1)
    return est, per_epoch * rerun_epochs


def refine(results, data, build_fn, deadline, epochs_recipe=100, log=print, folds=None, cache=None):
    """Estimate hyperparameters per finalist; adopt them only when clearly better on validation.

    Same epochs, seeds and splits as the search, so the numbers are comparable. A proposal is adopted if
    its mean validation score beats the searched one by more than the larger seed std.
    Each result gets r['hparams'] with the searched, estimated and validated values.
    """
    from .search import eval_runs, summarize

    folds = folds or [data]
    for r in results:
        cfg = r["cfg"]
        searched = {"lr": cfg["lr"], "batch": cfg["batch"], "wd": cfg["wd"]}
        r["hparams"] = {
            "searched": searched,
            "estimated": None,
            "adopted": "searched",
            "val_searched": r["val_mean"],
            "val_estimated": None,
        }
        if not clock.fits(sum(_refine_cost(r, data)), deadline):  # would end after the budget
            r["hparams"]["skipped"] = "budget"
            continue
        try:
            est = estimate(r, data, build_fn, epochs_recipe)
        except (
            RuntimeError,
            ValueError,
        ) as e:  # never lose a finalist because the estimator failed
            log(f"[hparams] {cfg['family']}: estimate failed ({e}); keeping searched")
            continue
        r["hparams"]["estimated"] = est
        if (est["lr"], est["batch"]) == (cfg["lr"], cfg["batch"]) or clock.now() > deadline:
            continue
        cand = {**cfg, "lr": est["lr"], "batch": est["batch"], "wd": est["wd"]}
        vals, tests, _, _ = eval_runs(
            cand, folds, build_fn, r["epochs"], r["n_seeds"], deadline, cache=cache,
            run_cost=_refine_cost(r, data)[1] / max(r["n_seeds"], 1),
        )  # fmt: skip
        new = summarize(vals, tests)
        v, sd = new["val_mean"], new["val_std"]
        r["hparams"]["val_estimated"] = v
        if v - r["val_mean"] > max(sd, r["val_std"], 1e-9):
            r["cfg"] = cand
            r.update(new)
            r["hparams"]["adopted"] = "estimated"
        log(
            f"[hparams] {cfg['family']}: searched val={r['hparams']['val_searched']:.4f} "
            f"estimated val={v:.4f} -> {r['hparams']['adopted']}"
        )
    results.sort(key=lambda r: -r["val_mean"])
    return results
