"""Zero-cost proxies: scores of an UNTRAINED network (standard algorithms, see ref/optimazeDL/).

Rule (agent.md sec. 16.1 item 4): a proxy may prune candidates only after its rank correlation
with real training results was measured on DLNet data and compared with the "parameter count"
baseline. So this module only SCORES and MEASURES (`correlation_report`); nothing is pruned.

Implemented (each cites its source; "not verified" marks what was not checked against it):
  * naswot     Mellor et al., "Neural Architecture Search without Training" (arXiv:2006.04647).
               Follows the authors' score_networks.py (ref/optimazeDL/official_code/
               nas-without-training): for every ReLU, binary code x = (input > 0) per sample,
               K += x x^T + (1 - x)(1 - x)^T, score = log|det K|.
  * synflow    Tanaka et al., "Pruning neural networks without any data by iteratively conserving
               synaptic flow" (arXiv:2006.05467): R = 1^T (prod |theta|) 1, i.e. all parameters
               replaced by their absolute value, an all-ones input, R = sum of outputs;
               score = sum |theta * dR/dtheta|. Using it as a whole-network score and running it
               in eval mode (BatchNorm with running statistics) follows Abdelfattah et al.
               (arXiv:2101.08134) from memory: not verified against their code.
  * grad_norm  Abdelfattah et al. (arXiv:2101.08134): Euclidean norm of the loss gradient on one
               minibatch (their code may sum per-layer norms instead: not verified).
  * params     the baseline every proxy has to beat: the parameter count.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

PROXIES = ("naswot", "synflow", "grad_norm", "params")


def naswot(model: nn.Module, X: torch.Tensor) -> float | None:
    """log|det K| of the ReLU activation-code kernel on the batch X. None if the net has no ReLU."""
    m = copy.deepcopy(model).train()
    n = len(X)
    K = np.zeros((n, n))
    hits = 0

    def hook(_mod, inp, _out):
        nonlocal K, hits
        x = (inp[0].reshape(n, -1) > 0).float()
        K = K + (x @ x.t()).numpy() + ((1.0 - x) @ (1.0 - x.t())).numpy()
        hits += 1

    handles = [mod.register_forward_hook(hook) for mod in m.modules() if isinstance(mod, nn.ReLU)]
    with torch.no_grad():
        m(X)
    for h in handles:
        h.remove()
    if not hits:
        return None
    return float(np.linalg.slogdet(K)[1])


def synflow(model: nn.Module, x_shape: tuple[int, ...]) -> float:
    """log of sum |theta * dR/dtheta| with |theta| weights and an all-ones input (data-free)."""
    m = copy.deepcopy(model).double().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.abs_()
    m.zero_grad()
    m(torch.ones((1, *x_shape), dtype=torch.float64)).sum().backward()
    s = sum(float((p.detach() * p.grad).abs().sum()) for p in m.parameters() if p.grad is not None)
    return math.log(s) if s > 0 and math.isfinite(s) else -math.inf


def grad_norm(model: nn.Module, X, y, task: str) -> float:
    m = copy.deepcopy(model).train()
    out = m(X)
    if task == "regression":
        loss = nn.functional.mse_loss(out.squeeze(-1), y)
    else:
        loss = nn.functional.cross_entropy(out, y)
    m.zero_grad()
    loss.backward()
    return float(
        math.sqrt(sum(float((p.grad**2).sum()) for p in m.parameters() if p.grad is not None))
    )


def score(model: nn.Module, data: dict, n: int = 64, seed: int = 0) -> dict:
    """All proxies for one untrained model on the first n (seeded) training samples."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(data["Xtr"]), generator=gen)[:n]
    X, y = data["Xtr"][idx], data["ytr"][idx]
    out = {"params": float(sum(p.numel() for p in model.parameters()))}
    for name, fn in (
        ("naswot", lambda: naswot(model, X)),
        ("synflow", lambda: synflow(model, tuple(X.shape[1:]))),
        ("grad_norm", lambda: grad_norm(model, X, y, data["task"])),
    ):
        try:
            out[name] = fn()
        except (RuntimeError, ValueError):  # a proxy that cannot handle a block is just missing
            out[name] = None
    return out


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=float)
    for v in np.unique(a):
        tie = a == v
        r[tie] = r[tie].mean()
    return r


def spearman(a, b) -> float | None:
    """Spearman rank correlation = Pearson correlation of the average ranks. None if undefined."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def correlation_report(scores: list[dict], vals: list[float], epochs: int) -> dict:
    """Spearman of every proxy with the real validation score after `epochs` of training.

    `scores[i]` and `vals[i]` belong to the same candidate. Candidates where a proxy is missing
    or not finite are left out of that proxy's correlation (n says how many were used).
    """
    rep = {}
    for name in PROXIES:
        pairs = [
            (s[name], v)
            for s, v in zip(scores, vals, strict=True)
            if s.get(name) is not None and math.isfinite(s[name])
        ]
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
        rep[name] = {"spearman": None if rho is None else round(rho, 3), "n": len(pairs)}
    base = abs(rep["params"]["spearman"] or 0.0)
    better = [k for k, v in rep.items() if k != "params" and abs(v["spearman"] or 0.0) > base + 0.1]
    return {
        "target": f"validation score after {epochs} epochs (rung 1)",
        "proxies": rep,
        "better_than_param_count": better,
        "used_for_pruning": False,
        "note": "information only: one dataset and short training are not enough evidence to prune",
    }
