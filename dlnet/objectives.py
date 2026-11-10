"""Optimisation objectives: what "best" means when DLNet ranks candidates (`--objective`).

  accuracy   validation score only (the default; DLNet <= 0.14 behaviour).
  params     the SMALLEST model whose validation score is within `tol` of the best one
             (epsilon-constraint method: optimise one objective, the other becomes a constraint;
             Haimes et al. 1971, from memory). `flops` is the same with FLOPs per sample.
  balanced   MnasNet reward (Tan et al., CVPR 2019, arXiv:1807.11626, eq. 2):
                 reward = ACC(m) * (COST(m) / T)^w,   w = alpha if COST <= T else beta,
             with the paper's soft-constraint setting alpha = beta = -0.07 (Sec. 3) and
             COST = parameters (or FLOPs with --cost flops). T = --target-cost, by default the
             median cost of the pool. `--cost-weight` changes w. For scores that are not in
             [0, 1] (regression: -RMSE, reconstruction: -MSE) ACC is replaced by the score
             min-max normalised over the ranked candidates (DLNet choice: the product needs a
             positive accuracy).
  pareto     Pareto ranking of (score up, cost down): non-dominated sorting, then crowding
             distance inside a front (NSGA-II, Deb et al., IEEE TEC 2002, from memory; LEMONADE,
             Elsken et al., arXiv:1804.09081, uses the same idea for NAS). The leaderboard keeps the
             first front.

The same `order` function ranks inside successive halving / Hyperband rungs, picks the
aging-evolution tournament winner and orders the finalists, so the whole search follows the
chosen objective.
"""

from __future__ import annotations

import math

import numpy as np

OBJECTIVES = ("accuracy", "params", "flops", "balanced", "pareto")


def cost_of(cfg: dict, cost: str = "params") -> float:
    v = cfg.get("flops" if cost == "flops" else "params")
    return float(v) if v else float("nan")


def _norm(vals):
    v = np.asarray(vals, float)
    if np.all((v >= 0) & (v <= 1)):
        return v
    lo, hi = np.nanmin(v), np.nanmax(v)
    return np.full_like(v, 1.0) if hi <= lo else 0.01 + 0.99 * (v - lo) / (hi - lo)


def mnas_reward(acc: float, cost: float, target: float, alpha=-0.07, beta=-0.07) -> float:
    """MnasNet eq. 2: ACC * (COST / T)^w with w = alpha if COST <= T else beta."""
    if not (cost > 0 and target > 0) or not math.isfinite(cost):
        return acc
    w = alpha if cost <= target else beta
    return acc * (cost / target) ** w


def dominates(a, b) -> bool:
    """a, b = (score, cost): a is at least as good in both and strictly better in one."""
    return a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1])


def nondominated_sort(points) -> list[list[int]]:
    """Fronts of indices (front 0 = Pareto front) for maximising score and minimising cost."""
    n = len(points)
    dom_by = [0] * n
    dominated = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and dominates(points[i], points[j]):
                dominated[i].append(j)
            elif i != j and dominates(points[j], points[i]):
                dom_by[i] += 1
    fronts, cur = [], [i for i in range(n) if dom_by[i] == 0]
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            for j in dominated[i]:
                dom_by[j] -= 1
                if dom_by[j] == 0:
                    nxt.append(j)
        cur = nxt
    return fronts


def crowding(points, front) -> dict[int, float]:
    """NSGA-II crowding distance of the members of one front (boundary points get infinity)."""
    d = dict.fromkeys(front, 0.0)
    if len(front) <= 2:
        return dict.fromkeys(front, math.inf)
    for k in (0, 1):
        f = sorted(front, key=lambda i: points[i][k])
        lo, hi = points[f[0]][k], points[f[-1]][k]
        d[f[0]] = d[f[-1]] = math.inf
        if hi > lo:
            for a in range(1, len(f) - 1):
                d[f[a]] += (points[f[a + 1]][k] - points[f[a - 1]][k]) / (hi - lo)
    return d


def order(entries, objective="accuracy", cost="params", tol=0.01, target=None, weight=-0.07):
    """Indices of `entries` = [(cfg, score), ...] from best to worst under `objective`.

    Candidates without a cost (not costed) are ranked by score after the costed ones when the
    objective needs a cost.
    """
    n = len(entries)
    scores = [float(v) for _, v in entries]
    by_score = sorted(range(n), key=lambda i: -scores[i])
    if objective == "accuracy" or n < 2:
        return by_score
    c = "flops" if objective == "flops" else cost
    costs = [cost_of(cfg, c) for cfg, _ in entries]
    known = [i for i in range(n) if math.isfinite(costs[i])]
    unknown = [i for i in by_score if i not in known]
    if not known:
        return by_score
    if objective in ("params", "flops"):
        best = max(scores[i] for i in known)
        ok = [i for i in known if scores[i] >= best - tol]
        rest = [i for i in known if i not in ok]
        return (
            sorted(ok, key=lambda i: (costs[i], -scores[i]))
            + sorted(rest, key=lambda i: -scores[i])
            + unknown
        )
    if objective == "balanced":
        acc = _norm([scores[i] for i in known])
        t = target or float(np.median([costs[i] for i in known]))
        r = {
            i: mnas_reward(a, costs[i], t, weight, weight) for i, a in zip(known, acc, strict=True)
        }
        return sorted(known, key=lambda i: -r[i]) + unknown
    if objective == "pareto":
        pts = [(scores[i], costs[i]) for i in known]
        out = []
        for front in nondominated_sort(pts):
            cd = crowding(pts, front)
            out += [known[j] for j in sorted(front, key=lambda j: (-cd[j], -pts[j][0]))]
        return out + unknown
    raise ValueError(f"unknown objective {objective!r}; use one of {OBJECTIVES}")


def pareto_front(entries, cost="params") -> list[int]:
    """Indices of the non-dominated (score, cost) entries."""
    known = [i for i, (cfg, _) in enumerate(entries) if math.isfinite(cost_of(cfg, cost))]
    pts = [(float(entries[i][1]), cost_of(entries[i][0], cost)) for i in known]
    fronts = nondominated_sort(pts)
    return [known[j] for j in fronts[0]] if fronts else []


def parse_target(s) -> float | None:
    """'200k', '1.5M', '3e6' -> number (None for empty)."""
    if not s:
        return None
    from .zoo import cost

    return cost.parse_size(s)
