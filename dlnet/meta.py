"""Meta-learning recommender: learn which model families win from a store of past runs.

Opt-in with `dlnet design --meta-store store.jsonl`. After a run, one record is appended:
the data profile (meta-features) and the best rung-1 validation score of every family. Before
the next run, the k nearest past datasets (same plugin) vote on the family order.

This is the classical "meta-features + k-nearest datasets" warm start (in the spirit of
Feurer et al., "Initializing Bayesian Hyperparameter Optimization via Meta-Learning", AAAI 2015;
written from memory, the paper is not in ref/ and nothing here was checked against it).
The recommendation only ORDERS the candidate families (which matters when the budget ends
inside rung 1); it never removes a family. Measure it with scripts/bench_meta.py first.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import jsonsafe

MIN_RECORDS = 3


def features(prof: dict) -> list[float]:
    """Meta-features of a dataset, from its DLNet profile."""
    d_int = prof.get("intrinsic_dim") or 0.0
    return [
        math.log10(max(prof.get("n_train") or 1, 1)),
        math.log10(max(prof.get("n_features") or 1, 1)),
        float(d_int) if math.isfinite(d_int) else 0.0,
        float(prof.get("n_classes") or 0),  # 0 = regression
    ]


def normalized_ranks(scores: dict[str, float]) -> dict[str, float]:
    """0 = best family of this dataset, 1 = worst (ties share the mean rank)."""
    fams = sorted(scores, key=lambda f: -scores[f])
    if len(fams) < 2:
        return dict.fromkeys(fams, 0.0)
    out = {}
    for f in fams:
        better = sum(scores[g] > scores[f] for g in fams)
        same = sum(scores[g] == scores[f] for g in fams) - 1
        out[f] = (better + same / 2) / (len(fams) - 1)
    return out


def make_record(
    plugin: str, prof: dict, rung1: list, name: str = "", card=None, epochs=None
) -> dict | None:
    """One store record from the rung-1 trace [(candidate cfg, validation score), ...].

    With `card` (data card) the record also keeps every candidate's features and score, the
    meta-dataset of the zero-shot surrogate (dlnet/surrogate.py)."""
    best: dict[str, float] = {}
    for cfg, val in rung1:
        if math.isfinite(val):
            best[cfg["family"]] = max(val, best.get(cfg["family"], -math.inf))
    if len(best) < 2:
        return None  # nothing to learn from a single family
    rec = {"plugin": plugin, "name": name, "features": features(prof), "family_scores": best}
    if card is not None:
        from .surrogate import candidate_record

        rec["card"] = card
        rec["epochs"] = epochs
        rec["candidates"] = [candidate_record(c, v) for c, v in rung1 if math.isfinite(v)]
    return rec


def load(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue  # a damaged line must not break a design run
        if isinstance(r, dict) and {"plugin", "features", "family_scores"} <= set(r):
            out.append(r)
    return out


def append(path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(jsonsafe.dumps(record) + "\n")


def recommend(records: list[dict], plugin: str, feats: list[float], k: int = 5) -> dict | None:
    """Family order voted by the k nearest past datasets of the same plugin.

    Distance: Euclidean on z-scored meta-features (statistics of the store). Vote: mean
    normalized rank of each family over the neighbours, weighted by 1 / (distance + 0.1).
    Returns None with fewer than MIN_RECORDS usable records.
    """
    recs = [r for r in records if r["plugin"] == plugin and len(r["features"]) == len(feats)]
    if len(recs) < MIN_RECORDS:
        return None
    n = len(feats)
    mu = [sum(r["features"][j] for r in recs) / len(recs) for j in range(n)]
    sd = [
        math.sqrt(sum((r["features"][j] - mu[j]) ** 2 for r in recs) / len(recs)) or 1.0
        for j in range(n)
    ]

    def dist(r):
        return math.sqrt(sum(((r["features"][j] - feats[j]) / sd[j]) ** 2 for j in range(n)))

    near = sorted(recs, key=dist)[:k]
    votes: dict[str, list[tuple[float, float]]] = {}
    for r in near:
        w = 1.0 / (dist(r) + 0.1)
        for fam, rank in normalized_ranks(r["family_scores"]).items():
            votes.setdefault(fam, []).append((w, rank))
    score = {f: sum(w * x for w, x in v) / sum(w for w, _ in v) for f, v in votes.items()}
    order = sorted(score, key=lambda f: (score[f], f))
    return {
        "family_order": order,
        "expected_rank": {f: round(score[f], 3) for f in order},
        "neighbours": [r.get("name", "") for r in near],
        "n_records": len(recs),
        "used_for_pruning": False,
    }


def reorder(cands: list[dict], family_order: list[str]) -> list[dict]:
    """Stable sort of the candidates by the recommended family order (unknown families last)."""
    pos = {f: i for i, f in enumerate(family_order)}
    return sorted(cands, key=lambda c: pos.get(c["family"], len(pos)))


def leave_one_out(records: list[dict], plugin: str, topk: int = 3, k: int = 5) -> dict:
    """How often the true best family of a held-out dataset is inside the recommended top-k.

    Compared with two baselines on the same held-out datasets: the global order (mean rank over
    all other datasets, no meta-features) and chance (topk / number of families).
    """
    recs = [r for r in records if r["plugin"] == plugin]
    hit = glob = chance = n = 0
    for i, r in enumerate(recs):
        rest = recs[:i] + recs[i + 1 :]
        rec = recommend(rest, plugin, r["features"], k=k)
        if rec is None:
            continue
        truth = max(r["family_scores"], key=r["family_scores"].get)
        g = recommend(rest, plugin, r["features"], k=len(rest))  # all datasets: nearly global
        n += 1
        hit += truth in rec["family_order"][:topk]
        glob += truth in g["family_order"][:topk]
        chance += min(topk / len(r["family_scores"]), 1.0)
    return {
        "datasets": n,
        "topk": topk,
        "hit_rate_knn": hit / n if n else None,
        "hit_rate_all_datasets": glob / n if n else None,
        "hit_rate_chance": chance / n if n else None,
    }


def portfolio(records: list[dict], plugin: str, feats=None, size: int = 4, k: int = 5) -> list[str]:
    """Greedy portfolio of families (Auto-sklearn 2.0, Feurer et al., JMLR 2022, arXiv:2007.04074,
    Algorithm 1): start empty and repeatedly add the family that most lowers the mean, over the
    meta-datasets, of the error of the best family already in the portfolio.

    Error of a family on a dataset = its normalized rank there (0 = best, 1 = worst; a family
    not run on that dataset counts as 1). With `feats`, the meta-datasets are the k nearest past
    datasets (z-scored meta-features, as `recommend`); otherwise every record of the plugin.
    Ties go to the alphabetically first family (the paper breaks ties by training order).
    """
    recs = [r for r in records if r["plugin"] == plugin]
    if feats is not None and len(recs) > k:
        n = len(feats)
        recs = [r for r in recs if len(r["features"]) == n]
        mu = [sum(r["features"][j] for r in recs) / len(recs) for j in range(n)]
        sd = [
            math.sqrt(sum((r["features"][j] - mu[j]) ** 2 for r in recs) / len(recs)) or 1.0
            for j in range(n)
        ]
        dist = lambda r: math.sqrt(
            sum(((r["features"][j] - feats[j]) / sd[j]) ** 2 for j in range(n))
        )
        recs = sorted(recs, key=dist)[:k]
    if not recs:
        return []
    ranks = [normalized_ranks(r["family_scores"]) for r in recs]
    cands = sorted({f for r in ranks for f in r})
    port: list[str] = []
    while cands and len(port) < size:

        def err(fams, ranks=ranks):
            return sum(min((r.get(f, 1.0) for f in fams), default=1.0) for r in ranks) / len(ranks)

        best = min(cands, key=lambda f: (err([*port, f]), f))
        port.append(best)
        cands.remove(best)
    return port
