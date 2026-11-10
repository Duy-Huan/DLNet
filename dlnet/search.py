"""Search and rank: successive halving, then multi-seed evaluation of the survivors."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from .core import Trainer


def _new_trainer(cfg, data, seed, build_fn):
    torch.manual_seed(seed)  # the seed must control the weight init too, not only the batch order
    m = build_fn(cfg, data["in_dim"], data["out_dim"])
    return Trainer(m, data, lr=cfg["lr"], wd=cfg["wd"], batch=cfg["batch"], seed=seed)


def eval_runs(cfg, folds, build_fn, epochs, n_runs, deadline, first=None):
    """Train `cfg` n_runs times: run i uses seed i on split folds[i % len(folds)].

    With one fold this is plain multi-seed evaluation; with several folds (repeated splits) the
    spread also contains the split noise, which seed std alone misses (agent.md sec. 9 item 7).
    `first` is an already trained seed-0 trainer on folds[0], reused as run 0.
    Returns (val scores, test metric dicts, seconds, number of distinct splits used).
    """
    vals, tests, secs = [], [], 0.0
    for i in range(n_runs):
        if vals and time.time() > deadline:
            break
        t = first if i == 0 and first is not None else None
        if t is None:
            t = _new_trainer(cfg, folds[i % len(folds)], i, build_fn)
            t.fit(epochs)
        vals.append(t.evaluate("va")["score"])
        tests.append(t.evaluate("te"))
        secs += t.seconds
    return vals, tests, secs, min(len(vals), len(folds))


def summarize(vals, tests) -> dict:
    return {
        "n_seeds": len(vals),
        "val_mean": float(np.mean(vals)),
        "val_std": float(np.std(vals)),
        "test": {k: float(np.mean([t[k] for t in tests])) for k in tests[0]},
        "test_std": float(np.std([t["score"] for t in tests])),
    }


def run_search(
    data, cands, deadline, topk, build_fn, rungs=(4, 12, 30), eta=3, seeds=3, log=print, folds=None
):
    """`folds`: optional list of data dicts (repeated splits, folds[0] is `data`) for stage 2."""
    folds = folds or [data]
    # --- stage 1: successive halving (resumable trainers, seed 0)
    if not cands:
        raise ValueError("no candidates to search")
    alive = [{"cfg": c, "tr": _new_trainer(c, data, 0, build_fn)} for c in cands]
    log(f"[search] {len(alive)} candidates, rungs={rungs}")
    for r, ep in enumerate(rungs):
        finished = []
        for i, a in enumerate(alive):
            # The very first candidate always trains, so even a spent budget yields a result.
            if time.time() > deadline and not (r == 0 and i == 0):
                break
            a["tr"].fit(ep - a["tr"].epochs_done)
            a["val0"] = a["tr"].evaluate("va")["score"]
            finished.append(a)
        if not finished:  # budget spent before this rung started: rank what the last rung left
            log("[search] time budget reached; ranking with what is done")
            break
        # Only candidates that completed THIS rung compete: never rank different epoch counts.
        alive = sorted(finished, key=lambda a: -a["val0"])
        keep = max(math.ceil(len(alive) / eta), min(len(alive), topk * 2))
        if r < len(rungs) - 1:
            alive = alive[:keep]
        log(f"[search] rung {r + 1} ({ep} ep): kept {len(alive)}; best val={alive[0]['val0']:.4f}")
        if time.time() > deadline:
            log("[search] time budget reached; ranking with what is done")
            break
    finalists = alive[: max(topk * 2, topk + 2)]

    # --- stage 2: finalists on several seeds (and splits); run 0 reuses the stage-1 trainer
    results = []
    n_runs = max(seeds, len(folds))
    for a in finalists:
        cfg, ep = a["cfg"], a["tr"].epochs_done
        vals, tests, secs, n_splits = eval_runs(
            cfg, folds, build_fn, ep, n_runs, deadline, first=a["tr"]
        )
        results.append(
            {
                "cfg": cfg,
                **summarize(vals, tests),
                "n_splits": n_splits,
                "train_seconds": round(secs, 1),
                "epochs": ep,
                "trainer": a["tr"],  # seed-0 trained model, used by hparams.estimate (not exported)
            }
        )
    return assign_groups(results)


def assign_groups(results):
    """Sort by validation and mark 'indistinguishable' groups: neighbours closer than the larger std
    (std over seeds, and over splits when repeated splits are used)."""
    results.sort(key=lambda r: -r["val_mean"])
    g = 0
    for i, r in enumerate(results):
        if i and results[i - 1]["val_mean"] - r["val_mean"] > max(
            results[i - 1]["val_std"], r["val_std"], 1e-9
        ):
            g += 1
        r["group"] = g
    return results


def select_topk(results, k):
    """Diversity-aware selection: best per family first, then fill by rank."""
    chosen, fams = [], set()
    for r in results:
        if r["cfg"]["family"] not in fams and len(chosen) < k:
            chosen.append(r)
            fams.add(r["cfg"]["family"])
    for r in results:
        if r not in chosen and len(chosen) < k:
            chosen.append(r)
    chosen.sort(key=lambda r: -r["val_mean"])
    top1 = chosen[0]
    roles = {id(top1): ["top1"]}
    rest = chosen[1:]
    if rest:
        roles.setdefault(id(min(rest, key=lambda r: r["cfg"]["params"])), []).append("best-small")
        roles.setdefault(id(min(chosen, key=lambda r: r["val_std"])), []).append("most-stable")
        diff = [r for r in rest if r["cfg"]["family"] != top1["cfg"]["family"]]
        if diff:
            roles.setdefault(id(diff[0]), []).append("most-different")
    for r in chosen:
        r["roles"] = roles.get(id(r), [])
    return chosen
