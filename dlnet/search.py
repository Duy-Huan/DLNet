"""Search and rank: successive halving, then multi-seed evaluation of the survivors."""

from __future__ import annotations

import math
import time

import numpy as np

from .core import Trainer
from .library import build_model


def _new_trainer(cfg, data, seed):
    m = build_model(cfg, data["in_dim"], data["out_dim"])
    return Trainer(m, data, lr=cfg["lr"], wd=cfg["wd"], batch=cfg["batch"], seed=seed)


def run_search(data, cands, deadline, topk, rungs=(4, 12, 30), eta=3, seeds=3, log=print):
    # --- stage 1: successive halving (resumable trainers, seed 0)
    alive = [{"cfg": c, "tr": _new_trainer(c, data, 0)} for c in cands]
    log(f"[search] {len(alive)} candidates, rungs={rungs}")
    for r, ep in enumerate(rungs):
        for a in alive:
            if time.time() > deadline:
                break
            a["tr"].fit(ep - a["tr"].epochs_done)
            a["val0"] = a["tr"].evaluate("va")["score"]
        alive = [a for a in alive if "val0" in a]
        alive.sort(key=lambda a: -a["val0"])
        keep = max(math.ceil(len(alive) / eta), min(len(alive), topk * 2))
        if r < len(rungs) - 1:
            alive = alive[:keep]
        log(f"[search] rung {r + 1} ({ep} ep): kept {len(alive)}; best val={alive[0]['val0']:.4f}")
        if time.time() > deadline:
            log("[search] time budget reached; ranking with what is done")
            break
    finalists = alive[: max(topk * 2, topk + 2)]

    # --- stage 2: multi-seed for finalists (seed 0 reused, extra seeds trained from scratch)
    results = []
    for a in finalists:
        cfg, ep = a["cfg"], a["tr"].epochs_done
        vals = [a["tr"].evaluate("va")["score"]]
        tests = [a["tr"].evaluate("te")]
        secs = a["tr"].seconds
        for s in range(1, seeds):
            if time.time() > deadline:
                break
            t = _new_trainer(cfg, data, s)
            t.fit(ep)
            vals.append(t.evaluate("va")["score"])
            tests.append(t.evaluate("te"))
            secs += t.seconds
        results.append(
            {
                "cfg": cfg,
                "n_seeds": len(vals),
                "val_mean": float(np.mean(vals)),
                "val_std": float(np.std(vals)),
                "test": {k: float(np.mean([t[k] for t in tests])) for k in tests[0]},
                "train_seconds": round(secs, 1),
                "epochs": ep,
            }
        )
    results.sort(key=lambda r: -r["val_mean"])
    # "indistinguishable" groups: neighbours closer than the larger std share a group
    g = 0
    for i, r in enumerate(results):
        if i and results[i - 1]["val_mean"] - r["val_mean"] > max(results[i - 1]["val_std"], r["val_std"], 1e-9):
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
