"""Search and rank: successive halving, then multi-seed evaluation of the survivors."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .core import Trainer


def _new_trainer(cfg, data, seed, build_fn):
    torch.manual_seed(seed)  # the seed must control the weight init too, not only the batch order
    m = build_fn(cfg, data["in_dim"], data["out_dim"])
    return Trainer(m, data, lr=cfg["lr"], wd=cfg["wd"], batch=cfg["batch"], seed=seed)


def interleave(cands):
    """Round-robin by family, so a budget that ends inside rung 1 does not favour the first families."""
    by: dict[str, list] = {}
    for c in cands:
        by.setdefault(c["family"], []).append(c)
    out = []
    for i in range(max(map(len, by.values()))):
        out += [v[i] for v in by.values() if i < len(v)]
    return out


class RunCache:
    """On-disk cache of from-scratch runs, so a repeated or interrupted command resumes.

    Key = DLNet version + config + fingerprint of the split (train/val arrays) + seed + epochs.
    Only stage-2 runs and the hyperparameter check use it; successive halving keeps its trainers
    in memory and is repeated on a re-run.
    """

    def __init__(self, path):
        self.path, self.hits = Path(path), 0
        try:
            self.d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.d = {}

    @staticmethod
    def key(cfg, data, seed, epochs) -> str:
        from . import __version__

        if "_fingerprint" not in data:
            h = hashlib.sha1()
            for a in (data["Xnp"]["tr"], data["y_raw"]["tr"], data["y_raw"]["va"]):
                h.update(np.ascontiguousarray(a).tobytes())
            data["_fingerprint"] = h.hexdigest()
        blob = json.dumps([__version__, cfg, data["_fingerprint"], seed, epochs], sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()

    def get(self, key):
        if key in self.d:
            self.hits += 1
        return self.d.get(key)

    def put(self, key, value):
        self.d[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d))
        tmp.replace(self.path)


def eval_runs(cfg, folds, build_fn, epochs, n_runs, deadline, first=None, cache=None):
    """Train `cfg` n_runs times: run i uses seed i on split folds[i % len(folds)].

    With one fold this is plain multi-seed evaluation; with several folds (repeated splits) the
    spread also contains the split noise, which seed std alone misses (agent.md sec. 9 item 7).
    `first` is an already trained seed-0 trainer on folds[0], reused as run 0.
    `cache` (RunCache) returns stored results of identical runs instead of training again.
    Returns (val scores, test metric dicts, seconds, number of distinct splits used).
    """
    vals, tests, secs = [], [], 0.0
    for i in range(n_runs):
        if vals and time.time() > deadline:
            break
        t = first if i == 0 and first is not None else None
        fold = folds[i % len(folds)]
        key = cache and RunCache.key(cfg, fold, i, epochs)
        hit = cache.get(key) if cache and t is None else None
        if hit is None:
            if t is None:
                t = _new_trainer(cfg, fold, i, build_fn)
                t.fit(epochs)
            hit = {"val": t.evaluate("va")["score"], "test": t.evaluate("te"), "secs": t.seconds}
            if cache:
                cache.put(key, hit)
        vals.append(hit["val"])
        tests.append(hit["test"])
        secs += hit["secs"]
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
    data, cands, deadline, topk, build_fn, rungs=(4, 12, 30), eta=3, seeds=3, log=print, folds=None,
    cache=None, trace=None,
):  # fmt: skip
    """`folds`: optional list of data dicts (repeated splits, folds[0] is `data`) for stage 2.

    `trace`: optional dict; receives "rung1" = [(cfg, val score)] of every candidate that finished
    the first rung (the real training signal zero-cost proxies are compared with).
    """
    folds = folds or [data]
    # --- stage 1: successive halving (resumable trainers, seed 0)
    if not cands:
        raise ValueError("no candidates to search")
    cands = interleave(cands)
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
        if r == 0 and trace is not None:
            trace["rung1"], trace["rung1_epochs"] = [(a["cfg"], a["val0"]) for a in finished], ep
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
            cfg, folds, build_fn, ep, n_runs, deadline, first=a["tr"], cache=cache
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
