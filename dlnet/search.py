"""Search and rank: successive halving, then multi-seed evaluation of the survivors."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import clock, jsonsafe
from .core import Trainer


def _new_trainer(cfg, data, seed, build_fn):
    torch.manual_seed(seed)  # the seed must control the weight init too, not only the batch order
    m = build_fn(cfg, data["in_dim"], data["out_dim"])
    return Trainer(m, data, lr=cfg["lr"], wd=cfg["wd"], batch=cfg["batch"], seed=seed)


class Pace:
    """Predicts what training a candidate for some epochs will cost on the budget clock.

    Work mode: the number of epochs (exact). Time mode: seconds per epoch measured on this
    trainer, or, for an untrained candidate, the median seconds per epoch and per FLOP of the
    candidates trained so far times its FLOPs (the median seconds per epoch without FLOPs).
    0 before anything was measured, so the very first run always starts. A run is only started
    when it is predicted to end before its deadline (agent.md 9.2: budgets were overrun by
    runs started a moment before the deadline).
    """

    def __init__(self):
        self.per_epoch, self.per_flop = [], []

    def record(self, tr, cfg):
        if tr.epochs_done:
            sec = tr.seconds / tr.epochs_done
            self.per_epoch.append(sec)
            if cfg.get("flops"):
                self.per_flop.append(sec / cfg["flops"])

    def cost(self, cfg, epochs, tr=None) -> float:
        if epochs <= 0:
            return 0.0
        if clock.mode() == "work":
            return float(epochs)
        if tr is not None and tr.epochs_done:
            return epochs * tr.seconds / tr.epochs_done
        if cfg.get("flops") and self.per_flop:
            return epochs * float(np.median(self.per_flop)) * cfg["flops"]
        return epochs * float(np.median(self.per_epoch)) if self.per_epoch else 0.0


def _ranked(items, rank=None):
    """Order trained candidates best first: by validation score, or by `rank` (an objective, see
    dlnet/objectives.py) applied to [(cfg, val), ...]."""
    if rank is None:
        return sorted(items, key=lambda a: -a["val0"])
    return [items[i] for i in rank([(a["cfg"], a["val0"]) for a in items])]


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
        tmp.write_text(jsonsafe.dumps(self.d))
        tmp.replace(self.path)


def eval_runs(cfg, folds, build_fn, epochs, n_runs, deadline, first=None, cache=None, run_cost=0.0):
    """Train `cfg` n_runs times: run i uses seed i on split folds[i % len(folds)].

    With one fold this is plain multi-seed evaluation; with several folds (repeated splits) the
    spread also contains the split noise, which seed std alone misses (agent.md sec. 9 item 7).
    `first` is an already trained seed-0 trainer on folds[0], reused as run 0.
    `cache` (RunCache) returns stored results of identical runs instead of training again.
    `run_cost`: predicted cost of one run on the budget clock; a run that would end after
    `deadline` is not started (cached runs are always used: they cost nothing).
    Returns (val scores, test metric dicts, seconds, number of distinct splits used).
    """
    vals, tests, secs = [], [], 0.0
    for i in range(n_runs):
        t = first if i == 0 and first is not None else None
        fold = folds[i % len(folds)]
        key = cache and RunCache.key(cfg, fold, i, epochs)
        hit = cache.get(key) if cache and t is None else None
        if hit is None and vals and not clock.fits(0.0 if t else run_cost, deadline):
            break
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
    cache=None, trace=None, rank=None,
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
    pace = Pace()
    for r, ep in enumerate(rungs):
        finished = []
        for i, a in enumerate(alive):
            # The very first candidate always trains, so even a spent budget yields a result.
            if clock.now() > deadline and not (r == 0 and i == 0):
                break
            if (r or i) and not clock.fits(
                pace.cost(a["cfg"], ep - a["tr"].epochs_done, a["tr"]), deadline
            ):
                continue  # would end after the deadline; a cheaper candidate may still fit
            a["tr"].fit(ep - a["tr"].epochs_done)
            pace.record(a["tr"], a["cfg"])
            a["val0"] = a["tr"].evaluate("va")["score"]
            finished.append(a)
        if not finished:  # budget spent before this rung started: rank what the last rung left
            log("[search] time budget reached; ranking with what is done")
            break
        if r == 0 and trace is not None:
            trace["rung1"], trace["rung1_epochs"] = [(a["cfg"], a["val0"]) for a in finished], ep
        # Only candidates that completed THIS rung compete: never rank different epoch counts.
        alive = _ranked(finished, rank)
        keep = max(math.ceil(len(alive) / eta), min(len(alive), topk * 2))
        if r < len(rungs) - 1:
            alive = alive[:keep]
        log(f"[search] rung {r + 1} ({ep} ep): kept {len(alive)}; best val={alive[0]['val0']:.4f}")
        if clock.now() > deadline:
            log("[search] time budget reached; ranking with what is done")
            break
    finalists = alive[: max(topk * 2, topk + 2)]
    return stage2(finalists, folds, build_fn, seeds, deadline, cache)


def stage2(finalists, folds, build_fn, seeds, deadline, cache=None):
    """Finalists on several seeds (and splits); run 0 reuses the stage-1 trainer."""
    results = []
    n_runs = max(seeds, len(folds))
    for a in finalists:
        cfg, ep = a["cfg"], a["tr"].epochs_done
        cost = float(ep) if clock.mode() == "work" else a["tr"].seconds
        vals, tests, secs, n_splits = eval_runs(
            cfg, folds, build_fn, ep, n_runs, deadline, first=a["tr"], cache=cache, run_cost=cost
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


def hyperband_brackets(R: int, eta: int = 3, unit: int = 4) -> list[list[tuple[int, int]]]:
    """Brackets of Hyperband (Li et al., JMLR 2018, arXiv:1603.06560, Algorithm 1).

    Resource = epochs, counted in units of `unit` epochs: R_units = R / unit, s_max =
    floor(log_eta(R_units)), B = (s_max + 1) R_units. Bracket s starts n = ceil(B / R_units *
    eta^s / (s + 1)) configurations at r = R_units eta^-s units; rung i has n_i = floor(n eta^-i)
    configurations trained to r_i = r eta^i units. Returns, per bracket, [(n_i, epochs_i), ...].

    This is Algorithm 1 as written. Table 1 of the paper (R = 81, eta = 3) lists different bracket
    sizes for s = 3, 2, 1 (27, 9, 6 instead of 34, 15, 8): the table matches
    floor((s_max + 1) / (s + 1)) * eta^s, not the ceil of line 2. DLNet follows the algorithm.
    """
    Ru = max(R // unit, 1)
    s_max = math.floor(math.log(Ru, eta) + 1e-9)
    B = (s_max + 1) * Ru
    out = []
    for s in range(s_max, -1, -1):
        n = math.ceil(B / Ru * eta**s / (s + 1))
        r = Ru * eta ** (-s)
        out.append([(int(n * eta ** (-i)), round(r * eta**i * unit)) for i in range(s + 1)])
    return out


def run_hyperband(
    data, cands, deadline, topk, build_fn, R=36, eta=3, unit=4, seeds=3, log=print, folds=None,
    cache=None, trace=None, rank=None,
):  # fmt: skip
    """Hyperband over a FIXED, ordered candidate pool, then the multi-seed stage of `run_search`.

    Hyperband samples fresh random configurations per bracket; here bracket after bracket takes
    the next unused candidates of the pool (the seed selector's order), and the whole schedule is
    repeated while candidates and time remain ("we recommend repeating it indefinitely", Sec. 3).
    Only configurations that reached the full resource R compete in the final ranking (the same
    number of epochs). `trace["rung1"]` holds (cfg, val) of the configurations of the most
    aggressive bracket after `unit` epochs (the signal proxies and meta-learning are compared with).
    """
    folds = folds or [data]
    if not cands:
        raise ValueError("no candidates to search")
    pool = list(cands)
    brackets = hyperband_brackets(R, eta, unit)
    log(f"[hyperband] {len(pool)} candidates; brackets (n, epochs): {brackets}")
    done, first_rung, it, pace = [], [], 0, Pace()
    while pool and (clock.now() < deadline or not done):  # a spent budget still yields one result
        it += 1
        for b, rungs in enumerate(brackets):
            if not pool or (clock.now() > deadline and done):
                break
            take, pool = pool[: rungs[0][0]], pool[rungs[0][0] :]
            alive = [{"cfg": c, "tr": _new_trainer(c, data, 0, build_fn)} for c in take]
            for i, (n_i, ep) in enumerate(rungs):
                if i and clock.now() > deadline:  # budget spent: keep what the last rung gave
                    break
                finished = []
                for a in alive[: max(n_i, 1)]:
                    if clock.now() > deadline and (done or finished):
                        break
                    left = ep - a["tr"].epochs_done
                    if (done or finished) and not clock.fits(
                        pace.cost(a["cfg"], left, a["tr"]), deadline
                    ):
                        continue
                    a["tr"].fit(left)
                    pace.record(a["tr"], a["cfg"])
                    a["val0"] = a["tr"].evaluate("va")["score"]
                    finished.append(a)
                if not finished:
                    break
                if ep == unit:
                    first_rung += [(a["cfg"], a["val0"]) for a in finished]
                alive = _ranked(finished, rank)
                if i < len(rungs) - 1:
                    alive = alive[: max(rungs[i + 1][0], 1)]
            full = [a for a in alive if a["tr"].epochs_done >= R]
            done += full or alive[:1]
            log(f"[hyperband] iteration {it} bracket {b + 1}: {len(take)} configs -> "
                f"{len(full)} at {R} epochs; best val={alive[0]['val0']:.4f}")  # fmt: skip
    if trace is not None:
        trace["rung1"], trace["rung1_epochs"] = first_rung, unit
    best_ep = max(a["tr"].epochs_done for a in done)
    pool_final = _ranked([a for a in done if a["tr"].epochs_done == best_ep], rank)
    finalists = pool_final[: max(topk * 2, topk + 2)]
    return stage2(finalists, folds, build_fn, seeds, deadline, cache)


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


def select_topk(results, k, keep_order=False):
    """Diversity-aware selection: best per family first, then fill by rank."""
    chosen, fams = [], set()
    for r in results:
        if r["cfg"]["family"] not in fams and len(chosen) < k:
            chosen.append(r)
            fams.add(r["cfg"]["family"])
    for r in results:
        if r not in chosen and len(chosen) < k:
            chosen.append(r)
    if keep_order:  # results were already ordered by an objective: keep that order
        pos = {id(r): i for i, r in enumerate(results)}
        chosen.sort(key=lambda r: pos[id(r)])
    else:
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
