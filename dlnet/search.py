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
    trainer, or, for an untrained candidate, one timed training step times the steps of an
    epoch (Trainer.probe_epoch_seconds; FLOPs alone missed Python-loop families by 10-100x,
    agent.md 9.10). A run is only started when it is predicted to end before its deadline
    (agent.md 9.2: budgets were overrun by runs started a moment before the deadline).
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
        if tr is not None and hasattr(tr, "probe_epoch_seconds"):
            if not hasattr(tr, "probed"):
                try:
                    tr.probed = tr.probe_epoch_seconds()
                except Exception:  # noqa: BLE001  a broken candidate fails (and is recorded) in _train_step
                    tr.probed = 0.0
            return epochs * tr.probed
        if cfg.get("flops") and self.per_flop:
            return epochs * float(np.median(self.per_flop)) * cfg["flops"]
        return epochs * float(np.median(self.per_epoch)) if self.per_epoch else 0.0


def _open(a, data, build_fn, failures, log=print) -> bool:
    """Build the candidate's model and trainer on first use; False if it cannot be built.

    Candidates are built one at a time, when their turn comes: building all of them before the
    search (as until v1.0.1) kept every model in memory and let ONE model that cannot be allocated
    end the whole run (segmentation at 256x256: an attention block asked for 60 GB; agent.md 9.10).
    """
    if a.get("tr") is not None:
        return True
    if a.get("dead"):
        return False
    try:
        a["tr"] = _new_trainer(a["cfg"], data, 0, build_fn)
    except Exception as e:  # noqa: BLE001  one candidate that cannot be built must not end the run
        a["dead"] = True
        failures.append(
            {"candidate": _label(a["cfg"]), "error": f"{type(e).__name__}: {str(e)[:200]}"}
        )
        log(f"[search] {_label(a['cfg'])} could not be built: {failures[-1]['error']}")
        return False
    return True


def _label(cfg) -> str:
    return f"{cfg['family']} ({cfg.get('origin', 'family')})"


def _train_step(a, epochs, failures, log=print) -> bool:
    """Train candidate `a` for `epochs` more epochs and score it on validation.

    A candidate that raises (shape bug, out of memory, ...) or ends with a non-finite validation
    score (diverged) is dropped and recorded in `failures` instead of ending the whole run; a NaN
    score would also corrupt every later sort (agent.md 9.4, audit R9)."""
    try:
        a["tr"].fit(epochs)
        v = a["tr"].evaluate("va")["score"]
    except Exception as e:  # noqa: BLE001  one broken candidate must not end the run
        failures.append(
            {"candidate": _label(a["cfg"]), "error": f"{type(e).__name__}: {str(e)[:200]}"}
        )
        log(f"[search] {_label(a['cfg'])} failed: {failures[-1]['error']}")
        return False
    if not math.isfinite(v):
        failures.append(
            {"candidate": _label(a["cfg"]), "error": f"non-finite validation score ({v})"}
        )
        log(f"[search] {_label(a['cfg'])} diverged (validation score {v}); dropped")
        return False
    a["val0"] = v
    return True


def sampling_noise(score: float, primary: str, n: int) -> float:
    """Standard error of ONE validation score caused by the finite validation set.

    accuracy-like metrics: binomial sqrt(p (1 - p) / n); RMSE: RMSE / sqrt(2 n) and MSE:
    MSE sqrt(2 / n) (normal errors, delta method). With a single run per model this is the only
    noise estimate there is: reporting +/- 0 would claim a precision the data does not have
    (audit S4). n = number of validation samples (images for dense tasks: pixels are correlated).
    """
    n = max(int(n), 1)
    if primary in ("accuracy", "balanced_accuracy", "miou", "kappa"):
        p = min(max(score, 0.01), 0.99)
        return float((p * (1 - p) / n) ** 0.5)
    if primary == "neg_rmse":
        return float(abs(score) / (2 * n) ** 0.5)
    if primary == "neg_mse":
        return float(abs(score) * (2 / n) ** 0.5)
    return 0.0


def annotate_noise(results, data):
    """r["val_noise"] = the larger of the seed/split std and the validation sampling error."""
    n, primary = len(data["yva"]), data["primary"]
    for r in results:
        sn = sampling_noise(r["val_mean"], primary, n)
        r["val_noise"] = max(r["val_std"], sn)
        r["noise_source"] = (
            "runs" if r["n_seeds"] > 1 and r["val_std"] >= sn else "validation sampling"
        )
    return results


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
    Runs that raise or end with a non-finite validation score are skipped.
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
            try:
                if t is None:
                    t = _new_trainer(cfg, fold, i, build_fn)
                    t.fit(epochs)
                hit = {
                    "val": t.evaluate("va")["score"],
                    "test": t.evaluate("te"),
                    "secs": t.seconds,
                }
            except Exception:  # noqa: BLE001, S112  a failed run is skipped, not fatal (audit R9)
                continue
            if not math.isfinite(hit["val"]):
                continue
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
    alive = [{"cfg": c, "tr": None} for c in cands]  # built lazily, see _open
    log(f"[search] {len(alive)} candidates, rungs={rungs}")
    pace, failures = Pace(), (trace if trace is not None else {}).setdefault("failed", [])
    # Successive halving gives every rung the same share of the budget (Jamieson & Talwalkar 2016,
    # Algorithm 1: B / ceil(log_eta n) per round); unused time carries over to the next rung. Before
    # v0.21 rung 1 could take the whole budget, so top-1 was often trained only 4 epochs (audit A9).
    start, by_rung = clock.now(), []
    for r, ep in enumerate(rungs):
        rung_deadline = start + (r + 1) / len(rungs) * (deadline - start)
        finished, tried = [], set()
        for a in alive:
            if clock.now() > rung_deadline:
                break
            if not _open(a, data, build_fn, failures, log):
                continue
            cost = pace.cost(a["cfg"], ep - a["tr"].epochs_done, a["tr"])
            if not clock.fits(cost, rung_deadline):
                if r == 0:  # keep the price, free the model (rebuilt if nothing else fits)
                    a["cost"], a["tr"] = cost, None
                continue  # would end after the deadline; a cheaper candidate may still fit
            tried.add(id(a))
            if _train_step(a, ep - a["tr"].epochs_done, failures, log):
                pace.record(a["tr"], a["cfg"])
                finished.append(a)
        # Nothing fitted rung 1: train the cheapest candidates (predicted) until one has a score,
        # so even a spent budget yields a result without starting with the slowest one.
        if r == 0 and not finished:
            untried = [a for a in alive if id(a) not in tried and not a.get("dead")]
            for a in sorted(untried, key=lambda a: a.get("cost", math.inf)):
                if not _open(a, data, build_fn, failures, log):
                    continue
                if _train_step(a, ep - a["tr"].epochs_done, failures, log):
                    pace.record(a["tr"], a["cfg"])
                    finished.append(a)
                    break
        if r == 0 and not finished:
            raise SystemExit(f"every candidate failed to train ({len(failures)} failures, first: "
                             f"{failures[0]['error'] if failures else '-'})")  # fmt: skip
        if not finished:  # budget spent before this rung started: rank what the last rung left
            log("[search] time budget reached; ranking with what is done")
            break
        if r == 0 and trace is not None:
            trace["rung1"], trace["rung1_epochs"] = [(a["cfg"], a["val0"]) for a in finished], ep
        # Only candidates that completed THIS rung compete: never rank different epoch counts.
        alive = _ranked(finished, rank)
        by_rung.append((ep, alive))
        keep = max(math.ceil(len(alive) / eta), min(len(alive), topk * 2))
        if r < len(rungs) - 1:
            alive = alive[:keep]
        log(f"[search] rung {r + 1} ({ep} ep): kept {len(alive)}; best val={alive[0]['val0']:.4f}")
        if clock.now() > deadline:
            log("[search] time budget reached; ranking with what is done")
            break
    # Finalists: the most-trained survivors first; when fewer than the Top-K reached the last rung,
    # the best of earlier rungs fill it (ranked apart by final_order, flagged in the leaderboard)
    n_final, finalists = max(topk * 2, topk + 2), []
    for ep, ranked in reversed(by_rung):
        finalists += [a for a in ranked if a["tr"].epochs_done == ep and a not in finalists]
    finalists = finalists[:n_final]
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
        if not vals:  # its stage-1 run failed on test or every run diverged
            continue
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
    failures = (trace if trace is not None else {}).setdefault("failed", [])
    while pool and (clock.now() < deadline or not done):  # a spent budget still yields one result
        it += 1
        for b, rungs in enumerate(brackets):
            if not pool or (clock.now() >= deadline and done):  # >=: a work budget ends exactly
                break
            take, pool = pool[: rungs[0][0]], pool[rungs[0][0] :]
            alive = [{"cfg": c, "tr": None} for c in take]
            alive = [a for a in alive if _open(a, data, build_fn, failures, log)]
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
                    if _train_step(a, left, failures, log):
                        pace.record(a["tr"], a["cfg"])
                        finished.append(a)
                if not finished:
                    break
                if ep == unit:
                    first_rung += [(a["cfg"], a["val0"]) for a in finished]
                alive = _ranked(finished, rank)
                if i < len(rungs) - 1:
                    alive = alive[: max(rungs[i + 1][0], 1)]
            alive = [
                a for a in alive if "val0" in a
            ]  # a bracket may train nobody (budget, failures)
            if not alive:
                continue
            full = [a for a in alive if a["tr"].epochs_done >= R]
            done += full or alive[:1]
            log(f"[hyperband] iteration {it} bracket {b + 1}: {len(take)} configs -> "
                f"{len(full)} at {R} epochs; best val={alive[0]['val0']:.4f}")  # fmt: skip
    if trace is not None:
        trace["rung1"], trace["rung1_epochs"] = first_rung, unit
    if not done:
        raise SystemExit(f"every candidate failed to train ({len(failures)} failures)")
    pool_final = []  # most-trained first; lower epochs only fill the Top-K (final_order ranks apart)
    for ep in sorted({a["tr"].epochs_done for a in done}, reverse=True):
        pool_final += _ranked([a for a in done if a["tr"].epochs_done == ep], rank)
    finalists = pool_final[: max(topk * 2, topk + 2)]
    return stage2(finalists, folds, build_fn, seeds, deadline, cache)


def final_order(results, rank=None):
    """Order results for the leaderboard: models trained for more epochs first, then by `rank`
    (an objective, dlnet/objectives.py) or by validation inside each epoch group.

    A model trained 4 epochs and one trained 12 cannot be compared on their scores (the principle
    of successive halving); results of the first generation and of evolution can differ in epochs
    when the budget ran out (audit S3). `equalize` first tries to bring them to the same epochs."""
    by: dict[int, list] = {}
    for r in results:
        by.setdefault(r["epochs"], []).append(r)
    out = []
    for ep in sorted(by, reverse=True):
        g = by[ep]
        out += (
            [g[i] for i in rank([(r["cfg"], r["val_mean"]) for r in g])]
            if rank
            else sorted(g, key=lambda r: -r["val_mean"])
        )
    return out


def equalize(results, folds, build_fn, deadline, cache=None, log=print):
    """Train results with fewer epochs on to the largest epoch count, while the budget allows
    (best first), re-running their seeds/splits; the others keep their epochs and are ranked
    after (final_order). Needs the stage-1 trainer of each result (r["trainer"])."""
    if not results:
        return results
    target = max(r["epochs"] for r in results)
    for r in sorted(results, key=lambda r: -r["val_mean"]):
        tr = r.get("trainer")
        if r["epochs"] >= target or tr is None or not tr.epochs_done:
            continue
        per_epoch = 1.0 if clock.mode() == "work" else tr.seconds / tr.epochs_done
        cost = per_epoch * ((target - r["epochs"]) + (r["n_seeds"] - 1) * target)
        if not clock.fits(cost, deadline):
            continue
        try:
            tr.fit(target - tr.epochs_done)
        except Exception:  # noqa: BLE001, S112  keep the shorter result rather than lose the model
            continue
        vals, tests, secs, n_splits = eval_runs(
            r["cfg"], folds, build_fn, target, r["n_seeds"], deadline, first=tr, cache=cache,
            run_cost=per_epoch * target,
        )  # fmt: skip
        if vals:
            r.update(summarize(vals, tests), epochs=target, n_splits=n_splits,
                     train_seconds=round(secs, 1))  # fmt: skip
            log(
                f"[equalize] {r['cfg']['family']}: trained on to {target} epochs, val={r['val_mean']:.4f}"
            )
    return results


def more_runs(results, folds, build_fn, deadline, runs=3, top=6, cache=None, log=print):
    """Repeat the best single-run results (other seeds / splits) while the budget allows, so their
    spread is measured instead of estimated, and noise-aware choices (groups, hyperparameters) have
    data (audit R4, R11). Best first; a result keeps its single run when the next run does not fit."""
    for r in sorted(results, key=lambda r: (-r["epochs"], -r["val_mean"]))[:top]:
        tr = r.get("trainer")
        if r["n_seeds"] >= runs or tr is None or not tr.epochs_done:
            continue
        per_epoch = 1.0 if clock.mode() == "work" else tr.seconds / tr.epochs_done
        if not clock.fits(per_epoch * r["epochs"], deadline):
            continue
        vals, tests, secs, n_splits = eval_runs(
            r["cfg"], folds, build_fn, r["epochs"], runs, deadline, first=tr, cache=cache,
            run_cost=per_epoch * r["epochs"],
        )  # fmt: skip
        if len(vals) > r["n_seeds"]:
            r.update(summarize(vals, tests), n_splits=n_splits, train_seconds=round(secs, 1))
            log(
                f"[runs] {r['cfg']['family']}: {r['n_seeds']} runs, val={r['val_mean']:.4f} +/- {r['val_std']:.4f}"
            )
    return results


def assign_groups(results, ordered=False):
    """Mark 'indistinguishable' groups: neighbours (in ranking order) of the same epoch count whose
    validation scores differ by less than the larger noise (val_noise: seed/split std, or the
    validation sampling error when there is one run). `ordered=False` sorts by epochs then val."""
    if not ordered:
        results.sort(key=lambda r: (-r["epochs"], -r["val_mean"]))
    g = 0
    for i, r in enumerate(results):
        if i:
            q = results[i - 1]
            noise = max(q.get("val_noise", q["val_std"]), r.get("val_noise", r["val_std"]), 1e-9)
            if q["epochs"] != r["epochs"] or abs(q["val_mean"] - r["val_mean"]) > noise:
                g += 1
        r["group"] = g
    return results


def select_topk(results, k, keep_order=True):
    """Diversity-aware selection: best per family first, then fill by rank; the chosen models keep
    the order of `results` (final_order). Roles: best-small only if smaller than top1, most-stable
    only when every chosen model was run at least twice (one run has no measured spread)."""
    chosen, fams = [], set()
    for r in results:
        if r["cfg"]["family"] not in fams and len(chosen) < k:
            chosen.append(r)
            fams.add(r["cfg"]["family"])
    for r in results:
        if r not in chosen and len(chosen) < k:
            chosen.append(r)
    if keep_order:
        pos = {id(r): i for i, r in enumerate(results)}
        chosen.sort(key=lambda r: pos[id(r)])
    else:
        chosen.sort(key=lambda r: -r["val_mean"])
    top1 = chosen[0]
    roles = {id(top1): ["top1"]}
    rest = chosen[1:]
    if rest:
        smaller = [r for r in rest if r["cfg"]["params"] < top1["cfg"]["params"]]
        if smaller:
            roles.setdefault(id(min(smaller, key=lambda r: r["cfg"]["params"])), []).append(
                "best-small"
            )
        if all(r["n_seeds"] >= 2 for r in chosen):
            roles.setdefault(id(min(chosen, key=lambda r: r["val_std"])), []).append("most-stable")
        diff = [r for r in rest if r["cfg"]["family"] != top1["cfg"]["family"]]
        if diff:
            roles.setdefault(id(diff[0]), []).append("most-different")
    for r in chosen:
        r["roles"] = roles.get(id(r), [])
    return chosen
