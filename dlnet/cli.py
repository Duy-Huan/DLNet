"""Command line: dlnet design --data <file> [plugin options] --budget 5m --topk 3"""

from __future__ import annotations

import argparse
import copy
import re
import time

from . import __version__


def parse_budget(s: str) -> float:
    """Seconds for 90s / 5m / 2h / 300; for 400ep a number of training epochs (see dlnet/clock.py)."""
    return parse_budget_mode(s)[0]


def parse_budget_mode(s: str) -> tuple[float, str]:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(s|m|h|ep)?", s.strip())
    if not m:
        raise SystemExit(
            f"bad --budget {s!r} (use e.g. 90s, 5m, 2h, or 400ep for a number of epochs)"
        )
    unit = m.group(2) or "s"
    if unit == "ep":
        return float(m.group(1)), "work"
    return float(m.group(1)) * {"s": 1, "m": 60, "h": 3600}[unit], "time"


def _budget_report(text, mode, budget, wall, threads) -> dict:
    """Where the budget went: seconds and epochs per stage, and the overrun of a seconds budget."""
    from . import clock

    stages = clock.breakdown()
    from . import core

    rep = {"budget": text, "mode": mode, "threads": threads, "device": core.DEVICE, "amp": core.AMP, "wall_seconds": round(wall, 1),
           "epochs_trained": round(clock.epochs(), 1), "stages": stages}  # fmt: skip
    spent = wall if mode == "time" else clock.epochs()
    rep["overrun_pct"] = round(100 * (spent - budget) / budget, 1) if budget else 0.0
    shown = ", ".join(f"{d['stage']} {d['seconds']:.1f}s/{d['epochs']:.0f}ep" for d in stages)
    print(f"[budget] {shown}")
    if rep["overrun_pct"] > 10:
        print(f"[budget] WARNING: spent {spent:.0f} of a {text} budget (+{rep['overrun_pct']}%)")
    return rep


def _seconds_per_epoch(results) -> float:
    """Budget-clock cost of one epoch: 1 in work mode, else the median measured seconds/epoch."""
    from . import clock

    if clock.mode() == "work":
        return 1.0
    xs = [r["trainer"].seconds / r["trainer"].epochs_done for r in results
          if r.get("trainer") is not None and r["trainer"].epochs_done]  # fmt: skip
    return float(sorted(xs)[len(xs) // 2]) if xs else 0.0


def _warnings(data, chosen, failed) -> list[str]:
    """Plain-language caveats shown at the top of the leaderboard (audit S3, S4, Q2, R9)."""
    out = list(data.get("warnings", []))
    if any(r["n_seeds"] < 2 for r in chosen):
        out.append("Some models were trained only once: their +/- is the sampling error of the "
                   "validation set, not run-to-run noise. A larger --budget repeats the finalists.")  # fmt: skip
    eps = sorted({r["epochs"] for r in chosen})
    if len(eps) > 1:
        out.append(f"The Top-K mixes models trained {eps} epochs (the budget ended before they could be "
                   "trained to the same length); compare scores only within the same epochs.")  # fmt: skip
    if chosen[0]["epochs"] < 12:
        out.append(f"top1 was trained only {chosen[0]['epochs']} epochs: its rank is a rough early "
                   "estimate. Increase --budget, or reduce --zoo / --per-family.")  # fmt: skip
    if failed:
        out.append(
            f"{len(failed)} candidate(s) failed or diverged and were dropped (failed_candidates)."
        )
    return out


def _apply_cost_model(plugin, data, cands, args):
    """Measure params / FLOPs / VRAM of every candidate; drop the ones over the limits."""
    from .zoo import cost

    inp = plugin.zoo_input(data)
    if inp is None:
        return cands
    max_flops, max_vram = cost.parse_size(args.max_flops), cost.parse_size(args.max_vram)
    kept, dropped = [], 0
    for c in cands:
        try:
            c.update(
                cost.estimate(
                    plugin.build_model, c, data["in_dim"], data["out_dim"], inp[1], kind=inp[0]
                )
            )
        except (RuntimeError, ValueError, TypeError) as e:  # a candidate we cannot cost still runs
            print(f"[cost] {c['family']}: not costed ({str(e)[:60]})")
            kept.append(c)
            continue
        if cost.within(c, max_flops, max_vram) and (
            not args.max_params or c.get("params", 0) <= cost.parse_size(args.max_params)
        ):
            kept.append(c)
        else:
            dropped += 1
    if dropped:
        print(
            f"[cost] dropped {dropped} candidates over --max-flops {args.max_flops} / --max-vram {args.max_vram}"
        )
    if kept and "flops" in kept[0]:
        fl = sorted(c["flops"] for c in kept if "flops" in c)
        print(
            f"[cost] {len(kept)} candidates; FLOPs/sample min {fl[0]:,} median {fl[len(fl) // 2]:,} max {fl[-1]:,}"
        )
    return kept


def design(args) -> int:
    from . import core, threads

    try:
        dev = core.pick_device(args.device)
    except ValueError as e:
        raise SystemExit(f"--device: {e}") from None
    amp = core.amp_dtype(dev, args.amp)
    saved = core.DEVICE, core.AMP, core.RECIPE
    core.DEVICE, core.AMP, core.RECIPE = (
        str(dev),
        args.amp,
        args.recipe,
    )  # every Trainer of this run
    print(f"[device] {dev}" + (f" ({torch_name(dev)})" if dev.type == "cuda" else "")
          + f", precision {'float32' if amp is None else str(amp).split('.')[-1]} (--device, --amp)")  # fmt: skip
    try:
        with threads.limit(threads.resolve(args.threads)) as n:
            print(f"[threads] {n} (of {threads.available_cpus()} CPUs; --threads to change)")
            return _design(args, n)
    finally:
        core.DEVICE, core.AMP, core.RECIPE = saved


def torch_name(dev) -> str:
    import torch

    return torch.cuda.get_device_name(dev)


def _design(args, n_threads) -> int:
    import functools
    from pathlib import Path

    import numpy as np
    import torch

    from . import (
        advisor,
        clock,
        export,
        hparams,
        meta,
        objectives,
        plugins,
        proxies,
        search,
        seeds,
        zoo,
    )
    from .zoo import space

    t_wall = time.time()
    from .zoo import cost as _cost

    for k in ("max_params", "max_flops", "max_vram"):  # fail before any work, not with a traceback
        try:
            _cost.parse_size(getattr(args, k))
        except (ValueError, KeyError):
            raise SystemExit(f"bad --{k.replace('_', '-')} {getattr(args, k)!r}: use a number such as "
                             "500k, 50M or 2G") from None  # fmt: skip
    budget, clock_mode = parse_budget_mode(args.budget)
    t0 = clock.start(clock_mode)
    clock.stage("load")
    deadline = t0 + budget
    search_deadline = t0 + 0.8 * budget  # the last 20% goes to hyperparameter estimation
    export.check_out_dir(args.out)  # fail before spending the budget, not after
    cache = None if args.no_cache else search.RunCache(Path(args.out) / export.CACHE_NAME)
    plugin = plugins.route(args.data) if args.plugin == "auto" else plugins.get_plugin(args.plugin)
    print(f"[plugin] {plugin.name}  data={args.data}")

    data = plugin.prepare(args)
    print(
        f"[load] task={data['task']} features={data['in_dim']} classes={data['out_dim']} seed={args.seed}"
    )
    prof = plugin.profile(data)
    shown = {k: v for k, v in prof.items() if k not in ("task",)}
    print(f"[profile] {shown}")
    card = {"n_train": prof.get("n_train", 0)}
    if data.get("kind") in ("image", "segmentation") and data["Xtr"].dim() == 4:
        from . import datacard

        card = datacard.image_card(data, seed=args.seed)
        print(f"[datacard] {card}")

    n_cv = args.cv_splits or plugin.default_cv_splits(args)
    folds = [data]
    for i in range(
        1, n_cv
    ):  # repeated splits: same data, split seed + i (preprocessing refit on train)
        a = copy.copy(args)
        a.seed = args.seed + i
        try:
            f = plugin.prepare(a)
        except (ValueError, SystemExit) as e:
            print(f"[cv] split {i} skipped: {e}")
            continue
        if (f["in_dim"], f["out_dim"]) == (data["in_dim"], data["out_dim"]):
            folds.append(f)
    if n_cv > 1:
        print(f"[cv] ranking finalists on {len(folds)} different splits")

    clock.stage("reference")
    print("[reference] classical baselines on the same split(s)")
    ref = export.mean_reference([plugin.reference(f) for f in folds])
    for r in ref:
        print(f"[reference] {r['name']:<26} val={r['va']['score']:.4f}")

    clock.stage("generate")
    cands = plugin.generate(prof, args.per_family, args.seed)
    zoo_cands = plugin.generate_zoo(
        data, prof, args.zoo, args.seed, not args.no_zoo_presets, args.zoo_width
    )
    if zoo_cands:
        origins = sorted({c["origin"].split(":")[0] for c in zoo_cands})
        print(f"[zoo] {len(zoo_cands)} architectures from the module zoo ({', '.join(origins)})")
    cands += zoo_cands
    n_generated = len(cands)
    smallest = min((c.get("params") or 0 for c in cands), default=0)
    cands = _apply_cost_model(plugin, data, cands, args)
    if not cands:
        limits = [f"--{k.replace('_', '-')} {getattr(args, k)}" for k in ("max_params", "max_flops", "max_vram")
                  if getattr(args, k)]  # fmt: skip
        raise SystemExit(
            f"no valid candidate: {n_generated} were generated" + (
                f" but none fits {', '.join(limits)} (the smallest has {smallest:,} parameters); loosen the limit"
                if limits and n_generated else "; raise --per-family / --zoo")
        )  # fmt: skip
    fams = sorted({c["family"] for c in cands})
    print(f"[generate] {len(cands)} valid candidates from families {fams}")
    meta_rec = None
    if args.meta_store:  # past runs vote on the family ORDER; no family is removed
        meta_rec = meta.recommend(meta.load(args.meta_store), plugin.name, meta.features(prof))
        if meta_rec:
            cands = meta.reorder(cands, meta_rec["family_order"])
            print(
                f"[meta] {meta_rec['n_records']} past runs; family order {meta_rec['family_order']}"
            )
        else:
            print(f"[meta] fewer than {meta.MIN_RECORDS} past {plugin.name} runs: default order")

    # Zero-cost proxies of the untrained candidates: measured against real training below; they
    # vote in the seed selection only with --seed-proxies (agent.md 16.1). At most 10% of the budget.
    clock.stage("proxies")
    zc, t_px = {}, time.time()  # the proxy stage gets 10% of the budget from ITS start
    px_limit = 0.1 * budget if clock_mode == "time" else float("inf")  # proxies train nothing
    for c in cands:
        if args.no_proxies or time.time() > t_px + px_limit:
            break
        torch.manual_seed(args.seed)
        zc[id(c)] = proxies.score(
            plugin.build_model(c, data["in_dim"], data["out_dim"]), data, seed=args.seed
        )
    clock.stage("seed_selection")
    zinp = plugin.zoo_input(data)
    if args.extra_specs and zinp:  # user- or AI-written architectures (JSON list of zoo specs)
        import json as _json

        rng_x = np.random.default_rng(args.seed)
        extra = [space._candidate(sp, zinp[0], zinp[1], data["out_dim"], zinp[2], plugin.zoo_train,
                                  rng_x, prof["max_params"], "user_spec")
                 for sp in _json.loads(Path(args.extra_specs).read_text())]  # fmt: skip
        extra = _apply_cost_model(plugin, data, [c for c in extra if c], args)
        print(f"[extra] {len(extra)} valid architectures from {args.extra_specs}")
        cands += extra
    adv_cmd, adv, adv_log = advisor.command(args.advisor, args.advisor_cmd), None, []
    if adv_cmd and zinp:
        adv = advisor.advise(
            adv_cmd, card, args.context, args.objective, cands, zinp[0], zinp[1], data["out_dim"],
            zinp[2], plugin.zoo_train, prof["max_params"], np.random.default_rng(args.seed),
            args.advisor_proposals, timeout=args.advisor_timeout,
        )  # fmt: skip
        if adv:
            n0 = len(cands)
            cands += _apply_cost_model(plugin, data, adv["proposals"], args)
            adv["scores"] = advisor.scores_from_ranking(
                len(cands), adv["ranking"] + list(range(n0, len(cands)))
            )
    sur = None
    if len(card) > 1 and args.surrogate_weight:  # zero-shot surrogate (ZAP) learned from past runs
        from . import surrogate

        past = (meta.load(args.meta_store) if args.meta_store else []) + (
            [] if args.no_builtin_meta else surrogate.builtin_records(plugin.name)
        )
        sur = surrogate.rank_candidates(past, plugin.name, card, cands)
        n_ds = len(surrogate.datasets_from_records(past, plugin.name))
        print(f"[surrogate] ZAP ranking learned from {n_ds} past datasets" if sur
              else "[surrogate] not enough past datasets")  # fmt: skip
    # Seed selection: data card + context prior, past runs, (optionally) the proxy vote
    port = None
    if args.meta_store:
        port = meta.portfolio(meta.load(args.meta_store), plugin.name, meta.features(prof)) or None
    use_px = args.seed_proxies and len(zc) == len(cands)
    goal = args.goal if args.goal != "accuracy" else {
        "params": "small", "flops": "small", "balanced": "balanced"}.get(args.objective, "accuracy")  # fmt: skip
    cands, seed_report = seeds.select(
        cands, card, args.seeds, args.context or "", goal, meta_rec, port,
        [zc[id(c)] for c in cands] if use_px else None,
        # the hand-written prior ranked real validation NEGATIVELY on 3 image sets (audit A6) and
        # lowered the learned ranking (A7): it only votes when nothing learned is available
        (0.0 if sur is not None and not args.context else 1.0, 1.0, 1.0 if use_px else 0.0),
        order=args.seed_order, seed=args.seed,
        surrogate_scores=sur, surrogate_weight=args.surrogate_weight,
        advice=adv["scores"] if adv else None, advice_weight=args.advisor_weight,
    )  # fmt: skip
    if len(zc) == len(cands) and len(cands) > 2:  # the vote and the prior are measured like proxies
        from .seeds import proxy_vote

        for c, v in zip(cands, proxy_vote([zc[id(c)] for c in cands]), strict=True):
            zc[id(c)]["vote"] = v
    for c in cands:  # the prior and the parameter count are measured for every candidate
        zc.setdefault(id(c), {"params": float(c.get("params") or 0)})["seed_prior"] = c[
            "seed_prior"
        ]
    for r in seed_report["rules_fired"]:
        print(f"[seeds] {r}")
    print(
        f"[seeds] {seed_report['kept']}/{seed_report['pool']} candidates, rankers {seed_report['rankers']}; "
        f"first: {[t['origin'] if t['origin'] != 'family' else t['family'] for t in seed_report['top'][:5]]}"
    )
    rank = functools.partial(
        objectives.order, objective=args.objective, cost=args.cost, tol=args.tolerance,
        target=objectives.parse_target(args.target_cost), weight=args.cost_weight,
    )  # fmt: skip
    trace = {}
    clock.stage("search")
    t_search = clock.now()
    gens = args.evolve if any(c.get("spec") for c in cands) else 0
    gen0_deadline = t_search + (0.55 if gens else 1.0) * (search_deadline - t_search)
    runner = search.run_hyperband if args.scheduler == "hyperband" else search.run_search
    results = runner(
        data, cands, gen0_deadline, args.topk, plugin.build_model, folds=folds,
        cache=cache, trace=trace, rank=rank,
    )  # fmt: skip
    evolution = []
    seen = {zoo.signature(c["spec"]) for c in cands if c.get("spec")}
    # aging evolution (Real et al. 2019): population = the last P evaluated models, oldest first
    pop_size = args.evolve_population or max(len(trace.get("rung1", [])), 4)
    population = [cv for cv in trace.get("rung1", []) if cv[0].get("spec")][-pop_size:]
    if gens:
        clock.stage("evolution")
    for g in range(1, gens + 1):
        if clock.now() > search_deadline - (5 if clock_mode == "time" else 1) or not population:
            break
        # Aging evolution needs many children (Real et al. 2019); children that cannot reach the
        # second rung only add 4-epoch noise (audit A4: lost on test 5/5 at 150 epochs)
        need = args.evolve_pop * 12 * _seconds_per_epoch(results)
        if not clock.fits(need, search_deadline):
            print(f"[evolve] skipped: the remaining search budget cannot train {args.evolve_pop} "
                  "children to 12 epochs (raise --budget or lower --evolve-pop)")  # fmt: skip
            break
        S = args.tournament or max(2, len(population) // 4)
        pos = {i: p for p, i in enumerate(rank(population))}  # tournament fitness = objective rank
        kids = space.aging_children(
            [(c, -pos[i]) for i, (c, _) in enumerate(population)],
            args.evolve_pop,
            S,
            args.seed + g,
            prof["max_params"],
            plugin.zoo_train,
            seen,
        )
        parents = sorted(population, key=lambda cv: -cv[1])[:3]
        parents = [p for p, _ in parents]
        if adv_cmd and zinp:  # the advisor sees the results so far and may propose children
            res_tab = [{"id": f"c{i}", "val": round(v, 4)} for i, (_, v) in enumerate(population)]
            a2 = advisor.advise(
                adv_cmd, card, args.context, args.objective, [c for c, _ in population], zinp[0],
                zinp[1], data["out_dim"], zinp[2], plugin.zoo_train, prof["max_params"],
                np.random.default_rng(args.seed + g), args.advisor_proposals, results=res_tab,
                timeout=args.advisor_timeout,
            )  # fmt: skip
            if a2:
                kids += a2["proposals"]
                adv_log.append({"generation": g, "rationale": a2["rationale"],
                                "proposals": len(a2["proposals"])})  # fmt: skip
        kids = _apply_cost_model(
            plugin, data, kids, args
        )  # children are costed and filtered like the rest
        if not kids:
            break
        seen |= {zoo.signature(k["spec"]) for k in kids}
        gen_deadline = clock.now() + (search_deadline - clock.now()) / (gens - g + 1)
        sub = {}
        print(
            f"[evolve] generation {g}: {len(kids)} children of {[p['family'] for p in parents[:3]]}"
        )
        res = search.run_search(
            data,
            kids,
            gen_deadline,
            args.topk,
            plugin.build_model,
            folds=folds,
            cache=cache,
            trace=sub,
            rank=rank,
        )
        results = results + res
        trace.setdefault("failed", []).extend(sub.get("failed", []))
        population = (population + sub.get("rung1", []))[-pop_size:]  # the oldest die
        best = max((v for _, v in sub.get("rung1", [])), default=None)
        evolution.append({"generation": g, "children": len(kids), "best_rung1_val": best})
    if len({r["epochs"] for r in results}) > 1:  # never rank 4-epoch against 12-epoch models
        clock.stage("equalize")
        results = search.equalize(results, folds, plugin.build_model, deadline, cache=cache)
    clock.stage("runs")  # measure the spread of the best models instead of estimating it
    results = search.more_runs(
        results, folds, plugin.build_model, deadline, top=2 * args.topk, cache=cache
    )
    clock.stage("hparams")
    print("[hparams] estimating lr / batch size / weight decay for the finalists")
    results = hparams.refine(
        results, data, plugin.build_model, deadline, log=print, folds=folds, cache=cache,
        mode=args.hparams,
    )  # fmt: skip
    search.annotate_noise(results, data)
    results = search.final_order(results, rank if args.objective != "accuracy" else None)
    search.assign_groups(results, ordered=True)
    top_ep = results[0]["epochs"]
    entries = [(r["cfg"], r["val_mean"]) for r in results if r["epochs"] == top_ep]
    chosen = search.select_topk(results, args.topk)
    n_trained = len(trace.get("rung1", []))
    extras = {"warnings": _warnings(data, chosen, trace.get("failed", []))}
    extras["search_coverage"] = {"candidates": len(cands), "trained": n_trained,
                                 "top_k_requested": args.topk, "top_k_exported": len(chosen)}  # fmt: skip
    if n_trained < max(3, len(cands) // 4):
        extras["warnings"].append(f"Only {n_trained} of {len(cands)} candidates could be trained within the budget; "
                                  "raise --budget, or lower --zoo / --per-family, or use --seeds N.")  # fmt: skip
    if len(chosen) < args.topk:
        extras["warnings"].append(f"Only {len(chosen)} of the {args.topk} requested models could be exported "
                                  "(too few trained candidates).")  # fmt: skip
    if trace.get("failed"):
        extras["failed_candidates"] = trace["failed"]
    for w in extras["warnings"]:
        print(f"[warning] {w}")
    front = objectives.pareto_front(entries, "flops" if args.objective == "flops" else args.cost)
    extras["objective"] = {
        "objective": args.objective, "cost": args.cost, "tolerance": args.tolerance,
        "target_cost": args.target_cost, "cost_weight": args.cost_weight,
        "pareto_front": sorted(
            ({"family": results[i]["cfg"]["family"], "val_mean": round(results[i]["val_mean"], 5),
              "params": results[i]["cfg"].get("params"), "flops": results[i]["cfg"].get("flops")}
             for i in front), key=lambda r: r["params"] or 0),
    }  # fmt: skip
    if args.objective != "accuracy":
        print(f"[objective] {args.objective}: top-1 {chosen[0]['cfg']['family']} "
              f"val={chosen[0]['val_mean']:.4f} params={chosen[0]['cfg'].get('params'):,}; "
              f"Pareto front has {len(front)} models")  # fmt: skip
    if zoo_cands or evolution:
        extras["module_zoo"] = {
            "initial_candidates": len(zoo_cands),
            "evolution": evolution,
            "top_origins": [c["cfg"].get("origin", "legacy family") for c in chosen],
            "ops_registered": len(zoo.OPS),
        }
    if meta_rec:
        extras["meta_recommendation"] = meta_rec
    if adv_cmd:  # reported even when a call failed, so the user sees what the advisor did
        extras["ai_advisor"] = {
            "command": adv_cmd.split()[0],
            "seed_selection": None if adv is None else {
                k: adv[k] for k in ("rationale", "dropped_ids", "invalid_proposals")
            } | {"proposals": len(adv["proposals"])},
            "evolution": adv_log,
            "top1_from_advisor": chosen[0]["cfg"].get("origin") == "ai_advisor",
        }  # fmt: skip
    extras["seed_selection"] = {**seed_report, "data_card": card, "context": args.context or "",
                                "goal": goal, "portfolio": port, "scheduler": args.scheduler}  # fmt: skip
    if args.meta_store:
        rec = meta.make_record(
            plugin.name, prof, trace.get("rung1", []), Path(args.data).name,
            card=card if len(card) > 1 else None, epochs=trace.get("rung1_epochs"),
        )  # fmt: skip
        if rec:
            meta.append(args.meta_store, rec)
            print(f"[meta] stored this run in {args.meta_store}")
    pairs = [(zc[id(c)], val) for c, val in trace.get("rung1", []) if id(c) in zc]
    if len(pairs) >= 3:
        rep = proxies.correlation_report(
            *map(list, zip(*pairs, strict=True)), trace["rung1_epochs"]
        )
        extras["zero_cost_proxies"] = rep
        shown = {k: v["spearman"] for k, v in rep["proxies"].items()}
        print(f"[proxies] Spearman with rung-1 validation (n={len(pairs)}): {shown}")
    if cache and cache.hits:
        print(f"[cache] reused {cache.hits} finished runs from {cache.path}")
    clock.stage("export")
    extras["budget"] = _budget_report(
        args.budget, clock_mode, budget, time.time() - t_wall, n_threads
    )
    v = export.export(args.out, plugin, data, prof, chosen, ref, args.seed, extras=extras)
    spent = f"{clock.now() - t0:.0f} epochs, " if clock_mode == "work" else ""
    print(f"[done] {spent}{time.time() - t_wall:.0f}s  -> {args.out}/leaderboard.md")
    print(
        f"[verdict] beats_reference={v['beats_reference']} "
        f"val_margin={v['val_margin']} test_margin={v['test_margin']}"
    )
    return 0


def selfcheck() -> int:
    """Prove the installation works: design Top-1 on a tiny synthetic CSV, check the package files."""
    import json
    import tempfile
    from pathlib import Path

    import numpy as np
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(240, 6))
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        df["y"] = (X[:, 0] + X[:, 1] > 0).astype(int)
        csv, out = Path(tmp) / "selfcheck.csv", Path(tmp) / "out"
        df.to_csv(csv, index=False)
        argv = ["design", "--data", str(csv), "--target", "y", "--budget", "30s", "--topk", "1"]
        rc = main([*argv, "--per-family", "1", "--zoo", "4", "--evolve", "0", "--out", str(out)])
        board = json.loads((out / "leaderboard.json").read_text())
        pkg = next(out.glob("top1_*"))
        ok = (
            rc == 0
            and bool(board["top_k"])
            and all((pkg / f).exists() for f in ("model.py", "train.py"))
        )
    print(f"selfcheck {'OK' if ok else 'FAILED'} (dlnet {__version__})")
    return 0 if ok else 1


def advisor_presets():
    from .advisor import PRESETS

    return PRESETS


def main(argv=None) -> int:
    from . import gan_run, plugins, unfold_run

    ap = argparse.ArgumentParser(prog="dlnet")
    ap.add_argument("--version", action="version", version=f"dlnet {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck", help="verify the installation end to end (about 20 seconds)")
    unfold_run.add_parser(sub)
    gan_run.add_parser(sub)
    sub.add_parser("zoo", help="list every module of the zoo with kinds, tags and sources")
    sub.add_parser(
        "mcp",
        help="run DLNet as an MCP server on stdio for AI clients (Claude Code, Codex, Antigravity)",
    )
    d = sub.add_parser("design", help="propose Top-K deep learning baselines for a dataset")
    d.add_argument(
        "--data",
        required=True,
        help="data file (CSV: tabular; .mat/.npz/.npy: hsi; .npz with X, y: images, masks, series)",
    )
    d.add_argument(
        "--plugin",
        default="auto",
        help="auto | tabular | hsi | image_classification (image) | image_segmentation (segmentation) | "
        "image_reconstruction (autoencoder) | timeseries | graph | recsys",
    )
    d.add_argument(
        "--budget", default="5m",
        help="budget of the whole run: wall clock (90s, 10m, 2h) or training epochs (400ep, reproducible)",
    )  # fmt: skip
    d.add_argument("--topk", type=int, default=3, help="number of exported baselines")
    d.add_argument(
        "--device", default="auto",
        help="auto (CUDA GPU, else Apple MPS, else CPU) | cpu | cuda | cuda:1 | mps",
    )  # fmt: skip
    d.add_argument(
        "--amp", default="auto", choices=["auto", "off", "bf16", "fp16"],
        help="mixed precision: auto = bf16 (or fp16) on a CUDA GPU, float32 elsewhere",
    )  # fmt: skip
    d.add_argument(
        "--threads", type=int, default=0,
        help="CPU threads for torch, BLAS and OpenMP (0 = auto: free CPUs; fix it for identical reruns)",
    )  # fmt: skip
    d.add_argument(
        "--per-family", type=int, default=8, help="random configurations per model family"
    )
    d.add_argument(
        "--seed", type=int, default=0, help="random seed (split, sampling, initialisation)"
    )
    d.add_argument(
        "--cv-splits",
        type=int,
        default=0,
        help="rank finalists on this many different splits (0 = plugin default; 1 = off)",
    )
    d.add_argument(
        "--out", default="dlnet_out", help="output folder (leaderboard + one package per baseline)"
    )
    d.add_argument("--no-cache", action="store_true", help="do not reuse finished runs of --out")
    d.add_argument(
        "--meta-store",
        help="JSONL file of past runs: used to order the families, and this run is appended",
    )
    d.add_argument(
        "--zoo", type=int, default=12, help="architectures from the module zoo (presets + random)"
    )
    d.add_argument("--no-zoo-presets", action="store_true", help="random zoo architectures only")
    d.add_argument("--zoo-width", type=int, default=16, help="base width of the zoo presets")
    d.add_argument(
        "--evolve", type=int, default=1, help="generations of mutation/crossover (0 = off)"
    )
    d.add_argument("--evolve-pop", type=int, default=8, help="children per generation")
    d.add_argument(
        "--max-flops", help="drop candidates above this many forward FLOPs per sample, e.g. 50M"
    )
    d.add_argument(
        "--max-vram", help="drop candidates above this estimated training VRAM in MB, e.g. 4000"
    )
    d.add_argument("--no-proxies", action="store_true", help="skip the zero-cost proxy report")
    d.add_argument(
        "--context",
        help='task context in words, e.g. "medical CT, small lesions" or "mobile, real-time"',
    )
    d.add_argument(
        "--goal", default="accuracy", choices=["accuracy", "small", "balanced"],
        help="seed-selection bias (default: follows --objective)",
    )  # fmt: skip
    d.add_argument(
        "--objective", default="accuracy", choices=["accuracy", "params", "flops", "balanced", "pareto"],
        help="accuracy: best validation | params/flops: smallest model within --tolerance of the best | "
        "balanced: MnasNet reward acc*(cost/T)^w | pareto: non-dominated (accuracy, cost) first",
    )  # fmt: skip
    d.add_argument(
        "--cost", default="params", choices=["params", "flops"], help="cost used by balanced/pareto"
    )
    d.add_argument(
        "--tolerance", type=float, default=0.01, help="params/flops: allowed score loss vs the best"
    )
    d.add_argument(
        "--target-cost", help="balanced: target cost T, e.g. 200k params (default: pool median)"
    )
    d.add_argument(
        "--cost-weight", type=float, default=-0.07, help="balanced: exponent w (MnasNet: -0.07)"
    )
    d.add_argument("--max-params", help="drop candidates with more parameters, e.g. 500k")
    d.add_argument(
        "--extra-specs", help="JSON file with a list of zoo specs to add to the candidates"
    )
    d.add_argument(
        "--surrogate-weight", type=float, default=1.0,
        help="weight of the zero-shot ZAP surrogate in the seed ranking (0 = off)",
    )  # fmt: skip
    d.add_argument(
        "--no-builtin-meta", action="store_true",
        help="do not use the meta-dataset shipped with DLNet for the surrogate (image classification)",
    )  # fmt: skip
    d.add_argument(
        "--advisor", choices=sorted(advisor_presets()),
        help="optional AI advisor CLI that ranks seeds and proposes architectures (doc/ai-advisor.md)",
    )  # fmt: skip
    d.add_argument(
        "--advisor-cmd", help='custom advisor command; prompt on stdin or "{prompt_file}"'
    )
    d.add_argument(
        "--advisor-weight", type=float, default=1.0, help="weight of the advisor ranking"
    )
    d.add_argument(
        "--advisor-proposals", type=int, default=3, help="new architectures asked per call"
    )
    d.add_argument("--advisor-timeout", type=float, default=120.0, help="seconds per advisor call")
    d.add_argument(
        "--seeds", type=int, default=0,
        help="keep only this many seed architectures (data-aware ranking); 0 = keep all, ranked",
    )  # fmt: skip
    d.add_argument(
        "--seed-proxies", action="store_true",
        help="let the zero-cost proxy vote (synflow, naswot, snip) rank the seeds too",
    )  # fmt: skip
    d.add_argument(
        "--scheduler", default="sh", choices=["sh", "hyperband"],
        help="sh: one successive-halving bracket (4/12/30 epochs) | hyperband: Li et al. 2018",
    )  # fmt: skip
    d.add_argument(
        "--seed-order", default="data", choices=["data", "random"],
        help="random = control arm that keeps --seeds candidates at random (for measuring the selector)",
    )  # fmt: skip
    d.add_argument(
        "--recipe", default="modern", choices=["modern", "plain"],
        help="training recipe: modern = image augmentation (shifts, flips for colour) + label smoothing 0.1",
    )  # fmt: skip
    d.add_argument(
        "--hparams", default="auto", choices=["auto", "on", "off"],
        help="lr/batch/wd estimation: auto = only for finalists run >= 2 times (noise measurable)",
    )  # fmt: skip
    d.add_argument(
        "--evolve-population",
        type=int,
        default=0,
        help="aging-evolution population (0 = rung 1 size)",
    )
    d.add_argument(
        "--tournament", type=int, default=0, help="tournament size S (0 = population / 4)"
    )
    for p in plugins._registry().values():
        p.add_arguments(d)
    args = ap.parse_args(argv)
    if args.cmd == "zoo":
        from . import zoo

        print(zoo.describe())
        return 0
    if args.cmd == "mcp":
        from .mcp_server import serve

        return serve()
    if args.cmd == "unfold":
        return unfold_run.run(args)
    if args.cmd == "gan":
        return gan_run.run(args)
    return selfcheck() if args.cmd == "selfcheck" else design(args)


if __name__ == "__main__":
    raise SystemExit(main())
