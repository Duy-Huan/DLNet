"""`dlnet plan`: what a `design` run with the same options will do, before spending the budget.

    dlnet plan --data cube.mat --gt gt.mat --budget 10m --papers papers/
    dlnet plan ... --json                  # the same as JSON (the web UI reads this)

Steps (seconds, nothing is trained to convergence): load and split the data as `design` does,
fit the classical reference baselines (the score a deep baseline has to beat), generate the
candidates with the same options and cost limits, time ONE training step of a sample of them and
of every paper network (Trainer.probe_epoch_seconds: the same measurement `design` uses to keep
to a time budget), then estimate:

- how many candidates reach each rung of successive halving within the budget,
- the wall-clock time (for an epoch budget such as 400ep: epochs x measured seconds per epoch),
- the time of the paper comparison (`--papers`: the top-1 protocol, n runs x last-rung epochs,
  plus the authors' recipe with --paper-recipe), which comes on top of --budget,
- warnings: paper cards that will be skipped or fail, a budget too small for the Top-K.

It is an estimate. Timings come from one step; the number of candidates follows the budget rules
of `design` (search 80% of the budget, rungs equal shares, keep 1/3, at least 2 x Top-K) but not
their every detail (evolution, early stops, failures). Scores are not predicted: the only score
shown is the measured classical reference, and what each paper reports on its own split.
"""

from __future__ import annotations

import math
import time

import numpy as np

SAMPLE = 8  # candidates timed


def _probe(cfg, data, build_fn) -> float | None:
    from . import search

    try:
        tr = search._new_trainer(cfg, data, 0, build_fn)
        return float(tr.probe_epoch_seconds())
    except Exception:  # noqa: BLE001 - an untimed candidate is reported, not fatal
        return None


def estimate_rungs(n_cands, sec_per_epoch, search_seconds, rungs, topk, eta=3) -> list[dict]:
    """Candidates trained to each rung when every rung gets an equal share of the search time."""
    out, alive = [], n_cands
    share, prev = search_seconds / len(rungs), 0
    for r, ep in enumerate(rungs):
        step = (ep - prev) * sec_per_epoch
        fit = int(share // step) if step > 0 else alive
        trained = min(alive, fit)
        out.append({"rung": r + 1, "epochs": ep, "candidates": trained})
        if trained == 0:
            break
        alive = max(math.ceil(trained / eta), min(trained, topk * 2))
        prev = ep
    return out


def _papers(cards, plugin, data, build_fn, runs, epochs, recipe, recipe_cap):
    from pathlib import Path

    from . import paper

    kind, dims, head = plugin.zoo_input(data) or (None, (), None)
    paths = sorted(
        p for c in cards for p in ([Path(c)] if Path(c).is_file() else Path(c).glob("*.y*ml"))
    )
    rows = []
    for path in paths:
        # status: English for the terminal and AI clients; code + detail: for the UI to translate
        row = {"card": str(path), "status": "ok", "code": "ok", "detail": {}}
        try:
            card = paper.load(path)
            row["paper"] = card["paper"]["name"]
            row["reported"] = card["paper"].get("reported") or []
            if "architecture" not in card:
                row["code"] = "level_a"
                row["status"] = (
                    "skipped: level A card (the authors' class), build it with `dlnet paper build`"
                )
            elif card["input"]["kind"] != kind or tuple(card["input"]["dims"]) != tuple(dims):
                row["code"] = "input"
                row["detail"] = {"card": f"{card['input']['kind']} {list(card['input']['dims'])}",
                                 "data": f"{kind} {list(dims)}"}  # fmt: skip
                row["status"] = (f"skipped: input {card['input']['kind']} {list(card['input']['dims'])}"
                                 f" is not this data's {kind} {list(dims)}")  # fmt: skip
            else:
                from . import zoo

                arch = {
                    **paper.to_arch(card, data["out_dim"]),
                    "head": head,
                    "lr": 1e-3,
                    "wd": 0.05,
                    "batch": 64,
                }
                row["params"] = zoo.count_params(zoo.build_model(arch))
                sec = _probe(arch, data, build_fn)
                row["sec_per_epoch"] = sec
                ep_recipe = 0
                if recipe and card.get("training"):
                    ep_recipe = min(int(card["training"].get("epochs", recipe_cap)), recipe_cap)
                row["epochs"] = runs * epochs + ep_recipe
                row["seconds"] = None if sec is None else sec * row["epochs"]
                if sec is None:
                    row["code"], row["detail"] = (
                        "fail",
                        {"error": "one training step raised an error"},
                    )
                    row["status"] = "fails: one training step raised an error"
        except Exception as e:  # noqa: BLE001
            row["code"], row["detail"] = "fail", {"error": f"{type(e).__name__}: {e}"}
            row["status"] = f"fails: {type(e).__name__}: {e}"
        rows.append(row)
    return rows


def _warn(code: str, text: str, **params) -> dict:
    """A warning: `text` in English (terminal, AI clients), `code` + `params` (the UI translates)."""
    return {"code": code, "text": text, "params": params}


def make(args) -> dict:
    from . import cli, plugins
    from .cli import parse_budget_mode, parse_rungs

    t_start = time.time()
    budget, mode = parse_budget_mode(args.budget)
    plugin = plugins.route(args.data) if args.plugin == "auto" else plugins.get_plugin(args.plugin)
    data = plugin.prepare(args)
    prof = plugin.profile(data)
    ref = [
        {"name": r["name"], "val": r["va"]["score"], "test": r["te"]["score"]}
        for r in plugin.reference(data)
    ]
    cands = plugin.generate(prof, args.per_family, args.seed)
    cands += plugin.generate_zoo(
        data, prof, args.zoo, args.seed, not args.no_zoo_presets, args.zoo_width
    )
    n_generated = len(cands)
    cands = cli._apply_cost_model(plugin, data, cands, args)
    if args.seeds:
        cands = cands[: args.seeds]
    # time a sample spread over the parameter counts (small, typical and large networks)
    order = sorted(cands, key=lambda c: c.get("params") or 0)
    pick = [
        order[int(i)]
        for i in np.unique(np.linspace(0, len(order) - 1, min(SAMPLE, len(order))).round())
    ]
    timed = [(c, _probe(c, data, plugin.build_model)) for c in pick]
    secs = [s for _, s in timed if s]
    sec_epoch = float(np.median(secs)) if secs else None
    rungs = parse_rungs(args.rungs) if args.scheduler != "hyperband" else (4, 12, 36)
    runs = 3  # finalists and paper networks: search.run_search(seeds=3)
    plan = {
        "data": args.data,
        "plugin": plugin.name,
        "task": data["task"],
        "metric": data["primary"],
        "shape": {
            "n_train": prof.get("n_train"),
            "in_dim": data["in_dim"],
            "out_dim": data["out_dim"],
        },
        "budget": {
            "text": args.budget,
            "mode": "epochs" if mode == "work" else "seconds",
            "value": budget,
        },
        "candidates": {
            "generated": n_generated,
            "valid": len(cands),
            "families": sorted({c["family"] for c in cands}),
        },
        "timing": {
            "timed": len(pick),
            "failed": sum(s is None for _, s in timed),
            "sec_per_epoch_median": sec_epoch,
            "sec_per_epoch_range": [min(secs), max(secs)] if secs else None,
        },
        "reference": sorted(ref, key=lambda r: -r["val"]),
        "warnings": [],
    }
    if sec_epoch:
        if mode == "work":
            search_epochs = 0.8 * budget
            plan["rungs"] = estimate_rungs(len(cands), 1.0, search_epochs, rungs, args.topk)
            plan["seconds_design"] = budget * sec_epoch
        else:
            plan["rungs"] = estimate_rungs(len(cands), sec_epoch, 0.8 * budget, rungs, args.topk)
            plan["seconds_design"] = budget
        last = plan["rungs"][-1]
        if plan["rungs"][0]["candidates"] < 2 * args.topk:
            n1, need = plan["rungs"][0]["candidates"], 2 * args.topk
            plan["warnings"].append(_warn(
                "few_candidates", f"the budget trains only {n1} candidate(s) in rung 1 (Top-{args.topk} "
                f"needs {need}); raise --budget or lower --zoo / --per-family", n=n1, topk=args.topk, need=need))  # fmt: skip
        top_epochs = last["epochs"]
    else:
        plan["warnings"].append(
            _warn("untimed", "no candidate could be timed: design would fail on this data")
        )
        top_epochs = rungs[-1]
        plan["seconds_design"] = None
    if getattr(args, "papers", None):
        rows = _papers(args.papers, plugin, data, plugin.build_model, runs, top_epochs,
                       args.paper_recipe, args.paper_epochs)  # fmt: skip
        plan["papers"] = rows
        plan["seconds_papers"] = sum(r.get("seconds") or 0 for r in rows)
        bad = [r for r in rows if r["code"] != "ok"]
        if bad:
            plan["warnings"].append(_warn(
                "papers_skipped", f"{len(bad)} of {len(rows)} paper card(s) will be skipped or fail (see papers)",
                n=len(bad), total=len(rows)))  # fmt: skip
    plan["seconds_total"] = (plan.get("seconds_design") or 0) + plan.get("seconds_papers", 0)
    plan["plan_seconds"] = round(time.time() - t_start, 1)
    return plan


def _dur(s) -> str:
    if s is None:
        return "?"
    s = float(s)
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


def text(p: dict) -> str:
    out = [
        f"Plan for {p['data']} ({p['plugin']}, {p['task']}, metric {p['metric']}), budget {p['budget']['text']}"
    ]
    sh = p["shape"]
    out.append(
        f"  data: {sh['n_train']} training samples, {sh['in_dim']} inputs, {sh['out_dim']} outputs"
    )
    c = p["candidates"]
    out.append(
        f"  candidates: {c['valid']} valid of {c['generated']} generated, {len(c['families'])} families"
    )
    t = p["timing"]
    if t["sec_per_epoch_median"]:
        lo, hi = t["sec_per_epoch_range"]
        out.append(
            f"  one epoch: {t['sec_per_epoch_median']:.2f} s median ({lo:.2f}-{hi:.2f} s over {t['timed']} timed)"
        )
    for r in p.get("rungs", []):
        out.append(
            f"  rung {r['rung']} ({r['epochs']} epochs): about {r['candidates']} candidate(s)"
        )
    out.append(f"  design: about {_dur(p['seconds_design'])}")
    if p["reference"]:
        b = p["reference"][0]
        out.append(
            f"  to beat: {b['name']} val {b['val']:.4f}, test {b['test']:.4f} (classical, measured now)"
        )
    if "papers" in p:
        out.append(
            f"  papers: {len(p['papers'])} card(s), about {_dur(p['seconds_papers'])} after the search"
        )
        for r in p["papers"]:
            rep = "; ".join(
                f"{x.get('dataset', '?')} {x['metric']} {x['value']}"
                for x in r.get("reported", [])
                if "metric" in x
            )
            what = (f"{r.get('params', 0):,} params, about {_dur(r.get('seconds'))}" if r["code"] == "ok"
                    else r["status"])  # fmt: skip
            out.append(
                f"    - {r.get('paper', r['card'])}: {what}"
                + (f" (paper reports {rep})" if rep else "")
            )
    out.append(f"  total: about {_dur(p['seconds_total'])}")
    for w in p["warnings"]:
        out.append(f"  warning: {w['text']}")
    out.append("  (an estimate from one timed step per network; scores are not predicted)")
    return "\n".join(out)


def run(args) -> int:
    from . import core, threads
    from .jsonsafe import dumps

    core.DEVICE = str(core.pick_device(args.device))
    with threads.limit(threads.resolve(args.threads)):
        p = make(args)
    if getattr(args, "save", None):
        from pathlib import Path

        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(dumps(p, indent=2))
    print(dumps(p, indent=2) if args.json else text(p))
    return 0
