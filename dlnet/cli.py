"""Command line: dlnet design --data <file> [plugin options] --budget 5m --topk 3"""

from __future__ import annotations

import argparse
import copy
import re
import time

from . import __version__


def parse_budget(s: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", s.strip())
    if not m:
        raise SystemExit(f"bad --budget {s!r} (use e.g. 90s, 5m, 2h)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def design(args) -> int:
    from pathlib import Path

    import torch

    from . import export, hparams, plugins, proxies, search

    t0 = time.time()
    budget = parse_budget(args.budget)
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

    print("[reference] classical baselines on the same split(s)")
    ref = export.mean_reference([plugin.reference(f) for f in folds])
    for r in ref:
        print(f"[reference] {r['name']:<26} val={r['va']['score']:.4f}")

    cands = plugin.generate(prof, args.per_family, args.seed)
    if not cands:
        raise SystemExit("no valid candidate was generated (check --per-family and the budget)")
    fams = sorted({c["family"] for c in cands})
    print(f"[generate] {len(cands)} valid candidates from families {fams}")

    # Zero-cost proxies of the untrained candidates: measured against real training below,
    # never used to prune (agent.md 16.1). At most 10% of the budget.
    zc = {}
    for c in cands:
        if args.no_proxies or time.time() > t0 + 0.1 * budget:
            break
        torch.manual_seed(args.seed)
        zc[id(c)] = proxies.score(
            plugin.build_model(c, data["in_dim"], data["out_dim"]), data, seed=args.seed
        )
    trace = {}
    results = search.run_search(
        data, cands, search_deadline, args.topk, plugin.build_model, folds=folds,
        cache=cache, trace=trace,
    )  # fmt: skip
    print("[hparams] estimating lr / batch size / weight decay for the finalists")
    results = hparams.refine(
        results, data, plugin.build_model, deadline, log=print, folds=folds, cache=cache
    )
    results = search.assign_groups(results)
    chosen = search.select_topk(results, args.topk)
    extras = {}
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
    v = export.export(args.out, plugin, data, prof, chosen, ref, args.seed, extras=extras)
    print(f"[done] {time.time() - t0:.0f}s  -> {args.out}/leaderboard.md")
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
        rc = main([*argv, "--per-family", "1", "--out", str(out)])
        board = json.loads((out / "leaderboard.json").read_text())
        pkg = next(out.glob("top1_*"))
        ok = (
            rc == 0
            and bool(board["top_k"])
            and all((pkg / f).exists() for f in ("model.py", "train.py"))
        )
    print(f"selfcheck {'OK' if ok else 'FAILED'} (dlnet {__version__})")
    return 0 if ok else 1


def main(argv=None) -> int:
    from . import plugins, unfold_run

    ap = argparse.ArgumentParser(prog="dlnet")
    ap.add_argument("--version", action="version", version=f"dlnet {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck", help="verify the installation end to end (about 20 seconds)")
    unfold_run.add_parser(sub)
    d = sub.add_parser("design", help="propose Top-K deep learning baselines for a dataset")
    d.add_argument(
        "--data", required=True, help="data file (CSV for tabular; .mat/.npz/.npy for hsi)"
    )
    d.add_argument("--plugin", default="auto", help="auto | tabular | hsi")
    d.add_argument("--budget", default="5m")
    d.add_argument("--topk", type=int, default=3)
    d.add_argument("--per-family", type=int, default=8)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument(
        "--cv-splits",
        type=int,
        default=0,
        help="rank finalists on this many different splits (0 = plugin default; 1 = off)",
    )
    d.add_argument("--out", default="dlnet_out")
    d.add_argument("--no-cache", action="store_true", help="do not reuse finished runs of --out")
    d.add_argument("--no-proxies", action="store_true", help="skip the zero-cost proxy report")
    for p in plugins._registry().values():
        p.add_arguments(d)
    args = ap.parse_args(argv)
    if args.cmd == "unfold":
        return unfold_run.run(args)
    return selfcheck() if args.cmd == "selfcheck" else design(args)


if __name__ == "__main__":
    raise SystemExit(main())
