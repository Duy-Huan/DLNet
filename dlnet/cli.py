"""Command line: dlnet design --data data.csv --target label --budget 5m --topk 3"""

from __future__ import annotations

import argparse
import re
import time


def parse_budget(s: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", s.strip())
    if not m:
        raise SystemExit(f"bad --budget {s!r} (use e.g. 90s, 5m, 2h)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def design(args) -> int:
    from . import core, export, generate, profile, reference, search

    t0 = time.time()
    deadline = t0 + parse_budget(args.budget)
    print(f"[load] {args.data}")
    X, y, _ = core.load_csv(args.data, args.target)
    task = core.detect_task(y) if args.task == "auto" else args.task
    data = core.prepare(X, y, task, seed=args.seed)
    print(f"[load] task={task} n={len(X)} features={X.shape[1]} split=60/20/20 seed={args.seed}")

    prof = profile.profile(data)
    print(
        f"[profile] intrinsic_dim={prof['intrinsic_dim']} alpha_hat={prof['scaling_exponent_hat']} "
        f"pca95={prof['pca_dims_for_95pct_energy']} max_params={prof['max_params']:,} "
        f"metric={prof['primary_metric']}"
    )

    print("[reference] classical baselines on the same split")
    ref = reference.run_reference(data)
    for r in ref:
        print(f"[reference] {r['name']:<24} val={r['va']['score']:.4f}")

    cands = generate.generate(prof, n_per_family=args.per_family, seed=args.seed)
    fams = sorted({c["family"] for c in cands})
    print(f"[generate] {len(cands)} valid candidates from families {fams}")

    results = search.run_search(data, cands, deadline, args.topk)
    chosen = search.select_topk(results, args.topk)
    v = export.export(args.out, data, prof, chosen, ref, args.seed)
    print(f"[done] {time.time() - t0:.0f}s  -> {args.out}/leaderboard.md")
    print(f"[verdict] beats_reference={v['beats_reference']} val_margin={v['val_margin']} test_margin={v['test_margin']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dlnet")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design", help="propose Top-K deep learning baselines for a dataset")
    d.add_argument("--data", required=True, help="CSV file")
    d.add_argument("--target", required=True, help="target column")
    d.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    d.add_argument("--budget", default="5m")
    d.add_argument("--topk", type=int, default=3)
    d.add_argument("--per-family", type=int, default=8)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", default="dlnet_out")
    args = ap.parse_args(argv)
    return design(args)


if __name__ == "__main__":
    raise SystemExit(main())
