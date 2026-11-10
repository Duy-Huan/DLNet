"""Command line: dlnet design --data <file> [plugin options] --budget 5m --topk 3"""

from __future__ import annotations

import argparse
import re
import time

from . import __version__


def parse_budget(s: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", s.strip())
    if not m:
        raise SystemExit(f"bad --budget {s!r} (use e.g. 90s, 5m, 2h)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def design(args) -> int:
    from . import export, plugins, search

    t0 = time.time()
    deadline = t0 + parse_budget(args.budget)
    plugin = plugins.route(args.data) if args.plugin == "auto" else plugins.get_plugin(args.plugin)
    print(f"[plugin] {plugin.name}  data={args.data}")

    data = plugin.prepare(args)
    print(
        f"[load] task={data['task']} features={data['in_dim']} classes={data['out_dim']} seed={args.seed}"
    )
    prof = plugin.profile(data)
    shown = {k: v for k, v in prof.items() if k not in ("task",)}
    print(f"[profile] {shown}")

    print("[reference] classical baselines on the same split")
    ref = plugin.reference(data)
    for r in ref:
        print(f"[reference] {r['name']:<26} val={r['va']['score']:.4f}")

    cands = plugin.generate(prof, args.per_family, args.seed)
    if not cands:
        raise SystemExit("no valid candidate was generated (check --per-family and the budget)")
    fams = sorted({c["family"] for c in cands})
    print(f"[generate] {len(cands)} valid candidates from families {fams}")

    results = search.run_search(data, cands, deadline, args.topk, plugin.build_model)
    chosen = search.select_topk(results, args.topk)
    v = export.export(args.out, plugin, data, prof, chosen, ref, args.seed)
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
    from . import plugins

    ap = argparse.ArgumentParser(prog="dlnet")
    ap.add_argument("--version", action="version", version=f"dlnet {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck", help="verify the installation end to end (about 20 seconds)")
    d = sub.add_parser("design", help="propose Top-K deep learning baselines for a dataset")
    d.add_argument(
        "--data", required=True, help="data file (CSV for tabular; .mat/.npz/.npy for hsi)"
    )
    d.add_argument("--plugin", default="auto", help="auto | tabular | hsi")
    d.add_argument("--budget", default="5m")
    d.add_argument("--topk", type=int, default=3)
    d.add_argument("--per-family", type=int, default=8)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", default="dlnet_out")
    for p in plugins._registry().values():
        p.add_arguments(d)
    args = ap.parse_args(argv)
    return selfcheck() if args.cmd == "selfcheck" else design(args)


if __name__ == "__main__":
    raise SystemExit(main())
