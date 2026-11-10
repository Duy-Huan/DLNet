"""Does data-aware seed selection help? Same budget, same data, four arms (agent.md 16.5).

    python scripts/bench_seeds.py --data a.npz b.npz --budget 150s --n 10 --reps 2 --out bench_seeds

Arms (all with the same --budget, --zoo and evolution settings):
  all        every candidate, ranked by the seed selector but none removed (--seeds 0)
  random     --seeds N chosen at random (--seed-order random): the control arm
  data       --seeds N chosen by the data card + prior (dlnet/seeds.py)
  data_hb    as `data`, with Hyperband instead of a single successive-halving bracket

Reported per arm: validation and test score of top-1, wall time. From the `all` arm: Spearman
correlation with the rung-1 validation score of every proxy, of the proxy vote and of the
seed prior (the signal the selector uses), next to the parameter-count baseline.
Writes <out>/results.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ARMS = {
    "all": ["--seeds", "0"],
    "random": ["--seeds", "{n}", "--seed-order", "random"],
    "data": ["--seeds", "{n}"],
    "data_hb": ["--seeds", "{n}", "--scheduler", "hyperband"],
}


def run(data, arm, n, rep, a, out):
    d = out / f"{Path(data).stem}_{arm}_r{rep}"
    args = [sys.executable, "-m", "dlnet", "design", "--data", data, "--budget", a.budget,
            "--topk", "1", "--seed", str(rep), "--zoo", str(a.zoo), "--evolve", str(a.evolve),
            "--out", str(d), "--no-cache", *[x.format(n=n) for x in ARMS[arm]]]  # fmt: skip
    if a.context:
        args += ["--context", a.context]
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, text=True, check=False)
    if p.returncode:
        return {"error": p.stderr[-400:]}
    b = json.loads((d / "leaderboard.json").read_text())
    top = b["top_k"][0]
    return {
        "val": top["val_mean"],
        "test": top["test"]["score"],
        "family": top["name"],
        "origin": top["origin"],
        "seconds": round(time.time() - t0, 1),
        "reference_best_test": max(r["test"]["score"] for r in b["reference"]),
        "proxies": {
            k: v["spearman"] for k, v in b.get("zero_cost_proxies", {}).get("proxies", {}).items()
        },
        "rules": b.get("seed_selection", {}).get("rules_fired", []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--budget", default="150s")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--zoo", type=int, default=40)
    ap.add_argument("--evolve", type=int, default=1)
    ap.add_argument("--context", default="")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="bench_seeds")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for data in a.data:
        for arm in a.arms.split(","):
            for rep in range(a.reps):
                r = run(data, arm, a.n, rep, a, out)
                res.setdefault(Path(data).stem, {}).setdefault(arm, []).append(r)
                print(
                    Path(data).stem,
                    arm,
                    rep,
                    {k: r.get(k) for k in ("val", "test", "origin", "seconds", "error")},
                    flush=True,
                )
                (out / "results.json").write_text(json.dumps(res, indent=2))
    print("\n| data | arm | val (mean of reps) | test (mean of reps) | top-1 origins |")
    print("|---|---|---|---|---|")
    for ds, arms in res.items():
        for arm, rs in arms.items():
            ok = [r for r in rs if "val" in r]
            if ok:
                v = sum(r["val"] for r in ok) / len(ok)
                t = sum(r["test"] for r in ok) / len(ok)
                print(
                    f"| {ds} | {arm} | {v:.4f} | {t:.4f} | {', '.join(r['origin'] or '-' for r in ok)} |"
                )


if __name__ == "__main__":
    main()
