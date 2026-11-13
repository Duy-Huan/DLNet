"""Does ranking rungs by the training speed estimate (TSE) give better designs? (v1.5, agent.md 16.7)

    python scripts/bench_rungs.py --data a.npz b.npz --budget 240ep --reps 2 --out bench_rungs

Same data, same seed pool, same budget in EPOCHS (reproducible on any machine), one design run per
arm and repetition; only the successive-halving schedule and the rung score differ:
  val        --rungs 4,12,30   --rung-score val       DLNet <= 1.4
  val+tse    --rungs 4,12,30   --rung-score val+tse   Borda of validation and TSE
  screen     --rungs 1,4,12,30 --rung-score tse       a 1-epoch screening rung ranked by TSE
  screen+    --rungs 1,4,12,30 --rung-score val+tse
Reported: validation (mean over the repeated runs of stage 2) and the single test score of top-1,
and how many candidates were trained at all. Writes <out>/results.json and a markdown table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ARMS = {
    "val": ["--rungs", "4,12,30", "--rung-score", "val"],
    "val+tse": ["--rungs", "4,12,30", "--rung-score", "val+tse"],
    "screen": ["--rungs", "1,4,12,30", "--rung-score", "tse"],
    "screen+": ["--rungs", "1,4,12,30", "--rung-score", "val+tse"],
}


def run(data, arm, rep, a, out):
    d = out / f"{Path(data).stem}_{arm}_r{rep}"
    if (d / "leaderboard.json").exists():  # resumable
        b = json.loads((d / "leaderboard.json").read_text())
        return summary(b, None)
    args = [sys.executable, "-m", "dlnet", "design", "--data", data, "--budget", a.budget, "--topk", "1",
            "--seed", str(rep), "--zoo", str(a.zoo), "--evolve", "0", "--out", str(d), "--no-cache",
            "--device", "cpu", "--threads", str(a.threads), *ARMS[arm]]  # fmt: skip
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, text=True, check=False)
    if p.returncode:
        return {"error": p.stderr[-400:]}
    return summary(json.loads((d / "leaderboard.json").read_text()), round(time.time() - t0, 1))


def summary(b, seconds):
    top = b["top_k"][0]
    cov = b.get("search_coverage", {})
    return {"val": top["val_mean"], "test": top["test"]["score"], "name": top["name"], "seconds": seconds,
            "epochs": top.get("epochs"), "trained": cov.get("trained") or cov.get("rung1"),
            "reference_best_test": max(r["test"]["score"] for r in b["reference"])}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--budget", default="240ep")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--zoo", type=int, default=40)
    ap.add_argument(
        "--threads",
        type=int,
        default=4,
        help="fixed: identical reruns, no slowdown when the machine is busy",
    )
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="bench_rungs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for data in a.data:
        for rep in range(a.reps):
            for arm in a.arms.split(","):
                r = run(data, arm, rep, a, out)
                res.setdefault(Path(data).stem, {}).setdefault(arm, []).append(r)
                print(Path(data).stem, arm, rep, r, flush=True)
                (out / "results.json").write_text(json.dumps(res, indent=2))
    print("\n| data | arm | val | test | trained | top-1 |")
    print("|---|---|---|---|---|---|")
    for ds, arms in res.items():
        for arm, rs in arms.items():
            ok = [r for r in rs if "val" in r]
            if ok:
                v, t = (sum(r[k] for r in ok) / len(ok) for k in ("val", "test"))
                print(
                    f"| {ds} | {arm} | {v:.4f} | {t:.4f} | {[r['trained'] for r in ok]} | {', '.join(r['name'] for r in ok)} |"
                )


if __name__ == "__main__":
    main()
