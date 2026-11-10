"""Does the seed prior (or any zero-cost proxy) predict real training results? (agent.md 16.5)

    python scripts/bench_prior.py --data a.npz b.npz --n 20 --epochs 4 --reps 2

For each dataset: the same candidate pool as `dlnet design` (families + 40 zoo architectures),
`n` candidates drawn at random (seeded), each scored by the seed prior, every zero-cost proxy,
the proxy vote and the parameter count, then trained for `epochs` epochs (seed 0) and evaluated
on validation. Reports the Spearman correlation of each score with the validation score, and the
top-3 overlap: how many of the 3 best-trained candidates are among the 3 best by that score.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import types
from pathlib import Path

import numpy as np
import torch

from dlnet import datacard, plugins, proxies, search, seeds


def study(path, n, epochs, rep, context=""):
    plugin = plugins.get_plugin("image")
    args = types.SimpleNamespace(data=path, seed=rep, x_key=None, y_key=None, task="auto")
    data = plugin.prepare(args)
    prof = plugin.profile(data)
    card = datacard.image_card(data, seed=rep)
    pool = plugin.generate(prof, 8, rep) + plugin.generate_zoo(data, prof, 40, rep, True, 16)
    rng = np.random.default_rng(100 + rep)
    cands = [pool[i] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)]
    prior, why = seeds.prior_scores(cands, card, context)
    scores, vals = [], []
    for c, p in zip(cands, prior, strict=True):
        torch.manual_seed(rep)
        s = proxies.score(plugin.build_model(c, data["in_dim"], data["out_dim"]), data, seed=rep)
        s["seed_prior"] = p
        scores.append(s)
        t = search._new_trainer(c, data, 0, plugin.build_model)
        t.fit(epochs)
        vals.append(t.evaluate("va")["score"])
    for s, v in zip(scores, seeds.proxy_vote(scores), strict=True):
        s["vote"] = v
    best3 = set(np.argsort(vals)[-3:])
    out = {}
    for k in (*proxies.PROXIES, "vote", "seed_prior"):
        xs = [s.get(k) for s in scores]
        ok = [i for i, x in enumerate(xs) if x is not None and math.isfinite(x)]
        rho = proxies.spearman([xs[i] for i in ok], [vals[i] for i in ok]) if len(ok) >= 3 else None
        top = set(sorted(ok, key=lambda i: xs[i])[-3:])
        out[k] = {"spearman": None if rho is None else round(rho, 3), "n": len(ok),
                  "top3_overlap": len(top & best3)}  # fmt: skip
    return {"scores": out, "val_range": [round(min(vals), 4), round(max(vals), 4)], "rules": why,
            "families": [c["family"] for c in cands]}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="bench_prior.json")
    a = ap.parse_args()
    res = {}
    for path in a.data:
        for rep in range(a.reps):
            t0 = time.time()
            r = study(path, a.n, a.epochs, rep)
            res.setdefault(Path(path).stem, []).append(r)
            sp = {k: v["spearman"] for k, v in r["scores"].items()}
            print(Path(path).stem, rep, f"{time.time() - t0:.0f}s", sp, flush=True)
            Path(a.out).write_text(json.dumps(res, indent=2))
    keys = (*proxies.PROXIES, "vote", "seed_prior")
    print("\n| data | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for ds, rs in res.items():
        cells = []
        for k in keys:
            v = [r["scores"][k]["spearman"] for r in rs if r["scores"][k]["spearman"] is not None]
            o = [r["scores"][k]["top3_overlap"] for r in rs]
            cells.append(
                ("-" if not v else " / ".join(f"{x:+.2f}" for x in v))
                + f" ({sum(o)}/{3 * len(rs)})"
            )
        print(f"| {ds} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
