"""Which early signal keeps the truly best candidates when successive halving cuts after rung 1?

Target: validation score after the last recorded epoch (6). Signals at epoch e (1 or 2):
val@e, TSE (sum of epoch-mean training losses, Ru et al. 2021), TSE-E (E=1: the last epoch's
loss), TSE-EMA (mu 0.9), zero-cost proxies, params, prior; and Borda combinations.
Metrics per (dataset, rep): Spearman with the target; whether the best is kept when keeping the
top third; regret@keep = best target - best target among kept.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

d = json.loads(Path(sys.argv[1]).read_text())


def epoch_losses(r):
    L = np.array(r["losses"], float)
    n = len(r["val"])
    k = len(L) // n
    return np.array([L[i * k : (i + 1) * k].mean() for i in range(n)])


def signals(r, e):
    el = epoch_losses(r)
    ema = el[0]
    for x in el[1:e]:
        ema = ema * 0.1 + 0.9 * x
    s = {
        f"val@{e}": r["val"][e - 1],
        f"TSE@{e}": -el[:e].sum(),
        f"TSE-E@{e}": -el[e - 1],
        f"TSE-EMA@{e}": -ema,
        # within-epoch: mean loss over the last half of epoch e's steps (finer TSE-E)
        f"TSE-late@{e}": -np.array(
            r["losses"][: e * (len(r["losses"]) // len(r["val"]))][
                -max(1, len(r["losses"]) // len(r["val"]) // 2) :
            ]
        ).mean(),
    }
    for k in ("synflow", "snip", "naswot", "zico", "grad_norm"):
        v = r["proxies"].get(k)
        s[k] = v if v is not None and np.isfinite(v) else np.nan
    s["params"] = r["proxies"]["params"]
    s["prior"] = r["prior"]
    return s


def ranks(x):
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, -np.inf)
    o = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return o.astype(float)


res = defaultdict(lambda: defaultdict(list))
for key, v in d.items():
    rows = [r for r in v["rows"] if len(r["val"]) >= 6 and np.isfinite(r["val"][-1])]
    if len(rows) < 8:
        continue
    y = np.array([r["val"][-1] for r in rows])
    keep = max(1, int(np.ceil(len(rows) / 3)))
    best = y.max()
    for e in (1, 2):
        S = [signals(r, e) for r in rows]
        names = list(S[0])
        cols = {n: np.array([s[n] for s in S], float) for n in names}
        cols[f"val+TSE@{e}"] = ranks(cols[f"val@{e}"]) + ranks(cols[f"TSE@{e}"])
        cols[f"val+TSE-E@{e}"] = ranks(cols[f"val@{e}"]) + ranks(cols[f"TSE-E@{e}"])
        cols[f"val+TSE-late@{e}"] = ranks(cols[f"val@{e}"]) + ranks(cols[f"TSE-late@{e}"])
        for n, x in cols.items():
            ok = np.isfinite(x)
            rho = spearmanr(x[ok], y[ok]).statistic if ok.sum() > 3 else np.nan
            top = np.argsort(-np.where(ok, x, -np.inf), kind="stable")[:keep]
            res[n]["rho"].append(rho)
            res[n]["kept_best"].append(float(y[top].max() == best))
            res[n]["regret"].append(best - y[top].max())
            res[n]["top1_regret"].append(best - y[top[0]])
    res["_n"]["sets"].append(key)

print("sets:", res.pop("_n")["sets"])
print(f"{'signal':<18}{'spearman':>9}{'best kept':>11}{'regret@1/3':>12}{'top1 regret':>13}")
for n, m in sorted(res.items(), key=lambda kv: -np.nanmean(kv[1]["rho"])):
    print(
        f"{n:<18}{np.nanmean(m['rho']):>+9.3f}{np.mean(m['kept_best']):>11.2f}{np.mean(m['regret']):>12.4f}{np.mean(m['top1_regret']):>13.4f}"
    )
