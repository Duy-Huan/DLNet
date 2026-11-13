"""Significance of a difference between two baselines measured on the same runs.

`design` trains every finalist and every paper network with run i = seed i on split i (search.
eval_runs), so the scores of two networks pair up run by run and a paired test applies. With the
usual 3 runs a test has little power: a large p means "not shown", never "equal".

Paired t-test (Student 1908; scipy.stats.ttest_rel) on the per-run differences; Holm (1979)
step-down correction when one baseline is compared with many (10-20 papers): it controls the
family-wise error rate without assuming independence, and is uniformly more powerful than Bonferroni.
"""

from __future__ import annotations

import math

import numpy as np


def paired(a, b) -> dict:
    """a - b on paired runs: {n, diff (mean of a - b), t, p (two-sided)}; p is None when it cannot
    be computed (fewer than 2 pairs, or every difference identical)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0, "diff": None, "t": None, "p": None}
    d = a[:n] - b[:n]
    out = {"n": int(n), "diff": float(d.mean()), "t": None, "p": None}
    if n < 2:
        return out
    sd = float(d.std(ddof=1))
    # rounding leaves ~1e-17 of spread on equal differences: that is no variance, not certainty
    if not math.isfinite(sd) or sd <= 1e-9 * max(1.0, abs(float(d.mean()))):
        # all differences equal: zero difference is "no evidence", a constant nonzero one has no
        # variance to test against; report the difference, not a p-value
        return out
    from scipy import stats

    t = float(d.mean() / (sd / math.sqrt(n)))
    out["t"], out["p"] = t, float(2 * stats.t.sf(abs(t), n - 1))
    return out


def holm(pvalues) -> list:
    """Holm-adjusted p-values in the input order; None stays None (not counted in the family)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    out = list(pvalues)
    running = 0.0
    for k, i in enumerate(sorted(idx, key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (m - k) * pvalues[i]))
        out[i] = running
    return out


def verdict(diff, p_adj, alpha: float = 0.05) -> str:
    """Words for one comparison of a paper network with the top-1 (diff = paper - top-1)."""
    if diff is None:
        return "-"
    if p_adj is None:
        return "not tested"
    if p_adj >= alpha:
        return "no significant difference"
    return "better" if diff > 0 else "worse"
