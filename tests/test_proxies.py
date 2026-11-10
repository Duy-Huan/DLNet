"""Zero-cost proxies: closed-form answers (agent.md 16.1 item 3), not only shape checks."""

import math

import pytest

pytest.importorskip("torch")

import numpy as np
import torch
from torch import nn

from dlnet import core, proxies
from dlnet.library import build_model


def test_naswot_matches_closed_form():
    """Identity layer + ReLU, inputs with codes [1,0] and [0,1]: K = 2 I, log|det K| = 2 ln 2."""
    lin = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(2))
    m = nn.Sequential(lin, nn.ReLU())
    X = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    assert proxies.naswot(m, X) == pytest.approx(2 * math.log(2))
    # identical codes -> singular kernel -> -inf: the proxy says "cannot tell the inputs apart"
    assert proxies.naswot(m, torch.tensor([[1.0, -1.0], [2.0, -3.0]])) == -math.inf
    assert proxies.naswot(nn.Sequential(nn.Linear(2, 2), nn.GELU()), X) is None  # no ReLU


def test_synflow_matches_closed_form():
    """y = w2 * w1 * x (scalars, no bias): R = |w1 w2|, each layer contributes |w1 w2|."""
    m = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        m[0].weight.fill_(-3.0)
        m[1].weight.fill_(0.5)
    assert proxies.synflow(m, (1,)) == pytest.approx(math.log(2 * 1.5))
    assert float(m[0].weight) == -3.0  # the model itself is untouched (signs not flipped)


def test_grad_norm_matches_autograd():
    torch.manual_seed(0)
    m = nn.Linear(3, 2)
    X, y = torch.randn(8, 3), torch.randint(0, 2, (8,))
    nn.functional.cross_entropy(m(X), y).backward()
    want = math.sqrt(sum(float((p.grad**2).sum()) for p in m.parameters()))
    assert proxies.grad_norm(m, X, y, "classification") == pytest.approx(want, rel=1e-6)


def test_spearman_closed_form_and_ties():
    assert proxies.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert proxies.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # textbook formula 1 - 6 sum d^2 / (n (n^2 - 1)) with d = (0, 1, -1, 0, 0): 1 - 12/120
    assert proxies.spearman([1, 2, 3, 4, 5], [1, 3, 2, 4, 5]) == pytest.approx(0.9)
    assert proxies.spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)  # ties
    assert proxies.spearman([1, 1, 1], [1, 2, 3]) is None and proxies.spearman([1], [1]) is None


def test_score_and_report_never_prune():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)).astype("float32")
    d = core.prepare(X, (X[:, 0] > 0).astype(int), "classification", seed=0)
    cfgs = [
        {"family": f, "width": w, "depth": 2, "dropout": 0.0, "rank": 4}
        for f in ("mlp", "resmlp")
        for w in (16, 32)
    ]
    scores = [proxies.score(build_model(c, 4, 2), d) for c in cfgs]
    assert all(s["params"] > 0 and math.isfinite(s["synflow"]) for s in scores)
    assert all(s["naswot"] is None for s in scores)  # GELU networks: NASWOT is undefined
    rep = proxies.correlation_report(scores, [0.9, 0.8, 0.7, 0.6], epochs=4)
    assert rep["used_for_pruning"] is False
    assert rep["proxies"]["params"]["n"] == 4 and rep["proxies"]["naswot"]["n"] == 0
    assert -1 <= rep["proxies"]["synflow"]["spearman"] <= 1


def test_run_cache_resumes_identical_runs_only(tmp_path):
    import time

    from dlnet import search

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)).astype("float32")
    y = (X[:, 0] > 0).astype(int)
    d = core.prepare(X, y, "classification", seed=0)
    cfg = {"family": "mlp", "width": 16, "depth": 2, "dropout": 0.0, "lr": 1e-2, "wd": 1e-4,
           "batch": 32}  # fmt: skip
    path = tmp_path / "c.json"
    far = time.time() + 60
    a = search.eval_runs(cfg, [d], build_model, 2, 2, far, cache=search.RunCache(path))
    c2 = search.RunCache(path)  # a new process: read from disk
    calls = []
    orig = search._new_trainer
    search._new_trainer = lambda *k: calls.append(1) or orig(*k)
    try:
        b = search.eval_runs(cfg, [d], build_model, 2, 2, far, cache=c2)
        assert not calls and c2.hits == 2 and a[:2] == b[:2]  # same numbers, nothing trained
        search.eval_runs({**cfg, "width": 32}, [d], build_model, 2, 1, far, cache=c2)
        search.eval_runs(cfg, [d], build_model, 3, 1, far, cache=c2)  # other epochs
        d2 = core.prepare(X, y, "classification", seed=1)  # other split
        search.eval_runs(cfg, [d2], build_model, 2, 1, far, cache=c2)
        assert len(calls) == 3
    finally:
        search._new_trainer = orig


def test_cli_writes_proxy_report_and_reuses_cache(tmp_path, tabular_csv, capsys):
    import json

    from dlnet import cli

    out = tmp_path / "o"
    argv = ["design", "--data", str(tabular_csv), "--target", "y", "--budget", "40s", "--topk",
            "1", "--per-family", "1", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    board = json.loads((out / "leaderboard.json").read_text())
    zc = board["zero_cost_proxies"]
    assert zc["used_for_pruning"] is False and zc["proxies"]["params"]["n"] >= 3
    assert "Zero-cost proxies" in (out / "leaderboard.md").read_text()
    capsys.readouterr()
    assert cli.main(argv) == 0  # same command again: stage-2 runs come from the cache
    assert "[cache] reused" in capsys.readouterr().out
    assert (out / ".dlnet_cache.json").exists()
