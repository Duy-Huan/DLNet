"""Regression tests of the v0.18 stability fixes: budgets that hold, thread limits, valid JSON.

Each test names the failure it guards against (agent.md 9.2).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from dlnet import cli, clock, jsonsafe, search, threads

pytestmark = pytest.mark.regression


def _strict_json(text):
    """json.loads that refuses the non-standard NaN / Infinity tokens Python writes by default."""

    def bad(tok):
        raise ValueError(f"non-standard JSON token {tok}")

    return json.loads(text, parse_constant=bad)


# ------------------------------------------------------------------------------ valid JSON


def test_jsonsafe_writes_standard_json():
    """Was: profile.json could contain the bare token NaN (not JSON; jq and browsers reject it)."""
    x = {"a": float("nan"), "b": [float("inf"), -float("inf"), 1.5], "c": np.float32(2.0),
         "d": np.int64(3), "e": np.array([1.0, np.nan]), "f": (1, None), "g": "NaN"}  # fmt: skip
    out = _strict_json(jsonsafe.dumps(x))
    assert out == {"a": None, "b": [None, None, 1.5], "c": 2.0, "d": 3, "e": [1.0, None],
                   "f": [1, None], "g": "NaN"}  # fmt: skip
    with pytest.raises(ValueError):
        _strict_json(json.dumps(float("nan")))  # the check itself works


def test_every_file_of_a_run_on_duplicated_samples_is_standard_json(tmp_path):
    """Duplicated samples give a NaN intrinsic dimension (profile.py); it must not reach a file."""
    rng = np.random.default_rng(0)
    X = np.repeat(rng.normal(size=(6, 5)), 15, axis=0).astype("float32")[:, :, None]
    p = tmp_path / "c.npz"
    np.savez(p, X=X, y=(X[:, 0, 0] > 0).astype(int))
    out = tmp_path / "o"
    argv = ["design", "--data", str(p), "--budget", "40ep", "--topk", "1", "--per-family", "1",
            "--zoo", "2", "--evolve", "0", "--out", str(out), "--no-builtin-meta"]  # fmt: skip
    assert cli.main(argv) == 0
    files = list(out.rglob("*.json")) + list(out.rglob("config.yaml"))
    assert len(files) >= 4
    for f in files:
        _strict_json(f.read_text())
    assert _strict_json((out / "profile.json").read_text())["intrinsic_dim"] is None


# ---------------------------------------------------------------------------- threads


def test_threads_resolve_and_limit_restore():
    """Was: torch, OpenBLAS and both OpenMP runtimes took every core; four runs side by side
    stalled for more than 10 minutes in HistGradientBoosting (0.2 s each with one thread)."""
    from threadpoolctl import threadpool_info

    assert threads.resolve(3) == 3
    assert 1 <= threads.resolve(0) <= threads.available_cpus()
    with pytest.raises(SystemExit, match="bad --threads"):
        threads.resolve(-1)
    before = torch.get_num_threads()
    assert threads.current() == -1
    with threads.limit(1) as n:
        assert n == 1 and torch.get_num_threads() == 1 and threads.current() == 1
        assert all(p["num_threads"] == 1 for p in threadpool_info())
    assert torch.get_num_threads() == before and threads.current() == -1


def test_reference_forests_use_the_thread_limit():
    from dlnet import reference

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3)).astype("float32")
    y = (X[:, 0] > 0).astype(int)
    d = {"task": "classification", "primary": "accuracy",
         "Xnp": {"tr": X[:40], "va": X[40:50], "te": X[50:]},
         "y_raw": {"tr": y[:40], "va": y[40:50], "te": y[50:]}}  # fmt: skip
    seen = []
    import sklearn.ensemble as ens

    orig = ens.RandomForestClassifier.fit

    def spy(self, *a, **k):
        seen.append(self.n_jobs)
        return orig(self, *a, **k)

    ens.RandomForestClassifier.fit = spy
    try:
        with threads.limit(2):
            reference.run_reference(d)
    finally:
        ens.RandomForestClassifier.fit = orig
    assert seen == [2]


# ----------------------------------------------------------------------------- budgets


def test_clock_stages_and_fits():
    clock.start("work")
    clock.stage("a")
    clock.spend(3)
    clock.stage("b")
    clock.spend(2)
    clock.stage("a")
    clock.spend(1)
    rows = {r["stage"]: r for r in clock.breakdown()}
    assert rows["a"]["epochs"] == 4 and rows["b"]["epochs"] == 2 and list(rows) == ["a", "b"]
    assert clock.fits(4, 10) and not clock.fits(5, 10)
    clock.start("time")
    assert clock.epochs() == 0 and clock.breakdown() == []


class _Tr:
    def __init__(self, seconds, epochs):
        self.seconds, self.epochs_done = seconds, epochs


def test_pace_predicts_from_the_trainer_then_from_flops():
    p = search.Pace()
    assert p.cost({"flops": 10}, 4) == 0.0  # nothing measured: the first run always starts
    p.record(_Tr(8.0, 4), {"flops": 100})  # 2 s/epoch, 0.02 s/epoch/FLOP
    assert p.cost({}, 3, _Tr(6.0, 2)) == 9.0
    assert p.cost({"flops": 400}, 2) == pytest.approx(16.0)
    assert p.cost({}, 2) == pytest.approx(4.0)
    assert p.cost({}, 0) == 0.0
    clock.start("work")
    assert p.cost({"flops": 400}, 7) == 7.0


def test_search_never_starts_a_run_that_would_end_after_the_deadline():
    """Was: a run was started whenever the clock was before the deadline, so the last one could
    end far after it (30 s budgets took 35-138 s)."""
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.normal(size=(60, 4)), dtype=torch.float32)
    y = torch.tensor((X[:, 0] > 0).numpy().astype("int64"))
    data = {"task": "classification", "primary": "accuracy", "in_dim": 4, "out_dim": 2,
            "Xtr": X[:40], "ytr": y[:40], "Xva": X[40:50], "yva": y[40:50], "Xte": X[50:],
            "yte": y[50:], "y_raw": {"tr": y[:40].numpy(), "va": y[40:50].numpy(),
                                     "te": y[50:].numpy()}}  # fmt: skip

    def build(cfg, i, o):
        return torch.nn.Linear(i, o)

    cands = [
        {"family": f"f{i}", "lr": 1e-2, "wd": 0.0, "batch": 16, "params": 10} for i in range(5)
    ]
    clock.start("work")
    res = search.run_search(data, cands, 10, 1, build, rungs=(4, 12), seeds=3, log=lambda *_: None)
    assert clock.epochs() <= 10  # no run was started that would end after epoch 10
    assert res and sum(r["n_seeds"] * r["epochs"] for r in res) <= 10


def test_design_reports_where_the_budget_went_and_keeps_a_work_budget(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6)).astype("float32")
    p = tmp_path / "t.npz"
    np.savez(p, X=X.reshape(120, 6, 1), y=(X[:, 0] > 0).astype(int))
    out = tmp_path / "o"
    argv = ["design", "--data", str(p), "--budget", "60ep", "--topk", "2", "--per-family", "1",
            "--zoo", "2", "--evolve", "1", "--evolve-pop", "2", "--out", str(out), "--no-cache",
            "--no-builtin-meta", "--threads", "1"]  # fmt: skip
    assert cli.main(argv) == 0
    b = _strict_json((out / "leaderboard.json").read_text())["budget"]
    assert b["mode"] == "work" and b["threads"] == 1
    assert b["epochs_trained"] <= 60 and b["overrun_pct"] <= 0
    stages = [s["stage"] for s in b["stages"]]
    assert stages[:3] == ["load", "reference", "generate"] and "search" in stages
    assert math.isclose(sum(s["epochs"] for s in b["stages"]), b["epochs_trained"])
    assert "## Budget: 60ep" in (out / "leaderboard.md").read_text()
