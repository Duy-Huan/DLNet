"""Regression tests of the v0.20 fixes of the quality audit (qa/reports/v0.19.0/SUMMARY.md, S1-S6).

Each test names the audit finding it guards against.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from dlnet import cli, clock, core, search
from dlnet.export import verdict

pytestmark = pytest.mark.regression


def _data(n=80, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = torch.tensor(rng.normal(size=(n, d)), dtype=torch.float32)
    y = torch.tensor((X[:, 0] > 0).numpy().astype("int64"))
    a, b = n // 2, 3 * n // 4
    return {"task": "classification", "primary": "accuracy", "in_dim": d, "out_dim": 2,
            "Xtr": X[:a], "ytr": y[:a], "Xva": X[a:b], "yva": y[a:b], "Xte": X[b:], "yte": y[b:],
            "y_raw": {"tr": y[:a].numpy(), "va": y[a:b].numpy(), "te": y[b:].numpy()}}  # fmt: skip


def _cands(n, **kw):
    return [
        {"family": f"f{i}", "lr": 1e-2, "wd": 0.0, "batch": 16, "params": 10, **kw}
        for i in range(n)
    ]


def _linear(cfg, i, o):
    return torch.nn.Linear(i, o)


class _Broken(torch.nn.Module):
    def __init__(self, i, o, mode):
        super().__init__()
        self.lin, self.mode = torch.nn.Linear(i, o), mode

    def forward(self, x):
        if self.mode == "raise":
            raise RuntimeError("shape bug in this candidate")
        return self.lin(x) * float("nan")


def _maybe_broken(cfg, i, o):
    return _Broken(i, o, cfg["broken"]) if cfg.get("broken") else torch.nn.Linear(i, o)


def test_hyperband_does_not_crash_when_a_work_budget_ends_exactly():
    """S1: a new bracket was opened at clock == deadline, trained nobody, then KeyError 'val0'."""
    clock.start("work")
    # R=8, eta=2, unit=4: bracket 1 trains 2 x 4 + 1 x 4 = 12 epochs, exactly the deadline
    res = search.run_hyperband(_data(), _cands(6), 12, 1, _linear, R=8, eta=2, unit=4, seeds=1,
                               log=lambda *_: None)  # fmt: skip
    assert res and clock.epochs() <= 12


def test_a_failing_or_diverging_candidate_is_dropped_not_fatal():
    """S2 (audit R9): an exception ended the run; a NaN score corrupted the sort and top-1."""
    cands = _cands(4)
    cands[0]["broken"], cands[2]["broken"] = "raise", "nan"
    trace = {}
    res = search.run_search(_data(), cands, float("inf"), 2, _maybe_broken, rungs=(2,), seeds=1,
                            log=lambda *_: None, trace=trace)  # fmt: skip
    assert {r["cfg"]["family"] for r in res} == {"f1", "f3"}
    assert [f["candidate"].split()[0] for f in trace["failed"]] == ["f0", "f2"]
    assert (
        "RuntimeError" in trace["failed"][0]["error"] and "diverged" in trace["failed"][1]["error"]
    )
    assert all(np.isfinite(r["val_mean"]) for r in res)
    with pytest.raises(SystemExit, match="every candidate failed"):
        search.run_search(_data(), _cands(2, broken="raise"), float("inf"), 1, _maybe_broken,
                          rungs=(2,), seeds=1, log=lambda *_: None)  # fmt: skip


def _result(tr, val, epochs, n=1):
    return {"cfg": {**_cands(1)[0], "family": f"e{epochs}_{val}"}, "trainer": tr, "val_mean": val,
            "val_std": 0.0, "n_seeds": n, "epochs": epochs, "test": {"score": val}}  # fmt: skip


def test_models_of_different_epochs_are_equalized_or_ranked_apart():
    """S3: 4-epoch models were ranked against 12-epoch ones by their scores."""
    d = _data()
    short = core.Trainer(torch.nn.Linear(4, 2), d, lr=1e-2, batch=16)
    short.fit(4)
    long_ = core.Trainer(torch.nn.Linear(4, 2), d, lr=1e-2, batch=16)
    long_.fit(12)
    res = [_result(short, 0.95, 4), _result(long_, 0.60, 12)]
    order = search.final_order([dict(r) for r in res])
    assert [r["epochs"] for r in order] == [12, 4]  # the higher 4-epoch score does not jump ahead
    clock.start("work")
    eq = search.equalize([dict(r) for r in res], [d], _linear, deadline=100, log=lambda *_: None)
    assert {r["epochs"] for r in eq} == {12} and short.epochs_done == 12
    clock.start("work")
    kept = search.equalize([dict(r) for r in res], [d], _linear, deadline=3, log=lambda *_: None)
    assert sorted(r["epochs"] for r in kept) == [4, 12]  # no budget: left as they are


def test_single_run_noise_is_the_validation_sampling_error():
    """S4: one run printed +/- 0.0 and split models differing by one validation sample."""
    assert search.sampling_noise(0.9, "accuracy", 100) == pytest.approx(0.03)
    assert search.sampling_noise(-2.0, "neg_rmse", 50) == pytest.approx(0.2)
    assert search.sampling_noise(-1.0, "neg_mse", 8) == pytest.approx(0.5)
    rs = [
        {"val_mean": v, "val_std": 0.0, "n_seeds": 1, "epochs": 12, "cfg": {}}
        for v in (0.98873, 0.98592)
    ]
    search.annotate_noise(rs, {"yva": np.zeros(355), "primary": "accuracy"})
    assert rs[0]["val_noise"] > 0 and rs[0]["noise_source"] == "validation sampling"
    search.assign_groups(rs, ordered=True)
    assert rs[0]["group"] == rs[1]["group"]  # 1/355 apart: indistinguishable


def test_verdict_uses_noise_on_regression_and_on_test():
    """Audit R2: regression had zero noise with one run; any positive test margin was a win."""
    ref = [{"name": "ridge", "va": {"score": -1.00}, "te": {"score": -1.00}}]
    top = {"val_mean": -0.99, "val_std": 0.0, "test": {"score": -0.99}}
    v = verdict(top, ref, 100, "regression", "neg_rmse", 100)
    assert not v["beats_reference"] and v["val_noise"] > 0.01 and v["test_noise"] > 0
    top = {"val_mean": -0.5, "val_std": 0.0, "test": {"score": -0.5}}
    assert verdict(top, ref, 100, "regression", "neg_rmse", 100)["beats_reference"]
    clf = {"val_mean": 0.95, "val_std": 0.0, "test": {"score": 0.905}}
    cref = [{"name": "rf", "va": {"score": 0.80}, "te": {"score": 0.90}}]
    assert not verdict(clf, cref, 400, "classification", "accuracy", 400)["beats_reference"]


def test_numeric_text_columns_are_read_as_numbers(tmp_path):
    """S6: a numeric column with a few '?' was one-hot encoded into one column per value."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    col = [f"{v:.3f}" for v in x]
    col[3] = col[17] = "?"
    ids = [f"id{i}" for i in range(60)]
    p = tmp_path / "t.csv"
    p.write_text(
        "x,uid,y\n"
        + "\n".join(f"{a},{b},{int(v > 0)}" for a, b, v in zip(col, ids, x, strict=True))
    )
    warns = []
    X, _, cols = core.load_csv(str(p), "y", warns)
    assert (
        "x" in cols and np.isnan(X[3, cols.index("x")]) and abs(X[0, cols.index("x")] - x[0]) < 1e-3
    )
    assert any("'x' is numeric" in w for w in warns) and any(
        "'uid' has 60 distinct" in w for w in warns
    )
    with pytest.raises(SystemExit, match="is not a column"):
        core.load_csv(str(p), "label")


def test_hsi_pixel_split_warns_about_spatial_leakage(hsi_npz, tmp_path):
    out = tmp_path / "o"
    argv = ["design", "--data", str(hsi_npz), "--budget", "20ep", "--topk", "1", "--per-family", "1",
            "--zoo", "2", "--evolve", "0", "--no-proxies", "--threads", "1", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    board = json.loads((out / "leaderboard.json").read_text())
    assert any("spatial leakage" in w for w in board["warnings"])
    md = (out / "leaderboard.md").read_text()
    assert "**Warnings**" in md and "| epochs | runs |" in md
    assert board["top_k"][0]["epochs"] > 0 and board["top_k"][0]["val_noise"] > 0


def test_common_input_mistakes_get_messages_that_say_what_to_do(tmp_path, tabular_csv):
    """Audit F4: 8/16 common mistakes gave a traceback or a misleading hint."""
    from dlnet import plugins

    rng = np.random.default_rng(0)
    p = tmp_path / "odd.npz"  # per-sample arrays with other names: was sent to the HSI plugin
    np.savez(p, images=rng.normal(size=(30, 8, 8)), labels=rng.integers(0, 2, 30))
    with pytest.raises(SystemExit, match=r"images \(30, 8, 8\).*Name them X and y"):
        plugins.route(str(p))
    base = ["design", "--data", str(tabular_csv), "--target", "y", "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="bad --max-params 'abc'"):
        cli.main([*base, "--max-params", "abc"])
    with pytest.raises(SystemExit, match="none fits --max-params 10 .*smallest has"):
        cli.main([*base, "--max-params", "10", "--budget", "4ep", "--no-proxies", "--threads", "1"])
    with pytest.raises(SystemExit, match=r"did you mean \['y'\]"):
        core.load_csv(str(tabular_csv), "Y")
    with pytest.raises(SystemExit, match="file not found"):
        cli.main(["design", "--data", str(tmp_path / "nope.csv"), "--target", "y"])


def test_leaderboard_says_how_many_candidates_were_trained(tmp_path, tabular_csv):
    """Audit F7: the leaderboard did not say that only 8 of 76 candidates were trained."""
    out = tmp_path / "o"
    argv = ["design", "--data", str(tabular_csv), "--target", "y", "--budget", "8ep", "--topk", "3",
            "--no-proxies", "--threads", "1", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    b = json.loads((out / "leaderboard.json").read_text())
    cov = b["search_coverage"]
    assert cov["trained"] < cov["candidates"] and any(
        "could be trained" in w for w in b["warnings"]
    )


class _Slow(torch.nn.Module):
    """Tiny FLOPs, slow step: what a Python-loop family (MambaVision, KAT, xLSTM) looks like."""

    def __init__(self, i, o):
        super().__init__()
        self.lin = torch.nn.Linear(i, o)

    def forward(self, x):
        import time

        time.sleep(0.1)
        return self.lin(x)


def test_time_budget_is_not_blown_by_a_slow_first_candidate():
    """QA v1.0 (examples/): `--budget 90s` ran 12+ minutes. The first candidate always started and
    an untrained one was priced by FLOPs, which miss Python-loop cost; now one timed step prices it
    and rung 1 starts with what fits."""
    import time

    clock.start("time")
    cands = _cands(3, flops=10)
    cands[0]["slow"] = True  # interleave keeps f0 first: before the fix it trained first, ~4 s
    build = lambda cfg, i, o: _Slow(i, o) if cfg.get("slow") else torch.nn.Linear(i, o)
    t0 = time.time()
    res = search.run_search(_data(), cands, clock.now() + 1.0, 1, build, rungs=(4,), seeds=1,
                            log=lambda *_: None)  # fmt: skip
    assert time.time() - t0 < 2.0
    assert res and all(not r["cfg"].get("slow") for r in res)


def test_probe_leaves_the_model_and_the_rng_untouched():
    d = _data()
    m = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8), torch.nn.Dropout(0.5), torch.nn.Linear(8, 2)
    )
    tr = core.Trainer(m, d, lr=1e-2, wd=0.0, batch=16)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    rng = torch.get_rng_state()
    assert tr.probe_epoch_seconds() > 0
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in m.parameters())
