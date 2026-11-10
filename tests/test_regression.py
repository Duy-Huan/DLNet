"""Regression tests: each one pins a behaviour found in the audit (see skills/audit-and-regression).

Every test below failed on the code before the audit fixes, or locks a rule that must not drift
(acceptance verdict, split leakage, deterministic splits, numeric golden values).
"""

from __future__ import annotations

import json
import time
import types

import pytest

pytestmark = pytest.mark.regression

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")
pytest.importorskip("scipy")

import numpy as np

from dlnet import cli, core, export, hsi_data, plugins, search
from dlnet.library import TTLinear, build_model


# ----------------------------------------------------------------------------- helpers
def _cfg(**kw):
    c = {
        "family": "mlp",
        "width": 16,
        "depth": 2,
        "dropout": 0.0,
        "lr": 1e-2,
        "wd": 1e-4,
        "batch": 32,
    }
    c.update(kw)
    c["params"] = sum(p.numel() for p in build_model(c, 4, 2).parameters())
    return c


def _tab_data(n=160, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype("float32")
    return core.prepare(X, (X[:, 0] > 0).astype(int), "classification", seed=0)


# ============================================================ search: budget / fairness
def test_search_with_expired_deadline_still_returns_a_result():
    """Was: IndexError (alive[0] of an empty list) when the budget ran out before rung 1."""
    res = search.run_search(
        _tab_data(), [_cfg(), _cfg(width=32)], time.time() - 1, 1, build_model,
        rungs=(1, 2), seeds=1, log=lambda *_: None,
    )  # fmt: skip
    assert len(res) >= 1


def test_search_does_not_compare_candidates_with_different_epochs(monkeypatch):
    """Was: when the budget expired mid-rung, candidates left behind kept a stale val from an
    earlier rung and were ranked against candidates trained longer."""
    tick = {"t": 0.0}

    def fake_time():
        tick["t"] += 1.0
        return tick["t"]

    monkeypatch.setattr(search, "time", types.SimpleNamespace(time=fake_time))
    cands = [_cfg(width=w) for w in (8, 16, 24, 32)]
    res = search.run_search(
        _tab_data(), cands, 7.5, 2, build_model, rungs=(1, 2, 3), eta=1, seeds=1,
        log=lambda *_: None,
    )  # fmt: skip
    assert len({r["epochs"] for r in res}) == 1, [r["epochs"] for r in res]


def test_search_is_deterministic_for_a_fixed_seed():
    def go():
        res = search.run_search(
            _tab_data(), [_cfg()], time.time() + 60, 1, build_model,
            rungs=(2,), seeds=1, log=lambda *_: None,
        )  # fmt: skip
        return res[0]["val_mean"], res[0]["test"]["accuracy"]

    assert go() == go()


def test_select_topk_prefers_family_diversity_and_assigns_roles():
    def r(fam, val, params, std=0.01):
        return {"cfg": {"family": fam, "params": params}, "val_mean": val, "val_std": std}

    results = [r("a", 0.9, 100), r("a", 0.89, 50), r("b", 0.8, 30, 0.001), r("c", 0.7, 10)]
    chosen = search.select_topk(results, 3)
    assert [c["cfg"]["family"] for c in chosen] == ["a", "b", "c"]  # a2 loses to b and c
    assert chosen[0]["roles"] == ["top1"]
    assert "best-small" in chosen[2]["roles"] and "most-stable" in chosen[1]["roles"]


# ================================================================== verdict (frozen rule)
def _top(val, val_std, test_score):
    return {"val_mean": val, "val_std": val_std, "test": {"score": test_score}}


def _ref(name, va, te):
    return {"name": name, "va": {"score": va}, "te": {"score": te}}


@pytest.mark.parametrize(
    ("top", "refs", "n_val", "expected"),
    [
        # clear win on val and on test against every reference
        (_top(0.95, 0.005, 0.94), [_ref("rf", 0.85, 0.84), _ref("svm", 0.80, 0.82)], 500, True),
        # wins on val by less than the noise floor -> no win
        (_top(0.851, 0.005, 0.90), [_ref("rf", 0.85, 0.84)], 500, False),
        # wins on val but loses on test to a NON-validation-winner reference -> no win
        (_top(0.95, 0.005, 0.83), [_ref("rf", 0.85, 0.80), _ref("svm", 0.80, 0.84)], 500, False),
        # tiny validation set: binomial noise floor blocks a 3-point margin
        (_top(0.90, 0.001, 0.95), [_ref("rf", 0.87, 0.80)], 30, False),
    ],
)
def test_verdict_rule(top, refs, n_val, expected):
    refs = sorted(refs, key=lambda r: -r["va"]["score"])
    assert export.verdict(top, refs, n_val, "classification")["beats_reference"] is expected


# ================================================================ export: safety / layout
def _export_minimal(out):
    data = _tab_data()
    prof = {"n_features": 4}
    cands = [_cfg()]
    res = search.run_search(
        data, cands, time.time() + 60, 1, build_model, rungs=(1,), seeds=1, log=lambda *_: None
    )
    chosen = search.select_topk(res, 1)
    ref = [
        {
            "name": "m",
            "seconds": 0,
            "va": {"score": 0.5},
            "te": {"score": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5},
        }
    ]
    plugin = plugins.get_plugin("tabular")
    export.export(out, plugin, data, prof, chosen, ref, seed=0)


def test_export_refuses_to_wipe_a_foreign_directory(tmp_path):
    """Was: shutil.rmtree(out) -> `--out ~/work` deleted everything in it."""
    out = tmp_path / "work"
    out.mkdir()
    precious = out / "precious.txt"
    precious.write_text("keep me")
    with pytest.raises(SystemExit):
        _export_minimal(out)
    assert precious.read_text() == "keep me"


def test_export_overwrites_only_its_own_previous_output(tmp_path):
    out = tmp_path / "o"
    _export_minimal(out)
    (out / "notes.txt").write_text("user file")  # not produced by DLNet
    _export_minimal(out)  # second run replaces DLNet output
    assert (out / "notes.txt").exists()
    assert json.loads((out / "leaderboard.json").read_text())["top_k"]


def test_exported_package_is_self_contained(tmp_path):
    out = tmp_path / "o"
    _export_minimal(out)
    pkg = next(out.glob("top1_*"))
    for f in ("model.py", "core.py", "train.py", "config.yaml", "report.md", "tests/test_model.py"):
        assert (pkg / f).exists(), f
    for f in pkg.glob("*.py"):
        assert "from dlnet" not in f.read_text() and "import dlnet" not in f.read_text(), f


# ================================================================ tabular data handling
def test_split_is_deterministic_and_stratified():
    y = np.array([0] * 50 + [1] * 30)
    a = core.stratified_split(y, "classification", seed=3)
    b = core.stratified_split(y, "classification", seed=3)
    assert all((x == z).all() for x, z in zip(a, b, strict=True))
    tr, va, te = a
    assert (len(tr), len(va), len(te)) == (48, 16, 16)
    assert len(set(tr) | set(va) | set(te)) == 80 and not (set(tr) & set(te))
    for part in a:  # every class in every split
        assert set(y[part]) == {0, 1}


def test_prepare_raises_clear_error_when_a_split_would_be_empty():
    """Was: evaluate() crashed with torch.cat([]) on an empty validation split."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="too few"):
        core.prepare(rng.normal(size=(4, 3)), np.array([0, 1] * 2), "classification", seed=0)
    d = core.prepare(rng.normal(size=(8, 3)), np.array([0, 1] * 4), "classification", seed=0)
    assert min(len(d["ytr"]), len(d["yva"]), len(d["yte"])) >= 1  # small but usable


def test_missing_values_are_imputed_with_train_statistics_only(tmp_path):
    """Was: load_csv filled NaN with the median of ALL rows (test rows leaked into train)."""
    import pandas as pd

    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    df.loc[::7, "a"] = np.nan
    df["y"] = (rng.normal(size=120) > 0).astype(int)
    p = tmp_path / "m.csv"
    df.to_csv(p, index=False)
    X, y, _ = core.load_csv(str(p), "y")
    assert np.isnan(X).any(), "load_csv must not impute with global statistics"
    d = core.prepare(X, y, "classification", seed=0)
    assert not np.isnan(d["Xnp"]["tr"]).any() and not np.isnan(d["Xnp"]["te"]).any()


def test_rows_with_missing_target_are_dropped(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": np.arange(10.0), "y": [0, 1] * 5})
    df.loc[3, "y"] = np.nan
    p = tmp_path / "n.csv"
    df.to_csv(p, index=False)
    X, y, _ = core.load_csv(str(p), "y")
    assert len(X) == len(y) == 9 and not np.isnan(y.astype(float)).any()


# ============================================================================ golden values
def test_compute_metrics_golden_values():
    m = core.compute_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), "classification", "accuracy"
    )
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["balanced_accuracy"] == pytest.approx(0.75)
    assert m["kappa"] == pytest.approx(0.5)
    r = core.compute_metrics(
        np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]), "regression", "neg_rmse"
    )
    assert r["rmse"] == pytest.approx((4 / 3) ** 0.5) and r["score"] == pytest.approx(-r["rmse"])


def test_ttlinear_matches_its_defining_formula():
    """W[(i1,i2),(o1,o2)] = sum_r G1[i1,o1,r] G2[r,i2,o2]  (agent.md section 4)."""
    layer = TTLinear(12, 6, rank=3)
    W = torch.einsum("aor,rbp->abop", layer.g1, layer.g2).reshape(12, 6)  # (i1 i2) x (o1 o2)
    x = torch.randn(5, 12)
    assert torch.allclose(layer(x), x @ W + layer.bias, atol=1e-5)


def test_patchstore_matches_direct_slicing(cube_gt):
    cube, _ = cube_gt
    p = 5
    pad = torch.from_numpy(np.pad(cube, ((2, 2), (2, 2), (0, 0)), mode="reflect"))
    store = hsi_data.PatchStore(pad, np.array([0, 10, 47]), np.array([0, 20, 47]), p)
    got = store[torch.arange(3)]
    for i, (r, c) in enumerate([(0, 0), (10, 20), (47, 47)]):
        want = pad[r : r + p, c : c + p, :].permute(2, 0, 1)
        assert torch.equal(got[i], want)


def test_twonn_recovers_a_low_dimensional_manifold():
    from dlnet.profile import twonn_intrinsic_dim

    rng = np.random.default_rng(0)
    Z = rng.uniform(size=(1500, 2))
    X = Z @ rng.normal(size=(2, 12))  # a 2-D plane embedded in 12-D
    assert 1.6 < twonn_intrinsic_dim(X) < 2.5


# ==================================================================== HSI: round trip / guards
def test_hsi_data_config_rebuilds_the_identical_split(cube_gt):
    """train.py rebuilds data from config.yaml; it must equal the data used in the search."""
    from dlnet.plugins.hsi import HSIPlugin

    cube, gt = cube_gt
    for split in ("pixel", "block"):
        d = hsi_data.build(cube, gt, split=split, seed=2, patch=5, pca=12, block=8)
        cfg = json.loads(json.dumps(HSIPlugin().data_config(d)))  # via JSON, like config.yaml
        d2 = hsi_data.build(cube, gt, split=cfg["split"], seed=2, patch=cfg["patch"],
                            pca=cfg["pca"], frac=cfg["frac"], block=cfg["block"])  # fmt: skip
        for sp in ("tr", "va", "te"):
            assert np.array_equal(d["y_raw"][sp], d2["y_raw"][sp]), (split, sp)
            assert np.allclose(d["Xnp"][sp], d2["Xnp"][sp]), (split, sp)


def test_hsi_pca_stem_is_fit_on_train_pixels_only(cube_gt):
    cube, gt = cube_gt
    d = hsi_data.build(cube, gt, split="pixel", seed=0, patch=5, pca=8)
    tr = d["Xnp"]["tr"]
    assert np.allclose(tr.mean(0), 0, atol=1e-4) and np.allclose(tr.std(0), 1, atol=1e-3)


def test_hsi_block_split_with_empty_part_raises_clear_error(cube_gt):
    """Was: block larger than the image put everything in train; evaluate() then crashed."""
    cube, gt = cube_gt
    with pytest.raises(ValueError, match="empty"):
        hsi_data.build(cube, gt, split="block", seed=0, patch=5, pca=8, block=200)


# ================================================================================= CLI / router
@pytest.mark.parametrize(
    ("s", "sec"), [("90s", 90), ("5m", 300), ("2h", 7200), ("30", 30), ("1.5m", 90)]
)
def test_parse_budget(s, sec):
    assert cli.parse_budget(s) == sec


@pytest.mark.parametrize("bad", ["", "m5", "5x", "-1m"])
def test_parse_budget_rejects_garbage(bad):
    with pytest.raises(SystemExit):
        cli.parse_budget(bad)


def test_router_picks_plugin_by_extension():
    assert plugins.route("a.csv").name == "tabular"
    assert plugins.route("a.mat").name == "hsi"
    with pytest.raises(SystemExit):
        plugins.route("a.unknown")


def test_cli_without_candidates_fails_with_a_clear_message(tabular_csv, tmp_path):
    """Was: IndexError inside the search when no candidate was generated (--per-family 0)."""
    argv = ["design", "--data", str(tabular_csv), "--target", "y", "--per-family", "0",
            "--out", str(tmp_path / "o")]  # fmt: skip
    with pytest.raises(SystemExit, match="candidate"):
        cli.main(argv)
