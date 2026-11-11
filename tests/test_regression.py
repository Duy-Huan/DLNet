"""Regression tests: each one pins a behaviour found in the audit (see skills/audit-and-regression).

Every test below failed on the code before the audit fixes, or locks a rule that must not drift
(acceptance verdict, split leakage, deterministic splits, numeric golden values).
"""

from __future__ import annotations

import json
import time

import pytest

pytestmark = pytest.mark.regression

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")
pytest.importorskip("scipy")

import numpy as np

from dlnet import cli, core, export, hparams, hsi_data, plugins, search
from dlnet.zoo.families import tabular_model as build_model
from dlnet.zoo.ops_vec import TTLinear


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


def _tab_xy(n=160, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype("float32")
    return X, (X[:, 0] > 0).astype(int)


def _tab_data(n=160, seed=0):
    return core.prepare(*_tab_xy(n, seed), "classification", seed=0)


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

    monkeypatch.setattr(search.clock, "now", fake_time)
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
    def r(fam, val, params, std=0.01, n=3):
        return {
            "cfg": {"family": fam, "params": params},
            "val_mean": val,
            "val_std": std,
            "n_seeds": n,
        }

    results = [r("a", 0.9, 100), r("a", 0.89, 50), r("b", 0.8, 30, 0.001), r("c", 0.7, 10)]
    chosen = search.select_topk(results, 3)
    assert [c["cfg"]["family"] for c in chosen] == ["a", "b", "c"]  # a2 loses to b and c
    assert chosen[0]["roles"] == ["top1"]
    assert "best-small" in chosen[2]["roles"] and "most-stable" in chosen[1]["roles"]


def test_roles_are_not_given_without_evidence():
    """Was (audit S5, S4): best-small went to a model LARGER than top1 (171k vs 72k params), and
    most-stable was picked from std 0.0 of single runs."""

    def r(fam, val, params, n=1):
        return {
            "cfg": {"family": fam, "params": params},
            "val_mean": val,
            "val_std": 0.0,
            "n_seeds": n,
        }

    chosen = search.select_topk([r("a", 0.9, 72_000), r("b", 0.8, 171_000), r("c", 0.7, 90_000)], 3)
    assert all(role not in c["roles"] for c in chosen for role in ("best-small", "most-stable"))


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
    argv = ["design", "--data", str(tabular_csv), "--target", "y", "--per-family", "0", "--zoo", "0",
            "--out", str(tmp_path / "o")]  # fmt: skip
    with pytest.raises(SystemExit, match="candidate"):
        cli.main(argv)


# ============================================================ hyperparameter estimation
def test_noise_scale_estimator_matches_mccandlish_closed_form():
    """E|G_B|^2 = |G|^2 + S/B (arXiv:1812.06162 A.1); inverting it must give |G|^2, S and B_simple."""
    g2_true, s_true, bs, bb = 4.0, 100.0, 16, 256
    g2, s, b_simple = hparams.noise_scale_from_norms(
        g2_true + s_true / bs, bs, g2_true + s_true / bb, bb
    )
    assert g2 == pytest.approx(g2_true) and s == pytest.approx(s_true)
    assert b_simple == pytest.approx(25.0)
    # pure noise (|G|^2 <= 0) must not produce a bogus batch size
    assert hparams.noise_scale_from_norms(1.0, 16, 1.0 + 1e-9, 256)[2] is None


def test_normalized_weight_decay_scales_with_sqrt_batch():
    """AdamW appendix B.1: lambda = lambda_norm * sqrt(b / (B*T)); lambda_norm kept when b changes."""
    assert hparams.normalized_wd(0.01, 64, 256) == pytest.approx(0.02)
    assert hparams.normalized_wd(0.01, 64, 64) == pytest.approx(0.01)


def test_suggest_batch_is_power_of_two_and_clipped():
    assert hparams.suggest_batch(216.0, 5000, 64) == 128
    assert hparams.suggest_batch(5.0, 5000, 64) == hparams.MIN_BATCH
    assert hparams.suggest_batch(1e6, 100000, 64) == hparams.MAX_BATCH
    assert hparams.suggest_batch(300.0, 100, 64) == hparams.MIN_BATCH  # n_train // 4 caps it
    assert hparams.suggest_batch(None, 5000, 64) == 64  # no estimate: keep the searched batch


def test_lr_range_test_is_deterministic_and_inside_the_sweep():
    d = _tab_data(400)
    a = hparams.lr_range_test(build_model, _cfg(), d, 32, seed=0)
    b = hparams.lr_range_test(build_model, _cfg(), d, 32, seed=0)
    assert a == b
    assert 1e-5 <= a["lr_base"] < a["lr"] < a["lr_max"] <= 1.0
    assert a["lr_max"] == pytest.approx(2 * a["lr"]) and a["lr_max"] == pytest.approx(
        4 * a["lr_base"]
    )


def test_gradient_noise_scale_on_trained_model_is_finite():
    d = _tab_data(400)
    tr = core.Trainer(build_model(_cfg(), 4, 2), d, lr=1e-2, wd=1e-4, batch=32)
    tr.fit(3)
    g = hparams.gradient_noise_scale(tr.model, d, n_meas=8)
    assert g["grad_sq"] > 0 and g["trace_sigma"] > 0 and np.isfinite(g["b_simple"])
    assert hparams.gradient_noise_scale(tr.model, _tab_data(20), b_big=256) is None  # too few rows


def test_refine_keeps_searched_setting_unless_clearly_better(monkeypatch):
    """The estimate is a proposal: adoption needs a margin larger than the seed std."""
    d = _tab_data(300)
    cands = [_cfg(), _cfg(width=32)]
    res = search.run_search(d, cands, time.time() + 60, 1, build_model, rungs=(1, 2), seeds=2,
                            log=lambda *_: None)  # fmt: skip
    searched = {id(r): dict(r["cfg"]) for r in res}
    # estimate says: much worse proposal (val 0.0) -> must be rejected
    monkeypatch.setattr(
        search, "eval_runs", lambda *a, **k: ([0.0, 0.0], [{"score": 0.0}] * 2, 0.0, 1)
    )
    out = hparams.refine(res, d, build_model, time.time() + 60, log=lambda *_: None)
    for r in out:
        assert r["hparams"]["adopted"] == "searched"
        assert r["cfg"] == searched[id(r)]
        assert set(r["hparams"]["searched"]) == {"lr", "batch", "wd"}


def test_refine_adopts_a_clearly_better_proposal(monkeypatch):
    d = _tab_data(300)
    res = search.run_search(d, [_cfg()], time.time() + 60, 1, build_model, rungs=(1, 2), seeds=2,
                            log=lambda *_: None)  # fmt: skip
    monkeypatch.setattr(
        hparams, "estimate",
        lambda *a, **k: {"lr": 5e-3, "batch": 64, "wd": 2e-4, "gradient_noise_scale": None},
    )  # fmt: skip
    monkeypatch.setattr(
        search,
        "eval_runs",
        lambda *a, **k: ([2.0, 2.0], [{"score": 2.0, "accuracy": 2.0}] * 2, 0.0, 1),
    )
    out = hparams.refine(res, d, build_model, time.time() + 60, log=lambda *_: None)
    r = out[0]
    assert r["hparams"]["adopted"] == "estimated"
    assert (r["cfg"]["lr"], r["cfg"]["batch"], r["cfg"]["wd"]) == (5e-3, 64, 2e-4)
    assert r["val_mean"] == 2.0  # numbers of the adopted setting replace the searched ones


def test_exported_outputs_carry_hyperparameters(tmp_path):
    """Every Top-K entry lists lr, weight decay, batch size, epochs, schedule and where they came from."""
    d = _tab_data(200)
    plugin = plugins.get_plugin("tabular")
    res = search.run_search(d, [_cfg()], time.time() + 60, 1, build_model, rungs=(1, 2), seeds=1,
                            log=lambda *_: None)  # fmt: skip
    res = hparams.refine(res, d, build_model, time.time() + 60, log=lambda *_: None)
    chosen = search.select_topk(search.assign_groups(res), 1)
    ref = [
        {
            "name": "r",
            "va": {"score": 0.5},
            "te": {"score": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5},
        }
    ]
    export.export(tmp_path / "o", plugin, d, {"task": "classification"}, chosen, ref, 0)
    row = json.loads((tmp_path / "o" / "leaderboard.json").read_text())["top_k"][0]
    hp = row["hyperparameters"]
    assert {"optimizer", "lr", "weight_decay", "batch_size", "epochs", "lr_schedule"} <= set(hp)
    assert hp["source"]["adopted"] in ("searched", "estimated")
    cfg = json.loads(next((tmp_path / "o").glob("top1_*")).joinpath("config.yaml").read_text())
    assert cfg["hyperparameters"]["lr"] == cfg["arch"]["lr"]  # train.py reads arch; must agree
    assert "lr" in (tmp_path / "o" / "leaderboard.md").read_text()


def test_seed_controls_weight_initialisation():
    """Was: models were built before torch.manual_seed, so the same seed gave different weights."""
    d = _tab_data()
    a = search._new_trainer(_cfg(), d, 3, build_model).model
    torch.manual_seed(999)  # disturb the global RNG between the two builds
    b = search._new_trainer(_cfg(), d, 3, build_model).model
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters(), strict=True))


# ---------------------------------------------------------------- repeated-split ranking (M3)


def test_finalists_are_ranked_on_several_splits():
    """agent.md 9.7: seed std misses split noise, so stage 2 must really use the other splits."""
    X, y = _tab_xy(300)
    folds = [core.prepare(X, y, "classification", seed=s) for s in range(3)]
    assert not np.array_equal(folds[0]["Xnp"]["va"], folds[1]["Xnp"]["va"])
    used = []

    def build(cfg, i, o):
        return build_model(cfg, i, o)

    orig = search._new_trainer

    def spy(cfg, data, seed, build_fn):
        used.append((seed, next(i for i, f in enumerate(folds) if f is data)))
        return orig(cfg, data, seed, build_fn)

    search._new_trainer = spy
    try:
        res = search.run_search(folds[0], [_cfg()], time.time() + 60, 1, build, rungs=(1, 2),
                                seeds=2, log=lambda *_: None, folds=folds)  # fmt: skip
    finally:
        search._new_trainer = orig
    assert used == [(0, 0), (1, 1), (2, 2)]  # run i = seed i on split i; run 0 reuses stage 1
    r = res[0]
    assert r["n_splits"] == 3 and r["n_seeds"] == 3 and r["test_std"] >= 0


def test_mean_reference_averages_over_splits():
    a = [{"name": "m", "seconds": 1.0, "va": {"score": 0.8}, "te": {"score": 0.6}}]
    b = [{"name": "m", "seconds": 1.0, "va": {"score": 0.6}, "te": {"score": 0.8}}]
    out = export.mean_reference([a, b])
    assert out[0]["va"]["score"] == pytest.approx(0.7) and out[0]["n_splits"] == 2
    assert export.mean_reference([a]) is a


def test_cli_block_split_uses_three_splits_by_default(tmp_path, hsi_npz):
    out = tmp_path / "o"
    argv = ["design", "--data", str(hsi_npz), "--split", "block", "--block-size", "8", "--patch",
            "3", "--pca", "8", "--frac", "0.4,0.2,0.4", "--budget", "40s", "--topk", "1",
            "--per-family", "1", "--zoo", "0", "--evolve", "0", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    board = json.loads((out / "leaderboard.json").read_text())
    assert board["top_k"][0]["n_splits"] >= 2
    assert "split-to-split" in (out / "leaderboard.md").read_text()


def test_hyperband_with_expired_deadline_still_returns_one_result():
    """Was: ValueError (max of an empty list) when the budget was spent before Hyperband started."""
    res = search.run_hyperband(
        _tab_data(), [_cfg(), _cfg(width=32)], time.time() - 1, 1, build_model,
        R=4, unit=1, seeds=1, log=lambda *_: None,
    )  # fmt: skip
    assert len(res) == 1 and res[0]["epochs"] == 1


def test_hyperband_finalists_all_reached_the_full_resource():
    cands = [_cfg(width=w) for w in (8, 12, 16, 20, 24, 28, 32, 36, 40)]
    trace = {}
    res = search.run_hyperband(
        _tab_data(), cands, time.time() + 120, 2, build_model, R=9, unit=1, seeds=1,
        log=lambda *_: None, trace=trace,
    )  # fmt: skip
    assert res and {r["epochs"] for r in res} == {9}
    assert (
        trace["rung1_epochs"] == 1 and len(trace["rung1"]) == 9
    )  # bracket s=2 starts 9 at 1 epoch


def test_image_tensors_are_contiguous():
    """Was: (N, H, W, 3) images became channels-last strided tensors (transposed view kept by fancy
    indexing); torch 2.15 crashed with "double free or corruption" in multi-threaded conv backward."""
    from dlnet import array_data

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8, 8, 3)).astype("float32")
    d = array_data.build(X, np.repeat([0, 1, 2], 10), "image", "classification", 0)
    assert all(d["X" + k].is_contiguous() for k in ("tr", "va", "te"))
    s = array_data.build_segmentation(X, rng.integers(0, 2, (30, 8, 8)), 0)
    assert all(s["X" + k].is_contiguous() for k in ("tr", "va", "te"))


def test_work_budget_is_reproducible_and_bounded(tmp_path):
    """Was: every budget was wall-clock seconds, so the same command gave a different leaderboard
    on a slower machine (and quality-asserting tests failed under coverage or a busy CI runner)."""
    import json

    from dlnet import cli, clock

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6)).astype("float32")
    p = tmp_path / "t.npz"
    np.savez(p, X=X.reshape(120, 6, 1), y=(X[:, 0] > 0).astype(int))

    def go(out):
        argv = ["design", "--data", str(p), "--budget", "60ep", "--topk", "2", "--per-family", "1",
                "--zoo", "2", "--evolve", "1", "--evolve-pop", "2", "--out", str(out), "--no-cache",
                "--no-builtin-meta"]  # fmt: skip
        assert cli.main(argv) == 0
        b = json.loads((out / "leaderboard.json").read_text())
        return [(r["name"], r["val_mean"]) for r in b["top_k"]], clock.now()

    a, spent_a = go(tmp_path / "a")
    b, spent_b = go(tmp_path / "b")
    assert a == b and spent_a == spent_b
    assert spent_a <= 60  # no run is started that would end after the budget (v0.18)
    assert cli.parse_budget_mode("400ep") == (400.0, "work")
    assert cli.parse_budget_mode("2m") == (120.0, "time")
    clock.start("time")


def test_profile_of_constant_or_duplicated_data_has_no_warnings_and_valid_bounds():
    """Was: RuntimeWarnings (0/0) and pca_dims_for_95pct_energy > n_features on constant data
    (seen with id-only recommender inputs)."""
    import warnings

    from dlnet import profile

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = profile.profile({"Xnp": {"tr": np.zeros((40, 3))}, "task": "classification",
                             "classes": [0, 1], "primary": "accuracy"})  # fmt: skip
        assert np.isnan(profile.twonn_intrinsic_dim(np.ones((30, 2))))
    assert p["pca_dims_for_95pct_energy"] <= 3 and p["scaling_exponent_hat"] is None
