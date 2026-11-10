"""Image and time-series plugins and the meta-learning recommender (M9)."""

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from dlnet import array_data, cli, meta, plugins, zoo
from dlnet.zoo import families


def _images(n=240, s=8, seed=0):
    """Two classes: a bright horizontal or vertical bar on noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.5, size=(n, s, s)).astype(np.float32)
    y = rng.integers(0, 2, n)
    for i in range(n):
        k = rng.integers(1, s - 1)
        if y[i]:
            X[i, k, :] += 2.0
        else:
            X[i, :, k] += 2.0
    return X, y


def _series(n=240, t=32, seed=0):
    """Two classes: slow or fast sine with a random phase, 2 channels."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    tt = np.arange(t)[None, :, None]
    freq = np.where(y, 0.9, 0.3)[:, None, None]
    X = np.sin(freq * tt + rng.uniform(0, 6.28, (n, 1, 1))) + rng.normal(scale=0.3, size=(n, t, 2))
    return X.astype(np.float32), y


@pytest.fixture
def image_npz(tmp_path):
    X, y = _images()
    np.savez(tmp_path / "img.npz", X=X, y=y)
    return tmp_path / "img.npz"


@pytest.fixture
def series_npz(tmp_path):
    X, y = _series()
    np.savez(tmp_path / "ts.npz", X=X, y=y)
    return tmp_path / "ts.npz"


def test_router_tells_images_series_and_hsi_apart(image_npz, series_npz, hsi_npz, tabular_csv):
    assert plugins.route(str(image_npz)).name == "image_classification"
    assert plugins.get_plugin("image").name == "image_classification"  # old name still works
    assert plugins.route(str(series_npz)).name == "timeseries"
    assert plugins.route(str(hsi_npz)).name == "hsi"
    assert plugins.route(str(tabular_csv)).name == "tabular"


def test_build_shapes_and_train_only_statistics():
    X, y = _images()
    d = array_data.build(X, y, "image", "classification", seed=0)
    assert d["Xtr"].shape[1:] == (1, 8, 8) and d["shape"] == [1, 8, 8] and d["in_dim"] == 1
    assert len(d["Xtr"]) + len(d["Xva"]) + len(d["Xte"]) == len(X)
    assert float(d["Xtr"].mean()) == pytest.approx(0.0, abs=1e-4)  # standardised on train
    assert float(d["Xtr"].std()) == pytest.approx(1.0, abs=1e-3)
    # an outlier in the test split must not change the train statistics (no leakage)
    X2 = X.copy()
    from dlnet import core

    _, _, te = core.stratified_split(
        core.encode_target(y, "classification")[0], "classification", 0
    )
    X2[te[0]] += 1000.0
    d2 = array_data.build(X2, y, "image", "classification", seed=0)
    assert torch.allclose(d["Xtr"], d2["Xtr"])
    s = array_data.build(*_series(), "timeseries", "classification", seed=0)
    assert s["Xtr"].shape[1:] == (2, 32) and s["Xnp"]["tr"].shape[1] == 64
    rgb = array_data.build(np.zeros((30, 6, 6, 3)) + X[:30, :6, :6, None], y[:30], "image",
                           "classification")  # fmt: skip
    assert rgb["Xtr"].shape[1:] == (3, 6, 6)


@pytest.mark.parametrize(
    ("domain", "fam"),
    [("image", f) for f in families.IMAGE]
    + [("timeseries", f) for f in families.TIMESERIES]
    + [("segmentation", f) for f in families.SEGMENTATION],
)
def test_models_forward_and_backward(domain, fam):
    shape = (2, 32) if domain == "timeseries" else (3, 8, 8)
    cfg = {"family": fam, "width": 32, "depth": 2, "dropout": 0.0, "patch": 4}
    m = families.build(cfg, domain, shape, 5)
    out = m(torch.randn(4, *shape))
    assert out.shape == ((4, 5, 8, 8) if domain == "segmentation" else (4, 5))
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_vit_rejects_patch_that_does_not_divide_and_sinusoid_formula():
    with pytest.raises(ValueError):
        families.build(
            {"family": "vit", "width": 32, "depth": 1, "dropout": 0.0, "patch": 3},
            "image",
            (1, 8, 8),
            2,
        )
    pe = zoo.ops_basic.SinePos(10, 8).pe[0]
    assert pe[0].tolist() == [0.0, 1.0] * 4  # sin(0), cos(0)
    assert float(pe[3, 0]) == pytest.approx(np.sin(3.0)) and float(pe[3, 1]) == pytest.approx(
        np.cos(3.0)
    )


@pytest.mark.parametrize("which", ["image_classification", "timeseries"])
def test_generate_only_valid_candidates(which, image_npz, series_npz):
    import argparse

    plugin = plugins.get_plugin(which)
    args = argparse.Namespace(
        data=str(image_npz if which == "image_classification" else series_npz), seed=0
    )
    data = plugin.prepare(args)
    prof = plugin.profile(data)
    cands = plugin.generate(prof, 2, 0)
    got = {c["family"] for c in cands}
    assert got <= set(plugin.families) and len(got) >= min(len(plugin.families) - 1, 6)
    assert all(c["params"] <= prof["max_params"] for c in cands)
    assert all(8 % c["patch"] == 0 for c in cands if c["family"] == "vit")
    ref = plugin.reference(data)
    assert ref and "score" in ref[0]["va"]


@pytest.mark.e2e
@pytest.mark.parametrize("which", ["image_classification", "timeseries"])
def test_cli_design_end_to_end_and_package_runs(which, image_npz, series_npz, tmp_path):
    data = image_npz if which == "image_classification" else series_npz
    out = tmp_path / "o"
    argv = ["design", "--data", str(data), "--budget", "45s", "--topk", "1", "--per-family", "1",
            "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    board = json.loads((out / "leaderboard.json").read_text())
    assert board["top_k"] and board["top_k"][0]["val_mean"] > 0.6  # the classes are separable
    pkg = next(out.glob("top1_*"))
    cfg = json.loads((pkg / "config.yaml").read_text())
    assert cfg["plugin"] == which and cfg["data"]["kind"] in ("image", "timeseries")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=pkg, capture_output=True,
                       check=False)  # fmt: skip
    assert r.returncode == 0, r.stdout.decode()[-500:]
    r = subprocess.run([sys.executable, "train.py", "--data", str(data), "--epochs", "2"], cwd=pkg,
                       capture_output=True, text=True, check=False)  # fmt: skip
    assert r.returncode == 0 and "test:" in r.stdout, r.stderr[-800:]


# ------------------------------------------------------------------------------ meta


def test_normalized_ranks_and_features():
    assert meta.normalized_ranks({"a": 0.9, "b": 0.7, "c": 0.5}) == {"a": 0.0, "b": 0.5, "c": 1.0}
    assert meta.normalized_ranks({"a": 0.9, "b": 0.9, "c": 0.5}) == {"a": 0.25, "b": 0.25, "c": 1.0}
    f = meta.features({"n_train": 1000, "n_features": 10, "intrinsic_dim": 4.2, "n_classes": None})
    assert f == [3.0, 1.0, 4.2, 0.0]


def _store():
    """Small datasets: family 'a' wins. Large datasets: family 'b' wins."""
    small = [{"plugin": "tabular", "name": f"s{i}", "features": [2.0 + 0.01 * i, 1.0, 3.0, 2.0],
              "family_scores": {"a": 0.9, "b": 0.6, "c": 0.5}} for i in range(4)]  # fmt: skip
    large = [{"plugin": "tabular", "name": f"l{i}", "features": [5.0 + 0.01 * i, 1.0, 3.0, 2.0],
              "family_scores": {"a": 0.6, "b": 0.9, "c": 0.5}} for i in range(4)]  # fmt: skip
    return small + large


def test_recommend_follows_the_nearest_datasets_and_never_prunes():
    recs = _store()
    r = meta.recommend(recs, "tabular", [2.0, 1.0, 3.0, 2.0], k=3)
    assert r["family_order"] == ["a", "b", "c"] and r["used_for_pruning"] is False
    assert meta.recommend(recs, "tabular", [5.0, 1.0, 3.0, 2.0], k=3)["family_order"][0] == "b"
    assert meta.recommend(recs, "hsi", [2.0, 1.0, 3.0, 2.0]) is None  # other plugin: no records
    assert meta.recommend(recs[:2], "tabular", [2.0, 1.0, 3.0, 2.0]) is None  # too few
    cands = [{"family": f, "i": i} for i in range(2) for f in ("c", "zzz", "a", "b")]
    out = meta.reorder(cands, ["a", "b", "c"])
    assert [c["family"] for c in out] == ["a", "a", "b", "b", "c", "c", "zzz", "zzz"]
    assert len(out) == len(cands)  # nothing removed
    loo = meta.leave_one_out(recs, "tabular", topk=1, k=3)
    assert loo["datasets"] == 8 and loo["hit_rate_knn"] == 1.0
    assert loo["hit_rate_chance"] == pytest.approx(1 / 3)


def test_store_roundtrip_ignores_damaged_lines(tmp_path):
    p = tmp_path / "m" / "store.jsonl"
    rec = meta.make_record("tabular", {"n_train": 100, "n_features": 4, "intrinsic_dim": 2.0},
                           [({"family": "a"}, 0.5), ({"family": "a"}, 0.7), ({"family": "b"}, 0.6),
                            ({"family": "c"}, float("nan"))], "d.csv")  # fmt: skip
    assert rec["family_scores"] == {"a": 0.7, "b": 0.6}
    assert meta.make_record("tabular", {}, [({"family": "a"}, 0.5)]) is None
    meta.append(p, rec)
    with p.open("a") as f:
        f.write("not json\n{}\n")
    assert meta.load(p) == [rec] and meta.load(tmp_path / "missing.jsonl") == []


def test_cli_meta_store_records_runs_and_reorders(tmp_path, tabular_csv, capsys):
    store = tmp_path / "store.jsonl"
    for r in _store()[:4]:  # past runs say: put tt_resmlp first
        meta.append(store, {**r, "family_scores": {"tt_resmlp": 0.9, "mlp": 0.5, "resmlp": 0.4}})
    argv = ["design", "--data", str(tabular_csv), "--target", "y", "--budget", "30s", "--topk", "1",
            "--per-family", "1", "--no-proxies", "--meta-store", str(store),
            "--out", str(tmp_path / "o")]  # fmt: skip
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "[meta] 4 past runs; family order ['tt_resmlp'" in out
    recs = meta.load(store)
    assert len(recs) == 5 and recs[-1]["name"] == "t.csv" and len(recs[-1]["family_scores"]) >= 2
    board = json.loads((tmp_path / "o" / "leaderboard.json").read_text())
    assert board["meta_recommendation"]["used_for_pruning"] is False
