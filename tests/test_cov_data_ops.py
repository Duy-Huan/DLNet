"""Data builders (dlnet/hsi_data.py, dlnet/array_data.py), hyperparameter refinement
(dlnet/hparams.py) and the refusals of two op libraries (dlnet/zoo/ops_more.py, ops_basic.py):
the paths the end-to-end runs do not take (v1.9 coverage)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_audit_fixes import _cands, _data, _linear

from dlnet import array_data, clock, hparams, hsi_data
from dlnet.zoo import ops_basic
from dlnet.zoo.core import _build

# ----------------------------------------------------------------------------- HSI


def test_hsi_cube_loading_split_and_no_pca(tmp_path, cube_gt):
    cube, gt = cube_gt
    f = tmp_path / "scene.npz"
    np.savez(f, cube=cube, other=cube[..., :3], gt=gt)
    assert hsi_data.load_cube(str(f), "cube").shape == cube.shape
    with pytest.raises(ValueError, match="cannot pick the cube"):
        hsi_data.load_cube(str(f))
    np.savez(tmp_path / "one.npz", cube=cube, gt=gt)
    assert hsi_data.load_cube(str(tmp_path / "one.npz")).dtype == np.float32
    with pytest.raises(ValueError, match="'pixel' or 'block'"):
        hsi_data.make_split(gt, "random", 0, None, 5, 8)
    d = hsi_data.build(cube, gt, pca=0, patch=5)
    assert d["info"]["pca_energy_kept"] == 1.0


# ----------------------------------------------------------------------------- arrays


def test_array_data_runs_outside_the_package(monkeypatch):
    here = Path(array_data.__file__).parent
    monkeypatch.syspath_prepend(str(here))
    spec = importlib.util.spec_from_file_location("array_data_copy", here / "array_data.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        assert mod.core.__name__ == "core"
    finally:
        sys.modules.pop("core", None)


def test_array_builders_refuse_too_few_samples_and_bad_values():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="too few samples"):
        array_data.build(
            rng.normal(size=(2, 3, 4, 4)).astype("float32"),
            np.array([0, 1]),
            "image",
            "classification",
        )
    d = array_data.build(
        rng.normal(size=(20, 8)).astype("float32"), rng.normal(size=20), "series", "regression"
    )
    assert d["out_dim"] == 1 and d["primary"] == "neg_rmse"
    X = rng.normal(size=(4, 4, 4, 3)).astype("float32")  # channels last
    X[0, 0, 0, 0] = np.nan
    with pytest.raises(SystemExit, match="NaN"):
        array_data.build_segmentation(X, np.zeros((4, 4, 4)))
    with pytest.raises(ValueError, match="too few images"):
        array_data.build_segmentation(
            rng.normal(size=(2, 4, 4, 3)).astype("float32"), np.zeros((2, 4, 4))
        )
    with pytest.raises(ValueError, match="too few graphs"):
        array_data.build_graph(
            rng.normal(size=(2, 3, 7)).astype("float32"), np.array([0, 1]), "classification"
        )
    with pytest.raises(ValueError, match="too few interactions"):
        array_data.build_ids(np.array([[0, 1], [1, 0]]), np.array([0.0, 1.0]), "regression")


# ----------------------------------------------------------------------------- hyperparameters


class NanNet(torch.nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.lin = torch.nn.Linear(i, o)

    def forward(self, x):
        return self.lin(x) * float("nan")


def test_lr_range_test_without_a_usable_step():
    cfg = {**_cands(1)[0]}
    r = hparams.lr_range_test(
        _linear, cfg, _data(), batch=1, n_iter=4
    )  # batches of one: never stepped
    assert r["diverged"] and r["n_iter"] == 0
    r = hparams.lr_range_test(lambda c, i, o: NanNet(i, o), cfg, _data(), batch=8, n_iter=4)
    assert r["diverged"] and r["n_iter"] == 0


def test_refine_keeps_the_searched_values_when_the_estimate_fails_agrees_or_cannot_be_trained(
    monkeypatch,
):
    clock.start("time")
    d = _data()

    def result():
        return {
            "cfg": {**_cands(1)[0]},
            "val_mean": 0.8,
            "val_std": 0.01,
            "n_seeds": 2,
            "epochs": 2,
        }

    def boom(*a, **k):
        raise RuntimeError("estimator failed")

    monkeypatch.setattr(hparams, "estimate", boom)
    logs = []
    r = result()
    hparams.refine([r], d, _linear, math.inf, log=logs.append)
    assert r["hparams"]["adopted"] == "searched" and "estimate failed" in logs[0]
    monkeypatch.setattr(
        hparams,
        "estimate",
        lambda r, *a: {"lr": r["cfg"]["lr"], "batch": r["cfg"]["batch"], "wd": 0.0},
    )
    r = result()
    hparams.refine([r], d, _linear, math.inf, log=logs.append)
    assert r["hparams"]["estimated"]["lr"] == r["cfg"]["lr"]
    monkeypatch.setattr(hparams, "estimate", lambda r, *a: {"lr": 0.5, "batch": 8, "wd": 0.0})
    from dlnet import search

    monkeypatch.setattr(search, "eval_runs", lambda *a, **k: ([], [], 0.0, 0))
    r = result()
    hparams.refine([r], d, _linear, math.inf, log=logs.append)
    assert r["hparams"]["adopted"] == "searched" and r["hparams"]["val_estimated"] is None


# ----------------------------------------------------------------------------- ops refusals


@pytest.mark.parametrize(("spec", "kind", "dims", "match"), [
    ({"op": "gatv2", "out": 6, "heads": 4}, "graph", (5, 12), "heads must divide"),
    ({"op": "autoint_attn", "out": 6, "heads": 4}, "seq", (3, 8), "heads must divide"),
    ({"op": "bilinear_fields"}, "seq", (1, 8), "at least two fields"),
    ({"op": "isab", "out": 6, "heads": 4}, "seq", (3, 8), "heads must divide"),
    ({"op": "attn_pool", "heads": 4}, "seq", (3, 6), "heads must divide"),
    ({"op": "inception_next_block", "branch_ratio": 0.125}, "map", (4, 8, 8), "leaves no branch"),
    ({"op": "pvt_block", "heads": 4}, "map", (6, 8, 8), "heads must divide"),
    ({"op": "blur_pool", "filt_size": 5}, "map", (4, 2, 2), "smaller than the reflect padding"),
    ({"op": "simam"}, "map", (4, 1, 1), "more than one position"),
    ({"op": "feature_tokens"}, "vec", (200,), "too many features"),
])  # fmt: skip
def test_op_refusals(spec, kind, dims, match):
    with pytest.raises(ValueError, match=match):
        _build(spec, kind, dims)


def test_basic_helpers():
    with pytest.raises(ValueError, match="unknown activation"):
        ops_basic.act("nope")
    with pytest.raises(ValueError, match="unknown norm"):
        ops_basic.norm_layer("map", 8, "nope")
    m = ops_basic.MLPBlock(4, 8, 1, "relu", 0.0, out=3)
    assert m(torch.randn(2, 4)).shape == (2, 3)
