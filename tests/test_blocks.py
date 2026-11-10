"""Block library (dlnet/blocks.py) and the image-segmentation plugin."""

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from dlnet import array_data, cli, core, plugins
from dlnet.plugins.arrays import pixel_features
from dlnet.zoo import count_params
from dlnet.zoo.families import tabular_model as build_model


def _seg_data(n=60, s=16, seed=0):
    """Background 0, one bright square (class 1) and one dark square (class 2) per image."""
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.4, size=(n, s, s)).astype(np.float32)
    y = np.zeros((n, s, s), dtype=np.int64)
    for i in range(n):
        for cls, val in ((1, 2.0), (2, -2.0)):
            r, c = rng.integers(0, s - 5, 2)
            y[i, r : r + 5, c : c + 5] = cls
            X[i, r : r + 5, c : c + 5] += val
    return X, y


@pytest.fixture
def seg_npz(tmp_path):
    X, y = _seg_data()
    np.savez(tmp_path / "seg.npz", X=X, y=y)
    return tmp_path / "seg.npz"


# ------------------------------------------------------------- recipes over the zoo


@pytest.mark.parametrize("fam", ["moe_mlp", "fm_mlp"])
def test_new_tabular_families_forward_backward(fam):
    cfg = {"family": fam, "width": 32, "depth": 2, "dropout": 0.0, "rank": 4, "experts": 3,
           "routing": "topk"}  # fmt: skip
    m = build_model(cfg, 6, 3)
    out = m(torch.randn(5, 6))
    assert out.shape == (5, 3) and count_params(m) > 0
    out.sum().backward()


def test_second_order_interaction_closed_form():
    """openrec SecondOrderFeatureInteraction through the zoo op: x, then the pairwise dot products."""
    from dlnet.zoo.ops_vec import FMInteraction

    f = FMInteraction(3, 2, False)
    with torch.no_grad():
        f.v.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]))
    x = torch.ones(1, 3)  # x_i v_i = v_i: pairs (0,1), (0,2), (1,2) -> 0, 3, 8
    assert f(x).tolist() == [[1.0, 1.0, 1.0, 0.0, 3.0, 8.0]]


# ------------------------------------------------------------------ segmentation


def test_miou_closed_form_and_ignored_pixels():
    y = np.array([[[0, 0], [1, 1]]])
    pred = np.array([[[0, 1], [1, 1]]])
    m = core.compute_metrics(y, pred, "classification", "miou")
    assert m["miou"] == pytest.approx((1 / 2 + 2 / 3) / 2) and m["score"] == m["miou"]
    assert m["accuracy"] == 0.75
    y2 = np.array([[[0, -100], [1, 1]]])  # the wrong pixel is ignored
    assert core.compute_metrics(y2, pred, "classification", "miou")["miou"] == 1.0
    assert "miou" not in core.compute_metrics(y.ravel(), pred.ravel(), "classification", "accuracy")


def test_segmentation_build_splits_whole_images_and_remaps_labels():
    X, y = _seg_data()
    y = np.where(y == 2, 7, y)  # labels 0, 1, 7 -> 0, 1, 2
    y[0, 0, 0] = -1  # ignored pixel
    d = array_data.build_segmentation(X, y, seed=0)
    assert d["classes"] == [0, 1, 7] and d["out_dim"] == 3 and d["primary"] == "miou"
    assert d["Xtr"].shape[1:] == (1, 16, 16) and d["ytr"].shape[1:] == (16, 16)
    assert len(d["Xtr"]) + len(d["Xva"]) + len(d["Xte"]) == len(X)  # images, not pixels
    allv = torch.cat([d["ytr"], d["yva"], d["yte"]])
    assert set(allv.unique().tolist()) == {-100, 0, 1, 2}
    assert array_data.kind_of(X, y) == "segmentation" and array_data.kind_of(X, y[:, :8]) is None
    assert pixel_features(d["Xtr"].numpy()).shape == (len(d["Xtr"]), 16, 16, 25)


def test_router_and_generate_for_segmentation(seg_npz):
    import argparse

    plugin = plugins.route(str(seg_npz))
    assert plugin.name == "image_segmentation"
    assert plugins.get_plugin("segmentation").name == "image_segmentation"
    data = plugin.prepare(argparse.Namespace(data=str(seg_npz), seed=0))
    prof = plugin.profile(data)
    cands = plugin.generate(prof, 2, 0)
    from dlnet.zoo.families import SEGMENTATION

    assert {c["family"] for c in cands} == set(SEGMENTATION)
    ref = plugin.reference(data)
    assert ref[0]["va"]["miou"] > 0.5  # the squares are easy for a 5x5 pixel classifier


@pytest.mark.e2e
def test_cli_segmentation_end_to_end_and_package_runs(seg_npz, tmp_path):
    out = tmp_path / "o"
    argv = ["design", "--data", str(seg_npz), "--budget", "60s", "--topk", "1", "--per-family", "1",
            "--zoo", "0", "--evolve", "0", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    board = json.loads((out / "leaderboard.json").read_text())
    assert board["metric"] == "miou" and board["top_k"][0]["val_mean"] > 0.5
    assert "miou" in board["top_k"][0]["test"] and "mIoU" in (out / "leaderboard.md").read_text()
    pkg = next(out.glob("top1_*"))
    assert (pkg / "zoo" / "core.py").exists()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=pkg, capture_output=True,
                       check=False)  # fmt: skip
    assert r.returncode == 0, r.stdout.decode()[-500:]
    r = subprocess.run([sys.executable, "train.py", "--data", str(seg_npz), "--epochs", "2"],
                       cwd=pkg, capture_output=True, text=True, check=False)  # fmt: skip
    assert r.returncode == 0 and "miou" in r.stdout, r.stderr[-800:]
