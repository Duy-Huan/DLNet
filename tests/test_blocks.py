"""Block library (dlnet/blocks.py) and the image-segmentation plugin."""

import itertools
import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from dlnet import array_data, blocks, cli, core, plugins
from dlnet.library import build_model, count_params
from dlnet.plugins.arrays import pixel_features


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


# ------------------------------------------------------------------------ blocks


@pytest.mark.parametrize("kind", sorted(blocks.STAGES))
@pytest.mark.parametrize("size", [8, 13, 28])
def test_encoder_levels_have_declared_channels_and_halving_sizes(kind, size):
    enc = blocks.Encoder(3, 8, 2, size, kind, attention="se")
    feats = enc(torch.randn(2, 3, size, size))
    assert [f.shape[1] for f in feats] == enc.channels and len(feats) == blocks.n_stages(size, 3)
    for a, b in itertools.pairwise(feats):
        assert b.shape[-1] == (a.shape[-1] + 1) // 2  # stride 2 per level, odd sizes round up


@pytest.mark.parametrize("decoder", sorted(blocks.DECODERS))
@pytest.mark.parametrize("hw", [(16, 16), (13, 20)])
def test_segmenter_returns_input_resolution_and_trains(decoder, hw):
    net = blocks.Segmenter(blocks.Encoder(1, 8, 1, min(hw), "resnet"), decoder, 3)
    out = net(torch.randn(2, 1, *hw))
    assert out.shape == (2, 3, *hw)
    out.sum().backward()
    assert all(p.grad is not None for p in net.parameters())


def test_attention_blocks_closed_form():
    se = blocks.SqueezeExcite(4)
    for p in se.parameters():
        nn.init.zeros_(p)  # MLP outputs 0 -> sigmoid 0.5 -> x / 2
    x = torch.randn(2, 4, 3, 3)
    assert torch.allclose(se(x), 0.5 * x)
    sa = blocks.SelfAttention2d(8)
    assert sa(torch.randn(2, 8, 4, 4)).shape == (2, 8, 4, 4)
    assert isinstance(blocks.make_attention("none", 8), nn.Identity)
    with pytest.raises(ValueError):
        blocks.make_attention("nope", 8)


def test_mixture_of_experts_soft_topk_and_product_closed_form():
    class Const(nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = torch.tensor(v)

        def forward(self, x):
            return self.v.expand(len(x), -1)

    class Gate(nn.Module):
        def forward(self, x):
            return torch.tensor([[0.0, math_log(3.0), -5.0]]).expand(len(x), -1)

    import math

    math_log = math.log
    experts = [Const([1.0, 0.0]), Const([0.0, 1.0]), Const([9.0, 9.0])]
    x = torch.zeros(2, 4)
    soft = blocks.MixtureOfExperts(experts, Gate(), "soft")
    p = torch.softmax(torch.tensor([0.0, math.log(3.0), -5.0]), 0)
    want = p[0] * torch.tensor([1.0, 0.0]) + p[1] * torch.tensor([0.0, 1.0]) + p[2] * 9.0
    assert torch.allclose(soft(x)[0], want) and torch.allclose(
        soft.gate_probs.sum(1), torch.ones(2)
    )
    top1 = blocks.MixtureOfExperts(experts, Gate(), "topk", k=1)
    assert torch.allclose(top1(x)[0], torch.tensor([0.0, 1.0]))  # hard routing to expert 2
    top2 = blocks.MixtureOfExperts(experts, Gate(), "topk", k=2)
    assert torch.allclose(top2(x)[0], torch.tensor([0.25, 0.75]))  # gate 1 : 3, expert 3 masked
    assert torch.equal(top2.gate_probs[0] > 0, torch.tensor([True, True, False]))
    prod = blocks.MixtureOfExperts(experts[:2], mode="product")
    # product of (e/(1+e), 1/(1+e)) and its mirror is uniform after renormalisation
    assert torch.allclose(torch.softmax(prod(x), 1)[0], torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError):
        blocks.MixtureOfExperts(experts, None, "soft")


def test_second_order_interaction_and_pairwise_loss_closed_form():
    f = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]])  # one sample, 3 features of dim 2
    assert blocks.SecondOrderInteraction()(f).tolist() == [[0.0, 3.0, 8.0]]  # pairs 01, 02, 12
    assert blocks.SecondOrderInteraction(True)(f).tolist() == [[1.0, 0.0, 3.0, 4.0, 8.0, 25.0]]
    u, p, n = torch.tensor([[1.0, 1.0]]), torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    assert float(blocks.pairwise_log_loss(u, p, n)) == pytest.approx(np.log(1 + np.exp(-1.0)))
    assert blocks.LatentFactor(10, 4)(torch.tensor([1, 2])).shape == (2, 4)


def test_neural_tree_and_mean_pooling():
    tree = blocks.NeuralTree(
        roots={"a": nn.Linear(3, 5), "b": nn.Linear(2, 7)},
        trunk=blocks.SumTrunk([5, 7], 6),
        branches={"shared": nn.Linear(6, 4)},
        leaves={"cls": nn.Linear(4, 3), "reg": nn.Linear(4, 1)},
        leaf_branch={"cls": "shared", "reg": "shared"},
    )
    out = tree({"a": torch.randn(2, 3), "b": torch.randn(2, 2)})
    assert out["cls"].shape == (2, 3) and out["reg"].shape == (2, 1)
    assert set(tree({"a": torch.randn(2, 3), "b": torch.randn(2, 2)}, leaf_keys=["reg"])) == {"reg"}
    with pytest.raises(KeyError):
        tree({"zzz": torch.randn(2, 3)})
    with pytest.raises(ValueError):
        blocks.NeuralTree({}, nn.Identity(), {}, {"l": nn.Identity()}, {"l": "missing"})
    x = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[True, True, False]])
    assert float(blocks.MeanPooling()(x, mask)) == 2.0 and float(
        blocks.MeanPooling()(x)
    ) == pytest.approx(104 / 3)


def test_catalog_lists_every_source_and_cli_prints_it(capsys):
    kinds = {c["kind"] for c in blocks.CATALOG}
    assert {"stage", "backbone", "decoder", "attention", "routing", "interaction"} <= kinds
    text = blocks.describe()
    for src in ("rasbt", "cortex", "openrec", "modular_deep_learning"):
        assert src in text
    assert cli.main(["blocks"]) == 0 and "mixture_of_experts" in capsys.readouterr().out


@pytest.mark.parametrize("fam", ["moe_mlp", "fm_mlp"])
def test_new_tabular_families_forward_backward(fam):
    cfg = {"family": fam, "width": 32, "depth": 2, "dropout": 0.0, "rank": 4, "experts": 3,
           "routing": "topk"}  # fmt: skip
    m = build_model(cfg, 6, 3)
    out = m(torch.randn(5, 6))
    assert out.shape == (5, 3) and count_params(m) > 0
    out.sum().backward()


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
    assert {c["family"] for c in cands} == {"fcn", "unet", "deconvnet"}
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
    assert (pkg / "blocks.py").exists()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=pkg, capture_output=True,
                       check=False)  # fmt: skip
    assert r.returncode == 0, r.stdout.decode()[-500:]
    r = subprocess.run([sys.executable, "train.py", "--data", str(seg_npz), "--epochs", "2"],
                       cwd=pkg, capture_output=True, text=True, check=False)  # fmt: skip
    assert r.returncode == 0 and "miou" in r.stdout, r.stderr[-800:]
