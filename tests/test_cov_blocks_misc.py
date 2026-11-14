"""Refusals and alternate paths of the HSI, segmentation, v1.8, CNN and sequence blocks, the GAN
command, seed selection and the zero-cost proxies (v1.9 coverage)."""

from __future__ import annotations

import time
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dlnet import gan_run, proxies, seeds
from dlnet.zoo import dense, ops_hsi
from dlnet.zoo.core import _build


@pytest.mark.parametrize(("spec", "kind", "dims", "match"), [
    ({"op": "mambahsi_plus_block", "token_num": 3, "group_num": 2}, "map", (8, 5, 5), "divisible"),
    ({"op": "ssftt", "tokens": 4, "dim": 6, "heads": 4, "depth": 1}, "map", (8, 5, 5), "not divisible"),
    ({"op": "fahm", "size": "tiny", "groups": 2}, "map", (8, 5, 4), "square patches"),
    ({"op": "pconv_block", "n_div": 4}, "map", (2, 6, 6), "channels < n_div"),
    ({"op": "gated_cnn_block", "expansion": 0.5, "conv_ratio": 1.0}, "map", (8, 6, 6), "exceed the hidden width"),
    ({"op": "mobilevit_block", "tdim": 6, "heads": 4}, "map", (8, 6, 6), "not divisible"),
    ({"op": "sdta_block", "heads": 4, "scales": 2}, "map", (6, 6, 6), "channels"),
    ({"op": "regnet_block", "out": 24, "group_width": 16, "ratio": 1.0}, "map", (8, 6, 6), "group width"),
    ({"op": "shuffle_unit", "out": 32, "s": 1}, "map", (7, 6, 6), "even channel count"),
    ({"op": "shuffle_unit", "out": 33, "s": 2}, "map", (8, 6, 6), "out must be even"),
    ({"op": "inception", "out": 16}, "map", (8, 6, 6), "out >= 32"),
    ({"op": "patch_merge"}, "map", (8, 5, 6), "even height and width"),
    ({"op": "window_attn", "heads": 3}, "map", (8, 8, 8), "heads must divide"),
    ({"op": "gated_deltanet_block", "heads": 4}, "seq", (5, 6), "not divisible"),
    ({"op": "mlstm_block", "heads": 3, "proj_factor": 1.0}, "seq", (5, 64), "not divisible"),
    ({"op": "mambavision_block"}, "seq", (5, 7), "must be even"),
])  # fmt: skip
def test_block_refusals(spec, kind, dims, match):
    with pytest.raises(ValueError, match=match):
        _build(spec, kind, dims)


def test_fahm_and_fgem_group_checks():
    with pytest.raises(ValueError, match="not divisible by"):
        _build({"op": "fahm", "size": "tiny", "groups": 7}, "map", (8, 5, 5))
    with pytest.raises(ValueError, match="FGEM"):
        ops_hsi.FGEM(7, 8, n_groups=2)


def test_hsi_blocks_without_learned_branch_weights():
    m = ops_hsi.BothMamba(8, token_num=4, group_num=4, use_att=False)
    assert m(torch.randn(2, 8, 5, 5)).shape == (2, 8, 5, 5)
    rc = ops_hsi.SpaMambaPlus(8, 2)
    rc.use_att = False
    assert rc(torch.randn(2, 8, 5, 5)).shape == (2, 8, 5, 5)
    with pytest.raises(ValueError, match="GatedDeltaNet"):
        from dlnet.zoo.ops_seq2 import GatedDeltaNet

        GatedDeltaNet(6, heads=4)
    from dlnet.zoo.ops_seq2 import MambaVisionMixer

    with pytest.raises(ValueError, match="must be even"):
        MambaVisionMixer(5)


def test_mobilevit_on_a_map_that_needs_resizing():
    m, _, d = _build(
        {"op": "mobilevit_block", "tdim": 8, "heads": 2, "depth": 1, "patch": 2}, "map", (8, 7, 7)
    )
    assert d == (8, 7, 7) and m(torch.randn(2, 8, 7, 7)).shape == (2, 8, 7, 7)


def test_segmentation_decoder_refusals():
    assert dense.atrous_rates(64) == (12, 24, 36)
    with pytest.raises(ValueError, match="at least 4 channels"):
        dense.PSPDecoder([3, 2])
    with pytest.raises(ValueError, match="at least 4 input channels"):
        dense.LinkBlock(2, 8)
    st = [_build({"op": "conv", "out": 8, "k": 3, "s": 2, "pad": 1}, "map", (3, 16, 16))[0]]
    with pytest.raises(ValueError, match="unknown decoder attention"):
        dense.EncDec(st, [(8, 8, 8)], "unet", 8, (3, 16, 16), attention="nope")
    with pytest.raises(ValueError, match="at least 2 encoder stages"):
        dense.EncDec(st, [(8, 8, 8)], "unetpp", 8, (3, 16, 16))


def test_gan_without_a_valid_generator_and_a_spent_deadline(tmp_path, monkeypatch):
    np.savez(tmp_path / "x.npz", X=np.random.default_rng(0).random((30, 1, 5, 5)).astype("float32"))
    args = types.SimpleNamespace(out=str(tmp_path / "o"), data=str(tmp_path / "x.npz"), x_key=None, seed=0,
                                 candidates=3, budget_s=5.0, epochs=1)  # fmt: skip
    with pytest.raises(SystemExit, match="no valid generator"):
        gan_run.run(args)
    from dlnet.zoo import core

    real = core.check
    monkeypatch.setattr(
        core, "check", lambda *a: {**real(*a), "dims": (1, 4, 4)}
    )  # a wrong-sized image
    with pytest.raises(ValueError, match="generator output"):
        gan_run.build_pair({"width": 16, "gen_c": 32, "zdim": 16, "levels": 2, "batch": 32}, 1, 8)
    monkeypatch.setattr(core, "check", real)
    G, D = gan_run.build_pair(
        {"width": 16, "gen_c": 32, "zdim": 16, "levels": 2, "batch": 32}, 1, 8
    )
    X = torch.rand(40, 1, 8, 8)
    out = gan_run.train_gan(G, D, X, X, 16, epochs=3, batch=16, deadline=time.time() - 1)
    assert len(out["curve"]) == 1  # stopped after the first epoch


def test_seed_selection_corners():
    cands = [{"family": "resnet", "params": 10}] + [
        {"family": "mlp", "params": 10 + i} for i in range(4)
    ]
    card = {"position_dependence": 0.5}
    out, _ = seeds.select(cands, card, n=4, advice=[0.0, 1.0, 2.0, 3.0, 4.0], max_per_family=2)
    assert len(out) == 4 and out[0]["family"] == "resnet"  # the anchor first, then mlp past its cap


def test_proxies_for_regression_and_reconstruction():
    m = torch.nn.Linear(4, 1)
    X, y = torch.randn(8, 4), torch.randn(8)
    assert proxies.grad_norm(m, X, y, "regression") > 0
    r = torch.nn.Linear(4, 4)
    assert proxies.grad_norm(r, X, X, "reconstruction") > 0
    assert proxies.snip(r, X, X, "reconstruction") > 0
    assert proxies.zico(m, [(X, y)], "regression") is None  # one batch: no spread to measure
