import time

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

import numpy as np
import torch

from dlnet import export, hsi_data, hsi_lib, search
from dlnet.plugins.hsi import HSIPlugin


def synth(h=48, w=48, bands=40, k=4, seed=0):
    """Tiny synthetic cube: k spatial regions, smooth class spectra, noise."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((h, w), dtype=np.int64)
    gt[: h // 2, : w // 2], gt[: h // 2, w // 2 :] = 1, 2
    gt[h // 2 :, : w // 2], gt[h // 2 :, w // 2 :] = 3, 4
    base = np.cumsum(rng.normal(size=(k, bands)), axis=1)
    cube = base[gt - 1] + rng.normal(scale=0.8, size=(h, w, bands))
    return cube.astype(np.float32), gt


def test_load_roundtrip(tmp_path):
    cube, gt = synth()
    np.savez(tmp_path / "x.npz", cube=cube, gt=gt)
    c2, g2 = hsi_data.load(str(tmp_path / "x.npz"))
    assert c2.shape == cube.shape and (g2 == gt).all()


def test_pixel_split_stratified():
    cube, gt = synth()
    d = hsi_data.build(cube, gt, split="pixel", patch=5, pca=10)
    assert d["info"]["classes_missing_train"] == [] and d["info"]["n_dropped_by_margin"] == 0
    assert d["Xtr"][torch.arange(3)].shape == (3, 10, 5, 5)


def test_block_split_has_no_patch_leakage():
    _, gt = synth(h=64, w=64)
    patch, block = 5, 8
    rows, cols, _, sid, keep, info = hsi_data.make_split(
        gt, "block", 0, (0.3, 0.2, 0.5), patch, block
    )
    assert info["n_dropped_by_margin"] > 0
    smap = np.full(gt.shape, -1)
    smap[rows, cols] = sid
    r = patch // 2
    for y, x, s in zip(rows[keep], cols[keep], sid[keep], strict=True):
        win = smap[max(y - r, 0) : y + r + 1, max(x - r, 0) : x + r + 1]
        assert set(np.unique(win[win >= 0])) <= {s}


def test_multilinear_rank_low_rank_cube():
    rng = np.random.default_rng(0)
    cube = np.einsum(
        "ir,jr,kr->ijk",
        rng.normal(size=(30, 2)),
        rng.normal(size=(30, 2)),
        rng.normal(size=(20, 2)),
    )
    assert max(hsi_data.multilinear_rank(cube.astype("float32"), 0.99)) <= 3


@pytest.mark.parametrize("family", hsi_lib.FAMILIES)
def test_models_forward(family):
    cfg = {"family": family, "width": 8, "depth": 2, "dropout": 0.0, "patch": 9, "rank_ratio": 0.5}
    if family == "composed":
        cfg.update(spectral="cnn1d", spatial="res", fusion="gated")
    m = hsi_lib.build_model(cfg, 20, 5).eval()
    assert m(torch.randn(3, 20, 9, 9)).shape == (3, 5)


def test_end_to_end(tmp_path):
    cube, gt = synth()
    np.savez(tmp_path / "x.npz", cube=cube, gt=gt)
    plugin = HSIPlugin()
    cube, gt = hsi_data.load(str(tmp_path / "x.npz"))
    data = hsi_data.build(cube, gt, split="pixel", patch=9, pca=15)
    data["info"]["multilinear_rank_99"] = hsi_data.multilinear_rank(cube)
    data["info"]["cube_shape"] = list(cube.shape)
    prof = plugin.profile(data)
    cands = plugin.generate(prof, 1, 0)
    assert {c["family"] for c in cands} >= {"cnn1d", "cnn2d", "tucker_cnn2d"}
    res = search.run_search(
        data,
        cands,
        time.time() + 300,
        2,
        plugin.build_model,
        rungs=(1, 2),
        seeds=2,
        log=lambda *_: None,
    )
    chosen = search.select_topk(res, 2)
    ref = plugin.reference(data)
    v = export.export(tmp_path / "out", plugin, data, prof, chosen, ref, seed=0)
    assert "beats_reference" in v
    pkg = next((tmp_path / "out").glob("top1_*"))
    assert (pkg / "hsi_data.py").exists() and (pkg / "model.py").exists()


# ------------------------------------------------------------ registry + grammar (M4)


def _prof(n_features=12, patch=5):
    return {"patch": patch, "n_features": n_features, "n_classes": 4, "max_params": 10**7}


def test_registry_generates_at_least_20_valid_candidates():
    """M4 criterion: >= 20 valid candidates, presets and grammar compositions, all runnable."""
    cands = HSIPlugin().generate(_prof(), 4, 0)
    new = [c for c in cands if c["family"] in hsi_lib.PRESETS or c["family"] == "composed"]
    assert len(new) >= 20 and {c["family"] for c in new} == {*hsi_lib.PRESETS, "composed"}
    x = torch.randn(2, 12, 5, 5)
    for c in new:
        m = hsi_lib.build_model(c, 12, 4)
        out = m(x)
        assert out.shape == (2, 4)
        out.sum().backward()
        assert hsi_lib.count_params(m) == c["params"]


def test_static_shape_inference_matches_the_built_blocks():
    x = torch.randn(2, 12, 5, 5)
    for cls in hsi_lib.BLOCKS.values():
        b = cls(12, 16, 2, 0.0, 5).eval()
        assert b(x).shape == (2, cls.features(12, 16, 2, 5))


@pytest.mark.parametrize(
    "bad",
    [
        {"spectral": "cnn1d", "spatial": "res"},  # two branches, no fusion
        {"spatial": "res", "fusion": "concat"},  # fusion with one branch
        {"joint": "ssrn", "spatial": "res", "fusion": "concat"},  # joint + branch
        {"spatial": "nope"},  # unknown block
        {},  # no encoder
    ],
)
def test_grammar_rejects_ill_typed_models_before_building(bad):
    with pytest.raises(ValueError):
        hsi_lib.plan_composed({"width": 16, "depth": 1, "patch": 5, **bad}, 12)


def test_ssrn_rejected_statically_when_too_few_bands():
    with pytest.raises(ValueError):
        hsi_lib.plan_composed({"width": 8, "depth": 1, "patch": 5, "joint": "ssrn"}, 5)
    cands = HSIPlugin().generate(_prof(n_features=5), 2, 0)
    assert "ssrn" not in {c["family"] for c in cands}


def test_search_interleaves_families():
    c = [{"family": f, "i": i} for f in "aab" for i in range(1)]
    assert [x["family"] for x in search.interleave(c)] == ["a", "b", "a"]
