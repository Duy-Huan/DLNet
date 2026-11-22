"""Hyperspectral super-resolution (plugin `hsi_sr`, v2.0): data protocol, metrics, models, search, package."""

from __future__ import annotations

import json
import subprocess
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from dlnet import (
    cli,
    core,
    diagram,
    export,
    hparams,
    hsi_sr_data,
    inference,
    plugins,
    proxies,
    search,
    seeds,
)
from dlnet.plugins.hsi_sr import HSISuperResolutionPlugin
from dlnet.zoo import core as zcore
from dlnet.zoo import families, space


def scene(h=64, w=64, bands=8, seed=0):
    """Smooth synthetic cube: a few spatial patterns mixed by random spectra, plus noise (>= 0)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    pats = np.stack(
        [np.sin(5 * xx * (k + 1) + 2 * yy) * np.cos(3 * yy * (k + 1)) for k in range(3)]
    )
    cube = np.einsum("khw,kb->hwb", pats, rng.random((3, bands))) + 0.05 * rng.random((h, w, bands))
    return (cube - cube.min()).astype(np.float32)


def args(path, **kw):
    a = {
        "data": str(path),
        "cube_key": None,
        "scale": 2,
        "sr_patch": 16,
        "degrade": "bicubic",
        "seed": 0,
    }
    return types.SimpleNamespace(**{**a, **kw})


# ----------------------------------------------------------------------------- data


def test_load_every_file_kind_and_key_rule(tmp_path):
    from scipy.io import savemat

    c = scene(16, 16, 4)
    np.savez(tmp_path / "a.npz", hr=c, other=np.zeros(3))
    np.savez(tmp_path / "b.npz", HR=np.stack([c, c]))
    np.savez(tmp_path / "c.npz", cube=c, vec=np.zeros(5))
    np.savez(tmp_path / "d.npz", a=c, b=c)
    np.save(tmp_path / "e.npy", c)
    savemat(tmp_path / "f.mat", {"img": c})
    assert hsi_sr_data.load(tmp_path / "a.npz").shape == (1, 16, 16, 4)
    assert hsi_sr_data.load(tmp_path / "b.npz").shape == (2, 16, 16, 4)
    assert hsi_sr_data.load(tmp_path / "c.npz").shape == (1, 16, 16, 4)
    assert hsi_sr_data.load(tmp_path / "d.npz", "b").shape == (1, 16, 16, 4)
    assert hsi_sr_data.load(tmp_path / "e.npy").shape == (1, 16, 16, 4)
    assert hsi_sr_data.load(tmp_path / "f.mat").shape == (1, 16, 16, 4)
    with pytest.raises(ValueError, match="cannot pick"):
        hsi_sr_data.load(tmp_path / "d.npz")
    with pytest.raises(ValueError, match="not found"):
        hsi_sr_data.load(tmp_path / "d.npz", "zz")
    with pytest.raises(ValueError, match="got shape"):
        hsi_sr_data.load(tmp_path / "c.npz", "vec")
    np.savez(tmp_path / "nan.npz", hr=np.full((4, 4, 2), np.nan))
    with pytest.raises(ValueError, match="NaN"):
        hsi_sr_data.load(tmp_path / "nan.npz")
    with pytest.raises(ValueError, match="unsupported"):
        hsi_sr_data.load(tmp_path / "x.csv")


def test_degradations_and_interpolation_keep_the_scale():
    hr = torch.rand(2, 3, 16, 16)
    for mode in hsi_sr_data.DEGRADATIONS:
        assert hsi_sr_data.degrade(hr, 2, mode).shape == (2, 3, 8, 8)
    flat = torch.full((1, 2, 12, 12), 0.7)  # a blur kernel sums to one: a constant stays constant
    assert torch.allclose(hsi_sr_data.degrade(flat, 3, "gauss"), torch.tensor(0.7))
    with pytest.raises(ValueError, match="unknown degradation"):
        hsi_sr_data.degrade(hr, 2, "box")
    lr = torch.rand(1, 3, 4, 4)
    for mode in ("nearest", "bilinear", "bicubic"):
        assert hsi_sr_data.upsample(lr, 3, mode).shape == (1, 3, 12, 12)


def test_tiles_do_not_overlap():
    c = np.arange(10 * 9 * 2, dtype=np.float32).reshape(10, 9, 2)
    t = hsi_sr_data.tiles(c, 4)
    assert t.shape == (4, 2, 4, 4)
    assert np.array_equal(t[1, 1], c[0:4, 4:8, 1])  # tile (0, 1), band 1


def test_build_splits_tiles_or_whole_cubes():
    d = hsi_sr_data.build(scene(64, 64)[None], scale=2, patch=16, seed=0)
    i = d["info"]
    assert i["split_by"] == "tile" and (i["n_train"], i["n_val"], i["n_test"]) == (10, 3, 3)
    assert d["Xtr"].shape == (10, 8, 8, 8) and d["ytr"].shape == (10, 8, 16, 16)
    assert d["preprocess"]["lo"] == 0.0 and d["ytr"].max() <= 1.0  # scaled by the training maximum
    cubes = np.stack([scene(32, 32, 4, s) - 0.5 for s in range(5)])  # negative values: shifted
    d = hsi_sr_data.build(cubes, scale=4, patch=16, degradation="gauss")
    assert d["info"]["split_by"] == "cube" and d["preprocess"]["lo"] < 0
    assert d["Xtr"].shape[-1] == 4 and len(d["Xtr"]) == 3 * 4  # 3 training cubes x 4 tiles


@pytest.mark.parametrize(("kw", "match"), [
    ({"scale": 1}, "at least 2"),
    ({"patch": 15}, "multiple"),
    ({"patch": 6, "scale": 2}, "at least 4"),
    ({"degradation": "box"}, "unknown degradation"),
    ({"patch": 128}, "smaller than one tile"),
    ({"patch": 64}, "too few tiles"),
])  # fmt: skip
def test_build_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        hsi_sr_data.build(scene(64, 64, 2)[None], **{"scale": 2, "patch": 16, **kw})


def test_tiles_stitch_back_a_whole_cube():
    pre = {"lo": 0.0, "hi": 2.0, "scale": 2, "tile": 4, "bands": 3}
    cube = np.random.default_rng(0).random((10, 7, 3)).astype(np.float32)
    t = hsi_sr_data.Tiles(cube, pre)
    assert len(t) == len(t.pos) and t[0:2].shape == (2, 3, 4, 4)
    up = torch.nn.functional.interpolate(t.tiles, scale_factor=2, mode="nearest")
    want = cube.repeat(2, 0).repeat(2, 1)
    assert np.allclose(t.stitch(up), want, atol=1e-6)  # overlaps average to the same values
    small = hsi_sr_data.Tiles(cube[:3, :2], pre)  # smaller than a tile: padded, then cropped
    up = torch.nn.functional.interpolate(small.tiles, scale_factor=2, mode="nearest")
    assert small.stitch(up).shape == (6, 4, 3)
    with pytest.raises(ValueError, match="need a cube"):
        hsi_sr_data.Tiles(cube[..., :2], pre)


# ----------------------------------------------------------------------- metrics, loss


def test_sr_metrics():
    t = np.random.default_rng(0).random((2, 4, 8, 8))
    m = core.compute_metrics(t, t, "super_resolution", "psnr", 2)
    assert m["psnr"] == pytest.approx(120.0) and m["ssim"] == pytest.approx(1.0)
    assert (
        m["sam"] == pytest.approx(0.0, abs=1e-5) and m["ergas"] == 0.0 and m["score"] == m["psnr"]
    )
    noisy = core.sr_metrics(t, t + 0.1, 4)
    assert noisy["psnr"] == pytest.approx(20.0) and noisy["rmse"] == pytest.approx(0.1)
    assert noisy["ssim"] < 1.0 and noisy["sam"] > 0 and noisy["ergas"] > 0
    z = np.zeros((1, 2, 3, 3))
    assert core.sr_metrics(z, z)["sam"] == 0.0  # no spectrum with a direction: no angle


def test_losses_and_noise_of_super_resolution():
    assert isinstance(core.loss_for("super_resolution"), torch.nn.L1Loss)
    assert isinstance(core.loss_for("reconstruction"), torch.nn.MSELoss)
    assert isinstance(core.loss_for("classification", 0.1), torch.nn.CrossEntropyLoss)
    assert isinstance(hparams._loss_fn({"task": "super_resolution"}), torch.nn.L1Loss)
    out, y = torch.zeros(2, 3), torch.ones(2, 3)
    assert float(proxies._loss(out, y, "super_resolution")) == 1.0
    assert search.sampling_noise(40.0, "psnr", 8) == pytest.approx(10 / np.log(10) * 0.5)


def test_trainer_flips_input_and_target_together():
    d = hsi_sr_data.build(scene(64, 64, 4)[None], scale=2, patch=16)
    m = families.build({"family": "vdsr", "width": 8, "depth": 1, "dropout": 0.0, "scale": 2}, "hsi_sr",
                       d["shape"], 4)  # fmt: skip
    tr = core.Trainer(m, d, batch=4, recipe="modern")
    assert tr.augment
    x = torch.arange(2 * 16, dtype=torch.float32).reshape(2, 1, 4, 4)
    y = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
    xa, ya = tr._dihedral(x, y)
    assert torch.equal(torch.nn.functional.interpolate(xa, scale_factor=2, mode="nearest"), ya)
    tr.fit(1)
    assert tr.evaluate("va")["psnr"] > 0


# --------------------------------------------------------------------------- models


def test_every_family_upsamples_and_is_drawn():
    for fam in families.HSI_SR:
        cfg = families.complete({"family": fam, "width": 8, "depth": 2, "dropout": 0.0, "scale": 3},
                                "hsi_sr", (5, 6, 6), 5)  # fmt: skip
        assert cfg["head"] == "sr3"
        assert zcore.build_model(cfg)(torch.randn(2, 5, 6, 6)).shape == (2, 5, 18, 18)
        assert "pixel shuffle x3" in diagram.svg(cfg)
    with pytest.raises(ValueError, match="unknown super-resolution family"):
        families.hsi_sr({"family": "nope", "width": 8, "depth": 1}, (5, 6, 6))


def test_sr_head_refusals():
    with pytest.raises(ValueError, match="needs a map"):
        zcore.Head("seq", (4, 8), 3, "sr2")
    down = {"spec": {"op": "conv_bn_act", "out": 8, "k": 3, "s": 2}, "input": {"kind": "map", "dims": [3, 8, 8]},
            "head": "sr2", "out_dim": 3}  # fmt: skip
    with pytest.raises(ValueError, match="keep the spatial size"):
        zcore.build_model(down)
    rng = np.random.default_rng(0)
    assert space._candidate(down["spec"], "map", (3, 8, 8), 3, "sr2", {}, rng, 10**6, "t") is None
    keep = {"op": "conv_bn_act", "out": 64, "k": 3, "s": 1}
    assert space._candidate(keep, "map", (3, 8, 8), 3, "sr2", {}, rng, 10**6, "t")["head"] == "sr2"
    with pytest.raises(ValueError, match="depth >= 1"):
        zcore._build({"op": "sr3d", "depth": 0}, "map", (3, 4, 4))


def test_seed_rule_message_without_classes():
    _, why = seeds.trait_weights({"n_train": 5})
    assert "5 training images: global attention" in why[0]
    _, why = seeds.trait_weights({"n_train": 5, "samples_per_class_min": 2})
    assert "(2 per class at least)" in why[0]


# --------------------------------------------------------------------------- plugin


def test_plugin_detect_prepare_profile_generate_reference(tmp_path):
    p = plugins.get_plugin("super_resolution")
    assert isinstance(p, HSISuperResolutionPlugin)
    np.savez(tmp_path / "s.npz", hr=scene(64, 64, 6))
    np.savez(tmp_path / "x.npz", X=np.zeros((4, 3)))
    (tmp_path / "bad.npz").write_bytes(b"not a zip")
    assert p.detect(tmp_path / "s.npz") == 0.97 and p.detect(tmp_path / "x.npz") == 0.0
    assert p.detect(tmp_path / "bad.npz") == 0.0 and p.detect(tmp_path / "s.mat") == 0.0
    assert plugins.route(str(tmp_path / "s.npz")).name == "hsi_sr"
    d = p.prepare(args(tmp_path / "s.npz"))
    assert d["warnings"] and d["keys"] == {"cube_key": None}
    assert p.data_config(d) == {"scale": 2, "patch": 16, "degradation": "bicubic", "cube_key": None}
    assert p.zoo_input(d) == ("map", (6, 8, 8), "sr2")
    prof = p.profile(d)
    assert prof["bands"] == 6 and prof["max_params"] > 2e5
    cands = p.generate(prof, 1, 0)
    assert {c["family"] for c in cands} == set(families.HSI_SR)
    many = p.generate(prof, 6, 0)  # srcnn has few settings: repeated draws are skipped
    assert len(many) > len(cands) and len(
        {tuple(sorted((k, str(v)) for k, v in c.items())) for c in many}
    ) == len(many)
    assert p.build_model(cands[0], 6, 6)(d["Xva"][:2]).shape == (2, 6, 16, 16)
    ref = p.reference(d)
    assert ref[0]["name"] == "bicubic_interpolation" and "psnr" in ref[0]["te"]
    with pytest.raises(SystemExit, match="hsi_sr: .*multiple"):
        p.prepare(args(tmp_path / "s.npz", sr_patch=15))
    np.savez(tmp_path / "many.npz", hr=np.stack([scene(32, 32, 3, s) for s in range(5)]))
    assert "warnings" not in p.prepare(args(tmp_path / "many.npz"))


def test_export_formats_super_resolution():
    m = {"psnr": 40.123, "ssim": 0.98, "sam": 1.5, "ergas": 2.0}
    assert export._fmt(m, "super_resolution").startswith("PSNR 40.12 dB")
    assert export._short(m, "super_resolution") == "psnr 40.123 dB, ssim 0.98, sam 1.5"


def test_inference_upsamples_a_low_resolution_cube(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(hsi_sr_data.__file__).rsplit("/", 1)[0]
    )  # the package's flat imports
    pre = {"type": "hsi_sr", "lo": 0.0, "hi": 1.0, "scale": 2, "tile": 4, "bands": 3}
    meta, cfg = {"task": "super_resolution", "preprocess": pre}, {"data": {"cube_key": None}}
    np.save(tmp_path / "lr.npy", np.random.default_rng(0).random((6, 5, 3)).astype(np.float32))
    X = inference.prepare_input(meta, cfg, str(tmp_path / "lr.npy"))
    model = torch.nn.Upsample(scale_factor=2, mode="nearest")
    assert inference.predict(model, meta, X, batch=3).shape == (12, 10, 3)
    np.save(tmp_path / "two.npy", np.zeros((2, 6, 5, 3), np.float32))
    with pytest.raises(SystemExit, match="one low-resolution cube"):
        inference.prepare_input(meta, cfg, str(tmp_path / "two.npy"))
    np.save(tmp_path / "bands.npy", np.zeros((6, 5, 4), np.float32))
    with pytest.raises(SystemExit, match="need a cube"):
        inference.prepare_input(meta, cfg, str(tmp_path / "bands.npy"))


@pytest.mark.e2e
def test_design_exports_a_package_that_trains_and_upscales(tmp_path):
    np.savez(tmp_path / "scene.npz", hr=scene(64, 64, 6))
    out = tmp_path / "out"
    rc = cli.main(["design", "--data", str(tmp_path / "scene.npz"), "--budget", "20", "--topk", "1",
                   "--sr-patch", "16", "--per-family", "1", "--zoo", "2", "--evolve", "0", "--out", str(out)])  # fmt: skip
    assert rc == 0
    board = json.loads((out / "leaderboard.json").read_text())
    assert board["task"] == "super_resolution" and board["metric"] == "psnr"
    assert (out / "baselines.ipynb").exists()
    pkg = next(out.glob("top1_*"))
    cfg = json.loads((pkg / "config.yaml").read_text())
    assert cfg["plugin"] == "hsi_sr" and cfg["data"]["scale"] == 2
    run = lambda *a: subprocess.run(
        [sys.executable, *a], cwd=pkg, capture_output=True, text=True, check=False
    )
    r = run("-m", "pytest", "-q", "-p", "no:cacheprovider", "tests")
    assert r.returncode == 0, r.stdout[-800:]
    r = run("train.py", "--data", str(tmp_path / "scene.npz"), "--epochs", "1")
    assert r.returncode == 0 and "test:" in r.stdout, r.stderr[-800:]
    np.save(tmp_path / "lr.npy", scene(20, 12, 6))
    r = run("predict.py", "--data", str(tmp_path / "lr.npy"), "--out", str(tmp_path / "sr.npy"))
    assert r.returncode == 0, r.stderr[-800:]
    assert np.load(tmp_path / "sr.npy").shape == (40, 24, 6)
