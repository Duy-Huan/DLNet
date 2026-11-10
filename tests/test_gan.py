"""DCGAN command: MMD^2 closed forms, generator/discriminator shapes, training, exported package."""

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from dlnet import cli, gan_run


def blobs(n=300, s=8, seed=0):
    """Images that are one bright blob on a dark background at one of two places."""
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.1, size=(n, s, s)).astype(np.float32)
    left = rng.random(n) < 0.5
    X[left, 1:4, 1:4] += 1.0
    X[~left, 4:7, 4:7] += 1.0
    return X


def test_mmd_is_near_zero_for_the_same_distribution_and_large_for_different_ones():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(300, 10, generator=g), torch.randn(300, 10, generator=g)
    same = gan_run.mmd2(a, b)
    shifted = gan_run.mmd2(a, b + 2.0)
    assert (
        abs(same) < 0.05 and shifted > 20 * abs(same) + 0.5
    )  # unbiased estimator: noise around 0, clear signal when shifted
    assert gan_run.mmd2(a, b) == pytest.approx(gan_run.mmd2(b, a), abs=1e-9)  # symmetric


def test_generator_and_discriminator_shapes_and_invalid_sizes():
    cfg = {"width": 16, "gen_c": 32, "zdim": 8, "levels": 2, "batch": 16}
    g, d = gan_run.build_pair(cfg, 1, 8)
    assert g(torch.randn(3, 8)).shape == (3, 1, 8, 8) and d(torch.randn(3, 1, 8, 8)).shape == (3, 1)
    assert float(g(torch.randn(3, 8)).abs().max()) <= 1.0  # tanh output
    with pytest.raises(ValueError, match="do not reach"):
        gan_run.build_pair({**cfg, "levels": 3}, 1, 12)


def test_training_improves_the_validation_mmd():
    X = torch.from_numpy(
        2 * (blobs() - blobs().min()) / (blobs().max() - blobs().min()) - 1
    ).unsqueeze(1)
    tr, va = X[:200], X[200:]
    cfg = {"width": 16, "gen_c": 32, "zdim": 8, "levels": 2, "batch": 32}
    torch.manual_seed(0)
    g, d = gan_run.build_pair(cfg, 1, 8)
    g.eval()
    with torch.no_grad():
        before = gan_run.mmd2(g(torch.randn(len(va), 8)), va)
    res = gan_run.train_gan(g, d, tr, va, 8, epochs=15, batch=32, seed=0)
    assert res["best_mmd"] < before and len(res["curve"]) == 15 and res["state"]


@pytest.mark.e2e
def test_cli_gan_end_to_end_and_package_runs(tmp_path):
    np.savez(tmp_path / "x.npz", X=blobs())
    out = tmp_path / "o"
    argv = [
        "gan",
        "--data",
        str(tmp_path / "x.npz"),
        "--candidates",
        "2",
        "--epochs",
        "4",
        "--seeds",
        "1",
        "--budget-s",
        "60",
        "--topk",
        "1",
        "--out",
        str(out),
    ]
    assert cli.main(argv) == 0
    board = json.loads((out / "leaderboard.json").read_text())
    assert (
        set(board["baselines_test_mmd"]) == {"gaussian_fit", "training_images"}
        and board["top_k"][0]["test_mmd"] > 0
    )
    assert (
        board["baselines_test_mmd"]["training_images"] < board["baselines_test_mmd"]["gaussian_fit"]
    )  # memorising beats a Gaussian
    pkg = next(out.glob("top1_*"))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"], cwd=pkg, capture_output=True, check=False
    )
    assert r.returncode == 0, r.stdout.decode()[-600:]
    r = subprocess.run(
        [sys.executable, "sample.py", "--n", "5"],
        cwd=pkg,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0 and np.load(pkg / "samples.npy").shape[0] == 5, r.stderr[-600:]
    r = subprocess.run(
        [sys.executable, "train.py", "--data", str(tmp_path / "x.npz"), "--epochs", "1"],
        cwd=pkg,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0 and "val MMD" in r.stdout, r.stderr[-600:]
