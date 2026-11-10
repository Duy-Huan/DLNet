"""Shared fixtures: tiny synthetic datasets (no downloads, a few seconds each)."""

from __future__ import annotations

import numpy as np
import pytest


def synth_cube(h=48, w=48, bands=40, k=4, seed=0):
    rng = np.random.default_rng(seed)
    gt = np.zeros((h, w), dtype=np.int64)
    gt[: h // 2, : w // 2], gt[: h // 2, w // 2 :] = 1, 2
    gt[h // 2 :, : w // 2], gt[h // 2 :, w // 2 :] = 3, 4
    base = np.cumsum(rng.normal(size=(k, bands)), axis=1)
    cube = base[gt - 1] + rng.normal(scale=0.8, size=(h, w, bands))
    return cube.astype(np.float32), gt


@pytest.fixture
def cube_gt():
    return synth_cube()


@pytest.fixture
def tabular_csv(tmp_path):
    """200 rows, 6 numeric features, one NaN-containing column, binary target."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["y"] = y
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def hsi_npz(tmp_path, cube_gt):
    cube, gt = cube_gt
    p = tmp_path / "cube.npz"
    np.savez(p, cube=cube, gt=gt)
    return p
