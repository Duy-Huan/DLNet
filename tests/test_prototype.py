import time

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")

import numpy as np

from dlnet import core, export, generate, profile, reference, search
from dlnet.plugins.tabular import TabularPlugin
from dlnet.zoo.ops_vec import TTLinear


def _data(task="classification"):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 8)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 0).astype(int) if task == "classification" else X[:, 0] * 2 + X[:, 1]
    return core.prepare(X, y, task, seed=0)


def test_tt_linear_shape_and_params():
    layer = TTLinear(64, 64, rank=4)
    import torch

    assert layer(torch.randn(3, 64)).shape == (3, 64)
    assert sum(p.numel() for p in layer.parameters()) < 64 * 64


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_end_to_end_topk(tmp_path, task):
    data = _data(task)
    prof = profile.profile(data)
    cands = generate.generate(prof, n_per_family=1)
    assert cands and all(c["params"] <= prof["max_params"] for c in cands)
    res = search.run_search(
        data,
        cands,
        time.time() + 120,
        2,
        TabularPlugin().build_model,
        rungs=(1, 2),
        seeds=2,
        log=lambda *_: None,
    )
    chosen = search.select_topk(res, 2)
    ref = reference.run_reference(data)
    v = export.export(tmp_path / "out", TabularPlugin(), data, prof, chosen, ref, seed=0)
    assert "beats_reference" in v
    assert (tmp_path / "out" / "leaderboard.md").exists()
    assert list((tmp_path / "out").glob("top1_*"))
