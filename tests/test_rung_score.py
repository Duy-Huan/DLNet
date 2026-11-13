"""v1.5: ranking a rung of successive halving by the training speed estimate (dlnet/search.py)."""

from __future__ import annotations

import math
import types

import pytest

from dlnet import search


def _a(val, losses, name):
    return {
        "cfg": {"family": name},
        "val0": val,
        "tr": types.SimpleNamespace(epoch_losses=list(losses)),
    }


def test_tse_is_the_negated_sum_of_epoch_losses_like_the_authors_code():
    """rubinxin/TSE rank_correlation_comparison.py, method TSE: np.sum(train_loss[:epoch])."""
    tr = types.SimpleNamespace(epoch_losses=[2.0, 1.0, 0.5])
    assert search.tse(tr) == -3.5
    assert search.tse(types.SimpleNamespace(epoch_losses=[])) == -math.inf
    assert search.tse(types.SimpleNamespace(epoch_losses=[1.0, float("nan")])) == -math.inf


def test_rung_orders_by_val_tse_or_both():
    items = [
        _a(0.90, [1.0, 0.8], "fast_low_val"),
        _a(0.95, [2.0, 1.9], "slow_high_val"),
        _a(0.80, [1.5, 1.2], "mid"),
    ]
    assert [a["cfg"]["family"] for a in search._ranked(items)] == [
        "slow_high_val",
        "fast_low_val",
        "mid",
    ]
    assert [a["cfg"]["family"] for a in search._ranked(items, score="tse")] == [
        "fast_low_val",
        "mid",
        "slow_high_val",
    ]
    both = search._ranked(items, score="val+tse")
    assert both[0]["cfg"]["family"] == "fast_low_val"  # 1 + 2 points beats 2 + 0 and 0 + 1
    assert both[0]["rung_score"] == 3.0
    with pytest.raises(ValueError, match="rung score"):
        search._ranked(items, score="nope")
    # an objective decides alone (its trade-offs are in validation units)
    assert (
        search._ranked(items, rank=lambda pairs: [2, 0, 1], score="tse")[0]["cfg"]["family"]
        == "mid"
    )
    # ties: Borda points shared, validation breaks the tie
    tie = [_a(0.5, [1.0], "a"), _a(0.7, [1.0], "b")]
    assert [a["cfg"]["family"] for a in search._ranked(tie, score="tse")] == ["b", "a"]


def test_trainer_records_the_mean_loss_of_every_epoch():
    import numpy as np
    import torch

    from dlnet import core

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(64, 4)).astype("float32"), rng.integers(0, 2, 64)
    data = core.prepare(X, y, "classification", seed=0)
    tr = core.Trainer(torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)), data,
                      lr=1e-2, wd=0.0, batch=16, seed=0)  # fmt: skip
    tr.fit(2)
    tr.fit(1)
    assert len(tr.epoch_losses) == 3 and all(x > 0 for x in tr.epoch_losses)
    assert tr.epoch_losses[-1] == tr.last_train_loss and search.tse(tr) == -sum(tr.epoch_losses)
