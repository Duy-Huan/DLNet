"""Shared training utilities (dlnet/core.py, also copied into every exported package): the paths
the search does not take (v1.9 coverage)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from test_audit_fixes import _data

from dlnet import core


def test_core_runs_outside_the_package_without_a_budget():
    """An exported package has core.py next to train.py, without dlnet/clock.py."""
    spec = importlib.util.spec_from_file_location("core_copy", Path(core.__file__))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._spend(3) is None


def test_load_features_matches_the_training_columns(tmp_path):
    f = tmp_path / "new.csv"
    a = [str(i) for i in range(9)] + ["x"]  # 90 % numbers: a numeric column with one bad cell
    pd.DataFrame({"a": a, "c": ["u", "v"] * 5, "y": [0, 1] * 5}).to_csv(f, index=False)
    X = core.load_features(str(f), ["a", "b", "c_u", "c_v", "c_w"], target="y")
    assert X.shape == (10, 5) and X.flags.writeable
    assert (
        X[3, 0] == 3.0 and np.isnan(X[9, 0]) and np.isnan(X[0, 1])
    )  # "x" and b are left to impute
    assert X[0, 4] == 0.0  # a category never seen in this file is 0


def test_detect_task_of_labels():
    assert core.detect_task(np.array(["a", "b"])) == "classification"
    assert core.detect_task(np.array([True, False])) == "classification"


@pytest.mark.skipif(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
                    reason="this machine has MPS")  # fmt: skip
def test_pick_device_refuses_a_missing_mps():
    with pytest.raises(ValueError, match="MPS"):
        core.pick_device("mps")


def test_members_of_an_ensemble_output():
    m = torch.nn.Linear(2, 2)
    m.ensemble_k = 3
    out, y = core.members(m, torch.zeros(4, 3, 2), torch.arange(4))
    assert out.shape == (12, 2) and y.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_tiny_splits_and_a_last_batch_of_one():
    d = _data(n=10)
    d["Xtr"], d["ytr"] = d["Xtr"][:5], d["ytr"][:5]
    tr = core.Trainer(torch.nn.Linear(4, 2), d, lr=1e-2, batch=4)  # 5 rows: batches of 4 and 1
    tr.fit(1)
    d1 = _data(n=10)
    d1["Xtr"], d1["ytr"] = d1["Xtr"][:1], d1["ytr"][:1]
    assert core.Trainer(torch.nn.Linear(4, 2), d1, batch=4).probe_epoch_seconds() == 0.0


def test_save_artifacts_writes_plain_json(tmp_path):
    d = {"task": "classification", "kind": "vec", "classes": np.array([0, 1]), "y_mean": np.float32(0.5),
         "y_std": None, "shape": (4,), "preprocess": {"median": np.array([1.0, 2.0]), "cols": ("a", "b")}}  # fmt: skip
    core.save_artifacts(torch.nn.Linear(4, 2), d, tmp_path)
    meta = json.loads((tmp_path / "preprocess.json").read_text())
    assert (
        meta["classes"] == [0, 1]
        and meta["preprocess"]["cols"] == ["a", "b"]
        and meta["y_mean"] == 0.5
    )
    assert (tmp_path / "model.pt").exists()
