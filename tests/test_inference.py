"""dlnet/inference.py, the file copied into every output package (load, prepare_input, predict, export).

The end-to-end tests run it as a copy inside a package (predict.py), so its own lines were not
counted; here it is called in-process on each preprocessing type with small hand-made metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

DLNET = Path(__file__).resolve().parent.parent / "dlnet"


@pytest.fixture
def inf(monkeypatch, tmp_path_factory):
    # inside a package, core.py and hsi_data.py are copies next to inference.py
    import shutil
    import sys

    pkg = tmp_path_factory.mktemp("pkg")
    for f in ("core.py", "hsi_data.py"):
        shutil.copy(DLNET / f, pkg / f)
    monkeypatch.syspath_prepend(str(pkg))
    for m in ("core", "hsi_data", "model"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    from dlnet import inference

    return inference


class Lin(torch.nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.f = torch.nn.Flatten()
        self.l = torch.nn.LazyLinear(o)

    def forward(self, x):
        return self.l(self.f(x.float()))


def test_load_needs_trained_weights(inf, tmp_path):
    with pytest.raises(SystemExit, match="train.py"):
        inf.load(tmp_path)


def test_load_builds_the_package_model(inf, tmp_path, monkeypatch):
    (tmp_path / "model.py").write_text(
        "import torch\ndef build_model(arch, i, o):\n    return torch.nn.Linear(i, o)\n"
    )
    torch.save(torch.nn.Linear(3, 2).state_dict(), tmp_path / "model.pt")
    (tmp_path / "config.yaml").write_text(json.dumps({"arch": "x", "in_dim": 3, "out_dim": 2}))
    (tmp_path / "preprocess.json").write_text(json.dumps({"task": "classification"}))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(__import__("sys").modules, "model", raising=False)
    model, meta, cfg = inf.load(tmp_path)
    assert not model.training and meta["task"] == "classification" and cfg["in_dim"] == 3


def test_table(inf, tmp_path):
    csv = tmp_path / "new.csv"
    csv.write_text("a,b\n1.0,\n3.0,4.0\n")
    pre = {"type": "table", "columns": ["a", "b"], "target": None,
           "median": [2.0, 5.0], "mean": [2.0, 4.0], "std": [1.0, 2.0]}  # fmt: skip
    X = inf.prepare_input({"preprocess": pre}, {}, csv)
    assert torch.allclose(X, torch.tensor([[-1.0, 0.5], [1.0, 0.0]]))  # NaN -> train median 5


def test_channels_image_and_series(inf, tmp_path):
    p = tmp_path / "x.npz"
    np.savez(p, X=np.ones((4, 5, 6), np.uint8))
    pre = {"type": "channels", "layout": "image", "mean": [0.5], "std": [0.5]}
    X = inf.prepare_input({"preprocess": pre}, {}, p)
    assert X.shape == (4, 1, 5, 6) and torch.allclose(X, torch.ones_like(X))
    np.savez(p, X=np.zeros((3, 10), np.float32))
    pre = {"type": "channels", "layout": "series", "mean": [1.0], "std": [2.0]}
    X = inf.prepare_input({"preprocess": pre}, {"x_key": "X"}, p)
    assert X.shape == (3, 1, 10) and float(X[0, 0, 0]) == -0.5


def test_missing_array_key(inf, tmp_path):
    p = tmp_path / "x.npz"
    np.savez(p, Z=np.ones(3))
    with pytest.raises(SystemExit, match="'X' not found"):
        inf.prepare_input({"preprocess": {"type": "channels", "layout": "image"}}, {}, p)


def test_graph(inf, tmp_path):
    p = tmp_path / "g.npz"
    np.savez(p, X=np.ones((2, 4, 3), np.float32), A=np.eye(4, dtype=np.float32)[None].repeat(2, 0))
    pre = {"type": "graph", "features": 3, "mean": [0.0] * 3, "std": [1.0] * 3}
    X = inf.prepare_input({"preprocess": pre}, {}, p)
    assert X.shape == (2, 4, 7)
    pre["features"] = 5
    with pytest.raises(SystemExit, match="expects 5"):
        inf.prepare_input({"preprocess": pre}, {}, p)


def test_ids(inf, tmp_path):
    p = tmp_path / "r.npz"
    pre = {"type": "ids", "cards": [3, 4]}
    np.savez(p, ids=np.array([[0, 1], [2, 3]]))
    assert inf.prepare_input({"preprocess": pre}, {}, p).dtype == torch.int64
    np.savez(p, ids=np.array([[0, 1, 2]]))
    with pytest.raises(SystemExit, match="non-negative"):
        inf.prepare_input({"preprocess": pre}, {}, p)
    np.savez(p, ids=np.array([[3, 1]]))
    with pytest.raises(SystemExit, match="above the training range"):
        inf.prepare_input({"preprocess": pre}, {}, p)


def test_unknown_type(inf, tmp_path):
    with pytest.raises(SystemExit, match="unknown preprocessing"):
        inf.prepare_input({"preprocess": {"type": "audio"}}, {}, tmp_path / "x")


def test_hsi_class_map(inf, tmp_path):
    p = tmp_path / "cube.npz"
    np.savez(p, cube=np.random.default_rng(0).random((6, 5, 4)).astype(np.float32))
    pre = {"type": "hsi", "mean": [0.5] * 4, "std": [0.3] * 4, "components": np.eye(4)[:, :2].tolist(),
           "scale": [1.0, 1.0], "patch": 3}  # fmt: skip
    meta = {"preprocess": pre, "task": "classification", "classes": [1, 7]}
    store = inf.prepare_input(meta, {}, p)
    assert len(store) == 30 and store.map_shape == (6, 5)
    torch.manual_seed(0)
    pred = inf.predict(Lin(18, 2), meta, store, batch=7)
    assert pred.shape == (6, 5) and set(np.unique(pred)) <= {1, 7}
    pre["mean"] = [0.5] * 3
    with pytest.raises(SystemExit, match="4 bands"):
        inf.prepare_input(meta, {}, p)


def test_predict_tasks(inf, tmp_path):
    X = torch.randn(5, 3)
    m = torch.nn.Linear(3, 1)
    y = inf.predict(m, {"task": "regression", "y_std": 2.0, "y_mean": 10.0}, X, batch=2)
    assert y.shape == (5,) and np.allclose(y, m(X).detach().squeeze(-1).numpy() * 2 + 10, atol=1e-5)
    c = inf.predict(torch.nn.Linear(3, 2), {"task": "classification", "classes": ["a", "b"]}, X)
    assert set(c) <= {"a", "b"}
    img = torch.randn(2, 1, 4, 4)
    meta = {"task": "reconstruction", "preprocess": {"mean": [1.0], "std": [2.0]}}
    out = inf.predict(torch.nn.Identity(), meta, img)
    assert np.allclose(out, img.numpy() * 2 + 1)
    p = tmp_path / "x.npz"
    np.savez(p, X=np.zeros((2, 3), np.float32))
    meta = {"task": "classification", "classes": [0, 1],
            "preprocess": {"type": "channels", "layout": "series", "mean": [0.0], "std": [1.0]}}  # fmt: skip
    assert inf.predict_file(Lin(3, 2), meta, {}, p).shape == (2,)


def test_export_pt2_detaches_the_sample(inf, tmp_path):
    m = torch.nn.Linear(3, 2).eval()
    big = torch.randn(1000, 3)
    out = inf.export(m, big[:2], tmp_path / "m.pt2")
    assert out.exists() and out.stat().st_size < 200_000  # not the 1000-row storage
    prog = torch.export.load(str(out))
    x = torch.randn(2, 3)
    assert torch.allclose(prog.module()(x), m(x))
