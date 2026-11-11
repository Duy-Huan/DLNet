"""Exported packages are usable on their own: train -> model.pt + preprocess.json -> predict.py
on a new file without labels -> labels in the original units (audit F5, qa/reports/v0.19.0).

Every plugin: `dlnet design` (tiny epoch budget), then `train.py` and `predict.py` in a
subprocess, started from ANOTHER folder (train.py used to need the package as working directory).
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from dlnet import cli

pytestmark = pytest.mark.e2e


def _run(args, cwd):
    p = subprocess.run(
        [sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=600, check=False
    )
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    return p.stdout


def _design(data, out, *extra):
    argv = ["design", "--data", str(data), "--out", str(out), "--budget", "30ep", "--topk", "1",
            "--per-family", "1", "--zoo", "2", "--evolve", "0", "--no-proxies", "--threads", "1",
            "--no-builtin-meta", *extra]  # fmt: skip
    assert cli.main(argv) == 0
    return next(out.glob("top1_*"))


def _image_npz(path, n=90, rgb=False, labels=True):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), n // 3)
    X = rng.normal(scale=0.3, size=(n, 8, 8, 3) if rgb else (n, 8, 8)).astype("float32")
    for i, c in enumerate(y):
        X[i, (c // 2) * 4 : (c // 2) * 4 + 4, (c % 2) * 4 : (c % 2) * 4 + 4] += 2.0
    np.savez(path, **({"X": X, "y": y * 10 + 1} if labels else {"X": X}))  # labels 1, 11, 21
    return X, y * 10 + 1


@pytest.mark.parametrize("rgb", [False, True])
def test_image_package_trains_saves_and_predicts_original_labels(tmp_path, rgb):
    _image_npz(tmp_path / "train.npz", rgb=rgb)
    pkg = _design(tmp_path / "train.npz", tmp_path / "o")
    _run([str(pkg / "train.py"), "--data", str(tmp_path / "train.npz"), "--epochs", "30"], tmp_path)
    assert (pkg / "model.pt").exists() and (pkg / "preprocess.json").exists()
    cfg = json.loads((pkg / "config.yaml").read_text())
    assert cfg["recipe"]["name"] == "modern" and cfg["recipe"]["label_smoothing"] == 0.1
    _, yn = _image_npz(tmp_path / "new.npz", n=30, rgb=rgb, labels=False)
    _run([str(pkg / "predict.py"), "--data", str(tmp_path / "new.npz"), "--out", str(tmp_path / "p.npy"),
          "--export", str(tmp_path / "m.pt2")], tmp_path)  # fmt: skip
    pred = np.load(tmp_path / "p.npy")
    assert set(pred) <= {1, 11, 21} and (pred == yn).mean() > 0.6
    import torch

    prog = torch.export.load(str(tmp_path / "m.pt2"))  # the exported graph runs without model.py
    assert prog.module()(torch.zeros(2, 3 if rgb else 1, 8, 8)).shape == (2, 3)


def test_tabular_package_predicts_from_a_csv_without_the_target(tmp_path, tabular_csv):
    import pandas as pd

    pkg = _design(tabular_csv, tmp_path / "o", "--target", "y")
    _run(
        [str(pkg / "train.py"), "--data", str(tabular_csv), "--target", "y", "--epochs", "5"],
        tmp_path,
    )
    df = pd.read_csv(tabular_csv)
    new = df.drop(columns=["y"]).head(40)
    new.loc[0, "f1"] = np.nan  # missing value: imputed with the TRAIN median
    new.to_csv(tmp_path / "new.csv", index=False)
    _run(
        [
            str(pkg / "predict.py"),
            "--data",
            str(tmp_path / "new.csv"),
            "--out",
            str(tmp_path / "p.csv"),
        ],
        tmp_path,
    )
    pred = np.loadtxt(tmp_path / "p.csv", delimiter=",")
    assert len(pred) == 40 and set(pred) <= {0, 1} and (pred == df["y"].head(40)).mean() > 0.6


def test_other_plugins_predict(tmp_path, hsi_npz):
    rng = np.random.default_rng(0)
    cases = []
    # time series
    y = rng.integers(0, 2, 90)
    t = np.linspace(0, 1, 32)
    X = np.stack([np.sin(2 * np.pi * (2 + 4 * c) * t) for c in y])[:, :, None] + 0.1 * rng.normal(
        size=(90, 32, 1)
    )
    np.savez(tmp_path / "ts.npz", X=X.astype("float32"), y=y)
    np.savez(tmp_path / "ts_new.npz", X=X[:10].astype("float32"))
    cases.append(("ts.npz", "ts_new.npz", [], (10,)))
    # segmentation
    Xs = rng.normal(size=(30, 8, 8)).astype("float32")
    m = (Xs > 0).astype(int) * 2  # mask values 0 / 2
    np.savez(tmp_path / "seg.npz", X=Xs, y=m)
    np.savez(tmp_path / "seg_new.npz", X=Xs[:4])
    cases.append(("seg.npz", "seg_new.npz", [], (4, 8, 8)))
    # graphs
    A = (rng.random((60, 6, 6)) > 0.5).astype("float32")
    Xg = rng.normal(size=(60, 6, 2)).astype("float32")
    np.savez(tmp_path / "g.npz", X=Xg, A=A, y=rng.integers(0, 2, 60))
    np.savez(tmp_path / "g_new.npz", X=Xg[:5], A=A[:5])
    cases.append(("g.npz", "g_new.npz", [], (5,)))
    # recommender
    ids = rng.integers(0, 10, (300, 2))
    np.savez(tmp_path / "rs.npz", ids=ids, y=rng.normal(size=300).astype("float32"))
    np.savez(tmp_path / "rs_new.npz", ids=ids[:7])
    cases.append(("rs.npz", "rs_new.npz", [], (7,)))
    # reconstruction
    Xr = rng.normal(size=(60, 8, 8)).astype("float32")
    np.savez(tmp_path / "rec.npz", X=Xr)
    np.savez(tmp_path / "rec_new.npz", X=Xr[:3])
    cases.append(("rec.npz", "rec_new.npz", ["--plugin", "image_reconstruction"], (3, 1, 8, 8)))
    # hyperspectral: a class map of the whole scene
    cases.append((str(hsi_npz), str(hsi_npz), [], (48, 48)))
    for i, (train, new, extra, shape) in enumerate(cases):
        pkg = _design(tmp_path / train, tmp_path / f"o{i}", *extra)
        _run([str(pkg / "train.py"), "--data", str(tmp_path / train), "--epochs", "2"], tmp_path)
        _run(
            [
                str(pkg / "predict.py"),
                "--data",
                str(tmp_path / new),
                "--out",
                str(tmp_path / f"p{i}.npy"),
            ],
            tmp_path,
        )
        pred = np.load(tmp_path / f"p{i}.npy")
        assert pred.shape == shape, (train, pred.shape)
        if train == "seg.npz":
            assert set(np.unique(pred)) <= {0, 2}  # original mask values
