"""End-to-end: run the real CLI in a subprocess, then use the exported package on its own.

This is the user's actual path (data -> Top-K baselines -> train.py), so it guards the contract
that exported packages are self-contained and reproduce the data used in the search.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")
pytest.importorskip("scipy")


def run(args, cwd, timeout=600):
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def check_package(pkg, train_args):
    for f in ("model.py", "core.py", "train.py", "config.yaml", "report.md"):
        assert (pkg / f).exists(), f
    cfg = json.loads((pkg / "config.yaml").read_text())
    assert cfg["arch"]["family"] in pkg.name
    t = run(["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], pkg)
    assert t.returncode == 0, t.stdout + t.stderr
    r = run(["train.py", *train_args, "--epochs", "2"], pkg)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "test:" in r.stdout and "val :" in r.stdout


def check_outputs(out):
    lb = json.loads((out / "leaderboard.json").read_text())
    assert lb["top_k"] and lb["reference"] and "beats_reference" in lb["verdict"]
    assert (out / "leaderboard.md").exists() and (out / "profile.json").exists()
    ranks = [r["rank"] for r in lb["top_k"]]
    assert ranks == list(range(1, len(ranks) + 1))
    assert len({r["name"] for r in lb["top_k"]}) == len(ranks)
    return lb


def test_tabular_cli_to_package(tabular_csv, tmp_path):
    out = tmp_path / "o"
    p = run(["-m", "dlnet", "design", "--data", str(tabular_csv), "--target", "y",
             "--budget", "40s", "--topk", "2", "--per-family", "1", "--out", str(out)],
            tmp_path)  # fmt: skip
    assert p.returncode == 0, p.stdout + p.stderr
    assert "[verdict]" in p.stdout
    lb = check_outputs(out)
    pkg = out / lb["top_k"][0]["name"]
    check_package(pkg, ["--data", str(tabular_csv), "--target", "y"])


def test_hsi_cli_to_package(hsi_npz, tmp_path):
    out = tmp_path / "o"
    p = run(["-m", "dlnet", "design", "--data", str(hsi_npz), "--budget", "60s", "--topk", "2",
             "--per-family", "1", "--patch", "5", "--pca", "12", "--out", str(out)],
            tmp_path)  # fmt: skip
    assert p.returncode == 0, p.stdout + p.stderr
    lb = check_outputs(out)
    pkg = out / lb["top_k"][0]["name"]
    assert (pkg / "hsi_data.py").exists()
    check_package(pkg, ["--data", str(hsi_npz)])


def test_cli_rerun_into_same_out_dir_keeps_layout(tabular_csv, tmp_path):
    out = tmp_path / "o"
    args = ["-m", "dlnet", "design", "--data", str(tabular_csv), "--target", "y", "--budget", "15s",
            "--topk", "1", "--per-family", "1", "--out", str(out)]  # fmt: skip
    assert run(args, tmp_path).returncode == 0
    assert run(args, tmp_path).returncode == 0
    assert len(list(out.glob("top*_*"))) == 1
