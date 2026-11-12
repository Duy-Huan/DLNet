"""Block diagrams (dlnet/diagram.py) and the training notebook (dlnet/notebook.py) of an output."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_standalone import DIMS, FAMILIES
from test_zoo import CASES, _cfg, _probe

from dlnet import cli, diagram, notebook
from dlnet.zoo import families


def _texts(svg):
    root = ET.fromstring(svg)  # well-formed XML
    return [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]


@pytest.mark.parametrize(("name", "kind"), CASES)
def test_every_op_is_drawn_with_its_output_shape(name, kind):
    cfg = _cfg({"op": name}, kind, _probe(name, kind))
    texts = _texts(diagram.svg(cfg))
    assert "Input" in texts and any(t and "->" in t for t in texts)  # input box and the head


@pytest.mark.parametrize(("domain", "family"), FAMILIES)
def test_every_family_is_drawn(domain, family):
    cfg = {
        "family": family,
        "width": 32,
        "depth": 2,
        "dropout": 0.1,
        "patch": 4,
        "rank_ratio": 0.5,
        "latent": 8,
    }
    c = families.complete(cfg, domain, DIMS[domain], 3)
    assert len(_texts(diagram.svg(c))) > 5


def test_residual_parallel_and_repeat_are_drawn_as_such():
    spec = [{"op": "conv_bn_act", "out": 8},
            {"op": "res", "body": [{"op": "conv_bn_act", "out": 8}, {"op": "conv_bn_act", "out": 8, "act": "identity"}]},
            {"op": "par", "merge": "concat", "branches": [[{"op": "conv_bn_act", "out": 4}], [{"op": "conv_bn_act", "out": 6, "k": 5}]]},
            {"op": "rep", "n": 3, "share": True, "body": {"op": "res", "body": [{"op": "conv_bn_act", "out": 10}]}},
            {"op": "pool", "mode": "max", "k": 2, "s": 2}, {"op": "gap"}]  # fmt: skip
    texts = _texts(diagram.svg(_cfg(spec, "map", (3, 16, 16))))
    assert texts.count("residual") == 2 and texts.count("+") == 2  # one merge node per residual
    assert "concat" in texts and "x 3 (shared weights)" in texts
    assert (
        "10 x 16 x 16" in texts and "10 x 8 x 8" in texts
    )  # traced shapes, before / after the pool
    assert "Conv 5x5, 6" in texts and "BN" in texts  # identity activation is not written


def test_notebook_cells_are_valid_python(tmp_path):
    out = _design(tmp_path)
    nb = json.loads((out / "baselines.ipynb").read_text())
    assert nb["nbformat"] == 4
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "cell", "exec")
    assert all((p / "architecture.svg").exists() for p in out.glob("top*_*"))


def _design(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 150)
    X = rng.normal(size=(150, 12, 12)).astype(np.float32)
    X[np.arange(150), y * 4, y * 4] += 4  # a bright pixel per class
    np.savez(tmp_path / "img.npz", X=X, y=y)
    out = tmp_path / "o"
    argv = ["design", "--data", str(tmp_path / "img.npz"), "--budget", "60ep", "--topk", "2", "--per-family", "1",
            "--zoo", "2", "--evolve", "0", "--no-proxies", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    return out


@pytest.mark.e2e
def test_notebook_runs_end_to_end_and_reloads_what_it_trained(tmp_path):
    """The notebook of a real `design` run, executed as a script (1 epoch per baseline)."""
    out = _design(tmp_path)
    code = notebook.script(out / "baselines.ipynb").replace("EPOCHS = None", "EPOCHS = 1")
    run = tmp_path / "run.py"
    run.write_text(
        "try:\n    import matplotlib\n    matplotlib.use('Agg')\nexcept ImportError:\n    pass\n"
        + code
    )
    env = {**os.environ, "DLNET_NO_INSTALL": "1", "OMP_NUM_THREADS": "2"}
    r = subprocess.run(
        [sys.executable, str(run)], cwd=out, capture_output=True, text=True, env=env, check=False
    )
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert "train / val / test" in r.stdout and "test accuracy" in r.stdout
    names = [x["name"] for x in json.loads((out / "leaderboard.json").read_text())["top_k"]]
    assert all((out / n / "model.pt").exists() for n in names)
