"""Paper cards (dlnet/paper.py): every mistake a card can hold, and the build / verify / compare paths
the other tests do not take (v1.9 coverage)."""

from __future__ import annotations

import json
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

from test_paper import CARD, VERIFIED, _card, _official

from dlnet import paper
from dlnet.paper import CardError

VEC = """
paper: {name: Vec Net, reported: [{params: 5}]}
input: {kind: vec, dims: [6]}
classes: 3
architecture:
  - {id: a, module: linear_act, args: {out: 8, act: relu}}
"""


def err(tmp_path, text, match, name="c.yaml"):
    with pytest.raises((CardError, SystemExit), match=match):
        card = paper.load(_card(tmp_path, text, name=name))
        r = paper.check(card)
        if not r["ok"]:
            raise CardError("; ".join(r["issues"]))


def test_loading(tmp_path, monkeypatch):
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"paper": {"name": "J"}, "input": {"kind": "vec", "dims": [3]}, "classes": 2,
                             "architecture": [{"id": "a", "module": "linear", "args": {"out": 4}}]}))  # fmt: skip
    assert paper.load(j)["paper"]["name"] == "J"
    (tmp_path / "list.yaml").write_text("- a\n- b\n")
    with pytest.raises(CardError, match="a card is a mapping"):
        paper.load(tmp_path / "list.yaml")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(SystemExit, match="PyYAML"):
        paper.load(_card(tmp_path))


@pytest.mark.parametrize(("old", "new", "match"), [
    ("blocks:\n  res8:\n", "blocks:\n  empty: []\n  res8:\n", "non-empty list"),
    ("- {id: pool, module: gap}", "- {id: pool}", "at least `module`"),
    ("- {id: pool, module: gap}", "- {id: stem, module: gap}", "used twice"),
    ("- {id: pool, module: gap}", "- {id: pool, module: gap, args: [1]}", "`args` is a mapping"),
    ("- {id: cat, from: [body, side], module: concat}", "- {id: cat, from: [body], module: concat}", "needs `from"),
    ("input: {kind: map, dims: [3, 8, 8]}", "", "`input:"),
    ("    - {id: c1, module: conv_bn_act, args: {out: 8, k: 3}}", "    - {id: c1, module: res8}", "uses itself"),
    ("    - {id: c1, module: conv_bn_act, args: {out: 8, k: 3}}",
     "    - {id: c1, module: conv_bn_act, args: {out: '$nope', k: 3}}", "nope"),
    ("    - {id: c1, module: conv_bn_act, args: {out: 8, k: 3}}",
     "    - {id: c1, module: conv_bn_act, args: {out: '$w ** 2', k: 3}}", "not a simple expression"),
    ("    - {id: c1, module: conv_bn_act, args: {out: 8, k: 3}}",
     "    - {id: c1, module: conv_bn_act, args: {out: '$w +', k: 3}}", "cannot read"),
])  # fmt: skip
def test_card_mistakes(tmp_path, old, new, match):
    text = CARD.replace("PARAMS", "2692")
    assert old in text
    text = text.replace(old, new)
    if "$w" in new or "$nope" in new:
        text = text.replace("module: res8, repeats: 2}", "module: res8, repeats: 2, args: {w: 8}}")
    err(tmp_path, text, match)


def test_block_arguments_in_lists(tmp_path):
    text = CARD.replace("PARAMS", "2692").replace(
        "    - {id: c1, module: conv_bn_act, args: {out: 8, k: 3}}",
        "    - {id: c1, module: conv, args: {out: '$w', k: [3, '$k'], pad: 1}}").replace(
        "module: res8, repeats: 2}", "module: res8, repeats: 2, args: {w: 8, k: 3}}")  # fmt: skip
    assert paper.check(paper.load(_card(tmp_path, text)))["params"]


def test_official_and_arch_errors(tmp_path):
    err(
        tmp_path,
        VERIFIED.replace("entry: src/net.py:Net", "entry: src/net.py"),
        "path/to/file.py:ClassName",
    )
    card = paper.load(_card(tmp_path, VERIFIED.split("architecture:")[0], name="a.yaml"))
    with pytest.raises(CardError, match="no `architecture`"):
        paper.to_arch(card)
    card = paper.load(_card(tmp_path, VERIFIED.replace("classes: 4\n", ""), name="nc.yaml"))
    with pytest.raises(CardError, match="number of outputs"):
        paper.to_arch(card)
    card = paper.load(_card(tmp_path, VERIFIED, name="v.yaml"))
    with pytest.raises(SystemExit, match="not found"):
        paper.verify(card, tmp_path / "no_checkout")
    bad = paper.load(
        _card(tmp_path, VERIFIED.replace("[[unsqueeze, 1]]", "[[explode, 1]]"), name="b.yaml")
    )
    with pytest.raises(CardError, match="is not one of"):
        paper.verify(bad, _official(tmp_path))


def test_missing_constants_file(tmp_path):
    text = VEC.replace("architecture:\n", "architecture:\n  - {id: k, module: static, args: {tensors: "
                       "[{shape: [8], learned: false, npz: {file: gone.npz, key: k_t0}}], program: []}}\n")  # fmt: skip
    with pytest.raises(CardError, match="gone.npz not found"):
        paper.load(_card(tmp_path, text))


def test_verify_reports_different_output_shapes(tmp_path):
    for mod in [m for m in sys.modules if m.startswith("official_")]:
        del sys.modules[mod]
    card = paper.load(_card(tmp_path, VERIFIED.replace("input_ops: [[unsqueeze, 1]]",
                                                       "input_ops: [[unsqueeze, 1]]\n  output_ops: [[unsqueeze, 1]]"),
                            name="shape.yaml"))  # fmt: skip
    r = paper.verify(card, _official(tmp_path))
    assert r["level"] == "C" and "output shapes differ" in r["note"]


def test_build_reports_and_fidelity_copy(tmp_path):
    for mod in [m for m in sys.modules if m.startswith("official_")]:
        del sys.modules[mod]
    card = paper.load(_card(tmp_path, VERIFIED.replace("  input_ops", "  skip_verify: no GPU here\n  input_ops"),
                            name="fid.yaml"))  # fmt: skip
    paper.verify(card, _official(tmp_path))
    logs = []
    pkg = paper.build(paper.load(card["_path"]), tmp_path / "out", log=logs.append)
    assert (pkg / "fidelity.json").exists()
    report = (pkg / "report.md").read_text()
    assert (
        "Not compared with the authors' code: no GPU here" in report and "Check against" in report
    )
    far = paper.load(
        _card(tmp_path, VEC, name="far.yaml")
    )  # reported 5 parameters: far from the card
    paper.build(far, tmp_path / "out", log=logs.append)
    assert any("warning" in m for m in logs)
    broken = paper.load(_card(tmp_path, VEC.replace("args: {out: 8, act: relu}", "args: {out: 8, act: nope}"),
                              name="broken.yaml"))  # fmt: skip
    with pytest.raises(CardError):
        paper.build(broken, tmp_path / "out", log=logs.append)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """One small DLNet output folder on a 6-column table (vec input of 6 features, 3 classes)."""
    import numpy as np
    import pandas as pd

    from dlnet import cli

    d = tmp_path_factory.mktemp("run")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 6))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["y"] = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1).astype(int)
    df.to_csv(d / "t.csv", index=False)
    assert cli.main(["design", "--data", str(d / "t.csv"), "--target", "y", "--budget", "30ep", "--topk", "1",
                     "--zoo", "1", "--per-family", "1", "--evolve", "0", "--no-proxies", "--threads", "1",
                     "--out", str(d / "o")]) == 0  # fmt: skip
    return d / "o"


def test_build_into_a_run(tmp_path, run_dir):
    logs = []
    card = paper.load(_card(tmp_path, VEC.replace("classes: 3", "classes: 5"), name="into.yaml"))
    board = json.loads((run_dir / "leaderboard.json").read_text())
    board.pop("notebook", None)  # an output folder written by an older DLNet
    (run_dir / "leaderboard.json").write_text(json.dumps(board))
    paper.build(card, into=run_dir, log=logs.append)
    assert any("this data has" in m for m in logs) and any("not updated" in m for m in logs)
    wrong = paper.load(_card(tmp_path, VEC.replace("dims: [6]", "dims: [7]"), name="wrong.yaml"))
    with pytest.raises(CardError, match="is not the input of this run"):
        paper.build(wrong, into=run_dir, log=logs.append)


def test_compare_skips_and_records_failures(tmp_path, run_dir, monkeypatch):
    from dlnet import search

    class Plugin:
        def zoo_input(self, data):
            return "vec", (6,), "cls"

        build_model = None

    board = json.loads((run_dir / "leaderboard.json").read_text())
    top = {"cfg": {"lr": 1e-3, "wd": 0.0, "batch": 16}, "epochs": 1, "n_seeds": 1, "runs": {}}
    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "a.yaml").write_text(VERIFIED.split("architecture:")[0])  # level A: not trained here
    (cards / "b.yaml").write_text(VEC)
    (cards / "c.yaml").write_text(VEC.replace("Vec Net", "Other Net"))
    monkeypatch.setattr(search, "eval_runs", lambda *a, **k: ([], [], 0.0, None))
    logs = []
    rows = paper.compare(
        [str(cards)], run_dir, Plugin(), {"out_dim": 3}, [], [top], False, 1, log=logs.append
    )
    assert (
        rows == []
        and any("level A card" in m for m in logs)
        and any("could not be trained" in m for m in logs)
    )

    def boom(*a, **k):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(search, "eval_runs", boom)
    paper.compare(
        [str(cards / "b.yaml")],
        run_dir,
        Plugin(),
        {"out_dim": 3},
        [],
        [top],
        False,
        1,
        log=logs.append,
    )
    assert any("failed, RuntimeError" in m for m in logs)
    assert json.loads((run_dir / "leaderboard.json").read_text())["papers_failed"]
    assert board["top_k"]
