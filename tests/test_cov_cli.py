"""The CLI (dlnet/cli.py): helpers on their edge cases, and one small `design` run in which the rare
branches are made to happen (a cross-validation split that cannot be prepared, an advisor that
proposes, evolution without children, failed candidates, a notebook and a LaTeX table that cannot be
written). v1.9 coverage."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from dlnet import cli

ROOT = Path(__file__).resolve().parents[1]


def test_budget_report_warns_on_an_overrun(capsys):
    rep = cli._budget_report("10s", "time", 10.0, 30.0, 1)
    assert rep["overrun_pct"] > 10 and "WARNING" in capsys.readouterr().out


def test_warnings_about_mixed_epochs_and_failures():
    data = {"n_train": 100, "task": "classification"}
    out = cli._warnings(
        data, [{"epochs": 4, "n_seeds": 2}, {"epochs": 12, "n_seeds": 2}], [{"family": "x"}]
    )
    assert any("mixes models" in w for w in out) and any("failed or diverged" in w for w in out)


def test_cost_model_without_a_zoo_input_or_with_an_uncostable_candidate(capsys, monkeypatch):
    class P:
        def __init__(self, inp):
            self.inp = inp

        def zoo_input(self, data):
            return self.inp

        build_model = None

    args = type("A", (), {"max_flops": None, "max_vram": None})()
    assert cli._apply_cost_model(P(None), {}, [{"family": "f"}], args) == [{"family": "f"}]
    from dlnet.zoo import cost

    def boom(*a, **k):
        raise RuntimeError("cannot cost this one")

    monkeypatch.setattr(cost, "estimate", boom)
    kept = cli._apply_cost_model(
        P(("vec", (3,), "cls")), {"in_dim": 3, "out_dim": 2}, [{"family": "f", "spec": []}], args
    )
    assert kept == [{"family": "f", "spec": []}] and "not costed" in capsys.readouterr().out


def test_device_name_and_rung_parsing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev: "Test GPU")
    assert cli.torch_name("cuda:0") == "Test GPU"
    with pytest.raises(SystemExit, match="comma-separated"):
        cli.parse_rungs("4,x")
    with pytest.raises(SystemExit, match="positive and increasing"):
        cli.parse_rungs("12,4")


def test_design_rare_branches(tabular_csv, tmp_path, monkeypatch, capsys):
    from dlnet import advisor, latex, notebook, search
    from dlnet.plugins import tabular
    from dlnet.zoo import space

    calls = {"n": 0}
    real_prepare = tabular.TabularPlugin.prepare

    def prepare(self, args):
        calls["n"] += 1
        if calls["n"] == 2:  # the second cross-validation split
            raise ValueError("this split cannot be made")
        return real_prepare(self, args)

    monkeypatch.setattr(tabular.TabularPlugin, "prepare", prepare)
    monkeypatch.setattr(advisor, "command", lambda name, cmd: "fake-advisor")
    monkeypatch.setattr(
        advisor,
        "advise",
        lambda *a, **k: {
            "proposals": [],
            "ranking": [],
            "rationale": "none",
            "dropped_ids": [],
            "invalid_proposals": 0,
        },
    )
    monkeypatch.setattr(space, "aging_children", lambda *a, **k: [])
    real_search = search.run_search

    def run_search(*a, **k):
        out = real_search(*a, **k)
        if k.get("trace") is not None:
            k["trace"].setdefault("failed", []).append({"family": "broken", "error": "diverged"})
        return out

    monkeypatch.setattr(search, "run_search", run_search)

    def fail(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(notebook, "write", fail)
    monkeypatch.setattr(latex, "write", fail)
    rc = cli.main(["design", "--data", str(tabular_csv), "--target", "y", "--budget", "200ep", "--topk", "1",
                   "--zoo", "2", "--per-family", "1", "--evolve", "1", "--evolve-pop", "1", "--cv-splits", "2", "--no-proxies",
                   "--threads", "1", "--advisor", "claude", "--out", str(tmp_path / "o")])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0
    assert (
        "[cv] split 1 skipped" in out
        and "[notebook] not written" in out
        and "[latex] not written" in out
    )
    import json

    board = json.loads((tmp_path / "o" / "leaderboard.json").read_text())
    assert board["failed_candidates"]


def test_paper_command_errors(tmp_path, monkeypatch, capsys):
    from dlnet import paper

    assert pytest.raises(SystemExit, cli.main, ["paper", "import"]).match("paper import needs")
    card = str(ROOT / "papers" / "resnet20_cifar.yaml")
    assert pytest.raises(SystemExit, cli.main, ["paper", "build", card]).match("--out DIR")
    assert pytest.raises(SystemExit, cli.main, ["paper", "verify", card]).match("--official-root")
    monkeypatch.setattr(paper, "verify", lambda c, root: {"paper": "ResNet-20", "level": "B", "params_card": 1,
                                                          "params_official": 1, "max_abs_diff": 0.0})  # fmt: skip
    assert cli.main(["paper", "verify", card, "--official-root", str(tmp_path)]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("paper: {}\n")
    assert cli.main(["paper", "check", str(bad)]) == 1
    assert "[paper]" in capsys.readouterr().out


def test_notebook_command(tmp_path, monkeypatch):
    from dlnet import notebook

    seen = {}
    monkeypatch.setattr(notebook, "refresh_support", lambda out: seen.setdefault("support", out))
    monkeypatch.setattr(notebook, "write", lambda out, src, url, sub: seen.update(src=src, url=url))
    assert (
        cli.main(["notebook", str(tmp_path), "--data", "d.csv", "--target", "y", "--git-url", "u"])
        == 0
    )
    assert seen["support"] == str(tmp_path) and seen["src"]["target"] == "y" and seen["url"] == "u"


def test_module_runs_as_a_script(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dlnet", "--version"])
    with pytest.raises(SystemExit):
        runpy.run_module("dlnet.cli", run_name="__main__")


def test_evolution_stops_when_nothing_reached_rung_one(tabular_csv, tmp_path, monkeypatch, capsys):
    """An empty population (every rung-1 result without a spec) ends the evolution loop at once."""
    from dlnet import search

    real_search = search.run_search

    def run_search(*a, **k):
        out = real_search(*a, **k)
        if k.get("trace") is not None:
            k["trace"]["rung1"] = []
        return out

    monkeypatch.setattr(search, "run_search", run_search)
    rc = cli.main(["design", "--data", str(tabular_csv), "--target", "y", "--budget", "200ep", "--topk", "1",
                   "--zoo", "2", "--per-family", "1", "--evolve", "3", "--no-proxies",
                   "--threads", "1", "--out", str(tmp_path / "o")])  # fmt: skip
    assert rc == 0
    assert "[evolve] generation" not in capsys.readouterr().out
