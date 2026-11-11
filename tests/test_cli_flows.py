"""In-process runs of `dlnet design` covering every option branch (tiny data, budgets of seconds).

Each test checks the outcome a user relies on (leaderboard written, option reflected in it), not
only that the run finishes. The CLI-in-a-subprocess and exported-package checks are in test_e2e.py.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from dlnet import cli

FAST = ["--budget", "40ep", "--topk", "1", "--zoo", "4", "--per-family", "1", "--evolve", "1",
        "--evolve-pop", "2", "--no-cache", "--no-builtin-meta"]  # fmt: skip


@pytest.fixture(scope="module")
def img_npz(tmp_path_factory):
    """90 grey 8x8 images, 3 classes told apart by a bright quadrant."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 30)
    X = rng.normal(scale=0.3, size=(90, 8, 8)).astype("float32")
    for i, c in enumerate(y):
        X[i, (c // 2) * 4 : (c // 2) * 4 + 4, (c % 2) * 4 : (c % 2) * 4 + 4] += 2.0
    p = tmp_path_factory.mktemp("d") / "img.npz"
    np.savez(p, X=X, y=y)
    return p


def run(data, out, *extra):
    rc = cli.main(["design", "--data", str(data), "--out", str(out), *FAST, *extra])
    assert rc == 0
    return json.loads((out / "leaderboard.json").read_text())


@pytest.mark.parametrize("objective", ["params", "flops", "balanced", "pareto"])
def test_every_objective_orders_the_topk_and_reports_the_front(img_npz, tmp_path, objective):
    b = run(img_npz, tmp_path / "o", "--objective", objective, "--target-cost", "20k")
    assert b["objective"]["objective"] == objective
    assert b["objective"]["pareto_front"] and b["top_k"]
    md = (tmp_path / "o" / "leaderboard.md").read_text()
    assert f"Objective: {objective}" in md and "Pareto front" in md


def test_hyperband_seeds_random_order_and_proxy_vote(img_npz, tmp_path):
    b = run(img_npz, tmp_path / "h", "--scheduler", "hyperband", "--seeds", "4", "--seed-proxies",
            "--seed-order", "random", "--context", "handwritten digits on mobile")  # fmt: skip
    s = b["seed_selection"]
    assert s["scheduler"] == "hyperband" and s["kept"] == 4 and s["used_for_pruning"]
    assert any("context matches" in r for r in s["rules_fired"])


def test_max_params_removes_everything_and_says_so(img_npz, tmp_path):
    with pytest.raises(SystemExit, match="no valid candidate"):
        cli.main(["design", "--data", str(img_npz), "--out", str(tmp_path / "x"), *FAST,
                  "--max-params", "10"])  # fmt: skip


def test_extra_specs_join_the_pool(img_npz, tmp_path):
    specs = tmp_path / "specs.json"
    good = {"op": "seq", "body": [{"op": "conv_bn_act", "out": 8}, {"op": "basic_block", "out": 8}]}
    specs.write_text(json.dumps([good, {"op": "does_not_exist"}]))
    b = run(img_npz, tmp_path / "e", "--extra-specs", str(specs), "--seeds", "0")
    assert b["seed_selection"]["pool"] >= 1


def _flaky_advisor(tmp_path):
    """Fails on the first call, answers on the next ones."""
    f = tmp_path / "flaky.py"
    flag = tmp_path / "called"
    f.write_text(
        "import json, pathlib, re, sys\n"
        f"flag = pathlib.Path({str(flag)!r})\n"
        "p = sys.stdin.read()\n"
        "if not flag.exists():\n    flag.write_text('1'); sys.exit(3)\n"
        'ids = re.findall(r\'"id": "(c\\d+)"\', p)\n'
        "print(json.dumps({'ranking': ids[:2], 'proposals': [{'op': 'seq', 'body': ["
        "{'op': 'conv_bn_act', 'out': 8}, {'op': 'eca', 'k': 0}]}], 'rationale': 'r'}))\n"
    )
    return f"{sys.executable} {f}"


@pytest.mark.regression
def test_advisor_failing_first_then_answering_in_evolution_does_not_crash(img_npz, tmp_path):
    """Was: AttributeError ('NoneType' has no setdefault) when the seed-selection call failed but
    the evolution call succeeded."""
    b = run(
        img_npz,
        tmp_path / "a",
        "--advisor-cmd",
        _flaky_advisor(tmp_path),
        "--evolve",
        "2",
        "--budget",
        "80ep",
    )
    assert "ai_advisor" in b and b["ai_advisor"]["evolution"]


def test_meta_store_records_candidates_and_feeds_the_surrogate(img_npz, tmp_path):
    store = tmp_path / "runs.jsonl"
    for i in range(3):
        run(img_npz, tmp_path / f"m{i}", "--meta-store", str(store), "--seed", str(i))
    recs = [json.loads(line) for line in store.read_text().splitlines()]
    assert len(recs) == 3 and all(r["candidates"] and r["card"] for r in recs)
    b = run(img_npz, tmp_path / "m3", "--meta-store", str(store))
    assert "zap_surrogate" in b["seed_selection"]["rankers"]
    assert b["meta_recommendation"]["n_records"] == 3


def test_repeated_splits_and_the_tabular_plugin(tabular_csv, tmp_path):
    rc = cli.main(["design", "--data", str(tabular_csv), "--target", "y", "--out", str(tmp_path / "t"),
                   *FAST, "--cv-splits", "2", "--budget", "600ep"])  # fmt: skip
    assert rc == 0
    b = json.loads((tmp_path / "t" / "leaderboard.json").read_text())
    assert b["top_k"][0]["n_splits"] == 2


def test_other_commands_and_bad_inputs(capsys):
    assert cli.main(["zoo"]) == 0
    assert "conv_bn_act" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="bad --budget"):
        cli.parse_budget("five minutes")
    assert cli.parse_budget("2h") == 7200 and cli.parse_budget("90") == 90
    with pytest.raises(SystemExit):
        cli.main(["design", "--data", "x.csv", "--plugin", "nope"])
