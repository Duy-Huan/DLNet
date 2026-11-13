"""v1.7 CLI for papers: paired significance (dlnet/stats.py), the LaTeX table (dlnet/latex.py) and the
pre-run estimate (dlnet/plan.py)."""

from __future__ import annotations

import json
import math

import pytest

from dlnet import latex, plan, stats

# ----------------------------------------------------------------------------- stats


def test_paired_t_matches_scipy():
    from scipy import stats as ss

    a, b = [0.91, 0.93, 0.92, 0.95], [0.90, 0.90, 0.91, 0.92]
    r = stats.paired(a, b)
    ref = ss.ttest_rel(a, b)
    assert r["n"] == 4 and math.isclose(r["diff"], 0.02, abs_tol=1e-12)
    assert math.isclose(r["t"], ref.statistic, rel_tol=1e-9) and math.isclose(
        r["p"], ref.pvalue, rel_tol=1e-9
    )


def test_paired_edge_cases():
    assert stats.paired([], [1.0])["p"] is None
    one = stats.paired([0.9], [0.8])
    assert one["n"] == 1 and math.isclose(one["diff"], 0.1) and one["p"] is None
    same = stats.paired([0.9, 0.8], [0.8, 0.7])  # constant difference: no variance to test
    assert math.isclose(same["diff"], 0.1) and same["p"] is None
    assert stats.paired([0.9, 0.8, 0.7], [0.1, 0.2])["n"] == 2  # unequal lengths: the common runs


def test_holm_step_down():
    # Holm 1979: sorted p (0.01, 0.02, 0.04) x (3, 2, 1) = (0.03, 0.04, 0.04), monotone
    assert stats.holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert stats.holm([0.5, None, 0.9]) == pytest.approx([1.0, None, 1.0])
    assert stats.holm([]) == []


def test_verdict_words():
    assert stats.verdict(None, None) == "-"
    assert stats.verdict(0.01, None) == "not tested"
    assert stats.verdict(0.01, 0.2) == "no significant difference"
    assert stats.verdict(0.01, 0.01) == "better" and stats.verdict(-0.01, 0.01) == "worse"


# ----------------------------------------------------------------------------- latex


def _board():
    return {
        "metric": "accuracy",
        "top_k": [
            {"name": "top1_cnn_small", "params": 12_345, "val_mean": 0.9, "val_std": 0.01, "n_runs": 3,
             "test": {"score": 0.89}, "test_std": 0.005},
            {"name": "top2_resnet", "params": 1_500_000, "val_mean": 0.88, "val_std": 0.02, "n_runs": 3,
             "test": {"score": 0.87}, "test_std": 0.01},
        ],
        "reference": [{"name": "random_forest", "val": 0.7, "test": {"score": 0.69}},
                      {"name": "log_reg", "val": 0.6, "test": {"score": 0.61}}],
        "papers": [
            {"paper": "Net_A & co", "params": 270_000, "common": {"val_mean": 0.85, "val_std": 0.01,
             "test": {"score": 0.84}, "test_std": 0.01}, "vs_top1": {"verdict": "worse"}},
            {"paper": "NetB", "params": 50_000, "common": {"val_mean": 0.9, "val_std": 0.0,
             "test": {"score": 0.91}, "test_std": 0.0}, "vs_top1": {"verdict": "no significant difference"}},
            {"paper": "skipped"},
        ],
    }  # fmt: skip


def test_latex_table():
    tex = latex.table(_board(), label="tab:x")
    assert r"\begin{tabular}{llrcc}" in tex and r"\toprule" in tex and r"\label{tab:x}" in tex
    assert r"cnn\_small" in tex and "1.50M" in tex and "12.3K" in tex
    assert r"Net\_A \& co$^\dagger$" in tex  # escaped, marked significantly worse
    assert r"\textbf{0.9100}" in tex  # best test in bold
    assert "random\\_forest" in tex and "log\\_reg" not in tex  # only the best classical reference
    assert "skipped" not in tex and tex.count(r"\midrule") == 3
    assert "Holm" in tex


def test_latex_write(tmp_path):
    with pytest.raises(SystemExit, match="leaderboard.json"):
        latex.write(tmp_path)
    (tmp_path / "leaderboard.json").write_text(json.dumps(_board()))
    out = latex.write(tmp_path)
    assert out.name == "results.tex" and r"\end{table}" in out.read_text()


def test_latex_cli(tmp_path, capsys):
    from dlnet import cli

    (tmp_path / "leaderboard.json").write_text(json.dumps(_board()))
    assert cli.main(["latex", str(tmp_path)]) == 0
    assert r"\begin{table}" in capsys.readouterr().out


# ----------------------------------------------------------------------------- plan


def test_estimate_rungs_follows_successive_halving():
    # 40 candidates, 1 s/epoch, 120 s of search, rungs (4, 12, 30): 40 s per rung
    r = plan.estimate_rungs(40, 1.0, 120.0, (4, 12, 30), topk=3)
    assert [x["candidates"] for x in r] == [10, 5, 2]
    # plenty of time: halving (keep 1/3, at least 2 x Top-K) decides
    r = plan.estimate_rungs(30, 0.01, 1e6, (4, 12, 30), topk=1)
    assert [x["candidates"] for x in r] == [30, 10, 4]
    # nothing fits rung 1: the estimate stops there
    assert plan.estimate_rungs(10, 100.0, 60.0, (4, 12), topk=1) == [
        {"rung": 1, "epochs": 4, "candidates": 0}
    ]


def test_plan_end_to_end_on_a_table(tmp_path, capsys):
    import numpy as np

    from dlnet import cli

    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 5))
    rows = [
        "a,b,c,d,e,y",
        *(",".join(f"{v:.3f}" for v in x) + f",{int(x[0] + x[1] > 0)}" for x in X),
    ]
    (tmp_path / "d.csv").write_text("\n".join(rows) + "\n")
    card = tmp_path / "card.yaml"
    card.write_text("paper: {name: Broken}\n")
    save = tmp_path / "plan.json"
    code = cli.main(["plan", "--data", str(tmp_path / "d.csv"), "--target", "y", "--budget", "60ep",
                     "--zoo", "2", "--per-family", "1", "--threads", "1", "--papers", str(card), "--save", str(save)])  # fmt: skip
    assert code == 0
    out = capsys.readouterr().out
    assert "rung 1" in out and "to beat" in out and "scores are not predicted" in out
    p = json.loads(save.read_text())
    assert p["budget"]["mode"] == "epochs" and p["candidates"]["valid"] > 0
    assert p["timing"]["sec_per_epoch_median"] > 0 and p["rungs"][0]["candidates"] > 0
    assert p["reference"] and p["papers"][0]["status"].startswith("fails")
    assert p["papers"][0]["code"] == "fail" and p["papers"][0]["detail"]["error"]
    w = next(w for w in p["warnings"] if w["code"] == "papers_skipped")
    assert w["params"] == {"n": 1, "total": 1} and "paper card" in w["text"]
    assert p["seconds_total"] >= p["seconds_design"] > 0


def test_plan_times_paper_networks_on_images(tmp_path):
    """A matching card is timed; one with another input is skipped with the reason; level A is skipped."""
    import shutil
    import types
    from pathlib import Path

    import numpy as np

    rng = np.random.default_rng(0)
    np.savez(
        tmp_path / "img.npz",
        X=rng.random((80, 32, 32, 3), dtype=np.float32),
        y=rng.integers(0, 2, 80),
    )
    papers = Path(__file__).resolve().parent.parent / "papers"
    cards = tmp_path / "cards"
    cards.mkdir()
    for c in ("resnet20_cifar.yaml", "ssftt.yaml"):
        shutil.copy(papers / c, cards / c)
    (cards / "official_only.yaml").write_text(
        "paper: {name: OnlyOfficial}\ninput: {kind: map, dims: [3, 32, 32]}\n"
        "official: {repo: https://example.org/x, commit: abc, class: x.Net}\n"
    )
    from dlnet import plugins

    plugin = plugins.route(str(tmp_path / "img.npz"))
    args = types.SimpleNamespace(
        data=str(tmp_path / "img.npz"), seed=0, x_key=None, y_key=None, task="auto"
    )
    data = plugin.prepare(args)
    rows = {r.get("paper", r["card"]): r for r in plan._papers([str(cards)], plugin, data, plugin.build_model,
                                                             3, 12, True, 50)}  # fmt: skip
    res = rows["ResNet-20"]
    assert res["code"] == "ok" and res["params"] > 100_000 and res["sec_per_epoch"] > 0
    assert (
        res["epochs"] == 3 * 12 + 50 and res["seconds"] > 0
    )  # common protocol + capped authors' recipe
    assert rows["SSFTT"]["code"] == "input" and rows["SSFTT"]["detail"]["data"].startswith("map")
    other = rows.get("OnlyOfficial") or next(
        r for r in rows.values() if r["card"].endswith("official_only.yaml")
    )
    assert other["code"] in ("level_a", "fail")
