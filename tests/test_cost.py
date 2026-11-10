"""Cost model: closed-form FLOPs and memory, limits, and the leaderboard columns."""

import json

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from dlnet import cli, zoo
from dlnet.zoo import cost, families


def test_linear_and_conv_flops_are_two_times_macs():
    lin = cost.model_cost(nn.Linear(7, 10), (7,))
    assert lin["flops"] == 2 * 7 * 10 and lin["params"] == 7 * 10 + 10
    conv = cost.model_cost(nn.Conv2d(3, 8, 3, padding=1), (3, 5, 5))
    assert conv["flops"] == 2 * 3 * 8 * 9 * 25  # out channels x in channels x kernel x positions
    two = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    assert cost.model_cost(two, (4,))["flops"] == 2 * (4 * 6 + 6 * 2)


def test_memory_formula_and_meta_equals_real():
    m = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    c = cost.model_cost(m, (4,), batch=10)
    act_bytes = 4 * (6 + 6 + 2)  # outputs of the three leaf modules, float32, one sample
    assert c["act_mb"] == pytest.approx(act_bytes / 1e6)
    assert c["vram_mb"] == pytest.approx((16 * c["params"] + 2 * 10 * act_bytes) / 1e6)
    with torch.device("meta"):
        meta = cost.model_cost(
            nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)), (4,), batch=10
        )
    assert meta == c


@pytest.mark.parametrize(
    "op", ["transformer_block", "mamba_block", "rnn", "basic_block", "tt_linear"]
)
def test_every_kind_of_block_can_be_costed_on_the_meta_device(op):
    kind = "map" if op == "basic_block" else "vec" if op == "tt_linear" else "seq"
    dims = {"map": (8, 8, 8), "vec": (32,), "seq": (8, 16)}[kind]
    spec = {"op": op}
    m = zoo.build_model({"spec": spec, "input": {"kind": kind, "dims": dims}, "out_dim": 3})
    c = cost.estimate(
        lambda cfg, i, o: zoo.build_model(cfg),
        {"spec": spec, "input": {"kind": kind, "dims": dims}, "out_dim": 3},
        0,
        3,
        dims,
    )
    assert c["flops"] > 0 and c["params"] == zoo.count_params(m)


def test_limits_and_size_parsing():
    assert (
        cost.parse_size("50M") == 5e7
        and cost.parse_size("2G") == 2e9
        and cost.parse_size("1500") == 1500
    )
    assert cost.parse_size(None) is None and cost.parse_size("") is None
    c = {"flops": 10, "vram_mb": 5.0}
    assert (
        cost.within(c, None, None)
        and cost.within(c, 10, 5.0)
        and not cost.within(c, 9, None)
        and not cost.within(c, None, 4.9)
    )


def test_attention_costs_more_than_a_pointwise_layer_of_the_same_width():
    d = (16, 32)
    att = cost.estimate(
        lambda c, i, o: zoo.build_model(c),
        {
            "spec": {"op": "attn_seq", "heads": 4, "dropout": 0.0, "causal": 0},
            "input": {"kind": "seq", "dims": d},
            "out_dim": 2,
        },
        0,
        2,
        d,
    )
    lin = cost.estimate(
        lambda c, i, o: zoo.build_model(c),
        {
            "spec": {"op": "linear_act", "out": 32, "act": "identity"},
            "input": {"kind": "seq", "dims": d},
            "out_dim": 2,
        },
        0,
        2,
        d,
    )
    assert att["flops"] > 4 * lin["flops"] > 0  # qkv + output projection + the T x T score matmuls


def test_cli_filters_by_flops_and_writes_cost_columns(tmp_path, tabular_csv, capsys):
    out = tmp_path / "o"
    argv = ["design", "--data", str(tabular_csv), "--target", "y", "--budget", "40s", "--topk", "1", "--per-family", "1",
            "--zoo", "4", "--evolve", "0", "--no-proxies", "--no-cache", "--max-flops", "20K", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert "[cost]" in text and "dropped" in text
    board = json.loads((out / "leaderboard.json").read_text())
    row = board["top_k"][0]
    assert row["flops"] is not None and row["flops"] <= 20_000 and row["vram_mb"] > 0
    md = (out / "leaderboard.md").read_text()
    assert "MFLOPs" in md and "training memory" in (out / row["name"] / "report.md").read_text()


def test_family_param_counts_match_the_earlier_libraries():
    """Parameter counts of the recipes equal those recorded from the earlier implementations (width 16, depth 2)."""
    base = {
        "width": 16,
        "depth": 2,
        "dropout": 0.1,
        "rank": 4,
        "patch": 4,
        "experts": 3,
        "routing": "topk",
        "rank_ratio": 0.5,
        "attention": "none",
        "encoder": "resnet",
    }
    golden = {("tabular", (20,), "mlp"): 693, ("tabular", (20,), "resmlp"): 1605, ("tabular", (20,), "tt_resmlp"): 1093,
              ("tabular", (20,), "tr_resmlp"): 2629, ("tabular", (20,), "lowrank_mlp"): 565, ("tabular", (20,), "fm_mlp"): 3813,
              ("image", (3, 16, 16), "cnn"): 16917, ("image", (3, 16, 16), "resnet"): 174933, ("image", (3, 16, 16), "lenet"): 205813,
              ("image", (3, 16, 16), "mobilenetv2"): 70805, ("timeseries", (4, 48), "gru"): 7077,
              ("segmentation", (3, 16, 16), "unet"): 222685,  # smp decoder since v0.13 (was 220965)
              ("hsi", (30, 9, 9), "cnn1d"): 2901, ("hsi", (30, 9, 9), "cnn3d"): 8853,
              ("hsi", (30, 9, 9), "ssrn"): 19973, ("hsi", (30, 9, 9), "tucker_cnn2d"): 5941}  # fmt: skip
    for (domain, dims, fam), n in golden.items():
        cfg = families.complete({"family": fam, **base}, domain, dims, 5)
        assert families.count_params(cfg) == n, (domain, fam)
