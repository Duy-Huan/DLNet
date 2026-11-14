"""`dlnet plan` (dlnet/plan.py), the v1.0 drop-in blocks (dlnet/zoo/ops_next.py) and the helper
refresh of old output folders (dlnet/notebook.py): their refusals and fallbacks (v1.9 coverage)."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dlnet import cli, notebook, plan
from dlnet.zoo import ops_next
from dlnet.zoo.core import _build


@pytest.mark.parametrize(("spec", "kind", "dims", "match"), [
    ({"op": "gated_attn_block", "heads": 4}, "seq", (5, 6), "not divisible by 4 heads"),
    ({"op": "diff_attn_block", "heads": 2}, "seq", (5, 6), "2 x 2 heads"),
    ({"op": "wtconv", "levels": 3}, "map", (4, 3, 3), "wavelet levels"),
    ({"op": "wtconvnext_block", "levels": 3}, "map", (4, 3, 3), "wavelet levels"),
    ({"op": "mobile_mqa", "kv_stride": 2}, "map", (8, 1, 1), "at least 2x2"),
    ({"op": "lsnet_block", "mixer": "ls"}, "map", (12, 4, 4), "divisible by 8"),
])  # fmt: skip
def test_block_refusals(spec, kind, dims, match):
    with pytest.raises(ValueError, match=match):
        _build(spec, kind, dims)


def test_small_maps_use_quadratic_attention_and_divisible_rounding():
    m, _, _ = _build(
        {"op": "efficientvit_block", "dim": 8}, "map", (16, 2, 2)
    )  # 2 x 2 <= 8: quadratic form
    assert m(torch.randn(2, 16, 2, 2)).shape == (2, 16, 2, 2)
    assert ops_next.make_divisible(10, 8) == 16  # 8 would lose more than 10 %


def _table(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    rows = ["a,b,c,d,y", *(",".join(f"{v:.3f}" for v in x) + f",{int(x[0] > 0)}" for x in X)]
    (tmp_path / "d.csv").write_text("\n".join(rows) + "\n")
    return tmp_path / "d.csv"


def test_plan_when_nothing_can_be_timed(tmp_path, monkeypatch):
    from dlnet import search

    def broken(*a, **k):
        raise RuntimeError("no memory")

    monkeypatch.setattr(search, "_new_trainer", broken)
    (tmp_path / "a.yaml").write_text("paper: {name: Authors}\ninput: {kind: vec, dims: [4]}\nofficial: "
                                      "{repo: x, commit: y, entry: 'm.py:M'}\n")  # fmt: skip
    (tmp_path / "v.yaml").write_text("paper: {name: Vec}\ninput: {kind: vec, dims: [4]}\nclasses: 2\n"
                                      "architecture:\n  - {id: a, module: linear_act, args: {out: 8, act: relu}}\n")  # fmt: skip
    save = tmp_path / "plan.json"
    assert cli.main(["plan", "--data", str(_table(tmp_path)), "--target", "y", "--budget", "30s", "--zoo", "1",
                     "--per-family", "1", "--seeds", "2", "--threads", "1", "--papers", str(tmp_path / "a.yaml"),
                     str(tmp_path / "v.yaml"), "--save", str(save)]) == 0  # fmt: skip
    p = json.loads(save.read_text())
    assert p["seconds_design"] is None and any(w["code"] == "untimed" for w in p["warnings"])
    codes = {r["paper"]: r["code"] for r in p["papers"]}
    assert codes == {"Authors": "level_a", "Vec": "fail"}
    assert plan._dur(None) == "?"


def test_plan_on_a_time_budget(tmp_path):
    save = tmp_path / "plan.json"
    assert cli.main(["plan", "--data", str(_table(tmp_path)), "--target", "y", "--budget", "30s", "--zoo", "1",
                     "--per-family", "1", "--threads", "1", "--save", str(save)]) == 0  # fmt: skip
    p = json.loads(save.read_text())
    assert p["budget"]["mode"] == "seconds" and p["seconds_design"] == 30


def test_refresh_support_updates_old_packages(tmp_path):
    pkg = tmp_path / "top1_mlp"
    pkg.mkdir()
    (pkg / "core.py").write_text("# an older core.py\n")
    (pkg / "inference.py").write_bytes(
        (notebook.Path(notebook.__file__).parent / "inference.py").read_bytes()
    )
    logs = []
    assert notebook.refresh_support(tmp_path, logs.append) == 1 and "refreshed 1" in logs[0]
    assert notebook.refresh_support(tmp_path, logs.append) == 0
