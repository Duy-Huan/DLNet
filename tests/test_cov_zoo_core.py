"""The zoo's grammar (dlnet/zoo/core.py): every composition it refuses, and the combinator paths the
search rarely builds (v1.9 coverage)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from dlnet.zoo import core
from dlnet.zoo.core import _build


def bad(spec, kind, dims, match):
    with pytest.raises(ValueError, match=match):
        _build(spec, kind, dims)


def test_op_registry_refuses_a_duplicate_name():
    with pytest.raises(ValueError, match="duplicate op"):
        core.op("linear", core.KINDS)(lambda dims, kind: nn.Identity())


def test_leaf_modules_must_return_one_non_empty_tensor(monkeypatch):
    class Two(nn.Module):
        def forward(self, x):
            return x, x

    with pytest.raises(ValueError, match="one tensor"):
        core._trace(Two(), (4,))

    class Empty(nn.Module):
        def forward(self, x):
            return x[:, :0]

    with pytest.raises(ValueError, match="empty output"):
        core._trace(Empty(), (4,))


def test_residual_projections():
    m, _, d = _build({"op": "res", "body": {"op": "linear", "out": 6}}, "vec", (4,))
    assert isinstance(m.proj, nn.Linear) and d == (6,)
    m, _, d = _build({"op": "res", "body": {"op": "linear", "out": 6}}, "seq", (5, 4))
    assert isinstance(m.proj, nn.Linear) and d == (5, 6)
    m, _k, d = _build(
        {"op": "res", "body": {"op": "conv", "out": 6, "k": 3, "s": 2, "pad": 1}}, "map", (4, 8, 8)
    )
    assert isinstance(m.proj, nn.Conv2d) and m.proj.stride == (2, 2)
    shorter = [{"op": "slice", "axis": 1, "start": 0, "stop": 3}, {"op": "as_kind", "to": "seq"}]
    bad({"op": "res", "body": shorter}, "seq", (6, 4), "sequence length")
    bad(
        {"op": "res", "body": {"op": "conv", "out": 4, "k": 1, "pad": 2}},
        "map",
        (4, 5, 5),
        "spatial size",
    )
    narrower = [{"op": "slice", "axis": 2, "start": 0, "stop": 3}, {"op": "as_kind", "to": "graph"}]
    bad({"op": "res", "body": narrower}, "graph", (8, 12), "not defined for kind graph")
    m.train()
    m.p = 0.5
    assert m(torch.randn(2, 4, 8, 8)).shape == (2, 6, 4, 4)


def test_parallel_merges():
    two = [{"op": "linear", "out": 4}, {"op": "linear", "out": 4}]
    for merge in ("sum", "gate"):
        m, _, _d = _build({"op": "par", "merge": merge, "branches": two}, "vec", (3,))
        assert m(torch.randn(2, 3)).shape == (2, 4)
    bad({"op": "par", "merge": "concat", "branches": [{"op": "conv", "out": 4, "k": 1},
         {"op": "conv", "out": 4, "k": 3, "s": 2, "pad": 1}]}, "map", (3, 8, 8), "cannot concatenate")  # fmt: skip
    bad(
        {"op": "par", "merge": "gated", "width": 4, "branches": two + two},
        "vec",
        (3,),
        "exactly two",
    )


def test_mixture_of_experts_paths():
    experts = {"n": 2, "body": {"op": "linear", "out": 4}}
    m, _, _ = _build({"op": "moe", "experts": experts}, "seq", (5, 3))
    assert m(torch.randn(2, 5, 3)).shape == (2, 5, 4)
    bad({"op": "moe", "experts": experts}, "ids", (5, 5), "real-valued")


def test_encdec_and_graph_refusals():
    bad({"op": "encdec", "stages": []}, "vec", (4,), "map inputs")
    g = {"op": "graph"}
    bad({**g, "nodes": [{"id": "a", "op": "linear", "out": 4}, {"id": "a", "op": "linear", "out": 4}]}, "vec", (3,),
        "used twice")  # fmt: skip
    bad({**g, "nodes": [{"id": "s", "op": "static", "from": "input", "tensors": [], "program": []}]}, "vec", (3,),
        "param / static nodes only")  # fmt: skip
    bad(
        {
            **g,
            "nodes": [
                {"id": "p", "op": "param", "shape": [2]},
                {
                    "id": "s",
                    "op": "static",
                    "from": "p",
                    "tensors": [],
                    "program": [{"op": "item", "in": [0], "index": [5]}],
                },
            ],
        },
        "vec",
        (3,),
        "static program",
    )
    bad({**g, "nodes": [{"id": "p", "op": "param", "shape": []}]}, "vec", (3,), "leading axis")
    bad(
        {**g, "nodes": [{"id": "a", "op": "linear", "out": 4, "from": "zz"}]},
        "vec",
        (3,),
        "not defined before",
    )
    bad(
        {**g, "nodes": [{"id": "a", "op": "add", "from": ["input"]}]},
        "vec",
        (3,),
        "needs >= 2 inputs",
    )
    bad({**g, "nodes": [{"id": "a", "op": "linear", "out": 3}, {"id": "b", "op": "linear", "out": 3, "from": ["input", "a"]}]},
        "vec", (3,), "takes one input")  # fmt: skip
    bad({**g, "nodes": [{"id": "p", "op": "param", "shape": [1, 3]}, {"id": "b", "op": "linear", "out": 3, "from": "p"}]},
        "vec", (3,), "param node is used through a merge")  # fmt: skip
    bad(
        {**g, "nodes": [{"id": "a", "op": "linear", "out": 3, "share": "zz"}]},
        "vec",
        (3,),
        "not a node defined",
    )
    bad({**g, "nodes": [{"id": "a", "op": "linear", "out": 3}, {"id": "b", "op": "linear", "out": 4, "share": "a"}]},
        "vec", (3,), "different layer")  # fmt: skip
    bad({**g, "nodes": []}, "vec", (3,), "at least one node")
    bad(
        {**g, "nodes": [{"id": "a", "op": "linear", "out": 3}], "out": "zz"},
        "vec",
        (3,),
        "is not defined",
    )
    bad({**g, "nodes": [{"id": "a", "op": "reshape", "shape": [3]}, {"id": "b", "op": "reshape", "shape": [-1, 3]}]},
        "vec", (6,), "still merged")  # fmt: skip


def test_ensemble_head_paths():
    with pytest.raises(ValueError, match="members, features"):
        core.Head("vec", (4,), 2, "ensemble")
    h = core.Head("seq", (3, 4), 1, "ensemble")
    x = torch.randn(2, 3, 4)
    h.train()
    assert h(x).shape == (2, 3, 1)  # training: one output per member
    h.eval()
    assert h(x).shape == (2, 1)  # regression: the members' mean


def test_more_combinator_paths():
    two = [{"op": "linear", "out": 4}, {"op": "linear", "out": 4}]
    m, _, d = _build({"op": "par", "merge": "gated", "width": 5, "branches": two}, "vec", (3,))
    assert d == (5,) and m(torch.randn(2, 3)).shape == (2, 5)
    bad({"op": "par", "branches": two[:1]}, "vec", (3,), ">= 2 branches")
    bad({"op": "par", "merge": "max", "branches": two}, "vec", (3,), "unknown merge")
    bad({"op": "rep", "n": 0, "body": two[0]}, "vec", (3,), "n >= 1")
    bad(
        {"op": "moe", "experts": [{"op": "linear", "out": 4}, {"op": "linear", "out": 5}]},
        "vec",
        (3,),
        "identical",
    )
    bad(
        {"op": "moe", "experts": {"n": 2, "body": two[0]}, "mode": "vote"},
        "vec",
        (3,),
        "unknown moe mode",
    )
    bad({"op": "encdec", "stages": [{"op": "gap"}]}, "map", (3, 8, 8), "keep the map kind")
    assert core.leaf_ops({"op": "par", "branches": two}) == ["linear", "linear", "par"]


def test_graph_frees_dead_nodes_and_reads_old_concat_cards():
    g = {"op": "graph", "nodes": [
        {"id": "a", "op": "linear", "out": 4},
        {"id": "dead", "op": "linear", "out": 2, "from": "input"},  # read by nobody, not the output
        {"id": "c", "op": "concat", "from": ["a", "input"]},  # a card written before v1.8: no axis
    ]}  # fmt: skip
    m, _k, d = _build(g, "seq", (5, 3))
    assert d == (5, 7) and m(torch.randn(2, 5, 3)).shape == (2, 5, 7)


def test_check_without_meta_building_and_ensemble_classification(monkeypatch):
    real = core._build

    def meta_fails(spec, kind, dims):
        if torch.get_default_device().type == "meta":
            raise NotImplementedError("no meta kernel")
        return real(spec, kind, dims)

    monkeypatch.setattr(core, "_build", meta_fails)
    assert core.check([{"op": "linear", "out": 4}], "vec", (3,))["dims"] == (4,)
    h = core.Head("seq", (3, 4), 5, "ensemble").eval()
    p = h(torch.randn(2, 3, 4)).exp().sum(1)
    assert torch.allclose(p, torch.ones(2), atol=1e-5)  # log of the members' mean probability


def test_concat_shared_and_graph_errors_run():
    two = [{"op": "linear", "out": 4}, {"op": "linear", "out": 2}]
    m, _, _d = _build({"op": "par", "branches": two}, "vec", (3,))
    assert m(torch.randn(2, 3)).shape == (2, 6)
    m, _, _ = _build(
        {"op": "rep", "n": 2, "share": True, "body": {"op": "linear", "out": 3}}, "vec", (3,)
    )
    assert m(torch.randn(2, 3)).shape == (2, 3)
    g = {
        "op": "graph",
        "nodes": [
            {"id": "a", "op": "linear", "out": 4},
            {"id": "b", "op": "add", "from": ["a", "input"]},
        ],
    }
    bad(g, "vec", (3,), "needs identical shapes")
    bad(
        {"op": "graph", "nodes": [{"id": "a", "op": "nope"}]},
        "vec",
        (3,),
        "graph node 'a': unknown op",
    )
