"""The basic layer library (dlnet/zoo/ops_prim.py): every argument it refuses, and the small paths the
importer rarely writes (v1.9 coverage)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dlnet.zoo import ops_prim as P
from dlnet.zoo.core import _build

M = ("map", (4, 6, 6))


@pytest.mark.parametrize(("spec", "match"), [
    ({"op": "conv", "out": 4, "k": [3, 3, 3]}, "2 values expected"),
    ({"op": "activation", "fn": "nope"}, "unknown activation"),
    ({"op": "pooling", "mode": "median", "k": 2}, "pool mode"),
    ({"op": "index_axis", "axis": 0, "index": [0]}, "batch axis"),
    ({"op": "transpose", "a": 0, "b": 1}, "batch axis"),
    ({"op": "flatten_axes", "start": 0, "end": 1}, "batch axis"),
    ({"op": "unflatten", "axis": 0, "sizes": [1, 2]}, "batch axis"),
    ({"op": "unsqueeze", "axis": 0}, "batch axis"),
    ({"op": "slice", "axis": 0, "start": 0, "stop": 1}, "batch axis"),
    ({"op": "select", "axis": 0, "index": 0}, "batch axis"),
    ({"op": "reduce", "fn": "mean", "axes": [0]}, "batch axis"),
    ({"op": "reduce", "fn": "median", "axes": [1]}, "reduce fn"),
    ({"op": "reduce", "fn": "prod", "axes": [1, 2]}, "one axis"),
    ({"op": "scalar", "fn": "xor", "value": 1}, "scalar fn"),
    ({"op": "unary", "fn": "gamma"}, "unary fn"),
    ({"op": "cast", "dtype": "float7"}, "torch dtype name"),
    ({"op": "fft", "fn": "dct", "axes": [2], "norm": "ortho", "size": None}, "fft fn"),
    ({"op": "unfold_axis", "axis": 0, "size": 2, "step": 1}, "batch axis"),
    ({"op": "unfold_axis", "axis": 2, "size": 9, "step": 1}, "window 9"),
])  # fmt: skip
def test_bad_arguments_are_refused(spec, match):
    with pytest.raises(ValueError, match=match):
        _build(spec, *M)


def test_merges_refuse_bad_arguments():
    with pytest.raises(ValueError, match="mha: unknown argument"):
        P.merge_module("mha", {"heads": 2, "colour": 1}, [(3, 8), (3, 8), (3, 8)])
    with pytest.raises(ValueError, match="unknown merge"):
        P.merge_module("xor", {}, [(3,), (3,)])
    with pytest.raises(ValueError, match="takes 3 inputs"):
        P.merge_module("where", {}, [(3,), (3,)])


def test_reprs_parameters_buffers_and_static_index_inputs():
    assert "relu" in repr(P.Fn("relu", torch.relu))
    assert repr(P.Cast("float32")).endswith("(float32)")
    assert "shape=(1, 3)" in repr(P.Param((1, 3), "normal"))
    assert torch.equal(P.Param((2,), "ones").weight, torch.ones(2))
    assert torch.equal(P.Param((2,), 0.5).weight, torch.full((2,), 0.5))
    assert "add" in repr(P.merge_module("add", {}, [(3,), (3,)]))
    assert "axis=2" in repr(P.UnfoldAxis(2, 3, 1))
    s = P.StaticExpr([{"shape": [4]}, {"shape": [2], "learned": False, "dtype": "int64"}],
                     [{"op": "item", "in": [0, 1], "index": [{"t": 1}]}])  # fmt: skip
    assert "t1" in dict(s.named_buffers()) and s().shape == (2,)
    bad = P.StaticExpr([{"shape": [1]}], [{"op": "svd", "in": [0]}])
    with pytest.raises(ValueError, match="static step"):
        bad()
