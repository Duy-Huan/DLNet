"""Standalone model.py (dlnet/standalone.py): the source collector, the expression writer and the
verifier on the cases real networks rarely hit (v1.9 coverage). The collector is driven on a small
package written for the test, so each rule shows on two or three lines of code."""

from __future__ import annotations

import functools
import json
import sys
from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")

from test_zoo import _cfg
from torch import nn

from dlnet import standalone
from dlnet.standalone import Collector, Unsupported, Writer, _Module

A = """
import math as m
from collections import OrderedDict as OD
from . import sub
from .b import helper
from .b import helper as renamed

try:
    import json
except ImportError:
    json = None

if True:
    FLAG = 1


def op(**kw):
    return lambda f: f


@op(kind="x")
def decorated(x):
    return helper(x)


@functools.lru_cache
def cached(x):
    return x


def uses_global():
    global FLAG
    import os
    FLAG = 2
    return os.sep, m.pi, OD


def nested_relative():
    from .b import helper
    return helper


def twice():
    return 1
"""

B = """
def helper(x):
    return x


def twice():
    return 2
"""


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    root = tmp_path / "fakezoo_pkg"
    (root / "sub").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "sub" / "__init__.py").write_text("")
    (root / "a.py").write_text("import functools\n" + A)
    (root / "b.py").write_text(B)
    monkeypatch.syspath_prepend(str(tmp_path))
    _Module.cache.clear()
    yield "fakezoo_pkg"
    _Module.cache.clear()
    for k in [k for k in sys.modules if k.startswith("fakezoo_pkg")]:
        del sys.modules[k]


def test_module_scan_reads_if_and_try_blocks_and_decorators(pkg):
    m = _Module.get(f"{pkg}.a")
    assert {"json", "FLAG", "decorated", "cached"} <= set(m.defs)
    node = m.defs["cached"][0]
    assert m.segment(node).startswith("@functools.lru_cache")
    assert m.segment(node, strip_op=True).startswith("def cached")


def test_collector_follows_names_and_strips_the_op_decorator(pkg):
    c = Collector()
    assert c.need(f"{pkg}.a", "decorated")
    src = dict(c.items)
    assert "@op" not in src[(f"{pkg}.a", "decorated")] and (f"{pkg}.b", "helper") in src
    assert c.need(f"{pkg}.a", "uses_global") and "import math as m" in c.imports
    assert any("OrderedDict as OD" in i for i in c.imports)
    assert c.need(f"{pkg}.a", "decorated")  # asked again: already done
    assert not c.need(f"{pkg}.a", "nothing_by_that_name")


def test_collector_refusals(pkg):
    with pytest.raises(Unsupported, match="relative import inside a definition"):
        Collector().need(f"{pkg}.a", "nested_relative")
    c = Collector()
    c.need(f"{pkg}.a", "twice")
    with pytest.raises(Unsupported, match="defined twice"):
        c.need(f"{pkg}.b", "twice")
    with pytest.raises(Unsupported, match="relative import renamed"):
        Collector().need(f"{pkg}.a", "renamed")
    with pytest.raises(Unsupported, match="is a module, not a definition"):
        Collector().need(f"{pkg}.a", "sub")
    c = Collector()
    c._import("import torch", "torch")
    with pytest.raises(Unsupported, match="means two things"):
        c._import("import numpy as torch", "torch")
    with pytest.raises(Unsupported, match="not a top-level definition of the zoo"):
        Collector().need_object(json.dumps)


def test_writer_expressions():
    w = Writer(Collector())
    assert w.expr(()) == "()" and w.expr([]) == "[]"
    assert w.expr((1,)) == "(1,)"
    assert w.expr(float("inf")) == 'float("inf")'
    assert w.expr(OrderedDict(a=1)).startswith("OrderedDict(")
    text = w.expr(
        functools.partial(torch.relu, inplace=False) if False else functools.partial(nn.ReLU, True)
    )
    assert text.startswith("functools.partial(nn.ReLU") and "import functools" in w.c.imports
    assert w.expr(torch.tensor([1.0, 2.0])).startswith("torch.tensor(")
    assert w.expr(nn.functional.relu) == "nn.functional.relu" and w.expr(torch.tanh) == "torch.tanh"
    with pytest.raises(Unsupported, match="cannot be written as source"):
        w.expr(object())
    named = nn.Sequential(OrderedDict(a=nn.ReLU()))
    with pytest.raises(Unsupported, match="named children"):
        w.expr(named)
    with standalone.recording():
        ml, md = nn.ModuleList([nn.ReLU()]), nn.ModuleDict({"r": nn.ReLU()})
    assert w.expr(ml).startswith("nn.ModuleList(") and w.expr(md).startswith("nn.ModuleDict(")
    with pytest.raises(Unsupported, match="built outside the recorder"):
        w.expr(nn.ReLU())


def test_source_imports_ordered_dict_when_a_constructor_needs_it(monkeypatch):
    real = standalone.Writer.expr

    def expr(self, v, depth=0):
        text = real(self, v, depth)
        return text + "  # OrderedDict(" if depth == 1 else text

    monkeypatch.setattr(standalone.Writer, "expr", expr)
    text = standalone.source(_cfg([{"op": "linear_act", "out": 4, "act": "relu"}], "vec", (6,)))
    assert "from collections import OrderedDict" in text


def _patched(text, body):
    """The standalone text with build_model wrapped by `body` (a function of the built model)."""
    return (
        text
        + f"\n\n_orig_build = build_model\n\n\ndef build_model(*_):\n    m = _orig_build()\n{body}\n    return m\n"
    )


def test_verify_tells_identical_equivalent_and_different():
    cfg = _cfg(
        [{"op": "linear_act", "out": 8, "act": "relu"}, {"op": "layer_norm", "axes": 1}],
        "vec",
        (6,),
    )
    text = standalone.source(cfg)
    assert standalone.verify(text, cfg) == "identical"
    redraw = "    g = torch.Generator().manual_seed(7)\n    for n, p in m.named_parameters():\n" \
             "        if p.dim() > 1:\n            p.data = p.data[torch.randperm(p.shape[0], generator=g)]"  # fmt: skip
    assert standalone.verify(_patched(text, redraw), cfg) == "equivalent"
    frozen = "    next(m.parameters()).requires_grad_(False)"
    with pytest.raises(Unsupported, match="trains other parameters"):
        standalone.verify(_patched(text, frozen), cfg)
    const = "    for n, p in m.named_parameters():\n        if p.dim() == 1 and bool((p == 1).all()):\n" \
            "            p.data.fill_(2.0)"  # fmt: skip
    with pytest.raises(Unsupported, match="initial value"):
        standalone.verify(_patched(text, const), cfg)
    spread = "    for p in m.parameters():\n        if p.dim() > 1:\n            p.data.mul_(10.0)"
    with pytest.raises(Unsupported, match="initial spread"):
        standalone.verify(_patched(text, spread), cfg)


def test_verified_source_wraps_other_errors(monkeypatch):
    def boom(*a, **k):
        raise KeyError("x")

    monkeypatch.setattr(standalone, "source", boom)
    with pytest.raises(Unsupported, match="KeyError"):
        standalone.verified_source({})

    def refuse(*a, **k):
        raise Unsupported("as it is")

    monkeypatch.setattr(standalone, "source", refuse)
    with pytest.raises(Unsupported, match="as it is"):
        standalone.verified_source({})


def test_convert_leaves_packages_it_cannot_rewrite(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(json.dumps({"arch": {"family": "mlp"}, "out_dim": 2}))
    logs = []
    assert standalone.convert(tmp_path, logs.append) is False and "no zoo spec" in logs[-1]
    cfg = _cfg([{"op": "linear_act", "out": 4, "act": "relu"}], "vec", (6,))
    (tmp_path / "config.yaml").write_text(json.dumps({"arch": cfg, "out_dim": 2}))

    def refuse(*a, **k):
        raise Unsupported("not today")

    monkeypatch.setattr(standalone, "write_package", refuse)
    assert standalone.convert(tmp_path, logs.append) is False and "not today" in logs[-1]
