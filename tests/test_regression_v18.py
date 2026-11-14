"""Regression tests for the bugs found and fixed in v1.8.0 - v1.8.2 (skills/audit-and-regression).

Each test failed on the code before its fix. They are small on purpose: one model or one call per bug.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

torch = pytest.importorskip("torch")

import torch.nn.functional as F
from torch import nn

from dlnet import paper, paper_import, zoo


def _rebuild(model, dims, classes=None, batch=3):
    """Import, build, copy weights, compare on `batch` rows (not the batch of the import run)."""
    model.eval()
    kind = "map" if len(dims) == 3 else "nd"
    card = paper_import.card_of(model, kind, dims, classes)
    mine = zoo.build_model(paper.to_arch(card, classes or 1)).eval()
    copied = paper_import.copy_weights(model, mine, card)
    assert not copied["unmatched_official"] and not copied["unmatched_card"], copied
    x = torch.randn(batch, *dims)
    with torch.no_grad():
        ya, yb = model(x), mine(x)
    assert ya.shape == yb.shape
    return float((ya - yb).abs().max()), card


# ------------------------------------------------------------------- zoo head (v1.8.1)
def test_head_of_a_one_axis_nd_feature_keeps_the_batch():
    """`x.mean(())` reduced every axis: a (B, 4) feature of kind nd gave one number for the batch."""
    from dlnet.zoo.core import Head

    h = Head("nd", (4,), 2, "cls")
    y = h(torch.randn(5, 4))
    assert y.shape == (5, 2)


# ------------------------------------------------------------ graph memory (v1.8.1)
def test_graph_frees_outputs_but_keeps_the_ones_read_later():
    """Graph.forward frees each output after its last reader; a node read twice, far apart, and
    the output node itself must survive."""
    arch = {"op": "graph", "nodes": [
        {"id": "a", "op": "linear", "out": 4},
        {"id": "b", "op": "activation", "fn": "relu"},
        {"id": "c", "op": "activation", "fn": "tanh"},
        {"id": "d", "op": "add", "from": ["a", "c"]},
    ]}  # fmt: skip
    from dlnet.zoo.core import _build

    m, _, _ = _build(arch, "vec", (3,))
    x = torch.randn(2, 3)
    a = m.mods[0](x)
    want = a + torch.tanh(torch.relu(a))
    assert torch.allclose(m(x), want)
    assert torch.allclose(m(x), want)  # twice: no state kept between calls


# ------------------------------------------------------------ importer (v1.8.1)
def test_in_place_op_result_is_what_later_reads_see():
    """NFNet: `out.mul_(gain)` then `out * alpha`: the card skipped the gain (max diff 0.025)."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.gain = nn.Parameter(torch.tensor(0.3))

        def forward(self, x):
            out = self.fc(x)
            out.mul_(self.gain)
            return out * 0.5 + x

    diff, _ = _rebuild(M(), (4,))
    assert diff <= 1e-6


def test_batch_sized_expand_of_a_parameter_is_not_frozen_to_the_import_batch():
    """`cls.expand(B, -1, -1)`: B came from the input, but the card wrote B of the import run."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls = nn.Parameter(torch.randn(1, 1, 4))
            self.fc = nn.Linear(4, 4)

        def forward(self, x):  # (B, 3, 4)
            t = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1)
            return self.fc(t)

    diff, _ = _rebuild(M(), (3, 4), batch=5)
    assert diff <= 1e-6


def test_trace_leaves_no_proxy_on_the_model():
    """A forward that caches a tensor (self.rel = ..., a dict cache) kept torch.fx proxies on the model:
    the next forward call failed, and the second trace read a stale proxy (KeyError)."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.cache = {}
            self.register_buffer("rel", torch.zeros(1), persistent=False)

        def forward(self, x):
            if (
                x.shape[-1] == 4
            ):  # control flow on a size: the plain trace fails, the concrete one runs
                self.rel = self.fc.weight.sum(0) * 0.0 + torch.arange(4.0)
                self.cache["b"] = self.fc.bias * 2.0
            return self.fc(x) + self.rel + self.cache["b"]

    m = M().eval()
    paper_import.card_of(m, "nd", (4,), None)
    assert not isinstance(m.rel, torch.fx.Proxy)
    assert not any(isinstance(v, torch.fx.Proxy) for v in m.cache.values())
    m(torch.randn(2, 4))  # the model still runs


def test_scalar_parameter_weights_are_copied():
    """A 0-d parameter is written [1] in the card: copy_weights refused the shapes (C level)."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.Parameter(torch.tensor(1.7))
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x * self.g)

    diff, _ = _rebuild(M(), (4,))
    assert diff <= 1e-6


def test_stack_on_axis_zero_then_sum_is_rebuilt():
    """EfficientViM: stack(xs, 0).sum(0) made the card reduce the batch axis (ValueError)."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            y = self.fc(x)
            return torch.stack([y, 2 * y, x], 0).sum(0)

    diff, _ = _rebuild(M(), (4,))
    assert diff <= 1e-6


def test_parameter_repeated_to_the_batch_then_a_linear():
    """XCiT / PE-Core: a parameter made batch-sized (repeat / expand) and given to a Linear."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Parameter(torch.randn(1, 2, 4))
            self.lin = nn.Linear(4, 4)

        def forward(self, x):  # (B, 2, 4)
            return (
                self.lin(self.q.repeat(x.shape[0], 1, 1))
                + self.lin(self.q.expand(x.shape[0], -1, -1))
                + x
            )

    diff, _ = _rebuild(M(), (2, 4), batch=4)
    assert diff <= 1e-6


def test_static_nodes_share_their_parameter_inputs():
    """Two computations on the same table made two copies of it: the card had more parameters."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.t = nn.Parameter(torch.randn(4))

        def forward(self, x):
            return x * self.t.sigmoid() + x * self.t.exp()

    m = M().eval()
    card = paper_import.card_of(m, "nd", (4,), None)
    built = zoo.build_model(paper.to_arch(card, 1))
    assert sum(p.numel() for n, p in built.named_parameters() if not n.startswith("head")) == 4


def test_functional_rms_norm_and_unfold_are_imported():
    """F.rms_norm, Tensor.unfold, F.unfold / F.fold had no card equivalent (Sapiens, HaloNet, VOLO)."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.rand(4) + 0.5)

        def forward(self, x):  # (B, 4, 6, 6)
            y = F.rms_norm(x.permute(0, 2, 3, 1), (4,), self.w).permute(0, 3, 1, 2)
            cols = F.unfold(y, 2, stride=2)
            y = F.fold(cols, (6, 6), 2, stride=2) + y.unfold(2, 3, 3).unfold(3, 3, 3).mean(
                (-1, -2)
            ).repeat_interleave(3, 2).repeat_interleave(3, 3)
            return y

    diff, _ = _rebuild(M(), (4, 6, 6))
    assert diff <= 1e-5


# ------------------------------------------------------------ UI in a container (v1.8.2)
def test_ui_listens_on_localhost_unless_told_otherwise(monkeypatch, tmp_path):
    import argparse

    from dlnet import ui_server

    monkeypatch.delenv("DLNET_UI_BIND", raising=False)
    sub = argparse.ArgumentParser().add_subparsers()
    assert ui_server.add_parser(sub).parse_args([]).bind == "127.0.0.1"


def test_ui_refuses_a_request_naming_another_host(tmp_path):
    """Bound to 0.0.0.0 in a container, the UI must still refuse a page that names another host
    (DNS rebinding)."""
    import threading
    import urllib.error
    import urllib.request

    from dlnet import ui_server

    srv = ui_server.make_server(tmp_path / "ws", 0, "127.0.0.1")
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/state", headers={"Host": f"evil.example:{port}"}
        )
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        assert e.value.code == 403
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/", headers={"Host": f"evil.example:{port}"}
        )
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        assert e.value.code == 403
    finally:
        srv.shutdown()
        srv.server_close()
