"""Regression tests for the bugs found while raising coverage to 100 % (v1.9.0).

Each test failed on the code before its fix.
"""

from __future__ import annotations

import socket

import pytest

pytestmark = pytest.mark.regression

torch = pytest.importorskip("torch")

from torch import nn

from dlnet import mcp_http, mcp_tools, paper, paper_import, zoo


class Fn(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.f = f

    def forward(self, x):
        return self.f(self.lin(x))


def _rebuild(model, dims=(4,)):
    model.eval()
    card = paper_import.card_of(model, "nd", dims, None)
    mine = zoo.build_model(paper.to_arch(card, 1)).eval()
    paper_import.copy_weights(model, mine, card)
    x = torch.randn(3, *dims)
    with torch.no_grad():
        return float((model(x) - mine(x)).abs().max())


def test_float_of_a_mask_is_a_cast_not_an_alias():
    """`(x > a).float() + (x > b).float()` was rebuilt as bool + bool, which torch computes as OR."""
    m = Fn(lambda x: (x > 0).float() + (x > -0.5).float())
    assert _rebuild(m) == 0.0


def test_shape_of_an_attribute_proxy_is_known():
    """`len(x.shape)` and `*x.shape` failed: torch.fx attribute proxies carried no value."""
    m = Fn(lambda x: x.reshape(*x.shape) * len(x.shape))
    assert _rebuild(m) == 0.0


def test_index_true_is_refused_not_read_as_one():
    """`x[:, True]` was imported as `x[:, 1]`."""
    with pytest.raises(paper_import.ImportUnsupported, match="index True"):
        paper_import.card_of(Fn(lambda x: x[:, True]).eval(), "nd", (4,), None)


def test_ipv6_server_leaves_the_ipv4_class_alone(tmp_path):
    """An IPv6 `dlnet serve` changed `Server.address_family` for every later server."""
    svc = mcp_tools.Service(tmp_path / "ws")
    try:
        mcp_http.make_server(
            svc, mcp_http.Config(host="::1", port=0, token="t" * 32)
        ).server_close()
    except OSError:
        pass
    assert mcp_http.Server.address_family == socket.AF_INET
    svc.close()
