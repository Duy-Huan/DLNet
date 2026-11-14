"""`dlnet paper import` (dlnet/paper_import.py): every mapping and every refusal the other tests do not
reach (v1.9 coverage). Small models, each written with one pattern; rebuilt models must give the
same output as the original, refused ones must name the reason."""

from __future__ import annotations

import operator

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F
from test_import_prim import roundtrip
from torch import nn

from dlnet import paper_import as pi
from dlnet.paper_import import ImportUnsupported


def unsup(model, dims, match, kind=None):
    kind = kind or ("map" if len(dims) == 3 else "nd")
    with pytest.raises(ImportUnsupported, match=match):
        pi.card_of(model.eval(), kind, dims, None)


class Fn(nn.Module):
    """A model whose forward is a function (and optional layers)."""

    def __init__(self, fn, **layers):
        super().__init__()
        self.fn = fn
        for k, v in layers.items():
            setattr(self, k, v)

    def forward(self, x):
        return self.fn(self, x)


# ----------------------------------------------------------------------------- layers


def test_layer_options():
    lstm = nn.LSTM(4, 6, num_layers=2, dropout=0.1, proj_size=3, batch_first=True)
    m = Fn(lambda s, x: s.fc(s.rnn(s.lstm(x)[0])[0]), lstm=lstm,
           rnn=nn.RNN(3, 5, nonlinearity="relu", batch_first=True), fc=nn.Linear(5, 2))  # fmt: skip
    card = roundtrip(m, (7, 4), None, tol=1e-5)
    rec = [n for n in card["architecture"] if n["module"] == "recurrent"]
    assert rec[0]["args"]["proj"] == 3 and rec[1]["args"]["nonlinearity"] == "relu"
    emb = Fn(lambda s, x: s.e(x).mean(1), e=nn.Embedding(10, 4))
    card = pi.card_of(emb.eval(), "ids", (10, 10, 10), None)
    assert card["architecture"][0]["module"] == "embedding"
    unsup(Fn(lambda s, x: s.p(x)[0], p=nn.AdaptiveMaxPool2d(2, return_indices=True)), (3, 4, 4),
          "return_indices")  # fmt: skip


def test_torchvision_window_attention_is_a_library_leaf():
    from dlnet.zoo.ops_lib import WindowAttention

    class ShiftedWindowAttention(
        WindowAttention
    ):  # what torchvision's class looks like to the importer
        pass

    ShiftedWindowAttention.__module__ = "torchvision.models.swin_transformer"
    m = Fn(lambda s, x: s.a(x), a=ShiftedWindowAttention(8, [4, 4], [2, 2], 2))
    assert pi.is_leaf(m.a)
    card = roundtrip(m, (8, 8, 8), None, tol=1e-5)
    assert card["architecture"][0]["module"] == "window_attention"


def test_multihead_attention_options():
    mha = nn.MultiheadAttention(
        8, 2, batch_first=True, add_bias_kv=True, add_zero_attn=True, kdim=6, vdim=5
    )
    m = Fn(lambda s, x: s.a(x, x[..., :6], x[..., :5])[0], a=mha)
    card = roundtrip(m, (3, 8), None, tol=1e-5)
    args = next(n for n in card["architecture"] if n["module"] == "mha")["args"]
    assert args["kdim"] == 6 and args["add_zero_attn"]
    unsup(Fn(lambda s, x: s.a(x, x, x)[1], a=nn.MultiheadAttention(8, 2, batch_first=True)), (3, 8),
          "only the output")  # fmt: skip
    mask = torch.zeros(3, 3, dtype=torch.bool)
    unsup(Fn(lambda s, x: s.a(x, x, x, attn_mask=mask)[0], a=nn.MultiheadAttention(8, 2, batch_first=True)),
          (3, 8), "attention masks")  # fmt: skip


def test_tuple_items():
    m = Fn(lambda s, x: x.max(1)[0] + x.min(1, keepdim=False)[0])
    roundtrip(m, (3, 4), None)
    unsup(Fn(lambda s, x: x.max(1)[1].float()), (3, 4), "indices")
    unsup(Fn(lambda s, x: torch.sort(x, 1)[0]), (3, 4), "indices")
    unsup(Fn(lambda s, x: torch.linalg.qr(x)[0]), (4, 4), "no card equivalent")


# ----------------------------------------------------------------------------- tracing


def test_trace_failures_are_named():
    with pytest.raises(ImportUnsupported, match="example input"):
        pi.nodes_of(nn.Linear(2, 2), None, None)

    class Raises(nn.Module):
        def forward(self, x):
            raise ImportUnsupported("my own reason")

    unsup(Raises(), (4,), "my own reason")

    class NeedsATensor(nn.Module):
        def forward(self, x):
            if type(x) is not torch.Tensor:
                raise TypeError("only real tensors")
            return x

    unsup(NeedsATensor(), (4,), "torch.fx cannot trace")


def test_concrete_tracing_of_python_on_sizes():
    """Control flow on sizes, iterating a shape, float() / index() of a size, a device read, in-place
    item assignment on a tensor made in forward, and attributes set in forward."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.scratch = None

        def forward(self, x):  # (B, 3, 4)
            dims = [d for d in x.shape]  # Tracer.iter
            if dims[1] == 3:
                scale = float(x.shape[2]) / 4.0
                grid = torch.zeros(1, x.shape[1], 4, device=x.device)  # made from a size: a proxy
                grid[..., 0] = 1.0  # __setitem__ on a tensor made from sizes
                w = torch.ones(4)[operator.index(x.shape[1]) - 3 :]  # __index__
                self.scratch = x * 2  # an attribute set in forward (taken back after tracing)
                self.extra = x + 1
                return self.fc(x * scale + grid) + w
            return x

    m = M().eval()
    pi.card_of(m, "nd", (3, 4), None)
    assert m.scratch is None and not hasattr(
        m, "extra"
    )  # the proxies set in forward were taken back
    roundtrip(m, (3, 4), None)


def test_concrete_tracer_keys():
    t = pi._ConcreteTracer(torch.zeros(1))
    node = type("N", (), {})()
    t.values[node] = {"a": 1}
    assert list(t.keys(type("P", (), {"node": node})())) == ["a"]


# ----------------------------------------------------------------------------- the main loop


def test_constants_from_sizes_used_directly():
    """A constant computed from sizes only (no weight) added to the activations: written once."""

    def f(s, x):
        pos = torch.arange(x.shape[1], dtype=torch.float32)[None, :, None] / 3.0
        idx = torch.arange(x.shape[1])[None, :, None].expand(1, -1, 4)
        return s.fc(x + pos) + idx

    card = roundtrip(Fn(f, fc=nn.Linear(4, 4)), (3, 4), None)
    consts = [
        n for n in card["architecture"] if n["module"] == "static" and not n["args"]["program"]
    ]
    assert consts and any(t.get("dtype") == "int64" for n in consts for t in n["args"]["tensors"])


def test_large_constants_are_refused(monkeypatch):
    monkeypatch.setattr(pi, "INLINE_MAX", 2)

    def f(s, x):
        return x + torch.arange(x.shape[1], dtype=torch.float32)

    unsup(Fn(f), (4,), "constant tensor of 4 values")

    class Attr(nn.Module):
        def __init__(self):
            super().__init__()
            self.t = torch.ones(4)  # a plain tensor attribute

        def forward(self, x):
            return x + self.t

    unsup(Attr(), (4,), "used as data")


def test_arithmetic_comparisons_and_aliases():
    def f(s, x):
        a = (
            (x > 0.1).float() + (0.2 < x).float() + (x >= x * 0.5).float()
        )  # tensor-tensor, flipped scalar
        n = x.shape[1] * 2  # arithmetic on sizes
        y = torch.cat([x], 1) + x.contiguous() + n
        y = (
            y.expand(*y.shape)
            + y.unsqueeze(1).expand(y.shape[0], 1, -1).squeeze(1).unsqueeze(1).squeeze()
        )
        y = y + y.permute(0, 1).flatten(0).reshape(y.shape) + (~(x > 0)).float()
        return (
            y + a + F.gelu(x, approximate="tanh") + torch.tensor(2.0) * x + torch.full_like(x, 3.0)
        )

    roundtrip(Fn(f), (4,), None, tol=1e-5)


def test_main_loop_refusals():
    unsup(Fn(lambda s, x: torch.div(x, 2, rounding_mode="floor")), (4,), "rounding_mode")
    unsup(Fn(lambda s, x: x.unsqueeze(0).expand(3, -1, -1)), (4,), "expand of the batch axis")
    unsup(Fn(lambda s, x: x.norm()), (4,), "norm over every axis")
    unsup(
        Fn(lambda s, x: x.repeat_interleave(torch.tensor([1, 2, 1, 1]), 1)), (4,), "tensor repeats"
    )
    unsup(Fn(lambda s, x: x + x.new_zeros(4)), (4,), "without the batch axis first")
    unsup(Fn(lambda s, x: x[0]), (4,), "indexing the batch axis")
    unsup(Fn(lambda s, x: x[[0, 1]]), (4,), "indexing the batch axis")
    unsup(Fn(lambda s, x: x[1:]), (4,), "slicing the batch axis")
    unsup(Fn(lambda s, x: x[:, True]), (4,), "index True")
    unsup(Fn(lambda s, x: x[:, torch.tensor([0, 1])]), (4,), "indexing with a tensor")
    unsup(Fn(lambda s, x: x.svd()[0]), (4, 4), "")
    unsup(Fn(lambda s, x: x), (4,), "not computed by a layer")
    unsup(Fn(lambda s, x: (x, x)), (4,), "several outputs")
    unsup(Fn(lambda s, x: {"y": x * 2}), (4,), "dict")
    unsup(Fn(lambda s, x: x.tolist() and x), (4,), "")


def test_output_read_from_an_earlier_node_and_head_detection():
    """The output is an earlier node (a later one is unused): read with as_kind. A Linear head onto
    the classes read from an earlier node gets a pre_head reader."""

    def f(s, x):
        y = s.a(x)
        s.b(x)
        return y

    card = roundtrip(Fn(f, a=nn.Linear(4, 4), b=nn.Linear(4, 3)), (4,), None)
    assert card["architecture"][-1]["module"] == "as_kind"

    def g(s, x):
        y = s.a(x)
        s.b(y)  # computed, not used: the head reads an earlier node
        return s.head(y)

    card = pi.card_of(Fn(g, a=nn.Linear(4, 4), b=nn.Linear(4, 4), head=nn.Linear(4, 3)).eval(),
                      "nd", (4,), 3)  # fmt: skip
    assert any(n["id"] == "pre_head" for n in card["architecture"])


def test_pooling_and_upsampling_functions():
    def f(s, x):
        y = F.avg_pool2d(x, 2, 2, 0, False, False)
        y = F.interpolate(y, size=(4, 4), mode="nearest") + F.interpolate(
            y, scale_factor=(2, 2), mode="nearest"
        )
        return y

    roundtrip(Fn(f), (3, 4, 4), None)


def test_time_major_tensors():
    """LSTM(batch_first=False) on (L, B, C): items of the output, a transpose back, other uses refused."""

    def f(s, x):
        t = x.transpose(0, 1)  # (L, B, C)
        y = s.rnn(t)[0]
        return y[-1] + y[0, :] + y.permute(1, 0, 2).mean(1)

    roundtrip(Fn(f, rnn=nn.GRU(4, 4)), (5, 4), None, tol=1e-5)
    unsup(Fn(lambda s, x: s.rnn(x.transpose(0, 1))[1], rnn=nn.GRU(4, 4)), (5, 4), "only the output")
    unsup(
        Fn(lambda s, x: s.rnn(x)[0], rnn=nn.GRU(4, 4)), (5, 4), "batch_first=False on a batch-first"
    )
    unsup(Fn(lambda s, x: s.rnn(x.transpose(0, 1))[0].transpose(1, 2), rnn=nn.GRU(4, 4)), (5, 4),
          "time-major")  # fmt: skip


def test_moved_batch_axis():
    def f(s, x):  # (B, 2, 3, 4): permute moves the batch, then [i]
        p = x.permute(1, 0, 2, 3)
        return p[0] + p[1]

    roundtrip(Fn(f), (2, 3, 4), None)
    unsup(Fn(lambda s, x: x.permute(1, 2, 0) * 2), (2, 3), "moves the batch axis")
    unsup(Fn(lambda s, x: x.permute(1, 2, 0)[0]), (2, 3), "batch axis does not come back first")


# ----------------------------------------------------------------------------- weights (static)


def test_weight_computations():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.t = nn.Parameter(torch.randn(2, 3, 4))
            self.i = nn.Parameter(torch.randn(6))

        def forward(self, x):
            w = self.t.squeeze(0) if self.t.shape[0] == 1 else self.t.sum(0)
            w = w + self.t.split([1, 1], 0)[1].squeeze(0) + self.t.unbind(0)[0]
            w = w + (2.0 - self.t[0]) + (1.0 / (self.t[1].abs() + 1)) + self.t[0].long().float()
            w = w + self.t.mean(0).repeat_interleave(1, 0) + F.dropout(self.t[1], 0.0, False)
            v = self.i.view(2, 3).sum(0, keepdim=True)
            return x * w.sum(0) + v.sum(1)

    roundtrip(M(), (4,), None, tol=1e-5)


def test_weight_computation_refusals(monkeypatch):
    class P(nn.Module):
        def __init__(self, f, **kw):
            super().__init__()
            self.t = nn.Parameter(torch.randn(1, 2, 4, 4))
            self.f = f
            for k, v in kw.items():
                setattr(self, k, v)

        def forward(self, x):
            return x + self.f(self)

    unsup(
        P(lambda s: F.pad(s.t, (1, 1, 1, 1), mode="reflect")[..., 1:-1, 1:-1]),
        (2, 4, 4),
        "pad mode",
    )
    unsup(P(lambda s: s.t.repeat_interleave(torch.tensor([1, 1]), 1)), (2, 4, 4), "tensor repeats")
    unsup(
        P(lambda s: s.n(s.t), n=nn.BatchNorm2d(2)), (2, 4, 4), "inside a computation on the weights"
    )
    unsup(P(lambda s: torch.linalg.inv(s.t)), (2, 4, 4), "no static equivalent")


def get_relative_position_bias(table, index, window):
    n = window[0] * window[1]
    return table[index].view(n, n, -1).permute(2, 0, 1).contiguous().unsqueeze(0)


torch.fx.wrap("get_relative_position_bias")  # as torchvision does


def test_torchvision_relative_position_bias_helper():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(torch.randn(9, 2))
            self.register_buffer("idx", torch.randint(0, 9, (16,)))

        def forward(self, x):  # (B, 2, 4, 4)
            return x + get_relative_position_bias(self.table, self.idx, [2, 2])[0]

    roundtrip(M(), (2, 4, 4), None, tol=1e-5)


# ----------------------------------------------------------------------------- functional layers


def test_functional_layer_refusals():
    class C(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 4, 3, 3))
            self.b = nn.Parameter(torch.randn(4))
            self.rm, self.rv = torch.zeros(4), torch.ones(4)
            self.f = f

        def forward(self, x):
            return self.f(self, x)

    unsup(C(lambda s, x: F.conv2d(x, s.w, padding="same")), (4, 5, 5), "padding='same'")
    unsup(
        C(lambda s, x: F.batch_norm(x, s.rm, s.rv, s.b, s.b, training=True)),
        (4, 5, 5),
        "training mode",
    )
    unsup(
        C(lambda s, x: F.layer_norm(x, (5,), s.b[:1].expand(5) * 1.0)),
        (4, 5, 5),
        "computed, not stored",
    )


# ----------------------------------------------------------------------------- draft / yaml / checks


TINY = """
from torch import nn


class Tiny(nn.Sequential):
    def __init__(self, head=True):
        super().__init__(nn.Linear(4, 4), nn.ReLU(), *([nn.Linear(4, 3)] if head else []))
"""


def test_draft_metadata_and_errors(tmp_path):
    (tmp_path / "tiny.py").write_text(TINY)
    off = {"repo": "https://x", "commit": "abc", "entry": "tiny.py:Tiny", "sys_path": ["."]}
    text = pi.draft(off, tmp_path, "nd", [4], "Tiny", classes=3, meta={"year": 2024, "venue": ""})
    assert "year: 2024" in text and 'sys_path: ["."]' in text
    with pytest.raises(ImportUnsupported, match="no final Linear"):
        pi.draft({**off, "init_args": {"head": False}}, tmp_path, "nd", [4], "NoHead", classes=3)
    m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 3))
    card = pi.card_of(m.eval(), "nd", [4], 3)
    card["official"] = {"repo": "https://x"}
    assert "official:" in pi.to_yaml(card)
    assert pi._flow(float("nan")) == ".nan"
    g = torch.fx.symbolic_trace(m).graph
    assert repr(pi._Resolved(next(iter(g.nodes)), {})).startswith("<")


def test_rebuild_check_reports_other_levels():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.unused = nn.Parameter(torch.ones(3))

        def forward(self, x):
            return self.fc(x)

    r = pi.rebuild_check(Odd().eval(), "nd", (4,))
    assert r["level"] == "C" and "same tensors" in r["reason"]


def test_copy_weights_reports_cards_without_names_and_shape_clashes():
    from dlnet import paper, zoo

    m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    card = pi.card_of(m.eval(), "nd", (4,), None)
    mine = zoo.build_model(paper.to_arch(card, 1))
    del card["architecture"][0]["ref"]
    assert pi.copy_weights(m, mine, card)["unmatched_card"]
    card = pi.card_of(m.eval(), "nd", (4,), None)
    card["architecture"][0]["ref"] = "2"  # points at the other Linear: shapes differ
    assert any("vs" in u for u in pi.copy_weights(m, mine, card)["unmatched_card"])


def _patch_merging_pad(x):
    h, w, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
    return torch.cat(
        [
            x[..., 0::2, 0::2, :],
            x[..., 1::2, 0::2, :],
            x[..., 0::2, 1::2, :],
            x[..., 1::2, 1::2, :],
        ],
        -1,
    )


torch.fx.wrap("_patch_merging_pad")  # as torchvision does


def test_more_main_loop_mappings():
    def f(s, x):  # (B, 3, 4)
        y = x.unsqueeze(1).expand(-1, 2, -1, -1).sum(1)  # expand of a non-batch axis
        y = (
            y + x.squeeze(1) + x * x.shape[1]
        )  # squeeze of an axis that is not 1; a size as a number
        y = y + x.unflatten(2, (2, 2)).flatten(2) + x[...] + x[:]
        y = y + torch.full((x.shape[0], 3, 4), 2.0)  # a constant as large as the batch
        return y

    roundtrip(Fn(f), (3, 4), None)
    m = Fn(lambda s, x: s.l(_patch_merging_pad(x)), l=nn.Linear(16, 4))
    card = roundtrip(m, (4, 4, 4), None)
    assert any(n["module"] == "patch_merge_pad" for n in card["architecture"])
    unsup(
        Fn(lambda s, x: s.p(x)[0], p=nn.MaxPool2d(2, return_indices=True)),
        (3, 4, 4),
        "tuple|return_indices",
    )


def test_time_major_round_trips():
    def f(s, x):  # (B, L, C) -> (L, B, C) by permute -> GRU -> slice of time -> back to batch first
        y = s.rnn(x.permute(1, 0, 2))[0]
        return y[1:].permute(1, 0, 2).mean(1)

    roundtrip(Fn(f, rnn=nn.GRU(4, 4)), (5, 4), None, tol=1e-5)


def test_static_node_that_only_renames_a_weight():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.t = nn.Parameter(torch.randn(4))

        def forward(self, x):
            return x * self.t.contiguous().detach() * self.t.sigmoid()

    roundtrip(M(), (4,), None)


def test_rebuild_check_reports_different_output_shapes(monkeypatch):
    from dlnet import zoo

    class Wider(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)[:, :1]

    real = zoo.build_model
    monkeypatch.setattr(zoo, "build_model", lambda arch: Wider(real(arch)))
    monkeypatch.setattr(
        pi, "copy_weights", lambda *a, **k: {"unmatched_official": [], "unmatched_card": []}
    )
    r = pi.rebuild_check(nn.Sequential(nn.Linear(4, 3)).eval(), "nd", (4,))
    assert r["level"] == "C" and "output shapes differ" in r["reason"]
