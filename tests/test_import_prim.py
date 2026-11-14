"""v1.8: the basic layer library (dlnet/zoo/ops_prim.py) and the importer rebuild networks EXACTLY.

Every model here is written with the patterns published networks use; each is imported into a
card, written to YAML and read back (large constants in the .npz sidecar), built by the zoo, given
the original weights by layer name, and compared: the outputs must be equal (fidelity level B)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from dlnet import paper, paper_import, zoo


def roundtrip(model, dims, classes=None, tmp_path=None, tol=1e-6):
    model.eval()
    card = paper_import.card_of(model, "map" if len(dims) == 3 else "nd", dims, classes)
    if tmp_path is not None:  # through YAML and back, as `paper import` writes it
        f = tmp_path / "card.yaml"
        f.write_text(paper_import.to_yaml(card, f))
        card = {**paper.load(f), "head_ref": card.get("head_ref")}
    arch = paper.to_arch(card, classes or 1)
    mine = zoo.build_model(arch).eval()
    copied = paper_import.copy_weights(model, mine, card)
    assert not copied["unmatched_official"] and not copied["unmatched_card"], copied
    assert sum(p.numel() for p in mine.parameters()) == sum(p.numel() for p in model.parameters())
    x = torch.randn(3, *dims)
    with torch.no_grad():
        ya, yb = model(x), mine(x)
    assert ya.shape == yb.shape
    assert float((ya - yb).abs().max()) <= tol * max(1.0, float(ya.abs().max()))
    return card


class ConvMix(nn.Module):
    """Per-axis kernels, BN with another eps, LayerNorm2d (functional, channels-last), GroupNorm,
    InstanceNorm, many activations, pooling with every option, upsample, pixel shuffle, padding."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, (3, 1), padding=(1, 0), bias=False)
        self.bn = nn.BatchNorm2d(8, eps=1e-3, momentum=0.01)
        self.dw = nn.Conv2d(8, 8, 5, padding=4, dilation=2, groups=8, padding_mode="reflect")
        self.ln_w, self.ln_b = nn.Parameter(torch.randn(8)), nn.Parameter(torch.randn(8))
        self.gn = nn.GroupNorm(4, 8, eps=1e-6)
        self.inorm = nn.InstanceNorm2d(8, affine=True)
        self.acts = nn.ModuleList(
            [nn.LeakyReLU(0.2), nn.Hardswish(), nn.ELU(0.5), nn.Softplus(2.0, 10.0), nn.Mish()]
        )
        self.pool = nn.MaxPool2d(3, 2, 1, ceil_mode=True)
        self.avg = nn.AvgPool2d(2, count_include_pad=False, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.ps = nn.PixelShuffle(2)
        self.ct = nn.ConvTranspose2d(2, 4, 3, stride=2, padding=1, output_padding=1)
        self.fc = nn.Linear(4 * 8 * 8, 10)

    def forward(self, x):
        x = self.bn(self.c1(x))
        x = x + self.dw(x)
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y, (8,), self.ln_w, self.ln_b, 1e-6).permute(0, 3, 1, 2)
        x = self.inorm(self.gn(y))
        for a in self.acts:
            x = a(x)
        x = F.pad(self.avg(self.pool(x)), (1, 0, 0, 1), mode="replicate")
        x = self.ps(self.up(x))
        x = torch.clamp(self.ct(x), min=-2.0, max=3.0) * 0.5 + 1.0
        x = F.adaptive_avg_pool2d(x, (8, 8))
        return self.fc(torch.flatten(x, 1))


class TinyViT(nn.Module):
    """Patch embedding, class token and position embedding (parameters), a qkv permute that moves the
    batch axis then unbind(0), scaled dot-product attention, torch.nn.MultiheadAttention, layer
    scale, x[:, 0], a shared layer and a bias computed from parameters (F.linear)."""

    def __init__(self, d=16, heads=2):
        super().__init__()
        self.heads = heads
        self.patch = nn.Conv2d(3, d, 4, 4)
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 17, d))
        self.ln = nn.LayerNorm(d, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.randn(d)), nn.Parameter(torch.randn(d))
        self.proj = nn.Linear(d, d)
        self.gamma = nn.Parameter(torch.randn(d) * 0.1)
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(approximate="tanh"), nn.Linear(2 * d, d)
        )
        self.head = nn.Linear(d, 7)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        b, n, c = x.shape
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1) + self.pos
        h = self.ln(x)
        bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias), self.v_bias])
        qkv = (
            F.linear(h, self.qkv.weight, bias)
            .reshape(b, n + 1, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n + 1, c)
        x = x + self.gamma * self.proj(a)
        h = self.ln(x)  # the same LayerNorm again: shared weights
        x = x + self.mha(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.ln(x))
        return self.head(x[:, 0])


class Windows(nn.Module):
    """Window partition (the batch axis merges with the windows and comes back), a relative
    position bias gathered from a table with an index buffer, a shift with torch.roll, a mask kept as
    a plain tensor attribute, einsum, softmax."""

    def __init__(self, c=8, w=4, heads=2):
        super().__init__()
        self.w, self.heads = w, heads
        self.table = nn.Parameter(torch.randn((2 * w - 1) ** 2, heads))
        coords = torch.stack(
            torch.meshgrid(torch.arange(w), torch.arange(w), indexing="ij")
        ).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (w - 1)
        self.register_buffer("index", (rel[..., 0] * (2 * w - 1) + rel[..., 1]).flatten())
        self.mask = torch.zeros(4, w * w, w * w)  # plain attribute, not a buffer
        self.mask[1:, :, : w * w // 2] = -100.0
        self.qkv = nn.Linear(c, 3 * c)
        self.out = nn.Linear(c, c)
        self.fc = nn.Linear(c, 5)

    def forward(self, x):  # x: (B, C, H, W) with H = W = 2w
        b, c, h, w = x.shape
        x = torch.roll(x.permute(0, 2, 3, 1), shifts=(-1, -1), dims=(1, 2))
        win = (
            x.reshape(b, h // self.w, self.w, w // self.w, self.w, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, self.w * self.w, c)
        )
        q, k, v = (
            self.qkv(win)
            .reshape(win.shape[0], -1, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        bias = (
            self.table[self.index]
            .view(self.w * self.w, self.w * self.w, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) * (c // self.heads) ** -0.5 + bias
        attn = attn.view(b, 4, self.heads, self.w * self.w, self.w * self.w) + self.mask.unsqueeze(
            1
        ).unsqueeze(0)
        attn = attn.view(-1, self.heads, self.w * self.w, self.w * self.w).softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(win.shape[0], self.w * self.w, c)
        y = (
            self.out(y)
            .reshape(b, h // self.w, w // self.w, self.w, self.w, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, h, w, c)
        )
        return self.fc(y.mean((1, 2)))


class Seq(nn.Module):
    """Recurrent layers on (B, C, L): LSTM batch_first=False with a transpose, GRU bidirectional."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 6, num_layers=2)
        self.gru = nn.GRU(6, 3, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(6, 2)

    def forward(self, x):  # (B, 4, 12)
        y, _ = self.lstm(x.permute(2, 0, 1))
        z, _ = self.gru(y.transpose(0, 1))
        return self.fc(z[:, -1])


def test_convolutional_patterns(tmp_path):
    card = roundtrip(ConvMix(), (3, 16, 16), 10, tmp_path)
    mods = {n["module"] for n in card["architecture"]}
    assert {
        "conv",
        "batch_norm",
        "layer_norm",
        "group_norm",
        "instance_norm",
        "activation",
        "pooling",
        "pad",
        "upsample",
        "pixel_shuffle",
        "conv_t",
        "clamp",
        "scalar",
        "adaptive_pool",
        "permute",
    } <= mods


def test_transformer_patterns(tmp_path):
    card = roundtrip(TinyViT(), (3, 16, 16), 7, tmp_path)
    mods = [n["module"] for n in card["architecture"]]
    assert "mha" in mods and "attention" in mods and "static" in mods and "linear_fn" in mods
    assert any("share" in n for n in card["architecture"])  # the LayerNorm used three times
    assert card["head_ref"] == "head"


def test_window_patterns(tmp_path):
    card = roundtrip(Windows(), (8, 8, 8), 5, tmp_path, tol=1e-5)
    assert (tmp_path / "card.constants.npz").exists() is False  # the mask is small: inline
    statics = [n for n in card["architecture"] if n["module"] == "static"]
    assert statics and any(
        t.get("value") is not None for n in statics for t in n["args"]["tensors"]
    )


def test_large_constants_go_to_a_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_import, "SIDECAR_MIN", 10)
    roundtrip(Windows(), (8, 8, 8), 5, tmp_path, tol=1e-5)
    assert (tmp_path / "card.constants.npz").exists()
    assert "npz" in (tmp_path / "card.yaml").read_text()


def test_recurrent_patterns(tmp_path):
    card = roundtrip(Seq(), (4, 12), 2, tmp_path, tol=1e-5)
    assert [n["args"]["cell"] for n in card["architecture"] if n["module"] == "recurrent"] == [
        "lstm",
        "gru",
    ]


def test_unsupported_is_named():
    class Bad(nn.Module):
        def forward(self, x):
            return torch.linalg.svdvals(x.flatten(2))

    with pytest.raises(paper_import.ImportUnsupported, match="svdvals"):
        paper_import.card_of(Bad(), "map", (3, 4, 4))


@pytest.mark.parametrize("v, text", [(None, "null"), (1e-05, "1.0e-05"), (0.5, "0.5"), (True, "true"), (3, "3"),
                                     ("bhnd,bhmd->bhnm", '"bhnd,bhmd->bhnm"'), ("relu", "relu"), ("null", '"null"'),
                                     (math.inf, ".inf")])  # fmt: skip
def test_yaml_values_read_back(v, text):
    import yaml

    assert paper_import._flow(v) == text
    assert yaml.safe_load(f"a: {text}")["a"] == v


def test_primitive_ops_are_never_searched():
    from dlnet.zoo import OPS

    prim = [n for n, o in OPS.items() if "primitive" in o.tags]
    assert len(prim) >= 40
    assert all(o.tags == {"primitive"} for n, o in OPS.items() if n in prim)


# ----------------------------------------------------------------------------- every primitive

M = ("map", (6, 8, 8))
S = ("seq", (10, 6))
PRIM = [
    ({"op": "activation", "fn": "leaky_relu", "a": 0.2}, *M),
    ({"op": "activation", "fn": "softmax", "a": 1}, *M),
    ({"op": "activation", "fn": "glu", "a": 1}, *M),
    ({"op": "activation", "fn": "hardtanh", "a": -0.5, "b": 2.0}, *M),
    ({"op": "adaptive_pool", "mode": "max", "size": [3, 2]}, *M),
    ({"op": "as_kind", "to": "nd"}, *M),
    ({"op": "avgpool", "k": 2, "s": 2, "pad": 0, "ceil": False}, *M),
    ({"op": "batch_norm", "eps": 1e-3, "momentum": 0.01}, *M),
    ({"op": "channel_shuffle", "groups": 3}, *M),
    ({"op": "chunk_part", "chunks": 2, "axis": 1, "i": 1}, *M),
    ({"op": "clamp", "min": -0.5, "max": 0.5}, *M),
    (
        {
            "op": "conv",
            "out": 4,
            "k": [3, 1],
            "pad": [1, 0],
            "s": [2, 1],
            "padding_mode": "circular",
        },
        *M,
    ),
    ({"op": "conv_t", "out": 4, "k": 3, "s": 2, "pad": 1, "out_pad": 1}, *M),
    ({"op": "cumsum", "axis": 2}, *M),
    ({"op": "embedding", "num": 50, "dim": 4}, "ids", (50, 20, 7)),
    ({"op": "expand", "shape": [6, 8, 8]}, "map", (6, 1, 8)),
    ({"op": "fft", "fn": "rfft2", "axes": [2, 3], "norm": "ortho"}, *M),
    ({"op": "flatten_axes", "start": 2, "end": 3}, *M),
    ({"op": "flip", "axes": [3]}, *M),
    ({"op": "group_norm", "groups": 3, "eps": 1e-6}, *M),
    ({"op": "index_axis", "axis": 2, "index": [3, 1, 0]}, *M),
    ({"op": "instance_norm", "affine": True}, *M),
    ({"op": "layer_norm", "axes": 2, "eps": 1e-6, "bias": False}, *M),
    ({"op": "linear", "out": 5}, *M),
    ({"op": "local_response_norm", "size": 3}, *M),
    ({"op": "maxpool", "k": 3, "s": 2, "pad": 1, "ceil": True}, *M),
    ({"op": "normalize", "axis": 1}, *M),
    ({"op": "pad", "pads": [1, 2, 0, 1], "mode": "reflect"}, *M),
    ({"op": "patch_merge_pad"}, "nd", (7, 7, 4)),
    ({"op": "permute", "order": [0, 2, 3, 1]}, *M),
    ({"op": "pixel_shuffle", "r": 2, "inverse": True}, *M),
    ({"op": "pixel_shuffle", "r": 2}, "map", (8, 4, 4)),
    ({"op": "pooling", "mode": "lp", "p": 2.0, "k": 2}, *M),
    (
        {"op": "pooling", "mode": "avg", "k": [3, 1], "s": 1, "pad": [1, 0], "count_pad": False},
        *M,
    ),
    ({"op": "prelu", "n": 6}, *M),
    ({"op": "fill", "shape": [3, 4], "value": 1.5}, *M),
    ({"op": "repeat_interleave", "repeats": 2, "axis": 2}, *M),
    ({"op": "recurrent", "cell": "lstm", "hidden": 5, "layers": 2, "bidirectional": True}, *S),
    ({"op": "recurrent", "cell": "gru", "hidden": 4, "batch_first": True}, *S),
    ({"op": "reduce", "fn": "std", "axes": [2, 3], "keepdim": True}, *M),
    ({"op": "reshape", "shape": [6, 64]}, *M),
    ({"op": "rms_norm", "axes": 1}, *M),
    ({"op": "roll", "shifts": [2, -1], "axes": [2, 3]}, *M),
    ({"op": "scalar", "fn": "rdiv", "value": 2.0}, *M),
    ({"op": "select", "axis": 2, "index": 3}, *M),
    ({"op": "slice", "axis": 3, "start": 1, "stop": 7, "step": 2}, *M),
    ({"op": "split_part", "sizes": [2, 4], "axis": 1, "i": 1}, *M),
    ({"op": "squeeze", "axis": 2}, "map", (6, 1, 8)),
    ({"op": "transpose", "a": 2, "b": 3}, *M),
    ({"op": "unary", "fn": "erf"}, *M),
    ({"op": "unflatten", "axis": 1, "sizes": [2, 3]}, *M),
    ({"op": "unsqueeze", "axis": 1}, *M),
    ({"op": "upsample", "scale": 2, "mode": "bicubic", "align_corners": True}, *M),
    ({"op": "upsample", "size": [5, 7], "scale": None, "mode": "nearest"}, *M),
    (
        {"op": "window_attention", "heads": 2, "window": [4, 4], "shift": [2, 2], "version": 2},
        "nd",
        (8, 8, 6),
    ),
    ({"op": "window_attention", "heads": 2, "window": [4, 4], "shift": [0, 0]}, "nd", (8, 8, 6)),
]


def test_every_primitive_is_covered():
    from dlnet.zoo import OPS

    names = {n for n, o in OPS.items() if "primitive" in o.tags}
    assert names == {c[0]["op"] for c in PRIM}, names ^ {c[0]["op"] for c in PRIM}


@pytest.mark.parametrize(
    ("spec", "kind", "dims"), PRIM, ids=lambda v: v["op"] if isinstance(v, dict) else None
)
def test_every_primitive_builds_trains_and_is_written_as_standalone_source(spec, kind, dims):
    from dlnet import standalone

    after = (
        [{"op": "unary", "fn": "real"}] if spec["op"] == "fft" else []
    )  # complex -> real for the head
    cfg = {"spec": {"op": "graph", "nodes": [spec, *after, {"op": "as_kind", "to": "nd"}, {"op": "flatten_axes", "start": 1}]},
           "input": {"kind": kind, "dims": list(dims)}, "head": "cls", "out_dim": 3}  # fmt: skip
    zoo.check(cfg["spec"], kind, dims)
    m = zoo.build_model(cfg)
    x = (
        torch.stack([torch.randint(0, c, (2,)) for c in dims], 1)
        if kind == "ids"
        else torch.randn(2, *dims)
    )
    out = m(x)
    assert out.shape == (2, 3)
    if out.dtype.is_floating_point:
        out.sum().backward()
    assert standalone.verified_source(cfg)[1] == "identical"
