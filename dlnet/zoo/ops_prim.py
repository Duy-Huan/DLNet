"""The basic layers of torch.nn and the tensor operations of torch, as card nodes (v1.8).

A published network is first of all built from these: convolutions with any kernel / stride /
padding per axis, every normalisation and activation of torch.nn with its exact settings, pooling,
resampling, recurrent layers, attention, and the tensor operations between them (reshape, permute,
slicing, reductions, arithmetic with a constant, learned tensors such as a class token or a
position embedding). `dlnet paper import` writes the authors' model as nodes of this library and
`paper verify` checks the result against their code, so each op does exactly what torch does with
the same arguments (each wraps the torch.nn module or the torch function of the same name).

Every op accepts every tensor kind, plus "nd": a tensor whose layout is none of the named kinds
(after a permute to channels-last, a reshape into attention heads, ...). Spatial ops on "nd" use
the number of axes: (C, L) -> 1-d, (C, H, W) -> 2-d, (C, D, H, W) -> 3-d. Axis numbers (`dim`,
`order`, ...) count the batch axis as 0, as in torch. Ops that change the layout return "nd" (a
1-d result is "vec"); `as_kind` relabels a tensor for the blocks of the zoo.

All carry the tag "primitive" only: the random grammar and the mutations of the search never pick
them; they exist to write networks exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .core import KINDS, ND, op

ANY = KINDS
P = {"primitive"}


def nd_of(kind, dims) -> int:
    """Spatial dimensionality: from the kind, or from the number of axes of an "nd" tensor."""
    if kind in ND:
        return ND[kind]
    n = len(dims) - 1
    if n not in (1, 2, 3):
        raise ValueError(
            f"a spatial layer needs (C, L), (C, H, W) or (C, D, H, W), got {tuple(dims)}"
        )
    return n


def tup(v, n):
    """An int, or one value per axis (a list), as the tuple torch.nn takes."""
    if isinstance(v, list | tuple):
        if len(v) != n:
            raise ValueError(f"{list(v)}: {n} values expected")
        return tuple(int(x) for x in v)
    return (int(v),) * n


class Fn(nn.Module):
    """A parameter-free torch function with fixed arguments (shown by repr for the diagrams)."""

    def __init__(self, name, fn, **kw):
        super().__init__()
        self.name, self.fn, self.kw = name, fn, kw

    def forward(self, x):
        return self.fn(x, **self.kw)

    def extra_repr(self):
        return f"{self.name}, {self.kw}"


class Reshape(nn.Module):
    def __init__(self, shape, keep_batch=False):
        super().__init__()
        self.shape, self.keep_batch = tuple(shape), keep_batch

    def forward(self, x):
        return x.reshape(x.shape[0] if self.keep_batch else -1, *self.shape)


class Expand(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.expand(x.shape[0], *self.shape)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c = x.shape[:2]
        return (
            x.reshape(b, self.groups, c // self.groups, *x.shape[2:])
            .transpose(1, 2)
            .reshape(x.shape)
        )


class Slice(nn.Module):
    def __init__(self, axis, start, stop, step):
        super().__init__()
        self.axis, self.start, self.stop, self.step = axis, start, stop, step

    def forward(self, x):
        return x[(slice(None),) * self.axis + (slice(self.start, self.stop, self.step),)]


class ChunkPart(nn.Module):
    def __init__(self, chunks, axis, i):
        super().__init__()
        self.chunks, self.axis, self.i = chunks, axis, i

    def forward(self, x):
        return torch.chunk(x, self.chunks, self.axis)[self.i]


class SplitPart(nn.Module):
    def __init__(self, sizes, axis, i):
        super().__init__()
        self.sizes, self.axis, self.i = sizes, axis, i

    def forward(self, x):
        return torch.split(x, self.sizes, self.axis)[self.i]


class Reduce(nn.Module):
    def __init__(self, fn, axes, keepdim=False, correction=1):
        super().__init__()
        self.fn, self.axes, self.keepdim, self.correction = fn, tuple(axes), keepdim, correction

    def forward(self, x):
        f, ax, kd = self.fn, self.axes, self.keepdim
        if f == "mean":
            return x.mean(ax, keepdim=kd)
        if f == "sum":
            return x.sum(ax, keepdim=kd)
        if f == "amax":
            return x.amax(ax, keepdim=kd)
        if f == "amin":
            return x.amin(ax, keepdim=kd)
        if f == "std":
            return x.std(ax, correction=self.correction, keepdim=kd)
        if f == "var":
            return x.var(ax, correction=self.correction, keepdim=kd)
        if f == "logsumexp":
            return torch.logsumexp(x, ax, keepdim=kd)
        if f == "norm":
            return torch.linalg.vector_norm(x, dim=ax, keepdim=kd)
        return x.prod(ax[0], keepdim=kd)


class Scalar(nn.Module):
    def __init__(self, fn, value):
        super().__init__()
        self.fn, self.value = fn, value

    def forward(self, x):
        f, v = self.fn, self.value
        if f == "add":
            return x + v
        if f == "sub":
            return x - v
        if f == "rsub":
            return v - x
        if f == "mul":
            return x * v
        if f == "div":
            return x / v
        if f == "rdiv":
            return v / x
        if f == "pow":
            return x**v
        if f == "rpow":
            return v**x
        if f == "floordiv":
            return torch.div(x, v, rounding_mode="floor")
        if f == "max":
            return torch.clamp(x, min=v)
        if f == "min":
            return torch.clamp(x, max=v)
        if f == "fmod":
            return torch.fmod(x, v)
        if f in ("gt", "lt", "ge", "le", "eq", "ne"):
            return getattr(torch, f)(x, v)
        return torch.remainder(x, v)


# ----------------------------------------------------------------------------- layers


@op("conv_t", ANY, space={"out": [16, 32], "k": [2, 3, 4], "s": [2, 1], "pad": [0, 1], "out_pad": [0, 1],
                          "groups": [1], "dilation": [1], "bias": [True, False]},
    tags=P, source="torch.nn.ConvTranspose1d/2d/3d")  # fmt: skip
def _conv_t(dims, kind, out, k, s, pad, out_pad, groups, dilation, bias):
    """Transposed convolution (learned upsampling); every size may be one value per axis."""
    n = nd_of(kind, dims)
    cls = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)[n - 1]
    return cls(
        dims[0],
        out,
        tup(k, n),
        tup(s, n),
        tup(pad, n),
        tup(out_pad, n),
        int(groups),
        bool(bias),
        tup(dilation, n),
    )


@op("batch_norm", ANY, space={"eps": [1e-5, 1e-3], "momentum": [0.1], "affine": [True, False], "track": [True, False]},
    tags=P, source="torch.nn.BatchNorm1d/2d/3d")  # fmt: skip
def _batch_norm(dims, kind, eps, momentum, affine, track):
    """Batch normalisation over the channel axis (axis 1), with torch's eps / momentum / affine."""
    n = 1 if kind in ("vec", "seq") or (kind == "nd" and len(dims) <= 2) else nd_of(kind, dims)
    cls = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)[n - 1]
    return cls(
        dims[0],
        float(eps),
        None if momentum is None else float(momentum),
        bool(affine),
        bool(track),
    )


@op("layer_norm", ANY, space={"axes": [1, 2], "eps": [1e-5, 1e-6], "affine": [True, False], "bias": [True, False]},
    tags=P, source="torch.nn.LayerNorm")  # fmt: skip
def _layer_norm(dims, kind, axes, eps, affine, bias):
    """Layer normalisation over the last `axes` axes (torch's normalized_shape)."""
    return nn.LayerNorm(tuple(dims[-int(axes) :]), float(eps), bool(affine), bias=bool(bias))


@op("group_norm", ANY, space={"groups": [32, 8, 1], "eps": [1e-5, 1e-6], "affine": [True, False]},
    tags=P, source="torch.nn.GroupNorm")  # fmt: skip
def _group_norm(dims, kind, groups, eps, affine):
    """Group normalisation over channels (axis 1); groups=1 is layer norm over (C, ...), groups=C instance norm."""
    return nn.GroupNorm(int(groups), dims[0], float(eps), bool(affine))


@op("instance_norm", ANY, space={"eps": [1e-5], "affine": [False, True], "track": [False, True]},
    tags=P, source="torch.nn.InstanceNorm1d/2d/3d")  # fmt: skip
def _instance_norm(dims, kind, eps, affine, track):
    """Instance normalisation: each channel of each sample over its spatial axes."""
    n = nd_of(kind, dims)
    cls = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)[n - 1]
    return cls(dims[0], float(eps), affine=bool(affine), track_running_stats=bool(track))


@op("rms_norm", ANY, space={"axes": [1], "eps": [1e-6, 1e-5, None], "affine": [True, False]},
    tags=P, source="torch.nn.RMSNorm")  # fmt: skip
def _rms_norm(dims, kind, axes, eps, affine):
    """Root-mean-square normalisation over the last `axes` axes (eps None: torch's default)."""
    return nn.RMSNorm(tuple(dims[-int(axes) :]), None if eps is None else float(eps), bool(affine))


@op("local_response_norm", ANY, space={"size": [5], "alpha": [1e-4], "beta": [0.75], "k": [1.0, 2.0]},
    tags=P, source="torch.nn.LocalResponseNorm")  # fmt: skip
def _lrn(dims, kind, size, alpha, beta, k):
    """Local response normalisation across channels (AlexNet, GoogLeNet)."""
    return nn.LocalResponseNorm(int(size), float(alpha), float(beta), float(k))


ACT = {
    "relu": lambda a, b: nn.ReLU(), "relu6": lambda a, b: nn.ReLU6(),
    "leaky_relu": lambda a, b: nn.LeakyReLU(0.01 if a is None else a), "elu": lambda a, b: nn.ELU(1.0 if a is None else a),
    "selu": lambda a, b: nn.SELU(), "celu": lambda a, b: nn.CELU(1.0 if a is None else a),
    "gelu": lambda a, b: nn.GELU(), "gelu_tanh": lambda a, b: nn.GELU(approximate="tanh"),
    "silu": lambda a, b: nn.SiLU(), "mish": lambda a, b: nn.Mish(), "hardswish": lambda a, b: nn.Hardswish(),
    "hardsigmoid": lambda a, b: nn.Hardsigmoid(),
    "hardtanh": lambda a, b: nn.Hardtanh(-1.0 if a is None else a, 1.0 if b is None else b),
    "sigmoid": lambda a, b: nn.Sigmoid(), "tanh": lambda a, b: nn.Tanh(),
    "softplus": lambda a, b: nn.Softplus(1.0 if a is None else a, 20.0 if b is None else b),
    "softsign": lambda a, b: nn.Softsign(), "tanhshrink": lambda a, b: nn.Tanhshrink(), "logsigmoid": lambda a, b: nn.LogSigmoid(),
    "hardshrink": lambda a, b: nn.Hardshrink(0.5 if a is None else a), "softshrink": lambda a, b: nn.Softshrink(0.5 if a is None else a),
    "threshold": lambda a, b: nn.Threshold(a, b), "softmax": lambda a, b: nn.Softmax(int(a)),
    "log_softmax": lambda a, b: nn.LogSoftmax(int(a)), "glu": lambda a, b: nn.GLU(-1 if a is None else int(a)),
    "identity": lambda a, b: nn.Identity(),
}  # fmt: skip


@op("activation", ANY, space={"fn": list(ACT), "a": [None], "b": [None]}, tags=P,
    source="torch.nn activations (ReLU ... GLU)")  # fmt: skip
def _activation(dims, kind, fn, a, b):
    """Any activation of torch.nn by name; `a` / `b` are its arguments (negative_slope, alpha,
    min/max, beta/threshold, lambd, dim for softmax / log_softmax / glu)."""
    if fn not in ACT:
        raise ValueError(f"unknown activation {fn!r}; one of {sorted(ACT)}")
    return ACT[fn](a, b)


@op("prelu", ANY, space={"n": [1], "init": [0.25]}, tags=P, source="torch.nn.PReLU")
def _prelu(dims, kind, n, init):
    """PReLU: a learned negative slope, one (n=1) or one per channel (n=C)."""
    return nn.PReLU(int(n), float(init))


@op("adaptive_pool", ANY, space={"mode": ["avg", "max"], "size": [1, 7]}, tags=P,
    source="torch.nn.AdaptiveAvgPool / AdaptiveMaxPool 1d/2d/3d")  # fmt: skip
def _adaptive_pool(dims, kind, mode, size):
    """Pool to a fixed output size (one value, or one per axis; None keeps that axis)."""
    n = nd_of(kind, dims)
    size = tuple(size) if isinstance(size, list | tuple) else size
    cls = {"avg": (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d),
           "max": (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)}[mode][n - 1]  # fmt: skip
    return cls(size)


@op("pooling", ANY, space={"mode": ["max", "avg", "lp"], "k": [2, 3], "s": [None], "pad": [0], "dilation": [1],
                        "ceil": [False, True], "count_pad": [True, False], "divisor": [None], "p": [2.0]},
    tags=P, source="torch.nn.MaxPool / AvgPool / LPPool 1d/2d/3d")  # fmt: skip
def _pool(dims, kind, mode, k, s, pad, dilation, ceil, count_pad, divisor, p):
    """Pooling with every torch.nn option; sizes may be one value per axis (s None: = k)."""
    n = nd_of(kind, dims)
    kk = tup(k, n)
    ss = kk if s is None else tup(s, n)
    if mode == "max":
        return (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)[n - 1](
            kk, ss, tup(pad, n), tup(dilation, n), ceil_mode=bool(ceil)
        )
    if mode == "avg":
        return (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)[n - 1](kk, ss, tup(pad, n), bool(ceil), bool(count_pad),
                                                                 **({} if n == 1 else {"divisor_override": divisor}))  # fmt: skip
    if mode == "lp":
        if n == 3:
            return nn.LPPool3d(float(p), kk, ss, bool(ceil))
        return (nn.LPPool1d, nn.LPPool2d)[n - 1](
            float(p), kk if n > 1 else kk[0], ss if n > 1 else ss[0], bool(ceil)
        )
    raise ValueError(f"pool mode {mode!r}")


@op("upsample", ANY, space={"scale": [2], "size": [None], "mode": ["nearest", "bilinear", "bicubic", "linear",
                            "trilinear", "nearest-exact", "area"], "align_corners": [None]},
    tags=P, source="torch.nn.Upsample / torch.nn.functional.interpolate")  # fmt: skip
def _upsample(dims, kind, scale, size, mode, align_corners):
    """Resize the spatial axes: by a factor (`scale`) or to a `size`."""
    sz = tuple(size) if isinstance(size, list | tuple) else size
    sc = (
        tuple(float(x) for x in scale)
        if isinstance(scale, list | tuple)
        else (None if scale is None or sz else float(scale))
    )
    return nn.Upsample(size=sz, scale_factor=sc, mode=mode, align_corners=align_corners)


@op("pixel_shuffle", ANY, space={"r": [2], "inverse": [False, True]}, tags=P,
    source="torch.nn.PixelShuffle / PixelUnshuffle")  # fmt: skip
def _pixel_shuffle(dims, kind, r, inverse):
    """Sub-pixel rearrangement: (C r^2, H, W) <-> (C, H r, W r)."""
    return nn.PixelUnshuffle(int(r)) if inverse else nn.PixelShuffle(int(r))


@op(
    "channel_shuffle",
    ANY,
    space={"groups": [2, 4]},
    tags=P,
    source="torch.nn.ChannelShuffle (ShuffleNet)",
)
def _channel_shuffle(dims, kind, groups):
    """Interleave the channels of `groups` groups (ShuffleNet)."""
    return ChannelShuffle(int(groups))


@op("pad", ANY, space={"pads": [[1, 1]], "mode": ["constant", "reflect", "replicate", "circular"], "value": [0.0]},
    tags=P, source="torch.nn.functional.pad (ZeroPad / ReflectionPad / ReplicationPad / CircularPad)")  # fmt: skip
def _pad(dims, kind, pads, mode, value):
    """Pad the last axes: [left, right, (top, bottom, (front, back))] as torch.nn.functional.pad."""
    kw = {"pad": tuple(int(p) for p in pads), "mode": mode}
    if mode == "constant":
        kw["value"] = float(value)
    return Fn("pad", F.pad, **kw)


class Embedding(nn.Embedding):
    def forward(self, x):
        return super().forward(x.long())


@op(
    "embedding",
    ANY,
    "nd",
    space={"num": [1000], "dim": [64], "padding_idx": [None]},
    tags=P,
    source="torch.nn.Embedding",
)
def _embedding(dims, kind, num, dim, padding_idx):
    """A lookup table: integer ids -> vectors (a new last axis of size `dim`)."""
    return Embedding(int(num), int(dim), padding_idx)


class Recurrent(nn.Module):
    """torch.nn.LSTM / GRU / RNN returning only the output sequence (what `out, _ = rnn(x)` keeps)."""

    def __init__(self, cell, *a, **kw):
        super().__init__()
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[cell](*a, **kw)

    def forward(self, x):
        return self.rnn(x)[0]


@op("recurrent", ANY, space={"cell": ["lstm", "gru", "rnn"], "hidden": [64], "layers": [1], "bias": [True],
                             "batch_first": [True, False], "bidirectional": [False, True], "nonlinearity": ["tanh"],
                             "proj": [0]},
    tags=P, source="torch.nn.LSTM / GRU / RNN")  # fmt: skip
def _recurrent(
    dims, kind, cell, hidden, layers, bias, batch_first, bidirectional, nonlinearity, proj
):
    """A recurrent layer exactly as torch.nn builds it; returns the output sequence only."""
    kw = {
        "num_layers": int(layers),
        "bias": bool(bias),
        "batch_first": bool(batch_first),
        "bidirectional": bool(bidirectional),
    }
    if cell == "rnn":
        kw["nonlinearity"] = nonlinearity
    if cell == "lstm" and proj:
        kw["proj_size"] = int(proj)
    return Recurrent(cell, dims[-1], int(hidden), **kw)


# ----------------------------------------------------------------------------- layout


def _norm_axis(a, rank):
    a = int(a)
    return a + rank if a < 0 else a


@op("reshape", ANY, "nd", space={"shape": [[-1]]}, tags=P, source="torch.Tensor.reshape")
def _reshape(dims, kind, shape):
    """Reshape to (batch, *shape). With every size given, the leading axis is -1: the batch axis may
    then merge with others or split (window partition (B, H, W, C) -> (B nW, w, w, C) and back);
    with a -1 inside `shape`, the batch axis is kept as it is."""
    shape = tuple(int(s) for s in shape)
    return Reshape(shape, keep_batch=-1 in shape)


@op("expand", ANY, "nd", space={"shape": [[-1]]}, tags=P, source="torch.Tensor.expand")
def _expand(dims, kind, shape):
    """Broadcast size-1 axes to `shape` (the batch axis is kept; -1 keeps an axis)."""
    return Expand(tuple(int(s) for s in shape))


@op(
    "fill",
    ANY,
    "nd",
    space={"shape": [[1]], "value": [0.0]},
    tags=P,
    source="torch.zeros / ones / full",
)
def _fill(dims, kind, shape, value):
    """A constant tensor (batch, *shape) filled with `value`; its input only gives the batch size."""
    return Fill(tuple(int(s) for s in shape), float(value))


class Fill(nn.Module):
    def __init__(self, shape, value):
        super().__init__()
        self.shape, self.value = tuple(shape), value

    def forward(self, x):
        dt = x.dtype if x.is_floating_point() else torch.float32
        return torch.full((x.shape[0], *self.shape), self.value, dtype=dt, device=x.device)


@op(
    "repeat_interleave",
    ANY,
    "nd",
    space={"repeats": [2], "axis": [1]},
    tags=P,
    source="torch.repeat_interleave",
)
def _repeat_interleave(dims, kind, repeats, axis):
    """Repeat every element of one axis `repeats` times in place (1 1 2 2 3 3); on axis 0 every
    sample is repeated (a global query shared by the windows of an image, GC ViT)."""
    return RepeatInterleave(
        int(repeats), _norm_axis(axis, len(dims) + 1)
    )  # axis 0: the batch grows


class RepeatInterleave(nn.Module):
    def __init__(self, repeats, axis):
        super().__init__()
        self.repeats, self.axis = repeats, axis

    def forward(self, x):
        return torch.repeat_interleave(x, self.repeats, self.axis)


@op(
    "index_axis",
    ANY,
    "nd",
    space={"axis": [1], "index": [[0]]},
    tags=P,
    source="torch.index_select (x[:, [i, j, ...]])",
)
def _index_axis(dims, kind, axis, index):
    """Take the given positions of one axis, in that order (a fixed permutation or subset of tokens)."""
    a = _norm_axis(axis, len(dims) + 1)
    if a < 1:
        raise ValueError("index_axis cannot pick from the batch axis")
    idx = torch.tensor([int(i) for i in index], dtype=torch.long)
    return IndexAxis(a, idx)


class IndexAxis(nn.Module):
    def __init__(self, axis, index):
        super().__init__()
        self.axis = axis
        self.register_buffer("index", index, persistent=False)

    def forward(self, x):
        return x.index_select(self.axis, self.index.to(x.device))


@op("permute", ANY, "nd", space={"order": [[0, 2, 3, 1]]}, tags=P, source="torch.permute")
def _permute(dims, kind, order):
    """Reorder the axes (`order` includes the batch axis 0, which must stay first)."""
    order = tuple(_norm_axis(a, len(dims) + 1) for a in order)
    if order[0] != 0 or sorted(order) != list(range(len(dims) + 1)):
        raise ValueError(
            f"permute order {list(order)} must keep the batch axis first and name every axis once"
        )
    return Fn("permute", torch.permute, dims=order)


@op("transpose", ANY, "nd", space={"a": [1], "b": [2]}, tags=P, source="torch.transpose")
def _transpose(dims, kind, a, b):
    """Swap two axes (counted with the batch axis 0)."""
    r = len(dims) + 1
    a, b = _norm_axis(a, r), _norm_axis(b, r)
    if 0 in (a, b):
        raise ValueError("transpose cannot move the batch axis")
    return Fn("transpose", torch.transpose, dim0=a, dim1=b)


@op("flatten_axes", ANY, "nd", space={"start": [1], "end": [-1]}, tags=P, source="torch.flatten")
def _flatten_axes(dims, kind, start, end):
    """Merge the axes start..end (batch axis 0 excluded) into one."""
    r = len(dims) + 1
    s, e = _norm_axis(start, r), _norm_axis(end, r)
    if s < 1:
        raise ValueError("flatten_axes cannot merge the batch axis")
    return Fn("flatten", torch.flatten, start_dim=s, end_dim=e)


@op(
    "unflatten",
    ANY,
    "nd",
    space={"axis": [1], "sizes": [[-1, 2]]},
    tags=P,
    source="torch.unflatten",
)
def _unflatten(dims, kind, axis, sizes):
    """Split one axis into several of the given sizes."""
    a = _norm_axis(axis, len(dims) + 1)
    if a < 1:
        raise ValueError("unflatten cannot split the batch axis")
    return Fn("unflatten", torch.unflatten, dim=a, sizes=tuple(int(s) for s in sizes))


@op("unsqueeze", ANY, "nd", space={"axis": [1]}, tags=P, source="torch.unsqueeze")
def _unsqueeze(dims, kind, axis):
    """Add an axis of size 1."""
    a = _norm_axis(axis, len(dims) + 2)
    if a < 1:
        raise ValueError("unsqueeze cannot insert before the batch axis")
    return Fn("unsqueeze", torch.unsqueeze, dim=a)


@op("squeeze", ANY, "nd", space={"axis": [1]}, tags=P, source="torch.squeeze")
def _squeeze(dims, kind, axis):
    """Remove an axis of size 1."""
    a = _norm_axis(axis, len(dims) + 1)
    if a < 1 or dims[a - 1] != 1:
        raise ValueError(f"squeeze axis {axis}: not an axis of size 1 in {tuple(dims)}")
    return Fn("squeeze", torch.squeeze, dim=a)


@op("slice", ANY, "nd", space={"axis": [1], "start": [0], "stop": [None], "step": [1]}, tags=P,
    source="Tensor indexing x[..., start:stop:step]")  # fmt: skip
def _slice(dims, kind, axis, start, stop, step):
    """Keep start:stop:step of one axis."""
    a = _norm_axis(axis, len(dims) + 1)
    if a < 1:
        raise ValueError("slice cannot cut the batch axis")
    return Slice(a, start, stop, step)


@op("select", ANY, "nd", space={"axis": [1], "index": [0]}, tags=P, source="torch.select (x[:, i])")
def _select(dims, kind, axis, index):
    """Take one index of an axis and drop that axis (x[:, 0] = the class token)."""
    a = _norm_axis(axis, len(dims) + 1)
    if a < 1:
        raise ValueError("select cannot pick from the batch axis")
    return Fn("select", torch.select, dim=a, index=int(index))


@op(
    "chunk_part",
    ANY,
    "nd",
    space={"chunks": [2], "axis": [1], "i": [0]},
    tags=P,
    source="torch.chunk(...)[i]",
)
def _chunk_part(dims, kind, chunks, axis, i):
    """Piece i of torch.chunk(x, chunks, axis)."""
    return ChunkPart(int(chunks), _norm_axis(axis, len(dims) + 1), int(i))


@op(
    "split_part",
    ANY,
    "nd",
    space={"sizes": [[1]], "axis": [1], "i": [0]},
    tags=P,
    source="torch.split(...)[i]",
)
def _split_part(dims, kind, sizes, axis, i):
    """Piece i of torch.split(x, sizes, axis) (`sizes`: one int, or a list)."""
    sz = [int(s) for s in sizes] if isinstance(sizes, list | tuple) else int(sizes)
    return SplitPart(sz, _norm_axis(axis, len(dims) + 1), int(i))


@op("reduce", ANY, "nd", space={"fn": ["mean", "sum", "amax", "amin", "prod", "std", "var", "logsumexp", "norm"],
                                "axes": [[1]], "keepdim": [False, True], "correction": [1]},
    tags=P, source="torch.mean / sum / amax / amin / prod / std / var / logsumexp / linalg.vector_norm")  # fmt: skip
def _reduce(dims, kind, fn, axes, keepdim, correction):
    """Reduce over the given axes (batch axis excluded)."""
    r = len(dims) + 1
    ax = tuple(_norm_axis(a, r) for a in (axes if isinstance(axes, list | tuple) else [axes]))
    if 0 in ax:
        raise ValueError("reduce cannot reduce the batch axis")
    if fn not in ("mean", "sum", "amax", "amin", "prod", "std", "var", "logsumexp", "norm"):
        raise ValueError(f"reduce fn {fn!r}")
    if fn == "prod" and len(ax) != 1:
        raise ValueError("prod reduces one axis")
    return Reduce(fn, ax, bool(keepdim), int(correction))


@op("as_kind", ANY, space={"to": ["seq", "map", "vec", "ser", "vol", "nd"]}, tags=P, source="DLNet")
def _as_kind(dims, kind, to):
    """Relabel a tensor (nothing is computed): e.g. an "nd" (T, D) as "seq" for the zoo's blocks."""
    return nn.Identity()


# ----------------------------------------------------------------------------- elementwise

SCALAR = {
    "add": lambda x, v: x + v, "sub": lambda x, v: x - v, "rsub": lambda x, v: v - x, "mul": lambda x, v: x * v,
    "div": lambda x, v: x / v, "rdiv": lambda x, v: v / x, "pow": lambda x, v: x**v, "rpow": lambda x, v: v**x,
    "floordiv": lambda x, v: torch.div(x, v, rounding_mode="floor"), "max": lambda x, v: torch.clamp(x, min=v),
    "min": lambda x, v: torch.clamp(x, max=v), "fmod": torch.fmod, "remainder": torch.remainder,
    "gt": torch.gt, "lt": torch.lt, "ge": torch.ge, "le": torch.le, "eq": torch.eq, "ne": torch.ne,
}  # fmt: skip


@op(
    "scalar",
    ANY,
    space={"fn": list(SCALAR), "value": [1.0]},
    tags=P,
    source="Tensor arithmetic with a constant",
)
def _scalar(dims, kind, fn, value):
    """x (op) constant: add, sub, rsub (c - x), mul, div, rdiv (c / x), pow, rpow (c ** x), floordiv,
    max / min (clamp), fmod, remainder."""
    if fn not in SCALAR:
        raise ValueError(f"scalar fn {fn!r}")
    return Scalar(fn, value)


UNARY = {
    "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt, "rsqrt": torch.rsqrt, "abs": torch.abs, "neg": torch.neg,
    "sin": torch.sin, "cos": torch.cos, "reciprocal": torch.reciprocal, "square": torch.square, "sign": torch.sign,
    "floor": torch.floor, "ceil": torch.ceil, "round": torch.round, "erf": torch.erf, "log1p": torch.log1p,
    "expm1": torch.expm1, "sigmoid": torch.sigmoid, "tanh": torch.tanh, "atan": torch.atan, "log2": torch.log2,
    "real": torch.real, "imag": torch.imag, "angle": torch.angle, "conj": torch.conj_physical,
    "logical_not": torch.logical_not,
}  # fmt: skip


@op("unary", ANY, space={"fn": list(UNARY)}, tags=P, source="torch elementwise functions")
def _unary(dims, kind, fn):
    """An elementwise torch function without arguments (exp, log, sqrt, rsqrt, abs, erf, ...)."""
    if fn not in UNARY:
        raise ValueError(f"unary fn {fn!r}")
    return Fn(fn, UNARY[fn])


@op("clamp", ANY, space={"min": [None, 0.0], "max": [None, 6.0]}, tags=P, source="torch.clamp")
def _clamp(dims, kind, min, max):
    """Clip to [min, max] (either may be None)."""
    return Fn("clamp", torch.clamp, min=min, max=max)


@op(
    "normalize",
    ANY,
    space={"p": [2.0], "axis": [1], "eps": [1e-12]},
    tags=P,
    source="torch.nn.functional.normalize",
)
def _normalize(dims, kind, p, axis, eps):
    """Divide by the p-norm along one axis."""
    return Fn(
        "normalize", F.normalize, p=float(p), dim=_norm_axis(axis, len(dims) + 1), eps=float(eps)
    )


@op(
    "roll",
    ANY,
    space={"shifts": [[1]], "axes": [[2]]},
    tags=P,
    source="torch.roll (Swin's shifted windows)",
)
def _roll(dims, kind, shifts, axes):
    """Cyclic shift along axes."""
    return Fn(
        "roll",
        torch.roll,
        shifts=tuple(int(s) for s in shifts),
        dims=tuple(_norm_axis(a, len(dims) + 1) for a in axes),
    )


@op("flip", ANY, space={"axes": [[2]]}, tags=P, source="torch.flip")
def _flip(dims, kind, axes):
    """Reverse the order along axes."""
    return Fn("flip", torch.flip, dims=tuple(_norm_axis(a, len(dims) + 1) for a in axes))


@op("cumsum", ANY, space={"axis": [1]}, tags=P, source="torch.cumsum")
def _cumsum(dims, kind, axis):
    """Cumulative sum along an axis."""
    return Fn("cumsum", torch.cumsum, dim=_norm_axis(axis, len(dims) + 1))


@op("fft", ANY, space={"fn": ["rfft", "irfft", "fft", "ifft", "rfft2", "irfft2", "fft2", "ifft2"], "axes": [[-1]],
                       "norm": ["backward", "ortho", "forward"], "size": [None]},
    tags=P, source="torch.fft")  # fmt: skip
def _fft(dims, kind, fn, axes, norm, size):
    """A torch.fft transform over the given axes (complex tensors stay complex)."""
    if fn not in ("rfft", "irfft", "fft", "ifft", "rfft2", "irfft2", "fft2", "ifft2"):
        raise ValueError(f"fft fn {fn!r}")
    ax = tuple(_norm_axis(a, len(dims) + 1) for a in axes)
    return FFT(
        fn, ax, norm, None if size is None else (tuple(size) if fn.endswith("2") else int(size))
    )


class FFT(nn.Module):
    """A torch.fft transform with fixed axes / norm / size."""

    def __init__(self, fn, axes, norm, size=None):
        super().__init__()
        self.fn, self.axes, self.norm, self.size = fn, tuple(axes), norm, size

    def forward(self, x):
        f = getattr(torch.fft, self.fn)
        if self.fn.endswith("2"):
            return f(x, s=self.size, dim=self.axes, norm=self.norm)
        return f(x, n=self.size, dim=self.axes[0], norm=self.norm)


# ----------------------------------------------------------------------------- learned tensors


class Param(nn.Module):
    """A learned tensor (a class token, a position embedding, a layer-scale vector): a graph node
    with no input; merges broadcast it against the batch."""

    def __init__(self, shape, init):
        super().__init__()
        t = torch.zeros(shape)
        if init == "normal":
            nn.init.trunc_normal_(t, std=0.02)
        elif init == "ones":
            t.fill_(1.0)
        elif isinstance(init, int | float):
            t.fill_(float(init))
        self.weight = nn.Parameter(t)

    def forward(self, x=None):
        return self.weight

    def extra_repr(self):
        return f"shape={tuple(self.weight.shape)}"


class Buffer(nn.Module):
    """A constant tensor of the network (not learned): saved with the weights, copied by verify."""

    def __init__(self, shape, dtype="float32"):
        super().__init__()
        self.register_buffer("value", torch.zeros(shape, dtype=getattr(torch, dtype)))

    def forward(self, x=None):
        return self.value


def param_module(shape, init="normal", learned=True, dtype="float32"):
    shape = tuple(int(s) for s in shape)
    return Param(shape, init) if learned else Buffer(shape, dtype)


# ----------------------------------------------------------------------------- merges (several inputs)


def _sdpa(q, k, v, causal=False, scale=None, mask=None):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=bool(causal), scale=scale
    )


class MHA(nn.Module):
    """torch.nn.MultiheadAttention returning the attention output only (query, key, value)."""

    def __init__(
        self,
        dim,
        heads,
        bias=True,
        kdim=None,
        vdim=None,
        batch_first=True,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, bias=bias, kdim=kdim, vdim=vdim, batch_first=batch_first,
                                          add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)  # fmt: skip

    def forward(self, q, k, v):
        return self.attn(q, k, v, need_weights=False)[0]


MERGE_FNS = {
    "add": lambda xs, a: _fold(xs, torch.add), "mul": lambda xs, a: _fold(xs, torch.mul),
    "sub": lambda xs, a: xs[0] - xs[1], "div": lambda xs, a: xs[0] / xs[1],
    "maximum": lambda xs, a: _fold(xs, torch.maximum), "minimum": lambda xs, a: _fold(xs, torch.minimum),
    "matmul": lambda xs, a: torch.matmul(xs[0], xs[1]),
    "concat": lambda xs, a: torch.cat(_batch(xs), a.get("axis", 1)),
    "stack": lambda xs, a: torch.stack(_batch(xs), a.get("axis", 1)),
    "attention": lambda xs, a: _sdpa(*xs[:3], a.get("causal", False), a.get("scale"), xs[3] if len(xs) > 3 else None),
    "where": lambda xs, a: torch.where(xs[0].bool(), xs[1], xs[2]),
    "pow2": lambda xs, a: torch.pow(xs[0], xs[1]),
    "complex": lambda xs, a: torch.complex(xs[0], xs[1]),
    "einsum": lambda xs, a: torch.einsum(a["equation"], *xs),
    "linear_fn": lambda xs, a: F.linear(xs[0], xs[1], xs[2] if len(xs) > 2 else None),
    "compare": lambda xs, a: getattr(torch, a["fn"])(xs[0], xs[1]),
    "conv_fn": lambda xs, a: _conv_fn(xs, a),
}  # fmt: skip


def _conv_fn(xs, a):
    """F.conv1d/2d/3d with weights that are inputs of the node (computed from parameters)."""
    f = (F.conv1d, F.conv2d, F.conv3d)[xs[0].dim() - 3]
    pad = a.get("padding", 0)
    return f(xs[0], xs[1], xs[2] if len(xs) > 2 else None, a.get("stride", 1), pad if isinstance(pad, str) else tuple(pad)
             if isinstance(pad, list) else pad, a.get("dilation", 1), a.get("groups", 1))  # fmt: skip


MERGE_ARITY = {
    "sub": 2,
    "div": 2,
    "matmul": 2,
    "attention": (3, 4),
    "where": 3,
    "pow2": 2,
    "complex": 2,
    "linear_fn": (2, 3),
    "conv_fn": (2, 3),
}


def _fold(xs, f):
    y = xs[0]
    for t in xs[1:]:
        y = f(y, t)
    return y


def _batch(xs):
    """Concatenating a learned tensor (batch 1) with a batch: expand it (the class token)."""
    b = max(x.shape[0] for x in xs)
    return [x.expand(b, *x.shape[1:]) if x.shape[0] == 1 and b > 1 else x for x in xs]


class Merge(nn.Module):
    """A node with several inputs: broadcasting arithmetic, concat / stack on an axis, matmul,
    scaled dot-product attention, multi-head attention (`mha`, with its weights)."""

    multi = True

    def __init__(self, how, args=None, mha=None):
        super().__init__()
        self.how, self.args = how, dict(args or {})
        self.mha = mha

    def forward(self, *xs):
        if self.how == "mha":
            return self.mha(*xs[:3]) if len(xs) >= 3 else self.mha(xs[0], xs[0], xs[0])
        return MERGE_FNS[self.how](list(xs), self.args)

    def extra_repr(self):
        return f"{self.how}, {self.args}" if self.args else self.how


def merge_module(how, args, in_dims):
    """(Merge module) for a graph node with several inputs; in_dims: the inputs' shapes (no batch)."""
    if how == "mha":
        a = dict(args)
        q = in_dims[0]
        mha = MHA(int(a.pop("dim", q[-1])), int(a.pop("heads")), bool(a.pop("bias", True)), a.pop("kdim", None),
                  a.pop("vdim", None), bool(a.pop("batch_first", True)), bool(a.pop("add_bias_kv", False)),
                  bool(a.pop("add_zero_attn", False)))  # fmt: skip
        if a:
            raise ValueError(f"mha: unknown argument(s) {sorted(a)}")
        return Merge("mha", {}, mha)
    if how not in MERGE_FNS:
        raise ValueError(f"unknown merge {how!r}")
    n = MERGE_ARITY.get(how)
    if n is not None and len(in_dims) not in ((n,) if isinstance(n, int) else n):
        raise ValueError(f"{how} takes {n} inputs, got {len(in_dims)}")
    return Merge(how, args)


MERGES = (*MERGE_FNS, "mha")


def meta_shape(module, in_dims, dtypes=None):
    """Output shape (no batch) of a module / merge applied to inputs of these shapes, on meta tensors."""
    xs = [
        torch.zeros(2, *d, device="meta", dtype=(dtypes or {}).get(i, torch.float32))
        for i, d in enumerate(in_dims)
    ]
    with torch.no_grad():
        y = module(*xs)
    return tuple(int(s) for s in y.shape[1:]), y.dtype


def _check_math():
    """Guard for the docs: these names are what torch calls them."""
    assert math and F


# ----------------------------------------------------------------------------- static sub-graphs


STATIC_UNARY = {"exp": torch.exp, "sigmoid": torch.sigmoid, "log": torch.log, "log2": torch.log2, "sign": torch.sign,
                "sin": torch.sin, "cos": torch.cos,
                "abs": torch.abs, "sqrt": torch.sqrt, "neg": torch.neg, "tanh": torch.tanh, "relu": torch.relu,
                "contiguous": lambda t: t.contiguous(), "float": lambda t: t.float(), "long": lambda t: t.long()}  # fmt: skip
STATIC_BINARY = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div, "pow": torch.pow,
                 "rsub": lambda a, b: b - a, "rdiv": lambda a, b: b / a, "matmul": torch.matmul}  # fmt: skip


def _idx(spec, vals=()):
    """A JSON index ([1, null, [0, 4, 2], "...", {"t": 1}]) -> a Python index for x[...];
    {"t": i} is the i-th input of the step (an integer tensor)."""
    out = []
    for s in spec:
        if isinstance(s, dict):
            out.append(vals[int(s["t"])])
        elif s is None:
            out.append(None)
        elif s == "...":
            out.append(Ellipsis)
        elif isinstance(s, list):
            out.append(slice(*s))
        else:
            out.append(int(s))
    return tuple(out)


class StaticExpr(nn.Module):
    """A computation on the network's own tensors only (no input): e.g. a relative position bias
    gathered from a learned table with a constant index, or Swin v2's bias MLP. `tensors` are the
    learned parameters and constant buffers it reads (copied from the original model by verify);
    `program` is a list of steps, each reading earlier values by number (the tensors first):

        {"op": "index", "in": [0, 1]}                     a[b]   (b an integer tensor)
        {"op": "item", "in": [2], "index": [[0, 4], 1]}   a[0:4, 1]
        {"op": "reshape", "in": [2], "shape": [49, 49, -1]}   permute / transpose / unsqueeze /
        squeeze / flatten / expand: their torch arguments, with every axis counted
        {"op": "mul", "in": [3], "value": 16.0}           add / sub / mul / div / pow, with a constant or two values
        {"op": "sigmoid", "in": [4]}                      exp, sigmoid, log, sign, abs, sqrt, ...
        {"op": "clamp", "in": [4], "min": null, "max": 4.6}   softmax {"dim"}
        {"op": "linear", "in": [4], "layer": 0}           the layer `layers[0]` (linear / activation)

    The value of the last step is the node's output."""

    def __init__(self, tensors, program, layers=()):
        super().__init__()
        self.program = [dict(s) for s in program]
        self.kinds = []
        for i, t in enumerate(tensors):
            shape, dtype = (
                tuple(int(s) for s in t["shape"]),
                getattr(torch, t.get("dtype", "float32")),
            )
            if "value" in t:  # a constant written in the card (not a weight): not saved, not copied
                self.register_buffer(
                    f"t{i}",
                    torch.as_tensor(t["value"], dtype=dtype).reshape(shape).clone(),
                    persistent=False,
                )
            elif t.get("learned", True):
                self.register_parameter(f"t{i}", nn.Parameter(torch.zeros(shape, dtype=dtype)))
            else:
                self.register_buffer(f"t{i}", torch.zeros(shape, dtype=dtype))
        self.n = len(tensors)
        self.layers = nn.ModuleList()
        for spec in layers:
            if spec["op"] == "linear":
                self.layers.append(
                    nn.Linear(int(spec["in"]), int(spec["out"]), bias=bool(spec.get("bias", True)))
                )
            elif spec["op"] == "activation":
                self.layers.append(ACT[spec["fn"]](spec.get("a"), spec.get("b")))
            else:
                raise ValueError(f"static layer {spec['op']!r}: linear or activation")

    def forward(self, *inputs):
        # inputs: the param / static nodes it reads (the network's weights, shared with other nodes);
        # then its own constants; then one value per step
        vals = [*inputs, *(getattr(self, f"t{i}") for i in range(self.n))]
        for s in self.program:
            a = [vals[i] for i in s["in"]]
            o = s["op"]
            if o == "index":
                y = a[0][a[1]]
            elif o == "item":
                y = a[0][_idx(s["index"], a)]
            elif o == "reshape":
                y = a[0].reshape(*s["shape"])
            elif o == "permute":
                y = a[0].permute(*s["order"])
            elif o == "transpose":
                y = a[0].transpose(s["a"], s["b"])
            elif o == "unsqueeze":
                y = a[0].unsqueeze(s["axis"])
            elif o == "squeeze":
                y = a[0].squeeze(s["axis"])
            elif o == "flatten":
                y = a[0].flatten(s.get("start", 0), s.get("end", -1))
            elif o == "expand":
                y = a[0].expand(*s["shape"])
            elif o in STATIC_BINARY:
                y = STATIC_BINARY[o](a[0], a[1] if len(a) > 1 else s["value"])
            elif o in STATIC_UNARY:
                y = STATIC_UNARY[o](a[0])
            elif o == "clamp":
                y = torch.clamp(a[0], s.get("min"), s.get("max"))
            elif o == "softmax":
                y = torch.softmax(a[0], s["dim"])
            elif o == "layer":
                y = self.layers[s["layer"]](a[0])
            elif o == "cat":
                y = torch.cat(a, s.get("dim", 0))
            elif o == "stack":
                y = torch.stack(a, s.get("dim", 0))
            elif o == "reduce":
                y = getattr(a[0], s["fn"])(s["dim"], keepdim=bool(s.get("keepdim", False)))
            elif o == "zeros_like":
                y = torch.zeros_like(a[0])
            elif o == "ones_like":
                y = torch.ones_like(a[0])
            else:
                raise ValueError(f"static step {o!r}")
            vals.append(y)
        return vals[-1]


def static_module(args):
    for s in args["program"]:
        known = {"index", "item", "reshape", "permute", "transpose", "unsqueeze", "squeeze", "flatten", "expand",
                 "clamp", "softmax", "layer", "cat", "stack", "reduce", "zeros_like", "ones_like", *STATIC_BINARY, *STATIC_UNARY}  # fmt: skip
        if s.get("op") not in known:
            raise ValueError(f"static step {s.get('op')!r}; one of {sorted(known)}")
    return StaticExpr(args["tensors"], args["program"], args.get("layers") or ())
