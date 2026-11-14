"""Draft a paper card from the AUTHORS' code (`dlnet paper import`), so nobody copies a figure by hand.

    python -m dlnet paper import --official-root <checkout> --repo URL --commit C \\
        --entry path/file.py:Class --init-args '{"num_classes": 10}' --kind map --dims 3,32,32 \\
        --name "Some Net" --out papers/some_net.yaml

The authors' model is traced with torch.fx (the layers of torch.nn are the leaves), and every node
of the traced graph becomes a node of the card: Conv -> conv, BatchNorm -> norm, Linear -> linear,
activations -> act, x + y -> add, x * y -> mul, torch.cat -> concat, global average pooling -> gap.
A last Linear onto the classes becomes the card's head. Anything else (a custom tensor operation,
attention written with einsum, ...) stops the import with the name of the node: such a network
is either written by hand with a zoo block, or used as it is (a card without `architecture`,
fidelity level A).

The draft is a claim, not a proof: `dlnet paper verify` then compares it with the same code.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from . import paper


class ImportUnsupported(ValueError):
    """The authors' model holds something the importer cannot write as card nodes."""


# ----------------------------------------------------------------------------- torch.nn layers

ACT_MODULES = {
    nn.ReLU: ("relu",), nn.ReLU6: ("relu6",), nn.SELU: ("selu",), nn.SiLU: ("silu",), nn.Mish: ("mish",),
    nn.Hardswish: ("hardswish",), nn.Hardsigmoid: ("hardsigmoid",), nn.Sigmoid: ("sigmoid",), nn.Tanh: ("tanh",),
    nn.Softsign: ("softsign",), nn.Tanhshrink: ("tanhshrink",), nn.LogSigmoid: ("logsigmoid",),
    nn.LeakyReLU: ("leaky_relu", "negative_slope"), nn.ELU: ("elu", "alpha"), nn.CELU: ("celu", "alpha"),
    nn.Hardtanh: ("hardtanh", "min_val", "max_val"), nn.Softplus: ("softplus", "beta", "threshold"),
    nn.Hardshrink: ("hardshrink", "lambd"), nn.Softshrink: ("softshrink", "lambd"),
    nn.Threshold: ("threshold", "threshold", "value"), nn.Softmax: ("softmax", "dim"),
    nn.LogSoftmax: ("log_softmax", "dim"), nn.GLU: ("glu", "dim"),
}  # fmt: skip
SKIP_MODULES = (
    nn.Identity,
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)
TUPLE_MODULES = (
    nn.LSTM,
    nn.GRU,
    nn.RNN,
    nn.MultiheadAttention,
)  # return (output, state): only [0] is supported


def _axes(v, n):
    """One value, or a list when the axes differ (what the card's ops take)."""
    v = tuple(v) if isinstance(v, tuple | list) else (v,) * n
    return v[0] if len(set(v)) == 1 else list(v)


def _module(m, name, x_shape):
    """(card op, args) of a torch.nn layer, or None when it does nothing at inference.
    x_shape: the input's shape with the batch axis."""
    n = len(x_shape) - 2  # spatial axes
    if isinstance(m, nn.Conv1d | nn.Conv2d | nn.Conv3d):
        k, d = tuple(m.kernel_size), tuple(m.dilation)
        p = m.padding
        if isinstance(p, str):
            if p == "valid":
                p = (0,) * len(k)
            elif all((di * (ki - 1)) % 2 == 0 for ki, di in zip(k, d, strict=True)):
                p = tuple(di * (ki - 1) // 2 for ki, di in zip(k, d, strict=True))
            else:
                raise ImportUnsupported(
                    f"{name}: padding='same' with an even kernel pads unevenly: write it with a pad node"
                )
        args = {"out": m.out_channels, "k": _axes(k, len(k)), "pad": _axes(p, len(k))}
        if set(m.stride) != {1}:
            args["s"] = _axes(m.stride, len(k))
        if m.groups != 1:
            args["groups"] = (
                "depthwise" if m.groups == m.in_channels == m.out_channels else m.groups
            )
        if set(d) != {1}:
            args["dilation"] = _axes(d, len(k))
        if m.bias is None:
            args["bias"] = False
        if m.padding_mode != "zeros":
            args["padding_mode"] = m.padding_mode
        return "conv", args
    if isinstance(m, nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d):
        k = len(m.kernel_size)
        return "conv_t", {"out": m.out_channels, "k": _axes(m.kernel_size, k), "s": _axes(m.stride, k),
                          "pad": _axes(m.padding, k), "out_pad": _axes(m.output_padding, k), "groups": m.groups,
                          "dilation": _axes(m.dilation, k), "bias": m.bias is not None}  # fmt: skip
    if isinstance(m, nn.Linear):
        return "linear", {"out": m.out_features} if m.bias is not None else {
            "out": m.out_features,
            "bias": False,
        }
    if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d | nn.SyncBatchNorm):
        return "batch_norm", {
            "eps": m.eps,
            "momentum": m.momentum,
            "affine": m.affine,
            "track": m.track_running_stats,
        }
    if isinstance(m, nn.LayerNorm):
        return "layer_norm", {"axes": len(m.normalized_shape), "eps": m.eps, "affine": m.elementwise_affine,
                              "bias": m.bias is not None}  # fmt: skip
    if isinstance(m, nn.GroupNorm):
        return "group_norm", {"groups": m.num_groups, "eps": m.eps, "affine": m.affine}
    if isinstance(m, nn.InstanceNorm1d | nn.InstanceNorm2d | nn.InstanceNorm3d):
        return "instance_norm", {"eps": m.eps, "affine": m.affine, "track": m.track_running_stats}
    if isinstance(m, nn.RMSNorm):
        return "rms_norm", {
            "axes": len(m.normalized_shape),
            "eps": m.eps,
            "affine": m.elementwise_affine,
        }
    if isinstance(m, nn.LocalResponseNorm):
        return "local_response_norm", {"size": m.size, "alpha": m.alpha, "beta": m.beta, "k": m.k}
    if isinstance(m, nn.GELU):
        return "activation", {"fn": "gelu" if m.approximate == "none" else "gelu_tanh"}
    if type(m) in ACT_MODULES:
        fn, *keys = ACT_MODULES[type(m)]
        args = {"fn": fn}
        for key, slot in zip(keys, ("a", "b"), strict=False):
            v = getattr(m, key)
            if v is None:
                raise ImportUnsupported(
                    f"{name}: {type(m).__name__}({key}=None): give the axis explicitly"
                )
            args[slot] = v
        return "activation", args
    if isinstance(m, nn.PReLU):
        return "prelu", {"n": m.num_parameters, "init": float(m.init)}
    if isinstance(m, SKIP_MODULES) or type(m).__name__ in ("DropPath", "StochasticDepth"):
        return None
    if isinstance(m, nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d):
        return "pooling", {"mode": "max", "k": _axes(m.kernel_size, n), "s": _axes(m.stride, n), "pad": _axes(m.padding, n),
                           "dilation": _axes(m.dilation, n), "ceil": bool(m.ceil_mode)}  # fmt: skip
    if isinstance(m, nn.AvgPool1d | nn.AvgPool2d | nn.AvgPool3d):
        args = {"mode": "avg", "k": _axes(m.kernel_size, n), "s": _axes(m.stride, n), "pad": _axes(m.padding, n),
                "ceil": bool(m.ceil_mode), "count_pad": bool(m.count_include_pad)}  # fmt: skip
        if getattr(m, "divisor_override", None):
            args["divisor"] = m.divisor_override
        return "pooling", args
    if isinstance(m, nn.LPPool1d | nn.LPPool2d | nn.LPPool3d):
        return "pooling", {"mode": "lp", "p": float(m.norm_type), "k": _axes(m.kernel_size, n),
                           "s": _axes(m.stride or m.kernel_size, n), "ceil": bool(m.ceil_mode)}  # fmt: skip
    if isinstance(m, nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d | nn.AdaptiveMaxPool1d
                  | nn.AdaptiveMaxPool2d | nn.AdaptiveMaxPool3d):  # fmt: skip
        if getattr(m, "return_indices", False):
            raise ImportUnsupported(f"{name}: adaptive max pooling with return_indices")
        return "adaptive_pool", {"mode": "max" if "Max" in type(m).__name__ else "avg",
                                 "size": list(m.output_size) if isinstance(m.output_size, tuple) else m.output_size}  # fmt: skip
    if isinstance(m, nn.Flatten):
        return "flatten_axes", {"start": m.start_dim, "end": m.end_dim}
    if isinstance(m, nn.Unflatten):
        return "unflatten", {"axis": m.dim, "sizes": list(m.unflattened_size)}
    if isinstance(m, nn.Upsample):
        args = {"mode": m.mode, "align_corners": m.align_corners}
        if m.size is not None:
            args.update(size=list(m.size) if isinstance(m.size, tuple) else m.size, scale=None)
        else:
            args["scale"] = (
                list(m.scale_factor) if isinstance(m.scale_factor, tuple) else m.scale_factor
            )
        return "upsample", args
    if isinstance(m, nn.PixelShuffle):
        return "pixel_shuffle", {"r": m.upscale_factor}
    if isinstance(m, nn.PixelUnshuffle):
        return "pixel_shuffle", {"r": m.downscale_factor, "inverse": True}
    if isinstance(m, nn.ChannelShuffle):
        return "channel_shuffle", {"groups": m.groups}
    if isinstance(
        m,
        nn.ZeroPad1d
        | nn.ZeroPad2d
        | nn.ZeroPad3d
        | nn.ConstantPad1d
        | nn.ConstantPad2d
        | nn.ConstantPad3d,
    ):
        return "pad", {
            "pads": list(m.padding),
            "mode": "constant",
            "value": float(getattr(m, "value", 0.0)),
        }
    if isinstance(m, nn.ReflectionPad1d | nn.ReflectionPad2d | nn.ReflectionPad3d):
        return "pad", {"pads": list(m.padding), "mode": "reflect"}
    if isinstance(m, nn.ReplicationPad1d | nn.ReplicationPad2d | nn.ReplicationPad3d):
        return "pad", {"pads": list(m.padding), "mode": "replicate"}
    if isinstance(m, nn.CircularPad1d | nn.CircularPad2d | nn.CircularPad3d):
        return "pad", {"pads": list(m.padding), "mode": "circular"}
    if isinstance(m, nn.Embedding):
        if m.max_norm is not None:
            raise ImportUnsupported(
                f"{name}: Embedding(max_norm=...) renormalises its weights in forward"
            )
        return "embedding", {
            "num": m.num_embeddings,
            "dim": m.embedding_dim,
            "padding_idx": m.padding_idx,
        }
    if isinstance(m, nn.LSTM | nn.GRU | nn.RNN):
        args = {"cell": type(m).__name__.lower(), "hidden": m.hidden_size, "layers": m.num_layers, "bias": m.bias,
                "batch_first": m.batch_first, "bidirectional": m.bidirectional}  # fmt: skip
        if isinstance(m, nn.RNN):
            args["nonlinearity"] = m.nonlinearity
        if getattr(m, "proj_size", 0):
            args["proj"] = m.proj_size
        if m.dropout:
            pass  # dropout between layers: nothing at inference
        return "recurrent", args
    if type(m).__name__ in LIB_LEAVES and type(m).__module__.startswith("torchvision"):
        return "window_attention", {"heads": m.num_heads, "window": list(m.window_size), "shift": list(m.shift_size),
                                    "qkv_bias": m.qkv.bias is not None, "proj_bias": m.proj.bias is not None,
                                    "version": 2 if type(m).__name__.endswith("V2") else 1}  # fmt: skip
    raise ImportUnsupported(f"{name}: layer {type(m).__name__} has no card equivalent")


LEAF_TYPES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
    nn.InstanceNorm2d, nn.InstanceNorm3d, nn.RMSNorm, nn.LocalResponseNorm, nn.GELU, nn.PReLU, *ACT_MODULES,
    *SKIP_MODULES, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.LPPool1d,
    nn.LPPool2d, nn.LPPool3d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d, nn.Flatten, nn.Unflatten, nn.Upsample, nn.PixelShuffle,
    nn.PixelUnshuffle, nn.ChannelShuffle, nn.ZeroPad1d, nn.ZeroPad2d, nn.ZeroPad3d, nn.ConstantPad1d,
    nn.ConstantPad2d, nn.ConstantPad3d, nn.ReflectionPad1d, nn.ReflectionPad2d, nn.ReflectionPad3d,
    nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d, nn.CircularPad1d, nn.CircularPad2d,
    nn.CircularPad3d, nn.Embedding, *TUPLE_MODULES,
)  # fmt: skip


class _Tracer(torch.fx.Tracer):
    """torch.nn layers the card can write are leaves; everything else is traced into."""

    def is_leaf_module(self, m, qualname):
        return is_leaf(m)


LIB_LEAVES = (
    "ShiftedWindowAttention",
    "ShiftedWindowAttentionV2",
)  # torchvision classes ported to ops_lib


def is_leaf(m) -> bool:
    """A torch.nn layer the card can write: its own class, or a subclass that keeps its forward
    (a subclass with another forward, like torchvision's LayerNorm2d, is traced into)."""
    if type(m).__name__ in ("DropPath", "StochasticDepth", *LIB_LEAVES):
        return True
    return any(isinstance(m, t) and type(m).forward is t.forward for t in LEAF_TYPES)


class _Values(torch.fx.Interpreter):
    """Runs the traced graph once and keeps every node's value (shapes, constants)."""

    def __init__(self, gm):
        super().__init__(gm)
        self.values = {}

    def run_node(self, n):
        v = super().run_node(n)
        self.values[n.name] = v
        return v


# ----------------------------------------------------------------------------- functions and methods

FUNC_ACT = {
    "relu": "relu", "relu6": "relu6", "silu": "silu", "sigmoid": "sigmoid", "tanh": "tanh", "hardswish": "hardswish",
    "hardsigmoid": "hardsigmoid", "mish": "mish", "selu": "selu", "softsign": "softsign", "logsigmoid": "logsigmoid",
}  # fmt: skip
UNARY_F = ("exp", "log", "sqrt", "rsqrt", "abs", "neg", "sin", "cos", "reciprocal", "square", "sign", "floor", "ceil",
           "round", "erf", "log1p", "expm1", "atan", "log2")  # fmt: skip
ALIAS_F = ("contiguous", "clone", "detach", "dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout",
           "stochastic_depth", "drop_path", "float", "to", "type_as", "requires_grad_")  # fmt: skip
INLINE_MAX = 2_000_000  # values of a constant tensor written with the card (masks, index tables)
NON_TENSOR = (
    int,
    float,
    bool,
    str,
    type(None),
    torch.Size,
    torch.dtype,
    torch.device,
    slice,
    type(Ellipsis),
)


def _fname(n):
    if n.op == "call_method":
        return n.target
    t = n.target
    name = getattr(t, "__name__", str(t))
    return name.strip("_")


def _arg(n, i, key, default=None):
    if len(n.args) > i:
        return n.args[i]
    return n.kwargs.get(key, default)


def nodes_of(model: nn.Module, classes: int | None = None, x=None):
    """(card nodes, whether the model's last Linear became the head, the head's layer name).

    `x`: an example input (batch included). Every node of the traced graph is mapped on its
    concrete value: shapes, sizes and constants come from one run on `x`."""
    try:
        gm = torch.fx.GraphModule(model, _Tracer().trace(model))
    except ImportUnsupported:
        raise
    except Exception as e:  # einops, data-dependent control flow, ...: torch.fx cannot follow it
        raise ImportUnsupported(
            f"torch.fx cannot trace this model ({type(e).__name__}: {str(e)[:120]}): write the card by hand"
        ) from e
    if x is None:
        raise ImportUnsupported("an example input is needed to read the shapes")
    run = _Values(gm)
    with torch.no_grad():
        run.run(x)
    vals = run.values
    mods = dict(model.named_modules())
    static = _static_nodes(gm.graph)
    alias, nodes, tuples = {}, [], {}
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def is_t(a):
        return isinstance(a, torch.fx.Node) and isinstance(vals.get(a.name), torch.Tensor)

    def shape(a):
        return tuple(vals[a.name].shape)

    pending = {}  # get_attr tensors: a param node is written where one is first used as data
    done_static = set()
    first_use = {}  # (layer name, op, args) -> the first node written from it
    tmajor = set()  # tensors that are (L, B, ...) in the original; the card keeps them (B, L, ...)
    tm_ok = {"v": False}  # set while an operation that understands time-major tensors reads one
    moved = {}  # permutes that move the batch axis: written as select + permute where they are indexed

    def from_moved(n, p_node, i):
        """permute(order)[i] == select(order[0], i) then a permute of the other axes."""
        x, order = moved[p_node.name]
        j = order[0]
        rest = [a if a < j else a - 1 for a in order[1:]]
        if rest[0] != 0:
            raise ImportUnsupported(
                f"{n.name}: the batch axis does not come back first after indexing"
            )
        name = getattr(n, "name", n)
        if rest == list(range(len(rest))):
            return add(n, "select", [src(x)], {"axis": j, "index": int(i)})
        sel = add(f"{name}_sel", "select", [src(x)], {"axis": j, "index": int(i)})
        return add(n, "permute", [sel], {"order": rest})

    def src(a):
        if not isinstance(a, torch.fx.Node):
            raise ImportUnsupported(f"a constant ({a!r}) is used where a tensor is expected")
        if a.name in moved:
            raise ImportUnsupported(
                f"{a.name}: a permute that moves the batch axis is used other than by [i] / unbind(0)"
            )
        if a.name in tmajor and not tm_ok["v"]:
            raise ImportUnsupported(f"{a.name}: a time-major tensor (batch on axis 1) is used by an operation other "
                                    "than a recurrent layer, a transpose back to batch first, or indexing")  # fmt: skip
        if a.name in pending:
            args, ref = pending.pop(a.name)
            add(a, "param", [], args, ref=ref)
        elif a.name in static and a.op != "get_attr" and a.name not in done_static:
            args, refs = _static_program(a, mods, vals, params, buffers)
            add(a, "static", [], args, ref=refs)
            done_static.add(a.name)
        return alias.get(a.name, a.name)

    def rank(a):
        return len(shape(a))

    def axis(a, d):
        d = int(d)
        return d + rank(a) if d < 0 else d

    def add(n, module, sources, args=None, ref=None):
        node = {"id": getattr(n, "name", n), "module": module}
        prev = nodes[-1]["id"] if nodes else "input"
        if module in ("param", "static"):
            pass
        elif len(sources) > 1 or module in ("mha",):
            node["from"] = sources
        elif sources[0] != prev:
            node["from"] = sources[0]
        if args:
            node["args"] = args
        if ref:
            node["ref"] = ref
            key = (str(ref), module, str(args))
            if isinstance(ref, str) and key in first_use:  # a layer called again: shared weights
                node["share"] = first_use[key]
            else:
                first_use.setdefault(key, node["id"])
        nodes.append(node)
        return node["id"]

    def scalar_or_merge(n, fn, a, b, rev_fn=None):
        if is_t(a) and is_t(b):
            add(n, {"truediv": "div", "floordiv": "div"}.get(fn, fn), [src(a), src(b)])
        elif is_t(a):
            add(n, "scalar", [src(a)], {"fn": fn, "value": _num(b)})
        elif is_t(b):
            add(n, "scalar", [src(b)], {"fn": rev_fn or fn, "value": _num(a)})
        else:
            alias[n.name] = None  # arithmetic on sizes: not a tensor

    def _num(v):
        if isinstance(v, torch.fx.Node):
            v = vals[v.name]
        if isinstance(v, torch.Tensor):
            if v.numel() != 1:
                raise ImportUnsupported(
                    "a constant tensor is combined with the activations: write this part by hand"
                )
            v = v.item()
        return float(v) if isinstance(v, float) else v

    head_ref = None
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            alias[n.name] = "input"
            continue
        if n.op == "output":
            outv = n.args[0]
            if isinstance(outv, tuple | list):
                raise ImportUnsupported(
                    "the model returns several outputs (an auxiliary head?): give the module that returns one"
                )
            if isinstance(outv, dict):
                raise ImportUnsupported(
                    "the model returns a dict: give the module that returns the tensor"
                )
            out = src(outv)
            continue
        v = vals.get(n.name)
        if n.op in ("call_function", "call_method", "call_module"):
            n = _Resolved(n, vals)  # sizes computed from shapes -> their values in this run
        if n.name in static and n.op != "get_attr":
            continue  # computed from the network's own tensors only: a static node, written where it is used
        if n.op == "get_attr":
            t = n.target
            if t in params:
                pending[n.name] = ({"shape": list(params[t].shape) or [1]}, t)
            elif t in buffers:
                b = buffers[t]
                if b.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                    alias[n.name] = (
                        None  # an integer index table: only usable by the ops that read it
                    )
                    continue
                pending[n.name] = ({"shape": list(b.shape) or [1], "learned": False}, t)
            elif isinstance(
                v, torch.Tensor
            ):  # a plain tensor attribute (not a weight): its values go into the card
                if v.numel() > INLINE_MAX:
                    raise ImportUnsupported(
                        f"{t}: a constant tensor of {v.numel():,} values used as data"
                    )
                entry = {"shape": list(v.shape), "learned": False, "value": v.tolist()}
                if v.dtype != torch.float32:
                    entry["dtype"] = str(v.dtype).replace("torch.", "")
                add(n, "static", [], {"tensors": [entry], "program": []})
            else:
                alias[n.name] = None
            continue
        if not isinstance(v, torch.Tensor):
            if isinstance(v, tuple | list) and any(isinstance(x_, torch.Tensor) for x_ in v):
                tuples[n.name] = n
                continue
            if isinstance(v, NON_TENSOR) or (
                isinstance(v, tuple | list) and all(isinstance(x_, NON_TENSOR) for x_ in v)
            ):
                alias[n.name] = (
                    None  # sizes, shapes, dims: read from the values, never computed by the card
                )
                continue
            raise ImportUnsupported(f"{n.name}: produces {type(v).__name__}, not a tensor")
        if n.op == "call_module":
            m = mods[n.target]
            x0 = n.args[0]
            if isinstance(m, nn.MultiheadAttention):
                raise ImportUnsupported(
                    f"{n.target}: MultiheadAttention returns a tuple"
                )  # handled through getitem
            got = _module(m, n.target, shape(x0))
            if got is None:
                alias[n.name] = src(x0)
                continue
            add(n, got[0], [src(x0)], got[1], ref=n.target)
            continue
        f = _fname(n)
        deps: list = []
        torch.fx.node.map_arg(
            (getattr(n, "node", n).args, getattr(n, "node", n).kwargs), deps.append
        )
        if f in ("arange", "linspace", "logspace", "eye", "tensor", "as_tensor", "meshgrid", "ones", "zeros", "full") \
                and not any(isinstance(vals.get(d.name), torch.Tensor) for d in deps) \
                and (v.dim() == 0 or v.shape[0] != x.shape[0]):  # fmt: skip
            # a constant made from sizes (positions, a grid): its values in this run go into the card
            if v.numel() > INLINE_MAX:
                raise ImportUnsupported(f"{n.name}: a constant tensor of {v.numel():,} values")
            entry = {
                "shape": list(v.shape) or [1],
                "learned": False,
                "value": v.reshape(v.shape or (1,)).tolist(),
            }
            if v.dtype != torch.float32:
                entry["dtype"] = str(v.dtype).replace("torch.", "")
            add(n, "static", [], {"tensors": [entry], "program": []})
            continue
        if f.endswith("_") and f[:-1] in (
            "add",
            "sub",
            "mul",
            "div",
            "clamp",
            "relu",
            "sigmoid",
            "tanh",
            "neg",
        ):
            f = f[:-1]  # in-place: the same value (verify compares the outputs)
        tens = [a for a in n.args if is_t(a)]
        if f == "getitem" and not getattr(n, "node", n).users:
            continue  # an unused item (`y, _ = rnn(x)`)
        if f == "getitem":
            base, idx = n.args
            if base.name in moved and isinstance(idx, int):
                from_moved(n, base, idx)
                continue
            if base.name in tmajor:  # x[t] of (L, B, ...): the card's tensor is (B, L, ...)
                ix = idx if isinstance(idx, tuple) else (idx,)
                ix = (*ix, slice(None)) if len(ix) == 1 else ix
                ix = (ix[1], ix[0], *ix[2:])
                tm_ok["v"] = True
                _getitem_tensor(n, base, ix, add, src, axis, rank, shape, vals)
                tm_ok["v"] = False
                if isinstance(ix[1], slice):
                    tmajor.add(n.name)
                continue
            tb = tuples.get(base.name)
            if (
                tb is not None
                and tb.op == "call_module"
                and isinstance(mods[tb.target], nn.LSTM | nn.GRU | nn.RNN)
            ):
                if idx != 0:
                    raise ImportUnsupported(
                        f"{tb.target}: only the output ([0]) of a recurrent layer is supported"
                    )
                got = _module(mods[tb.target], tb.target, shape(tb.args[0]))
                x_in = tb.args[0]
                args = dict(got[1])
                if x_in.name in tmajor and not args["batch_first"]:
                    args["batch_first"] = (
                        True  # the card's tensor is batch first; the output stays so
                    )
                    tmajor.add(n.name)
                elif not args["batch_first"]:
                    raise ImportUnsupported(
                        f"{tb.target}: batch_first=False on a batch-first tensor"
                    )
                tm_ok["v"] = True
                add(n, "recurrent", [src(x_in)], args, ref=tb.target)
                tm_ok["v"] = False
                continue
            if (
                tb is not None
                and _fname(tb) == "unbind"
                and tb.args[0].name in moved
                and _arg(tb, 1, "dim", 0) == 0
            ):
                from_moved(n, tb.args[0], idx)
                continue
            if base.name in tuples:
                _getitem_tuple(n, base, idx, tuples, mods, vals, add, src, axis, shape)
                continue
            _getitem_tensor(n, base, idx, add, src, axis, rank, shape, vals)
            continue
        if f in ALIAS_F or (f in ("expand", "expand_as") and n.args[0].op == "get_attr"):
            alias[n.name] = src(n.args[0])
            continue
        if f in ("add", "iadd", "radd"):
            scalar_or_merge(n, "add", n.args[0], n.args[1])
        elif f in ("sub", "isub", "rsub"):
            a, b = (n.args[1], n.args[0]) if f == "rsub" else n.args[:2]
            scalar_or_merge(n, "sub", a, b, "rsub")
        elif f in ("mul", "imul", "rmul"):
            scalar_or_merge(n, "mul", n.args[0], n.args[1])
        elif f in ("truediv", "div", "itruediv"):
            if n.kwargs.get("rounding_mode"):
                raise ImportUnsupported(f"{n.name}: division with rounding_mode")
            scalar_or_merge(n, "div", n.args[0], n.args[1], "rdiv")
        elif f in ("pow",):
            scalar_or_merge(n, "pow", n.args[0], n.args[1], "rpow") if not (
                is_t(n.args[0]) and is_t(n.args[1])
            ) else add(n, "pow2", [src(n.args[0]), src(n.args[1])])
        elif f in ("matmul", "bmm"):
            add(n, "matmul", [src(n.args[0]), src(n.args[1])])
        elif f in ("cat", "concat", "concatenate") and len(n.args[0]) == 1:
            alias[n.name] = src(n.args[0][0])
        elif f in ("cat", "concat", "concatenate", "stack"):
            seq = n.args[0]
            d = _arg(n, 1, "dim", 0)
            add(
                n,
                "stack" if f == "stack" else "concat",
                [src(a) for a in seq],
                {"axis": axis(n, d) if f != "stack" else int(d) % (rank(seq[0]) + 1)},
            )
        elif f in FUNC_ACT:
            add(n, "activation", [src(tens[0])], {"fn": FUNC_ACT[f]})
        elif f == "gelu":
            add(
                n,
                "activation",
                [src(tens[0])],
                {"fn": "gelu_tanh" if n.kwargs.get("approximate") == "tanh" else "gelu"},
            )
        elif f in ("leaky_relu",):
            add(
                n,
                "activation",
                [src(tens[0])],
                {"fn": "leaky_relu", "a": _arg(n, 1, "negative_slope", 0.01)},
            )
        elif f in ("elu", "celu"):
            add(n, "activation", [src(tens[0])], {"fn": f, "a": _arg(n, 1, "alpha", 1.0)})
        elif f == "softplus":
            add(
                n,
                "activation",
                [src(tens[0])],
                {
                    "fn": "softplus",
                    "a": _arg(n, 1, "beta", 1.0),
                    "b": _arg(n, 2, "threshold", 20.0),
                },
            )
        elif f in ("hardtanh",):
            add(
                n,
                "activation",
                [src(tens[0])],
                {
                    "fn": "hardtanh",
                    "a": _arg(n, 1, "min_val", -1.0),
                    "b": _arg(n, 2, "max_val", 1.0),
                },
            )
        elif f in ("softmax", "log_softmax"):
            d = _arg(n, 1, "dim")
            if d is None:
                raise ImportUnsupported(f"{n.name}: {f} without dim")
            add(n, "activation", [src(tens[0])], {"fn": f, "a": axis(n, d)})
        elif f == "glu":
            add(
                n,
                "activation",
                [src(tens[0])],
                {"fn": "glu", "a": axis(n.args[0], _arg(n, 1, "dim", -1))},
            )
        elif f in ("clamp", "clip", "clamp_min", "clamp_max"):
            lo = _arg(n, 1, "min") if f != "clamp_max" else None
            hi = (
                _arg(n, 2, "max")
                if f in ("clamp", "clip")
                else (_arg(n, 1, "max") if f == "clamp_max" else None)
            )
            add(
                n,
                "clamp",
                [src(tens[0])],
                {
                    "min": _num(lo) if lo is not None else None,
                    "max": _num(hi) if hi is not None else None,
                },
            )
        elif f in ("invert", "logical_not") and v.dtype == torch.bool:
            add(n, "unary", [src(tens[0])], {"fn": "logical_not"})
        elif f in UNARY_F:
            add(n, "unary", [src(tens[0])], {"fn": f})
        elif f in ("reshape", "view", "reshape_as", "view_as"):
            add(n, "reshape", [src(n.args[0])], {"shape": list(v.shape[1:])})
        elif f in ("expand", "expand_as", "broadcast_to"):
            if list(v.shape) == list(shape(n.args[0])):
                alias[n.name] = src(n.args[0])
            elif v.shape[0] != shape(n.args[0])[0]:
                raise ImportUnsupported(f"{n.name}: expand of the batch axis")
            else:
                add(n, "expand", [src(n.args[0])], {"shape": list(v.shape[1:])})
        elif f == "flatten":
            st = axis(n.args[0], _arg(n, 1, "start_dim", 0))
            if st == 0:  # the batch axis merges with the next ones: a reshape whose batch grows
                add(n, "reshape", [src(n.args[0])], {"shape": list(v.shape[1:])})
            else:
                add(
                    n,
                    "flatten_axes",
                    [src(n.args[0])],
                    {"start": st, "end": axis(n.args[0], _arg(n, 2, "end_dim", -1))},
                )
        elif f == "permute":
            order = (
                n.args[1]
                if len(n.args) == 2 and isinstance(n.args[1], tuple | list)
                else n.args[1:] or n.kwargs.get("dims")
            )
            order = [axis(n.args[0], d) for d in order]
            tm_in = n.args[0].name in tmajor
            if tm_in:  # the card's tensor is batch first: swap axes 0 and 1 of the order
                order = [{0: 1, 1: 0}.get(a, a) for a in order]
                tm_ok["v"] = True
            users = list(getattr(n, "node", n).users)
            split_only = bool(users) and all(_fname(u) in ("unbind", "getitem") for u in users)
            if (
                order[0] != 0 and order[1] == 0 and not split_only
            ):  # -> time-major: the card keeps the batch first
                bf = [0, order[0], *order[2:]]
                if bf == list(range(len(bf))):
                    alias[n.name] = src(n.args[0])
                else:
                    add(n, "permute", [src(n.args[0])], {"order": bf})
                tmajor.add(n.name)
                tm_ok["v"] = False
                continue
            if (
                order[0] != 0
            ):  # the batch axis moves: only for unbind(0) / [i] (q, k, v = qkv.permute(2, 0, ...))
                moved[n.name] = (n.args[0], order)
                tm_ok["v"] = False
                continue
            if order == list(range(len(order))):
                alias[n.name] = src(n.args[0])
            else:
                add(n, "permute", [src(n.args[0])], {"order": order})
            tm_ok["v"] = False
        elif f in ("transpose", "swapaxes"):
            pair = {axis(n.args[0], _arg(n, 1, "dim0")), axis(n.args[0], _arg(n, 2, "dim1"))}
            tm_in = n.args[0].name in tmajor
            if pair == {0, 1}:  # batch first <-> time-major: the card's tensor stays batch first
                tm_ok["v"] = True
                alias[n.name] = src(n.args[0])
                tm_ok["v"] = False
                if not tm_in:
                    tmajor.add(n.name)
                continue
            if tm_in:
                raise ImportUnsupported(
                    f"{n.name}: a transpose of a time-major tensor other than back to batch first"
                )
            a_, b_ = sorted(pair)
            add(n, "transpose", [src(n.args[0])], {"a": a_, "b": b_})
        elif f == "unsqueeze":
            add(n, "unsqueeze", [src(n.args[0])], {"axis": axis(n, _arg(n, 1, "dim"))})
        elif f == "squeeze":
            d = _arg(n, 1, "dim")
            if d is None or isinstance(d, tuple | list):
                add(n, "reshape", [src(n.args[0])], {"shape": list(v.shape[1:])})
            else:
                d = axis(n.args[0], d)
                if shape(n.args[0])[d] == 1:
                    add(n, "squeeze", [src(n.args[0])], {"axis": d})
                else:
                    alias[n.name] = src(n.args[0])
        elif f in ("norm", "vector_norm") and _arg(n, 1, "p" if f == "norm" else "ord", 2) in (
            2,
            2.0,
            "fro",
        ):
            d = _arg(n, 2, "dim")
            if d is None:
                raise ImportUnsupported(f"{n.name}: norm over every axis (batch included)")
            dl = [d] if isinstance(d, int) else list(d)
            add(n, "reduce", [src(n.args[0])], {"fn": "norm", "axes": [axis(n.args[0], a) for a in dl],
                                                "keepdim": bool(_arg(n, 3, "keepdim", False))})  # fmt: skip
        elif f in ("mean", "sum", "amax", "amin", "std", "var", "logsumexp"):
            d = _arg(n, 1, "dim")
            if d is None:
                raise ImportUnsupported(f"{n.name}: {f} over every axis (batch included)")
            dl = [d] if isinstance(d, int) else list(d)
            args = {
                "fn": f,
                "axes": [axis(n.args[0], a) for a in dl],
                "keepdim": bool(_arg(n, 2, "keepdim", False)),
            }
            if f in ("std", "var"):
                corr = n.kwargs.get("correction", 0 if n.kwargs.get("unbiased") is False else 1)
                args["correction"] = corr
            add(n, "reduce", [src(n.args[0])], args)
        elif f in (
            "adaptive_avg_pool1d",
            "adaptive_avg_pool2d",
            "adaptive_avg_pool3d",
            "adaptive_max_pool2d",
        ):
            size = _arg(n, 1, "output_size")
            add(n, "adaptive_pool", [src(n.args[0])], {"mode": "max" if "max" in f else "avg",
                                                        "size": list(size) if isinstance(size, tuple | list) else size})  # fmt: skip
        elif f in (
            "avg_pool2d",
            "max_pool2d",
            "avg_pool1d",
            "max_pool1d",
            "avg_pool3d",
            "max_pool3d",
        ):
            k = _arg(n, 1, "kernel_size")
            st = _arg(n, 2, "stride") or k
            args = {"mode": "max" if "max" in f else "avg", "k": k if isinstance(k, int) else list(k),
                    "s": st if isinstance(st, int) else list(st), "pad": _arg(n, 3, "padding", 0),
                    "ceil": bool(_arg(n, 5 if "max" in f else 4, "ceil_mode", False))}  # fmt: skip
            if "avg" in f:
                args["count_pad"] = bool(_arg(n, 5, "count_include_pad", True))
            add(n, "pooling", [src(n.args[0])], args)
        elif f in ("interpolate", "upsample"):
            size, sf = _arg(n, 1, "size"), _arg(n, 2, "scale_factor")
            args = {
                "mode": _arg(n, 3, "mode", "nearest"),
                "align_corners": _arg(n, 4, "align_corners"),
            }
            if size is not None:
                args.update(size=list(v.shape[2:]), scale=None)
            else:
                args["scale"] = sf if not isinstance(sf, tuple | list) else list(sf)
            add(n, "upsample", [src(n.args[0])], args)
        elif f == "pad":
            add(n, "pad", [src(n.args[0])], {"pads": list(_arg(n, 1, "pad")), "mode": _arg(n, 2, "mode", "constant"),
                                             "value": float(_arg(n, 3, "value", 0.0) or 0.0)})  # fmt: skip
        elif f == "roll":
            sh, ds = _arg(n, 1, "shifts"), _arg(n, 2, "dims")
            sh = list(sh) if isinstance(sh, tuple | list) else [sh]
            ds = list(ds) if isinstance(ds, tuple | list) else [ds]
            add(
                n,
                "roll",
                [src(n.args[0])],
                {"shifts": sh, "axes": [axis(n.args[0], d) for d in ds]},
            )
        elif f == "flip":
            add(
                n,
                "flip",
                [src(n.args[0])],
                {"axes": [axis(n.args[0], d) for d in _arg(n, 1, "dims")]},
            )
        elif f == "normalize":
            add(n, "normalize", [src(n.args[0])], {"p": float(_arg(n, 1, "p", 2.0)), "axis": axis(n.args[0], _arg(n, 2, "dim", 1)),
                                                   "eps": float(_arg(n, 3, "eps", 1e-12))})  # fmt: skip
        elif f == "scaled_dot_product_attention":
            mask = _arg(n, 3, "attn_mask")
            ins = [src(a) for a in n.args[:3]] + ([src(mask)] if is_t(mask) else [])
            add(
                n,
                "attention",
                ins,
                {"causal": bool(_arg(n, 5, "is_causal", False)), "scale": n.kwargs.get("scale")},
            )
        elif f == "where" and len(n.args) == 3 and all(is_t(a) for a in n.args):
            add(n, "where", [src(a) for a in n.args])
        elif f in ("maximum", "minimum"):
            add(n, f, [src(a) for a in n.args[:2]])
        elif f in (
            "layer_norm",
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "batch_norm",
            "group_norm",
        ):
            _functional_layer(n, f, add, src, shape, vals, params)
        elif f == "einsum":
            eq = n.args[0]
            ops_ = (
                n.args[1]
                if len(n.args) == 2 and isinstance(n.args[1], tuple | list)
                else n.args[1:]
            )
            add(n, "einsum", [src(a) for a in ops_], {"equation": eq})
        elif f == "patch_merging_pad":
            add(n, "patch_merge_pad", [src(n.args[0])])
        elif f in (
            "zeros",
            "ones",
            "full",
            "zeros_like",
            "ones_like",
            "full_like",
            "new_zeros",
            "new_ones",
        ):
            if v.dim() == 0 or v.shape[0] != x.shape[0]:
                raise ImportUnsupported(f"{n.name}: {f} without the batch axis first")
            fill = (
                1.0
                if "ones" in f
                else float(_arg(n, 1, "fill_value", 0.0))
                if f.startswith("full")
                else 0.0
            )
            if f == "full":
                fill = float(_arg(n, 1, "fill_value"))
            # a constant tensor as large as the batch: computed from the input's batch size
            add(n, "fill", ["input"], {"shape": list(v.shape[1:]), "value": fill})
        elif f == "addcmul":
            a, b, c = n.args[:3]
            val = float(n.kwargs.get("value", 1.0))
            m1 = add(f"{n.name}_mul", "mul", [src(b), src(c)])
            m2 = (
                add(f"{n.name}_scale", "scalar", [m1], {"fn": "mul", "value": val})
                if val != 1.0
                else m1
            )
            add(n, "add", [src(a), m2])
        elif f == "repeat_interleave":
            reps, d = _arg(n, 1, "repeats"), _arg(n, 2, "dim")
            if not isinstance(reps, int) or d is None:
                raise ImportUnsupported(
                    f"{n.name}: repeat_interleave with tensor repeats or without dim"
                )
            add(
                n,
                "repeat_interleave",
                [src(n.args[0])],
                {"repeats": reps, "axis": axis(n.args[0], d)},
            )
        elif f == "cumsum":
            add(n, "cumsum", [src(n.args[0])], {"axis": axis(n.args[0], _arg(n, 1, "dim"))})
        elif f in ("chunk", "split", "unbind", "unflatten"):
            if f == "unflatten":
                add(
                    n,
                    "unflatten",
                    [src(n.args[0])],
                    {
                        "axis": axis(n.args[0], _arg(n, 1, "dim")),
                        "sizes": list(_arg(n, 2, "sizes")),
                    },
                )
            else:
                tuples[n.name] = n
        else:
            raise ImportUnsupported(f"{n.name}: operation {f!r} has no card equivalent")
        if n.name in alias and alias[n.name] is None and isinstance(v, torch.Tensor):
            raise ImportUnsupported(f"{n.name}: tensor operation {f!r} on constants")
    if not nodes or out != nodes[-1]["id"]:
        if out in alias.values() or out == "input":
            raise ImportUnsupported("the model's output is not computed by a layer")
        # the output is an earlier node: read it last
        nodes.append({"id": "out", "module": "as_kind", "from": out, "args": {"to": "nd"}})
    head = False
    last = nodes[-1]
    if (
        classes
        and last["module"] == "linear"
        and last["args"].get("out") == classes
        and "bias" not in last["args"]
    ):
        nodes.pop()
        if "from" in last and (not nodes or last["from"] != nodes[-1]["id"]):
            nodes.append(
                {"id": "pre_head", "module": "as_kind", "from": last["from"], "args": {"to": "nd"}}
            )
        head, head_ref = True, last.get("ref")
    return nodes, head, head_ref


class _Resolved:
    """A traced node whose non-tensor arguments (sizes computed from shapes, paddings, ...) are
    replaced by their values in the import run; tensor arguments stay nodes."""

    def __init__(self, n, vals):
        self.node, self.name, self.op, self.target = n, n.name, n.op, n.target

        def val(a):
            v = vals.get(a.name)
            if isinstance(v, torch.Tensor) or (
                isinstance(v, tuple | list) and any(isinstance(x, torch.Tensor) for x in v)
            ):
                return a
            return a if a.op == "get_attr" else v

        self.args = torch.fx.node.map_arg(n.args, val)
        self.kwargs = torch.fx.node.map_arg(n.kwargs, val)

    def __repr__(self):
        return f"<{self.name}>"


def _static_nodes(graph) -> set:
    """Nodes computed from the network's own tensors and constants only (no path from the input)."""
    out = set()
    for n in graph.nodes:
        if n.op == "get_attr":
            out.add(n.name)
        elif n.op in ("call_function", "call_method", "call_module"):
            deps: list = []
            torch.fx.node.map_arg((n.args, n.kwargs), deps.append)
            if deps and all(d.name in out for d in deps):
                out.add(n.name)
    return out


def _static_program(node, mods, vals, params, buffers):
    """The `static` card node (ops_prim.StaticExpr) computing `node` from the network's tensors."""
    order, seen = [], set()

    def visit(n):
        if n.name in seen:
            return
        seen.add(n.name)
        torch.fx.node.map_arg((n.args, n.kwargs), visit)
        order.append(n)

    visit(node)
    attrs = [n for n in order if n.op == "get_attr"]
    learned = [n for n in attrs if n.target in params]
    consts = [n for n in attrs if n.target in buffers]
    inline = [
        n for n in attrs if n.target not in params and n.target not in buffers
    ]  # plain tensor attributes
    pos, tensors, refs = {}, [], []
    for n in learned + consts + inline:
        t = params.get(n.target, buffers.get(n.target, vals.get(n.name)))
        if not isinstance(t, torch.Tensor):
            raise ImportUnsupported(f"{n.target}: a constant that is not a tensor")
        pos[n.name] = len(tensors)
        entry = {"shape": list(t.shape), "learned": n.target in params}
        if t.dtype != torch.float32:
            entry["dtype"] = str(t.dtype).replace("torch.", "")
        if n in inline:
            if t.numel() > INLINE_MAX:
                raise ImportUnsupported(
                    f"{n.target}: a constant tensor of {t.numel():,} values is not a parameter or buffer"
                )
            entry["value"] = t.tolist()  # written into the card: not a weight of the model
        else:
            refs.append(n.target)
        tensors.append(entry)
    program, layers, layer_refs = [], [], []

    def ref_of(a):
        return pos[a.name]

    def step(n, op_, ins, **kw):
        program.append({"op": op_, "in": [ref_of(a) for a in ins], **kw})
        pos[n.name] = len(tensors) + len(program) - 1

    for n in order:
        if n.op == "get_attr":
            continue
        f = _fname(n)
        v = vals[n.name]
        a0 = n.args[0] if n.args else None
        if n.op == "call_module":
            m = mods[n.target]
            if isinstance(m, SKIP_MODULES) or type(m).__name__ in ("DropPath", "StochasticDepth"):
                pos[n.name] = ref_of(a0)  # nothing at inference
                continue
            if isinstance(m, nn.Linear):
                layers.append(
                    {
                        "op": "linear",
                        "in": m.in_features,
                        "out": m.out_features,
                        "bias": m.bias is not None,
                    }
                )
                layer_refs.append(n.target)
            elif type(m) in ACT_MODULES or isinstance(m, nn.GELU):
                got = _module(m, n.target, tuple(vals[a0.name].shape))
                layers.append({"op": "activation", **got[1]})
            else:
                raise ImportUnsupported(
                    f"{n.target}: {type(m).__name__} inside a computation on the weights"
                )
            step(n, "layer", [a0], layer=len(layers) - 1)
        elif f == "getitem":
            idx = n.args[1]
            if isinstance(idx, torch.fx.Node):
                step(n, "index", [a0, idx])
            else:
                idx = idx if isinstance(idx, tuple) else (idx,)
                tens_ix = [i for i in idx if isinstance(i, torch.fx.Node)]
                spec = [{"t": 1 + tens_ix.index(i)} if isinstance(i, torch.fx.Node) else None if i is None
                        else "..." if i is Ellipsis else [i.start, i.stop, i.step] if isinstance(i, slice)
                        else int(i) for i in idx]  # fmt: skip
                step(n, "item", [a0, *tens_ix], index=spec)
        elif (
            f == "get_relative_position_bias"
        ):  # torchvision's wrapped helper: table[index] -> (1, heads, N, N)
            table, index, window = n.args[:3]
            nn_ = int(window[0]) * int(window[1])
            step(n, "index", [table, index])
            program.append({"op": "reshape", "in": [pos[n.name]], "shape": [nn_, nn_, -1]})
            program.append(
                {"op": "permute", "in": [len(tensors) + len(program) - 1], "order": [2, 0, 1]}
            )
            program.append({"op": "unsqueeze", "in": [len(tensors) + len(program) - 1], "axis": 0})
            pos[n.name] = len(tensors) + len(program) - 1
        elif f in ("view", "reshape"):
            step(n, "reshape", [a0], shape=list(v.shape))
        elif f == "permute":
            order_ = (
                n.args[1]
                if len(n.args) == 2 and isinstance(n.args[1], tuple | list)
                else n.args[1:]
            )
            step(n, "permute", [a0], order=[int(d) for d in order_])
        elif f == "transpose":
            step(n, "transpose", [a0], a=int(n.args[1]), b=int(n.args[2]))
        elif f == "unsqueeze":
            step(n, "unsqueeze", [a0], axis=int(_arg(n, 1, "dim")))
        elif f == "squeeze":
            step(n, "reshape", [a0], shape=list(v.shape))
        elif f == "flatten":
            step(
                n,
                "flatten",
                [a0],
                start=int(_arg(n, 1, "start_dim", 0)),
                end=int(_arg(n, 2, "end_dim", -1)),
            )
        elif f in ("expand", "expand_as", "repeat"):
            step(
                n, "expand" if f != "repeat" else "reshape", [a0], shape=list(v.shape)
            ) if f != "repeat" else None
            if f == "repeat":
                raise ImportUnsupported(f"{n.name}: repeat on the weights")
        elif f in ("contiguous", "float", "clone", "detach", "to"):
            pos[n.name] = ref_of(a0)
        elif f in ("long", "int"):
            step(n, "long", [a0])
        elif f in ("add", "sub", "mul", "truediv", "div", "pow", "iadd", "imul", "rsub", "matmul"):
            g = {"truediv": "div", "iadd": "add", "imul": "mul"}.get(f, f)
            a, b = n.args[:2]
            if isinstance(a, torch.fx.Node) and isinstance(b, torch.fx.Node):
                step(n, g, [a, b])
            elif isinstance(a, torch.fx.Node):
                step(n, g, [a], value=b)
            else:
                step(n, {"sub": "rsub", "div": "rdiv"}.get(g, g), [b], value=a)
        elif f in ("dropout", "identity"):
            pos[n.name] = ref_of(a0)
        elif f in (
            "exp",
            "sigmoid",
            "log",
            "log2",
            "sign",
            "abs",
            "sqrt",
            "neg",
            "tanh",
            "relu",
            "sin",
            "cos",
        ):
            step(n, f, [a0])
        elif f in ("clamp", "clip"):
            step(n, "clamp", [a0], min=_arg(n, 1, "min"), max=_arg(n, 2, "max"))
        elif f == "softmax":
            step(n, "softmax", [a0], dim=int(_arg(n, 1, "dim")))
        elif f in ("cat", "concat", "stack"):
            step(
                n,
                "stack" if f == "stack" else "cat",
                list(n.args[0]),
                dim=int(_arg(n, 1, "dim", 0)),
            )
        elif f in ("zeros_like", "ones_like"):
            step(n, f, [a0])
        else:
            raise ImportUnsupported(
                f"{n.name}: operation {f!r} on the weights has no static equivalent"
            )
    if pos[node.name] != len(tensors) + len(program) - 1:
        program.append({"op": "contiguous", "in": [pos[node.name]]})
    return {"tensors": tensors, "program": program} | (
        {"layers": layers} if layers else {}
    ), refs + layer_refs


def _getitem_tuple(n, base, idx, tuples, mods, vals, add, src, axis, shape):
    """`chunk(...)[i]`, `split(...)[i]`, `unbind(...)[i]`, `lstm(x)[0]`, `mha(q, k, v)[0]`."""
    b = tuples[base.name]
    if b.op == "call_module":
        m = mods[b.target]
        if idx != 0:
            raise ImportUnsupported(
                f"{b.target}: only the output ([0]) of {type(m).__name__} is supported"
            )
        if isinstance(m, nn.MultiheadAttention):
            q, k, v = (b.args + (None, None, None))[:3]
            k, v = k or b.kwargs.get("key"), v or b.kwargs.get("value")
            if (
                b.kwargs.get("attn_mask") is not None
                or b.kwargs.get("key_padding_mask") is not None
            ):
                raise ImportUnsupported(f"{b.target}: attention masks are not supported")
            args = {
                "heads": m.num_heads,
                "bias": m.in_proj_bias is not None,
                "batch_first": m.batch_first,
            }
            if m.add_zero_attn or m.bias_k is not None:
                args.update(add_bias_kv=m.bias_k is not None, add_zero_attn=m.add_zero_attn)
            if not m._qkv_same_embed_dim:
                args.update(kdim=m.kdim, vdim=m.vdim)
            add(n, "mha", [src(q), src(k), src(v)], args, ref=b.target)
            return
        got = _module(m, b.target, shape(b.args[0]))
        add(n, got[0], [src(b.args[0])], got[1], ref=b.target)
        return
    f = _fname(b)
    x0 = b.args[0]
    if f == "chunk":
        add(
            n,
            "chunk_part",
            [src(x0)],
            {"chunks": _arg(b, 1, "chunks"), "axis": axis(x0, _arg(b, 2, "dim", 0)), "i": idx},
        )
    elif f == "split":
        sz = (
            _arg(b, 1, "split_size_or_sections")
            if b.op == "call_function"
            else _arg(b, 1, "split_size")
        )
        add(n, "split_part", [src(x0)], {"sizes": list(sz) if isinstance(sz, tuple | list) else sz,
                                         "axis": axis(x0, _arg(b, 2, "dim", 0)), "i": idx})  # fmt: skip
    elif f == "unbind":
        add(n, "select", [src(x0)], {"axis": axis(x0, _arg(b, 1, "dim", 0)), "index": idx})
    elif f in ("max", "min", "sort", "topk"):
        if f in ("max", "min") and idx == 0:
            add(n, "reduce", [src(x0)], {"fn": "amax" if f == "max" else "amin", "axes": [axis(x0, _arg(b, 1, "dim"))],
                                         "keepdim": bool(_arg(b, 2, "keepdim", False))})  # fmt: skip
        else:
            raise ImportUnsupported(f"{n.name}: {f}(...)[{idx}] (indices) has no card equivalent")
    else:
        raise ImportUnsupported(f"{n.name}: item {idx} of {f!r} has no card equivalent")


def _getitem_tensor(n, base, idx, add, src, axis, rank, shape, vals):
    """x[:, 0], x[..., :k], x[:, None]: a chain of select / slice / unsqueeze nodes."""
    idx = idx if isinstance(idx, tuple) else (idx,)
    if any(isinstance(i, torch.fx.Node) for i in idx):
        raise ImportUnsupported(f"{n.name}: indexing with a tensor has no card equivalent")
    r = rank(base)
    if Ellipsis in idx:
        k = idx.index(Ellipsis)
        fill = r - (len([i for i in idx if i is not None]) - 1)
        idx = idx[:k] + (slice(None),) * fill + idx[k + 1 :]
    cur, ax, out_rank, steps = src(base), 0, r, []
    for i in idx:
        if i is None:
            steps.append(("unsqueeze", {"axis": ax}))
            ax += 1
            out_rank += 1
        elif isinstance(i, list | tuple) and all(isinstance(j, int) for j in i):
            if ax == 0:
                raise ImportUnsupported(f"{n.name}: indexing the batch axis")
            steps.append(("index_axis", {"axis": ax, "index": list(i)}))
            ax += 1
        elif isinstance(i, int):
            if ax == 0:
                raise ImportUnsupported(f"{n.name}: indexing the batch axis")
            steps.append(("select", {"axis": ax, "index": i}))
            out_rank -= 1
        elif isinstance(i, slice):
            if ax == 0 and i != slice(None):
                raise ImportUnsupported(f"{n.name}: slicing the batch axis")
            if i != slice(None):
                steps.append(
                    (
                        "slice",
                        {"axis": ax, "start": i.start or 0, "stop": i.stop, "step": i.step or 1},
                    )
                )
            ax += 1
        else:
            raise ImportUnsupported(f"{n.name}: index {i!r}")
    if not steps:
        return add(n, "as_kind", [cur], {"to": "nd"})
    for k, (op_, args) in enumerate(steps):
        name = n.name if k == len(steps) - 1 else f"{n.name}_{k}"
        cur = add(name, op_, [cur], args)
    return cur


def _functional_layer(n, f, add, src, shape, vals, params=None):
    """F.layer_norm / F.linear / F.conv2d / F.batch_norm / F.group_norm with the module's own
    parameters (get_attr): the card's layer, its weights taken from those names."""
    x0 = n.args[0]

    def name(a):
        if a is None:
            return None
        if not isinstance(a, torch.fx.Node) or a.op != "get_attr":
            raise ImportUnsupported(f"{n.name}: {f} with weights that are computed, not stored")
        return a.target

    w_args = [
        a
        for a in (*n.args[1:3], n.kwargs.get("weight"), n.kwargs.get("bias"))
        if isinstance(a, torch.fx.Node)
    ]
    # stored = the layer's own learned parameters; a buffer (a fixed filter) or a computed tensor is an input
    stored = all(a.op == "get_attr" and (params is None or a.target in params) for a in w_args)
    if not stored and f in (
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    ):  # weights computed from parameters
        w, b = _arg(n, 1, "weight"), _arg(n, 2, "bias")
        ins = [src(x0), src(w)] + ([src(b)] if isinstance(b, torch.fx.Node) else [])
        if f == "linear":
            add(n, "linear_fn", ins)
        else:
            add(n, "conv_fn", ins, {"stride": _arg(n, 3, "stride", 1), "padding": _arg(n, 4, "padding", 0),
                                    "dilation": _arg(n, 5, "dilation", 1), "groups": int(_arg(n, 6, "groups", 1))})  # fmt: skip
        return
    if f == "layer_norm":
        ns = _arg(n, 1, "normalized_shape")
        w, b = name(_arg(n, 2, "weight")), name(_arg(n, 3, "bias"))
        add(n, "layer_norm", [src(x0)], {"axes": len(ns), "eps": float(_arg(n, 4, "eps", 1e-5)), "affine": w is not None,
                                         "bias": b is not None}, ref=[r for r in (w, b) if r])  # fmt: skip
    elif f == "linear":
        w, b = name(_arg(n, 1, "weight")), name(_arg(n, 2, "bias"))
        out = vals[n.name].shape[-1]
        add(
            n,
            "linear",
            [src(x0)],
            {"out": int(out)} | ({} if b else {"bias": False}),
            ref=[r for r in (w, b) if r],
        )
    elif f in ("conv1d", "conv2d", "conv3d"):
        wt = vals[_arg(n, 1, "weight").name]
        w, b = name(_arg(n, 1, "weight")), name(_arg(n, 2, "bias"))
        k = list(wt.shape[2:])
        nd = len(k)

        def ax(v):
            v = tuple(v) if isinstance(v, tuple | list) else (v,) * nd
            return v[0] if len(set(v)) == 1 else list(v)

        pad = _arg(n, 4, "padding", 0)
        if isinstance(pad, str):
            raise ImportUnsupported(f"{n.name}: functional conv with padding={pad!r}")
        g = int(_arg(n, 6, "groups", 1))
        args = {"out": int(wt.shape[0]), "k": ax(k), "s": ax(_arg(n, 3, "stride", 1)), "pad": ax(pad),
                "dilation": ax(_arg(n, 5, "dilation", 1)), "groups": g, "bias": b is not None}  # fmt: skip
        add(n, "conv", [src(x0)], args, ref=[r for r in (w, b) if r])
    elif f == "batch_norm":
        rm, rv = name(_arg(n, 1, "running_mean")), name(_arg(n, 2, "running_var"))
        w, b = name(_arg(n, 3, "weight")), name(_arg(n, 4, "bias"))
        if _arg(n, 5, "training", False):
            raise ImportUnsupported(f"{n.name}: batch_norm in training mode")
        add(n, "batch_norm", [src(x0)], {"eps": float(_arg(n, 7, "eps", 1e-5)), "affine": w is not None, "track": rm is not None},
            ref=[r for r in (w, b, rm, rv) if r])  # fmt: skip
    elif f == "group_norm":
        w, b = name(_arg(n, 2, "weight")), name(_arg(n, 3, "bias"))
        add(n, "group_norm", [src(x0)], {"groups": int(_arg(n, 1, "num_groups")), "eps": float(_arg(n, 4, "eps", 1e-5)),
                                         "affine": w is not None}, ref=[r for r in (w, b) if r])  # fmt: skip


def draft(
    official: dict,
    root,
    kind: str,
    dims,
    name: str,
    classes: int | None = None,
    meta: dict | None = None,
    out_path=None,
) -> str:
    """YAML text of a card drafted from the authors' model (see module docstring)."""
    card = {
        "paper": {"name": name},
        "official": official,
        "input": {"kind": kind, "dims": list(dims)},
        "_path": "draft",
    }
    model = paper._official_class(card, root)(**(official.get("init_args") or {})).eval()
    nodes, head, head_ref = nodes_of(model, classes, example_input(kind, dims))
    if out_path is not None:
        externalize_constants(nodes, out_path)
    if not head:
        raise ImportUnsupported(
            "no final Linear onto `--classes` was found: give --classes, or write the head by hand"
        )
    lines = [
        f"# {name}: drafted by `dlnet paper import` from {official['repo']} @ {official['commit']}",
        f"# ({official['entry']}). A draft is a claim: `dlnet paper verify` compares it with that code.",
        "paper:",
        f"  name: {name}",
    ]
    for k, v in (meta or {}).items():
        if v:
            lines.append(f"  {k}: {v}")
    lines += [
        "  reported: []          # fill in from the paper: {dataset, metric, value, protocol}, {params: N}",
        "",
    ]
    lines += [
        "official:",
        f"  repo: {official['repo']}",
        f"  commit: {official['commit']}",
        f"  entry: {official['entry']}",
    ]
    if official.get("init_args"):
        lines.append(f"  init_args: {_flow(official['init_args'])}")
    for k in ("sys_path", "stub_modules", "input_ops", "output_ops"):
        if official.get(k):
            lines.append(f"  {k}: {_flow(official[k])}")
    lines += [
        "",
        f"input: {{kind: {kind}, dims: {list(dims)}}}",
        "head: cls",
        f"classes: {classes}",
        *([f"head_ref: {head_ref}"] if head_ref else []),
        "",
        "architecture:",
    ]
    lines += [f"  - {_flow(n)}" for n in nodes]
    return "\n".join(lines) + "\n"


SIDECAR_MIN = 4096  # constants larger than this go to <card>.constants.npz instead of the YAML


def externalize_constants(nodes, out_path) -> None:
    """Move the values of large constants (an attention mask, an index table) out of the card into
    <card>.constants.npz next to it; paper.load reads them back."""
    import numpy as np

    out_path = Path(out_path)
    side = out_path.with_name(out_path.stem + ".constants.npz")
    arrays = {}
    for n in nodes:
        if n.get("module") != "static":
            continue
        for j, t in enumerate(n["args"]["tensors"]):
            if "value" in t and np.asarray(t["value"]).size > SIDECAR_MIN:
                key = f"{n['id']}_t{j}"
                arrays[key] = np.asarray(t.pop("value"))
                t["npz"] = {"file": side.name, "key": key}
    if arrays:
        np.savez_compressed(side, **arrays)


def _flow(v) -> str:
    """YAML flow style that reads back as the same value (null, floats with a dot, quoted strings)."""
    import json
    import math
    import re

    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_flow(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_flow(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, float):
        if math.isnan(v):
            return ".nan"
        if math.isinf(v):
            return ".inf" if v > 0 else "-.inf"
        r = repr(v)
        return r.replace("e", ".0e") if "e" in r and "." not in r else r  # YAML 1.1 needs the dot
    if isinstance(v, int):
        return str(v)
    s = str(v)
    plain = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./-]*", s) and s.lower() not in (
        "true",
        "false",
        "null",
        "yes",
        "no",
        "on",
        "off",
        "none",
    )
    return s if plain else json.dumps(s)


def example_input(kind: str, dims, batch: int = 2) -> torch.Tensor:
    """A random input of the card's kind (integer ids for `ids`)."""
    if kind == "ids":
        return torch.stack([torch.randint(0, int(c), (batch,)) for c in dims], 1)
    return torch.randn(batch, *[int(d) for d in dims])


def card_of(
    model: nn.Module, kind: str, dims, classes: int | None = None, name: str = "imported"
) -> dict:
    """The card (as a dict) drafted from an nn.Module instance."""
    nodes, head, head_ref = nodes_of(model, classes, example_input(kind, dims))
    card = {"paper": {"name": name}, "input": {"kind": kind, "dims": list(dims)},
            "architecture": nodes, "head": "cls" if head else "none"}  # fmt: skip
    if classes:
        card["classes"] = classes
    if head_ref:
        card["head_ref"] = head_ref
    return card


def rebuild_check(
    model: nn.Module, kind: str, dims, classes: int | None = None, tol: float = 1e-5
) -> dict:
    """Import `model`, build the card's network, copy the weights, compare outputs (in memory).

    The same test as `dlnet paper verify`, without a repository checkout: level "B" when the
    parameter counts are equal and the outputs agree within tol x max(1, |y|)."""
    from . import zoo

    card = card_of(model, kind, dims, classes)
    arch = paper.to_arch(card, classes or 1)
    mine = zoo.build_model(arch).eval()
    res = {"nodes": len(card["architecture"]), "params_official": sum(p.numel() for p in model.parameters()),
           "params_card": sum(p.numel() for p in mine.parameters())}  # fmt: skip
    copied = copy_weights(model, mine, card)
    res.update(copied)
    if copied["unmatched_official"] or copied["unmatched_card"]:
        return {
            **res,
            "level": "C",
            "max_abs_diff": None,
            "reason": "the two networks do not hold the same tensors",
        }
    x = example_input(kind, dims)
    with torch.no_grad():
        ya, yb = model(x), mine(x)
    if ya.shape != yb.shape:
        return {
            **res,
            "level": "C",
            "max_abs_diff": None,
            "reason": f"output shapes differ: {tuple(ya.shape)} vs {tuple(yb.shape)}",
        }
    diff = float((ya - yb).abs().max())
    ok = (
        diff <= tol * max(1.0, float(ya.abs().max()))
        and res["params_official"] == res["params_card"]
    )
    return {**res, "level": "B" if ok else "C", "max_abs_diff": diff}


def _tensors(model: nn.Module, ref) -> list:
    """The tensors (parameters, then buffers, without counters) behind a layer name or a list of
    parameter / buffer names of the authors' model."""
    out = []
    for r in [ref] if isinstance(ref, str) else ref:
        try:
            sub = model.get_submodule(r)
        except AttributeError:
            sub = None
        if isinstance(sub, nn.Module):
            out += [
                (f"{r}.{k}", v)
                for k, v in sub.state_dict().items()
                if not k.endswith("num_batches_tracked")
            ]
        else:
            owner, _, attr = r.rpartition(".")
            out.append((r, getattr(model.get_submodule(owner) if owner else model, attr)))
    return out


def copy_weights(ref: nn.Module, mine: nn.Module, card: dict | None = None, ignore=()) -> dict:
    """Copy the authors' tensors into the card's network.

    Imported cards name, for every node, the layer of the authors' model it was written from
    (`ref`; `head_ref` for the classifier): each node's tensors are taken from that layer, in order,
    shapes checked. Cards written by hand fall back to the first unused tensor of the same shape,
    in order. Returns the tensors left without a partner on either side."""
    from .zoo.core import Graph

    graph = next((m for m in mine.modules() if isinstance(m, Graph)), None)
    nodes = (card or {}).get("architecture") or []
    if (
        graph is not None
        and nodes
        and any("ref" in n for n in nodes)
        and len(graph.mods) == len(nodes)
    ):
        used, unmatched_card, sd = set(), [], {}
        pairs = [(graph.mods[i], n.get("ref")) for i, n in enumerate(nodes)]
        head = getattr(mine, "head", None)
        if card.get("head_ref") and head is not None:
            pairs.append((head, card["head_ref"]))
        for mod, r in pairs:
            mine_t = [
                (k, v) for k, v in mod.state_dict().items() if not k.endswith("num_batches_tracked")
            ]
            if not mine_t:
                continue
            if r is None:
                unmatched_card += [k for k, _ in mine_t]
                continue
            theirs = _tensors(ref, r)
            rel = {k[len(r) + 1 :]: (k, v) for k, v in theirs} if isinstance(r, str) else {}
            if rel and set(rel) == {
                k for k, _ in mine_t
            }:  # same names (a ported library layer): by name
                theirs = [rel[k] for k, _ in mine_t]
            if [tuple(v.shape) for _, v in theirs] != [tuple(v.shape) for _, v in mine_t]:
                unmatched_card += [
                    f"{r}: {[tuple(v.shape) for _, v in mine_t]} vs {[tuple(v.shape) for _, v in theirs]}"
                ]
                continue
            for (k, v), (rk, rv) in zip(mine_t, theirs, strict=True):
                sd[id(v)] = (v, rv)
                used.add(rk)
        # every learned parameter must have a place in the card; a buffer the forward pass does not
        # read (an index table kept next to the constant actually used) is not a weight
        unmatched_official = [
            k
            for k, _ in ref.named_parameters()
            if k not in used and not k.startswith(tuple(ignore))
        ]
        if not unmatched_card and not unmatched_official:
            with torch.no_grad():
                for v, rv in sd.values():
                    v.copy_(rv)
        return {
            "unmatched_official": unmatched_official[:10],
            "unmatched_card": unmatched_card[:10],
        }
    a = {k: v for k, v in ref.state_dict().items() if not k.endswith("num_batches_tracked")}
    b = {k: v for k, v in mine.state_dict().items() if not k.endswith("num_batches_tracked")}
    free, mapping, unmatched = list(b), {}, []
    for k, v in a.items():
        hit = next((m for m in free if b[m].shape == v.shape), None)
        if hit is None:
            unmatched.append(k)
        else:
            mapping[hit] = v
            free.remove(hit)
    if not unmatched and not free:
        mine.load_state_dict({**mine.state_dict(), **mapping})
    return {"unmatched_official": unmatched[:10], "unmatched_card": free[:10]}


def to_yaml(card: dict, out_path=None) -> str:
    """YAML text of a card dict (as `card_of` returns); with `out_path`, large constants go to the
    sidecar <card>.constants.npz (externalize_constants)."""
    import copy

    card = copy.deepcopy(card)
    if out_path is not None:
        externalize_constants(card["architecture"], out_path)
    lines = ["paper:", f"  name: {_flow(card['paper']['name'])}", ""]
    if card.get("official"):
        lines += ["official:"] + [f"  {k}: {_flow(v)}" for k, v in card["official"].items()] + [""]
    lines.append(f"input: {_flow(card['input'])}")
    for k in ("head", "classes", "head_ref"):
        if card.get(k) is not None:
            lines.append(f"{k}: {_flow(card[k])}")
    lines += ["", "architecture:"] + [f"  - {_flow(n)}" for n in card["architecture"]]
    return "\n".join(lines) + "\n"
