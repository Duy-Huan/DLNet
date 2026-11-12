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

import operator

import torch
from torch import nn

from . import paper


class ImportUnsupported(ValueError):
    """The authors' model holds something the importer cannot write as card nodes."""


ACTS = {nn.ReLU: "relu", nn.ReLU6: "relu6", nn.GELU: "gelu", nn.SiLU: "swish", nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid", nn.Hardswish: "hswish"}  # fmt: skip
FUNC_ACTS = {"relu": "relu", "relu6": "relu6", "gelu": "gelu", "silu": "swish", "tanh": "tanh",
             "sigmoid": "sigmoid", "hardswish": "hswish"}  # fmt: skip


def _one(v, what):
    v = tuple(v) if isinstance(v, tuple | list) else (v,)
    if len(set(v)) != 1:
        raise ImportUnsupported(f"{what} differs per axis ({v}): write this layer by hand")
    return v[0]


def _module(m, name):
    """(card module, args) of a torch.nn layer; None = the layer does nothing at inference."""
    if isinstance(m, nn.Conv1d | nn.Conv2d | nn.Conv3d):
        k, s, d = (
            _one(m.kernel_size, "kernel"),
            _one(m.stride, "stride"),
            _one(m.dilation, "dilation"),
        )
        p = m.padding if isinstance(m.padding, str) else _one(m.padding, "padding")
        if isinstance(p, str) and p != "same":
            p = 0
        if m.padding_mode != "zeros":
            raise ImportUnsupported(f"{name}: padding_mode {m.padding_mode}")
        g = "depthwise" if m.groups == m.in_channels and m.groups > 1 else m.groups
        args = {"out": m.out_channels, "k": k}
        if s != 1:
            args["s"] = s
        if p != "same":
            args["pad"] = "same" if p == d * (k - 1) // 2 and s == 1 else p
        if g != 1:
            args["groups"] = g
        if d != 1:
            args["dilation"] = d
        if m.bias is None:
            args["bias"] = False
        return "conv", args
    if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d):
        if not m.affine or m.eps != 1e-5 or not m.track_running_stats:
            raise ImportUnsupported(f"{name}: batch norm with affine={m.affine}, eps={m.eps}")
        return "norm", {"norm": "batch"}
    if isinstance(m, nn.Linear):
        return "linear", (
            {"out": m.out_features}
            if m.bias is not None
            else {"out": m.out_features, "bias": False}
        )
    if type(m) in ACTS:
        if isinstance(m, nn.GELU) and m.approximate != "none":
            raise ImportUnsupported(f"{name}: GELU(approximate={m.approximate!r})")
        return "act", {"act": ACTS[type(m)]}
    if (
        isinstance(m, nn.Identity)
        or (isinstance(m, nn.Dropout | nn.Dropout2d) and m.p == 0)
        or type(m).__name__ == "DropPath"
    ):
        return None
    if isinstance(m, nn.Dropout):
        return "dropout", {"p": m.p}
    if (
        isinstance(m, nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d)
        and _one(m.output_size, "size") == 1
    ):
        return "gap", {}
    if isinstance(
        m, nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d | nn.AvgPool1d | nn.AvgPool2d | nn.AvgPool3d
    ):
        k = _one(m.kernel_size, "kernel")
        args = {
            "k": k,
            "s": _one(m.stride or k, "stride"),
            "pad": _one(m.padding, "padding"),
            "ceil": bool(m.ceil_mode),
        }
        return ("maxpool" if "Max" in type(m).__name__ else "avgpool"), args
    if isinstance(m, nn.Flatten) and m.start_dim == 1:
        return "flatten", {}
    raise ImportUnsupported(f"{name}: layer {type(m).__name__} has no card equivalent")


class _Tracer(torch.fx.Tracer):
    def is_leaf_module(self, m, qualname):
        if isinstance(m, nn.Sequential | nn.ModuleList | nn.ModuleDict):
            return False  # containers are traced into
        return m.__module__.startswith("torch.nn") or type(m).__name__ == "DropPath"


def nodes_of(model: nn.Module, classes: int | None = None):
    """(card nodes, whether the model's last Linear became the head)."""
    graph = _Tracer().trace(model)
    mods = dict(model.named_modules())
    alias, nodes, pooled = {}, [], set()

    def src(a):
        if not isinstance(a, torch.fx.Node):
            raise ImportUnsupported(
                f"a constant ({a!r}) is combined with a tensor: write this part by hand"
            )
        return alias.get(a.name, a.name)

    def add(n, module, sources, args=None):
        node = {"id": n.name, "module": module}
        prev = nodes[-1]["id"] if nodes else "input"
        if len(sources) > 1:
            node["from"] = sources
        elif sources[0] != prev:
            node["from"] = sources[0]
        if args:
            node["args"] = args
        nodes.append(node)

    for n in graph.nodes:
        if n.op == "placeholder":
            alias[n.name] = "input"
        elif n.op == "output":
            out = src(n.args[0])
        elif n.op == "call_module":
            got = _module(mods[n.target], n.target)
            if got is None or (got[0] == "flatten" and src(n.args[0]) in pooled):
                alias[n.name] = src(n.args[0])
                continue
            add(n, got[0], [src(n.args[0])], got[1])
            if got[0] == "gap":
                pooled.add(n.name)
        else:
            f = n.target if n.op == "call_method" else getattr(n.target, "__name__", str(n.target))
            f = f.strip("_")
            tensors = [a for a in n.args if isinstance(a, torch.fx.Node)]
            if f in ("add", "iadd") or n.target in (operator.add, operator.iadd):
                add(n, "add", [src(a) for a in n.args[:2]])
            elif f in ("mul", "imul") or n.target in (operator.mul, operator.imul):
                add(n, "mul", [src(a) for a in n.args[:2]])
            elif f == "cat":
                if n.kwargs.get("dim", n.args[1] if len(n.args) > 1 else 0) != 1:
                    raise ImportUnsupported(f"{n.name}: torch.cat on an axis other than 1")
                add(n, "concat", [src(a) for a in n.args[0]])
            elif f in FUNC_ACTS:
                add(n, "act", [src(tensors[0])], {"act": FUNC_ACTS[f]})
            elif f == "flatten" and src(tensors[0]) in pooled:
                alias[n.name] = src(tensors[0])
            elif f == "flatten" and (n.args[1:] == (1,) or n.kwargs.get("start_dim") == 1):
                add(n, "flatten", [src(tensors[0])])
            elif f == "mean" and set(
                n.args[1] if len(n.args) > 1 and isinstance(n.args[1], tuple | list) else ()
            ) in ({2, 3}, {-1, -2}):
                add(n, "gap", [src(tensors[0])])
                pooled.add(n.name)
            elif f in ("contiguous", "clone", "dropout") and (f != "dropout" or not model.training):
                alias[n.name] = src(tensors[0])
            else:
                raise ImportUnsupported(f"{n.name}: operation {f!r} has no card equivalent")
    if not nodes or out != nodes[-1]["id"]:
        raise ImportUnsupported("the model's output is not its last layer")
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
            raise ImportUnsupported("the classifier does not read the last feature layer")
        head = True
    return nodes, head


def draft(
    official: dict,
    root,
    kind: str,
    dims,
    name: str,
    classes: int | None = None,
    meta: dict | None = None,
) -> str:
    """YAML text of a card drafted from the authors' model (see module docstring)."""
    card = {
        "paper": {"name": name},
        "official": official,
        "input": {"kind": kind, "dims": list(dims)},
        "_path": "draft",
    }
    model = paper._official_class(card, root)(**(official.get("init_args") or {})).eval()
    nodes, head = nodes_of(model, classes)
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
        "",
        "architecture:",
    ]
    lines += [f"  - {_flow(n)}" for n in nodes]
    return "\n".join(lines) + "\n"


def _flow(v) -> str:
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_flow(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_flow(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
