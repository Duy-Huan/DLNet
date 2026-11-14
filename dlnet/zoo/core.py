"""Typed module registry and architecture specs (torch only, standalone).

Every module is registered with `@op(name, kin, ...)`: the kinds of tensor it accepts, the kind it
returns, a discrete search space for its parameters, functional tags used by the grammar, and the
source it was taken from. A tensor kind is the layout without the batch axis:

    vec (D)      seq (T, D)      map (C, H, W)      ser (C, T)      vol (C, D, H, W)
    graph (N, F+N)   a dense graph: N nodes, F features per node followed by the N x N adjacency row
    ids (c_1, ..., c_F)   F categorical fields, dims = their cardinalities; the tensor is (B, F) integers

An ARCHITECTURE is a plain JSON tree (a "spec"). A node is either a leaf {"op": name, **params}
or a combinator:

    {"op": "seq", "body": [node, ...]}                       sequential composition
    {"op": "res", "body": node, "drop_path": 0.0}            x + body(x), 1x1 projection if shapes differ
    {"op": "par", "branches": [node, ...], "merge": "concat|sum|gate|gated"}  (gated: two vectors, "width")
    {"op": "rep", "n": 3, "body": node, "share": false}      n copies (fresh weights, or one shared)
    {"op": "moe", "experts": [node, ...], "mode": "soft|topk|product", "k": 2}
    {"op": "encdec", "stages": [node, ...], "dec": "unet|unetpp|fpn|deeplabv3|deeplabv3plus|pspnet|linknet|upernet|segformer|fcn|deconv",
     "width": 32, "attention": "none|scse"}

Shapes are inferred statically by building the model on the `meta` device and tracing one fake
batch, so an invalid composition (kind mismatch, a map that shrinks to nothing, heads that do not
divide the width) is rejected in milliseconds, before any training. A head is NOT part of the spec:
`build_model` appends it from the task (`cls`: global pooling + linear, `dense`: 1x1 conv per pixel).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import torch
from torch import nn

KINDS = (
    "vec",
    "seq",
    "map",
    "ser",
    "vol",
    "graph",
    "ids",
    "nd",
)  # nd: any other layout (v1.8, ops_prim)
ND = {"ser": 1, "map": 2, "vol": 3}  # spatial dimensionality of the convolutional kinds
OPS: dict[str, OpDef] = {}
COMBINATORS = ("seq", "res", "par", "rep", "moe", "encdec", "graph")


@dataclass
class OpDef:
    name: str
    kin: tuple[str, ...]
    kout: str | None  # None: same kind as the input
    space: dict
    tags: frozenset
    source: str
    build: callable
    doc: str = ""
    defaults: dict = field(default_factory=dict)


def op(name, kin, kout=None, space=None, tags=(), source=""):
    """Register a builder `fn(dims, kind, **params) -> nn.Module` (dims: input shape, no batch)."""
    kin = (kin,) if isinstance(kin, str) else tuple(kin)

    def deco(fn):
        if name in OPS or name in COMBINATORS:
            raise ValueError(f"duplicate op {name!r}")
        space_ = space or {}
        OPS[name] = OpDef(
            name, kin, kout, space_, frozenset(tags), source, fn, (fn.__doc__ or "").strip(),
            {k: v[0] for k, v in space_.items()},
        )  # fmt: skip
        return fn

    return deco


def ops_with(tag: str, kind: str | None = None) -> list[str]:
    return sorted(n for n, o in OPS.items() if tag in o.tags and (kind is None or kind in o.kin))


# ------------------------------------------------------------------ shape tracing


PROBE = {
    "n": 2,
    "last": 2,
}  # batch of the shape-tracing probe (a graph whose batch axis merges / splits)


def probe(kind: str, dims, n: int = 2, device=None) -> torch.Tensor:
    """A zero tensor of batch n for the kind (integer ids for `ids`, whose dims are cardinalities)."""
    if kind == "ids":
        return torch.zeros(n, len(dims), dtype=torch.long, device=device)
    return torch.zeros(n, *dims, device=device)


def _trace(m: nn.Module, dims, kind: str = "vec") -> tuple[int, ...]:
    p = next(itertools.chain(m.parameters(), m.buffers()), None)
    dev = p.device if p is not None else torch.device("cpu")
    was = m.training
    m.eval()
    try:
        with torch.no_grad():
            y = m(probe(kind, dims, PROBE["n"], dev))
    except (
        RuntimeError,
        IndexError,
        AssertionError,
        TypeError,
        ZeroDivisionError,
        NotImplementedError,
    ) as e:
        raise ValueError(f"cannot apply to shape {tuple(dims)}: {e}") from e
    finally:
        m.train(was)
    if not isinstance(y, torch.Tensor):
        raise ValueError("a leaf module must return one tensor")  # noqa: TRY004
    PROBE["last"] = int(y.shape[0])
    if y.numel() == 0 or min(y.shape[1:], default=1) < 1:
        raise ValueError(f"empty output {tuple(y.shape)} for input {tuple(dims)}")
    return tuple(int(s) for s in y.shape[1:])


def _leaf(node, kind, dims):
    name = node["op"]
    if name not in OPS:
        raise ValueError(f"unknown op {name!r}")
    o = OPS[name]
    if kind not in o.kin:
        raise ValueError(f"{name} takes {o.kin}, got kind {kind}")
    unknown = set(node) - {"op"} - set(o.defaults)
    if unknown:
        raise ValueError(f"{name}: unknown parameter(s) {sorted(unknown)}")
    params = {**o.defaults, **{k: v for k, v in node.items() if k != "op"}}
    kw = {"kind": kind}
    try:
        m = o.build(tuple(dims), **kw, **params)
    except (TypeError, KeyError, RuntimeError, IndexError, ZeroDivisionError, AssertionError) as e:
        raise ValueError(f"{name}: bad parameters {params}: {e}") from e
    out_kind = o.kout or kind
    out_dims = _trace(m, dims, kind)
    if name == "as_kind":
        out_kind = params["to"]
        want = {"vec": 1, "seq": 2, "ser": 2, "map": 3, "vol": 4}.get(out_kind)
        if want is not None and len(out_dims) != want:
            raise ValueError(f"as_kind {out_kind}: a {want}-axis tensor is needed, got {out_dims}")
    elif out_kind == "nd" and len(out_dims) == 1:
        out_kind = "vec"  # a layout op that ends with one axis: a vector
    return m, out_kind, out_dims


# -------------------------------------------------------------------- combinators


class Residual(nn.Module):
    def __init__(self, body, proj=None, drop_path=0.0):
        super().__init__()
        self.body, self.proj, self.p = body, proj, drop_path

    def forward(self, x):
        y = self.body(x)
        if self.training and self.p > 0:  # stochastic depth (Huang et al., arXiv:1603.09382)
            keep = (torch.rand(y.shape[0], *[1] * (y.dim() - 1), device=y.device) >= self.p).to(y)
            y = y * keep / (1 - self.p)
        return (x if self.proj is None else self.proj(x)) + y


def _projection(kind, din, dout):
    if din == dout:
        return None
    if kind not in ("vec", "seq", *ND):
        raise ValueError(
            f"residual projection is not defined for kind {kind}: the body must keep the shape"
        )
    if kind in ("vec",):
        return nn.Linear(din[0], dout[0]) if din != dout else None
    if kind == "seq":
        if din[0] != dout[0]:
            raise ValueError("residual body changed the sequence length")
        return nn.Linear(din[1], dout[1]) if din[1] != dout[1] else None
    if din == dout:
        return None
    nd = ND[kind]
    stride = []
    for a, b in zip(din[1:], dout[1:], strict=True):
        if b > a or a % b:
            raise ValueError(f"residual body changed the spatial size {din[1:]} -> {dout[1:]}")
        stride.append(a // b)
    conv = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[nd - 1]
    return conv(din[0], dout[0], 1, tuple(stride), bias=False)


class GatedFusion(nn.Module):
    """Gated fusion of two vectors: g * A(a) + (1 - g) * B(b), g = sigmoid(W [a, b]) (earlier HSI `Gated`)."""

    def __init__(self, da, db, d):
        super().__init__()
        self.a, self.b, self.g = nn.Linear(da, d), nn.Linear(db, d), nn.Linear(da + db, d)

    def forward(self, a, b):
        g = torch.sigmoid(self.g(torch.cat([a, b], 1)))
        return g * self.a(a) + (1 - g) * self.b(b)


class Par(nn.Module):
    def __init__(self, branches, merge, dim, gated=None):
        super().__init__()
        self.branches, self.merge, self.dim, self.fuse = nn.ModuleList(branches), merge, dim, gated
        if merge == "gate":
            self.w = nn.Parameter(torch.zeros(len(branches)))

    def forward(self, x):
        ys = [b(x) for b in self.branches]
        if self.merge == "gated":
            return self.fuse(*ys)
        if self.merge == "concat":
            return torch.cat(ys, self.dim)
        if self.merge == "sum":
            return sum(ys)
        w = torch.softmax(self.w, 0)  # "gate": learned convex combination of the branches
        return sum(wi * y for wi, y in zip(w, ys, strict=True))


class Shared(nn.Module):
    def __init__(self, body, n):
        super().__init__()
        self.body, self.n = body, n

    def forward(self, x):
        for _ in range(self.n):
            x = self.body(x)
        return x


class MoE(nn.Module):
    """Mixture of experts (paper 2302.11529: learned routing + aggregation of functions).

    soft:    sum_e softmax(g(x))_e * expert_e(x)
    topk:    the same with all but the k largest gate logits set to -inf before the softmax
    product: product of experts for vector outputs: sum_e log_softmax(expert_e(x))
    Method of yamsgithub/modular_deep_learning (soft = their expectation model, topk = top-k model);
    re-implemented for any tensor kind and mixing outputs as they are, not after a softmax.
    """

    def __init__(self, experts, gate_in, mode="soft", k=2):
        super().__init__()
        self.experts, self.mode, self.k = nn.ModuleList(experts), mode, k
        self.gate = None if mode == "product" else nn.Linear(gate_in, len(experts))
        self.gate_probs = None

    def forward(self, x):
        y = torch.stack([e(x) for e in self.experts], 1)  # (B, E, ...)
        if self.mode == "product":
            return torch.log_softmax(y, -1).sum(1)
        if x.dim() == 2:
            g = x
        elif self._seq:
            g = x.mean(1)
        else:
            g = x.mean(tuple(range(2, x.dim())))
        logits = self.gate(g)
        if self.mode == "topk" and self.k < logits.shape[1]:
            keep = torch.topk(logits, self.k, 1).indices
            mask = torch.ones_like(logits, dtype=torch.bool).scatter(1, keep, False)
            logits = logits.masked_fill(mask, float("-inf"))
        p = torch.softmax(logits, 1)
        self.gate_probs = p.detach()
        return (p.view(*p.shape, *[1] * (y.dim() - 2)) * y).sum(1)

    _seq = False


class Graph(nn.Module):
    """A directed acyclic graph of blocks, as drawn in a paper: every node reads the output of one
    earlier node, or merges several. Written by `{"op": "graph", "nodes": [{"id", "from", "op", ...},
    ...], "out": id}` (dlnet/paper.py). Merges (ops_prim.MERGES): add, mul, sub, div, maximum,
    minimum (broadcasting), concat / stack on an axis, matmul, attention (scaled dot-product), mha
    (torch.nn.MultiheadAttention), where. A `param` node has no input: a learned tensor.

    `wiring[i] = (how, sources)`: how is "op" (mods[i] applied to its one source), "merge"
    (mods[i] applied to all its sources) or "param" (mods[i] alone); a source is the index of an
    earlier node, -1 for the input of the graph."""

    def __init__(self, mods, wiring, axis=1, out=-1):
        super().__init__()
        self.mods = nn.ModuleList([nn.Identity() if m is None else m for m in mods])
        self.wiring, self.axis, self.out = [(h, tuple(s)) for h, s in wiring], axis, out

    def forward(self, x):
        ys = []
        for m, (how, src) in zip(self.mods, self.wiring, strict=True):
            xs = [x if i < 0 else ys[i] for i in src]
            if how == "param":
                ys.append(m())
            elif how == "merge":
                ys.append(m(*xs))
            elif how == "concat":  # cards written before v1.8
                ys.append(torch.cat(xs, self.axis))
            elif how == "add":
                ys.append(sum(xs[1:], xs[0]))
            elif how == "mul":
                y = xs[0]
                for t in xs[1:]:
                    y = y * t
                ys.append(y)
            else:
                ys.append(m(xs[0]))
        return ys[self.out]


def _graph(node, kind, dims):
    from .ops_prim import MERGES, merge_module, param_module, static_module

    known = {"input": (-1, kind, tuple(dims))}
    mult = {"input": 1}  # batch of each node / batch of the input (window partition multiplies it)
    full = {}  # param node id -> its whole shape (with its leading broadcast axis)
    mods, wiring = [], []
    prev = "input"
    for i, n in enumerate(node["nodes"]):
        nid = str(n.get("id", f"n{i}"))
        if nid in known:
            raise ValueError(f"graph: node id {nid!r} is used twice")
        body = {k: v for k, v in n.items() if k not in ("id", "from", "share")}
        how = body.get("op")
        if how == "static":  # a computation on the network's own tensors only (ops_prim.StaticExpr)
            m = static_module({k: v for k, v in body.items() if k != "op"})
            try:
                with torch.no_grad():
                    y = m()
            except (RuntimeError, IndexError) as e:
                raise ValueError(f"graph node {nid!r}: static program: {e}") from e
            shape = tuple(int(s) for s in y.shape)
            mods.append(m)
            wiring.append(("param", []))
            known[nid], full[nid], mult[nid] = (i, "param", shape[1:]), shape, 1
            prev = nid
            continue
        if how == "param":  # a learned tensor (or a constant buffer): no input
            shape = tuple(int(s) for s in body["shape"])
            if not shape:
                raise ValueError(
                    f"graph node {nid!r}: a param needs a shape with a leading axis (1 to broadcast)"
                )
            mods.append(
                param_module(
                    shape,
                    body.get("init", "normal"),
                    body.get("learned", True),
                    body.get("dtype", "float32"),
                )
            )
            wiring.append(("param", []))
            known[nid], full[nid], mult[nid] = (i, "param", shape[1:]), shape, 1
            prev = nid
            continue
        src = n.get("from", prev)
        src = [src] if isinstance(src, str | int) else list(src)
        missing = [str(s) for s in src if str(s) not in known]
        if missing:
            raise ValueError(f"graph node {nid!r}: 'from' names {missing}, not defined before it")
        ins = [known[str(s)] for s in src]
        if how in MERGES:
            args = {k: v for k, v in body.items() if k != "op"}
            if (
                how == "concat" and "axis" not in args
            ):  # pre-v1.8 cards: the feature axis of the kind
                args["axis"] = 2 if ins[0][1] == "seq" else 1
            if how not in ("mha", "einsum") and len(ins) < 2:
                raise ValueError(f"graph node {nid!r}: {how} needs >= 2 inputs")
            shapes = [
                full[str(s)] if k == "param" else (2 * mult[str(s)], *d)
                for (_, k, d), s in zip(ins, src, strict=True)
            ]
            try:
                m = merge_module(how, args, [sh[1:] for sh in shapes])
                w = next(itertools.chain(m.parameters(), m.buffers()), None)
                dev = w.device if w is not None else "meta"
                with torch.no_grad():
                    y = m(*[torch.zeros(sh, device=dev) for sh in shapes])
                d2 = tuple(int(v) for v in y.shape[1:])
            except (RuntimeError, ValueError, IndexError, TypeError) as e:
                got = [list(sh[1:]) for sh in shapes]
                need = (
                    " needs identical shapes (or shapes that broadcast)"
                    if how in ("add", "mul", "sub", "div")
                    else ""
                )
                raise ValueError(f"graph node {nid!r}: {how}{need}, got {got}: {e}") from e
            mods.append(m)
            wiring.append(("merge", [i_ for i_, _, _ in ins]))
            real = [x for x in ins if x[1] != "param"] or ins
            k0, d0 = real[0][1], real[0][2]
            same = len({x[1] for x in real}) == 1 and len(d2) == len(
                d0
            )  # e.g. a channel concat of maps
            known[nid] = (i, k0 if same else ("vec" if len(d2) == 1 else "nd"), d2)
            mult[nid] = max(1, int(y.shape[0]) // 2)
            prev = nid
            continue
        if len(ins) != 1:
            raise ValueError(
                f"graph node {nid!r}: {how} takes one input; merge with concat / add first"
            )
        if ins[0][1] == "param":
            raise ValueError(
                f"graph node {nid!r}: a param node is used through a merge (add, mul, concat, ...)"
            )
        mi = mult[str(src[0])]
        PROBE["n"] = 2 * mi
        try:
            m, k2, d2 = _build(body, ins[0][1], ins[0][2])
        except ValueError as e:
            raise ValueError(f"graph node {nid!r}: {e}") from e
        finally:
            PROBE["n"] = 2
        mult[nid] = max(1, PROBE["last"] // 2)
        if (
            n.get("share") is not None
        ):  # the same layer called again: one module, one set of weights
            first = str(n["share"])
            if first not in known or known[first][0] < 0:
                raise ValueError(
                    f"graph node {nid!r}: share names {first!r}, not a node defined before it"
                )
            other = mods[known[first][0]]
            if type(other) is not type(m) or [p.shape for p in other.parameters()] != [
                p.shape for p in m.parameters()
            ]:
                raise ValueError(f"graph node {nid!r}: share {first!r} is a different layer")
            m = other
        mods.append(m)
        wiring.append(("op", [i_ for i_, _, _ in ins]))
        known[nid] = (i, k2, tuple(d2))
        prev = nid
    if not mods:
        raise ValueError("graph needs at least one node")
    out = str(node.get("out", prev))
    if out not in known or out == "input":
        raise ValueError(f"graph: output node {out!r} is not defined")
    if mult.get(out, 1) != 1:
        raise ValueError(
            f"graph: the output's batch axis is still merged with another axis (x{mult[out]})"
        )
    return Graph(mods, wiring, 1, known[out][0]), known[out][1], known[out][2]


def _build(node, kind, dims):
    """-> (module, out_kind, out_dims). Raises ValueError for any invalid composition."""
    if isinstance(node, list):
        node = {"op": "seq", "body": node}
    name = node.get("op")
    if name == "seq":
        mods = []
        for n in node["body"]:
            m, kind, dims = _build(n, kind, dims)
            mods.append(m)
        return nn.Sequential(*mods), kind, dims
    if name == "res":
        body, k2, d2 = _build(node["body"], kind, dims)
        if k2 != kind:
            raise ValueError("residual body must keep the tensor kind")
        return Residual(body, _projection(kind, dims, d2), node.get("drop_path", 0.0)), kind, d2
    if name == "par":
        parts = [_build(b, kind, dims) for b in node["branches"]]
        kinds = {p[1] for p in parts}
        if len(kinds) != 1 or len(parts) < 2:
            raise ValueError("par needs >= 2 branches with the same output kind")
        out_kind, merge = parts[0][1], node.get("merge", "concat")
        ds = [p[2] for p in parts]
        axis = 2 if out_kind == "seq" else 1
        if merge == "concat":
            rest = {tuple(d[: axis - 1] + d[axis:]) for d in ds}
            if len(rest) != 1:
                raise ValueError(f"cannot concatenate shapes {ds}")
            od = list(ds[0])
            od[axis - 1] = sum(d[axis - 1] for d in ds)
            od = tuple(od)
        elif merge == "gated":
            if out_kind != "vec" or len(parts) != 2:
                raise ValueError("gated fusion needs exactly two vector branches")
            od = (int(node["width"]),)
        elif merge in ("sum", "gate"):
            if len(set(ds)) != 1:
                raise ValueError(f"{merge} needs identical branch shapes, got {ds}")
            od = ds[0]
        else:
            raise ValueError(f"unknown merge {merge!r}")
        fuse = GatedFusion(ds[0][0], ds[1][0], od[0]) if merge == "gated" else None
        return Par([p[0] for p in parts], merge, axis, fuse), out_kind, od
    if name == "rep":
        n, share = int(node["n"]), bool(node.get("share", False))
        if n < 1:
            raise ValueError("rep needs n >= 1")
        if share:
            body, k2, d2 = _build(node["body"], kind, dims)
            if (k2, d2) != (kind, tuple(dims)):
                raise ValueError("a shared body must keep the shape")
            return Shared(body, n), kind, d2
        mods = []
        for _ in range(n):
            m, kind, dims = _build(node["body"], kind, dims)
            mods.append(m)
        return nn.Sequential(*mods), kind, dims
    if name == "moe":
        if kind == "ids":
            raise ValueError("the moe gate needs a real-valued input, not integer ids")
        spec = node["experts"]
        spec = [spec["body"]] * int(spec["n"]) if isinstance(spec, dict) else spec
        parts = [_build(e, kind, dims) for e in spec]
        if len(parts) < 2 or len({(p[1], p[2]) for p in parts}) != 1:
            raise ValueError("moe needs >= 2 experts with identical output shapes")
        mode = node.get("mode", "soft")
        if mode not in ("soft", "topk", "product"):
            raise ValueError(f"unknown moe mode {mode!r}")
        if mode == "product" and parts[0][1] != "vec":
            raise ValueError("product of experts needs vector outputs")
        gate_in = dims[0] if kind != "seq" else dims[1]
        m = MoE([p[0] for p in parts], gate_in, mode, int(node.get("k", 2)))
        m._seq = kind == "seq"
        return m, parts[0][1], parts[0][2]
    if name == "graph":
        return _graph(node, kind, dims)
    if name == "encdec":
        from .dense import EncDec

        if kind != "map":
            raise ValueError("encdec works on map inputs")
        stages, feats, k, d = [], [], kind, dims
        for s in node["stages"]:
            m, k, d = _build(s, k, d)
            if k != "map":
                raise ValueError("encoder stages must keep the map kind")
            stages.append(m)
            feats.append(d)
        net = EncDec(
            stages, feats, node.get("dec", "unet"), int(node.get("width", 32)), dims,
            node.get("attention", "none"),
        )  # fmt: skip
        return net, "map", (net.out_ch, *dims[1:])
    return _leaf(node, kind, dims)


# ------------------------------------------------------------------------- public API


def check(spec, kind: str, dims) -> dict:
    """Static validation without allocating weights. Returns {"params", "kind", "dims"}.

    Raises ValueError when the spec cannot be applied to an input of this kind and shape.
    """
    try:
        with torch.device("meta"):
            m, k, d = _build(spec, kind, tuple(dims))
    except ValueError:
        raise
    except (RuntimeError, NotImplementedError, TypeError):  # an op that cannot be built on meta
        m, k, d = _build(spec, kind, tuple(dims))
    return {"params": sum(p.numel() for p in m.parameters()), "kind": k, "dims": d}


class Head(nn.Module):
    """Global pooling + dropout + linear (cls), a 1x1 convolution per pixel (dense), or one output
    layer per ensemble member (ensemble: TabM, arXiv:2410.24210)."""

    def __init__(self, kind, dims, out_dim, mode="cls", dropout=0.0):
        super().__init__()
        self.kind, self.mode = kind, mode
        if mode == "ensemble":
            # tabm.LinearEnsemble: k separate linear layers on the k member representations (k, d).
            # Training: (B, k, out), one loss per member (core.Trainer); eval: the members' mean
            # probability (classification, as log) or mean value (out_dim = 1).
            if kind != "seq":
                raise ValueError("an ensemble head needs (members, features)")
            k, d = dims
            b = d**-0.5
            self.drop = nn.Dropout(dropout)
            self.weight = nn.Parameter(torch.empty(k, d, out_dim).uniform_(-b, b))
            self.bias = nn.Parameter(torch.empty(k, out_dim).uniform_(-b, b))
            self.k = k
            return
        if mode == "dense":
            if kind != "map":
                raise ValueError("dense prediction needs a map feature")
            self.f = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(dims[0], out_dim, 1))
        else:
            feat = dims[1] if kind == "seq" else dims[0]
            self.f = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, out_dim))

    def forward(self, x, size=None):
        if self.mode == "ensemble":
            y = (self.drop(x).transpose(0, 1) @ self.weight).transpose(
                0, 1
            ) + self.bias  # (B, k, out)
            if self.training:
                return y
            if y.shape[-1] == 1:
                return y.mean(1)
            return torch.logsumexp(torch.log_softmax(y, -1), 1) - math.log(self.k)  # log mean prob
        if self.mode == "dense":
            y = self.f(x)
            return nn.functional.interpolate(y, size=size, mode="bilinear", align_corners=False) \
                if size is not None and y.shape[-2:] != size else y  # fmt: skip
        if self.kind == "seq":
            x = x.mean(1)
        elif self.kind != "vec":
            x = x.mean(tuple(range(2, x.dim())))
        return self.f(x)


class ZooNet(nn.Module):
    def __init__(self, body, head, in_dims, kl_weight=0.0):
        super().__init__()
        self.body, self.head, self.size, self.kl_weight = body, head, tuple(in_dims[-2:]), kl_weight
        # members of an ensemble head: in training the output is (B, k, out), one loss per member
        self.ensemble_k = head.k if head.mode == "ensemble" else 0

    def kl_loss(self):
        """Sum of the KL terms stored by variational modules in the last training forward pass (0 if none)."""
        kls = [m.last_kl for m in self.modules() if getattr(m, "last_kl", None) is not None]
        return sum(kls) if kls else None

    def forward(self, x):
        z = self.body(x)
        return self.head(z, x.shape[-2:]) if self.head.mode == "dense" else self.head(z)


def build_model(cfg: dict, in_dim=None, out_dim=None) -> nn.Module:
    """Model of a candidate {"spec": ..., "input": {"kind", "dims"}, "head": "cls|dense", ...}."""
    inp = cfg["input"]
    body, kind, dims = _build(cfg["spec"], inp["kind"], tuple(inp["dims"]))
    if cfg.get("head") == "none":  # the spec already produces the output (a generator)
        return body
    head = Head(
        kind, dims, out_dim or cfg["out_dim"], cfg.get("head", "cls"), cfg.get("dropout", 0.0)
    )
    return ZooNet(body, head, inp["dims"], cfg.get("beta", 1e-3))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


_MERGE_NAMES = (
    "concat",
    "add",
    "mul",
    "sub",
    "div",
    "maximum",
    "minimum",
    "matmul",
    "stack",
    "attention",
    "where",
    "pow2",
    "complex",
    "mha",
    "param",
    "einsum",
    "static",
    "linear_fn",
    "conv_fn",
)


def leaf_ops(spec) -> list[str]:
    """Names of every leaf op in a spec, in order."""
    if isinstance(spec, list):
        return [n for s in spec for n in leaf_ops(s)]
    name = spec["op"]
    if name in ("seq",):
        return leaf_ops(spec["body"])
    if name in ("res", "rep"):
        return leaf_ops(spec["body"])
    if name == "par":
        return [n for b in spec["branches"] for n in leaf_ops(b)] + ["par"]
    if name == "moe":
        e = spec["experts"]
        return (leaf_ops(e["body"]) if isinstance(e, dict) else leaf_ops(e)) + ["moe"]
    if name == "encdec":
        return leaf_ops(spec["stages"]) + [spec.get("dec", "unet")]
    if name == "graph":
        nodes = [{k: v for k, v in n.items() if k not in ("id", "from")} for n in spec["nodes"]]
        return [x for n in nodes if n.get("op") not in _MERGE_NAMES for x in leaf_ops(n)]
    return [name]


def label(spec, n: int = 3) -> str:
    """Short family label of a spec: its most frequent distinct leaf ops (for diversity grouping)."""
    from collections import Counter

    skip = {
        "conv_bn_act",
        "norm",
        "act",
        "dropout",
        "pool",
        "gap",
        "flatten",
        "tokens",
        "linear_act",
    }
    c = Counter(o for o in leaf_ops(spec) if o not in skip) or Counter(leaf_ops(spec))
    return "zoo_" + "_".join(k for k, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


def signature(spec) -> str:
    import hashlib
    import json

    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def describe() -> str:
    """Markdown table of the whole registry."""
    rows = ["| op | in kinds | out | tags | source |", "|---|---|---|---|---|"]
    for n in sorted(OPS):
        o = OPS[n]
        rows.append(
            f"| `{n}` | {','.join(o.kin)} | {o.kout or 'same'} | {','.join(sorted(o.tags))} | {o.source} |"
        )
    return "\n".join(rows)


__all__ = ["OPS", "build_model", "check", "label", "math", "op"]
