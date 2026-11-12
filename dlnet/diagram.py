"""Paper-style block diagram of a generated network, as a self-contained SVG (no dependency).

    svg = diagram.svg(arch)            # arch = config.yaml["arch"]: spec, input kind / dims, head
    diagram.write(arch, "architecture.svg")

The diagram is drawn from the spec, the same tree `model.py` builds, top to bottom:

- every block is a coloured box (colour = kind of block: convolution, attention / transformer,
  frequency domain, recurrent / state space, MLP, pooling / reshape, normalisation) with its main
  hyper-parameters, and the shape of its output on the right (C x H x W, tokens x d, ...);
- a residual block is a dashed frame with the skip connection drawn around it into a (+) node;
- parallel branches are drawn side by side and merged (concat, sum, gate);
- a block repeated n times is drawn once with "x n" (shared weights say so);
- an encoder-decoder is drawn as its encoder stages and the decoder with its skip inputs;
- the input and the head (pooling + linear, or the per-pixel classifier) close the figure.

Shapes come from tracing the spec (`dlnet.zoo.check`), so they are the real ones.
"""

from __future__ import annotations

import html
from pathlib import Path

from .zoo import core

FONT = "font-family='Helvetica, Arial, sans-serif'"
CW, PAD, GAP, ROW = (
    6.6,
    10,
    18,
    34,
)  # char width (11 px font), box padding, vertical gap, box height

COLOURS = {  # category: (fill, stroke)
    "conv": ("#dbeafe", "#2563eb"),
    "attn": ("#fde7d4", "#ea580c"),
    "freq": ("#ede9fe", "#7c3aed"),
    "rnn": ("#dcfce7", "#16a34a"),
    "mlp": ("#ccfbf1", "#0d9488"),
    "pool": ("#fef9c3", "#ca8a04"),
    "norm": ("#f1f5f9", "#64748b"),
    "io": ("#ffffff", "#111827"),
    "head": ("#fee2e2", "#dc2626"),
    "graph": ("#fce7f3", "#db2777"),
}
LEGEND = [("conv", "convolution"), ("attn", "attention / transformer"), ("freq", "frequency (FFT)"),
          ("rnn", "recurrent / state space"), ("mlp", "MLP / linear"), ("pool", "pooling / reshape"),
          ("norm", "norm / activation"), ("graph", "graph / interaction")]  # fmt: skip

FREQ = (
    "fft",
    "fourier",
    "gfnet",
    "afno",
    "fno",
    "ffc",
    "fca",
    "fnet",
    "spectral",
    "spec_",
    "wavelet",
    "wtconv",
)
ATTN = ("attn", "attention", "transformer", "vit", "swin", "mqa", "msa", "former", "mixer", "token", "patch",
        "pos", "cls", "diff_", "rope", "ssftt", "efficientvit", "lsnet", "msca")  # fmt: skip
RNN = ("mamba", "lstm", "gru", "rnn", "deltanet", "ssm", "scan", "fahm", "xlstm")
GRAPH = ("graph", "gcn", "gat", "gin", "gps", "sage", "polynormer", "sgformer", "fm_", "cross", "dcn",
         "finalmlp", "wukong", "embed", "interaction")  # fmt: skip
POOL = (
    "pool",
    "gap",
    "flatten",
    "reshape",
    "to_",
    "vec_to",
    "upsample",
    "pixel_shuffle",
    "down",
    "stem",
)
NORM = ("norm", "act", "dropout", "dyt", "bn", "ln")
MLP = ("linear", "mlp", "ffn", "glu", "kan", "realmlp", "tabm", "plr", "tt_", "tr_", "lowrank")

PRETTY = {
    "conv_bn_act": "Conv", "dw_sep": "DW-sep conv", "gap": "Global pool", "pool": "Pool",
    "linear_act": "Linear", "mlp": "MLP", "res": "Residual", "patchify": "Patch embed", "pos": "Pos. embed",
    "transformer": "Transformer block", "gfnet_block": "GFNet block", "afno_block": "AFNO block",
    "spectral_gf": "Spectral global filter", "spectral_rfft": "Spectral rFFT", "mbconv": "MBConv",
    "basic_block": "Basic block", "bottleneck": "Bottleneck", "dense_block": "Dense block",
    "unet_res_block": "Res block", "mambahsi_block": "MambaHSI block", "se": "SE", "cbam": "CBAM",
}  # fmt: skip
NORMS = {"batch": "BN", "group": "GN", "layer": "LN", "instance": "IN", "none": ""}


CONV = (
    "conv",
    "block",
    "res",
    "dense",
    "bottleneck",
    "inception",
    "fire",
    "ghost",
    "star",
    "uib",
    "nin",
)
ACTS = {"relu": "ReLU", "gelu": "GELU", "swish": "SiLU", "silu": "SiLU", "relu6": "ReLU6", "tanh": "Tanh",
        "sigmoid": "Sigmoid", "leaky_relu": "LeakyReLU", "elu": "ELU", "mish": "Mish", "hardswish": "HardSwish"}  # fmt: skip


def category(name: str) -> str:
    n = name.lower()
    for cat, keys in (("freq", FREQ), ("rnn", RNN), ("attn", ATTN), ("graph", GRAPH), ("pool", POOL),
                      ("conv", CONV), ("mlp", MLP), ("norm", NORM)):  # fmt: skip
        if any(k in n for k in keys):
            return cat
    return "conv"


def label(node: dict) -> tuple[str, str]:
    """(title, detail) of a leaf block, e.g. ("Conv 3x3, 64", "BN, ReLU, stride 2")."""
    name, p = node["op"], {k: v for k, v in node.items() if k != "op"}
    title = PRETTY.get(name, name.replace("_", " ").capitalize())
    detail = []
    if name == "conv_bn_act":
        k, out = p.pop("k", 3), p.pop("out", None)
        title = f"Conv {k}x{k}" + (f", {out}" if out else "")
        norm, act = NORMS.get(p.pop("norm", "batch"), "?"), p.pop("act", "relu")
        detail = [x for x in (norm, "" if act == "identity" else ACTS.get(act, act)) if x]
        if p.get("s", 1) != 1:
            detail.append(f"stride {p['s']}")
        p.pop("s", None)
    elif name in ("pool", "gap"):
        mode = p.pop("mode", "avg")
        title = f"{'Global ' if name == 'gap' else ''}{mode} pool" + (
            f" {p.pop('k')}x" if "k" in p else ""
        )
        p.pop("s", None)
    else:
        if "out" in p:
            title += f", {p.pop('out')}"
        if "dim" in p:
            title += f", d={p.pop('dim')}"
    if "act" in p:
        a = p.pop("act")
        if a != "identity":
            detail.append(ACTS.get(a, a))
    for k, v in p.items():
        if k in ("dropout", "drop_path") and not v:
            continue
        if len(detail) < 4:
            detail.append(f"{k}={v}")
    return title, ", ".join(detail)


def shape_text(kind, dims) -> str:
    d = [str(x) for x in dims]
    if kind == "vec":
        return d[0]
    if kind == "seq":
        return f"{d[0]} tok x {d[1]}"
    if kind == "graph":
        return f"{d[0]} nodes x {dims[1] - dims[0]}"
    return " x ".join(d)


# ------------------------------------------------------------------------------- layout


class Box:
    """A laid-out piece of the diagram. `core` is the width of its main column (centred on the
    flow of arrows); `w` adds what sits to the right of it (skip lanes, shape labels)."""

    def __init__(self, w, h, draw, out="", core=None):
        self.w, self.h, self.draw, self.out = w, h, draw, out
        self.core = w if core is None else core


def _text(x, y, s, size=11, weight="normal", anchor="middle", colour="#111827"):
    return (f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' font-weight='{weight}' "
            f"text-anchor='{anchor}' fill='{colour}'>{html.escape(str(s))}</text>")  # fmt: skip


def _arrow(x1, y1, x2, y2, dash=False):
    d = " stroke-dasharray='4 3'" if dash else ""
    return (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='#374151' "
            f"stroke-width='1.3' marker-end='url(#arr)'{d}/>")  # fmt: skip


def block(title, detail, cat, out="") -> Box:
    w = max(len(title) * CW * 1.05, len(detail) * CW * 0.85) + 2 * PAD
    w = max(w, 120)
    h = ROW + (12 if detail else 0)
    fill, stroke = COLOURS[cat]

    def draw(x, y):
        s = [(f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h}' rx='6' fill='{fill}' "
              f"stroke='{stroke}' stroke-width='1.4'/>"),
             _text(x + w / 2, y + 20, title, 12, "bold")]  # fmt: skip
        if detail:
            s.append(_text(x + w / 2, y + 35, detail, 10, colour="#374151"))
        return s

    return Box(w, h, draw, out)


def _tag_w(b: Box) -> float:
    return len(b.out) * CW * 0.9 + 10 if b.out else 0


def stack(boxes: list[Box], shapes=True) -> Box:
    """Boxes top to bottom on one centre line, joined by arrows; each output shape on its right."""
    if not boxes:
        return Box(0, 0, lambda x, y: [])
    left = max(b.core / 2 for b in boxes)
    right = max(b.w - b.core / 2 + (_tag_w(b) if shapes else 0) for b in boxes)
    h = sum(b.h for b in boxes) + GAP * (len(boxes) - 1)

    def draw(x, y):
        s, cy, cx = [], y, x + left
        for i, b in enumerate(boxes):
            s += b.draw(cx - b.core / 2, cy)
            if shapes and b.out:
                s.append(_text(cx - b.core / 2 + b.w + 8, cy + b.h / 2 + 4, b.out, 10, anchor="start",
                               colour="#6b7280"))  # fmt: skip
            if i < len(boxes) - 1:
                s.append(_arrow(cx, cy + b.h, cx, cy + b.h + GAP - 2))
            cy += b.h + GAP
        return s

    return Box(left + right, h, draw, boxes[-1].out, core=2 * left)


def frame(inner: Box, caption: str, skip=False, dashed=True, merge="+") -> Box:
    """A frame around `inner` (caption top left); with `skip`, the residual path around it into a
    merge node below."""
    m, lane = 12, 26 if skip else 0
    extra = 34 if skip else 0
    cap_w = len(caption) * CW * 0.9 + 16
    fw = max(inner.w + 2 * m, cap_w)  # frame width
    h = inner.h + 2 * m + 16 + extra
    off = m + (fw - inner.w - 2 * m) / 2 * 0  # inner drawn at the left margin
    core_c = off + inner.core / 2  # centre line, from the frame's left edge

    def draw(x, y):
        dash = " stroke-dasharray='5 4'" if dashed else ""
        s = [(f"<rect x='{x:.1f}' y='{y:.1f}' width='{fw:.1f}' height='{h - extra:.1f}' rx='8' "
              f"fill='none' stroke='#9ca3af' stroke-width='1.2'{dash}/>"),
             _text(x + 8, y + 13, caption, 10, "bold", "start", "#4b5563")]  # fmt: skip
        s += inner.draw(x + off, y + m + 16)
        if skip:
            cx, by, ny, rx = x + core_c, y + h - extra, y + h - 14, x + fw + lane - 6
            s.append(_arrow(cx, by, cx, ny - 10))
            s.append(
                f"<circle cx='{cx:.1f}' cy='{ny:.1f}' r='10' fill='#fff' stroke='#374151' stroke-width='1.3'/>"
            )
            s.append(_text(cx, ny + 4.5, merge, 14, "bold"))
            s.append(f"<path d='M {cx:.1f} {y - 6:.1f} L {rx:.1f} {y - 6:.1f} L {rx:.1f} {ny:.1f} "
                     f"L {cx + 12:.1f} {ny:.1f}' fill='none' stroke='#374151' stroke-width='1.3' "
                     f"marker-end='url(#arr)'/>")  # fmt: skip
        return s

    # the centre line must be the box centre for stack(): pad the core symmetrically
    core = 2 * core_c
    return Box(max(fw + lane, core), h, draw, inner.out, core=core)


def side(boxes: list[Box], merge: str, out: str) -> Box:
    """Branches side by side, fanned out from one point and merged into one node."""
    gap, top = 22, 16
    w = sum(b.w for b in boxes) + gap * (len(boxes) - 1)
    h = max(b.h for b in boxes) + top + 44

    def draw(x, y):
        s, cx, bx = [], x + w / 2, x
        for b in boxes:
            mid = bx + b.core / 2
            s.append(f"<path d='M {cx:.1f} {y:.1f} L {mid:.1f} {y + top - 2:.1f}' stroke='#374151' "
                     f"stroke-width='1.2' fill='none' marker-end='url(#arr)'/>")  # fmt: skip
            s += b.draw(bx, y + top)
            s.append(_arrow(mid, y + top + b.h, cx, y + h - 26))
            bx += b.w + gap
        node_w = max(len(merge) * CW + 18, 30)
        s.append(f"<rect x='{cx - node_w / 2:.1f}' y='{y + h - 24:.1f}' width='{node_w:.1f}' height='22' "
                 f"rx='11' fill='#fff' stroke='#374151' stroke-width='1.3'/>")  # fmt: skip
        s.append(_text(cx, y + h - 9, merge, 11, "bold"))
        return s

    return Box(w, h, draw, out)


# ------------------------------------------------------------------------------ the tree


def _out(spec, kind, dims):
    info = core.check(spec, kind, dims)
    return info["kind"], tuple(info["dims"])


def layout(node, kind, dims) -> tuple[Box, str, tuple]:
    """Box for `node` applied to (kind, dims); returns (box, out kind, out dims)."""
    if isinstance(node, list):
        node = {"op": "seq", "body": node}
    name = node.get("op")
    if name == "seq":
        boxes = []
        for n in node["body"]:
            b, kind, dims = layout(n, kind, dims)
            boxes += b.parts if hasattr(b, "parts") else [b]
        b = stack(boxes)
        b.parts = boxes  # nested sequences are flattened into their parent
        return b, kind, dims
    if name == "res":
        inner, k2, d2 = layout(node["body"], kind, dims)
        cap = "residual" + (f", drop path {node['drop_path']}" if node.get("drop_path") else "")
        b = frame(inner, cap, skip=True)
        b.out = shape_text(k2, d2)
        return b, k2, d2
    if name == "par":
        parts = [layout(br, kind, dims) for br in node["branches"]]
        k2, d2 = _out(node, kind, dims)
        merge = {"concat": "concat", "sum": "+", "gate": "gate", "gated": "gated fusion"}.get(
            node.get("merge", "concat"), node.get("merge", "concat")
        )
        return side([p[0] for p in parts], merge, shape_text(k2, d2)), k2, d2
    if name == "rep":
        inner, k2, d2 = layout(node["body"], kind, dims)
        n, share = int(node["n"]), bool(node.get("share", False))
        if n > 1 and not share:  # each copy maps the previous output: trace them all for the shape
            k2, d2 = _out(node, kind, dims)
        b = frame(inner, f"x {n}" + (" (shared weights)" if share else ""), dashed=False)
        b.out = shape_text(k2, d2)
        return b, k2, d2
    if name == "moe":
        experts = node["experts"]
        experts = [experts["body"]] * int(experts["n"]) if isinstance(experts, dict) else experts
        shown = [layout(e, kind, dims)[0] for e in experts[:3]]
        if len(experts) > 3:
            shown.append(block(f"... {len(experts) - 3} more", "", "norm"))
        k2, d2 = _out(node, kind, dims)
        mode = node.get("mode", "soft")
        merge = f"gate ({mode}{', top-' + str(node.get('k', 2)) if mode == 'topk' else ''})"
        return frame(side(shown, merge, shape_text(k2, d2)), "mixture of experts"), k2, d2
    if name == "encdec":
        return _encdec(node, kind, dims)
    if name == "graph":
        return _graph(node, kind, dims)
    k2, d2 = _out(node, kind, dims)
    title, detail = label(node)
    return block(title, detail, category(name), shape_text(k2, d2)), k2, d2


def _graph(node, kind, dims):
    """Nodes top to bottom in the order of the card; an edge from the node just above is the
    straight arrow, every other edge (skip, branch) runs down a lane on the left."""
    nodes = node["nodes"]
    ids = [str(n.get("id", f"n{i}")) for i, n in enumerate(nodes)]
    shape = {"input": (kind, tuple(dims))}
    boxes, srcs, prev = [], [], "input"
    for i, n in enumerate(nodes):
        src = n.get("from", prev)
        src = [str(src)] if isinstance(src, str | int) else [str(x) for x in src]
        k2, d2 = _out({"op": "graph", "nodes": nodes[: i + 1]}, kind, dims)
        body = {k: v for k, v in n.items() if k not in ("id", "from")}
        if body.get("op") in ("concat", "add", "mul"):
            sign = {"concat": "concat", "add": "+  add", "mul": "x  multiply"}[body["op"]]
            b = block(sign, "", "norm", shape_text(k2, d2))
        else:
            b = layout(body, *shape[src[0]])[0]
            if body.get("op") == "graph":  # a block of the card: framed, named by the node
                b = frame(b, ids[i])
            b.out = "" if body.get("op") in ("graph", "rep") else shape_text(k2, d2)
        boxes.append(b)
        srcs.append(src)
        shape[ids[i]] = (k2, d2)
        prev = ids[i]
    extra = [(s, i) for i, ss in enumerate(srcs) for s in ss if s != (ids[i - 1] if i else "input")]
    lane = 14 * len(extra) + (12 if extra else 0)
    col = stack(boxes)
    tops, y = {}, 0.0
    for i, b in enumerate(boxes):
        tops[ids[i]] = (y, b)
        y += b.h + GAP

    def draw(x, y0):
        s = col.draw(x + lane, y0)
        cx = x + lane + col.core / 2
        for i, ss in enumerate(
            srcs
        ):  # the straight arrow above a node that does not use it: cover it
            above = ids[i - 1] if i else "input"
            if i and above not in ss:
                ty = y0 + tops[ids[i]][0]
                s.append(
                    f"<rect x='{cx - 9:.1f}' y='{ty - GAP + 1:.1f}' width='18' height='{GAP - 2}' fill='#ffffff'/>"
                )
        for j, (src, i) in enumerate(extra):
            ty, tb = tops[ids[i]]
            lx = x + lane - 12 - 14 * j
            if src == "input":
                sy, sx = y0 - GAP / 2, cx
            else:
                st, sb = tops[src]
                sy, sx = y0 + st + sb.h / 2, cx - sb.core / 2
            ey, ex = y0 + ty + tb.h / 2, cx - tb.core / 2
            s.append(f"<path d='M {sx:.1f} {sy:.1f} L {lx:.1f} {sy:.1f} L {lx:.1f} {ey:.1f} L {ex - 2:.1f} {ey:.1f}' "
                     f"fill='none' stroke='#374151' stroke-width='1.3' marker-end='url(#arr)'/>")  # fmt: skip
        return s

    out_id = str(node.get("out", ids[-1]))
    k2, d2 = shape[out_id]
    return Box(col.w + lane, col.h, draw, "", core=col.core + 2 * lane), k2, d2


def _encdec(node, kind, dims):
    """Encoder stages on the left, the decoder on the right, skip connections between them."""
    stages, k, d = [], kind, dims
    for i, s in enumerate(node["stages"]):
        b, k, d = layout(s, k, d)
        stages.append(frame(b, f"encoder stage {i + 1} -> {shape_text(k, d)}", dashed=True))
    enc = stack(stages, shapes=False)
    dec_name = node.get("dec", "unet")
    att = node.get("attention", "none")
    dec = block(f"{dec_name} decoder", f"width {node.get('width', 32)}" + ("" if att == "none" else f", {att}"),
                "conv", f"{node.get('width', 32)} x {' x '.join(map(str, dims[1:]))}")  # fmt: skip
    k2, d2 = _out(node, kind, dims)
    gap = 70
    w = enc.w + gap + dec.w
    h = max(enc.h, dec.h)

    def draw(x, y):
        s = enc.draw(x, y)
        dx, dy = x + enc.w + gap, y + (h - dec.h) / 2
        s += dec.draw(dx, dy)
        cy = y
        for st in stages:  # every stage output feeds the decoder (skip connections)
            sy = cy + st.h / 2
            s.append(f"<path d='M {x + st.w:.1f} {sy:.1f} C {dx - 30:.1f} {sy:.1f} "
                     f"{dx - 30:.1f} {dy + dec.h / 2:.1f} {dx:.1f} {dy + dec.h / 2:.1f}' fill='none' "
                     f"stroke='#6b7280' stroke-width='1.1' stroke-dasharray='4 3' marker-end='url(#arr)'/>")  # fmt: skip
            cy += st.h + GAP
        s.append(_text(dx + dec.w / 2, dy + dec.h + 16, "skip connections from every stage", 10,
                       colour="#6b7280"))  # fmt: skip
        ex = x + enc.core / 2  # the decoder output joins the main line below the encoder
        s.append(f"<path d='M {dx + dec.w / 2:.1f} {dy + dec.h + 22:.1f} L {dx + dec.w / 2:.1f} {y + h + 10:.1f} "
                 f"L {ex + 8:.1f} {y + h + 10:.1f}' fill='none' stroke='#374151' stroke-width='1.3' "
                 f"marker-end='url(#arr)'/>")  # fmt: skip
        return s

    return Box(w, h + 20, draw, shape_text(k2, d2), core=enc.core), k2, d2


def _head(arch, kind, dims) -> Box:
    mode, out = arch.get("head", "cls"), arch.get("out_dim")
    drop = arch.get("dropout", 0.0)
    if mode == "dense":
        return block(f"1x1 conv -> {out} classes", "per pixel, resized to the input", "head",
                     f"{out} x {' x '.join(map(str, arch['input']['dims'][1:]))}")  # fmt: skip
    if mode == "ensemble":
        return block(
            f"{dims[0]} linear heads -> {out}",
            "ensemble members, mean of predictions",
            "head",
            str(out),
        )
    if mode == "none":
        return block("output", "", "head", shape_text(kind, dims))
    pool = (
        ""
        if kind == "vec"
        else ("mean over tokens + " if kind == "seq" else "global average pool + ")
    )
    return block(f"{pool}Linear -> {out}", f"dropout {drop}" if drop else "", "head", str(out))


def svg(arch: dict, title: str | None = None) -> str:
    """SVG text of the block diagram of `arch` (config.yaml["arch"] or a zoo candidate)."""
    kind, dims = arch["input"]["kind"], tuple(arch["input"]["dims"])
    body, k2, d2 = layout(arch["spec"], kind, dims)
    inp = block("Input", shape_text(kind, dims), "io", "")
    parts = [inp, *(body.parts if hasattr(body, "parts") else [body]), _head(arch, k2, d2)]
    fig = stack(parts)
    params = arch.get("params")
    head_txt = title or f"{arch.get('family', 'network')}"
    sub = f"{params:,} parameters" if params else ""
    legend_h = 22 * ((len(LEGEND) + 1) // 2) + 10
    W = max(fig.w + 2 * 24, 470)
    H = fig.h + 70 + legend_h
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W:.0f}' height='{H:.0f}' viewBox='0 0 {W:.0f} {H:.0f}'>",
         ("<defs><marker id='arr' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
          "orient='auto-start-reverse'><path d='M 0 0 L 10 5 L 0 10 z' fill='#374151'/></marker></defs>"),
         "<rect width='100%' height='100%' fill='#ffffff'/>",
         _text(W / 2, 22, head_txt, 15, "bold"), _text(W / 2, 40, sub, 11, colour="#4b5563")]  # fmt: skip
    s += fig.draw((W - fig.w) / 2, 56)
    ly = 56 + fig.h + 18
    for i, (cat, name) in enumerate(LEGEND):
        lx = 24 + (i % 2) * (W / 2 - 10)
        yy = ly + (i // 2) * 22
        fill, stroke = COLOURS[cat]
        s.append(
            f"<rect x='{lx:.1f}' y='{yy:.1f}' width='16' height='12' rx='3' fill='{fill}' stroke='{stroke}'/>"
        )
        s.append(_text(lx + 22, yy + 10, name, 10, anchor="start", colour="#374151"))
    s.append("</svg>")
    return "\n".join(s)


def write(arch: dict, path, title: str | None = None) -> Path:
    path = Path(path)
    path.write_text(svg(arch, title), encoding="utf-8")
    return path
