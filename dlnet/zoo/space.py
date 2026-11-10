"""Grammar-based sampling, mutation and crossover of architecture specs (torch only).

This is the search space of the platform. Everything is driven by the registry: a new op with the
right tags and kinds is sampled, swapped in and inserted without touching this file.

  sample       random architecture for an input kind and shape (seeded, validated by `core.check`)
  mutate       one of: param, swap, insert, delete, wrap_res, repeat, widen, to_moe
  crossover    child = prefix of one parent + suffix of another (stage-level), validated
  population   presets + random samples as candidates for `search.run_search`
  offspring    children of the best candidates (mutation and crossover)

Every candidate is checked statically (`core.check`: kinds, shapes, parameter budget) before it is
returned, so a generation never wastes training time on an invalid network.
"""

from __future__ import annotations

import copy

import numpy as np

from . import core
from .core import OPS, ops_with
from .presets import N, S, n_stages, presets

WIDTH_KEYS = ("out", "dim", "hidden", "width", "filters")
MUTATIONS = ("param", "swap", "insert", "delete", "wrap_res", "repeat", "widen", "to_moe")
MIN_PARAMS = 1_000


def _pick(rng, xs):
    return xs[int(rng.integers(len(xs)))]


def rand_params(name, rng, **fixed):
    o = OPS[name]
    p = {k: _pick(rng, v) for k, v in o.space.items()}
    p.update({k: v for k, v in fixed.items() if k in o.space})
    return {"op": name, **p}


def _sets_width(name):
    return any(k in OPS[name].space for k in WIDTH_KEYS)


def sample(kind, dims, rng, head="cls", width=None):
    """Random body spec for an input of this kind and shape (not yet validated)."""
    w = width or int(_pick(rng, [8, 16, 32]))
    if kind == "vec":
        return _sample_vec(dims, rng, w)
    if kind == "seq":
        return _sample_seq(dims, rng, w)
    if head == "dense" and kind == "map":
        return _sample_dense(dims, rng, w)
    return _sample_conv(kind, dims, rng, w)


def _stage(kind, rng, width, first, extra_ok):
    """One resolution level: a block that sets the width (strided if not first), extra blocks, attention."""
    setters = [
        n
        for n in ops_with("block", kind)
        if "out" in OPS[n].space and "s" in OPS[n].space and n != "shuffle_unit"
    ]
    keep = [n for n in ops_with("block", kind) if not _sets_width(n) and n not in ("window_attn",)]
    blocks = [rand_params(_pick(rng, setters), rng, out=width, s=1 if first else 2)]
    for _ in range(int(rng.integers(0, 3))):
        if extra_ok and rng.random() < 0.5:
            blocks.append(rand_params(_pick(rng, keep), rng))
        else:
            blocks.append(rand_params(_pick(rng, setters), rng, out=width, s=1))
    if rng.random() < 0.3:
        attn = [n for n in ops_with("attn", kind) if n not in ("window_attn",)]
        blocks.append(rand_params(_pick(rng, attn), rng))
    return S(*blocks)


def _sample_conv(kind, dims, rng, w):
    n = n_stages(dims, cap=3)
    if kind == "ser" and rng.random() < 0.35:  # sequence model on the time axis
        seq = [N("conv_bn_act", out=max(w, 16), k=1, s=1, act="identity", norm="none"), N("tokens")]
        return S(*seq, *_sample_seq((dims[1], max(w, 16)), rng, max(w, 16))["body"])
    if kind == "map" and rng.random() < 0.2:  # patch-token transformer
        for patch in (4, 2, 1):
            if all(d % patch == 0 for d in dims[1:]):
                dim = int(_pick(rng, [32, 64]))
                body = _sample_seq((int(np.prod(dims[1:]) // patch**2), dim), rng, dim)["body"]
                return S(N("patchify", dim=dim, patch=patch), *body)
    stem = rand_params("conv_bn_act", rng, out=w, k=_pick(rng, [3, 5]), s=1)
    stages = [_stage(kind, rng, w * 2**i, i == 0, True) for i in range(n)]
    return S(stem, *stages)


def _sample_seq(dims, rng, w):
    d = dims[1]
    pos = [rand_params("pos", rng)] if rng.random() < 0.7 else []
    blocks = ops_with("block", "seq")
    body = []
    for _ in range(int(rng.integers(1, 4))):
        name = _pick(rng, blocks)
        if name in ("transformer_block", "rnn", "mixer_block", "mamba_block", "attn_seq", "ffn"):
            node = rand_params(name, rng)
        else:
            node = rand_params(name, rng)
        body.append(node)
    return S(*pos, *body) if d else S(*body)


def _sample_vec(dims, rng, w):
    if rng.random() < 0.2 and dims[0] <= 128:  # tokens per feature + transformer
        body = _sample_seq((dims[0] + 1, 2 * w), rng, 2 * w)["body"]
        body = [b for b in body if b["op"] != "rnn"] or [rand_params("transformer_block", rng)]
        return S(N("feature_tokens", dim=2 * w), N("cls_token"), *body)
    pre = []
    if rng.random() < 0.25 and dims[0] <= 128:
        pre.append(rand_params(_pick(rng, ["fm_interaction", "cross_net"]), rng))
    layers = []
    for _ in range(int(rng.integers(1, 4))):
        name = _pick(
            rng,
            [
                n
                for n in ops_with("block", "vec")
                if n not in ("fm_interaction", "cross_net", "ffn")
            ],
        )
        node = rand_params(name, rng, out=4 * w, width=4 * w)
        if rng.random() < 0.3:
            node = {"op": "moe", "experts": {"n": int(_pick(rng, [2, 3, 4])), "body": node},
                    "mode": _pick(rng, ["soft", "topk"]), "k": 2}  # fmt: skip
        layers.append(node)
    return S(*pre, *layers)


def _sample_dense(dims, rng, w):
    n = max(2, n_stages(dims, cap=3))
    stages = [
        S(
            N("conv_bn_act", out=w, k=3, s=1, act="relu", norm="batch"),
            _stage("map", rng, w, True, True),
        )
    ]
    stages += [_stage("map", rng, w * 2**i, False, True) for i in range(1, n)]
    return {
        "op": "encdec",
        "stages": stages,
        "dec": _pick(rng, ["unet", "fcn", "fpn", "aspp", "deconv"]),
        "width": w,
    }


# ----------------------------------------------------------------------- tree helpers


def children(node):
    """(key, child) pairs: child is a node dict or a list of nodes."""
    if isinstance(node, list):
        return [(i, c) for i, c in enumerate(node)]
    out = []
    for key in ("body", "branches", "experts", "stages"):
        if key in node:
            v = node[key]
            if isinstance(v, dict) and key == "experts":
                out.append((key, v["body"]))  # shorthand {"n", "body"}: expose the body
            else:
                out.append((key, v))
    return out


def walk(node, path=()):
    """Yield (path, node) for every dict node (leaves and combinators)."""
    if isinstance(node, list):
        for i, c in enumerate(node):
            yield from walk(c, (*path, i))
        return
    yield path, node
    for key, ch in children(node):
        if isinstance(ch, list):
            for i, c in enumerate(ch):
                yield from walk(c, (*path, key, i))
        else:
            yield from walk(ch, (*path, key))


def _get(spec, path):
    for p in path:
        if (
            isinstance(p, str)
            and p == "experts"
            and isinstance(spec, dict)
            and isinstance(spec["experts"], dict)
        ):
            spec = spec["experts"]["body"]
        else:
            spec = spec[p]
    return spec


def _set(spec, path, new):
    if not path:
        return new
    spec = copy.deepcopy(spec)
    cur = spec
    for p in path[:-1]:
        cur = (
            cur["experts"]["body"]
            if p == "experts" and isinstance(cur, dict) and isinstance(cur["experts"], dict)
            else cur[p]
        )
    last = path[-1]
    if last == "experts" and isinstance(cur, dict) and isinstance(cur["experts"], dict):
        cur["experts"]["body"] = new
    else:
        cur[last] = new
    return spec


def _leaves(spec):
    return [(p, n) for p, n in walk(spec) if n["op"] in OPS]


# ---------------------------------------------------------------------------- mutation


def _swap_candidates(name, kind_hint):
    o = OPS[name]
    keep = (
        o.tags & {"block", "attn", "down", "norm", "act", "stem", "adapt", "tensor", "interaction"}
    ) or o.tags
    return [
        n
        for n, q in OPS.items()
        if n != name
        and q.kin == o.kin
        and (q.kout or q.kin[0]) == (o.kout or o.kin[0])
        and q.tags & keep
    ]


def mutate(spec, kind, dims, rng, op=None):
    """One random structural or parametric change. Returns (new_spec, op_name) or None if inapplicable."""
    op = op or _pick(rng, MUTATIONS)
    leaves = _leaves(spec)
    if op == "param":
        cand = [(p, n) for p, n in leaves if OPS[n["op"]].space]
        if not cand:
            return None
        p, n = _pick(rng, cand)
        key = _pick(rng, list(OPS[n["op"]].space))
        new = {**n, key: _pick(rng, OPS[n["op"]].space[key])}
        return _set(spec, p, new), op
    if op == "swap":
        cand = [(p, n) for p, n in leaves if _swap_candidates(n["op"], kind)]
        if not cand:
            return None
        p, n = _pick(rng, cand)
        name = _pick(rng, _swap_candidates(n["op"], kind))
        fixed = {
            k: v
            for k, v in n.items()
            if k != "op" and k in OPS[name].space and v in OPS[name].space[k]
        }
        return _set(spec, p, rand_params(name, rng, **fixed)), op
    if op in ("insert", "delete"):
        lists = [(p, n) for p, n in walk(spec) if n["op"] == "seq" and isinstance(n["body"], list)]
        if not lists:
            return None
        p, n = _pick(rng, lists)
        body = list(n["body"])
        if op == "delete":
            if len(body) < 2:
                return None
            body.pop(int(rng.integers(len(body))))
        else:
            pool = [
                m
                for m in OPS
                if OPS[m].tags & {"block", "attn", "norm", "act"}
                and kind in OPS[m].kin
                and not _sets_width(m)
            ]
            pool = pool or [m for m in OPS if OPS[m].tags & {"block", "attn"}]
            body.insert(int(rng.integers(len(body) + 1)), rand_params(_pick(rng, pool), rng))
        return _set(spec, (*p, "body"), body), op
    if op in ("wrap_res", "repeat", "to_moe"):
        cand = [(p, n) for p, n in walk(spec) if p and n["op"] not in ("encdec",)]
        if not cand:
            return None
        p, n = _pick(rng, cand)
        if op == "wrap_res":
            if n["op"] == "res":
                return None
            new = {"op": "res", "body": n, "drop_path": _pick(rng, [0.0, 0.1])}
        elif op == "repeat":
            new = (
                {**n, "n": n["n"] + 1}
                if n["op"] == "rep"
                else {"op": "rep", "n": 2, "body": n, "share": False}
            )
        else:
            if n["op"] not in OPS or n["op"] == "moe":
                return None
            new = {
                "op": "moe",
                "experts": {"n": int(_pick(rng, [2, 3, 4])), "body": n},
                "mode": _pick(rng, ["soft", "topk"]),
                "k": 2,
            }
        return _set(spec, p, new), op
    if op == "widen":
        factor = _pick(rng, [0.5, 2.0])

        def scale(node):
            if isinstance(node, list):
                return [scale(x) for x in node]
            if not isinstance(node, dict):
                return node
            new = {}
            for k, v in node.items():
                if k in WIDTH_KEYS and isinstance(v, int) and v > 0 and node.get("op") in OPS:
                    new[k] = max(4, int(v * factor))
                elif isinstance(v, (dict, list)):
                    new[k] = scale(v)
                else:
                    new[k] = v
            return new

        return scale(spec), op
    return None


def crossover(a, b, rng):
    """Stage-level crossover of two top-level sequential specs. Returns the child or None."""

    def top(s):
        return list(s["body"]) if s["op"] == "seq" else None

    A, B = top(a), top(b)
    if not A or not B:
        return None
    i, j = int(rng.integers(1, len(A) + 1)), int(rng.integers(0, len(B)))
    return S(*(A[:i] + B[j:]))


# ------------------------------------------------------------------------- candidates


def _candidate(spec, kind, dims, out_dim, head, train, rng, max_params, origin):
    try:
        info = core.check(spec, kind, dims)
    except ValueError:
        return None
    if not (MIN_PARAMS <= info["params"] <= max_params):
        return None
    if head == "dense" and info["kind"] != "map":
        return None
    cfg = {"family": core.label(spec), "spec": spec, "input": {"kind": kind, "dims": list(dims)},
           "out_dim": out_dim, "head": head, "dropout": 0.0, "params": info["params"], "origin": origin}  # fmt: skip
    for k, vs in train.items():
        cfg[k] = _pick(rng, vs)
    return cfg


def population(
    kind, dims, out_dim, n, seed, max_params, head="cls", train=None, use_presets=True, width=16
):
    """`n` validated candidates: up to n // 2 presets (seeded shuffle) and random grammar samples for the rest."""
    rng = np.random.default_rng(seed)
    train = train or {"lr": [1e-3, 3e-3], "wd": [1e-4, 1e-2], "batch": [32, 64]}
    out, seen = [], set()

    def add(spec, origin):
        c = _candidate(spec, kind, dims, out_dim, head, train, rng, max_params, origin)
        if c and core.signature(spec) not in seen:
            seen.add(core.signature(spec))
            out.append(c)

    if use_presets:
        items = list(presets(kind, dims, head, width).items())
        for i in rng.permutation(len(items)):
            if len(out) >= n // 2:
                break
            add(items[int(i)][1], f"preset:{items[int(i)][0]}")
    tries = 0
    while len(out) < n and tries < 40 * max(n, 1):
        tries += 1
        add(sample(kind, dims, rng, head), "random")
    return out


def offspring(parents, n, seed, max_params, train=None, exclude=()):
    """`n` children of `parents` (candidate dicts): mutation, and crossover when two parents exist."""
    rng = np.random.default_rng(seed)
    if not parents:
        return []
    p0 = parents[0]
    kind, dims, out_dim, head = (
        p0["input"]["kind"],
        tuple(p0["input"]["dims"]),
        p0["out_dim"],
        p0.get("head", "cls"),
    )
    train = train or {k: [p0[k]] for k in ("lr", "wd", "batch") if k in p0}
    seen = set(exclude) | {core.signature(p["spec"]) for p in parents}
    out, tries = [], 0
    while len(out) < n and tries < 60 * n:
        tries += 1
        a = _pick(rng, parents)
        if len(parents) > 1 and rng.random() < 0.3:
            b = _pick(rng, parents)
            spec, how = crossover(a["spec"], b["spec"], rng), "crossover"
            res = (spec, how) if spec else None
        else:
            res = mutate(a["spec"], kind, dims, rng)
        if not res or core.signature(res[0]) in seen:
            continue
        c = _candidate(
            res[0], kind, dims, out_dim, head, train, rng, max_params, f"{res[1]}<-{a['family']}"
        )
        if c:
            c.update(
                {k: a[k] for k in ("lr", "wd", "batch") if k in a}
            )  # children keep the parent's recipe
            seen.add(core.signature(res[0]))
            out.append(c)
    return out
