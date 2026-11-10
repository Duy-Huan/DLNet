"""Well-known architectures written as specs of zoo ops (seeds for the search).

Each preset adapts to the input: the number of downsampling stages follows the spatial size.
They are re-creations in the spirit of the originals, scaled down for small data; none is a
verified reproduction of the published model. The search treats them like any other candidate.
"""

from __future__ import annotations

import math

from .core import ND

N = lambda op, **kw: {"op": op, **kw}
R = lambda body, **kw: {"op": "res", "body": body, **kw}
S = lambda *body: {"op": "seq", "body": list(body)}


def n_stages(dims, cap=3, floor=2):
    """Downsampling levels such that the smallest spatial side stays >= `floor` * 2."""
    side = min(dims[1:])
    return max(1, min(cap, int(math.log2(max(side // floor, 1)))))


def _cnn_presets(kind, dims, w):
    s = n_stages(dims)
    st = range(s)
    c = lambda i: w * 2**i
    stem = N("conv_bn_act", out=w, k=3, s=1, act="relu", norm="batch")
    p = {
        "lenet": S(*[N("lenet_stage", out=c(i) if i else 6 * w // 8, k=5) for i in st]),
        "vgg": S(*[N("vgg_stage", out=c(i), n=2, norm="batch", pool=2) for i in st]),
        "resnet": S(
            stem,
            *[
                S(N("basic_block", out=c(i), s=2 if i else 1), N("basic_block", out=c(i), s=1))
                for i in st
            ],
        ),
        "resnet_se": S(
            stem,
            *[
                S(
                    N("basic_block", out=c(i), s=2 if i else 1),
                    N("basic_block", out=c(i)),
                    N("se", ratio=0.25),
                )
                for i in st
            ],
        ),
        "preact_resnet": S(
            stem,
            *[
                S(N("preact_block", out=c(i), s=2 if i else 1), N("preact_block", out=c(i)))
                for i in st
            ],
        ),
        "resnext": S(
            stem,
            *[
                S(
                    N("bottleneck", out=4 * c(i), s=2 if i else 1, ratio=4, groups=8, base_width=4),
                    N("bottleneck", out=4 * c(i), s=1, ratio=4, groups=8, base_width=4),
                )
                for i in st
            ],
        ),
        "wide_resnet": S(
            stem,
            *[
                S(N("bottleneck", out=4 * c(i), s=2 if i else 1, ratio=4, groups=1, base_width=128))
                for i in st
            ],
        ),
        "densenet": S(
            stem,
            *[
                S(
                    N("dense_block", layers=3, growth=w // 2 or 4, bn_size=2, dropout=0.0),
                    N("transition", compression=0.5, pool=2),
                )
                for _ in range(s - 1)
            ],
            N("dense_block", layers=3, growth=w // 2 or 4, bn_size=2, dropout=0.0),
        ),
        "mobilenet": S(
            stem,
            *[
                S(
                    N("mbconv", out=c(i), expand=4, k=3, s=2 if i else 1, se=0, act="relu6"),
                    N("mbconv", out=c(i), expand=4, k=3, s=1, se=0, act="relu6"),
                )
                for i in st
            ],
        ),
        "efficientnet": S(
            stem,
            *[
                S(
                    N("fused_mbconv", out=c(i), expand=2, k=3, s=2 if i else 1, se=0, act="swish"),
                    N("mbconv", out=c(i), expand=4, k=3, s=1, se=1, act="swish"),
                )
                for i in st
            ],
        ),
        "convnext": S(
            N("conv_bn_act", out=w, k=3, s=1, act="identity", norm="layer"),
            *[
                S(
                    *(
                        [
                            N(
                                "conv_bn_act",
                                out=c(i),
                                k=2 if min(dims[1:]) >= 8 else 3,
                                s=2,
                                act="identity",
                                norm="layer",
                            )
                        ]
                        if i
                        else []
                    ),
                    N("convnext_block", k=7, ratio=4, layer_scale=1e-6),
                    N("convnext_block", k=7, ratio=4, layer_scale=1e-6),
                )
                for i in st
            ],
        ),
        "regnet": S(
            stem,
            *[
                S(
                    N("regnet_block", out=c(i), s=2 if i else 1, group_width=8, ratio=1.0, se=0.25),
                    N("regnet_block", out=c(i), s=1, group_width=8, ratio=1.0, se=0.25),
                )
                for i in st
            ],
        ),
        "squeezenet": S(
            stem,
            *[
                S(
                    N("fire", out=c(i), squeeze_ratio=4),
                    N("fire", out=c(i), squeeze_ratio=4),
                    N("pool", mode="max", k=2, s=2) if i < s - 1 else N("dropout", p=0.0),
                )
                for i in st
            ],
        ),
        "nin": S(
            *[
                S(
                    N("nin_block", out=c(i), k=3, dropout=0.0),
                    N("pool", mode="max", k=2, s=2) if i < s - 1 else N("dropout", p=0.0),
                )
                for i in st
            ]
        ),
        "inception": S(
            stem,
            *[
                S(
                    N("inception", out=4 * c(i)),
                    N("pool", mode="max", k=2, s=2) if i < s - 1 else N("dropout", p=0.0),
                )
                for i in st
            ],
        ),
        "dw_cnn": S(
            stem,
            *[
                S(
                    N("dw_sep", out=c(i), k=3, s=2 if i else 1, act="relu"),
                    N("dw_sep", out=c(i), k=3, s=1, act="relu"),
                )
                for i in st
            ],
        ),
    }
    if kind == "map":
        p["shufflenet"] = S(
            N("conv_bn_act", out=2 * w, k=3, s=1, act="relu", norm="batch"),
            *[
                S(
                    N("shuffle_unit", out=2 * c(i), s=2 if i else 1),
                    N("shuffle_unit", out=2 * c(i), s=1),
                )
                for i in st
            ],
        )
        side = min(dims[1:])
        win = next(
            (x for x in (8, 4, 2) if all(d % (x * 2 ** (s - 1)) == 0 for d in dims[1:])), None
        )
        if win and side >= 4:
            p["swin_like"] = S(
                N("conv_bn_act", out=w, k=3, s=1, act="identity", norm="layer"),
                *[
                    S(*([N("patch_merge")] if i else []), N("window_attn", heads=2, window=win, shift=0),
                      N("window_attn", heads=2, window=win, shift=1))
                    for i in range(s)
                ],
            )  # fmt: skip
        for patch in (4, 2):
            if all(d % patch == 0 for d in dims[1:]):
                dim = 4 * w
                p["vit"] = S(
                    N("patchify", dim=dim, patch=patch),
                    N("pos", mode="learned"),
                    {
                        "op": "rep",
                        "n": 2,
                        "body": N(
                            "transformer_block", heads=4, ratio=2, out=0, dropout=0.0, causal=0
                        ),
                    },
                )
                p["hybrid_cnn_transformer"] = S(stem, N("basic_block", out=2 * w, s=2), N("tokens"), N("pos", mode="learned"),
                                                {"op": "rep", "n": 2, "body": N("transformer_block", heads=4, ratio=2, out=0, dropout=0.0, causal=0)}) \
                    if (2 * w) % 4 == 0 and min(dims[1:]) >= 4 else None  # fmt: skip
                break
    return {k: v for k, v in p.items() if v is not None}


def _dense_presets(dims, w):
    s = max(2, n_stages(dims, cap=3, floor=2))
    c = lambda i: w * 2**i

    def enc(kind):
        if kind == "res":
            return [
                S(
                    N("conv_bn_act", out=w, k=3, s=1, act="relu", norm="batch"),
                    N("basic_block", out=w, s=1),
                )
                if i == 0
                else S(N("basic_block", out=c(i), s=2), N("basic_block", out=c(i), s=1))
                for i in range(s)
            ]
        if kind == "vgg":
            return [
                N("vgg_stage", out=c(i), n=2, norm="batch", pool=1 if i == 0 else 2)
                for i in range(s)
            ]
        return [
            S(
                N("conv_bn_act", out=w, k=3, s=1, act="relu", norm="batch"),
                N("mbconv", out=w, expand=4, k=3, s=1, se=0, act="relu6"),
            )
            if i == 0
            else N("mbconv", out=c(i), expand=4, k=3, s=2, se=0, act="relu6")
            for i in range(s)
        ]

    return {
        "unet_res": {"op": "encdec", "stages": enc("res"), "dec": "unet", "width": w},
        "unet_vgg": {"op": "encdec", "stages": enc("vgg"), "dec": "unet", "width": w},
        "unet_mobile": {"op": "encdec", "stages": enc("mb"), "dec": "unet", "width": w},
        "fcn_vgg": {"op": "encdec", "stages": enc("vgg"), "dec": "fcn", "width": w},
        "fpn_res": {"op": "encdec", "stages": enc("res"), "dec": "fpn", "width": w},
        "deeplab_res": {"op": "encdec", "stages": enc("res"), "dec": "aspp", "width": w},
        "autoencoder_deconv": {"op": "encdec", "stages": enc("vgg"), "dec": "deconv", "width": w},
    }


def _seq_presets(t, d):
    blk = lambda **kw: N("transformer_block", heads=4, ratio=2, out=0, dropout=0.0, causal=0, **kw)
    p = {
        "transformer": S(N("pos", mode="sine"), {"op": "rep", "n": 2, "body": blk()}),
        "gru": S(N("rnn", cell="gru", hidden=64, layers=1, bidir=1, dropout=0.0)),
        "lstm": S(N("rnn", cell="lstm", hidden=64, layers=2, bidir=1, dropout=0.0)),
        "mixer": S(
            {"op": "rep", "n": 2, "body": N("mixer_block", token_ratio=1.0, channel_ratio=2)}
        ),
        "mlp_seq": S(N("mlp", width=128, depth=2, act="gelu", dropout=0.0)),
    }
    if d % 4 == 0:
        p["mamba"] = S(
            {"op": "rep", "n": 2, "body": N("mamba_block", d_state=16, d_conv=4, expand=2)}
        )
        p["transformer_cls"] = S(
            N("cls_token"), N("pos", mode="learned"), {"op": "rep", "n": 2, "body": blk()}
        )
    return p


def _ser_presets(c, t, w):
    to_seq = [N("tokens")]
    d = c
    p = {
        "tcn": S(
            *[
                N("conv1d_resid", out=w, k=3, dilation=2**i, norm=1, act="swish", dropout=0.0)
                for i in range(3)
            ]
        ),
        "textcnn": S(N("textcnn", filters=w, ks=[3, 4, 5], dropout=0.0)),
        "resnet1d": S(
            N("conv_bn_act", out=w, k=7, s=1, act="relu", norm="batch"),
            N("basic_block", out=w, s=1),
            N("basic_block", out=2 * w, s=2),
        ),
        "gru": S(*to_seq, N("rnn", cell="gru", hidden=w, layers=1, bidir=1, dropout=0.0)),
        "lstm": S(*to_seq, N("rnn", cell="lstm", hidden=w, layers=1, bidir=1, dropout=0.0)),
        "transformer": S(
            N("conv_bn_act", out=w, k=1, s=1, act="identity", norm="none"),
            *to_seq,
            N("pos", mode="sine"),
            {
                "op": "rep",
                "n": 2,
                "body": N("transformer_block", heads=4, ratio=2, out=0, dropout=0.0, causal=0),
            },
        ),
        "mamba": S(
            N("conv_bn_act", out=w, k=1, s=1, act="identity", norm="none"),
            *to_seq,
            {"op": "rep", "n": 2, "body": N("mamba_block", d_state=16, d_conv=4, expand=2)},
        ),
        "mbconv1d": S(
            N("conv_bn_act", out=w, k=3, s=1, act="swish", norm="batch"),
            N("mbconv", out=w, expand=4, k=5, s=1, se=1, act="swish"),
            N("mbconv", out=2 * w, expand=4, k=5, s=2, se=1, act="swish"),
        ),
    }
    del d
    return p


def _vec_presets(d, w):
    mlp = lambda depth: S(N("mlp", width=4 * w, depth=depth, act="gelu", dropout=0.1))
    res = lambda lin, n: S(
        N("linear_act", out=4 * w, act="identity"),
        *[
            R(
                S(
                    N("norm", norm="layer"),
                    lin,
                    N("act", act="gelu"),
                    N("linear_act", out=4 * w, act="identity"),
                )
            )
            for _ in range(n)
        ],
    )
    p = {
        "mlp": mlp(3),
        "resmlp": S(
            N("linear_act", out=4 * w, act="identity"),
            *[R(N("ffn", ratio=2, act="gelu", dropout=0.1)) for _ in range(2)],
        ),
        "tt_resmlp": res(N("tt_linear", out=4 * w, rank=8), 2),
        "tr_resmlp": res(N("tr_linear", out=4 * w, rank=4), 2),
        "lowrank_mlp": S(
            N("linear_act", out=4 * w, act="gelu"),
            N("lowrank_linear", out=4 * w, rank=16),
            N("act", act="gelu"),
            N("lowrank_linear", out=4 * w, rank=16),
            N("act", act="gelu"),
        ),
        "glu_net": S(N("glu", out=4 * w), N("glu", out=4 * w)),
        "dcn": S(N("cross_net", layers=2), N("mlp", width=4 * w, depth=2, act="relu", dropout=0.1)),
        "moe_mlp": S(
            {
                "op": "moe",
                "experts": {
                    "n": 4,
                    "body": N("mlp", width=2 * w, depth=1, act="gelu", dropout=0.0),
                },
                "mode": "soft",
                "k": 2,
            },
            N("linear_act", out=4 * w, act="gelu"),
        ),
        "moe_topk": S(
            {
                "op": "moe",
                "experts": {
                    "n": 4,
                    "body": N("mlp", width=2 * w, depth=1, act="gelu", dropout=0.0),
                },
                "mode": "topk",
                "k": 2,
            },
            N("linear_act", out=4 * w, act="gelu"),
        ),
    }
    if d <= 128:
        p["fm_mlp"] = S(
            N("fm_interaction", rank=8, self_interaction=0),
            N("mlp", width=4 * w, depth=2, act="gelu", dropout=0.1),
        )
        p["ft_transformer"] = S(
            N("feature_tokens", dim=2 * w),
            N("cls_token"),
            {
                "op": "rep",
                "n": 2,
                "body": N("transformer_block", heads=4, ratio=2, out=0, dropout=0.1, causal=0),
            },
        )
    return p


def _graph_presets(dims, w):
    ro = lambda m: N("graph_readout", mode=m)
    return {
        "gcn": S(N("gcn", out=2 * w), N("gcn", out=2 * w), ro("mean")),
        "gcn_deep": S(*[N("gcn", out=2 * w) for _ in range(3)], ro("max")),
        "gat": S(N("gat", out=2 * w, heads=2), N("gat", out=2 * w, heads=2), ro("mean")),
        "graphsage": S(N("sage", out=2 * w), N("sage", out=2 * w), ro("mean")),
        "gin": S(N("gin", out=2 * w), N("gin", out=2 * w), ro("sum")),
        "graph_mlp": S(N("flatten"), N("mlp", width=4 * w, depth=2, act="gelu", dropout=0.1)),
    }


def _ids_presets(dims, w):
    emb = N("embed_fields", dim=w)
    both = {
        "op": "par",
        "branches": [N("flatten"), N("pair_dots", self_interaction=0)],
        "merge": "concat",
    }
    p = {
        "ncf": S(emb, N("flatten"), N("mlp", width=2 * w, depth=2, act="relu", dropout=0.0)),
        "fm": S(emb, N("pair_dots", self_interaction=0)),
        "dlrm": S(emb, both, N("mlp", width=2 * w, depth=2, act="relu", dropout=0.0)),
    }
    if len(dims) == 2:
        p["gmf"] = S(emb, N("gmf"))
    return p


def presets(kind: str, dims, head: str = "cls", width: int = 16) -> dict[str, dict]:
    """name -> spec for this input. Candidates that do not fit the shape are dropped by the caller."""
    if kind == "map" and head == "dense":
        return _dense_presets(dims, width)
    if kind == "graph":
        return _graph_presets(dims, width)
    if kind == "ids":
        return _ids_presets(dims, width)
    if kind == "vec":
        return _vec_presets(dims[0], width)
    if kind == "seq":
        return _seq_presets(*dims)
    if kind == "ser":
        return _ser_presets(dims[0], dims[1], width * 2)
    if kind in ("map", "vol"):
        if kind == "vol":
            p = _cnn_presets(kind, dims, width)
            return {k: v for k, v in p.items() if k not in ("lenet", "squeezenet", "inception")} | {
                k: v for k, v in p.items() if k in ("lenet", "squeezenet", "inception")
            }
        return _cnn_presets(kind, dims, width)
    raise ValueError(f"unknown kind {kind!r}")


_ = ND
