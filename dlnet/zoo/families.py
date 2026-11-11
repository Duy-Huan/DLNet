"""Model families as RECIPES over the zoo (torch only).

A family (`mlp`, `resnet`, `hybridsn`, `unet`, ...) is a function (config -> spec) of a few sizes:
width, depth, dropout, rank, ... so the search keeps its small discrete spaces while every network is a
zoo architecture: one registry of blocks, one static checker, one cost model, and the same mutation
and crossover operators apply to a family's spec. These recipes replace the separate libraries
(`library.py`, `array_lib.py`, `hsi_lib.py`, `blocks.py`) of earlier versions.

The recipes follow the structure of those earlier families; they are not bit-identical to them
(for example a mixture of experts now mixes features and a shared head reads the mixture).
The head (global pooling + dropout + linear, or a 1x1 convolution per pixel) is added by
`core.build_model` from `cfg["dropout"]`, so a recipe describes the feature extractor only.
"""

from __future__ import annotations

import math

import torch

from . import core
from .presets import N, R, S

TABULAR = (
    "mlp",
    "resmlp",
    "tt_resmlp",
    "tr_resmlp",
    "lowrank_mlp",
    "ft_transformer",
    "moe_mlp",
    "fm_mlp",
    "tabm",
    "tabm_plr",
    "realmlp",
)
ENSEMBLE_FAMILIES = (
    "tabm",
    "tabm_plr",
)  # one output layer and one loss per member (head "ensemble")
IMAGE = ("cnn", "resnet", "vit", "lenet", "allconv", "vgg", "densenet", "mobilenetv2")
SEGMENTATION = (
    "fcn", "unet", "deconvnet", "unetpp", "fpn", "deeplabv3", "deeplabv3plus", "pspnet", "linknet",
    "upernet", "segformer",
)  # fmt: skip
TIMESERIES = ("tcn", "gru", "ts_transformer", "patchtst", "itransformer", "timemixer", "timexer")
HSI = (
    "cnn1d",
    "cnn2d",
    "tucker_cnn2d",
    "cnn3d",
    "hybridsn",
    "resnet2d",
    "dwsep2d",
    "convnext2d",
    "spec_transformer",
    "ssrn",
    "gfnet2d",
    "fno2d",
    "afno2d",
    "ffc2d",
    "fcanet2d",
    "spec_fft2d",
    "fourier_ss",
    "mambahsi",
    "mambahsi_plus",
    "ssftt",
    "fahm",
    "composed",
)
HSI_PRESETS = {
    "resnet2d": {"spatial": "res"}, "dwsep2d": {"spatial": "dwsep"}, "convnext2d": {"spatial": "convnext"},
    "spec_transformer": {"spectral": "transformer"}, "ssrn": {"joint": "ssrn"},
    # frequency domain (dlnet/zoo/ops_freq.py)
    "gfnet2d": {"spatial": "gfnet"}, "fno2d": {"spatial": "fno"}, "afno2d": {"spatial": "afno"},
    "ffc2d": {"spatial": "ffc"}, "fcanet2d": {"spatial": "fca"},
}  # fmt: skip
HSI_SPECTRAL = ("cnn1d", "transformer")
HSI_SPATIAL = ("conv", "res", "dwsep", "convnext", "tucker", "gfnet", "fno", "afno", "ffc", "fca")
FUSIONS = ("concat", "gated")
GRAPH = (
    "gcn_net",
    "gat_net",
    "sage_net",
    "gin_net",
    "graph_mlp",
    "gps_net",
    "polynormer_net",
    "sgformer_net",
)
RECSYS = ("mf", "ncf", "fm", "dlrm", "finalmlp", "dcnv3", "wukong")
RECONSTRUCTION = ("conv_ae", "vae")
BACKBONES = ("lenet", "allconv", "vgg", "resnet", "densenet", "mobilenetv2")


def TB(heads, ratio, p):
    return N(
        "transformer_block",
        heads=heads,
        ratio=ratio,
        out=0,
        dropout=p,
        causal=0,
        bias=0,
        cortex_skip=0,
    )


def rep(n, body, **kw):
    return {"op": "rep", "n": int(n), "body": body, **kw}


def cba(out, k=3, s=1, act="relu", norm="batch", **kw):
    return N("conv_bn_act", out=out, k=k, s=s, act=act, norm=norm, **kw)


# ------------------------------------------------------------------------ tabular


def tabular(cfg, dims):
    f, w, d, p, r = cfg["family"], cfg["width"], cfg["depth"], cfg["dropout"], cfg.get("rank", 8)
    lin = lambda out, act="identity": N("linear_act", out=out, act=act)
    if f == "mlp":
        return S(N("mlp", width=w, depth=d, act="gelu", dropout=p))
    if f in ("resmlp", "tt_resmlp", "tr_resmlp"):
        inner = {
            "resmlp": lambda: lin(w),
            "tt_resmlp": lambda: N("tt_linear", out=w, rank=r),
            "tr_resmlp": lambda: N("tr_linear", out=w, rank=r),
        }[f]
        first = lin(w) if f == "resmlp" else inner()
        block = R(
            S(N("norm", norm="layer"), first, N("act", act="gelu"), N("dropout", p=p), inner())
        )
        return S(lin(w), rep(d, block), N("norm", norm="layer"))
    if f == "lowrank_mlp":
        return S(
            lin(w, "gelu"),
            N("dropout", p=p),
            *[
                S(N("lowrank_linear", out=w, rank=r), N("act", act="gelu"), N("dropout", p=p))
                for _ in range(d - 1)
            ],
        )
    if f == "ft_transformer":
        return S(
            N("feature_tokens", dim=w),
            N("cls_token"),
            rep(d, TB(4, 2, p)),
            N("seq_pool", mode="cls"),
            N("norm", norm="layer"),
        )
    if f == "moe_mlp":
        experts = {
            "n": cfg.get("experts", 4),
            "body": N("mlp", width=max(w // 2, 8), depth=d, act="gelu", dropout=p),
        }
        return S(
            {"op": "moe", "experts": experts, "mode": cfg.get("routing", "soft"), "k": 2},
            lin(w, "gelu"),
        )
    if f == "realmlp":  # RealMLP-TD (NeurIPS 2024): PBLD embedding, front scale, NTK layers
        return S(
            N(
                "realmlp",
                width=w,
                depth=d,
                act=cfg.get("act", "selu"),
                p_drop=p,
                emb="pbld",
                smooth_clip=1,
            )
        )
    if f in ("tabm", "tabm_plr"):  # TabM (arXiv:2410.24210): plain, or on PLR(lite) embeddings
        body = N("tabm", k=cfg.get("k", 16), width=w, depth=d, dropout=p,
                 init="normal" if f == "tabm_plr" else "random-signs")  # fmt: skip
        return S(N("plr_embed", d=cfg.get("emb", 16), freq=16, sigma=cfg.get("sigma", 0.01)), body) \
            if f == "tabm_plr" else S(body)  # fmt: skip
    if f == "fm_mlp":
        return S(
            N("fm_interaction", rank=r, self_interaction=0),
            N("mlp", width=w, depth=d, act="gelu", dropout=p),
        )
    raise ValueError(f"unknown tabular family {f!r}")


# -------------------------------------------------------------------- image / series


def _attn(kind, ch):
    if kind in (None, "none"):
        return []
    if kind == "se":
        return [N("se", ratio=0.25)]
    if kind == "self":
        return [N("self_attn", heads=4 if ch and ch % 4 == 0 else 1)]
    raise ValueError(f"unknown attention {kind!r}")


def n_levels(size, cap):
    """Resolution levels so that the deepest map is at least 2 pixels wide."""
    return max(1, min(cap, int(math.log2(max(size, 2)))))


def encoder_stages(kind, width, depth, size, attention="none", cap=3):
    """List of stage specs of a backbone; stage s has stride 2 relative to s - 1 (stage 0 includes the stem)."""
    if kind not in BACKBONES:
        raise ValueError(f"unknown backbone {kind!r}; use one of {BACKBONES}")
    stages = []
    for s in range(n_levels(size, cap)):
        o, down = width * 2**s, s > 0
        if kind == "lenet":
            body = ([N("pool", mode="max", k=2, s=2)] if down else []) + [
                cba(o, 5, 1, "tanh", "none") for _ in range(depth)
            ]
        elif kind == "allconv":
            body = [cba(o, 3, 2 if down else 1)] + [cba(o) for _ in range(depth - 1)]
        elif kind == "vgg":
            body = ([N("pool", mode="max", k=2, s=2)] if down else []) + [
                cba(o) for _ in range(depth)
            ]
        elif kind == "resnet":
            body = [
                N("basic_block", out=o, s=2 if (down and b == 0) else 1, act="relu", norm="batch")
                for b in range(depth)
            ]
        elif kind == "densenet":
            c0 = max(o // 2, 4)
            growth = max((o - c0) // depth, 2)
            body = (
                [cba(c0, 1)]
                + ([N("pool", mode="avg", k=2, s=2)] if down else [])
                + [N("dense_block", layers=depth, growth=growth, bn_size=2, dropout=0.0)]
            )
        else:
            body = [
                N(
                    "mbconv",
                    out=o,
                    expand=4,
                    k=3,
                    s=2 if (down and b == 0) else 1,
                    se=0,
                    act="relu6",
                )
                for b in range(depth)
            ]
        att = _attn(attention, None if kind == "densenet" else o)
        if s == 0:
            body = [cba(width)] + body  # stem
        stages.append(S(*body, *att))
    return stages


def image(cfg, dims):
    f, w, d, p = cfg["family"], cfg["width"], cfg["depth"], cfg["dropout"]
    size = min(dims[1:])
    if f == "cnn":
        body, s, ww = [], size, w
        for i in range(d):
            body += [cba(ww), cba(ww)]
            if s >= 8 and i < d - 1:
                body.append(N("pool", mode="max", k=2, s=2))
                s //= 2
                ww *= 2
        return S(*body)
    if f == "resnet" and "encoder" not in cfg and cfg.get("attention") is None:
        s, body = size, [cba(w)]
        for stage in range(3):
            o, down = w * 2**stage, stage > 0 and s >= 8
            body += [
                N("basic_block", out=o, s=2 if (down and b == 0) else 1, act="relu", norm="batch")
                for b in range(d)
            ]
            s = s // 2 if down else s
        return S(*body)
    if f == "vit":
        return S(
            N("patchify", dim=w, patch=cfg["patch"]),
            N("cls_token"),
            N("pos", mode="learned"),
            rep(d, TB(4, 2, p)),
            N("seq_pool", mode="cls"),
            N("norm", norm="layer"),
        )
    if f in BACKBONES:
        return S(*encoder_stages(f, w, d, size, cfg.get("attention", "none")))
    raise ValueError(f"unknown image family {f!r}")


def segmentation(cfg, dims):
    f = cfg["family"]
    dec = {"deconvnet": "deconv"}.get(f, f)
    if dec not in SEGMENTATION and dec != "deconv":
        raise ValueError(f"unknown segmentation family {f!r}")
    stages = encoder_stages(
        cfg.get("encoder", "resnet"),
        cfg["width"],
        cfg["depth"],
        min(dims[1:]),
        cfg.get("attention", "none"),
    )
    spec = {"op": "encdec", "stages": stages, "dec": dec, "width": cfg["width"]}
    if dec in ("unet", "unetpp") and cfg.get("dec_attention", "none") != "none":
        spec["attention"] = cfg["dec_attention"]
    return spec


def timeseries(cfg, dims):
    f, w, d, p = cfg["family"], cfg["width"], cfg["depth"], cfg["dropout"]
    if f == "tcn":
        return S(
            *[
                N(
                    "conv1d_resid",
                    out=w,
                    k=cfg.get("kernel", 3),
                    dilation=2**i,
                    norm=1,
                    act="relu",
                    dropout=p,
                )
                for i in range(d)
            ]
        )
    if f == "gru":
        return S(N("tokens"), N("rnn", cell="gru", hidden=w, layers=d, bidir=1, dropout=p))
    if f == "ts_transformer":
        return S(
            cba(w, 1, 1, "identity", "none"),
            N("tokens"),
            N("pos", mode="sine"),
            rep(d, TB(4, 2, p)),
        )
    if f == "patchtst":  # PatchTST (arXiv:2211.14730)
        return S(N("patchtst", d=w, layers=d, heads=4, patch=cfg.get("patch", 8), dropout=p))
    if f == "timemixer":  # TimeMixer (ICLR 2024): decomposable multi-scale mixing
        return S(
            N(
                "timemixer",
                d_model=w,
                e_layers=d,
                down_layers=cfg.get("down", 1),
                moving_avg=cfg.get("moving_avg", 25),
                dropout=p,
            )
        )
    if (
        f == "timexer"
    ):  # TimeXer (NeurIPS 2024): patch tokens + global token + variate cross-attention
        return S(
            N("timexer", patch_len=cfg.get("patch", 8), d_model=w, heads=4, e_layers=d, dropout=p)
        )
    if f == "itransformer":  # iTransformer (arXiv:2310.06625)
        return S(N("itransformer", d=w, layers=d, heads=4, dropout=p))
    raise ValueError(f"unknown time-series family {f!r}")


# ------------------------------------------------------------------------------ HSI


def _spectral(name, w, d, p, bands):
    if name == "cnn1d":
        body = []
        for i in range(d):
            body.append(cba(w * min(2**i, 4), 5))
            if i < d - 1:
                body.append(N("pool", mode="max", k=2, s=2))
        return S(N("center_spectrum"), *body, N("gap", mode="avg"))
    if name == "transformer":
        if w % 4:
            raise ValueError("transformer width must be divisible by the 4 heads")
        return S(N("center_spectrum"), N("tokens"), N("linear_act", out=w, act="identity"), N("cls_token"), N("pos", mode="learned"),
                 rep(d, TB(4, 2, p)), N("seq_pool", mode="cls"), N("norm", norm="layer"))  # fmt: skip
    raise ValueError(f"unknown spectral block {name!r}")


def _spatial(name, w, d, patch):
    if name == "conv":
        block = lambda: cba(w)
    elif name == "res":
        block = lambda: R(S(cba(w), cba(w, act="identity")))
    elif name == "dwsep":
        block = lambda: N("dw_sep", out=w, k=3, s=1, act="relu")
    elif name == "convnext":
        block = lambda: N("convnext_block", k=7 if patch >= 7 else 3, ratio=4, layer_scale=1e-6)
    elif name == "tucker":
        block = lambda: N("tucker_conv", out=0, ratio=0.5, k=3)
    elif name == "gfnet":  # GFNet: global filter in the 2-D Fourier domain of the patch
        block = lambda: N("gfnet_block", mlp_ratio=4, dropout=0.0)
    elif name == "fno":  # FNO layer: lowest frequencies mixed across channels
        block = lambda: N("fno_block", modes=4)
    elif name == "afno":  # AFNO: block-diagonal complex MLP on every frequency
        block = lambda: N("afno_block", num_blocks=8 if w % 8 == 0 else 1, mlp_ratio=4,
                          double_skip=0, dropout=0.0)  # fmt: skip
    elif name == "ffc":  # FFC: local 3x3 conv branch + global Fourier branch
        lfu = int(patch % 2 == 0 and (int(w * 0.5) // 2) % 4 == 0)
        block = lambda: N("ffc_block", ratio=0.5, lfu=lfu)
    elif name == "fca":  # FcaNet basic block: conv-conv, DCT channel attention, residual, ReLU
        att = N("fca", reduction=16 if w >= 32 else 4, method="top16" if w % 16 == 0 else "top1")
        block = lambda: S(R(S(cba(w), cba(w, act="identity"), att)), N("act", act="relu"))
    else:
        raise ValueError(f"unknown spatial block {name!r}")
    return S(cba(w), *[block() for _ in range(d)], N("gap", mode="avg"))


def _res3d(w, k):
    return S(R(S(cba(w, k), cba(w, k, act="identity"))), N("act", act="relu"))


def _joint(name, w, d, bands):
    if name == "conv3d":
        body = []
        for i in range(d):
            body.append(cba(w, 3))
            if i < d - 1:
                body.append(N("pool", mode="max", k=[2, 1, 1], s=[2, 1, 1]))
        return S(N("to_vol"), *body, N("pool_flatten", size=[2, 3, 3]))
    if name == "ssrn":
        if bands < 7:
            raise ValueError("ssrn needs at least 7 spectral components")
        left = (bands - 7) // 2 + 1
        return S(N("to_vol"), cba(w, [7, 1, 1], [2, 1, 1], pad="valid"), rep(d, _res3d(w, [7, 1, 1])), cba(w, [left, 1, 1], 1, pad="valid"),
                 rep(d, _res3d(w, [1, 3, 3])), N("gap", mode="avg"))  # fmt: skip
    raise ValueError(f"unknown joint block {name!r}")


def hsi(cfg, dims):
    f, w, d, p = cfg["family"], cfg["width"], cfg.get("depth", 3), cfg["dropout"]
    bands, patch = dims[0], dims[1]
    if f == "cnn1d":
        return _spectral("cnn1d", w, d, p, bands)
    if f == "cnn2d":
        return S(*[cba(w) for _ in range(d)], N("pool_flatten", size=3))
    if f == "tucker_cnn2d":
        return S(
            cba(w),
            *[N("tucker_conv", out=w, ratio=cfg["rank_ratio"], k=3) for _ in range(d - 1)],
            N("pool_flatten", size=3),
        )
    if f == "cnn3d":
        return _joint("conv3d", w, d, bands)
    if f in (
        "mambahsi",
        "mambahsi_plus",
    ):  # 1x1 embedding, GN, SiLU; blocks with 2x2 pooling between
        blk = N(f + "_block", token_num=4, group_num=4 if f == "mambahsi" else 2)
        body, side = [], patch
        for i in range(d):
            body.append(blk)
            if i < d - 1 and side >= 4:
                body.append(N("pool", mode="avg", k=2, s=2))
                side //= 2
        return S(cba(w, 1, act="swish", norm="group"), *body, N("gap", mode="avg"))
    if f == "ssftt":  # SSFTT (TGRS 2022): 3-D/2-D conv, Gaussian tokenizer, transformer
        return S(N("ssftt", tokens=4, dim=64, heads=8, depth=1))
    if f == "fahm":  # FAHM (2025): frequency-domain embedding + RoPE + Mamba, 3 stages
        return S(N("fahm", size=cfg.get("size", "small"), groups=2))
    if f == "spec_fft2d":  # each pixel's spectrum -> its Fourier coefficients, then a ResNet
        return S(N("spectral_rfft"), _spatial("res", w, d, patch))
    if f == "fourier_ss":  # spectral-axis Fourier filter, then spatial GFNet blocks
        return S(N("spectral_gf"), _spatial("gfnet", w, d, patch))
    if f == "hybridsn":
        if patch < 7 or bands < 13:
            raise ValueError("hybridsn needs patch >= 7 and >= 13 spectral components")
        pad = "same" if patch - 6 < 3 else "valid"
        return S(N("to_vol"), cba(w, [7, 3, 3], 1, pad="valid"), cba(2 * w, [5, 3, 3], 1, pad="valid"), cba(4 * w, [3, 3, 3], 1, pad="valid"),
                 N("vol_to_map"), cba(64, 3, 1, pad=pad), N("flatten"), N("linear_act", out=256, act="relu"), N("dropout", p=p),
                 N("linear_act", out=128, act="relu"), N("dropout", p=p))  # fmt: skip
    cfg = {**cfg, **HSI_PRESETS.get(f, {})}
    spectral, spatial, joint = (
        cfg.get("spectral") or "none",
        cfg.get("spatial") or "none",
        cfg.get("joint") or "none",
    )
    fusion = cfg.get("fusion") or "none"
    used = [
        k
        for k, v in (("spectral", spectral), ("spatial", spatial), ("joint", joint))
        if v != "none"
    ]
    if not used:
        raise ValueError("a composed model needs at least one encoder")
    if "joint" in used and len(used) > 1:
        raise ValueError("a joint encoder cannot be combined with spectral/spatial branches")
    if (len(used) == 2) != (fusion != "none"):
        raise ValueError("fusion is required for exactly two branches and forbidden otherwise")
    if fusion != "none" and fusion not in FUSIONS:
        raise ValueError(f"unknown fusion {fusion!r}")
    if joint != "none":
        return _joint(joint, w, d, bands)
    branches = []
    if spectral != "none":
        branches.append(_spectral(spectral, w, d, p, bands))
    if spatial != "none":
        branches.append(_spatial(spatial, w, d, patch))
    if len(branches) == 1:
        return branches[0]
    merge = (
        {"op": "par", "branches": branches, "merge": "concat"}
        if fusion == "concat"
        else {"op": "par", "branches": branches, "merge": "gated", "width": w}
    )
    return S(merge)


def sample_composed(rng) -> dict:
    """Block choice of one grammar-composed HSI model (always type-correct)."""
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    return {
        "spectral": pick(HSI_SPECTRAL),
        "spatial": pick(HSI_SPATIAL),
        "joint": "none",
        "fusion": pick(FUSIONS),
    }


# ------------------------------------------------------------------ autoencoders


def code_size(cfg, dims) -> int:
    """Number of elements of the bottleneck code of an autoencoder recipe."""
    h, w = dims[1], dims[2]
    for _ in range(n_levels(min(dims[1:]), 3) - 1):
        h, w = -(-h // 2), -(-w // 2)
    return cfg["latent"] * h * w


def reconstruction(cfg, dims):
    """Encoder backbone, a bottleneck (plain 1x1 convolution, or variational) and a transposed-convolution decoder, no skips."""
    f = cfg["family"]
    stages = encoder_stages(cfg.get("encoder", "vgg"), cfg["width"], cfg["depth"], min(dims[1:]))
    if f == "conv_ae":
        stages.append(S(cba(cfg["latent"], 1, 1, "identity", "none")))
    elif f == "vae":
        stages.append(S(N("vae_bottleneck", latent=cfg["latent"])))
    else:
        raise ValueError(f"unknown autoencoder family {f!r}")
    return {"op": "encdec", "stages": stages, "dec": "deconv", "width": cfg["width"]}


# ---------------------------------------------------------------- graphs and ids


def graph(cfg, dims):
    f, w, d, p = cfg["family"], cfg["width"], cfg["depth"], cfg["dropout"]
    layer = {
        "gcn_net": lambda: N("gcn", out=w, act="relu"),
        "gat_net": lambda: N("gat", out=w, heads=2),
        "sage_net": lambda: N("sage", out=w),
        "gin_net": lambda: N("gin", out=w),
    }
    ro = N("graph_readout", mode=cfg.get("readout", "mean"))
    if (
        f == "gps_net"
    ):  # GraphGPS: RWSE, input projection, GPS layers (local GIN + global attention)
        heads = 4 if w % 4 == 0 else 1
        return S(
            N("rwse", k=8),
            N("graph_lin", out=w),
            *[N("gps_layer", heads=heads, dropout=p) for _ in range(d)],
            ro,
        )
    if f == "polynormer_net":  # Polynormer (ICLR 2024): local then global polynomial attention
        return S(
            N("polynormer", hidden=w, heads=1, local_layers=max(d, 1), global_layers=2, dropout=p),
            ro,
        )
    if f == "sgformer_net":  # SGFormer (NeurIPS 2023): simple global attention + GCN branch
        return S(N("sgformer", hidden=w, layers=max(d, 1), heads=1, dropout=p), ro)
    if f == "graph_mlp":  # no message passing: the flattened features and adjacency
        return S(N("flatten"), N("mlp", width=w, depth=d, act="gelu", dropout=p))
    if f not in layer:
        raise ValueError(f"unknown graph family {f!r}")
    return S(*[layer[f]() for _ in range(d)], N("graph_readout", mode=cfg.get("readout", "mean")))


def recsys(cfg, dims):
    f, w, d, p = cfg["family"], cfg["width"], cfg["depth"], cfg["dropout"]
    emb = N("embed_fields", dim=w)
    if f == "mf":
        return S(emb, N("gmf"))
    if f == "ncf":
        return S(emb, N("flatten"), N("mlp", width=2 * w, depth=d, act="relu", dropout=p))
    if f == "fm":
        return S(emb, N("pair_dots", self_interaction=0))
    if f == "dlrm":
        both = {
            "op": "par",
            "branches": [N("flatten"), N("pair_dots", self_interaction=0)],
            "merge": "concat",
        }
        return S(emb, both, N("mlp", width=2 * w, depth=d, act="relu", dropout=p))
    if f == "finalmlp":  # FinalMLP (AAAI 2023): two gated MLP streams, bilinear aggregation
        return S(emb, N("finalmlp", hidden=4 * w, depth=d + 1, heads=1, out=16, dropout=p))
    if f == "dcnv3":  # DCNv3 / FCN: exponential + linear cross networks
        return S(emb, N("dcnv3", exp_layers=d + 1, lin_layers=d + 1, dropout=p))
    if f == "wukong":  # WuKong (ICML 2024): stacked FM blocks + linear compression, then an MLP
        return S(emb, N("wukong", layers=d + 1, lcb=16, fmb=16, rank_k=8, dropout=p), N("flatten"),
                 N("mlp", width=2 * w, depth=2, act="relu", dropout=p))  # fmt: skip
    raise ValueError(f"unknown recsys family {f!r}")


# ------------------------------------------------------------------------ dispatch

DOMAINS = {
    "tabular": (TABULAR, tabular, "vec", "cls"),
    "image": (IMAGE, image, "map", "cls"),
    "segmentation": (SEGMENTATION, segmentation, "map", "dense"),
    "timeseries": (TIMESERIES, timeseries, "ser", "cls"),
    "hsi": (HSI, hsi, "map", "cls"),
    "graph": (GRAPH, graph, "graph", "cls"),
    "recsys": (RECSYS, recsys, "ids", "cls"),
    "reconstruction": (RECONSTRUCTION, reconstruction, "map", "dense"),
}


def complete(cfg: dict, domain: str, dims, out_dim: int) -> dict:
    """Add `spec`, `input`, `head`, `out_dim` to a family config (in place) and return it.

    Raises ValueError when the family does not fit the input.
    """
    fams, recipe, kind, head = DOMAINS[domain]
    if cfg["family"] not in fams:
        raise ValueError(f"unknown {domain} family {cfg['family']!r}")
    cfg["spec"] = recipe(cfg, tuple(dims))
    if cfg["family"] in ENSEMBLE_FAMILIES:
        head = "ensemble"
    cfg.update(input={"kind": kind, "dims": list(dims)}, head=head, out_dim=out_dim)
    return cfg


def count_params(cfg: dict) -> int:
    """Parameters of the whole model (head included) from the meta device: no memory, no time."""
    try:
        with torch.device("meta"):
            return sum(p.numel() for p in core.build_model(cfg).parameters())
    except (RuntimeError, NotImplementedError, TypeError):
        return sum(p.numel() for p in core.build_model(cfg).parameters())


def build(cfg: dict, domain: str, dims, out_dim: int):
    """The model of a family config (does not modify `cfg`)."""
    return core.build_model(complete(dict(cfg), domain, dims, out_dim))


def tabular_model(cfg: dict, in_dim: int, out_dim: int):
    """`build_model(cfg, in_dim, out_dim)` for a tabular family config (used by tests and scripts)."""
    return build(cfg, "tabular", (in_dim,), out_dim)
