"""Numerical equivalence of zoo blocks with the ORIGINAL implementations (weights copied across).

    PYTHONPATH=<torchvision> python scripts/verify_numeric.py --mamba DIR --cortex DIR

Every block is built twice (original and zoo) with the same configuration; parameters and buffers are
copied across in definition order (shapes must agree one by one) and the outputs on the same random
input are compared in eval mode. Missing optional dependencies of the upstream packages are stubbed
(they are not needed for the blocks checked here). Results: THIRD_PARTY.md.
"""

from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock

import torch
from torch import nn

from dlnet.zoo.core import _build


def import_with_stubs(fn, tries=40):
    """Call fn(); on ModuleNotFoundError stub the missing module and retry."""
    for _ in range(tries):
        try:
            return fn()
        except ModuleNotFoundError as e:
            name = e.name or ""
            sys.modules[name] = MagicMock()
            parts = name.split(".")
            for i in range(1, len(parts)):
                sys.modules.setdefault(".".join(parts[:i]), MagicMock())
    raise RuntimeError("too many missing modules")


def tensors(m):
    return [p for p in m.parameters()] + [b for n, b in m.named_buffers() if "num_batches" not in n]


def tensors_proj_last(m):
    """RegNet defines the shortcut first; our block defines it last."""
    ps = [(n, p) for n, p in m.named_parameters()]
    bs = [(n, b) for n, b in m.named_buffers() if "num_batches" not in n]
    key = lambda nb: nb[0].startswith("proj")
    return [p for _, p in sorted(ps, key=key)] + [b for _, b in sorted(bs, key=key)]


def copy_in_order(src, dst, fn=None):
    a, b = (fn or tensors)(src), tensors(dst)
    if [tuple(x.shape) for x in a] != [tuple(x.shape) for x in b]:
        raise ValueError(
            "parameter shapes differ in order: "
            + str([(tuple(x.shape), tuple(y.shape)) for x, y in zip(a, b, strict=False)][:6])
        )
    with torch.no_grad():
        for x, y in zip(a, b, strict=True):
            y.copy_(x)


def copy_by_name(src, dst, mapping):
    """mapping: upstream state_dict key -> zoo key (shapes may differ by size-1 axes)."""
    s, d = src.state_dict(), dst.state_dict()
    unmapped = [k for k in s if k not in mapping and "num_batches" not in k]
    if unmapped or set(mapping.values()) != {k for k in d if "num_batches" not in k}:
        raise ValueError(
            f"mapping incomplete: upstream {unmapped}, zoo {sorted(set(d) - set(mapping.values()))}"
        )
    with torch.no_grad():
        for k, v in mapping.items():
            dst.state_dict()[v].copy_(s[k].reshape(dst.state_dict()[v].shape))


def compare(
    name,
    theirs,
    ours_spec,
    kind,
    dims,
    x=None,
    atol=1e-5,
    rows=None,
    wrap=None,
    mapping=None,
    fn=None,
):
    theirs.eval()
    mine, _, _ = _build(ours_spec, kind, dims)
    mine.eval()
    try:
        if mapping:
            copy_by_name(theirs.m if hasattr(theirs, "m") else theirs, mine, mapping)
        else:
            copy_in_order(theirs, mine, fn)
    except ValueError as e:
        rows.append((name, "ORDER MISMATCH", str(e)[:90]))
        return
    x = torch.randn(2, *dims) if x is None else x
    with torch.no_grad():
        a, b = theirs(x), mine(x)
    if isinstance(a, tuple):
        a = a[0]
    err = float((a - b).abs().max())
    rows.append((name, "OK" if err <= atol else "DIFFERENT", f"max abs diff {err:.2e}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mamba", required=True)
    ap.add_argument("--cortex", required=True)
    a = ap.parse_args()
    sys.path[:0] = [a.mamba, a.cortex]
    rows = []
    torch.manual_seed(0)

    # ---- state-spaces/mamba: Mamba mixer (slow path) and selective_scan_ref
    ref = import_with_stubs(
        lambda: (
            __import__("mamba_ssm.ops.selective_scan_interface", fromlist=["x"]).selective_scan_ref
        )
    )
    from dlnet.zoo.ops_seq import selective_scan

    b, d, n, L = 2, 6, 8, 12
    u, delta, A = torch.randn(b, d, L), torch.rand(b, d, L), -torch.rand(d, n)
    B, C, D = torch.randn(b, n, L), torch.randn(b, n, L), torch.randn(d)
    err = float((ref(u, delta, A, B, C, D) - selective_scan(u, delta, A, B, C, D)).abs().max())
    rows.append(
        (
            "mamba selective_scan_ref vs selective_scan",
            "OK" if err < 1e-5 else "DIFFERENT",
            f"max abs diff {err:.2e}",
        )
    )
    mod = import_with_stubs(lambda: __import__("mamba_ssm.modules.mamba_simple", fromlist=["x"]))
    mod.selective_scan_fn, mod.causal_conv1d_fn = (
        lambda x, dt, A, B, C, D, z, delta_bias, delta_softplus, return_last_state=False: ref(
            x, dt, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus
        ),
        None,
    )
    for dm, ds, dc, ex in ((16, 16, 4, 2), (32, 8, 3, 1)):
        up = mod.Mamba(dm, d_state=ds, d_conv=dc, expand=ex, use_fast_path=False)
        mine = _build(
            {"op": "mamba_block", "d_state": ds, "d_conv": dc, "expand": ex}, "seq", (10, dm)
        )[0]
        # our block = x + mixer(RMSNorm(x)); copy the mixer weights, norm weight stays 1
        copy_in_order(up, mine.mixer)
        x = torch.randn(2, 10, dm)
        with torch.no_grad():
            err = float((x + up(mine.norm(x)) - mine(x)).abs().max())
        rows.append(
            (
                f"mamba_simple.Mamba d={dm} state={ds} conv={dc} expand={ex}",
                "OK" if err < 1e-5 else "DIFFERENT",
                f"max abs diff {err:.2e}",
            )
        )

    # ---- Mamba-2: SSD and the mixer (the upstream Triton kernel is replaced by the repo's own minimal SSD)
    ssd = import_with_stubs(lambda: __import__("mamba_ssm.modules.ssd_minimal", fromlist=["x"]))
    from dlnet.zoo.ops_seq import ssd_minimal

    bb, ll, hh, pp, nn_, ch = 2, 32, 4, 8, 16, 8
    Xs, As = torch.randn(bb, ll, hh, pp), -torch.rand(bb, ll, hh)
    Bs, Cs = torch.randn(bb, ll, hh, nn_), torch.randn(bb, ll, hh, nn_)
    err = float(
        (ssd.ssd_minimal_discrete(Xs, As, Bs, Cs, ch)[0] - ssd_minimal(Xs, As, Bs, Cs, ch))
        .abs()
        .max()
    )
    rows.append(
        (
            "mamba2 ssd_minimal_discrete vs ssd_minimal",
            "OK" if err < 1e-5 else "DIFFERENT",
            f"max abs diff {err:.2e}",
        )
    )
    m2 = import_with_stubs(lambda: __import__("mamba_ssm.modules.mamba2_simple", fromlist=["x"]))
    rms_ref = import_with_stubs(
        lambda: __import__("mamba_ssm.ops.triton.layernorm_gated", fromlist=["x"]).rms_norm_ref
    )

    class GatedRMS(
        nn.Module
    ):  # upstream RMSNormGated (Triton) -> its own pure-PyTorch reference, norm_before_gate=False
        def __init__(self, d, eps=1e-5, norm_before_gate=False, **kw):
            super().__init__()
            self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

        def forward(self, x, z=None):
            return rms_ref(x, self.weight, None, z=z, eps=self.eps, norm_before_gate=False)

    m2.RMSNormGated = GatedRMS
    m2.mamba_chunk_scan_combined = (
        lambda x, dt, A, B, C, chunk_size, D=None, z=None, seq_idx=None, initial_states=None: (
            ssd.ssd_minimal_discrete(x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)[0]
            + x * D.unsqueeze(-1)
        )
    )
    m2.causal_conv1d_fn = None
    for dm, ds, ex, hd, chunk, length in ((32, 16, 2, 16, 8, 24), (32, 64, 1, 16, 16, 32)):
        up = m2.Mamba2Simple(
            dm,
            d_state=ds,
            d_conv=4,
            expand=ex,
            headdim=hd,
            chunk_size=chunk,
            use_mem_eff_path=False,
        )
        mine = _build(
            {
                "op": "mamba2_block",
                "d_state": ds,
                "d_conv": 4,
                "expand": ex,
                "headdim": hd,
                "chunk": chunk,
            },
            "seq",
            (length, dm),
        )[0]
        mine.eval()
        copy_by_name(up, mine.mixer, {k: k for k in up.state_dict()})
        x = torch.randn(2, length, dm)
        with torch.no_grad():
            err = float((x + up(mine.norm(x)) - mine(x)).abs().max())
        rows.append(
            (
                f"mamba2_simple.Mamba2Simple d={dm} state={ds} expand={ex} headdim={hd} chunk={chunk} len={length}",
                "OK" if err < 1e-4 else "DIFFERENT",
                f"max abs diff {err:.2e}",
            )
        )

    # ---- prescient-design/cortex blocks
    TB = import_with_stubs(
        lambda: __import__("cortex.model.block", fromlist=["x"]).TransformerBlock
    )
    C1 = import_with_stubs(
        lambda: __import__("cortex.model.block", fromlist=["x"]).Conv1dResidBlock
    )

    class TBW(nn.Module):  # cortex blocks take (x, padding_mask)
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m((x, torch.ones(x.shape[:2])))[0]

    for dd, h, out, causal in ((32, 4, 32, False), (32, 4, 64, False), (32, 2, 32, True)):
        t = TB(dd, out, num_heads=h, bias=False, dropout_p=0.0, is_causal=causal)
        spec = {
            "op": "transformer_block",
            "heads": h,
            "ratio": 4,
            "out": 0 if out == dd else out,
            "dropout": 0.0,
            "causal": int(causal),
        }
        mp = {"ln_1.weight": "ln1.weight", "attn.c_attn.weight": "attn.qkv.weight", "attn.c_proj.weight": "attn.proj.weight",
              "ln_2.weight": "ln2.weight", "mlp.0.weight": "mlp.0.weight", "mlp.2.weight": "mlp.2.weight"}  # fmt: skip
        if out != dd:
            mp["proj.weight"] = "proj.weight"
        spec["bias"], spec["cortex_skip"] = 0, 1
        compare(
            f"cortex TransformerBlock d={dd} out={out} heads={h} causal={causal}",
            TBW(t),
            spec,
            "seq",
            (9, dd),
            rows=rows,
            atol=1e-5,
            mapping=mp,
        )

    class CW(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m((x, torch.ones(x.shape[0], x.shape[2])))[0]

    for i, o, k, dil in ((16, 16, 3, 1), (16, 32, 5, 2)):
        c = C1(i, o, k, layernorm=True, dropout_p=0.0, act_fn="swish", dilation=dil)
        mp = {"conv_1.weight": "f.2.weight", "conv_2.weight": "f.5.weight", "norm_1.weight": "f.0.w", "norm_1.bias": "f.0.b",
              "norm_2.weight": "f.3.w", "norm_2.bias": "f.3.b"}  # fmt: skip
        if i != o:
            mp.update({"proj.weight": "proj.weight", "proj.bias": "proj.bias"})
        compare(
            f"cortex Conv1dResidBlock {i}->{o} k={k} dil={dil}",
            CW(c),
            {
                "op": "conv1d_resid",
                "out": 0 if i == o else o,
                "k": k,
                "dilation": dil,
                "norm": 1,
                "act": "swish",
                "dropout": 0.0,
            },
            "ser",
            (i, 20),
            rows=rows,
            atol=1e-4,
            mapping=mp,
        )

    # ---- pytorch/vision blocks (weights copied in definition order)
    from torchvision.models import convnext, efficientnet, resnet, shufflenetv2, squeezenet
    from torchvision.ops import SqueezeExcitation
    from torchvision.ops.misc import Conv2dNormActivation

    def tv(name, t, spec, dims=(64, 16, 16)):
        compare(name, t, spec, "map", dims, rows=rows)

    sc = nn.Sequential(nn.Conv2d(64, 128, 1, 2, bias=False), nn.BatchNorm2d(128))
    tv(
        "resnet BasicBlock 64->128 s2",
        resnet.BasicBlock(64, 128, 2, sc),
        {"op": "basic_block", "out": 128, "s": 2, "act": "relu", "norm": "batch"},
    )
    sc = nn.Sequential(nn.Conv2d(64, 256, 1, 1, bias=False), nn.BatchNorm2d(256))
    tv(
        "resnext50 Bottleneck 64->256",
        resnet.Bottleneck(64, 64, 1, sc, 32, 4),
        {
            "op": "bottleneck",
            "out": 256,
            "s": 1,
            "ratio": 4,
            "groups": 32,
            "base_width": 4,
            "act": "relu",
            "norm": "batch",
        },
    )
    cn = {"block.0.weight": "dw.weight", "block.0.bias": "dw.bias", "block.2.weight": "ln.weight", "block.2.bias": "ln.bias",
          "block.3.weight": "p1.weight", "block.3.bias": "p1.bias", "block.5.weight": "p2.weight", "block.5.bias": "p2.bias", "layer_scale": "gamma"}  # fmt: skip
    compare(
        "convnext CNBlock",
        convnext.CNBlock(64, 0.0, 1e-3),
        {"op": "convnext_block", "k": 7, "ratio": 4, "layer_scale": 1e-6},
        "map",
        (64, 16, 16),
        rows=rows,
        mapping=cn,
    )
    tv(
        "efficientnet MBConv",
        efficientnet.MBConv(efficientnet.MBConvConfig(4, 3, 1, 64, 64, 1), 0.0, nn.BatchNorm2d),
        {"op": "mbconv", "out": 64, "expand": 4, "k": 3, "s": 1, "se": 1, "act": "swish"},
    )
    tv(
        "efficientnet FusedMBConv",
        efficientnet.FusedMBConv(
            efficientnet.FusedMBConvConfig(4, 3, 1, 64, 64, 1), 0.0, nn.BatchNorm2d
        ),
        {"op": "fused_mbconv", "out": 64, "expand": 4, "k": 3, "s": 1, "se": 0, "act": "swish"},
    )
    tv(
        "shufflenetv2 unit s2",
        shufflenetv2.InvertedResidual(64, 128, 2),
        {"op": "shuffle_unit", "out": 128, "s": 2},
    )
    tv(
        "squeezenet Fire",
        squeezenet.Fire(64, 16, 64, 64),
        {"op": "fire", "out": 128, "squeeze_ratio": 8},
    )
    tv("SqueezeExcitation", SqueezeExcitation(64, 16), {"op": "se", "ratio": 0.25})
    tv(
        "Conv2dNormActivation",
        Conv2dNormActivation(64, 128, 3, 2),
        {"op": "conv_bn_act", "out": 128, "k": 3, "s": 2, "act": "relu", "norm": "batch"},
    )

    # ---- blocks that were first written from the papers; torchvision has the original
    from functools import partial

    from torchvision.models import regnet
    from torchvision.models.mobilenetv3 import InvertedResidual as V3Block
    from torchvision.models.mobilenetv3 import InvertedResidualConfig
    from torchvision.models.swin_transformer import PatchMerging
    from torchvision.models.vision_transformer import EncoderBlock

    cfg3 = InvertedResidualConfig(64, 3, 256, 64, True, "HS", 1, 1, 1.0)
    tv(
        "mobilenetv3 InvertedResidual e4 HS se (default BN eps; torchvision uses 1e-3)",
        V3Block(cfg3, nn.BatchNorm2d),
        {"op": "mbconv", "out": 64, "expand": 4, "k": 3, "s": 1, "se": 1, "act": "hswish"},
    )
    rb = regnet.ResBottleneckBlock(
        64, 128, 2, nn.BatchNorm2d, nn.ReLU, group_width=8, bottleneck_multiplier=1.0, se_ratio=0.25
    )
    compare(
        "regnet ResBottleneckBlock 64->128 s2 se",
        rb,
        {"op": "regnet_block", "out": 128, "s": 2, "group_width": 8, "ratio": 1.0, "se": 0.25},
        "map",
        (64, 16, 16),
        rows=rows,
        fn=tensors_proj_last,
    )

    class VW(
        nn.Module
    ):  # torchvision EncoderBlock expects (B, T, D) and has a no-op dropout in eval
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)

    vit = EncoderBlock(4, 32, 128, 0.0, 0.0, norm_layer=partial(nn.LayerNorm, eps=1e-5))
    mp = {"ln_1.weight": "ln1.weight", "ln_1.bias": "ln1.bias", "self_attention.in_proj_weight": "attn.qkv.weight", "self_attention.in_proj_bias": "attn.qkv.bias",
          "self_attention.out_proj.weight": "attn.proj.weight", "self_attention.out_proj.bias": "attn.proj.bias", "ln_2.weight": "ln2.weight", "ln_2.bias": "ln2.bias",
          "mlp.0.weight": "mlp.0.weight", "mlp.0.bias": "mlp.0.bias", "mlp.3.weight": "mlp.2.weight", "mlp.3.bias": "mlp.2.bias"}  # fmt: skip
    compare(
        "torchvision ViT EncoderBlock d=32 heads=4 mlp=128",
        VW(vit),
        {
            "op": "transformer_block",
            "heads": 4,
            "ratio": 4,
            "out": 0,
            "dropout": 0.0,
            "causal": 0,
            "bias": 1,
            "cortex_skip": 0,
        },
        "seq",
        (9, 32),
        rows=rows,
        atol=1e-5,
        mapping=mp,
    )

    class PM(nn.Module):  # torchvision PatchMerging is channels-last
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    compare("torchvision Swin PatchMerging", PM(PatchMerging(64)), {"op": "patch_merge"}, "map", (64, 16, 16), rows=rows,
            mapping={"norm.weight": "norm.weight", "norm.bias": "norm.bias", "reduction.weight": "red.weight"})  # fmt: skip

    print("| block | result | detail |\n|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    bad = [r for r in rows if r[1] != "OK"]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} numerically equivalent")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
