"""Numerical equivalence of zoo blocks with timm, torchvision, segmentation_models_pytorch and MONAI.

Install the reference libraries outside the project (they are not DLNet dependencies), e.g.

    pip install --no-deps --target /tmp/vlibs timm==1.0.31 segmentation-models-pytorch==0.6.0 \
        monai==1.6.1 torchvision==0.29.2 safetensors huggingface_hub pyyaml
    PYTHONPATH=/tmp/vlibs python scripts/verify_vision_libs.py

Every block is built twice with the same configuration; weights (randomised, BN statistics
included) are copied from the reference into the zoo block in definition order, and the eval
outputs on the same input are compared. Results: THIRD_PARTY.md.
"""

import sys
from functools import partial

import torch
from torch import nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from monai.networks.blocks.dynunet_block import UnetResBlock
from monai.networks.blocks.segresnet_block import ResBlock
from segmentation_models_pytorch.base.modules import SCSEModule
from timm.layers import BatchNormAct2d, PatchEmbed, create_conv2d
from timm.layers.cbam import CbamModule
from timm.layers.coord_attn import CoordAttn
from timm.layers.eca import EcaModule
from timm.layers.global_context import GlobalContext
from timm.layers.grn import GlobalResponseNorm
from timm.layers.norm import GroupNorm1
from timm.layers.selective_kernel import SelectiveKernel
from timm.layers.split_attn import SplitAttn
from timm.models._efficientnet_blocks import DepthwiseSeparableConv
from timm.models.byobnet import RepVggBlock
from timm.models.ghostnet import GhostBottleneck
from timm.models.metaformer import MetaFormerBlock, Pooling
from timm.models.mlp_mixer import MixerBlock
from timm.models.res2net import Bottle2neck
from timm.models.resnetv2 import DownsampleConv, PreActBasic
from torchvision.models.googlenet import Inception
from torchvision.models.swin_transformer import ShiftedWindowAttention

from dlnet.zoo.core import _build


def tens(m):
    return list(m.parameters()) + [b for n, b in m.named_buffers() if "num_batches" not in n]


def short_first(m):
    ps = [(n, p) for n, p in m.named_parameters()]
    bs = [(n, b) for n, b in m.named_buffers() if "num_batches" not in n]
    key = lambda nb: not nb[0].startswith("short")
    return [p for _, p in sorted(ps, key=key)] + [b for _, b in sorted(bs, key=key)]


def copy(a, b, fb=None):
    ta, tb = tens(a), (fb or tens)(b)
    if [t.shape for t in ta] != [t.shape for t in tb]:
        raise ValueError(
            f"{[tuple(t.shape) for t in ta][:8]} vs {[tuple(t.shape) for t in tb][:8]}"
        )
    with torch.no_grad():
        for x, y in zip(ta, tb):
            y.copy_(x if x.dtype.is_floating_point or not y.dtype.is_floating_point else x.float())


def rnd(m):
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        for n, b in m.named_buffers():
            if "running_var" in n:
                b.copy_(torch.rand(b.shape, generator=g) + 0.5)
            elif "running_mean" in n:
                b.copy_(torch.randn(b.shape, generator=g) * 0.1)


def mine(spec, kind, dims):
    return _build(spec, kind, dims)[0]


C, H = 32, 16
cases = [
    ("cbam", CbamModule(C, rd_ratio=1 / 8), {"op": "cbam", "ratio": 8}),
    ("eca adaptive", EcaModule(C), {"op": "eca", "k": 0}),
    ("eca k5", EcaModule(kernel_size=5), {"op": "eca", "k": 5}),
    ("coord_attn", CoordAttn(C), {"op": "coord_attn", "rd_ratio": 1 / 16}),
    ("gc_block", GlobalContext(C), {"op": "gc_block", "rd_ratio": 1 / 8}),
    ("grn", GlobalResponseNorm(C, channels_last=False), {"op": "grn"}),
    ("scse", SCSEModule(C, 16), {"op": "scse", "reduction": 16}),
    ("sk_conv", SelectiveKernel(C, 64, stride=2), {"op": "sk_conv", "out": 64, "s": 2}),
    (
        "split_attn",
        SplitAttn(C, 64, radix=2, norm_layer=nn.BatchNorm2d),
        {"op": "split_attn", "out": 64, "s": 1, "radix": 2},
    ),
    (
        "split_attn s2",
        SplitAttn(C, 64, stride=2, radix=2, norm_layer=nn.BatchNorm2d),
        {"op": "split_attn", "out": 64, "s": 2, "radix": 2},
    ),
    (
        "ghost_bottleneck",
        GhostBottleneck(C, 96, 48, 3, 2),
        {"op": "ghost_bottleneck", "out": 48, "expand": 3, "s": 2, "k": 3},
    ),
    (
        "ghost_bottleneck id",
        GhostBottleneck(C, 64, C, 3, 1),
        {"op": "ghost_bottleneck", "out": C, "expand": 2, "s": 1, "k": 3},
    ),
    (
        "res2net_block",
        Bottle2neck(
            C,
            16,
            2,
            downsample=nn.Sequential(nn.Conv2d(C, 64, 1, 2, bias=False), nn.BatchNorm2d(64)),
            norm_layer=nn.BatchNorm2d,
        ),
        {"op": "res2net_block", "out": 64, "s": 2, "scale": 4, "base_width": 26},
    ),
    (
        "res2net_block id",
        Bottle2neck(64, 16, 1, norm_layer=nn.BatchNorm2d),
        {"op": "res2net_block", "out": 64, "s": 1, "scale": 4, "base_width": 26},
    ),
    ("repvgg_block", RepVggBlock(C, C), {"op": "repvgg_block", "out": C, "s": 1}),
    ("repvgg_block s2", RepVggBlock(C, 64, stride=2), {"op": "repvgg_block", "out": 64, "s": 2}),
    (
        "poolformer_block",
        MetaFormerBlock(
            C,
            token_mixer=Pooling,
            mlp_act=nn.GELU,
            mlp_bias=True,
            norm_layer=GroupNorm1,
            layer_scale_init_value=1e-5,
        ),
        {"op": "poolformer_block", "pool": 3, "ratio": 4},
    ),
    (
        "dw_sep",
        DepthwiseSeparableConv(C, 64, 3, stride=2, pw_act=True, noskip=True),
        {"op": "dw_sep", "out": 64, "k": 3, "s": 2, "act": "relu"},
    ),
    (
        "preact_block",
        PreActBasic(
            C,
            64,
            stride=2,
            conv_layer=partial(create_conv2d, bias=False),
            norm_layer=BatchNormAct2d,
            proj_layer=DownsampleConv,
        ),
        {"op": "preact_block", "out": 64, "s": 2, "act": "relu", "norm": "batch"},
    ),
    ("inception", Inception(C, 32, 48, 64, 8, 16, 16), {"op": "inception", "out": 128, "k5": 3}),
    ("segres_block", ResBlock(2, C, ("GROUP", {"num_groups": 8})), {"op": "segres_block", "k": 3}),
    (
        "unet_res_block",
        UnetResBlock(2, C, 64, 3, 2, "instance"),
        {"op": "unet_res_block", "out": 64, "k": 3, "s": 2},
    ),
    (
        "unet_res_block 3d",
        UnetResBlock(3, C, C, 3, 1, "instance"),
        {"op": "unet_res_block", "out": C, "k": 3, "s": 1},
    ),
]
ok = 0
for name, ref, spec in cases:
    kind = "vol" if "3d" in name else "map"
    dims = (C, 8, 8, 8) if kind == "vol" else (C, H, H)
    if name == "res2net_block id":
        dims = (64, H, H)
    try:
        m = mine(spec, kind, dims)
        rnd(ref)
        copy(ref, m, short_first if name.startswith("preact") else None)
        ref.eval()
        m.eval()
        x = torch.randn(2, *dims, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            d = (ref(x) - m(x)).abs().max().item()
        ok += d < 1e-4
        print(f"{name:22s} maxdiff={d:.2e}")
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"{name:22s} FAILED {str(e)[:200]}")
# repvgg fuse
m = mine({"op": "repvgg_block", "out": C, "s": 1}, "map", (C, H, H))
rnd(m)
m.eval()
x = torch.randn(2, C, H, H)
with torch.no_grad():
    y0 = m(x)
    m.fuse()
    d = (m(x) - y0).abs().max().item()
print("repvgg fuse", d)
ok += d < 1e-4
# mixer
ref = MixerBlock(48, 64)
m = mine({"op": "mixer_block", "token_ratio": 0.5, "channel_ratio": 4}, "seq", (64, 48))
rnd(ref)
copy(ref, m)
x = torch.randn(2, 64, 48)
with torch.no_grad():
    d = (ref(x) - m(x)).abs().max().item()
print("mixer_block", d)
ok += d < 1e-4
# patchify
ref = PatchEmbed(16, 4, C, 64)
m = mine({"op": "patchify", "dim": 64, "patch": 4}, "map", (C, 16, 16))
rnd(ref)
copy(ref, m)
x = torch.randn(2, C, 16, 16)
with torch.no_grad():
    d = (ref(x) - m(x)).abs().max().item()
print("patchify", d)
ok += d < 1e-5
# glu
m = mine({"op": "glu", "out": 32}, "vec", (24,))
x = torch.randn(4, 24)
with torch.no_grad():
    d = (nn.functional.glu(m.f(x), -1) - m(x)).abs().max().item()
print("glu", d)
ok += d < 1e-6
# window attention (shifted and not, with padding-free windows)
for shift in (0, 2):
    ref = ShiftedWindowAttention(C, [4, 4], [shift, shift], 4)
    m = mine(
        {"op": "window_attn", "heads": 4, "window": 4, "shift": int(shift > 0)}, "map", (C, H, H)
    )
    rnd(ref)
    copy(ref, m.attn)
    x = torch.randn(2, C, H, H)
    with torch.no_grad():
        z = x.permute(0, 2, 3, 1)
        want = x + ref(m.ln(z)).permute(0, 3, 1, 2)
        d = (want - m(x)).abs().max().item()
    print("window_attn shift", shift, d)
    ok += d < 1e-4
n_ops = len(cases) + 6
print("blocks OK", ok, "of", n_ops)
ok_blocks = ok


def _decoders():
    from segmentation_models_pytorch.decoders.deeplabv3.decoder import (
        DeepLabV3Decoder,
        DeepLabV3PlusDecoder,
    )
    from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
    from segmentation_models_pytorch.decoders.linknet.decoder import LinknetDecoder
    from segmentation_models_pytorch.decoders.pspnet.decoder import PSPDecoder
    from segmentation_models_pytorch.decoders.segformer.decoder import SegformerDecoder
    from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
    from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
    from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder

    from dlnet.zoo import dense as D

    def feats(ch, strides, s=64):
        g = torch.Generator().manual_seed(0)
        return [torch.randn(2, c, s // st, s // st, generator=g) for c, st in zip(ch, strides)]

    ch = (3, 32, 24, 40, 64, 128)
    st = (1, 2, 4, 8, 16, 32)
    cases = {
        "unet": (UnetDecoder(ch, (64, 48, 32, 24, 16), 5), D.UnetDecoder(ch, (64, 48, 32, 24, 16))),
        "unet_scse": (
            UnetDecoder(ch, (64, 48, 32, 24, 16), 5, attention_type="scse"),
            D.UnetDecoder(ch, (64, 48, 32, 24, 16), "scse"),
        ),
        "unetpp": (
            UnetPlusPlusDecoder(ch, (64, 48, 32, 24, 16), 5),
            D.UnetPPDecoder(ch, (64, 48, 32, 24, 16)),
        ),
        "fpn": (FPNDecoder(ch, 5, 64, 32, 0.2), D.FPNDecoder(ch, 64, 32, 0.2)),
        "deeplabv3": (
            DeepLabV3Decoder(128, 32, (12, 24, 36), False, 0.5),
            D.DeepLabV3Decoder(ch, 32, (12, 24, 36)),
        ),
        "pspnet": (PSPDecoder(ch, out_channels=32), D.PSPDecoder(ch, 32)),
        "linknet": (LinknetDecoder(ch, 16, 5), D.LinknetDecoder(ch, 16)),
        "upernet": (UPerNetDecoder(ch, 5, 32), D.UPerNetDecoder(ch, 32)),
        "segformer": (SegformerDecoder(ch, 5, 32), D.SegformerDecoder(ch, 32)),
    }
    ok = 0
    for name, (a, b) in cases.items():
        copy(a, b)
        a.eval()
        b.eval()
        f = feats(ch, st, 128 if name == "deeplabv3" else 64)
        with torch.no_grad():
            ya, yb = a([t.clone() for t in f]), b([t.clone() for t in f])
        d = (ya - yb).abs().max().item()
        ok += d < 1e-5
        print(f"{name:12s} {tuple(ya.shape)} maxdiff={d:.2e}")
    # deeplabv3+ with output stride 16
    st16 = (1, 2, 4, 8, 16, 16)
    a = DeepLabV3PlusDecoder(ch, 5, 32, (12, 24, 36), 16, True, 0.5)
    b = D.DeepLabV3PlusDecoder(ch, 32, (12, 24, 36), 2)
    copy(a, b)
    a.eval()
    b.eval()
    f = feats(ch, st16, 128)
    with torch.no_grad():
        d = (a(f) - b(f)).abs().max().item()
    print("deeplabv3plus", d)
    ok += d < 1e-5
    print("OK", ok, "of", len(cases) + 1)

    return ok, len(cases) + 1


d_ok, d_n = _decoders()
print("all OK", ok_blocks + d_ok, "of", n_ops + d_n)
raise SystemExit(0 if ok_blocks + d_ok == n_ops + d_n else 1)
