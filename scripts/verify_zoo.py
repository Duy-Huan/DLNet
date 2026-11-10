"""Compare ported zoo blocks with the original torchvision blocks (parameter counts, output shapes).

    PYTHONPATH=/path/to/torchvision python scripts/verify_zoo.py

torchvision is NOT a dependency of DLNet: install it somewhere temporary, for example
    pip install --no-deps --target /tmp/tvpkg torchvision && PYTHONPATH=/tmp/tvpkg python scripts/verify_zoo.py
A match of parameter count and output shape means the structure is the same; it does not prove
the arithmetic is (weights are not copied across). Results: THIRD_PARTY.md.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from dlnet import zoo


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def ours(node, dims):
    return zoo.build_model(
        {"spec": node, "input": {"kind": "map", "dims": dims}, "head": "cls", "out_dim": 1}
    )


def body_params(node, dims):
    """Parameters of the block alone (spec body, no head)."""
    from dlnet.zoo.core import _build

    m, _, d = _build(node, "map", dims)
    return n_params(m), d


def main():
    from torchvision.models import convnext, efficientnet, regnet, resnet, shufflenetv2, squeezenet
    from torchvision.models.mobilenetv3 import InvertedResidual as V3Block
    from torchvision.models.mobilenetv3 import InvertedResidualConfig
    from torchvision.ops import SqueezeExcitation
    from torchvision.ops.misc import Conv2dNormActivation

    rows, X = [], torch.randn(2, 64, 16, 16)

    def add(name, theirs, node, dims=(64, 16, 16), x=None):
        t = theirs.eval()
        y = t(X if x is None else x)
        p, d = body_params(node, dims)
        rows.append((name, n_params(t), p, tuple(y.shape[1:]), tuple(d)))

    # ResNet basic block: conv3x3-bn-relu-conv3x3-bn + 1x1 projection shortcut
    sc = nn.Sequential(nn.Conv2d(64, 128, 1, 2, bias=False), nn.BatchNorm2d(128))
    add(
        "resnet BasicBlock 64->128 s2",
        resnet.BasicBlock(64, 128, 2, sc),
        {"op": "basic_block", "out": 128, "s": 2, "act": "relu", "norm": "batch"},
    )
    # ResNet / ResNeXt / Wide bottleneck; torchvision `planes` = our out / 4
    for name, g, bw in (
        ("resnet50", 1, 64),
        ("resnext50_32x4d", 32, 4),
        ("wide_resnet50_2", 1, 128),
    ):
        sc = nn.Sequential(nn.Conv2d(64, 256, 1, 1, bias=False), nn.BatchNorm2d(256))
        add(
            f"{name} Bottleneck 64->256",
            resnet.Bottleneck(64, 64, 1, sc, g, bw),
            {
                "op": "bottleneck",
                "out": 256,
                "s": 1,
                "ratio": 4,
                "groups": g,
                "base_width": bw,
                "act": "relu",
                "norm": "batch",
            },
        )
    # ConvNeXt block, dim 64
    add(
        "convnext CNBlock 64",
        convnext.CNBlock(64, 0.0, 1e-6),
        {"op": "convnext_block", "k": 7, "ratio": 4, "layer_scale": 1e-6},
    )
    # EfficientNet MBConv 64->64 expand 4 k3 s1 with SE
    cfg = efficientnet.MBConvConfig(4, 3, 1, 64, 64, 1)
    add(
        "efficientnet MBConv e4 k3 se",
        efficientnet.MBConv(cfg, 0.0, nn.BatchNorm2d),
        {"op": "mbconv", "out": 64, "expand": 4, "k": 3, "s": 1, "se": 1, "act": "swish"},
    )
    fcfg = efficientnet.FusedMBConvConfig(4, 3, 1, 64, 64, 1)
    add(
        "efficientnet FusedMBConv e4 k3",
        efficientnet.FusedMBConv(fcfg, 0.0, nn.BatchNorm2d),
        {"op": "fused_mbconv", "out": 64, "expand": 4, "k": 3, "s": 1, "se": 0, "act": "swish"},
    )
    # RegNet block (group width 8, bottleneck ratio 1.0, SE ratio 0.25)
    rb = regnet.ResBottleneckBlock(
        64, 128, 2, nn.BatchNorm2d, nn.ReLU, group_width=8, bottleneck_multiplier=1.0, se_ratio=0.25
    )
    add(
        "regnet ResBottleneckBlock 64->128 s2 se",
        rb,
        {"op": "regnet_block", "out": 128, "s": 2, "group_width": 8, "ratio": 1.0, "se": 0.25},
    )
    # ShuffleNetV2 unit stride 2 and stride 1
    add(
        "shufflenetv2 InvertedResidual 64->128 s2",
        shufflenetv2.InvertedResidual(64, 128, 2),
        {"op": "shuffle_unit", "out": 128, "s": 2},
    )
    add(
        "shufflenetv2 InvertedResidual 64->64 s1",
        shufflenetv2.InvertedResidual(64, 64, 1),
        {"op": "shuffle_unit", "out": 64, "s": 1},
    )
    # SqueezeNet Fire 64 -> squeeze 16, expand 64 + 64
    add(
        "squeezenet Fire 64 (16, 64, 64)",
        squeezenet.Fire(64, 16, 64, 64),
        {"op": "fire", "out": 128, "squeeze_ratio": 8},
    )
    # Squeeze-and-excitation: torchvision SqueezeExcitation(64, 16) vs ours ratio 0.25
    add("SqueezeExcitation 64 (16)", SqueezeExcitation(64, 16), {"op": "se", "ratio": 0.25})
    # Conv2dNormActivation 3x3 64->128 s2
    add(
        "Conv2dNormActivation 3x3 s2",
        Conv2dNormActivation(64, 128, 3, 2),
        {"op": "conv_bn_act", "out": 128, "k": 3, "s": 2, "act": "relu", "norm": "batch"},
    )
    # MobileNetV3 inverted residual with hard-swish and SE (expand 4: 64 -> 256)
    cfg3 = InvertedResidualConfig(64, 3, 256, 64, True, "HS", 1, 1, 1.0)
    add(
        "mobilenetv3 InvertedResidual e4 HS se",
        V3Block(cfg3, partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)),
        {"op": "mbconv", "out": 64, "expand": 4, "k": 3, "s": 1, "se": 1, "act": "hswish"},
    )

    print("| block | torchvision params | zoo params | same? | torchvision out | zoo out |")
    print("|---|---|---|---|---|---|")
    for name, a, b, ya, yb in rows:
        print(
            f"| {name} | {a:,} | {b:,} | {'yes' if a == b else f'NO ({b - a:+,})'} | {ya} | {yb} |"
        )


if __name__ == "__main__":
    main()
