"""Primitive layers for paper cards (dlnet/paper.py): exactly what a block diagram names.

The blocks of the zoo bundle choices (conv_bn_act: convolution without bias + norm + activation).
A published network often differs in such a detail (a bias before batch norm, an explicit
padding), and a card must be able to write it as the authors did. These ops carry the tag
"primitive" only, so the random grammar and the mutations of the search never pick them.
"""

from __future__ import annotations

from torch import nn

from .core import ND, op
from .ops_basic import CONV


@op("conv", ("ser", "map", "vol"), space={"out": [16, 32, 64], "k": [3, 1, 5, 7], "s": [1, 2], "pad": ["same", 0, 1],
                                          "groups": [1, "depthwise"], "dilation": [1, 2], "bias": [True, False]},
    tags={"primitive"}, source="torch.nn.Conv1d/2d/3d, for paper cards")  # fmt: skip
def _conv(dims, kind, out, k, s, pad, groups, dilation, bias):
    """A plain convolution: no norm, no activation. pad "same" = dilation * (k - 1) // 2."""
    c = dims[0]
    g = c if groups == "depthwise" else int(groups)
    if c % g or out % g:
        raise ValueError(f"conv: {c} -> {out} channels cannot be split into {g} groups")
    p = dilation * (k - 1) // 2 if pad == "same" else pad
    return CONV[ND[kind]](c, out, k, s, p, dilation, g, bias=bool(bias))


@op("linear", ("vec", "seq"), space={"out": [64, 32, 128, 256], "bias": [True, False]}, tags={"primitive"},
    source="torch.nn.Linear, for paper cards")  # fmt: skip
def _linear(dims, kind, out, bias):
    """A plain linear layer on the last axis: no activation."""
    return nn.Linear(dims[-1], out, bias=bool(bias))


@op("maxpool", ("ser", "map", "vol"), space={"k": [2, 3], "s": [2, 1], "pad": [0, 1], "ceil": [False, True]},
    tags={"primitive"}, source="torch.nn.MaxPool1d/2d/3d, for paper cards")  # fmt: skip
def _maxpool(dims, kind, k, s, pad, ceil):
    """Max pooling exactly as torch.nn writes it (kernel, stride, padding, ceil_mode)."""
    from .ops_basic import MAXP

    return MAXP[ND[kind]](k, s, pad, ceil_mode=bool(ceil))


@op("avgpool", ("ser", "map", "vol"), space={"k": [2, 3], "s": [2, 1], "pad": [0, 1], "ceil": [False, True]},
    tags={"primitive"}, source="torch.nn.AvgPool1d/2d/3d, for paper cards")  # fmt: skip
def _avgpool(dims, kind, k, s, pad, ceil):
    """Average pooling exactly as torch.nn writes it (kernel, stride, padding, ceil_mode)."""
    from .ops_basic import AVGP

    return AVGP[ND[kind]](k, s, pad, ceil_mode=bool(ceil))
