"""Super-resolution blocks (v2.0, hyperspectral super-resolution plugin `hsi_sr`).

Every block keeps the shape of its input (kind `map`: features, H, W); the upsampling is the
`srN` head of `core.Head` (convolution + pixel shuffle + bicubic global residual). Written after
the structure of the papers and of the authors' public code named below; NOT compared numerically
with that code (THIRD_PARTY.md, "v2.0"):

- edsr_block: EDSR residual block without batch norm, residual scaled by `res_scale` (Lim et al.,
  CVPRW 2017; sanghyun-son/EDSR-PyTorch src/model/common.py ResBlock).
- rcab: residual channel attention block (Zhang et al., ECCV 2018; yulunzhang/RCAN
  RCAN_TrainCode/code/model/rcan.py RCAB, CALayer).
- ssb: spatial-spectral block of SSPSR (Jiang et al., IEEE TCI 2020; junjun-jiang/SSPSR
  SSPSR.py SSB = spatial ResBlock (3x3) then spectral ResAttentionBlock (1x1 + channel attention)).
- sr3d: 3-D fully convolutional network on the cube (Mei et al., Remote Sensing 2017, 3D-FCNN):
  the bands are a third convolution axis, so neighbouring bands share filters; output is one
  feature map per band, added to the input (residual).
"""

from __future__ import annotations

from torch import nn

from .core import op

SR = {"sr"}  # tag: not in the random grammar of the other domains


def _conv(cin, cout, k=3):
    return nn.Conv2d(cin, cout, k, padding=k // 2)


class EDSRBlock(nn.Module):
    def __init__(self, ch, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(_conv(ch, ch), nn.ReLU(inplace=True), _conv(ch, ch))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class ChannelAttention(nn.Module):
    """RCAN CALayer: global average pool, 1x1 down, ReLU, 1x1 up, sigmoid gate."""

    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, mid, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(mid, ch, 1), nn.Sigmoid())  # fmt: skip

    def forward(self, x):
        return x * self.gate(x)


class RCAB(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.body = nn.Sequential(_conv(ch, ch), nn.ReLU(inplace=True), _conv(ch, ch),
                                  ChannelAttention(ch, reduction))  # fmt: skip

    def forward(self, x):
        return x + self.body(x)


class SSB(nn.Module):
    def __init__(self, ch, res_scale=0.1, reduction=16):
        super().__init__()
        self.spa = EDSRBlock(ch, res_scale)
        self.spc = nn.Sequential(_conv(ch, ch, 1), nn.ReLU(inplace=True), _conv(ch, ch, 1),
                                 ChannelAttention(ch, reduction))  # fmt: skip
        self.res_scale = res_scale

    def forward(self, x):
        x = self.spa(x)
        return x + self.spc(x) * self.res_scale


class SR3D(nn.Module):
    def __init__(self, width=16, depth=3):
        super().__init__()
        layers, c = [], 1
        for _ in range(depth):
            layers += [nn.Conv3d(c, width, 3, padding=1), nn.ReLU(inplace=True)]
            c = width
        layers.append(nn.Conv3d(c, 1, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.body(x.unsqueeze(1)).squeeze(1)


@op("edsr_block", "map", space={"res_scale": [0.1, 1.0]}, tags=SR,
    source="sanghyun-son/EDSR-PyTorch src/model/common.py ResBlock (EDSR, CVPRW 2017)")  # fmt: skip
def _edsr_block(dims, kind, res_scale):
    """conv3x3 - ReLU - conv3x3, no batch norm, x + res_scale * body(x)."""
    return EDSRBlock(dims[0], float(res_scale))


@op("rcab", "map", space={"reduction": [16, 8]}, tags=SR,
    source="yulunzhang/RCAN RCAN_TrainCode/code/model/rcan.py RCAB, CALayer (ECCV 2018)")  # fmt: skip
def _rcab(dims, kind, reduction):
    """Residual channel attention block: conv - ReLU - conv - channel attention, plus the input."""
    return RCAB(dims[0], int(reduction))


@op("ssb", "map", space={"res_scale": [0.1, 1.0], "reduction": [16, 8]}, tags=SR,
    source="junjun-jiang/SSPSR SSPSR.py SSB (IEEE TCI 2020)")  # fmt: skip
def _ssb(dims, kind, res_scale, reduction):
    """Spatial residual block (3x3) then spectral residual attention block (1x1 + channel attention)."""
    return SSB(dims[0], float(res_scale), int(reduction))


@op("sr3d", "map", space={"width": [8, 16], "depth": [2, 3]}, tags=SR,
    source="3D-FCNN (Mei et al., Remote Sensing 2017), written from the paper")  # fmt: skip
def _sr3d(dims, kind, width, depth):
    """3-D convolutions over (bands, H, W) with one input channel, residual."""
    if int(depth) < 1:
        raise ValueError("sr3d needs depth >= 1")
    return SR3D(int(width), int(depth))
