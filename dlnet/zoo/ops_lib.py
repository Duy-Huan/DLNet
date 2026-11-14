"""Library layers that published networks use as one unit (v1.8), ported from their source.

torchvision's Swin Transformer (BSD-3-Clause, pytorch/vision torchvision/models/swin_transformer.py,
v0.29): `shifted_window_attention` with its relative position bias (Swin v1: a learned table
indexed by relative position; Swin v2: cosine attention with a learned logit scale and a
continuous position bias from a small MLP), and `_patch_merging_pad`. The code below follows that
file line by line, with the same attribute names, so the weights of a torchvision model load into
it by name (`dlnet paper import`, `paper verify`).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .core import KINDS, op


def _patch_merging_pad(x):
    h, w, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
    x0 = x[..., 0::2, 0::2, :]
    x1 = x[..., 1::2, 0::2, :]
    x2 = x[..., 0::2, 1::2, :]
    x3 = x[..., 1::2, 1::2, :]
    return torch.cat([x0, x1, x2, x3], -1)


@op("patch_merge_pad", KINDS, "nd", space={}, tags={"primitive"},
    source="torchvision swin_transformer._patch_merging_pad (BSD-3)")  # fmt: skip
def _patch_merge_pad(dims, kind):
    """(..., H, W, C) -> (..., H/2, W/2, 4C): the 2x2 neighbours side by side (Swin patch merging)."""
    return PatchMergePad()


class PatchMergePad(nn.Module):
    """torchvision's _patch_merging_pad as a layer."""

    def forward(self, x):
        return _patch_merging_pad(x)


def _rel_bias(table, index, window):
    n = window[0] * window[1]
    b = table[index].view(n, n, -1)
    return b.permute(2, 0, 1).contiguous().unsqueeze(0)


def shifted_window_attention(x, qkv_weight, proj_weight, bias, window, heads, shift, qkv_bias=None,
                             proj_bias=None, logit_scale=None):  # fmt: skip
    """torchvision.models.swin_transformer.shifted_window_attention at inference (no dropout)."""
    b, h, w, c = x.shape
    pad_r = (window[1] - w % window[1]) % window[1]
    pad_b = (window[0] - h % window[0]) % window[0]
    x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
    _, ph, pw, _ = x.shape
    shift = list(shift)
    if window[0] >= ph:
        shift[0] = 0
    if window[1] >= pw:
        shift[1] = 0
    if sum(shift) > 0:
        x = torch.roll(x, shifts=(-shift[0], -shift[1]), dims=(1, 2))
    nw = (ph // window[0]) * (pw // window[1])
    x = x.view(b, ph // window[0], window[0], pw // window[1], window[1], c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * nw, window[0] * window[1], c)
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        n = qkv_bias.numel() // 3
        qkv_bias[n : 2 * n].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, heads, c // heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn * torch.clamp(logit_scale, max=math.log(100.0)).exp()
    else:
        q = q * (c // heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    attn = attn + bias
    if sum(shift) > 0:
        mask = x.new_zeros((ph, pw))
        hs = ((0, -window[0]), (-window[0], -shift[0]), (-shift[0], None))
        ws = ((0, -window[1]), (-window[1], -shift[1]), (-shift[1], None))
        count = 0
        for a in hs:
            for z in ws:
                mask[a[0] : a[1], z[0] : z[1]] = count
                count += 1
        mask = mask.view(ph // window[0], window[0], pw // window[1], window[1])
        mask = mask.permute(0, 2, 1, 3).reshape(nw, window[0] * window[1])
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        attn = attn.view(x.size(0) // nw, nw, heads, x.size(1), x.size(1))
        attn = attn + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, heads, x.size(1), x.size(1))
    attn = F.softmax(attn, dim=-1)
    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
    x = F.linear(x, proj_weight, proj_bias)
    x = x.view(b, ph // window[0], pw // window[1], window[0], window[1], c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, ph, pw, c)
    if sum(shift) > 0:
        x = torch.roll(x, shifts=(shift[0], shift[1]), dims=(1, 2))
    return x[:, :h, :w, :].contiguous()


class WindowAttention(nn.Module):
    """torchvision ShiftedWindowAttention (version 1) / ShiftedWindowAttentionV2 (version 2) on a
    channels-last map (H, W, C), with torchvision's attribute names."""

    def __init__(self, dim, window, shift, heads, qkv_bias=True, proj_bias=True, version=1):
        super().__init__()
        self.window_size, self.shift_size, self.num_heads, self.version = (
            list(window),
            list(shift),
            heads,
            version,
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        ws = self.window_size
        if version == 1:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * ws[0] - 1) * (2 * ws[1] - 1), heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            ch = torch.arange(-(ws[0] - 1), ws[0], dtype=torch.float32)
            cw = torch.arange(-(ws[1] - 1), ws[1], dtype=torch.float32)
            t = (
                torch.stack(torch.meshgrid([ch, cw], indexing="ij"))
                .permute(1, 2, 0)
                .contiguous()
                .unsqueeze(0)
            )
            t[:, :, :, 0] /= max(ws[0] - 1, 1)
            t[:, :, :, 1] /= max(ws[1] - 1, 1)
            t *= 8
            t = torch.sign(t) * torch.log2(torch.abs(t) + 1.0) / 3.0
            self.register_buffer("relative_coords_table", t)
        coords = torch.stack(
            torch.meshgrid(torch.arange(ws[0]), torch.arange(ws[1]), indexing="ij")
        )
        flat = torch.flatten(coords, 1)
        rel = (flat[:, :, None] - flat[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws[0] - 1
        rel[:, :, 1] += ws[1] - 1
        rel[:, :, 0] *= 2 * ws[1] - 1
        self.register_buffer("relative_position_index", rel.sum(-1).flatten())
        if version == 2:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((heads, 1, 1))))
            self.cpb_mlp = nn.Sequential(
                nn.Linear(2, 512, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(512, heads, bias=False),
            )
            if qkv_bias:
                n = self.qkv.bias.numel() // 3
                self.qkv.bias[n : 2 * n].data.zero_()

    def relative_position_bias(self):
        if self.version == 1:
            return _rel_bias(
                self.relative_position_bias_table, self.relative_position_index, self.window_size
            )
        table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        return 16 * torch.sigmoid(_rel_bias(table, self.relative_position_index, self.window_size))

    def forward(self, x):
        return shifted_window_attention(
            x, self.qkv.weight, self.proj.weight, self.relative_position_bias(), self.window_size, self.num_heads,
            self.shift_size, self.qkv.bias, self.proj.bias, self.logit_scale if self.version == 2 else None,
        )  # fmt: skip


@op("window_attention", KINDS, space={"heads": [3], "window": [[7, 7]], "shift": [[0, 0]], "qkv_bias": [True],
                                      "proj_bias": [True], "version": [1, 2]},
    tags={"primitive"}, source="torchvision ShiftedWindowAttention / V2 (Swin v1 / v2, BSD-3)")  # fmt: skip
def _window_attention(dims, kind, heads, window, shift, qkv_bias, proj_bias, version):
    """Shifted-window multi-head self-attention with relative position bias on (H, W, C)."""
    if len(dims) != 3:
        raise ValueError(f"window_attention takes a channels-last map (H, W, C), got {tuple(dims)}")
    return WindowAttention(
        dims[-1], window, shift, int(heads), bool(qkv_bias), bool(proj_bias), int(version)
    )
