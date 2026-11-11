"""Sequence mixers of 2024-2025 and rational KAN layers, written from their OFFICIAL code (v1.0).

Pure PyTorch: the CUDA/Triton kernels of the originals are replaced by their reference forms
(Gated DeltaNet: the repository's own `recurrent_gated_delta_rule_ref`; mLSTM: xlstm's
`parallel_stabilized_simple`; KAT: rational_kat_cu's `Rational_CUDA_A_1DGroup` torch path;
MambaVision: DLNet's selective scan, equal to mamba_ssm's `selective_scan_ref`). Sequential
recurrences are slow on long sequences: meant for the short sequences DLNet builds (<= a few
hundred steps). Numerical check: scripts/verify_seq2.py (job `verify-seq2`).

- Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025, arXiv:2412.06464; NVlabs/GatedDeltaNet
  lit_gpt/gated_delta_net.py GatedDeltaNet, use_mamba_gate, qk_norm "l2", fused RMSNorm-swish gate).
- mLSTM (Beck et al., NeurIPS 2024, arXiv:2405.04517; NX-AI/xlstm blocks/mlstm layer.py, cell.py,
  backends.py, components; block = xLSTM mLSTMBlock: LayerNorm, mLSTMLayer, residual).
- GR-KAN (Yang & Wang, ICLR 2025, arXiv:2409.10594; Adamdad/kat katransformer.py KAN with
  rational_kat_cu KAT_Group: rational P(x)/Q(x), 8 groups, numerator shared across groups,
  identity / GELU initialisation from init.json). FlashKAT (AAAI 2026) reports KAT trains far
  slower than an MLP of the same FLOPs: measure before using it widely.
- MambaVision (Hatamizadeh & Kautz, CVPR 2025, arXiv:2407.08083; NVlabs/MambaVision
  mamba_vision.py MambaVisionMixer: SSM branch + symmetric non-SSM conv branch, concatenated;
  Block with LayerNorm, layer scale, MLP; the hybrid places attention in the later blocks).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import op
from .ops_next import Mlp, RMSNorm, SwiGLU
from .ops_seq import selective_scan

# ------------------------------------------------------------------------- Gated DeltaNet


def recurrent_gated_delta_rule(q, k, v, beta, g):
    """lit_gpt recurrent_gated_delta_rule_ref: S_t = exp(g_t) S_{t-1}, v' = beta (v - S k),
    S += k v'^T, o_t = S^T (q / sqrt(d_k)). q, k, v (B, H, L, D); beta, g (B, H, L)."""
    q, k, v, beta, g = (x.float() for x in (q, k, v, beta, g))
    b, h, _, dk = q.shape
    s = q.new_zeros(b, h, dk, v.shape[-1])
    q = q * dk**-0.5
    out = []
    for q_i, k_i, v_i, b_i, g_i in zip(
        *(x.unbind(2) for x in (q, k, v, beta, g))
    ):  # unbind: see selective_scan
        s = s * g_i.exp()[..., None, None]
        vi = (v_i - (s * k_i[..., None]).sum(-2)) * b_i[..., None]
        s = s + k_i[..., None] * vi[:, :, None, :]
        out.append(torch.einsum("bhd,bhdm->bhm", q_i, s))
    return torch.stack(out, 2)


class ShortConvolution(nn.Conv1d):
    """fla ShortConvolution: causal depthwise conv (no bias) + SiLU on (B, L, D)."""

    def __init__(self, d, k):
        super().__init__(d, d, k, groups=d, padding=k - 1, bias=False)

    def forward(self, x):
        return F.silu(super().forward(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2))


class FusedRMSNormSwishGate(nn.Module):
    """fla FusedRMSNormSwishGate: RMSNorm(x) * weight * g * sigmoid(g)."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, g):
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(x.dtype) * F.silu(g)


def l2_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + eps).to(x.dtype)


class GatedDeltaNet(nn.Module):
    """GatedDeltaNet (use_mamba_gate, l2 qk norm, fused norm gate, no kv groups): (B, L, D)."""

    def __init__(self, hidden, heads=4, expand_k=1.0, expand_v=2.0, conv_size=4):
        super().__init__()
        self.h = heads
        self.key_dim, self.value_dim = int(hidden * expand_k), int(hidden * expand_v)
        if self.key_dim % heads or self.value_dim % heads:
            raise ValueError(
                f"GatedDeltaNet: key/value widths {self.key_dim}/{self.value_dim} not divisible by {heads} heads"
            )
        self.head_v = self.value_dim // heads
        self.q_proj = nn.Linear(hidden, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden, self.value_dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size)
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size)
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size)
        self.gk_proj = nn.Linear(hidden, heads, bias=False)
        self.b_proj = nn.Linear(hidden, heads, bias=True)
        self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v)
        self.o_proj = nn.Linear(self.value_dim, hidden, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(0, 16)))
        self.D = nn.Parameter(
            torch.ones(heads)
        )  # declared by the original, unused without use_residual
        dt = torch.exp(
            torch.rand(heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = (
            self.q_conv1d(self.q_proj(x)),
            self.k_conv1d(self.k_proj(x)),
            self.v_conv1d(self.v_proj(x)),
        )
        gk = (
            -self.A_log.float().exp() * F.softplus(self.gk_proj(x).float() + self.dt_bias)
        ).transpose(1, 2)
        beta = self.b_proj(x).float().sigmoid().transpose(1, 2)
        sh = lambda t: t.view(b, n, self.h, -1).transpose(1, 2)
        q, k, v = l2_norm(sh(q)).to(v), l2_norm(sh(k)).to(v), sh(v)
        o = (
            recurrent_gated_delta_rule(q, k, v, beta, gk).to(x.dtype).transpose(1, 2)
        )  # (b, n, h, dv)
        o = self.g_norm_swish_gate(o, self.g_proj(x).view(b, n, self.h, -1))
        return self.o_proj(o.reshape(b, n, -1))


class GDNBlock(nn.Module):
    """x + GatedDeltaNet(RMSNorm x); x + SwiGLU(RMSNorm x) (the LM block of the repository)."""

    def __init__(self, d, heads, ratio):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(d), GatedDeltaNet(d, heads)
        self.mlp_norm, self.mlp = RMSNorm(d), SwiGLU(d, d * ratio)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


@op("gated_deltanet_block", "seq", space={"heads": [4, 2, 1], "ratio": [2, 4]}, tags={"block"},
    source="NVlabs/GatedDeltaNet lit_gpt/gated_delta_net.py GatedDeltaNet (Yang et al., ICLR 2025), recurrent reference")  # fmt: skip
def _gdn(dims, kind, heads, ratio):
    if dims[1] % heads:
        raise ValueError(f"GatedDeltaNet: width {dims[1]} not divisible by {heads} heads")
    m = GDNBlock(dims[1], heads, ratio)
    m.attn.D.requires_grad_(False)  # never used in the forward pass (use_residual False)
    return m


# --------------------------------------------------------------------------------- mLSTM


class XLayerNorm(nn.Module):
    """xlstm components.ln.LayerNorm(weight, no bias, residual weight: scale = 1 + weight)."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), weight=1.0 + self.weight, eps=self.eps)


class MultiHeadLayerNorm(XLayerNorm):
    def forward(self, x):  # (B, NH, S, DH)
        b, nh, s, dh = x.shape
        y = F.group_norm(
            x.transpose(1, 2).reshape(b * s, nh * dh), nh, weight=1.0 + self.weight, eps=self.eps
        )
        return y.view(b, s, nh, dh).transpose(1, 2)


class LinearHeadwiseExpand(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.weight = nn.Parameter(torch.empty(heads, d // heads, d // heads))
        nn.init.normal_(self.weight, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))

    def forward(self, x):
        s = x.shape
        return torch.einsum(
            "...hd,hod->...ho", x.view(*s[:-1], self.heads, -1), self.weight
        ).reshape(*s[:-1], -1)


class CausalConv1d(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.pad = k - 1
        self.conv = nn.Conv1d(d, d, k, padding=self.pad, groups=d, bias=True)

    def forward(self, x):
        return self.conv(x.transpose(2, 1))[..., : -self.pad].transpose(1, 2) if self.pad else x


def parallel_stabilized_simple(q, k, v, ig, fg, eps=1e-6):
    """xlstm backends.parallel_stabilized_simple (row-wise stabilised), q/k/v (B, NH, S, DH)."""
    b, nh, s, dh = q.shape
    log_f = F.logsigmoid(fg)
    cum = torch.cat([q.new_zeros(b, nh, 1, 1), torch.cumsum(log_f, dim=-2)], dim=-2)
    rep = cum.repeat(1, 1, 1, s + 1)
    ltr = torch.tril(torch.ones(s, s, dtype=torch.bool, device=q.device))
    log_fg = torch.where(ltr, (rep - rep.transpose(-2, -1))[:, :, 1:, 1:], -float("inf"))
    log_d = log_fg + ig.transpose(-2, -1)
    max_d, _ = torch.max(log_d, dim=-1, keepdim=True)
    dm = torch.exp(log_d - max_d)
    c = (q @ (k / math.sqrt(dh)).transpose(-2, -1)) * dm
    norm = torch.maximum(c.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_d))
    return (c / (norm + eps)) @ v


class mLSTMCell(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.igate = nn.Linear(3 * d, heads)
        self.fgate = nn.Linear(3 * d, heads)
        self.outnorm = MultiHeadLayerNorm(d)
        with torch.no_grad():  # mLSTMCell.reset_parameters
            self.fgate.weight.zero_()
            self.fgate.bias.copy_(torch.linspace(3.0, 6.0, heads))
            self.igate.weight.zero_()
            self.igate.bias.normal_(0.0, 0.1)

    def forward(self, q, k, v):
        b, s, _ = q.shape
        gin = torch.cat([q, k, v], dim=-1)
        sh = lambda t: t.view(b, s, self.heads, -1).transpose(1, 2)
        ig = self.igate(gin).transpose(-1, -2).unsqueeze(-1)
        fg = self.fgate(gin).transpose(-1, -2).unsqueeze(-1)
        h = parallel_stabilized_simple(sh(q), sh(k), sh(v), ig, fg)
        return self.outnorm(h).transpose(1, 2).reshape(b, s, -1)


class mLSTMLayer(nn.Module):
    def __init__(self, d, heads=4, proj_factor=2.0, conv_k=4, qkv_blocksize=4):
        super().__init__()
        inner = int(
            math.ceil(proj_factor * d / 64) * 64
        )  # round_proj_up_to_multiple_of 64, round up
        self.inner = inner
        self.proj_up = nn.Linear(d, 2 * inner, bias=False)
        nh = inner // qkv_blocksize
        self.q_proj, self.k_proj, self.v_proj = (LinearHeadwiseExpand(inner, nh) for _ in range(3))
        self.conv1d = CausalConv1d(inner, conv_k)
        self.mlstm_cell = mLSTMCell(inner, heads)
        self.learnable_skip = nn.Parameter(torch.ones(inner))
        self.proj_down = nn.Linear(inner, d, bias=False)

    def forward(self, x):
        xm, z = torch.split(self.proj_up(x), self.inner, dim=-1)
        xc = F.silu(self.conv1d(xm))
        h = self.mlstm_cell(self.q_proj(xc), self.k_proj(xc), self.v_proj(xm))
        return self.proj_down((h + self.learnable_skip * xc) * F.silu(z))


class MLSTMBlock(nn.Module):
    """xLSTM mLSTMBlock: x + mLSTMLayer(LayerNorm(x))."""

    def __init__(self, d, heads=4, proj_factor=2.0):
        super().__init__()
        self.xlstm_norm = XLayerNorm(d)
        self.xlstm = mLSTMLayer(d, heads, proj_factor)

    def forward(self, x):
        return x + self.xlstm(self.xlstm_norm(x))


@op("mlstm_block", "seq", space={"heads": [4, 2, 8], "proj_factor": [2.0, 1.0]}, tags={"block"},
    source="NX-AI/xlstm blocks/mlstm (Beck et al., NeurIPS 2024) mLSTMBlock, parallel stabilised backend")  # fmt: skip
def _mlstm(dims, kind, heads, proj_factor):
    inner = int(math.ceil(proj_factor * dims[1] / 64) * 64)
    if inner % heads:
        raise ValueError(f"mLSTM: inner width {inner} not divisible by {heads} heads")
    return MLSTMBlock(dims[1], heads, proj_factor)


# ---------------------------------------------------------------------------- GR-KAN (KAT)

KAT_INIT = {  # rational_kat_cu kat_rational/init.json (order (5, 4))
    "identity": ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
    "gelu": ([-0.00153969, 0.51692871, 0.44075827, 0.0972553, -0.00884418, -0.00378675],
             [-0.14171056, -0.06481231, -0.0444695, 0.01283776]),
}  # fmt: skip


def rational_1d_group(x, w_num, w_den, groups):
    """P(x) / (1 + |b1| |x| + ... + |bm| |x|^m) per group, as the Triton kernel of rational_kat_cu
    (kat_1dgroup_triton / rational_triton: `tl.abs` on the coefficients AND on x) and the paper.
    Its torch CPU path (Rational_CUDA_A_1DGroup) uses signed powers x^i in the denominator, which
    differs for x < 0; DLNet follows the kernel used to train KAT."""
    b, n, d = x.shape
    ln, ld = w_num.shape[1], w_den.shape[1]
    z = x.view(b, n, groups, d // groups).permute(2, 0, 1, 3).reshape(groups, -1)
    zp, ap = [torch.ones_like(z)], [torch.ones_like(z)]
    for _ in range(max(ln, ld + 1) - 1):
        zp.append(zp[-1] * z)
        ap.append(ap[-1] * z.abs())
    num = torch.einsum("gk,gkn->gn", w_num, torch.stack(zp[:ln], dim=1))
    den = 1 + torch.einsum("gk,gkn->gn", w_den.abs(), torch.stack(ap[1 : ld + 1], dim=1))
    return (num / den).view(groups, b, n, d // groups).permute(1, 2, 0, 3).reshape(b, n, d)


class KATGroup(nn.Module):
    """KAT_Group: numerator (1, 6) shared by the groups, denominator (groups, 4)."""

    def __init__(self, groups=8, mode="gelu"):
        super().__init__()
        self.groups = groups
        num, den = KAT_INIT[mode]
        self.weight_numerator = nn.Parameter(torch.tensor(num).view(1, -1))
        self.weight_denominator = nn.Parameter(torch.tensor([den] * groups))

    def forward(self, x):
        squeeze = x.dim() == 2
        x = x.unsqueeze(1) if squeeze else x
        y = rational_1d_group(
            x, self.weight_numerator.repeat(self.groups, 1), self.weight_denominator, self.groups
        )
        return y.squeeze(1) if squeeze else y


class KAN(nn.Module):
    """katransformer.KAN: rational(identity init), fc1, rational(GELU init), fc2."""

    def __init__(self, d, hidden, out=None, drop=0.0, groups=8):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.act1 = KATGroup(groups, "identity")
        self.drop1 = nn.Dropout(drop)
        self.act2 = KATGroup(groups, "gelu")
        self.fc2 = nn.Linear(hidden, out or d)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.fc2(self.drop2(self.act2(self.fc1(self.drop1(self.act1(x))))))


@op("gr_kan", ("vec", "seq"), space={"ratio": [4, 2, 1], "groups": [8, 4], "out": [0, 64, 128]}, tags={"block"},
    source="Adamdad/kat katransformer.py KAN + rational_kat_cu KAT_Group (GR-KAN, Yang & Wang, ICLR 2025)")  # fmt: skip
def _grkan(dims, kind, ratio, groups, out):
    d = dims[-1]
    o = out or d
    if d % groups or (d * ratio) % groups:
        raise ValueError(f"GR-KAN: width {d} not divisible by {groups} groups")
    return KAN(d, d * ratio, o, 0.0, groups)


class KATBlock(nn.Module):
    """KAT Block: x + Attention(LN x); x + KAN(LN x) (timm ViT block with the MLP replaced)."""

    def __init__(self, d, heads, ratio=4, groups=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.mlp = KAN(d, d * ratio, d, 0.0, groups)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


@op("kat_block", "seq", space={"heads": [4, 2, 8], "ratio": [4, 2], "groups": [8, 4]}, tags={"block"},
    source="Adamdad/kat katransformer.py Block (ViT block with GR-KAN instead of the MLP)")  # fmt: skip
def _kat(dims, kind, heads, ratio, groups):
    d = dims[1]
    if d % heads or d % groups:
        raise ValueError(f"KAT: width {d} not divisible by {heads} heads / {groups} groups")
    return KATBlock(d, heads, ratio, groups)


# ---------------------------------------------------------------------------- MambaVision


class MambaVisionMixer(nn.Module):
    """mamba_vision.MambaVisionMixer: in_proj -> (x, z); x: conv + SiLU + selective scan; z: conv +
    SiLU (no SSM); concat; out_proj. Defaults of its Block: d_state 8, d_conv 3, expand 1."""

    def __init__(self, d, d_state=8, d_conv=3, expand=1):
        super().__init__()
        self.d_inner = int(expand * d)
        if self.d_inner % 2:
            raise ValueError("MambaVision mixer: expand * width must be even")
        half = self.d_inner // 2
        self.d_state, self.dt_rank = d_state, math.ceil(d / 16)
        self.in_proj = nn.Linear(d, self.d_inner, bias=False)
        self.x_proj = nn.Linear(half, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, half, bias=True)
        std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(half) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(half, 1))
        )
        self.D = nn.Parameter(torch.ones(half))
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)
        self.conv1d_x = nn.Conv1d(
            half, half, d_conv, groups=half, bias=False
        )  # bias = conv_bias // 2 = 0
        self.conv1d_z = nn.Conv1d(half, half, d_conv, groups=half, bias=False)

    def forward(self, h):
        b, n, _ = h.shape
        x, z = self.in_proj(h).transpose(1, 2).chunk(2, dim=1)
        x = F.silu(F.conv1d(x, self.conv1d_x.weight, None, padding="same", groups=x.shape[1]))
        z = F.silu(F.conv1d(z, self.conv1d_z.weight, None, padding="same", groups=z.shape[1]))
        dbl = self.x_proj(x.transpose(1, 2).reshape(b * n, -1))
        dt, bb, cc = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # the original calls self.dt_proj(dt) (bias included) AND passes delta_bias=dt_proj.bias to
        # the scan, so the bias is added twice: kept, to compute the same function
        dt = self.dt_proj(dt).view(b, n, -1).transpose(1, 2) + self.dt_proj.bias[None, :, None]
        dt = F.softplus(dt)
        bb = bb.view(b, n, -1).transpose(1, 2)
        cc = cc.view(b, n, -1).transpose(1, 2)
        y = selective_scan(x, dt, -torch.exp(self.A_log.float()), bb, cc, self.D.float())
        return self.out_proj(torch.cat([y, z], dim=1).transpose(1, 2))


class MambaVisionBlock(nn.Module):
    """mamba_vision.Block with the MambaVision mixer (no layer scale by default)."""

    def __init__(self, d, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.mixer = nn.LayerNorm(d), MambaVisionMixer(d)
        self.norm2, self.mlp = nn.LayerNorm(d), Mlp(d, int(d * mlp_ratio))

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


@op("mambavision_block", "seq", space={"mlp_ratio": [4, 2]}, tags={"block"},
    source="NVlabs/MambaVision mambavision/models/mamba_vision.py Block + MambaVisionMixer (CVPR 2025)")  # fmt: skip
def _mv(dims, kind, mlp_ratio):
    if dims[1] % 2:
        raise ValueError("MambaVision: width must be even")
    return MambaVisionBlock(dims[1], mlp_ratio)
