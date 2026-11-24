"""Numerical equivalence of the v2.1 mixers (dlnet/zoo/ops_v21.py) with their reference code.

Run on GitHub Actions (job `verify-v21`); the reference code is pinned and only lives on the runner:

    python scripts/verify_v21.py --fla /tmp/v21/fla --mamba /tmp/v21/mamba

--fla   the `fla` package of the unpacked wheels fla-core 0.6.0 and flash-linear-attention 0.6.0
--mamba state-spaces/mamba at commit e9594ce

Each original LAYER file is executed unchanged (projections, gating, reshapes, initialisation). What
needs a GPU is replaced by what the same repository ships as its reference for it, read from its
own source with `ast` (no package import):

    KDA         chunk_kda             -> naive_kda_gate + l2 norm + sigmoid + naive_recurrent_kda
    RWKV-7      chunk_rwkv7           -> dplr_recurrence (scale 1), token_shift_ref,
                fused_addcmul_rwkv7   -> torch_addcmul_rwkv7, fused_k_rwkv7 -> k_update_ref,
                gate_output_correction -> gate_output_correction_ref
    FoX         parallel_forgetting_attn -> naive_forgetting_attn
    MLA         flash_attn_func       -> exact causal softmax attention,
                RotaryEmbedding       -> rotary_embedding_ref with the class's inverse frequencies
    Mamba-3     mamba3_siso_combined  -> tests/ops/triton/test_mamba3_siso.py mamba3_siso_step_ref,
                RMSNorm (layernorm_gated) -> rms_norm_ref
    norms / short convolution: their textbook definition (RMSNorm, sigmoid-gated RMSNorm, causal
    depthwise conv + SiLU), as in scripts/verify_seq2.py.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import logging
import math
import sys
import types
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from dlnet.zoo import ops_v21 as v21

NS = {"torch": torch, "nn": nn, "F": F, "math": math, "rearrange": rearrange, "repeat": repeat,
      "Optional": Optional, "Tuple": tuple}  # fmt: skip


def extract(path, *names, ns=None):
    """The functions / classes `names` of a source file, executed alone (no module imports)."""
    tree = ast.parse(Path(path).read_text())
    ns = {**NS, **(ns or {})}
    nodes = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    if {n.name for n in nodes} != set(names):
        raise SystemExit(f"{path}: missing {set(names) - {n.name for n in nodes}}")
    exec(compile(ast.Module(nodes, []), path, "exec"), ns)  # noqa: S102  pinned reference code
    return [ns[n] for n in names]


def module(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__dict__.update(attrs)
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules.get(parent) or module(parent), child, m)
    return m


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def report(name, ref, out, tol=1e-5):
    err = (ref.float() - out.float()).abs().max().item()
    ok = err < tol * max(1.0, ref.abs().max().item())
    print(f"{name:<52} max|diff| = {err:.2e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def copy(ref, mine, scale=0.1):
    """Perturb the reference weights (so zero / constant initial values are not trivially equal),
    then load them into the DLNet module: same names and shapes, nothing missing or extra."""
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * scale)
    mine.load_state_dict(ref.state_dict(), strict=True)


# ------------------------------------------------------------- textbook modules for the stubs


class RMSNormRef(nn.Module):
    def __init__(
        self,
        hidden_size,
        elementwise_affine=True,
        bias=False,
        eps=1e-5,
        device=None,
        dtype=None,
        **_,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, *a, **k):
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(x.dtype)


class GatedNormRef(RMSNormRef):
    """fla FusedRMSNormGated: rms_norm(x) * weight * act(g), act sigmoid or swish."""

    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **_):
        super().__init__(hidden_size, eps=eps)
        self.activation = activation

    def forward(self, x, g, *a, **k):
        y = super().forward(x).float()
        return (
            y * (torch.sigmoid(g.float()) if self.activation == "sigmoid" else F.silu(g.float()))
        ).to(x.dtype)


class ShortConvRef(nn.Conv1d):
    """fla ShortConvolution: causal depthwise conv1d + activation on (B, T, D)."""

    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", **_):
        super().__init__(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=bias,
        )
        self.act = activation

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **_):
        y = super().forward(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return (F.silu(y) if self.act in ("silu", "swish") else y), None


def common_stubs():
    lg = types.SimpleNamespace(get_logger=logging.getLogger)
    module("transformers.utils", logging=lg)
    module("fla.layers.utils", get_layer_cache=lambda *a, **k: None, update_layer_cache=lambda *a, **k: None,
           unpad_hidden_states=lambda h, cu, m, q: (h, None, cu), repad_hidden_states=lambda o, i, b, q: o,
           pad_input=None, unpad_input=None)  # fmt: skip
    module("fla.modules", FusedRMSNormGated=GatedNormRef, ShortConvolution=ShortConvRef, GroupNorm=None,
           RMSNorm=RMSNormRef)  # fmt: skip


# ------------------------------------------------------------------------------ checks


def check_kda(fla):
    (gate,) = extract(f"{fla}/ops/kda/gate.py", "naive_kda_gate")
    (rec,) = extract(f"{fla}/ops/kda/naive.py", "naive_recurrent_kda")

    def kda_ref(q, k, v, g, beta, A_log, dt_bias, use_qk_l2norm_in_kernel, use_gate_in_kernel,
                use_beta_sigmoid_in_kernel, allow_neg_eigval=False, **_):  # fmt: skip
        assert use_qk_l2norm_in_kernel and use_gate_in_kernel and use_beta_sigmoid_in_kernel
        l2 = lambda t: (
            t / torch.sqrt((t.float() ** 2).sum(-1, keepdim=True) + 1e-6)
        )  # l2norm_fwd_kernel
        beta = beta.float().sigmoid() * (2 if allow_neg_eigval else 1)
        return rec(l2(q), l2(k), v, gate(g, A_log, dt_bias), beta)[0], None

    module("fla.ops.kda", chunk_kda=kda_ref, fused_recurrent_kda=kda_ref)
    ref = load(f"{fla}/layers/kda.py", "fla_kda").KimiDeltaAttention(
        hidden_size=32, head_dim=8, num_heads=4
    )
    mine = v21.KimiDeltaAttention(32, 4)
    copy(ref, mine)
    x = torch.randn(2, 11, 32)
    return report("KimiDeltaAttention (Kimi Linear)", ref(x)[0], mine(x))


def check_rwkv7(fla):
    (dplr,) = extract(f"{fla}/ops/generalized_delta_rule/dplr/naive.py", "dplr_recurrence")
    (shift,) = extract(f"{fla}/modules/token_shift.py", "token_shift_ref")
    (addcmul,) = extract(f"{fla}/ops/rwkv7/fused_addcmul.py", "torch_addcmul_rwkv7")
    (kup,) = extract(f"{fla}/ops/rwkv7/fused_k_update.py", "k_update_ref")
    (goc,) = extract(f"{fla}/ops/rwkv7/gate_output_correction.py", "gate_output_correction_ref")
    (LoRA,) = extract(f"{fla}/layers/rwkv6.py", "LoRA")

    def chunk(r, w, k, v, a, b, scale=1.0, **_):  # dplr_recurrence scales q by d_k^-0.5 itself
        t = lambda x: x.transpose(1, 2)
        o, _s = dplr(t(r) * scale * r.shape[-1] ** 0.5, t(k), t(v), t(a), t(b), t(w))
        return t(o), None

    module("fla.layers.rwkv6", LoRA=LoRA)
    module("fla.modules.l2norm", l2_norm=None)
    module(
        "fla.modules.token_shift",
        token_shift=lambda x, cu=None, output_cache=False, cache=None, **k: (shift(x), None),
    )
    module("fla.ops.rwkv7", chunk_rwkv7=chunk, fused_mul_recurrent_rwkv7=None)
    module("fla.ops.rwkv7.fused_addcmul", fused_addcmul_rwkv7=addcmul)
    module("fla.ops.rwkv7.fused_k_update", fused_k_rwkv7=kup)
    module("fla.ops.rwkv7.gate_output_correction", gate_output_correction=goc)
    ref = load(f"{fla}/layers/rwkv7.py", "fla_rwkv7").RWKV7Attention(
        mode="chunk", hidden_size=32, head_dim=8, layer_idx=0, num_hidden_layers=2)  # fmt: skip
    mine = v21.RWKV7Attention(32, 4)
    ok = True
    for (
        n,
        p,
    ) in mine.named_parameters():  # the deterministic initialisation (orthogonal / zero init aside)
        if n.startswith(
            ("x_", "k_k", "k_a", "r_k", "g_norm", "w_lora.lora.2.bias", "a_lora.lora.2.bias")
        ):
            ok &= report(f"RWKV-7 initial {n}", dict(ref.named_parameters())[n], p, 1e-6)
    copy(ref, mine)
    x = torch.randn(2, 9, 32)
    ok &= report("RWKV7Attention (RWKV-7 time mixing)", ref(x)[0], mine(x))
    ff_src = f"{fla}/models/rwkv7/modeling_rwkv7.py"
    FF, = extract(ff_src, "RWKV7FeedForward", ns={"ACT2FN": {"sqrelu": lambda t: F.relu(t) ** 2},
                                                  "token_shift": lambda x, cu=None, output_cache=False, cache=None:
                                                  (shift(x), None), "Cache": None})  # fmt: skip
    rf, mf = FF(32, layer_idx=0, num_hidden_layers=2), v21.RWKV7FeedForward(32)
    ok &= report("RWKV-7 initial x_k", rf.x_k, mf.x_k, 1e-6)
    copy(rf, mf)
    return ok & report("RWKV7FeedForward (channel mixing)", rf(x)[0], mf(x))


def check_fox(fla):
    (naive,) = extract(f"{fla}/ops/forgetting_attn/naive.py", "naive_forgetting_attn")
    module("fla.ops.attn.decoding", attn_decoding_one_step=None)
    module("fla.ops.forgetting_attn.parallel", parallel_forgetting_attn=lambda q, k, v, f, cu_seqlens=None:
           naive(q, k, v, f).to(v.dtype))  # fmt: skip
    ok = True
    for gate in (False, True):
        ref = load(f"{fla}/layers/forgetting_attn.py", "fla_fox").ForgettingAttention(
            hidden_size=32, num_heads=4, use_output_gate=gate)  # fmt: skip
        mine = v21.ForgettingAttention(32, 4, gate)
        copy(ref, mine)
        x = torch.randn(2, 10, 32)
        ok &= report(f"ForgettingAttention (FoX), output gate {gate}", ref(x)[0], mine(x))
    return ok


def check_mla(fla):
    rot, half = extract(f"{fla}/modules/rotary/ops.py", "rotary_embedding_ref", "rotate_half")
    rot.__globals__["rotate_half"] = half

    class RotaryRef(nn.Module):
        def __init__(self, dim, base=10000.0, **_):
            super().__init__()
            self.dim, self.base = dim, base

        def forward(self, q, k, seqlen_offset=0, max_seqlen=None, cu_seqlens=None, **_):
            inv = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
            )  # _compute_inv_freq
            f = torch.outer(torch.arange(q.shape[1], dtype=torch.float32), inv)
            cos, sin = f.cos()[None], f.sin()[None]
            return rot(q, cos, sin), rot(k, cos, sin)

    def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), **_):  # exact attention
        s = torch.einsum("bqhd,bkhd->bhqk", q.float() * q.shape[-1] ** -0.5, k.float())
        n = q.shape[1]
        s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), float("-inf"))
        return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v.float()).to(v.dtype)

    module("flash_attn", flash_attn_func=flash_attn_func, flash_attn_varlen_func=None)
    module("fla.ops.utils.index", prepare_lens_from_mask=None)
    sys.modules["fla.modules"].RotaryEmbedding = RotaryRef
    ok = True
    for q_rank in (None, 16):
        ref = load(f"{fla}/layers/mla.py", "fla_mla").MultiheadLatentAttention(
            hidden_size=32, num_heads=4, q_lora_rank=q_rank, qk_rope_head_dim=4, kv_lora_rank=16,
            v_head_dim=8, qk_nope_head_dim=8, qk_head_dim=12)  # fmt: skip
        mine = v21.MultiheadLatentAttention(32, 4, kv_rank=16, q_rank=q_rank or 0, rope_dim=4)
        copy(ref, mine)
        x = torch.randn(2, 10, 32)
        ok &= report(
            f"MultiheadLatentAttention (DeepSeek-V2), q_rank {q_rank}", ref(x, None)[0], mine(x)
        )
    return ok


def check_mamba3(root):
    (rms,) = extract(f"{root}/mamba_ssm/ops/triton/layernorm_gated.py", "rms_norm_ref")
    (step,) = extract(f"{root}/tests/ops/triton/test_mamba3_siso.py", "mamba3_siso_step_ref")

    class RMSNormGated(nn.Module):
        def __init__(
            self,
            hidden_size,
            eps=1e-5,
            norm_before_gate=True,
            group_size=None,
            device=None,
            dtype=None,
        ):
            super().__init__()
            self.eps, self.group_size, self.nbg = eps, group_size, norm_before_gate
            self.weight = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x, z=None):
            return rms(
                x,
                self.weight,
                None,
                z=z,
                eps=self.eps,
                group_size=self.group_size,
                norm_before_gate=self.nbg,
            )

    def siso(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D=None, Z=None, **_):
        return step(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z)[0]

    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
        module(name)
    module("mamba_ssm.ops.triton.layernorm_gated", RMSNorm=RMSNormGated)
    module("mamba_ssm.ops.triton.mamba3.mamba3_siso_combined", mamba3_siso_combined=siso)
    module(
        "mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step", apply_rotary_qk_inference_fwd=None
    )
    ref = load(f"{root}/mamba_ssm/modules/mamba3.py", "mamba3_ref").Mamba3(
        d_model=32, d_state=16, expand=2, headdim=16
    )
    mine = v21.Mamba3(32, 16, 2, 16)
    copy(ref, mine)
    x = torch.randn(2, 13, 32)
    return report("Mamba3 (SISO)", ref(x), mine(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fla", required=True, help="directory of the `fla` package (fla-core 0.6.0)")
    ap.add_argument("--mamba", required=True, help="state-spaces/mamba checkout at e9594ce")
    a = ap.parse_args()
    torch.manual_seed(0)
    common_stubs()
    results = [
        check_kda(a.fla),
        check_rwkv7(a.fla),
        check_fox(a.fla),
        check_mla(a.fla),
        check_mamba3(a.mamba),
    ]
    print(f"{sum(results)}/{len(results)} layers match")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
