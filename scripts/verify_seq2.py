"""Numerical equivalence of the v1.0 sequence mixers / KAN (dlnet/zoo/ops_seq2.py) with OFFICIAL code.

Run on GitHub Actions (job `verify-seq2`), repositories cloned at pinned commits:

    python scripts/verify_seq2.py --root /tmp/seq

The originals call CUDA/Triton kernels that do not run on a CPU runner; each is replaced by the
reference the repository itself ships for testing those kernels, and that substitution is what
DLNet implements: Gated DeltaNet -> recurrent_gated_delta_rule_ref (its chunk.py; the fla norm /
short-conv modules -> their definitions), KAT -> kat_1dgroup_torch.Rational_CUDA_A_1DGroup,
MambaVision -> selective_scan_ref semantics (DLNet's scan, verified against mamba_ssm in
scripts/verify_numeric.py). mLSTM runs the official xlstm code unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from unittest.mock import MagicMock

import torch
from torch.nn import functional as F

from dlnet.zoo import ops_seq2 as s2
from dlnet.zoo.ops_seq import selective_scan


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def module(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def report(name, ref, out, tol=1e-5):
    err = (ref - out).abs().max().item()
    ok = err < tol * max(1.0, ref.abs().max().item())
    print(f"{name:<46} max|diff| = {err:.2e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def perturb(m, scale=0.1):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * scale)


def copy(ref, mine, skip=()):
    sd = {n: x for n, x in ref.state_dict().items() if not any(s in n for s in skip)}
    missing, unexpected = mine.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError(f"state_dict mismatch: missing {missing} unexpected {unexpected}")


def _triton_stub():
    tl = MagicMock()
    triton = MagicMock()
    triton.language = tl
    triton.jit = lambda f=None, **k: f if callable(f) else (lambda g: g)
    triton.autotune = lambda *a, **k: lambda g: g
    triton.heuristics = lambda *a, **k: lambda g: g
    sys.modules.update({"triton": triton, "triton.language": tl})


def check_gated_deltanet(root):
    _triton_stub()
    base = f"{root}/GatedDeltaNet/lit_gpt"
    module("fla")
    module("fla.utils", contiguous=lambda f: f, autocast_custom_fwd=lambda f=None, **k: f if f else (lambda g: g),
           autocast_custom_bwd=lambda f=None, **k: f if f else (lambda g: g))  # fmt: skip
    module("fla.ops")
    module(
        "fla.ops.utils", chunk_global_reversed_cumsum=MagicMock(), chunk_local_cumsum=MagicMock()
    )
    module("fla.ops.simple_gla", chunk_simple_gla=MagicMock())

    class ShortConvolution(s2.ShortConvolution):
        def __init__(self, hidden_size, kernel_size, activation="silu", **kw):
            super().__init__(hidden_size, kernel_size)

        def forward(self, x, mask=None, cache=None):
            return super().forward(x)

    class FusedRMSNormSwishGate(s2.FusedRMSNormSwishGate):
        def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5):
            super().__init__(hidden_size, eps)

    module(
        "fla.modules",
        FusedRMSNormSwishGate=FusedRMSNormSwishGate,
        RMSNorm=MagicMock(),
        ShortConvolution=ShortConvolution,
    )
    module("fla.modules.activations", ACT2FN={})
    module("fla.modules.l2norm", l2_norm=s2.l2_norm, l2_norm_fn=s2.l2_norm)
    module("fla.models")
    module("fla.models.utils", Cache=object)
    pkg = module("gdn_pkg")
    pkg.__path__ = [base]
    ops = module("gdn_pkg.gated_delta_rule_ops")
    ops.__path__ = [f"{base}/gated_delta_rule_ops"]
    try:
        chunk = load(f"{base}/gated_delta_rule_ops/chunk.py", "gdn_pkg.gated_delta_rule_ops.chunk")
        ref_rule = chunk.recurrent_gated_delta_rule_ref
    except Exception as e:  # noqa: BLE001
        print(f"{'Gated delta rule reference':<46} ERROR loading chunk.py: {e}")
        return False
    ok = True
    torch.manual_seed(0)
    b, h, n, dk, dv = 2, 3, 9, 8, 6
    q, k, v = (
        torch.randn(b, h, n, dk),
        F.normalize(torch.randn(b, h, n, dk), dim=-1),
        torch.randn(b, h, n, dv),
    )
    beta, g = torch.rand(b, h, n), torch.empty(b, h, n).uniform_(0.01, 0.3).log()
    ok &= report("Gated delta rule == recurrent_gated_delta_rule_ref", ref_rule(q, k, v, beta, g),
                 s2.recurrent_gated_delta_rule(q, k, v, beta, g))  # fmt: skip
    ops.chunk_gated_delta_rule = (
        lambda q, k, v, beta, g, initial_state=None, output_final_state=False: (
            ref_rule(q, k, v, beta, g),
            None,
        )
    )
    m = load(f"{base}/gated_delta_net.py", "gdn_pkg.gated_delta_net")
    for hidden, heads in ((32, 4), (24, 2)):
        ref = m.GatedDeltaNet(
            hidden_size=hidden, expand_k=1.0, expand_v=2.0, num_heads=heads, use_mamba_gate=True
        ).eval()
        mine = s2.GatedDeltaNet(hidden, heads, 1.0, 2.0).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(2, 11, hidden)
        ok &= report(f"GatedDeltaNet layer d={hidden} heads={heads}", ref(x)[0], mine(x), 1e-4)
    return ok


def check_mlstm(root):
    sys.path.insert(0, f"{root}/xlstm")
    from xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig

    ok = True
    torch.manual_seed(0)
    for d, heads, s in ((32, 4, 10), (64, 2, 7)):
        cfg = mLSTMLayerConfig(embedding_dim=d, num_heads=heads, context_length=s, proj_factor=2.0)
        ref = mLSTMLayer(cfg).eval()
        mine = s2.mLSTMLayer(d, heads, 2.0).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(2, s, d)
        ok &= report(f"mLSTM layer d={d} heads={heads} S={s}", ref(x), mine(x), 1e-4)
    sys.path.pop(0)
    return ok


def check_kat(root):
    t = load(f"{root}/rational_kat_cu/kat_rational/kat_1dgroup_torch.py", "kat_torch_official")

    class KAT_Group(t.KAT_Group_Torch):
        def __init__(self, num_groups=8, mode="gelu", device="cpu"):
            super().__init__(num_groups, mode)

    module("kat_rational", KAT_Group=KAT_Group)
    m = load(f"{root}/kat/katransformer.py", "katransformer_official")
    ok = True
    torch.manual_seed(0)
    for d, ratio in ((32, 4), (64, 2)):
        ref = m.KAN(d, d * ratio, device="cpu").eval()
        mine = s2.KAN(d, d * ratio, d).eval()
        perturb(ref, 0.05)
        copy(ref, mine)
        x = torch.randn(2, 5, d)
        ok &= report(f"GR-KAN (KAT KAN) d={d} ratio={ratio}", ref(x), mine(x), 1e-4)
    return ok


def check_mambavision(root):
    def selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=None,
    ):
        if delta_bias is not None:
            delta = delta + delta_bias[None, :, None]
        if delta_softplus:
            delta = F.softplus(delta)
        return selective_scan(u.float(), delta.float(), A.float(), B.float(), C.float(), D.float())

    module("mamba_ssm")
    module("mamba_ssm.ops")
    module(
        "mamba_ssm.ops.selective_scan_interface",
        selective_scan_fn=selective_scan_fn,
        mamba_inner_fn=MagicMock(),
    )
    import timm.models.registry as reg

    module("mv_pkg", __path__=[f"{root}/MambaVision/mambavision"])
    module("mv_pkg.models", __path__=[f"{root}/MambaVision/mambavision/models"])
    module("mv_pkg.models.registry", register_pip_model=lambda f: f)
    keep, reg.register_model = reg.register_model, lambda f: f  # its pretrained cfgs are not needed
    try:
        m = load(
            f"{root}/MambaVision/mambavision/models/mamba_vision.py", "mv_pkg.models.mamba_vision"
        )
    finally:
        reg.register_model = keep
    ok = True
    torch.manual_seed(0)
    for d in (32, 48):
        ref = m.MambaVisionMixer(d_model=d, d_state=8, d_conv=3, expand=1).eval()
        mine = s2.MambaVisionMixer(d, 8, 3, 1).eval()
        perturb(ref, 0.05)
        copy(ref, mine)
        x = torch.randn(2, 13, d)
        ok &= report(f"MambaVision mixer d={d}", ref(x), mine(x), 1e-4)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    good = True
    for chk in (check_gated_deltanet, check_mlstm, check_kat, check_mambavision):
        try:
            good &= chk(a.root)
        except Exception as e:  # noqa: BLE001  (report every family, then fail)
            print(f"{chk.__name__:<46} ERROR {type(e).__name__}: {e}")
            good = False
    print("ALL OK" if good else "SOME MISMATCH")
    raise SystemExit(0 if good else 1)
