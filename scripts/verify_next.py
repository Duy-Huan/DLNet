"""Numerical equivalence of the v1.0 drop-in operators (dlnet/zoo/ops_next.py) with OFFICIAL code.

Run on GitHub Actions (.github/workflows/verify-modern.yml, job `verify-next`), which clones the
repositories at pinned commits into one folder and installs transformers, timm and PyWavelets:

    python scripts/verify_next.py --root /tmp/v1

Weights (and BatchNorm statistics, set to the same random values) are copied by name with
load_state_dict(strict=True); outputs on the same random input are compared in eval mode.
Packages the official files import only for parts that are not compared (flash_attn, the Triton
kernels, mmcv/mmseg registries) are replaced by stubs; the compared computation is the original.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from unittest.mock import MagicMock

import torch
from torch import nn

from dlnet.zoo import ops_next as nx


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def package(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


def stub(*names):
    for n in names:
        sys.modules.setdefault(n, MagicMock())


def report(name, ref, out, tol=1e-5):
    err = (ref - out).abs().max().item()
    ok = err < tol * max(1.0, ref.abs().max().item())
    print(f"{name:<44} max|diff| = {err:.2e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def random_bn(m):
    with torch.no_grad():
        for b in m.modules():
            if isinstance(b, nn.modules.batchnorm._BatchNorm):
                b.running_mean.normal_()
                b.running_var.uniform_(0.5, 2)
                if b.affine:
                    b.weight.normal_()
                    b.bias.normal_()


def copy(ref, mine, rules=(), skip=()):
    sd = {}
    for n, x in ref.state_dict().items():
        if any(n.startswith(s) for s in skip) or n.endswith("num_batches_tracked"):
            continue
        for a, b in rules:
            n = n.replace(a, b)
        sd[n] = x
    missing, unexpected = mine.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise ValueError(f"state_dict mismatch: missing {missing} unexpected {unexpected}")


def perturb(m):
    """Layer scales / gates start at tiny or constant values: randomise every parameter so that
    the comparison exercises the whole computation, not a near-identity."""
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)


# ------------------------------------------------------------------------- checks


def check_dyt(root):
    d = load(f"{root}/DyT/dynamic_tanh.py", "dyt_official")
    ok = True
    for last, shape in ((True, (3, 5, 16)), (False, (2, 16, 4, 4))):
        ref = d.DynamicTanh(16, last)
        mine = nx.DynamicTanh(16, last, 0.5, 2)
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(*shape)
        ok &= report(f"DyT channels_last={last}", ref(x), mine(x))
    return ok


def check_gated(root):
    sys.path.insert(0, f"{root}/gated_attention")
    package("qwen3_gated", f"{root}/gated_attention")
    load(f"{root}/gated_attention/configuration_qwen3.py", "qwen3_gated.configuration_qwen3")
    q = load(f"{root}/gated_attention/modeling_qwen3.py", "qwen3_gated.modeling_qwen3")
    ok = True
    torch.manual_seed(0)
    for gate in ("headwise", "elementwise", "none"):
        cfg = q.Qwen3Config(hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=4,
                            head_dim=8, use_qk_norm=True, headwise_attn_output_gate=gate == "headwise",
                            elementwise_attn_output_gate=gate == "elementwise")  # fmt: skip
        cfg.qkv_bias = False
        cfg._attn_implementation = "eager"
        ref = q.Qwen3DecoderLayer(cfg, 0).eval()
        mine = nx.GatedBlock(32, 4, 2, gate, False).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(2, 7, 32)
        cos, sin = (
            torch.ones(2, 7, 8),
            torch.zeros(2, 7, 8),
        )  # identity rotary: DLNet adds no RoPE here
        y = ref(x, attention_mask=None, position_ids=None, position_embeddings=(cos, sin))[0]
        ok &= report(f"Gated attention block gate={gate}", y, mine(x))
    sys.path.pop(0)
    return ok


def check_diff(root):
    stub("flash_attn")  # without apex the file falls back to the repo's own rms_norm.RMSNorm
    kernel = package("kernel", f"{root}/unilm/Diff-Transformer/kernel")
    rot = types.ModuleType("kernel.rotary")
    rot.apply_rotary_emb = lambda x, cos, sin, interleaved=True: x  # rotary is not used by DLNet
    sys.modules["kernel.rotary"], kernel.rotary = rot, rot
    sys.path.insert(0, f"{root}/unilm/Diff-Transformer")
    m = load(f"{root}/unilm/Diff-Transformer/multihead_diffattn.py", "diffattn_official")
    ok = True
    torch.manual_seed(0)
    for d, heads, depth in ((32, 2, 1), (64, 4, 5)):
        ref = m.MultiheadDiffAttn(d, depth, heads).eval()
        mine = nx.MultiheadDiffAttn(d, depth, heads).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(2, 6, d)
        y = ref(x, rel_pos=(None, None), attn_mask=torch.zeros(6, 6))  # non-causal
        ok &= report(f"Differential attention d={d} heads={heads} depth={depth}", y, mine(x))
        ycz = ref(x, rel_pos=(None, None))  # official default mask: causal
        mine.causal = True
        ok &= report(f"Differential attention causal d={d}", ycz, mine(x))
    sys.path.pop(0)
    return ok


def check_rope(root):
    sys.path.insert(0, f"{root}/rope-vit")  # vit_rope imports deit.models_v2
    r = load(f"{root}/rope-vit/models/vit_rope.py", "vit_rope_official")
    ok = True
    torch.manual_seed(0)
    dim, heads, h, w = 32, 4, 5, 6
    for mode in ("axial", "mixed"):
        mine = nx.RoPEBlock(dim, heads, h, w, mode).eval()
        perturb(mine)
        if mode == "axial":
            cis = r.compute_axial_cis(dim // heads, w, h, 100.0)
        else:
            t_x, t_y = r.init_t_xy(w, h)
            f = mine.freqs.detach().reshape(2, 1, -1)  # official layout (2, depth, heads * hd/2)
            cis = r.compute_mixed_cis(f, t_x, t_y, heads)[0]
        ok &= report(
            f"RoPE-ViT freqs_cis {mode}",
            torch.view_as_real(cis),
            torch.view_as_real(mine.freqs_cis()),
        )
        # attention with the official rotary on the same q, k (a class token at 0 is skipped by it)
        ref = r.RoPEAttention(dim, num_heads=heads, qkv_bias=True).eval()
        copy(ref, mine.attn)
        x = torch.randn(2, h * w, dim)
        y = ref(torch.cat([torch.zeros(2, 1, dim), x], 1), cis)[:, 1:]
        xc = torch.cat([torch.zeros(2, 1, dim), x], 1)
        y_mine_full = _rope_attn_with_cls(mine.attn, xc, cis)
        ok &= report(f"RoPE attention {mode}", y, y_mine_full)
    sys.path.pop(0)
    return ok


def _rope_attn_with_cls(attn, x, cis):
    """dlnet RoPEAttention on tokens [cls, pixels] with the class token left unrotated."""
    B, N, C = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0].clone(), qkv[1].clone(), qkv[2]
    q[:, :, 1:], k[:, :, 1:] = nx.apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], cis)
    a = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    return attn.proj((a @ v).transpose(1, 2).reshape(B, N, C))[:, 1:]


def check_wtconv(root):
    package("wtconv", f"{root}/WTConv/wtconv")
    package("wtconv.util", f"{root}/WTConv/wtconv/util")
    load(f"{root}/WTConv/wtconv/util/wavelet.py", "wtconv.util.wavelet")
    sys.modules["wtconv.util"].wavelet = sys.modules["wtconv.util.wavelet"]
    w = load(f"{root}/WTConv/wtconv/wtconv2d.py", "wtconv.wtconv2d")
    ok = True
    torch.manual_seed(0)
    for c, k, lv, hw in ((8, 5, 1, 8), (16, 3, 3, 9), (8, 5, 2, 7)):
        ref = w.WTConv2d(c, c, k, wt_levels=lv).eval()
        mine = nx.WTConv2d(c, k, True, lv).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(2, c, hw, hw)
        ok &= report(f"WTConv2d c={c} k={k} levels={lv} {hw}x{hw}", ref(x), mine(x))
    return ok


def check_timm_mnv4():
    from timm.models._efficientnet_blocks import MobileAttention, UniversalInvertedResidual

    ok = True
    torch.manual_seed(0)
    for variant, (s, m, e) in nx.UIB_VARIANTS.items():
        ks = [3 if v else 0 for v in (s, m, e)]
        ref = UniversalInvertedResidual(16, 16, ks[0], ks[1], ks[2], exp_ratio=4.0).eval()
        mine = nx.UIB(16, ks[0], ks[1], ks[2], 4.0).eval()
        perturb(ref)
        random_bn(ref)
        copy(ref, mine)
        x = torch.randn(2, 16, 6, 6)
        ok &= report(f"MobileNetV4 UIB {variant}", ref(x), mine(x))
    for kv in (1, 2):
        ref = MobileAttention(
            32, 32, num_heads=4, key_dim=16, value_dim=16, use_multi_query=True, kv_stride=kv
        ).eval()
        mine = nx.MobileMQA(32, 4, 16, kv).eval()
        perturb(ref)
        random_bn(ref)
        copy(ref, mine)
        x = torch.randn(2, 32, 6, 6)
        ok &= report(f"MobileNetV4 Mobile MQA kv_stride={kv}", ref(x), mine(x))
    return ok


def check_lsnet(root):
    from timm.layers import SqueezeExcite  # noqa: F401  (the official file imports it from timm)

    # model/ska.py is a Triton kernel: replaced by DLNet's unfold version, checked in tests
    ska = types.ModuleType("lsnet_pkg.ska")

    class SKA(nn.Module):
        def forward(self, x, w):
            return nx.ska(x, w)

    ska.SKA = SKA
    package("lsnet_pkg", f"{root}/lsnet/model")
    sys.modules["lsnet_pkg.ska"] = ska
    m = load(f"{root}/lsnet/model/lsnet.py", "lsnet_pkg.lsnet")
    ok = True
    torch.manual_seed(0)
    for depth, mixer in ((1, "ls"), (0, "repvgg")):
        ref = m.Block(32, 16, 8, 4, 8, stage=0, depth=depth).eval()
        mine = nx.LSBlock(32, mixer).eval()
        perturb(ref)
        random_bn(ref)
        copy(ref, mine)
        x = torch.randn(2, 32, 8, 8)
        ok &= report(f"LSNet block mixer={mixer}", ref(x), mine(x))
    return ok


def check_starnet(root):
    s = load(f"{root}/Rewrite-the-Stars/imagenet/starnet.py", "starnet_official")
    ok = True
    torch.manual_seed(0)
    for r in (3, 4):
        ref = s.Block(16, r).eval()
        mine = nx.StarBlock(16, r).eval()
        perturb(ref)
        random_bn(ref)
        copy(ref, mine)
        x = torch.randn(2, 16, 9, 9)
        ok &= report(f"StarNet block mlp_ratio={r}", ref(x), mine(x))
    return ok


def check_efficientvit(root):
    base = f"{root}/efficientvit/efficientvit"
    package("efficientvit", base)
    package("efficientvit.models", f"{base}/models")
    package("efficientvit.models.nn", f"{base}/models/nn")
    stub("efficientvit.models.nn.triton_rms_norm")
    load(f"{base}/models/utils/__init__.py", "efficientvit.models.utils")
    for n in ("act", "norm"):
        load(f"{base}/models/nn/{n}.py", f"efficientvit.models.nn.{n}")
    o = load(f"{base}/models/nn/ops.py", "efficientvit.models.nn.ops")
    ok = True
    torch.manual_seed(0)
    for c, dim, hw in ((32, 8, 6), (32, 16, 3), (64, 32, 5)):
        ref = o.EfficientViTBlock(c, dim=dim, expand_ratio=4).eval()
        mine = nx.EfficientViTBlock(c, dim, 4).eval()
        perturb(ref)
        random_bn(ref)
        copy(ref, mine)
        x = torch.randn(2, c, hw, hw)
        ok &= report(f"EfficientViT block c={c} dim={dim} {hw}x{hw}", ref(x), mine(x), 1e-4)
    return ok


def check_segnext(root):
    mmcv = MagicMock()
    sys.modules.update({"mmcv": mmcv, "mmcv.cnn": mmcv.cnn, "mmcv.runner": mmcv.runner,
                        "mmcv.cnn.bricks": mmcv.cnn.bricks, "mmcv.cnn.utils": mmcv.cnn.utils,
                        "mmcv.cnn.utils.weight_init": mmcv.cnn.utils.weight_init,
                        "mmseg": MagicMock(), "mmseg.models": MagicMock(), "mmseg.models.builder": MagicMock()})  # fmt: skip
    mmcv.runner.BaseModule = nn.Module
    mmcv.cnn.build_norm_layer = lambda cfg, c: ("bn", nn.BatchNorm2d(c))
    mmcv.cnn.bricks.DropPath = nn.Identity
    m = load(f"{root}/SegNeXt/mmseg/models/backbones/mscan.py", "mscan_official")
    ok = True
    torch.manual_seed(0)
    for dim, r in ((32, 4), (16, 8)):
        ref = m.Block(dim, mlp_ratio=r).eval()
        mine = nx.MSCABlock(dim, r).eval()
        perturb(ref)
        random_bn(ref)
        copy(ref, mine)
        x = torch.randn(2, dim, 7, 9)
        y = ref(x.flatten(2).transpose(1, 2), 7, 9).transpose(1, 2).reshape(2, dim, 7, 9)
        ok &= report(f"SegNeXt MSCAN block dim={dim} ratio={r}", y, mine(x))
    return ok


def check_pywt_haar():
    import pywt

    w = pywt.Wavelet("db1")
    ok = all(
        abs(a - b) < 1e-12 for k in nx.HAAR for a, b in zip(nx.HAAR[k], getattr(w, k), strict=True)
    )
    print(f"{'Haar filters == pywt db1':<44} {'OK' if ok else 'MISMATCH'}")
    return ok


CHECKS = (check_dyt, check_gated, check_diff, check_rope, check_wtconv, check_lsnet, check_starnet,
          check_efficientvit, check_segnext)  # fmt: skip

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding the cloned official repositories")
    a = ap.parse_args()
    good = check_pywt_haar() & check_timm_mnv4()
    for chk in CHECKS:
        try:
            good &= chk(a.root)
        except Exception as e:  # noqa: BLE001  (report every family, then fail)
            print(f"{chk.__name__:<44} ERROR {type(e).__name__}: {e}")
            good = False
    print("ALL OK" if good else "SOME MISMATCH")
    raise SystemExit(0 if good else 1)
