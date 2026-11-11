"""Numerical equivalence of the frequency-domain blocks (dlnet/zoo/ops_freq.py) with OFFICIAL code.

    git clone https://github.com/raoyongming/GFNet /tmp/GFNet             # 2de48b6
    git clone https://github.com/NVlabs/AFNO-transformer /tmp/AFNO        # 941e924
    git clone https://github.com/pkumivision/FFC /tmp/FFC                 # 39dcc99
    git clone https://github.com/cfzd/FcaNet /tmp/FcaNet                  # aa5fb63
    python scripts/verify_freq.py --gfnet /tmp/GFNet --afno /tmp/AFNO --ffc /tmp/FFC --fcanet /tmp/FcaNet

Weights (and BatchNorm statistics, set to the same random values) are copied from the official
module by name; outputs on the same random input are compared in eval mode. Run on GitHub Actions
by .github/workflows/verify-modern.yml. FNO and FNet have no runnable official PyTorch code here:
closed-form tests in tests/test_freq.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import torch

from dlnet.zoo import ops_freq as fq


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def report(name, ref, out):
    err = (ref - out).abs().max().item()
    ok = err < 1e-5 * max(1.0, ref.abs().max().item())
    print(f"{name:<40} max|diff| = {err:.2e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def random_bn(m):
    with torch.no_grad():
        for b in m.modules():
            if isinstance(b, torch.nn.BatchNorm2d):
                b.running_mean.normal_()
                b.running_var.uniform_(0.5, 2)
                b.weight.normal_()
                b.bias.normal_()


def copy_by_name(src, dst, rules=()):
    sd = {}
    for n, x in src.state_dict().items():
        for a, b in rules:
            n = n.replace(a, b)
        sd[n] = x
    dst.load_state_dict(sd, strict=True)  # every tensor matched by name and shape


def check_gfnet(repo):
    # gfnet.py imports timm (only DropPath/trunc_normal_ for the full network) and a numpy path
    # removed in numpy 2; neither is used by GlobalFilter / Block with drop_path 0
    for name in ("timm", "timm.data", "timm.models", "timm.models.layers"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules.setdefault("numpy.lib.arraypad", types.SimpleNamespace(pad=np.pad))
    g = load(f"{repo}/gfnet.py", "gfnet_official")
    ok = True
    torch.manual_seed(0)
    for dim, h in ((16, 6), (32, 9)):
        ref = g.Block(dim, mlp_ratio=4.0, h=h, w=h // 2 + 1).eval()
        mine = fq.GFNetBlock(dim, (h, h), 4.0).eval()
        copy_by_name(ref, mine)
        x = torch.randn(2, h, h, dim)
        ok &= report(
            f"GFNet Block dim={dim} {h}x{h}",
            ref(x.reshape(2, h * h, dim)),
            mine(x).reshape(2, h * h, dim),
        )
    return ok


def check_afno(repo):
    a = load(f"{repo}/afno/afno2d.py", "afno2d_official")
    ok = True
    torch.manual_seed(0)
    for dim, blocks, h in ((16, 4, 6), (32, 8, 9), (24, 1, 8)):
        ref = a.AFNO2D(dim, num_blocks=blocks).eval()
        mine = fq.AFNO2D(dim, blocks).eval()
        copy_by_name(ref, mine)
        x = torch.randn(2, h, h, dim)
        ok &= report(
            f"AFNO2D dim={dim} blocks={blocks} {h}x{h}",
            ref(x.reshape(2, h * h, dim)),
            mine(x).reshape(2, h * h, dim),
        )
    return ok


def _old_fft_api():
    """torch.rfft / torch.irfft (removed in torch 1.8) used by the FFC code, on top of torch.fft."""

    def rfft(x, signal_ndim, normalized=False, onesided=True):
        dims = tuple(range(-signal_ndim, 0))
        return torch.view_as_real(
            torch.fft.rfftn(x, dim=dims, norm="ortho" if normalized else "backward")
        )

    def irfft(x, signal_ndim, normalized=False, onesided=True, signal_sizes=None):
        dims = tuple(range(-signal_ndim, 0))
        z = torch.view_as_complex(x.contiguous())
        return torch.fft.irfftn(
            z, s=signal_sizes, dim=dims, norm="ortho" if normalized else "backward"
        )

    torch.rfft, torch.irfft = rfft, irfft


def check_ffc(repo):
    _old_fft_api()
    pkg = types.ModuleType("model_zoo")
    pkg.__path__ = [f"{repo}/model_zoo"]
    sys.modules["model_zoo"] = pkg
    load(f"{repo}/model_zoo/ffc.py", "model_zoo.ffc")
    r = load(f"{repo}/model_zoo/ffc_resnet.py", "model_zoo.ffc_resnet")
    ok = True
    torch.manual_seed(0)
    for c, h, ratio, lfu in ((32, 8, 0.5, True), (32, 9, 0.5, False), (64, 8, 0.25, True)):
        ref = r.BasicBlock(c, c, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu).eval()
        random_bn(ref)
        mine = fq.FFCBasicBlock(c, ratio, lfu).eval()
        copy_by_name(ref, mine)
        x = torch.randn(2, c, h, h)
        cg = int(c * ratio)
        yl, yg = ref((x[:, : c - cg], x[:, c - cg :]))
        ok &= report(
            f"FFC BasicBlock c={c} {h}x{h} r={ratio} lfu={lfu}", torch.cat([yl, yg], 1), mine(x)
        )
    return ok


def check_fcanet(repo):
    f = load(f"{repo}/model/layer.py", "fcanet_layer_official")
    ok = True
    torch.manual_seed(0)
    for c, method, h in ((64, "top16", 7), (32, "low4", 9), (32, "bot8", 5)):
        ref = f.MultiSpectralAttentionLayer(c, 7, 7, reduction=16, freq_sel_method=method).eval()
        mine = fq.MultiSpectralAttention(c, 7, 7, 16, method).eval()
        copy_by_name(ref, mine, (("dct_layer.weight", "weight"),))
        x = torch.randn(2, c, h, h)
        ok &= report(f"FcaNet attention c={c} {method} {h}x{h}", ref(x), mine(x))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("gfnet", "afno", "ffc", "fcanet"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    good = check_gfnet(a.gfnet) & check_afno(a.afno) & check_ffc(a.ffc) & check_fcanet(a.fcanet)
    print("ALL OK" if good else "SOME MISMATCH")
    raise SystemExit(0 if good else 1)
