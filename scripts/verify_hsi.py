"""Numerical equivalence of the v1.0 HSI networks (dlnet/zoo/ops_hsi.py) with OFFICIAL code.

Run on GitHub Actions (job `verify-hsi`), repositories cloned at pinned commits into one folder:

    python scripts/verify_hsi.py --root /tmp/hsi

`mamba_ssm.Mamba` (CUDA) is replaced by DLNet's Mamba, itself equal to mamba_ssm's reference path
(scripts/verify_numeric.py), so what is compared here is the code AROUND Mamba: token layout,
scans, flips, normalisation, fusion, embeddings. Weights and BatchNorm statistics copied by name.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

import torch
from torch import nn

from dlnet.zoo import ops_hsi as hs
from dlnet.zoo.ops_seq import Mamba


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def report(name, ref, out, tol=1e-5):
    err = (ref - out).abs().max().item()
    ok = err < tol * max(1.0, ref.abs().max().item())
    print(f"{name:<44} max|diff| = {err:.2e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def perturb(m):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        for b in m.modules():
            if isinstance(b, nn.modules.batchnorm._BatchNorm):
                b.running_mean.normal_()
                b.running_var.uniform_(0.5, 2)


def copy(ref, mine, skip=()):
    sd = {n: x for n, x in ref.state_dict().items() if not any(n.startswith(s) for s in skip)}
    missing, unexpected = mine.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError(f"state_dict mismatch: missing {missing} unexpected {unexpected}")


def mamba_stub():
    ms = types.ModuleType("mamba_ssm")

    class OfficialSignatureMamba(Mamba):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kw):
            super().__init__(d_model, d_state, d_conv, expand)

    ms.Mamba = OfficialSignatureMamba
    sys.modules["mamba_ssm"] = ms


def check_mambahsi(root):
    m = load(f"{root}/MambaHSI/model/MambaHSI.py", "mambahsi_official")
    ok = True
    torch.manual_seed(0)
    for c, tok, g in ((32, 4, 4), (24, 8, 2)):
        ref = m.BothMamba(c, tok, True, g, True).eval()
        mine = hs.BothMamba(c, tok, True, g, True).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(1, c, 5, 5)  # batch 1: the original scans all pixels of the batch at once
        ok &= report(f"MambaHSI BothMamba c={c} tokens={tok}", ref(x), mine(x))
    return ok


def check_mambahsi_plus(root):
    chdir, os.chdir = (
        os.chdir,
        lambda p: None,
    )  # the original chdir()s to its author's home at import
    try:
        pkg = types.ModuleType("mhp")
        pkg.__path__ = [f"{root}/MambaHSI_Plus/model"]
        sys.modules["mhp"] = pkg
        load(f"{root}/MambaHSI_Plus/model/Utils.py", "mhp.Utils")
        m = load(f"{root}/MambaHSI_Plus/model/MambaHSI_Plus.py", "mhp.MambaHSI_Plus")
    finally:
        os.chdir = chdir
    ok = True
    torch.manual_seed(0)
    for c, tok, g in ((32, 4, 2), (24, 8, 4)):
        ref = m.BothMamba(c, tok, True, g).eval()
        mine = hs.BothMambaPlus(c, tok, g).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(2, c, 5, 4)
        ok &= report(f"MambaHSI+ BothMamba c={c} tokens={tok}", ref(x), mine(x))
    return ok


def check_ssftt(root):
    m = load(f"{root}/HSI_SSFTT/cls_SSFTT_IP/SSFTTnet.py", "ssftt_official")
    ok = True
    torch.manual_seed(0)
    for bands, p in ((30, 13), (16, 9)):
        ref = m.SSFTTnet(in_channels=1, num_classes=5).eval()
        if bands != 30:  # the original hard-codes 8 * 28 input channels of the 2-D conv (30 bands)
            ref.conv2d_features[0] = nn.Conv2d(8 * (bands - 2), 64, (3, 3))
        mine = hs.SSFTT(bands).eval()
        perturb(ref)
        copy(ref, mine, skip=("nn1.",))
        x = torch.randn(2, bands, p, p)
        feats = {}
        ref.nn1.register_forward_pre_hook(lambda mod, a, f=feats: f.setdefault("x", a[0]))
        ref(x.unsqueeze(1))
        ok &= report(f"SSFTT bands={bands} patch={p}", feats["x"], mine(x))
    return ok


def check_fahm(root):
    sys.modules.setdefault("torchsummaryX", MagicMock())
    sys.modules.setdefault("fvcore", MagicMock())
    sys.modules.setdefault("fvcore.nn", MagicMock())
    sys.path.insert(0, f"{root}/FAHM")
    m = load(f"{root}/FAHM/model/fahm.py", "fahm_official")
    ok = True
    torch.manual_seed(0)
    for c, p, e in ((30, 7, (64, 32, 16)), (31, 9, (32, 16, 8))):
        ref = m.FAHM(
            img_size=p,
            in_chans=c,
            num_classes=4,
            n_groups=[2, 2, 2],
            embed_dims=list(e),
            depths=[1, 1, 1],
        ).eval()
        mine = hs.FAHM(c, p, (2, 2, 2), e).eval()
        perturb(ref)
        skip = (
            "head.",
            "rope.",
            "pre_conv.bn4.",
            *(f"merge.{i}.batch_norm." for i in range(3)),
        )  # unused in its forward
        copy(ref, mine, skip=skip)
        x = torch.randn(2, c, p, p)
        y = ref.forward_features(x.flatten(2)).mean(dim=1)
        ok &= report(f"FAHM bands={c} patch={p} dims={e}", y, mine(x), 1e-4)
    sys.path.pop(0)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    mamba_stub()
    good = True
    for chk in (check_mambahsi, check_mambahsi_plus, check_ssftt, check_fahm):
        try:
            good &= chk(a.root)
        except Exception as e:  # noqa: BLE001  (report every network, then fail)
            print(f"{chk.__name__:<44} ERROR {type(e).__name__}: {e}")
            good = False
    print("ALL OK" if good else "SOME MISMATCH")
    raise SystemExit(0 if good else 1)
