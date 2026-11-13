"""Numerical equivalence of the v1.4 modules (dlnet/zoo/ops_more.py) with OFFICIAL code.

Run on GitHub Actions (job `verify-more`), repositories cloned at pinned commits into one folder,
timm and torch_geometric installed:

    python scripts/verify_more.py --root /tmp/more

Weights are copied by name from the official module into DLNet's and the outputs compared in eval
mode on random inputs. The graph layers are compared on random symmetric graphs (PyG works on an
edge list, DLNet on the dense adjacency).
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from types import SimpleNamespace

import torch
from torch import nn
from verify_domain import _edge_index, _graph, copy, load, module, perturb, pre_input, report

from dlnet.zoo import ops_more as om

# ------------------------------------------------------------------------- time series (TSLib)


def check_tslib(root):
    sys.path.insert(0, f"{root}/Time-Series-Library")
    dl = load(f"{root}/Time-Series-Library/models/DLinear.py", "tslib_dlinear")
    mx = load(f"{root}/Time-Series-Library/models/TSMixer.py", "tslib_tsmixer")
    td = load(f"{root}/Time-Series-Library/models/TiDE.py", "tslib_tide")
    ok = True
    torch.manual_seed(0)
    for c, t, k, individual in ((3, 24, 5, False), (4, 36, 25, True), (2, 16, 13, False)):
        cfg = SimpleNamespace(
            task_name="anomaly_detection", seq_len=t, pred_len=0, moving_avg=k, enc_in=c
        )
        ref = dl.Model(cfg, individual=individual).eval()
        mine = om.DLinear(c, t, k, individual).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(4, t, c)
        ok &= report(f"DLinear C={c} T={t} k={k} individual={individual}", ref.encoder(x),
                     mine(x.transpose(1, 2)).transpose(1, 2))  # fmt: skip
    for c, t, d in ((3, 24, 16), (7, 12, 32)):
        ref = mx.ResBlock(SimpleNamespace(seq_len=t, d_model=d, enc_in=c, dropout=0.1)).eval()
        mine = om.TSMixerBlock(c, t, d, 0.1).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(4, t, c)
        ok &= report(
            f"TSMixer ResBlock C={c} T={t} d={d}", ref(x), mine(x.transpose(1, 2)).transpose(1, 2)
        )
    for i, h, o in ((8, 16, 8), (12, 32, 5)):
        ref = td.ResBlock(i, h, o, 0.1).eval()
        mine = om.TiDEBlock(i, h, o, 0.1).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(4, 6, i)
        ok &= report(f"TiDE ResBlock {i}->{h}->{o}", ref(x), mine(x))
    return ok


# ------------------------------------------------------------------------- graphs (PyG)


def _dense(x, a):
    return torch.cat([x, a], -1)[None]


def check_pyg(root):
    from torch_geometric.nn import APPNP, GATv2Conv, GCN2Conv, SGConv

    ok = True
    torch.manual_seed(0)
    for f, o, heads, seed in ((5, 8, 1, 0), (6, 12, 4, 1), (4, 6, 2, 2)):
        x, a = _graph(9, f, 0.35, seed)
        ref = GATv2Conv(f, o // heads, heads=heads).eval()
        mine = om.GATv2(9, f, o, heads).eval()
        perturb(ref)
        copy(ref, mine)
        ok &= report(
            f"GATv2Conv F={f} out={o} heads={heads}",
            ref(x, _edge_index(a)),
            mine(_dense(x, a))[0, :, :o],
        )
    for f, hidden, layers, alpha, theta, seed in ((5, 16, 4, 0.1, 0.5, 3), (7, 8, 6, 0.2, 1.0, 4)):
        x, a = _graph(8, f, 0.4, seed)
        ei = _edge_index(a)
        ref = nn.Module()
        ref.lin = nn.Linear(f, hidden)
        ref.convs = nn.ModuleList(GCN2Conv(hidden, alpha, theta, k + 1) for k in range(layers))
        ref.eval()
        mine = om.GCNII(8, f, hidden, layers, alpha, theta, 0.0).eval()
        perturb(ref)
        copy(ref, mine)
        h = h0 = ref.lin(x).relu()
        for conv in ref.convs:
            h = conv(h, h0, ei).relu()
        ok &= report(f"GCN2Conv stack F={f} hidden={hidden} layers={layers} alpha={alpha} theta={theta}",
                     h, mine(_dense(x, a))[0, :, :hidden])  # fmt: skip
    for k, alpha, seed in ((10, 0.1, 5), (3, 0.3, 6)):
        x, a = _graph(8, 6, 0.4, seed)
        mine = om.APPNP(8, 6, 6, 6, k, alpha).eval()
        ok &= report(
            f"APPNP K={k} alpha={alpha}",
            APPNP(k, alpha).eval()(x, _edge_index(a)),
            mine.propagate(x[None], a[None])[0],
        )
    for f, o, k, seed in ((5, 8, 2, 7), (6, 4, 3, 8)):
        x, a = _graph(8, f, 0.4, seed)
        ref = SGConv(f, o, K=k).eval()
        mine = om.SGC(8, f, o, k).eval()
        perturb(ref)
        copy(ref, mine)
        ok &= report(
            f"SGConv F={f} out={o} K={k}", ref(x, _edge_index(a)), mine(_dense(x, a))[0, :, :o]
        )
    return ok


# ------------------------------------------------------------------------- field embeddings (FuxiCTR)


def check_fuxictr(root):
    layers = f"{root}/FuxiCTR/fuxictr/pytorch/layers"
    cin = load(f"{layers}/interactions/compressed_interaction_net.py", "fuxi_cin")
    bil = load(f"{layers}/interactions/bilinear_interaction.py", "fuxi_bilinear")
    se = load(f"{layers}/attentions/squeeze_excitation.py", "fuxi_se")
    dpa = load(f"{layers}/attentions/dot_product_attention.py", "fuxi_dpa")
    module("fuxictr")
    module("fuxictr.pytorch")
    module("fuxictr.pytorch.models", BaseModel=nn.Module)
    module("fuxictr.pytorch.layers", FeatureEmbedding=None, MLP_Block=None, LogisticRegression=None,
           ScaledDotProductAttention=dpa.ScaledDotProductAttention)  # fmt: skip
    autoint = load(f"{root}/FuxiCTR/model_zoo/AutoInt/src/AutoInt.py", "fuxi_autoint")
    ok = True
    torch.manual_seed(0)
    for fields, emb, units in ((5, 8, [16, 16]), (4, 6, [8, 12, 4])):
        ref = cin.CompressedInteractionNet(fields, units).eval()
        mine = om.CIN(fields, units).eval()
        perturb(ref)
        copy(ref, mine, skip=("fc.",))
        seen = {}
        pre_input(ref.fc, seen)
        e = torch.randn(4, fields, emb)
        ref(e)
        ok &= report(f"CIN fields={fields} emb={emb} units={units}", seen["x"], mine(e))
    for i, o, heads, scale, ln in (
        (16, 16, 2, False, False),
        (8, 16, 4, True, True),
        (12, 12, 1, True, False),
    ):
        ref = autoint.MultiHeadSelfAttention(i, o, heads, 0.0, True, scale, ln).eval()
        mine = om.AutoIntAttention(i, o, heads, 0.0, scale, ln).eval()
        perturb(ref)
        copy(ref, mine)
        e = torch.randn(4, 6, i)
        ok &= report(
            f"AutoInt attention {i}->{o} heads={heads} scale={scale} ln={ln}", ref(e), mine(e)
        )
    for fields, red, act in ((9, 3, "ReLU"), (5, 2, "Sigmoid")):
        ref = se.SqueezeExcitation(fields, red, act).eval()
        mine = om.FieldSE(fields, red, act.lower()).eval()
        perturb(ref)
        copy(ref, mine)
        e = torch.randn(4, fields, 8)
        ok &= report(f"FiBiNET SENET fields={fields} reduction={red} {act}", ref(e), mine(e))
    for kind in ("field_interaction", "field_all", "field_each"):
        ref = bil.BilinearInteractionV2(6, 8, kind).eval()
        ref1 = bil.BilinearInteraction(6, 8, kind).eval()
        mine = om.BilinearInteraction(6, 8, kind).eval()
        with torch.no_grad():
            ref.bilinear_W.normal_()
            ref1.bilinear_W.copy_(ref.bilinear_W)
        copy(ref, mine, skip=("triu_index",))
        e = torch.randn(4, 6, 8)
        ok &= report(f"FiBiNET bilinear V2 {kind}", ref(e), mine(e))
        ok &= report(f"FiBiNET bilinear (pairwise loop) {kind}", ref1(e), mine(e))
    return ok


# ------------------------------------------------------------------------- Set Transformer


def check_set_transformer(root):
    st = load(f"{root}/set_transformer/modules.py", "set_transformer_modules")
    ok = True
    torch.manual_seed(0)
    for i, o, heads, inds, ln in ((6, 16, 4, 5, False), (8, 8, 2, 3, True)):
        ref = st.ISAB(i, o, heads, inds, ln=ln).eval()
        mine = om.ISAB(i, o, heads, inds, ln).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(3, 11, i)
        ok &= report(
            f"Set Transformer ISAB {i}->{o} heads={heads} inds={inds} ln={ln}", ref(x), mine(x)
        )
    for d, heads, seeds, ln in ((16, 4, 1, False), (8, 2, 3, True)):
        ref = st.PMA(d, heads, seeds, ln=ln).eval()
        mine = om.PMA(d, heads, seeds, ln).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(3, 11, d)
        ok &= report(
            f"Set Transformer PMA d={d} heads={heads} seeds={seeds} ln={ln}",
            ref(x).flatten(1),
            mine(x),
        )
    return ok


# ------------------------------------------------------------------------- timm


def check_timm(root):
    from timm.layers import AttentionPoolLatent, BlurPool2d, NonLocalAttn
    from timm.models.focalnet import FocalNetBlock
    from timm.models.inception_next import InceptionDWConv2d, MetaNeXtBlock
    from timm.models.pvt_v2 import Block

    ok = True
    torch.manual_seed(0)
    for c, band, ratio, mlp in ((16, 11, 0.125, 4.0), (24, 7, 0.25, 3.0)):
        mixer = partial(InceptionDWConv2d, band_kernel_size=band, branch_ratio=ratio)
        ref = MetaNeXtBlock(c, token_mixer=mixer, mlp_ratio=mlp).eval()
        mine = om.MetaNeXtBlock(c, band, ratio, mlp).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(2, c, 14, 14)
        ok &= report(f"timm MetaNeXtBlock C={c} band={band} ratio={ratio}", ref(x), mine(x))
    for c, window, level, mlp in ((16, 3, 2, 4.0), (12, 5, 3, 2.0), (8, 3, 1, 4.0)):
        ref = FocalNetBlock(c, mlp_ratio=mlp, focal_level=level, focal_window=window).eval()
        mine = om.FocalNetBlock(c, window, level, mlp).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(2, c, 12, 12)
        ok &= report(f"timm FocalNetBlock C={c} window={window} level={level}", ref(x), mine(x))
    for c, scale, ratio in ((32, True, 1 / 8), (16, False, 1 / 2)):
        ref = NonLocalAttn(c, use_scale=scale, rd_ratio=ratio).eval()
        mine = om.NonLocalAttn(c, scale, ratio).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(2, c, 9, 9)
        ok &= report(f"timm NonLocalAttn C={c} scale={scale} rd={ratio}", ref(x), mine(x))
    for c, heads, sr, mlp in ((16, 2, 2, 4.0), (24, 4, 1, 8.0), (16, 1, 4, 2.0)):
        ref = Block(c, heads, sr_ratio=sr, mlp_ratio=mlp, qkv_bias=True).eval()
        mine = om.PVTv2Block(c, heads, sr, mlp).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(2, c, 8, 12)
        y = ref(x.flatten(2).transpose(1, 2), [8, 12]).transpose(1, 2).reshape(2, c, 8, 12)
        ok &= report(f"timm pvt_v2 Block C={c} heads={heads} sr={sr}", y, mine(x))
    for d, heads, mlp, norm in ((32, 8, 4.0, True), (24, 4, 2.0, False)):
        ref = AttentionPoolLatent(
            d, num_heads=heads, mlp_ratio=mlp, norm_layer=nn.LayerNorm if norm else None
        ).eval()
        mine = om.AttentionPoolLatent(d, heads, mlp, norm).eval()
        perturb(ref)
        copy(ref, mine)
        x = torch.randn(3, 10, d)
        ok &= report(f"timm AttentionPoolLatent d={d} heads={heads} norm={norm}", ref(x), mine(x))
    for c, size, stride in ((6, 3, 2), (4, 5, 2), (3, 3, 1)):
        x = torch.randn(2, c, 11, 12)
        ok &= report(f"timm BlurPool2d C={c} filt={size} stride={stride}", BlurPool2d(c, size, stride)(x),
                     om.BlurPool2d(c, size, stride)(x))  # fmt: skip
    return ok


# ------------------------------------------------------------------------- SimAM, GeM


def check_simam_gem(root):
    sm = load(f"{root}/SimAM/networks/attentions/simam_module.py", "simam_official")
    module("cirtorch")
    module("cirtorch.layers")
    load(f"{root}/cirtorch/cirtorch/layers/functional.py", "cirtorch.layers.functional")
    module("cirtorch.layers.normalization", L2N=nn.Identity)
    pool = load(f"{root}/cirtorch/cirtorch/layers/pooling.py", "cirtorch_pooling")
    ok = True
    torch.manual_seed(0)
    x = torch.randn(3, 6, 7, 9)
    for lam in (1e-4, 1e-2):
        ok &= report(f"SimAM lambda={lam}", sm.simam_module(e_lambda=lam)(x), om.SimAM(lam)(x))
    for p in (3.0, 1.7):
        ok &= report(f"GeM p={p}", pool.GeM(p=p)(x).flatten(1), om.GeM(p)(x))
    return ok


CHECKS = {
    "tslib": check_tslib,
    "pyg": check_pyg,
    "fuxictr": check_fuxictr,
    "set_transformer": check_set_transformer,
    "timm": check_timm,
    "simam_gem": check_simam_gem,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--only", nargs="*", default=list(CHECKS))
    a = ap.parse_args()
    ok = True
    with torch.no_grad():
        for name in a.only:
            ok &= CHECKS[name](a.root)
    print("ALL EQUIVALENT" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
