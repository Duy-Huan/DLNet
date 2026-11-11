"""Numerical equivalence of the modern families (dlnet/zoo/ops_modern.py) with their OFFICIAL code.

    pip install --target /tmp/m tabm==0.0.3 rtdl_num_embeddings==0.0.12
    git clone https://github.com/yuqinie98/PatchTST /tmp/PatchTST
    git clone https://github.com/thuml/iTransformer /tmp/iTransformer
    PYTHONPATH=/tmp/m python scripts/verify_modern.py --patchtst /tmp/PatchTST --itransformer /tmp/iTransformer

Weights are copied across in definition order; outputs on the same random input are compared in
eval mode (dropout off; BatchNorm with the same random running statistics). Results: THIRD_PARTY.md.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import torch

from dlnet.zoo.core import _build
from dlnet.zoo.ops_modern import ITransformer, PatchTST, PeriodicLite, TabMBackbone


def tensors(m):
    return list(m.parameters()) + [b for n, b in m.named_buffers() if "num_batches" not in n]


def copy_list(a, b):
    sa, sb = [tuple(x.shape) for x in a], [tuple(x.shape) for x in b]
    if sa != sb:
        raise ValueError(f"parameter shapes differ: {sa[:8]} vs {sb[:8]}")
    with torch.no_grad():
        for x, y in zip(a, b, strict=True):
            y.copy_(x)


def report(name, ref, out):
    err = (ref - out).abs().max().item()
    ok = err < 1e-5 * max(1.0, ref.abs().max().item())
    print(f"{name:<36} max|diff| = {err:.2e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def check_tabm():
    import rtdl_num_embeddings
    import tabm

    ok = True
    torch.manual_seed(0)
    for k, w, d in ((8, 64, 3), (4, 32, 2)):
        ref = tabm.TabM(n_num_features=10, cat_cardinalities=None, d_out=None, n_blocks=d, d_block=w,
                        dropout=0.1, k=k, arch_type="tabm", start_scaling_init="random-signs").eval()  # fmt: skip
        mine = TabMBackbone(10, k, w, d, 0.1).eval()
        copy_list(tensors(ref), tensors(mine))
        x = torch.randn(5, 10)
        ok &= report(f"TabM backbone k={k} w={w} d={d}", ref(x), mine(x))
    emb = rtdl_num_embeddings.PeriodicEmbeddings(
        7, 24, n_frequencies=16, frequency_init_scale=0.1, lite=True
    )
    mine = PeriodicLite(7, 24, 16, 0.1)
    copy_list(tensors(emb), tensors(mine))
    x = torch.randn(5, 7)
    ok &= report("PLR(lite) embeddings", emb(x).flatten(1), mine(x))
    _, _, dims = _build({"op": "tabm", "k": 4, "width": 32, "depth": 2}, "vec", (10,))
    assert tuple(dims) == (4, 32), dims  # the zoo op builds the same backbone
    return ok


def _bn_same_random_stats(ref, mine):
    with torch.no_grad():
        for s, t in zip([m for m in ref.modules() if isinstance(m, torch.nn.BatchNorm1d)],
                        [m for m in mine.modules() if isinstance(m, torch.nn.BatchNorm1d)], strict=True):  # fmt: skip
            t.running_mean.normal_()
            t.running_var.uniform_(0.5, 2)
            s.running_mean.copy_(t.running_mean)
            s.running_var.copy_(t.running_var)


def check_patchtst(repo):
    sys.path.insert(0, f"{repo}/PatchTST_supervised")
    from layers.PatchTST_backbone import PatchTST_backbone

    ok = True
    torch.manual_seed(0)
    for c, t, d, heads, p, layers in ((3, 48, 32, 4, 8, 2), (1, 64, 64, 8, 16, 1)):
        ref = PatchTST_backbone(c_in=c, context_window=t, target_window=4, patch_len=p, stride=p // 2,
                                n_layers=layers, d_model=d, n_heads=heads, d_ff=2 * d, dropout=0.1,
                                head_dropout=0, revin=True, affine=True).eval()  # fmt: skip
        mine = PatchTST(c, t, d, layers, heads, p, p // 2, 2 * d, 0.1).eval()
        a = [x for n, x in ref.named_parameters() if not n.startswith("head")]
        a += [
            x for n, x in ref.named_buffers() if "num_batches" not in n and not n.startswith("head")
        ]
        copy_list(a, tensors(mine))
        _bn_same_random_stats(ref, mine)
        x = torch.randn(4, c, t)
        z = ref.revin_layer(x.permute(0, 2, 1), "norm").permute(0, 2, 1)
        z = ref.backbone(z.unfold(-1, p, p // 2).permute(0, 1, 3, 2))  # (B, C, d, n)
        z = z.permute(0, 1, 3, 2).reshape(4, -1, d)
        ok &= report(f"PatchTST c={c} t={t} d={d} p={p} L={layers}", z, mine(x))
    sys.path.pop(0)
    for k in [k for k in sys.modules if k == "layers" or k.startswith("layers.")]:
        del sys.modules[k]  # both repositories have a package named `layers`
    return ok


def check_itransformer(repo):
    from unittest.mock import MagicMock

    sys.path.insert(0, repo)
    for name in ("reformer_pytorch", "einops"):  # only used by attention variants not checked here
        try:
            __import__(name)
        except ModuleNotFoundError:
            sys.modules[name] = MagicMock()
    from model.iTransformer import Model

    ok = True
    torch.manual_seed(0)
    for c, t, d, heads, layers in ((5, 48, 32, 4, 2), (2, 96, 64, 8, 1)):
        cfg = SimpleNamespace(seq_len=t, pred_len=4, output_attention=False, use_norm=True, d_model=d,
                              embed="fixed", freq="h", dropout=0.1, class_strategy=None, factor=1,
                              n_heads=heads, e_layers=layers, d_ff=2 * d, activation="gelu")  # fmt: skip
        ref = Model(cfg).eval()
        mine = ITransformer(t, d, layers, heads, 2 * d, 0.1).eval()
        copy_list(
            [x for n, x in ref.named_parameters() if not n.startswith("projector")], tensors(mine)
        )
        x = torch.randn(3, c, t)
        xe = x.transpose(1, 2)  # official input (B, T, C), normalised as in Model.forecast
        xe = xe - xe.mean(1, keepdim=True)
        xe = xe / torch.sqrt(torch.var(xe, dim=1, keepdim=True, unbiased=False) + 1e-5)
        enc, _ = ref.encoder(ref.enc_embedding(xe, None), attn_mask=None)
        ok &= report(f"iTransformer c={c} t={t} d={d} L={layers}", enc, mine(x))
    sys.path.pop(0)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patchtst", required=True)
    ap.add_argument("--itransformer", required=True)
    a = ap.parse_args()
    good = check_tabm() & check_patchtst(a.patchtst) & check_itransformer(a.itransformer)
    print("ALL OK" if good else "SOME MISMATCH")
    raise SystemExit(0 if good else 1)
