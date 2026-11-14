"""Numerical equivalence of the v1.8 search blocks (dlnet/zoo/ops_v18.py) with timm.

    pip install "timm==1.0.31"
    python scripts/verify_v18.py

Each block is built by timm and by DLNet with the same settings; timm's weights (perturbed, so a
zero-initialised layer cannot hide a difference) are copied into DLNet's block in order, shapes
checked, and the outputs compared in eval mode on random inputs. Exit code 1 on any mismatch.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dlnet.zoo import ops_v18 as v

TOL = 1e-5


def copy_in_order(src: nn.Module, dst: nn.Module) -> None:
    a = [t for k, t in src.state_dict().items() if not k.endswith("num_batches_tracked")]
    b = [(k, t) for k, t in dst.state_dict().items() if not k.endswith("num_batches_tracked")]
    if [tuple(t.shape) for t in a] != [tuple(t.shape) for _, t in b]:
        raise AssertionError(
            f"tensors differ:\n timm {[tuple(t.shape) for t in a]}\n dlnet {[tuple(t.shape) for _, t in b]}"
        )
    dst.load_state_dict({k: t for (k, _), t in zip(b, a, strict=True)}, strict=False)


def perturb(m: nn.Module) -> None:
    with torch.no_grad():
        for t in m.state_dict().values():
            if t.is_floating_point():
                t.add_(torch.randn_like(t) * 0.1)
            if t.dim() == 1 and t.is_floating_point() and (t < 0).any():
                pass
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_var.abs_().add_(0.5)


def compare(name, ref, mine, x) -> bool:
    torch.manual_seed(0)
    perturb(ref)
    copy_in_order(ref, mine)
    ref.eval(), mine.eval()
    with torch.no_grad():
        ya, yb = ref(x), mine(x)
    diff = float((ya - yb).abs().max()) if ya.shape == yb.shape else float("inf")
    ok = diff <= TOL * max(1.0, float(ya.abs().max()))
    print(
        f"{'ok ' if ok else 'BAD'} {name:<46} {tuple(x.shape)} -> {tuple(yb.shape)}  max|diff| {diff:.2e}"
    )
    return ok


def main() -> int:
    from timm.layers import ConvNormAct
    from timm.models import fasternet, mambaout, mobilevit
    from timm.models.byobnet import LayerFn

    torch.manual_seed(0)
    ok = True
    for dim, n_div, ratio, ls in ((32, 4, 2.0, 0.0), (48, 2, 1.0, 1e-5), (24, 4, 2.0, 0.5)):
        ref = fasternet.MLPBlock(dim, n_div, ratio, 0.0, ls)
        ok &= compare(f"pconv_block d={dim} n_div={n_div} ls={ls}", ref, v.PConvBlock(dim, n_div, ratio, ls),
                      torch.randn(2, dim, 9, 11))  # fmt: skip
    for dim, exp, k, cr in ((32, 8 / 3, 7, 1.0), (48, 2.0, 3, 0.5)):
        ref = mambaout.GatedConvBlock(dim, exp, k, cr)

        class ChannelsLast(nn.Module):  # timm's stage feeds the block [B, H, W, C]
            def __init__(self, b):
                super().__init__()
                self.b = b

            def forward(self, x):
                return self.b(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        ok &= compare(f"gated_cnn_block d={dim} exp={exp:.2f} k={k} cr={cr}", ChannelsLast(ref),
                      v.GatedCNNBlock(dim, exp, k, cr), torch.randn(2, dim, 10, 10))  # fmt: skip
    layers = LayerFn(conv_norm_act=partial(ConvNormAct, act_layer=nn.SiLU), act=nn.SiLU)
    for cin, t, depth, patch, heads, hw in ((32, 48, 2, 2, 4, (8, 8)), (24, 24, 1, 2, 2, (9, 7))):
        ref = mobilevit.MobileVitBlock(cin, cin, transformer_dim=t, transformer_depth=depth, patch_size=patch,
                                       num_heads=heads, layers=layers)  # fmt: skip
        ok &= compare(f"mobilevit_block c={cin} t={t} depth={depth} {hw}", ref,
                      v.MobileViTBlock(cin, t, depth, patch, heads), torch.randn(2, cin, *hw))  # fmt: skip
    for cin, t, depth, patch, hw in ((32, 32, 2, 2, (8, 8)), (24, 48, 1, 2, (10, 6))):
        ref = mobilevit.MobileVitV2Block(cin, cin, transformer_dim=t, transformer_depth=depth, patch_size=patch,
                                         layers=layers)  # fmt: skip
        ok &= compare(f"mobilevitv2_block c={cin} t={t} depth={depth} {hw}", ref,
                      v.MobileViTv2Block(cin, t, depth, patch), torch.randn(2, cin, *hw))  # fmt: skip
    from timm.models import edgenext

    for dim, scales, heads, pos in ((48, 2, 8, True), (64, 4, 4, False), (36, 3, 4, True)):
        ref = edgenext.SplitTransposeBlock(
            dim, num_scales=scales, num_heads=heads, use_pos_emb=pos, ls_init_value=0.5
        )
        ok &= compare(f"sdta_block d={dim} scales={scales} heads={heads} pos={pos}", ref,
                      v.SDTABlock(dim, scales, heads, 4.0, pos, 0.5), torch.randn(2, dim, 7, 9))  # fmt: skip
    print("verify_v18:", "all equal" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
