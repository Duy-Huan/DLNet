"""v1.4 modules (dlnet/zoo/ops_more.py): properties that hold in closed form, written from the papers'
definitions and independent of the ported code. Equivalence with the authors' code, weights copied:
scripts/verify_more.py (job `verify-more`)."""

from __future__ import annotations

import itertools

import pytest
import torch
from torch import nn

from dlnet import zoo
from dlnet.zoo import families
from dlnet.zoo import ops_more as om
from dlnet.zoo.ops_graph import renormalize


def _graph(n=7, f=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(n, n, generator=g) < 0.4).float().triu(1)
    a = a + a.T
    return torch.randn(2, n, f, generator=g), a.expand(2, n, n)


def test_dlinear_starts_as_the_mean_over_time():
    """Both linear maps start at 1/T, and seasonal + trend = the series: every output is the mean."""
    for individual in (False, True):
        m = om.DLinear(3, 20, 5, individual)
        for lin in (
            [m.Linear_Seasonal, m.Linear_Trend]
            if not individual
            else [*m.Linear_Seasonal, *m.Linear_Trend]
        ):
            nn.init.zeros_(lin.bias)
        x = torch.randn(4, 3, 20)
        assert torch.allclose(m(x), x.mean(-1, keepdim=True).expand_as(x), atol=1e-5)


def test_series_decomposition_of_a_constant_has_no_seasonal_part():
    s, t = om.SeriesDecomp(7)(torch.full((2, 16, 3), 2.5))
    assert s.abs().max() < 1e-6 and torch.allclose(t, torch.full_like(t, 2.5))


def test_tsmixer_and_tide_are_their_formulas():
    m = om.TSMixerBlock(3, 12, 8, 0.0).eval()
    x = torch.randn(2, 3, 12)
    h = x + m.temporal(x)
    assert torch.allclose(m(x), h + m.channel(h.transpose(1, 2)).transpose(1, 2))
    t = om.TiDEBlock(6, 10, 4, 0.0).eval()
    v = torch.randn(5, 6)
    assert torch.allclose(t(v), t.ln(t.fc2(torch.relu(t.fc1(v))) + t.fc3(v)))


def test_gatv2_matches_a_per_node_loop_over_neighbours():
    x, a = _graph()
    m = om.GATv2(7, 4, 6, heads=2).eval()
    out = m(torch.cat([x, a], -1))[..., :6]
    xl, xr = m.lin_l(x[0]).view(7, 2, 3), m.lin_r(x[0]).view(7, 2, 3)
    for i in range(7):
        nb = [j for j in range(7) if a[0, i, j] > 0 or j == i]
        e = torch.stack(
            [(nn.functional.leaky_relu(xr[i] + xl[j], 0.2) * m.att[0]).sum(-1) for j in nb]
        )
        w = torch.softmax(e, 0)  # (neighbours, heads)
        ref = sum(w[k].unsqueeze(-1) * xl[j] for k, j in enumerate(nb)).reshape(-1) + m.bias
        assert torch.allclose(out[0, i], ref, atol=1e-5)


def test_graph_layers_carry_the_adjacency_and_ignore_other_nodes_when_isolated():
    x, a = _graph()
    a = a.clone()
    a[:, 0, :] = 0
    a[:, :, 0] = 0  # node 0 is isolated
    g = torch.cat([x, a], -1)
    for m in (
        om.GATv2(7, 4, 6),
        om.SGC(7, 4, 6, 3),
        om.APPNP(7, 4, 8, 6, 4, 0.2),
        om.GCNII(7, 4, 6, 3),
    ):
        m.eval()
        out = m(g)
        assert torch.equal(out[..., 6:], a)
        g2 = g.clone()
        g2[:, 1:, :4] += 1.0  # change every other node's features
        assert torch.allclose(m(g2)[:, 0, :6], out[:, 0, :6], atol=1e-6)


def test_sgc_and_appnp_and_gcnii_formulas():
    x, a = _graph()
    ah = renormalize(a)
    sgc = om.SGC(7, 4, 5, 2).eval()
    assert torch.allclose(sgc(torch.cat([x, a], -1))[..., :5], sgc.lin(ah @ (ah @ x)), atol=1e-6)
    ap = om.APPNP(7, 4, 8, 5, 50, 1.0).eval()  # alpha = 1: no propagation at all
    h = ap.lin2(torch.relu(ap.lin1(x)))
    assert torch.allclose(ap(torch.cat([x, a], -1))[..., :5], h, atol=1e-6)
    # the K -> infinity limit is personalised PageRank: alpha (I - (1 - alpha) A_hat)^-1 h
    ap = om.APPNP(7, 4, 8, 5, 200, 0.2).eval()
    h = ap.lin2(torch.relu(ap.lin1(x)))
    ppr = 0.2 * torch.linalg.solve(torch.eye(7) - 0.8 * ah, h)
    assert torch.allclose(ap(torch.cat([x, a], -1))[..., :5], ppr, atol=1e-4)
    layer = om.GCN2Layer(4, alpha=0.3, theta=0.5, layer=2)
    hh = 0.7 * (ah @ x) + 0.3 * x
    assert torch.allclose(
        layer(x, x, ah), (1 - layer.beta) * hh + layer.beta * hh @ layer.weight1, atol=1e-6
    )
    assert layer.beta == pytest.approx(torch.log(torch.tensor(1.25)).item())


def test_cin_first_layer_is_a_1x1_convolution_of_all_pairwise_products():
    m = om.CIN(3, [4]).eval()
    e = torch.randn(2, 3, 5)
    conv = m.cin_layer["layer_1"]
    pairs = torch.stack([e[:, h] * e[:, k] for h in range(3) for k in range(3)], 1)  # (B, 9, D)
    ref = torch.einsum("oc,bcd->bod", conv.weight[:, :, 0], pairs) + conv.bias[None, :, None]
    assert torch.allclose(m(e), ref.sum(-1), atol=1e-5)
    assert om.CIN(3, [4, 6, 2])(e).shape == (2, 12)


@pytest.mark.parametrize("kind", ["field_interaction", "field_all", "field_each"])
def test_bilinear_interaction_is_vi_w_times_vj_for_every_pair(kind):
    m = om.BilinearInteraction(4, 3, kind)
    e = torch.randn(2, 4, 3)
    out = m(e)
    for n, (i, j) in enumerate(itertools.combinations(range(4), 2)):
        w = (
            m.bilinear_W
            if kind == "field_all"
            else m.bilinear_W[n if kind == "field_interaction" else i]
        )
        assert torch.allclose(out[:, n], (e[:, i] @ w) * e[:, j], atol=1e-5)


def test_autoint_attention_and_field_se():
    m = om.AutoIntAttention(8, 8, 1, use_scale=True).eval()
    e = torch.randn(2, 5, 8)
    att = torch.softmax(m.W_q(e) @ m.W_k(e).transpose(1, 2) / 8**0.5, -1)
    assert torch.allclose(m(e), torch.relu(att @ m.W_v(e) + e), atol=1e-5)
    assert om.AutoIntAttention(8, 12, 3)(e).shape == (2, 5, 12)  # projected residual
    se = om.FieldSE(5, 2)
    w = se.excitation(e.mean(-1))
    assert torch.allclose(se(e), e * w.unsqueeze(-1)) and w.shape == (2, 5)


def test_set_transformer_blocks_are_permutation_equivariant_and_invariant():
    torch.manual_seed(0)
    x = torch.randn(2, 9, 6)
    perm = torch.randperm(9)
    isab = om.ISAB(6, 8, 2, 4, ln=True).eval()
    assert torch.allclose(isab(x)[:, perm], isab(x[:, perm]), atol=1e-5)
    pma = om.PMA(6, 2, 3).eval()
    assert pma(x).shape == (2, 18) and torch.allclose(pma(x), pma(x[:, perm]), atol=1e-5)
    pool = om.AttentionPoolLatent(6, 2).eval()
    assert pool(x).shape == (2, 6) and torch.allclose(pool(x), pool(x[:, perm]), atol=1e-5)


def test_image_blocks_start_at_or_near_the_identity():
    x = torch.randn(2, 16, 12, 12)
    assert torch.equal(om.NonLocalAttn(16).eval()(x), x)  # BatchNorm weight starts at zero
    assert (om.MetaNeXtBlock(16).eval()(x) - x).abs().max() < 1e-3  # layer scale 1e-6
    assert (om.FocalNetBlock(16).eval()(x) - x).abs().max() < 1e-2  # layer scale 1e-4


def test_inception_mixer_leaves_the_identity_channels_untouched():
    m = om.InceptionDWConv2d(16, 3, 11, 0.125)
    x = torch.randn(1, 16, 9, 9)
    y = m(x)
    assert (
        m.split_indexes == (10, 2, 2, 2)
        and torch.equal(y[:, :10], x[:, :10])
        and not torch.equal(y[:, 10:], x[:, 10:])
    )


def test_pvt_attention_reduces_the_keys_but_not_the_queries():
    m = om.PVTv2Block(8, 2, sr_ratio=4).eval()
    seen = {}
    m.attn.kv.register_forward_pre_hook(lambda mod, a: seen.update(n=a[0].shape[1]))
    x = torch.randn(1, 8, 8, 12)
    assert m(x).shape == x.shape and seen["n"] == (8 // 4) * (12 // 4)
    with pytest.raises(ValueError, match="cannot be reduced"):
        zoo.check({"op": "pvt_block", "sr_ratio": 4}, "map", (8, 2, 2))


def test_blur_pool_keeps_constants_and_removes_the_nyquist_pattern():
    m = om.BlurPool2d(3, 3, 2)
    assert torch.allclose(m(torch.full((1, 3, 8, 8), 1.5)), torch.full((1, 3, 4, 4), 1.5))
    checker = ((torch.arange(8)[:, None] + torch.arange(8)[None, :]) % 2).float().expand(1, 3, 8, 8)
    assert torch.allclose(
        m(checker), torch.full((1, 3, 4, 4), 0.5), atol=1e-6
    )  # plain stride 2 would give 0 or 1
    assert abs(float(m.filt.sum()) - 3.0) < 1e-6 and not list(m.state_dict())


def test_simam_and_gem_closed_forms():
    x = torch.randn(2, 3, 5, 5)
    y = om.SimAM(1e-4)(x)
    d = (x - x.mean((2, 3), keepdim=True)) ** 2
    energy = d / (4 * (d.sum((2, 3), keepdim=True) / 24 + 1e-4)) + 0.5
    assert torch.allclose(y, x * torch.sigmoid(energy)) and not list(om.SimAM().parameters())
    pos = torch.rand(2, 3, 5, 5) + 0.1
    assert torch.allclose(om.GeM(1.0)(pos), pos.mean((2, 3)), atol=1e-5)  # p = 1: average pooling
    assert torch.allclose(
        om.GeM(300.0)(pos), pos.amax((2, 3)), atol=0.02
    )  # p -> infinity: max pooling


NEW_FAMILIES = {
    "tabular": (("tide_mlp",), (20,)),
    "image": (("inceptionnext", "focalnet", "pvt"), (3, 16, 16)),
    "timeseries": (("dlinear", "tsmixer"), (4, 32)),
    "graph": (("gatv2_net", "gcnii_net", "appnp_net", "sgc_net"), (8, 12)),
    "recsys": (("xdeepfm", "autoint", "fibinet"), (7, 9, 11, 5)),
}


@pytest.mark.parametrize(
    ("domain", "family"), [(d, f) for d, (fs, _) in NEW_FAMILIES.items() for f in fs]
)
def test_new_families_build_and_train_one_step(domain, family):
    dims = NEW_FAMILIES[domain][1]
    cfg = {"family": family, "width": 16, "depth": 2, "dropout": 0.1}
    model = families.build(cfg, domain, dims, 3)
    kind = families.DOMAINS[domain][2]
    if kind == "ids":
        x = torch.stack([torch.randint(0, c, (4,)) for c in dims], 1)
    elif kind == "graph":
        a = (torch.rand(4, 8, 8) < 0.4).float()
        x = torch.cat([torch.randn(4, 8, 4), ((a + a.transpose(1, 2)) > 0).float()], -1)
    else:
        x = torch.randn(4, *dims)
    out = model(x)
    assert out.shape == (4, 3)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert family in getattr(families, {"tabular": "TABULAR", "image": "IMAGE", "timeseries": "TIMESERIES",
                                        "graph": "GRAPH", "recsys": "RECSYS"}[domain])  # fmt: skip


@pytest.mark.parametrize(
    ("domain", "family"), [(d, f) for d, (fs, _) in NEW_FAMILIES.items() for f in fs]
)
def test_new_families_export_as_standalone_pytorch_and_draw(domain, family):
    """model.py without dlnet, same weights and outputs. focalnet mixes modules that import
    torch.nn.functional in two spellings (`import a.b as F`, `from a import b as F`)."""
    from dlnet import diagram, standalone

    cfg = families.complete(
        {"family": family, "width": 16, "depth": 1, "dropout": 0.0},
        domain,
        NEW_FAMILIES[domain][1],
        3,
    )
    text, status = standalone.verified_source(cfg)
    assert status == "identical" and "dlnet" not in text.split('"""', 2)[2]
    assert diagram.svg(cfg).startswith("<svg")
