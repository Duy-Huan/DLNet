"""v1.0 blocks: closed-form answers (the comparison with the authors' code is in scripts/verify_*.py,
run on GitHub Actions) and properties every block must have in DLNet (export, autocast, causality)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dlnet import zoo
from dlnet.zoo import families
from dlnet.zoo import ops_graph2 as og
from dlnet.zoo import ops_hsi as hs
from dlnet.zoo import ops_next as nx
from dlnet.zoo import ops_seq2 as s2


def test_haar_wavelet_transform_is_perfectly_inverted():
    w = nx.WTConv2d(3, 5, True, 1)
    x = torch.randn(2, 3, 8, 6)
    torch.testing.assert_close(w._iwt(w._wt(x)), x, atol=1e-6, rtol=1e-6)


def test_wtconv_with_zero_wavelet_branch_is_its_depthwise_conv():
    w = nx.WTConv2d(4, 3, True, 2)
    with torch.no_grad():
        for s in w.wavelet_scale:
            s.weight.zero_()
    x = torch.randn(1, 4, 9, 9)
    torch.testing.assert_close(w(x), w.base_conv(x))


def test_ska_equals_the_direct_sum_of_its_definition():
    x, wt = torch.randn(1, 8, 5, 4), torch.randn(1, 2, 9, 5, 4)  # 8 channels share 2 kernel groups
    out = nx.ska(x, wt)
    xp = torch.nn.functional.pad(x, (1, 1, 1, 1))
    ref = torch.zeros_like(x)
    for c in range(8):
        for i in range(5):
            for j in range(4):
                for k in range(9):
                    ref[0, c, i, j] += xp[0, c, i + k // 3, j + k % 3] * wt[0, c % 2, k, i, j]
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_dyt_is_tanh_of_alpha_x_with_affine():
    d = nx.DynamicTanh(4, True, 0.7)
    x = torch.randn(3, 4)
    torch.testing.assert_close(d(x), torch.tanh(0.7 * x))


def test_gated_attention_without_gate_is_softmax_attention():
    a = nx.GatedAttention(8, 2, "none", qk_norm=False)
    x = torch.randn(1, 5, 8)
    q, k, v = (m(x).view(1, 5, 2, 4).transpose(1, 2) for m in (a.q_proj, a.k_proj, a.v_proj))
    ref = torch.softmax(q @ k.transpose(-1, -2) / 2.0, -1) @ v
    torch.testing.assert_close(a(x), a.o_proj(ref.transpose(1, 2).reshape(1, 5, 8)))


@pytest.mark.parametrize("block", ["mlstm", "gdn", "diff", "gated"])
def test_causal_mixers_do_not_look_ahead(block):
    torch.manual_seed(0)
    m = {"mlstm": lambda: s2.MLSTMBlock(16, 4), "gdn": lambda: s2.GDNBlock(16, 4, 2),
         "diff": lambda: nx.DiffBlock(16, 2, 2, 1, True), "gated": lambda: nx.GatedBlock(16, 2, 2, "headwise", True)}[block]().eval()  # fmt: skip
    x = torch.randn(1, 10, 16)
    y = x.clone()
    y[:, 6:] = torch.randn(1, 4, 16)  # change the future only
    torch.testing.assert_close(m(x)[:, :6], m(y)[:, :6], atol=1e-5, rtol=1e-5)


def test_gated_delta_rule_one_step_closed_form():
    q, k, v = (
        torch.randn(1, 1, 1, 4),
        torch.nn.functional.normalize(torch.randn(1, 1, 1, 4), dim=-1),
        torch.randn(1, 1, 1, 3),
    )
    beta, g = torch.full((1, 1, 1), 0.5), torch.zeros(1, 1, 1)
    o = s2.recurrent_gated_delta_rule(q, k, v, beta, g)
    s = 0.5 * k[0, 0, 0, :, None] * v[0, 0, 0, None, :]  # S = beta k v^T from S0 = 0
    torch.testing.assert_close(o[0, 0, 0], (q[0, 0, 0] / 2.0) @ s)


def test_kat_rational_initialised_as_identity_and_gelu():
    x = torch.linspace(-2.5, 2.5, 64).view(1, 8, 8)
    torch.testing.assert_close(s2.KATGroup(8, "identity")(x), x)
    near = torch.linspace(-1, 1, 64).view(1, 8, 8)  # the fitted GELU init is close only near 0
    assert (s2.KATGroup(8, "gelu")(near) - torch.nn.functional.gelu(near)).abs().max() < 0.03


def test_kat_denominator_uses_absolute_powers_like_the_kernel():
    k = s2.KATGroup(1, "gelu")
    x = torch.tensor([[[-1.5]]])
    a, b = k.weight_numerator[0].detach(), k.weight_denominator[0].detach()
    p = sum(c * (-1.5) ** i for i, c in enumerate(a))
    q = 1 + sum(abs(c) * 1.5 ** (i + 1) for i, c in enumerate(b))
    torch.testing.assert_close(k(x)[0, 0, 0], p / q)


def test_rwse_matches_numpy_matrix_powers():
    a = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
    p = a / a.sum(1, keepdims=True)
    expect = np.stack([np.diag(np.linalg.matrix_power(p, k)) for k in (1, 2, 3)], -1)
    np.testing.assert_allclose(
        og.rw_landing_probs(torch.tensor(a)[None], [1, 2, 3])[0].numpy(), expect, atol=1e-9
    )


def test_dense_gat_gives_isolated_nodes_zero_like_pyg():
    gat = og.DenseGATConv(3, 4, 2)
    a = torch.zeros(1, 3, 3)
    a[0, 0, 1] = a[0, 1, 0] = 1
    y = gat(torch.randn(1, 3, 3), a)
    assert torch.all(y[0, 2] == 0) and torch.isfinite(y).all()


def test_dcnv3_split_embedding_layout():
    e = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    x = og.fcn_embedding(e)
    torch.testing.assert_close(
        x[0, :6], torch.tensor([0.0, 1, 4, 5, 8, 9])
    )  # first halves of each field


def test_mambahsi_rejects_what_the_original_cannot_run():
    with pytest.raises(ValueError):
        zoo.check({"op": "mambahsi_block", "token_num": 8}, "map", (30, 9, 9))


@pytest.mark.parametrize(
    ("domain", "family", "dims"),
    [
        ("hsi", "mambahsi", (32, 9, 9)), ("hsi", "ssftt", (30, 9, 9)), ("hsi", "fahm", (30, 7, 7)),
        ("tabular", "realmlp", (10,)), ("timeseries", "timemixer", (3, 32)), ("timeseries", "timexer", (3, 32)),
        ("timeseries", "xlstm", (2, 24)), ("timeseries", "gated_deltanet", (2, 24)),
        ("image", "mambavision", (3, 16, 16)), ("image", "kat", (3, 16, 16)),
        ("graph", "gps_net", (6, 9)), ("graph", "polynormer_net", (6, 9)), ("graph", "sgformer_net", (6, 9)),
    ],
)  # fmt: skip
def test_v1_families_train_export_and_run_under_autocast(domain, family, dims):
    builder, kind = families.DOMAINS[domain][1], families.DOMAINS[domain][2]
    cfg = {"family": family, "width": 32, "depth": 2, "dropout": 0.0, "patch": 4}
    m = zoo.build_model(
        {
            "spec": builder(cfg, dims),
            "input": {"kind": kind, "dims": list(dims)},
            "head": "cls",
            "out_dim": 3,
        }
    )
    if kind == "graph":
        n = dims[0]
        a = (torch.rand(2, n, n) > 0.5).float()
        x = torch.cat([torch.randn(2, n, dims[1] - n), ((a + a.transpose(1, 2)) > 0).float()], -1)
    else:
        x = torch.randn(2, *dims)
    m(x).sum().backward()
    m.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert m(x).shape == (2, 3)
    if family not in (
        "fahm",
        "mambahsi",
        "xlstm",
        "gated_deltanet",
        "mambavision",
    ):  # python loops: slow to trace
        prog = torch.export.export(m, (x,))
        torch.testing.assert_close(prog.module()(x), m(x), atol=1e-5, rtol=1e-4)


def test_ctr_families_on_ids():
    for f in ("finalmlp", "dcnv3", "wukong"):
        spec = families.recsys({"family": f, "width": 8, "depth": 1, "dropout": 0.0}, (10, 7, 5))
        m = zoo.build_model(
            {
                "spec": spec,
                "input": {"kind": "ids", "dims": [10, 7, 5]},
                "head": "cls",
                "out_dim": 1,
            }
        )
        ids = torch.stack([torch.randint(0, c, (4,)) for c in (10, 7, 5)], 1)
        assert m(ids).shape == (4, 1)


def test_ssftt_and_fahm_forward_shapes():
    assert hs.SSFTT(30)(torch.randn(2, 30, 9, 9)).shape == (2, 64)
    assert hs.FAHM(30, 7, (2, 2, 2), (32, 16, 8))(torch.randn(2, 30, 7, 7)).shape == (2, 8)
