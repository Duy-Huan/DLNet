"""Tensor layers, Tensor-LoRA and HOSVD: closed-form checks (agent.md sec. 4, M7)."""

import math

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from dlnet import tensor
from dlnet import tensor_lora as tl
from dlnet.zoo import count_params
from dlnet.zoo.families import tabular_model as build_model
from dlnet.zoo.ops_vec import TRLinear, TTLinear

# ------------------------------------------------------------------ HOSVD (data side)


def _tucker_tensor(shape=(12, 10, 8), ranks=(3, 2, 4), seed=0):
    rng = np.random.default_rng(seed)
    core = rng.normal(size=ranks)
    return tensor.tucker_to_tensor(core, [rng.normal(size=(n, r)) for n, r in zip(shape, ranks)])


def test_hosvd_recovers_exact_multilinear_rank():
    T = _tucker_tensor()
    assert tensor.multilinear_rank(T, energy=1 - 1e-12) == [3, 2, 4]
    core, factors = tensor.hosvd(T, (3, 2, 4))
    assert core.shape == (3, 2, 4)
    assert np.allclose(tensor.tucker_to_tensor(core, factors), T, atol=1e-9)
    for u in factors:
        assert np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-10)  # orthonormal factors


def test_hosvd_truncation_error_respects_bound():
    T = _tucker_tensor() + 0.1 * np.random.default_rng(1).normal(size=(12, 10, 8))
    ranks = (2, 2, 2)
    core, factors = tensor.hosvd(T, ranks)
    err = np.linalg.norm(T - tensor.tucker_to_tensor(core, factors))
    assert 0 < err <= tensor.hosvd_error_bound(T, ranks) + 1e-9
    rep = tensor.compress_report(T, energy=0.99)
    assert rep["relative_error"] <= rep["error_bound"] + 1e-9 and 0 < rep["stored_fraction"] < 1
    with pytest.raises(ValueError):
        tensor.hosvd(T, (2, 2))


def test_mode_product_matches_definition():
    rng = np.random.default_rng(0)
    T, M = rng.normal(size=(3, 4, 5)), rng.normal(size=(6, 4))
    want = np.einsum("ajb,ij->aib", T, M)
    assert np.allclose(tensor.mode_product(T, M, 1), want)


# -------------------------------------------------------------- TT / TR dense layers


def test_tr_linear_equals_its_dense_matrix_and_param_formula():
    torch.manual_seed(0)
    lay = TRLinear(12, 6, rank=3)  # 12 = 3*4, 6 = 2*3
    W = torch.einsum("aiob,bjpa->opij", lay.g1, lay.g2).reshape(6, 12)
    x = torch.randn(5, 12)
    assert torch.allclose(lay(x), x @ W.t() + lay.bias, atol=1e-5)
    assert count_params(lay) == 3 * 3 * (3 * 2 + 4 * 3) + 6
    tt = TTLinear(12, 6, rank=3)
    assert count_params(tt) == 3 * (3 * 2 + 4 * 3) + 6  # TT: r instead of r^2


def test_tr_resmlp_trains_and_is_smaller_than_dense():
    cfg = {"family": "tr_resmlp", "width": 64, "depth": 2, "dropout": 0.0, "rank": 2}
    m = build_model(cfg, 10, 3)
    dense = build_model({**cfg, "family": "resmlp"}, 10, 3)
    assert m(torch.randn(4, 10)).shape == (4, 3)
    assert count_params(m) < 0.5 * count_params(dense)
    m(torch.randn(4, 10)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


# ------------------------------------------------------------------------ Tensor-LoRA


@pytest.mark.parametrize("fmt", tl.FORMATS)
def test_delta_shape_param_count_zero_start_and_gradients(fmt):
    torch.manual_seed(0)
    base = nn.Linear(12, 8)
    lay = tl.TensorLoRALinear(base, fmt, rank=3)
    x = torch.randn(5, 12)
    assert lay.delta().shape == (8, 12)
    assert lay.n_trainable() == tl.expected_params(fmt, 12, 8, 3)
    assert torch.allclose(lay(x), base(x))  # dW = 0 at the start
    assert not base.weight.requires_grad and not base.bias.requires_grad
    opt = torch.optim.SGD([p for p in lay.parameters() if p.requires_grad], lr=0.1)
    for _ in range(3):  # after the zero factor moved, every core must receive gradient
        opt.zero_grad()
        (lay(x) ** 2).sum().backward()
        opt.step()
    opt.zero_grad()
    (lay(x) ** 2).sum().backward()
    for name, p in lay.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert not torch.allclose(lay(x), base(x))


def test_tt_with_trivial_factors_is_exactly_lora():
    """in = (in, 1), out = (1, out): dW[o, i] = sum_r G1[i, r] G2[r, o], i.e. B A of LoRA."""
    torch.manual_seed(0)
    lay = tl.TensorLoRALinear(nn.Linear(6, 4), "tt", rank=3, in_factors=(6, 1), out_factors=(1, 4))
    with torch.no_grad():
        lay.cores[1].normal_()
    A = lay.cores[0].reshape(6, 3).t()  # (r, in)
    B = lay.cores[1].reshape(3, 4).t()  # (out, r)
    assert torch.allclose(lay.delta(), B @ A, atol=1e-6)
    assert lay.n_trainable() == tl.expected_params("lora", 6, 4, 3)


def test_kronecker_matches_torch_kron_and_ring_matches_trace():
    torch.manual_seed(0)
    lay = tl.TensorLoRALinear(nn.Linear(6, 4), "kronecker", rank=2, n=2)
    with torch.no_grad():
        lay.s.normal_()
    want = sum(torch.kron(lay.A_k[i], lay.s[i] @ lay.t[i].t()) for i in range(2))
    assert torch.allclose(lay.delta(), want, atol=1e-6)
    g1, g2 = torch.randn(3, 2, 2, 3), torch.randn(3, 3, 2, 3)
    W = tl.cores_to_matrix([g1, g2], ring=True)
    # entry (o1=1,o2=0 ; i1=0,i2=2) = Tr(G1[0,1] G2[2,0])
    assert W[1 * 2 + 0, 0 * 3 + 2] == pytest.approx(float(torch.trace(g1[:, 0, 1] @ g2[:, 2, 0])))
    with pytest.raises(ValueError):
        tl.TensorLoRALinear(nn.Linear(7, 4), "kronecker", n=2)


def test_tt_svd_exact_at_full_rank_and_bounded_when_truncated():
    torch.manual_seed(0)
    W = torch.randn(8, 12, dtype=torch.float64)
    m, n = (3, 4), (2, 4)
    full = tl.tt_svd(W, m, n, rank=64)
    assert torch.allclose(tl.cores_to_matrix(full), W, atol=1e-10)
    # one SVD step for 2 cores: error = sqrt(sum of dropped singular values squared)
    C = W.reshape(2, 4, 3, 4).permute(2, 0, 3, 1).reshape(6, 16)
    s = torch.linalg.svdvals(C)
    err = torch.linalg.norm(tl.cores_to_matrix(tl.tt_svd(W, m, n, rank=2)) - W)
    assert float(err) == pytest.approx(math.sqrt(float((s[2:] ** 2).sum())), rel=1e-8)
    three = tl.tt_svd(torch.randn(8, 8, dtype=torch.float64), (2, 2, 2), (2, 2, 2), rank=64)
    assert len(three) == 3 and three[0].shape[0] == 1 and three[-1].shape[-1] == 1


def test_apply_and_merge_on_a_model():
    torch.manual_seed(0)
    model = build_model({"family": "mlp", "width": 16, "depth": 2, "dropout": 0.0}, 8, 3)
    x = torch.randn(4, 8)
    before = model(x)
    n_all = count_params(model)
    tl.apply_tensor_lora(
        model, "tr", rank=2, skip=("head.f.1",)
    )  # the zoo head is Head.f = [Dropout, Linear]
    assert torch.allclose(model(x), before)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < trainable < n_all and model.head.f[1].weight.requires_grad
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.1)
    after = model(x)
    tl.merge_tensor_lora(model)
    assert not any(isinstance(m, tl.TensorLoRALinear) for m in model.modules())
    assert torch.allclose(model(x), after, atol=1e-5)
