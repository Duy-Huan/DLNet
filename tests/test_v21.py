"""v2.1 mixers (dlnet/zoo/ops_v21.py): behaviour pinned by the definitions of the papers, refusals,
families. Numerical equality with the reference code: scripts/verify_v21.py (CI job verify-v21)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from dlnet.zoo import core, families
from dlnet.zoo import ops_v21 as v
from dlnet.zoo.core import _build

SEQ_OPS = ("rwkv7_block", "kda_block", "fox_block", "mla_block", "mamba3_block")


@pytest.mark.regression
@pytest.mark.parametrize("name", SEQ_OPS)
def test_every_mixer_is_causal(name):
    """Output at step t depends on steps <= t only (all five are causal language-model mixers)."""
    torch.manual_seed(0)
    m = _build({"op": name}, "seq", (10, 16))[0].eval()
    for p in m.parameters():  # zero-initialised output projections would hide a leak
        p.data.add_(torch.randn_like(p) * 0.05)
    x = torch.randn(2, 10, 16)
    y = x.clone()
    y[:, 6:] = torch.randn(2, 4, 16)
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5) and not torch.allclose(a[:, 6:], b[:, 6:])


@pytest.mark.regression
def test_delta_rule_retrieves_what_it_stored():
    """KDA with no decay and beta = 1: after writing v under a unit key k, reading with k gives v."""
    k = torch.nn.functional.normalize(torch.randn(1, 1, 1, 8), dim=-1)
    val = torch.randn(1, 1, 1, 4)
    q = k * math.sqrt(8)  # recurrent_kda scales q by 1 / sqrt(d_k)
    out = v.recurrent_kda(q, k, val, torch.zeros(1, 1, 1, 8), torch.ones(1, 1, 1))
    assert torch.allclose(out, val, atol=1e-5)
    # a second write under the same key replaces the value (delta rule), it does not add to it
    k2, v2 = k.repeat(1, 2, 1, 1), torch.cat([val, -val], 1)
    out = v.recurrent_kda(
        q.repeat(1, 2, 1, 1), k2, v2, torch.zeros(1, 2, 1, 8), torch.ones(1, 2, 1)
    )
    assert torch.allclose(out[:, 1], -val[:, 0], atol=1e-5)


@pytest.mark.regression
def test_forget_gate_closes_attention_to_the_past():
    """FoX: a forget gate of ~0 (logsigmoid -> -inf) leaves each token attending to itself only."""
    m = v.ForgettingAttention(8, 2)
    with torch.no_grad():
        m.f_proj.weight.zero_()
        m.f_proj.bias.fill_(-60.0)
        m.o_proj.weight.copy_(torch.eye(8))
        m.v_proj.weight.copy_(torch.eye(8))
    x = torch.randn(1, 5, 8)
    assert torch.allclose(m(x), x, atol=1e-5)


@pytest.mark.regression
def test_mamba3_first_step_by_hand():
    """One step from a zero state: S = dt * sigmoid(trap) k v^T, y = S q + D v, gated by silu(z)."""
    q, k = torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4)
    val, z = torch.randn(1, 1, 1, 3), torch.randn(1, 1, 1, 3)
    dt, trap = torch.tensor([[[0.3]]]), torch.tensor([[[0.4]]])
    zeros = torch.zeros(1, 4)
    y = v.mamba3_recurrence(q, k, val, torch.zeros(1, 1, 1), dt, trap, zeros, zeros, torch.zeros(1, 1, 1, 1),
                            torch.full((1,), 0.5), z)  # fmt: skip
    gamma = torch.sigmoid(trap) * dt  # angle 0 at the first step: no rotation
    s = gamma[..., None] * (k[0, 0, 0][None, :] * val[0, 0, 0][:, None])
    want = (s @ q[0, 0, 0] + 0.5 * val[0, 0, 0]) * torch.nn.functional.silu(z[0, 0, 0])
    assert torch.allclose(y[0, 0, 0], want.flatten(), atol=1e-6)


@pytest.mark.regression
def test_rwkv7_block_starts_as_identity_on_its_time_mixing():
    """RWKV-7 zero-initialises the output projections (time and channel mixing): x in, x out."""
    m = v.RWKV7Block(16, 2)
    x = torch.randn(2, 5, 16)
    assert torch.allclose(m(x), x)


def test_scan2d_runs_every_mixer_on_a_map():
    for mixer in v.MIXERS:
        m = _build({"op": "scan2d_block", "mixer": mixer, "heads": 2}, "map", (8, 3, 4))[0]
        assert m(torch.randn(2, 8, 3, 4)).shape == (2, 8, 3, 4)


@pytest.mark.parametrize(("spec", "dims", "match"), [
    ({"op": "rwkv7_block", "heads": 3}, (4, 16), "not divisible"),
    ({"op": "mamba3_block", "headdim": 16}, (4, 12), "not divisible by headdim"),
    ({"op": "mamba3_block", "d_state": 3, "headdim": 8}, (4, 16), "no rotary angle"),
])  # fmt: skip
def test_refusals(spec, dims, match):
    with pytest.raises(ValueError, match=match):
        _build(spec, "seq", dims)
    with pytest.raises(ValueError, match="unknown mixer"):
        v.make_block("lstm", 8)
    with pytest.raises(ValueError, match="must be even"):
        v.MultiheadLatentAttention(8, 2, kv_rank=4, rope_dim=3)


def test_gated_norm_swish_and_kda_shapes():
    n = v.GatedRMSNorm(4, "swish")
    x, g = torch.randn(2, 4), torch.randn(2, 4)
    want = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * torch.nn.functional.silu(g)
    assert torch.allclose(n(x, g), want, atol=1e-6)
    assert v.KimiDeltaAttention(16, 2, expand_v=2.0)(torch.randn(1, 3, 16)).shape == (1, 3, 16)


@pytest.mark.parametrize("family", ["rwkv7", "kimi_linear", "fox", "mamba3"])
def test_time_series_families(family):
    cfg = families.complete(
        {"family": family, "width": 16, "depth": 1, "dropout": 0.0}, "timeseries", (2, 12), 3
    )
    assert core.build_model(cfg)(torch.randn(4, 2, 12)).shape == (4, 3)
    if family == "kimi_linear":
        assert core.leaf_ops(cfg["spec"]).count("mla_block") == 1


def test_image_family():
    cfg = families.complete({"family": "mamba3_vision", "width": 16, "depth": 1, "dropout": 0.0, "patch": 4},
                            "image", (1, 8, 8), 4)  # fmt: skip
    assert core.build_model(cfg)(torch.randn(2, 1, 8, 8)).shape == (2, 4)


def test_dense_samples_sometimes_carry_decoder_attention():
    """space._sample_dense adds scSE to some U-Net decoders (lost by chance when the grammar grew)."""
    import numpy as np

    from dlnet.zoo import space

    specs = [space._sample_dense((3, 16, 16), np.random.default_rng(s), 8) for s in range(60)]
    assert any(sp.get("attention") == "scse" for sp in specs)


def test_mobilevitv2_resizes_a_map_that_is_not_a_patch_multiple():
    m = _build({"op": "mobilevitv2_block", "patch": 2}, "map", (8, 5, 5))[0]
    assert m(torch.randn(2, 8, 5, 5)).shape == (2, 8, 6, 6)  # rounded up to the patch, as timm
