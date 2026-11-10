"""M15: data card, seed selection, Hyperband, aging evolution, portfolio, new proxies (closed-form answers)."""

import numpy as np
import pytest
import torch
from torch import nn

from dlnet import datacard, meta, proxies, search, seeds
from dlnet.zoo import space


def test_hyperband_brackets_follow_algorithm_1():
    """R = 81, eta = 3 (paper sec. 3.3): s_max = 4, B = 405, n = ceil(B / R * eta^s / (s + 1))."""
    b = search.hyperband_brackets(81, 3, 1)
    assert b[0] == [(81, 1), (27, 3), (9, 9), (3, 27), (1, 81)]
    assert [br[0][0] for br in b] == [81, 34, 15, 8, 5]  # Algorithm 1 (Table 1 shows 27, 9, 6)
    assert all(br[-1][1] == 81 for br in b)  # every bracket ends at the full resource
    assert search.hyperband_brackets(36, 3, 4) == [
        [(9, 4), (3, 12), (1, 36)],
        [(5, 12), (1, 36)],
        [(3, 36)],
    ]


def test_proxy_vote_is_pairwise_majority():
    s = [
        {"synflow": 3, "naswot": 3, "snip": 1},  # wins 2 of 3 against both others
        {"synflow": 2, "naswot": 1, "snip": 3},
        {"synflow": 1, "naswot": 2, "snip": 2},
    ]
    wins = seeds.proxy_vote(s)
    assert wins[0] == 2 and wins[0] > wins[1] and wins[0] > wins[2]
    assert seeds.proxy_vote([{"synflow": None}, {"synflow": None}]) == [0.5, 0.5]  # no vote: tie


def test_borda_weights_and_ties():
    total = seeds.borda([([3, 1, 2], 1.0), ([0, 0, 1], 2.0)])
    # list 1 points: 2, 0, 1 ; list 2 points: 0.5, 0.5, 2 (ties share)
    assert np.allclose(total, [2 + 1, 0 + 1, 1 + 4])


def test_portfolio_greedy_picks_complementary_families():
    recs = [
        {"plugin": "p", "features": [0.0], "family_scores": {"a": 1.0, "b": 0.5, "c": 0.0}},
        {"plugin": "p", "features": [1.0], "family_scores": {"a": 0.0, "b": 0.5, "c": 1.0}},
    ]
    # alone, a, b and c all have mean rank 0.5: the tie goes to "a"; then c removes all error
    assert meta.portfolio(recs, "p", size=1) == ["a"]
    assert meta.portfolio(recs, "p", size=3) == ["a", "c", "b"]
    recs.append({"plugin": "p", "features": [2.0], "family_scores": {"a": 0.0, "b": 0.9, "c": 0.5}})
    assert meta.portfolio(recs, "p", size=1) == ["b"]  # mean ranks a 0.67, b 0.33, c 0.5


def test_tournament_with_full_sample_returns_the_best():
    pop = [({"id": i}, float(v)) for i, v in enumerate([0.2, 0.9, 0.5])]
    rng = np.random.default_rng(0)
    assert space.tournament(pop, 50, rng)["id"] == 1


def test_spectral_slope_recovers_power_law():
    rng = np.random.default_rng(0)
    H = 64
    fy, fx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(H), indexing="ij")
    f = np.hypot(fy, fx)
    f[0, 0] = 1
    X = []
    for _ in range(20):
        spec = (
            rng.normal(size=(H, H)) + 1j * rng.normal(size=(H, H))
        ) / f  # amplitude 1/f -> P ~ f^-2
        X.append(np.fft.ifft2(spec).real)
    card = datacard.spectrum_features(np.array(X)[:, None])
    assert abs(card["spectral_slope"] - 2.0) < 0.3
    white = rng.normal(size=(20, 1, H, H))
    assert abs(datacard.spectrum_features(white)["spectral_slope"]) < 0.2
    assert datacard.correlation_length(white) <= 1.0


def test_position_dependence_extremes():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 50)
    X = np.zeros((100, 1, 4, 4))
    X[y == 1, 0, 0, 0] = 1.0  # the class is a fixed pixel: all variance is between classes
    assert datacard.position_dependence(X, y) == pytest.approx(1.0)
    Z = rng.normal(size=(100, 1, 4, 4))
    assert datacard.position_dependence(Z, rng.permutation(y)) < 0.1


def test_zico_matches_the_official_formula_on_a_linear_layer():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(3, 2))
    batches = [(torch.randn(4, 3), torch.randint(0, 2, (4,))) for _ in range(3)]
    grads = []
    for X, y in batches:
        m.zero_grad()
        nn.functional.cross_entropy(m(X), y).backward()
        grads.append(m[0].weight.grad.reshape(-1).numpy().copy())
    g = np.array(grads)
    sd = g.std(0)
    nz = np.nonzero(sd)[0]
    want = np.log(np.sum(np.abs(g).mean(0)[nz] / sd[nz]))
    assert proxies.zico(m, batches, "classification") == pytest.approx(want, rel=1e-6)


def test_snip_is_sum_of_abs_grad_times_weight():
    torch.manual_seed(0)
    m = nn.Linear(3, 2, bias=False)
    X, y = torch.randn(5, 3), torch.randint(0, 2, (5,))
    m.zero_grad()
    nn.functional.cross_entropy(m(X), y).backward()
    want = float((m.weight.grad * m.weight.detach()).abs().sum())
    assert proxies.snip(m, X, y, "classification") == pytest.approx(want, rel=1e-6)


def _cands():
    from dlnet.zoo import presets

    out = []
    for name, spec in presets.presets("map", (3, 16, 16), "cls", 8).items():
        out.append({"family": name, "origin": f"preset:{name}", "spec": spec, "params": 1000})
    return out


def test_select_keeps_n_with_an_anchor_and_reacts_to_the_data_card():
    cands = _cands()
    texture = {"n_train": 800, "samples_per_class_min": 200, "conv_bias_gain": 0.6,
               "high_freq_energy": 0.7, "spectral_slope": 0.0}  # fmt: skip
    kept, rep = seeds.select(cands, texture, n=6)
    assert len(kept) == 6 and any(c["family"] == "resnet" for c in kept)
    assert rep["used_for_pruning"] and rep["rules_fired"]
    assert not any("global_attention" in c["seed_traits"] for c in kept[:3])
    # context words move light models up
    kept2, _ = seeds.select(_cands(), {"n_train": 800}, n=4, context="real-time mobile camera")
    assert sum("light" in c["seed_traits"] for c in kept2) >= 2


def test_image_card_runs_on_tiny_data():
    from dlnet import array_data

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8, 8)).astype(np.float32)
    y = np.repeat(np.arange(3), 20)
    d = array_data.build(X, y, "image", "classification", 0)
    card = datacard.image_card(d)
    assert card["n_classes"] == 3 and card["height"] == 8 and "conv_bias_gain" in card
