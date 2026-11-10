"""Intrinsic-dimension estimators: known answers and stability (agent.md M6, 16.4)."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from dlnet import profile


def _subspace(d, n=1000, D=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) @ rng.normal(size=(d, D))


@pytest.mark.parametrize("d", [2, 5])
def test_estimators_recover_known_dimension(d):
    X = _subspace(d)
    assert profile.twonn_intrinsic_dim(X) == pytest.approx(d, rel=0.15)
    assert profile.mle_intrinsic_dim(X) == pytest.approx(d, rel=0.15)


def test_mle_matches_formula_on_a_line():
    """Points 0, 1, 2, ... on a line: neighbour distances are known, so 1/m_k is closed form."""
    X = np.arange(41, dtype=float)[:, None]
    k = 4
    # interior point: T = (1, 1, 2, 2) -> mean_j<k log(T_k / T_j) = (2 ln 2 + 0) / 3
    T = np.sort(np.abs(X - X.T), axis=1)[:, 1 : k + 1]
    want = 1.0 / np.mean(np.mean(np.log(T[:, -1:] / T[:, :-1]), axis=1))
    assert profile.mle_intrinsic_dim(X, k=k) == pytest.approx(want)
    assert np.mean(np.log(T[20, -1] / T[20, :-1])) == pytest.approx(2 * np.log(2) / 3)


def test_report_is_stable_on_clean_data_and_flags_disagreement():
    r = profile.intrinsic_dim_report(_subspace(5))
    assert r["stable"] and r["value"] == pytest.approx(5, rel=0.15)
    assert r["twonn"]["std"] < 0.5 and r["rel_disagreement"] < 0.3
    assert profile.intrinsic_dim_report(_subspace(5)) == r  # deterministic for a fixed seed
    tiny = profile.intrinsic_dim_report(np.random.default_rng(0).normal(size=(12, 3)))
    assert tiny["stable"] in (True, False) and set(tiny) >= {"twonn", "mle", "value"}


def test_duplicates_and_tiny_inputs_do_not_crash():
    X = np.repeat(_subspace(3, n=50), 2, axis=0)  # every point duplicated
    assert np.isfinite(profile.mle_intrinsic_dim(X)) and np.isfinite(profile.twonn_intrinsic_dim(X))
    assert np.isnan(profile.mle_intrinsic_dim(np.zeros((3, 2))))


def test_capacity_hint_is_information_only():
    h = profile.capacity_hint(5.0, 1000, 50)
    assert h == {"min_width": 16, "scaling_exponent": 0.8, "used_for_pruning": False}
    assert profile.capacity_hint(40.0, 1000, 8)["min_width"] == 8  # capped by the feature count
    assert profile.capacity_hint(float("nan"), 10, 4)["min_width"] >= 2


def test_profile_contains_estimator_report():
    from dlnet import core

    X = _subspace(4, n=300, D=10).astype("float32")
    d = core.prepare(X, (X[:, 0] > 0).astype(int), "classification", seed=0)
    p = profile.profile(d)
    assert set(p["intrinsic_dim_estimators"]) >= {"twonn", "mle", "value", "stable"}
    assert p["capacity_hint"]["used_for_pruning"] is False
    import json

    json.dumps(p)  # must stay serialisable
