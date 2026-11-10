"""M16: optimisation objectives (closed-form answers)."""

import math

import pytest

from dlnet import objectives as O


def E(val, params, flops=None):
    return ({"params": params, "flops": flops or params * 10, "family": f"m{params}"}, val)


def test_mnasnet_reward_equation_2():
    # ACC * (COST / T)^w, w = -0.07 (soft): half the cost of the target buys 2^0.07 = +5% reward
    assert O.mnas_reward(0.5, 50, 100) == pytest.approx(0.5 * 0.5**-0.07)
    assert O.mnas_reward(0.5, 200, 100) == pytest.approx(0.5 * 2**-0.07)
    assert O.mnas_reward(0.5, 100, 100) == 0.5
    # hard constraint of the paper: alpha = 0, beta = -1
    assert O.mnas_reward(0.8, 50, 100, 0.0, -1.0) == 0.8
    assert O.mnas_reward(0.8, 200, 100, 0.0, -1.0) == pytest.approx(0.4)


def test_params_objective_takes_smallest_within_tolerance():
    e = [E(0.95, 1000), E(0.945, 100), E(0.90, 10)]
    assert O.order(e, "params", tol=0.01) == [1, 0, 2]
    assert O.order(e, "params", tol=0.0) == [0, 1, 2]
    assert O.order(e, "accuracy") == [0, 1, 2]


def test_nondominated_sort_and_crowding():
    pts = [(0.9, 100), (0.8, 10), (0.85, 50), (0.7, 60), (0.95, 1000)]
    fronts = O.nondominated_sort(pts)
    assert sorted(fronts[0]) == [0, 1, 2, 4] and fronts[1] == [3]
    cd = O.crowding(pts, fronts[0])
    assert math.isinf(cd[1]) and math.isinf(cd[4]) and 0 < cd[2] < math.inf


def test_pareto_order_puts_the_front_first():
    e = [E(0.9, 100), E(0.8, 10), E(0.85, 50), E(0.7, 60), E(0.95, 1000)]
    order = O.order(e, "pareto")
    assert set(order[:4]) == {0, 1, 2, 4} and order[4] == 3
    assert sorted(O.pareto_front(e)) == [0, 1, 2, 4]


def test_balanced_prefers_cheaper_when_accuracy_is_close():
    e = [E(0.90, 100_000), E(0.895, 10_000)]
    assert O.order(e, "balanced", target=50_000) == [1, 0]
    e = [E(0.95, 100_000), E(0.70, 10_000)]  # a large accuracy gap is not bought by size
    assert O.order(e, "balanced", target=50_000) == [0, 1]


def test_uncosted_candidates_go_last_and_regression_scores_work():
    e = [({"family": "x"}, 0.99), E(-0.5, 100), E(-0.4, 1000)]
    assert O.order(e, "balanced")[-1] == 0
    assert O.order(e, "pareto")[-1] == 0
    assert O.parse_target("200k") == 200_000 and O.parse_target(None) is None
