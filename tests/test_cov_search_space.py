"""Search (dlnet/search.py), the mutation space (dlnet/zoo/space.py) and the family recipes
(dlnet/zoo/families.py): the failure and corner paths (v1.9 coverage)."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_audit_fixes import _cands, _data, _linear, _maybe_broken

from dlnet import clock, search
from dlnet.zoo import families, space


def quiet(*a):
    return None


# ----------------------------------------------------------------------------- search


class FakeTrainer:
    def __init__(self, score=float("nan"), fit_raises=False):
        self.score, self.fit_raises, self.epochs_done, self.seconds = score, fit_raises, 2, 1.0

    def fit(self, epochs):
        if self.fit_raises:
            raise RuntimeError("out of memory")
        self.epochs_done += epochs

    def evaluate(self, split):
        return {"score": self.score}


def test_small_search_helpers():
    assert search._open({"dead": True, "cfg": {"family": "x"}}, None, None, [], quiet) is False
    failures = []
    a = {"tr": FakeTrainer(), "cfg": {"family": "x"}}
    assert (
        search._train_step(a, 1, failures, quiet) is False and "non-finite" in failures[0]["error"]
    )
    assert search.sampling_noise(0.5, "r2", 100) == 0.0
    with pytest.raises(ValueError, match="no candidates"):
        search.run_search(_data(), [], math.inf, 1, _linear)
    with pytest.raises(ValueError, match="no candidates"):
        search.run_hyperband(_data(), [], math.inf, 1, _linear)


def test_eval_runs_skips_failed_and_diverged_runs():
    cfg = {**_cands(1)[0], "broken": "raise"}
    vals, *_ = search.eval_runs(cfg, [_data()], _maybe_broken, 1, 2, math.inf)
    assert vals == []
    vals, *_ = search.eval_runs(
        _cands(1)[0], [_data()], _linear, 1, 1, math.inf, first=FakeTrainer()
    )
    assert vals == []


def test_stage2_and_equalize_edges(monkeypatch):
    assert search.equalize([], [_data()], _linear, math.inf) == []
    r = {
        "cfg": _cands(1)[0],
        "epochs": 2,
        "val_mean": 0.5,
        "n_seeds": 1,
        "trainer": FakeTrainer(0.5, True),
    }
    longer = {**r, "epochs": 6, "trainer": None}
    out = search.equalize([r, longer], [_data()], _linear, math.inf)
    assert out[0]["epochs"] == 2  # its training failed: it keeps the shorter result
    monkeypatch.setattr(search, "eval_runs", lambda *a, **k: ([], [], 0.0, 0))
    assert (
        search.stage2(
            [{"cfg": _cands(1)[0], "tr": FakeTrainer(0.5)}], [_data()], _linear, 1, math.inf
        )
        == []
    )


def test_hyperband_when_everything_fails_or_the_budget_is_tight():
    clock.start("time")
    with pytest.raises(SystemExit, match="every candidate failed"):
        search.run_hyperband(_data(), _cands(3, broken="raise"), math.inf, 1, _maybe_broken, R=8, eta=2,
                             unit=4, seeds=1, log=quiet)  # fmt: skip
    clock.start("work")
    res = search.run_hyperband(
        _data(), _cands(6), 5, 1, _linear, R=8, eta=2, unit=4, seeds=1, log=quiet
    )
    assert res and clock.epochs() <= 8


def test_select_topk_fills_and_orders_by_validation():
    rs = [
        {"cfg": {"family": "a", "params": 10}, "val_mean": v, "n_seeds": 1, "epochs": 4}
        for v in (0.5, 0.9)
    ]
    chosen = search.select_topk(rs, 2, keep_order=False)
    assert [r["val_mean"] for r in chosen] == [0.9, 0.5]  # same family: filled by rank


# ----------------------------------------------------------------------------- space


class Rng:
    """Deterministic stand-in for numpy's Generator: always the first choice, random() fixed."""

    def __init__(self, r=0.1):
        self.r = r

    def integers(self, n, *a, **k):
        return 0

    def random(self):
        return self.r

    def choice(self, xs, *a, **k):
        return xs[0]


def test_walk_and_children_of_lists():
    spec = [{"op": "linear", "out": 4}, {"op": "linear", "out": 2}]
    assert [p for p, _ in space.walk(spec)] == [(0,), (1,)]
    assert [k for k, _ in space.children(spec)] == [0, 1]


def test_mutations_that_do_not_apply():
    leaf = {"op": "flatten"}  # an op without searchable parameters
    assert space.mutate([leaf], "map", (4, 8, 8), Rng(), "param") is None
    assert (
        space.mutate(
            {"op": "encdec", "stages": [], "dec": "unet"}, "map", (4, 8, 8), Rng(), "insert"
        )
        is None
    )
    assert space.mutate({"op": "flatten"}, "map", (4, 8, 8), Rng(), "wrap_res") is None
    assert (
        space.mutate(
            [{"op": "res", "body": {"op": "linear", "out": 4}}], "vec", (4,), Rng(), "wrap_res"
        )
        is None
    )
    assert space.mutate([{"op": "linear", "out": 4}], "vec", (4,), Rng(), "nonsense") is None
    assert space.mutate([{"op": "flatten"}], "vec", (4,), Rng(), "swap") is None
    w = space.mutate([{"op": "conv", "out": 8, "k": [3, 3]}], "map", (3, 8, 8), Rng(0.9), "widen")
    assert w[0][0]["k"] == [3, 3]  # a list parameter is copied as it is


def test_decoder_mutation_toggles_attention():
    enc = {"op": "encdec", "stages": [[{"op": "conv", "out": 8}]], "dec": "unet", "width": 8}
    new, _ = space.mutate(enc, "map", (3, 16, 16), Rng(0.1), "decoder")
    assert new["attention"] == "scse"


def test_candidates_need_a_head_they_can_use():
    from dlnet.zoo.core import probe  # noqa: F401  (imports the zoo)

    rng = np.random.default_rng(0)
    assert (
        space._candidate(
            [{"op": "linear", "out": 4}], "vec", (3,), 2, "dense", None, rng, 10**6, "t"
        )
        is None
    )
    assert space._candidate([{"op": "slice", "axis": 1, "start": 0, "stop": 1}, {"op": "as_kind", "to": "ids"}],
                            "vec", (3,), 2, "cls", None, rng, 10**6, "t") is None  # fmt: skip
    assert space.offspring([], 3, 0, 10**6) == []


# ----------------------------------------------------------------------------- families


@pytest.mark.parametrize("domain", ["tabular", "image", "segmentation", "timeseries", "graph", "recsys",
                                    "reconstruction"])  # fmt: skip
def test_unknown_family_names_are_refused(domain):
    _, recipe, *_ = families.DOMAINS[domain]
    with pytest.raises(ValueError, match="unknown"):
        recipe({"family": "nope", "width": 8, "depth": 1, "dropout": 0.0, "latent": 4}, (3, 8, 8))
    with pytest.raises(ValueError, match="unknown"):
        families.complete({"family": "nope"}, domain, (3, 8, 8), 2)


def test_family_building_blocks_refuse_bad_names():
    with pytest.raises(ValueError, match="unknown attention"):
        families._attn("nope", 8)
    assert families._attn("self", 8)[0]["op"] == "self_attn"
    with pytest.raises(ValueError, match="unknown backbone"):
        families.encoder_stages("nope", 8, 1, 32)
    with pytest.raises(ValueError, match="divisible by the 4 heads"):
        families._spectral("transformer", 6, 1, 0.0, 30)
    with pytest.raises(ValueError, match="unknown spectral"):
        families._spectral("nope", 8, 1, 0.0, 30)
    with pytest.raises(ValueError, match="unknown joint"):
        families._joint("nope", 8, 1, 30)
    with pytest.raises(ValueError, match="unknown fusion"):
        families.hsi({"family": "x", "spectral": "mlp", "spatial": "cnn2d", "fusion": "nope", "joint": "none",
                      "width": 8, "depth": 1, "dropout": 0.0}, (30, 9, 9))  # fmt: skip


def test_count_params_falls_back_when_meta_building_fails(monkeypatch):
    from dlnet.zoo import core

    real = core.build_model

    def picky(cfg, **k):
        if torch.get_default_device().type == "meta":
            raise NotImplementedError("no meta kernel")
        return real(cfg, **k)

    monkeypatch.setattr(core, "build_model", picky)
    cfg = families.complete(
        {"family": "mlp", "width": 8, "depth": 1, "dropout": 0.0}, "tabular", (6,), 2
    )
    assert families.count_params(cfg) > 0


def test_rung_one_fallback_skips_a_candidate_that_cannot_be_rebuilt():
    """Nothing fits rung 1: the cheapest candidates are rebuilt and trained until one has a score;
    one that cannot be rebuilt is skipped."""
    built = {}

    def flaky(cfg, i, o):
        built[cfg["family"]] = built.get(cfg["family"], 0) + 1
        if cfg["family"] == "f0" and built["f0"] > 1:
            raise RuntimeError("out of memory on the second build")
        return torch.nn.Linear(i, o)

    clock.start("work")
    res = search.run_search(_data(), _cands(2), 0, 1, flaky, rungs=(2,), seeds=1, log=quiet)
    assert [r["cfg"]["family"] for r in res] == ["f1"]
    clock.start("time")


def test_candidates_with_enough_parameters_still_need_a_usable_head():
    rng = np.random.default_rng(0)
    wide = [{"op": "linear", "out": 64}]
    assert space._candidate(wide, "vec", (32,), 2, "dense", {}, rng, 10**6, "t") is None
    ids = [*wide, {"op": "as_kind", "to": "ids"}]
    assert space._candidate(ids, "vec", (32,), 2, "cls", {}, rng, 10**6, "t") is None
