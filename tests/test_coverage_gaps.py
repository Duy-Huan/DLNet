"""Branches no other test reached (found with `coverage`, see agent.md sec. 9 and doc/README.md).

Mostly input validation (every user-facing error must say what is wrong), exported-package helpers
that only ran inside subprocesses, and the MCP tools that start a design run.
"""

from __future__ import annotations

import io
import json
import sys

import numpy as np
import pytest
import torch

from dlnet import array_data, hsi_data, meta, plugins, seeds, surrogate

pytestmark = pytest.mark.regression

# ------------------------------------------------------------------ array_data validation


def test_array_loader_errors_name_the_problem(tmp_path):
    np.save(tmp_path / "a.npy", np.zeros((3, 4)))
    with pytest.raises(SystemExit, match="need a .npz"):
        array_data.load(tmp_path / "a.npy")
    np.savez(tmp_path / "b.npz", data=np.zeros((4, 3)))
    with pytest.raises(SystemExit, match="not found"):
        array_data.load(tmp_path / "b.npz")
    np.savez(tmp_path / "c.npz", X=np.zeros((4, 3)), y=np.zeros(5))
    with pytest.raises(SystemExit, match="one row per sample"):
        array_data.load(tmp_path / "c.npz")
    with pytest.raises(SystemExit, match="need a .npz"):
        array_data.load_x(tmp_path / "b.npz")
    np.savez(tmp_path / "d.npz", data=np.zeros((4, 3)), lab=np.zeros(4))
    X, _ = array_data.load(tmp_path / "d.npz", x_key="data", y_key="lab")
    assert X.shape == (4, 3)


def test_kind_detection_and_shape_errors():
    assert array_data.kind_of(np.zeros((5, 8, 8))) == "image"
    assert array_data.kind_of(np.zeros((5, 3, 40))) == "timeseries"
    assert array_data.kind_of(np.zeros((5, 8, 8, 3))) == "image"
    assert array_data.kind_of(np.zeros((5, 8, 8)), np.zeros((5, 8, 8))) == "segmentation"
    assert array_data.kind_of(np.zeros((5, 8, 8)), np.zeros((5, 4, 4))) is None
    assert array_data.kind_of(np.zeros(5)) is None
    with pytest.raises(SystemExit, match="image data"):
        array_data.to_channels_first(np.zeros((5, 2, 3, 4, 5)), "image")
    with pytest.raises(SystemExit, match="time series"):
        array_data.to_channels_first(np.zeros((5, 2, 3, 4)), "timeseries")
    assert array_data.to_channels_first(np.zeros((5, 7)), "timeseries").shape == (5, 1, 7)


def test_builders_reject_bad_values_and_tiny_data():
    X = np.zeros((20, 8, 8), "float32")
    X[0, 0, 0] = np.nan
    with pytest.raises(SystemExit, match="NaN"):
        array_data.build(X, np.arange(20) % 2, "image", "classification")
    with pytest.raises(SystemExit, match="NaN"):
        array_data.build_reconstruction(X)
    with pytest.raises(ValueError, match="too few"):
        array_data.build_reconstruction(np.zeros((2, 8, 8), "float32"))
    with pytest.raises(SystemExit, match="masks"):
        array_data.build_segmentation(np.zeros((6, 8, 8)), np.zeros((6, 4, 4), int))
    with pytest.raises(SystemExit, match="packed graphs"):
        array_data.build_graph(np.zeros((6, 4, 3)), np.arange(6) % 2, "classification")
    g = np.zeros((6, 4, 6))
    g[0, 0, 0] = np.inf
    with pytest.raises(SystemExit, match="NaN"):
        array_data.build_graph(g, np.arange(6) % 2, "classification")
    with pytest.raises(SystemExit, match="non-negative integers"):
        array_data.build_ids(-np.ones((10, 2), int), np.arange(10) % 2, "classification")
    with pytest.raises(SystemExit, match="5000"):
        big = np.stack([np.arange(3000), np.arange(3000)], 1)
        array_data.build_ids(big, np.arange(3000) % 2, "classification")


@pytest.mark.parametrize("kind", ["image", "segmentation", "graph", "ids", "reconstruction"])
def test_build_any_dispatches_like_the_exported_train_py(kind):
    rng = np.random.default_rng(0)
    n = 40
    if kind == "segmentation":
        X, y = rng.normal(size=(n, 8, 8)), rng.integers(0, 2, (n, 8, 8))
    elif kind == "graph":
        a = (rng.random((n, 5, 5)) < 0.4).astype(float)
        X, y = (
            np.concatenate([rng.normal(size=(n, 5, 2)), np.maximum(a, a.transpose(0, 2, 1))], -1),
            rng.integers(0, 2, n),
        )
    elif kind == "ids":
        X, y = rng.integers(0, 6, (n, 2)), rng.normal(size=n)
    else:
        X, y = rng.normal(size=(n, 8, 8)), rng.integers(0, 2, n)
    task = "regression" if kind == "ids" else "classification"
    d = array_data.build_any(X, y, kind, task)
    assert len(d["Xtr"]) + len(d["Xva"]) + len(d["Xte"]) == n


# ----------------------------------------------------------------------------- hsi_data


def test_hsi_loader_picks_arrays_or_asks_for_keys(tmp_path, cube_gt):
    cube, gt = cube_gt
    np.savez(tmp_path / "two.npz", a=cube, b=cube, gt=gt)
    with pytest.raises(ValueError, match="--cube-key"):
        hsi_data.load(tmp_path / "two.npz")
    c, g = hsi_data.load(tmp_path / "two.npz", cube_key="a", gt_key="gt")
    assert c.shape == cube.shape and g.dtype == np.int64
    np.savez(tmp_path / "nogt.npz", cube=cube)
    with pytest.raises(ValueError, match="ground truth"):
        hsi_data.load(tmp_path / "nogt.npz")
    np.save(tmp_path / "cube.npy", cube)
    np.save(tmp_path / "gt.npy", gt[:10])
    with pytest.raises(ValueError):
        hsi_data.load(tmp_path / "cube.npy", gt_path=tmp_path / "gt.npy", gt_key="array")
    with pytest.raises(ValueError, match="unsupported"):
        hsi_data.load(tmp_path / "x.txt")
    from scipy.io import savemat

    savemat(tmp_path / "c.mat", {"cube": cube, "gt": gt})
    assert hsi_data.load(tmp_path / "c.mat")[0].shape == cube.shape
    assert len(hsi_data.multilinear_rank(cube, max_dim=50)) == 3


# --------------------------------------------------------------------------- plugins


def test_plugin_registry_router_and_base_class(tmp_path):
    assert plugins.get_plugin("recommendation").name == "recsys"
    assert plugins.get_plugin("autoencoder").name == "image_reconstruction"
    with pytest.raises(SystemExit, match="unknown plugin"):
        plugins.get_plugin("nope")
    (tmp_path / "x.txt").write_text("hello")
    with pytest.raises(SystemExit, match="cannot tell"):
        plugins.route(str(tmp_path / "x.txt"))
    base = plugins.DomainPlugin()
    for fn, a in ((base.prepare, (None,)), (base.profile, ({},)), (base.generate, ({}, 1, 0)),
                  (base.build_model, ({}, 1, 1)), (base.reference, ({},))):  # fmt: skip
        with pytest.raises(NotImplementedError):
            fn(*a)
    assert base.detect("x") == 0.0 and base.zoo_input({}) is None and base.data_config({}) == {}
    assert base.generate_zoo({}, {}, 4, 0) == [] and base.default_cv_splits(None) == 1


def test_array_plugins_refuse_files_that_are_not_theirs(tmp_path):
    reg = plugins._registry()
    (tmp_path / "bad.npz").write_text("not a zip")
    np.savez(tmp_path / "noy.npz", X=np.zeros((4, 8, 8)))
    np.savez(tmp_path / "flat.npz", X=np.zeros((4, 8)), y=np.zeros(4))
    for name in ("image_classification", "timeseries", "graph", "recsys", "image_reconstruction"):
        p = reg[name]
        assert p.detect(str(tmp_path / "data.csv")) == 0.0
        assert p.detect(str(tmp_path / "bad.npz")) == 0.0
    assert reg["image_classification"].detect(str(tmp_path / "noy.npz")) == 0.0
    assert reg["image_classification"].detect(str(tmp_path / "flat.npz")) == 0.0
    assert reg["image_reconstruction"].detect(str(tmp_path / "noy.npz")) > 0.9
    import argparse

    np.savez(tmp_path / "lab.npz", X=np.zeros((10, 8, 8)), y=np.zeros(10))
    with pytest.raises(SystemExit, match="masks"):
        reg["image_segmentation"].prepare(
            argparse.Namespace(data=str(tmp_path / "lab.npz"), seed=0)
        )


def test_recsys_with_three_id_fields_drops_the_two_field_mf_family(tmp_path):
    import argparse

    rng = np.random.default_rng(0)
    p = tmp_path / "r.npz"
    np.savez(p, ids=rng.integers(0, 8, (200, 3)), y=rng.normal(size=200))
    plug = plugins.get_plugin("recsys")
    d = plug.prepare(argparse.Namespace(data=str(p), seed=0, task="auto"))
    cands = plug.generate(plug.profile(d), 1, 0)
    assert cands and "mf" not in {c["family"] for c in cands}


# -------------------------------------------------------------- exported GAN package


def test_exported_gan_helpers_match_the_dlnet_versions(tmp_path):
    """gan_data.py is a copy-free re-implementation inside exported packages: it must agree."""
    from dlnet import gan_data, gan_run

    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(30, 1, 4, 4, generator=g), torch.randn(30, 1, 4, 4, generator=g) + 1
    assert gan_data.mmd2(a, b) == pytest.approx(gan_run.mmd2(a, b), rel=1e-9)
    np.savez(tmp_path / "g.npz", X=np.random.default_rng(0).random((40, 8, 8)))
    X = gan_data.load(tmp_path / "g.npz", None, 0)
    assert X["train"].shape == (24, 1, 8, 8) and float(X["train"].abs().max()) <= 1.0
    cfg = {"width": 8, "zdim": 8, "levels": 1, "gen_c": 16}
    G, D = gan_run.build_pair(cfg, 1, 8)
    res = gan_data.train_gan(G, D, X["train"], X["val"], 8, epochs=1, batch=8)
    assert np.isfinite(res["best_mmd"])


# ------------------------------------------------------------------ meta / surrogate


def test_meta_store_tolerates_damaged_lines_and_portfolio_uses_neighbours(tmp_path):
    p = tmp_path / "s.jsonl"
    recs = [{"plugin": "p", "name": f"d{i}", "features": [float(i)], "family_scores": {"a": i, "b": -i}}
            for i in range(8)]  # fmt: skip
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n{broken\n[1, 2]\n")
    loaded = meta.load(p)
    assert len(loaded) == 8 and meta.load(tmp_path / "missing.jsonl") == []
    assert meta.portfolio(loaded, "p", feats=[7.0], size=1, k=3) == [
        "a"
    ]  # neighbours d5..d7 favour a
    assert meta.portfolio(loaded, "q") == [] and meta.recommend(loaded[:2], "p", [0.0]) is None


def test_surrogate_needs_ordered_pairs_and_enough_datasets():
    with pytest.raises(ValueError, match="ordered pair"):
        surrogate.Surrogate().fit([{"phi": [0.0], "candidates": [{"x": [1.0], "val": 0.5}] * 3}])
    assert surrogate.rank_candidates([], "image_classification", {}, []) is None
    assert surrogate.builtin_records("tabular") == []
    assert len(surrogate.card_vector({"channel_correlation": None})) == len(surrogate.CARD_KEYS)


def test_seed_rules_for_large_structures_colour_and_goals():
    w, _ = seeds.trait_weights({"n_train": 10**5, "samples_per_class_min": 10**4,
                                  "corr_length_rel": 0.4, "channel_correlation": 0.2}, goal="balanced")  # fmt: skip
    assert w["context"] > 0 and w["channel_attention"] > 0 and w["light"] > 0
    assert w["global_attention"] == 0  # enough data: no penalty
    assert seeds.borda([([1, 2], 0.0)]).tolist() == [0.0, 0.0]


# ------------------------------------------------------------------------ MCP tools


def test_mcp_profile_design_and_leaderboard_round_trip(tmp_path, tabular_csv):
    from dlnet import mcp_server

    def call(name, args):
        r = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                               "params": {"name": name, "arguments": args}})  # fmt: skip
        return r["result"]

    prof = json.loads(
        call("dlnet_profile", {"data": str(tabular_csv), "target": "y"})["content"][0]["text"]
    )
    assert prof["plugin"] == "tabular" and prof["profile"]["n_train"] > 0
    spec = {"op": "seq", "body": [{"op": "linear_act", "out": 16}]}
    res = call("dlnet_design", {"data": str(tabular_csv), "target": "y", "out": str(tmp_path / "o"),
                                "budget": "20ep", "topk": 1, "objective": "params", "extra_specs": [spec]})  # fmt: skip
    body = json.loads(res["content"][0]["text"])
    assert body["ok"] and body["top_k"] and "Leaderboard" in body["leaderboard_md"]
    assert (tmp_path / "o.extra_specs.json").exists()
    bad = call("dlnet_leaderboard", {"out": str(tmp_path / "missing")})
    assert bad["isError"]
    failed = json.loads(
        call("dlnet_design", {"data": str(tmp_path / "nope.csv"), "out": str(tmp_path / "p")})[
            "content"
        ][0]["text"]
    )
    assert failed["ok"] is False
    assert "zoo_ops" not in call("dlnet_zoo", {"kind": "map"})["content"][0]["text"]
    assert mcp_server.handle({"jsonrpc": "2.0", "id": 9, "method": "ping"})["result"] == {}


def test_mcp_stdout_carries_only_json_rpc():
    from dlnet import mcp_server

    out = io.StringIO()
    mcp_server.serve(io.StringIO("not json\n\n[1, 2]\n"), out)
    lines = out.getvalue().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["error"]["code"] == -32700


# ------------------------------------------------------------------------- advisor


def test_advisor_commands_and_prompt_placeholders(tmp_path):
    from dlnet import advisor

    assert advisor.command("claude", None).startswith("claude -p")
    assert advisor.command(None, "x {prompt_file}") == "x {prompt_file}"
    assert advisor.command(None, None) is None
    with pytest.raises(SystemExit, match="unknown --advisor"):
        advisor.command("nope", None)
    echo = tmp_path / "echo.py"
    echo.write_text("import sys; print(open(sys.argv[1]).read()[:5] if len(sys.argv) > 1 else '')")
    assert advisor.run(f"{sys.executable} {echo} {{prompt_file}}", "HELLO WORLD").strip() == "HELLO"
    with pytest.raises(ValueError):
        advisor.parse("no json here [1, 2]")


# ------------------------------------------------------------------ zoo invariants


def test_repvgg_fuse_keeps_the_eval_output_closed_form():
    from dlnet.zoo.core import _build

    for s, out in ((1, 8), (2, 16)):
        m = _build({"op": "repvgg_block", "out": out, "s": s}, "map", (8, 6, 6))[0]
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.randn_like(p) * 0.3)
            for name, b in m.named_buffers():
                if "running_var" in name:
                    b.uniform_(0.5, 1.5)
        m.eval()
        x = torch.randn(2, 8, 6, 6)
        y0 = m(x)
        assert torch.allclose(m.fuse()(x), y0, atol=1e-5)


@pytest.mark.parametrize(
    "spec, msg",
    [
        ({"op": "res", "body": {"op": "flatten"}}, "keep the tensor kind"),
        ({"op": "par", "branches": [{"op": "gap"}]}, ">= 2 branches"),
        (
            {"op": "par", "branches": [{"op": "gap"}, {"op": "gap"}], "merge": "max"},
            "unknown merge",
        ),
        (
            {"op": "par", "branches": [{"op": "gap"}, {"op": "flatten"}], "merge": "sum"},
            "identical branch shapes",
        ),
        ({"op": "rep", "n": 0, "body": {"op": "se"}}, "n >= 1"),
        ({"op": "moe", "experts": [{"op": "se"}]}, ">= 2 experts"),
        (
            {"op": "moe", "experts": {"n": 2, "body": {"op": "se"}}, "mode": "vote"},
            "unknown moe mode",
        ),
        (
            {"op": "moe", "experts": {"n": 2, "body": {"op": "se"}}, "mode": "product"},
            "vector outputs",
        ),
        ({"op": "encdec", "stages": [{"op": "gap"}], "dec": "unet"}, "keep the map kind"),
    ],
)
def test_invalid_specs_are_rejected_with_a_reason(spec, msg):
    from dlnet import zoo

    with pytest.raises(ValueError, match=msg):
        zoo.check(spec, "map", (4, 8, 8))


def test_random_grammar_for_graphs_and_ids_builds_and_trains():
    from dlnet import zoo
    from dlnet.zoo import space

    for kind, dims in (("graph", (6, 3 + 6)), ("ids", (5, 7))):
        pop = space.population(kind, dims, 2, 6, 0, 10**6, use_presets=False)
        assert pop, kind
        for c in pop[:3]:
            m = zoo.build_model(c)
            if kind == "ids":
                x = torch.stack([torch.randint(0, n, (4,)) for n in dims], 1)
            else:
                x = torch.cat([torch.randn(4, 6, 3), (torch.rand(4, 6, 6) < 0.3).float()], -1)
            m(x).sum().backward()


def test_train_full_keeps_the_best_validation_checkpoint():
    from dlnet import core
    from dlnet.zoo import families

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5)).astype("float32")
    d = core.prepare(X, (X[:, 0] > 0).astype(int), "classification", seed=0)
    cfg = families.complete(
        {"family": "mlp", "width": 8, "depth": 1, "dropout": 0.0}, "tabular", (5,), 2
    )
    model = families.tabular_model(cfg, 5, 2)
    tr, best = core.train_full(model, d, 3e-3, 1e-4, 32, 5)
    assert tr.evaluate("va")["score"] == pytest.approx(
        best
    )  # the restored weights give the best score
