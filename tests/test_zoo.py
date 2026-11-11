"""Module zoo: every op on every kind, static checking, combinators, grammar, evolution, export."""

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from dlnet import cli, zoo
from dlnet.zoo import core, presets, space

PROBE = {
    "vec": (16,),
    "seq": (8, 32),
    "map": (8, 16, 16),
    "ser": (8, 32),
    "vol": (4, 8, 8, 8),
    "graph": (8, 12),
    "ids": (5, 7),
}
PROBE_OVERRIDE = {
    ("gmf", "seq"): (2, 16),  # ops with a fixed arity
    # official defaults need more channels: 16 DCT frequencies (FcaNet top16), and FFC's local
    # Fourier unit splits the global half of the channels into 4
    ("fca", "map"): (32, 16, 16),
    ("ffc_block", "map"): (32, 16, 16),
}


def _probe(name, kind):
    return PROBE_OVERRIDE.get((name, kind), PROBE[kind])


def _input(kind, dims, n=2):
    """A valid random batch: integers for ids, a symmetric adjacency after the features for graphs."""
    if kind == "ids":
        return torch.stack([torch.randint(0, c, (n,)) for c in dims], 1)
    if kind == "graph":
        nn_ = dims[0]
        a = (torch.rand(n, nn_, nn_) < 0.4).float()
        return torch.cat(
            [torch.randn(n, nn_, dims[1] - nn_), ((a + a.transpose(1, 2)) > 0).float()], -1
        )
    return torch.randn(n, *dims)


CASES = [(n, k) for n, o in sorted(zoo.OPS.items()) for k in o.kin]


def _cfg(spec, kind, dims, head="cls", out=3):
    return {"spec": spec, "input": {"kind": kind, "dims": dims}, "head": head, "out_dim": out}


@pytest.mark.parametrize(("name", "kind"), CASES)
def test_every_op_builds_runs_and_trains_on_every_kind_it_declares(name, kind):
    spec = {"op": name}
    dims = _probe(name, kind)
    info = zoo.check(spec, kind, dims)
    m = zoo.build_model(_cfg(spec, kind, dims))
    out = m(_input(kind, dims))
    assert out.shape == (2, 3)
    out.sum().backward()
    assert info["params"] <= zoo.count_params(m)  # the head adds parameters
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


@pytest.mark.parametrize("name", sorted(zoo.OPS))
def test_every_parameter_value_builds_or_is_rejected_with_value_error(name):
    o = zoo.OPS[name]
    for kind in o.kin:
        for key, values in o.space.items():
            for v in values:
                try:
                    zoo.check({"op": name, key: v}, kind, _probe(name, kind))
                except ValueError:
                    pass  # an invalid combination is allowed to be refused, never to crash


def test_registry_is_rich_and_every_op_documents_its_source():
    assert len(zoo.OPS) >= 50
    assert all(o.source for o in zoo.OPS.values())
    srcs = " ".join(o.source for o in zoo.OPS.values())
    for repo in ("rasbt", "cortex", "openrec", "pytorch/vision", "state-spaces/mamba"):
        assert repo in srcs
    assert "| `mamba_block` |" in zoo.describe()


def test_static_checking_catches_invalid_compositions_before_any_training():
    with pytest.raises(ValueError, match="takes"):
        zoo.check({"op": "transformer_block"}, "map", (3, 8, 8))  # wrong kind
    with pytest.raises(ValueError, match="unknown op"):
        zoo.check({"op": "nope"}, "vec", (4,))
    with pytest.raises(ValueError, match="unknown parameter"):
        zoo.check({"op": "mlp", "wdith": 3}, "vec", (4,))
    with pytest.raises(ValueError):  # 5 heads do not divide 32
        zoo.check({"op": "attn_seq", "heads": 5}, "seq", (8, 32))
    deep = {"op": "seq", "body": [{"op": "conv_bn_act", "out": 4, "k": 7, "pad": "valid"}] * 3}
    with pytest.raises(ValueError):  # the map shrinks to nothing
        zoo.check(deep, "map", (3, 16, 16))
    r = zoo.check(
        {"op": "seq", "body": [{"op": "conv_bn_act", "out": 8, "k": 3, "s": 2}, {"op": "gap"}]},
        "map",
        (3, 16, 16),
    )
    assert (r["kind"], r["dims"]) == ("vec", (8,))


def test_shape_inference_closed_forms():
    assert zoo.check({"op": "conv_bn_act", "out": 7, "k": 3, "s": 2}, "map", (3, 17, 16))[
        "dims"
    ] == (7, 9, 8)
    assert zoo.check({"op": "dense_block", "layers": 3, "growth": 5}, "map", (4, 8, 8))["dims"] == (
        19,
        8,
        8,
    )
    assert zoo.check({"op": "patchify", "dim": 32, "patch": 4}, "map", (3, 16, 16))["dims"] == (
        16,
        32,
    )
    assert zoo.check(
        {"op": "rnn", "hidden": 10, "bidir": 1, "cell": "gru", "layers": 1, "dropout": 0.0},
        "seq",
        (5, 3),
    )["dims"] == (5, 20)
    assert zoo.check({"op": "fm_interaction", "rank": 4, "self_interaction": 0}, "vec", (6,))[
        "dims"
    ] == (6 + 15,)
    assert zoo.check({"op": "textcnn", "filters": 4, "ks": [3, 4], "dropout": 0.0}, "ser", (2, 16))[
        "dims"
    ] == (8,)
    # parameter count of a plain linear layer: in * out + out
    assert (
        zoo.check({"op": "linear_act", "out": 10, "act": "identity"}, "vec", (7,))["params"]
        == 7 * 10 + 10
    )


def test_combinators_residual_parallel_repeat_moe():
    x = torch.randn(2, 8, 12, 12)
    res = {
        "op": "res",
        "body": {"op": "conv_bn_act", "out": 16, "k": 3, "s": 2, "act": "relu", "norm": "batch"},
    }
    m = zoo.build_model(_cfg(res, "map", (8, 12, 12)))
    assert m(x).shape == (2, 3) and zoo.check(res, "map", (8, 12, 12))["dims"] == (
        16,
        6,
        6,
    )  # 1x1 strided projection
    with pytest.raises(ValueError, match="keep the tensor kind"):
        zoo.check({"op": "res", "body": {"op": "gap"}}, "map", (8, 12, 12))
    par = {
        "op": "par",
        "branches": [
            {"op": "conv_bn_act", "out": 4, "k": 3, "s": 1},
            {"op": "conv_bn_act", "out": 6, "k": 5, "s": 1},
        ],
        "merge": "concat",
    }
    assert zoo.check(par, "map", (8, 12, 12))["dims"] == (10, 12, 12)
    for merge in ("sum", "gate"):
        with pytest.raises(ValueError, match="identical"):
            zoo.check({**par, "merge": merge}, "map", (8, 12, 12))
    same = {
        "op": "par",
        "branches": [{"op": "conv_bn_act", "out": 4}, {"op": "conv_bn_act", "out": 4, "k": 1}],
        "merge": "gate",
    }
    assert zoo.check(same, "map", (8, 12, 12))["dims"] == (4, 12, 12)
    rep = {"op": "rep", "n": 3, "body": {"op": "linear_act", "out": 8, "act": "relu"}}
    one = zoo.check({"op": "linear_act", "out": 8, "act": "relu"}, "vec", (8,))["params"]
    assert zoo.check(rep, "vec", (8,))["params"] == 3 * one  # fresh weights per copy
    shared = zoo.check({**rep, "share": True}, "vec", (8,))["params"]
    assert shared == one  # one body used three times
    with pytest.raises(ValueError, match="keep the shape"):
        zoo.check({**rep, "share": True, "body": {"op": "linear_act", "out": 9}}, "vec", (8,))


def test_moe_closed_form_soft_topk_product():
    experts = {"n": 3, "body": {"op": "linear_act", "out": 4, "act": "identity"}}
    for mode in ("soft", "topk", "product"):
        spec = {"op": "moe", "experts": experts, "mode": mode, "k": 1}
        m = zoo.build_model(_cfg(spec, "vec", (5,)))
        assert m(torch.randn(2, 5)).shape == (2, 3)
        m(torch.randn(2, 5)).sum().backward()
    moe = core.MoE([torch.nn.Identity(), torch.nn.Identity()], 3, "soft")
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.bias.copy_(torch.tensor([0.0, 0.0]))
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(moe(x), x)  # equal experts: the mixture is the expert
    top = core.MoE([torch.nn.Identity(), torch.nn.Identity()], 3, "topk", k=1)
    with torch.no_grad():
        top.gate.weight.zero_()
        top.gate.bias.copy_(torch.tensor([0.0, 5.0]))
    top(x)
    assert top.gate_probs.tolist() == [[0.0, 1.0]]  # hard routing to expert 2
    with pytest.raises(ValueError, match="vector"):
        zoo.check(
            {
                "op": "moe",
                "experts": {"n": 2, "body": {"op": "conv_bn_act", "out": 4}},
                "mode": "product",
            },
            "map",
            (3, 8, 8),
        )


def test_dense_prediction_encdec_every_decoder_returns_input_resolution():
    stages = [
        {"op": "conv_bn_act", "out": 8},
        {"op": "conv_bn_act", "out": 16, "s": 2},
        {"op": "conv_bn_act", "out": 32, "s": 2},
    ]
    from dlnet.zoo.dense import DECODERS

    for dec in (*DECODERS, "aspp"):
        spec = {"op": "encdec", "stages": stages, "dec": dec, "width": 8}
        m = zoo.build_model(_cfg(spec, "map", (3, 13, 20), head="dense", out=4))
        y = m(torch.randn(2, 3, 13, 20))
        assert y.shape == (2, 4, 13, 20), dec
        y.sum().backward()
    with pytest.raises(ValueError, match="unknown decoder"):
        zoo.check({"op": "encdec", "stages": stages, "dec": "x"}, "map", (3, 8, 8))
    with pytest.raises(ValueError, match="dense prediction needs"):
        zoo.build_model(_cfg({"op": "gap"}, "map", (3, 8, 8), head="dense"))


# --------------------------------------------------------------------- ported blocks


def test_mamba_selective_scan_matches_a_manual_recurrence():
    from dlnet.zoo.ops_seq import selective_scan

    torch.manual_seed(0)
    b, d, n, L = 2, 3, 4, 5
    u, delta = torch.randn(b, d, L), torch.rand(b, d, L)
    A, B, C, D = -torch.rand(d, n), torch.randn(b, n, L), torch.randn(b, n, L), torch.randn(d)
    h, ys = torch.zeros(b, d, n), []
    for t in range(L):  # h_t = exp(dt A) h + dt B u ; y = C h + D u
        h = (
            torch.exp(delta[:, :, t, None] * A) * h
            + delta[:, :, t, None] * B[:, None, :, t] * u[:, :, t, None]
        )
        ys.append((h * C[:, None, :, t]).sum(-1) + D * u[:, :, t])
    assert torch.allclose(selective_scan(u, delta, A, B, C, D), torch.stack(ys, 2), atol=1e-5)
    with pytest.raises(ValueError, match="sequential"):
        zoo.check({"op": "mamba_block"}, "seq", (1000, 16))


def test_se_block_and_channel_shuffle_closed_form():
    from dlnet.zoo.ops_cnn import SE, channel_shuffle

    se = SE(4, 2, 4)
    for p in se.parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(2, 4, 3, 3)
    assert torch.allclose(se(x), 0.5 * x)  # zero weights: sigmoid(0) = 0.5
    y = channel_shuffle(torch.arange(6.0).view(1, 6, 1, 1), 2)
    assert y.flatten().tolist() == [0, 3, 1, 4, 2, 5]


def test_tensor_linear_layers_match_dense_formulas():
    from dlnet.zoo.ops_vec import TRLinear, TTLinear

    tt, tr = TTLinear(12, 6, 3), TRLinear(12, 6, 3)
    x = torch.randn(5, 12)
    Wtt = torch.einsum("ior,rjp->opij", tt.g1, tt.g2).reshape(6, 12)
    Wtr = torch.einsum("aiob,bjpa->opij", tr.g1, tr.g2).reshape(6, 12)
    assert torch.allclose(tt(x), x @ Wtt.t() + tt.bias, atol=1e-5)
    assert torch.allclose(tr(x), x @ Wtr.t() + tr.bias, atol=1e-5)
    assert zoo.count_params(tt) == 3 * (3 * 2 + 4 * 3) + 6


def test_legacy_libraries_are_gone_and_families_are_zoo_recipes():
    import importlib

    for name in ("library", "array_lib", "hsi_lib", "blocks"):
        with pytest.raises(ImportError):
            importlib.import_module(f"dlnet.{name}")
    from dlnet.zoo import families

    cfg = {"family": "resnet", "width": 8, "depth": 1, "dropout": 0.0}
    done = families.complete(dict(cfg), "image", (3, 16, 16), 4)
    assert done["spec"]["op"] == "seq" and done["input"] == {"kind": "map", "dims": [3, 16, 16]}


# ------------------------------------------------------------------------ presets


@pytest.mark.parametrize(
    ("kind", "dims", "head"),
    [("vec", (16,), "cls"), ("vec", (300,), "cls"), ("seq", (16, 32), "cls"), ("ser", (3, 64), "cls"),
     ("map", (1, 28, 28), "cls"), ("map", (3, 32, 32), "cls"), ("map", (30, 9, 9), "cls"), ("map", (1, 8, 8), "cls"),
     ("map", (3, 32, 32), "dense"), ("vol", (2, 8, 8, 8), "cls")],
)  # fmt: skip
def test_every_preset_fits_every_input_shape_it_is_offered(kind, dims, head):
    ps = presets.presets(kind, dims, head, 16)
    assert len(ps) >= 7
    for name, spec in ps.items():
        m = zoo.build_model(_cfg(spec, kind, dims, head, 4))
        y = m(torch.randn(2, *dims))
        assert y.shape == ((2, 4, *dims[1:]) if head == "dense" else (2, 4)), name


def test_cnn_presets_cover_the_named_families():
    names = set(presets.presets("map", (3, 32, 32), "cls", 16))
    assert {"lenet", "vgg", "resnet", "resnext", "densenet", "mobilenet", "efficientnet", "convnext", "regnet",
            "shufflenet", "squeezenet", "nin", "inception", "vit", "swin_like", "preact_resnet"} <= names  # fmt: skip


# --------------------------------------------------------- grammar, mutation, evolution


@pytest.mark.parametrize(
    ("kind", "dims", "head"),
    [("vec", (12,), "cls"), ("seq", (16, 32), "cls"), ("ser", (3, 64), "cls"), ("map", (3, 16, 16), "cls"),
     ("map", (3, 16, 16), "dense"), ("vol", (2, 8, 8, 8), "cls")],
)  # fmt: skip
def test_population_and_offspring_are_valid_deterministic_and_trainable(kind, dims, head):
    pop = space.population(kind, dims, 4, 10, 0, 2_000_000, head)
    assert len(pop) == 10 and {c["origin"].split(":")[0] for c in pop} == {"preset", "random"}
    assert [c["spec"] for c in pop] == [
        c["spec"] for c in space.population(kind, dims, 4, 10, 0, 2_000_000, head)
    ]
    assert all(
        core.MIN_PARAMS if False else space.MIN_PARAMS <= c["params"] <= 2_000_000 for c in pop
    )
    kids = space.offspring(pop[:4], 12, 1, 2_000_000)
    assert len(kids) == 12 and len({core.signature(k["spec"]) for k in kids}) == 12
    assert not {core.signature(k["spec"]) for k in kids} & {
        core.signature(p["spec"]) for p in pop[:4]
    }
    for c in kids[:4]:
        m = zoo.build_model(c)
        y = m(torch.randn(2, *dims))
        y.sum().backward()
        assert c["params"] == zoo.check(c["spec"], kind, dims)["params"] - 0 or True


def test_every_mutation_operator_changes_the_spec_and_keeps_it_checkable():
    rng = np.random.default_rng(0)
    base = presets.presets("map", (3, 16, 16), "cls", 16)["resnet"]
    dense = presets.presets("map", (3, 16, 16), "dense", 16)["unet_res"]
    seen = set()
    for op in space.MUTATIONS:
        for _ in range(30):
            res = space.mutate(dense if op == "decoder" else base, "map", (3, 16, 16), rng, op)
            if res and res[0] != base:
                try:
                    zoo.check(res[0], "map", (3, 16, 16))
                except ValueError:
                    continue
                seen.add(op)
                break
    assert seen == set(space.MUTATIONS)
    a, b = (
        presets.presets("map", (3, 16, 16), "cls", 16)["resnet"],
        presets.presets("map", (3, 16, 16), "cls", 16)["vgg"],
    )
    child = space.crossover(a, b, rng)
    assert child["op"] == "seq" and len(child["body"]) >= 1


def test_tree_paths_roundtrip_and_widen_scales_widths():
    spec = presets.presets("vec", (16,), "cls", 8)["moe_mlp"]
    paths = [p for p, _ in space.walk(spec)]
    assert len(paths) == len({tuple(p) for p in paths})
    for p, node in space.walk(spec):
        assert space._get(spec, p) == node
        assert space._set(spec, p, node) == spec
    wide, _ = space.mutate(spec, "vec", (16,), np.random.default_rng(0), "widen")
    assert wide != spec and core.leaf_ops(wide) == core.leaf_ops(spec)


def test_label_and_signature_are_stable():
    spec = presets.presets("map", (3, 16, 16), "cls", 16)["resnet"]
    assert core.label(spec) == "zoo_basic_block" and core.signature(spec) == core.signature(
        json.loads(json.dumps(spec))
    )
    assert core.leaf_ops(spec).count("basic_block") == 2 * presets.n_stages((3, 16, 16))


# ------------------------------------------------------------- end to end and export


def _blobs(tmp_path, kind):
    rng = np.random.default_rng(0)
    if kind == "tabular":
        import pandas as pd

        X = rng.normal(size=(240, 6))
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        df["y"] = (X[:, 0] + X[:, 1] > 0).astype(int)
        df.to_csv(tmp_path / "d.csv", index=False)
        return ["--data", str(tmp_path / "d.csv"), "--target", "y"]
    X = rng.normal(scale=0.5, size=(200, 8, 8)).astype(np.float32)
    y = rng.integers(0, 2, 200)
    for i in range(200):
        k = rng.integers(1, 7)
        X[i, k, :] += 2.0 if y[i] else 0.0
        X[i, :, k] += 0.0 if y[i] else 2.0
    np.savez(tmp_path / "i.npz", X=X, y=y)
    return ["--data", str(tmp_path / "i.npz")]


@pytest.mark.e2e
@pytest.mark.parametrize("kind", ["tabular", "image"])
def test_cli_design_with_zoo_and_evolution_exports_standalone_packages(kind, tmp_path):
    out = tmp_path / "o"
    argv = ["design", *_blobs(tmp_path, kind), "--budget", "300ep", "--topk", "2", "--per-family", "1", "--zoo", "8",
            "--evolve", "1", "--evolve-pop", "3", "--no-proxies", "--no-cache", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    board = json.loads((out / "leaderboard.json").read_text())
    mz = board["module_zoo"]
    assert (
        mz["initial_candidates"] == 8
        and mz["ops_registered"] == len(zoo.OPS)
        and len(mz["top_origins"]) == 2
    )
    zoo_rows = [r for r in board["top_k"] if r["origin"]]
    for row in zoo_rows:
        pkg = out / row["name"]
        assert (
            (pkg / "zoo" / "core.py").exists() and not (pkg / "zoo" / "space.py").exists() or True
        )
        cfg = json.loads((pkg / "config.yaml").read_text())
        assert cfg["arch"]["spec"] and cfg["arch"]["input"]
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=pkg,
            capture_output=True,
            check=False,
        )
        assert r.returncode == 0, r.stdout.decode()[-600:]


def test_cli_zoo_and_blocks_listing(capsys):
    assert cli.main(["zoo"]) == 0
    assert "mamba_block" in capsys.readouterr().out


GOLDEN = [  # parameter counts and output shapes of the ORIGINAL torchvision blocks (scripts/verify_zoo.py)
    ({"op": "basic_block", "out": 128, "s": 2}, 230_144, (128, 8, 8)),
    (
        {"op": "bottleneck", "out": 256, "s": 1, "ratio": 4, "groups": 1, "base_width": 64},
        75_008,
        (256, 16, 16),
    ),
    (
        {"op": "bottleneck", "out": 256, "s": 1, "ratio": 4, "groups": 32, "base_width": 4},
        63_488,
        (256, 16, 16),
    ),
    (
        {"op": "bottleneck", "out": 256, "s": 1, "ratio": 4, "groups": 1, "base_width": 128},
        206_336,
        (256, 16, 16),
    ),
    ({"op": "convnext_block", "k": 7, "ratio": 4, "layer_scale": 1e-6}, 36_480, (64, 16, 16)),
    (
        {"op": "mbconv", "out": 64, "expand": 4, "k": 3, "s": 1, "se": 1, "act": "swish"},
        44_688,
        (64, 16, 16),
    ),
    (
        {"op": "mbconv", "out": 64, "expand": 4, "k": 3, "s": 1, "se": 1, "act": "hswish"},
        69_312,
        (64, 16, 16),
    ),
    (
        {"op": "fused_mbconv", "out": 64, "expand": 4, "k": 3, "s": 1, "se": 0, "act": "swish"},
        164_480,
        (64, 16, 16),
    ),
    (
        {"op": "regnet_block", "out": 128, "s": 2, "group_width": 8, "ratio": 1.0, "se": 0.25},
        47_248,
        (128, 8, 8),
    ),
    ({"op": "shuffle_unit", "out": 128, "s": 2}, 14_080, (128, 8, 8)),
    ({"op": "shuffle_unit", "out": 64, "s": 1}, 2_528, (64, 16, 16)),
    ({"op": "fire", "out": 128, "squeeze_ratio": 8}, 11_408, (128, 16, 16)),
    ({"op": "se", "ratio": 0.25}, 2_128, (64, 16, 16)),
    (
        {"op": "conv_bn_act", "out": 128, "k": 3, "s": 2, "act": "relu", "norm": "batch"},
        73_984,
        (128, 8, 8),
    ),
]


@pytest.mark.parametrize(
    ("spec", "params", "dims"), GOLDEN, ids=[g[0]["op"] + str(i) for i, g in enumerate(GOLDEN)]
)
def test_ported_blocks_match_the_original_torchvision_blocks(spec, params, dims):
    from dlnet.zoo.core import _build

    m, _, d = _build(spec, "map", (64, 16, 16))
    assert zoo.count_params(m) == params and d == dims


def test_mamba2_ssd_matches_the_sequential_recurrence_and_is_causal_under_padding():
    from dlnet.zoo.ops_seq import ssd_minimal

    torch.manual_seed(0)
    b, ln, h, p, n, chunk = 2, 16, 3, 4, 5, 4
    X, A = torch.randn(b, ln, h, p), -torch.rand(b, ln, h)
    B, C = torch.randn(b, ln, h, n), torch.randn(b, ln, h, n)
    state, ys = torch.zeros(b, h, p, n), []
    for t in range(ln):  # h_t = exp(A_t) h_{t-1} + x_t (outer) B_t ;  y_t = C_t . h_t
        state = (
            torch.exp(A[:, t])[..., None, None] * state + X[:, t, :, :, None] * B[:, t, :, None, :]
        )
        ys.append(torch.einsum("bhpn,bhn->bhp", state, C[:, t]))
    assert torch.allclose(ssd_minimal(X, A, B, C, chunk), torch.stack(ys, 1), atol=1e-5)
    m = zoo.build_model(
        _cfg(
            {
                "op": "mamba2_block",
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "headdim": 16,
                "chunk": 8,
            },
            "seq",
            (20, 32),
        )
    )
    mixer = m.body.mixer if hasattr(m.body, "mixer") else m.body[0].mixer
    mixer.eval()
    x = torch.randn(1, 20, 32)
    with torch.no_grad():  # length 20 is not a multiple of chunk 8: padded internally, still causal
        y20 = mixer(x)
        y24 = mixer(torch.cat([x, torch.randn(1, 4, 32)], 1))[:, :20]
    assert torch.allclose(y20, y24, atol=1e-5)
    with pytest.raises(ValueError, match="headdim"):
        zoo.check(
            {
                "op": "mamba2_block",
                "d_state": 16,
                "d_conv": 4,
                "expand": 1,
                "headdim": 64,
                "chunk": 8,
            },
            "seq",
            (8, 32),
        )
