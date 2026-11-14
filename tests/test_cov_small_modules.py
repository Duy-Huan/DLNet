"""Refusals, fallbacks and size limits of the smaller modules: the array and HSI plugins, export,
threads, data card, profile, objectives, generation, meta-learning, cost, the MCP protocol and a
dozen op libraries (v1.9 coverage)."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dlnet import (
    clock,
    datacard,
    diagram,
    export,
    generate,
    inference,
    latex,
    mcp_server,
    meta,
    objectives,
    profile,
    tensor_lora,
    threads,
    unfold_run,
)
from dlnet.plugins import arrays, hsi, tabular
from dlnet.zoo import cost, families, ops_img, ops_lib, presets
from dlnet.zoo.core import _build

# ----------------------------------------------------------------------------- op refusals


@pytest.mark.parametrize(("spec", "kind", "dims", "match"), [
    ({"op": "gat", "out": 6, "heads": 4}, "graph", (5, 12), "heads must divide"),
    ({"op": "gat", "out": 8, "heads": 1}, "graph", (5, 5), "F >= 1"),
    ({"op": "grid_spectral_conv", "k": 10, "filters": 2}, "map", (2, 40, 40), "at most 1024"),
    ({"op": "pair_dots"}, "seq", (1, 8), "at least two fields"),
    ({"op": "finalmlp", "hidden": 6, "depth": 1, "heads": 4}, "seq", (3, 8), "not divisible"),
    ({"op": "dcnv3"}, "seq", (3, 7), "must be even"),
    ({"op": "ffc_block", "ratio": 0.25, "lfu": 0}, "map", (4, 8, 8), "no local or global part"),
    ({"op": "fca", "reduction": 16}, "map", (8, 7, 7), "reduction"),
    ({"op": "timemixer", "down_layers": 3}, "ser", (2, 12), "too short"),
    ({"op": "timexer", "patch_len": 4, "d_model": 6, "heads": 4}, "ser", (2, 16), "not divisible"),
    ({"op": "fm_interaction"}, "vec", (200,), "quadratic"),
])  # fmt: skip
def test_op_refusals(spec, kind, dims, match):
    with pytest.raises(ValueError, match=match):
        _build(spec, kind, dims)


def test_op_helpers():
    from dlnet.zoo.ops_freq import MultiSpectralAttention
    from dlnet.zoo.ops_graph2 import mlp_block

    with pytest.raises(ValueError, match="not divisible by 16 frequencies"):
        MultiSpectralAttention(24)
    assert isinstance(mlp_block(4, (8,), bn=True)[1], torch.nn.BatchNorm1d)
    assert ops_img.make_divisible(10, 8) == 16
    with pytest.raises(ValueError, match="splits the input"):
        ops_img.SelectiveKernel(5, 8)
    with pytest.raises(ValueError, match="unknown kind"):
        presets.presets("nope", (3,))
    # a window as wide as the map: no shift along that axis
    x = torch.randn(1, 4, 4, 8)
    qkv, proj, bias = torch.randn(24, 8), torch.randn(8, 8), torch.zeros(1, 2, 8, 8)
    out = ops_lib.shifted_window_attention(x, qkv, proj, bias, [2, 4], 2, [1, 1])
    assert out.shape == x.shape


def test_zoo_model_file_imports_the_zoo(monkeypatch):
    import importlib.util

    import dlnet.zoo

    monkeypatch.setitem(sys.modules, "zoo", dlnet.zoo)
    p = Path(arrays.__file__).parent.parent / "zoo_model.py"
    spec = importlib.util.spec_from_file_location("zoo_model_copy", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build_model is dlnet.zoo.build_model


# ----------------------------------------------------------------------------- plugins


def test_array_plugin_warns_about_an_ambiguous_target_and_reduces_wide_inputs(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 1, 24, 24)).astype("float32")  # 576 features > 512: PCA references
    y = (np.arange(120) % 30).astype("float64")  # 30 whole numbers, read as a regression
    np.savez(tmp_path / "d.npz", X=X, y=y)
    p = arrays.ImageClassificationPlugin()
    args = types.SimpleNamespace(
        data=str(tmp_path / "d.npz"), x_key=None, y_key=None, task="auto", seed=0
    )
    d = p.prepare(args)
    assert any("distinct whole numbers" in w for w in d["warnings"])
    assert p.reference(d)


def test_hsi_and_tabular_plugins_refuse_bad_options(tmp_path):
    (tmp_path / "bad.npz").write_bytes(b"not a zip")
    assert hsi.HSIPlugin().detect(tmp_path / "bad.npz") == 0.0
    with pytest.raises(SystemExit, match="must be odd"):
        hsi.HSIPlugin().prepare(types.SimpleNamespace(patch=4))
    with pytest.raises(SystemExit, match="needs --target"):
        tabular.TabularPlugin().prepare(types.SimpleNamespace(target=None))


def test_unfolding_needs_enough_samples(tmp_path):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 4))
    X = rng.normal(size=(4, 4))
    np.savez(tmp_path / "u.npz", A=A, Y=X @ A.T, X=X)
    args = types.SimpleNamespace(out=str(tmp_path / "o"), data=str(tmp_path / "u.npz"), seed=0)
    with pytest.raises(SystemExit, match="too few samples"):
        unfold_run.run(args)


# ----------------------------------------------------------------------------- export, inference


def test_mean_reference_keeps_the_code():
    fold = [
        {
            "name": "ridge",
            "seconds": 1.0,
            "code": "x = 1",
            "va": {"score": 0.5},
            "te": {"score": 0.4},
        }
    ]
    m = export.mean_reference([fold, fold])
    assert m[0]["code"] == "x = 1" and m[0]["n_splits"] == 2


def test_onnx_export_paths(tmp_path, monkeypatch):
    m = torch.nn.Linear(3, 2)

    def fake(model, args, path, dynamo):
        Path(path).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake)
    assert inference.export(m, torch.randn(1, 3), tmp_path / "m.onnx").read_bytes() == b"onnx"

    def missing(*a, **k):
        raise ImportError("no onnxscript")

    monkeypatch.setattr(torch.onnx, "export", missing)
    with pytest.raises(SystemExit, match="onnxscript"):
        inference.export(m, torch.randn(1, 3), tmp_path / "n.onnx")


# ----------------------------------------------------------------------------- environment


def test_cpu_counts_without_affinity_or_proc(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert threads.available_cpus() >= 1

    def no_proc(*a, **k):
        raise OSError("no /proc")

    monkeypatch.setattr("builtins.open", no_proc)
    assert threads.busy_cpus() == 0


# ----------------------------------------------------------------------------- data analysis


def test_data_card_on_constant_and_large_inputs():
    X = np.zeros((6, 1, 8, 8), dtype="float32")
    assert datacard.correlation_length(X) == 0.0
    assert datacard.position_dependence(X, np.arange(6) % 2) == 0.0
    rng = np.random.default_rng(0)
    data = {"Xtr": torch.from_numpy(rng.normal(size=(40, 1, 8, 8)).astype("float32")),
            "y_raw": {"tr": np.arange(40) % 2}}  # fmt: skip
    card = datacard.image_card(data, max_images=20)
    assert card


def test_intrinsic_dimension_subsamples_large_inputs():
    X = np.random.default_rng(0).normal(size=(60, 3))
    assert profile.twonn_intrinsic_dim(X, max_points=30) > 0
    assert profile.mle_intrinsic_dim(X, max_points=30) > 0
    big = np.random.default_rng(0).normal(size=(2100, 3))
    assert profile.intrinsic_dim_report(big, n_boot=2)


# ----------------------------------------------------------------------------- choosing models


def test_objectives_edges():
    assert objectives.mnas_reward(0.9, 0.0, 10) == 0.9
    entries = [({"family": "a"}, 0.5), ({"family": "b"}, 0.7)]  # no costs known
    assert objectives.order(entries, "params") == [1, 0]
    costed = [({"params": 10}, 0.5), ({"params": 20}, 0.7)]
    with pytest.raises(ValueError, match="unknown objective"):
        objectives.order(costed, "nope")


def test_generate_skips_wide_quadratic_families_and_configurations_that_do_not_fit(monkeypatch):
    prof = {"n_features": 200, "n_classes": 2, "max_params": 10**7}
    cands = generate.generate(prof, n_per_family=1)
    assert all(c["family"] not in ("ft_transformer", "fm_mlp") for c in cands)

    def no_fit(*a, **k):
        raise ValueError("does not fit")

    monkeypatch.setattr(families, "complete", no_fit)
    assert generate.generate({**prof, "n_features": 4}, n_per_family=1) == []


def test_meta_with_too_few_records():
    assert meta.normalized_ranks({"mlp": 0.5}) == {"mlp": 0.0}
    recs = [{"plugin": "tabular", "features": [float(i)], "family_scores": {"mlp": 0.5, "resnet": 0.4}}
            for i in range(3)]  # fmt: skip
    assert meta.leave_one_out(recs, "tabular")["datasets"] == 0


def test_cost_falls_back_to_a_real_device():
    assert cost._first_tensor("text") is None

    class NoMeta(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 2)

        def forward(self, x):
            if x.device.type == "meta":
                raise NotImplementedError("needs real data")
            return self.lin(x)

    c = cost.estimate(lambda cfg, i, o: NoMeta(), {"batch": 2}, 4, 2, (4,))
    assert c["params"] == 10


def test_tensor_lora_refusals():
    with pytest.raises(ValueError, match="does not match factors"):
        tensor_lora.tt_svd(torch.randn(4, 6), (2, 3), (4,), 2)
    with pytest.raises(ValueError, match="unknown format"):
        tensor_lora.TensorLoRALinear(torch.nn.Linear(4, 4), fmt="nope")
    with pytest.raises(ValueError, match="do not match"):
        tensor_lora.TensorLoRALinear(torch.nn.Linear(4, 4), fmt="tt", in_factors=(3, 2))


# ----------------------------------------------------------------------------- small helpers


def test_small_helpers():
    with pytest.raises(ValueError, match="unknown clock mode"):
        clock.start("nope")
    clock.start("time")
    assert diagram.stack([]).w == 0
    assert latex._pm(None, None) == "--"


class FakeService:
    def __init__(self, cancel_then_fail):
        self.cancel_then_fail = cancel_then_fail
        self.proto = None

    def call(self, name, args, progress, cancelled):
        progress(1, total=4, message="step")
        if self.cancel_then_fail:
            self.proto.cancel(None, 9)
            raise RuntimeError("stopped")
        return {"ok": True}


def test_mcp_progress_with_a_total_and_a_cancelled_failure():
    svc = FakeService(False)
    proto = mcp_server.Protocol(svc)
    svc.proto = proto
    seen = []
    msg = {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
           "params": {"name": "dlnet_info", "arguments": {}, "_meta": {"progressToken": "p"}}}  # fmt: skip
    proto.handle(msg, seen.append)
    assert seen[0]["params"]["total"] == 4
    svc.cancel_then_fail = True
    assert proto.handle(msg, seen.append) is None  # cancelled: no response, even on an error


def test_export_bundles_the_zoo_when_the_network_cannot_be_written_standalone(
    tmp_path, monkeypatch
):
    from test_regression import _export_minimal

    from dlnet import standalone

    def refuse(*a, **k):
        raise standalone.Unsupported("an op without a standalone form")

    from dlnet import search

    real = search.select_topk

    def with_spec(*a, **k):  # the searched network as a zoo spec
        out = real(*a, **k)
        for r in out:
            r["cfg"] = families.complete({**r["cfg"]}, "tabular", (4,), 2)
        return out

    monkeypatch.setattr(search, "select_topk", with_spec)
    monkeypatch.setattr(standalone, "write_package", refuse)
    _export_minimal(tmp_path / "o")
    assert list((tmp_path / "o").glob("top1_*/zoo/__init__.py"))
