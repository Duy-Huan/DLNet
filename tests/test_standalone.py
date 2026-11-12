"""Exported model.py is the generated network as plain PyTorch source (dlnet/standalone.py)."""

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from test_zoo import CASES, _cfg, _input, _probe

from dlnet import cli, standalone, zoo
from dlnet.zoo import families

DIMS = {"tabular": (16,), "image": (3, 32, 32), "segmentation": (3, 32, 32), "timeseries": (3, 64),
        "hsi": (32, 9, 9), "graph": (8, 12), "recsys": (10, 7), "reconstruction": (3, 32, 32)}  # fmt: skip
FAMILIES = [(d, f) for d, (fams, *_) in families.DOMAINS.items() for f in fams if f != "composed"]


def _code(text):  # the file without its docstring (which names DLNet as the generator)
    return text.split('"""', 2)[2]


@pytest.mark.parametrize(("name", "kind"), CASES)
def test_every_op_is_written_as_source_identical_to_the_zoo_model(name, kind):
    text, how = standalone.verified_source(_cfg({"op": name}, kind, _probe(name, kind)))
    assert how == "identical"  # same seed -> the same initial weights, the same outputs
    assert "dlnet" not in _code(text) and "import zoo" not in text and "from ." not in _code(text)


@pytest.mark.parametrize(("domain", "family"), FAMILIES)
def test_every_family_is_written_as_source_identical_to_the_zoo_model(domain, family):
    cfg = {
        "family": family,
        "width": 32,
        "depth": 2,
        "dropout": 0.1,
        "patch": 4,
        "rank_ratio": 0.5,
        "latent": 8,
    }
    c = families.complete(cfg, domain, DIMS[domain], 3)
    assert standalone.verified_source(c, 3)[1] == "identical"


def test_combinators_and_shared_weights_are_reproduced():
    spec = [{"op": "linear_act", "out": 16, "act": "relu"},
            {"op": "rep", "n": 2, "share": True, "body": {"op": "res", "body": [{"op": "linear_act", "out": 16, "act": "relu"}]}},
            {"op": "par", "merge": "concat", "branches": [[{"op": "linear_act", "out": 8, "act": "relu"}], [{"op": "linear_act", "out": 4, "act": "relu"}]]}]  # fmt: skip
    text, how = standalone.verified_source(_cfg(spec, "vec", (10,)))
    assert how == "identical" and "Shared(" in text and "Par(" in text


def test_standalone_file_runs_alone_and_loads_the_weights_of_the_zoo_model(tmp_path):
    cfg = _cfg(
        [{"op": "conv_bn_act", "out": 8}, {"op": "gfnet_block"}, {"op": "gap"}], "map", (4, 8, 8)
    )
    torch.manual_seed(1)
    ref = zoo.build_model(cfg).eval()
    (tmp_path / "model.py").write_text(standalone.source(cfg))
    torch.save(ref.state_dict(), tmp_path / "w.pt")
    x = _input("map", (4, 8, 8))
    torch.save(x, tmp_path / "x.pt")
    code = ("import torch, model, sys; m = model.build_model().eval(); "
            "m.load_state_dict(torch.load('w.pt')); torch.save(m(torch.load('x.pt')), 'y.pt'); "
            "assert not any(k == 'dlnet' or k.startswith('dlnet.') or k == 'zoo' for k in sys.modules)")  # fmt: skip
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr[-800:]
    torch.testing.assert_close(torch.load(tmp_path / "y.pt"), ref(x), atol=1e-6, rtol=1e-5)


def test_a_network_that_differs_is_rejected_not_exported():
    cfg = _cfg({"op": "linear_act", "out": 8, "act": "relu"}, "vec", (6,))
    text = standalone.source(cfg)
    with pytest.raises(standalone.Unsupported):
        standalone.verify(text.replace("nn.Linear(6, 8", "nn.Linear(6, 8, bias=False"), cfg)
    wrong = text.replace("return self.f(x)", "return self.f(x) * 2")
    assert wrong != text
    with pytest.raises(standalone.Unsupported):
        standalone.verify(wrong, cfg)


def test_recorder_restores_every_constructor():
    before = (torch.nn.Linear.__init__, zoo.core.Residual.__init__)
    with standalone.recording():
        assert torch.nn.Linear.__init__ is not before[0]
    assert (torch.nn.Linear.__init__, zoo.core.Residual.__init__) == before


def test_convert_rewrites_an_old_package_and_keeps_its_weights(tmp_path):
    """Packages exported before v1.0 hold `from zoo import build_model` + a zoo/ copy."""
    import shutil
    from pathlib import Path

    import dlnet

    cfg = _cfg([{"op": "linear_act", "out": 8, "act": "relu"}], "vec", (6,))
    pkg = tmp_path / "top1_x"
    pkg.mkdir()
    (pkg / "config.yaml").write_text(json.dumps({"arch": cfg, "in_dim": 6, "out_dim": 3}))
    root = Path(dlnet.__file__).parent
    shutil.copy(root / "zoo_model.py", pkg / "model.py")
    shutil.copytree(root / "zoo", pkg / "zoo")
    ref = zoo.build_model(cfg).eval()
    torch.save(ref.state_dict(), pkg / "model.pt")
    assert cli.main(["standalone", str(pkg)]) == 0
    assert not (pkg / "zoo").exists()
    m = (
        standalone.load((pkg / "model.py").read_text()).build_model(cfg, 6, 3).eval()
    )  # old call signature
    m.load_state_dict(torch.load(pkg / "model.pt"))
    x = torch.randn(4, 6)
    torch.testing.assert_close(m(x), ref(x))
