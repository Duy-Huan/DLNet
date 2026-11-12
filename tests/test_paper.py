"""Paper cards (dlnet/paper.py): a published network written as YAML, rebuilt, checked, verified."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

from dlnet import cli, diagram, paper, standalone, zoo

ROOT = Path(__file__).resolve().parent.parent
CARDS = sorted((ROOT / "papers").glob("*.yaml"))

CARD = """
paper:
  name: Toy Net
  url: https://example.org/toy
  reported:
    - {params: PARAMS}
    - {dataset: toy, metric: accuracy, value: 0.9, protocol: "as in the paper"}
input: {kind: map, dims: [3, 8, 8]}
classes: 4
blocks:
  res8:
    - {id: c1, module: conv_bn_act, args: {out: 8, k: 3}}
    - {id: c2, module: conv_bn_act, args: {out: 8, k: 3, act: identity}}
    - {id: sum, from: [c2, input], module: add}
architecture:
  - {id: stem, module: conv_bn_act, args: {out: 8}}
  - {id: body, module: res8, repeats: 2}
  - {id: side, from: stem, module: conv_bn_act, args: {out: 4, k: 1}}
  - {id: cat, from: [body, side], module: concat}
  - {id: pool, module: gap}
training: {optimizer: adam, lr: 0.002, batch: 16, epochs: 5}
"""


def _card(tmp_path, text=CARD, params=2692, name="toy.yaml"):
    p = tmp_path / name
    p.write_text(text.replace("PARAMS", str(params)))
    return p


@pytest.mark.parametrize("path", CARDS, ids=lambda p: p.stem)
def test_shipped_cards_are_valid_and_match_the_reported_parameters(path):
    card = paper.load(path)
    r = paper.check(card)
    assert r["ok"], r["issues"]
    if "architecture" in card:
        assert standalone.verified_source(paper.to_arch(card))[1] == "identical"
        assert "Input" in diagram.svg(paper.to_arch(card))


def test_shipped_cards_exist():
    assert {p.stem for p in CARDS} >= {"resnet20_cifar", "ssftt"}


def test_card_becomes_a_graph_with_skips_blocks_and_repeats(tmp_path):
    card = paper.load(_card(tmp_path))
    arch = paper.to_arch(card)
    model = zoo.build_model(arch)
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 4)
    r = paper.check(card)
    assert r["ok"] and r["params"] == 2692 and r["level"] == "C"
    assert [n["id"] for n in r["nodes"]] == ["stem", "body", "side", "cat", "pool"]
    assert r["nodes"][3]["shape"] == [12, 8, 8]  # concat of 8 and 4 channels
    assert arch["lr"] == 0.002 and arch["batch"] == 16  # the authors' recipe


def test_parameter_count_far_from_the_paper_is_reported(tmp_path):
    r = paper.check(paper.load(_card(tmp_path, params=5000)))
    assert not r["ok"] and "the paper reports 5,000" in r["issues"][0]
    assert cli.main(["paper", "check", str(_card(tmp_path, params=5000))]) == 1
    assert cli.main(["paper", "check", str(_card(tmp_path))]) == 0


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("from: [c2, input]", "from: [c2, nowhere]", "not a node defined above"),
        ("module: gap}", "module: gap, colour: red}", "unknown key"),
        ("args: {out: 4, k: 1}", "args: {out: 4, kernel: 1}", "unknown parameter"),
        ("module: concat}", "module: add}", "identical shapes"),
        ("classes: 4", "klasses: 4", "unknown top-level"),
        ("  name: Toy Net\n", "", "paper.name"),
    ],
)
def test_mistakes_in_a_card_are_named(tmp_path, old, new, message):
    assert old in CARD
    path = _card(tmp_path, CARD.replace(old, new))
    try:
        r = paper.check(paper.load(path))
        text = " ".join(r["issues"])
        assert not r["ok"]
    except paper.CardError as e:
        text = str(e)
    assert message in text, text


def test_build_writes_a_standalone_package(tmp_path):
    out = tmp_path / "pkgs"
    assert cli.main(["paper", "build", str(_card(tmp_path)), "--out", str(out)]) == 0
    pkg = out / "paper_toy_net"
    assert {"model.py", "architecture.svg", "card.yaml", "config.yaml", "report.md"} <= {
        f.name for f in pkg.iterdir()
    }
    assert "NOT verified" in (pkg / "report.md").read_text()  # level C says so
    code = "import model, torch; print(model.build_model()(torch.zeros(1, 3, 8, 8)).shape)"
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=pkg, capture_output=True, text=True, check=False
    )
    assert "torch.Size([1, 4])" in r.stdout, r.stderr[-400:]


OFFICIAL = """
import torch
from torch import nn
class Net(nn.Module):  # the "authors' code": takes (N, 1, C, H, W)
    def __init__(self, classes=4, wide=8):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, wide, 3, padding=1, bias=False), nn.BatchNorm2d(wide), nn.ReLU())
        self.fc = nn.Linear(wide, classes)
    def forward(self, x):
        return self.fc(self.features(x.squeeze(1)EXTRA).mean((2, 3)))
"""
VERIFIED = """
paper: {name: Tiny Official}
official:
  repo: https://example.org/authors/tiny
  commit: abc1234
  entry: src/net.py:Net
  init_args: {classes: 4, wide: 8}
  input_ops: [[unsqueeze, 1]]
input: {kind: map, dims: [3, 8, 8]}
classes: 4
architecture:
  - {id: conv, module: conv_bn_act, args: {out: 8, k: 3}}
  - {id: pool, module: gap}
"""


def _official(tmp_path, extra=""):
    (tmp_path / "repo" / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "src" / "net.py").write_text(OFFICIAL.replace("EXTRA", extra))
    return tmp_path / "repo"


def test_verify_gives_level_b_only_when_outputs_match_the_authors_model(tmp_path):
    card = paper.load(_card(tmp_path, VERIFIED, name="tiny.yaml"))
    assert paper.level(card) == "C"
    r = paper.verify(card, _official(tmp_path))
    assert r["level"] == "B" and r["params_match"] and r["max_abs_diff"] < 1e-5
    assert (
        paper.level(card) == "B"
        and json.loads((tmp_path / "tiny.fidelity.json").read_text())["level"] == "B"
    )
    # the same card against authors' code that computes something else: not level B
    for mod in [m for m in sys.modules if m.startswith("official_")]:
        del sys.modules[mod]
    r = paper.verify(card, _official(tmp_path, extra=" * 1.5"))
    assert r["level"] == "C" and r["max_abs_diff"] > 1e-3
    # and against a model with other tensors: named, not level B
    wide = paper.load(_card(tmp_path, VERIFIED.replace("wide: 8", "wide: 16"), name="wide.yaml"))
    r = paper.verify(wide, _official(tmp_path))
    assert r["level"] == "C" and not r["params_match"] and r["unmatched_official"]


def test_level_a_package_uses_the_authors_class(tmp_path):
    text = VERIFIED.split("architecture:")[0]
    card = paper.load(_card(tmp_path, text, name="a.yaml"))
    assert paper.level(card) == "A"
    pkg = paper.build(card, tmp_path / "pkgs", log=lambda *_: None)
    code = ("import model, torch; m = model.build_model(); "
            "print(m(torch.zeros(2, 3, 8, 8)).shape, model.count_params(m))")  # fmt: skip
    env = {"DLNET_OFFICIAL_ROOT": str(_official(tmp_path)), "PATH": "/usr/bin:/bin"}
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=pkg, capture_output=True, text=True, env=env, check=False
    )
    assert "torch.Size([2, 4]) 268" in r.stdout, r.stderr[-400:]
    assert "identical by construction" in (pkg / "report.md").read_text()
