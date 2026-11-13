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


def test_stubbed_imports_do_not_break_code_that_walks_sys_modules(tmp_path):
    """CI run 1 of verify-papers: a stub answered `__file__` with a function, and every later
    card failed with "'function' object has no attribute 'endswith'"."""
    import inspect

    repo = _official(tmp_path)
    src = repo / "src" / "net.py"
    src.write_text("import some_missing_tool\n" + src.read_text())
    text = VERIFIED.replace("  input_ops", "  stub_modules: [some_missing_tool]\n  input_ops")
    card = paper.load(_card(tmp_path, text, name="stub.yaml"))
    assert paper.verify(card, repo)["level"] == "B"
    stub = sys.modules.pop("some_missing_tool")
    assert stub.anything() is None and not hasattr(stub, "__file__")
    assert inspect.getmodule(paper) is paper  # walks sys.modules


AUTHORS_RESNET = """
import torch
from torch import nn
import torch.nn.functional as F
class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.drop = nn.Dropout(0.0)
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return torch.relu(x + self.drop(self.conv2(y)))
class Net(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU6())
        self.blocks = nn.Sequential(Block(8), Block(8))
        self.side = nn.Conv2d(8, 4, 1)
        self.pool = nn.MaxPool2d(2)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(12, num_classes)
    def forward(self, x):
        s = self.stem(x)
        y = torch.cat([self.blocks(s), self.side(s) * 2.0 if False else self.side(s)], 1)
        return self.fc(torch.flatten(self.avg(self.pool(y)), 1))
"""


def test_import_drafts_a_card_from_the_authors_code_and_verify_proves_it(tmp_path):
    repo = tmp_path / "authors"
    repo.mkdir()
    (repo / "net.py").write_text(AUTHORS_RESNET)
    out = tmp_path / "drafted.yaml"
    argv = ["paper", "import", "--official-root", str(repo), "--repo", "https://example.org/a/net", "--commit", "abc",
            "--entry", "net.py:Net", "--init-args", '{"num_classes": 5}', "--dims", "3,16,16", "--classes", "5",
            "--name", "Authors Net", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    card = paper.load(out)
    mods = [n["module"] for n in card["architecture"]]
    assert mods.count("add") == 2 and "concat" in mods and "maxpool" in mods and mods[-1] == "gap"
    assert "linear" not in mods  # the classifier became the card's head
    r = paper.verify(card, repo)
    assert r["level"] == "B" and r["params_match"] and r["max_abs_diff"] < 1e-5


def test_import_refuses_what_it_cannot_write(tmp_path):
    repo = tmp_path / "authors"
    repo.mkdir()
    (repo / "net.py").write_text(
        AUTHORS_RESNET.replace("return torch.relu(x +", "return torch.sin(x +")
    )
    argv = ["paper", "import", "--official-root", str(repo), "--repo", "u", "--commit", "c", "--entry", "net.py:Net",
            "--dims", "3,16,16", "--classes", "5", "--name", "X", "--out", str(tmp_path / "x.yaml")]  # fmt: skip
    assert cli.main(argv) == 1 and not (tmp_path / "x.yaml").exists()


def test_import_reports_a_model_torch_fx_cannot_trace(tmp_path):
    """Data-dependent control flow (or einops on a proxy) is an 'import not possible', not a crash."""
    repo = tmp_path / "authors"
    repo.mkdir()
    (repo / "net.py").write_text(
        AUTHORS_RESNET.replace(
            "return torch.relu(x +", "return (x if x.sum() > 0 else -x) + torch.relu(x +"
        )
    )
    argv = ["paper", "import", "--official-root", str(repo), "--repo", "u", "--commit", "c", "--entry", "net.py:Net",
            "--dims", "3,16,16", "--classes", "5", "--name", "X", "--out", str(tmp_path / "x.yaml")]  # fmt: skip
    assert cli.main(argv) == 1 and not (tmp_path / "x.yaml").exists()


SMALL = """
paper:
  name: Small Paper Net
  reported: [{dataset: somewhere else, metric: accuracy, value: 0.99, protocol: "their split"}]
input: {kind: map, dims: [1, 12, 12]}
classes: 3
architecture:
  - {id: c1, module: conv_bn_act, args: {out: 8, k: 3}}
  - {id: c2, module: conv_bn_act, args: {out: 8, k: 3, act: identity}}
  - {id: sum, from: [c2, c1], module: add}
  - {id: pool, module: gap}
training: {optimizer: sgd, lr: 0.05, weight_decay: 0.0005, batch: 32, epochs: 300}
"""


@pytest.mark.e2e
def test_design_with_papers_trains_them_under_the_same_protocol(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 150)
    X = rng.normal(size=(150, 12, 12)).astype(np.float32)
    X[np.arange(150), y * 4, y * 4] += 4
    np.savez(tmp_path / "img.npz", X=X, y=y)
    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "small.yaml").write_text(SMALL)
    (cards / "other_input.yaml").write_text(
        SMALL.replace("[1, 12, 12]", "[3, 32, 32]").replace("Small", "Big")
    )
    (cards / "zz_broken.yaml").write_text("paper: {name: Broken}\n")  # recorded, the others go on
    out = tmp_path / "o"
    argv = ["design", "--data", str(tmp_path / "img.npz"), "--budget", "60ep", "--topk", "2", "--per-family", "1",
            "--zoo", "2", "--evolve", "0", "--no-proxies", "--out", str(out), "--papers", str(cards),
            "--paper-recipe", "--paper-epochs", "3"]  # fmt: skip
    assert cli.main(argv) == 0
    board = json.loads((out / "leaderboard.json").read_text())
    (row,) = board["papers"]  # the card with another input size was skipped
    top = board["top_k"][0]
    assert row["paper"] == "Small Paper Net" and row["level"] == "C"
    assert row["common"]["epochs"] == top["epochs"] and row["common"]["n_seeds"] == top["n_runs"]
    assert row["authors"]["epochs"] == 3 and row["authors"]["recipe"]["optimizer"] == "sgd"
    md = (out / "leaderboard.md").read_text()
    assert "## Paper baselines" in md and "somewhere else: accuracy 0.99" in md
    # v1.7: failures recorded, per-run scores, paired comparison with the top-1, LaTeX table
    assert [f["card"].endswith("zz_broken.yaml") for f in board["papers_failed"]] == [True]
    assert "Not trained:" in md and "zz_broken.yaml" in md and "p (Holm)" in md
    assert len(top["runs"]["test"]) == top["n_runs"] == len(row["common"]["runs"]["test"])
    vs = row["vs_top1"]
    assert vs["n"] == top["n_runs"] and vs["verdict"] in (
        "better",
        "worse",
        "no significant difference",
        "not tested",
    )
    assert abs(vs["diff"] - (row["common"]["test"]["score"] - top["test"]["score"])) < 2e-4
    tex = (out / "results.tex").read_text()
    assert r"\begin{tabular}" in tex and "Small Paper Net" in tex
    pkg = out / "paper_small_paper_net"
    assert (pkg / "train.py").exists() and (pkg / "architecture.svg").exists()
    assert "paper_small_paper_net" in (out / "baselines.ipynb").read_text()
    r = subprocess.run([sys.executable, "train.py", "--data", str(tmp_path / "img.npz"), "--epochs", "1"],
                       cwd=pkg, capture_output=True, text=True, check=False)  # fmt: skip
    assert r.returncode == 0 and (pkg / "model.pt").exists(), r.stderr[-600:]


def test_trainer_optimizers_of_paper_recipes():
    from dlnet import core

    X = torch.randn(40, 4)
    y = (X[:, 0] > 0).long()
    d = {"task": "classification", "primary": "accuracy", "in_dim": 4, "out_dim": 2, "Xtr": X, "ytr": y,
         "Xva": X, "yva": y, "Xte": X, "yte": y}  # fmt: skip
    for name, cls in (
        ("sgd", torch.optim.SGD),
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
    ):
        tr = core.Trainer(torch.nn.Linear(4, 2), d, batch=8, optimizer=name)
        assert type(tr.opt) is cls
        tr.fit(1)
    with pytest.raises(ValueError, match="unknown optimizer"):
        core.Trainer(torch.nn.Linear(4, 2), d, optimizer="lion")
