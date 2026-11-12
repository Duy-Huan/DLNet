"""Tabular plugin (CSV with one target column)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from .. import core, generate, profile, reference, zoo
from . import DomainPlugin

HERE = Path(__file__).parent.parent

TRAIN_PY = '''"""Full training recipe for this baseline. Usage:
    python train.py --data data.csv --target <column> [--epochs 100] [--seed 0] [--device auto] [--amp auto]
Requires: torch, numpy, pandas.
"""
import argparse
import json
from pathlib import Path

import core
from model import build_model, count_params

HERE = Path(__file__).resolve().parent  # runs from any folder
cfg = json.load(open(HERE / "config.yaml"))  # JSON is valid YAML
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--epochs", type=int, default=cfg["recipe"]["epochs"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1 | mps")
ap.add_argument("--amp", default="auto", choices=["auto", "off", "bf16", "fp16"])
a = ap.parse_args()

X, y, cols = core.load_csv(a.data, a.target)
data = core.prepare(X, y, cfg["task"], seed=cfg["split_seed"])
data["preprocess"].update(columns=cols, target=a.target)
model = build_model(cfg["arch"], data["in_dim"], data["out_dim"])
print(f"params={count_params(model):,} epochs={a.epochs}")
tr, best_val = core.train_full(
    model, data, cfg["arch"]["lr"], cfg["arch"]["wd"], cfg["arch"]["batch"], a.epochs, seed=a.seed,
    device=a.device, amp=a.amp, recipe=cfg["recipe"].get("name"),
)
print("val :", tr.evaluate("va"))
print("test:", tr.evaluate("te"))
core.save_artifacts(model, data, HERE)
print(f"saved {HERE / 'model.pt'} and preprocess.json; predict with: python predict.py --data <new file>")
'''

TEST_PY = """import json
from pathlib import Path

import torch

from model import build_model


def test_forward_shape():
    cfg = json.load(open(Path(__file__).parent.parent / "config.yaml"))
    m = build_model(cfg["arch"], cfg["in_dim"], cfg["out_dim"]).eval()
    out = m(torch.randn(4, cfg["in_dim"]))
    assert out.shape == (4, cfg["out_dim"])
"""


class TabularPlugin(DomainPlugin):
    name = "tabular"
    model_file = HERE / "zoo_model.py"
    extra_dirs: ClassVar[dict[str, Path]] = {"zoo": HERE / "zoo"}
    train_template = TRAIN_PY
    test_template = TEST_PY
    metric_keys = ("accuracy", "balanced_accuracy", "rmse", "r2")

    def add_arguments(self, parser):
        parser.add_argument("--target", help="[tabular] target column")
        parser.add_argument(
            "--task", default="auto", choices=["auto", "classification", "regression"]
        )

    def detect(self, path):
        return 0.9 if str(path).lower().endswith((".csv", ".tsv")) else 0.0

    def prepare(self, args):
        if not args.target:
            raise SystemExit(
                "tabular data (CSV) needs --target <column>; for .npz arrays leave --plugin auto"
            )
        warns = []
        X, y, cols = core.load_csv(args.data, args.target, warns)
        task = core.detect_task(y) if args.task == "auto" else args.task
        d = core.prepare(X, y, task, seed=args.seed)
        note = core.task_note(y, task) if args.task == "auto" else None
        d["warnings"] = warns + ([note] if note else [])
        d["preprocess"].update(columns=cols, target=args.target)
        return d

    def profile(self, data):
        return profile.profile(data)

    def generate(self, prof, n_per_family, seed):
        return generate.generate(prof, n_per_family=n_per_family, seed=seed)

    def zoo_input(self, data):
        return "vec", (data["in_dim"],), "cls"

    def build_model(self, cfg, in_dim, out_dim):
        return zoo.build_model(cfg, in_dim, out_dim)

    def reference(self, data):
        return reference.run_reference(data)
