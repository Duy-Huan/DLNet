"""Image and time-series plugins: .npz with X and y (see dlnet/array_data.py for the shapes)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import numpy as np

from .. import array_data, array_lib, core, profile, reference
from . import DomainPlugin

HERE = Path(__file__).parent.parent

TRAIN_PY = '''"""Full training recipe for this baseline. Usage:
    python train.py --data data.npz [--x-key X] [--y-key y] [--epochs 100] [--seed 0]
Requires: torch, numpy.
"""
import argparse
import json

import array_data
import core
from model import build_model, count_params

cfg = json.load(open("config.yaml"))  # JSON is valid YAML
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--x-key", default=cfg["data"]["x_key"])
ap.add_argument("--y-key", default=cfg["data"]["y_key"])
ap.add_argument("--epochs", type=int, default=cfg["recipe"]["epochs"])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

X, y = array_data.load(a.data, a.x_key, a.y_key)
data = array_data.build(X, y, cfg["data"]["kind"], cfg["task"], seed=cfg["split_seed"])
model = build_model(cfg["arch"], data["in_dim"], data["out_dim"])
print(f"params={count_params(model):,} epochs={a.epochs}")
tr, best_val = core.train_full(
    model, data, cfg["arch"]["lr"], cfg["arch"]["wd"], cfg["arch"]["batch"], a.epochs, seed=a.seed
)
print("val :", tr.evaluate("va"))
print("test:", tr.evaluate("te"))
'''

TEST_PY = """import json
from pathlib import Path

import torch

from model import build_model


def test_forward_shape():
    cfg = json.load(open(Path(__file__).parent.parent / "config.yaml"))
    m = build_model(cfg["arch"], cfg["in_dim"], cfg["out_dim"]).eval()
    out = m(torch.randn(4, *cfg["data"]["shape"]))
    assert out.shape == (4, cfg["out_dim"])
"""

TRAIN = {"lr": [1e-3, 3e-3], "wd": [1e-4, 1e-2], "batch": [32, 64]}
SPACE = {
    "cnn": {"width": [16, 32], "depth": [2, 3], "dropout": [0.0, 0.2]},
    "resnet": {"width": [8, 16, 32], "depth": [1, 2], "dropout": [0.0, 0.2]},
    "vit": {"width": [32, 64], "depth": [2, 4], "dropout": [0.0, 0.1], "patch": [2, 4, 7, 8]},
    "tcn": {"width": [32, 64], "depth": [2, 3, 4], "dropout": [0.0, 0.1], "kernel": [3, 5, 7]},
    "gru": {"width": [32, 64], "depth": [1, 2], "dropout": [0.0, 0.1]},
    "ts_transformer": {"width": [32, 64], "depth": [1, 2], "dropout": [0.0, 0.1]},
}
REF_MAX_FEATURES = 512  # flattened inputs larger than this are PCA-reduced for the references
REF_PCA = 64


class ArrayPlugin(DomainPlugin):
    """Shared code of the image and time-series plugins."""

    kind = ""
    families: tuple[str, ...] = ()
    model_file = HERE / "array_lib.py"
    extra_files: ClassVar[dict[str, Path]] = {"array_data.py": HERE / "array_data.py"}
    train_template = TRAIN_PY
    test_template = TEST_PY
    metric_keys = ("accuracy", "balanced_accuracy", "rmse", "r2")

    def add_arguments(self, parser):
        try:  # both array plugins share these options; --task is registered by tabular
            parser.add_argument("--x-key", help="[image/timeseries] array name of the inputs (X)")
            parser.add_argument("--y-key", help="[image/timeseries] array name of the targets (y)")
        except argparse.ArgumentError:
            pass

    def detect(self, path):
        if not str(path).lower().endswith(".npz"):
            return 0.0
        try:
            with np.load(path) as z:
                if "X" not in z.files or "y" not in z.files:
                    return 0.0
                guess = array_data.kind_of(z["X"])
        except (OSError, ValueError):
            return 0.0
        if guess is None:
            return 0.0
        return 0.95 if guess == self.kind else 0.3  # 0.3: still usable with --plugin

    def prepare(self, args):
        x_key, y_key = getattr(args, "x_key", None), getattr(args, "y_key", None)
        X, y = array_data.load(args.data, x_key, y_key)
        task = getattr(args, "task", "auto")
        task = core.detect_task(y) if task == "auto" else task
        d = array_data.build(X, y, self.kind, task, seed=args.seed)
        d["keys"] = {"x_key": x_key or "X", "y_key": y_key or "y"}
        return d

    def profile(self, data):
        p = profile.profile(data)
        p["kind"], p["shape"] = self.kind, data["shape"]
        # Heuristic parameter budget (NOT derived from a paper): ~300 params per training sample.
        p["max_params"] = int(np.clip(300 * p["n_train"], 5e4, 3e6))
        return p

    def generate(self, prof, n_per_family, seed):
        import torch

        rng = np.random.default_rng(seed)
        shape, out_dim = prof["shape"], prof["n_classes"] or 1
        cands, seen = [], set()
        for fam in self.families:
            for _ in range(n_per_family * 6):
                if sum(c["family"] == fam for c in cands) >= n_per_family:
                    break
                cfg = {"family": fam}
                for k, v in {**SPACE[fam], **TRAIN}.items():
                    cfg[k] = v[int(rng.integers(len(v)))]
                key = tuple(sorted(cfg.items()))
                if key in seen:
                    continue
                seen.add(key)
                cfg["shape"] = list(shape)
                try:  # cost model (v0): instantiate, run one tiny batch, count parameters
                    m = array_lib.build_model(cfg, shape[0], out_dim).eval()
                    with torch.no_grad():
                        m(torch.zeros(2, *shape))
                    cfg["params"] = array_lib.count_params(m)
                except (RuntimeError, ValueError):
                    continue
                if cfg["params"] <= prof["max_params"]:
                    cands.append(cfg)
        return cands

    def build_model(self, cfg, in_dim, out_dim):
        return array_lib.build_model(cfg, in_dim, out_dim)

    def reference(self, data):
        """Classical baselines on the flattened input (PCA to 64 dims, fit on train, if large)."""
        Xnp = data["Xnp"]
        if Xnp["tr"].shape[1] > REF_MAX_FEATURES:
            mu = Xnp["tr"].mean(0)
            _, _, vt = np.linalg.svd(Xnp["tr"] - mu, full_matrices=False)
            V = vt[:REF_PCA].T
            data = {**data, "Xnp": {k: (v - mu) @ V for k, v in Xnp.items()}}
        return reference.run_reference(data)

    def data_config(self, data):
        return {"kind": self.kind, "shape": data["shape"], **data["keys"]}


class ImagePlugin(ArrayPlugin):
    name = kind = "image"
    families = array_lib.IMAGE_FAMILIES


class TimeSeriesPlugin(ArrayPlugin):
    name = kind = "timeseries"
    families = array_lib.TS_FAMILIES
