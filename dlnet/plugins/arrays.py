"""Image classification, image segmentation and time-series plugins: .npz with X and y.

Shapes: see dlnet/array_data.py. Models: recipes in dlnet/zoo/families.py (the module zoo).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import numpy as np

from .. import array_data, core, profile, reference, threads, zoo
from ..zoo import families as recipes
from . import DomainPlugin

HERE = Path(__file__).parent.parent

RECON_TRAIN_PY = '''"""Full training recipe for this autoencoder. Usage:
    python train.py --data images.npz [--x-key X] [--epochs 100] [--seed 0]
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
ap.add_argument("--epochs", type=int, default=cfg["recipe"]["epochs"])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

data = array_data.build_reconstruction(array_data.load_x(a.data, a.x_key), seed=cfg["split_seed"])
model = build_model(cfg["arch"], data["in_dim"], data["out_dim"])
print(f"params={count_params(model):,} epochs={a.epochs}")
tr, best_val = core.train_full(
    model, data, cfg["arch"]["lr"], cfg["arch"]["wd"], cfg["arch"]["batch"], a.epochs, seed=a.seed
)
print("val :", tr.evaluate("va"))
print("test:", tr.evaluate("te"))
'''

RECON_TEST_PY = """import json
from pathlib import Path

import torch

from model import build_model


def test_forward_shape():
    cfg = json.load(open(Path(__file__).parent.parent / "config.yaml"))
    m = build_model(cfg["arch"], cfg["in_dim"], cfg["out_dim"]).eval()
    x = torch.randn(3, *cfg["data"]["shape"])
    assert m(x).shape == x.shape  # reconstruction: same shape as the input
"""

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
data = array_data.build_any(X, y, cfg["data"]["kind"], cfg["task"], seed=cfg["split_seed"])
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
    kind, shape = cfg["data"]["kind"], cfg["data"]["shape"]
    if kind == "ids":  # shape = cardinalities of the id fields
        x = torch.stack([torch.randint(0, c, (4,)) for c in shape], 1)
    elif kind == "graph":  # shape = (N, F + N): features, then a symmetric adjacency
        n = shape[0]
        a = (torch.rand(4, n, n) < 0.3).float()
        x = torch.cat([torch.randn(4, n, shape[1] - n), ((a + a.transpose(1, 2)) > 0).float()], -1)
    else:
        x = torch.randn(4, *shape)
    out = m(x)
    if cfg["data"]["kind"] == "segmentation":
        assert out.shape == (4, cfg["out_dim"], *shape[1:])
    else:
        assert out.shape == (4, cfg["out_dim"])
"""

TRAIN = {"lr": [1e-3, 3e-3], "wd": [1e-4, 1e-2], "batch": [32, 64]}
GRAPH_SPACE = {
    f: {
        "width": [16, 32, 64],
        "depth": [1, 2, 3],
        "dropout": [0.0, 0.1],
        "readout": ["mean", "max", "sum"],
    }
    for f in recipes.GRAPH
}
RECSYS_SPACE = {
    f: {"width": [8, 16, 32], "depth": [1, 2], "dropout": [0.0, 0.1]} for f in recipes.RECSYS
}
SPACE = {
    "conv_ae": {
        "width": [8, 16, 32],
        "depth": [2, 3],
        "dropout": [0.0],
        "latent": [2, 4, 8],
        "encoder": ["vgg", "resnet", "allconv"],
    },
    "vae": {
        "width": [8, 16, 32],
        "depth": [2, 3],
        "dropout": [0.0],
        "latent": [2, 4, 8],
        "encoder": ["vgg", "resnet", "allconv"],
        "beta": [1e-4, 1e-3, 1e-2],
    },
    "cnn": {"width": [16, 32], "depth": [2, 3], "dropout": [0.0, 0.2]},
    "resnet": {"width": [8, 16, 32], "depth": [1, 2], "dropout": [0.0, 0.2]},
    "vit": {"width": [32, 64], "depth": [2, 4], "dropout": [0.0, 0.1], "patch": [2, 4, 7, 8]},
    **{
        f: {
            "width": [8, 16, 32],
            "depth": [1, 2],
            "dropout": [0.0, 0.2],
            "attention": ["none", "none", "se"],
        }
        for f in [f for f in recipes.IMAGE if f in recipes.BACKBONES]
    },
    **{
        f: {
            "width": [8, 16, 32],
            "depth": [1, 2],
            "dropout": [0.0, 0.1],
            "encoder": ["vgg", "resnet", "densenet", "mobilenetv2"],
            "attention": ["none", "none", "se"],
            **({"dec_attention": ["none", "scse"]} if f in ("unet", "unetpp") else {}),
        }
        for f in recipes.SEGMENTATION
    },
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
    model_file = HERE / "zoo_model.py"
    extra_files: ClassVar[dict[str, Path]] = {"array_data.py": HERE / "array_data.py"}
    extra_dirs: ClassVar[dict[str, Path]] = {"zoo": HERE / "zoo"}
    head = "cls"

    def zoo_input(self, data):
        kind = "ser" if self.kind == "timeseries" else "map"
        return kind, tuple(data["shape"]), self.head

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
                if "X" not in z.files or "y" not in z.files or "A" in z.files:
                    return 0.0
                guess = array_data.kind_of(z["X"], z["y"])
        except (OSError, ValueError):
            return 0.0
        if guess is None:
            return 0.0
        return 0.95 if guess == self.kind else 0.0

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

    domain = ""

    def generate(self, prof, n_per_family, seed):
        rng = np.random.default_rng(seed)
        shape, out_dim = prof["shape"], prof["n_classes"] or 1
        cands, seen = [], set()
        for fam in self.families:
            for _ in range(n_per_family * 6):
                if sum(c["family"] == fam for c in cands) >= n_per_family:
                    break
                cfg = {"family": fam}
                for k, v in {
                    **SPACE.get(fam, GRAPH_SPACE.get(fam, RECSYS_SPACE.get(fam))),
                    **TRAIN,
                }.items():
                    cfg[k] = v[int(rng.integers(len(v)))]
                key = tuple(sorted(cfg.items()))
                if key in seen:
                    continue
                seen.add(key)
                try:  # static check, then cost model (v0): parameters of the whole model
                    recipes.complete(cfg, self.domain, shape, out_dim)
                    cfg["params"] = recipes.count_params(cfg)
                except (RuntimeError, ValueError):
                    continue
                if cfg["params"] <= prof["max_params"]:
                    cands.append(cfg)
        return cands

    def build_model(self, cfg, in_dim, out_dim):
        return zoo.build_model(cfg, in_dim, out_dim)

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


class ImageClassificationPlugin(ArrayPlugin):
    """One label (or one number) per image. `image` is accepted as an alias of the name."""

    name = "image_classification"
    kind = "image"
    families = recipes.IMAGE
    domain = "image"


class TimeSeriesPlugin(ArrayPlugin):
    name = kind = "timeseries"
    families = recipes.TIMESERIES
    domain = "timeseries"


class GraphClassificationPlugin(ArrayPlugin):
    """Graph-level prediction: .npz with X (G, N, F) node features, A (G, N, N) adjacency, y (G)."""

    name, kind, domain = "graph", "graph", "graph"
    families = recipes.GRAPH

    def detect(self, path):
        if not str(path).lower().endswith(".npz"):
            return 0.0
        try:
            with np.load(path) as z:
                ok = (
                    {"X", "A", "y"} <= set(z.files)
                    and z["X"].ndim == 3
                    and z["A"].shape == (*z["X"].shape[:2], z["X"].shape[1])
                )
        except (OSError, ValueError):
            return 0.0
        return 0.97 if ok else 0.0

    def zoo_input(self, data):
        return "graph", tuple(data["shape"]), "cls"

    def prepare(self, args):
        X, y = array_data.load(
            args.data, getattr(args, "x_key", None), getattr(args, "y_key", None)
        )
        task = getattr(args, "task", "auto")
        d = array_data.build_graph(
            X, y, core.detect_task(y) if task == "auto" else task, seed=args.seed
        )
        d["keys"] = {
            "x_key": getattr(args, "x_key", None) or "X",
            "y_key": getattr(args, "y_key", None) or "y",
        }
        return d

    def profile(self, data):
        p = profile.profile(data)
        p.update(kind="graph", shape=data["shape"], n_nodes=data["shape"][0])
        p["max_params"] = int(np.clip(300 * p["n_train"], 5e4, 3e6))
        return p


class RecommendationPlugin(ArrayPlugin):
    """Interactions of categorical fields (user, item, ...): .npz with ids (N, F) integers and y (N) rating or click."""

    name, kind, domain = "recsys", "ids", "recsys"
    families = recipes.RECSYS

    def detect(self, path):
        if not str(path).lower().endswith(".npz"):
            return 0.0
        try:
            with np.load(path) as z:
                return (
                    0.97
                    if {"ids", "y"} <= set(z.files)
                    and z["ids"].ndim == 2
                    and z["ids"].dtype.kind in "iu"
                    else 0.0
                )
        except (OSError, ValueError):
            return 0.0

    def zoo_input(self, data):
        return "ids", tuple(data["shape"]), "cls"

    def prepare(self, args):
        xk = getattr(args, "x_key", None) or "ids"
        X, y = array_data.load(args.data, xk, getattr(args, "y_key", None))
        task = getattr(args, "task", "auto")
        d = array_data.build_ids(
            X, y, core.detect_task(y) if task == "auto" else task, seed=args.seed
        )
        d["keys"] = {"x_key": xk, "y_key": getattr(args, "y_key", None) or "y"}
        return d

    def profile(self, data):
        p = profile.profile(data)
        p.update(kind="ids", shape=data["shape"], cards=data["cards"])
        p["max_params"] = int(np.clip(200 * p["n_train"], 5e4, 5e6))
        return p

    def generate(self, prof, n_per_family, seed):
        if len(prof["shape"]) != 2:  # the matrix-factorisation recipe reads exactly (user, item)
            self.families = tuple(f for f in recipes.RECSYS if f != "mf")
        return super().generate(prof, n_per_family, seed)


class ImageReconstructionPlugin(ArrayPlugin):
    """Autoencoders and variational autoencoders: .npz with images X only. Metric: reconstruction MSE (and PSNR)."""

    name, kind, domain = "image_reconstruction", "reconstruction", "reconstruction"
    families = recipes.RECONSTRUCTION
    train_template, test_template = RECON_TRAIN_PY, RECON_TEST_PY
    metric_keys = ("mse", "psnr")

    def detect(self, path):
        if not str(path).lower().endswith(".npz"):
            return 0.0
        try:
            with np.load(path) as z:
                label_free = "X" in z.files and not {"y", "A"} & set(z.files)
                return 0.95 if label_free and z["X"].ndim in (3, 4) else 0.0
        except (OSError, ValueError):
            return 0.0

    def zoo_input(self, data):
        return "map", tuple(data["shape"]), "dense"

    def prepare(self, args):
        xk = getattr(args, "x_key", None)
        d = array_data.build_reconstruction(array_data.load_x(args.data, xk), seed=args.seed)
        d["keys"] = {"x_key": xk or "X"}
        return d

    def profile(self, data):
        p = profile.profile(data)
        p.update(kind="image", shape=data["shape"], n_pixels=int(np.prod(data["shape"])))
        p["max_params"] = int(np.clip(300 * p["n_train"], 5e4, 3e6))
        return p

    def generate(self, prof, n_per_family, seed):
        cands = super().generate(prof, n_per_family, seed)
        for c in cands:
            c["code"] = recipes.code_size(c, prof["shape"])
        return cands

    def reference(self, data):
        """PCA reconstructions with 0 (the mean image), 8 and 32 components, fit on the training images only."""
        import time

        Xtr = data["Xnp"]["tr"]
        mu = Xtr.mean(0)
        _, _, vt = np.linalg.svd(Xtr - mu, full_matrices=False)
        out = []
        sp = SPACE["conv_ae"]
        codes = {  # every code size a candidate can have, so each one meets a PCA of its size
            recipes.code_size({"latent": lat, "depth": d}, data["shape"])
            for lat in sp["latent"] for d in sp["depth"]
        }  # fmt: skip
        for k in sorted({0, 8, 32} | {c for c in codes if c < min(Xtr.shape)}):
            t0 = time.time()
            V = vt[:k]
            res = {
                "name": "mean_image" if k == 0 else f"pca_{k}_components",
                "seconds": 0.0,
                "code": k,
            }
            for sp in ("va", "te"):
                X = data["Xnp"][sp]
                rec = mu + (X - mu) @ V.T @ V if k else np.tile(mu, (len(X), 1))
                res[sp] = core.compute_metrics(X, rec, "reconstruction", "neg_mse")
            res["seconds"] = round(time.time() - t0, 2)
            out.append(res)
        out.sort(key=lambda r: -r["va"]["score"])
        return out

    def reference_for(self, cfg, reference):
        """Judge an autoencoder against PCA with at most as many components as its code has elements."""
        code = cfg.get("code")
        if not code:
            return reference
        fair = [r for r in reference if r.get("code", 0) <= code]
        return sorted(fair, key=lambda r: -r["va"]["score"]) or reference

    def data_config(self, data):
        return {"kind": "reconstruction", "shape": data["shape"], **data["keys"]}


def pixel_features(X: np.ndarray, k: int = 5) -> np.ndarray:
    """(N, C, H, W) -> (N, H, W, C k k): the k x k neighbourhood of every pixel (reflect padding)."""
    p = k // 2
    Xp = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(Xp, (k, k), axis=(2, 3))  # N, C, H, W, k, k
    return win.transpose(0, 2, 3, 1, 4, 5).reshape(*X.shape[:1], *X.shape[2:], -1)


class ImageSegmentationPlugin(ArrayPlugin):
    """One class per pixel: X (N, H, W[, C]) and masks y (N, H, W). Metric: mean IoU."""

    name = "image_segmentation"
    kind = "segmentation"
    families = recipes.SEGMENTATION
    domain = "segmentation"
    head = "dense"
    metric_keys = ("miou", "accuracy", "balanced_accuracy", "kappa")
    REF_PIXELS = 20000  # training pixels sampled for the classical per-pixel baselines

    def prepare(self, args):
        x_key, y_key = getattr(args, "x_key", None), getattr(args, "y_key", None)
        X, y = array_data.load(args.data, x_key, y_key)
        if y.ndim != 3:
            raise SystemExit(f"segmentation needs masks y of shape (N, H, W), got {y.shape}")
        d = array_data.build_segmentation(X, y, seed=args.seed)
        d["keys"] = {"x_key": x_key or "X", "y_key": y_key or "y"}
        return d

    def profile(self, data):
        p = super().profile(data)
        p["n_pixels_train"] = int(np.sum(data["y_raw"]["tr"] >= 0))
        return p

    def reference(self, data):
        """Per-pixel classical classifiers on the 5x5 neighbourhood of each pixel.

        Same information level as a small CNN's first layers (local context), far less than a
        deep encoder's receptive field. Trained on at most REF_PIXELS labelled training pixels.
        """
        import time

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        feats = {k: pixel_features(data["X" + k].numpy()) for k in ("tr", "va", "te")}
        ytr = data["y_raw"]["tr"]
        idx = np.flatnonzero(ytr.reshape(-1) >= 0)
        rng = np.random.default_rng(data["seed"])
        idx = rng.choice(idx, min(self.REF_PIXELS, len(idx)), replace=False)
        Ftr, ltr = feats["tr"].reshape(-1, feats["tr"].shape[-1])[idx], ytr.reshape(-1)[idx]
        out = []
        for name, m in (
            ("pixel_logistic_regression_5x5", LogisticRegression(max_iter=500)),
            (
                "pixel_random_forest_5x5",
                RandomForestClassifier(100, n_jobs=threads.current(), random_state=0),
            ),
        ):
            t0 = time.time()
            m.fit(Ftr, ltr)
            res = {"name": name, "seconds": round(time.time() - t0, 1)}
            for sp in ("va", "te"):
                y = data["y_raw"][sp]
                pred = m.predict(feats[sp].reshape(-1, feats[sp].shape[-1])).reshape(y.shape)
                res[sp] = core.compute_metrics(y, pred, "classification", "miou")
            out.append(res)
        out.sort(key=lambda r: -r["va"]["score"])
        return out
