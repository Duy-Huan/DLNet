"""Hyperspectral image plugin: cube + ground-truth map -> pixel classification."""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

import numpy as np

from .. import hsi_data, threads, zoo
from ..core import compute_metrics
from ..profile import capacity_hint, intrinsic_dim_report, rounded, twonn_intrinsic_dim
from ..zoo import families
from . import DomainPlugin

HERE = Path(__file__).parent.parent

TRAIN_PY = '''"""Full training recipe for this HSI baseline. Usage:
    python train.py --data cube.mat --gt gt.mat [--cube-key K] [--gt-key K] [--epochs 100] [--seed 0]
Requires: torch, numpy, scipy.
"""
import argparse
import json

import core
import hsi_data
from model import build_model, count_params

cfg = json.load(open("config.yaml"))  # JSON is valid YAML
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--gt")
ap.add_argument("--cube-key")
ap.add_argument("--gt-key")
ap.add_argument("--epochs", type=int, default=cfg["recipe"]["epochs"])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dc = cfg["data"]
cube, gt = hsi_data.load(a.data, a.gt, a.cube_key, a.gt_key)
data = hsi_data.build(
    cube, gt, split=dc["split"], seed=cfg["split_seed"], patch=dc["patch"], pca=dc["pca"],
    frac=dc["frac"], block=dc["block"],
)
print("data:", {k: v for k, v in data["info"].items() if k != "classes"})
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
    p = cfg["data"]["patch"]
    m = build_model(cfg["arch"], cfg["in_dim"], cfg["out_dim"]).eval()
    out = m(torch.randn(2, cfg["in_dim"], p, p))
    assert out.shape == (2, cfg["out_dim"])
"""

SPACE = {
    "cnn1d": {"width": [16, 32, 64], "depth": [2, 3], "dropout": [0.0, 0.2, 0.4]},
    "cnn2d": {"width": [32, 64, 128], "depth": [2, 3], "dropout": [0.0, 0.2, 0.4]},
    "tucker_cnn2d": {
        "width": [64, 128, 256],
        "depth": [2, 3, 4],
        "dropout": [0.0, 0.2, 0.4],
        "rank_ratio": [0.25, 0.5],
    },
    "cnn3d": {"width": [8, 16], "depth": [2, 3], "dropout": [0.0, 0.2, 0.4]},
    "hybridsn": {"width": [4, 8], "dropout": [0.2, 0.4]},
    "resnet2d": {"width": [32, 64, 128], "depth": [1, 2, 3], "dropout": [0.0, 0.2, 0.4]},
    "dwsep2d": {"width": [32, 64, 128], "depth": [2, 3, 4], "dropout": [0.0, 0.2, 0.4]},
    "convnext2d": {"width": [32, 64], "depth": [1, 2, 3], "dropout": [0.0, 0.2]},
    "spec_transformer": {"width": [32, 64], "depth": [1, 2], "dropout": [0.0, 0.1]},
    "ssrn": {"width": [8, 16, 24], "depth": [1, 2], "dropout": [0.0, 0.2, 0.4]},
    # block choice comes from the grammar (hsi_lib.sample_composed), sizes from here
    "composed": {"width": [32, 64], "depth": [1, 2], "dropout": [0.0, 0.2, 0.4]},
}
TRAIN = {"lr": [1e-3, 3e-3], "wd": [1e-4, 1e-3], "batch": [32, 64]}


class HSIPlugin(DomainPlugin):
    name = "hsi"
    model_file = HERE / "zoo_model.py"
    extra_files: ClassVar[dict[str, Path]] = {"hsi_data.py": HERE / "hsi_data.py"}
    train_template = TRAIN_PY
    test_template = TEST_PY
    metric_keys = ("accuracy", "balanced_accuracy", "kappa")  # OA, AA, Kappa

    def add_arguments(self, parser):
        parser.add_argument("--gt", help="[hsi] ground-truth file (omit if inside --data)")
        parser.add_argument("--cube-key", help="[hsi] array name of the cube")
        parser.add_argument("--gt-key", help="[hsi] array name of the ground truth")
        parser.add_argument(
            "--split", default="pixel", choices=["pixel", "block"], help="[hsi] split mode"
        )
        parser.add_argument("--frac", help="[hsi] train,val,test fractions, e.g. 0.1,0.1,0.8")
        parser.add_argument("--patch", type=int, default=9, help="[hsi] odd patch size")
        parser.add_argument(
            "--pca", type=int, default=30, help="[hsi] spectral components (0 = keep all bands)"
        )
        parser.add_argument(
            "--block-size", type=int, default=16, help="[hsi] block size for --split block"
        )

    def default_cv_splits(self, args):
        # A block split leaves only a few validation blocks: one split ranks noisily (agent.md 9.7).
        return 3 if args.split == "block" else 1

    def detect(self, path):
        return 0.9 if str(path).lower().endswith((".mat", ".npz", ".npy")) else 0.0

    def prepare(self, args):
        if args.patch % 2 == 0:
            raise SystemExit("--patch must be odd")
        cube, gt = hsi_data.load(args.data, args.gt, args.cube_key, args.gt_key)
        frac = tuple(float(x) for x in args.frac.split(",")) if args.frac else None
        d = hsi_data.build(
            cube,
            gt,
            split=args.split,
            seed=args.seed,
            patch=args.patch,
            pca=args.pca,
            frac=frac,
            block=args.block_size,
        )
        d["info"]["multilinear_rank_99"] = hsi_data.multilinear_rank(cube)
        d["info"]["multilinear_rank_levels"] = {
            str(e): hsi_data.multilinear_rank(cube, energy=e) for e in (0.9, 0.95, 0.99)
        }
        d["info"]["cube_shape"] = list(cube.shape)
        return d

    def profile(self, data):
        i = data["info"]
        spectra = data["Xnp"]["tr"]
        d_int = twonn_intrinsic_dim(spectra)
        rep = intrinsic_dim_report(spectra)
        n = i["n_train"]
        p = {
            "task": "classification",
            "primary_metric": "accuracy (OA); also AA, Kappa",
            "cube_shape": i["cube_shape"],
            "n_classes": data["out_dim"],
            "n_train": n,
            "n_val": i["n_val"],
            "n_test": i["n_test"],
            "n_features": data["in_dim"],  # spectral components after the stem
            "patch": data["patch"],
            "split": data["split_mode"],
            "frac": i["frac"],
            "dropped_by_margin": i["n_dropped_by_margin"],
            "pca_energy_kept": round(i["pca_energy_kept"], 4),
            "multilinear_rank_99": i["multilinear_rank_99"],  # (rows, cols, bands), Tucker/HOSVD
            "multilinear_rank_levels": i.get("multilinear_rank_levels"),  # energy -> rank
            "intrinsic_dim": round(d_int, 2),
            "scaling_exponent_hat": round(4.0 / max(d_int, 1e-6), 3),
            "intrinsic_dim_estimators": rounded(rep),
            "capacity_hint": capacity_hint(rep["value"], n, data["in_dim"]),
            "classes_missing_train": i["classes_missing_train"],
            "classes_missing_test": i["classes_missing_test"],
        }
        # Heuristic parameter budget (NOT derived from the paper): ~500 params per training pixel.
        p["max_params"] = int(np.clip(500 * n, 1e5, 4e6))
        return p

    def generate(self, prof, n_per_family, seed):
        rng = np.random.default_rng(seed)
        cands, seen = [], set()
        for fam in families.HSI:
            for _ in range(n_per_family * 6):
                if sum(c["family"] == fam for c in cands) >= n_per_family:
                    break
                cfg = {"family": fam, "patch": prof["patch"]}
                for k, v in {**SPACE[fam], **TRAIN}.items():
                    cfg[k] = v[int(rng.integers(len(v)))]
                if fam == "composed":
                    cfg.update(families.sample_composed(rng))
                key = tuple(sorted(cfg.items()))
                if key in seen:
                    continue
                seen.add(key)
                try:  # static check, then cost model (v0): the parameter count of the whole model
                    families.complete(
                        cfg,
                        "hsi",
                        (prof["n_features"], prof["patch"], prof["patch"]),
                        prof["n_classes"],
                    )
                    cfg["params"] = families.count_params(cfg)
                except (RuntimeError, ValueError):
                    continue
                if cfg["params"] <= prof["max_params"]:
                    cands.append(cfg)
        return cands

    extra_dirs: ClassVar[dict[str, Path]] = {"zoo": HERE / "zoo"}

    def zoo_input(self, data):
        return "map", (data["in_dim"], data["patch"], data["patch"]), "cls"

    def build_model(self, cfg, in_dim, out_dim):
        return zoo.build_model(cfg, in_dim, out_dim)

    def reference(self, data):
        """Classical baselines on the same split: spectral-only AND spectral+spatial (patch-mean).

        A spectral-only reference would be unfair against patch-based CNNs. SVM's C is picked on
        validation from {1, 10, 100}.
        """
        import torch
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        def spatial(split):
            st, out = data["X" + split], []
            for i in range(0, len(st), 2048):
                out.append(st[torch.arange(i, min(i + 2048, len(st)))].mean(dim=(2, 3)).numpy())
            return np.hstack([data["Xnp"][split], np.vstack(out)])

        feats = {
            "spectral": {sp: data["Xnp"][sp] for sp in ("tr", "va", "te")},
            "spectral_spatial": {sp: spatial(sp) for sp in ("tr", "va", "te")},
        }
        out = []
        for fname, F in feats.items():
            fits = {f"svm_rbf_{fname}": [SVC(C=c, gamma="scale") for c in (1, 10, 100)]}
            fits[f"random_forest_{fname}"] = [
                RandomForestClassifier(300, n_jobs=threads.current(), random_state=0)
            ]
            for name, models in fits.items():
                t0, best = time.time(), None
                for m in models:
                    m.fit(F["tr"], data["y_raw"]["tr"])
                    res = {"name": name, "seconds": round(time.time() - t0, 1)}
                    for sp in ("va", "te"):
                        pred = np.asarray(m.predict(F[sp]))
                        res[sp] = compute_metrics(
                            data["y_raw"][sp], pred, "classification", "accuracy"
                        )
                    if best is None or res["va"]["score"] > best["va"]["score"]:
                        best = res
                out.append(best)
        out.sort(key=lambda r: -r["va"]["score"])
        return out

    def data_config(self, data):
        i = data["info"]
        return {
            "split": data["split_mode"],
            "frac": i["frac"],
            "patch": data["patch"],
            "pca": i["pca_components"] if i["pca_components"] < i["bands"] else 0,
            "block": i["block"],
        }
