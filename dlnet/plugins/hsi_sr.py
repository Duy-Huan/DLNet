"""Hyperspectral super-resolution plugin: a high-resolution cube -> networks that upsample its bands.

Data protocol, split and metrics: dlnet/hsi_sr_data.py and core.sr_metrics. Models: the `hsi_sr`
recipes of dlnet/zoo/families.py (SRCNN, VDSR, EDSR, RCAN, SSPSR, 3D-FCNN) and zoo architectures
that keep the spatial size, all closed by the `srN` head (pixel shuffle + bicubic residual).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from .. import hsi_sr_data, zoo
from ..core import sr_metrics
from ..zoo import families
from . import DomainPlugin

HERE = Path(__file__).parent.parent

TRAIN_PY = '''"""Full training recipe for this hyperspectral super-resolution baseline. Usage:
    python train.py --data cube.mat [--cube-key K] [--epochs 100] [--seed 0] [--device auto] [--amp auto]
The cube is the high-resolution ground truth; its low-resolution input is simulated as in the search.
Requires: torch, numpy (scipy for .mat).
"""
import argparse
import json
from pathlib import Path

import core
import hsi_sr_data
from model import build_model, count_params

HERE = Path(__file__).resolve().parent  # runs from any folder
cfg = json.load(open(HERE / "config.yaml"))  # JSON is valid YAML
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--cube-key", default=cfg["data"]["cube_key"])
ap.add_argument("--epochs", type=int, default=cfg["recipe"]["epochs"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1 | mps")
ap.add_argument("--amp", default="auto", choices=["auto", "off", "bf16", "fp16"])
a = ap.parse_args()

dc = cfg["data"]
data = hsi_sr_data.build(hsi_sr_data.load(a.data, a.cube_key), scale=dc["scale"], patch=dc["patch"],
                         seed=cfg["split_seed"], degradation=dc["degradation"])
print("data:", data["info"])
model = build_model(cfg["arch"], data["in_dim"], data["out_dim"])
print(f"params={count_params(model):,} epochs={a.epochs}")
tr, best_val = core.train_full(
    model, data, cfg["arch"]["lr"], cfg["arch"]["wd"], cfg["arch"]["batch"], a.epochs, seed=a.seed,
    device=a.device, amp=a.amp, recipe=cfg["recipe"].get("name"),
)
print("val :", tr.evaluate("va"))
print("test:", tr.evaluate("te"))
core.save_artifacts(model, data, HERE)
print(f"saved {HERE / 'model.pt'} and preprocess.json; upscale a low-resolution cube with: "
      "python predict.py --data <low-resolution cube> --out sr.npy")
'''

TEST_PY = """import json
from pathlib import Path

import torch

from model import build_model


def test_forward_shape():
    cfg = json.load(open(Path(__file__).parent.parent / "config.yaml"))
    m = build_model(cfg["arch"], cfg["in_dim"], cfg["out_dim"]).eval()
    b, h, w = cfg["arch"]["input"]["dims"]
    s = cfg["data"]["scale"]
    assert m(torch.randn(2, b, h, w)).shape == (2, cfg["out_dim"], h * s, w * s)
"""

SPACE = {
    "srcnn": {"width": [32, 64], "depth": [2]},
    "vdsr": {"width": [32, 64], "depth": [4, 6, 8]},
    "edsr": {"width": [32, 64], "depth": [2, 4, 8]},
    "rcan": {"width": [32, 64], "depth": [2, 4]},
    "sspsr": {"width": [32, 64], "depth": [2, 4]},
    "fcnn3d": {"width": [8, 16], "depth": [2, 3]},
}
# Adam-type learning rates of the SR literature are 1e-4..2e-4 for 300+ epochs; a short search budget
# needs larger steps. Small batches: one tile already holds every band.
TRAIN = {"lr": [3e-4, 1e-3], "wd": [0.0, 1e-4], "batch": [8, 16, 32]}
REFERENCES = ("nearest", "bilinear", "bicubic")


class HSISuperResolutionPlugin(DomainPlugin):
    name = "hsi_sr"
    model_file = HERE / "zoo_model.py"
    extra_files: ClassVar[dict[str, Path]] = {"hsi_sr_data.py": HERE / "hsi_sr_data.py"}
    extra_dirs: ClassVar[dict[str, Path]] = {"zoo": HERE / "zoo"}
    train_template = TRAIN_PY
    test_template = TEST_PY
    metric_keys = ("psnr", "ssim", "sam", "ergas")
    zoo_train: ClassVar[dict] = TRAIN

    def add_arguments(self, parser):
        parser.add_argument(
            "--scale", type=int, default=2, help="[hsi_sr] upsampling factor (2, 3, 4, ...)"
        )
        parser.add_argument("--sr-patch", type=int, default=32,
                            help="[hsi_sr] side of a high-resolution tile (a multiple of --scale)")  # fmt: skip
        parser.add_argument("--degrade", default="bicubic", choices=hsi_sr_data.DEGRADATIONS,
                            help="[hsi_sr] how the low-resolution input is simulated")  # fmt: skip

    def detect(self, path):
        """Only a .npz with an `hr` array: a .mat / .npy cube is classification data unless
        `--plugin hsi_sr` is given."""
        if not str(path).lower().endswith(".npz"):
            return 0.0
        try:
            with np.load(path) as z:
                return 0.97 if {"hr", "HR"} & set(z.files) else 0.0
        except (OSError, ValueError):
            return 0.0

    def prepare(self, args):
        key = getattr(args, "cube_key", None)
        try:
            cubes = hsi_sr_data.load(args.data, key)
            d = hsi_sr_data.build(cubes, scale=args.scale, patch=args.sr_patch, seed=args.seed,
                                  degradation=args.degrade)  # fmt: skip
        except ValueError as e:
            raise SystemExit(f"hsi_sr: {e}") from None
        d["keys"] = {"cube_key": key}
        i = d["info"]
        print(f"[hsi_sr] {i['cubes']} cube(s) {i['cube_shape']}, x{i['scale']} ({i['degradation']}), tiles "
              f"{i['patch']} px split by {i['split_by']}: train {i['n_train']} / val {i['n_val']} / test {i['n_test']}")  # fmt: skip
        if i["split_by"] == "tile":
            d["warnings"] = [(
                "Fewer than 5 cubes: the tiles of one scene were split, so validation and test tiles are "
                "unseen pixels of the SAME scene. Scores on a new scene can be lower; give several cubes "
                "(.npz with hr (N, H, W, bands)) to test on unseen scenes."
            )]  # fmt: skip
        return d

    def profile(self, data):
        i = data["info"]
        p = {
            "task": "super_resolution", "primary_metric": "psnr (mean over bands, dB); also SSIM, SAM, ERGAS",
            "cube_shape": i["cube_shape"], "cubes": i["cubes"], "bands": data["in_dim"], "scale": i["scale"],
            "degradation": i["degradation"], "patch": i["patch"], "split_by": i["split_by"],
            "n_train": i["n_train"], "n_val": i["n_val"], "n_test": i["n_test"], "n_classes": data["out_dim"],
            "shape": data["shape"],
        }  # fmt: skip
        # Heuristic parameter budget (NOT from a paper): ~1000 parameters per training tile, and room for
        # the upsampling head (3 x 3 x width x bands x scale^2) of many-band cubes.
        head = 9 * 64 * data["out_dim"] * i["scale"] ** 2
        p["max_params"] = int(np.clip(1000 * i["n_train"], 2e5, 4e6) + head)
        return p

    def generate(self, prof, n_per_family, seed):
        rng = np.random.default_rng(seed)
        cands, seen = [], set()
        for fam in families.HSI_SR:
            for _ in range(n_per_family * 6):
                if sum(c["family"] == fam for c in cands) >= n_per_family:
                    break
                cfg = {"family": fam, "dropout": 0.0, "scale": prof["scale"]}
                for k, v in {**SPACE[fam], **TRAIN}.items():
                    cfg[k] = v[int(rng.integers(len(v)))]
                key = tuple(sorted(cfg.items()))
                if key in seen:
                    continue
                seen.add(key)
                families.complete(cfg, "hsi_sr", prof["shape"], prof["n_classes"])
                cfg["params"] = families.count_params(cfg)
                if cfg["params"] <= prof["max_params"]:
                    cands.append(cfg)
        return cands

    def zoo_input(self, data):
        return "map", tuple(data["shape"]), f"sr{data['scale']}"

    def build_model(self, cfg, in_dim, out_dim):
        return zoo.build_model(cfg, in_dim, out_dim)

    def reference(self, data):
        """Classical interpolation of the same low-resolution tiles (nearest, bilinear, bicubic)."""
        out = []
        for mode in REFERENCES:
            t0, res = time.time(), {"name": f"{mode}_interpolation"}
            for sp in ("va", "te"):
                with torch.no_grad():
                    up = hsi_sr_data.upsample(data["X" + sp], data["scale"], mode)
                res[sp] = sr_metrics(data["y_raw"][sp], up.numpy(), data["scale"])
            res["seconds"] = round(time.time() - t0, 2)
            out.append(res)
        out.sort(key=lambda r: -r["va"]["score"])
        return out

    def data_config(self, data):
        i = data["info"]
        return {
            "scale": i["scale"],
            "patch": i["patch"],
            "degradation": i["degradation"],
            **data["keys"],
        }
