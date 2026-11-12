"""Full training recipe for this HSI baseline. Usage:
    python train.py --data cube.mat --gt gt.mat [--cube-key K] [--gt-key K] [--epochs 100] [--seed 0] [--device auto] [--amp auto]
Requires: torch, numpy, scipy.
"""
import argparse
import json
from pathlib import Path

import core
import hsi_data
from model import build_model, count_params

HERE = Path(__file__).resolve().parent  # runs from any folder
cfg = json.load(open(HERE / "config.yaml"))  # JSON is valid YAML
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--gt")
ap.add_argument("--cube-key")
ap.add_argument("--gt-key")
ap.add_argument("--epochs", type=int, default=cfg["recipe"]["epochs"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1 | mps")
ap.add_argument("--amp", default="auto", choices=["auto", "off", "bf16", "fp16"])
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
    model, data, cfg["arch"]["lr"], cfg["arch"]["wd"], cfg["arch"]["batch"], a.epochs, seed=a.seed,
    device=a.device, amp=a.amp, recipe=cfg["recipe"].get("name"),
)
print("val :", tr.evaluate("va"))
print("test:", tr.evaluate("te"))
core.save_artifacts(model, data, HERE)
print(f"saved {HERE / 'model.pt'} and preprocess.json; predict with: python predict.py --data <new file>")
