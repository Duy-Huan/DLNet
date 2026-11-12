"""Full training recipe for this baseline. Usage:
    python train.py --data data.npz [--x-key X] [--y-key y] [--epochs 100] [--seed 0] [--device auto] [--amp auto]
Requires: torch, numpy.
"""
import argparse
import json
from pathlib import Path

import array_data
import core
from model import build_model, count_params

HERE = Path(__file__).resolve().parent  # runs from any folder
cfg = json.load(open(HERE / "config.yaml"))  # JSON is valid YAML
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--x-key", default=cfg["data"]["x_key"])
ap.add_argument("--y-key", default=cfg["data"]["y_key"])
ap.add_argument("--epochs", type=int, default=cfg["recipe"]["epochs"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1 | mps")
ap.add_argument("--amp", default="auto", choices=["auto", "off", "bf16", "fp16"])
a = ap.parse_args()

X, y = array_data.load(a.data, a.x_key, a.y_key)
data = array_data.build_any(X, y, cfg["data"]["kind"], cfg["task"], seed=cfg["split_seed"])
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
