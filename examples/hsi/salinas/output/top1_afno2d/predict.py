"""Predict with this baseline after `python train.py` (which writes model.pt, preprocess.json).

    python predict.py --data new_file [--out predictions.npy|.csv] [--export model.pt2|model.onnx]

new_file has the same format as the training data, without labels (CSV: the same columns;
.npz: the same array names; HSI: a cube). Output: class labels / target values in the original
units (segmentation: masks, reconstruction: images, HSI: a class map). Requires: torch, numpy.
"""
import argparse
from pathlib import Path

import numpy as np

import inference

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="predictions.npy", help=".npy, or .csv for one value per row")
ap.add_argument("--device", default="cpu")
ap.add_argument("--export", help="also save a deployable graph: model.pt2 (torch.export) or model.onnx")
a = ap.parse_args()

model, meta, cfg = inference.load(HERE, a.device)
X = inference.prepare_input(meta, cfg, a.data)
pred = inference.predict(model, meta, X, device=a.device)
if a.out.endswith(".csv"):
    np.savetxt(a.out, np.asarray(pred).reshape(len(pred), -1), delimiter=",", fmt="%s")
else:
    np.save(a.out, pred)
print(f"{len(pred)} predictions -> {a.out}")
if a.export:
    idx = np.arange(min(2, len(X)))
    sample = X[idx] if hasattr(X, "map_shape") else X[: len(idx)]
    print("exported", inference.export(model, sample.to(a.device), a.export))
