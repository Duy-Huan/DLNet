"""Does the module zoo find better architectures than the fixed families? Same budget, both ways.

    python scripts/bench_zoo.py [--budget 100s] [--seeds 0,1]

For every dataset and seed: `dlnet design` with the legacy families only (--zoo 0 --evolve 0), then
with the zoo (presets + random + one evolution generation). Reports top-1 validation / test of
each, where the zoo winner came from, and the verdict against the classical references.
Results: agent.md sec. 13.6.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from dlnet import cli


def datasets(tmp: Path):
    from sklearn import datasets as D

    out = {}
    for name in ("breast_cancer", "wine"):
        b = getattr(D, f"load_{name}")()
        df = pd.DataFrame(b.data, columns=[f"f{i}" for i in range(b.data.shape[1])])
        df["y"] = b.target
        df.to_csv(tmp / f"{name}.csv", index=False)
        out[f"tabular/{name}"] = ["--data", str(tmp / f"{name}.csv"), "--target", "y"]
    X, y = D.make_classification(1500, 30, n_informative=10, n_classes=4, random_state=1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(30)])
    df["y"] = y
    df.to_csv(tmp / "mc.csv", index=False)
    out["tabular/make_classification_1500x30"] = ["--data", str(tmp / "mc.csv"), "--target", "y"]
    d = D.load_digits()
    np.savez(tmp / "digits.npz", X=d.images.astype("float32"), y=d.target)
    out["image/digits_8x8"] = ["--data", str(tmp / "digits.npz")]
    rng = np.random.default_rng(0)
    n, t = 600, 48
    y = rng.integers(0, 3, n)
    tt = np.arange(t)[None, :, None]
    freq = np.array([0.25, 0.4, 0.4])[y][
        :, None, None
    ]  # classes 1 and 2 differ only by an amplitude burst
    X = np.sin(freq * tt + rng.uniform(0, 6.28, (n, 1, 1))) + 0.8 * rng.normal(size=(n, t, 2))
    burst = (y == 2)[:, None, None] * np.exp(-((tt - 24) ** 2) / 20.0)
    np.savez(tmp / "ts.npz", X=(X + 1.5 * burst).astype("float32"), y=y)
    out["timeseries/synthetic_3class"] = ["--data", str(tmp / "ts.npz")]
    rng = np.random.default_rng(1)
    Xs = rng.normal(scale=0.6, size=(300, 24, 24)).astype("float32")
    ys = np.zeros((300, 24, 24), dtype=np.int64)
    for i in range(300):
        for cls, val in ((1, 1.2), (2, -1.2)):
            r, c = rng.integers(0, 18, 2)
            ys[i, r : r + 6, c : c + 6] = cls
            Xs[i, r : r + 6, c : c + 6] += val
    np.savez(tmp / "seg.npz", X=Xs, y=ys)
    out["segmentation/synthetic_squares"] = ["--data", str(tmp / "seg.npz")]
    return out


def run(argv, out):
    cli.main(
        [
            "design",
            *argv,
            "--out",
            str(out),
            "--topk",
            "3",
            "--per-family",
            "2",
            "--no-proxies",
            "--no-cache",
        ]
    )
    b = json.loads((out / "leaderboard.json").read_text())
    t = b["top_k"][0]
    return {"val": t["val_mean"], "val_std": t["val_std"], "test": t["test"].get("miou", t["test"].get("accuracy", t["test"].get("score"))),
            "name": t["name"], "origin": t.get("origin"), "wins": b["verdict"]["beats_reference"]}  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="100s")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, argv in datasets(tmp).items():
            if a.only and a.only not in name:
                continue
            for seed in map(int, a.seeds.split(",")):
                base = [*argv, "--budget", a.budget, "--seed", str(seed)]
                leg = run([*base, "--zoo", "0", "--evolve", "0"], tmp / "o_leg")
                zoo = run(
                    [*base, "--zoo", "14", "--evolve", "1", "--evolve-pop", "8"], tmp / "o_zoo"
                )
                rows.append((name, seed, leg, zoo))
                print(f"RESULT {name} seed={seed} legacy={leg} zoo={zoo}", flush=True)
    print(
        "\n| dataset | seed | legacy top-1 val | legacy test | zoo top-1 val | zoo test | zoo winner | origin | beats refs (legacy / zoo) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for name, seed, lg, zo in rows:
        print(f"| {name} | {seed} | {lg['val']:.4f} ± {lg['val_std']:.4f} | {lg['test']:.4f} | {zo['val']:.4f} ± {zo['val_std']:.4f} | "
              f"{zo['test']:.4f} | {zo['name']} | {zo['origin'] or 'legacy family'} | {lg['wins']} / {zo['wins']} |")  # fmt: skip
