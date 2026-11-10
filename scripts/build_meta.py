"""Build a meta-dataset for the zero-shot surrogate (dlnet/surrogate.py) from image datasets.

    python scripts/build_meta.py --data a.npz b.npz ... --n 24 --epochs 4 --out meta_image.jsonl

For each dataset: the same candidate pool as `dlnet design` (families + 40 zoo architectures),
`n` candidates drawn at random (seeded), costed, trained for `epochs` epochs (seed 0) and
evaluated on validation. One meta-store record per dataset (data card + every candidate's
features and score) is appended to --out, in the format `dlnet design --meta-store` writes.
"""

from __future__ import annotations

import argparse
import math
import time
import types
from pathlib import Path

import numpy as np
import torch

from dlnet import datacard, meta, plugins, search
from dlnet.zoo import cost


def build(path, n, epochs, seed):
    plugin = plugins.get_plugin("image")
    data = plugin.prepare(
        types.SimpleNamespace(data=path, seed=seed, x_key=None, y_key=None, task="auto")
    )
    prof = plugin.profile(data)
    card = datacard.image_card(data, seed=seed)
    pool = plugin.generate(prof, 8, seed) + plugin.generate_zoo(data, prof, 40, seed, True, 16)
    rng = np.random.default_rng(1000 + seed)
    cands = [pool[i] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)]
    kind, dims, _ = plugin.zoo_input(data)
    trace = []
    for c in cands:
        try:
            c.update(
                cost.estimate(
                    plugin.build_model, c, data["in_dim"], data["out_dim"], dims, kind=kind
                )
            )
        except (RuntimeError, ValueError, TypeError):
            pass
        t = search._new_trainer(c, data, 0, plugin.build_model)
        t.fit(epochs)
        v = t.evaluate("va")["score"]
        trace.append((c, v if math.isfinite(v) else float("nan")))
    return meta.make_record(plugin.name, prof, trace, Path(path).name, card=card, epochs=epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="meta_image.jsonl")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    for p in a.data:
        t0 = time.time()
        rec = build(p, a.n, a.epochs, a.seed)
        if rec:
            meta.append(a.out, rec)
            vals = [c["val"] for c in rec["candidates"]]
            print(
                f"{Path(p).name}: {len(vals)} candidates, val {min(vals):.3f}..{max(vals):.3f}, {time.time() - t0:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
