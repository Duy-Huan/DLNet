"""Raw material for the v1.5 seed study: per candidate, per-step training losses, validation per
epoch, zero-cost proxies and the seed prior. Analysis is done afterwards on the JSON."""

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from dlnet import datacard, plugins, proxies, search, seeds

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--out")
a = ap.parse_args()
res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
plugin = plugins.get_plugin("image")
for path in a.data:
    for rep in range(a.reps):
        key = f"{Path(path).stem}|{rep}"
        if res.get(key, {}).get("done"):
            continue
        args = types.SimpleNamespace(data=path, seed=rep, x_key=None, y_key=None, task="auto")
        data = plugin.prepare(args)
        prof = plugin.profile(data)
        card = datacard.image_card(data, seed=rep)
        pool = plugin.generate(prof, 8, rep) + plugin.generate_zoo(data, prof, 40, rep, True, 16)
        rng = np.random.default_rng(100 + rep)
        cands = [pool[i] for i in rng.choice(len(pool), min(a.n, len(pool)), replace=False)]
        prior, _ = seeds.prior_scores(cands, card, "")
        rows = res.get(key, {}).get("rows", [])
        for ci, (c, p) in enumerate(zip(cands, prior)):
            if ci < res.get(key, {}).get("next", 0):
                continue
            t0 = time.time()
            torch.manual_seed(rep)
            try:
                s = proxies.score(
                    plugin.build_model(c, data["in_dim"], data["out_dim"]), data, seed=rep
                )
                tr = search._new_trainer(c, data, 0, plugin.build_model)
                est = tr.probe_epoch_seconds()
                if est > 12:
                    raise RuntimeError(f"too slow for the study: {est:.0f}s per epoch")
                tr = search._new_trainer(c, data, 0, plugin.build_model)
                losses, orig = [], tr.loss_fn

                def rec(o, y, _f=orig, _l=losses):
                    v = _f(o, y)
                    _l.append(float(v.detach()))
                    return v

                tr.loss_fn = rec
                vals, secs = [], []
                for _ in range(a.epochs):
                    tr.fit(1)
                    vals.append(tr.evaluate("va")["score"])
                    secs.append(tr.seconds)
            except Exception as e:  # noqa: BLE001 - a candidate that fails is recorded and skipped
                print("skip", c.get("family"), type(e).__name__, str(e)[:80], flush=True)
                res[key] = {"rows": rows, "next": ci + 1}
                Path(a.out).write_text(json.dumps(res))
                continue
            rows.append(
                {
                    "family": c["family"],
                    "origin": c.get("origin", "family"),
                    "params": c.get("params"),
                    "batch": c["batch"],
                    "lr": c["lr"],
                    "prior": p,
                    "proxies": s,
                    "losses": losses,
                    "val": vals,
                    "secs": secs,
                    "n_train": len(data["Xtr"]),
                }
            )
            res[key] = {"rows": rows, "next": ci + 1}
            Path(a.out).write_text(json.dumps(res))
            print(
                key,
                c["family"],
                len(losses),
                [round(v, 3) for v in vals[::3]],
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
        res[key] = {"rows": rows, "next": len(cands), "done": True}
        Path(a.out).write_text(json.dumps(res))
