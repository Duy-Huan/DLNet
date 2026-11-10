"""Leave-one-dataset-out evaluation of the seed rankers (agent.md 16.6).

    python scripts/bench_surrogate.py --store meta_a.jsonl meta_b.jsonl

For every dataset of the meta-dataset (scripts/build_meta.py): fit on the OTHER datasets, rank
the held-out dataset's candidates, compare with their real validation scores.
Rankers:
  zap          zero-shot surrogate with meta-features (ZAP, arXiv:2206.08476)
  zap_no_meta  the same surrogate without meta-features (one global ranking for every dataset)
  knn_family   families ordered by the k nearest past datasets (meta.recommend, Feurer 2015)
  prior        DLNet data-card rules (dlnet/seeds.py)
  params       parameter count (larger first)
  borda_learned  Borda count of zap, zap_no_meta and knn_family (what `design` uses with a store)
  random       expected value of a random order
Metrics per held-out dataset: Spearman, regret@1 (best val - val of the ranker's first pick),
hit@3 (the best candidate is among the ranker's first 3), and their means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dlnet import meta, proxies, seeds, surrogate


def prior_from_x(rec_card, xs):
    w, _ = seeds.trait_weights(rec_card)
    t = seeds.TRAITS
    n = max(rec_card.get("n_train", 1), 1)
    out = []
    for x in xs:
        s = sum(w[name] for name, on in zip(t, x[: len(t)], strict=True) if on)
        p = 10 ** x[len(t)]
        if p > 100 * n:
            s -= 0.5 * np.log10(p / (100 * n))
        out.append(s)
    return out


def metrics(pred, vals):
    pred, vals = np.asarray(pred, float), np.asarray(vals, float)
    order = np.argsort(-pred, kind="stable")
    best = vals.max()
    return {
        "spearman": proxies.spearman(pred, vals),
        "regret1": float(best - vals[order[0]]),
        "hit3": float(vals.argmax() in order[:3]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", nargs="+", required=True)
    ap.add_argument("--out", default="bench_surrogate.json")
    a = ap.parse_args()
    recs = [r for p in a.store for r in meta.load(p)]
    plugin = recs[0]["plugin"]
    ds = surrogate.datasets_from_records(recs, plugin)
    by_name = {r["name"]: r for r in recs}
    res = {}
    for i, d in enumerate(ds):
        rest = ds[:i] + ds[i + 1 :]
        xs = [c["x"] for c in d["candidates"]]
        vals = [c["val"] for c in d["candidates"]]
        r = by_name[d["name"]]
        rankers = {
            "zap": surrogate.Surrogate().fit(rest).score(d["phi"], xs),
            "zap_no_meta": surrogate.Surrogate(use_meta=False).fit(rest).score(d["phi"], xs),
            "prior": prior_from_x(r["card"], xs),
            "params": [x[len(seeds.TRAITS)] for x in xs],
        }
        others = [o for o in recs if o["name"] != d["name"]]
        rec = meta.recommend(others, plugin, r["features"], k=5)
        if rec:
            pos = {f: k for k, f in enumerate(rec["family_order"])}
            rankers["knn_family"] = [-pos.get(c["family"], len(pos)) for c in d["candidates"]]
        learned = [(rankers[k], 1.0) for k in ("zap", "zap_no_meta", "knn_family") if k in rankers]
        rankers["borda_learned"] = seeds.borda(learned).tolist()
        m = {k: metrics(v, vals) for k, v in rankers.items()}
        n = len(vals)
        m["random"] = {
            "spearman": 0.0,
            "regret1": float(max(vals) - np.mean(vals)),
            "hit3": min(3 / n, 1.0),
        }
        res[d["name"]] = m
        print(
            d["name"],
            {
                k: (
                    None if v["spearman"] is None else round(v["spearman"], 2),
                    round(v["regret1"], 3),
                    v["hit3"],
                )
                for k, v in m.items()
            },
            flush=True,
        )
    names = ["borda_learned", "zap", "zap_no_meta", "knn_family", "prior", "params", "random"]
    print("\n| ranker | mean Spearman | mean regret@1 | hit@3 |\n|---|---|---|---|")
    summary = {}
    for k in names:
        rows = [m[k] for m in res.values() if k in m]
        if not rows:
            continue
        sp = [x["spearman"] for x in rows if x["spearman"] is not None]
        summary[k] = {
            "spearman": float(np.mean(sp)) if sp else None,
            "regret1": float(np.mean([x["regret1"] for x in rows])),
            "hit3": f"{sum(x['hit3'] for x in rows):.1f}/{len(rows)}",
        }
        s = summary[k]
        print(
            f"| {k} | {s['spearman']:+.3f} | {s['regret1']:.4f} | {s['hit3']} |"
            if s["spearman"] is not None
            else f"| {k} | - | {s['regret1']:.4f} | {s['hit3']} |"
        )
    Path(a.out).write_text(json.dumps({"per_dataset": res, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
