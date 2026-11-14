"""How many published networks can DLNet rebuild exactly? (`paper import` + `paper verify`, in memory)

    python scripts/bench_import.py --source torchvision            # every torchvision classifier
    python scripts/bench_import.py --source timm --limit 200       # a deterministic sample of timm
    python scripts/bench_import.py --source torchvision --only resnet18,vit_b_16 -v

For every model (random weights, nothing is downloaded): trace it into card nodes
(dlnet/paper_import.py), build the card's network with the zoo, copy the original weights by layer
name, and compare the outputs on random input. A model counts as rebuilt when the outputs agree
within 1e-5 x max(1, |y|) and the parameter counts are equal: fidelity level B of a paper card.
Writes a JSON report (--out) with the reason of every failure, grouped, so the missing modules show.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlnet import paper_import

MAX_PARAMS = 160e6  # larger networks are skipped (memory of the CI runner), reported as such


def torchvision_models():
    from torchvision import models

    out = []
    for name in sorted(models.list_models(module=models)):
        size = 299 if name == "inception_v3" else 224
        # googlenet / inception_v3: the auxiliary classifiers are training-only heads, not used at inference
        kw = (
            {"aux_logits": False, "init_weights": True}
            if name in ("googlenet", "inception_v3")
            else {}
        )
        out.append(
            (
                name,
                lambda n=name, kw=kw: models.get_model(n, weights=None, num_classes=1000, **kw),
                (3, size, size),
            )
        )
    return out


def timm_models(limit, seed=0):
    import random

    import timm

    names = sorted(timm.list_models())
    if limit and limit < len(names):
        names = sorted(random.Random(seed).sample(names, limit))
    out = []
    for name in names:

        def make(n=name):
            return timm.create_model(n, pretrained=False, num_classes=1000)

        out.append((name, make, None))
    return out


def one(name, make, dims, tol=1e-5):
    t0 = time.time()
    torch.manual_seed(0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = make().eval()
    except Exception as e:  # noqa: BLE001
        return {"status": "skip", "reason": f"cannot create: {type(e).__name__}: {str(e)[:120]}"}
    n = sum(p.numel() for p in model.parameters())
    if n > MAX_PARAMS:
        return {
            "status": "skip",
            "reason": f"{n / 1e6:.0f}M parameters (> {MAX_PARAMS / 1e6:.0f}M)",
        }
    if dims is None:  # timm: the model's own input size
        cfg = getattr(model, "pretrained_cfg", None) or {}
        dims = tuple(cfg.get("input_size") or (3, 224, 224))
    try:
        res = paper_import.rebuild_check(model, "map", dims, classes=1000, tol=tol)
    except paper_import.ImportUnsupported as e:
        return {
            "status": "unsupported",
            "reason": str(e)[:300],
            "params": n,
            "seconds": round(time.time() - t0, 1),
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:250]}",
                "trace": traceback.format_exc().splitlines()[-3:], "params": n}  # fmt: skip
    res["seconds"] = round(time.time() - t0, 1)
    res["status"] = "rebuilt" if res["level"] == "B" else "differs"
    return res


def key(reason: str) -> str:
    """Group failures: the module or operation named in the reason."""
    import re

    m = re.search(
        r"(layer \w+|operation '[^']+'|method '[^']+'|function '[^']+'|torch.fx cannot trace)",
        reason,
    )
    return m.group(1) if m else reason[:60]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["torchvision", "timm"], default="torchvision")
    ap.add_argument("--limit", type=int, default=0, help="timm: number of models sampled (0 = all)")
    ap.add_argument("--only", help="comma-separated model names")
    ap.add_argument("--out", help="JSON report")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--min-rate", type=float, default=0.0, help="exit 1 when fewer are rebuilt (CI floor)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    models = torchvision_models() if a.source == "torchvision" else timm_models(a.limit)
    if a.only:
        want = set(a.only.split(","))
        models = [m for m in models if m[0] in want]
    results = {}
    for name, make, dims in models:
        r = one(name, make, dims)
        results[name] = r
        line = f"{name:<40} {r['status']:<11} {r.get('max_abs_diff', '')!s:<24} {r.get('reason', '')[:110]}"
        print(line, flush=True)
        if a.v and r.get("trace"):
            print("   ", "\n    ".join(r["trace"]))
    counts = collections.Counter(r["status"] for r in results.values())
    tried = sum(v for k, v in counts.items() if k != "skip")
    print(f"\n{a.source}: {counts['rebuilt']} rebuilt of {tried} tried "
          f"({100 * counts['rebuilt'] / max(tried, 1):.0f}%); {dict(counts)}")  # fmt: skip
    why = collections.Counter(
        key(r["reason"]) for r in results.values() if r["status"] in ("unsupported", "error")
    )
    for k, v in why.most_common(25):
        print(f"  {v:>4}  {k}")
    if a.out:
        Path(a.out).write_text(json.dumps({"source": a.source, "counts": dict(counts), "missing": why.most_common(),
                                           "models": results}, indent=1, default=str))  # fmt: skip
    rate = counts["rebuilt"] / max(tried, 1)
    if rate < a.min_rate:
        print(f"FAILED: {rate:.0%} rebuilt, the floor is {a.min_rate:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
