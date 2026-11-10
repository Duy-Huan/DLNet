"""`dlnet unfold`: design unfolded networks for y = A x + noise and compare them honestly.

For every depth K: train LISTA (several seeds), and compare with ISTA and FISTA run for the SAME
number of iterations K, each with its lambda tuned on the validation split. The acceptance rule
is on the held-out test split.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

HERE = Path(__file__).parent

TRAIN_PY = '''"""Re-train this unfolded network. Usage:
    python train.py --data problem.npz [--epochs 200] [--seed 0]
problem.npz holds A (m x n), Y (N x m), X (N x n). Requires: torch, numpy.
"""
import argparse
import json

import torch

import model as U

cfg = json.load(open("config.yaml"))  # JSON is valid YAML
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--epochs", type=int, default=cfg["recipe"]["epochs"])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

A, Y, X = U.load_problem(a.data)
tr, va, te = U.split_indices(len(Y), cfg["split_seed"])
torch.manual_seed(a.seed)
net = U.LISTA(A, cfg["arch"]["K"], lam=cfg["arch"]["lam_init"], shared=cfg["arch"]["shared"])
best = U.train_lista(net, Y[tr], X[tr], Y[va], X[va], epochs=a.epochs, lr=cfg["arch"]["lr"],
                     batch=cfg["arch"]["batch"], seed=a.seed)
with torch.no_grad():
    print(f"val NMSE {best:.2f} dB   test NMSE {U.nmse_db(net(Y[te]), X[te]):.2f} dB")
torch.save(net.state_dict(), "weights.pt")
'''

TEST_PY = """import json
from pathlib import Path

import torch

import model as U


def test_untrained_network_equals_ista():
    cfg = json.load(open(Path(__file__).parent.parent / "config.yaml"))
    torch.manual_seed(0)
    m, n, K = cfg["m"], cfg["n"], cfg["arch"]["K"]
    A = torch.randn(m, n) / m**0.5
    Y = torch.randn(4, m)
    net = U.LISTA(A, K, lam=0.1, shared=cfg["arch"]["shared"])
    out = net(Y)
    assert out.shape == (4, n)
    assert torch.allclose(out, U.ista(A, Y, 0.1, K), atol=1e-5)
"""


def _stats(v):
    import numpy as np

    return float(np.mean(v)), float(np.std(v))


def run(args) -> int:
    import torch

    from . import export, unfold

    t0 = time.time()
    export.check_out_dir(args.out)
    A, Y, X = unfold.load_problem(args.data)
    tr, va, te = unfold.split_indices(len(Y), args.seed)
    if min(len(tr), len(va), len(te)) < 2:
        raise SystemExit(f"too few samples ({len(Y)}) for a 60/20/20 split")
    depths = sorted({int(k) for k in args.layers.split(",")})
    print(
        f"[unfold] A {tuple(A.shape)}  train/val/test = {len(tr)}/{len(va)}/{len(te)}  K={depths}"
    )

    rows = []
    for K in depths:
        lam_i, ista_va = unfold.tune_lambda(unfold.ista, A, Y[va], X[va], K)
        lam_f, fista_va = unfold.tune_lambda(unfold.fista, A, Y[va], X[va], K)
        base = {
            "ista": {"lam": lam_i, "val": ista_va,
                     "test": unfold.nmse_db(unfold.ista(A, Y[te], lam_i, K), X[te])},
            "fista": {"lam": lam_f, "val": fista_va,
                      "test": unfold.nmse_db(unfold.fista(A, Y[te], lam_f, K), X[te])},
        }  # fmt: skip
        vals, tests, best_state = [], [], None
        for s in range(args.seeds):
            torch.manual_seed(args.seed + s)
            net = unfold.LISTA(A, K, lam=lam_i, shared=args.shared)
            v = unfold.train_lista(
                net, Y[tr], X[tr], Y[va], X[va], epochs=args.epochs, lr=args.lr, batch=args.batch,
                seed=args.seed + s,
            )  # fmt: skip
            with torch.no_grad():
                tests.append(unfold.nmse_db(net(Y[te]), X[te]))
            if not vals or v < min(vals):
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            vals.append(v)
        (vm, vs), (tm, ts) = _stats(vals), _stats(tests)
        best_classical = min(base["ista"]["test"], base["fista"]["test"])
        row = {
            "K": K, "params": sum(p.numel() for p in net.parameters()),
            "val_mean": vm, "val_std": vs, "test_mean": tm, "test_std": ts, "n_seeds": args.seeds,
            "baselines_same_K": base,
            # acceptance: better than BOTH classical solvers on test by more than the seed std
            "beats_iterative": bool(tm + ts < best_classical),
            "gain_db_vs_best_classical": round(best_classical - tm, 3),
            "_state": best_state,
        }  # fmt: skip
        rows.append(row)
        print(
            f"[unfold] K={K:<3} LISTA test {tm:7.2f} ± {ts:.2f} dB | ISTA {base['ista']['test']:7.2f}"
            f" | FISTA {base['fista']['test']:7.2f} | beats={row['beats_iterative']}"
        )

    rows.sort(key=lambda r: r["val_mean"])  # lower NMSE is better
    out = Path(args.out)
    export._prepare_out_dir(out)
    for i, r in enumerate(rows[: args.topk], 1):
        d = out / f"top{i}_lista_K{r['K']}"
        (d / "tests").mkdir(parents=True)
        shutil.copy(HERE / "unfold.py", d / "model.py")
        torch.save(r["_state"], d / "weights.pt")
        cfg = {
            "plugin": "unfold", "task": "sparse linear inverse problem", "split_seed": args.seed,
            "m": A.shape[0], "n": A.shape[1],
            "arch": {"family": "lista_cp", "K": r["K"], "shared": args.shared,
                     "lam_init": r["baselines_same_K"]["ista"]["lam"], "lr": args.lr,
                     "batch": args.batch},
            "recipe": {"epochs": args.epochs, "schedule": "cosine", "select": "best-val weights"},
            "search_result": {k: r[k] for k in ("val_mean", "val_std", "test_mean", "test_std")},
        }  # fmt: skip
        (d / "config.yaml").write_text(json.dumps(cfg, indent=2))
        (d / "train.py").write_text(TRAIN_PY)
        (d / "tests" / "test_model.py").write_text(TEST_PY)
    for r in rows:
        del r["_state"]
    board = {
        "task": "unfold", "metric": "NMSE in dB (lower is better)", "top_k": rows,
        "verdict": {"beats_iterative_at_every_K": all(r["beats_iterative"] for r in rows)},
    }  # fmt: skip
    (out / "leaderboard.json").write_text(json.dumps(board, indent=2))
    md = [
        "# Leaderboard (deep unfolding, NMSE in dB on the test split, lower is better)\n",
        "| rank | model | params | LISTA (mean ± std) | ISTA, same K | FISTA, same K | gain | beats both |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        b = r["baselines_same_K"]
        md.append(
            f"| {i} | lista_K{r['K']} | {r['params']:,} | {r['test_mean']:.2f} ± {r['test_std']:.2f} | "
            f"{b['ista']['test']:.2f} | {b['fista']['test']:.2f} | "
            f"{r['gain_db_vs_best_classical']:+.2f} dB | {'yes' if r['beats_iterative'] else 'no'} |"
        )
    md.append(
        f"\nRanked by validation NMSE. ISTA and FISTA ran for the same number of iterations K with "
        f"lambda tuned on the validation split. LISTA: {args.seeds} seed(s), {args.epochs} epochs; "
        "'beats both' requires mean + std below the better classical solver on test."
    )
    (out / "leaderboard.md").write_text("\n".join(md) + "\n")
    print(f"[done] {time.time() - t0:.0f}s  -> {args.out}/leaderboard.md")
    return 0


def add_parser(sub) -> None:
    u = sub.add_parser(
        "unfold", help="deep unfolding (LISTA) for y = A x + noise with a known forward model A"
    )
    u.add_argument("--data", required=True, help=".npz with A (m x n), Y (N x m), X (N x n)")
    u.add_argument("--layers", default="4,8,16", help="depths K to try, comma separated")
    u.add_argument("--epochs", type=int, default=100)
    u.add_argument("--seeds", type=int, default=3)
    u.add_argument("--lr", type=float, default=1e-3)
    u.add_argument("--batch", type=int, default=128)
    u.add_argument(
        "--shared", action="store_true", help="share W between layers (fewer parameters)"
    )
    u.add_argument("--topk", type=int, default=3)
    u.add_argument("--seed", type=int, default=0)
    u.add_argument("--out", default="dlnet_out")
