"""Packager: write Top-K baseline packages, leaderboard and profile."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent


def _fmt(m: dict, task: str) -> str:
    if task == "regression":
        return f"{m['rmse']:.4f} (R2 {m['r2']:.3f})"
    extra = f", kappa {m['kappa']:.4f}" if "kappa" in m else ""
    return f"{m['accuracy']:.4f} (bal {m['balanced_accuracy']:.4f}{extra})"


def _short(t: dict, task: str) -> str:
    if task == "regression":
        return f"rmse {t['rmse']}"
    k = f", kappa {t['kappa']}" if "kappa" in t else ""
    return f"acc {t['accuracy']}, bal {t['balanced_accuracy']}{k}"


def verdict(top1: dict, reference: list[dict], n_val: int, task: str) -> dict:
    """Acceptance rule (conservative).

    top1 must beat the best reference (picked on validation) on BOTH
      - validation, by more than its seed std (floored by the binomial noise for tiny
        classification validation sets), and
      - the held-out test split, against EVERY reference model (not only the validation winner).
    Comparing on validation alone is optimistic: top1 is the best of many candidates.
    """
    best_ref = reference[0]
    margin = top1["val_mean"] - best_ref["va"]["score"]
    noise = top1["val_std"]
    if task == "classification":
        p = min(max(top1["val_mean"], 0.01), 0.99)
        noise = max(noise, (p * (1 - p) / n_val) ** 0.5)
    test_margin = top1["test"]["score"] - max(
        r["te"]["score"] for r in reference
    )  # vs every reference
    ok = margin > noise and test_margin > 0
    return {
        "beats_reference": bool(ok),
        "val_margin": round(margin, 5),
        "val_noise": round(noise, 5),
        "test_margin": round(test_margin, 5),
        "top1_val": round(top1["val_mean"], 5),
        "best_reference": best_ref["name"],
        "best_reference_val": round(best_ref["va"]["score"], 5),
    }


_OWN = ("leaderboard.md", "leaderboard.json", "profile.json", "reference")


def _prepare_out_dir(out: Path) -> None:
    """Create `out`; replace only what a previous DLNet run wrote, never foreign files."""
    if out.exists() and any(out.iterdir()):
        if not (out / "leaderboard.json").exists():
            raise SystemExit(
                f"--out {out} is not empty and does not look like a DLNet output; "
                "pick an empty or new directory (nothing was deleted)"
            )
        for p in [*(out / n for n in _OWN), *out.glob("top[0-9]*_*")]:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)


def export(out_dir, plugin, data, prof, chosen, reference, seed, epochs_recipe=100):
    out = Path(out_dir)
    _prepare_out_dir(out)
    task = data["task"]
    v = verdict(chosen[0], reference, len(data["yva"]), task)

    (out / "profile.json").write_text(json.dumps(prof, indent=2))
    (out / "reference").mkdir()
    (out / "reference" / "results.json").write_text(json.dumps(reference, indent=2))

    rows = []
    for i, r in enumerate(chosen, 1):
        cfg = r["cfg"]
        d = out / f"top{i}_{cfg['family']}"
        (d / "tests").mkdir(parents=True)
        shutil.copy(plugin.model_file, d / "model.py")
        shutil.copy(HERE / "core.py", d / "core.py")
        for name, src in plugin.extra_files.items():
            shutil.copy(src, d / name)
        arch = {k: v for k, v in cfg.items() if k != "params"}
        config = {
            "plugin": plugin.name,
            "task": task,
            "split_seed": seed,
            "in_dim": data["in_dim"],
            "out_dim": data["out_dim"],
            "primary_metric": data["primary"],
            "arch": arch,
            "data": plugin.data_config(data),
            "recipe": {
                "epochs": epochs_recipe,
                "schedule": "cosine",
                "select": "best-val checkpoint",
            },
            "search_result": {
                "val_mean": r["val_mean"],
                "val_std": r["val_std"],
                "epochs_used": r["epochs"],
                "seeds": r["n_seeds"],
            },
        }
        (d / "config.yaml").write_text(json.dumps(config, indent=2))
        (d / "train.py").write_text(plugin.train_template)
        (d / "tests" / "test_model.py").write_text(plugin.test_template)
        (d / "report.md").write_text(
            f"# top{i}: {cfg['family']}  {r['roles']}\n\n"
            f"- Config: `{arch}`\n- Params: {cfg['params']:,}\n"
            f"- Validation ({data['primary']}): {r['val_mean']:.4f} +/- {r['val_std']:.4f} "
            f"({r['n_seeds']} seeds, {r['epochs']} epochs)\n"
            f"- Test (mean over seeds): {_fmt(r['test'], task)}\n"
            f"- Search time spent on this candidate: {r['train_seconds']}s\n"
            f"- Indistinguishable-group id: {r['group']} (same id = difference within seed noise)\n\n"
            f"Short search budget only. Re-train with the full recipe (see `python train.py -h`); "
            f"neighbouring ranks may reorder.\n"
        )
        rows.append(
            {
                "rank": i,
                "name": d.name,
                "roles": r["roles"],
                "group": r["group"],
                "params": cfg["params"],
                "val_mean": round(r["val_mean"], 5),
                "val_std": round(r["val_std"], 5),
                "test": {k: round(x, 5) for k, x in r["test"].items()},
            }
        )

    ref_rows = [
        {
            "name": x["name"],
            "val": round(x["va"]["score"], 5),
            "test": {k: round(a, 5) for k, a in x["te"].items()},
        }
        for x in reference
    ]
    (out / "leaderboard.json").write_text(
        json.dumps(
            {
                "task": task,
                "metric": data["primary"],
                "verdict": v,
                "top_k": rows,
                "reference": ref_rows,
            },
            indent=2,
        )
    )
    md = [f"# Leaderboard ({task}, primary metric = {data['primary']}, higher is better)\n"]
    md.append("| rank | model | roles | group | params | val (mean+/-std) | test |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        ts = _short(r["test"], task)
        md.append(
            f"| {r['rank']} | {r['name']} | {', '.join(r['roles'])} | {r['group']} | {r['params']:,} | "
            f"{r['val_mean']} +/- {r['val_std']} | {ts} |"
        )
    md.append("\n## Reference baselines (same split)\n")
    md.append("| model | val | test |\n|---|---|---|")
    for r in ref_rows:
        ts = _short(r["test"], task)
        md.append(f"| {r['name']} | {r['val']} | {ts} |")
    md.append("\n## Verdict\n")
    if v["beats_reference"]:
        md.append(
            f"top1 beats the best reference ({v['best_reference']}): val margin {v['val_margin']} "
            f"(> noise {v['val_noise']}) and test margin {v['test_margin']}."
        )
    else:
        md.append(
            f"**No significant win over the best reference ({v['best_reference']}).** "
            f"Val margin {v['val_margin']} vs noise {v['val_noise']}; test margin {v['test_margin']}. "
            "Consider the reference baseline first; the deep-learning candidates below are still valid to try."
        )
    (out / "leaderboard.md").write_text("\n".join(md) + "\n")
    return v
