"""Packager: write Top-K baseline packages, leaderboard and profile."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent


def _fmt(m: dict, task: str) -> str:
    if task == "reconstruction":
        return f"MSE {m['mse']:.4f} (PSNR {m['psnr']:.2f} dB)"
    if task == "regression":
        return f"{m['rmse']:.4f} (R2 {m['r2']:.3f})"
    extra = f", kappa {m['kappa']:.4f}" if "kappa" in m else ""
    if "miou" in m:
        return f"mIoU {m['miou']:.4f} (pixel acc {m['accuracy']:.4f}{extra})"
    return f"{m['accuracy']:.4f} (bal {m['balanced_accuracy']:.4f}{extra})"


def _short(t: dict, task: str) -> str:
    if task == "reconstruction":
        return f"mse {t['mse']}, psnr {t['psnr']} dB"
    if task == "regression":
        return f"rmse {t['rmse']}"
    k = f", kappa {t['kappa']}" if "kappa" in t else ""
    if "miou" in t:
        return f"mIoU {t['miou']}, pixel acc {t['accuracy']}{k}"
    return f"acc {t['accuracy']}, bal {t['balanced_accuracy']}{k}"


def _cost_line(cfg: dict) -> str:
    if cfg.get("flops") is None:
        return "not measured"
    return (
        f"{cfg['flops']:,} forward FLOPs per sample (matrix multiply and convolution only), about "
        f"{cfg['vram_mb']:.0f} MB training memory at batch {cfg['batch']} (estimate, not measured on a GPU)"
    )


def _hp_note(hp: dict) -> str:
    src = hp.get("source")
    if not src or src["estimated"] is None:
        return " (searched; not estimated)"
    e = src["estimated"]
    gns = (e.get("gradient_noise_scale") or {}).get("b_simple")
    gns = f", B_simple {gns:.0f}" if gns else ""
    return (
        f"\n  - Adopted: {src['adopted']}. Estimated lr {e['lr']:.2g}, batch {e['batch']}, "
        f"wd {e['wd']:.2g}{gns}; validation {src['val_searched']:.4f} (searched) vs "
        f"{'n/a' if src['val_estimated'] is None else format(src['val_estimated'], '.4f')} (estimated)"
    )


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


def mean_reference(per_fold: list[list[dict]]) -> list[dict]:
    """Average each reference model over the splits (one list of results per split)."""
    if len(per_fold) == 1:
        return per_fold[0]
    out = []
    for name in [r["name"] for r in per_fold[0]]:
        rs = [r for fold in per_fold for r in fold if r["name"] == name]
        m = {"name": name, "seconds": round(sum(r["seconds"] for r in rs), 1), "n_splits": len(rs)}
        for sp in ("va", "te"):
            m[sp] = {k: float(sum(r[sp][k] for r in rs) / len(rs)) for k in rs[0][sp]}
        out.append(m)
    out.sort(key=lambda r: -r["va"]["score"])
    return out


def hyperparameters(r: dict, recipe_epochs: int) -> dict:
    """The full training setting of one baseline, plus how it was obtained."""
    cfg = r["cfg"]
    hp = {
        "optimizer": "AdamW",
        "lr": cfg["lr"],
        "weight_decay": cfg["wd"],
        "batch_size": cfg["batch"],
        "epochs": recipe_epochs,
        "lr_schedule": "cosine",
        "grad_clip_norm": 5.0,
        "dropout": cfg["dropout"],
        **{
            k: cfg[k]
            for k in (
                "width",
                "depth",
                "rank",
                "rank_ratio",
                "patch",
                "encoder",
                "attention",
                "experts",
            )
            if k in cfg
        },
    }
    est = r.get("hparams")
    if est:
        hp["source"] = {
            "adopted": est["adopted"],
            "searched": est["searched"],
            "estimated": est["estimated"],
            "val_searched": est["val_searched"],
            "val_estimated": est["val_estimated"],
            "methods": [
                "LR range test (Smith 1506.01186)",
                "gradient noise scale B_simple (McCandlish 1812.06162)",
                "normalized weight decay (Loshchilov & Hutter 1711.05101)",
            ],
        }
    return hp


_OWN = ("leaderboard.md", "leaderboard.json", "profile.json", "reference")
CACHE_NAME = ".dlnet_cache.json"  # kept between runs (resume); see search.RunCache


def check_out_dir(out) -> bool:
    """Refuse a non-empty directory that is not a DLNet output. True if old output is present."""
    out = Path(out)
    files = [p for p in out.iterdir() if p.name != CACHE_NAME] if out.exists() else []
    if files and not (out / "leaderboard.json").exists():
        raise SystemExit(
            f"--out {out} is not empty and does not look like a DLNet output; "
            "pick an empty or new directory (nothing was deleted)"
        )
    return bool(files)


def _prepare_out_dir(out: Path) -> None:
    """Create `out`; replace only what a previous DLNet run wrote, never foreign files."""
    if check_out_dir(out):
        for p in [*(out / n for n in _OWN), *out.glob("top[0-9]*_*")]:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)


def export(out_dir, plugin, data, prof, chosen, reference, seed, epochs_recipe=100, extras=None):
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
        if cfg.get("spec"):  # a zoo architecture needs the zoo next to model.py
            for name, src in plugin.extra_dirs.items():
                shutil.copytree(src, d / name, ignore=shutil.ignore_patterns("__pycache__"))
        arch = {k: v for k, v in cfg.items() if k != "params"}
        hp = hyperparameters(r, epochs_recipe)
        config = {
            "plugin": plugin.name,
            "task": task,
            "split_seed": seed,
            "in_dim": data["in_dim"],
            "out_dim": data["out_dim"],
            "primary_metric": data["primary"],
            "arch": arch,
            "hyperparameters": hp,
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
                "splits": r.get("n_splits", 1),
                "test_std": r.get("test_std"),
            },
        }
        (d / "config.yaml").write_text(json.dumps(config, indent=2))
        (d / "train.py").write_text(plugin.train_template)
        (d / "tests" / "test_model.py").write_text(plugin.test_template)
        (d / "report.md").write_text(
            f"# top{i}: {cfg['family']}  {r['roles']}\n\n"
            f"- Config: `{arch}`\n- Origin: {cfg.get('origin', 'legacy family')}\n- Params: {cfg['params']:,}\n"
            f"- Cost model: {_cost_line(cfg)}\n"
            f"- Validation ({data['primary']}): {r['val_mean']:.4f} +/- {r['val_std']:.4f} "
            f"({r['n_seeds']} runs on {r.get('n_splits', 1)} split(s), {r['epochs']} epochs)\n"
            f"- Test (mean over runs): {_fmt(r['test'], task)}, "
            f"primary metric +/- {r.get('test_std', 0.0):.4f}\n"
            f"- Hyperparameters: lr {hp['lr']:.2g}, weight decay {hp['weight_decay']:.2g}, "
            f"batch {hp['batch_size']}, dropout {hp['dropout']}, AdamW, cosine, "
            f"{hp['epochs']} epochs (full recipe), clip {hp['grad_clip_norm']}"
            f"{_hp_note(hp)}\n"
            f"- Search time spent on this candidate: {r['train_seconds']}s\n"
            f"- Indistinguishable-group id: {r['group']} (same id = difference within run-to-run noise)\n\n"
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
                "hyperparameters": hp,
                "val_mean": round(r["val_mean"], 5),
                "val_std": round(r["val_std"], 5),
                "origin": cfg.get("origin"),
                "flops": cfg.get("flops"),
                "vram_mb": None if cfg.get("vram_mb") is None else round(cfg["vram_mb"], 1),
                "n_runs": r["n_seeds"],
                "n_splits": r.get("n_splits", 1),
                "test_std": round(r.get("test_std", 0.0), 5),
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
                **(extras or {}),
            },
            indent=2,
        )
    )
    md = [f"# Leaderboard ({task}, primary metric = {data['primary']}, higher is better)\n"]
    md.append(
        "| rank | model | roles | group | params | MFLOPs | MB | lr | wd | batch | val (mean+/-std) | test |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ts, h = _short(r["test"], task), r["hyperparameters"]
        md.append(
            f"| {r['rank']} | {r['name']} | {', '.join(r['roles'])} | {r['group']} | {r['params']:,} | "
            f"{'-' if r['flops'] is None else format(r['flops'] / 1e6, '.1f')} | "
            f"{'-' if r['vram_mb'] is None else format(r['vram_mb'], '.0f')} | {h['lr']:.2g} | {h['weight_decay']:.2g} | {h['batch_size']} | "
            f"{r['val_mean']} +/- {r['val_std']} | {ts} (+/- {r['test_std']}) |"
        )
    ns = max(r["n_splits"] for r in rows)
    md.append(
        f"\nEach number is a mean over runs (seeds); +/- is the std over those runs. "
        f"Finalists were ranked on {ns} split(s)"
        + (", so the std includes split-to-split noise." if ns > 1 else ".")
    )
    md.append("\n## Reference baselines (same split(s))\n")
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
    zc = (extras or {}).get("zero_cost_proxies")
    if zc:
        md.append("\n## Zero-cost proxies (information only, nothing was pruned)\n")
        md.append(f"Spearman rank correlation with the {zc['target']}:\n")
        md.append("| proxy | Spearman | candidates |\n|---|---|---|")
        md += [f"| {k} | {v['spearman']} | {v['n']} |" for k, v in zc["proxies"].items()]
        md.append(
            f"\nClearly better than the parameter count: {zc['better_than_param_count'] or 'none'}."
        )
    (out / "leaderboard.md").write_text("\n".join(md) + "\n")
    return v
