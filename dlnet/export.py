"""Packager: write Top-K baseline packages, leaderboard and profile."""

from __future__ import annotations

import shutil
from pathlib import Path

from . import jsonsafe, standalone

HERE = Path(__file__).parent

PREDICT_PY = '''"""Predict with this baseline after `python train.py` (which writes model.pt, preprocess.json).

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
'''


def core_recipe() -> str:
    from . import core

    return core.RECIPE


def core_recipe_detail() -> dict:
    from . import core

    return dict(core.RECIPES[core.RECIPE])


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


def verdict(top1: dict, reference: list[dict], n_val: int, task: str, primary: str = "accuracy",
            n_test: int | None = None) -> dict:  # fmt: skip
    """Acceptance rule (conservative).

    top1 must beat the best reference (picked on validation) on BOTH
      - validation, by more than its noise: the run-to-run std, floored by the sampling error of
        the validation set (search.sampling_noise; also for regression, audit R2), and
      - the held-out test split, against EVERY reference model (not only the validation winner),
        by more than the sampling error of the test set.
    Comparing on validation alone is optimistic: top1 is the best of many candidates.
    """
    from .search import sampling_noise

    best_ref = reference[0]
    margin = top1["val_mean"] - best_ref["va"]["score"]
    noise = max(top1["val_std"], sampling_noise(top1["val_mean"], primary, n_val))
    test_margin = top1["test"]["score"] - max(
        r["te"]["score"] for r in reference
    )  # vs every reference
    test_noise = sampling_noise(top1["test"]["score"], primary, n_test or n_val)
    ok = margin > noise and test_margin > test_noise
    return {
        "beats_reference": bool(ok),
        "val_margin": round(margin, 5),
        "val_noise": round(noise, 5),
        "test_margin": round(test_margin, 5),
        "test_noise": round(test_noise, 5),
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
        if "code" in rs[0]:
            m["code"] = rs[0]["code"]
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


_OWN = ("leaderboard.md", "leaderboard.json", "profile.json", "reference", "results.tex")
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
    fair = plugin.reference_for(chosen[0]["cfg"], reference)  # e.g. PCA with the same code size
    v = verdict(chosen[0], fair, len(data["yva"]), task, data["primary"], len(data["yte"]))
    if fair is not reference:
        v["reference_matched_to"] = [r["name"] for r in fair]

    (out / "profile.json").write_text(jsonsafe.dumps(prof, indent=2))
    (out / "reference").mkdir()
    (out / "reference" / "results.json").write_text(jsonsafe.dumps(reference, indent=2))

    rows = []
    for i, r in enumerate(chosen, 1):
        cfg = r["cfg"]
        d = out / f"top{i}_{cfg['family']}"
        (d / "tests").mkdir(parents=True)
        shutil.copy(HERE / "core.py", d / "core.py")
        shutil.copy(HERE / "inference.py", d / "inference.py")
        for name, src in plugin.extra_files.items():
            shutil.copy(src, d / name)
        # model.py: the network itself as plain PyTorch source (dlnet/standalone.py), checked against
        # the searched model. Only if that is not possible: the spec interpreter + a copy of the zoo.
        model_note = "`model.py` is plain PyTorch (no DLNet import): the network's own source"
        try:
            if not cfg.get("spec"):
                raise standalone.Unsupported("no zoo spec")
            how = standalone.write_package(
                d, {k: v for k, v in cfg.items() if k != "params"}, data["out_dim"],
                f"Network `{cfg['family']}` (top{i})",
            )  # fmt: skip
            model_note += f", {how} to the searched network"
        except standalone.Unsupported as e:
            shutil.copy(plugin.model_file, d / "model.py")
            if cfg.get("spec"):
                for name, src in plugin.extra_dirs.items():
                    shutil.copytree(src, d / name, ignore=shutil.ignore_patterns("__pycache__"))
            model_note = (
                f"`model.py` builds the network from config.yaml with the bundled `zoo/` ({e})"
            )
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
                "schedule": "linear warm-up (5 %) then cosine",
                "select": "best-val checkpoint",
                "name": core_recipe(),
                **core_recipe_detail(),
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
        (d / "config.yaml").write_text(jsonsafe.dumps(config, indent=2))
        (d / "train.py").write_text(plugin.train_template)
        (d / "predict.py").write_text(PREDICT_PY)
        (d / "tests" / "test_model.py").write_text(plugin.test_template)
        (d / "report.md").write_text(
            f"# top{i}: {cfg['family']}  {r['roles']}\n\n"
            f"- Config: `{arch}`\n- Origin: {cfg.get('origin', 'legacy family')}\n- Params: {cfg['params']:,}\n"
            f"- Cost model: {_cost_line(cfg)}\n"
            f"- Validation ({data['primary']}): {r['val_mean']:.4f} +/- {r.get('val_noise', r['val_std']):.4f} "
            f"({r['n_seeds']} runs on {r.get('n_splits', 1)} split(s), {r['epochs']} epochs)\n"
            f"- Test (mean over runs): {_fmt(r['test'], task)}, "
            f"primary metric +/- {r.get('test_std', 0.0):.4f}\n"
            f"- Hyperparameters: lr {hp['lr']:.2g}, weight decay {hp['weight_decay']:.2g}, "
            f"batch {hp['batch_size']}, dropout {hp['dropout']}, AdamW, cosine, "
            f"{hp['epochs']} epochs (full recipe), clip {hp['grad_clip_norm']}"
            f"{_hp_note(hp)}\n"
            f"- Model file: {model_note}\n"
            f"- Search time spent on this candidate: {r['train_seconds']}s\n"
            f"- Indistinguishable-group id: {r['group']} (same id = difference within run-to-run noise)\n\n"
            f"Short search budget only. Re-train with the full recipe (see `python train.py -h`); "
            f"neighbouring ranks may reorder.\n\n"
            f"Use: `python train.py --data ...` writes model.pt and preprocess.json; then "
            f"`python predict.py --data new_file` predicts in the original label units "
            f"(`--export model.pt2` saves a deployable graph). Training recipe: {core_recipe()} "
            f"{core_recipe_detail()}.\n"
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
                "val_noise": round(r.get("val_noise", r["val_std"]), 5),
                "noise_source": r.get("noise_source", "runs"),
                "epochs": r["epochs"],
                "origin": cfg.get("origin"),
                "flops": cfg.get("flops"),
                "vram_mb": None if cfg.get("vram_mb") is None else round(cfg["vram_mb"], 3),
                "n_runs": r["n_seeds"],
                "n_splits": r.get("n_splits", 1),
                "test_std": round(r.get("test_std", 0.0), 5),
                "test": {k: round(x, 5) for k, x in r["test"].items()},
                "runs": {k: [round(x, 5) for x in v] for k, v in (r.get("runs") or {}).items()},
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
        jsonsafe.dumps(
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
    warns = (extras or {}).get("warnings") or []
    if warns:
        md.append("**Warnings**\n")
        md += [f"- {w}" for w in warns]
        md.append("")
    md.append(
        "| rank | model | roles | group | epochs | runs | params | MFLOPs | MB | lr | wd | batch | val (mean+/-noise) | test |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ts, h = _short(r["test"], task), r["hyperparameters"]
        md.append(
            f"| {r['rank']} | {r['name']} | {', '.join(r['roles'])} | {r['group']} | {r['epochs']} | "
            f"{r['n_runs']} | {r['params']:,} | "
            f"{'-' if r['flops'] is None else format(r['flops'] / 1e6, '.1f')} | "
            f"{'-' if r['vram_mb'] is None else format(r['vram_mb'], '.0f')} | {h['lr']:.2g} | {h['weight_decay']:.2g} | {h['batch_size']} | "
            f"{r['val_mean']} +/- {r['val_noise']} | {ts} (+/- {r['test_std']}) |"
        )
    ns = max(r["n_splits"] for r in rows)
    md.append(
        f"\nEach number is a mean over `runs` (seeds); +/- is the larger of the std over those runs "
        f"and the sampling error of the validation set (the only noise estimate when runs = 1). "
        f"Models are ranked within the same `epochs` (more epochs first). "
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
            f"(> noise {v['val_noise']}) and test margin {v['test_margin']} (> noise {v['test_noise']})."
        )
    else:
        md.append(
            f"**No significant win over the best reference ({v['best_reference']}).** "
            f"Val margin {v['val_margin']} vs noise {v['val_noise']}; test margin {v['test_margin']} "
            f"vs noise {v['test_noise']}. "
            "Consider the reference baseline first; the deep-learning candidates below are still valid to try."
        )
    ob = (extras or {}).get("objective")
    if ob:
        md.append(f"\n## Objective: {ob['objective']}\n")
        if ob["objective"] != "accuracy":
            md.append(
                "Top-K is ordered by this objective, not by validation alone (see doc/user-guide.md).\n"
            )
        md.append("Pareto front of (validation, parameters) among the ranked finalists:\n")
        md.append("| family | val | params | FLOPs |\n|---|---|---|---|")
        for r in ob["pareto_front"]:
            fl = "-" if r["flops"] is None else f"{r['flops']:,}"
            md.append(f"| {r['family']} | {r['val_mean']} | {r['params']:,} | {fl} |")
    zc = (extras or {}).get("zero_cost_proxies")
    if zc:
        md.append("\n## Zero-cost proxies (information only, nothing was pruned)\n")
        md.append(f"Spearman rank correlation with the {zc['target']}:\n")
        md.append("| proxy | Spearman | candidates |\n|---|---|---|")
        md += [f"| {k} | {v['spearman']} | {v['n']} |" for k, v in zc["proxies"].items()]
        md.append(
            f"\nClearly better than the parameter count: {zc['better_than_param_count'] or 'none'}."
        )
    bu = (extras or {}).get("budget")
    if bu:
        unit = "seconds" if bu["mode"] == "time" else "training epochs"
        md.append(f"\n## Budget: {bu['budget']} ({unit})\n")
        md.append(f"Wall clock {bu['wall_seconds']} s, {bu['epochs_trained']} epochs trained, "
                  f"overrun {bu['overrun_pct']}%.\n")  # fmt: skip
        md.append("| stage | seconds | epochs |\n|---|---|---|")
        md += [f"| {d['stage']} | {d['seconds']} | {d['epochs']:.0f} |" for d in bu["stages"]]
    (out / "leaderboard.md").write_text("\n".join(md) + "\n")
    return v
