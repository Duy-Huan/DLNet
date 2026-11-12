"""Jupyter / Google Colab notebook that trains the Top-K baselines of one DLNet output folder.

    notebook.write(out_dir, source={"data": "images.npz"})       # -> out_dir/baselines.ipynb
    python -m dlnet notebook dlnet_out --data images.npz          # same, for an existing output

Sections of the notebook (every cell is plain Python, it also runs outside Jupyter):
  0. what the run found (leaderboard, the baselines and their block diagrams);
  1. setup: get the output folder on Colab (it is already there locally), install what is missing,
     pick the device (GPU on Colab);
  2. data: the same file, split and preprocessing as the search (built by the package's own code);
  3. one section per baseline: paper-style block diagram (architecture.svg), the layer graph from
     torchview, then full training with the package's recipe (warm-up + cosine, best validation
     checkpoint), learning curves and test metrics; model.pt / preprocess.json are saved;
  4. comparison of the baselines (table, curves, bar chart);
  5. the saved baselines loaded back one by one and used on the data (predictions, pictures).

Packages are self-contained (model.py is plain PyTorch); their helper modules share names
(model, core, inference, ...), so the notebook imports one package at a time (`use`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PLUGINS_ARRAYS = ("image_classification", "image_segmentation", "timeseries", "graph", "recsys")


def _md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}  # fmt: skip


SETUP = """
# 1. Setup ---------------------------------------------------------------------------------
# Local Jupyter: open this notebook inside the DLNet output folder; nothing to change here.
# Google Colab: the output folder (and the data) must be on the Colab machine first. Choose:
#   SOURCE = "git"  -> clone GIT_URL and go to GIT_SUBDIR (a private repo needs a token in the URL)
#   SOURCE = "zip"  -> upload a .zip of the output folder (with the data inside or next to it)
import os, subprocess, sys, zipfile
from pathlib import Path

SOURCE = {source!r}          # "here" | "git" | "zip"
GIT_URL = {git_url!r}
GIT_SUBDIR = {git_subdir!r}
IN_COLAB = "google.colab" in sys.modules

if SOURCE == "git" and not Path("leaderboard.json").exists():
    repo = Path("/content/repo" if IN_COLAB else "repo")
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", GIT_URL, str(repo)], check=True)
    os.chdir(repo / GIT_SUBDIR)
elif SOURCE == "zip" and not Path("leaderboard.json").exists():
    from google.colab import files  # Colab only
    for name in files.upload():
        zipfile.ZipFile(name).extractall(".")
    os.chdir(next(p.parent for p in Path(".").rglob("leaderboard.json")))
OUT = Path.cwd()
assert (OUT / "leaderboard.json").exists(), f"{{OUT}} is not a DLNet output folder (no leaderboard.json)"

# what the packages need; torch, numpy, pandas, scipy and matplotlib are already on Colab
needed = {packages}
missing = []
for mod, pip_name in needed.items():
    try:
        __import__(mod)
    except ImportError:
        missing.append(pip_name)
if missing and os.environ.get("DLNET_NO_INSTALL") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("output folder:", OUT)
print("torch", torch.__version__, "| device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
"""

HELPERS = """
# Helpers: every package has its own model.py / core.py / inference.py ..., so one package is
# imported at a time. `use(pkg)` returns its modules.
import importlib, json, time, types

SHARED = ("model", "core", "inference", "array_data", "hsi_data")
_ACTIVE = []

def use(pkg):
    pkg = (OUT / pkg).resolve()
    for name in SHARED:
        sys.modules.pop(name, None)
    while _ACTIVE:
        old = _ACTIVE.pop()
        if old in sys.path:
            sys.path.remove(old)
    sys.path.insert(0, str(pkg))
    _ACTIVE.append(str(pkg))
    mods = types.SimpleNamespace(dir=pkg, cfg=json.loads((pkg / "config.yaml").read_text()))
    for name in SHARED:
        if (pkg / f"{name}.py").exists():
            setattr(mods, name, importlib.import_module(name))
    return mods

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    from IPython.display import SVG, Markdown, display
except ImportError:  # plain Python (no IPython): say where the picture is
    def SVG(filename=None, data=None):
        return f"[picture: {filename}]"
    Markdown = str
    def display(x):
        print(x)
"""

OVERVIEW = """
# 0. What the search found
board = json.loads(Path("leaderboard.json").read_text())
BASELINES = [r["name"] for r in board["top_k"]]
PAPERS = [r["name"] for r in board.get("papers", [])]   # networks rebuilt from papers (dlnet paper build)
print("task:", board["task"], "| metric:", board["metric"])
for r in board["top_k"]:
    print(f"  {{r['name']:<45}} params {{r['params']:>10,}}   val {{r['val_mean']:.4f}} +- {{r['val_std']:.4f}}"
          f"   (search: {{r['epochs']}} epochs)")
for r in board.get("papers", []):
    print(f"  {{r['name']:<45}} params {{(r['params'] or 0):>10,}}   paper baseline, fidelity level {{r['level']}}")
print("verdict:", board.get("verdict"))
"""

DATA = {
    "hsi": """
# 2. Data: the same cube, split, PCA and patches as the search (the package's hsi_data.py)
DATA, GT = {data!r}, {gt!r}
CUBE_KEY, GT_KEY = {cube_key!r}, {gt_key!r}
p = use(BASELINES[0])
dc = p.cfg["data"]
cube, gt = p.hsi_data.load(DATA, GT, CUBE_KEY, GT_KEY)
data = p.hsi_data.build(cube, gt, split=dc["split"], seed=p.cfg["split_seed"], patch=dc["patch"],
                        pca=dc["pca"], frac=dc["frac"], block=dc["block"])
print("cube", cube.shape, "| classes", len(np.unique(gt[gt > 0])),
      "| train / val / test:", len(data["ytr"]), len(data["yva"]), len(data["yte"]))
""",
    "tabular": """
# 2. Data: the same columns, split and scaling as the search (the package's core.py)
DATA, TARGET = {data!r}, {target!r}
p = use(BASELINES[0])
X, y, cols = p.core.load_csv(DATA, TARGET)
data = p.core.prepare(X, y, p.cfg["task"], seed=p.cfg["split_seed"])
data["preprocess"].update(columns=cols, target=TARGET)
print("rows", len(y), "| features", data["in_dim"], "| train / val / test:", len(data["ytr"]), len(data["yva"]), len(data["yte"]))
""",
    "arrays": """
# 2. Data: the same arrays, split and normalisation as the search (the package's array_data.py)
DATA = {data!r}
p = use(BASELINES[0])
X, y = p.array_data.load(DATA, p.cfg["data"].get("x_key"), p.cfg["data"].get("y_key"))
data = p.array_data.build_any(X, y, p.cfg["data"]["kind"], p.cfg["task"], seed=p.cfg["split_seed"])
print("X", X.shape, "| y", y.shape, "| train / val / test:", len(data["ytr"]), len(data["yva"]), len(data["yte"]))
if plt is not None and p.cfg["data"]["kind"] in ("image", "segmentation"):
    fig, ax = plt.subplots(1, 8, figsize=(14, 2))
    for i, a in enumerate(ax):
        img = X[i] if X.ndim == 3 or X.shape[-1] in (1, 3) else X[i, ..., :3]
        a.imshow(img.squeeze(), cmap="gray" if img.ndim == 2 or img.shape[-1] == 1 else None)
        a.set_title(str(y[i]) if y.ndim == 1 else "", fontsize=8); a.axis("off")
    plt.show()
""",
    "image_reconstruction": """
# 2. Data: the same images, split and normalisation as the search (the package's array_data.py)
DATA = {data!r}
p = use(BASELINES[0])
data = p.array_data.build_reconstruction(p.array_data.load_x(DATA, p.cfg["data"].get("x_key")),
                                         seed=p.cfg["split_seed"])
print("train / val / test:", len(data["Xtr"]), len(data["Xva"]), len(data["Xte"]))
""",
}

TRAIN_HELPERS = '''
# Training of one baseline: the package's own recipe (core.train_full: AdamW, linear warm-up then
# cosine, gradient clipping, best-validation checkpoint), on DEVICE (GPU on Colab).
EPOCHS = None   # None = the recipe of each package (config.yaml -> recipe.epochs); e.g. 30 for a quick run
RESULTS = {}

def show_architecture(name):
    """Paper-style block diagram (architecture.svg, drawn by DLNet from the spec)."""
    svg = OUT / name / "architecture.svg"
    if svg.exists():
        display(SVG(filename=str(svg)))

def show_layers(name, depth=2):
    """Layer-by-layer graph drawn by torchview (needs the graphviz program; Colab has it)."""
    p = use(name)
    model = p.model.build_model()
    kind, shape = p.cfg["arch"]["input"]["kind"], p.cfg["arch"]["input"]["dims"]
    try:
        from torchview import draw_graph
        if kind == "ids":
            x = torch.stack([torch.randint(0, c, (1,)) for c in shape], 1)
            g = draw_graph(model, input_data=x, depth=depth, expand_nested=True, graph_name=name)
        else:
            g = draw_graph(model, input_size=(1, *shape), depth=depth, expand_nested=True, graph_name=name)
        g.visual_graph.render(str(OUT / name / "layers"), format="svg", cleanup=True)
        display(SVG(filename=str(OUT / name / "layers.svg")))
    except Exception as e:  # torchview or graphviz missing, or a block it cannot trace
        print(f"torchview graph not drawn ({type(e).__name__}: {e}); the module tree instead:")
        print(model)

def train(name, epochs=None, seed=0):
    p = use(name)
    arch, recipe = p.cfg["arch"], p.cfg["recipe"]
    epochs = epochs or EPOCHS or recipe["epochs"]
    torch.manual_seed(seed)
    model = p.model.build_model(arch, data["in_dim"], data["out_dim"])
    n = sum(q.numel() for q in model.parameters())
    print(f"{name}: {n:,} parameters, {epochs} epochs, lr {arch['lr']}, wd {arch['wd']}, batch {arch['batch']}")
    t0 = time.time()
    tr, best_val = p.core.train_full(model, data, arch["lr"], arch["wd"], arch["batch"], epochs,
                                     seed=seed, device=DEVICE, recipe=recipe.get("name"))
    secs = time.time() - t0
    val, test = tr.evaluate("va"), tr.evaluate("te")
    p.core.save_artifacts(model.cpu(), data, p.dir)  # model.pt + preprocess.json next to model.py
    RESULTS[name] = {"params": n, "epochs": epochs, "seconds": round(secs, 1), "val": val, "test": test,
                     "history": getattr(tr, "history", [])}
    print(f"  {secs:.0f}s | val {val} | test {test}")
    plot_history(name)
    return RESULTS[name]

def plot_history(name):
    h = RESULTS.get(name, {}).get("history")
    if not h or plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h]); ax[0].set_title("training loss")
    ax[1].plot([r["epoch"] for r in h], [r["val"] for r in h], color="tab:orange")
    ax[1].set_title(f"validation {board['metric']}")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.suptitle(name); plt.tight_layout(); plt.show()
'''

BASELINE = """
NAME = {name!r}
show_architecture(NAME)
show_layers(NAME)
"""

TRAIN_ONE = """
train(NAME)
"""

COMPARE = """
# 4. Comparison of the baselines (this notebook's training, same data and split)
metric = board["metric"]
rows = [(n, r["params"], r["epochs"], r["seconds"], r["val"].get("score"), r["test"].get("score"))
        for n, r in RESULTS.items()]
print(f"{'baseline':<45}{'params':>11}{'epochs':>8}{'sec':>8}{'val':>9}{'test':>9}")
for n, pa, ep, sec, v, t in sorted(rows, key=lambda r: -(r[4] or 0)):
    print(f"{n:<45}{pa:>11,}{ep:>8}{sec:>8}{v:>9.4f}{t:>9.4f}")
print("(score = the primary metric:", metric, "; regression scores are minus the RMSE)")
if plt is not None and RESULTS:
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    for n, r in RESULTS.items():
        if r["history"]:
            ax[0].plot([h["epoch"] for h in r["history"]], [h["val"] for h in r["history"]], label=n)
    ax[0].set_title(f"validation {metric} per epoch"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    names = list(RESULTS)
    ax[1].barh(names, [RESULTS[n]["test"]["score"] for n in names], color="tab:green")
    ax[1].set_title(f"test {metric}"); ax[1].grid(alpha=0.3, axis="x")
    plt.tight_layout(); plt.show()
"""

RELOAD = """
# 5. Load the saved baselines back, one by one, and use them (model.pt + preprocess.json)
def evaluate_saved(name):
    p = use(name)
    if not (p.dir / "model.pt").exists():
        print(f"{name}: no model.pt yet (train it in section 3)"); return None
    model, meta, cfg = p.inference.load(p.dir, "cpu")
    a = cfg["arch"]
    test = p.core.Trainer(model, data, lr=a["lr"], wd=a["wd"], batch=a["batch"]).evaluate("te")
    print(f"{name}: test {test}")
    return p, model.cpu().eval(), meta, cfg

for NAME in BASELINES + PAPERS:
    loaded = evaluate_saved(NAME)
"""

SHOW = {
    "hsi": """
# Class map of the whole scene from the best saved baseline (predict.py does the same)
best = max(RESULTS, key=lambda n: RESULTS[n]["val"]["score"]) if RESULTS else BASELINES[0]
p, model, meta, cfg = evaluate_saved(best)
cmap = p.inference.predict_file(model, meta, cfg, DATA)
if plt is not None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(np.where(gt > 0, gt, np.nan), cmap="tab20"); ax[0].set_title("ground truth")
    ax[1].imshow(cmap, cmap="tab20"); ax[1].set_title(f"{best}: every pixel")
    ax[2].imshow(np.where(gt > 0, cmap, np.nan), cmap="tab20"); ax[2].set_title("labelled pixels")
    for a in ax: a.axis("off")
    plt.show()
""",
    "segmentation": """
# Masks predicted on test images by the best saved baseline
best = max(RESULTS, key=lambda n: RESULTS[n]["val"]["score"]) if RESULTS else BASELINES[0]
p, model, meta, cfg = evaluate_saved(best)
with torch.no_grad():
    pred = model(data["Xte"][:4]).argmax(1).numpy()
if plt is not None:
    fig, ax = plt.subplots(3, 4, figsize=(12, 8))
    for i in range(4):
        img = data["Xte"][i].numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (np.ptp(img) + 1e-9)
        ax[0, i].imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
        ax[1, i].imshow(np.where(data["yte"][i].numpy() < 0, np.nan, data["yte"][i].numpy()), cmap="tab20")
        ax[2, i].imshow(pred[i], cmap="tab20")
    for a, t in zip(ax[:, 0], ("image", "truth", best)):
        a.set_ylabel(t)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    plt.show()
""",
    "classification": """
# Confusion matrix on the test split of the best saved baseline
best = max(RESULTS, key=lambda n: RESULTS[n]["val"]["score"]) if RESULTS else BASELINES[0]
p, model, meta, cfg = evaluate_saved(best)
with torch.no_grad():
    out = torch.cat([model(data["Xte"][i:i + 512]) for i in range(0, len(data["Xte"]), 512)])
pred, true = out.argmax(1).numpy(), data["yte"].numpy()
k = int(max(pred.max(), true.max())) + 1
cm = np.zeros((k, k), int); np.add.at(cm, (true, pred), 1)
print("test accuracy", (pred == true).mean())
if plt is not None:
    plt.figure(figsize=(6, 5)); plt.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues")
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"{best}: confusion (row-normalised)")
    plt.colorbar(); plt.show()
""",
    "regression": """
# Predicted against true values on the test split of the best saved baseline
best = max(RESULTS, key=lambda n: RESULTS[n]["val"]["score"]) if RESULTS else BASELINES[0]
p, model, meta, cfg = evaluate_saved(best)
with torch.no_grad():
    pred = model(data["Xte"]).squeeze(-1).numpy() * meta["y_std"] + meta["y_mean"]
true = data["y_raw"]["te"] if "y_raw" in data else data["yte"].numpy()
if plt is not None:
    plt.figure(figsize=(5, 5)); plt.scatter(true, pred, s=6, alpha=0.6)
    plt.xlabel("true"); plt.ylabel("predicted"); plt.title(best); plt.show()
""",
}

INTRO = """
# DLNet baselines: train, compare and reuse

This notebook belongs to the DLNet output folder it sits in ({n} baselines for a **{task}** task,
metric **{metric}**). It runs on a laptop (Jupyter) or on **Google Colab** (GPU: *Runtime > Change
runtime type > GPU*).

Every baseline is a self-contained package `top<k>_<family>/`: `model.py` is the network as plain
PyTorch (no DLNet import), `config.yaml` its hyper-parameters, `architecture.svg` its block diagram.

1. **Setup**: get the folder onto Colab, install what is missing, choose the device.
2. **Data**: the same file, split and preprocessing as the search.
3. **One section per baseline**: block diagram, layer graph (torchview), full training, curves.
4. **Comparison** of the baselines.
5. **Reuse**: the saved weights loaded back and used on the data.

The search ranked the baselines on a short budget; full training can reorder neighbours.
"""


def _packages(plugin: str) -> dict:
    need = {"torchview": "torchview", "graphviz": "graphviz", "matplotlib": "matplotlib"}
    if plugin == "hsi":
        need["scipy"] = "scipy"
    if plugin == "tabular":
        need["pandas"] = "pandas"
    return need


def cells(out_dir, source: dict, git_url: str = "", git_subdir: str = "") -> list[dict]:
    out = Path(out_dir)
    board = json.loads((out / "leaderboard.json").read_text())
    names = [r["name"] for r in board["top_k"]]
    cfg = json.loads((out / names[0] / "config.yaml").read_text())
    plugin = cfg["plugin"]
    kind = "hsi" if plugin == "hsi" else "tabular" if plugin == "tabular" else (
        "image_reconstruction" if plugin == "image_reconstruction" else "arrays")  # fmt: skip
    rel = {k: (os.path.relpath(v, out) if v and Path(v).exists() else v) for k, v in source.items()}
    c = [
        _md(INTRO.format(n=len(names), task=board["task"], metric=board["metric"])),
        _code(
            SETUP.format(
                source="git" if git_url else "here",
                git_url=git_url,
                git_subdir=git_subdir,
                packages=_packages(plugin),
            )
        ),
        _code(HELPERS),
        _code(OVERVIEW.format()),
        _md("## 2. Data\n\nEdit the path(s) if the data is somewhere else on this machine."),
        _code(
            DATA[kind].format(
                data=rel.get("data", ""),
                gt=rel.get("gt"),
                target=rel.get("target"),
                cube_key=rel.get("cube_key"),
                gt_key=rel.get("gt_key"),
            )
        ),
        _md("## 3. The baselines\n\nTraining helpers first; then one section per baseline."),
        _code(TRAIN_HELPERS),
    ]
    for r in board["top_k"]:
        n = r["name"]
        c.append(_md(f"### {n}\n\n{r['params']:,} parameters; search: validation "
                     f"{r['val_mean']:.4f} +- {r['val_std']:.4f} after {r['epochs']} epochs "
                     f"(group {r['group']}: same group = indistinguishable within the noise). "
                     f"Block diagram (DLNet) and layer graph (torchview), then full training."))  # fmt: skip
        c.append(_code(BASELINE.format(name=n)))
        c.append(_code(TRAIN_ONE))
    for r in board.get("papers", []):
        rep = "; ".join(
            f"{x.get('dataset', '?')}: {x['metric']} {x['value']}"
            for x in r["reported"]
            if "metric" in x
        )
        c.append(_md(f"### {r['name']} (from the paper: {r['paper']})\n\nFidelity level **{r['level']}** "
                     f"(A = the authors' class, B = verified against the authors' code, C = written from the "
                     f"paper, not verified). Reported by the paper: {rep or 'not given in the card'}. "
                     f"Trained here on the same data and split as the baselines above."))  # fmt: skip
        c.append(_code(BASELINE.format(name=r["name"])))
        c.append(_code(TRAIN_ONE))
    c += [
        _md("## 4. Comparison"),
        _code(COMPARE),
        _md("## 5. Reuse the saved baselines"),
        _code(RELOAD),
    ]
    show = "hsi" if plugin == "hsi" else (
        "segmentation" if plugin == "image_segmentation" else
        "regression" if board["task"] == "regression" else
        "classification" if board["task"] == "classification" and plugin != "image_reconstruction" else None
    )  # fmt: skip
    if show:
        c.append(_code(SHOW[show]))
    return c


def write(out_dir, source: dict, git_url: str = "", git_subdir: str = "", log=print) -> Path:
    """Write <out>/baselines.ipynb and <pkg>/architecture.svg for every exported baseline."""
    from . import diagram

    out = Path(out_dir)
    board = json.loads((out / "leaderboard.json").read_text())
    for r in board["top_k"]:
        pkg = out / r["name"]
        arch = json.loads((pkg / "config.yaml").read_text())["arch"]
        if arch.get("spec"):
            arch = {**arch, "params": r.get("params")}
            diagram.write(arch, pkg / "architecture.svg", f"{r['name']}  (rank {r['rank']})")
            rep = pkg / "report.md"
            if rep.exists() and "architecture.svg" not in rep.read_text():
                rep.write_text(rep.read_text() + "\nBlock diagram: `architecture.svg`. Training notebook "
                               "(Jupyter / Colab): `../baselines.ipynb`.\n")  # fmt: skip
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                     "language_info": {"name": "python"}, "accelerator": "GPU",
                     "colab": {"provenance": []}},
        "cells": cells(out, source, git_url, git_subdir),
    }  # fmt: skip
    board["notebook"] = {"source": {k: (str(v) if v is not None else None) for k, v in source.items()},
                         "git_url": git_url, "git_subdir": git_subdir}  # fmt: skip
    (out / "leaderboard.json").write_text(json.dumps(board, indent=2))
    path = out / "baselines.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"[notebook] {path} and architecture.svg in {len(board['top_k'])} package(s)")
    return path


def script(nb_path) -> str:
    """The code cells of a notebook as one Python script (tests, or running without Jupyter)."""
    nb = json.loads(Path(nb_path).read_text())
    return "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


SUPPORT = ("core.py", "inference.py", "array_data.py", "hsi_data.py")


def refresh_support(out_dir, log=print) -> int:
    """Copy the current DLNet versions of the packages' helper modules (training loop, data,
    inference) into an older output folder; model.py, config.yaml and weights are not touched."""
    import shutil

    here, n = Path(__file__).parent, 0
    for pkg in Path(out_dir).glob("top*_*"):
        for name in SUPPORT:
            dst = pkg / name
            if dst.exists() and dst.read_bytes() != (here / name).read_bytes():
                shutil.copyfile(here / name, dst)
                n += 1
    if n:
        log(
            f"[notebook] refreshed {n} helper module(s) (core.py, inference.py, ...) in the packages"
        )
    return n
