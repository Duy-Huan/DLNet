"""Paper cards: a published network written down once (YAML), rebuilt as a DLNet baseline.

    python -m dlnet paper check  papers/resnet20.yaml
    python -m dlnet paper build  papers/resnet20.yaml --out pkgs            # or --into dlnet_out
    python -m dlnet paper verify papers/ssftt.yaml --official-root /path/to/the/authors/repo

A card has four parts (doc/user-guide.md, "Mạng từ paper"):

    paper:        name, title, venue, url, code, reported: [{dataset, metric, value, protocol}, {params: N}]
    official:     repo, commit, entry "file.py:Class", init_args  (the authors' code: the reference)
    input / head / classes
    architecture: [{id, from, module, repeats, args}, ...]   one entry per box of the paper's figure
    blocks:       {name: [nodes]}                            blocks the paper defines, reused by name
    training:     optimizer, lr, weight_decay, epochs, batch (the authors' recipe)

`from` names the node(s) whose output a node reads (default: the node above; "input" = the input
of the network or of the block); `module` is a zoo op, a block of `blocks`, or `concat` / `add`.

How faithful the rebuilt network is, is MEASURED, never assumed (fidelity level):

    A  the authors' own class is used (no architecture in the card): identical by construction
    B  the card's network, with the weights of the authors' model copied in, gives the same
       outputs (`paper verify`, run where the authors' code can run: GitHub Actions)
    C  written from the paper only: checked for shapes and the reported number of parameters
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
MERGES = ("concat", "add", "mul")
TOP = {
    "paper",
    "official",
    "input",
    "head",
    "classes",
    "dropout",
    "architecture",
    "blocks",
    "training",
    "tolerance",
}
NODE = {"id", "from", "module", "repeats", "args"}


class CardError(ValueError):
    """The card is not valid; the message says where."""


def load(path) -> dict:
    """Read a card (.yaml / .yml / .json) and validate its structure."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        card = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            raise SystemExit(
                "paper cards in YAML need PyYAML: pip install pyyaml (or write the card as .json)"
            ) from None
        card = yaml.safe_load(text)
    if not isinstance(card, dict):
        raise CardError(f"{path}: a card is a mapping with the keys {sorted(TOP)}")
    card["_path"] = str(path)
    validate(card)
    return card


def _nodes(nodes, where, blocks):
    if not isinstance(nodes, list) or not nodes:
        raise CardError(f"{where}: a non-empty list of nodes is needed")
    seen = {"input"}
    for i, n in enumerate(nodes):
        at = f"{where}[{i}]"
        if not isinstance(n, dict) or "module" not in n:
            raise CardError(f"{at}: every node is a mapping with at least `module`")
        unknown = set(n) - NODE
        if unknown:
            raise CardError(f"{at}: unknown key(s) {sorted(unknown)}; a node has {sorted(NODE)}")
        nid = str(n.get("id", f"n{i}"))
        if nid in seen:
            raise CardError(f"{at}: id {nid!r} is used twice")
        src = n.get("from")
        for s in [] if src is None else ([src] if isinstance(src, str | int) else src):
            if str(s) not in seen:
                raise CardError(f"{at} ({nid}): `from: {s}` is not a node defined above it")
        if not isinstance(n.get("args", {}), dict):
            raise CardError(f"{at} ({nid}): `args` is a mapping of the module's parameters")
        if n["module"] in MERGES and (src is None or isinstance(src, str | int) or len(src) < 2):
            raise CardError(f"{at} ({nid}): {n['module']} needs `from: [a, b, ...]`")
        seen.add(nid)


def validate(card: dict) -> None:
    unknown = {k for k in card if not k.startswith("_")} - TOP
    if unknown:
        raise CardError(f"unknown top-level key(s) {sorted(unknown)}; a card has {sorted(TOP)}")
    paper = card.get("paper")
    if not isinstance(paper, dict) or not paper.get("name"):
        raise CardError("`paper.name` is required")
    off = card.get("official")
    if off is not None:
        for k in ("repo", "commit", "entry"):
            if not off.get(k):
                raise CardError(
                    f"`official.{k}` is required (the authors' repository, the pinned commit, file.py:Class)"
                )
        if ":" not in off["entry"]:
            raise CardError("`official.entry` is written path/to/file.py:ClassName")
    if "architecture" not in card and off is None:
        raise CardError(
            "a card needs `architecture` (the network) or `official` (the authors' code), or both"
        )
    inp = card.get("input")
    if not isinstance(inp, dict) or "kind" not in inp or "dims" not in inp:
        raise CardError("`input: {kind: map|ser|vec|seq|..., dims: [...]}` is required")
    blocks = card.get("blocks") or {}
    for name, nodes in blocks.items():
        _nodes(nodes, f"blocks.{name}", blocks)
    if "architecture" in card:
        _nodes(card["architecture"], "architecture", blocks)


def slug(card) -> str:
    return re.sub(r"[^a-z0-9]+", "_", card["paper"]["name"].lower()).strip("_")


def level(card) -> str:
    """Fidelity level: A (authors' class), B (verified against it), C (from the paper only)."""
    fid = fidelity(card)
    if fid and fid.get("level") in ("A", "B"):
        return fid["level"]
    return "A" if "architecture" not in card else "C"


def fidelity_path(card) -> Path:
    p = Path(card["_path"])
    return p.with_name(p.stem + ".fidelity.json")


def fidelity(card) -> dict | None:
    p = fidelity_path(card)
    return json.loads(p.read_text()) if p.exists() else None


# -------------------------------------------------------------------------- card -> spec


def _node_spec(n, blocks, stack=()):
    module, args, reps = n["module"], dict(n.get("args") or {}), int(n.get("repeats", 1))
    if module in MERGES:
        body = {"op": module}
    elif module in blocks:
        if module in stack:
            raise CardError(f"block {module!r} uses itself")
        nodes = [{**b, "args": _subst(b.get("args") or {}, args, module)} for b in blocks[module]]
        nodes = [{**b, "repeats": _subst(b.get("repeats", 1), args, module)} for b in nodes]
        body = _graph_spec(nodes, blocks, (*stack, module))
    else:
        body = {"op": module, **args}
    if reps > 1:
        body = {"op": "rep", "n": reps, "body": body}
    out = {k: n[k] for k in ("id", "from") if k in n}
    return {**out, **body}


def _subst(value, env, block):
    """Block parameters: a string "$dim" or "$dim * 4" in a block is computed from the args the
    block is used with (numbers, names, + - * // only)."""
    import ast
    import operator

    if isinstance(value, dict):
        return {k: _subst(v, env, block) for k, v in value.items()}
    if isinstance(value, list):
        return [_subst(v, env, block) for v in value]
    if not (isinstance(value, str) and value.startswith("$")):
        return value
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
    }

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, int | float):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise CardError(
                    f"block {block!r} uses ${n.id}, which is not in the args it is called with {sorted(env)}"
                )
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        raise CardError(
            f"block {block!r}: {value!r} is not a simple expression (numbers, names, + - * //)"
        )

    try:
        return ev(ast.parse(value.replace("$", ""), mode="eval").body)
    except SyntaxError:
        raise CardError(f"block {block!r}: cannot read {value!r}") from None


def _graph_spec(nodes, blocks, stack=()):
    return {"op": "graph", "nodes": [_node_spec(n, blocks, stack) for n in nodes]}


def to_arch(card: dict, out_dim: int | None = None) -> dict:
    """The zoo candidate of a card: {"spec", "input", "head", "out_dim", "family", ...}."""
    if "architecture" not in card:
        raise CardError(
            "this card has no `architecture` (level A: the authors' class is the network)"
        )
    tr = card.get("training") or {}
    k = out_dim or card.get("classes")
    if not k:
        raise CardError("the number of outputs is needed: `classes:` in the card or --classes")
    return {
        "family": f"paper_{slug(card)}",
        "spec": _graph_spec(card["architecture"], card.get("blocks") or {}),
        "input": {"kind": card["input"]["kind"], "dims": list(card["input"]["dims"])},
        "head": card.get("head", "cls"),
        "out_dim": int(k),
        "dropout": float(card.get("dropout", 0.0)),
        "origin": f"paper:{card['paper']['name']}",
        "lr": float(tr.get("lr", 1e-3)),
        "wd": float(tr.get("weight_decay", 1e-4)),
        "batch": int(tr.get("batch", 64)),
    }


def reported(card, key):
    for r in card["paper"].get("reported") or []:
        if key in r:
            return r[key]
    return None


def check(card: dict, out_dim: int | None = None) -> dict:
    """Static check of a card: every node's shape, the parameter count against the paper's.
    Returns {"ok", "level", "params", "reported_params", "params_diff", "nodes", "issues"}."""
    from . import zoo
    from .zoo import core

    res = {"name": card["paper"]["name"], "level": level(card), "issues": [], "nodes": []}
    if "architecture" not in card:
        res.update(
            ok=True,
            params=None,
            note="level A: the network is the authors' class; nothing to check statically",
        )
        return res
    try:
        arch = to_arch(card, out_dim)
        kind, dims = arch["input"]["kind"], tuple(arch["input"]["dims"])
        nodes = arch["spec"]["nodes"]
        for i, n in enumerate(nodes):  # shapes node by node: the first failing node is named
            info = core.check({"op": "graph", "nodes": nodes[: i + 1]}, kind, dims)
            res["nodes"].append(
                {"id": str(n.get("id", f"n{i}")), "kind": info["kind"], "shape": list(info["dims"])}
            )
        model = zoo.build_model(arch)
        res["params"] = zoo.count_params(model)
    except (ValueError, KeyError, TypeError) as e:
        res.update(ok=False, params=None)
        res["issues"].append(str(e))
        return res
    rp = reported(card, "params")
    res["reported_params"] = rp
    if rp:
        diff = (res["params"] - rp) / rp
        res["params_diff"] = round(diff, 4)
        tol = float(card.get("tolerance", 0.02))
        if abs(diff) > tol:
            res["issues"].append(
                f"{res['params']:,} parameters, the paper reports {rp:,} ({diff:+.1%}, tolerance {tol:.0%}): "
                "a block is probably read differently from the paper"
            )
    res["ok"] = not res["issues"]
    return res


# ----------------------------------------------------------------------- authors' code


def _stub_attr(attr):
    """Attributes of a stubbed import: any callable, but no dunder (code that walks sys.modules
    reads __file__ / __spec__ and must get AttributeError, not a function)."""
    if attr.startswith("__"):
        raise AttributeError(attr)
    return lambda *a, **k: None


def _official_class(card, root):
    """The authors' model class, imported from a checkout of their repository at `root`."""
    off = card["official"]
    file, cls = off["entry"].split(":")
    root = Path(root)
    path = root / file
    if not path.exists():
        raise SystemExit(
            f"{path} not found: --official-root must be a checkout of {off['repo']} at {off['commit']}"
        )
    for name in off.get("stub_modules") or []:  # imports the model file does not need to run
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__getattr__ = _stub_attr
            sys.modules[name] = m
    for p in [path.parent, *(root / x for x in off.get("sys_path") or [])]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(f"official_{slug(card)}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls)


def official_model(card, root):
    """(model, adapt): the authors' network and the function mapping DLNet's input to theirs."""
    off = card["official"]
    model = _official_class(card, root)(**(off.get("init_args") or {}))
    call = getattr(model, off["method"]) if off.get("method") else model

    def run(x):
        return _apply(call(_apply(x, off.get("input_ops"))), off.get("output_ops"))

    return model, run


TENSOR_OPS = ("unsqueeze", "squeeze", "flatten", "mean", "transpose", "permute", "reshape", "index")


def _apply(x, ops):
    """`[[name, arg, ...], ...]` applied in order: the few tensor methods that adapt DLNet's
    input / output convention to the authors' (e.g. [[unsqueeze, 1]]; index picks y[i])."""
    for name, *args in ops or []:
        if name not in TENSOR_OPS:
            raise CardError(f"official.input_ops / output_ops: {name!r} is not one of {TENSOR_OPS}")
        x = x[args[0]] if name == "index" else getattr(x, name)(*args)
    return x


def verify(card: dict, official_root, tol: float = 1e-5, seed: int = 0) -> dict:
    """Compare the card's network with the authors' model: the same number of parameters, and,
    with the authors' (randomly perturbed) weights copied in, the same outputs. Writes
    <card>.fidelity.json and returns it."""
    import torch

    from . import zoo

    torch.manual_seed(seed)
    ref, run = official_model(card, official_root)
    ref.eval()
    with torch.no_grad():
        for p in ref.parameters():  # perturb: zero-initialised layers must not hide a difference
            p.add_(torch.randn_like(p) * 0.05)
    off = card["official"]
    features = off.get("compare", "output") == "features"  # the network without its classifier
    arch = to_arch(card)
    mine = zoo.build_model({**arch, "head": "none"} if features else arch).eval()
    ignore = tuple(off.get("ignore_params") or ())  # tensors the authors' forward pass never uses
    a = {k: v for k, v in ref.state_dict().items()
         if not k.endswith("num_batches_tracked") and not k.startswith(ignore)}  # fmt: skip
    b = {k: v for k, v in mine.state_dict().items() if not k.endswith("num_batches_tracked")}
    n_ref = sum(p.numel() for k, p in ref.named_parameters() if not k.startswith(ignore))
    n_mine = sum(p.numel() for p in mine.parameters())
    res = {"paper": card["paper"]["name"], "official": {k: card["official"][k] for k in ("repo", "commit", "entry")},
           "compared": "features (without the classifier)" if features else "the whole network",
        "ignored_official_params": list(ignore),
        "params_official": n_ref, "params_card": n_mine, "params_match": n_ref == n_mine}  # fmt: skip
    free, mapping, unmatched = list(b), {}, []
    for k, v in a.items():  # the first unused tensor of the same shape, in order
        hit = next((m for m in free if b[m].shape == v.shape), None)
        if hit is None:
            unmatched.append(k)
        else:
            mapping[hit] = v
            free.remove(hit)
    res["unmatched_official"], res["unmatched_card"] = unmatched[:10], free[:10]
    if unmatched or free:
        res.update(level="C", max_abs_diff=None,
                   note="the two networks do not hold the same tensors; see unmatched_*")  # fmt: skip
    else:
        mine.load_state_dict({**mine.state_dict(), **mapping})
        kind, dims = card["input"]["kind"], tuple(card["input"]["dims"])
        x = (
            torch.stack([torch.randint(0, c, (4,)) for c in dims], 1)
            if kind == "ids"
            else torch.randn(4, *dims)
        )
        with torch.no_grad():
            ya, yb = run(x), mine(x)
        if ya.shape != yb.shape:
            res.update(
                level="C",
                max_abs_diff=None,
                note=f"output shapes differ: {tuple(ya.shape)} vs {tuple(yb.shape)}",
            )
        else:
            diff = float((ya - yb).abs().max())
            scale = max(1.0, float(ya.abs().max()))
            res.update(
                max_abs_diff=diff, level="B" if diff <= tol * scale and n_ref == n_mine else "C"
            )
    res["torch"] = torch.__version__
    fidelity_path(card).write_text(json.dumps(res, indent=1) + "\n")
    return res


# -------------------------------------------------------------------------------- build

OFFICIAL_MODEL_PY = '''"""{name}: the authors' own model class (fidelity level A: identical by construction).

Needs the authors' repository at the pinned commit:
    {repo} @ {commit}   ({entry})
It is cloned into ./official_src on first use; set DLNET_OFFICIAL_ROOT to use an existing checkout.
The code of that repository keeps its own licence.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from torch import nn

HERE = Path(__file__).resolve().parent
OFFICIAL = {official!r}


def _root() -> Path:
    env = os.environ.get("DLNET_OFFICIAL_ROOT")
    if env:
        return Path(env)
    root = HERE / "official_src"
    if not root.exists():
        subprocess.run(["git", "clone", "-q", OFFICIAL["repo"], str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "-q", OFFICIAL["commit"]], check=True)
    return root


class Official(nn.Module):
    """The authors' network behind DLNet's input / output conventions."""

    def __init__(self):
        super().__init__()
        file, cls = OFFICIAL["entry"].split(":")
        path = _root() / file
        for p in [path.parent, *(_root() / x for x in OFFICIAL.get("sys_path") or [])]:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        spec = importlib.util.spec_from_file_location("official_model", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.net = getattr(mod, cls)(**(OFFICIAL.get("init_args") or {{}}))
        self.ensemble_k = 0

    def forward(self, x):
        for name, *args in OFFICIAL.get("input_ops") or []:
            x = x[args[0]] if name == "index" else getattr(x, name)(*args)
        y = self.net(x)
        for name, *args in OFFICIAL.get("output_ops") or []:
            y = y[args[0]] if name == "index" else getattr(y, name)(*args)
        return y


def build_model(*_unused) -> nn.Module:
    return Official()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
'''

LEVEL_TEXT = {
    "A": "A: the authors' own class is used (identical by construction)",
    "B": "B: verified against the authors' code (same weights give the same outputs)",
    "C": "C: written from the paper, NOT verified against the authors' code",
}


def _report(card, res, lvl) -> str:
    p = card["paper"]
    lines = [f"# {p['name']} (paper baseline)", ""]
    for k in ("title", "venue", "url", "code"):
        if p.get(k):
            lines.append(f"- {k.capitalize()}: {p[k]}")
    lines.append(f"- Fidelity level: {LEVEL_TEXT[lvl]}")
    fid = fidelity(card)
    if fid:
        lines.append(f"- Check against {fid['official']['repo']} @ {fid['official']['commit']}: parameters "
                     f"{fid['params_card']:,} vs {fid['params_official']:,}, max |output difference| {fid.get('max_abs_diff')}")  # fmt: skip
    if res.get("params"):
        rp = res.get("reported_params")
        lines.append(
            f"- Parameters: {res['params']:,}"
            + (f" (paper: {rp:,}, {res['params_diff']:+.1%})" if rp else "")
        )
    rep = [r for r in p.get("reported") or [] if "metric" in r]
    if rep:
        lines += ["", "Reported by the paper (not measured here):", ""]
        lines += [
            f"- {r.get('dataset', '?')}: {r['metric']} {r['value']} ({r.get('protocol', 'protocol not given')})"
            for r in rep
        ]
    if card.get("training"):
        lines += ["", f"Authors' training recipe: `{card['training']}`"]
    return "\n".join(lines) + "\n"


def build(card: dict, out=None, into=None, out_dim: int | None = None, log=print) -> Path:
    """Write the package of a card: model.py (plain PyTorch, or the level-A wrapper),
    architecture.svg, card and report. `into` = a DLNet output folder: the package then gets the
    run's data pipeline (train.py, the same split) and joins its notebook."""
    from . import diagram, standalone

    lvl = level(card)
    run_cfg = None
    if into:
        into = Path(into)
        board = json.loads((into / "leaderboard.json").read_text())
        top1 = into / board["top_k"][0]["name"]
        run_cfg = json.loads((top1 / "config.yaml").read_text())
        out_dim = run_cfg["out_dim"]
        run_dims = list(run_cfg["arch"]["input"]["dims"])
        if (
            list(card["input"]["dims"]) != run_dims
            or card["input"]["kind"] != run_cfg["arch"]["input"]["kind"]
        ):
            raise CardError(
                f"the card's input {card['input']['kind']} {list(card['input']['dims'])} is not the input of this run "
                f"({run_cfg['arch']['input']['kind']} {run_dims}); run `dlnet design` with the paper's input size "
                "(for HSI: --patch / --pca), or write a card for this size"
            )
        if card.get("classes") and int(card["classes"]) != out_dim:
            log(
                f"[paper] the card says {card['classes']} classes, this data has {out_dim}: built for {out_dim}"
            )
        out = into
    pkg = Path(out) / f"paper_{slug(card)}"
    pkg.mkdir(parents=True, exist_ok=True)
    res = check(card, out_dim)
    if not res["ok"] and res.get("params") is None:
        raise CardError("; ".join(res["issues"]))
    arch = None
    if "architecture" in card:
        arch = to_arch(card, out_dim)
        how = standalone.write_package(
            pkg, arch, arch["out_dim"], f"{card['paper']['name']} (from the paper card)"
        )
        diagram.write(
            {**arch, "params": res["params"]}, pkg / "architecture.svg", card["paper"]["name"]
        )
        log(
            f"[paper] {pkg / 'model.py'}: plain PyTorch, {how} to the card's network; {res['params']:,} parameters"
        )
    else:
        (pkg / "model.py").write_text(
            OFFICIAL_MODEL_PY.format(
                name=card["paper"]["name"], official=card["official"], **card["official"]
            )
        )
        log(
            f"[paper] {pkg / 'model.py'}: wrapper around the authors' class ({card['official']['entry']})"
        )
    shutil.copyfile(card["_path"], pkg / ("card" + Path(card["_path"]).suffix))
    if fidelity(card):
        shutil.copyfile(fidelity_path(card), pkg / "fidelity.json")
    meta = {"paper": card["paper"], "official": card.get("official"), "training": card.get("training"),
            "fidelity_level": lvl, "fidelity": fidelity(card), "check": res}  # fmt: skip
    if run_cfg:
        tr = card.get("training") or {}
        a = arch or {"family": f"paper_{slug(card)}", "input": run_cfg["arch"]["input"], "lr": float(tr.get("lr", 1e-3)),
                     "wd": float(tr.get("weight_decay", 1e-4)), "batch": int(tr.get("batch", 64))}  # fmt: skip
        cfg = {**run_cfg, "arch": a, **meta}
        cfg.pop("search_result", None)
        cfg["hyperparameters"] = {**run_cfg.get("hyperparameters", {}), "lr": a["lr"], "weight_decay": a["wd"],
                                  "batch_size": a["batch"], "source": "paper card"}  # fmt: skip
        for f in top1.iterdir():  # the run's training / data / inference code
            if f.suffix == ".py" and f.name != "model.py":
                shutil.copyfile(f, pkg / f.name)
        if (top1 / "tests").exists():
            shutil.copytree(top1 / "tests", pkg / "tests", dirs_exist_ok=True)
        entry = {"name": pkg.name, "paper": card["paper"]["name"], "level": lvl, "params": res.get("params"),
                 "reported": card["paper"].get("reported") or []}  # fmt: skip
        board["papers"] = [p for p in board.get("papers", []) if p["name"] != pkg.name] + [entry]
        (into / "leaderboard.json").write_text(json.dumps(board, indent=2))
        _refresh_notebook(into, log)
    else:
        cfg = {
            "arch": arch,
            "in_dim": card["input"]["dims"][0],
            "out_dim": (arch or {}).get("out_dim"),
            **meta,
        }
    (pkg / "config.yaml").write_text(json.dumps(cfg, indent=2))
    (pkg / "report.md").write_text(_report(card, res, lvl))
    for issue in res["issues"]:
        log(f"[paper] warning: {issue}")
    log(f"[paper] {pkg}  (fidelity level {LEVEL_TEXT[lvl]})")
    return pkg


def _refresh_notebook(out, log):
    from . import notebook

    board = json.loads((Path(out) / "leaderboard.json").read_text())
    src = board.get("notebook")
    if src:
        notebook.write(
            out, src["source"], src.get("git_url", ""), src.get("git_subdir", ""), log=log
        )
    else:
        log(
            "[paper] baselines.ipynb not updated (written by an older DLNet): run `dlnet notebook <out> --data ...`"
        )
