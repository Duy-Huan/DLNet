"""The tools of the DLNet MCP server (`dlnet serve`).

`Service.call(name, arguments, progress, cancelled)` runs one tool and returns a JSON-able dict;
problems the caller can fix raise `ServiceError` (reported as a tool error, `isError: true`, so
the model can correct itself: MCP 2025-11-25, "input validation errors are tool execution
errors"). Tool definitions (`TOOLS`) carry JSON Schema 2020-12 input schemas and the standard
annotations (readOnlyHint, destructiveHint, idempotentHint, openWorldHint).

Typical session of an AI client:
    dlnet_info -> dlnet_files -> dlnet_profile -> (dlnet_zoo, dlnet_check_spec: write own specs)
    -> dlnet_design (extra_specs / papers) -> dlnet_job (poll) -> dlnet_leaderboard
    -> dlnet_read_file (report.md, model.py) -> dlnet_train -> dlnet_predict -> dlnet_archive
"""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import os
import re
import socket
import threading
import time
import types
import urllib.parse
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

from . import service
from .service import ServiceError

BUDGET = re.compile(r"^(\d+(?:\.\d+)?)(s|m|h|ep)$")
S, I, B, N, O, A = (
    {"type": t} for t in ("string", "integer", "boolean", "number", "object", "array")
)
DATA_ARGS = {
    "data": {**S, "description": "dataset file in the workspace, e.g. data/train.csv (.csv, .npz, .npy, .mat)"},
    "plugin": {**S, "description": "force a plugin: tabular | image | timeseries | hsi | segmentation | graph | recsys "
               "(default: detected from the file)"},
    "target": {**S, "description": "tabular: name of the target column"},
    "task": {**S, "enum": ["auto", "classification", "regression"], "description": "default auto"},
    "x_key": {**S, "description": "arrays (.npz): name of the inputs (default X)"},
    "y_key": {**S, "description": "arrays (.npz): name of the targets (default y)"},
    "gt": {**S, "description": "hyperspectral: ground-truth file (.mat / .npy) of the cube given as data"},
    "cube_key": {**S, "description": "hyperspectral: variable name of the cube in the .mat file"},
    "gt_key": {**S, "description": "hyperspectral: variable name of the ground truth in its file"},
    "patch": {**I, "description": "hyperspectral: spatial patch size (default 9)"},
    "pca": {**I, "description": "hyperspectral: PCA components kept (default 30; 0 = all bands)"},
    "split": {**S, "enum": ["pixel", "block"], "description": "hyperspectral: how train / test pixels are separated"},
}  # fmt: skip
WAIT = {**N, "description": "seconds to wait for the job before returning its id instead of the result "
        "(poll with dlnet_job); keep it below your client's tool timeout"}  # fmt: skip
RO = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
RW = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}


def tool(name, title, description, props, required=(), annotations=RO):
    return {"name": name, "title": title, "description": description, "annotations": annotations,
            "inputSchema": {"type": "object", "properties": props, "required": list(required),
                            "additionalProperties": False}}  # fmt: skip


TOOLS = [
    tool("dlnet_info", "DLNet service information",
         "Version, compute device, workspace layout, limits and the recommended order of the tools. Call it first.", {}),
    tool("dlnet_files", "List workspace files",
         "List files of the workspace (data/, runs/<run>/, a package folder). Paths are relative to the workspace.",
         {"folder": {**S, "description": "default: the workspace root"}, "depth": {**I, "default": 2},
          "limit": {**I, "default": 300}}),
    tool("dlnet_read_file", "Read a text file",
         "Read a text file of the workspace: report.md, leaderboard.md, config.yaml, model.py (the generated "
         "network as plain PyTorch), a job log, a paper card. Returns at most `limit` characters from `offset` "
         "(or the end with tail=true).",
         {"path": S, "offset": {**I, "default": 0}, "limit": {**I, "default": 20000}, "tail": {**B, "default": False}},
         ["path"]),
    tool("dlnet_write_file", "Write a small file",
         "Write a file into the workspace (a CSV dataset, a paper card .yaml, a JSON of zoo specs): `text`, or "
         "`base64` for binary content, at most 5 MB. Larger datasets are uploaded over HTTP: PUT /files/data/<name>.",
         {"path": {**S, "description": "e.g. data/iris.csv"}, "text": S, "base64": S,
          "overwrite": {**B, "default": False}}, ["path"],
         {**RW, "idempotentHint": True}),
    tool("dlnet_fetch", "Download a dataset",
         "Download a public https URL into data/ (only when the server was started with --allow-fetch; private "
         "and local addresses are refused).",
         {"url": S, "name": {**S, "description": "file name under data/ (default: taken from the URL)"},
          "overwrite": {**B, "default": False}}, ["url"], {**RW, "openWorldHint": True}),
    tool("dlnet_profile", "Profile a dataset",
         "Load a dataset with DLNet and return the detected plugin, its profile (sizes, classes, task) and, for "
         "images, the data card (spectrum, structure size, position dependence, colour, probes).",
         DATA_ARGS, ["data"]),
    tool("dlnet_candidates", "Seed architectures for a dataset",
         "The seed pool DLNet would search for this dataset (model families, zoo presets, random architectures), "
         "each with traits, size, FLOPs, the DLNet prior score and its zoo spec.",
         {**DATA_ARGS, "zoo": {**I, "default": 24}, "per_family": {**I, "default": 2}, "context": S,
          "limit": {**I, "default": 40}}, ["data"]),
    tool("dlnet_zoo", "Module catalogue",
         "Catalogue of zoo modules (ops) with input kinds, parameter values, tags and sources; use it to write "
         "new architectures as zoo specs. Give `kind` to get only the modules usable on that tensor kind.",
         {"kind": {**S, "description": "vec | seq | ser | map | vol | graph | ids"}}),
    tool("dlnet_check_spec", "Check an architecture",
         "Statically check a zoo spec for an input kind and shape (no training): returns parameters, FLOPs and "
         "output shape, or the error to fix. With diagram=true also returns the block diagram (SVG).",
         {"spec": {**O, "description": 'e.g. {"op": "seq", "body": [{"op": "conv_bn_act", "out": 32}, ...]}'},
          "kind": S, "dims": {**A, "items": I, "description": "input shape without the batch, e.g. [3, 32, 32]"},
          "diagram": {**B, "default": False}}, ["spec", "kind", "dims"]),
    tool("dlnet_design", "Design baselines (search + train)",
         "Run `dlnet design` on a dataset: generate candidate networks, search within the budget, train and "
         "evaluate the Top-K (validation mean ± std over repeated runs, one test evaluation), write one package "
         "per baseline (model.py as plain PyTorch, weights, train.py, predict.py, report.md, architecture.svg) "
         "plus leaderboard.md and a Colab notebook. Runs as a background job: returns the result when it ends "
         "within `wait` seconds, otherwise a job_id to poll with dlnet_job. `extra_specs` (zoo specs written by "
         "you) and `papers` (paper cards) join the comparison under the same protocol.",
         {**DATA_ARGS,
          "name": {**S, "description": "name of the run: results go to runs/<name> (default: made up)"},
          "budget": {**S, "default": "5m", "description": "wall clock (90s, 10m, 2h) or training epochs (400ep: "
                     "reproducible on any machine)"},
          "topk": {**I, "default": 3}, "objective": {**S, "enum": ["accuracy", "params", "flops", "balanced", "pareto"]},
          "goal": {**S, "enum": ["accuracy", "small", "balanced"]},
          "context": {**S, "description": "what the data is, in words (e.g. 'satellite images of crops'): steers the seeds"},
          "seeds": {**I, "description": "keep only the N best-ranked seed candidates (0 = all)"},
          "zoo": {**I, "description": "random zoo architectures added to the pool"},
          "evolve": {**I, "description": "children created by evolution"},
          "rung_score": {**S, "enum": ["val", "tse", "val+tse"], "description": "how successive halving ranks "
                         "a rung: validation, training speed estimate, or both"},
          "rungs": {**S, "description": "epochs at the end of each rung, e.g. 1,4,12,30"},
          "per_family": I, "seed": I, "cv_splits": I, "max_params": I,
          "device": {**S, "description": "auto | cpu | cuda | cuda:1 | mps"}, "threads": I,
          "papers": {**A, "items": S, "description": "paper cards (.yaml) or folders of cards in the workspace"},
          "paper_recipe": {**B, "description": "also train each paper network with its authors' recipe"},
          "extra_specs": {**A, "items": O, "description": "zoo specs to add to the candidates"},
          "meta_store": S, "wait": {**WAIT, "default": 45}}, ["data"], RW),
    tool("dlnet_job", "Status of a job",
         "Status, timing and the last log lines of a background job; when it has ended, its result (leaderboard "
         "of a design run, metrics of a training run, ...). `wait` blocks up to that many seconds for the end.",
         {"job_id": S, "wait": {**WAIT, "default": 0}, "log_lines": {**I, "default": 20}}, ["job_id"]),
    tool("dlnet_jobs", "List jobs", "Recent background jobs, newest first.", {"limit": {**I, "default": 30}}),
    tool("dlnet_cancel", "Cancel a job", "Stop a queued or running job (its partial output stays in the workspace).",
         {"job_id": S}, ["job_id"], {**RW, "destructiveHint": True, "idempotentHint": True}),
    tool("dlnet_leaderboard", "Results of a design run",
         "leaderboard.md, the verdict and the Top-K table (validation mean ± std, test, parameters, FLOPs, package "
         "folder) of a finished design run.",
         {"run": {**S, "description": "run name or folder, e.g. my-run or runs/my-run"}}, ["run"]),
    tool("dlnet_package", "Inspect a baseline package",
         "One package of a run (e.g. runs/my-run/top1_cnn): its files, config (architecture spec, recipe, "
         "preprocessing), metrics and the first part of report.md.",
         {"package": {**S, "description": "package folder"}}, ["package"]),
    tool("dlnet_train", "Train a baseline in full",
         "Run the package's own train.py (the full recipe) on a dataset and report validation and test metrics; "
         "the weights (model.pt) and preprocess.json are saved in the package. Background job, like dlnet_design.",
         {"package": S, **{k: DATA_ARGS[k] for k in ("data", "target", "x_key", "y_key", "gt", "cube_key", "gt_key")},
          "epochs": I, "seed": I, "device": S, "wait": {**WAIT, "default": 45}}, ["package", "data"], RW),
    tool("dlnet_predict", "Predict with a trained baseline",
         "Run the package's predict.py on new data (same columns / shape as the training data, labels not "
         "needed) and write the predictions (.npy, or .csv). Background job.",
         {"package": S, "data": S, "out": {**S, "description": "default: <package>/predictions.npy"},
          "device": S, "wait": {**WAIT, "default": 45}}, ["package", "data"], RW),
    tool("dlnet_paper", "Paper cards",
         "Work with paper cards (a published network written as YAML, see doc/user-guide.md): `check` validates "
         "cards and reports parameters and fidelity level; `build` writes the package of each card into a new run.",
         {"action": {**S, "enum": ["check", "build"]}, "cards": {**A, "items": S},
          "name": {**S, "description": "build: run name"}, "wait": {**WAIT, "default": 45}}, ["action", "cards"], RW),
    tool("dlnet_archive", "Zip a run or a package",
         "Zip a run or a package folder for one download; returns the zip's path and, over HTTP, its URL.",
         {"path": S}, ["path"], {**RW, "idempotentHint": True}),
]  # fmt: skip

INSTRUCTIONS = """DLNet turns a dataset into Top-K trained PyTorch baselines and compares them with \
networks from papers under one protocol. All paths are relative to the server's workspace: \
datasets live in data/, results in runs/<name>/. Start with dlnet_info and dlnet_files. Typical flow: \
dlnet_profile (what the data is) -> optionally dlnet_zoo + dlnet_check_spec (write your own \
architectures as zoo specs) -> dlnet_design (search, train, evaluate; long: it returns a job_id \
when it does not finish within `wait`) -> dlnet_job (poll) -> dlnet_leaderboard -> dlnet_read_file \
(report.md, model.py) -> dlnet_train / dlnet_predict on a package. Report numbers exactly as the \
tools return them (validation mean ± std and the single test evaluation); do not invent results."""

PROMPTS = [
    {"name": "design_baselines", "title": "Design baselines for a dataset",
     "description": "Profile a dataset, run a design within a budget and report the leaderboard honestly.",
     "arguments": [{"name": "data", "description": "dataset path in the workspace", "required": True},
                   {"name": "budget", "description": "e.g. 10m or 400ep", "required": False},
                   {"name": "context", "description": "what the data is", "required": False}]},
    {"name": "propose_architecture", "title": "Propose and test a new architecture",
     "description": "Write new zoo specs for a dataset, check them and compare them with DLNet's own baselines.",
     "arguments": [{"name": "data", "description": "dataset path in the workspace", "required": True},
                   {"name": "idea", "description": "the idea to try", "required": False}]},
    {"name": "compare_with_papers", "title": "Compare with published networks",
     "description": "Run a design with paper cards as extra baselines under the same protocol.",
     "arguments": [{"name": "data", "description": "dataset path in the workspace", "required": True},
                   {"name": "cards", "description": "paper cards or a folder of them", "required": True}]},
]  # fmt: skip


def prompt_text(name: str, a: dict) -> str:
    data = a.get("data", "<dataset>")
    if name == "design_baselines":
        return (f"Use the DLNet tools on the dataset `{data}`. 1) dlnet_info and dlnet_profile: say what the data "
                f"is. 2) dlnet_design with budget {a.get('budget', '10m')}"
                + (f" and context {a['context']!r}" if a.get("context") else "")
                + "; poll dlnet_job until it ends. 3) dlnet_leaderboard: report the Top-K with validation mean ± "
                "std, the test score, parameters, and the verdict against the classical reference, exactly as "
                "returned. 4) Read report.md of the top-1 package and summarise the architecture and the recipe.")  # fmt: skip
    if name == "propose_architecture":
        return (f"For the dataset `{data}`: dlnet_profile, then dlnet_zoo for its tensor kind. Write 2-4 new "
                f"architectures as zoo specs{' for this idea: ' + a['idea'] if a.get('idea') else ''}; validate each "
                "with dlnet_check_spec (fix what it reports). Run dlnet_design with them as extra_specs so they are "
                "trained under the same protocol as DLNet's own candidates, and report where they rank.")  # fmt: skip
    if name == "compare_with_papers":
        return (f"For the dataset `{data}`: dlnet_paper check on {a.get('cards', '<cards>')}, then dlnet_design with "
                "papers set to those cards. Report the 'Paper baselines' table next to DLNet's Top-K: the paper's "
                "reported number, the result under DLNet's protocol, and the fidelity level of each card.")  # fmt: skip
    raise ServiceError(f"unknown prompt {name!r}")


def _seconds(budget: str) -> float | None:
    """Seconds of a wall-clock budget; None for an epoch budget."""
    m = BUDGET.match(str(budget).strip().lower())
    if not m:
        raise ServiceError(f"budget {budget!r}: use 90s, 10m, 2h or 400ep")
    v, unit = float(m.group(1)), m.group(2)
    return None if unit == "ep" else v * {"s": 1, "m": 60, "h": 3600}[unit]


class Service:
    """Workspace + jobs + the tool handlers. One instance per server."""

    def __init__(self, workspace=None, sandbox: bool = True, max_jobs: int = 1, default_wait: float = 45.0,
                 max_budget: str = "12h", allow_fetch: bool = False, base_url: str | None = None):  # fmt: skip
        self._root = workspace or os.environ.get("DLNET_WORKSPACE") or "dlnet-workspace"
        self.sandbox, self.max_jobs = sandbox, max_jobs
        self.default_wait, self.allow_fetch, self.base_url = default_wait, allow_fetch, base_url
        self.max_budget = max_budget
        self.max_seconds = _seconds(max_budget)
        self._ws: service.Workspace | None = None
        self._jobs: service.Jobs | None = None
        self._lock = threading.Lock()  # workspace / jobs creation
        self._torch = threading.Lock()  # one in-process torch tool at a time
        self.started = time.time()

    # the workspace is created on first use
    @property
    def ws(self) -> service.Workspace:
        with self._lock:
            if self._ws is None:
                self._ws = service.Workspace(self._root, self.sandbox)
            return self._ws

    @property
    def jobs(self) -> service.Jobs:
        ws = self.ws
        with self._lock:
            if self._jobs is None:
                self._jobs = service.Jobs(ws, self.max_jobs)
            return self._jobs

    def close(self) -> None:
        if self._jobs is not None:
            self._jobs.shutdown()

    # ------------------------------------------------------------------ dispatch

    def call(self, name: str, a: dict, progress=None, cancelled=None) -> dict:
        fn = (
            getattr(self, "tool_" + name.removeprefix("dlnet_"), None)
            if name.startswith("dlnet_")
            else None
        )
        if fn is None:
            raise KeyError(name)
        if not isinstance(a, dict):
            raise ServiceError("arguments must be an object")
        spec = next(t for t in TOOLS if t["name"] == name)["inputSchema"]
        unknown = sorted(set(a) - set(spec["properties"]))
        if unknown:
            raise ServiceError(
                f"unknown argument(s) {unknown}; {name} takes {sorted(spec['properties'])}"
            )
        missing = [k for k in spec["required"] if a.get(k) in (None, "")]
        if missing:
            raise ServiceError(f"missing required argument(s) {missing}")
        self._progress, self._cancelled = progress, cancelled
        return fn(a, progress or (lambda *_: None), cancelled or (lambda: False))

    # ------------------------------------------------------------------ small tools

    def tool_info(self, a, progress, cancelled):
        import torch

        from . import __version__, zoo
        from .zoo import families

        cuda = torch.cuda.is_available()
        return {
            "dlnet": __version__, "torch": torch.__version__,
            "device": {"cuda": cuda, "gpu": torch.cuda.get_device_name(0) if cuda else None,
                       "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
                       "cpu_threads": os.cpu_count()},
            "workspace": {"root": str(self.ws.root) if not self.sandbox else "(server side)",
                          "layout": {"data/": "datasets and paper cards", "runs/": "one folder per design run",
                                     "jobs/": "job state and logs"},
                          "sandboxed": self.sandbox},
            "limits": {"max_parallel_jobs": self.max_jobs, "max_budget": self.max_budget,
                       "default_wait_seconds": self.default_wait, "write_file_bytes": service.WRITE_LIMIT,
                       "fetch_enabled": self.allow_fetch},
            "zoo": {"modules": len(zoo.OPS),
                    "families": {k: list(v[0]) for k, v in families.DOMAINS.items()}},
            "http": ({"upload": f"PUT {self.base_url}/files/data/<name>", "download": f"GET {self.base_url}/files/<path>"}
                     if self.base_url else None),
            "how_to": INSTRUCTIONS,
        }  # fmt: skip

    def tool_files(self, a, progress, cancelled):
        return self.ws.listing(
            a.get("folder") or ".", int(a.get("depth", 2)), int(a.get("limit", 300))
        )

    def tool_read_file(self, a, progress, cancelled):
        return self.ws.read_text(
            a["path"], int(a.get("offset", 0)), int(a.get("limit", 20000)), bool(a.get("tail"))
        )

    def tool_write_file(self, a, progress, cancelled):
        if (a.get("text") is None) == (a.get("base64") is None):
            raise ServiceError("give exactly one of `text` and `base64`")
        try:
            data = (
                a["text"].encode()
                if a.get("text") is not None
                else base64.b64decode(a["base64"], validate=True)
            )
        except (ValueError, AttributeError) as e:
            raise ServiceError(f"invalid content: {e}") from e
        if len(data) > service.WRITE_LIMIT:
            raise ServiceError(f"{len(data):,} bytes is over the {service.WRITE_LIMIT:,} byte limit of this tool; "
                               "upload the file over HTTP (PUT /files/data/<name>)")  # fmt: skip
        return self.ws.write(a["path"], data, bool(a.get("overwrite")))

    def tool_fetch(self, a, progress, cancelled):
        if not self.allow_fetch:
            raise ServiceError("downloads are disabled on this server (start it with --allow-fetch), "
                               "or upload the file: PUT /files/data/<name>")  # fmt: skip
        u = urllib.parse.urlsplit(a["url"])
        if u.scheme != "https" or not u.hostname:
            raise ServiceError("only https URLs are downloaded")
        try:
            addrs = {i[4][0] for i in socket.getaddrinfo(u.hostname, u.port or 443)}
        except OSError as e:
            raise ServiceError(f"cannot resolve {u.hostname}: {e}") from e
        for ip in addrs:
            if not ipaddress.ip_address(ip.split("%")[0]).is_global:
                raise ServiceError(f"{u.hostname} resolves to a non-public address ({ip}); refused")
        name = a.get("name") or Path(urllib.parse.unquote(u.path)).name or "download"
        if not service.NAME.match(name):
            raise ServiceError("name: letters, digits, '.', '_' and '-' only")
        dest = self.ws.path(f"data/{name}")
        if dest.exists() and not a.get("overwrite"):
            raise ServiceError(f"data/{name} exists; pass overwrite=true")
        cap, got = 4 * 1024**3, 0
        tmp = dest.with_name(dest.name + ".part")
        req = urllib.request.Request(a["url"], headers={"User-Agent": "dlnet-service"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    got += len(chunk)
                    if got > cap:
                        raise ServiceError("download larger than 4 GB: refused")
                    f.write(chunk)
                    progress(got, None, f"{got / 1e6:.0f} MB")
                    if cancelled():
                        raise ServiceError("cancelled")
            tmp.replace(dest)
        except OSError as e:
            raise ServiceError(f"download failed: {e}") from e
        finally:
            tmp.unlink(missing_ok=True)
        return {"path": self.ws.rel(dest), "bytes": got}

    # ------------------------------------------------------------------ in-process analysis

    def _namespace(self, a: dict):
        data = self.ws.path(a["data"], must_exist=True)
        gt = self.ws.path(a["gt"], must_exist=True) if a.get("gt") else None
        return types.SimpleNamespace(
            data=str(data), seed=int(a.get("seed", 0)), x_key=a.get("x_key"), y_key=a.get("y_key"),
            task=a.get("task", "auto"), target=a.get("target"), gt=str(gt) if gt else None,
            cube_key=a.get("cube_key"), gt_key=a.get("gt_key"), split=a.get("split", "pixel"), frac=None,
            patch=int(a.get("patch", 9)), pca=int(a.get("pca", 30)), block_size=16,
        )  # fmt: skip

    def _prepare(self, a: dict):
        from . import plugins

        ns = self._namespace(a)
        try:
            p = plugins.get_plugin(a["plugin"]) if a.get("plugin") else plugins.route(ns.data)
        except (KeyError, ValueError, SystemExit) as e:
            raise ServiceError(f"plugin: {e}") from e
        with redirect_stdout(io.StringIO()):
            data = p.prepare(ns)
            prof = p.profile(data)
        card = {"n_train": prof.get("n_train")}
        if data.get("kind") in ("image", "segmentation") and data["Xtr"].dim() == 4:
            from .datacard import image_card

            card = image_card(data)
        return p, data, prof, card

    def tool_profile(self, a, progress, cancelled):
        with self._torch:
            p, _, prof, card = self._prepare(a)
        return {"plugin": p.name, "profile": prof, "data_card": card}

    def tool_candidates(self, a, progress, cancelled):
        from . import advisor, seeds

        with self._torch:
            p, data, prof, card = self._prepare(a)
            with redirect_stdout(io.StringIO()):
                cands = p.generate(prof, int(a.get("per_family", 2)), 0) + p.generate_zoo(
                    data, prof, int(a.get("zoo", 24)), 0, True, 16
                )
                cands, _ = seeds.select(cands, card, 0, a.get("context", ""))
        table = advisor.candidate_table(cands)[: int(a.get("limit", 40))]
        for row, c in zip(table, cands, strict=False):
            row["spec"] = c.get("spec")
        inp = p.zoo_input(data)
        return {"plugin": p.name, "input": {"kind": inp[0], "dims": list(inp[1])} if inp else None,
                "data_card": card, "candidates": table}  # fmt: skip

    def tool_zoo(self, a, progress, cancelled):
        from .advisor import ops_catalogue
        from .zoo import KINDS, OPS

        kind = a.get("kind")
        if kind:
            if kind not in KINDS:
                raise ServiceError(f"kind {kind!r}: one of {list(KINDS)}")
            return {"ops": ops_catalogue(kind)}
        return {"ops": [{"op": n, "kinds": list(o.kin), "params": o.space, "tags": sorted(o.tags),
                         "source": o.source} for n, o in OPS.items()]}  # fmt: skip

    def tool_check_spec(self, a, progress, cancelled):
        from . import zoo

        try:
            with self._torch:
                info = zoo.check(a["spec"], a["kind"], tuple(a["dims"]))
        except (ValueError, KeyError, TypeError) as e:
            return {"valid": False, "error": str(e)}
        out = {
            "valid": True,
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in info.items()},
        }
        if a.get("diagram"):
            from . import diagram

            cfg = {
                "spec": a["spec"],
                "input": {"kind": a["kind"], "dims": list(a["dims"])},
                "head": "none",
            }
            try:
                out["diagram_svg"] = diagram.svg(cfg)
            except (ValueError, KeyError, TypeError) as e:
                out["diagram_error"] = str(e)
        return out

    # ------------------------------------------------------------------ jobs

    def _wait(self, job, a, progress, cancelled) -> dict:
        """Wait up to `wait` seconds for the job, reporting progress; then the job's description
        with its result when it has ended."""
        wait = a.get("wait")
        wait = self.default_wait if wait is None else float(wait)
        t0, last = time.time(), None
        while not job.done.is_set() and time.time() - t0 < wait and not cancelled():
            job.done.wait(min(2.0, max(wait - (time.time() - t0), 0.05)))
            tail = self.jobs.log_tail(job, 1)
            line = tail[0] if tail else job.status
            if line != last:
                last = line
            progress(round(time.time() - t0, 1), None, f"[{job.status}] {line}"[:300])
        return self._result(job, int(a.get("log_lines", 20)))

    def _result(self, job, log_lines: int = 20) -> dict:
        d = self.jobs.describe(job, log_lines)
        d["ok"] = job.status == "succeeded" if job.status in service.FINAL else None
        if job.status not in service.FINAL:
            d["next"] = (
                f"still {job.status}: call dlnet_job with job_id {job.id!r} (wait up to your tool timeout)"
            )
            return d
        if job.status == "succeeded":
            try:
                d.update(self._outcome(job))
            except (OSError, ValueError, KeyError) as e:
                d["result_error"] = f"the job ended but its result could not be read: {e}"
        return d

    def _outcome(self, job) -> dict:
        ws = self.ws
        if job.kind == "design":
            return self._board(ws.path(job.out))
        log = (
            (self.jobs.log_path(job).read_text(errors="replace"))
            if self.jobs.log_path(job).exists()
            else ""
        )
        if job.kind == "train":
            out = {"package": job.extra.get("package")}
            for key in ("val", "test"):
                m = re.findall(rf"^{key}\s*:\s*(\{{.*\}})\s*$", log, re.MULTILINE)
                if m:
                    try:
                        out[key] = json.loads(m[-1].replace("'", '"').replace("nan", "null"))
                    except ValueError:
                        out[key] = m[-1]
            m = re.search(r"^params=([\d,]+)", log, re.MULTILINE)
            if m:
                out["params"] = int(m.group(1).replace(",", ""))
            return out
        if job.kind == "predict":
            p = ws.path(job.extra["out"])
            return {"predictions": ws.rel(p), "bytes": p.stat().st_size if p.exists() else None,
                    **self._link(p)}  # fmt: skip
        if job.kind == "paper":
            return {"output": log[-6000:], "run": job.out}
        return {}

    def _link(self, p: Path) -> dict:
        if not self.base_url:
            return {}
        return {"url": f"{self.base_url}/files/{urllib.parse.quote(self.ws.rel(p))}"}

    def _board(self, out: Path) -> dict:
        f = out / "leaderboard.json"
        if not f.exists():
            raise ServiceError(f"{self.ws.rel(out)} has no leaderboard.json: not a finished design run "
                               "(see dlnet_jobs / dlnet_files)")  # fmt: skip
        b = json.loads(f.read_text())
        keys = (
            "rank",
            "name",
            "origin",
            "params",
            "flops",
            "val_mean",
            "val_std",
            "n_runs",
            "epochs",
            "test",
        )
        top = [{k: r.get(k) for k in keys if k in r} for r in b.get("top_k", [])]
        for r in top:
            if r.get("name") and (out / r["name"]).is_dir():
                r["package"] = self.ws.rel(out / r["name"])
        md = out / "leaderboard.md"
        return {"run": self.ws.rel(out), "task": b.get("task"), "metric": b.get("metric"),
                "verdict": b.get("verdict"), "top_k": top, "reference": b.get("reference"),
                "papers": b.get("papers"), "warnings": b.get("warnings"), "budget": b.get("budget"),
                "leaderboard_md": md.read_text()[:12000] if md.exists() else None,
                "files": [e["path"] for e in self.ws.listing(self.ws.rel(out), 1, 60)["entries"]]}  # fmt: skip

    def tool_design(self, a, progress, cancelled):
        ws = self.ws
        sec = _seconds(a.get("budget", "5m"))
        if sec is not None and self.max_seconds is not None and sec > self.max_seconds:
            raise ServiceError(
                f"budget {a['budget']} is over this server's limit ({self.max_budget})"
            )
        ns = self._namespace(a)
        out = ws.run_dir(a.get("name"))
        argv = [*service.python(), "design", "--data", ns.data, "--out", str(out),
                "--budget", str(a.get("budget", "5m")), "--topk", str(int(a.get("topk", 3)))]  # fmt: skip
        flags = {"objective": "--objective", "goal": "--goal", "context": "--context", "seeds": "--seeds",
                 "target": "--target", "plugin": "--plugin", "device": "--device", "threads": "--threads",
                 "zoo": "--zoo", "evolve": "--evolve", "per_family": "--per-family", "seed": "--seed",
                 "cv_splits": "--cv-splits", "max_params": "--max-params", "rung_score": "--rung-score", "rungs": "--rungs", "task": "--task", "x_key": "--x-key",
                 "y_key": "--y-key", "cube_key": "--cube-key", "gt_key": "--gt-key", "patch": "--patch",
                 "pca": "--pca", "split": "--split"}  # fmt: skip
        for k, flag in flags.items():
            if a.get(k) not in (None, ""):
                argv += [flag, str(a[k])]
        if ns.gt:
            argv += ["--gt", ns.gt]
        if a.get("meta_store"):
            argv += ["--meta-store", str(ws.path(a["meta_store"]))]
        if a.get("papers"):
            argv += ["--papers", *[str(ws.path(p, must_exist=True)) for p in a["papers"]]]
            if a.get("paper_recipe"):
                argv.append("--paper-recipe")
        if a.get("extra_specs"):
            out.parent.mkdir(parents=True, exist_ok=True)
            spec_file = out.parent / f"{out.name}.extra_specs.json"
            spec_file.write_text(json.dumps(a["extra_specs"]))
            argv += ["--extra-specs", str(spec_file)]
        job = self.jobs.submit(
            "design",
            argv,
            out=out,
            title=f"design {ws.rel(Path(ns.data))} ({a.get('budget', '5m')})",
        )
        return self._wait(job, a, progress, cancelled)

    def tool_job(self, a, progress, cancelled):
        job = self.jobs.get(a["job_id"])
        return self._wait(job, {**a, "wait": a.get("wait", 0)}, progress, cancelled)

    def tool_jobs(self, a, progress, cancelled):
        return {"jobs": self.jobs.listing(int(a.get("limit", 30)))}

    def tool_cancel(self, a, progress, cancelled):
        return self.jobs.describe(self.jobs.cancel(a["job_id"]), 5)

    def _run_folder(self, name: str) -> Path:
        ws = self.ws
        direct = ws.path(name)
        if (direct / "leaderboard.json").exists() or not service.NAME.match(name):
            return direct
        return ws.root / "runs" / name

    def tool_leaderboard(self, a, progress, cancelled):
        return self._board(self._run_folder(a["run"]))

    def _package(self, p: str) -> Path:
        pkg = self.ws.path(p, must_exist=True)
        if not (pkg / "config.yaml").exists():
            raise ServiceError(f"{self.ws.rel(pkg)} is not a package (no config.yaml); packages are the "
                               "top<k>_<name> folders of a run, see dlnet_leaderboard")  # fmt: skip
        return pkg

    def tool_package(self, a, progress, cancelled):
        pkg = self._package(a["package"])
        cfg = json.loads((pkg / "config.yaml").read_text())
        report = pkg / "report.md"
        return {"package": self.ws.rel(pkg), "files": sorted(f.name for f in pkg.iterdir()),
                "config": cfg, "trained": (pkg / "model.pt").exists(),
                "report_md": report.read_text()[:8000] if report.exists() else None,
                "model_py": self.ws.rel(pkg / "model.py") if (pkg / "model.py").exists() else None}  # fmt: skip

    def tool_train(self, a, progress, cancelled):
        pkg = self._package(a["package"])
        if not (pkg / "train.py").exists():
            raise ServiceError(f"{self.ws.rel(pkg)} has no train.py")
        ns = self._namespace(a)
        argv = [*service.package_python(), str(pkg / "train.py"), "--data", ns.data]
        for k, flag in (("target", "--target"), ("x_key", "--x-key"), ("y_key", "--y-key"), ("cube_key", "--cube-key"),
                        ("gt_key", "--gt-key"), ("epochs", "--epochs"), ("seed", "--seed"), ("device", "--device")):  # fmt: skip
            if a.get(k) not in (None, ""):
                argv += [flag, str(a[k])]
        if ns.gt:
            argv += ["--gt", ns.gt]
        job = self.jobs.submit("train", argv, cwd=pkg, out=pkg, title=f"train {self.ws.rel(pkg)}",
                               extra={"package": self.ws.rel(pkg)})  # fmt: skip
        return self._wait(job, a, progress, cancelled)

    def tool_predict(self, a, progress, cancelled):
        pkg = self._package(a["package"])
        if not (pkg / "model.pt").exists():
            raise ServiceError(
                f"{self.ws.rel(pkg)} has no trained weights (model.pt): run dlnet_train first"
            )
        data = self.ws.path(a["data"], must_exist=True)
        out = self.ws.path(a["out"]) if a.get("out") else pkg / "predictions.npy"
        argv = [
            *service.package_python(),
            str(pkg / "predict.py"),
            "--data",
            str(data),
            "--out",
            str(out),
        ]
        if a.get("device"):
            argv += ["--device", str(a["device"])]
        job = self.jobs.submit("predict", argv, cwd=pkg, out=pkg, title=f"predict {self.ws.rel(pkg)}",
                               extra={"out": self.ws.rel(out)})  # fmt: skip
        return self._wait(job, a, progress, cancelled)

    def tool_paper(self, a, progress, cancelled):
        if a["action"] not in ("check", "build"):
            raise ServiceError("action: check or build")
        if not isinstance(a["cards"], list) or not a["cards"]:
            raise ServiceError("cards: a non-empty list of card files")
        cards = [str(self.ws.path(c, must_exist=True)) for c in a["cards"]]
        argv = [*service.python(), "paper", a["action"], *cards]
        out = None
        if a["action"] == "build":
            out = self.ws.run_dir(a.get("name"), "papers")
            argv += ["--out", str(out)]
        job = self.jobs.submit("paper", argv, out=out, title=f"paper {a['action']}")
        return self._wait(job, a, progress, cancelled)

    def tool_archive(self, a, progress, cancelled):
        r = self.ws.archive(a["path"])
        return {**r, **self._link(self.ws.path(r["path"]))}
