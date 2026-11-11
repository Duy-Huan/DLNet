"""`dlnet mcp`: DLNet as a Model Context Protocol server (stdio), so an AI client such as
Claude Code, Codex CLI or Google Antigravity can profile data, inspect and write architectures,
run `design` and read the leaderboard. Optional; nothing else in DLNet depends on it.

Transport: JSON-RPC 2.0, one message per line on stdin/stdout (MCP stdio transport). Implemented
methods: initialize, notifications/initialized, ping, tools/list, tools/call. Written from the
MCP specification (modelcontextprotocol.io, revision 2025-06-18) without the MCP SDK, to avoid a
dependency; the protocol version the client asks for is echoed back. Logs go to stderr only.

Client configuration: doc/ai-advisor.md.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

from . import jsonsafe

SERVER = {"name": "dlnet", "version": None}
PROTOCOL = "2025-06-18"

TOOLS = [
    {
        "name": "dlnet_profile",
        "description": "Load a dataset with DLNet and return the detected plugin, its profile and, for images, "
        "the data card (spectrum, structure size, position dependence, colour, probes).",
        "inputSchema": {"type": "object", "required": ["data"], "properties": {
            "data": {"type": "string", "description": "path of the data file (.csv, .npz, .mat, .npy)"},
            "plugin": {"type": "string", "description": "force a plugin (default: auto)"},
            "target": {"type": "string", "description": "tabular: target column"}}},
    },
    {
        "name": "dlnet_candidates",
        "description": "The seed pool DLNet would search for this dataset (model families, zoo presets, random "
        "architectures), each with traits, size, FLOPs, the DLNet prior score and its zoo spec.",
        "inputSchema": {"type": "object", "required": ["data"], "properties": {
            "data": {"type": "string"}, "plugin": {"type": "string"}, "target": {"type": "string"},
            "zoo": {"type": "integer", "default": 24}, "per_family": {"type": "integer", "default": 2},
            "context": {"type": "string"}, "limit": {"type": "integer", "default": 40}}},
    },
    {
        "name": "dlnet_zoo",
        "description": "Catalogue of zoo modules (ops) with input kinds, parameter values and sources; use it to "
        "write new architectures as zoo specs.",
        "inputSchema": {"type": "object", "properties": {
            "kind": {"type": "string", "description": "vec | seq | ser | map | vol | graph | ids"}}},
    },
    {
        "name": "dlnet_check_spec",
        "description": "Statically check a zoo spec for an input kind and shape; returns parameters and output "
        "shape, or the error.",
        "inputSchema": {"type": "object", "required": ["spec", "kind", "dims"], "properties": {
            "spec": {"type": "object"}, "kind": {"type": "string"},
            "dims": {"type": "array", "items": {"type": "integer"}}}},
    },
    {
        "name": "dlnet_design",
        "description": "Run `dlnet design` (blocking, at most `budget`). Optional `extra_specs` (zoo specs written "
        "by you) join the candidates. Returns the leaderboard summary.",
        "inputSchema": {"type": "object", "required": ["data", "out"], "properties": {
            "data": {"type": "string"}, "out": {"type": "string", "description": "new output folder"},
            "budget": {"type": "string", "default": "5m"},
            "objective": {"type": "string", "enum": ["accuracy", "params", "flops", "balanced", "pareto"]},
            "context": {"type": "string"}, "topk": {"type": "integer", "default": 3},
            "seeds": {"type": "integer", "default": 0}, "target": {"type": "string"},
            "plugin": {"type": "string"}, "meta_store": {"type": "string"},
            "extra_specs": {"type": "array", "items": {"type": "object"}}}},
    },
    {
        "name": "dlnet_leaderboard",
        "description": "Read leaderboard.md (and the verdict) of a finished design run.",
        "inputSchema": {"type": "object", "required": ["out"], "properties": {"out": {"type": "string"}}},
    },
]  # fmt: skip


def _prepare(a: dict):
    from . import plugins

    ns = types.SimpleNamespace(
        data=a["data"], seed=0, x_key=None, y_key=None, task="auto", target=a.get("target"),
        gt=None, cube_key=None, gt_key=None, split="pixel", frac=None, patch=9, pca=30, block_size=16,
    )  # fmt: skip
    p = plugins.route(a["data"]) if not a.get("plugin") else plugins.get_plugin(a["plugin"])
    with redirect_stdout(io.StringIO()):
        data = p.prepare(ns)
        prof = p.profile(data)
    card = {"n_train": prof.get("n_train")}
    if data.get("kind") in ("image", "segmentation") and data["Xtr"].dim() == 4:
        from .datacard import image_card

        card = image_card(data)
    return p, data, prof, card


def tool_profile(a):
    p, _, prof, card = _prepare(a)
    return {"plugin": p.name, "profile": prof, "data_card": card}


def tool_candidates(a):
    from . import advisor, seeds

    p, data, prof, card = _prepare(a)
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


def tool_zoo(a):
    from .advisor import ops_catalogue
    from .zoo import OPS

    kind = a.get("kind")
    if kind:
        return {"ops": ops_catalogue(kind)}
    return {"ops": [{"op": n, "kinds": list(o.kin), "params": o.space, "tags": sorted(o.tags),
                     "source": o.source} for n, o in OPS.items()]}  # fmt: skip


def tool_check(a):
    from . import zoo

    try:
        info = zoo.check(a["spec"], a["kind"], tuple(a["dims"]))
    except (ValueError, KeyError, TypeError) as e:
        return {"valid": False, "error": str(e)}
    return {"valid": True, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in info.items()}}


def tool_design(a):
    out = Path(a["out"])
    args = [sys.executable, "-m", "dlnet", "design", "--data", a["data"], "--out", str(out),
            "--budget", a.get("budget", "5m"), "--topk", str(a.get("topk", 3))]  # fmt: skip
    for k, flag in (("objective", "--objective"), ("context", "--context"), ("seeds", "--seeds"),
                    ("target", "--target"), ("plugin", "--plugin"), ("meta_store", "--meta-store")):  # fmt: skip
        if a.get(k) not in (None, ""):
            args += [flag, str(a[k])]
    if a.get("extra_specs"):
        out.parent.mkdir(parents=True, exist_ok=True)
        spec_file = out.parent / f"{out.name}.extra_specs.json"
        spec_file.write_text(json.dumps(a["extra_specs"]))
        args += ["--extra-specs", str(spec_file)]
    p = subprocess.run(args, capture_output=True, text=True, check=False)
    if p.returncode:
        return {"ok": False, "stderr": p.stderr[-2000:], "stdout": p.stdout[-2000:]}
    return {"ok": True, **tool_leaderboard({"out": str(out)}),
            "log_tail": [line for line in p.stdout.splitlines() if line.startswith(("[seeds]", "[extra]", "[verdict]", "[objective]"))]}  # fmt: skip


def tool_leaderboard(a):
    out = Path(a["out"])
    b = json.loads((out / "leaderboard.json").read_text())
    return {
        "leaderboard_md": (out / "leaderboard.md").read_text()[:12000],
        "verdict": b.get("verdict"),
        "top_k": [{k: r.get(k) for k in ("name", "origin", "params", "flops", "val_mean", "val_std", "test")}
                  for r in b.get("top_k", [])],
    }  # fmt: skip


HANDLERS = {
    "dlnet_profile": tool_profile, "dlnet_candidates": tool_candidates, "dlnet_zoo": tool_zoo,
    "dlnet_check_spec": tool_check, "dlnet_design": tool_design, "dlnet_leaderboard": tool_leaderboard,
}  # fmt: skip


def handle(msg: dict) -> dict | None:
    """One JSON-RPC message -> response (None for notifications)."""
    from . import __version__

    mid, method = msg.get("id"), msg.get("method")
    if mid is None:  # notification (e.g. notifications/initialized)
        return None
    if method == "initialize":
        version = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL
        result = {"protocolVersion": version, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "dlnet", "version": __version__}}  # fmt: skip
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        fn = HANDLERS.get(params.get("name"))
        if fn is None:
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32602, "message": f"unknown tool {params.get('name')!r}"},
            }
        try:
            out = fn(params.get("arguments") or {})
            result = {
                "content": [{"type": "text", "text": jsonsafe.dumps(out, default=str)}],
                "isError": False,
            }
        except (OSError, KeyError, ValueError, TypeError, RuntimeError, SystemExit) as e:
            result = {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}
    else:
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(stdin=sys.stdin, stdout=sys.stdout) -> int:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }
        else:
            with redirect_stdout(sys.stderr):  # nothing but JSON-RPC may reach stdout
                resp = handle(msg) if isinstance(msg, dict) else None
        if resp is not None:
            stdout.write(jsonsafe.dumps(resp, default=str) + "\n")
            stdout.flush()
    return 0
