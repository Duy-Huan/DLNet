"""Optional AI advisor: an external AI CLI (Claude Code, Codex, Gemini / Antigravity, or any
command) takes part in seed selection and in evolution. OFF unless `--advisor` is given.

How it works
  * DLNet writes a prompt (data card, context, objective, the candidate list with traits, size
    and its own rankings; later the rung-1 results) and runs the advisor command with the prompt
    on stdin (or in a file, `{prompt_file}` in the command). The command must print JSON:
        {"ranking": ["c3", "c0", ...], "proposals": [<zoo spec>, ...], "rationale": "..."}
  * The reply is DATA, never code: only candidate ids from the list and zoo specs are read;
    every proposed spec is checked statically (zoo.check) and costed like any other candidate.
    Unknown ids, invalid specs and anything else are dropped and reported.
  * The ranking becomes one more ranker in the Borda count of dlnet/seeds.py
    (`--advisor-weight`, default 1); proposals join the pool / the evolution children.
  * Any failure (command missing, timeout, unparsable reply) is logged and the run continues
    without the advisor.

Presets (`--advisor NAME`); check the flags against your installed version:
  claude   claude -p --output-format text          (Claude Code, prompt on stdin)
  codex    codex exec --skip-git-repo-check -      (Codex CLI, prompt on stdin)
  gemini   gemini -p {prompt}                      (Gemini CLI; Antigravity: use --advisor-cmd)
Anything else: `--advisor-cmd "my-tool --flag {prompt_file}"`.
Nothing here was run against a real AI in DLNet's tests: they use a scripted fake advisor.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

PRESETS = {
    "claude": "claude -p --output-format text",
    "codex": "codex exec --skip-git-repo-check -",
    "gemini": "gemini -p {prompt}",
}

INSTRUCTIONS = """You are advising DLNet, an architecture-search tool, on which neural network
candidates to train for the dataset below. Reply with ONE JSON object and nothing else:
{"ranking": [candidate ids, most promising first; you may omit ids],
 "proposals": [optional new architectures as DLNet zoo specs, at most %d],
 "rationale": "one short paragraph"}
A zoo spec is a JSON tree: {"op": "seq", "body": [...]}, residual {"op": "res", "body": node},
leaf ops with parameters, e.g. {"op": "conv_bn_act", "out": 32, "k": 3, "s": 2},
{"op": "basic_block", "out": 64, "s": 1}, {"op": "se", "ratio": 0.25}. Valid leaf ops and their
parameters for this input kind are listed below. Do not include the classification head.
Judge from the data card: size, resolution, texture vs shape, position dependence, colour.
"""


def command(name: str | None, cmd: str | None) -> str | None:
    if cmd:
        return cmd
    if name:
        if name not in PRESETS:
            raise SystemExit(f"unknown --advisor {name!r}; use {sorted(PRESETS)} or --advisor-cmd")
        return PRESETS[name]
    return None


def candidate_table(cands: list[dict]) -> list[dict]:
    from .seeds import traits
    from .zoo import leaf_ops

    rows = []
    for i, c in enumerate(cands):
        spec = c.get("spec")
        rows.append({
            "id": f"c{i}", "family": c.get("family"), "origin": c.get("origin", "family"),
            "traits": sorted(traits(c)), "params": c.get("params"), "flops": c.get("flops"),
            "modules": sorted(set(leaf_ops(spec)))[:12] if spec else [],
            "dlnet_prior": c.get("seed_prior"),
        })  # fmt: skip
    return rows


def ops_catalogue(kind: str) -> list[dict]:
    from .zoo import OPS

    return [
        {"op": n, "params": {k: v for k, v in o.space.items()}, "tags": sorted(o.tags)}
        for n, o in OPS.items()
        if kind in o.kin and o.tags & {"block", "down", "attn", "stem"}
    ][:60]


def build_prompt(card, context, objective, cands, kind, n_proposals=3, results=None) -> str:
    payload = {
        "data_card": card, "context": context or "", "objective": objective,
        "candidates": candidate_table(cands), "zoo_ops": ops_catalogue(kind),
    }  # fmt: skip
    if results:
        payload["rung1_results"] = results
    return (INSTRUCTIONS % n_proposals) + "\n" + json.dumps(payload, default=str)


def run(cmd: str, prompt: str, timeout: float = 120.0) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "prompt.txt"
        pf.write_text(prompt)
        args = [
            a.replace("{prompt_file}", str(pf)).replace("{prompt}", prompt)
            for a in shlex.split(cmd)
        ]
        uses_arg = "{prompt_file}" in cmd or "{prompt}" in cmd
        p = subprocess.run(
            args, input=None if uses_arg else prompt, capture_output=True, text=True,
            timeout=timeout, check=False,
        )  # fmt: skip
    if p.returncode:
        raise RuntimeError(f"advisor exited with {p.returncode}: {p.stderr[-300:]}")
    return p.stdout


def parse(text: str) -> dict:
    """The first JSON object in the reply (AI CLIs often wrap it in prose or ``` fences)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else text[text.find("{") : text.rfind("}") + 1]
    d = json.loads(blob)
    if not isinstance(d, dict):
        raise TypeError("advisor reply is not a JSON object")
    return d


def validate(reply: dict, cands, kind, dims, out_dim, head, train, max_params, rng) -> dict:
    """Keep only known candidate ids and statically valid specs; report what was dropped."""
    from .zoo import space

    ids = {f"c{i}": i for i in range(len(cands))}
    raw = reply.get("ranking") or []
    raw = raw if isinstance(raw, list) else []
    ranking = [r for r in raw if isinstance(r, str) and r in ids]
    dropped = [r for r in raw if not (isinstance(r, str) and r in ids)]
    proposals, bad = [], 0
    for spec in (reply.get("proposals") or [])[:10]:
        try:  # an AI reply is untrusted data: any malformed spec is simply dropped
            c = (
                space._candidate(spec, kind, dims, out_dim, head, train, rng, max_params, "ai_advisor")
                if isinstance(spec, dict) else None
            )  # fmt: skip
        except (KeyError, TypeError, AttributeError, IndexError, RuntimeError, ValueError):
            c = None
        if c:
            proposals.append(c)
        else:
            bad += 1
    return {
        "ranking": [ids[r] for r in dict.fromkeys(ranking)], "proposals": proposals,
        "rationale": str(reply.get("rationale", ""))[:2000],
        "dropped_ids": dropped[:20], "invalid_proposals": bad,
    }  # fmt: skip


def scores_from_ranking(n: int, ranking: list[int]) -> list[float]:
    """Borda-ready scores: ranked candidates first (in order), unranked ones tie at the bottom."""
    s = [-float(n)] * n
    for pos, i in enumerate(ranking):
        s[i] = -float(pos)
    return s


def advise(cmd, card, context, objective, cands, kind, dims, out_dim, head, train, max_params,
           rng, n_proposals=3, results=None, timeout=120.0, log=print) -> dict | None:  # fmt: skip
    """Ask the advisor; None on any failure (the run continues without it)."""
    try:
        text = run(
            cmd, build_prompt(card, context, objective, cands, kind, n_proposals, results), timeout
        )
        out = validate(parse(text), cands, kind, dims, out_dim, head, train, max_params, rng)
    except (OSError, RuntimeError, ValueError, TypeError, subprocess.TimeoutExpired) as e:
        log(f"[advisor] skipped: {str(e)[:200]}")
        return None
    log(f"[advisor] ranked {len(out['ranking'])} candidates, {len(out['proposals'])} valid proposals"
        f" ({out['invalid_proposals']} invalid, {len(out['dropped_ids'])} unknown ids)")  # fmt: skip
    if out["rationale"]:
        log(f"[advisor] rationale: {out['rationale'][:300]}")
    return out
