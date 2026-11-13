"""`dlnet serve` against the OFFICIAL MCP Python SDK as the client (job `mcp-sdk`).

    pip install "mcp==2.3.0"
    python scripts/verify_mcp.py

Starts the server on a free port with a temporary workspace and a token, then connects twice:
  legacy  ClientSession over streamable HTTP, `initialize` handshake (revision 2025-11-25)
  modern  the SDK's dual-era Client in auto mode, which must select the stateless 2026-07-28
and runs the whole service flow: list tools and prompts, profile, check a spec, design with
progress notifications over SSE, poll the job, leaderboard, read model.py, train, predict,
archive, HTTP download and upload. The SDK validates every response against its schema, so a
malformed result fails here.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

TOKEN = "verify-mcp-token"
FINAL = ("succeeded", "failed", "cancelled", "lost")


def attr(o, *names):
    for n in names:  # SDK 1.x: camelCase, SDK 2.x: snake_case
        if hasattr(o, n):
            return getattr(o, n)
    raise AttributeError(names)


def body(r):
    assert not attr(r, "isError", "is_error"), r.content[0].text
    return json.loads(r.content[0].text)


async def legacy(url, http):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with (
        streamable_http_client(url, http_client=http) as streams,
        ClientSession(streams[0], streams[1]) as s,
    ):
        init = await s.initialize()
        version = attr(init, "protocolVersion", "protocol_version")
        assert version == "2025-11-25", version
        tools = (await s.list_tools()).tools
        assert len(tools) >= 19 and {
            "dlnet_design",
            "dlnet_train",
            "dlnet_predict",
            "dlnet_job",
        } <= {t.name for t in tools}
        assert len((await s.list_prompts()).prompts) == 3
        assert body(await s.call_tool("dlnet_info", {}))["zoo"]["modules"] > 150
        prof = body(await s.call_tool("dlnet_profile", {"data": "data/iris.csv", "target": "y"}))
        assert prof["plugin"] == "tabular" and prof["profile"]["task"] == "classification"
        bad = await s.call_tool("dlnet_profile", {"data": "../../etc/passwd"})
        assert attr(bad, "isError", "is_error") and "outside the workspace" in bad.content[0].text
        assert body(
            await s.call_tool(
                "dlnet_check_spec", {"spec": {"op": "tide_block"}, "kind": "vec", "dims": [4]}
            )
        )["valid"]
        seen = []

        async def progress(done, total, message):
            seen.append(message)

        d = body(await s.call_tool("dlnet_design", {"data": "data/iris.csv", "target": "y", "name": "legacy-run",
                                                    "budget": "60ep", "topk": 2, "zoo": 2, "evolve": 0, "per_family": 1,
                                                    "device": "cpu", "wait": 5}, progress_callback=progress))  # fmt: skip
        while d["status"] not in FINAL:
            d = body(
                await s.call_tool(
                    "dlnet_job", {"job_id": d["job_id"], "wait": 20}, progress_callback=progress
                )
            )
        assert d["status"] == "succeeded" and d["top_k"], d
        assert seen, "no progress notification arrived over SSE"
        pkg = body(await s.call_tool("dlnet_leaderboard", {"run": "legacy-run"}))["top_k"][0][
            "package"
        ]
        src = body(await s.call_tool("dlnet_read_file", {"path": pkg + "/model.py", "limit": 4000}))
        assert "import torch" in src["text"] and "dlnet" not in src["text"].split('"""', 2)[2]
        t = body(await s.call_tool("dlnet_train", {"package": pkg, "data": "data/iris.csv", "target": "y", "epochs": 20,
                                                   "device": "cpu", "wait": 300}))  # fmt: skip
        assert t["status"] == "succeeded" and 0 <= t["test"]["score"] <= 1, t
        p = body(await s.call_tool("dlnet_predict", {"package": pkg, "data": "data/iris.csv",
                                                     "out": "runs/legacy-run/pred.csv", "wait": 300}))  # fmt: skip
        assert p["status"] == "succeeded" and p["bytes"] > 0, p
        z = body(await s.call_tool("dlnet_archive", {"path": pkg}))
        r = await http.get(z["url"])
        assert r.status_code == 200 and r.content[:2] == b"PK"
        up = await http.put(url.replace("/mcp", "/files/data/up.csv"), content=b"a,y\n1,0\n2,1\n")
        assert up.status_code == 201 and up.json()["bytes"] == 12
        g = await s.get_prompt("design_baselines", {"data": "data/iris.csv"})
        assert "dlnet_design" in g.messages[0].content.text
        print(f"legacy  {version}: {len(tools)} tools, design -> train (test {t['test']['score']:.3f}) -> predict -> "
              f"download, {len(seen)} progress events  OK")  # fmt: skip


async def modern(url, http):
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    async with Client(streamable_http_client(url, http_client=http), raise_exceptions=True) as c:
        assert c.protocol_version == "2026-07-28", c.protocol_version
        assert c.server_info.name == "dlnet"
        tools = (await c.list_tools()).tools
        assert len(tools) >= 19
        r = await c.call_tool(
            "dlnet_check_spec", {"spec": {"op": "gem"}, "kind": "map", "dims": [8, 6, 6]}
        )
        assert json.loads(r.content[0].text)["valid"]
        seen = []

        async def progress(done, total, message):
            seen.append(done)

        r = await c.call_tool("dlnet_design", {"data": "data/iris.csv", "target": "y", "name": "modern-run",
                                               "budget": "40ep", "topk": 1, "zoo": 2, "evolve": 0, "per_family": 1,
                                               "device": "cpu", "wait": 600}, progress_callback=progress)  # fmt: skip
        d = json.loads(r.content[0].text)
        assert d["status"] == "succeeded" and d["top_k"], d
        assert seen == sorted(seen) and seen, "progress must arrive and increase"
        g = await c.get_prompt("propose_architecture", {"data": "data/iris.csv"})
        assert g.messages
        print(f"modern  {c.protocol_version}: {len(tools)} tools, design {d['top_k'][0]['name']} "
              f"(val {d['top_k'][0]['val_mean']}), {len(seen)} progress events  OK")  # fmt: skip


async def run(url):
    import httpx2 as httpx

    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {TOKEN}"}, timeout=700) as http:
        await legacy(url, http)
        await modern(url, http)


def main() -> int:
    from sklearn.datasets import load_iris

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        (ws / "data").mkdir(parents=True)
        X, y = load_iris(return_X_y=True)
        with open(ws / "data" / "iris.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["a", "b", "c", "d", "y"])
            w.writerows([*x, int(t)] for x, t in zip(X, y, strict=True))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        log = open(Path(tmp) / "serve.log", "w")  # noqa: SIM115 (closed below)
        srv = subprocess.Popen([sys.executable, "-m", "dlnet", "serve", "--port", str(port), "--workspace", str(ws)],
                               env={**os.environ, "DLNET_TOKEN": TOKEN}, stdout=log, stderr=subprocess.STDOUT)  # fmt: skip
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2).read()
                    break
                except OSError:
                    time.sleep(0.2)
            else:
                raise RuntimeError("the server did not start")
            asyncio.run(run(f"http://127.0.0.1:{port}/mcp"))
        except BaseException:
            log.flush()
            print(Path(tmp, "serve.log").read_text()[-3000:], file=sys.stderr)
            raise
        finally:
            srv.terminate()
            srv.wait(20)
            log.close()
    print("MCP SDK CLIENT: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
