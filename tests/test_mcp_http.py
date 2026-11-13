"""`dlnet serve`: the DLNet MCP server over Streamable HTTP (dlnet/mcp_http.py, mcp_tools.py, service.py).

A real server on a free port, a plain http.client on the other side. What is checked is what the
MCP specification asks of a server (revisions 2025-03-26 ... 2025-11-25 with sessions, and the
stateless 2026-07-28) and what a service needs: authentication, the workspace sandbox, jobs.
The same server against the official MCP SDK client: scripts/verify_mcp.py (CI job `mcp-sdk`).
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import sys
import threading
import time
import zipfile

import pytest

from dlnet import mcp_http, mcp_server, mcp_tools, service

TOKEN = "test-token-123"
V = "io.modelcontextprotocol/protocolVersion"


class Client:
    def __init__(self, httpd, token=TOKEN):
        self.addr, self.token, self.sid = httpd.server_address, token, None

    def request(self, method, path, body=None, headers=None, raw=False):
        h = {"Host": f"127.0.0.1:{self.addr[1]}"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
            h["Accept"] = "application/json, text/event-stream"
        h.update(headers or {})
        c = http.client.HTTPConnection("127.0.0.1", self.addr[1], timeout=60)
        c.request(method, path, body=body, headers={k: v for k, v in h.items() if v is not None})
        r = c.getresponse()
        data = r.read()
        c.close()
        if raw:
            return r, data
        ctype = r.getheader("Content-Type") or ""
        return r.status, (json.loads(data) if data and "json" in ctype else data), r

    def rpc(self, method, params=None, mid=1, headers=None):
        msg = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params is not None:
            msg["params"] = params
        h = {"Mcp-Session-Id": self.sid, **(headers or {})}
        return self.request("POST", "/mcp", msg, h)

    def initialize(self, version="2025-11-25"):
        code, body, r = self.rpc("initialize", {"protocolVersion": version, "capabilities": {},
                                                "clientInfo": {"name": "t", "version": "0"}})  # fmt: skip
        assert code == 200, body
        self.sid = r.getheader("Mcp-Session-Id")
        self.request("POST", "/mcp", {"jsonrpc": "2.0", "method": "notifications/initialized"},
                     {"Mcp-Session-Id": self.sid})  # fmt: skip
        return body["result"]

    def call(self, name, args=None, **kw):
        code, body, _ = self.rpc("tools/call", {"name": name, "arguments": args or {}}, **kw)
        assert code == 200, body
        res = body["result"]
        return res, (
            json.loads(res["content"][0]["text"])
            if not res["isError"]
            else res["content"][0]["text"]
        )

    def modern(self, method, params=None, mid=1, headers=None, version="2026-07-28"):
        params = {**(params or {}), "_meta": {V: version, **(params or {}).get("_meta", {})}}
        h = {"MCP-Protocol-Version": version, "Mcp-Method": method}
        if "name" in params:
            h["Mcp-Name"] = params["name"]
        h.update(headers or {})
        return self.request(
            "POST", "/mcp", {"jsonrpc": "2.0", "id": mid, "method": method, "params": params}, h
        )


@pytest.fixture
def server(tmp_path):
    svc = mcp_tools.Service(tmp_path / "ws", max_jobs=1, default_wait=30)
    httpd = mcp_http.make_server(svc, mcp_http.Config(port=0, token=TOKEN, quiet=True))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    svc.close()


# ------------------------------------------------------------------------- guard


def test_requests_need_the_token_an_allowed_origin_and_a_loopback_host(server):
    assert Client(server, token=None).rpc("ping")[0] == 401
    code, _, r = Client(server, token="wrong").rpc("ping")
    assert code == 401 and "Bearer" in r.getheader("WWW-Authenticate")
    c = Client(server)
    assert c.rpc("ping", headers={"Origin": "https://evil.example"})[0] == 403  # DNS rebinding
    assert c.rpc("ping", headers={"Host": "evil.example"})[0] == 403
    c.initialize()
    assert c.rpc("ping", headers={"Origin": "http://localhost:3000"})[0] == 200
    code, body, _ = Client(server, token=None).request("GET", "/healthz")
    assert code == 200 and body["status"] == "ok" and "version" in body
    assert Client(server, token=None).request("GET", "/files/data")[0] == 401


def test_a_public_bind_without_a_token_is_refused(tmp_path):
    svc = mcp_tools.Service(tmp_path / "ws")
    with pytest.raises(ValueError, match="without authentication"):
        mcp_http.make_server(svc, mcp_http.Config(host="0.0.0.0", port=0, token=None))


def test_serve_command_reads_the_token_and_reports_a_bad_bind(tmp_path, monkeypatch, capsys):
    from dlnet import cli

    monkeypatch.setenv("DLNET_TOKEN", "from-env")
    f = tmp_path / "tok"
    f.write_text("from-file\n")
    ns = lambda **kw: type("A", (), {"token": None, "token_file": None, **kw})()
    assert mcp_http.read_token(ns()) == "from-env"
    assert mcp_http.read_token(ns(token_file=str(f))) == "from-file"
    assert mcp_http.read_token(ns(token="arg")) == "arg"
    monkeypatch.delenv("DLNET_TOKEN")
    rc = cli.main(["serve", "--host", "0.0.0.0", "--port", "0", "--workspace", str(tmp_path / "w")])
    assert rc == 2 and "without authentication" in capsys.readouterr().err


# ------------------------------------------------------------------------- legacy era (sessions)


def test_legacy_handshake_sessions_and_methods(server):
    c = Client(server)
    assert c.rpc("tools/list")[0] == 400  # no session yet
    init = c.initialize("2025-06-18")
    assert init["protocolVersion"] == "2025-06-18" and init["serverInfo"]["name"] == "dlnet"
    assert (
        init["capabilities"]["tools"] == {"listChanged": False}
        and "dlnet_design" in init["instructions"]
    )
    assert len(c.sid) >= 32 and c.sid.isascii()
    assert (
        Client(server).initialize("1999-01-01")["protocolVersion"] == mcp_server.PROTOCOL
    )  # negotiation
    code, body, _ = c.rpc("tools/list")
    tools = body["result"]["tools"]
    assert (
        code == 200
        and len(tools) == len(mcp_tools.TOOLS) >= 19
        and "resultType" not in body["result"]
    )
    for t in tools:
        assert (
            t["inputSchema"]["type"] == "object"
            and t["description"]
            and "readOnlyHint" in t["annotations"]
        )
        assert hasattr(mcp_tools.Service, "tool_" + t["name"].removeprefix("dlnet_")), t["name"]
        assert set(t["inputSchema"]["required"]) <= set(t["inputSchema"]["properties"])
    assert c.rpc("ping")[1]["result"] == {}
    assert c.rpc("nope")[1]["error"]["code"] == -32601
    assert c.rpc("tools/call", {"name": "nope"})[1]["error"]["code"] == -32602
    code, body, _ = c.request("POST", "/mcp", {"jsonrpc": "2.0", "method": "notifications/initialized"},
                              {"Mcp-Session-Id": c.sid})  # fmt: skip
    assert code == 202 and body == b""
    assert c.rpc("ping", headers={"Mcp-Session-Id": "unknown"})[0] == 404
    code, body, _ = c.rpc("ping", headers={"MCP-Protocol-Version": "1999-01-01"})
    assert (
        code == 400
        and body["error"]["code"] == -32022
        and "2025-11-25" in body["error"]["data"]["supported"]
    )
    code, _, r = c.request("GET", "/mcp")
    assert code == 405 and "POST" in r.getheader("Allow")
    # a 2025-03-26 batch: two requests and a notification -> two responses
    batch = [{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/x"},
             {"jsonrpc": "2.0", "id": 2, "method": "prompts/list"}]  # fmt: skip
    code, body, _ = c.request("POST", "/mcp", batch, {"Mcp-Session-Id": c.sid})
    assert (
        code == 200 and [m["id"] for m in body] == [1, 2] and len(body[1]["result"]["prompts"]) == 3
    )
    assert c.request("DELETE", "/mcp", headers={"Mcp-Session-Id": c.sid})[0] == 204
    assert c.rpc("ping")[0] == 404  # the session is gone


def test_malformed_posts_get_http_errors(server):
    c = Client(server)
    c.initialize()
    h = {"Mcp-Session-Id": c.sid, "Content-Type": "application/json"}
    assert c.request("POST", "/mcp", b"{not json", h)[1]["error"]["code"] == -32700
    assert c.request("POST", "/mcp", b"[]", h)[0] == 400
    assert c.request("POST", "/mcp", b"{}", {**h, "Content-Type": "text/plain"})[0] == 415
    assert c.request("POST", "/elsewhere", {"jsonrpc": "2.0", "id": 1, "method": "ping"})[0] == 404
    r, _ = c.request(
        "POST", "/mcp", None, {**h, "Content-Length": str(mcp_http.MAX_BODY + 1)}, raw=True
    )
    assert r.status == 413


# ------------------------------------------------------------------------- modern era (stateless)


def test_modern_requests_need_no_session_and_carry_result_type(server):
    c = Client(server)
    code, body, r = c.modern("server/discover")
    res = body["result"]
    assert code == 200 and r.getheader("Mcp-Session-Id") is None
    assert res["resultType"] == "complete" and "2026-07-28" in res["supportedVersions"]
    assert (
        res["capabilities"]["tools"] is not None
        and res["ttlMs"] > 0
        and res["cacheScope"] == "public"
    )
    assert res["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "dlnet"
    code, body, _ = c.modern("tools/list")
    assert (
        code == 200 and body["result"]["resultType"] == "complete" and body["result"]["ttlMs"] > 0
    )
    code, body, _ = c.modern("tools/call", {"name": "dlnet_check_spec", "arguments": {
        "spec": {"op": "tide_block"}, "kind": "vec", "dims": [4]}})  # fmt: skip
    res = body["result"]
    assert (
        code == 200
        and res["resultType"] == "complete"
        and res["structuredContent"]["valid"] is True
    )
    assert json.loads(res["content"][0]["text"]) == res["structuredContent"]
    code, body, _ = c.modern(
        "prompts/get", {"name": "design_baselines", "arguments": {"data": "data/x.csv"}}
    )
    assert code == 200 and "data/x.csv" in body["result"]["messages"][0]["content"]["text"]
    code, body, _ = c.modern("ping")  # removed from the modern protocol
    assert code == 404 and body["error"]["code"] == -32601
    code, body, _ = c.modern("tools/list", version="2030-01-01")
    assert (
        code == 400
        and body["error"]["code"] == -32022
        and body["error"]["data"]["requested"] == "2030-01-01"
    )


def test_modern_headers_must_mirror_the_body(server):
    c = Client(server)
    call = {"name": "dlnet_zoo", "arguments": {"kind": "vec"}}
    for headers, word in (
        ({"Mcp-Method": None}, "Mcp-Method header is missing"),
        ({"Mcp-Method": "tools/list"}, "Mcp-Method"),
        ({"Mcp-Name": None}, "Mcp-Name header is missing"),
        ({"Mcp-Name": "dlnet_design"}, "does not match body value 'dlnet_zoo'"),
        ({"Mcp-Name": "=?base64?!!!?="}, "Base64"),
    ):
        code, body, _ = c.modern("tools/call", call, headers=headers)
        assert (
            code == 400 and body["error"]["code"] == -32020 and word in body["error"]["message"]
        ), headers
    encoded = "=?base64?" + base64.b64encode(b"dlnet_zoo").decode() + "?="
    assert c.modern("tools/call", call, headers={"Mcp-Name": encoded})[0] == 200
    # header says modern, body says nothing: mismatch
    code, body, _ = c.request("POST", "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                              {"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/list"})  # fmt: skip
    assert code == 400 and body["error"]["code"] == -32020


# ------------------------------------------------------------------------- workspace and files


def test_workspace_paths_cannot_leave_the_workspace(tmp_path):
    ws = service.Workspace(tmp_path / "ws")
    assert (
        ws.path("data/a.csv") == ws.root / "data" / "a.csv" and ws.rel(ws.root / "runs") == "runs"
    )
    for bad in ("../x", "/etc/passwd", "data/../../x", "", "a\x00b"):
        with pytest.raises(service.ServiceError):
            ws.path(bad)
    (tmp_path / "secret.txt").write_text("s")
    os.symlink(tmp_path / "secret.txt", ws.root / "data" / "link")
    with pytest.raises(service.ServiceError, match="outside"):
        ws.path("data/link")
    with pytest.raises(service.ServiceError, match="no such file"):
        ws.path("data/none.csv", must_exist=True)
    assert (
        service.Workspace(tmp_path / "ws", sandbox=False)
        .path(str(tmp_path / "secret.txt"))
        .exists()
    )
    for bad in ("a b", "../x", "x" * 101, ".hidden"):
        with pytest.raises(service.ServiceError):
            ws.run_dir(bad)
    assert ws.run_dir(None).parent == ws.root / "runs" and ws.run_dir("exp-1").name == "exp-1"


def test_file_tools_and_http_transfer(server):
    c = Client(server)
    c.initialize()
    res, body = c.call("dlnet_write_file", {"path": "data/a.csv", "text": "x,y\n1,0\n2,1\n"})
    assert not res["isError"] and body == {"path": "data/a.csv", "bytes": 12}
    assert "exists" in c.call("dlnet_write_file", {"path": "data/a.csv", "text": "z"})[1]
    assert (
        c.call("dlnet_write_file", {"path": "data/a.csv", "text": "z", "overwrite": True})[1][
            "bytes"
        ]
        == 1
    )
    blob = base64.b64encode(bytes(range(256))).decode()
    assert c.call("dlnet_write_file", {"path": "data/b.bin", "base64": blob})[1]["bytes"] == 256
    for args, word in (({"path": "../x", "text": "a"}, "outside"), ({"path": "jobs/x", "text": "a"}, "jobs/"),
                       ({"path": "data/c"}, "exactly one"), ({"path": "data/c", "base64": "@@"}, "invalid"),
                       ({"path": "data/c", "text": "a", "bogus": 1}, "unknown argument"), ({}, "missing required")):  # fmt: skip
        res, text = c.call("dlnet_write_file", args)
        assert res["isError"] and word in text, args
    files = c.call("dlnet_files", {"folder": "data"})[1]
    assert {e["path"] for e in files["entries"]} == {"data/a.csv", "data/b.bin"}
    assert c.call("dlnet_read_file", {"path": "data/a.csv"})[1]["text"] == "z"
    assert "binary" in c.call("dlnet_read_file", {"path": "data/b.bin"})[1]
    assert "outside" in c.call("dlnet_read_file", {"path": "/etc/passwd"})[1]
    # HTTP upload / download
    payload = os.urandom(300_000)
    code, body, _ = c.request(
        "PUT", "/files/data/big.npy", payload, {"Content-Type": "application/octet-stream"}
    )
    assert code == 201 and body == {"path": "data/big.npy", "bytes": 300_000}
    assert c.request("PUT", "/files/data/big.npy", b"x")[0] == 200  # replaced
    assert c.request("PUT", "/files/data/big.npy", b"x", {"If-None-Match": "*"})[0] == 409
    r, data = c.request("GET", "/files/data/b.bin", raw=True)
    assert (
        r.status == 200
        and data == bytes(range(256))
        and "attachment" in r.getheader("Content-Disposition")
    )
    assert c.request("GET", "/files/data")[1]["entries"]
    assert c.request("GET", "/files/data/none")[0] == 404
    assert c.request("GET", "/files/%2e%2e/%2e%2e/etc/passwd")[0] == 404
    assert (
        c.request("PUT", "/files/../../x", b"x")[0] == 400
        and c.request("PUT", "/files/jobs/x.json", b"x")[0] == 400
    )
    server.cfg.max_upload = 10
    r, _ = c.request("PUT", "/files/data/too-big", None, {"Content-Length": "11"}, raw=True)
    assert r.status == 413
    res, text = c.call("dlnet_fetch", {"url": "https://example.com/a.csv"})
    assert res["isError"] and "disabled" in text


def test_info_zoo_and_check_spec(server):
    c = Client(server)
    c.initialize()
    info = c.call("dlnet_info")[1]
    assert info["zoo"]["modules"] > 150 and "tabular" in info["zoo"]["families"]
    assert info["workspace"]["sandboxed"] is True and info["http"]["upload"].startswith(
        "PUT http://127.0.0.1"
    )
    assert any(o["op"] == "tide_block" for o in c.call("dlnet_zoo", {"kind": "vec"})[1]["ops"])
    assert "one of" in c.call("dlnet_zoo", {"kind": "nope"})[1]
    ok = c.call("dlnet_check_spec", {"spec": {"op": "conv_bn_act", "out": 8}, "kind": "map", "dims": [3, 8, 8],
                                     "diagram": True})[1]  # fmt: skip
    assert ok["valid"] and ok["dims"] == [8, 8, 8] and ok["diagram_svg"].startswith("<svg")
    bad = c.call("dlnet_check_spec", {"spec": {"op": "nope"}, "kind": "map", "dims": [3, 8, 8]})[1]
    assert bad["valid"] is False and "nope" in bad["error"]
    assert (
        "over this server's limit"
        in c.call("dlnet_design", {"data": "data/x.csv", "budget": "99h"})[1]
    )
    assert "90s, 10m" in c.call("dlnet_design", {"data": "data/x.csv", "budget": "soon"})[1]


# ------------------------------------------------------------------------- jobs


def _py(code):
    return [sys.executable, "-c", code]


def test_jobs_queue_run_cancel_and_survive_a_restart(tmp_path):
    ws = service.Workspace(tmp_path / "ws")
    jobs = service.Jobs(ws, max_jobs=1)
    a = jobs.submit(
        "demo", _py("import time; print('start', flush=True); time.sleep(30)"), title="slow"
    )
    b = jobs.submit("demo", _py("print('hello from b')"))
    c = jobs.submit("demo", _py("import sys; print('boom'); sys.exit(3)"))
    deadline = time.time() + 20
    while a.status != "running" and time.time() < deadline:
        time.sleep(0.05)
    assert a.status == "running" and b.status == "queued"
    assert jobs.describe(b)["queue_position"] == 1 and jobs.describe(c)["queue_position"] == 2
    t0 = time.time()
    assert (
        jobs.cancel(a.id).status == "cancelled" and time.time() - t0 < 15
    )  # the process group is killed
    assert b.done.wait(30) and b.status == "succeeded" and jobs.log_tail(b) == ["hello from b"]
    assert (
        c.done.wait(30) and c.status == "failed" and c.returncode == 3 and "exit code 3" in c.error
    )
    assert jobs.cancel(b.id).status == "succeeded"  # cancelling a finished job changes nothing
    with pytest.raises(service.ServiceError, match="unknown job"):
        jobs.get("job-none")
    assert [j["job_id"] for j in jobs.listing()] == [c.id, b.id, a.id]
    # a job that was running when the server died is reported as lost, not as running for ever
    state = {**a.state(), "id": "job-ghost", "status": "running", "finished": None}
    (ws.root / "jobs" / "job-ghost.json").write_text(json.dumps(state))
    again = service.Jobs(ws, max_jobs=1)
    assert again.get("job-ghost").status == "lost" and again.get(b.id).status == "succeeded"
    assert json.loads((ws.root / "jobs" / "job-ghost.json").read_text())["status"] == "lost"
    bad = again.submit("demo", ["/no/such/program"])
    assert bad.done.wait(10) and bad.status == "failed" and "could not start" in bad.error


def test_a_slow_call_returns_the_job_id_and_can_be_polled_and_cancelled(server):
    c = Client(server)
    c.initialize()
    svc = server.proto.svc
    job = svc.jobs.submit(
        "demo",
        _py("import time\nfor i in range(100):\n  print('step', i, flush=True); time.sleep(0.2)"),
    )
    _, body = c.call("dlnet_job", {"job_id": job.id, "wait": 1})
    assert body["status"] == "running" and body["ok"] is None and job.id in body["next"]
    assert any(line.startswith("step") for line in body["log_tail"])
    assert c.call("dlnet_jobs")[1]["jobs"][0]["job_id"] == job.id
    assert c.call("dlnet_cancel", {"job_id": job.id})[1]["status"] == "cancelled"
    _, body = c.call("dlnet_job", {"job_id": job.id})
    assert body["status"] == "cancelled" and body["ok"] is False
    assert "unknown job" in c.call("dlnet_job", {"job_id": "job-x"})[1]


def _sse(raw: bytes):
    events = []
    for block in raw.decode().split("\n\n"):
        ev = {}
        for line in block.splitlines():
            if line.startswith(":"):
                ev["comment"] = line[1:].strip()
            elif ":" in line:
                k, v = line.split(":", 1)
                ev[k] = v.strip()
        if ev:
            events.append(ev)
    return events


@pytest.mark.parametrize("era", ["legacy", "modern"])
def test_a_call_with_a_progress_token_is_answered_as_an_sse_stream(server, era, monkeypatch):
    monkeypatch.setattr(mcp_http, "KEEPALIVE", 0.3)
    c = Client(server)
    svc = server.proto.svc
    job = svc.jobs.submit(
        "demo",
        _py("import time\nfor i in range(6):\n  print('step', i, flush=True); time.sleep(0.4)"),
    )
    params = {
        "name": "dlnet_job",
        "arguments": {"job_id": job.id, "wait": 30},
        "_meta": {"progressToken": "tok-1"},
    }
    msg = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": params}
    if era == "legacy":
        c.initialize()
        headers = {"Mcp-Session-Id": c.sid}
    else:
        params["_meta"][V] = "2026-07-28"
        headers = {
            "MCP-Protocol-Version": "2026-07-28",
            "Mcp-Method": "tools/call",
            "Mcp-Name": "dlnet_job",
        }
    r, raw = c.request("POST", "/mcp", msg, headers, raw=True)
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    assert r.getheader("X-Accel-Buffering") == "no"
    events = _sse(raw)
    assert events[0].get("id") and events[0].get("data", "") == ""  # priming event
    assert len({e["id"] for e in events if "id" in e}) == len([e for e in events if "id" in e])
    assert any(e.get("comment") == "keep-alive" for e in events)
    msgs = [json.loads(e["data"]) for e in events if e.get("data")]
    prog = [m["params"] for m in msgs if m.get("method") == "notifications/progress"]
    assert prog and all(p["progressToken"] == "tok-1" for p in prog)
    values = [p["progress"] for p in prog]
    assert values == sorted(set(values))  # strictly increasing
    assert any("step" in p.get("message", "") for p in prog)
    final = msgs[-1]
    assert final["id"] == 7 and final["result"]["structuredContent"]["status"] == "succeeded"
    assert ("resultType" in final["result"]) == (era == "modern")
    # without a progress token the same call is a plain JSON response
    params["_meta"].pop("progressToken")
    r, raw = c.request("POST", "/mcp", msg, headers, raw=True)
    assert (
        r.getheader("Content-Type") == "application/json"
        and json.loads(raw)["result"]["isError"] is False
    )


def test_cancelled_notification_ends_a_waiting_call_without_a_response(server):
    c = Client(server)
    c.initialize()
    svc = server.proto.svc
    job = svc.jobs.submit("demo", _py("import time; time.sleep(20)"))
    out = {}

    def wait():
        out["r"] = c.request("POST", "/mcp", {"jsonrpc": "2.0", "id": 42, "method": "tools/call", "params": {
            "name": "dlnet_job", "arguments": {"job_id": job.id, "wait": 20}}}, {"Mcp-Session-Id": c.sid})  # fmt: skip

    t = threading.Thread(target=wait)
    t.start()
    time.sleep(0.8)
    code, _, _ = c.request("POST", "/mcp", {"jsonrpc": "2.0", "method": "notifications/cancelled",
                                            "params": {"requestId": 42}}, {"Mcp-Session-Id": c.sid})  # fmt: skip
    assert code == 202
    t.join(10)
    assert not t.is_alive() and out["r"][0] == 202  # no JSON-RPC response for a cancelled request
    assert job.status == "running"  # the job itself goes on: only the wait was cancelled
    svc.jobs.cancel(job.id)


# ------------------------------------------------------------------------- protocol layer, tools


def test_protocol_layer_answers_both_eras_without_a_transport(tmp_path):
    proto = mcp_server.Protocol(mcp_tools.Service(tmp_path / "ws"))
    modern = {"_meta": {V: "2026-07-28"}}
    r = proto.handle({"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": modern})
    assert (
        r["result"]["supportedVersions"][0] == "2026-07-28"
        and r["result"]["resultType"] == "complete"
    )
    r = proto.handle(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"_meta": {V: "2031-01-01"}}}
    )
    assert r["error"]["code"] == -32022
    r = proto.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        }
    )
    assert r["result"]["protocolVersion"] == "2025-03-26" and "resultType" not in r["result"]
    assert proto.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert (
        proto.handle({"jsonrpc": "2.0", "id": 4, "result": {}}) is None
    )  # a response: nothing to do
    assert proto.handle([1, 2])["error"]["code"] == -32600
    assert (
        proto.handle({"jsonrpc": "2.0", "id": 5, "method": "ping", "params": [1]})["error"]["code"]
        == -32602
    )
    assert (
        proto.handle({"jsonrpc": "2.0", "id": 6, "method": "prompts/get", "params": {"name": "x"}})[
            "error"
        ]["code"]
        == -32602
    )
    for name in ("design_baselines", "propose_architecture", "compare_with_papers"):
        assert "dlnet_" in mcp_tools.prompt_text(
            name, {"data": "data/d.csv", "idea": "i", "cards": "c"}
        )
    seen = []
    r = proto.handle({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {
        "name": "dlnet_info", "arguments": {}, "_meta": {"progressToken": 1}}}, seen.append)  # fmt: skip
    assert r["result"]["isError"] is False and r["result"]["structuredContent"]["dlnet"]


def test_candidates_tool_on_images(tmp_path):
    import numpy as np

    svc = mcp_tools.Service(tmp_path / "ws")
    rng = np.random.default_rng(0)
    np.savez(svc.ws.root / "data" / "img.npz", X=rng.normal(size=(60, 8, 8)).astype("float32"),
             y=np.repeat(np.arange(3), 20))  # fmt: skip
    body = svc.call("dlnet_candidates", {"data": "data/img.npz", "zoo": 4, "limit": 5})
    assert body["plugin"] == "image_classification" and body["input"]["kind"] == "map"
    assert 0 < len(body["candidates"]) <= 5 and all("spec" in c for c in body["candidates"])
    assert "conv_bias_gain" in body["data_card"]
    with pytest.raises(KeyError):
        svc.call("nope", {})
    with pytest.raises(service.ServiceError, match="plugin"):
        svc.call("dlnet_profile", {"data": "data/img.npz", "plugin": "no-such-plugin"})


# ------------------------------------------------------------------------- the whole flow


@pytest.mark.e2e
def test_design_train_predict_and_download_over_http(server, tabular_csv):
    c = Client(server)
    c.initialize()
    code, _, _ = c.request("PUT", "/files/data/t.csv", tabular_csv.read_bytes())
    assert code == 201
    prof = c.call("dlnet_profile", {"data": "data/t.csv", "target": "y"})[1]
    assert prof["plugin"] == "tabular" and prof["profile"]["n_train"] > 0
    spec = {"op": "seq", "body": [{"op": "tide_block", "hidden": 16, "out": 8}]}
    _, d = c.call("dlnet_design", {"data": "data/t.csv", "target": "y", "name": "exp1", "budget": "30ep", "topk": 1,
                                     "zoo": 2, "evolve": 0, "per_family": 1, "device": "cpu", "threads": 1,
                                     "extra_specs": [spec], "wait": 0})  # fmt: skip
    assert d["status"] in ("queued", "running") and d["kind"] == "design"
    while d["status"] not in service.FINAL:
        d = c.call("dlnet_job", {"job_id": d["job_id"], "wait": 20})[1]
    assert d["status"] == "succeeded" and d["ok"] and d["run"] == "runs/exp1" and d["top_k"], d
    assert "Leaderboard" in d["leaderboard_md"]
    assert "already exists" in c.call("dlnet_design", {"data": "data/t.csv", "name": "exp1"})[1]
    board = c.call("dlnet_leaderboard", {"run": "exp1"})[1]
    pkg = board["top_k"][0]["package"]
    assert pkg.startswith("runs/exp1/top1_") and board["verdict"] is not None
    info = c.call("dlnet_package", {"package": pkg})[1]
    assert "model.py" in info["files"] and info["config"]["arch"] and info["report_md"]
    assert "import torch" in c.call("dlnet_read_file", {"path": pkg + "/model.py"})[1]["text"]
    assert "not a package" in c.call("dlnet_package", {"package": "runs/exp1"})[1]
    tr = c.call(
        "dlnet_train",
        {"package": pkg, "data": "data/t.csv", "target": "y", "epochs": 5, "wait": 120},
    )[1]
    assert tr["status"] == "succeeded" and 0 <= tr["test"]["score"] <= 1 and tr["params"] > 0, tr
    pr = c.call(
        "dlnet_predict",
        {"package": pkg, "data": "data/t.csv", "out": "runs/exp1/pred.csv", "wait": 120},
    )[1]
    assert (
        pr["status"] == "succeeded"
        and pr["predictions"] == "runs/exp1/pred.csv"
        and pr["bytes"] > 0
    )
    z = c.call("dlnet_archive", {"path": pkg})[1]
    r, data = c.request("GET", "/files/" + z["path"], raw=True)
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert (
        r.status == 200 and any(n.endswith("model.py") for n in names) and z["url"].endswith(".zip")
    )
    assert "refusing" in c.call("dlnet_archive", {"path": "."})[1]
    kinds = [(j["kind"], j["status"]) for j in c.call("dlnet_jobs")[1]["jobs"]]
    assert kinds == [("predict", "succeeded"), ("train", "succeeded"), ("design", "succeeded")]


# --------------------------------------------------------------------- dlnet_fetch (network faked)


class _Body(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _fetch_svc(tmp_path, monkeypatch, ip="93.184.216.34", body=b"x" * 10):
    svc = mcp_tools.Service(tmp_path / "ws", allow_fetch=True)
    monkeypatch.setattr(mcp_tools.socket, "getaddrinfo", lambda h, p: [(0, 0, 0, "", (ip, p))])
    monkeypatch.setattr(mcp_tools.urllib.request, "urlopen", lambda req, timeout: _Body(body))
    return svc


def test_fetch_is_off_unless_enabled(tmp_path):
    svc = mcp_tools.Service(tmp_path / "ws")
    with pytest.raises(mcp_tools.ServiceError, match="--allow-fetch"):
        svc.tool_fetch({"url": "https://example.org/a.npz"}, lambda *a: None, lambda: False)


def test_fetch_downloads_into_data(tmp_path, monkeypatch):
    svc = _fetch_svc(tmp_path, monkeypatch)
    seen = []
    r = svc.tool_fetch({"url": "https://example.org/d/a%20b.npz", "name": "a.npz"},
                       lambda *a: seen.append(a), lambda: False)  # fmt: skip
    assert r == {"path": "data/a.npz", "bytes": 10} and seen
    assert (tmp_path / "ws" / "data" / "a.npz").read_bytes() == b"x" * 10
    with pytest.raises(mcp_tools.ServiceError, match="overwrite"):
        svc.tool_fetch({"url": "https://example.org/a.npz"}, lambda *a: None, lambda: False)
    svc.tool_fetch(
        {"url": "https://example.org/a.npz", "overwrite": True}, lambda *a: None, lambda: False
    )


@pytest.mark.parametrize(
    "url, ip, match",
    [
        ("http://example.org/a.npz", "93.184.216.34", "only https"),
        ("https://intranet/a.npz", "10.0.0.5", "non-public"),
        ("https://localhost/a.npz", "127.0.0.1", "non-public"),
        ("https://example.org/a$b.npz", "93.184.216.34", "letters, digits"),
    ],
)
def test_fetch_refuses(tmp_path, monkeypatch, url, ip, match):
    svc = _fetch_svc(tmp_path, monkeypatch, ip=ip)
    with pytest.raises(mcp_tools.ServiceError, match=match):
        svc.tool_fetch({"url": url}, lambda *a: None, lambda: False)


def test_fetch_unresolvable_cancelled_and_failed(tmp_path, monkeypatch):
    svc = _fetch_svc(tmp_path, monkeypatch)

    def nores(h, p):
        raise OSError("no such host")

    monkeypatch.setattr(mcp_tools.socket, "getaddrinfo", nores)
    with pytest.raises(mcp_tools.ServiceError, match="cannot resolve"):
        svc.tool_fetch({"url": "https://nowhere.example/a.npz"}, lambda *a: None, lambda: False)

    svc = _fetch_svc(tmp_path, monkeypatch)
    with pytest.raises(mcp_tools.ServiceError, match="cancelled"):
        svc.tool_fetch({"url": "https://example.org/b.npz"}, lambda *a: None, lambda: True)
    assert not (tmp_path / "ws" / "data" / "b.npz").exists()
    assert not (tmp_path / "ws" / "data" / "b.npz.part").exists()

    def down(req, timeout):
        raise OSError("connection reset")

    monkeypatch.setattr(mcp_tools.urllib.request, "urlopen", down)
    with pytest.raises(mcp_tools.ServiceError, match="download failed"):
        svc.tool_fetch({"url": "https://example.org/c.npz"}, lambda *a: None, lambda: False)


def test_paper_tool_rejects_bad_arguments(tmp_path):
    svc = mcp_tools.Service(tmp_path / "ws")
    with pytest.raises(mcp_tools.ServiceError, match="check or build"):
        svc.tool_paper({"action": "verify", "cards": ["x"]}, None, None)
    with pytest.raises(mcp_tools.ServiceError, match="non-empty list"):
        svc.tool_paper({"action": "check", "cards": []}, None, None)
