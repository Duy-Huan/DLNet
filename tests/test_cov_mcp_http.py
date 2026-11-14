"""`dlnet serve` (dlnet/mcp_http.py): the HTTP edge cases the main tests do not reach (v1.9 coverage).

Malformed transfers, expired sessions, batches, header mismatches, a client that drops an SSE stream,
HEAD, failed uploads, TLS, IPv6, and the `serve` command loop itself."""

from __future__ import annotations

import http.client
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
import types

import pytest
from test_mcp_http import TOKEN, Client, V, _py

from dlnet import mcp_http, mcp_tools


@pytest.fixture
def server(tmp_path):
    svc = mcp_tools.Service(tmp_path / "ws", max_jobs=1, default_wait=30)
    httpd = mcp_http.make_server(svc, mcp_http.Config(port=0, token=TOKEN, quiet=False))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    svc.close()


def _raw(httpd, data: bytes) -> bytes:
    """Send bytes as they are (no client library fixing the request) and read the answer."""
    with socket.create_connection(httpd.server_address, timeout=30) as s:
        s.sendall(data)
        s.shutdown(socket.SHUT_WR)
        out = b""
        while chunk := s.recv(65536):
            out += chunk
    return out


def _head(port, extra=""):
    return (f"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {TOKEN}\r\n"
            f"Content-Type: application/json\r\n{extra}").encode()  # fmt: skip


def test_bodies_without_a_length_are_refused(server):
    port = server.server_address[1]
    out = _raw(server, _head(port, "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"))
    assert b" 411 " in out.split(b"\r\n")[0] and b"chunked" in out
    out = _raw(server, _head(port, "Content-Length: abc\r\n\r\n"))
    assert b" 411 " in out.split(b"\r\n")[0]


def test_access_log_root_and_unknown_routes(server, capsys):
    c = Client(server, token=None)
    code, body, _ = c.request("GET", "/")
    assert code == 200 and body["mcp_endpoint"] == "/mcp"
    assert c.request("GET", "/nothing")[0] == 404
    c = Client(server)
    assert c.request("DELETE", "/nothing")[0] == 404
    assert c.request("DELETE", "/mcp")[0] == 400  # no session id
    assert c.request("DELETE", "/mcp", headers={"Mcp-Session-Id": "nope"})[0] == 404
    code, _, r = c.request("PUT", "/mcp", b"x")
    assert code == 405 and "POST" in r.getheader("Allow")
    assert c.request("PUT", "/elsewhere", b"x")[0] == 404
    assert "[serve]" in capsys.readouterr().err  # quiet=False: one access-log line per request


def test_sessions_expire(server, monkeypatch):
    c1, c2 = Client(server), Client(server)
    c1.initialize()
    c2.initialize()
    monkeypatch.setattr(mcp_http, "SESSION_IDLE", -1.0)
    assert c1.rpc("ping")[0] == 404  # expired when it is used
    Client(server).initialize()  # a new session clears the other expired ones (c2)
    assert c2.sid not in server.sessions.items


def test_batches_discovery_and_mismatched_modern_headers(server):
    c = Client(server)
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}  # fmt: skip
    assert c.request("POST", "/mcp", [init])[0] == 400  # initialize in a batch
    code, body, _ = c.request(
        "POST", "/mcp", {"jsonrpc": "2.0", "id": 2, "method": "server/discover"}
    )
    assert code == 200 and "result" in body  # answered without a session
    modern = {"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "ping"}
    ping = {"jsonrpc": "2.0", "id": 3, "method": "ping", "params": {"_meta": {V: "2026-07-28"}}}
    code, body, _ = c.request("POST", "/mcp", [ping, ping], modern)
    assert code == 400 and "single JSON-RPC message" in json.dumps(body)
    code, body, _ = c.request("POST", "/mcp", ping)  # body says modern, header missing
    assert code == 400 and "header is missing" in json.dumps(body)
    note = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {"_meta": {V: "2026-07-28"}},
    }
    assert c.request("POST", "/mcp", note, {"MCP-Protocol-Version": "2026-07-28"})[0] == 202


def test_a_client_that_drops_the_sse_stream_cancels_its_wait(server, monkeypatch):
    monkeypatch.setattr(mcp_http, "KEEPALIVE", 0.1)
    c = Client(server)
    c.initialize()
    svc = server.proto.svc
    job = svc.jobs.submit("demo", _py("import time; time.sleep(30)"))
    msg = {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {
        "name": "dlnet_job", "arguments": {"job_id": job.id, "wait": 30}, "_meta": {"progressToken": "p"}}}  # fmt: skip
    body = json.dumps(msg).encode()
    port = server.server_address[1]
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall((f"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {TOKEN}\r\n"
               f"Content-Type: application/json\r\nAccept: application/json, text/event-stream\r\n"
               f"Mcp-Session-Id: {c.sid}\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body)  # fmt: skip
    first = s.recv(4096)
    assert b"text/event-stream" in first
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    s.close()  # reset: the next keep-alive write fails
    deadline = time.time() + 20
    while time.time() < deadline and job.status == "running" and threading.active_count() > 3:
        time.sleep(0.2)
        if all("keepalive" not in (t.name or "") for t in threading.enumerate()):
            break
    time.sleep(1.0)  # the waiting call returns after the cancel
    assert job.status == "running"  # the job goes on: only the wait was cancelled
    svc.jobs.cancel(job.id)


def test_head_download_and_failed_uploads(server):
    c = Client(server)
    assert c.request("PUT", "/files/data/a.txt", b"hello")[0] == 201
    r, data = c.request("HEAD", "/files/data/a.txt", raw=True)
    assert r.status == 200 and r.getheader("Content-Length") == "5" and data == b""
    port = server.server_address[1]
    put = (f"PUT /files/data/b.txt HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {TOKEN}\r\n").encode()  # fmt: skip
    out = _raw(server, put + b"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
    assert b" 411 " in out.split(b"\r\n")[0]
    out = _raw(server, put + b"Content-Length: 100\r\n\r\nonly-a-part")  # the client stops early
    assert b" 400 " in out.split(b"\r\n")[0] and b"upload failed" in out
    assert not list((server.proto.svc.ws.root / "data").glob("b.txt*"))  # no partial file left


def test_dropped_connections_are_not_logged_but_errors_are(server, capsys):
    try:
        raise ConnectionResetError("gone")
    except ConnectionResetError:
        server.handle_error(None, ("127.0.0.1", 1))
    assert "Traceback" not in capsys.readouterr().err
    try:
        raise RuntimeError("bug")
    except RuntimeError:
        server.handle_error(None, ("127.0.0.1", 1))
    assert "RuntimeError" in capsys.readouterr().err


def test_ipv6_server_does_not_change_the_ipv4_class(tmp_path):
    svc = mcp_tools.Service(tmp_path / "ws")
    try:
        httpd = mcp_http.make_server(svc, mcp_http.Config(host="::1", port=0, token=TOKEN))
        httpd.server_close()
    except OSError:
        pass  # a machine without IPv6: the class is still made, the bind fails
    assert mcp_http.Server.address_family == socket.AF_INET
    v4 = mcp_http.make_server(svc, mcp_http.Config(port=0, token=TOKEN))
    v4.server_close()
    svc.close()


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl to make a certificate")
def test_tls(tmp_path):
    import ssl

    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out",
                    str(cert), "-days", "1", "-subj", "/CN=127.0.0.1"], check=True, capture_output=True)  # fmt: skip
    svc = mcp_tools.Service(tmp_path / "ws")
    httpd = mcp_http.make_server(svc, mcp_http.Config(port=0, token=TOKEN), str(cert), str(key))
    assert httpd.url.startswith("https://")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        ctx = ssl.create_default_context(cafile=str(cert))
        ctx.check_hostname = False
        conn = http.client.HTTPSConnection(
            "127.0.0.1", httpd.server_address[1], context=ctx, timeout=10
        )
        conn.request("GET", "/healthz", headers={"Host": f"127.0.0.1:{httpd.server_address[1]}"})
        assert json.loads(conn.getresponse().read())["status"] == "ok"
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        svc.close()


def test_serve_command_runs_until_interrupted(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DLNET_TOKEN", raising=False)

    def stop(self, *a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(mcp_http.Server, "serve_forever", stop)
    args = types.SimpleNamespace(token=None, token_file=None, generate_token=True, workspace=str(tmp_path / "w"),
                                 no_sandbox=False, max_jobs=1, wait=5, max_budget="12h", allow_fetch=False,
                                 public_url="http://example.test/", host="127.0.0.1", port=0, allow_origin=[],
                                 allow_host=[], max_upload_gb=1, stateless=False, quiet=True, certfile=None,
                                 keyfile=None)  # fmt: skip
    assert mcp_http.run(args) == 0
    err = capsys.readouterr().err
    assert "generated token" in err and "bearer token required" in err and "stopping" in err
    assert "http://example.test/mcp" in err
    args.generate_token = False
    assert mcp_http.run(args) == 0
    assert "NO AUTHENTICATION" in capsys.readouterr().err


def test_module_has_no_import_side_effects():
    assert "dlnet.mcp_http" in sys.modules
