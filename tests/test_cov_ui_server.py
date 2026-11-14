"""The web UI server (dlnet/ui_server.py): the MCP server it starts and stops, and the API edges the
other UI tests do not reach (v1.9 coverage)."""

from __future__ import annotations

import json
import socket
import subprocess

import pytest
from test_ui import _req, ui  # noqa: F401  (the fixture)

from dlnet import service, ui_server


def test_lan_address(monkeypatch):
    ip = ui_server.lan_address()
    assert ip is None or ip.count(".") == 3

    class NoNet(socket.socket):
        def connect(self, addr):
            raise OSError("no network")

    monkeypatch.setattr(ui_server.socket, "socket", NoNet)
    assert ui_server.lan_address() is None


class FakeProc:
    def __init__(self, alive=False, slow=False):
        self.alive, self.slow, self.returncode, self.killed = alive, slow, 3, False

    def poll(self):
        return None if self.alive else self.returncode

    def terminate(self):
        pass

    def wait(self, timeout=None):
        if self.slow:
            raise subprocess.TimeoutExpired("dlnet serve", timeout)
        return self.returncode

    def kill(self):
        self.killed = True


def test_mcp_process_start_failures_and_stop(tmp_path, monkeypatch):
    m = ui_server.McpProcess()
    assert m.tail() == []
    monkeypatch.setattr(ui_server.time, "sleep", lambda s: None)
    monkeypatch.setattr(ui_server.McpProcess, "_port_free", staticmethod(lambda port: False))
    with pytest.raises(service.ServiceError, match="used by another program"):
        m.start(tmp_path, {"port": 9})
    monkeypatch.setattr(ui_server.McpProcess, "_port_free", staticmethod(lambda port: True))
    monkeypatch.setattr(ui_server, "lan_address", lambda: "192.168.1.5")
    seen = {}

    def popen(argv, **kw):
        seen["argv"] = argv
        kw["stdout"].write(b"Traceback: it could not start\n")
        return FakeProc(alive=False)

    monkeypatch.setattr(ui_server.subprocess, "Popen", popen)
    with pytest.raises(service.ServiceError, match="stopped at start"):
        m.start(tmp_path, {"port": 9, "lan": True, "allow_fetch": True})
    assert "--allow-fetch" in seen["argv"] and "192.168.1.5:9" in seen["argv"]
    m.proc = FakeProc(alive=True, slow=True)
    m.stop()
    assert m.proc.killed


def test_mcp_test_reports_a_server_that_does_not_answer(tmp_path, monkeypatch):
    m = ui_server.McpProcess()
    m.proc = FakeProc(alive=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    m.cfg = {"port": free, "token": "t"}
    monkeypatch.setattr(ui_server.time, "sleep", lambda s: None)
    r = m.test()
    assert r["ok"] is False and "Error" in r["error"]


def test_api_edges(ui, tmp_path):  # noqa: F811
    srv, port = ui
    tok = {"X-DLNet-Token": srv.app.token}
    root = srv.app.ws.root
    (root / "runs" / "_hidden").mkdir(parents=True)
    run = root / "runs" / "r1"
    run.mkdir()
    (run / "results.tex").write_text("\\begin{table}\\end{table}")
    st, d = _req(port, "GET", "/api/run/r1", headers=tok)
    assert st == 200 and "begin{table}" in d["latex"]
    st, runs = _req(port, "GET", "/api/runs", headers=tok)
    assert all(r["name"] != "_hidden" for r in runs["runs"])
    big = {"X-DLNet-Token": srv.app.token, "Content-Length": "3000000"}
    st, r = _req(port, "POST", "/api/tool/dlnet_info", b"{}", big)
    assert st == 400 and "too large" in json.dumps(r)
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sk:  # the client stops early
        sk.sendall((f"PUT /files/data/cut.bin HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nX-DLNet-Token: {srv.app.token}"
                    "\r\nContent-Length: 10\r\n\r\nabc").encode())  # fmt: skip
        sk.shutdown(socket.SHUT_WR)
        out = b""
        while chunk := sk.recv(4096):
            out += chunk
    assert b" 400 " in out.split(b"\r\n")[0] and b"cut short" in out
    st, _ = _req(port, "GET", f"/files/data?t={srv.app.token}")
    assert st == 404
    (root / "data" / "a.txt").write_text("hello")
    c = __import__("http.client").client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(
        "GET",
        f"/files/data/a.txt?t={srv.app.token}&download=1",
        headers={"Host": f"127.0.0.1:{port}"},
    )
    r = c.getresponse()
    assert r.status == 200 and "attachment" in r.getheader("Content-Disposition")
    c.close()
    st, r = _req(port, "POST", "/api/tool/dlnet_job", {"job_id": "job-none"}, tok)
    assert st == 400
    st, r = _req(port, "GET", "/api/tools", headers=tok)
    assert any(t["name"] == "dlnet_design" for t in r["tools"])
