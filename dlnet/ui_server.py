"""`dlnet ui`: the optional web interface, on this machine only. The CLI stays the main tool.

    dlnet ui                                   # opens http://127.0.0.1:8766 in the browser
    dlnet ui --workspace ~/research/hsi-paper --port 8766 --no-browser

Six screens (doc/ui.md): workspace, runs, files, new run (with `dlnet plan`), run detail (Top-K,
paper rows with significance, LaTeX) and AI / MCP (start `dlnet serve`, client set-up, test).
Everything the UI does goes through the same tools as the MCP server (mcp_tools.Service), so a run
started here is an ordinary `dlnet design` job and a run started in a terminal shows up here.

Security: it listens on 127.0.0.1 only. The page carries a random token made at start; every API
call must send it (a page of another site cannot read it), and the Host header must be this
machine (DNS rebinding). The UI may browse folders of this computer to pick a workspace: it is a
local, single-user tool, never expose it on a network (use `dlnet serve` for remote AI clients).
"""

from __future__ import annotations

import contextlib
import json
import mimetypes
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__, mcp_tools, service

STATIC = Path(__file__).resolve().parent / "ui"
UPLOAD_LIMIT = 8 * 1024**3


def state_file() -> Path:
    return Path(os.environ.get("DLNET_UI_STATE") or Path.home() / ".dlnet" / "ui.json")


def load_state() -> dict:
    try:
        return json.loads(state_file().read_text())
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    f = state_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(st, indent=1))


def lan_address() -> str | None:
    """The address other machines of the local network reach this one at (no packet is sent)."""
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    return None


class McpProcess:
    """`dlnet serve` started and stopped from the AI screen."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.cfg: dict = {}
        self.log: Path | None = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> dict:
        lan = bool(self.cfg.get("lan"))
        port = self.cfg.get("port", 8765)
        host = (lan_address() or "0.0.0.0") if lan else "127.0.0.1"
        return {"running": self.running(), "url": f"http://{host}:{port}/mcp", **self.cfg,
                "exit_code": None if self.proc is None or self.running() else self.proc.returncode}  # fmt: skip

    @staticmethod
    def _port_free(port: int) -> bool:
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", port)) != 0

    def start(self, workspace: Path, cfg: dict) -> dict:
        self.stop()
        port = int(cfg.get("port") or 8765)
        for _ in range(40):  # a server stopped a moment ago may still hold the port
            if self._port_free(port):
                break
            time.sleep(0.25)
        else:
            raise service.ServiceError(f"port {port} is used by another program: pick another port")
        token = cfg.get("token") or secrets.token_urlsafe(24)
        lan = bool(cfg.get("lan"))
        self.cfg = {"port": port, "lan": lan, "token": token, "allow_fetch": bool(cfg.get("allow_fetch")),
                    "max_jobs": int(cfg.get("max_jobs") or 1), "max_budget": str(cfg.get("max_budget") or "12h")}  # fmt: skip
        argv = [*service.python(), "serve", "--workspace", str(workspace), "--port", str(port),
                "--host", "0.0.0.0" if lan else "127.0.0.1", "--max-jobs", str(self.cfg["max_jobs"]),
                "--max-budget", self.cfg["max_budget"]]  # fmt: skip
        if self.cfg["allow_fetch"]:
            argv.append("--allow-fetch")
        if lan:  # the advertised address must pass the server's Host check
            ip = lan_address()
            if ip:
                argv += ["--allow-host", f"{ip}:{port}"]
        (workspace / "jobs").mkdir(parents=True, exist_ok=True)
        self.log = workspace / "jobs" / "mcp-server.log"
        env = {**os.environ, "DLNET_TOKEN": token, "PYTHONUNBUFFERED": "1"}
        with open(self.log, "ab") as log:
            self.proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         env=env)  # fmt: skip
        for _ in range(240):  # until it answers /healthz (or exits)
            if not self.running():
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1).read()
                break
            except OSError:
                time.sleep(0.25)
        if not self.running():
            raise service.ServiceError("the MCP server stopped at start: " + " ".join(self.tail(3)))
        return self.status()

    def stop(self) -> None:
        if self.running():
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def tail(self, lines: int = 60) -> list[str]:
        if not self.log or not self.log.exists():
            return []
        text = self.log.read_bytes()[-32_000:].decode("utf-8", errors="replace")
        return text.splitlines()[-lines:]

    def test(self) -> dict:
        """What an AI client does first: initialize, then list the tools (MCP 2025-11-25)."""
        if not self.running():
            return {"ok": False, "error": "the server is not running"}
        port, token = self.cfg["port"], self.cfg["token"]
        url = f"http://127.0.0.1:{port}/mcp"
        head = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-11-25"}  # fmt: skip
        t0 = time.time()
        for _ in range(8):  # a server that has just started may not accept connections yet
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.25)
        try:
            init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "dlnet-ui", "version": __version__}}}  # fmt: skip
            req = urllib.request.Request(url, json.dumps(init).encode(), head)
            with urllib.request.urlopen(req, timeout=10) as r:
                info = json.loads(r.read())["result"]
                sid = r.headers.get("Mcp-Session-Id")
            h2 = {**head, **({"Mcp-Session-Id": sid} if sid else {})}
            note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            urllib.request.urlopen(
                urllib.request.Request(url, json.dumps(note).encode(), h2), timeout=10
            ).read()
            lst = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            with urllib.request.urlopen(
                urllib.request.Request(url, json.dumps(lst).encode(), h2), timeout=10
            ) as r:
                tools = json.loads(r.read())["result"]["tools"]
        except (OSError, ValueError, KeyError) as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "server": info.get("serverInfo"), "protocol": info.get("protocolVersion"),
                "tools": len(tools), "ms": round((time.time() - t0) * 1000)}  # fmt: skip


class App:
    """What the handlers share: the token, the open workspace and its Service, the MCP process."""

    def __init__(self, workspace: Path, port: int):
        self.token = secrets.token_urlsafe(24)
        self.port = port
        self.lock = threading.Lock()
        self.mcp = McpProcess()
        self.svc: mcp_tools.Service | None = None
        self.open(workspace, create=True)  # the first start: ./dlnet-workspace does not exist yet

    @property
    def ws(self) -> service.Workspace:
        return self.svc.ws

    def open(self, path: Path, create: bool = False) -> None:
        path = Path(path).expanduser().resolve()
        if not path.is_dir():
            if not create:
                raise service.ServiceError(f"{path} is not a folder")
            path.mkdir(parents=True)
        with self.lock:
            if self.svc is not None:
                self.svc.close()
            self.svc = mcp_tools.Service(str(path), max_jobs=1, default_wait=0)
            for sub in ("data", "runs", "papers"):
                (path / sub).mkdir(exist_ok=True)
        st = load_state()
        recent = [p for p in st.get("recent", []) if p != str(path)]
        st["recent"] = [str(path), *recent][:12]
        save_state(st)

    def runs(self) -> list[dict]:
        root = self.ws.root
        jobs = {}
        for j in self.svc.jobs.jobs.values():
            if j.kind == "design" and j.out:
                prev = jobs.get(j.out)
                if prev is None or j.created > prev.created:
                    jobs[j.out] = j
        out = []
        for d in sorted((root / "runs").glob("*")):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            rel = self.ws.rel(d)
            j = jobs.get(rel)
            row = {"name": d.name, "path": rel, "modified": d.stat().st_mtime, "status": "empty"}
            if j is not None:
                row["status"], row["job_id"] = j.status, j.id
                row["command"] = j.argv
                if j.status in ("running", "queued"):
                    tail = self.svc.jobs.log_tail(j, 1)
                    row["progress"] = tail[0] if tail else ""
            board = d / "leaderboard.json"
            if board.exists():
                with contextlib.suppress(OSError, ValueError, KeyError, IndexError):
                    b = json.loads(board.read_text())
                    t = b["top_k"][0]
                    row.update(status=row["status"] if j is not None and j.status in ("running", "queued")
                               else "done", task=b.get("task"), metric=b.get("metric"), top1=t["name"],
                               val=t.get("val_mean"), val_std=t.get("val_std"), n_runs=t.get("n_runs"),
                               test=(t.get("test") or {}).get("score"), papers=len(b.get("papers") or []),
                               budget=(b.get("budget") or {}).get("budget"))  # fmt: skip
            out.append(row)
        return sorted(out, key=lambda r: -r["modified"])

    def run_detail(self, name: str) -> dict:
        d = self.ws.root / "runs" / name
        if not service.NAME.match(name) or not d.is_dir():
            raise service.ServiceError(f"no run {name!r}")
        row = next((r for r in self.runs() if r["name"] == name), {"name": name})
        res = {"run": row, "board": None}
        if (d / "leaderboard.json").exists():
            res["board"] = json.loads((d / "leaderboard.json").read_text())
        if (d / "results.tex").exists():
            res["latex"] = (d / "results.tex").read_text(encoding="utf-8")
        res["files"] = [e for e in self.ws.listing(self.ws.rel(d), 2, 400)["entries"]]
        if row.get("job_id"):
            res["log"] = self.svc.jobs.log_tail(self.svc.jobs.get(row["job_id"]), 30)
        return res


def browse(path: str | None) -> dict:
    p = Path(path).expanduser() if path else Path.home()
    p = p.resolve()
    if not p.is_dir():
        raise service.ServiceError(f"{p} is not a folder")
    dirs = []
    with contextlib.suppress(OSError):
        for c in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if c.name.startswith(".") or not c.is_dir():
                continue
            with contextlib.suppress(OSError):
                dirs.append(
                    {"name": c.name, "workspace": (c / "runs").is_dir() or (c / "data").is_dir()}
                )
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "dirs": dirs[:500],
            "workspace": (p / "runs").is_dir() or (p / "data").is_dir()}  # fmt: skip


class Handler(BaseHTTPRequestHandler):
    server_version = f"dlnet-ui/{__version__}"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # quiet: the terminal is the user's
        pass

    def _send(self, code: int, body=b"", ctype="application/json", headers=None) -> None:
        if not isinstance(body, bytes):
            body = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in (f"127.0.0.1:{self.app.port}", f"localhost:{self.app.port}")

    def _guard(self) -> bool:
        if not self._host_ok():
            self._send(403, {"error": "this interface only answers on 127.0.0.1 / localhost"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin.lower() not in (
            f"http://127.0.0.1:{self.app.port}",
            f"http://localhost:{self.app.port}",
        ):
            self._send(403, {"error": "cross-origin request refused"})
            return False
        if not secrets.compare_digest(self.headers.get("X-DLNet-Token", ""), self.app.token):
            self._send(401, {"error": "missing or wrong UI token: reload the page"})
            return False
        return True

    def _json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2_000_000:
            raise service.ServiceError("request too large")
        raw = self.rfile.read(n) if n else b"{}"
        body = json.loads(raw or b"{}")
        if not isinstance(body, dict):
            raise service.ServiceError("the body must be a JSON object")
        return body

    # ------------------------------------------------------------------ static

    def _static(self, rel: str) -> None:
        if rel in ("", "index.html"):
            html = (STATIC / "index.html").read_text(encoding="utf-8")
            html = html.replace("__TOKEN__", self.app.token).replace("__VERSION__", __version__)
            return self._send(200, html.encode(), "text/html; charset=utf-8",
                              {"Content-Security-Policy": "default-src 'self'; img-src 'self' data:; "
                               "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"})  # fmt: skip
        f = (STATIC / rel).resolve()
        if STATIC not in f.parents or not f.is_file():
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        return self._send(200, f.read_bytes(), ctype)

    # ------------------------------------------------------------------ verbs

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        if not url.path.startswith(("/api/", "/files/")):
            if not self._host_ok():
                return self._send(
                    403, {"error": "this interface only answers on 127.0.0.1 / localhost"}
                )
            return self._static(url.path.lstrip("/").removeprefix("static/"))
        if url.path.startswith("/files/"):
            return self._download(urllib.parse.unquote(url.path[len("/files/") :]), url.query)
        if not self._guard():
            return None
        q = dict(urllib.parse.parse_qsl(url.query))
        return self._api("GET", url.path, q, {})

    def do_POST(self):
        url = urllib.parse.urlsplit(self.path)
        if not self._guard():
            return None
        try:
            body = self._json()
        except (ValueError, service.ServiceError) as e:
            return self._send(400, {"error": str(e)})
        return self._api("POST", url.path, {}, body)

    def do_PUT(self):
        url = urllib.parse.urlsplit(self.path)
        if not self._guard():
            return None
        rel = (
            urllib.parse.unquote(url.path[len("/files/") :])
            if url.path.startswith("/files/")
            else ""
        )
        if not rel.startswith(("data/", "papers/")):
            return self._send(400, {"error": "upload into data/ or papers/"})
        try:
            n = int(self.headers.get("Content-Length") or -1)
            if n < 0 or n > UPLOAD_LIMIT:
                raise service.ServiceError("Content-Length required, at most 8 GB")
            dest = self.app.ws.path(rel)
            if dest.exists() and self.headers.get("X-Overwrite") != "1":
                raise service.ServiceError(f"{rel} exists")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            left = n
            with open(tmp, "wb") as f:
                while left:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        raise service.ServiceError("upload cut short")
                    f.write(chunk)
                    left -= len(chunk)
            tmp.replace(dest)
        except service.ServiceError as e:
            return self._send(400, {"error": str(e)})
        return self._send(200, {"path": rel, "bytes": n})

    def _download(self, rel: str, query: str) -> None:
        q = dict(urllib.parse.parse_qsl(query))
        # links in <img>/<a> cannot send a header: the token comes in the query
        if not self._host_ok() or not secrets.compare_digest(q.get("t", ""), self.app.token):
            return self._send(401, {"error": "missing or wrong UI token"})
        try:
            f = self.app.ws.path(rel, must_exist=True)
        except service.ServiceError as e:
            return self._send(404, {"error": str(e)})
        if not f.is_file():
            return self._send(404, {"error": "not a file"})
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if f.suffix in (".py", ".md", ".yaml", ".yml", ".tex", ".log", ".csv", ".txt", ".json"):
            ctype = "text/plain; charset=utf-8"  # shown, never run, by the browser
        headers = {"Content-Security-Policy": "sandbox"}
        if q.get("download"):
            headers["Content-Disposition"] = f'attachment; filename="{f.name}"'
        return self._send(200, f.read_bytes(), ctype, headers)

    # ------------------------------------------------------------------ API

    def _api(self, method: str, path: str, q: dict, body: dict) -> None:
        app = self.app
        try:
            if path == "/api/state":
                st = load_state()
                return self._send(200, {"version": __version__, "workspace": str(app.ws.root),
                                        "recent": [p for p in st.get("recent", []) if Path(p).is_dir()],
                                        "mcp": app.mcp.status()})  # fmt: skip
            if path == "/api/browse":
                return self._send(200, browse(q.get("path")))
            if path == "/api/workspace" and method == "POST":
                app.open(Path(body["path"]), bool(body.get("create")))
                return self._send(200, {"workspace": str(app.ws.root)})
            if path == "/api/runs":
                return self._send(200, {"runs": app.runs()})
            if path.startswith("/api/run/"):
                return self._send(
                    200, app.run_detail(urllib.parse.unquote(path[len("/api/run/") :]))
                )
            if path.startswith("/api/tool/") and method == "POST":
                name = path[len("/api/tool/") :]
                if name in ("dlnet_design", "dlnet_plan", "dlnet_job", "dlnet_paper"):
                    body.setdefault("wait", 0)  # the page polls; a request never blocks
                return self._send(200, {"ok": True, "result": app.svc.call(name, body)})
            if path == "/api/tools":
                return self._send(
                    200,
                    {
                        "tools": [
                            {"name": t["name"], "title": t.get("title")} for t in mcp_tools.TOOLS
                        ]
                    },
                )
            if path == "/api/mcp":
                return self._send(200, {**app.mcp.status(), "log": app.mcp.tail()})
            if path == "/api/mcp/start" and method == "POST":
                return self._send(200, app.mcp.start(app.ws.root, body))
            if path == "/api/mcp/stop" and method == "POST":
                app.mcp.stop()
                return self._send(200, app.mcp.status())
            if path == "/api/mcp/test" and method == "POST":
                return self._send(200, app.mcp.test())
        except KeyError as e:
            return self._send(400, {"ok": False, "error": f"unknown or missing: {e}"})
        except (service.ServiceError, ValueError, OSError) as e:
            return self._send(400, {"ok": False, "error": str(e)})
        return self._send(404, {"error": f"no endpoint {method} {path}"})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(workspace, port: int = 8766) -> Server:
    srv = Server(("127.0.0.1", port), Handler)
    srv.app = App(Path(workspace), srv.server_address[1])  # type: ignore[attr-defined]
    return srv


def add_parser(sub):
    p = sub.add_parser(
        "ui", help="optional web interface on this machine (the CLI stays the main tool)"
    )
    p.add_argument("--workspace", help="folder with data/, runs/, papers/ (default: the last one used, "
                   "else ./dlnet-workspace)")  # fmt: skip
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    return p


def run(args) -> int:
    recent = load_state().get("recent") or []
    ws = args.workspace or next((p for p in recent if Path(p).is_dir()), None) or "dlnet-workspace"
    try:
        srv = make_server(ws, args.port)
    except OSError as e:
        print(f"[ui] cannot listen on 127.0.0.1:{args.port}: {e} (use --port)", file=sys.stderr)
        return 2
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(
        f"[ui] DLNet {__version__} at {url}  (workspace {srv.app.ws.root}; Ctrl+C to stop)",
        file=sys.stderr,
    )
    if not args.no_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[ui] stopping", file=sys.stderr)
    finally:
        srv.app.mcp.stop()
        srv.app.svc.close()
        srv.server_close()
    return 0
