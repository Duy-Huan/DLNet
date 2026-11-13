"""`dlnet serve`: DLNet as a remote MCP server over the Streamable HTTP transport.

One process serves many AI clients (Claude Code, Codex CLI, Google Antigravity, any MCP client):
they connect to `http://<host>:<port>/mcp`, and DLNet designs, trains and evaluates networks on
the machine the server runs on. Standard library only (http.server), no MCP SDK.

Endpoints
    POST   /mcp            JSON-RPC messages of MCP (the "MCP endpoint" of the specification)
    GET    /mcp            405: this server sends no unsolicited messages
    DELETE /mcp            ends the session named by MCP-Session-Id
    PUT    /files/<path>   upload a file into the workspace (datasets go to data/)
    GET    /files/<path>   download a file of the workspace (a zip, model.py, predictions)
    GET    /healthz        {"status": "ok", "version": ...}; no authentication

What the transport section of the specification (2025-11-25, also valid for 2025-06-18 and
2025-03-26) asks of a server, and where it is done here:
  * Origin validated on every request, 403 when present and not allowed (DNS rebinding).  `_guard`
  * bind to 127.0.0.1 by default; a non-loopback bind needs a token.                   `make_server`
  * authentication: static bearer token (`Authorization: Bearer ...`), constant-time compare.
  * a request gets `application/json`, or `text/event-stream` when the call can report progress
    (the client sent a progress token and accepts SSE): priming event with an id, progress
    notifications, the response, then the stream ends. Notifications / responses get 202.
  * legacy revisions (2025-03-26 ... 2025-11-25): sessions. `MCP-Session-Id` is issued with the
    InitializeResult (cryptographically random); later requests without it get 400, with an
    unknown or expired one 404; DELETE ends it. JSON-RPC batches are accepted (2025-03-26).
  * modern revision (2026-07-28): stateless. No session, no handshake; every POST must carry
    `MCP-Protocol-Version`, `Mcp-Method` and, for tools/call and prompts/get, `Mcp-Name`
    (plain or `=?base64?...?=`), and they must match the body: otherwise 400 with HeaderMismatch
    (-32020). An unimplemented method is 404 with -32601. A closed response stream cancels the
    request.
  * `MCP-Protocol-Version` (or the version in `_meta`) not supported: 400 with
    UnsupportedProtocolVersionError (-32022) listing the supported versions, which is also how
    a dual-era client recognises this server.
Not implemented (optional in the specification): resumable streams (Last-Event-ID, legacy only),
`subscriptions/listen` (the tool list never changes while the server runs), server requests
(sampling, elicitation), OAuth. Put a reverse proxy in front for TLS, or use --certfile.
"""

from __future__ import annotations

import base64
import contextlib
import hmac
import json
import mimetypes
import os
import secrets
import ssl
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import jsonsafe, mcp_server, mcp_tools
from .service import ServiceError

LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")
MAX_BODY = 16 * 1024 * 1024  # one JSON-RPC message
SESSION_IDLE = 24 * 3600.0
KEEPALIVE = 15.0


class Config:
    def __init__(self, host="127.0.0.1", port=8765, token=None, allow_origins=(), allow_hosts=(),
                 max_upload=8 * 1024**3, stateless=False, quiet=False):  # fmt: skip
        self.host, self.port, self.token = host, port, token
        self.allow_origins = {o.rstrip("/").lower() for o in allow_origins}
        self.allow_hosts = {h.lower() for h in allow_hosts}
        self.max_upload, self.stateless, self.quiet = max_upload, stateless, quiet
        self.loopback = host in LOOPBACK


class Sessions:
    def __init__(self):
        self.lock = threading.Lock()
        self.items: dict[str, dict] = {}

    def new(self, version: str) -> str:
        sid = secrets.token_urlsafe(32)
        now = time.time()
        with self.lock:
            for k in [k for k, v in self.items.items() if now - v["seen"] > SESSION_IDLE]:
                del self.items[k]
            self.items[sid] = {"version": version, "created": now, "seen": now}
        return sid

    def touch(self, sid: str) -> dict | None:
        with self.lock:
            s = self.items.get(sid)
            if s is not None:
                if time.time() - s["seen"] > SESSION_IDLE:
                    del self.items[sid]
                    return None
                s["seen"] = time.time()
            return s

    def end(self, sid: str) -> bool:
        with self.lock:
            return self.items.pop(sid, None) is not None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "dlnet-mcp"
    sys_version = ""

    # set by make_server on the server object
    @property
    def cfg(self) -> Config:
        return self.server.cfg

    @property
    def proto(self) -> mcp_server.Protocol:
        return self.server.proto

    def log_message(self, fmt, *args):
        if not self.cfg.quiet:
            sys.stderr.write(f"[serve] {self.address_string()} {fmt % args}\n")

    # ------------------------------------------------------------------ plumbing

    def _send(self, code: int, body=None, ctype="application/json", headers=None):
        raw = (
            b""
            if body is None
            else body
            if isinstance(body, bytes)
            else jsonsafe.dumps(body, default=str).encode()
        )
        self.send_response(code)
        if raw or code not in (202, 204):
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def _fail(self, code: int, message: str, headers=None):
        self._send(
            code, mcp_server.error(None, mcp_server.INVALID_REQUEST, message), headers=headers
        )

    def _guard(self, auth: bool = True) -> bool:
        """Origin, Host and (unless auth=False) the bearer token. Sends the error itself."""
        origin = self.headers.get("Origin")
        if origin is not None:
            o = origin.rstrip("/").lower()
            host = urllib.parse.urlsplit(o).hostname or ""
            if not (
                o in self.cfg.allow_origins
                or (self.cfg.loopback and host in ("127.0.0.1", "localhost", "::1"))
            ):
                self._fail(
                    403, f"origin {origin!r} is not allowed (start the server with --allow-origin)"
                )
                return False
        if (
            self.cfg.loopback
        ):  # a page on another site can only reach a loopback server under a foreign Host
            host = (self.headers.get("Host") or "").lower()
            name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
            if (
                name not in LOOPBACK
                and host not in self.cfg.allow_hosts
                and name not in self.cfg.allow_hosts
            ):
                self._fail(
                    403, f"host {host!r} is not allowed (start the server with --allow-host)"
                )
                return False
        if auth and self.cfg.token:
            got = self.headers.get("Authorization") or ""
            scheme, _, value = got.partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                value.strip().encode(), self.cfg.token.encode()
            ):
                self._fail(401, "missing or wrong bearer token",
                           {"WWW-Authenticate": 'Bearer realm="dlnet", error="invalid_token"'})  # fmt: skip
                return False
        return True

    def _body(self, limit: int) -> bytes | None:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self._fail(411, "chunked bodies are not supported: send Content-Length")
            return None
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._fail(411, "Content-Length required")
            return None
        if n < 0 or n > limit:
            self._fail(413, f"body of {n:,} bytes is over the limit of {limit:,}")
            self.close_connection = True
            return None
        return self.rfile.read(n)

    def _route(self) -> tuple[str, str]:
        path = urllib.parse.urlsplit(self.path).path
        if path.rstrip("/") == "/mcp":
            return "mcp", ""
        if path.startswith("/files/"):
            return "files", urllib.parse.unquote(path[len("/files/") :])
        if path in ("/healthz", "/health"):
            return "health", ""
        return ("root", "") if path in ("", "/") else ("none", "")

    # ------------------------------------------------------------------ verbs

    def do_GET(self):
        kind, rest = self._route()
        if kind == "health":
            from . import __version__

            if self._guard(auth=False):
                self._send(200, {"status": "ok", "name": "dlnet", "version": __version__})
        elif kind == "root":
            if self._guard(auth=False):
                self._send(200, {"name": "dlnet", "mcp_endpoint": "/mcp", "transport": "streamable-http",
                                 "protocol_versions": list(mcp_server.SUPPORTED), "docs": "doc/mcp-server.md"})  # fmt: skip
        elif kind == "mcp":
            if (
                self._guard()
            ):  # no server-initiated stream is offered (allowed by the specification)
                self._send(405, mcp_server.error(None, mcp_server.INVALID_REQUEST,
                                                 "this server offers no standalone SSE stream; use POST"),
                           headers={"Allow": "POST, DELETE"})  # fmt: skip
        elif kind == "files":
            if self._guard():
                self._download(rest)
        else:
            self._fail(404, "not found")

    do_HEAD = do_GET

    def do_DELETE(self):
        kind, _ = self._route()
        if kind != "mcp":
            self._fail(404, "not found")
        elif self._guard():
            sid = self.headers.get("Mcp-Session-Id")
            if not sid:
                self._fail(400, "MCP-Session-Id header required")
            elif self.server.sessions.end(sid):
                self._send(204)
            else:
                self._fail(404, "unknown session")

    def do_PUT(self):
        kind, rest = self._route()
        if kind != "files":
            self._fail(
                405 if kind == "mcp" else 404, "PUT is for /files/<path>", {"Allow": "POST, DELETE"}
            )
        elif self._guard():
            self._upload(rest)

    def do_POST(self):
        kind, _ = self._route()
        if kind != "mcp":
            self._fail(404, "POST goes to /mcp")
            return
        if not self._guard():
            return
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._fail(415, "Content-Type must be application/json")
            return
        raw = self._body(MAX_BODY)
        if raw is None:
            return
        try:
            msg = json.loads(raw)
        except ValueError:
            self._send(400, mcp_server.error(None, mcp_server.PARSE_ERROR, "parse error"))
            return
        msgs = msg if isinstance(msg, list) else [msg]
        if not msgs or not all(isinstance(m, dict) for m in msgs):
            self._fail(400, "the body must be a JSON-RPC message")
            return
        version = self.headers.get("Mcp-Protocol-Version")
        declared = mcp_server.request_version(msgs[0])
        if (version or declared) is not None and (version or declared) not in mcp_server.SUPPORTED:
            self._send(400, mcp_server.unsupported_version(msgs[0].get("id"), version or declared))
            return
        modern = version in mcp_server.MODERN or declared in mcp_server.MODERN
        sid = None if modern else self.headers.get("Mcp-Session-Id")
        headers = {}
        if modern:  # stateless era: one message per POST, mirrored headers checked against the body
            problem = self._mismatch(msg, version, declared)
            if problem:
                self._send(400, mcp_server.error(msgs[0].get("id"), mcp_server.HEADER_MISMATCH,
                                                 f"Header mismatch: {problem}"))  # fmt: skip
                return
        elif not self.cfg.stateless:
            if any(m.get("method") == "initialize" for m in msgs):
                if isinstance(msg, list):
                    self._fail(400, "initialize must not be part of a batch")
                    return
                sid = self.server.sessions.new(
                    self.proto.negotiate((msg.get("params") or {}).get("protocolVersion"))
                )
                headers["Mcp-Session-Id"] = sid
            elif msgs[0].get("method") == "server/discover":
                sid = None  # the probe of a dual-era client: answered without a session
            elif not sid:
                self._fail(400, "MCP-Session-Id header required (initialize first)")
                return
            elif self.server.sessions.touch(sid) is None:
                self._fail(404, "unknown or expired session: initialize again")
                return
        requests = [m for m in msgs if "method" in m and m.get("id") is not None]
        if not requests:  # notifications and responses only
            for m in msgs:
                self.proto.handle(m, scope=sid)
            self._send(202, headers=headers)
            return
        accept = (self.headers.get("Accept") or "").lower()
        one = msgs[0] if len(msgs) == 1 else None
        streaming = (
            one is not None
            and one.get("method") == "tools/call"
            and "text/event-stream" in accept
            and ((one.get("params") or {}).get("_meta") or {}).get("progressToken") is not None
        )
        if streaming:
            self._stream(
                one, sid if not modern else object(), headers
            )  # modern: a scope of its own
            return
        out = [r for r in (self.proto.handle(m, scope=sid) for m in msgs) if r is not None]
        if not out:  # every request was cancelled meanwhile
            self._send(202, headers=headers)
            return
        code = 200
        if modern and out[0].get("error", {}).get("code") == mcp_server.METHOD_NOT_FOUND:
            code = 404  # 2026-07-28: an unimplemented method is 404 with the JSON-RPC error
        self._send(code, out if isinstance(msg, list) else out[0], headers=headers)

    def _mismatch(self, msg, version, declared) -> str | None:
        """Why the mirrored headers of a modern request do not match its body (None: they do)."""
        if isinstance(msg, list):
            return "a POST carries a single JSON-RPC message"
        if version is None:
            return "the MCP-Protocol-Version header is missing"
        if "method" not in msg or msg.get("id") is None:
            return None  # header rules are defined for requests only
        if declared != version:
            return (f"MCP-Protocol-Version header value {version!r} does not match body value {declared!r} "
                    "(params._meta['io.modelcontextprotocol/protocolVersion'])")  # fmt: skip
        method = self.headers.get("Mcp-Method")
        if method is None:
            return "the Mcp-Method header is missing"
        if method != msg["method"]:
            return f"Mcp-Method header value {method!r} does not match body value {msg['method']!r}"
        if msg["method"] in ("tools/call", "prompts/get", "resources/read"):
            params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
            want = params.get("name", params.get("uri"))
            name = self.headers.get("Mcp-Name")
            if name is None:
                return "the Mcp-Name header is missing"
            if name.startswith("=?base64?") and name.endswith("?="):
                try:
                    name = base64.b64decode(name[9:-2], validate=True).decode("utf-8")
                except ValueError:
                    return "the Mcp-Name header is not valid Base64"
            if name != want:
                return f"Mcp-Name header value {name!r} does not match body value {want!r}"
        return None

    # ------------------------------------------------------------------ SSE

    def _stream(self, msg: dict, sid, headers: dict):
        """Answer one tools/call as an SSE stream: priming event, progress notifications, keep-alive
        comments, the response; then the stream (and the connection) ends."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.close_connection = True
        lock, stream, n, alive = threading.Lock(), secrets.token_hex(6), [0], [True]

        def write(data: str | None, comment: str | None = None):
            with lock:
                if not alive[0]:
                    return
                try:
                    if comment is not None:
                        self.wfile.write(f": {comment}\n\n".encode())
                    else:
                        n[0] += 1
                        self.wfile.write(f"id: {stream}-{n[0]}\n".encode())
                        if data:
                            self.wfile.write(b"event: message\n")
                        self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    # the client closed the stream: that cancels the request (2026-07-28). The tool
                    # stops waiting; a job it started goes on and keeps its result in the workspace.
                    alive[0] = False
                    self.proto.cancel(sid, msg.get("id"))

        write("")  # priming event (id, empty data)
        done = threading.Event()

        def keepalive():
            while not done.wait(KEEPALIVE):
                write(None, "keep-alive")

        threading.Thread(target=keepalive, daemon=True).start()
        try:
            resp = self.proto.handle(
                msg, lambda m: write(jsonsafe.dumps(m, default=str)), scope=sid
            )
        finally:
            done.set()
        if resp is not None:
            write(jsonsafe.dumps(resp, default=str))

    # ------------------------------------------------------------------ files

    def _download(self, rel: str):
        ws = self.proto.svc.ws
        try:
            full = ws.path(rel, must_exist=True)
        except ServiceError as e:
            self._fail(404, str(e))
            return
        if full.is_dir():
            self._send(200, ws.listing(rel, 1))
            return
        size = full.stat().st_size
        ctype = mimetypes.guess_type(full.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{full.name}"')
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(full, "rb") as f, contextlib.suppress(OSError):
            while chunk := f.read(1 << 20):
                self.wfile.write(chunk)

    def _upload(self, rel: str):
        ws = self.proto.svc.ws
        try:
            full = ws.path(rel)
            if (
                ws.root / "jobs" in [full, *full.parents]
                or full == ws.root
                or not rel
                or rel.endswith("/")
            ):
                raise ServiceError(
                    "upload to data/<name> (or another file path of the workspace, not jobs/)"
                )
            n = int(self.headers.get("Content-Length", "-1"))
        except (ServiceError, ValueError) as e:
            self._fail(400, str(e))
            self.close_connection = True
            return
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked" or n < 0:
            self._fail(411, "Content-Length required")
            self.close_connection = True
            return
        if n > self.cfg.max_upload:
            self._fail(413, f"{n:,} bytes is over the upload limit of {self.cfg.max_upload:,}")
            self.close_connection = True
            return
        existed = full.exists()
        if existed and (full.is_dir() or self.headers.get("If-None-Match") == "*"):
            self._fail(409, f"{ws.rel(full)} exists")
            self.close_connection = True
            return
        full.parent.mkdir(parents=True, exist_ok=True)
        tmp = full.with_name(full.name + f".part-{secrets.token_hex(3)}")
        left = n
        try:
            with open(tmp, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        raise OSError("connection closed before the whole body arrived")
                    f.write(chunk)
                    left -= len(chunk)
            tmp.replace(full)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            self._fail(400, f"upload failed: {e}")
            self.close_connection = True
            return
        self._send(200 if existed else 201, {"path": ws.rel(full), "bytes": n})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(
        self, request, client_address
    ):  # a dropped connection is not worth a traceback
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)


def make_server(svc: mcp_tools.Service, cfg: Config, certfile=None, keyfile=None) -> Server:
    """The HTTP server, bound but not yet serving. Refuses a non-loopback bind without a token."""
    if not cfg.loopback and not cfg.token:
        raise ValueError(
            f"refusing to listen on {cfg.host} without authentication: give --token / --token-file / "
            "DLNET_TOKEN (or --generate-token), or bind to 127.0.0.1"
        )
    family_v6 = ":" in cfg.host
    if family_v6:
        import socket

        Server.address_family = socket.AF_INET6
    httpd = Server((cfg.host, cfg.port), Handler)
    httpd.cfg, httpd.proto, httpd.sessions = cfg, mcp_server.Protocol(svc), Sessions()
    scheme = "http"
    if certfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(certfile, keyfile)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    shown = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
    shown = f"[{shown}]" if ":" in shown else shown
    httpd.url = f"{scheme}://{shown}:{httpd.server_address[1]}"
    if svc.base_url is None:
        svc.base_url = httpd.url
    return httpd


def read_token(args) -> str | None:
    if getattr(args, "token", None):
        return args.token
    if getattr(args, "token_file", None):
        return Path(args.token_file).read_text().strip() or None
    return os.environ.get("DLNET_TOKEN") or None


def add_parser(sub):
    p = sub.add_parser(
        "serve",
        aliases=["mcp"],
        help="run DLNet as a remote MCP server (Streamable HTTP) for Claude Code, Codex, Antigravity and other AI clients",
    )
    p.add_argument(
        "--host", default="127.0.0.1", help="address to listen on (default: this machine only)"
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--workspace", default="dlnet-workspace", help="folder for data/, runs/ and jobs/"
    )
    p.add_argument(
        "--token", help="bearer token the clients must send (or DLNET_TOKEN, or --token-file)"
    )
    p.add_argument("--token-file", help="file holding the bearer token")
    p.add_argument(
        "--generate-token", action="store_true", help="make a random token and print it once"
    )
    p.add_argument(
        "--allow-origin", action="append", default=[], help="extra allowed Origin (browser clients)"
    )
    p.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="extra allowed Host header (behind a proxy)",
    )
    p.add_argument("--public-url", help="URL the clients use (behind a proxy), for download links")
    p.add_argument(
        "--max-jobs", type=int, default=1, help="jobs running at the same time (default 1)"
    )
    p.add_argument(
        "--max-budget", default="12h", help="largest wall-clock budget a design may ask for"
    )
    p.add_argument(
        "--wait", type=float, default=45.0, help="default seconds a tool waits for its job"
    )
    p.add_argument("--max-upload-gb", type=float, default=8.0)
    p.add_argument(
        "--allow-fetch", action="store_true", help="enable dlnet_fetch (download public https URLs)"
    )
    p.add_argument(
        "--no-sandbox",
        action="store_true",
        help="accept paths outside the workspace (trusted clients only)",
    )
    p.add_argument("--stateless", action="store_true", help="do not issue MCP session ids")
    p.add_argument("--certfile", help="TLS certificate (PEM): serve https")
    p.add_argument("--keyfile", help="TLS private key (PEM)")
    p.add_argument("--quiet", action="store_true", help="no access log")
    return p


def run(args) -> int:
    token = read_token(args)
    if args.generate_token and not token:
        token = secrets.token_urlsafe(32)
        print(f"[serve] generated token (shown once): {token}", file=sys.stderr)
    svc = mcp_tools.Service(args.workspace, sandbox=not args.no_sandbox, max_jobs=args.max_jobs,
                            default_wait=args.wait, max_budget=args.max_budget, allow_fetch=args.allow_fetch,
                            base_url=args.public_url.rstrip("/") if args.public_url else None)  # fmt: skip
    cfg = Config(args.host, args.port, token, args.allow_origin, args.allow_host,
                 int(args.max_upload_gb * 1024**3), args.stateless, args.quiet)  # fmt: skip
    try:
        httpd = make_server(svc, cfg, args.certfile, args.keyfile)
    except (ValueError, OSError) as e:
        print(f"[serve] {e}", file=sys.stderr)
        return 2
    auth = "bearer token required" if token else "NO AUTHENTICATION (loopback only)"
    print(f"[serve] DLNet MCP server on {httpd.url}/mcp  ({auth})", file=sys.stderr)
    print(f"[serve] workspace: {svc.ws.root}   jobs at a time: {args.max_jobs}", file=sys.stderr)
    print(f"[serve] connect, e.g.: claude mcp add --transport http dlnet {svc.base_url}/mcp"
          + (' --header "Authorization: Bearer $DLNET_TOKEN"' if token else ""), file=sys.stderr)  # fmt: skip
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[serve] stopping", file=sys.stderr)
    finally:
        httpd.server_close()
        svc.close()
    return 0
