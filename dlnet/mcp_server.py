"""The protocol layer of the DLNet MCP server: JSON-RPC messages of MCP -> tool calls.

The server itself is `dlnet serve` (dlnet/mcp_http.py, Streamable HTTP); the tools are in
dlnet/mcp_tools.py. An AI client such as Claude Code, Codex CLI or Google Antigravity connects to
it to profile data, write and check architectures, run `design`, train, predict and read the
results. Optional; nothing else in DLNet depends on it. (The stdio server of DLNet <= 1.4 was
removed in 1.5: one server, reachable from the same machine or over the network.)

Written from the MCP specification without the MCP SDK, to avoid a dependency. Two eras of the
protocol are served side by side ("dual-era", specification 2026-07-28, basic/versioning):

  legacy   revisions 2024-11-05 ... 2025-11-25: `initialize` handshake (version negotiation: the
           client's revision when supported, else the newest legacy one), notifications/initialized,
           ping.
  modern   revision 2026-07-28: no handshake; every request carries its protocol version in
           `params._meta["io.modelcontextprotocol/protocolVersion"]`; `server/discover` advertises
           versions, capabilities and identity; every result has `resultType: "complete"` and the
           server identity in `_meta`; list results carry the caching hints `ttlMs`, `cacheScope`;
           an unsupported version is answered with UnsupportedProtocolVersionError (-32022).

Both: tools/list, tools/call (text content + structuredContent; tool errors as `isError`),
prompts/list, prompts/get, notifications/progress for a call that carries a progress token, and
cancellation (notifications/cancelled; in the modern era a closed response stream).

Client configuration: doc/mcp-server.md.
"""

from __future__ import annotations

import json
import threading

from . import jsonsafe, mcp_tools
from .mcp_tools import TOOLS  # noqa: F401  (re-exported: the tool list of the service)
from .service import ServiceError

MODERN = ("2026-07-28",)
LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SUPPORTED = (*MODERN, *LEGACY)
PROTOCOL = LEGACY[0]  # what `initialize` falls back to
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602
HEADER_MISMATCH, UNSUPPORTED_VERSION = -32020, -32022
META = "io.modelcontextprotocol/"
CAPABILITIES = {"tools": {"listChanged": False}, "prompts": {"listChanged": False}}
LIST_TTL_MS = 3_600_000  # the tool and prompt lists only change with the DLNet version


def unsupported_version(mid, requested) -> dict:
    e = error(mid, UNSUPPORTED_VERSION, "Unsupported protocol version")
    e["error"]["data"] = {"supported": list(SUPPORTED), "requested": requested}
    return e


def request_version(msg) -> str | None:
    """The protocol version a modern request declares in its `_meta` (None: a legacy message)."""
    params = msg.get("params") if isinstance(msg, dict) else None
    meta = params.get("_meta") if isinstance(params, dict) else None
    return meta.get(META + "protocolVersion") if isinstance(meta, dict) else None


def error(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


class Protocol:
    """JSON-RPC messages of MCP -> calls on a `mcp_tools.Service`. Transport-independent."""

    def __init__(self, svc: mcp_tools.Service):
        self.svc = svc
        self.lock = threading.Lock()
        self.cancelled: dict = {}  # (scope, request id) -> threading.Event of an in-flight call

    def negotiate(self, requested) -> str:
        return requested if requested in LEGACY else PROTOCOL

    @staticmethod
    def server_info() -> dict:
        from . import __version__

        return {"name": "dlnet", "title": "DLNet", "version": __version__}

    def initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": self.negotiate((params or {}).get("protocolVersion")),
            "capabilities": CAPABILITIES,
            "serverInfo": self.server_info(),
            "instructions": mcp_tools.INSTRUCTIONS,
        }

    def discover(self) -> dict:
        return {"supportedVersions": list(SUPPORTED), "capabilities": CAPABILITIES,
                "instructions": mcp_tools.INSTRUCTIONS, "ttlMs": LIST_TTL_MS, "cacheScope": "public"}  # fmt: skip

    def cancel(self, scope, request_id) -> None:
        with self.lock:
            ev = self.cancelled.get((scope, request_id))
        if ev is not None:
            ev.set()

    def handle(self, msg, notify=None, scope=None) -> dict | None:
        """One JSON-RPC message -> its response (None for a notification or a cancelled call).
        `notify(dict)` sends a server notification to the caller while the call runs."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" and "method" not in msg:
            return error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if method is None:  # a response to a server request: this server sends none
            return None
        if mid is None:  # notification
            if method == "notifications/cancelled":
                self.cancel(scope, params.get("requestId"))
            return None
        if not isinstance(params, dict):
            return error(mid, INVALID_PARAMS, "params must be an object")
        version = request_version(msg)
        modern = version is not None
        if modern and version not in MODERN:
            return unsupported_version(mid, version)
        cache = {"ttlMs": LIST_TTL_MS, "cacheScope": "public"} if modern else {}
        if method == "server/discover":
            result = self.discover()
        elif method == "initialize" and not modern:
            result = self.initialize(params)
        elif method == "ping" and not modern:
            result = {}
        elif method == "tools/list":
            result = {"tools": mcp_tools.TOOLS, **cache}
        elif method == "prompts/list":
            result = {"prompts": mcp_tools.PROMPTS, **cache}
        elif method == "prompts/get":
            try:
                text = mcp_tools.prompt_text(params.get("name"), params.get("arguments") or {})
            except ServiceError as e:
                return error(mid, INVALID_PARAMS, str(e))
            result = {"messages": [{"role": "user", "content": {"type": "text", "text": text}}]}
        elif method == "tools/call":
            resp = self._call(mid, params, notify, scope)
            return self._stamp(resp) if modern and resp is not None and "result" in resp else resp
        else:
            return error(mid, METHOD_NOT_FOUND, f"method not found: {method}")
        resp = {"jsonrpc": "2.0", "id": mid, "result": result}
        return self._stamp(resp) if modern or method == "server/discover" else resp

    def _stamp(self, resp: dict) -> dict:
        """What every modern result carries: `resultType` and the server's identity."""
        r = resp["result"]
        r["resultType"] = "complete"
        r.setdefault("_meta", {})[META + "serverInfo"] = self.server_info()
        return resp

    def _call(self, mid, params, notify, scope):
        name, token = params.get("name"), (params.get("_meta") or {}).get("progressToken")
        if not any(t["name"] == name for t in mcp_tools.TOOLS):
            return error(mid, INVALID_PARAMS, f"unknown tool {name!r}")
        ev = threading.Event()
        with self.lock:
            self.cancelled[(scope, mid)] = ev

        last = [float("-inf")]

        def progress(done, total=None, message=None):
            if notify is None or token is None or done <= last[0]:  # the value MUST increase
                return
            last[0] = done
            p = {"progressToken": token, "progress": done}
            if total is not None:
                p["total"] = total
            if message:
                p["message"] = message
            notify({"jsonrpc": "2.0", "method": "notifications/progress", "params": p})

        try:
            out = self.svc.call(name, params.get("arguments") or {}, progress, ev.is_set)
            if ev.is_set():  # cancelled by the client: no response (MCP cancellation)
                return None
            text = jsonsafe.dumps(out, default=str)
            result = {"content": [{"type": "text", "text": text}], "structuredContent": json.loads(text),
                      "isError": False}  # fmt: skip
        except (
            ServiceError,
            OSError,
            KeyError,
            ValueError,
            TypeError,
            RuntimeError,
            SystemExit,
        ) as e:
            if ev.is_set():
                return None
            result = {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}
        finally:
            with self.lock:
                self.cancelled.pop((scope, mid), None)
        return {"jsonrpc": "2.0", "id": mid, "result": result}
