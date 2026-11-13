"""v1.7 web UI (`dlnet ui`): the two languages are in sync (100%), every text of the page comes from
them, and the local server refuses what it must (other hosts, other origins, no token, paths
outside the workspace)."""

from __future__ import annotations

import http.client
import json
import re
import threading
from pathlib import Path

import pytest

from dlnet import ui_server

UI = Path(__file__).resolve().parent.parent / "dlnet" / "ui"
# keys built at run time in app.js: "<prefix>" + value; every key with the prefix counts as used
DYNAMIC = (
    "kind.",
    "status.",
    "theme.",
    "tool.",
    "verdict.",
    "ai.client.",
    "plan.status.",
    "plan.warn.",
)


def _dicts():
    return {
        lang: json.loads((UI / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))
        for lang in ("en", "vi")
    }


def test_languages_have_exactly_the_same_keys():
    d = _dicts()
    assert set(d["en"]) == set(d["vi"]), {
        "only_en": sorted(set(d["en"]) - set(d["vi"])),
        "only_vi": sorted(set(d["vi"]) - set(d["en"])),
    }


def test_every_text_is_translated_and_keeps_its_placeholders():
    d = _dicts()
    for k in d["en"]:
        for lang in ("en", "vi"):
            assert isinstance(d[lang][k], str) and d[lang][k].strip(), f"{lang}.{k} is empty"
        ph = {lang: sorted(re.findall(r"\{(\w+)\}", d[lang][k])) for lang in ("en", "vi")}
        assert ph["en"] == ph["vi"], f"{k}: placeholders differ {ph}"


def test_every_key_used_exists_and_every_key_is_used():
    d = _dicts()["en"]
    js = (UI / "app.js").read_text(encoding="utf-8")
    html = (UI / "index.html").read_text(encoding="utf-8")
    used = set(re.findall(r'\bt\("([a-z_.0-9]+)"', js)) | set(
        re.findall(r'data-i18n="([a-z_.0-9]+)"', html)
    )
    used -= {u for u in used if u.endswith(".")}  # the dynamic prefixes themselves
    missing = sorted(used - set(d))
    assert not missing, f"keys used but not in the dictionaries: {missing}"
    unused = sorted(k for k in d if k not in used and not k.startswith(DYNAMIC))
    assert not unused, f"keys in the dictionaries but never used: {unused}"
    for prefix in DYNAMIC:
        assert f'"{prefix}"' in js or f"'{prefix}'" in js, (
            f"dynamic prefix {prefix} not used in app.js"
        )


def test_dynamic_keys_cover_every_value():
    from dlnet import mcp_tools, plugins

    d = _dicts()["en"]
    assert {f"tool.{t['name']}" for t in mcp_tools.TOOLS} <= set(d)
    assert {f"kind.{n}" for n in plugins._registry()} <= set(d)
    from dlnet.service import FINAL

    assert {f"status.{s}" for s in (*FINAL, "queued", "running", "empty", "done")} <= set(d)


def test_no_hard_coded_vietnamese_in_the_code():
    js = (UI / "app.js").read_text(encoding="utf-8")
    viet = re.findall(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", js, re.IGNORECASE)
    assert not viet, "Vietnamese text belongs in i18n/vi.json"


def test_both_themes_define_every_colour_token():
    css = (UI / "app.css").read_text(encoding="utf-8")
    light = set(re.findall(r"(--[a-z-]+):", css.split("@media")[0]))
    dark_media = css.split("@media (prefers-color-scheme: dark)")[1].split("}\n}")[0]
    dark_attr = css.split(':root[data-theme="dark"]')[1].split("}")[0]
    for block in (dark_media, dark_attr):
        assert light == set(re.findall(r"(--[a-z-]+):", block))


# ----------------------------------------------------------------------------- the server


@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.setenv("DLNET_UI_STATE", str(tmp_path / "state.json"))
    srv = ui_server.make_server(tmp_path / "ws", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    port = srv.server_address[1]
    yield srv, port
    srv.shutdown()
    srv.app.svc.close()
    srv.server_close()


def _req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    h = {"Host": f"127.0.0.1:{port}", **(headers or {})}
    data = (
        body
        if isinstance(body, bytes)
        else (json.dumps(body).encode() if body is not None else None)
    )
    c.request(method, path, data, h)
    r = c.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw


def test_page_carries_a_token_and_the_api_requires_it(ui):
    srv, port = ui
    st, page = _req(port, "GET", "/")
    assert st == 200 and srv.app.token.encode() in page
    assert _req(port, "GET", "/api/state")[0] == 401
    assert _req(port, "GET", "/api/state", headers={"X-DLNet-Token": "wrong"})[0] == 401
    st, s = _req(port, "GET", "/api/state", headers={"X-DLNet-Token": srv.app.token})
    assert st == 200 and s["workspace"].endswith("ws") and s["mcp"]["running"] is False


def test_other_hosts_and_origins_are_refused(ui):
    srv, port = ui
    tok = {"X-DLNet-Token": srv.app.token}
    assert _req(port, "GET", "/", headers={"Host": "evil.example:80"})[0] == 403  # DNS rebinding
    assert _req(port, "GET", "/api/state", headers={**tok, "Host": "evil.example"})[0] == 403
    assert (
        _req(port, "POST", "/api/tool/dlnet_files", {}, {**tok, "Origin": "http://evil.example"})[0]
        == 403
    )
    assert (
        _req(port, "GET", "/api/state", headers={**tok, "Origin": f"http://localhost:{port}"})[0]
        == 200
    )


def test_static_files_and_no_escape(ui):
    _, port = ui
    assert _req(port, "GET", "/static/app.js")[0] == 200
    assert _req(port, "GET", "/static/i18n/vi.json")[0] == 200
    assert _req(port, "GET", "/static/../ui_server.py")[0] == 404


def test_workspace_files_tools_upload_download(ui, tmp_path):
    srv, port = ui
    tok = {"X-DLNet-Token": srv.app.token}
    # upload into data/ only
    st, r = _req(port, "PUT", "/files/data/x.csv", b"a,b\n1,2\n", tok)
    assert st == 200 and r["bytes"] == 8
    assert _req(port, "PUT", "/files/data/x.csv", b"again", tok)[0] == 400  # exists
    assert _req(port, "PUT", "/files/runs/x.csv", b"a", tok)[0] == 400
    assert _req(port, "PUT", "/files/data/y.csv", b"a")[0] == 401
    # tools are the MCP tools
    st, r = _req(port, "POST", "/api/tool/dlnet_files", {}, tok)
    assert st == 200 and any(e["path"] == "data/x.csv" for e in r["result"]["entries"])
    st, r = _req(port, "POST", "/api/tool/dlnet_read_file", {"path": "../../etc/passwd"}, tok)
    assert st == 400
    st, r = _req(port, "POST", "/api/tool/dlnet_nope", {}, tok)
    assert st == 400
    # downloads: token in the query (img / a links), shown as text, never run
    st, raw = _req(port, "GET", f"/files/data/x.csv?t={srv.app.token}")
    assert st == 200 and raw == b"a,b\n1,2\n"
    assert _req(port, "GET", "/files/data/x.csv")[0] == 401
    assert _req(port, "GET", f"/files/../secret?t={srv.app.token}")[0] == 404


def test_runs_listing_and_detail(ui):
    srv, port = ui
    tok = {"X-DLNet-Token": srv.app.token}
    run = srv.app.ws.root / "runs" / "demo"
    run.mkdir(parents=True)
    board = {"task": "classification", "metric": "accuracy", "verdict": {"beats_reference": True},
             "top_k": [{"rank": 1, "name": "top1_cnn", "params": 1000, "val_mean": 0.9, "val_std": 0.01,
                        "n_runs": 3, "test": {"score": 0.88}, "test_std": 0.01}],
             "reference": [{"name": "logreg", "val": 0.7, "test": {"score": 0.69}}], "budget": {"budget": "10m"},
             "papers": []}  # fmt: skip
    (run / "leaderboard.json").write_text(json.dumps(board))
    st, r = _req(port, "GET", "/api/runs", headers=tok)
    assert st == 200 and r["runs"][0]["name"] == "demo" and r["runs"][0]["status"] == "done"
    assert r["runs"][0]["val"] == 0.9 and r["runs"][0]["test"] == 0.88
    st, d = _req(port, "GET", "/api/run/demo", headers=tok)
    assert st == 200 and d["board"]["top_k"][0]["name"] == "top1_cnn"
    assert _req(port, "GET", "/api/run/..%2F..", headers=tok)[0] == 400


def test_browse_and_switch_workspace(ui, tmp_path):
    srv, port = ui
    tok = {"X-DLNet-Token": srv.app.token}
    (tmp_path / "other" / "runs").mkdir(parents=True)
    st, b = _req(port, "GET", f"/api/browse?path={tmp_path}", headers=tok)
    assert st == 200 and any(d["name"] == "other" and d["workspace"] for d in b["dirs"])
    st, r = _req(port, "POST", "/api/workspace", {"path": str(tmp_path / "other")}, tok)
    assert st == 200 and r["workspace"] == str((tmp_path / "other").resolve())
    assert (tmp_path / "other" / "data").is_dir() and (tmp_path / "other" / "papers").is_dir()
    assert _req(port, "POST", "/api/workspace", {"path": str(tmp_path / "missing")}, tok)[0] == 400
    st, r = _req(
        port, "POST", "/api/workspace", {"path": str(tmp_path / "new"), "create": True}, tok
    )
    assert st == 200 and (tmp_path / "new" / "runs").is_dir()
    st, s = _req(port, "GET", "/api/state", headers=tok)
    assert s["recent"][0] == str((tmp_path / "new").resolve())


def test_mcp_server_start_test_stop(ui):
    srv, port = ui
    tok = {"X-DLNet-Token": srv.app.token}
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        mport = s.getsockname()[1]
    st, r = _req(port, "POST", "/api/mcp/start", {"port": mport}, tok)
    try:
        assert (
            st == 200
            and r["running"]
            and r["token"]
            and r["url"] == f"http://127.0.0.1:{mport}/mcp"
        )
        st, t = _req(port, "POST", "/api/mcp/test", {}, tok)
        assert st == 200 and t["ok"] and t["tools"] >= 20, t
        st, s = _req(port, "GET", "/api/mcp", headers=tok)
        assert s["running"] and any("POST /mcp" in line for line in s["log"])
    finally:
        _req(port, "POST", "/api/mcp/stop", {}, tok)
    st, s = _req(port, "GET", "/api/mcp", headers=tok)
    assert not s["running"]
    assert _req(port, "POST", "/api/mcp/test", {}, tok)[1]["ok"] is False


def _tokens(block):
    return dict(re.findall(r"(--[a-z-]+):\s*(#[0-9A-Fa-f]{6})", block))


def _contrast(a, b):
    def lum(h):
        c = [int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_text_contrast_is_at_least_4_5_in_both_themes():
    """WCAG 2.1 AA for normal text, for every text colour on every background it is used on."""
    css = (UI / "app.css").read_text(encoding="utf-8")
    themes = {"light": _tokens(css.split("@media")[0]),
              "dark": _tokens(css.split(':root[data-theme="dark"]')[1].split("}")[0])}  # fmt: skip
    pairs = [("text", "bg"), ("text", "surface"), ("text", "sunken"), ("text", "accent-soft"),
             ("muted", "bg"), ("muted", "surface"), ("muted", "sunken"), ("accent", "bg"),
             ("accent", "surface"), ("accent", "accent-soft"), ("ok", "surface"), ("warn", "bg"),
             ("warn", "surface"), ("bad", "bg"), ("bad", "surface"), ("invert-text", "invert-bg")]  # fmt: skip
    for name, tok in themes.items():
        for fg, bg in pairs:
            r = _contrast(tok["--" + fg], tok["--" + bg])
            assert r >= 4.5, f"{name}: --{fg} on --{bg} is {r:.2f}:1"


def test_plan_codes_have_translations():
    """Warnings and paper statuses of `dlnet plan` reach the page as codes the UI translates."""
    src = (Path(__file__).resolve().parent.parent / "dlnet" / "plan.py").read_text(encoding="utf-8")
    d = _dicts()
    warns = set(re.findall(r'_warn\(\s*"(\w+)"', src))
    codes = set(re.findall(r'row\["code"\][^=]*= "(\w+)"', src)) | {"ok"}
    assert warns >= {"few_candidates", "untimed", "papers_skipped"} and codes >= {
        "ok",
        "level_a",
        "input",
        "fail",
    }
    for lang in ("en", "vi"):
        assert {f"plan.warn.{w}" for w in warns} <= set(d[lang])
        assert {f"plan.status.{c}" for c in codes} <= set(d[lang])


def test_run_in_progress_shows_job_log_and_can_be_stopped(ui):
    import sys
    import time

    srv, port = ui
    tok = {"X-DLNet-Token": srv.app.token}
    out = srv.app.ws.root / "runs" / "live"
    out.mkdir(parents=True)
    argv = [sys.executable, "-c", "import time; print('[search] rung 1', flush=True); time.sleep(60)",
            "design", "--data", "data/x.npz", "--budget", "5m"]  # fmt: skip
    job = srv.app.svc.jobs.submit("design", argv, out=out, title="design x")
    for _ in range(100):
        if srv.app.svc.jobs.log_tail(job, 1):
            break
        time.sleep(0.1)
    st, r = _req(port, "GET", "/api/runs", headers=tok)
    row = next(x for x in r["runs"] if x["name"] == "live")
    assert row["status"] == "running" and row["job_id"] == job.id and "rung 1" in row["progress"]
    assert row["command"][-4:] == ["--data", "data/x.npz", "--budget", "5m"]
    st, d = _req(port, "GET", "/api/run/live", headers=tok)
    assert st == 200 and d["board"] is None and any("rung 1" in line for line in d["log"])
    st, c = _req(port, "POST", "/api/tool/dlnet_cancel", {"job_id": job.id}, tok)
    assert st == 200 and c["result"]["status"] == "cancelled"
    assert _req(port, "GET", "/api/nothing", headers=tok)[0] == 404


def test_bad_state_file_and_requests(ui, tmp_path, monkeypatch):
    srv, port = ui
    tok = {"X-DLNet-Token": srv.app.token}
    ui_server.state_file().write_text("{not json")
    assert ui_server.load_state() == {}
    st, s = _req(port, "GET", "/api/state", headers=tok)
    assert st == 200 and s["recent"] == []
    assert _req(port, "POST", "/api/workspace", b"[1, 2]", tok)[0] == 400  # not an object
    assert _req(port, "POST", "/api/workspace", b"{bad", tok)[0] == 400
    assert _req(port, "GET", f"/api/browse?path={tmp_path / 'nope'}", headers=tok)[0] == 400
    assert _req(port, "PUT", "/files/data/z.bin", b"", {**tok, "Content-Length": "-1"})[0] == 400


def test_ui_command_starts_and_stops(tmp_path, monkeypatch, capsys):
    from dlnet import cli

    monkeypatch.setenv("DLNET_UI_STATE", str(tmp_path / "state.json"))
    opened = []
    monkeypatch.setattr(ui_server.webbrowser, "open", opened.append)

    def stop(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(ui_server.Server, "serve_forever", stop)
    assert cli.main(["ui", "--workspace", str(tmp_path / "w"), "--port", "0"]) == 0
    err = capsys.readouterr().err
    assert (
        "[ui] DLNet" in err
        and "stopping" in err
        and opened
        and opened[0].startswith("http://127.0.0.1:")
    )
    # the next start opens the last workspace; a port in use is reported, not a traceback
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert cli.main(["ui", "--port", str(busy), "--no-browser"]) == 2
    assert "cannot listen" in capsys.readouterr().err
    assert ui_server.load_state()["recent"][0] == str((tmp_path / "w").resolve())
