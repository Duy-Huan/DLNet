"""M18: AI advisor (scripted fake CLI) and the MCP server (JSON-RPC over stdio)."""

import io
import json
import sys

import numpy as np

from dlnet import advisor, mcp_server
from dlnet.zoo import presets


def _cands():
    out = []
    for name, spec in list(presets.presets("map", (1, 8, 8), "cls", 8).items())[:6]:
        out.append({"family": name, "origin": f"preset:{name}", "spec": spec, "params": 1000})
    return out


def _fake(tmp_path, reply: str):
    f = tmp_path / "fake_ai.py"
    f.write_text(
        "import sys\nprompt = sys.stdin.read()\nassert 'data_card' in prompt\n"
        f"print('Here is my answer:\\n```json\\n' + {reply!r} + '\\n```')\n"
    )
    return f"{sys.executable} {f}"


def test_parse_accepts_fenced_and_bare_json():
    assert advisor.parse('blah ```json\n{"ranking": ["c1"]}\n``` more') == {"ranking": ["c1"]}
    assert advisor.parse('ok {"ranking": []} done') == {"ranking": []}


def test_advisor_reply_is_validated_as_data(tmp_path):
    good = {"op": "seq", "body": [{"op": "conv_bn_act", "out": 8}, {"op": "basic_block", "out": 8}]}
    reply = json.dumps({
        "ranking": ["c2", "c0", "c99", 5, "c2"],                 # unknown id, wrong type, duplicate
        "proposals": [good, {"op": "no_such_op"}, "rm -rf /", {"op": "seq", "body": 3}],
        "rationale": "small images: shallow CNNs",
    })  # fmt: skip
    out = advisor.advise(
        _fake(tmp_path, reply), {"n_train": 100}, "", "accuracy", _cands(), "map", (1, 8, 8), 10,
        "cls", {"lr": [1e-3], "wd": [1e-4], "batch": [32]}, 10**6, np.random.default_rng(0),
        log=lambda *_: None,
    )  # fmt: skip
    assert out["ranking"] == [2, 0]
    assert len(out["proposals"]) == 1 and out["proposals"][0]["origin"] == "ai_advisor"
    assert out["invalid_proposals"] == 3 and out["dropped_ids"] == ["c99", 5]
    s = advisor.scores_from_ranking(4, [2, 0])
    assert s[2] > s[0] > s[1] == s[3]


def test_advisor_failure_does_not_stop_the_run(tmp_path):
    logs = []
    args = ({"n_train": 1}, "", "accuracy", _cands(), "map", (1, 8, 8), 10, "cls", {}, 10**6,
            np.random.default_rng(0))  # fmt: skip
    assert advisor.advise("no-such-binary-xyz", *args, log=logs.append) is None
    bad = tmp_path / "bad.py"
    bad.write_text("print('I cannot help with that')")
    assert advisor.advise(f"{sys.executable} {bad}", *args, log=logs.append) is None
    assert all("[advisor] skipped" in m for m in logs)


def _rpc(*msgs):
    out = io.StringIO()
    mcp_server.serve(io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n"), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_mcp_handshake_tools_and_errors():
    r = _rpc(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "dlnet_check_spec",
         "arguments": {"spec": {"op": "basic_block", "out": 16}, "kind": "map", "dims": [3, 8, 8]}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "dlnet_check_spec",
         "arguments": {"spec": {"op": "nope"}, "kind": "map", "dims": [3, 8, 8]}}},
        {"jsonrpc": "2.0", "id": 5, "method": "nope"},
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "nope"}},
    )  # fmt: skip
    assert [m["id"] for m in r] == [1, 2, 3, 4, 5, 6]  # the notification gets no reply
    assert (
        r[0]["result"]["protocolVersion"] == "2025-03-26"
        and "tools" in r[0]["result"]["capabilities"]
    )
    names = {t["name"] for t in r[1]["result"]["tools"]}
    assert {"dlnet_profile", "dlnet_candidates", "dlnet_design", "dlnet_leaderboard"} <= names
    assert all("inputSchema" in t for t in r[1]["result"]["tools"])
    ok = json.loads(r[2]["result"]["content"][0]["text"])
    assert ok["valid"] and ok["params"] > 0 and ok["dims"] == [16, 8, 8]
    assert json.loads(r[3]["result"]["content"][0]["text"])["valid"] is False
    assert r[4]["error"]["code"] == -32601 and r[5]["error"]["code"] == -32602


def test_mcp_candidates_on_images(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "img.npz"
    np.savez(p, X=rng.normal(size=(60, 8, 8)).astype("float32"), y=np.repeat(np.arange(3), 20))
    r = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": "dlnet_candidates", "arguments": {"data": str(p), "zoo": 4, "limit": 5}}})  # fmt: skip
    body = json.loads(r[0]["result"]["content"][0]["text"])
    assert body["plugin"] == "image_classification" and body["input"]["kind"] == "map"
    assert 0 < len(body["candidates"]) <= 5 and all("spec" in c for c in body["candidates"])
    assert "conv_bias_gain" in body["data_card"]
