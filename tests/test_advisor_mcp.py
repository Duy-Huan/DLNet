"""M18: AI advisor (scripted fake CLI). The MCP server is tested in tests/test_mcp_http.py."""

import json
import sys

import numpy as np

from dlnet import advisor
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
