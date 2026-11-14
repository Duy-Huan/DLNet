"""The MCP tools (dlnet/mcp_tools.py): argument checks, the outcome of every job kind, and the paths a
real client rarely takes (v1.9 coverage). Jobs here are small Python scripts, not real runs."""

from __future__ import annotations

import base64
import json
import sys

import pytest
from test_mcp_http import _fetch_svc

from dlnet import mcp_tools, service
from dlnet.service import ServiceError


def _py(code):
    return [sys.executable, "-c", code]


def nop(*a):
    return None


@pytest.fixture
def svc(tmp_path):
    s = mcp_tools.Service(tmp_path / "ws", default_wait=30)
    yield s
    s.close()


def test_arguments_and_small_tools(svc, monkeypatch):
    with pytest.raises(ServiceError, match="must be an object"):
        svc.call("dlnet_info", ["not", "a", "dict"])
    monkeypatch.setattr(service, "WRITE_LIMIT", 4)
    with pytest.raises(ServiceError, match="byte limit"):
        svc.call(
            "dlnet_write_file",
            {"path": "data/a.txt", "base64": base64.b64encode(b"12345").decode()},
        )
    ops = svc.call("dlnet_zoo", {})["ops"]
    assert any(o["op"] == "linear" for o in ops)


def test_check_spec_with_a_diagram_that_cannot_be_drawn(svc, monkeypatch):
    from dlnet import diagram

    def boom(cfg):
        raise ValueError("no room")

    monkeypatch.setattr(diagram, "svg", boom)
    out = svc.call("dlnet_check_spec", {"spec": [{"op": "linear", "out": 4}], "kind": "vec", "dims": [3],
                                        "diagram": True})  # fmt: skip
    assert out["valid"] and out["diagram_error"] == "no room"


def test_a_download_over_4_gb_is_refused(tmp_path, monkeypatch):
    svc = _fetch_svc(tmp_path, monkeypatch)

    class Huge(bytes):
        def __len__(self):
            return 5 * 1024**3

    class Body:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return Huge(b"x")

    monkeypatch.setattr(mcp_tools.urllib.request, "urlopen", lambda req, timeout: Body())
    with pytest.raises(ServiceError, match="larger than 4 GB"):
        svc.tool_fetch({"url": "https://example.org/big.npz"}, nop, lambda: False)


def _job(svc, kind, code, **kw):
    j = svc.jobs.submit(kind, _py(code), **kw)
    assert j.done.wait(30)
    return j


def test_outcomes_of_every_job_kind(svc):
    ws = svc.ws
    plan = ws.root / "jobs" / "p.json"
    j = _job(
        svc,
        "plan",
        f"open({str(plan)!r}, 'w').write('{{\"ok\": 1}}')",
        extra={"plan": ws.rel(plan)},
    )
    assert svc._result(j)["plan"] == {"ok": 1}
    j = _job(
        svc,
        "train",
        "print(\"val: {'score': bad}\"); print('params=1,234')",
        extra={"package": "p"},
    )
    out = svc._result(j)
    assert out["val"] == "{'score': bad}" and out["params"] == 1234  # unparsable: kept as text
    j = _job(svc, "paper", "print('level B')")
    assert "level B" in svc._result(j)["output"]
    j = _job(svc, "predict", "pass", extra={"out": "data/none.npy"})
    assert svc._result(j)["bytes"] is None and "url" not in svc._result(j)  # no public URL
    j = _job(svc, "design", "pass", out=ws.root / "runs" / "empty")
    assert "could not be read" in svc._result(j)["result_error"]
    j = _job(svc, "other", "pass")
    assert svc._result(j)["ok"] is True


def test_design_arguments(svc):
    (svc.ws.root / "data" / "x.npz").write_bytes(b"")
    (svc.ws.root / "data" / "gt.npz").write_bytes(b"")
    (svc.ws.root / "data" / "card.yaml").write_text("")
    argv = svc._design_argv({"data": "data/x.npz", "gt": "data/gt.npz", "meta_store": "data/meta.json",
                             "papers": ["data/card.yaml"], "paper_recipe": True}, "design", svc.ws.root / "runs" / "r")  # fmt: skip
    assert (
        "--gt" in argv
        and "--meta-store" in argv
        and "--papers" in argv
        and "--paper-recipe" in argv
    )


def test_plan_tool_runs_a_plan(svc, tabular_csv):
    import shutil

    shutil.copy(tabular_csv, svc.ws.root / "data" / "t.csv")
    out = svc.call(
        "dlnet_plan", {"data": "data/t.csv", "target": "y", "budget": "30ep", "wait": 120}
    )
    assert out["status"] == "succeeded" and "plan" in out, out


def test_package_tools_refuse_incomplete_packages(svc, tmp_path):
    pkg = svc.ws.root / "runs" / "r" / "top1_x"
    pkg.mkdir(parents=True)
    (pkg / "config.yaml").write_text("{}")
    with pytest.raises(ServiceError, match="no train.py"):
        svc.call("dlnet_train", {"package": "runs/r/top1_x", "data": "data/t.csv"})
    with pytest.raises(ServiceError, match="no trained weights"):
        svc.call("dlnet_predict", {"package": "runs/r/top1_x", "data": "data/t.csv"})
    (svc.ws.root / "data" / "t.csv").write_text("a\n1\n")
    (svc.ws.root / "data" / "gt.npz").write_bytes(b"")
    (pkg / "train.py").write_text("import sys; print(sys.argv)")
    out = svc.call(
        "dlnet_train",
        {"package": "runs/r/top1_x", "data": "data/t.csv", "gt": "data/gt.npz", "wait": 30},
    )
    assert out["status"] == "succeeded" and "--gt" in " ".join(out["log_tail"])
    (pkg / "model.pt").write_bytes(b"")
    (pkg / "predict.py").write_text("import sys; print(sys.argv)")
    out = svc.call(
        "dlnet_predict",
        {"package": "runs/r/top1_x", "data": "data/t.csv", "device": "cpu", "wait": 30},
    )
    assert "--device" in " ".join(out["log_tail"])
    with pytest.raises(ServiceError, match="no leaderboard.json"):
        svc.call("dlnet_leaderboard", {"run": "runs/r"})
    assert svc._run_folder("runs/r") == svc.ws.root / "runs" / "r"


def test_paper_tool_checks_and_builds(svc):
    from pathlib import Path

    card = Path(__file__).resolve().parents[1] / "papers" / "resnet20_cifar.yaml"
    (svc.ws.root / "data" / "r.yaml").write_text(card.read_text())
    out = svc.call("dlnet_paper", {"action": "check", "cards": ["data/r.yaml"], "wait": 120})
    assert out["status"] == "succeeded"
    out = svc.call(
        "dlnet_paper", {"action": "build", "cards": ["data/r.yaml"], "name": "pp", "wait": 120}
    )
    assert out["status"] == "succeeded" and json.dumps(out).count("paper") > 1
