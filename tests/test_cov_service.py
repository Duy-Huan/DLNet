"""The workspace and job queue behind `dlnet serve` and `dlnet ui` (dlnet/service.py): the edges of
listing, reading and writing files, and every way a job can be cancelled (v1.9 coverage)."""

from __future__ import annotations

import sys
import time

import pytest

from dlnet import service
from dlnet.service import Jobs, ServiceError, Workspace


def _py(code):
    return [sys.executable, "-c", code]


def test_workspace_edges(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "data").mkdir(exist_ok=True)
    for i in range(5):
        (ws.root / "data" / f"f{i}.txt").write_text("x")
    assert ws.rel(tmp_path / "elsewhere") == str(tmp_path / "elsewhere")
    one = ws.listing("data/f0.txt")
    assert one["entries"][0]["path"].endswith("f0.txt")
    assert ws.listing("data", limit=2)["truncated"] is True
    with pytest.raises(ServiceError, match="is a folder"):
        ws.read_text("data")
    with pytest.raises(ServiceError, match="is a folder"):
        ws.write("data", b"x", overwrite=True)
    with pytest.raises(ServiceError, match="is a file"):
        ws.archive("data/f0.txt")


def test_frozen_program_restarts_itself(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert service.python() == [sys.executable]
    assert service.package_python() == [sys.executable, "run-script"]


def test_cancel_queued_running_stubborn_and_racing_jobs(tmp_path, monkeypatch):
    jobs = Jobs(Workspace(tmp_path / "ws"), max_jobs=1)
    first = jobs.submit("demo", _py("import time; time.sleep(30)"))
    queued = jobs.submit("demo", _py("print('never')"))
    deadline = time.time() + 20
    while first.status != "running" and time.time() < deadline:
        time.sleep(0.05)
    assert jobs.cancel(queued.id).status == "cancelled"  # never started

    # a job that ignores SIGTERM: cancel escalates to SIGKILL
    class Impatient:
        def __init__(self, ev):
            self.ev, self.calls = ev, 0

        def wait(self, timeout=None):
            self.calls += 1
            return False if self.calls == 1 else self.ev.wait(timeout)

        def set(self):
            self.ev.set()

    first.done = Impatient(first.done)
    assert jobs.cancel(first.id).status == "cancelled"
    Jobs._kill(first)  # the process is gone: nothing to do
    # cancelled between "running" and the start of its process
    real = service.subprocess.Popen

    def popen(*a, **k):
        p = real(*a, **k)
        for j in jobs.jobs.values():
            if j.status == "running":
                j.cancel_requested = True
        return p

    monkeypatch.setattr(service.subprocess, "Popen", popen)
    racing = jobs.submit("demo", _py("import time; time.sleep(30)"))
    assert racing.done.wait(20) and racing.status == "cancelled"
    monkeypatch.setattr(service.subprocess, "Popen", real)
    last = jobs.submit("demo", _py("import time; time.sleep(30)"))
    while last.status != "running" and time.time() < deadline + 20:
        time.sleep(0.05)
    jobs.shutdown()
    assert last.status == "cancelled"
