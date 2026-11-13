"""What the DLNet service does behind its MCP tools: a workspace, background jobs, file access.

Used by both transports (`dlnet mcp`: stdio, dlnet/mcp_server.py; `dlnet serve`: HTTP,
dlnet/mcp_http.py). Nothing here speaks MCP.

Workspace (one folder, `--workspace`):
    data/    datasets and paper cards (uploaded, written by a tool, or copied in by the user)
    runs/    one folder per `design` run (leaderboard, packages, notebook)
    jobs/    one <id>.json (state) and <id>.log (stdout + stderr) per background job

Every path a tool receives is resolved inside the workspace (`Workspace.path`); with
`sandbox=False` (the stdio server on the user's own machine) absolute paths are accepted too.
Long work (design, train, predict, paper build) runs as a JOB: a subprocess of the same Python
(`python -m dlnet ...` or the package's own `train.py`), at most `max_jobs` at a time, the rest
queued. A tool waits a bounded time for its job and otherwise returns the job id, so a client
with a short tool timeout is never cut off in the middle of a run.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from . import jsonsafe

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
TEXT_LIMIT = 200_000  # characters returned by read_file / log tails at most
WRITE_LIMIT = 5_000_000  # bytes a tool call may write (bigger files: HTTP upload)
FINAL = ("succeeded", "failed", "cancelled", "lost")


class ServiceError(ValueError):
    """A problem the caller can fix (bad path, unknown job, ...): reported as a tool error."""


class Workspace:
    def __init__(self, root, sandbox: bool = True):
        self.root = Path(root).expanduser().resolve()
        self.sandbox = sandbox
        for d in ("data", "runs", "jobs"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def path(self, p, must_exist: bool = False) -> Path:
        """Resolve a path given by a client. Relative paths are relative to the workspace; a path
        that leaves the workspace (.., absolute, symlink) is refused in sandbox mode."""
        if not isinstance(p, str) or not p.strip() or "\x00" in p:
            raise ServiceError("a path must be a non-empty string")
        raw = Path(p).expanduser()
        full = (raw if raw.is_absolute() else self.root / raw).resolve()
        if self.sandbox and full != self.root and self.root not in full.parents:
            raise ServiceError(
                f"path {p!r} is outside the workspace; use a path relative to it (see dlnet_files)"
            )
        if must_exist and not full.exists():
            raise ServiceError(f"no such file or folder: {self.rel(full)} (see dlnet_files)")
        return full

    def rel(self, full: Path) -> str:
        full = Path(full)
        try:
            return full.resolve().relative_to(self.root).as_posix() or "."
        except ValueError:
            return str(full)

    def run_dir(self, name: str | None, prefix: str = "run") -> Path:
        """Folder of a NEW run under runs/ (a name is made up when none is given)."""
        if not name:
            name = f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
        if not NAME.match(name):
            raise ServiceError(
                "a run name is 1-100 characters of letters, digits, '.', '_' and '-'"
            )
        out = self.root / "runs" / name
        if out.exists() and any(out.iterdir()):
            raise ServiceError(f"run {name!r} already exists; choose another name")
        return out

    # ---------------------------------------------------------------- files

    def listing(self, folder: str = ".", depth: int = 2, limit: int = 300) -> dict:
        base = self.path(folder, must_exist=True)
        if base.is_file():
            st = base.stat()
            return {
                "folder": self.rel(base.parent),
                "entries": [self._entry(base, st)],
                "truncated": False,
            }
        entries, truncated = [], False
        for cur, dirs, files in os.walk(base):
            cur = Path(cur)
            level = len(cur.relative_to(base).parts)
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
            if level >= max(depth, 1):
                for d in dirs:
                    entries.append({"path": self.rel(cur / d) + "/", "type": "dir"})
                dirs[:] = []
            for f in sorted(files):
                if len(entries) >= limit:
                    truncated = True
                    break
                with contextlib.suppress(OSError):
                    entries.append(self._entry(cur / f, (cur / f).stat()))
            if truncated:
                break
        return {"folder": self.rel(base), "entries": entries, "truncated": truncated}

    def _entry(self, p: Path, st) -> dict:
        return {"path": self.rel(p), "type": "file", "bytes": st.st_size,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}  # fmt: skip

    def read_text(self, p: str, offset: int = 0, limit: int = 20_000, tail: bool = False) -> dict:
        full = self.path(p, must_exist=True)
        if not full.is_file():
            raise ServiceError(f"{self.rel(full)} is a folder; list it with dlnet_files")
        raw = full.read_bytes()
        if b"\x00" in raw[:4096]:
            raise ServiceError(
                f"{self.rel(full)} is a binary file ({len(raw):,} bytes); download it over HTTP (GET /files/<path>)"
            )
        text = raw.decode("utf-8", errors="replace")
        limit = max(1, min(int(limit), TEXT_LIMIT))
        start = max(len(text) - limit, 0) if tail else max(int(offset), 0)
        chunk = text[start : start + limit]
        return {"path": self.rel(full), "chars": len(text), "offset": start, "text": chunk,
                "more": start + len(chunk) < len(text)}  # fmt: skip

    def write(self, p: str, data: bytes, overwrite: bool = False) -> dict:
        full = self.path(p)
        if self.sandbox and self.root / "jobs" in [full, *full.parents]:
            raise ServiceError("jobs/ is written by the service only")
        if full.exists() and not overwrite:
            raise ServiceError(f"{self.rel(full)} exists; pass overwrite=true to replace it")
        if full.is_dir():
            raise ServiceError(f"{self.rel(full)} is a folder")
        full.parent.mkdir(parents=True, exist_ok=True)
        tmp = full.with_name(full.name + f".part-{secrets.token_hex(3)}")
        tmp.write_bytes(data)
        tmp.replace(full)
        return {"path": self.rel(full), "bytes": len(data)}

    def archive(self, p: str) -> dict:
        """Zip a folder of the workspace (a run, a package) next to it, for one download."""
        src = self.path(p, must_exist=True)
        if not src.is_dir():
            raise ServiceError(f"{self.rel(src)} is a file: download it directly")
        if src == self.root:
            raise ServiceError("refusing to zip the whole workspace; name a run or a package")
        out = src.with_name(src.name + ".zip")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(src.rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    z.write(f, f.relative_to(src.parent).as_posix())
        return {"path": self.rel(out), "bytes": out.stat().st_size}


# --------------------------------------------------------------------------- jobs


class Job:
    FIELDS = ("id", "kind", "title", "status", "argv", "cwd", "out", "created", "started", "finished",
              "returncode", "error", "extra")  # fmt: skip

    def __init__(self, **kw):
        self.id = kw.get("id") or f"job-{time.strftime('%m%d%H%M%S')}-{secrets.token_hex(3)}"
        self.kind, self.title = kw["kind"], kw.get("title", kw["kind"])
        self.status = kw.get("status", "queued")
        self.argv, self.cwd, self.out = list(kw.get("argv", [])), kw.get("cwd"), kw.get("out")
        self.created = kw.get("created", time.time())
        self.started, self.finished = kw.get("started"), kw.get("finished")
        self.returncode, self.error = kw.get("returncode"), kw.get("error")
        self.extra = kw.get("extra") or {}
        self.proc: subprocess.Popen | None = None
        self.done = threading.Event()
        self.cancel_requested = False
        if self.status in FINAL:
            self.done.set()

    def state(self) -> dict:
        return {k: getattr(self, k) for k in self.FIELDS}


class Jobs:
    """A queue of subprocess jobs with `max_jobs` workers; state survives a restart of the server
    (a job that was running when the server stopped is reported as `lost`)."""

    def __init__(self, ws: Workspace, max_jobs: int = 1, env: dict | None = None):
        self.ws, self.dir = ws, ws.root / "jobs"
        self.env = env
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue = queue.Queue()
        self.max_jobs = max(1, int(max_jobs))
        for f in sorted(self.dir.glob("*.json")):
            with contextlib.suppress(ValueError, KeyError, OSError):
                j = Job(**json.loads(f.read_text()))
                if j.status not in FINAL:
                    j.status, j.error = (
                        "lost",
                        "the server stopped while this job was queued or running",
                    )
                    j.finished = j.finished or time.time()
                    j.done.set()
                    self._save(j)
                self.jobs[j.id] = j
        self.workers = [threading.Thread(target=self._work, daemon=True, name=f"dlnet-job-{i}")
                        for i in range(self.max_jobs)]  # fmt: skip
        for w in self.workers:
            w.start()

    def log_path(self, j: Job) -> Path:
        return self.dir / f"{j.id}.log"

    def _save(self, j: Job) -> None:
        tmp = self.dir / f"{j.id}.json.tmp"
        tmp.write_text(jsonsafe.dumps(j.state(), indent=1))
        tmp.replace(self.dir / f"{j.id}.json")

    def submit(self, kind: str, argv: list[str], cwd: Path | None = None, out: Path | None = None,
               title: str | None = None, extra: dict | None = None) -> Job:  # fmt: skip
        j = Job(kind=kind, argv=[str(a) for a in argv], cwd=str(cwd or self.ws.root),
                out=self.ws.rel(out) if out else None, title=title or kind, extra=extra)  # fmt: skip
        with self.lock:
            self.jobs[j.id] = j
            self._save(j)
        self.queue.put(j.id)
        return j

    def _work(self) -> None:
        while True:
            j = self.jobs[self.queue.get()]
            with self.lock:
                if j.cancel_requested or j.status != "queued":
                    continue
                j.status, j.started = "running", time.time()
                self._save(j)
            code, err = None, None
            try:
                with open(self.log_path(j), "ab") as log:
                    env = {**os.environ, **(self.env or {}), "PYTHONUNBUFFERED": "1"}
                    kw = {"start_new_session": True} if os.name == "posix" else {
                        "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}  # fmt: skip
                    j.proc = subprocess.Popen(j.argv, cwd=j.cwd, stdout=log, stderr=subprocess.STDOUT,
                                              stdin=subprocess.DEVNULL, env=env, **kw)  # fmt: skip
                    if j.cancel_requested:  # cancelled between the status change and the start
                        self._kill(j)
                    code = j.proc.wait()
            except OSError as e:
                err = f"could not start the job: {e}"
            with self.lock:
                j.returncode, j.finished = code, time.time()
                j.status = (
                    "cancelled" if j.cancel_requested else "succeeded" if code == 0 else "failed"
                )
                j.error = err or (
                    None if code == 0 or j.cancel_requested else f"exit code {code}; see the log"
                )
                self._save(j)
                j.done.set()

    @staticmethod
    def _kill(j: Job) -> None:
        p = j.proc
        if p is None or p.poll() is not None:
            return
        with contextlib.suppress(OSError, ProcessLookupError):
            if os.name == "posix":
                os.killpg(p.pid, signal.SIGTERM)
            else:
                p.terminate()

    def get(self, job_id: str) -> Job:
        j = self.jobs.get(job_id)
        if j is None:
            raise ServiceError(f"unknown job {job_id!r} (see dlnet_jobs)")
        return j

    def cancel(self, job_id: str) -> Job:
        j = self.get(job_id)
        with self.lock:
            if j.status in FINAL:
                return j
            j.cancel_requested = True
            if j.status == "queued":
                j.status, j.finished = "cancelled", time.time()
                self._save(j)
                j.done.set()
                return j
        self._kill(j)
        if not j.done.wait(10) and j.proc is not None and j.proc.poll() is None:
            with contextlib.suppress(OSError, ProcessLookupError):
                if os.name == "posix":
                    os.killpg(j.proc.pid, signal.SIGKILL)
                else:
                    j.proc.kill()
            j.done.wait(5)
        return j

    def log_tail(self, j: Job, lines: int = 40) -> list[str]:
        p = self.log_path(j)
        if not p.exists():
            return []
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(f.tell() - 64_000, 0))
            text = f.read().decode("utf-8", errors="replace")
        return [line[:500] for line in text.splitlines()[-max(1, int(lines)) :]]

    def describe(self, j: Job, log_lines: int = 20) -> dict:
        now = time.time()
        d = {"job_id": j.id, "kind": j.kind, "title": j.title, "status": j.status, "out": j.out,
             "queued_seconds": round((j.started or j.finished or now) - j.created, 1),
             "run_seconds": round((j.finished or now) - j.started, 1) if j.started else None,
             "returncode": j.returncode, "error": j.error, "log": self.ws.rel(self.log_path(j))}  # fmt: skip
        if j.status == "queued":
            d["queue_position"] = sum(
                1 for o in self.jobs.values() if o.status == "queued" and o.created <= j.created
            )
        if log_lines:
            d["log_tail"] = self.log_tail(j, log_lines)
        return d

    def listing(self, limit: int = 30) -> list[dict]:
        js = sorted(self.jobs.values(), key=lambda j: -j.created)[: max(1, int(limit))]
        return [self.describe(j, 0) for j in js]

    def shutdown(self) -> None:
        for j in list(self.jobs.values()):
            if j.status not in FINAL:
                self.cancel(j.id)


def python() -> list[str]:
    """How to start DLNet again as a subprocess: `python -m dlnet`, or the frozen program itself."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "dlnet"]


def package_python() -> list[str]:
    """Interpreter for a package's own scripts (train.py, predict.py): the frozen program runs
    them through its `run-script` command, since it has no separate python."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "run-script"]
    return [sys.executable]
