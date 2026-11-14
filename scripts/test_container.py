"""Build the DLNet container image and test it as a user would (Podman, else Docker).

    python scripts/test_container.py                 # build + test
    python scripts/test_container.py --no-build      # test an image built before
    python scripts/test_container.py --engine docker

Environment: DLNET_CONTAINER_ENGINE_ARGS (options before every engine command, e.g.
--storage-driver=vfs), DLNET_CONTAINER_BUILD_ARGS (options of the build only, e.g. --network host).

Checks, each in a fresh container with a host folder mounted on /workspace:
  1. `dlnet --version` is the version of this checkout;
  2. `dlnet selfcheck` (a whole design on built-in data);
  3. `design` on a CSV of the host folder, then the package's train.py and predict.py through
     `run-script`: the files appear in the host folder, owned by the user who ran the test;
  4. `serve` published on 127.0.0.1: /healthz from the host, refuses to start on 0.0.0.0 without
     a token;
  5. `ui` published on 127.0.0.1: the page and its static files from the host, and a request that
     names another host is refused (the UI's DNS-rebinding guard still holds in a container).
Exit code 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlnet import __version__

IMAGE = "localhost/dlnet:test"


def engine(name: str | None) -> list[str]:
    for e in [name] if name else ["podman", "docker"]:
        if e and shutil.which(e):
            extra = os.environ.get("DLNET_CONTAINER_ENGINE_ARGS", "").split()
            return [e, *extra]
    raise SystemExit("neither podman nor docker is installed")


def run(cmd, timeout=1800, **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(
        list(map(str, cmd)), capture_output=True, text=True, timeout=timeout, check=False, **kw
    )
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise AssertionError(f"exit {r.returncode}: {' '.join(map(str, cmd))}")
    return r


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url, headers=None, tries=240):
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            return urllib.request.urlopen(req, timeout=3).read()
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise AssertionError(f"no answer from {url}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["podman", "docker"])
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--image", default=IMAGE)
    a = ap.parse_args()
    eng = engine(a.engine)
    docker = Path(eng[0]).name == "docker"
    t0 = time.time()
    if not a.no_build:
        # DLNET_CONTAINER_BUILD_ARGS: extra build options of this machine (a proxy, its CA, --network host)
        extra = os.environ.get("DLNET_CONTAINER_BUILD_ARGS", "").split()
        run(
            [*eng, "build", *extra, "-t", a.image, "-f", ROOT / "Containerfile", ROOT], timeout=3600
        )
        print(f"[container] built {a.image} in {time.time() - t0:.0f}s")
    size = run([*eng, "image", "inspect", a.image, "--format", "{{.Size}}"]).stdout.strip()
    print(f"[container] image size {int(size) / 1e9:.2f} GB")

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        os.chmod(ws, 0o777)  # the container's user may be another uid (Docker with --user)
        user = (
            ["--user", f"{os.getuid()}:{os.getgid()}"] if docker and hasattr(os, "getuid") else []
        )
        base = [*eng, "run", "--rm", *user, "-v", f"{ws}:/workspace"]

        out = run([*base, a.image, "--version"])
        assert __version__ in out.stdout + out.stderr, (out.stdout, out.stderr)
        print(f"[container] --version {__version__} OK")

        t = time.time()
        r = run([*base, a.image, "selfcheck"], timeout=1800)
        assert "selfcheck OK" in r.stdout, r.stdout[-2000:]
        print(f"[container] selfcheck OK in {time.time() - t:.0f}s")

        import random

        rng = random.Random(0)
        rows = ["a,b,c,d,y"]
        for _ in range(200):
            x = [rng.gauss(0, 1) for _ in range(4)]
            rows.append(",".join(f"{v:.4f}" for v in x) + f",{int(x[0] > 0)}")
        (ws / "d.csv").write_text("\n".join(rows) + "\n")
        run([*base, a.image, "design", "--data", "d.csv", "--target", "y", "--budget", "20ep", "--topk", "1",
             "--zoo", "2", "--evolve", "0", "--per-family", "1", "--out", "out"], timeout=1800)  # fmt: skip
        pkg = next((ws / "out").glob("top1_*"))
        rel = pkg.relative_to(ws)
        run([*base, a.image, "run-script", f"{rel}/train.py", "--data", "d.csv", "--target", "y",
             "--epochs", "3"], timeout=1800)  # fmt: skip
        run(
            [
                *base,
                a.image,
                "run-script",
                f"{rel}/predict.py",
                "--data",
                "d.csv",
                "--out",
                "pred.csv",
            ]
        )
        pred = ws / "pred.csv"
        assert pred.exists() and len(pred.read_text().splitlines()) >= 200, "no predictions"
        if hasattr(os, "getuid"):
            assert pred.stat().st_uid == os.getuid(), (
                f"pred.csv is owned by uid {pred.stat().st_uid}, not by the user ({os.getuid()})"
            )
        print(
            "[container] design -> train.py -> predict.py OK, files in the host folder, owned by the user"
        )

        # serve: refuses 0.0.0.0 without a token; with one, answers from the host
        r = subprocess.run([*base, a.image, "serve", "--host", "0.0.0.0", "--workspace", "/workspace"],
                           capture_output=True, text=True, timeout=300, check=False)  # fmt: skip
        assert r.returncode != 0 and "without authentication" in r.stdout + r.stderr, (
            r.stdout,
            r.stderr,
        )
        port = free_port()
        name = f"dlnet-serve-{port}"
        run([*eng, "run", "-d", "--rm", "--name", name, *user, "-v", f"{ws}:/workspace", "-e", "DLNET_TOKEN=t0k",
             "-p", f"127.0.0.1:{port}:8765", a.image, "serve", "--host", "0.0.0.0", "--workspace", "/workspace"])  # fmt: skip
        try:
            h = json.loads(wait_http(f"http://127.0.0.1:{port}/healthz"))
            assert h["version"] == __version__, h
            print("[container] serve: /healthz from the host OK; no token on 0.0.0.0 refused")
        finally:
            subprocess.run([*eng, "rm", "-f", name], capture_output=True, check=False)

        # ui: same port inside and out (the page checks the Host header)
        port = free_port()
        name = f"dlnet-ui-{port}"
        run([*eng, "run", "-d", "--rm", "--name", name, *user, "-v", f"{ws}:/workspace",
             "-p", f"127.0.0.1:{port}:{port}", a.image, "ui", "--port", port, "--no-browser",
             "--workspace", "/workspace"])  # fmt: skip
        try:
            page = wait_http(f"http://127.0.0.1:{port}/").decode()
            assert 'name="dlnet-token"' in page and __version__ in page, "no UI page"
            for f in ("app.js", "app.css", "i18n/en.json", "i18n/vi.json"):
                body = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/static/{f}", timeout=10
                ).read()
                assert len(body) > 100, f"static/{f} missing"
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/state", headers={"Host": f"evil.example:{port}"}
            )
            try:
                urllib.request.urlopen(req, timeout=10)
                raise AssertionError("the UI answered a request naming another host")
            except urllib.error.HTTPError as e:
                assert e.code == 403, e.code
            print("[container] ui: page and static files from the host OK; other Host refused")
        finally:
            subprocess.run([*eng, "rm", "-f", name], capture_output=True, check=False)
    print(f"[container] ALL OK ({Path(eng[0]).name}, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
