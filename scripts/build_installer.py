"""Build the installable DLNet program for THIS operating system, test it, package it.

    python scripts/build_installer.py            # build + smoke test + installer into dist/installers/
    python scripts/build_installer.py --skip-build   # reuse dist/dlnet (re-test and re-package)

Steps
  1. PyInstaller (packaging/dlnet.spec): one folder dist/dlnet/ with the dlnet program, Python,
     torch (CPU; MPS on Apple silicon) and every dependency of `design`, `serve`, `paper`.
  2. Smoke test of the FROZEN program, not of the source: `--version` must print this version,
     `selfcheck` runs a whole design, `run-script` runs a package's predict.py, and `serve` must
     answer /healthz and an MCP initialize. A failure stops the build: no installer is written.
  3. Installer for the OS, all named DLNet-<version>-<os>-<arch>.<ext> (scripts/release.py
     `assets` lists what a release must hold):
       Windows  -setup.exe (Inno Setup: Program Files or per-user, PATH, Start-menu entries) and .zip
       macOS    .pkg (pkgbuild: /Library/DLNet, /usr/local/bin/dlnet) and .tar.gz
       Linux    .deb (/opt/dlnet, /usr/bin/dlnet) and .tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from release import asset_names, pyproject_version

DIST = ROOT / "dist"
APP = DIST / "dlnet"
OUT = DIST / "installers"


def os_arch() -> tuple[str, str]:
    osn = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    m = platform.machine().lower()
    arch = "arm64" if m in ("arm64", "aarch64") else "x64"
    return osn, arch


def exe() -> Path:
    return APP / ("dlnet.exe" if sys.platform == "win32" else "dlnet")


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def build() -> None:
    shutil.rmtree(APP, ignore_errors=True)
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", DIST,
         "--workpath", ROOT / "build" / "pyinstaller", ROOT / "packaging" / "dlnet.spec"])  # fmt: skip


# ----------------------------------------------------------------------------- smoke test


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def smoke(version: str) -> None:
    out = subprocess.run(
        [exe(), "--version"], capture_output=True, text=True, timeout=300, check=False
    )
    assert out.returncode == 0 and version in out.stdout + out.stderr, (out.stdout, out.stderr)
    print(f"[smoke] --version: {(out.stdout + out.stderr).strip()}")
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        r = subprocess.run(
            [exe(), "selfcheck"], capture_output=True, text=True, timeout=900, cwd=tmp, check=False
        )
        print(r.stdout[-1500:], r.stderr[-1500:])
        assert r.returncode == 0 and "selfcheck OK" in r.stdout, "selfcheck failed"
        print(f"[smoke] selfcheck OK in {time.time() - t0:.0f}s")
        # a design whose package is then used through run-script, as the MCP tools do
        import numpy as np

        rng = np.random.default_rng(0)
        csv = Path(tmp) / "d.csv"
        X = rng.normal(size=(200, 4))
        rows = ["a,b,c,d,y", *(",".join(f"{v:.4f}" for v in x) + f",{int(x[0] > 0)}" for x in X)]
        csv.write_text("\n".join(rows) + "\n")
        run([exe(), "design", "--data", csv, "--target", "y", "--budget", "20ep", "--topk", "1", "--zoo", "2",
             "--evolve", "0", "--per-family", "1", "--out", Path(tmp) / "out"], cwd=tmp, timeout=900)  # fmt: skip
        pkg = next((Path(tmp) / "out").glob("top1_*"))
        run(
            [
                exe(),
                "run-script",
                pkg / "train.py",
                "--data",
                csv,
                "--target",
                "y",
                "--epochs",
                "3",
            ],
            cwd=tmp,
            timeout=900,
        )
        pred = Path(tmp) / "pred.csv"
        run(
            [exe(), "run-script", pkg / "predict.py", "--data", csv, "--out", pred],
            cwd=tmp,
            timeout=600,
        )
        assert pred.exists() and len(pred.read_text().splitlines()) >= 200, (
            "predict.py wrote no predictions"
        )
        src = (pkg / "model.py").read_text()
        assert "import torch" in src, "model.py is not the standalone source"
        print("[smoke] design -> run-script train.py -> predict.py OK")
        # the MCP server
        port = _free_port()
        env = {**os.environ, "DLNET_TOKEN": "smoke-token"}
        srv = subprocess.Popen([exe(), "serve", "--port", str(port), "--workspace", str(Path(tmp) / "ws"), "--quiet"],
                               env=env)  # fmt: skip
        try:
            for _ in range(240):
                try:
                    h = json.loads(
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2).read()
                    )
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                raise AssertionError("dlnet serve did not answer /healthz")
            assert h["version"] == version, h
            body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "smoke", "version": "0"}}}).encode()  # fmt: skip
            req = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", data=body, headers={
                "Authorization": "Bearer smoke-token", "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"})  # fmt: skip
            with urllib.request.urlopen(req, timeout=30) as r:
                init = json.loads(r.read())
                sid = r.headers["Mcp-Session-Id"]
            assert init["result"]["serverInfo"]["version"] == version and sid
            print("[smoke] serve: /healthz and MCP initialize OK")
        finally:
            srv.terminate()
            try:
                srv.wait(20)
            except subprocess.TimeoutExpired:
                srv.kill()


# ----------------------------------------------------------------------------- installers


def _tar(dest: Path) -> None:
    with tarfile.open(dest, "w:gz") as t:
        t.add(APP, arcname=f"dlnet-{dest.name.split('-')[1]}")


def _zip(dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(APP.rglob("*")):
            if f.is_file():
                z.write(f, Path("DLNet") / f.relative_to(APP))


def windows(version: str, names: dict) -> None:
    iscc = shutil.which("iscc") or r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    run([iscc, f"/DAppVersion={version}", f"/DSourceDir={APP}", f"/DOutputDir={OUT}",
         ROOT / "packaging" / "windows" / "dlnet.iss"])  # fmt: skip
    _zip(OUT / names["portable"])


def macos(version: str, names: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, scripts = Path(tmp) / "root" / "Library" / "DLNet", Path(tmp) / "scripts"
        shutil.copytree(APP, root, symlinks=True)
        scripts.mkdir()
        shutil.copy2(ROOT / "packaging" / "macos" / "postinstall", scripts / "postinstall")
        run(["pkgbuild", "--root", Path(tmp) / "root", "--identifier", "io.github.dlnet", "--version", version,
             "--install-location", "/", "--scripts", scripts, OUT / names["installer"]])  # fmt: skip
    _tar(OUT / names["portable"])


def linux(version: str, names: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg"
        shutil.copytree(APP, pkg / "opt" / "dlnet", symlinks=True)
        deb = pkg / "DEBIAN"
        deb.mkdir()
        size = sum(f.stat().st_size for f in (pkg / "opt").rglob("*") if f.is_file()) // 1024
        control = (
            (ROOT / "packaging" / "linux" / "control").read_text().replace("{version}", version)
        )
        (deb / "control").write_text(control + f"Installed-Size: {size}\n")
        for s in ("postinst", "prerm"):
            shutil.copy2(ROOT / "packaging" / "linux" / s, deb / s)
            (deb / s).chmod(0o755)
        run(["dpkg-deb", "--root-owner-group", "--build", pkg, OUT / names["installer"]])
    _tar(OUT / names["portable"])


def checksums() -> None:
    lines = []
    for f in sorted(OUT.iterdir()):
        if f.is_file() and f.suffix != ".txt":
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda fh=fh: fh.read(1 << 20), b""):
                    h.update(chunk)
            lines.append(f"{h.hexdigest()}  {f.name}")
    osn, arch = os_arch()
    (OUT / f"SHA256SUMS-{osn}-{arch}.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument(
        "--skip-smoke", action="store_true", help="only for debugging the packaging step"
    )
    a = ap.parse_args()
    version = pyproject_version()
    osn, arch = os_arch()
    names = asset_names(version)[(osn, arch)]
    if not a.skip_build:
        build()
    if not exe().exists():
        raise SystemExit(f"no program at {exe()}: build first")
    if not a.skip_smoke:
        smoke(version)
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    {"windows": windows, "macos": macos, "linux": linux}[osn](version, names)
    missing = [n for n in names.values() if not (OUT / n).exists()]
    if missing:
        raise SystemExit(f"installer files missing: {missing}")
    checksums()
    for f in sorted(OUT.iterdir()):
        print(f"{f.stat().st_size / 1e6:9.1f} MB  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
