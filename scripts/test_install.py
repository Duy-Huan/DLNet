"""Install the OS package built by build_installer.py the way a user would, use it, uninstall it.

    python scripts/test_install.py                  # the package of this OS in dist/installers/

Checks, on a CI runner of each OS (installers.yml, after the build):
  1. SHA256SUMS-<os>-<arch>.txt matches every file it lists.
  2. The installer runs unattended: Windows `setup.exe /VERYSILENT /CURRENTUSER`, macOS
     `sudo installer -pkg`, Linux `sudo apt-get install ./x.deb`.
  3. `dlnet` is found on the PATH a new terminal would get (on Windows: the user's PATH in the
     registry, written by the installer), `dlnet --version` prints the version of the file name,
     `dlnet selfcheck` passes in an empty folder, `dlnet serve` answers /healthz.
  4. Uninstall: Windows `unins000.exe /VERYSILENT`, macOS the documented commands, Linux
     `apt-get remove`; afterwards the program folder and the `dlnet` command are gone and the
     installer's PATH entry is removed.
Exits non-zero with the failed step on stderr.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "installers"


def run(cmd, **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def find(pattern: str) -> Path:
    hits = sorted(OUT.glob(pattern))
    assert len(hits) == 1, f"expected one {pattern} in {OUT}, found {[h.name for h in hits]}"
    return hits[0]


def version_of(installer: Path) -> str:
    m = re.match(r"DLNet-(\d+\.\d+\.\d+[^-]*)-", installer.name)
    assert m, f"no version in {installer.name}"
    return m.group(1)


def check_sums() -> None:
    sums = find("SHA256SUMS-*.txt")
    lines = [ln.split() for ln in sums.read_text().splitlines() if ln.strip()]
    assert lines, f"{sums.name} is empty"
    for digest, name in lines:
        h = hashlib.sha256()
        with open(OUT / name, "rb") as fh:
            for chunk in iter(lambda fh=fh: fh.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == digest, f"checksum of {name} does not match {sums.name}"
    print(f"[install] {sums.name}: {len(lines)} files match")


def use(dlnet: Path, version: str) -> None:
    """What a user does first: --version, selfcheck, serve."""
    r = run([dlnet, "--version"], capture_output=True, text=True, timeout=300)
    assert version in r.stdout, f"`dlnet --version` printed {r.stdout!r}, expected {version}"
    with tempfile.TemporaryDirectory() as tmp:
        r = run([dlnet, "selfcheck"], capture_output=True, text=True, timeout=900, cwd=tmp)
        assert "selfcheck OK" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        srv = subprocess.Popen([str(dlnet), "serve", "--port", str(port), "--workspace", str(Path(tmp) / "ws"), "--quiet"])  # fmt: skip
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
                raise AssertionError("the installed `dlnet serve` did not answer /healthz")
            assert h["version"] == version, h
        finally:
            srv.terminate()
            try:
                srv.wait(20)
            except subprocess.TimeoutExpired:
                srv.kill()
    print(f"[install] dlnet {version}: --version, selfcheck, serve OK")


# ----------------------------------------------------------------------------- Windows


def _user_path() -> str:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
        try:
            return winreg.QueryValueEx(k, "Path")[0]
        except FileNotFoundError:
            return ""


def _in_path(folder: Path, path: str) -> bool:
    want = os.path.normcase(str(folder)).rstrip("\\")
    return any(os.path.normcase(p).rstrip("\\") == want for p in path.split(";") if p)


def windows() -> None:
    setup = find("*-windows-*-setup.exe")
    version = version_of(setup)
    app = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "DLNet"
    run([setup, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER", f"/LOG={OUT.parent / 'setup.log'}"],
        timeout=900)  # fmt: skip
    assert (app / "dlnet.exe").exists(), f"dlnet.exe not installed in {app}"
    path = _user_path()
    assert _in_path(app, path), f"the installer did not add {app} to the user PATH: {path!r}"
    dlnet = shutil.which("dlnet", path=path)
    assert dlnet and Path(dlnet).parent == app, f"`dlnet` on the new PATH is {dlnet}"
    use(Path(dlnet), version)

    run([app / "unins000.exe", "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], timeout=600)
    # the uninstaller copies itself to %TEMP% and returns at once: wait for it to finish
    for _ in range(300):
        if not (app / "dlnet.exe").exists() and not _in_path(app, _user_path()):
            break
        time.sleep(1)
    assert not (app / "dlnet.exe").exists(), f"uninstall left {app / 'dlnet.exe'}"
    assert not _in_path(app, _user_path()), (
        f"uninstall left {app} in the user PATH: {_user_path()!r}"
    )
    print("[install] Windows: uninstall removed the program and its PATH entry")


# ----------------------------------------------------------------------------- macOS


def macos() -> None:
    pkg = find("*-macos-*.pkg")
    version = version_of(pkg)
    run(["sudo", "installer", "-pkg", pkg, "-target", "/"], timeout=900)
    link = Path("/usr/local/bin/dlnet")
    assert link.exists() and Path("/Library/DLNet/dlnet").exists(), "dlnet not installed"
    dlnet = shutil.which("dlnet", path="/usr/local/bin:/usr/bin:/bin")
    assert dlnet == str(link), f"`dlnet` on PATH is {dlnet}"
    use(link, version)
    # the uninstall commands of doc/install.md
    run(["sudo", "rm", "-rf", "/Library/DLNet", "/usr/local/bin/dlnet"])
    run(["sudo", "pkgutil", "--forget", "io.github.dlnet"])
    assert not link.exists() and not Path("/Library/DLNet").exists()
    print("[install] macOS: uninstall removed the program and the command")


# ----------------------------------------------------------------------------- Linux


def linux() -> None:
    deb = find("*-linux-*.deb")
    version = version_of(deb)
    # an absolute path, so apt reads a file and not a package name
    run(["sudo", "apt-get", "install", "-y", deb.resolve()], timeout=900)
    dlnet = shutil.which("dlnet", path="/usr/local/bin:/usr/bin:/bin")
    assert dlnet == "/usr/bin/dlnet", f"`dlnet` on PATH is {dlnet}"
    r = run(["dpkg-query", "-W", "-f=${Version}", "dlnet"], capture_output=True, text=True)
    assert r.stdout == version, f"dpkg reports version {r.stdout!r}, file name says {version}"
    use(Path(dlnet), version)
    run(["sudo", "apt-get", "remove", "-y", "dlnet"], timeout=600)
    assert not Path("/usr/bin/dlnet").exists(), "apt-get remove left /usr/bin/dlnet"
    assert not Path("/opt/dlnet/dlnet").exists(), "apt-get remove left /opt/dlnet/dlnet"
    print("[install] Linux: apt-get remove removed the program and the command")


def main() -> int:
    check_sums()
    {"win32": windows, "darwin": macos}.get(sys.platform, linux)()
    print("[install] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
