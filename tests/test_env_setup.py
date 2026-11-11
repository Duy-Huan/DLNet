"""Environment contract: the repo ships a reproducible setup, never a committed virtualenv.

A venv cannot be committed: it is gigabytes (libtorch_cuda alone is >500 MB, GitHub rejects files
over 100 MB), holds absolute paths and OS-specific binaries. See skills/setup-env.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parent.parent
MAX_TRACKED_MB = 50  # GitHub hard limit is 100 MB per file
# examples/<domain>/<case>/input/ holds the publishers' dataset files unchanged (examples/prepare_data.py,
# sha256 in SOURCE.md); Botswana.mat is 75 MB and losslessly compressed still 73 MB. Allowed up to
# 95 MB there, below GitHub's 100 MB hard limit; everything else stays under 50 MB.
MAX_DATASET_MB = 95


def tracked():
    try:
        out = subprocess.run(
            ["git", "ls-files", "-s", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    rows = [r.split("\t", 1) for r in out.split("\0") if r]
    return [(meta.split()[0], path) for meta, path in rows]


def test_no_virtualenv_or_oversized_file_is_tracked():
    bad = [
        p for _, p in tracked() if re.search(r"(^|/)(\.venv|venv|site-packages)/|pyvenv\.cfg$", p)
    ]
    assert not bad, f"virtualenv files are tracked: {bad[:5]}"

    def limit(p):
        return MAX_DATASET_MB if re.match(r"examples/[^/]+/[^/]+/input/", p) else MAX_TRACKED_MB

    big = [
        p
        for _, p in tracked()
        if (ROOT / p).is_file() and (ROOT / p).stat().st_size > limit(p) * 2**20
    ]
    assert not big, (
        f"files over the size limit ({MAX_TRACKED_MB} MB, {MAX_DATASET_MB} MB for example datasets): {big}"
    )


def test_gitignore_blocks_the_venv():
    assert ".venv/" in (ROOT / ".gitignore").read_text().split()


def test_setup_scripts_exist_and_unix_script_is_executable():
    assert (ROOT / "scripts/setup_env.ps1").exists()
    modes = {p: m for m, p in tracked()}
    if "scripts/setup_env.sh" in modes:  # tracked: git must keep the executable bit
        assert modes["scripts/setup_env.sh"] == "100755"
    else:  # not added yet (local work in progress): the file mode must already be right
        assert os.access(ROOT / "scripts/setup_env.sh", os.X_OK)


def test_setup_scripts_install_the_extras_ci_uses():
    for name in ("setup_env.sh", "setup_env.ps1"):
        text = (ROOT / "scripts" / name).read_text()
        assert ".[dev,design]" in text and "selfcheck" in text, name


def _pins(name):
    out = {}
    for line in (ROOT / name).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith(("-", "http")):
            m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([\w.+]+)", line)
            assert m, f"{name}: '{line}' must be pinned as name==version"
            out[m[1].lower().replace("_", "-")] = m[2]
    return out


def test_requirements_are_pinned_and_cover_the_design_extra():
    extras = re.search(
        r"^design = \[(.*?)\]", (ROOT / "pyproject.toml").read_text(), re.DOTALL | re.MULTILINE
    )
    needed = set(re.findall(r'"([A-Za-z0-9_.\-]+)"', extras.group(1)))
    assert needed and needed <= set(_pins("requirements.txt")), needed - set(
        _pins("requirements.txt")
    )
    dev = _pins("requirements-dev.txt")
    assert {"pytest", "ruff", "pyyaml"} <= set(dev)
    assert "-r requirements.txt" in (ROOT / "requirements-dev.txt").read_text()


def test_python_version_file_is_a_supported_minor_release():
    v = (ROOT / ".python-version").read_text().strip()
    assert re.fullmatch(r"3\.(10|11|12|13)", v), v


@pytest.mark.e2e
def test_cli_selfcheck_and_version():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    from dlnet import __version__, cli

    assert cli.main(["selfcheck"]) == 0
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and __version__
