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
    big = [
        p
        for _, p in tracked()
        if (ROOT / p).is_file() and (ROOT / p).stat().st_size > MAX_TRACKED_MB * 2**20
    ]
    assert not big, f"files over {MAX_TRACKED_MB} MB are tracked: {big}"


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


def test_requirements_files_point_at_existing_extras():
    extras = re.search(
        r"\[project\.optional-dependencies\](.*?)\n\[",
        (ROOT / "pyproject.toml").read_text(),
        re.DOTALL,
    )
    names = set(re.findall(r"^(\w+)\s*=", extras.group(1), re.MULTILINE))
    for f in ("requirements.txt", "requirements-dev.txt"):
        used = set(re.findall(r"\[([\w,]+)\]", (ROOT / f).read_text()))
        wanted = {x for u in used for x in u.split(",")}
        assert wanted and wanted <= names, (f, wanted, names)


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
