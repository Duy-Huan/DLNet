"""Release contract: version, CHANGELOG and notes must agree, or `release.yml` would fail or lie.

Pure Python (no torch), so it runs on every CI matrix entry.
"""

from __future__ import annotations

import importlib.util
import re
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("release_script", ROOT / "scripts" / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def make_repo(tmp_path, pyproject="0.2.0", package="0.2.0", changelog=None):
    (tmp_path / "dlnet").mkdir()
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{pyproject}"\n')
    (tmp_path / "dlnet" / "__init__.py").write_text(f'__version__ = "{package}"\n')
    if changelog is None:
        changelog = textwrap.dedent(
            """\
            # Changelog

            ## [Unreleased]

            ## [0.2.0] - 2026-02-01

            ### Thêm
            - b

            ## [0.1.0] - 2026-01-01

            - a

            [Unreleased]: https://example.com/compare
            """
        )
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    return tmp_path


def test_the_real_repo_is_release_ready():
    assert release.check() == []
    assert release.pyproject_version() == release.package_version()


def test_real_changelog_notes_are_non_empty_and_have_no_link_footer():
    body = release.notes(release.pyproject_version())
    assert body and not re.search(r"^\[[^\]]+\]: ", body, re.MULTILINE)


def test_valid_repo_passes_and_notes_are_extracted_per_version(tmp_path):
    r = make_repo(tmp_path)
    assert release.check(r) == []
    assert release.notes("0.2.0", r) == "### Thêm\n- b"
    assert release.notes("0.1.0", r) == "- a"


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"package": "0.1.9"}, "mismatch"),
        ({"pyproject": "0.2", "package": "0.2"}, "not X.Y.Z"),
        ({"pyproject": "0.3.0", "package": "0.3.0"}, "newest CHANGELOG"),  # bumped, no notes
        ({"changelog": "# C\n\n## [0.2.0] - 2026-02-01\n- b\n"}, "Unreleased"),
        ({"changelog": "# C\n\n## [Unreleased]\n\n## [0.2.0]\n- b\n"}, "needs a date"),
        ({"changelog": "# C\n\n## [Unreleased]\n\n## [0.2.0] - 2026-02-01\n\n"}, "is empty"),
        (
            {
                "changelog": "# C\n\n## [Unreleased]\n\n## [0.1.0] - 2026-01-01\n- a\n\n"
                "## [0.2.0] - 2026-02-01\n- b\n"
            },
            "newest first",
        ),
    ],
)
def test_check_rejects_inconsistent_releases(tmp_path, kwargs, needle):
    errors = release.check(make_repo(tmp_path, **kwargs))
    assert any(needle in e for e in errors), errors


def test_notes_for_unknown_version_fails_loudly(tmp_path):
    with pytest.raises(SystemExit):
        release.notes("9.9.9", make_repo(tmp_path))


def test_workflows_wire_release_after_promote():
    """promote -> release is the only automatic path (GITHUB_TOKEN merges trigger no other workflow)."""
    import yaml

    promote = yaml.safe_load((ROOT / ".github/workflows/promote.yml").read_text())
    rel = promote["jobs"]["release"]
    assert rel["needs"] == "promote" and rel["uses"].endswith("release.yml")
    assert rel["permissions"]["contents"] == "write"
    wf = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    triggers = wf.get("on", wf.get(True))
    assert {"workflow_call", "workflow_dispatch"} <= set(triggers)
