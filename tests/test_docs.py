"""doc/ stays in sync with the code (agent.md sec. 17)."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_generated_docs_are_up_to_date():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_docs.py"), "--check"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr


def test_doc_links_point_to_existing_files():
    for md in (ROOT / "doc").glob("*.md"):
        for target in re.findall(r"\]\(([^)#:]+)(?:#[^)]*)?\)", md.read_text()):
            assert (md.parent / target).exists(), f"{md.name}: broken link {target}"


def test_every_command_and_objective_is_documented():
    guide = (ROOT / "doc" / "user-guide.md").read_text()
    from dlnet.objectives import OBJECTIVES

    for o in OBJECTIVES:
        assert f"--objective {o}" in guide, o
    for cmd in ("design", "zoo", "unfold", "gan", "selfcheck", "standalone"):
        assert f"dlnet {cmd}" in guide or f"dlnet {cmd}" in (ROOT / "doc" / "cli.md").read_text()


def test_release_bump_updates_doc_version(tmp_path):
    import shutil

    sys.path.insert(0, str(ROOT / "scripts"))
    import release

    for f in ("pyproject.toml", "CHANGELOG.md"):
        shutil.copy(ROOT / f, tmp_path / f)
    (tmp_path / "dlnet").mkdir()
    shutil.copy(ROOT / "dlnet" / "__init__.py", tmp_path / "dlnet" / "__init__.py")
    (tmp_path / "doc").mkdir()
    shutil.copy(ROOT / "doc" / "README.md", tmp_path / "doc" / "README.md")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "gen_docs.py").write_text("")  # generation is tested above
    cl = tmp_path / "CHANGELOG.md"
    cl.write_text(
        cl.read_text().replace("## [Unreleased]\n", "## [Unreleased]\n\n- something\n", 1)
    )
    release.bump("9.9.9", "2030-01-01", tmp_path)
    assert "Phiên bản tài liệu: **9.9.9**" in (tmp_path / "doc" / "README.md").read_text()
    assert release.check(tmp_path) == []


def test_readme_module_count_matches_the_zoo():
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    from dlnet import zoo

    m = re.search(r"\((\d+) module", (ROOT / "README.md").read_text())
    assert m and int(m.group(1)) == len(zoo.OPS), (
        "README.md module count is stale (agent.md sec. 17)"
    )


def test_readme_links_point_to_existing_files():
    for target in re.findall(r"\]\(([^)#:]+)(?:#[^)]*)?\)", (ROOT / "README.md").read_text()):
        assert (ROOT / target).exists(), f"README.md: broken link {target}"


def test_docs_rule_requires_readme_and_a_hand_written_page():
    sys.path.insert(0, str(ROOT / "scripts"))
    import release

    assert release.docs_errors(["tests/x.py"]) == []
    assert len(release.docs_errors(["dlnet/core.py"])) == 2
    assert (
        len(release.docs_errors(["dlnet/core.py", "README.md", "doc/zoo.md"])) == 1
    )  # generated page
    assert release.docs_errors(["dlnet/core.py", "README.md", "doc/user-guide.md"]) == []
