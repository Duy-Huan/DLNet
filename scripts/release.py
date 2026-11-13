"""Release helpers (stdlib only). Single place that defines what a valid release looks like.

python scripts/release.py check          # version and CHANGELOG are consistent
python scripts/release.py version        # print the version from pyproject.toml
python scripts/release.py notes 0.1.0    # print the CHANGELOG section of that version
python scripts/release.py bump 0.2.0     # release commit edits: versions, CHANGELOG, doc/ (agent.md sec. 17)
python scripts/release.py docs [BASE]    # code changed since BASE (default: last tag) => README.md and doc/ too
python scripts/release.py assets DIR     # DIR holds every installer a release must ship (agent.md sec. 11)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"\d+\.\d+\.\d+")
HEADING = re.compile(
    r"^## \[(?P<name>[^\]]+)\](?: - (?P<date>\d{4}-\d{2}-\d{2}))?\s*$", re.MULTILINE
)


def _first(pattern: str, path: Path) -> str | None:
    m = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    return m.group(1) if m else None


def pyproject_version(root: Path = ROOT) -> str | None:
    return _first(r'^version\s*=\s*"([^"]+)"', root / "pyproject.toml")


def package_version(root: Path = ROOT) -> str | None:
    return _first(r'^__version__\s*=\s*"([^"]+)"', root / "dlnet" / "__init__.py")


def sections(text: str) -> list[dict]:
    """Split a changelog into [{'name', 'date', 'body'}] in file order."""
    marks = list(HEADING.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = re.split(r"^\[[^\]]+\]: ", text[m.end() : end], maxsplit=1, flags=re.MULTILINE)[0]
        out.append({"name": m["name"], "date": m["date"], "body": body.strip()})
    return out


def _key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def notes(version: str, root: Path = ROOT) -> str:
    for s in sections((root / "CHANGELOG.md").read_text(encoding="utf-8")):
        if s["name"] == version:
            return s["body"]
    raise SystemExit(f"CHANGELOG.md has no section for {version}")


def check(root: Path = ROOT) -> list[str]:
    errors = []
    v, pv = pyproject_version(root), package_version(root)
    if not v or not SEMVER.fullmatch(v):
        errors.append(f"pyproject.toml version {v!r} is not X.Y.Z")
    if v != pv:
        errors.append(f"version mismatch: pyproject.toml={v!r} dlnet/__init__.py={pv!r}")
    cl = root / "CHANGELOG.md"
    if not cl.exists():
        return [*errors, "CHANGELOG.md is missing"]
    secs = sections(cl.read_text(encoding="utf-8"))
    if not secs or secs[0]["name"] != "Unreleased":
        errors.append("CHANGELOG.md must start with a '## [Unreleased]' section")
    released = [s for s in secs if s["name"] != "Unreleased"]
    names = [s["name"] for s in released]
    bad = [n for n in names if not SEMVER.fullmatch(n)]
    if bad:
        errors.append(f"CHANGELOG.md has non X.Y.Z sections: {bad}")
        return errors
    if len(set(names)) != len(names):
        errors.append("CHANGELOG.md has duplicate version sections")
    if names != sorted(names, key=_key, reverse=True):
        errors.append("CHANGELOG.md versions must be newest first")
    for s in released:
        if not s["date"]:
            errors.append(f"CHANGELOG.md [{s['name']}] needs a date (YYYY-MM-DD)")
        if not s["body"]:
            errors.append(f"CHANGELOG.md [{s['name']}] is empty")
    if v and (not names or names[0] != v):
        errors.append(f"newest CHANGELOG.md section is {names[:1]}, but version is {v!r}")
    doc = root / "doc" / "README.md"
    if doc.exists():  # agent.md sec. 17: the docs state the version they describe
        dv = _first(DOC_VERSION, doc)
        if dv != v:
            errors.append(f"doc/README.md describes version {dv!r}, but version is {v!r}")
    return errors


DOC_VERSION = r"Phiên bản tài liệu: \*\*([^*]+)\*\*"


HAND_DOCS = (
    "doc/user-guide.md",
    "doc/architecture.md",
    "doc/algorithms.md",
    "doc/mcp-server.md",
    "doc/install.md",
)


def _git(*args: str, root: Path = ROOT) -> str:
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def docs_errors(changed: list[str]) -> list[str]:
    """agent.md sec. 17 rule 0: a change to the package comes with README.md and a hand-written
    doc/ page (doc/cli.md, doc/zoo.md are generated and do not count)."""
    if not any(f.startswith("dlnet/") for f in changed):
        return []
    errors = []
    if "README.md" not in changed:
        errors.append("dlnet/ changed but README.md was not updated (agent.md sec. 17)")
    if not any(f in changed for f in HAND_DOCS):
        errors.append(
            f"dlnet/ changed but none of {', '.join(HAND_DOCS)} was updated (agent.md sec. 17)"
        )
    return errors


def docs(base: str | None = None, root: Path = ROOT) -> tuple[list[str], bool]:
    """(errors, strict). Strict on a release commit (version not yet tagged), a warning otherwise,
    so work in progress on dev stays green while the release cannot skip the docs."""
    tags = _git("tag", "--list", "v*", "--sort=-v:refname", root=root).split()
    tag = base or (tags[0] if tags else None)  # tags live on main, not on dev: newest by version
    if tag is None:
        return [], False
    changed = _git("diff", "--name-only", f"{tag}...HEAD", root=root).split()
    strict = f"v{pyproject_version(root)}" != tag
    return docs_errors(changed), strict


def bump(new: str, today: str, root: Path = ROOT) -> None:
    """Release commit edits: version in pyproject / __init__ / doc/README.md, CHANGELOG section and
    links, regenerated doc/cli.md and doc/zoo.md."""
    old = pyproject_version(root)
    pp = root / "pyproject.toml"
    pp.write_text(pp.read_text().replace(f'version = "{old}"', f'version = "{new}"', 1))
    ip = root / "dlnet" / "__init__.py"
    ip.write_text(ip.read_text().replace(old, new))
    cp = root / "CHANGELOG.md"
    c = cp.read_text()
    c = c.replace("## [Unreleased]\n", f"## [Unreleased]\n\n## [{new}] - {today}\n", 1)
    c = c.replace(
        f"compare/v{old}...HEAD",
        f"compare/v{new}...HEAD\n[{new}]: https://github.com/Duy-Huan/DLNet/releases/tag/v{new}",
    )
    cp.write_text(c)
    doc = root / "doc" / "README.md"
    if doc.exists():
        doc.write_text(re.sub(DOC_VERSION, f"Phiên bản tài liệu: **{new}**", doc.read_text()))
        import subprocess

        subprocess.run([sys.executable, str(root / "scripts" / "gen_docs.py")], check=True)


# Every release ships the installable program for these platforms (agent.md sec. 11, doc/install.md).
# scripts/build_installer.py writes exactly these names; the release workflow refuses to tag a
# version when one is missing.
PLATFORMS = (("windows", "x64"), ("macos", "arm64"), ("linux", "x64"), ("linux", "arm64"))
EXT = {"windows": ("setup.exe", "zip"), "macos": ("pkg", "tar.gz"), "linux": ("deb", "tar.gz")}


def asset_names(version: str) -> dict[tuple[str, str], dict[str, str]]:
    out = {}
    for osn, arch in PLATFORMS:
        inst, port = EXT[osn]
        stem = f"DLNet-{version}-{osn}-{arch}"
        out[(osn, arch)] = {"installer": f"{stem}{'-' if inst == 'setup.exe' else '.'}{inst}",
                            "portable": f"{stem}.{port}"}  # fmt: skip
    return out


def missing_assets(folder: Path, version: str) -> list[str]:
    want = [n for d in asset_names(version).values() for n in d.values()]
    want += [f"SHA256SUMS-{o}-{a}.txt" for o, a in PLATFORMS]
    have = {f.name for f in Path(folder).rglob("*") if f.is_file()}
    return [n for n in want if n not in have]


def main(argv: list[str] | None = None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if a[:1] == ["check"]:
        errs = check()
        for e in errs:
            print(f"::error::{e}")
        print("release check: " + ("FAILED" if errs else f"ok ({pyproject_version()})"))
        return 1 if errs else 0
    if a[:1] == ["docs"] and len(a) <= 2:
        errs, strict = docs(a[1] if len(a) == 2 else None)
        for e in errs:
            print(f"::{'error' if strict else 'warning'}::{e}")
        print(
            "docs check: "
            + ("ok" if not errs else "FAILED" if strict else "warning (not a release commit)")
        )
        return 1 if errs and strict else 0
    if a[:1] == ["version"]:
        print(pyproject_version())
        return 0
    if a[:1] == ["bump"] and len(a) == 2:
        import datetime

        bump(a[1], datetime.datetime.now(datetime.UTC).date().isoformat())
        return 0
    if a[:1] == ["assets"] and len(a) == 2:
        missing = missing_assets(Path(a[1]), pyproject_version())
        for m in missing:
            print(f"::error::installer missing from the release: {m}")
        print("assets check: " + ("FAILED" if missing else f"ok ({len(PLATFORMS)} platforms)"))
        return 1 if missing else 0
    if a[:1] == ["notes"] and len(a) == 2:
        print(notes(a[1]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
