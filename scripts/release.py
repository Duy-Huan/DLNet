"""Release helpers (stdlib only). Single place that defines what a valid release looks like.

python scripts/release.py check          # version and CHANGELOG are consistent
python scripts/release.py version        # print the version from pyproject.toml
python scripts/release.py notes 0.1.0    # print the CHANGELOG section of that version
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
    return errors


def main(argv: list[str] | None = None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if a[:1] == ["check"]:
        errs = check()
        for e in errs:
            print(f"::error::{e}")
        print("release check: " + ("FAILED" if errs else f"ok ({pyproject_version()})"))
        return 1 if errs else 0
    if a[:1] == ["version"]:
        print(pyproject_version())
        return 0
    if a[:1] == ["notes"] and len(a) == 2:
        print(notes(a[1]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
