"""Run the CI gates locally BEFORE pushing, so GitHub Actions does not find the failure first.

    python scripts/ci_local.py            # default: what fails most often, about 3 minutes
    python scripts/ci_local.py --quick    # seconds: lint, format, release + docs checks, repo-hygiene tests
    python scripts/ci_local.py --full     # everything CI runs on Linux: all tests incl. e2e (10+ min)
    python scripts/ci_local.py --install  # git pre-push hook: every `git push` runs the gates first

Same commands as .github/workflows/ci.yml (jobs lint, regression, test, e2e). Not reproduced
locally: the macOS / Windows env-setup jobs and the coverage floor. The hook (scripts/hooks/pre-push)
runs --quick when the push only changes example data/results (examples/*/*/input|output|qc), the
default gates otherwise; `git push --no-verify` skips it and should be said out loud when used.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
PYTEST = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"]

QUICK = [
    ("ruff check", [PY, "-m", "ruff", "check", "."]),
    ("ruff format --check", [PY, "-m", "ruff", "format", "--check", "."]),
    ("release check (versions, CHANGELOG)", [PY, "scripts/release.py", "check"]),
    ("docs rule (README.md, doc/)", [PY, "scripts/release.py", "docs"]),
    ("repo hygiene + docs tests", [*PYTEST, "tests/test_env_setup.py", "tests/test_docs.py", "tests/test_release.py"]),
]  # fmt: skip
DEFAULT = [*QUICK, ("regression suite", [*PYTEST, "-m", "regression"])]
FULL = [
    *QUICK,
    ("unit + regression tests", [*PYTEST, "-m", "not e2e"]),
    ("e2e", [*PYTEST, "-m", "e2e"]),
]


def run(gates) -> int:
    failed = []
    for name, cmd in gates:
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        ok = r.returncode == 0
        print(f"[ci-local] {'ok  ' if ok else 'FAIL'} {name} ({time.time() - t0:.0f}s)", flush=True)
        if not ok:
            failed.append(name)
            print("\n".join((r.stdout + r.stderr).strip().splitlines()[-25:]), flush=True)
    print(f"[ci-local] {'ALL GATES PASSED' if not failed else 'FAILED: ' + ', '.join(failed)}")
    return 1 if failed else 0


def main(argv) -> int:
    if "--install" in argv:
        subprocess.run(["git", "config", "core.hooksPath", "scripts/hooks"], cwd=ROOT, check=True)
        print("pre-push hook installed (git config core.hooksPath scripts/hooks)")
        return 0
    return run(QUICK if "--quick" in argv else FULL if "--full" in argv else DEFAULT)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
