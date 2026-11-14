"""v1.8.2: the Containerfile stays in step with the repository (the image itself is built and used
by scripts/test_container.py in the CI job `container`)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_containerfile_pins_the_torch_of_requirements():
    req = (ROOT / "requirements.txt").read_text()
    pin = re.search(r"^torch==([\d.]+)", req, re.MULTILINE).group(1)
    cf = (ROOT / "Containerfile").read_text()
    assert f'"torch=={pin}"' in cf, "Containerfile installs another torch than requirements.txt"
    assert "pip install -r requirements.txt" in cf
    assert re.search(r"^FROM docker\.io/library/python:3\.12", cf, re.MULTILINE), (
        "same Python as .python-version"
    )
    assert "DLNET_UI_BIND=0.0.0.0" in cf and 'ENTRYPOINT ["dlnet"]' in cf


def test_ignore_files_agree_and_keep_the_build_inputs():
    def rules(name):
        return [
            ln for ln in (ROOT / name).read_text().splitlines() if ln and not ln.startswith("#")
        ]

    assert rules(".containerignore") == rules(".dockerignore")
    ignored = set(rules(".containerignore"))
    for needed in ("dlnet", "requirements.txt", "pyproject.toml", "README.md"):
        assert needed not in ignored
    assert {".git", "examples", "ref"} <= ignored  # hundreds of MB the build does not need


def test_ui_bind_option(monkeypatch, tmp_path):
    import argparse

    from dlnet import ui_server

    sub = argparse.ArgumentParser().add_subparsers()
    p = ui_server.add_parser(sub)
    assert p.parse_args([]).bind == "127.0.0.1"
    monkeypatch.setenv("DLNET_UI_BIND", "0.0.0.0")
    sub = argparse.ArgumentParser().add_subparsers()
    assert ui_server.add_parser(sub).parse_args([]).bind == "0.0.0.0"
    srv = ui_server.make_server(tmp_path / "ws", 0, "127.0.0.1")
    try:
        assert srv.server_address[0] == "127.0.0.1"
    finally:
        srv.server_close()
