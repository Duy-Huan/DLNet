"""v1.6: what a release must ship (scripts/release.py) and `dlnet run-script` (the installed program)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import release

from dlnet import cli


def test_every_release_lists_installers_for_three_platforms(tmp_path):
    names = release.asset_names("1.6.0")
    assert set(names) == {("windows", "x64"), ("macos", "arm64"), ("linux", "x64")}
    flat = [n for d in names.values() for n in d.values()]
    assert "DLNet-1.6.0-windows-x64-setup.exe" in flat and "DLNet-1.6.0-macos-arm64.pkg" in flat
    assert "DLNet-1.6.0-linux-x64.deb" in flat and all(n.startswith("DLNet-1.6.0-") for n in flat)
    assert len(release.missing_assets(tmp_path, "1.6.0")) == 9  # 6 files + 3 checksum lists
    for n in flat + [f"SHA256SUMS-{o}-{a}.txt" for o, a in release.PLATFORMS]:
        (tmp_path / n).write_text("x")
    assert release.missing_assets(tmp_path, "1.6.0") == []
    (tmp_path / "DLNet-1.6.0-linux-x64.deb").unlink()
    assert release.missing_assets(tmp_path, "1.6.0") == ["DLNet-1.6.0-linux-x64.deb"]
    assert release.main(["assets", str(tmp_path)]) == 1


def test_installer_definitions_match_the_names_the_release_expects():
    iss = (ROOT / "packaging" / "windows" / "dlnet.iss").read_text()
    assert "OutputBaseFilename=DLNet-{#AppVersion}-windows-x64-setup" in iss
    spec = (ROOT / "packaging" / "dlnet.spec").read_text()
    assert (
        'collect_data_files("dlnet", include_py_files=True' in spec
    )  # sources for packages / model.py
    wf = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "installers.yml" in wf and "release.py assets installers" in wf
    assert wf.index("release.py assets") < wf.index("gh release create")


def test_run_script_runs_a_package_script_with_its_folder_on_the_path(tmp_path, capsys):
    (tmp_path / "helper.py").write_text("VALUE = 7\n")
    (tmp_path / "s.py").write_text(
        "import sys, helper\nprint('got', sys.argv[1:], helper.VALUE, __name__)\nsys.exit(int(sys.argv[1]))\n"
    )
    old_path = sys.path[:]
    assert cli.main(["run-script", str(tmp_path / "s.py"), "3", "--flag"]) == 3
    assert "got ['3', '--flag'] 7 __main__" in capsys.readouterr().out
    assert sys.path == old_path
    (tmp_path / "ok.py").write_text("print('fine')\n")
    assert cli.main(["run-script", str(tmp_path / "ok.py")]) == 0
    assert cli.main(["run-script", str(tmp_path / "missing.py")]) == 2


# --------------------------------------------------------------------- scripts/test_install.py


def _test_install():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_install_script", ROOT / "scripts" / "test_install.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_install_version_from_file_name():
    ti = _test_install()
    assert ti.version_of(Path("DLNet-1.6.0-windows-x64-setup.exe")) == "1.6.0"
    assert ti.version_of(Path("DLNet-1.6.0rc1-linux-x64.deb")) == "1.6.0rc1"
    with pytest.raises(AssertionError):
        ti.version_of(Path("dlnet.deb"))


def test_install_checksums(tmp_path, monkeypatch):
    import hashlib

    ti = _test_install()
    monkeypatch.setattr(ti, "OUT", tmp_path)
    (tmp_path / "DLNet-1.0.0-linux-x64.deb").write_bytes(b"deb")
    digest = hashlib.sha256(b"deb").hexdigest()
    sums = tmp_path / "SHA256SUMS-linux-x64.txt"
    sums.write_text(f"{digest}  DLNet-1.0.0-linux-x64.deb\n")
    ti.check_sums()
    (tmp_path / "DLNet-1.0.0-linux-x64.deb").write_bytes(b"changed")
    with pytest.raises(AssertionError, match="does not match"):
        ti.check_sums()


def test_install_windows_path_matching():
    ti = _test_install()
    folder = Path("C:/Users/a/AppData/Local/Programs/DLNet")
    assert ti._in_path(folder, f"C:/x;{os.path.normcase(str(folder))};C:/y")
    assert not ti._in_path(folder, "C:/x;C:/y")
