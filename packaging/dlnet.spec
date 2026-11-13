# PyInstaller spec of the installed DLNet program (one folder: dlnet[.exe] + _internal/).
# Built by scripts/build_installer.py; see doc/install.md.
#
# The package's .py sources are shipped as data next to the compiled modules: DLNet copies its own
# source into every output package (core.py, zoo/) and the standalone model.py is written from the
# source of the zoo classes, so `dlnet.<module>.__file__` must point at a real file.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821 (defined by PyInstaller)

hidden = collect_submodules("dlnet") + collect_submodules("sklearn") + [
    "scipy.special._cdflib", "pandas", "yaml", "threadpoolctl",
]
datas = collect_data_files("dlnet", include_py_files=True, excludes=["**/__pycache__/**"])

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "IPython", "notebook", "jupyter", "pytest", "torchvision.datasets"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dlnet",
    console=True,
    icon=None,
    target_arch=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="dlnet", strip=False, upx=False)  # noqa: F821
