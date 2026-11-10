"""Domain plugins: everything that depends on the kind of data.

The core (search, ranking, export) only talks to this interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar


class DomainPlugin:
    name = "base"
    model_file: Path  # copied into each package as model.py
    extra_files: ClassVar[dict[str, Path]] = {}  # other standalone files copied into each package
    train_template = ""
    test_template = ""
    metric_keys: tuple[str, ...] = ()  # metrics shown in the leaderboard

    def add_arguments(self, parser) -> None:
        """Register plugin-specific CLI options."""

    def detect(self, path: str) -> float:
        """Confidence in [0, 1] that `path` is data for this plugin."""
        return 0.0

    def prepare(self, args) -> dict:
        """Load + split + preprocess. Returns the data dict consumed by Trainer."""
        raise NotImplementedError

    def profile(self, data: dict) -> dict:
        raise NotImplementedError

    def generate(self, prof: dict, n_per_family: int, seed: int) -> list[dict]:
        raise NotImplementedError

    def build_model(self, cfg: dict, in_dim: int, out_dim: int):
        raise NotImplementedError

    def reference(self, data: dict) -> list[dict]:
        raise NotImplementedError

    def data_config(self, data: dict) -> dict:
        """Data-side settings written to config.yaml so train.py can rebuild the same data."""
        return {}


def _registry() -> dict[str, DomainPlugin]:
    from .hsi import HSIPlugin
    from .tabular import TabularPlugin

    return {p.name: p for p in (TabularPlugin(), HSIPlugin())}


def get_plugin(name: str) -> DomainPlugin:
    reg = _registry()
    if name not in reg:
        raise SystemExit(f"unknown plugin {name!r}; available: {sorted(reg)}")
    return reg[name]


def route(path: str) -> DomainPlugin:
    """Plugin router: pick the plugin with the highest detect() confidence."""
    scored = sorted(((p.detect(path), p.name) for p in _registry().values()), reverse=True)
    if scored[0][0] <= 0:
        raise SystemExit(
            f"cannot tell the data type of {path!r}; pass --plugin {sorted(_registry())}"
        )
    return get_plugin(scored[0][1])
