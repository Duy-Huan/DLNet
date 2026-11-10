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

    def reference_for(self, cfg: dict, reference: list[dict]) -> list[dict]:
        """The references the top-1 config is judged against (default: all of them)."""
        return reference

    extra_dirs: ClassVar[dict[str, Path]] = {}  # whole packages copied into an exported baseline
    zoo_train: ClassVar[dict] = {"lr": [1e-3, 3e-3], "wd": [1e-4, 1e-2], "batch": [32, 64]}

    def zoo_input(self, data: dict):
        """(tensor kind, per-sample shape, head) the module zoo should build for, or None."""
        return

    def generate_zoo(
        self, data: dict, prof: dict, n: int, seed: int, presets=True, width=16
    ) -> list[dict]:
        """Candidates from the module zoo: presets and random grammar samples, statically checked."""
        spec = self.zoo_input(data)
        if spec is None or n <= 0:
            return []
        from ..zoo import space

        kind, dims, head = spec
        return space.population(
            kind,
            dims,
            data["out_dim"],
            n,
            seed,
            prof["max_params"],
            head,
            self.zoo_train,
            presets,
            width,
        )

    def default_cv_splits(self, args) -> int:
        """How many different splits rank the finalists when --cv-splits is not given."""
        return 1

    def data_config(self, data: dict) -> dict:
        """Data-side settings written to config.yaml so train.py can rebuild the same data."""
        return {}


ALIASES = {
    "image": "image_classification",
    "segmentation": "image_segmentation",
    "recommendation": "recsys",
}


def _registry() -> dict[str, DomainPlugin]:
    from .arrays import (
        GraphClassificationPlugin,
        ImageClassificationPlugin,
        ImageReconstructionPlugin,
        ImageSegmentationPlugin,
        RecommendationPlugin,
        TimeSeriesPlugin,
    )
    from .hsi import HSIPlugin
    from .tabular import TabularPlugin

    plugins = (
        TabularPlugin(),
        HSIPlugin(),
        ImageClassificationPlugin(),
        ImageSegmentationPlugin(),
        ImageReconstructionPlugin(),
        TimeSeriesPlugin(),
        GraphClassificationPlugin(),
        RecommendationPlugin(),
    )
    return {p.name: p for p in plugins}


def get_plugin(name: str) -> DomainPlugin:
    reg = _registry()
    name = ALIASES.get(name, name)
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
