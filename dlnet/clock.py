"""The budget clock of `dlnet design`: wall-clock seconds, or training epochs.

`--budget 10m` measures seconds (the result depends on the machine's speed). `--budget 400ep`
measures WORK: every epoch any candidate trains costs one unit, and the steps that do not train
(profiling, references, proxies) cost nothing. A work budget makes a run reproducible: the same
command gives the same candidates, the same training and the same leaderboard on a fast and on a
slow machine (up to floating-point non-determinism of the BLAS library). Stage fractions of the
budget (proxies, search, evolution, hyperparameters) are the same in both modes.
"""

from __future__ import annotations

import time

_mode = "time"
_spent = 0.0


def start(mode: str) -> float:
    """Select the mode ("time" or "work"), reset the work counter and return now()."""
    global _mode, _spent
    if mode not in ("time", "work"):
        raise ValueError(f"unknown clock mode {mode!r}")
    _mode, _spent = mode, 0.0
    return now()


def mode() -> str:
    return _mode


def now() -> float:
    return time.time() if _mode == "time" else _spent


def spend(epochs: float) -> None:
    """Called by the trainer after every epoch; a no-op in time mode."""
    global _spent
    if _mode == "work":
        _spent += epochs
