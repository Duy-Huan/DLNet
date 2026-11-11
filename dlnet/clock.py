"""The budget clock of `dlnet design`: wall-clock seconds, or training epochs.

`--budget 10m` measures seconds (the result depends on the machine's speed). `--budget 400ep`
measures WORK: every epoch any candidate trains costs one unit, and the steps that do not train
(profiling, references, proxies) cost nothing. A work budget makes a run reproducible: the same
command gives the same candidates, the same training and the same leaderboard on a fast and on a
slow machine (up to floating-point non-determinism of the BLAS library). Stage fractions of the
budget (proxies, search, evolution, hyperparameters) are the same in both modes.

`stage(name)` marks where the run is; `breakdown()` returns the seconds and epochs every stage
used, so a run that overshoots its budget says where the time went (leaderboard "budget").
"""

from __future__ import annotations

import time

_mode = "time"
_spent = 0.0  # epochs trained since start(); counted in both modes
_stages: list[list] = []  # [name, wall start, epochs at start, wall end, epochs at end]


def start(mode: str) -> float:
    """Select the mode ("time" or "work"), reset the counters and return now()."""
    global _mode, _spent
    if mode not in ("time", "work"):
        raise ValueError(f"unknown clock mode {mode!r}")
    _mode, _spent = mode, 0.0
    _stages.clear()
    return now()


def mode() -> str:
    return _mode


def now() -> float:
    return time.time() if _mode == "time" else _spent


def epochs() -> float:
    return _spent


def spend(n: float) -> None:
    """Called by the trainer after every epoch."""
    global _spent
    _spent += n


def stage(name: str) -> None:
    """Close the current stage and open `name`; a repeated name accumulates."""
    t = time.time()
    if _stages and _stages[-1][3] is None:
        _stages[-1][3:] = [t, _spent]
    _stages.append([name, t, _spent, None, None])


def breakdown() -> list[dict]:
    """Seconds and epochs per stage, in first-seen order (the open stage is measured up to now)."""
    out: dict[str, dict] = {}
    for name, t0, e0, t1, e1 in _stages:
        t1, e1 = (time.time(), _spent) if t1 is None else (t1, e1)
        d = out.setdefault(name, {"stage": name, "seconds": 0.0, "epochs": 0.0})
        d["seconds"] += t1 - t0
        d["epochs"] += e1 - e0
    return [{**d, "seconds": round(d["seconds"], 2)} for d in out.values()]


def fits(cost: float, deadline: float) -> bool:
    """True when work estimated at `cost` (seconds, or epochs in work mode) ends by `deadline`."""
    return now() + cost <= deadline
