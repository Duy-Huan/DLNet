"""How many CPU threads DLNet's numerical libraries may use (`--threads`).

torch, OpenBLAS and the OpenMP runtimes of torch and scikit-learn each start one thread per core
by default. When several programs do that on the same cores, OpenMP threads spin while they wait
for each other and the run can stall: four `dlnet design --budget 30s` started together on 4 cores
were still inside HistGradientBoosting after 10 minutes; the same model fitted in 0.2 s per process
with one thread each (agent.md 9.2). `limit(n)` caps every one of these pools for the whole run.

`--threads 0` (auto) = the CPUs this process may run on, minus the CPUs other processes keep busy
(read from /proc/loadavg on Linux), and at least 1. Fewer threads only slow a run down a little;
too many can stall it, so auto errs on the low side. Pass an explicit number for runs that must
be bitwise identical (the thread count changes the order of floating-point sums).
"""

from __future__ import annotations

import contextlib
import math
import os

_current = 0  # 0 = not limited by DLNet


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS, Windows
        return os.cpu_count() or 1


def busy_cpus() -> int:
    """CPUs used by OTHER processes: the larger of the runnable count now and the 1-minute load."""
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
        running = int(parts[3].split("/")[0]) - 1  # minus this process
        return max(running, math.floor(float(parts[0])), 0)
    except (OSError, ValueError, IndexError):
        return 0


def resolve(n: int) -> int:
    """`--threads` value -> thread count (0 = auto)."""
    if n < 0:
        raise SystemExit(f"bad --threads {n} (use 0 for auto or a positive number)")
    return n if n > 0 else max(1, available_cpus() - busy_cpus())


def current() -> int:
    """Threads granted to the current run (n_jobs for scikit-learn); -1 = all, when not limited."""
    return _current or -1


@contextlib.contextmanager
def limit(n: int):
    """Cap torch, BLAS and OpenMP to `n` threads inside the block; restore them afterwards."""
    global _current
    import torch
    from threadpoolctl import threadpool_limits

    before, prev = torch.get_num_threads(), _current
    torch.set_num_threads(n)
    _current = n
    try:
        with threadpool_limits(limits=n):
            yield n
    finally:
        torch.set_num_threads(before)
        _current = prev
