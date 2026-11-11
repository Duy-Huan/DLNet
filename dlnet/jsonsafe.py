"""Standard JSON out of DLNet: NaN / inf become null, numpy scalars become Python numbers.

Python's json writes float("nan") as the bare token NaN, which is not JSON (RFC 8259): jq, browsers
and most other languages refuse the whole file. Every file DLNet writes goes through `dumps`.
"""

from __future__ import annotations

import json
import math

import numpy as np


def clean(x):
    """A copy of `x` with non-finite floats replaced by None and numpy values made native."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def dumps(x, **kw) -> str:
    """json.dumps that never writes NaN / Infinity (allow_nan=False after cleaning)."""
    return json.dumps(clean(x), allow_nan=False, **kw)
