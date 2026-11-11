"""Exported into every baseline package as model.py: builds the architecture of config.yaml with the zoo.

`config.yaml` holds the full spec (`arch["spec"]`), the input kind and shape, the head and the output
size, so `build_model(cfg["arch"], in_dim, out_dim)` needs nothing else. The `zoo/` folder next to
this file is a copy of dlnet/zoo.
"""

from zoo import build_model, count_params  # noqa: F401
