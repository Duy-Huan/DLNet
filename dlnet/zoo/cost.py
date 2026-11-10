"""Cost model: parameters, forward FLOPs, activation memory and training VRAM (torch only).

Measured, not guessed, on the `meta` device (no memory, no time), so every candidate can be
costed before any training:

  flops     forward matrix-multiply and convolution FLOPs of ONE sample, counted by
            torch.utils.flop_counter.FlopCounterMode (2 x multiply-accumulates). Elementwise
            operations, normalisation, activations and Mamba's sequential scan recurrence are
            NOT counted, so this under-estimates models dominated by them.
  act_mb    bytes (MB) of the outputs of all leaf modules for one sample, float32. A proxy for what
            autograd keeps for the backward pass; fused kernels and in-place ops make real use
            lower, modules that reuse tensors make it higher.
  vram_mb   training estimate for batch size b: weights + gradients + two Adam moments
            (4 x 4 bytes per parameter) + 2 x b x act bytes. It ignores the CUDA context, the
            allocator's fragmentation and workspace buffers: treat it as a lower bound of +-30%.
Nothing here has been checked against measured GPU memory (no GPU in this environment).
"""

from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def _first_tensor(o):
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, (tuple, list)) and o:
        return _first_tensor(o[0])
    return None


def model_cost(model: nn.Module, dims, batch: int = 64) -> dict:
    """Cost of an already built model (meta or real) for per-sample input shape `dims`."""
    was = model.training
    model.eval()
    p = next(itertools.chain(model.parameters(), model.buffers()), None)
    dev = p.device if p is not None else torch.device("cpu")
    act = [0]

    def hook(_m, _i, out):
        t = _first_tensor(out)
        if t is not None:
            act[0] += t.numel()

    hooks = [m.register_forward_hook(hook) for m in model.modules() if not list(m.children())]
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(torch.zeros(1, *dims, device=dev))
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    n = sum(q.numel() for q in model.parameters())
    act_bytes = 4 * act[0]
    return {
        "params": n,
        "flops": int(fc.get_total_flops()),
        "act_mb": act_bytes / 1e6,
        "vram_mb": (16 * n + 2 * batch * act_bytes) / 1e6,
    }


def estimate(
    build_fn, cfg: dict, in_dim: int, out_dim: int, dims, batch: int | None = None
) -> dict:
    """Cost of the model `build_fn(cfg, in_dim, out_dim)`; meta device first, CPU if an op needs real data."""
    batch = batch or cfg.get("batch", 64)
    try:
        with torch.device("meta"):
            return model_cost(build_fn(cfg, in_dim, out_dim), dims, batch)
    except (RuntimeError, NotImplementedError, TypeError, ValueError):
        return model_cost(build_fn(cfg, in_dim, out_dim), dims, batch)


def parse_size(s: str | None) -> float | None:
    """'50M' -> 5e7, '2G' -> 2e9, '1500' -> 1500 (also for MB limits: '4G' = 4000 MB is written 4000)."""
    if s is None or s == "":
        return None
    s = str(s).strip().upper()
    mult = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    return float(s[:-1]) * mult[s[-1]] if s[-1] in mult else float(s)


def within(c: dict, max_flops: float | None, max_vram_mb: float | None) -> bool:
    return (max_flops is None or c["flops"] <= max_flops) and (
        max_vram_mb is None or c["vram_mb"] <= max_vram_mb
    )
