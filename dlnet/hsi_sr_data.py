"""Hyperspectral super-resolution data: high-resolution cubes -> (low-resolution tile, high-resolution tile) pairs.

Copied into every exported `hsi_sr` package (needs torch, numpy; scipy for .mat files).

Protocol (Wald et al. 1997, the usual one for single-image HSI super-resolution benchmarks): the given
cube is the ground truth; its low-resolution input is simulated by a known degradation, so every
metric compares the network output with real pixels.

    degrade "bicubic": bicubic downsampling with anti-aliasing (torch.nn.functional.interpolate)
    degrade "gauss":   Gaussian blur (sigma = scale / 2) then keeping every `scale`-th pixel

Split: with 5 or more cubes, whole cubes go to train / validation / test (60/20/20, unseen scenes);
otherwise every cube is cut into non-overlapping tiles and the tiles are split. Tiles never overlap,
so no test pixel is seen in training; neighbouring tiles of one scene are still correlated.
Scaling: values are divided by the largest training value (after shifting by the smallest one if it
is negative), so metrics are on a [0, 1]-like range: PSNR with peak 1, as in the literature.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

DEGRADATIONS = ("bicubic", "gauss")
SPLIT_BY_CUBE = 5  # at least this many cubes: split whole cubes, not tiles


def _arrays(path: str) -> dict[str, np.ndarray]:
    p = str(path).lower()
    if p.endswith(".mat"):
        from scipy.io import loadmat

        m = loadmat(path)
        return {k: v for k, v in m.items() if not k.startswith("__") and isinstance(v, np.ndarray)}
    if p.endswith(".npz"):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    if p.endswith(".npy"):
        return {"array": np.load(path)}
    raise ValueError(f"unsupported file type: {path} (use .mat, .npz or .npy)")


def load(path: str, key: str | None = None) -> np.ndarray:
    """(N, H, W, B) float32: one cube (H, W, B) or a stack of cubes (N, H, W, B).

    The array is `key`, else `hr` (or `HR`), else the only 3-D or 4-D array of the file."""
    arrs = _arrays(path)
    if key:
        if key not in arrs:
            raise ValueError(f"{path}: array {key!r} not found (has {sorted(arrs)})")
        a = arrs[key]
    elif "hr" in arrs or "HR" in arrs:
        a = arrs.get("hr", arrs.get("HR"))
    else:
        cands = [v for v in arrs.values() if v.ndim in (3, 4)]
        if len(cands) != 1:
            raise ValueError(f"cannot pick the cube among {sorted(arrs)}; pass --cube-key")
        a = cands[0]
    if a.ndim not in (3, 4):
        raise ValueError(f"a cube is (H, W, bands) or (N, H, W, bands), got shape {a.shape}")
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError("the cube contains NaN or infinite values")
    return a[None] if a.ndim == 3 else a


def degrade(hr: torch.Tensor, scale: int, mode: str = "bicubic") -> torch.Tensor:
    """(N, B, H, W) -> (N, B, H / scale, W / scale): the simulated low-resolution observation."""
    if mode == "bicubic":
        return F.interpolate(hr, scale_factor=1 / scale, mode="bicubic", align_corners=False,
                             antialias=True)  # fmt: skip
    if mode != "gauss":
        raise ValueError(f"unknown degradation {mode!r}: one of {DEGRADATIONS}")
    sigma = scale / 2
    r = math.ceil(2 * sigma)
    t = torch.arange(-r, r + 1, dtype=hr.dtype)
    g = torch.exp(-(t**2) / (2 * sigma**2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :]).expand(hr.shape[1], 1, -1, -1)
    blur = F.conv2d(F.pad(hr, (r, r, r, r), mode="reflect"), k, groups=hr.shape[1])
    o = scale // 2
    return blur[..., o::scale, o::scale].contiguous()


def upsample(lr: torch.Tensor, scale: int, mode: str = "bicubic") -> torch.Tensor:
    """Classical interpolation (the references): nearest, bilinear or bicubic."""
    if mode == "nearest":
        return F.interpolate(lr, scale_factor=scale, mode="nearest")
    return F.interpolate(lr, scale_factor=scale, mode=mode, align_corners=False)


def tiles(cube: np.ndarray, size: int) -> np.ndarray:
    """Non-overlapping (size x size) tiles of one cube (H, W, B) -> (n, B, size, size)."""
    h, w = cube.shape[0] // size, cube.shape[1] // size
    c = cube[: h * size, : w * size].reshape(h, size, w, size, -1)
    return np.ascontiguousarray(c.transpose(0, 2, 4, 1, 3).reshape(h * w, -1, size, size))


def _split(n: int, seed: int, frac=(0.6, 0.2, 0.2)):
    perm = np.random.default_rng(seed).permutation(n)
    a, b = round(frac[0] * n), round((frac[0] + frac[1]) * n)
    return perm[:a], perm[a:b], perm[b:]


def build(
    cubes: np.ndarray, scale: int = 2, patch: int = 32, seed: int = 0, degradation: str = "bicubic"
) -> dict:
    """The data dict of the trainer: X = low-resolution tiles, y = high-resolution tiles."""
    if scale < 2:
        raise ValueError("--scale must be at least 2")
    if patch % scale or patch // scale < 4:
        raise ValueError(f"--sr-patch {patch} must be a multiple of --scale {scale} and give low-resolution "
                         f"tiles of at least 4 pixels")  # fmt: skip
    if degradation not in DEGRADATIONS:
        raise ValueError(f"unknown degradation {degradation!r}: one of {DEGRADATIONS}")
    if min(cubes.shape[1:3]) < patch:
        raise ValueError(
            f"the cube ({cubes.shape[1]} x {cubes.shape[2]}) is smaller than one tile ({patch})"
        )
    by_cube = len(cubes) >= SPLIT_BY_CUBE
    if by_cube:
        parts = [
            np.concatenate([tiles(cubes[i], patch) for i in idx])
            for idx in _split(len(cubes), seed)
        ]
    else:
        allt = np.concatenate([tiles(c, patch) for c in cubes])
        parts = [allt[idx] for idx in _split(len(allt), seed)]
    if min(len(p) for p in parts) == 0:
        raise ValueError(f"too few tiles ({sum(len(p) for p in parts)}) of {patch} pixels for a 60/20/20 "
                         "split: use a smaller --sr-patch or a larger cube")  # fmt: skip
    lo = min(0.0, float(parts[0].min()))
    hi = float(parts[0].max()) - lo or 1.0
    bands = int(cubes.shape[-1])
    d = {
        "task": "super_resolution", "kind": "hsi_sr", "classes": None, "in_dim": bands, "out_dim": bands,
        "scale": int(scale), "primary": "psnr", "shape": [bands, patch // scale, patch // scale],
        "seed": seed, "keys": {}, "y_raw": {}, "Xnp": {}, "eval_batch": 64,
        "preprocess": {"type": "hsi_sr", "lo": lo, "hi": hi, "scale": int(scale), "tile": patch // scale,
                       "bands": bands, "degradation": degradation},
        "info": {"cubes": len(cubes), "cube_shape": list(cubes.shape[1:]), "split_by": "cube" if by_cube else "tile",
                 "patch": patch, "scale": int(scale), "degradation": degradation,
                 "n_train": len(parts[0]), "n_val": len(parts[1]), "n_test": len(parts[2])},
    }  # fmt: skip
    for k, p in zip(("tr", "va", "te"), parts, strict=True):
        hr = torch.from_numpy(((p - lo) / hi).astype(np.float32))
        lr = degrade(hr, scale, degradation)
        d["X" + k], d["y" + k] = lr, hr
        d["y_raw"][k] = hr.numpy()
        d["Xnp"][k] = lr.reshape(len(lr), -1).numpy()
    return d


# ----------------------------------------------------------------------------- prediction


class Tiles:
    """A low-resolution cube cut into overlapping tiles of the training size, for a fully trained
    network of any kind (also one that only accepts its training tile size). `stitch` averages
    the overlapping outputs back into one super-resolved cube (H * scale, W * scale, B)."""

    def __init__(self, cube: np.ndarray, pre: dict):
        if cube.ndim != 3 or cube.shape[2] != pre["bands"]:
            raise ValueError(f"need a cube (H, W, {pre['bands']}), got shape {cube.shape}")
        self.pre, t, self.shape = pre, pre["tile"], cube.shape[:2]
        x = torch.from_numpy(((cube - pre["lo"]) / pre["hi"]).astype(np.float32)).permute(2, 0, 1)
        ph, pw = max(t - cube.shape[0], 0), max(t - cube.shape[1], 0)
        if ph or pw:  # smaller than one tile: replicate the border
            x = F.pad(x[None], (0, pw, 0, ph), mode="replicate")[0]
        self.padded = x.shape[1:]

        def starts(n):
            s = list(range(0, n - t + 1, max(t // 2, 1)))
            return s if s[-1] == n - t else [*s, n - t]

        self.pos = [(i, j) for i in starts(x.shape[1]) for j in starts(x.shape[2])]
        self.tiles = torch.stack([x[:, i : i + t, j : j + t] for i, j in self.pos])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, k):
        return self.tiles[k]

    def stitch(self, out: torch.Tensor) -> np.ndarray:
        s, t = self.pre["scale"], self.pre["tile"] * self.pre["scale"]
        h, w = self.padded[0] * s, self.padded[1] * s
        acc = torch.zeros(out.shape[1], h, w)
        cnt = torch.zeros(1, h, w)
        for (i, j), o in zip(self.pos, out, strict=True):
            acc[:, i * s : i * s + t, j * s : j * s + t] += o
            cnt[:, i * s : i * s + t, j * s : j * s + t] += 1
        sr = (acc / cnt)[:, : self.shape[0] * s, : self.shape[1] * s]
        return (sr.permute(1, 2, 0).numpy() * self.pre["hi"] + self.pre["lo"]).astype(np.float32)
