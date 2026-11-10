"""Hyperspectral data handling (numpy + scipy + torch).

Copied verbatim into every exported HSI package so that train.py can rebuild the exact same data.
"""

from __future__ import annotations

import numpy as np
import torch

DEFAULT_FRAC = {"pixel": (0.1, 0.1, 0.8), "block": (0.25, 0.1, 0.65)}


# --------------------------------------------------------------------------- loading


def _arrays(path: str) -> dict[str, np.ndarray]:
    p = str(path).lower()
    if p.endswith(".mat"):
        from scipy.io import loadmat

        m = loadmat(path)
        return {k: v for k, v in m.items() if not k.startswith("__") and isinstance(v, np.ndarray)}
    if p.endswith(".npz"):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    if p.endswith(".npy"):
        return {"array": np.load(path)}
    raise ValueError(f"unsupported file type: {path} (use .mat, .npz or .npy)")


def load(
    path: str, gt_path: str | None = None, cube_key: str | None = None, gt_key: str | None = None
):
    """Return (cube float32 [H,W,B], gt int [H,W]; 0 = unlabeled)."""
    arrs = _arrays(path)
    cubes = {k: v for k, v in arrs.items() if v.ndim == 3}
    if cube_key:
        cube = arrs[cube_key]
    elif len(cubes) == 1:
        cube = next(iter(cubes.values()))
    else:
        raise ValueError(f"cannot pick the cube among {sorted(arrs)}; pass --cube-key")
    garrs = _arrays(gt_path) if gt_path else {k: v for k, v in arrs.items() if v.ndim == 2}
    gts = {k: v for k, v in garrs.items() if v.ndim == 2 and v.shape == cube.shape[:2]}
    if gt_key:
        gt = garrs[gt_key]
    elif len(gts) == 1:
        gt = next(iter(gts.values()))
    else:
        raise ValueError(
            f"cannot pick the ground truth among {sorted(garrs)}; pass --gt and/or --gt-key"
        )
    if gt.shape != cube.shape[:2]:
        raise ValueError(f"ground truth {gt.shape} does not match cube {cube.shape}")
    return cube.astype(np.float32), np.rint(gt).astype(np.int64)


# --------------------------------------------------------------------------- splitting


def make_split(gt, mode, seed, frac, patch, block):
    """Split labeled pixels into train/val/test (ids 0/1/2).

    pixel: stratified few-shot split per class (common in the literature, but the patches of
           train and test pixels overlap, so accuracy can be inflated).
    block: the image is cut into square blocks; whole blocks go to one split, and pixels whose
           patch window touches another split are dropped, so no patch crosses a split boundary.
    Returns rows, cols, labels (0-based dense), split_id, keep mask, info.
    """
    rows, cols = np.nonzero(gt > 0)
    raw = gt[rows, cols]
    classes = np.unique(raw)
    labels = np.searchsorted(classes, raw)
    K = len(classes)
    info = {"n_labeled": len(rows), "classes": classes.tolist()}
    if mode == "pixel":
        rng = np.random.default_rng(seed)
        sid = np.empty(len(rows), dtype=np.int64)
        for c in range(K):
            idx = rng.permutation(np.nonzero(labels == c)[0])
            n = len(idx)
            ntr = min(max(round(frac[0] * n), min(3, max(n - 2, 1))), n)
            nva = min(max(round(frac[1] * n), 1 if n >= 3 else 0), n - ntr)
            sid[idx[:ntr]] = 0
            sid[idx[ntr : ntr + nva]] = 1
            sid[idx[ntr + nva :]] = 2
        keep = np.ones(len(rows), dtype=bool)
    elif mode == "block":
        from scipy.ndimage import maximum_filter, minimum_filter

        H, W = gt.shape
        nby, nbx = -(-H // block), -(-W // block)
        nb = nby * nbx
        bid = (np.arange(H)[:, None] // block) * nbx + (np.arange(W)[None, :] // block)
        best = None
        for t in range(300):
            order = np.random.default_rng(seed * 1000 + t).permutation(nb)
            ntr, nva = max(1, round(frac[0] * nb)), max(1, round(frac[1] * nb))
            bsplit = np.full(nb, 2)
            bsplit[order[:ntr]] = 0
            bsplit[order[ntr : ntr + nva]] = 1
            smap = bsplit[bid]
            s = smap[rows, cols]
            cov = tuple(len(np.unique(labels[s == k])) for k in (0, 2, 1))
            if best is None or cov > best[0]:
                best = (cov, smap)
            if cov[0] == K and cov[1] == K:
                break
        smap = best[1]
        r = patch // 2
        same = minimum_filter(smap, size=2 * r + 1, mode="nearest") == maximum_filter(
            smap, size=2 * r + 1, mode="nearest"
        )
        sid, keep = smap[rows, cols], same[rows, cols]
    else:
        raise ValueError(f"split must be 'pixel' or 'block', got {mode!r}")
    info["n_dropped_by_margin"] = int((~keep).sum())
    return rows, cols, labels, sid, keep, info


# ------------------------------------------------------------------------ tensor rank


def multilinear_rank(cube: np.ndarray, energy: float = 0.99, max_dim: int = 4000) -> list[int]:
    """Estimate the multilinear (Tucker/HOSVD) rank: singular-value energy of each mode unfolding."""
    X = cube - cube.mean(axis=(0, 1), keepdims=True)
    out = []
    for mode in range(3):
        M = np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)
        if M.shape[1] > max_dim:  # subsample columns for speed
            M = M[:, np.random.default_rng(0).choice(M.shape[1], max_dim, replace=False)]
        w = np.linalg.eigvalsh(M @ M.T)[::-1].clip(0)
        out.append(int(np.searchsorted(np.cumsum(w) / w.sum(), energy) + 1))
    return out


# ---------------------------------------------------------------------------- patches


class PatchStore:
    """Lazy patches: indexing returns [n, C, p, p] gathered from the padded cube."""

    def __init__(self, cube_pad: torch.Tensor, rows, cols, patch: int):
        self.r = patch // 2
        self.cube = cube_pad  # [H+2r, W+2r, C]
        self.rows = torch.as_tensor(rows) + self.r
        self.cols = torch.as_tensor(cols) + self.r
        self.off = torch.arange(-self.r, self.r + 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        idx = torch.as_tensor(idx)
        ys = self.rows[idx][:, None] + self.off[None, :]
        xs = self.cols[idx][:, None] + self.off[None, :]
        return self.cube[ys[:, :, None], xs[:, None, :]].permute(0, 3, 1, 2).contiguous()


def build(cube, gt, split="pixel", seed=0, patch=9, pca=30, frac=None, block=16) -> dict:
    """Split, fit the spectral stem on TRAIN pixels only, and return the data dict for Trainer."""
    frac = tuple(frac) if frac else DEFAULT_FRAC[split]
    rows, cols, labels, sid, keep, info = make_split(gt, split, seed, frac, patch, block)
    r, c, y, s = rows[keep], cols[keep], labels[keep], sid[keep]
    tr = s == 0

    spec = cube[r[tr], c[tr]].astype(np.float64)
    mean, std = spec.mean(0), spec.std(0) + 1e-6
    Z = (spec - mean) / std
    B = cube.shape[2]
    if pca and pca < B:
        _, sv, vt = np.linalg.svd(Z, full_matrices=False)
        k = min(pca, len(sv))
        V = vt[:k].T
        info["pca_energy_kept"] = float(np.sum(sv[:k] ** 2) / np.sum(sv**2))
    else:
        k, V = B, np.eye(B)
        info["pca_energy_kept"] = 1.0
    scale = (Z @ V).std(0) + 1e-6
    cube_t = (((cube - mean) / std) @ V / scale).astype(np.float32)

    p = patch // 2
    pad = torch.from_numpy(np.pad(cube_t, ((p, p), (p, p), (0, 0)), mode="reflect"))
    K = len(info["classes"])
    d = {
        "task": "classification",
        "classes": info["classes"],
        "in_dim": int(k),
        "out_dim": K,
        "seed": seed,
        "primary": "accuracy",
        "patch": patch,
        "split_mode": split,
        "y_raw": {},
        "Xnp": {},
    }
    for name, sp in (("tr", 0), ("va", 1), ("te", 2)):
        m = s == sp
        d["X" + name] = PatchStore(pad, r[m], c[m], patch)
        d["y" + name] = torch.from_numpy(y[m].astype(np.int64))
        d["y_raw"][name] = y[m].astype(np.int64)
        d["Xnp"][name] = cube_t[r[m], c[m]]
    info.update(
        bands=int(B),
        pca_components=int(k),
        n_train=int(tr.sum()),
        n_val=int((s == 1).sum()),
        n_test=int((s == 2).sum()),
        classes_missing_train=sorted(set(range(K)) - set(y[tr].tolist())),
        classes_missing_test=sorted(set(range(K)) - set(y[s == 2].tolist())),
        frac=list(frac),
        block=block,
    )
    d["info"] = info
    return d
