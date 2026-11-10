"""Data and training helpers copied into an exported GAN package (torch + numpy only)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def load(path, key, seed):
    z = np.load(path)
    X = np.asarray(z[key or "X"]).astype(np.float32)
    if X.ndim == 3:
        X = X[..., None]
    X = X.transpose(0, 3, 1, 2)
    lo, hi = float(X.min()), float(X.max())
    X = (2 * (X - lo) / max(hi - lo, 1e-9) - 1).astype(np.float32)
    perm = np.random.default_rng(seed).permutation(len(X))
    a, b = int(0.6 * len(X)), int(0.8 * len(X))
    t = torch.from_numpy
    return {"train": t(X[perm[:a]]), "val": t(X[perm[a:b]]), "test": t(X[perm[b:]])}


def mmd2(x, y):
    x, y = x.flatten(1).double(), y.flatten(1).double()
    n, m = len(x), len(y)
    z = torch.cat([x, y])
    d2 = torch.cdist(z, z).square()
    med = d2[~torch.eye(len(z), dtype=torch.bool)].median().clamp_min(1e-9)
    k = sum(torch.exp(-d2 / (s * med)) for s in (0.5, 1.0, 2.0))
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return float(
        (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
        + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
        - 2 * kxy.mean()
    )


def train_gan(G, D, Xtr, Xva, zdim, epochs, batch=64, lr=2e-4, seed=0):
    torch.manual_seed(seed)
    og = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))
    bce, val, best = nn.BCEWithLogitsLoss(), Xva[:400], float("inf")
    for _ in range(epochs):
        G.train()
        D.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr) - batch + 1, batch):
            real, fake = Xtr[perm[i : i + batch]], G(torch.randn(batch, zdim))
            ld = bce(D(real).squeeze(1), torch.ones(batch)) + bce(
                D(fake.detach()).squeeze(1), torch.zeros(batch)
            )
            od.zero_grad()
            ld.backward()
            od.step()
            lg = bce(D(fake).squeeze(1), torch.ones(batch))
            og.zero_grad()
            lg.backward()
            og.step()
        G.eval()
        with torch.no_grad():
            best = min(best, mmd2(G(torch.randn(len(val), zdim)), val))
    return {"best_mmd": best}
