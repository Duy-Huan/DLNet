"""Data card of an image dataset: measurable characteristics that bear on the choice of architecture.

Every feature is computed on the TRAINING split only (standardised images, (N, C, H, W)) and on at
most `max_images` images. Each one cites where the measurement comes from; what is written from
memory says so. The card only DESCRIBES the data; how it is used to rank seed architectures is in
`dlnet/seeds.py`.

Features
  size        n_train, n_classes, channels, height, width, samples per class, imbalance ratio
              (largest / smallest class count).
  spectrum    `spectral_slope`: alpha of the radially averaged power spectrum P(f) ~ f^-alpha,
              least-squares fit in log-log over 1..Nyquist (natural images: alpha ~ 2, Field 1987,
              van der Schaaf & van Hateren 1996; from memory). Larger = smoother, more large-scale
              structure; smaller = more fine texture/noise.
              `high_freq_energy`: fraction of the (mean-removed) spectral energy above half the
              Nyquist frequency.
  scale       `corr_length`: distance (pixels) at which the spatial autocorrelation of an image,
              averaged over images and channels, first drops below 1/e (from memory); divided by
              the smaller side it gives `corr_length_rel`, a proxy of the size of the structures.
  edges       `edge_density`: share of pixels whose gradient magnitude exceeds twice the dataset
              mean gradient magnitude (high = sparse strong edges such as strokes and shapes on
              a flat background; low = gradients spread evenly, as in textures). DLNet heuristic.
  layout      `position_dependence`: variance across pixels of the per-class mean images divided
              by the total pixel variance (between-class variance of the mean images, from memory).
              High = the class is told apart by WHERE things are (centred digits); low = by what
              appears anywhere (textures, natural scenes).
  colour      `channel_correlation`: mean absolute Pearson correlation between channels (3+ channels;
              near 1 = almost grey).
  separability `knn_accuracy`: 1-nearest-neighbour accuracy on 64 PCA components, leave-one-out
              on a subsample (N1-type complexity measure of Ho & Basu 2002, from memory);
              `linear_probe_pixels` and `linear_probe_random_conv`: logistic-regression accuracy on
              PCA-64 pixels versus on globally pooled features of a fixed random convolution layer
              (random filters as feature extractors: Jarrett et al. 2009, Saxe et al. 2011, from
              memory), both by 3-fold cross-validation on a subsample. `conv_bias_gain` = the
              difference: positive means translation-invariant local features already help.
"""

from __future__ import annotations

import numpy as np


def _radial_profile(P: np.ndarray) -> np.ndarray:
    """Mean of a centred 2-D power spectrum over rings of integer radius."""
    h, w = P.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    s = np.bincount(r.ravel(), P.ravel())
    n = np.bincount(r.ravel())
    return s / np.maximum(n, 1)


def spectrum_features(X: np.ndarray) -> dict:
    """Spectral slope and high-frequency energy fraction of (N, C, H, W) images."""
    _, _, H, W = X.shape
    Z = X - X.mean(axis=(2, 3), keepdims=True)
    P = (np.abs(np.fft.fftshift(np.fft.fft2(Z), axes=(2, 3))) ** 2).mean(axis=(0, 1))
    prof = _radial_profile(P)
    nyq = min(H, W) // 2
    f = np.arange(1, nyq + 1)
    p = prof[1 : nyq + 1]
    ok = p > 0
    slope = (
        float(-np.polyfit(np.log(f[ok]), np.log(p[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")
    )
    yy, xx = np.indices((H, W))
    rr = np.hypot(yy - H // 2, xx - W // 2)
    total = P[rr > 0].sum()
    high = P[rr > nyq / 2].sum()
    return {
        "spectral_slope": round(slope, 3),
        "high_freq_energy": round(float(high / total), 4) if total > 0 else 0.0,
    }


def correlation_length(X: np.ndarray) -> float:
    """Pixels until the mean spatial autocorrelation falls below 1/e (Wiener-Khinchin, circular)."""
    _, _, H, W = X.shape
    Z = X - X.mean(axis=(2, 3), keepdims=True)
    F = np.fft.fft2(Z)
    ac = np.fft.ifft2(np.abs(F) ** 2).real.mean(axis=(0, 1))
    if ac[0, 0] <= 0:
        return 0.0
    ac = np.fft.fftshift(ac / ac[0, 0])
    prof = _radial_profile(ac)  # radius 0 is the centre (value 1)
    below = np.flatnonzero(prof[: min(H, W) // 2 + 1] < 1 / np.e)
    return float(below[0]) if len(below) else float(min(H, W) // 2)


def edge_density(X: np.ndarray) -> float:
    """Share of pixels whose gradient magnitude exceeds twice the dataset mean gradient magnitude."""
    g = np.zeros(X.shape[:1] + X.shape[2:])
    for c in range(X.shape[1]):
        gy, gx = np.gradient(X[:, c], axis=(1, 2))
        g += np.hypot(gx, gy)
    ref = g.mean()  # the median is 0 on flat backgrounds (digits), so the mean is the reference
    return float((g > 2 * ref).mean()) if ref > 0 else 0.0


def position_dependence(X: np.ndarray, y: np.ndarray) -> float:
    """Between-class variance of the class-mean images over the total pixel variance (0..1)."""
    flat = X.reshape(len(X), -1)
    tot = flat.var(axis=0).sum()
    if tot <= 0:
        return 0.0
    mu = flat.mean(0)
    between = 0.0
    for c in np.unique(y):
        m = y == c
        between += m.mean() * ((flat[m].mean(0) - mu) ** 2).sum()
    return float(between / tot)


def channel_correlation(X: np.ndarray) -> float | None:
    C = X.shape[1]
    if C < 2:
        return None
    flat = X.transpose(1, 0, 2, 3).reshape(C, -1)
    r = np.corrcoef(flat)
    return float(np.abs(r[np.triu_indices(C, 1)]).mean())


def _pca(X: np.ndarray, k: int = 64) -> np.ndarray:
    Z = X - X.mean(0)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ vt[: min(k, len(vt))].T


def random_conv_features(X: np.ndarray, n_filters: int = 64, seed: int = 0) -> np.ndarray:
    """Global average and max pooling of ReLU(random 5x5 convolution): translation-invariant features."""
    import torch

    g = torch.Generator().manual_seed(seed)
    C = X.shape[1]
    w = torch.randn(n_filters, C, 5, 5, generator=g) / (25 * C) ** 0.5
    with torch.no_grad():
        z = torch.relu(torch.nn.functional.conv2d(torch.from_numpy(X).float(), w, padding=2))
        return torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1).numpy()


def separability(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """1-NN accuracy and the two linear probes (pixels versus random-convolution features)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    pcs = _pca(X.reshape(len(X), -1))
    knn = KNeighborsClassifier(1)
    nn_acc = cross_val_score(knn, pcs, y, cv=3).mean()
    lr = LogisticRegression(max_iter=300)
    pix = cross_val_score(lr, StandardScaler().fit_transform(pcs), y, cv=3).mean()
    rc = StandardScaler().fit_transform(random_conv_features(X, seed=seed))
    conv = cross_val_score(lr, rc, y, cv=3).mean()
    return {
        "knn_accuracy": round(float(nn_acc), 4),
        "linear_probe_pixels": round(float(pix), 4),
        "linear_probe_random_conv": round(float(conv), 4),
        "conv_bias_gain": round(float(conv - pix), 4),
    }


def image_card(data: dict, max_images: int = 1500, seed: int = 0) -> dict:
    """Data card of an image-classification (or segmentation) data dict built by array_data."""
    X = data["Xtr"].numpy()
    y = np.asarray(data["y_raw"]["tr"])
    rng = np.random.default_rng(seed)
    if len(X) > max_images:
        idx = rng.choice(len(X), max_images, replace=False)
        X, ysub = X[idx], y[idx]
    else:
        ysub = y
    _, C, H, W = X.shape
    card = {"n_train": len(data["Xtr"]), "channels": int(C), "height": int(H), "width": int(W)}
    classification = data.get("task") == "classification" and y.ndim == 1
    if classification:
        counts = np.bincount(y)
        counts = counts[counts > 0]
        card.update(
            n_classes=len(counts),
            samples_per_class_min=int(counts.min()),
            samples_per_class_median=float(np.median(counts)),
            imbalance_ratio=round(float(counts.max() / counts.min()), 2),
        )
    card.update(spectrum_features(X))
    cl = correlation_length(X)
    card["corr_length"] = round(cl, 2)
    card["corr_length_rel"] = round(cl / min(H, W), 3)
    card["edge_density"] = round(edge_density(X), 4)
    cc = channel_correlation(X)
    if cc is not None:
        card["channel_correlation"] = round(cc, 3)
    if classification:
        card["position_dependence"] = round(position_dependence(X, ysub), 4)
        if len(np.unique(ysub)) > 1 and np.bincount(ysub)[np.bincount(ysub) > 0].min() >= 3:
            sub = rng.choice(len(X), min(len(X), 1000), replace=False)
            card.update(separability(X[sub], ysub[sub], seed))
    return card
