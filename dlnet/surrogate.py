"""Zero-shot surrogate: learn which seed architectures suit a dataset from past runs.

Algorithm: ZAP (Öztürk, Ferreira, Jomaa, Schmidt-Thieme, Grabocka, "Zero-Shot AutoML with
Pretrained Models", ICML 2022, arXiv:2206.08476, Sec. 3.2, eq. 3-4). A surrogate
f(x, phi; psi) scores configuration x on a dataset with meta-features phi; it is meta-learned
on a meta-dataset of (dataset, configuration, result) triples with the pairwise ranking loss

    min_psi  sum_{(i,j,k) in E}  log sigma( f_{i,j} - f_{i,k} ),   E = {(i,j,k): C(x_j,D_i) < C(x_k,D_i)}

(C = cost; DLNet uses score = -cost, so the loss is -log sigma(s_j - s_k) for s_j better than
s_k: the same objective with the sign of the score flipped). At test time the surrogate ranks
the candidates of a NEW dataset from its meta-features alone (zero-shot).

DLNet specifics (not from the paper):
  * x = traits of the architecture read from its spec (dlnet/seeds.py OP_TRAITS), log10 params,
    log10 FLOPs and the number of leaf modules; ZAP uses one-hot pretrained models and
    fine-tuning hyperparameters instead. phi = the image data card (dlnet/datacard.py).
  * f = MLP with two hidden layers of 32 units, an ensemble of 3 seeds averaged; inputs z-scored
    with the meta-dataset statistics. The paper's network size and training schedule were not
    copied (their meta-dataset is far larger).
  * The meta-dataset comes from `--meta-store` records with per-candidate results (written by
    `dlnet design`) or from scripts/build_meta.py.
"""

from __future__ import annotations

import math

import numpy as np

CARD_KEYS = (
    "n_train", "n_classes", "channels", "height", "spectral_slope", "high_freq_energy",
    "corr_length_rel", "edge_density", "position_dependence", "channel_correlation",
    "knn_accuracy", "linear_probe_pixels", "linear_probe_random_conv", "conv_bias_gain",
)  # fmt: skip
LOG_KEYS = {"n_train", "n_classes", "height"}


def card_vector(card: dict) -> list[float]:
    """Fixed-length meta-features phi from a data card (missing values: neutral defaults)."""
    default = {"channel_correlation": 1.0, "channels": 1.0, "n_classes": 2.0}
    out = []
    for k in CARD_KEYS:
        v = card.get(k, default.get(k, 0.0))
        v = float(v) if v is not None and np.isfinite(v) else default.get(k, 0.0)
        out.append(math.log10(max(v, 1.0)) if k in LOG_KEYS else v)
    return out


def config_vector(cfg: dict) -> list[float]:
    """x: architecture traits (multi-hot), log10 params, log10 FLOPs, number of leaf modules."""
    from .seeds import TRAITS, traits
    from .zoo import leaf_ops

    t = traits(cfg)
    spec = cfg.get("spec")
    n_ops = len(leaf_ops(spec)) if spec else 0
    return [
        *(1.0 if name in t else 0.0 for name in TRAITS),
        math.log10(max(float(cfg.get("params") or 1), 1.0)),
        math.log10(max(float(cfg.get("flops") or 1), 1.0)),
        math.log10(1 + n_ops),
    ]


def candidate_record(cfg: dict, val: float) -> dict:
    return {
        "x": [round(v, 5) for v in config_vector(cfg)],
        "family": cfg.get("family"),
        "val": float(val),
    }


class Surrogate:
    """Ensemble of MLPs trained with the ZAP pairwise ranking loss."""

    def __init__(self, hidden=32, members=3, steps=300, lr=3e-3, wd=1e-3, use_meta=True, seed=0):
        self.hidden, self.members, self.steps, self.lr, self.wd = hidden, members, steps, lr, wd
        self.use_meta, self.seed = use_meta, seed
        self.nets, self.mu, self.sd = [], None, None

    def _rows(self, datasets):
        X, groups, y = [], [], []
        for g, d in enumerate(datasets):
            phi = d["phi"] if self.use_meta else []
            for c in d["candidates"]:
                X.append([*c["x"], *phi])
                y.append(c["val"])
                groups.append(g)
        return np.array(X, float), np.array(groups), np.array(y, float)

    def fit(self, datasets: list[dict]) -> Surrogate:
        """datasets: [{"phi": [...], "candidates": [{"x": [...], "val": v}, ...]}, ...]."""
        import torch

        X, g, y = self._rows(datasets)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        pairs = [(j, k) for gi in np.unique(g) for j in np.flatnonzero(g == gi)
                 for k in np.flatnonzero(g == gi) if y[j] > y[k]]  # fmt: skip
        if not pairs:
            raise ValueError("the meta-dataset has no ordered pair")
        J = torch.tensor([p[0] for p in pairs])
        K = torch.tensor([p[1] for p in pairs])
        self.nets = []
        threads = torch.get_num_threads()
        torch.set_num_threads(1)  # a tiny MLP: thread start-up costs more than the maths
        try:
            self._train(X, Xt, J, K)
        finally:
            torch.set_num_threads(threads)
        return self

    def _train(self, X, Xt, J, K):
        import torch

        for m in range(self.members):
            torch.manual_seed(self.seed + m)
            net = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], self.hidden), torch.nn.ReLU(),
                torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU(),
                torch.nn.Linear(self.hidden, 1),
            )  # fmt: skip
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
            for _ in range(self.steps):
                s = net(Xt).squeeze(-1)
                loss = -torch.nn.functional.logsigmoid(s[J] - s[K]).mean()  # ZAP eq. 4, score form
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.nets.append(net.eval())

    def score(self, phi: list[float], xs: list[list[float]]) -> list[float]:
        """Higher = predicted better on the dataset with meta-features phi."""
        import torch

        X = np.array([[*x, *(phi if self.use_meta else [])] for x in xs], float)
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        with torch.no_grad():
            return np.mean([n(Xt).squeeze(-1).numpy() for n in self.nets], 0).tolist()


def datasets_from_records(records: list[dict], plugin: str) -> list[dict]:
    """Meta-store records that carry per-candidate results (format of `meta.make_record`)."""
    out = []
    for r in records:
        if (
            r.get("plugin") == plugin
            and r.get("card") is not None
            and len(r.get("candidates", [])) >= 3
        ):
            out.append(
                {
                    "name": r.get("name", ""),
                    "phi": card_vector(r["card"]),
                    "candidates": r["candidates"],
                }
            )
    return out


MIN_DATASETS = 3
BUILTIN = {"image_classification": "data/meta_image.jsonl"}


def builtin_records(plugin: str) -> list[dict]:
    """Meta-dataset shipped with DLNet (scripts/build_meta.py on 14 public / synthetic image sets,
    24 random candidates each, 4 epochs; agent.md 16.6). Empty for plugins without one."""
    from pathlib import Path

    from .meta import load

    rel = BUILTIN.get(plugin)
    return load(Path(__file__).parent / rel) if rel else []


def rank_candidates(records, plugin, card, cands) -> list[float] | None:
    """Scores of `cands` on the dataset described by `card`: Borda count of the ZAP surrogate with
    and without meta-features (the combination with the best leave-one-dataset-out results,
    agent.md 16.6). None with fewer than MIN_DATASETS usable past datasets."""
    from .seeds import borda

    ds = datasets_from_records(records, plugin)
    if len(ds) < MIN_DATASETS:
        return None
    xs = [config_vector(c) for c in cands]
    phi = card_vector(card)
    with_meta = Surrogate().fit(ds).score(phi, xs)
    without = Surrogate(use_meta=False).fit(ds).score(phi, xs)
    return borda([(with_meta, 1.0), (without, 1.0)]).tolist()
