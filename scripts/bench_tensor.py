"""Measure the tensor modules on real data (scikit-learn digits, bundled, no download).

    python scripts/bench_tensor.py

Part A, compression: the same residual MLP with dense / Tensor-Train / Tensor-Ring / low-rank
hidden layers; parameters saved versus test accuracy lost.
Part B, Tensor-LoRA: pretrain on digits 0-4, adapt to digits 5-9 from few labels with the
backbone frozen; trainable parameters versus test accuracy for each delta format.
Results: agent.md sec. 4 (table "Kết quả đo M7").
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from dlnet import core, tensor
from dlnet import tensor_lora as tl
from dlnet.zoo import build_model as _zoo_build
from dlnet.zoo import count_params
from dlnet.zoo.families import complete

SEEDS, EPOCHS = (0, 1, 2), 40


def build_model(cfg, in_dim, out_dim):
    """A tabular family of dlnet/zoo/families.py as a model."""
    cfg = dict(cfg)
    complete(cfg, "tabular", (in_dim,), out_dim)
    return _zoo_build(cfg)


def _fit(model, data, seed, lr=3e-3):
    tr, _ = core.train_full(model, data, lr, 1e-4, 64, EPOCHS, seed=seed)
    return tr.evaluate("te")["accuracy"]


def part_a(X, y):
    print("\n## A. Compression (digits, 10 classes, width 256, depth 2, 3 split+init seeds)\n")
    print("| hidden layers | params | saved vs dense | test acc | lost vs dense |")
    print("|---|---|---|---|---|")
    rows = []
    for fam, rank in (("resmlp", 0), ("tt_resmlp", 16), ("tt_resmlp", 4), ("tr_resmlp", 4),
                      ("tr_resmlp", 2), ("lowrank_mlp", 16)):  # fmt: skip
        cfg = {"family": fam, "width": 256, "depth": 2, "dropout": 0.1, "rank": rank}
        accs = []
        for s in SEEDS:
            data = core.prepare(X, y, "classification", seed=s)
            torch.manual_seed(s)
            m = build_model(cfg, data["in_dim"], data["out_dim"])
            accs.append(_fit(m, data, s))
        rows.append(
            (f"{fam} r={rank}" if rank else fam, count_params(m), np.mean(accs), np.std(accs))
        )
    p0, a0 = rows[0][1], rows[0][2]
    for name, p, a, sd in rows:
        print(f"| {name} | {p:,} | {1 - p / p0:.1%} | {a:.4f} ± {sd:.4f} | {a0 - a:+.4f} |")


def part_b(X, y):
    print("\n## B. Tensor-LoRA transfer (pretrain digits 0-4, adapt to 5-9 with 100 labels)\n")
    print("| method | trainable params | test acc on 5-9 |")
    print("|---|---|---|")
    cfg = {"family": "mlp", "width": 128, "depth": 3, "dropout": 0.0}
    src, tgt = y < 5, y >= 5
    res: dict[str, list] = {}
    for s in SEEDS:
        d_src = core.prepare(X[src], y[src], "classification", seed=s)
        torch.manual_seed(s)
        base = build_model(cfg, 64, 5)
        _fit(base, d_src, s)
        d = core.prepare(X[tgt], y[tgt] - 5, "classification", seed=s)
        keep = torch.randperm(len(d["Xtr"]), generator=torch.Generator().manual_seed(s))[:100]
        d["Xtr"], d["ytr"] = d["Xtr"][keep], d["ytr"][keep]  # few-label target task
        methods = {
            "from scratch (no pretraining)": None,
            "full fine-tuning": "full",
            "head only (frozen backbone)": "head",
            "LoRA r=4": ("lora", 4),
            "TT r=4": ("tt", 4),
            "TR r=2": ("tr", 2),
            "Kronecker n=2 r=1 (Compacter)": ("kronecker", 1),
        }
        for name, how in methods.items():
            torch.manual_seed(s)
            m = build_model(cfg, 64, 5) if how is None else copy.deepcopy(base)
            m.head = nn.Linear(128, 5)  # new task, new head (always trainable)
            if how == "head":
                for name_p, p in m.named_parameters():
                    p.requires_grad_(name_p.startswith("head."))
            elif isinstance(how, tuple):
                tl.apply_tensor_lora(m, how[0], rank=how[1], skip=("head",))
            n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
            res.setdefault(name, []).append((n_train, _fit(m, d, s)))
    for name, v in res.items():
        accs = [a for _, a in v]
        print(f"| {name} | {v[0][0]:,} | {np.mean(accs):.4f} ± {np.std(accs):.4f} |")


def part_c():
    print("\n## C. HOSVD of a data tensor (synthetic 48x48x40 cube, Tucker rank (6,6,4) + noise)\n")
    rng = np.random.default_rng(0)
    T = tensor.tucker_to_tensor(
        rng.normal(size=(6, 6, 4)),
        [rng.normal(size=(n, r)) for n, r in ((48, 6), (48, 6), (40, 4))],
    )
    T = T + 0.05 * T.std() * rng.normal(size=T.shape)
    print("| energy kept per mode | ranks | stored fraction | relative error | bound |")
    print("|---|---|---|---|---|")
    for e in (0.9, 0.99, 0.999):
        r = tensor.compress_report(T, e)
        print(f"| {e} | {r['ranks']} | {r['stored_fraction']:.2%} | {r['relative_error']:.4f} | "
              f"{r['error_bound']:.4f} |")  # fmt: skip


if __name__ == "__main__":
    from sklearn.datasets import load_digits

    ds = load_digits()
    X, y = ds.data.astype(np.float32), ds.target
    part_a(X, y)
    part_b(X, y)
    part_c()
