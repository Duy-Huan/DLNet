"""`dlnet gan`: DCGAN-style generative adversarial networks built from the zoo, ranked by MMD^2.

For images X (N, H, W[, C]) scaled to [-1, 1]:
  generator      latent vector -> feature map -> transposed convolutions -> tanh image (zoo ops)
  discriminator  strided convolutions with LeakyReLU -> one logit (zoo ops)
  training       Adam(2e-4, betas 0.5 / 0.999), non-saturating generator loss (Goodfellow et al., 2014;
                 Radford et al., arXiv:1511.06434), the best checkpoint by MMD^2 on the validation images
  metric         unbiased squared maximum mean discrepancy (Gretton et al. 2012) between generated and real
                 images in pixel space, Gaussian kernels with the median-heuristic bandwidth x {0.5, 1, 2}.
                 Lower is better; 0 up to noise means the two samples cannot be told apart by this test.
  baselines      a Gaussian fitted to the training images (full covariance), and the training images
                 themselves (the best a model that only memorises could do), against the same real images.
There is no Inception network here, so no FID: pixel-space MMD is a coarse measure that favours blurry samples.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import jsonsafe

HERE = Path(__file__).parent

TRAIN_PY = '''"""Re-train this GAN. Usage: python train.py --data images.npz [--epochs 60] [--seed 0]
Writes generator.pt. Requires: torch, numpy."""
import argparse
import json

import torch

import gan_data
from model import build_model

cfg = json.load(open("config.yaml"))  # JSON is valid YAML
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--epochs", type=int, default=cfg["recipe"]["epochs"])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
X = gan_data.load(a.data, cfg["data"]["x_key"], cfg["split_seed"])
torch.manual_seed(a.seed)
G, D = build_model(cfg["generator"]), build_model(cfg["discriminator"])
res = gan_data.train_gan(G, D, X["train"], X["val"], cfg["arch"]["zdim"], a.epochs, batch=cfg["arch"]["batch"], seed=a.seed)
print("val MMD^2:", res["best_mmd"])
torch.save(G.state_dict(), "generator.pt")
'''

SAMPLE_PY = '''"""Draw images from generator.pt. Usage: python sample.py [--n 16] -> samples.npy (N, C, H, W) in [-1, 1]."""
import argparse
import json

import numpy as np
import torch

from model import build_model

cfg = json.load(open("config.yaml"))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=16)
a = ap.parse_args()
G = build_model(cfg["generator"])
G.load_state_dict(torch.load("generator.pt"))
G.eval()
with torch.no_grad():
    np.save("samples.npy", G(torch.randn(a.n, cfg["arch"]["zdim"])).numpy())
'''


def mmd2(x: torch.Tensor, y: torch.Tensor) -> float:
    """Unbiased MMD^2 between two samples of flattened images, mixture of three Gaussian kernels."""
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


def generator_spec(c, s, levels, channels, width):
    layers = [{"op": "vec_to_map", "c": c, "s": s}]
    ch = c
    for _ in range(levels):
        ch = max(ch // 2, 8)
        layers.append({"op": "up_deconv", "out": ch})
    layers.append(
        {"op": "conv_bn_act", "out": channels, "k": 3, "s": 1, "act": "tanh", "norm": "none"}
    )
    return {"op": "seq", "body": layers}


def discriminator_spec(width, levels):
    layers = []
    for i in range(levels):
        layers.append(
            {
                "op": "conv_bn_act",
                "out": width * 2**i,
                "k": 3,
                "s": 2,
                "act": "leaky",
                "norm": "none" if i == 0 else "batch",
            }
        )
    return {"op": "seq", "body": [*layers, {"op": "gap", "mode": "avg"}]}


def make_pair(cfg, channels, size):
    """Generator and discriminator configs for an image size; ValueError if the size does not fit."""
    lv = cfg["levels"]
    if size % 2**lv or size // 2**lv < 1:
        raise ValueError(f"{lv} upsampling levels do not reach size {size}")
    g = {
        "spec": generator_spec(cfg["gen_c"], size // 2**lv, lv, channels, cfg["width"]),
        "input": {"kind": "vec", "dims": [cfg["zdim"]]},
        "head": "none",
        "out_dim": 1,
    }
    d = {
        "spec": discriminator_spec(cfg["width"], lv),
        "input": {"kind": "map", "dims": [channels, size, size]},
        "head": "cls",
        "out_dim": 1,
    }
    return g, d


def build_pair(cfg, channels, size):
    """(generator, discriminator) modules; ValueError if the generator does not produce a (channels, size, size) image."""
    from . import zoo
    from .zoo.core import check

    g, d = make_pair(cfg, channels, size)
    info = check(g["spec"], "vec", tuple(g["input"]["dims"]))
    if (info["kind"], info["dims"]) != ("map", (channels, size, size)):
        raise ValueError(
            f"generator output {info['kind']} {info['dims']}, expected a ({channels}, {size}, {size}) image"
        )
    return zoo.build_model(g), zoo.build_model(d)


def train_gan(G, D, Xtr, Xva, zdim, epochs, batch=64, lr=2e-4, seed=0, deadline=None, n_eval=400):
    """Standard GAN training; returns the best (lowest validation MMD^2) generator state and its score."""
    torch.manual_seed(seed)
    opt_g = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    val = Xva[: min(n_eval, len(Xva))]
    best, best_state, curve = float("inf"), None, []
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        G.train()
        D.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr) - batch + 1, batch):
            real = Xtr[perm[i : i + batch]]
            fake = G(torch.randn(batch, zdim))
            loss_d = bce(D(real).squeeze(1), torch.ones(batch)) + bce(
                D(fake.detach()).squeeze(1), torch.zeros(batch)
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            loss_g = bce(D(fake).squeeze(1), torch.ones(batch))  # non-saturating generator loss
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
        G.eval()
        with torch.no_grad():
            score = mmd2(G(torch.randn(len(val), zdim)), val)
        curve.append(score)
        if score < best:
            best, best_state = score, {k: v.clone() for k, v in G.state_dict().items()}
        if deadline is not None and time.time() > deadline:
            break
    return {"best_mmd": best, "state": best_state, "curve": curve}


def _data(path, key, seed):
    from . import array_data

    X = array_data.to_channels_first(array_data.load_x(path, key), "image")
    lo, hi = float(X.min()), float(X.max())
    X = (2 * (X - lo) / max(hi - lo, 1e-9) - 1).astype(np.float32)
    perm = np.random.default_rng(seed).permutation(len(X))
    a, b = int(0.6 * len(X)), int(0.8 * len(X))
    t = torch.from_numpy
    return {"train": t(X[perm[:a]]), "val": t(X[perm[a:b]]), "test": t(X[perm[b:]])}


def run(args) -> int:
    from . import export

    t0 = time.time()
    export.check_out_dir(args.out)
    data = _data(args.data, args.x_key, args.seed)
    n_tr, c, size = len(data["train"]), data["train"].shape[1], data["train"].shape[2]
    print(
        f"[gan] images {tuple(data['train'].shape[1:])}  train/val/test = {n_tr}/{len(data['val'])}/{len(data['test'])}"
    )
    rng = np.random.default_rng(args.seed)
    grid = {
        "width": [16, 32],
        "gen_c": [32, 64],
        "zdim": [16, 64],
        "levels": [1, 2, 3],
        "batch": [32, 64],
    }
    cfgs, seen = [], set()
    for _ in range(40 * args.candidates):
        if len(cfgs) >= args.candidates:
            break
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in grid.items()}
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        try:
            build_pair(cfg, c, size)
            cfgs.append(cfg)
        except ValueError:
            continue
    if not cfgs:
        raise SystemExit(f"no valid generator for image size {size}")
    per_deadline = (args.budget_s - (time.time() - t0)) / len(cfgs)
    rows = []
    for cfg in cfgs:
        vals, tests, best_state, best_val = [], [], None, float("inf")
        for s in range(args.seeds):
            G, D = build_pair(cfg, c, size)
            res = train_gan(
                G,
                D,
                data["train"],
                data["val"],
                cfg["zdim"],
                args.epochs,
                cfg["batch"],
                seed=args.seed + s,
                deadline=time.time() + per_deadline / args.seeds,
            )
            G.load_state_dict(res["state"])
            G.eval()
            with torch.no_grad():
                tests.append(
                    mmd2(
                        G(torch.randn(min(400, len(data["test"])), cfg["zdim"])), data["test"][:400]
                    )
                )
            vals.append(res["best_mmd"])
            if res["best_mmd"] < best_val:
                best_val, best_state = res["best_mmd"], res["state"]
        rows.append({"cfg": cfg, "val_mmd": float(np.mean(vals)), "val_std": float(np.std(vals)), "test_mmd": float(np.mean(tests)), "test_std": float(np.std(tests)),
                     "params": sum(p.numel() for p in G.parameters()) + sum(p.numel() for p in D.parameters()), "_state": best_state, "epochs": args.epochs})  # fmt: skip
        print(
            f"[gan] {cfg}  val MMD^2 {rows[-1]['val_mmd']:.4f} ± {rows[-1]['val_std']:.4f}  test {rows[-1]['test_mmd']:.4f}"
        )

    flat = lambda x: x.flatten(1)
    Xtr, Xte = flat(data["train"]).double(), data["test"][:400]
    mu, cov = Xtr.mean(0), torch.cov(Xtr.T) if Xtr.shape[1] <= 1024 else torch.diag(Xtr.var(0))
    mvn = torch.distributions.MultivariateNormal(
        mu, cov + 1e-4 * torch.eye(len(mu), dtype=torch.double)
    )
    n = len(Xte)
    base = {
        "gaussian_fit": mmd2(mvn.sample((n,)).float().view(n, *Xte.shape[1:]), Xte),
        "training_images": mmd2(
            data["train"][torch.randperm(n_tr, generator=torch.Generator().manual_seed(0))[:n]], Xte
        ),
    }
    rows.sort(key=lambda r: r["val_mmd"])
    out = Path(args.out)
    export._prepare_out_dir(out)
    for i, r in enumerate(rows[: args.topk], 1):
        d = out / f"top{i}_dcgan"
        (d / "tests").mkdir(parents=True)
        shutil.copy(HERE / "zoo_model.py", d / "model.py")
        shutil.copy(HERE / "gan_data.py", d / "gan_data.py")
        shutil.copytree(HERE / "zoo", d / "zoo", ignore=shutil.ignore_patterns("__pycache__"))
        g, dd = make_pair(r["cfg"], c, size)
        torch.save(r["_state"], d / "generator.pt")
        cfg = {"plugin": "gan", "split_seed": args.seed, "arch": r["cfg"], "generator": g, "discriminator": dd, "data": {"x_key": args.x_key, "size": size, "channels": c},
               "recipe": {"epochs": args.epochs}, "search_result": {k: r[k] for k in ("val_mmd", "val_std", "test_mmd", "test_std")}}  # fmt: skip
        (d / "config.yaml").write_text(jsonsafe.dumps(cfg, indent=2))
        (d / "train.py").write_text(TRAIN_PY)
        (d / "sample.py").write_text(SAMPLE_PY)
        (d / "tests" / "test_model.py").write_text(
            "import json\nfrom pathlib import Path\n\nimport torch\n\nfrom model import build_model\n\n\ndef test_generator_shape():\n"
            "    cfg = json.load(open(Path(__file__).parent.parent / 'config.yaml'))\n    g = build_model(cfg['generator']).eval()\n"
            "    z = cfg['arch']['zdim']\n    assert g(torch.randn(2, z)).shape == (2, cfg['data']['channels'], cfg['data']['size'], cfg['data']['size'])\n"
        )  # fmt: skip
    for r in rows:
        del r["_state"]
    best = rows[0]
    wins = best["test_mmd"] + best["test_std"] < base["gaussian_fit"]
    board = {
        "task": "gan",
        "metric": "MMD^2 in pixel space (lower is better)",
        "top_k": rows,
        "baselines_test_mmd": base,
        "verdict": {"beats_gaussian_fit": bool(wins)},
    }
    (out / "leaderboard.json").write_text(jsonsafe.dumps(board, indent=2))
    md = [
        "# Leaderboard (GAN, MMD^2 in pixel space, lower is better)\n",
        "| rank | config | params | val MMD^2 | test MMD^2 |",
        "|---|---|---|---|---|",
    ]
    md += [
        f"| {i} | {r['cfg']} | {r['params']:,} | {r['val_mmd']:.4f} ± {r['val_std']:.4f} | {r['test_mmd']:.4f} ± {r['test_std']:.4f} |"
        for i, r in enumerate(rows, 1)
    ]
    md += [
        "\n## Baselines (test MMD^2 against the same real images)\n",
        "| baseline | MMD^2 |",
        "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in base.items()],
    ]
    md.append(
        "\nPixel-space MMD^2 only; no FID. 'training_images' is what memorising the training set would score."
    )
    (out / "leaderboard.md").write_text("\n".join(md) + "\n")
    print(
        f"[done] {time.time() - t0:.0f}s  -> {args.out}/leaderboard.md   beats gaussian_fit: {wins}"
    )
    return 0


def add_parser(sub) -> None:
    g = sub.add_parser(
        "gan", help="train and rank DCGAN-style generators on an image set (MMD^2 in pixel space)"
    )
    g.add_argument("--data", required=True, help=".npz with X (N, H, W[, C])")
    g.add_argument("--x-key", default=None)
    g.add_argument("--candidates", type=int, default=4)
    g.add_argument("--epochs", type=int, default=40)
    g.add_argument("--seeds", type=int, default=2)
    g.add_argument("--budget-s", type=float, default=300.0, help="total seconds for all candidates")
    g.add_argument("--topk", type=int, default=2)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="dlnet_out")
