"""HSI model library (torch only). Copied into every exported HSI package as model.py.

All models take a patch batch [n, C, p, p] (C = spectral components after the stem) and return logits.

Families:
  cnn1d        spectral-only 1D CNN on the centre pixel
  cnn2d        2D CNN over the patch
  tucker_cnn2d 2D CNN whose 3x3 convs are Tucker-2 factorised (1x1 -> 3x3 -> 1x1), see ref/04_tucker-*
  cnn3d        3D CNN over (spectral, height, width)
  hybridsn     HybridSN-style: 3D convs -> reshape -> 2D conv -> FC
               (re-implemented from memory of Roy et al. 2019; not verified against the original code)

Typed registry + grammar (agent.md sec. 13): every encoder block is registered with a kind
(spectral / spatial / joint), maps a patch [n, C, p, p] to a feature vector [n, D], and can tell
D (or reject the input) WITHOUT building the model (`features`). Models are composed as
    Model := ( Spectral || Spatial | Spectral | Spatial | Joint ) -> Fusion -> Head
Preset families are configurations of the registry, not separate code:
  resnet2d, dwsep2d, convnext2d, spec_transformer, ssrn, and `composed` (sampled from the grammar).
None of the named blocks was checked against the authors' code; each docstring says so.
"""

from __future__ import annotations

import torch
from torch import nn

PRESETS = {
    "resnet2d": {"spatial": "res"},
    "dwsep2d": {"spatial": "dwsep"},
    "convnext2d": {"spatial": "convnext"},
    "spec_transformer": {"spectral": "transformer"},
    "ssrn": {"joint": "ssrn"},
}
FAMILIES = ("cnn1d", "cnn2d", "tucker_cnn2d", "cnn3d", "hybridsn", *PRESETS, "composed")


def _cbr2d(i, o, k=3):
    return [
        nn.Conv2d(i, o, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    ]


class CNN1D(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, patch):
        super().__init__()
        self.r = patch // 2
        layers, c = [], 1
        for i in range(depth):
            w = width * min(2**i, 4)
            layers += [
                nn.Conv1d(c, w, 5, padding=2, bias=False),
                nn.BatchNorm1d(w),
                nn.ReLU(inplace=True),
            ]
            if i < depth - 1:
                layers.append(nn.MaxPool1d(2, ceil_mode=True))
            c = w
        self.body = nn.Sequential(
            *layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(c, out_dim)

    def forward(self, x):
        return self.head(self.body(x[:, :, self.r, self.r].unsqueeze(1)))


class CNN2D(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, patch):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(depth):
            layers += _cbr2d(c, width)
            c = width
        self.body = nn.Sequential(
            *layers, nn.AdaptiveAvgPool2d(3), nn.Flatten(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(c * 9, out_dim)

    def forward(self, x):
        return self.head(self.body(x))


class TuckerConv(nn.Module):
    """3x3 conv with Tucker-2 kernel: 1x1 (in->r1), 3x3 (r1->r2), 1x1 (r2->out)."""

    def __init__(self, c_in, c_out, ratio):
        super().__init__()
        r1, r2 = max(4, int(c_in * ratio)), max(4, int(c_out * ratio))
        self.net = nn.Sequential(
            nn.Conv2d(c_in, r1, 1, bias=False),
            nn.Conv2d(r1, r2, 3, padding=1, bias=False),
            nn.Conv2d(r2, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TuckerCNN2D(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, patch, ratio):
        super().__init__()
        layers = [*_cbr2d(in_ch, width)]
        layers += [TuckerConv(width, width, ratio) for _ in range(depth - 1)]
        self.body = nn.Sequential(
            *layers, nn.AdaptiveAvgPool2d(3), nn.Flatten(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(width * 9, out_dim)

    def forward(self, x):
        return self.head(self.body(x))


class CNN3D(nn.Module):
    def __init__(self, in_ch, out_dim, width, depth, dropout, patch):
        super().__init__()
        layers, c = [], 1
        for i in range(depth):
            layers += [
                nn.Conv3d(c, width, 3, padding=1, bias=False),
                nn.BatchNorm3d(width),
                nn.ReLU(inplace=True),
            ]
            if i < depth - 1:
                layers.append(nn.MaxPool3d((2, 1, 1), ceil_mode=True))
            c = width
        self.body = nn.Sequential(
            *layers, nn.AdaptiveAvgPool3d((2, 3, 3)), nn.Flatten(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(width * 18, out_dim)

    def forward(self, x):
        return self.head(self.body(x.unsqueeze(1)))


class HybridSN(nn.Module):
    def __init__(self, in_ch, out_dim, width, dropout, patch):
        super().__init__()
        c1, c2, c3 = width, 2 * width, 4 * width
        self.c3d = nn.Sequential(
            nn.Conv3d(1, c1, (7, 3, 3)), nn.BatchNorm3d(c1), nn.ReLU(inplace=True),
            nn.Conv3d(c1, c2, (5, 3, 3)), nn.BatchNorm3d(c2), nn.ReLU(inplace=True),
            nn.Conv3d(c2, c3, (3, 3, 3)), nn.BatchNorm3d(c3), nn.ReLU(inplace=True),
        )  # fmt: skip
        with torch.no_grad():
            z = self.c3d(
                torch.zeros(2, 1, in_ch, patch, patch)
            )  # batch of 2: BatchNorm needs >1 value
        _, c, k, h, w = z.shape
        self.c2d = nn.Sequential(
            nn.Conv2d(c * k, 64, 3, padding=1 if min(h, w) < 3 else 0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat = self.c2d(z.reshape(2, c * k, h, w)).numel() // 2
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, out_dim),
        )  # fmt: skip

    def forward(self, x):
        z = self.c3d(x.unsqueeze(1))
        n, c, k, h, w = z.shape
        return self.fc(self.c2d(z.reshape(n, c * k, h, w)))


# ------------------------------------------------------------- typed registry + grammar

BLOCKS: dict[tuple[str, str], type] = {}
KINDS = ("spectral", "spatial", "joint")


def register(kind: str, name: str):
    """Register an encoder block. Contract: forward([n, C, p, p]) -> [n, features(...)]."""

    def deco(cls):
        cls.kind, cls.block_name = kind, name
        BLOCKS[(kind, name)] = cls
        return cls

    return deco


class Block(nn.Module):
    """Base encoder block. `features` is the static shape inference used to reject bad configs."""

    in_spec, out_spec = ("B", "C", "P", "P"), ("B", "D")

    @staticmethod
    def features(in_ch: int, width: int, depth: int, patch: int) -> int:
        return width


@register("spectral", "cnn1d")
class SpecCNN1D(Block):
    """1D CNN along the spectrum of the centre pixel."""

    def __init__(self, in_ch, width, depth, dropout, patch):
        super().__init__()
        self.net = CNN1D(in_ch, 1, width, depth, 0.0, patch)
        self.net.head = nn.Identity()

    @staticmethod
    def features(in_ch, width, depth, patch):
        return width * min(2 ** (depth - 1), 4)

    def forward(self, x):
        return self.net(x)


@register("spectral", "transformer")
class SpecTransformer(Block):
    """Transformer over band tokens of the centre pixel (one token per spectral component).

    The idea of treating bands as a token sequence follows SpectralFormer (Hong et al. 2021);
    this is a plain pre-norm encoder written from memory, NOT a re-implementation of that model.
    """

    def __init__(self, in_ch, width, depth, dropout, patch):
        super().__init__()
        self.features(in_ch, width, depth, patch)
        self.r = patch // 2
        self.embed = nn.Linear(1, width)
        self.pos = nn.Parameter(torch.zeros(1, in_ch + 1, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width, 4, width * 2, dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    @staticmethod
    def features(in_ch, width, depth, patch):
        if width % 4:
            raise ValueError("transformer width must be divisible by the 4 heads")
        return width

    def forward(self, x):
        t = self.embed(x[:, :, self.r, self.r].unsqueeze(-1))
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.norm(self.enc(t)[:, 0])


class _Spatial(Block):
    """Stem conv -> `depth` blocks from `_block` -> global average pool."""

    def __init__(self, in_ch, width, depth, dropout, patch):
        super().__init__()
        self.stem = nn.Sequential(*_cbr2d(in_ch, width))
        self.blocks = nn.ModuleList([self._block(width, patch) for _ in range(depth)])
        self.residual = False

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = x + b(x) if self.residual else b(x)
        return x.mean(dim=(2, 3))


@register("spatial", "conv")
class SpatConv(_Spatial):
    """Plain conv-BN-ReLU stack."""

    def _block(self, w, patch):
        return nn.Sequential(*_cbr2d(w, w))


@register("spatial", "res")
class SpatRes(_Spatial):
    """ResNet basic block (He et al. 2016): two 3x3 conv-BN with an identity skip.

    Written from memory; the ReLU after the addition is replaced by a ReLU inside the branch.
    """

    def __init__(self, *a):
        super().__init__(*a)
        self.residual = True

    def _block(self, w, patch):
        return nn.Sequential(
            *_cbr2d(w, w), nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w)
        )


@register("spatial", "dwsep")
class SpatDWSep(_Spatial):
    """Depthwise-separable conv (MobileNet-style): 3x3 depthwise then 1x1 pointwise."""

    def _block(self, w, patch):
        return nn.Sequential(
            nn.Conv2d(w, w, 3, padding=1, groups=w, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            *_cbr2d(w, w, 1),
        )


@register("spatial", "convnext")
class SpatConvNeXt(_Spatial):
    """ConvNeXt-style block (Liu et al. 2022): depthwise conv -> norm -> 1x1 (x4) -> GELU -> 1x1, skip.

    Written from memory, not verified: no layer scale / stochastic depth, GroupNorm(1) stands in
    for channel-last LayerNorm, and the kernel is 3 (not 7) for patches smaller than 7.
    """

    def __init__(self, *a):
        super().__init__(*a)
        self.residual = True

    def _block(self, w, patch):
        k = 7 if patch >= 7 else 3
        return nn.Sequential(
            nn.Conv2d(w, w, k, padding=k // 2, groups=w),
            nn.GroupNorm(1, w),
            nn.Conv2d(w, 4 * w, 1),
            nn.GELU(),
            nn.Conv2d(4 * w, w, 1),
        )


@register("spatial", "tucker")
class SpatTucker(_Spatial):
    """Tucker-2 factorised 3x3 convs (rank ratio 0.5), see ref/04_tucker-*."""

    def _block(self, w, patch):
        return TuckerConv(w, w, 0.5)


@register("joint", "conv3d")
class JointConv3D(Block):
    """3D convs over (spectral, height, width)."""

    def __init__(self, in_ch, width, depth, dropout, patch):
        super().__init__()
        self.net = CNN3D(in_ch, 1, width, depth, 0.0, patch)
        self.net.head = nn.Identity()

    @staticmethod
    def features(in_ch, width, depth, patch):
        return width * 18

    def forward(self, x):
        return self.net(x)


class _Res3d(nn.Module):
    def __init__(self, w, k):
        super().__init__()
        pad = tuple(i // 2 for i in k)
        self.net = nn.Sequential(
            nn.Conv3d(w, w, k, padding=pad, bias=False), nn.BatchNorm3d(w), nn.ReLU(inplace=True),
            nn.Conv3d(w, w, k, padding=pad, bias=False), nn.BatchNorm3d(w),
        )  # fmt: skip

    def forward(self, x):
        return torch.relu(x + self.net(x))


@register("joint", "ssrn")
class JointSSRN(Block):
    """SSRN-style (Zhong et al. 2018): spectral residual blocks (7x1x1 3D convs), a conv that
    collapses the spectral axis, then spatial residual blocks (1x3x3), global average pool.

    Re-implemented from memory; NOT verified against the authors' code (layer counts and widths
    are search hyperparameters here, not the paper's).
    """

    def __init__(self, in_ch, width, depth, dropout, patch):
        super().__init__()
        self.features(in_ch, width, depth, patch)
        self.stem = nn.Sequential(
            nn.Conv3d(1, width, (7, 1, 1), stride=(2, 1, 1), bias=False),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True),
        )
        self.spec = nn.Sequential(*[_Res3d(width, (7, 1, 1)) for _ in range(depth)])
        left = (in_ch - 7) // 2 + 1
        self.collapse = nn.Sequential(
            nn.Conv3d(width, width, (left, 1, 1), bias=False),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True),
        )
        self.spat = nn.Sequential(*[_Res3d(width, (1, 3, 3)) for _ in range(depth)])

    @staticmethod
    def features(in_ch, width, depth, patch):
        if in_ch < 7:
            raise ValueError("ssrn needs at least 7 spectral components")
        return width

    def forward(self, x):
        z = self.spat(self.collapse(self.spec(self.stem(x.unsqueeze(1)))))
        return z.mean(dim=(2, 3, 4))


class Gated(nn.Module):
    """Gated fusion of two vectors: g * A(a) + (1 - g) * B(b), g = sigmoid(W [a, b])."""

    def __init__(self, da, db, d):
        super().__init__()
        self.a, self.b, self.g = nn.Linear(da, d), nn.Linear(db, d), nn.Linear(da + db, d)

    def forward(self, a, b):
        g = torch.sigmoid(self.g(torch.cat([a, b], 1)))
        return g * self.a(a) + (1 - g) * self.b(b)


FUSIONS = ("concat", "gated")


def plan_composed(cfg: dict, in_ch: int) -> dict:
    """Static check of a grammar configuration; raises ValueError before any model is built.

    Returns {"branches": [(kind, name, features), ...], "fusion": name | None, "features": D}.
    """
    names = {k: cfg.get(k) or "none" for k in KINDS}
    fusion = cfg.get("fusion") or "none"
    used = [k for k in KINDS if names[k] != "none"]
    for k in used:
        if (k, names[k]) not in BLOCKS:
            raise ValueError(f"unknown {k} block {names[k]!r}")
    if not used:
        raise ValueError("a composed model needs at least one encoder")
    if "joint" in used and len(used) > 1:
        raise ValueError("a joint encoder cannot be combined with spectral/spatial branches")
    if (len(used) == 2) != (fusion != "none"):
        raise ValueError("fusion is required for exactly two branches and forbidden otherwise")
    if fusion != "none" and fusion not in FUSIONS:
        raise ValueError(f"unknown fusion {fusion!r}")
    w, d, patch = cfg["width"], cfg.get("depth", 2), cfg["patch"]
    br = [(k, names[k], BLOCKS[(k, names[k])].features(in_ch, w, d, patch)) for k in used]
    feats = br[0][2] if len(br) == 1 else (br[0][2] + br[1][2] if fusion == "concat" else w)
    return {"branches": br, "fusion": None if fusion == "none" else fusion, "features": feats}


def sample_composed(rng) -> dict:
    """Draw the block choice of one model from the grammar (always type-correct)."""
    spec = sorted(n for k, n in BLOCKS if k == "spectral")
    spat = sorted(n for k, n in BLOCKS if k == "spatial")
    return {
        "spectral": spec[int(rng.integers(len(spec)))],
        "spatial": spat[int(rng.integers(len(spat)))],
        "joint": "none",
        "fusion": FUSIONS[int(rng.integers(len(FUSIONS)))],
    }


class Composed(nn.Module):
    def __init__(self, cfg, in_ch, out_dim):
        super().__init__()
        plan = plan_composed(cfg, in_ch)
        w, d, p, patch = cfg["width"], cfg.get("depth", 2), cfg["dropout"], cfg["patch"]
        self.branches = nn.ModuleList(
            [BLOCKS[(k, n)](in_ch, w, d, p, patch) for k, n, _ in plan["branches"]]
        )
        f = [b[2] for b in plan["branches"]]
        self.gate = Gated(f[0], f[1], w) if plan["fusion"] == "gated" else None
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(plan["features"], out_dim)

    def forward(self, x):
        z = [b(x) for b in self.branches]
        z = z[0] if len(z) == 1 else (self.gate(*z) if self.gate else torch.cat(z, 1))
        return self.head(self.drop(z))


def build_model(cfg: dict, in_dim: int, out_dim: int) -> nn.Module:
    if cfg.get("spec"):  # an architecture of the module zoo (dlnet/zoo)
        try:
            from . import zoo
        except ImportError:  # exported package: zoo/ sits next to model.py
            import zoo
        return zoo.build_model(cfg, in_dim, out_dim)
    f, w, d, p, patch = (
        cfg["family"],
        cfg["width"],
        cfg.get("depth", 3),
        cfg["dropout"],
        cfg["patch"],
    )
    if f == "cnn1d":
        return CNN1D(in_dim, out_dim, w, d, p, patch)
    if f == "cnn2d":
        return CNN2D(in_dim, out_dim, w, d, p, patch)
    if f == "tucker_cnn2d":
        return TuckerCNN2D(in_dim, out_dim, w, d, p, patch, cfg["rank_ratio"])
    if f == "cnn3d":
        return CNN3D(in_dim, out_dim, w, d, p, patch)
    if f == "hybridsn":
        return HybridSN(in_dim, out_dim, w, p, patch)
    if f in PRESETS:
        return Composed({**cfg, **PRESETS[f]}, in_dim, out_dim)
    if f == "composed":
        return Composed(cfg, in_dim, out_dim)
    raise ValueError(f"unknown family {f}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
