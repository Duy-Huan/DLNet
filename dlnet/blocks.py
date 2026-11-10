"""Block library (torch only): backbones, decoders, attention, routing and interaction blocks.

Copied into exported packages as blocks.py. Every block is listed in CATALOG with the project it
was taken from; see THIRD_PARTY.md for licences and pinned commits. Nothing here is a verified
reproduction of a published model: blocks are sized down for small data and say what differs.

Sources
  rasbt/deeplearning-models (MIT)        LeNet-5, all-convolutional net, VGG, ResNet basic block,
                                          DenseNet layer / transition, deconvolution decoder.
                                          Adapted: widths and number of stages follow the input size.
  prescient-design/cortex (Apache-2.0)   root -> trunk -> branch -> leaf composition (NeuralTree,
                                          SumTrunk), mean pooling. Re-written without hydra.
  ylongqi/openrec (Apache-2.0)           latent factors, second-order feature interaction,
                                          pairwise log loss. Ported from TensorFlow to PyTorch.
  yamsgithub/modular_deep_learning       mixture of experts with soft and top-k gating. That
                                          repository has NO licence, so no code was copied: the
                                          blocks are re-implemented from the method.
Written from memory of the papers, not verified against the authors' code: MobileNetV2 inverted
residual (Sandler et al., arXiv:1801.04381), squeeze-and-excitation (Hu et al., arXiv:1709.01507),
FCN skip head (Long et al., arXiv:1411.4038), U-Net decoder (Ronneberger et al., arXiv:1505.04597),
product of experts (Hinton 2002).
"""

from __future__ import annotations

import math

import torch
from torch import nn

CATALOG: list[dict] = []
_RASBT, _CORTEX = "rasbt/deeplearning-models (MIT)", "prescient-design/cortex (Apache-2.0)"
_OPENREC = "ylongqi/openrec (Apache-2.0)"
_MOE = "yamsgithub/modular_deep_learning (no licence: re-implemented, no code copied)"
_PAPER = "paper, written from memory (not verified)"


def catalog(kind: str, name: str, source: str):
    """Class decorator: register a block in CATALOG."""

    def deco(obj):
        CATALOG.append({"name": name, "kind": kind, "source": source, "object": obj.__name__})
        return obj

    return deco


# ------------------------------------------------------------------ attention


@catalog("attention", "se", _PAPER)
class SqueezeExcite(nn.Module):
    """Channel attention: x * sigmoid(MLP(global average pool(x)))."""

    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.f = nn.Sequential(
            nn.Linear(ch, hidden), nn.ReLU(), nn.Linear(hidden, ch), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.f(x.mean((2, 3)))[:, :, None, None]


@catalog("attention", "self_attention_2d", "torch.nn.MultiheadAttention")
class SelfAttention2d(nn.Module):
    """Multi-head self-attention over the positions of a feature map, with a residual."""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.norm(x.flatten(2).transpose(1, 2))
        return x + self.attn(t, t, t, need_weights=False)[0].transpose(1, 2).reshape(b, c, h, w)


def make_attention(kind: str, ch: int) -> nn.Module:
    if kind in ("none", None):
        return nn.Identity()
    if kind == "se":
        return SqueezeExcite(ch)
    if kind == "self":
        return SelfAttention2d(ch, heads=4 if ch % 4 == 0 else 1)
    raise ValueError(f"unknown attention {kind!r}")


# ------------------------------------------------------------ backbone stages


def conv_bn_relu(i: int, o: int, k: int = 3, stride: int = 1) -> list[nn.Module]:
    return [nn.Conv2d(i, o, k, stride, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU()]


@catalog("stage", "lenet", _RASBT)
class LeNetStage(nn.Sequential):
    """LeNet-5 feature stage: 5x5 convolution, tanh, 2x2 pooling (padding added to keep sizes)."""

    def __init__(self, i, o, depth, stride):
        layers = [nn.MaxPool2d(2, ceil_mode=True)] if stride == 2 else []
        for d in range(depth):
            layers += [nn.Conv2d(i if d == 0 else o, o, 5, padding=2), nn.Tanh()]
        super().__init__(*layers)
        self.out_ch = o


@catalog("stage", "allconv", _RASBT)
class AllConvStage(nn.Sequential):
    """All-convolutional stage: a stride-2 convolution replaces pooling (BatchNorm added)."""

    def __init__(self, i, o, depth, stride):
        layers = conv_bn_relu(i, o, 3, stride)
        for _ in range(depth - 1):
            layers += conv_bn_relu(o, o)
        super().__init__(*layers)
        self.out_ch = o


@catalog("stage", "vgg", _RASBT)
class VGGStage(nn.Sequential):
    """VGG stage: max pooling, then `depth` 3x3 conv-BN-ReLU layers (the BatchNorm variant)."""

    def __init__(self, i, o, depth, stride):
        layers = [nn.MaxPool2d(2, ceil_mode=True)] if stride == 2 else []
        for d in range(depth):
            layers += conv_bn_relu(i if d == 0 else o, o)
        super().__init__(*layers)
        self.out_ch = o


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convolutions and an identity / 1x1-projection shortcut."""

    def __init__(self, i, o, stride):
        super().__init__()
        self.f = nn.Sequential(
            *conv_bn_relu(i, o, 3, stride), nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o)
        )
        self.short = None
        if stride != 1 or i != o:
            self.short = nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.f(x) + (x if self.short is None else self.short(x)))


@catalog("stage", "resnet", _RASBT)
class ResNetStage(nn.Sequential):
    def __init__(self, i, o, depth, stride):
        super().__init__(
            *[BasicBlock(i if d == 0 else o, o, stride if d == 0 else 1) for d in range(depth)]
        )
        self.out_ch = o


class DenseLayer(nn.Module):
    """DenseNet-BC layer: BN-ReLU-1x1 conv (bottleneck) then BN-ReLU-3x3 conv; output is concatenated."""

    def __init__(self, i, growth, bn_size=2):
        super().__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(i), nn.ReLU(), nn.Conv2d(i, bn_size * growth, 1, bias=False),
            nn.BatchNorm2d(bn_size * growth), nn.ReLU(),
            nn.Conv2d(bn_size * growth, growth, 3, padding=1, bias=False),
        )  # fmt: skip

    def forward(self, x):
        return torch.cat([x, self.f(x)], 1)


@catalog("stage", "densenet", _RASBT)
class DenseNetStage(nn.Sequential):
    """Transition (BN-ReLU-1x1 conv, average pooling when stride 2) followed by a dense block."""

    def __init__(self, i, o, depth, stride):
        c0 = max(o // 2, 4)
        growth = max((o - c0) // depth, 2)
        layers = [nn.BatchNorm2d(i), nn.ReLU(), nn.Conv2d(i, c0, 1, bias=False)]
        if stride == 2:
            layers.append(nn.AvgPool2d(2, ceil_mode=True))
        layers += [DenseLayer(c0 + d * growth, growth) for d in range(depth)]
        super().__init__(*layers)
        self.out_ch = c0 + depth * growth


class InvertedResidual(nn.Module):
    """MobileNetV2 block: 1x1 expand, 3x3 depthwise, 1x1 linear projection; residual if same shape."""

    def __init__(self, i, o, stride, expand=4):
        super().__init__()
        h = i * expand
        self.f = nn.Sequential(
            nn.Conv2d(i, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU6(),
            nn.Conv2d(h, h, 3, stride, 1, groups=h, bias=False), nn.BatchNorm2d(h), nn.ReLU6(),
            nn.Conv2d(h, o, 1, bias=False), nn.BatchNorm2d(o),
        )  # fmt: skip
        self.res = stride == 1 and i == o

    def forward(self, x):
        return x + self.f(x) if self.res else self.f(x)


@catalog("stage", "mobilenetv2", _PAPER)
class MobileNetV2Stage(nn.Sequential):
    def __init__(self, i, o, depth, stride):
        super().__init__(
            *[
                InvertedResidual(i if d == 0 else o, o, stride if d == 0 else 1)
                for d in range(depth)
            ]
        )
        self.out_ch = o


STAGES = {
    "lenet": LeNetStage, "allconv": AllConvStage, "vgg": VGGStage, "resnet": ResNetStage,
    "densenet": DenseNetStage, "mobilenetv2": MobileNetV2Stage,
}  # fmt: skip


def n_stages(size: int, cap: int = 4) -> int:
    """Number of resolution levels so that the deepest map is at least 2 pixels wide."""
    return max(1, min(cap, int(math.log2(max(size, 2)))))


@catalog("backbone", "encoder", "DLNet (stages from the sources above)")
class Encoder(nn.Module):
    """Stem + stages; level s has stride 2^s. forward returns the feature map of every level."""

    def __init__(self, in_ch, width, depth, size, kind="resnet", attention="none", max_stages=3):
        super().__init__()
        if kind not in STAGES:
            raise ValueError(f"unknown backbone {kind!r}; use one of {sorted(STAGES)}")
        self.stem = nn.Sequential(*conv_bn_relu(in_ch, width))
        self.stages, self.attn, self.channels = nn.ModuleList(), nn.ModuleList(), []
        c = width
        for s in range(n_stages(size, max_stages)):
            st = STAGES[kind](c, width * 2**s, depth, 2 if s > 0 else 1)
            c = st.out_ch
            self.stages.append(st)
            self.attn.append(make_attention(attention, c))
            self.channels.append(c)

    def forward(self, x):
        feats, x = [], self.stem(x)
        for st, at in zip(self.stages, self.attn, strict=True):
            x = at(st(x))
            feats.append(x)
        return feats


# ----------------------------------------------------------------- heads / decoders


@catalog("head", "classifier", _RASBT)
class Classifier(nn.Module):
    """Encoder + global average pooling + dropout + linear layer."""

    def __init__(self, encoder: Encoder, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.channels[-1], out_dim))

    def forward(self, x):
        return self.head(self.encoder(x)[-1].mean((2, 3)))


def _resize(x, ref):
    return nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


@catalog("decoder", "fcn", _PAPER)
class FCNDecoder(nn.Module):
    """FCN with skips: a 1x1 score layer per level, upsampled and summed from deep to shallow."""

    def __init__(self, channels, out_dim):
        super().__init__()
        self.score = nn.ModuleList(nn.Conv2d(c, out_dim, 1) for c in channels)

    def forward(self, feats):
        y = self.score[-1](feats[-1])
        for f, s in zip(feats[-2::-1], list(self.score)[-2::-1], strict=True):
            y = _resize(y, f) + s(f)
        return y


@catalog("decoder", "unet", _PAPER)
class UNetDecoder(nn.Module):
    """U-Net expanding path: upsample, concatenate the skip, two 3x3 conv-BN-ReLU; 1x1 output."""

    def __init__(self, channels, out_dim):
        super().__init__()
        self.up = nn.ModuleList()
        c = channels[-1]
        for skip in channels[-2::-1]:
            self.up.append(nn.Sequential(*conv_bn_relu(c + skip, skip), *conv_bn_relu(skip, skip)))
            c = skip
        self.out = nn.Conv2d(c, out_dim, 1)

    def forward(self, feats):
        y = feats[-1]
        for f, block in zip(feats[-2::-1], self.up, strict=True):
            y = block(torch.cat([_resize(y, f), f], 1))
        return self.out(y)


@catalog("decoder", "deconv", _RASBT)
class DeconvDecoder(nn.Module):
    """Transposed-convolution decoder without skips (the decoder of a convolutional autoencoder)."""

    def __init__(self, channels, out_dim):
        super().__init__()
        layers, c = [], channels[-1]
        for nxt in channels[-2::-1]:
            layers += [nn.ConvTranspose2d(c, nxt, 2, 2), nn.BatchNorm2d(nxt), nn.LeakyReLU()]
            c = nxt
        self.f = nn.Sequential(*layers)
        self.out = nn.Conv2d(c, out_dim, 1)

    def forward(self, feats):
        return self.out(_resize(self.f(feats[-1]), feats[0]))


DECODERS = {"fcn": FCNDecoder, "unet": UNetDecoder, "deconv": DeconvDecoder}


@catalog("head", "segmenter", "DLNet")
class Segmenter(nn.Module):
    """Encoder + decoder; logits (N, classes, H, W) at the input resolution."""

    def __init__(self, encoder: Encoder, decoder: str, out_dim: int, dropout: float = 0.0):
        super().__init__()
        if decoder not in DECODERS:
            raise ValueError(f"unknown decoder {decoder!r}; use one of {sorted(DECODERS)}")
        self.encoder, self.drop = encoder, nn.Dropout2d(dropout)
        self.decoder = DECODERS[decoder](encoder.channels, out_dim)

    def forward(self, x):
        feats = self.encoder(x)
        feats[-1] = self.drop(feats[-1])
        return _resize(self.decoder(feats), x)


# ------------------------------------------------------------- routing: experts


@catalog("routing", "mixture_of_experts", _MOE)
class MixtureOfExperts(nn.Module):
    """Experts combined by a gate (paper 2302.11529: learned routing + aggregation of functions).

    mode "soft":    sum_e p_e(x) y_e(x), p = softmax(gate(x))            (expectation model)
    mode "topk":    the same with every expert outside the top-k gate logits masked to -inf
                    before the softmax (hard routing for k = 1)
    mode "product": product of experts for classification: sum_e log_softmax(y_e(x)), i.e. the
                    experts' class distributions are multiplied; the gate is not used
    Expert outputs are mixed as logits (the source mixes them and then applies a softmax).
    The gate probabilities of the last forward pass are kept in `gate_probs`.
    """

    def __init__(self, experts, gate: nn.Module | None = None, mode: str = "soft", k: int = 1):
        super().__init__()
        if mode not in ("soft", "topk", "product"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode != "product" and gate is None:
            raise ValueError("soft and topk routing need a gate")
        self.experts, self.gate, self.mode, self.k = nn.ModuleList(experts), gate, mode, k
        self.gate_probs = None

    def forward(self, x):
        y = torch.stack([e(x) for e in self.experts], 1)  # (B, E, out)
        if self.mode == "product":
            return torch.log_softmax(y, -1).sum(1)
        logits = self.gate(x)
        if self.mode == "topk" and self.k < len(self.experts):
            keep = torch.topk(logits, self.k, dim=1).indices
            mask = torch.ones_like(logits, dtype=torch.bool).scatter(1, keep, False)
            logits = logits.masked_fill(mask, float("-inf"))
        p = torch.softmax(logits, 1)
        self.gate_probs = p.detach()
        return (p.unsqueeze(-1) * y).sum(1)


# ---------------------------------------------------- interaction (recommenders)


@catalog("interaction", "latent_factor", _OPENREC)
class LatentFactor(nn.Embedding):
    """Embedding table for ids (users, items, categories), small normal initialisation."""

    def __init__(self, num: int, dim: int, std: float = 0.01):
        super().__init__(num, dim)
        nn.init.normal_(self.weight, std=std)


@catalog("interaction", "second_order", _OPENREC)
class SecondOrderInteraction(nn.Module):
    """Dot products of every pair of feature vectors: (B, F, D) -> (B, F (F - 1) / 2).

    With self_interaction=True the F squared norms are included: F (F + 1) / 2 outputs.
    """

    def __init__(self, self_interaction: bool = False):
        super().__init__()
        self.self_interaction = self_interaction

    def forward(self, feats):
        g = feats @ feats.transpose(1, 2)
        n = feats.shape[1]
        i, j = torch.triu_indices(n, n, offset=0 if self.self_interaction else 1, device=g.device)
        return g[:, i, j]


@catalog("loss", "pairwise_log_loss", _OPENREC)
def pairwise_log_loss(user, pos_item, neg_item, pos_bias=None, neg_bias=None):
    """BPR-style loss: -mean log sigmoid(max(u.p - u.n, -30))."""
    diff = (user * pos_item).sum(1, keepdim=True) - (user * neg_item).sum(1, keepdim=True)
    if pos_bias is not None:
        diff = diff + pos_bias
    if neg_bias is not None:
        diff = diff - neg_bias
    return -nn.functional.logsigmoid(diff.clamp_min(-30.0)).mean()


# -------------------------------------------------- composition (cortex pattern)


@catalog("aggregation", "mean_pooling", _CORTEX)
class MeanPooling(nn.Module):
    """Mean over the sequence axis of (B, T, D), ignoring positions where mask is False."""

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(1)
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


@catalog("aggregation", "sum_trunk", _CORTEX)
class SumTrunk(nn.Module):
    """Project every root output to one width and sum them."""

    def __init__(self, in_dims: list[int], out_dim: int):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(d, out_dim) for d in in_dims)

    def forward(self, *roots):
        return sum(p(r) for p, r in zip(self.proj, roots, strict=True))


@catalog("composition", "neural_tree", _CORTEX)
class NeuralTree(nn.Module):
    """Multi-input, multi-task composition: roots (one per input) -> trunk -> branches -> leaves.

    `leaf_branch` maps a leaf name to the branch it reads. forward takes {root name: input} and
    returns {leaf name: output}; a branch is evaluated once even if several leaves share it.
    """

    def __init__(
        self, roots: dict, trunk: nn.Module, branches: dict, leaves: dict, leaf_branch: dict
    ):
        super().__init__()
        missing = {b for b in leaf_branch.values() if b not in branches}
        if missing or set(leaf_branch) != set(leaves):
            raise ValueError(f"leaf_branch must name every leaf and existing branches ({missing})")
        self.roots, self.trunk = nn.ModuleDict(roots), trunk
        self.branches, self.leaves = nn.ModuleDict(branches), nn.ModuleDict(leaves)
        self.leaf_branch = dict(leaf_branch)

    def forward(self, inputs: dict, leaf_keys=None) -> dict:
        unknown = set(inputs) - set(self.roots)
        if unknown:
            raise KeyError(f"root key(s) {sorted(unknown)} not found in root nodes")
        trunk = self.trunk(*[self.roots[k](v) for k, v in inputs.items()])
        cache, out = {}, {}
        for leaf in leaf_keys or self.leaves.keys():
            b = self.leaf_branch[leaf]
            if b not in cache:
                cache[b] = self.branches[b](trunk)
            out[leaf] = self.leaves[leaf](cache[b])
        return out


def describe() -> str:
    """Markdown table of every block in the library."""
    rows = ["| kind | block | class | source |", "|---|---|---|---|"]
    rows += [
        f"| {c['kind']} | {c['name']} | `{c['object']}` | {c['source']} |"
        for c in sorted(CATALOG, key=lambda c: (c["kind"], c["name"]))
    ]
    return "\n".join(rows)
