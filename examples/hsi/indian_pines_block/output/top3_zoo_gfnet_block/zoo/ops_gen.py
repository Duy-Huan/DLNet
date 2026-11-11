"""Generative blocks: variational bottleneck, DCGAN generator and discriminator pieces (torch only).

Sources (see THIRD_PARTY.md):
  rasbt/deeplearning-models   ConvVariationalAutoencoder (mean / log-variance heads, reparameterisation,
                              KL against a standard normal), DCGAN generator and discriminator layout
                              (transposed convolutions + BatchNorm + LeakyReLU, tanh output)
"""

from __future__ import annotations

import torch
from torch import nn

from .core import op


class VAEBottleneck(nn.Module):
    """Convolutional variational bottleneck: z = mu + exp(logvar / 2) * eps while training, z = mu in eval.

    `last_kl` holds KL(q(z|x) || N(0, I)) summed over the latent elements and averaged over the batch from the
    last training forward pass; `ZooNet.kl_loss()` collects it and the trainer adds `kl_weight * kl`.
    """

    def __init__(self, c, latent):
        super().__init__()
        self.mu, self.logvar = (
            nn.Conv2d(c, latent, 3, padding=1),
            nn.Conv2d(c, latent, 3, padding=1),
        )
        self.last_kl = None

    def forward(self, x):
        mu, logvar = self.mu(x), self.logvar(x).clamp(-10, 10)
        if not self.training:
            self.last_kl = None
            return mu
        self.last_kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).flatten(1).sum(1).mean()
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


@op("vae_bottleneck", "map", space={"latent": [2, 4, 8, 16]}, tags={"adapt"},
    source="rasbt/deeplearning-models ConvVariationalAutoencoder (reparameterisation trick, Kingma & Welling arXiv:1312.6114)")  # fmt: skip
def _vae(dims, kind, latent):
    return VAEBottleneck(dims[0], latent)


class VecToMap(nn.Module):
    def __init__(self, d, c, s):
        super().__init__()
        self.c, self.s = c, s
        self.f = nn.Sequential(nn.Linear(d, c * s * s), nn.LeakyReLU(0.2))

    def forward(self, z):
        return self.f(z).view(len(z), self.c, self.s, self.s)


@op(
    "vec_to_map",
    "vec",
    "map",
    space={"c": [32, 64, 128], "s": [2, 4]},
    tags={"adapt", "stem"},
    source="DCGAN generator stem: latent vector -> (c, s, s) feature map",
)
def _vec_to_map(dims, kind, c, s):
    return VecToMap(dims[0], c, s)
