"""DLNet module zoo: typed building blocks, architecture specs, grammar search (torch only).

Importing this package registers every op. Layout:
  core         registry, kinds, combinators, static checking, head, `build_model`
  ops_basic    primitives and adapters between kinds          ops_cnn   1-D/2-D/3-D convolutional blocks
  ops_seq      attention, transformer, RNN, Mamba, TextCNN     ops_vec   factorised linear, interaction
  ops_graph    GCN, GAT, SAGE, GIN, grid graphs, id embeddings  ops_gen   VAE bottleneck, DCGAN stem
  ops_img      image blocks ported from timm, smp and MONAI
  ops_domain   RealMLP (tabular), TimeMixer, TimeXer (time series)
  ops_graph2   GraphGPS + RWSE, Polynormer, SGFormer; FinalMLP, DCNv3, WuKong (CTR)
  ops_seq2     Gated DeltaNet, mLSTM, GR-KAN / KAT, MambaVision (reference implementations)
  ops_hsi      HSI networks: MambaHSI, MambaHSI+, SSFTT, FAHM
  ops_next     2023-2025 drop-ins: gated/diff attention, DyT, 2-D RoPE, WTConv, UIB, MQA, LSNet, StarNet, EfficientViT, MSCA
  ops_freq     frequency domain (FFT/iFFT): GFNet, AFNO, FNO, FFC, FcaNet, FNet, spectral filters
  dense        encoder-decoder composition, smp decoders                    space     sampling, mutation, crossover
  presets      well-known architectures written as specs
Sources and what was and was not ported: THIRD_PARTY.md.
"""

from . import (  # noqa: F401  (registers the ops)
    dense,
    ops_basic,
    ops_cnn,
    ops_domain,
    ops_freq,
    ops_gen,
    ops_graph,
    ops_graph2,
    ops_hsi,
    ops_img,
    ops_modern,
    ops_next,
    ops_paper,
    ops_seq,
    ops_seq2,
    ops_vec,
)
from .core import (
    COMBINATORS,
    KINDS,
    OPS,
    build_model,
    check,
    count_params,
    describe,
    label,
    leaf_ops,
    op,
    ops_with,
    signature,
)

__all__ = [
    "COMBINATORS", "KINDS", "OPS", "build_model", "check", "count_params", "describe", "label",
    "leaf_ops", "op", "ops_with", "signature",
]  # fmt: skip
