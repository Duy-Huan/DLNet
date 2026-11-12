import json
from pathlib import Path

import torch

from model import build_model


def test_forward_shape():
    cfg = json.load(open(Path(__file__).parent.parent / "config.yaml"))
    m = build_model(cfg["arch"], cfg["in_dim"], cfg["out_dim"]).eval()
    kind, shape = cfg["data"]["kind"], cfg["data"]["shape"]
    if kind == "ids":  # shape = cardinalities of the id fields
        x = torch.stack([torch.randint(0, c, (4,)) for c in shape], 1)
    elif kind == "graph":  # shape = (N, F + N): features, then a symmetric adjacency
        n = shape[0]
        a = (torch.rand(4, n, n) < 0.3).float()
        x = torch.cat([torch.randn(4, n, shape[1] - n), ((a + a.transpose(1, 2)) > 0).float()], -1)
    else:
        x = torch.randn(4, *shape)
    out = m(x)
    if cfg["data"]["kind"] == "segmentation":
        assert out.shape == (4, cfg["out_dim"], *shape[1:])
    else:
        assert out.shape == (4, cfg["out_dim"])
