import json
from pathlib import Path

import torch

from model import build_model


def test_forward_shape():
    cfg = json.load(open(Path(__file__).parent.parent / "config.yaml"))
    p = cfg["data"]["patch"]
    m = build_model(cfg["arch"], cfg["in_dim"], cfg["out_dim"]).eval()
    out = m(torch.randn(2, cfg["in_dim"], p, p))
    assert out.shape == (2, cfg["out_dim"])
