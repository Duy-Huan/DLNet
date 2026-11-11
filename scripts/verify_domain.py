"""Numerical equivalence of the v1.0 domain networks (ops_domain.py, ops_graph2.py) with OFFICIAL code.

Run on GitHub Actions (job `verify-domain`), repositories cloned at pinned commits into one folder,
pytabkit (from its clone) and torch_geometric installed:

    python scripts/verify_domain.py --root /tmp/dom

Weights copied by name; outputs compared in eval mode. For the CTR models and the RealMLP layers,
the official classes are built directly (the surrounding training frameworks, FuxiCTR's BaseModel
and pytabkit's Fitter, are not needed for the forward pass); FuxiCTR's `get_activation` is
replaced by the plain torch activations it returns for "ReLU" / "Sigmoid".
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

from dlnet.zoo import ops_domain as od
from dlnet.zoo import ops_graph2 as og


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def module(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def drop_modules(prefix):
    for k in [k for k in sys.modules if k == prefix or k.startswith(prefix + ".")]:
        del sys.modules[k]


def report(name, ref, out, tol=1e-5):
    err = (ref - out).abs().max().item()
    ok = err < tol * max(1.0, ref.abs().max().item())
    print(f"{name:<46} max|diff| = {err:.2e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def perturb(m):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        for b in m.modules():
            if isinstance(b, nn.modules.batchnorm._BatchNorm):
                b.running_mean.normal_()
                b.running_var.uniform_(0.5, 2)


def copy(ref, mine, skip=(), rules=()):
    sd = {}
    for n, x in ref.state_dict().items():
        if any(s in n for s in skip):
            continue
        for a, b in rules:
            n = n.replace(a, b)
        sd[n] = x
    missing, unexpected = mine.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise ValueError(f"state_dict mismatch: missing {missing} unexpected {unexpected}")


def pre_input(m, store):
    m.register_forward_pre_hook(lambda mod, a, s=store: s.setdefault("x", a[0]))


# ------------------------------------------------------------------------- RealMLP


def check_realmlp(root):
    sys.path.insert(0, f"{root}/pytabkit")
    from pytabkit.models.nn_models.activations import ParametricActivationLayer
    from pytabkit.models.nn_models.base import BiasLayer, ScaleLayer, Variable, WeightLayer
    from pytabkit.models.nn_models.nn import PLREmbeddingsLayerCosBias

    ok = True
    torch.manual_seed(0)
    for d, act in ((7, "selu"), (12, "mish")):
        mine = od.RealMLP(d, 32, 2, act, 0.0, "pbld").eval()
        perturb(mine)
        e = mine.emb
        V = lambda t: Variable(t.detach().clone())
        layers = [PLREmbeddingsLayerCosBias(None, V(e.weight_1), V(e.bias_1), V(e.weight_2), V(e.bias_2), "linear", True),
                  ScaleLayer(V(mine.scale))]  # fmt: skip
        mods = list(mine.layers)
        for i in range(0, len(mods), 3):
            lin, pa = mods[i], mods[i + 1]
            layers += [WeightLayer(V(lin.weight), factor=lin.factor), BiasLayer(V(lin.bias)),
                       ParametricActivationLayer(od.REALMLP_ACTS[act], V(pa.weight))]  # fmt: skip
        x = torch.randn(5, d)
        y = x
        for layer in layers:
            y = layer.forward_cont(y)
        ok &= report(f"RealMLP PBLD + NTK blocks d={d} act={act}", y, mine(x))
    sys.path.pop(0)
    return ok


# ---------------------------------------------------------------------- time series


def _ts_cfg(**kw):
    base = {"task_name": "classification", "label_len": 0, "pred_len": 0, "down_sampling_window": 2,
            "channel_independence": 0, "decomp_method": "moving_avg", "top_k": 5, "use_future_temporal_feature": 0,
            "embed": "timeF", "freq": "h", "use_norm": 1, "down_sampling_method": "avg", "num_class": 3, "factor": 1,
            "activation": "relu", "features": "M"}  # fmt: skip
    return SimpleNamespace(**{**base, **kw})


def check_timemixer(root):
    drop_modules("layers")
    sys.path.insert(0, f"{root}/TimeMixer")
    m = load(f"{root}/TimeMixer/models/TimeMixer.py", "timemixer_official")
    ok = True
    torch.manual_seed(0)
    for c, t, dl, el in ((3, 32, 1, 2), (2, 48, 2, 1)):
        cfg = _ts_cfg(seq_len=t, down_sampling_layers=dl, e_layers=el, d_model=16, d_ff=32, dropout=0.1, moving_avg=25,
                      enc_in=c, c_out=c)  # fmt: skip
        ref = m.Model(cfg).eval()
        mine = od.TimeMixer(c, t, 16, 32, el, dl, 2, 25, 0.1).eval()
        perturb(ref)
        last = f"pdm_blocks.{el - 1}.mixing_multi_scale_season"
        copy(ref, mine, skip=("projection", "normalize_layers", "temporal_embedding", "position_embedding",
                              ".layer_norm.", "out_cross_layer", last))  # fmt: skip
        x = torch.randn(2, c, t)
        s = {}
        pre_input(ref.projection, s)
        ref.classification(x.transpose(1, 2), torch.ones(2, t))
        ok &= report(f"TimeMixer c={c} T={t} down={dl} layers={el}", s["x"], mine(x))
    sys.path.pop(0)
    drop_modules("layers")
    return ok


def check_timexer(root):
    drop_modules("layers")
    sys.modules.setdefault("reformer_pytorch", MagicMock())  # only for the Reformer attention variant
    sys.path.insert(0, f"{root}/TimeXer")
    m = load(f"{root}/TimeXer/models/TimeXer.py", "timexer_official")
    ok = True
    torch.manual_seed(0)
    for c, t, p, el in ((3, 32, 8, 2), (2, 24, 4, 1)):
        cfg = _ts_cfg(task_name="long_term_forecast", seq_len=t, pred_len=4, patch_len=p, e_layers=el, d_model=32,
                      d_ff=64, n_heads=4, dropout=0.1, enc_in=c)  # fmt: skip
        ref = m.Model(cfg).eval()
        mine = od.TimeXer(c, t, p, 32, 4, 64, el, 0.1).eval()
        perturb(ref)
        copy(ref, mine, skip=("head.", "position_embedding.pe"))
        x = torch.randn(2, c, t)
        s = {}
        pre_input(ref.head, s)
        ref.forecast_multi(x.transpose(1, 2), None, None, None)
        ok &= report(f"TimeXer c={c} T={t} patch={p} layers={el}", s["x"].reshape(2, -1), mine(x))
    sys.path.pop(0)
    drop_modules("layers")
    return ok


# ---------------------------------------------------------------------------- graphs


def _graph(n=7, f=5, p=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(n, n, generator=g) < p).float().triu(1)
    a = a + a.T
    return torch.randn(n, f, generator=g), a


def _edge_index(a):
    i, j = a.nonzero(as_tuple=True)
    return torch.stack([j, i])  # message j -> i for A[i, j] > 0


def check_rwse(root):
    module("graphgps", __path__=[f"{root}/GraphGPS/graphgps"])
    module("graphgps.encoder")
    module("graphgps.encoder.graphormer_encoder", graphormer_pre_processing=MagicMock())
    m = load(f"{root}/GraphGPS/graphgps/transform/posenc_stats.py", "graphgps_posenc")
    _, a = _graph()
    ks = list(range(1, 9))
    ref = m.get_rw_landing_probs(ks, _edge_index(a), num_nodes=a.shape[0])
    return report("GraphGPS RWSE (8 steps)", ref, og.rw_landing_probs(a, ks))


def check_gps(root):
    from torch_geometric.data import Batch, Data
    from torch_geometric.graphgym import register

    register.act_dict.setdefault("relu", nn.ReLU)
    module("performer_pytorch", SelfAttention=MagicMock())
    for sub in ("bigbird_layer", "gatedgcn_layer", "gine_conv_layer"):
        module(f"graphgps.layer.{sub}", SingleBigBirdLayer=MagicMock(), GatedGCNLayer=MagicMock(),
               GINEConvESLapPE=MagicMock())  # fmt: skip
    module("graphgps.layer", __path__=[f"{root}/GraphGPS/graphgps/layer"])
    m = load(f"{root}/GraphGPS/graphgps/layer/gps_layer.py", "graphgps_gps_layer")
    ok = True
    torch.manual_seed(0)
    for dim, heads in ((16, 4), (32, 2)):
        ref = m.GPSLayer(
            dim, "GIN", "Transformer", heads, act="relu", batch_norm=True, layer_norm=False
        ).eval()
        mine = og.GPSLayer(7, dim, heads).eval()
        perturb(ref)
        copy(ref, mine, skip=("local_model.eps",))  # GINConv's eps buffer (0, not trained)
        graphs = [_graph(7, dim, 0.4, s) for s in (0, 1)]
        batch = Batch.from_data_list([Data(x=x, edge_index=_edge_index(a)) for x, a in graphs])
        y = ref(batch).x.view(2, 7, dim)
        dense = torch.stack([torch.cat([x, a], -1) for x, a in graphs])
        ok &= report(f"GraphGPS GPSLayer GIN+Transformer d={dim}", y, mine(dense)[..., :dim])
    return ok


def check_polynormer(root):
    m = load(f"{root}/Polynormer/model.py", "polynormer_official")
    ok = True
    torch.manual_seed(0)
    for hidden, heads in ((16, 1), (8, 2)):
        ref = m.Polynormer(5, hidden, 3, local_layers=2, global_layers=2, heads=heads).eval()
        ref._global = True
        mine = og.Polynormer(7, 5, hidden, heads, 2, 2).eval()
        perturb(ref)
        copy(ref, mine, skip=("pred_local", "pred_global"))
        x, a = _graph(7, 5, 0.4, 2)
        s = {}
        pre_input(ref.pred_global, s)
        ref(x, _edge_index(a))
        ok &= report(
            f"Polynormer hidden={hidden} heads={heads}",
            s["x"],
            mine(torch.cat([x, a], -1)[None])[0, :, : hidden * heads],
        )
    return ok


def check_sgformer(root):
    module("torch_sparse", SparseTensor=MagicMock(), matmul=MagicMock())
    sys.path.insert(0, f"{root}/SGFormer/medium")
    drop_modules("models")
    load(f"{root}/SGFormer/medium/models.py", "models")
    m = load(f"{root}/SGFormer/medium/ours.py", "sgformer_official")
    ok = True
    torch.manual_seed(0)
    for hidden, layers in ((16, 1), (8, 2)):
        gnn = sys.modules["models"].GCN(5, hidden, hidden, num_layers=2, dropout=0.5, use_bn=True)
        ref = m.SGFormer(
            5, hidden, 3, num_layers=layers, num_heads=1, alpha=0.5, dropout=0.5, gnn=gnn
        ).eval()
        mine = og.SGFormer(7, 5, hidden, layers, 1).eval()
        perturb(ref)
        copy(ref, mine, skip=("fc.",))
        x, a = _graph(7, 5, 0.4, 3)
        data = SimpleNamespace(graph={"node_feat": x, "edge_index": _edge_index(a)})
        s = {}
        pre_input(ref.fc, s)
        ref(data)
        ok &= report(
            f"SGFormer hidden={hidden} layers={layers}",
            s["x"],
            mine(torch.cat([x, a], -1)[None])[0, :, :hidden],
        )
    sys.path.pop(0)
    drop_modules("models")
    return ok


# ------------------------------------------------------------------------------- CTR


def _fuxictr(root):
    acts = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}
    get_activation = lambda a, h=None: ([acts[x.lower()]() for x in a] if isinstance(a, list)
                                        else acts[a.lower()]() if isinstance(a, str) else a)  # fmt: skip
    module("fuxictr")
    module("fuxictr.pytorch")
    module("fuxictr.pytorch.torch_utils", get_activation=get_activation)
    mlp = load(f"{root}/FuxiCTR/fuxictr/pytorch/layers/blocks/mlp_block.py", "fuxictr_mlp_block")
    module("fuxictr.pytorch.models", BaseModel=nn.Module)
    module(
        "fuxictr.pytorch.layers",
        MLP_Block=mlp.MLP_Block,
        FeatureEmbedding=lambda *a, **k: nn.Identity(),
    )


def check_finalmlp(root):
    _fuxictr(root)
    m = load(f"{root}/FuxiCTR/model_zoo/FinalMLP/src/FinalMLP.py", "finalmlp_official")
    ok = True
    torch.manual_seed(0)
    for fields, emb, heads, out in ((5, 8, 1, 1), (3, 16, 2, 16)):
        fd = fields * emb
        ref = nn.Module()
        ref.mlp1 = sys.modules["fuxictr.pytorch.layers"].MLP_Block(fd, hidden_units=[64, 64, 64])
        ref.mlp2 = sys.modules["fuxictr.pytorch.layers"].MLP_Block(fd, hidden_units=[64, 64, 64])
        ref.fs_module = m.FeatureSelection(None, fd, emb, [64], [], [])
        ref.fusion_module = m.InteractionAggregation(64, 64, out, heads)
        mine = og.FinalMLP(fields, emb, (64, 64, 64), (64,), heads, out).eval()
        ref.eval()
        perturb(ref)
        copy(ref, mine)
        e = torch.randn(4, fields, emb)
        flat = e.flatten(1)
        f1, f2 = ref.fs_module(None, flat)
        y = ref.fusion_module(ref.mlp1(f1), ref.mlp2(f2))
        ok &= report(f"FinalMLP fields={fields} emb={emb} heads={heads}", y, mine(e))
    return ok


def check_wukong(root):
    _fuxictr(root)
    m = load(f"{root}/FuxiCTR/model_zoo/WuKong/src/WuKong.py", "wukong_official")
    ok = True
    torch.manual_seed(0)
    for fields, emb, layers in ((5, 8, 2), (12, 16, 3)):
        ref = nn.Sequential(*[m.WuKongLayer(fields if i == 0 else 32, 16, 16, emb, 8, [32, 32], "relu", 0.0, True)
                              for i in range(layers)]).eval()  # fmt: skip
        mine = nn.Sequential(
            *[og.WuKongLayer(fields if i == 0 else 32, 16, 16, emb, 8) for i in range(layers)]
        ).eval()
        perturb(ref)
        copy(ref, mine)
        e = torch.randn(4, fields, emb)
        ok &= report(f"WuKong stack fields={fields} emb={emb} layers={layers}", ref(e), mine(e))
    return ok


def check_dcnv3(root):
    _fuxictr(root)
    m = load(f"{root}/FCN/src/FCNv1.py", "fcn_official")
    ok = True
    torch.manual_seed(0)
    for fields, emb, layers in ((4, 8, 3), (6, 16, 2)):
        d = fields * emb
        mhe = m.MultiHeadFeatureEmbedding(None, emb, 1)
        ecn = m.ExponentialCrossNetwork(
            d, layers, batch_norm=False, layer_norm=False, net_dropout=0, num_heads=1
        ).eval()
        lcn = m.LinearCrossNetwork(
            d, layers, batch_norm=False, layer_norm=False, net_dropout=0, num_heads=1
        ).eval()
        mine = og.DCNv3(fields, emb, layers, layers).eval()
        for r in (ecn, lcn):
            perturb(r)
        copy(ecn, mine.ECN, skip=("fc.",))
        copy(lcn, mine.LCN, skip=("fc.",))
        e = torch.randn(4, fields, emb)
        x = mhe(e)  # (B, 1, F*D)
        s1, s2 = {}, {}
        pre_input(ecn.fc, s1)
        pre_input(lcn.fc, s2)
        ecn(x)
        lcn(x)
        ok &= report(
            f"DCNv3 (FCN) fields={fields} emb={emb} layers={layers}",
            torch.cat([s1["x"][:, 0], s2["x"][:, 0]], -1),
            mine(e),
        )
    return ok


CHECKS = (check_realmlp, check_timemixer, check_timexer, check_rwse, check_gps, check_polynormer, check_sgformer,
          check_finalmlp, check_wukong, check_dcnv3)  # fmt: skip

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    good = True
    for chk in CHECKS:
        try:
            good &= chk(a.root)
        except Exception as e:  # noqa: BLE001  (report every network, then fail)
            print(f"{chk.__name__:<46} ERROR {type(e).__name__}: {e}")
            good = False
    print("ALL OK" if good else "SOME MISMATCH")
    raise SystemExit(0 if good else 1)
