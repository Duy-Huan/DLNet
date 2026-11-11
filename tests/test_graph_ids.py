"""Graph (GCN, GAT, GraphSAGE, GIN, rasbt grid graphs) and categorical-id (recommender) blocks and plugins."""

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from dlnet import array_data, cli, plugins, zoo
from dlnet.zoo import families
from dlnet.zoo.ops_graph import EmbedFields, PairDots, grid_adjacency, renormalize


def graphs(n_graphs=90, n=12, seed=0):
    """Class 0: Erdos-Renyi. Class 1: a sparser random graph plus planted triangles. Features: noise only."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_graphs, n, 2)).astype(np.float32)
    A = np.zeros((n_graphs, n, n), dtype=np.float32)
    y = rng.integers(0, 2, n_graphs)
    for g in range(n_graphs):
        p = 0.12 if y[g] else 0.2
        u = np.triu(rng.random((n, n)) < p, 1)
        if y[g]:
            for _ in range(5):
                i, j, k = rng.choice(n, 3, replace=False)
                u[min(i, j), max(i, j)] = u[min(i, k), max(i, k)] = u[min(j, k), max(j, k)] = True
        A[g] = (u | u.T).astype(np.float32)
    return X, A, y


def interactions(n_users=60, n_items=40, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    U, V = rng.normal(size=(n_users, 4)), rng.normal(size=(n_items, 4))
    u, i = rng.integers(0, n_users, n), rng.integers(0, n_items, n)
    return np.stack([u, i], 1), (3 + (U[u] * V[i]).sum(1) + 0.1 * rng.normal(size=n)).astype(
        np.float32
    )


@pytest.fixture
def graph_npz(tmp_path):
    X, A, y = graphs()
    np.savez(tmp_path / "g.npz", X=X, A=A, y=y)
    return tmp_path / "g.npz"


@pytest.fixture
def recsys_npz(tmp_path):
    ids, y = interactions()
    np.savez(tmp_path / "r.npz", ids=ids, y=y)
    return tmp_path / "r.npz"


# ------------------------------------------------------------------------- graph blocks


def test_renormalisation_closed_form():
    a = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    assert torch.allclose(
        renormalize(a), torch.full((1, 2, 2), 0.5)
    )  # (A + I) = ones, degree 2: 1 / sqrt(2 * 2)
    iso = renormalize(torch.zeros(1, 3, 3))
    assert torch.allclose(iso, torch.eye(3).unsqueeze(0))  # isolated nodes keep themselves


def _packed(b=3, n=7, f=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(b, n, n, generator=g) < 0.4).float()
    a = ((a + a.transpose(1, 2)) > 0).float() * (1 - torch.eye(n))
    return torch.cat([torch.randn(b, n, f, generator=g), a], -1), n, f


@pytest.mark.parametrize(
    "layer",
    [
        {"op": "gcn", "out": 8},
        {"op": "gat", "out": 8, "heads": 2},
        {"op": "sage", "out": 8},
        {"op": "gin", "out": 8},
    ],
)
@pytest.mark.parametrize("readout", ["mean", "max", "sum"])
def test_message_passing_nets_are_invariant_to_node_order(layer, readout):
    """A graph network must not depend on how the nodes are numbered: permute nodes and adjacency, same output."""
    x, n, f = _packed()
    spec = {"op": "seq", "body": [layer, layer, {"op": "graph_readout", "mode": readout}]}
    m = zoo.build_model(
        {"spec": spec, "input": {"kind": "graph", "dims": [n, f + n]}, "out_dim": 3}
    ).eval()
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
    h, a = x[..., :f], x[..., f:]
    xp = torch.cat([h[:, perm], a[:, perm][:, :, perm]], -1)
    with torch.no_grad():
        assert torch.allclose(m(x), m(xp), atol=1e-5)
    m(x).sum().backward()


def test_attention_rows_sum_to_one_and_isolated_nodes_attend_to_themselves():
    from dlnet.zoo.ops_graph import GAT

    layer = GAT(5, 4, 8, 2)
    h = torch.randn(2, 5, 4)
    a = torch.zeros(2, 5, 5)
    a[:, 0, 1] = a[:, 1, 0] = 1.0  # nodes 2, 3, 4 are isolated
    out = layer.nodes(h, a)
    assert out.shape == (2, 5, 8) and torch.isfinite(out).all()
    z = layer.lin(h).view(2, 5, 2, 4)
    assert torch.allclose(
        nn_elu(z[:, 4].reshape(2, -1)), out[:, 4], atol=1e-6
    )  # isolated: attends only to itself


def nn_elu(x):
    return torch.nn.functional.elu(x)


def test_sage_mean_and_gin_sum_aggregators_closed_form():
    from dlnet.zoo.ops_graph import GIN, SAGE

    h = torch.tensor([[[1.0], [2.0], [4.0]]])
    a = torch.tensor(
        [[[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    )  # node 0 has neighbours 1 and 2
    s = SAGE(3, 1, 1)
    with torch.no_grad():
        s.lin.weight.copy_(torch.tensor([[0.0, 1.0]]))
        s.lin.bias.zero_()
    assert torch.allclose(
        s.nodes(h, a)[0, :, 0], torch.relu(torch.tensor([3.0, 1.0, 1.0]))
    )  # the mean of the neighbours
    g = GIN(3, 1, 1)
    agg = (1 + g.eps) * h + a @ h
    assert torch.allclose(
        agg[0, :, 0], torch.tensor([1.0 + 6.0, 2.0 + 1.0, 4.0 + 1.0])
    )  # own + sum of neighbours


# ---------------------------------------------------------------------- rasbt grid graphs


def test_grid_adjacency_and_laplacian_spectrum():
    a = grid_adjacency(6, 6, normalize=False)
    assert torch.allclose(a, a.t()) and torch.allclose(
        torch.diagonal(a), torch.ones(36)
    )  # exp(0) = 1
    ahat = grid_adjacency(6, 6)
    assert torch.allclose(ahat, ahat.t(), atol=1e-6) and float(ahat.max()) <= 1.0 + 1e-6
    from dlnet.zoo.ops_graph import GridSpectral

    layer = GridSpectral(1, 6, 6, 10, 2)
    assert torch.allclose(
        layer.v.t() @ layer.v, torch.eye(10), atol=1e-4
    )  # orthonormal eigenvectors
    d = (grid_adjacency(6, 6, normalize=False).double().sum(1) + 1e-5) ** -0.5
    lap = (
        torch.eye(36, dtype=torch.double)
        - d[:, None] * grid_adjacency(6, 6, normalize=False).double() * d[None]
    )
    assert float(torch.linalg.eigvalsh(lap)[0]) == pytest.approx(
        0.0, abs=1e-3
    )  # the smallest Laplacian eigenvalue
    assert layer(torch.randn(2, 1, 6, 6)).shape == (2, 2, 6, 6)
    with pytest.raises(ValueError, match="1024"):
        zoo.check({"op": "grid_graph_avg"}, "map", (1, 40, 40))


# -------------------------------------------------------------------------------- ids


def test_embeddings_gmf_and_pair_dots_closed_form():
    emb = EmbedFields([3, 4], 2)
    with torch.no_grad():
        emb.tables[0].weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        emb.tables[1].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [1.0, 1.0]]))
    z = emb(torch.tensor([[2, 2], [0, 1]]))
    assert z.tolist() == [[[5.0, 6.0], [2.0, 2.0]], [[1.0, 2.0], [0.0, 1.0]]]
    assert (z[:, 0] * z[:, 1]).tolist() == [[10.0, 12.0], [0.0, 2.0]]  # what `gmf` computes
    assert PairDots(2, False)(z).tolist() == [[22.0], [2.0]]  # one pair: <u, v>
    assert PairDots(3, True)(torch.ones(1, 3, 2)).shape == (1, 6)
    assert zoo.check({"op": "embed_fields", "dim": 8}, "ids", (20, 30))["params"] == (20 + 30) * 8


def test_ids_kind_static_check_and_wrong_arity():
    with pytest.raises(ValueError, match="exactly two"):
        zoo.check(
            {"op": "seq", "body": [{"op": "embed_fields", "dim": 4}, {"op": "gmf"}]},
            "ids",
            (5, 6, 7),
        )
    with pytest.raises(ValueError, match="takes"):
        zoo.check({"op": "gcn", "out": 8}, "ids", (5, 6))


# ------------------------------------------------------------------------------- data


def test_graph_stats_closed_form_for_a_triangle_and_a_path():
    tri = np.ones((3, 3), dtype=np.float32) - np.eye(3, dtype=np.float32)
    path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    xp = np.zeros((2, 3, 1 + 3), dtype=np.float32)
    xp[0, :, 1:], xp[1, :, 1:] = tri, path
    s = array_data.graph_stats(xp)
    assert s[0, 0] == pytest.approx(1.0) and s[1, 0] == pytest.approx(4 / 6)  # density
    assert s[0, 4] == pytest.approx(1 / 3) and s[1, 4] == 0.0  # triangles per node
    assert s[0, 5] == pytest.approx(1.0) and s[1, 5] == 0.0  # global clustering


def test_build_graph_splits_whole_graphs_and_normalises_features_only():
    X, A, y = graphs()
    Xp = np.concatenate([X, A], -1)
    d = array_data.build_graph(Xp, y, "classification", seed=0)
    assert d["shape"] == [12, 14] and d["in_dim"] == 2 and d["kind"] == "graph"
    assert len(d["Xtr"]) + len(d["Xva"]) + len(d["Xte"]) == len(X)
    assert float(d["Xtr"][..., :2].mean()) == pytest.approx(
        0.0, abs=1e-4
    )  # standardised with train statistics
    assert torch.equal(
        d["Xtr"][..., 2:], torch.from_numpy(Xp)[..., 2:][_train_idx(d, y)]
    )  # adjacency untouched
    assert (
        d["Xnp"]["tr"].shape[1] == 6 + 5 - 5 + 5 + 2 * 2 - 5 + 5 - 6 + 6
        or d["Xnp"]["tr"].shape[1] == 15
    )


def _train_idx(d, y):
    from dlnet import core

    return core.stratified_split(core.encode_target(y, "classification")[0], "classification", 0)[0]


def test_build_ids_cardinalities_and_one_hot_reference_view():
    ids, y = interactions()
    d = array_data.build_ids(ids, y, "regression", seed=0)
    assert (
        d["cards"] == [int(ids[:, 0].max()) + 1, int(ids[:, 1].max()) + 1]
        and d["Xtr"].dtype == torch.long
    )
    assert (
        d["Xnp"]["tr"].shape[1] == sum(d["cards"]) and (d["Xnp"]["tr"].sum(1) == 2).all()
    )  # one hot per field
    with pytest.raises(SystemExit, match="non-negative"):
        array_data.build_ids(np.array([[-1, 2]]), np.array([1.0]), "regression")


# ----------------------------------------------------------------------- plugins / CLI


def test_router_and_every_recipe_for_graphs_and_ids(graph_npz, recsys_npz, image_npz_like):
    assert (
        plugins.route(str(graph_npz)).name == "graph"
        and plugins.route(str(recsys_npz)).name == "recsys"
    )
    assert (
        plugins.route(str(image_npz_like)).name == "image_classification"
    )  # a plain X, y file is still an image set
    for fam in families.GRAPH:
        m = families.build(
            {"family": fam, "width": 16, "depth": 2, "dropout": 0.0}, "graph", (12, 14), 2
        )
        x, _, _ = _packed(2, 12, 2)
        assert m(x).shape == (2, 2)
    for fam in families.RECSYS:
        m = families.build(
            {"family": fam, "width": 8, "depth": 1, "dropout": 0.0}, "recsys", (20, 30), 1
        )
        assert m(
            torch.stack([torch.randint(0, 20, (4,)), torch.randint(0, 30, (4,))], 1)
        ).shape == (4, 1)


@pytest.fixture
def image_npz_like(tmp_path):
    np.savez(
        tmp_path / "i.npz",
        X=np.random.default_rng(0).normal(size=(40, 8, 8)).astype(np.float32),
        y=np.arange(40) % 2,
    )
    return tmp_path / "i.npz"


@pytest.mark.e2e
@pytest.mark.parametrize("which", ["graph", "recsys"])
def test_cli_design_end_to_end_and_package_runs(which, graph_npz, recsys_npz, tmp_path):
    data = graph_npz if which == "graph" else recsys_npz
    out = tmp_path / "o"
    argv = ["design", "--data", str(data), "--budget", "240ep", "--topk", "1", "--per-family", "1", "--zoo", "0", "--evolve", "0",
            "--no-proxies", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    board = json.loads((out / "leaderboard.json").read_text())
    assert board["top_k"] and board["top_k"][0]["flops"] is not None
    pkg = next(out.glob("top1_*"))
    cfg = json.loads((pkg / "config.yaml").read_text())
    assert cfg["plugin"] == which and cfg["arch"]["input"]["kind"] == (
        "graph" if which == "graph" else "ids"
    )
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"], cwd=pkg, capture_output=True, check=False
    )
    assert r.returncode == 0, r.stdout.decode()[-600:]
    r = subprocess.run(
        [sys.executable, "train.py", "--data", str(data), "--epochs", "2"],
        cwd=pkg,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0 and "test:" in r.stdout, r.stderr[-800:]
