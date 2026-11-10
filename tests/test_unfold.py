"""Deep unfolding: closed-form checks and the comparison with the iterative solvers (M8)."""

import itertools
import json
import math
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from dlnet import cli, unfold


def _problem(m=20, n=40, N=600, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    X = rng.normal(size=(N, n)) * (rng.random((N, n)) < 0.1)
    Y = X @ A.T + 0.01 * rng.normal(size=(N, m))
    return (torch.from_numpy(a.astype(np.float32)) for a in (A, Y, X))


def test_soft_threshold_and_scalar_ista_closed_form():
    x = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    assert torch.equal(unfold.soft(x, 1.0), torch.tensor([-2.0, 0.0, 0.0, 0.0, 2.0]))
    # A = [[a]]: L = a^2 and one step from 0 gives soft(y/a, lam/a^2); it is already the fixed point
    A, Y, lam = torch.tensor([[2.0]]), torch.tensor([[3.0], [0.1]]), 1.0
    want = torch.tensor([[3.0 / 2 - lam / 4], [0.0]])
    assert torch.allclose(unfold.ista(A, Y, lam, 1), want)
    assert torch.allclose(unfold.ista(A, Y, lam, 25), want, atol=1e-6)
    assert torch.allclose(unfold.fista(A, Y, lam, 25), want, atol=1e-5)
    assert unfold.lipschitz(torch.tensor([[3.0, 0.0], [0.0, 1.0]])) == pytest.approx(9.0)


def test_ista_decreases_the_lasso_objective_and_fista_is_not_slower():
    A, Y, _ = _problem()
    lam = 0.05

    def obj(X):
        return float((0.5 * ((Y - X @ A.t()) ** 2).sum(1) + lam * X.abs().sum(1)).mean())

    f = [obj(unfold.ista(A, Y, lam, k)) for k in (0, 1, 5, 20, 80)]
    assert all(a >= b - 1e-6 for a, b in itertools.pairwise(f))  # monotone (step 1/L)
    assert obj(unfold.fista(A, Y, lam, 20)) <= f[3] + 1e-6


@pytest.mark.parametrize("shared", [False, True])
def test_untrained_lista_is_exactly_ista(shared):
    A, Y, _ = _problem()
    net = unfold.LISTA(A, 7, lam=0.05, shared=shared)
    with torch.no_grad():
        assert torch.allclose(net(Y), unfold.ista(A, Y, 0.05, 7), atol=1e-5)
        assert torch.allclose(net(Y, K=3), unfold.ista(A, Y, 0.05, 3), atol=1e-5)
    m, n = A.shape
    assert sum(p.numel() for p in net.parameters()) == (1 if shared else 7) * m * n + 7


def test_nmse_db_closed_form():
    X = torch.ones(2, 5)
    assert unfold.nmse_db(X * 0.9, X) == pytest.approx(10 * math.log10(0.01), abs=1e-4)
    assert unfold.nmse_db(torch.zeros(2, 5), X) == pytest.approx(0.0, abs=1e-6)


def test_trained_lista_beats_tuned_ista_and_fista_at_same_depth():
    A, Y, X = _problem(N=1500)
    tr, va, te = unfold.split_indices(len(Y), 0)
    K = 6
    lam_i, _ = unfold.tune_lambda(unfold.ista, A, Y[va], X[va], K)
    lam_f, _ = unfold.tune_lambda(unfold.fista, A, Y[va], X[va], K)
    torch.manual_seed(0)
    net = unfold.LISTA(A, K, lam=lam_i)
    with torch.no_grad():
        before = unfold.nmse_db(net(Y[va]), X[va])
    best = unfold.train_lista(net, Y[tr], X[tr], Y[va], X[va], epochs=30, seed=0)
    assert best <= before  # best-val weights are kept, so training can never make it worse
    with torch.no_grad():
        lista = unfold.nmse_db(net(Y[te]), X[te])
    classical = min(
        unfold.nmse_db(unfold.ista(A, Y[te], lam_i, K), X[te]),
        unfold.nmse_db(unfold.fista(A, Y[te], lam_f, K), X[te]),
    )
    assert lista < classical - 1.0  # at least 1 dB better at the same number of iterations
    assert float(net.theta.min()) >= 0.0


def test_cli_unfold_writes_leaderboard_and_runnable_package(tmp_path):
    A, Y, X = (t.numpy() for t in _problem(N=400))
    data, out = tmp_path / "p.npz", tmp_path / "o"
    np.savez(data, A=A, Y=Y, X=X)
    argv = ["unfold", "--data", str(data), "--layers", "2,4", "--epochs", "8", "--seeds", "2",
            "--topk", "1", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    board = json.loads((out / "leaderboard.json").read_text())
    assert [r["K"] for r in board["top_k"]] and board["top_k"][0]["n_seeds"] == 2
    row = board["top_k"][0]
    assert set(row["baselines_same_K"]) == {"ista", "fista"} and "_state" not in row
    pkg = next(out.glob("top1_lista_K*"))
    assert all((pkg / f).exists() for f in ("model.py", "train.py", "weights.pt", "config.yaml"))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"], cwd=pkg, capture_output=True, check=False
    )
    assert r.returncode == 0, r.stdout.decode()[-500:]
    r = subprocess.run([sys.executable, "train.py", "--data", str(data), "--epochs", "2"], cwd=pkg,
                       capture_output=True, text=True, check=False)  # fmt: skip
    assert r.returncode == 0 and "test NMSE" in r.stdout, r.stderr[-500:]
    assert cli.main(argv) == 0  # re-running into the same --out is allowed


def test_load_problem_rejects_bad_files(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, A=np.zeros((3, 5)), Y=np.zeros((4, 2)), X=np.zeros((4, 5)))
    with pytest.raises(SystemExit, match="do not fit"):
        unfold.load_problem(str(p))
    np.savez(p, A=np.zeros((3, 5)))
    with pytest.raises(SystemExit, match="missing"):
        unfold.load_problem(str(p))
