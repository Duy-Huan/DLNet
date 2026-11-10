"""Autoencoders and variational autoencoders: the reconstruction task, the VAE bottleneck, PCA references."""

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from dlnet import array_data, cli, core, plugins, zoo
from dlnet.zoo import families
from dlnet.zoo.ops_gen import VAEBottleneck


def images(n=120, s=8, seed=0):
    """Images on a 3-dimensional linear subspace of pixel space, so a 3-component PCA is exact."""
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(3, s * s))
    return (rng.normal(size=(n, 3)) @ basis).reshape(n, s, s).astype(np.float32)


@pytest.fixture
def img_npz(tmp_path):
    np.savez(tmp_path / "x.npz", X=images())
    return tmp_path / "x.npz"


def test_vae_kl_closed_form_and_eval_is_deterministic():
    b = VAEBottleneck(4, 2)
    with torch.no_grad():
        for m in (b.mu, b.logvar):
            m.weight.zero_()
            m.bias.zero_()
    b.train()
    b(torch.randn(3, 4, 5, 5))
    assert float(b.last_kl) == pytest.approx(0.0)  # mu = 0, logvar = 0: q = N(0, I), KL = 0
    with torch.no_grad():
        b.mu.bias.fill_(1.0)  # q = N(1, 1) per element: KL = 0.5 * 1^2
    b.train()
    b(torch.randn(3, 4, 5, 5))
    assert float(b.last_kl) == pytest.approx(
        0.5 * 2 * 5 * 5
    )  # summed over the 2 x 5 x 5 latent elements
    b.eval()
    x = torch.randn(3, 4, 5, 5)
    assert torch.equal(b(x), b(x)) and b.last_kl is None  # z = mu, no sampling, no KL term


def test_build_reconstruction_target_is_the_input_and_metrics_closed_form():
    d = array_data.build_reconstruction(images(), seed=0)
    assert d["task"] == "reconstruction" and d["Xtr"] is d["ytr"] and d["out_dim"] == 1
    assert len(d["Xtr"]) + len(d["Xva"]) + len(d["Xte"]) == 120
    assert float(d["Xtr"].mean()) == pytest.approx(0.0, abs=1e-4)
    y = np.array([[0.0, 2.0], [4.0, 0.0]])
    m = core.compute_metrics(y, y + 1.0, "reconstruction", "neg_mse")
    assert m["mse"] == pytest.approx(1.0) and m["score"] == pytest.approx(-1.0)
    assert m["psnr"] == pytest.approx(10 * np.log10(16.0 / 1.0))  # range 4


def test_pca_references_are_exact_on_a_low_dimensional_subspace():
    plugin = plugins.get_plugin("image_reconstruction")
    d = array_data.build_reconstruction(images(), seed=0)
    ref = {r["name"]: r for r in plugin.reference(d)}
    assert ref["pca_8_components"]["te"]["mse"] < 1e-8  # the data has rank 3
    assert ref["mean_image"]["te"]["mse"] > 0.5 and plugin.reference(d)[0]["name"].startswith("pca")


@pytest.mark.parametrize("fam", families.RECONSTRUCTION)
def test_autoencoder_families_reconstruct_the_input_shape_and_train(fam):
    cfg = {
        "family": fam,
        "width": 8,
        "depth": 2,
        "dropout": 0.0,
        "latent": 4,
        "encoder": "vgg",
        "beta": 1e-3,
    }
    done = families.complete(dict(cfg), "reconstruction", (1, 8, 8), 1)
    net = zoo.build_model(done)
    x = torch.randn(4, 1, 8, 8)
    out = net(x)
    assert out.shape == x.shape
    (out - x).square().mean().backward()
    assert (net.kl_loss() is None) == (fam == "conv_ae")
    # three resolution levels on 8 pixels (8 -> 4 -> 2): 4 latent channels x 2 x 2, checked on the built encoder
    code = net.body.stages[-1](_encode(net, x))
    assert families.code_size(cfg, (1, 8, 8)) == code[0].numel() == 4 * 2 * 2


def _encode(net, x):
    for s in net.body.stages[:-1]:
        x = s(x)
    return x


def test_trainer_adds_the_kl_term_and_reduces_reconstruction_error():
    d = array_data.build_reconstruction(images(n=200), seed=0)
    cfg = families.complete(
        {
            "family": "vae",
            "width": 8,
            "depth": 2,
            "dropout": 0.0,
            "latent": 4,
            "encoder": "vgg",
            "beta": 1e-3,
        },
        "reconstruction",
        (1, 8, 8),
        1,
    )
    torch.manual_seed(0)
    tr = core.Trainer(zoo.build_model(cfg), d, lr=3e-3, wd=0.0, batch=32, seed=0)
    before = tr.evaluate("va")["mse"]
    tr.fit(8)
    assert (
        tr.model.kl_loss() is not None
    )  # the last training forward recorded a KL (evaluate() clears it)
    assert tr.evaluate("va")["mse"] < before


def test_router_picks_the_autoencoder_plugin_for_label_free_images(img_npz, tmp_path):
    assert plugins.route(str(img_npz)).name == "image_reconstruction"
    np.savez(tmp_path / "labelled.npz", X=images(), y=np.arange(120) % 2)
    assert plugins.route(str(tmp_path / "labelled.npz")).name == "image_classification"


@pytest.mark.e2e
def test_cli_design_end_to_end_and_package_runs(img_npz, tmp_path):
    out = tmp_path / "o"
    argv = ["design", "--data", str(img_npz), "--budget", "60s", "--topk", "1", "--per-family", "1", "--zoo", "0", "--evolve", "0",
            "--no-proxies", "--out", str(out)]  # fmt: skip
    assert cli.main(argv) == 0
    board = json.loads((out / "leaderboard.json").read_text())
    assert (
        board["metric"] == "neg_mse"
        and "mse" in board["top_k"][0]["test"]
        and board["top_k"][0]["flops"] is not None
    )
    assert (
        "MSE" in (out / board["top_k"][0]["name"] / "report.md").read_text()
        or "mse" in (out / "leaderboard.md").read_text()
    )
    pkg = next(out.glob("top1_*"))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"], cwd=pkg, capture_output=True, check=False
    )
    assert r.returncode == 0, r.stdout.decode()[-600:]
    r = subprocess.run(
        [sys.executable, "train.py", "--data", str(img_npz), "--epochs", "2"],
        cwd=pkg,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0 and "test:" in r.stdout, r.stderr[-800:]


def test_autoencoder_is_judged_against_pca_of_its_own_code_size():
    """Was: the verdict compared every autoencoder with PCA-32, whatever its code size."""
    from dlnet.plugins.arrays import ImageReconstructionPlugin

    p = ImageReconstructionPlugin()
    ref = [{"name": f"pca_{k}", "code": k, "va": {"score": -1.0 / (k + 1)}} for k in (0, 4, 8, 32)]
    fair = p.reference_for({"code": 8}, ref)
    assert [r["name"] for r in fair] == ["pca_8", "pca_4", "pca_0"]
    assert p.reference_for({}, ref) is ref
