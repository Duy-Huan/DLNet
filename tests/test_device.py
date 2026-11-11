"""Device and mixed precision (v0.19, `--device`, `--amp`).

There is no GPU in CI. Two substitutes cover what can go wrong on one without it:
- the `meta` device rejects a CPU tensor mixed with its tensors exactly like CUDA does (CPU
  0-dim scalars allowed), so a model or training loop that creates a tensor without `device=`
  fails here as it would on a GPU;
- autocast with bfloat16 / float16 and the float16 GradScaler run on the CPU too.
What remains untested is the speed and numerics on real GPU kernels (agent.md 9.3).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dlnet import cli, core, hsi_data, zoo
from dlnet.zoo import space

pytestmark = pytest.mark.regression


def _data(n=48, shape=(1, 8, 8), task="classification"):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, *shape, generator=g)
    y = (X.flatten(1).mean(1) > 0).long() if task == "classification" else X.flatten(1)[:, 0]
    k = n // 4
    yr = y.numpy()
    return {"task": task, "primary": "accuracy" if task == "classification" else "neg_rmse",
            "Xtr": X[: 2 * k], "ytr": y[: 2 * k], "Xva": X[2 * k : 3 * k], "yva": y[2 * k : 3 * k],
            "Xte": X[3 * k :], "yte": y[3 * k :], "y_mean": 0.0, "y_std": 1.0,
            "y_raw": {"tr": yr[: 2 * k], "va": yr[2 * k : 3 * k], "te": yr[3 * k :]}}  # fmt: skip


def _cnn():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
                               torch.nn.Flatten(), torch.nn.Linear(4 * 36, 2))  # fmt: skip


def test_pick_device_and_amp_policy(monkeypatch):
    assert core.pick_device("cpu") == torch.device("cpu")
    if not torch.cuda.is_available():
        assert core.pick_device("auto").type in ("cpu", "mps")
        with pytest.raises(ValueError, match="no CUDA GPU"):
            core.pick_device("cuda:0")
    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    assert core.amp_dtype(cpu, "auto") is None and core.amp_dtype(cuda, "off") is None
    assert (
        core.amp_dtype(cpu, "bf16") is torch.bfloat16
        and core.amp_dtype(cpu, "fp16") is torch.float16
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    assert core.amp_dtype(cuda, "auto") is torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
    assert core.amp_dtype(cuda, "auto") is torch.float16  # older GPUs: fp16 with loss scaling


def test_cli_refuses_a_missing_gpu_before_spending_the_budget(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("a CUDA GPU is present")
    with pytest.raises(SystemExit, match="--device: .*no CUDA GPU"):
        cli.main(["design", "--data", "x.npz", "--device", "cuda", "--out", str(tmp_path / "o")])


def test_training_loop_moves_every_tensor_to_the_device():
    """meta stands in for a GPU: a batch left on the CPU raises 'not on the expected device'."""
    d = _data()
    m = _cnn()
    tr = core.Trainer(m, d, batch=16, device="meta")
    tr.fit(2)
    assert next(m.parameters()).device.type == "meta" and tr.epochs_done == 2
    assert {k for k, _ in d["_on_device"]} == {"Xtr", "ytr"}  # copied once, then reused
    assert d["Xtr"].device.type == "cpu"  # the CPU data the other steps use is unchanged


@pytest.mark.parametrize("amp", ["bf16", "fp16"])
def test_mixed_precision_trains_like_full_precision(amp):
    def run(a):
        tr = core.Trainer(_cnn(), _data(200), lr=1e-2, batch=16, amp=a)
        before = tr.evaluate("va")["score"]
        tr.fit(15)
        return tr, before, tr.evaluate("va")["score"]

    tr, before, after = run(amp)
    _, _, full = run("off")
    assert tr.scaler.is_enabled() == (amp == "fp16")  # loss scaling only for float16
    assert after > before + 0.2 and abs(after - full) <= 0.1  # measured: 0.68 for all three
    assert all(p.dtype == torch.float32 for p in tr.model.parameters())  # master weights stay fp32


def test_full_precision_cpu_path_is_unchanged():
    """The default path (CPU, float32) must give the numbers it gave before device support."""
    a, b = _data(), _data()
    t1 = core.Trainer(_cnn(), a, batch=16)
    t2 = core.Trainer(_cnn(), b, batch=16, device="cpu", amp="off")
    t1.fit(3)
    t2.fit(3)
    for p, q in zip(t1.model.parameters(), t2.model.parameters(), strict=True):
        assert torch.equal(p, q)
    assert "_on_device" not in a  # nothing is copied on the CPU


def test_channels_last_computes_the_same_function():
    d = _data()
    t = core.Trainer(_cnn(), d, batch=16)
    x = d["Xva"]
    t.model.eval()
    ref = t.model(x)
    t.channels_last = True  # what a CUDA device turns on
    t.model.to(memory_format=torch.channels_last)
    xb = t.batch_of(x, torch.arange(len(x)))
    assert xb.is_contiguous(memory_format=torch.channels_last)
    assert torch.allclose(t.model(xb), ref, atol=1e-5)
    t.fit(1)  # trains in that layout


def test_regression_and_hparams_on_the_trainer_device():
    from dlnet import hparams

    d = _data(64, (6,), task="regression")
    torch.manual_seed(0)
    tr = core.Trainer(torch.nn.Linear(6, 1), d, batch=16)
    tr.fit(2)
    gns = hparams.gradient_noise_scale(tr.model, d, b_small=4, b_big=16, n_meas=2)
    assert gns and np.isfinite(gns["grad_sq"])
    cfg = {"wd": 0.0}
    lrt = hparams.lr_range_test(
        lambda c, i, o: torch.nn.Linear(i, o), cfg, d | {"in_dim": 6, "out_dim": 1}, 16, n_iter=10
    )
    assert lrt["n_iter"] > 0


def test_hsi_patch_store_moves_to_the_device():
    rng = np.random.default_rng(0)
    cube = torch.tensor(rng.normal(size=(12, 12, 3)), dtype=torch.float32)
    ps = hsi_data.PatchStore(cube, [2, 3, 4], [5, 6, 7], 3)
    on = ps.to("meta")
    assert on.device.type == "meta" and on[torch.tensor([0, 2])].shape == (2, 3, 3, 3)
    assert ps.device.type == "cpu" and ps.nbytes > cube.nbytes
    assert torch.equal(ps[torch.tensor([1])][0, :, 1, 1], cube[3 + 1, 6 + 1])  # cube is padded by r


@pytest.mark.parametrize("kind,dims,head", [("map", (3, 16, 16), "cls"), ("map", (1, 16, 16), "dense"),
                                            ("seq", (12, 4), "cls"), ("ser", (2, 32), "cls"),
                                            ("vec", (10,), "cls"), ("vol", (1, 6, 8, 8), "cls")])  # fmt: skip
def test_zoo_architectures_create_no_cpu_tensor_in_forward(kind, dims, head):
    """Every sampled architecture runs forward in train and eval mode on the meta device (a GPU
    stand-in): an op creating a tensor without device= (positions, masks, noise) fails here."""
    cands = space.population(kind, dims, 3, 24, seed=1, max_params=2_000_000, head=head)
    assert len(cands) >= 12
    for c in cands:
        torch.manual_seed(0)
        m = zoo.build_model(c).to("meta")
        x = torch.randn(2, *dims, device="meta")
        for mode in (m.train, m.eval):
            mode()
            m(x)
        kl = getattr(m, "kl_loss", None)
        if kl:
            kl()
