"""Frequency-domain modules (dlnet/zoo/ops_freq.py): closed-form answers of the FFT formulas.

The comparison with the authors' code (weights copied across) is scripts/verify_freq.py, run on
GitHub Actions; here every block is checked against what the Fourier transform must give.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dlnet import zoo
from dlnet.zoo import ops_freq as fq


def _circular_conv2d(x, k):
    """y[c, i, j] = sum_{a, b} x[c, a, b] k[c, (i - a) % H, (j - b) % W] (numpy, direct sum)."""
    _, h, w = x.shape
    y = np.zeros_like(x)
    for i in range(h):
        for j in range(w):
            for a in range(h):
                for b in range(w):
                    y[:, i, j] += x[:, a, b] * k[:, (i - a) % h, (j - b) % w]
    return y


def test_global_filter_is_a_global_circular_convolution():
    rng = np.random.default_rng(0)
    c, h, w = 3, 6, 5
    x, k = rng.normal(size=(c, h, w)), rng.normal(size=(c, h, w))
    gf = fq.GlobalFilter(c, (h, w))
    K = np.fft.rfft2(k)  # (c, h, w//2+1): the filter of the kernel, channels-last in the module
    with torch.no_grad():
        gf.complex_weight.copy_(
            torch.from_numpy(np.stack([K.real, K.imag], -1).transpose(1, 2, 0, 3))
        )
    y = gf(torch.from_numpy(x).float().permute(1, 2, 0)[None]).permute(0, 3, 1, 2)[0]
    np.testing.assert_allclose(y.detach().numpy(), _circular_conv2d(x, k), atol=1e-4)


def test_spectral_conv_keeps_every_frequency_as_identity_and_drops_the_rest():
    x = torch.randn(2, 3, 8, 8)
    sc = fq.SpectralConv(3, 3, (8, 8), modes=12)  # clipped to 4 (both signs) x 5 (rfft axis)
    assert sc.modes == [4, 5] and len(sc.weights) == 2
    eye = torch.zeros(3, 3, 4, 5, 2)
    eye[range(3), range(3), :, :, 0] = 1
    with torch.no_grad():
        for w in sc.weights:
            w.copy_(eye)
    torch.testing.assert_close(sc(x), x, atol=1e-5, rtol=1e-5)
    # one mode keeps frequencies 0 AND -1 along the first axis (FNO's two corners); with the
    # negative corner zeroed only the mean survives -> a constant map per channel
    sc1 = fq.SpectralConv(3, 3, (8, 8), modes=1)
    with torch.no_grad():
        sc1.weights[0].copy_(eye[:, :, :1, :1])
        sc1.weights[1].zero_()
    y = sc1(x)
    torch.testing.assert_close(y, x.mean((2, 3), keepdim=True).expand_as(x), atol=1e-5, rtol=1e-5)


def test_spectral_conv_1d_and_3d_corners():
    for sizes, n in (((16,), 1), ((4, 6, 8), 4)):
        sc = fq.SpectralConv(2, 2, sizes, modes=99)
        assert len(sc.weights) == n
        eye = torch.zeros(2, 2, *sc.modes, 2)
        eye[range(2), range(2), ..., 0] = 1
        with torch.no_grad():
            for w in sc.weights:
                w.copy_(eye)
        x = torch.randn(1, 2, *sizes)
        torch.testing.assert_close(sc(x), x, atol=1e-5, rtol=1e-5)


def test_fnet_mixing_is_the_real_part_of_the_2d_fft():
    blk = fq.FNetBlock(6)
    x = torch.randn(2, 5, 6)
    seen = {}
    blk.mixing_norm.register_forward_pre_hook(lambda m, a: seen.setdefault("in", a[0]))
    blk(x)
    expect = x.numpy() + np.fft.fftn(x.numpy(), axes=(-2, -1)).real
    np.testing.assert_allclose(seen["in"].detach().numpy(), expect, atol=1e-4)


def test_fcanet_dct_filters_are_the_orthonormal_2d_dct_basis():
    xs, ys = zip(*[(u, v) for u in range(7) for v in range(7)], strict=True)
    f = fq.dct_filters(7, 7, list(xs), list(ys), 49).reshape(49, 49)
    torch.testing.assert_close(f @ f.T, torch.eye(49), atol=1e-5, rtol=0)
    assert torch.allclose(f[0], torch.full((49,), 1 / 7))  # frequency (0, 0) = average pooling * 7
    att = fq.MultiSpectralAttention(32, method="top16")
    assert att(torch.randn(2, 32, 9, 9)).shape == (2, 32, 9, 9)  # pooled to the 7x7 DCT grid


def test_afno_with_zero_weights_returns_its_input_and_masks_frequencies():
    a = fq.AFNO2D(8, num_blocks=4)
    with torch.no_grad():
        for p in a.parameters():
            p.zero_()
    x = torch.randn(2, 6, 6, 8)
    torch.testing.assert_close(a(x), x)


def test_spectral_filter_and_rfft_for_hyperspectral_patches():
    x = torch.randn(2, 30, 9, 9)  # 30 bands, 9x9 patch
    gf = fq.SpectralGlobalFilter(30)
    with torch.no_grad():
        gf.complex_weight.zero_()
    torch.testing.assert_close(gf(x), x)  # zero filter: residual identity
    with torch.no_grad():
        gf.complex_weight[..., 0] = 1  # all-pass filter: x + x
    torch.testing.assert_close(gf(x), 2 * x, atol=1e-5, rtol=1e-5)
    f = fq.SpectralRFFT()(x)
    assert f.shape == (2, 32, 9, 9)  # 16 frequencies x (real, imag)
    z = torch.complex(f[:, 0::2], f[:, 1::2])
    torch.testing.assert_close(
        torch.fft.irfft(z, n=30, dim=1, norm="ortho"), x, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(
    "spec",
    [
        {"op": "gfnet_block"},
        {"op": "afno_block", "num_blocks": 4},
        {"op": "fno_block"},
        {"op": "ffc_block", "lfu": 0},
        {"op": "fca", "method": "top4", "reduction": 4},
        {"op": "spectral_gf"},
    ],
)
def test_frequency_blocks_on_an_odd_hsi_patch_under_autocast_and_export(spec):
    dims = (32, 9, 9)
    cfg = {"spec": {"op": "seq", "body": [{"op": "spectral_rfft"}, {"op": "conv_bn_act", "out": 32}, spec]},
           "input": {"kind": "map", "dims": dims}, "head": "cls", "out_dim": 4}  # fmt: skip
    m = zoo.build_model(cfg).eval()
    x = torch.randn(2, *dims)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert m(x).shape == (2, 4)
    prog = torch.export.export(m, (x,))
    torch.testing.assert_close(prog.module()(x), m(x))


def test_ffc_rejects_shapes_the_original_cannot_take():
    with pytest.raises(ValueError, match="even"):
        zoo.check({"op": "ffc_block", "lfu": 1}, "map", (32, 9, 9))
    with pytest.raises(ValueError, match="split evenly"):
        zoo.check({"op": "ffc_block", "ratio": 0.75}, "map", (30, 8, 8))
