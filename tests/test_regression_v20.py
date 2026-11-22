"""Regression tests for the bugs found while adding hyperspectral super-resolution (v2.0.0).

Each test failed on v1.9.0.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

torch = pytest.importorskip("torch")

from dlnet import hparams, seeds


def test_hparams_measures_an_autoencoder_with_its_training_loss():
    """The LR range test and the gradient noise scale used cross-entropy for `reconstruction`
    (the image taken as soft labels): a different loss than the one the model trains with."""
    assert isinstance(hparams._loss_fn({"task": "reconstruction"}), torch.nn.MSELoss)


def test_seed_rule_does_not_print_a_sentinel_class_count():
    """Data without classes printed '(1000000000 per class at least)'."""
    _, why = seeds.trait_weights({"n_train": 5})
    assert "per class" not in why[0]
