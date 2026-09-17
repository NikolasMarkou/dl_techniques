"""Lock-in tests for ``train.vit.train_vit`` iter-2 invariant assertion.

DECISION plan_2026-05-12_f2d29729/D-007: these tests pin
``_assert_train_val_distribution_match`` against the iter-1 cliff-fingerprint
bug class. If the test stops raising on a |Δmean|≈2.0 mismatch, the guard is
silently broken — re-investigate `_cifar_augment` / `_cifar_normalize` ordering.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from train.vit.train_vit import TrainingConfig, _assert_train_val_distribution_match


def _make_ds(mean: float, std: float, *, shape=(8, 32, 32, 3), seed: int = 0):
    """Build a single-batch tf.data.Dataset of synthetic images with target stats."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(loc=mean, scale=std, size=shape).astype("float32")
    labels = np.zeros((shape[0],), dtype="int32")
    return tf.data.Dataset.from_tensors((tf.constant(arr), tf.constant(labels)))


def test_assert_train_val_distribution_match_raises_on_mismatch() -> None:
    """Iter-1 bug fingerprint: train mean≈+2.0, val mean≈0.0 → must raise."""
    train_ds = _make_ds(mean=2.0, std=1.0, seed=1)
    val_ds = _make_ds(mean=0.0, std=1.0, seed=2)
    with pytest.raises(RuntimeError, match="distribution mismatch"):
        _assert_train_val_distribution_match(train_ds, val_ds)


def test_assert_train_val_distribution_match_passes_on_match() -> None:
    """Healthy pipeline: train and val both mean≈0, std≈1 → must not raise."""
    train_ds = _make_ds(mean=0.0, std=1.0, seed=11)
    val_ds = _make_ds(mean=0.0, std=1.0, seed=22)
    # Should return None without raising.
    assert _assert_train_val_distribution_match(train_ds, val_ds) is None


def test_training_config_rejects_nonzero_label_smoothing() -> None:
    """DECISION plan-2026-09-17T032714-403de954/D-003: SparseCategoricalCrossentropy
    has no label_smoothing support, so a non-zero value must hard-raise at
    config-validation time rather than being silently ignored.
    """
    with pytest.raises(ValueError, match="label_smoothing"):
        TrainingConfig(label_smoothing=0.1)


def test_training_config_default_label_smoothing_is_zero_and_does_not_raise() -> None:
    """Default invocation (no --label-smoothing flag) must not raise."""
    config = TrainingConfig()
    assert config.label_smoothing == 0.0
