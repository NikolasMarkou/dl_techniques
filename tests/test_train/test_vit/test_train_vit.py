"""Tests for ``train.vit.train_vit``'s configuration.

The iter-1 cliff (D-006/D-007: augmenting after normalizing and clipping saturated the
pixels and made train and val disagree) was guarded here by a runtime train/val
distribution check. Both splits now come out of ``train.common.classification_data``,
which normalizes every pipeline with the same function after the augmentation hook;
``tests/test_train/test_common_classification_data.py`` pins that order
(``test_the_augment_hook_sees_unit_pixels_before_normalization``).
"""

from __future__ import annotations

import pytest

from train.vit.train_vit import TrainingConfig


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
