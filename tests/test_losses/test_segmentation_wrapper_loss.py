"""Tests for `SegmentationWrapperLoss`.

Mirrors the module name per F-004 (sibling-loss conventions): one
test file per loss module. Validates the contract the closure-based
`WrappedLoss` could not satisfy: lossless `get_config`/`from_config`
round-trips, full Keras model save/load round-trips without
`custom_objects` or `compile=False`, reduction forwarding, and class
identity stability.
"""

import os
import tempfile
from typing import Tuple

import keras
import numpy as np
import pytest

from dl_techniques.losses.segmentation_wrapper_loss import (
    SegmentationWrapperLoss,
    create_segmentation_wrapper_loss,
)
from dl_techniques.losses.segmentation_loss import LossConfig


# ---------------------------------------------------------------------
# Constants and fixtures
# ---------------------------------------------------------------------

LOSS_NAMES = [
    "cross_entropy",
    "dice",
    "focal",
    "tversky",
    "focal_tversky",
    "lovasz",
    "combo",
    "boundary",
    "hausdorff",
]


@pytest.fixture(scope="module")
def seg_data() -> Tuple[np.ndarray, np.ndarray]:
    """Small (2, 16, 16, 3) one-hot y_true and softmax y_pred."""
    rng = np.random.default_rng(seed=0)
    h, w, c = 16, 16, 3
    labels = rng.integers(low=0, high=c, size=(2, h, w))
    y_true = np.eye(c, dtype=np.float32)[labels]  # (2,16,16,3)
    logits = rng.standard_normal(size=(2, h, w, c)).astype(np.float32)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    y_pred = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    return y_true, y_pred


# ---------------------------------------------------------------------
# Test 1 — LossConfig serialization
# ---------------------------------------------------------------------


def test_loss_config_serialization() -> None:
    """LossConfig.get_config returns the full dict; from_config reconstructs."""
    original = LossConfig(num_classes=5, focal_gamma=3.0)
    cfg = original.get_config()
    assert cfg["num_classes"] == 5
    assert cfg["focal_gamma"] == 3.0
    # Every dataclass field must be present.
    assert set(cfg.keys()) >= {
        "num_classes",
        "smooth_factor",
        "focal_gamma",
        "focal_alpha",
        "tversky_alpha",
        "tversky_beta",
        "focal_tversky_gamma",
        "combo_alpha",
        "combo_beta",
        "boundary_theta",
    }
    rebuilt = LossConfig.from_config(cfg)
    assert rebuilt == original


# ---------------------------------------------------------------------
# Test 2 — Class identity
# ---------------------------------------------------------------------


def test_class_identity() -> None:
    """All factory-built instances share the single module-level class."""
    a = create_segmentation_wrapper_loss("dice")
    b = create_segmentation_wrapper_loss("focal")
    assert type(a) is SegmentationWrapperLoss
    assert type(b) is SegmentationWrapperLoss
    assert type(a) is type(b)


# ---------------------------------------------------------------------
# Test 3 — Unknown loss name
# ---------------------------------------------------------------------


def test_unknown_loss_name_raises() -> None:
    """Constructor raises ValueError on unknown loss name."""
    with pytest.raises(ValueError, match="Unknown loss function"):
        SegmentationWrapperLoss("not_a_real_loss")


# ---------------------------------------------------------------------
# Test 4 — Config round-trip (parametrized over all 9 names)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("loss_name", LOSS_NAMES)
def test_config_round_trip(loss_name: str) -> None:
    """get_config/from_config losslessly round-trips for every supported name."""
    cfg = LossConfig(num_classes=3)
    loss = SegmentationWrapperLoss(loss_name, cfg)
    serialized = loss.get_config()
    loss2 = SegmentationWrapperLoss.from_config(serialized)
    assert loss2.loss_name == loss_name
    assert loss2.config == cfg
    assert loss2.name == loss.name
    assert loss2.reduction == loss.reduction


# ---------------------------------------------------------------------
# Test 5 — Reduction forwarding
# ---------------------------------------------------------------------


def test_reduction_forwarding() -> None:
    """Explicit reduction kwarg is forwarded and survives round-trip."""
    loss = SegmentationWrapperLoss(
        "dice", LossConfig(num_classes=2), reduction="sum"
    )
    assert loss.reduction == "sum"
    loss2 = SegmentationWrapperLoss.from_config(loss.get_config())
    assert loss2.reduction == "sum"


# ---------------------------------------------------------------------
# Test 6 — Full model round-trip (parametrized)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("loss_name", LOSS_NAMES)
def test_full_model_round_trip(
    loss_name: str, seg_data: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Compiled model saves & reloads without custom_objects/compile=False."""
    y_true, y_pred = seg_data
    num_classes = y_true.shape[-1]
    cfg = LossConfig(num_classes=num_classes)
    original_loss = SegmentationWrapperLoss(loss_name, cfg)

    inputs = keras.Input(shape=y_pred.shape[1:])
    outputs = keras.layers.Conv2D(
        num_classes, 1, activation="softmax", padding="same"
    )(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss=original_loss)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"model_{loss_name}.keras")
        model.save(path)
        loaded = keras.models.load_model(path)  # NO custom_objects, NO compile=False

    assert type(loaded.loss).__name__ == "SegmentationWrapperLoss"
    assert loaded.loss.loss_name == loss_name

    orig_val = float(original_loss(y_true, y_pred))
    loaded_val = float(loaded.loss(y_true, y_pred))
    assert abs(orig_val - loaded_val) < 1e-6, (orig_val, loaded_val)


# ---------------------------------------------------------------------
# Test 7 — Default config
# ---------------------------------------------------------------------


def test_default_config() -> None:
    """Constructing without an explicit config defaults to LossConfig(num_classes=1)."""
    loss = SegmentationWrapperLoss("dice")
    assert loss.config.num_classes == 1
    assert isinstance(loss.config, LossConfig)
