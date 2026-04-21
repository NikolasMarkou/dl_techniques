"""Tests for CliffordDetectionLoss — stride-configurable YOLOv12 loss."""

import os

import keras
import numpy as np
import pytest

from dl_techniques.losses.clifford_detection_loss import CliffordDetectionLoss


def _synthetic_batch(
    batch: int = 2,
    input_hw: tuple = (256, 256),
    strides=(8, 16, 32),
    num_classes: int = 7,
    reg_max: int = 8,
    max_gt: int = 5,
):
    """Build synthetic (y_true, y_pred) shapes that match the loss contract.

    y_true: (B, max_gt, 5) = [class_id, x1, y1, x2, y2] xyxy-normalized; empty slots -1.0
    y_pred: (B, total_anchors, 4*reg_max + num_classes) logits
    """
    rng = np.random.default_rng(42)
    H, W = input_hw
    total_anchors = sum((H // s) * (W // s) for s in strides)
    y_pred = rng.standard_normal(
        (batch, total_anchors, 4 * reg_max + num_classes)
    ).astype(np.float32)
    # One real box per image, rest padded.
    y_true = -1.0 * np.ones((batch, max_gt, 5), dtype=np.float32)
    for b in range(batch):
        cls = rng.integers(0, num_classes)
        x1, y1 = rng.uniform(0.0, 0.4, size=2)
        x2, y2 = x1 + rng.uniform(0.1, 0.5), y1 + rng.uniform(0.1, 0.5)
        y_true[b, 0] = [cls, x1, y1, min(x2, 1.0), min(y2, 1.0)]
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_strides(self):
        loss = CliffordDetectionLoss(num_classes=7)
        assert loss._strides_config == (8, 16, 32)

    def test_natural_u_net_strides(self):
        loss = CliffordDetectionLoss(strides=(2, 4, 8), num_classes=7)
        assert loss._strides_config == (2, 4, 8)

    def test_wrong_stride_count_rejected(self):
        with pytest.raises(ValueError, match="exactly 3 strides"):
            CliffordDetectionLoss(strides=(8, 16), num_classes=7)

    def test_zero_or_negative_strides_rejected(self):
        with pytest.raises(ValueError, match="must all be positive"):
            CliffordDetectionLoss(strides=(0, 8, 16), num_classes=7)

    def test_degenerate_feature_map_raises(self):
        # Stride > input size → feature map has 0 height/width.
        with pytest.raises(ValueError, match="degenerate"):
            CliffordDetectionLoss(
                strides=(512, 1024, 2048),  # larger than 256
                input_shape=(256, 256),
                num_classes=7,
            )


# ---------------------------------------------------------------------------
# Anchor layout
# ---------------------------------------------------------------------------


class TestAnchorLayout:
    def test_yolo_default_anchor_count(self):
        """YOLO strides [8,16,32] at 256x256 = 32² + 16² + 8² = 1344 anchors (not 8400 — that's 640px)."""
        loss = CliffordDetectionLoss(
            strides=(8, 16, 32), input_shape=(256, 256), num_classes=7,
        )
        n_anchors = int(loss.anchors.shape[0])
        assert n_anchors == 32 * 32 + 16 * 16 + 8 * 8  # = 1344

    def test_natural_strides_anchor_count(self):
        """Natural U-Net strides [2,4,8] at 256 → 128² + 64² + 32² = 21504."""
        loss = CliffordDetectionLoss(
            strides=(2, 4, 8), input_shape=(256, 256), num_classes=7,
        )
        n_anchors = int(loss.anchors.shape[0])
        assert n_anchors == 128 * 128 + 64 * 64 + 32 * 32  # = 21504

    def test_stride_tensor_matches_anchor_count(self):
        loss = CliffordDetectionLoss(
            strides=(2, 4, 8), input_shape=(128, 128), num_classes=7,
        )
        assert loss.anchors.shape[0] == loss.strides.shape[0]
        # Stride tensor groups: 64² entries of 2, then 32² of 4, then 16² of 8
        stride_np = keras.ops.convert_to_numpy(loss.strides).ravel()
        assert np.allclose(stride_np[: 64 * 64], 2.0)
        assert np.allclose(stride_np[64 * 64 : 64 * 64 + 32 * 32], 4.0)
        assert np.allclose(stride_np[-16 * 16:], 8.0)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForward:
    def test_loss_finite_at_yolo_strides(self):
        num_classes, reg_max = 7, 8
        loss = CliffordDetectionLoss(
            strides=(8, 16, 32), input_shape=(256, 256),
            num_classes=num_classes, reg_max=reg_max,
        )
        y_true, y_pred = _synthetic_batch(
            input_hw=(256, 256), strides=(8, 16, 32),
            num_classes=num_classes, reg_max=reg_max,
        )
        val = loss(y_true, y_pred)
        val_np = float(keras.ops.convert_to_numpy(val))
        assert np.isfinite(val_np), f"loss={val_np}"
        assert val_np > 0.0

    def test_loss_finite_at_natural_strides(self):
        num_classes, reg_max = 7, 8
        loss = CliffordDetectionLoss(
            strides=(2, 4, 8), input_shape=(128, 128),
            num_classes=num_classes, reg_max=reg_max,
        )
        y_true, y_pred = _synthetic_batch(
            input_hw=(128, 128), strides=(2, 4, 8),
            num_classes=num_classes, reg_max=reg_max,
        )
        val = loss(y_true, y_pred)
        val_np = float(keras.ops.convert_to_numpy(val))
        assert np.isfinite(val_np)
        assert val_np > 0.0

    def test_loss_zero_gradient_on_all_pad(self):
        """All-padded batch (no real boxes) should return a finite loss — the
        classification branch still fires against all-negative targets."""
        num_classes, reg_max = 7, 8
        loss = CliffordDetectionLoss(
            strides=(8, 16, 32), input_shape=(256, 256),
            num_classes=num_classes, reg_max=reg_max,
        )
        batch, max_gt = 2, 5
        y_true = -1.0 * np.ones((batch, max_gt, 5), dtype=np.float32)
        total_anchors = 32 * 32 + 16 * 16 + 8 * 8
        y_pred = np.zeros(
            (batch, total_anchors, 4 * reg_max + num_classes), dtype=np.float32
        )
        val = loss(y_true, y_pred)
        val_np = float(keras.ops.convert_to_numpy(val))
        assert np.isfinite(val_np)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip(self):
        loss = CliffordDetectionLoss(
            strides=(2, 4, 8), num_classes=7, reg_max=8, input_shape=(128, 128),
        )
        config = loss.get_config()
        assert config["strides"] == [2, 4, 8]
        reloaded = CliffordDetectionLoss.from_config(config)
        assert reloaded._strides_config == (2, 4, 8)
        assert reloaded.num_classes == 7
        assert reloaded.reg_max == 8
