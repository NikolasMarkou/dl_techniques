"""Tests for MambaLCTBoxLoss: shapes, values, and serialization."""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.losses.mambalct_loss import MambaLCTBoxLoss


def test_per_sample_shape_and_weights() -> None:
    loss = MambaLCTBoxLoss(l1_weight=5.0, giou_weight=2.0)
    y_true = np.zeros((2, 2, 4), dtype=np.float32)
    y_true[..., 2:] = 0.5
    y_pred = np.zeros((2, 2, 4), dtype=np.float32)
    y_pred[..., 2:] = 0.5
    values = np.asarray(loss.call(y_true, y_pred))
    assert values.shape == (2,)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)


def test_perfect_boxes_give_zero() -> None:
    loss = MambaLCTBoxLoss()
    rng = np.random.default_rng(0)
    boxes = rng.uniform(0.2, 0.8, size=(3, 2, 4)).astype(np.float32)
    values = np.asarray(loss.call(boxes, boxes))
    # GIoU of identical boxes is 1.0 up to float32 union-eps noise.
    np.testing.assert_allclose(values, np.zeros(3), atol=1e-5)


def test_rows_are_independent() -> None:
    loss = MambaLCTBoxLoss()
    rng = np.random.default_rng(1)
    y_true = rng.uniform(0.2, 0.8, size=(3, 2, 4)).astype(np.float32)
    y_pred = rng.uniform(0.2, 0.8, size=(3, 2, 4)).astype(np.float32)
    full = np.asarray(loss.call(y_true, y_pred))
    assert full.shape == (3,)
    partial = np.asarray(loss.call(y_true[1:2], y_pred[1:2]))
    np.testing.assert_allclose(full[1:2], partial, atol=0.0)


def test_serialization_round_trip() -> None:
    loss = MambaLCTBoxLoss(l1_weight=3.0, giou_weight=1.0)
    config = loss.get_config()
    restored = MambaLCTBoxLoss.from_config(config)
    assert restored.l1_weight == 3.0
    assert restored.giou_weight == 1.0


def test_disjoint_boxes_known_value() -> None:
    # (cx, cy, w, h): true x in [0.3, 0.7], pred x in [0.8, 1.2].
    # Disjoint, so IoU = 0; enclosing box x in [0.3, 1.2] (w=0.9, h=0.4,
    # C=0.36), union = 0.32, GIoU = -(0.36-0.32)/0.36.
    y_true = np.array([[[0.5, 0.5, 0.4, 0.4]]], dtype=np.float32)
    y_pred = np.array([[[1.0, 0.5, 0.4, 0.4]]], dtype=np.float32)
    l1 = float(np.mean(np.abs(y_true - y_pred)))
    giou = -(0.36 - 0.32) / 0.36
    expected = 5.0 * l1 + 2.0 * (1.0 - giou)
    value = float(np.asarray(MambaLCTBoxLoss().call(y_true, y_pred))[0])
    np.testing.assert_allclose(value, expected, atol=1e-5, rtol=0)
    assert value > 2.0  # clearly positive, not a near-zero residual


def test_weights_are_live() -> None:
    rng = np.random.default_rng(2)
    y_true = rng.uniform(0.2, 0.8, size=(2, 2, 4)).astype(np.float32)
    y_pred = rng.uniform(0.2, 0.8, size=(2, 2, 4)).astype(np.float32)
    base = np.asarray(MambaLCTBoxLoss().call(y_true, y_pred))
    halved_l1 = np.asarray(
        MambaLCTBoxLoss(l1_weight=2.5, giou_weight=2.0).call(y_true, y_pred)
    )
    assert not np.allclose(base, halved_l1, atol=1e-7, rtol=0)
    halved_giou = np.asarray(
        MambaLCTBoxLoss(l1_weight=5.0, giou_weight=1.0).call(y_true, y_pred)
    )
    assert not np.allclose(base, halved_giou, atol=1e-7, rtol=0)


def test_negative_weights_raise() -> None:
    with pytest.raises(ValueError):
        MambaLCTBoxLoss(l1_weight=-1.0)
    with pytest.raises(ValueError):
        MambaLCTBoxLoss(giou_weight=-1.0)
