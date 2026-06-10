"""Tests for CoverageMetric and SharpnessMetric.

The metrics consume quantile predictions of shape ``[B, H, F, Q]`` (with an
explicit feature axis) or ``[B, H, Q]`` (no feature axis, the tirex output
layout). ``low_index``/``high_index`` select the two quantile slices along the
LAST axis bounding the central interval.
"""

import numpy as np
import pytest
import keras

from dl_techniques.metrics.probabilistic_forecast_metrics import (
    CoverageMetric,
    SharpnessMetric,
)


# ---------------------------------------------------------------------
# Fixtures: deterministic quantile tensors. Quantile axis layout used
# throughout: [low, median, high] -> low_index=0, high_index=2.
# ---------------------------------------------------------------------


@pytest.fixture
def low_high_indices():
    return {"low_index": 0, "high_index": 2}


def _make_pred(low, median, high, shape):
    """Build a [*shape, 3] quantile tensor with constant low/median/high."""
    base = np.empty(shape + (3,), dtype="float32")
    base[..., 0] = low
    base[..., 1] = median
    base[..., 2] = high
    return base


class TestCoverageMetric:
    """Tests for CoverageMetric (empirical interval coverage)."""

    def test_all_inside_is_one(self, low_high_indices):
        # interval [-10, 10] bounds every target -> coverage 1.0
        shape = (4, 6, 2)  # [B, H, F]
        y_true = np.zeros(shape, dtype="float32")
        y_pred = _make_pred(-10.0, 0.0, 10.0, shape)
        m = CoverageMetric(**low_high_indices)
        m.update_state(y_true, y_pred)
        assert abs(float(m.result()) - 1.0) < 1e-6

    def test_all_outside_is_zero(self, low_high_indices):
        # interval [-10, -5] excludes target 0.0 -> coverage 0.0
        shape = (4, 6, 2)
        y_true = np.zeros(shape, dtype="float32")
        y_pred = _make_pred(-10.0, -7.0, -5.0, shape)
        m = CoverageMetric(**low_high_indices)
        m.update_state(y_true, y_pred)
        assert abs(float(m.result()) - 0.0) < 1e-6

    def test_partial_half_inside_is_half(self, low_high_indices):
        # Half the targets are 0.0 (inside [-1,1]), half are 100.0 (outside).
        shape = (2, 4, 2)  # 16 elements; first half rows inside
        y_true = np.zeros(shape, dtype="float32")
        y_true[1, :, :] = 100.0  # second batch row entirely outside
        y_pred = _make_pred(-1.0, 0.0, 1.0, shape)
        m = CoverageMetric(**low_high_indices)
        m.update_state(y_true, y_pred)
        assert abs(float(m.result()) - 0.5) < 1e-6

    def test_boundary_inclusive(self, low_high_indices):
        # Targets exactly on the bounds count as inside (>=, <=).
        shape = (1, 2, 1)
        y_pred = _make_pred(-1.0, 0.0, 1.0, shape)
        y_true = np.array([[[1.0], [-1.0]]], dtype="float32")
        m = CoverageMetric(**low_high_indices)
        m.update_state(y_true, y_pred)
        assert abs(float(m.result()) - 1.0) < 1e-6

    def test_reset_state(self, low_high_indices):
        shape = (4, 6, 2)
        y_true = np.zeros(shape, dtype="float32")
        inside = _make_pred(-10.0, 0.0, 10.0, shape)
        m = CoverageMetric(**low_high_indices)
        m.update_state(y_true, inside)
        assert float(m.result()) > 0.0

        m.reset_state()
        assert abs(float(m.result()) - 0.0) < 1e-6

        # Fresh update gives the correct fresh value.
        outside = _make_pred(-10.0, -7.0, -5.0, shape)
        m.update_state(y_true, outside)
        assert abs(float(m.result()) - 0.0) < 1e-6

    def test_get_config_roundtrip(self):
        m = CoverageMetric(low_index=1, high_index=3, name="cov80")
        cfg = m.get_config()
        assert cfg["low_index"] == 1
        assert cfg["high_index"] == 3

        restored = keras.metrics.deserialize(keras.metrics.serialize(m))
        assert isinstance(restored, CoverageMetric)
        assert restored.get_config() == m.get_config()

        # Same data -> same result. Quantile axis with 4 levels; bounds 1 and 3.
        shape = (2, 3, 2)
        y_true = np.zeros(shape, dtype="float32")
        y_pred = np.zeros(shape + (4,), dtype="float32")
        y_pred[..., 1] = -2.0  # low bound
        y_pred[..., 3] = 2.0   # high bound -> inside
        m.update_state(y_true, y_pred)
        restored.update_state(y_true, y_pred)
        assert abs(float(m.result()) - float(restored.result())) < 1e-6
        assert abs(float(m.result()) - 1.0) < 1e-6

    def test_sample_weight_zeros_out_outside(self, low_high_indices):
        # 2 batch rows: row 0 inside, row 1 outside. Zero-weight the outside
        # samples -> coverage rises to 1.0 (only the inside samples count).
        shape = (2, 4, 2)
        y_true = np.zeros(shape, dtype="float32")
        y_true[1, :, :] = 100.0  # outside
        y_pred = _make_pred(-1.0, 0.0, 1.0, shape)
        sw = np.ones((2, 4, 2), dtype="float32")
        sw[1, :, :] = 0.0  # ignore the outside row

        m = CoverageMetric(**low_high_indices)
        m.update_state(y_true, y_pred, sample_weight=sw)
        assert abs(float(m.result()) - 1.0) < 1e-6

    def test_3d_no_feature_axis(self, low_high_indices):
        # [B, H, Q] layout (tirex output, no feature axis).
        shape = (3, 5)  # [B, H]
        y_true = np.zeros(shape, dtype="float32")
        y_pred = _make_pred(-1.0, 0.0, 1.0, shape)  # -> [3, 5, 3]
        assert y_pred.shape == (3, 5, 3)
        m = CoverageMetric(**low_high_indices)
        m.update_state(y_true, y_pred)
        assert abs(float(m.result()) - 1.0) < 1e-6


class TestSharpnessMetric:
    """Tests for SharpnessMetric (mean interval width)."""

    def test_known_width_two(self, low_high_indices):
        # high - low = 1 - (-1) = 2.0 everywhere.
        shape = (4, 6, 2)
        y_true = np.zeros(shape, dtype="float32")
        y_pred = _make_pred(-1.0, 0.0, 1.0, shape)
        m = SharpnessMetric(**low_high_indices)
        m.update_state(y_true, y_pred)
        assert abs(float(m.result()) - 2.0) < 1e-6

    def test_known_width_four(self, low_high_indices):
        # high - low = 2 - (-2) = 4.0 everywhere.
        shape = (4, 6, 2)
        y_true = np.zeros(shape, dtype="float32")
        y_pred = _make_pred(-2.0, 0.0, 2.0, shape)
        m = SharpnessMetric(**low_high_indices)
        m.update_state(y_true, y_pred)
        assert abs(float(m.result()) - 4.0) < 1e-6

    def test_reset_state(self, low_high_indices):
        shape = (4, 6, 2)
        y_true = np.zeros(shape, dtype="float32")
        y_pred = _make_pred(-1.0, 0.0, 1.0, shape)
        m = SharpnessMetric(**low_high_indices)
        m.update_state(y_true, y_pred)
        assert abs(float(m.result()) - 2.0) < 1e-6

        m.reset_state()
        assert abs(float(m.result()) - 0.0) < 1e-6

        # Fresh update with a different width gives the fresh value.
        y_pred2 = _make_pred(-2.0, 0.0, 2.0, shape)
        m.update_state(y_true, y_pred2)
        assert abs(float(m.result()) - 4.0) < 1e-6

    def test_get_config_roundtrip(self):
        m = SharpnessMetric(low_index=1, high_index=3, name="sharp80")
        cfg = m.get_config()
        assert cfg["low_index"] == 1
        assert cfg["high_index"] == 3

        restored = keras.metrics.deserialize(keras.metrics.serialize(m))
        assert isinstance(restored, SharpnessMetric)
        assert restored.get_config() == m.get_config()

        shape = (2, 3, 2)
        y_true = np.zeros(shape, dtype="float32")
        y_pred = np.zeros(shape + (4,), dtype="float32")
        y_pred[..., 1] = -1.5  # low bound
        y_pred[..., 3] = 1.5   # high bound -> width 3.0
        m.update_state(y_true, y_pred)
        restored.update_state(y_true, y_pred)
        assert abs(float(m.result()) - float(restored.result())) < 1e-6
        assert abs(float(m.result()) - 3.0) < 1e-6

    def test_sample_weight_weighted_mean(self, low_high_indices):
        # Two rows with widths 2.0 and 4.0. Zero-weight the width-4 row ->
        # weighted mean width = 2.0.
        shape = (2, 4, 2)
        y_true = np.zeros(shape, dtype="float32")
        y_pred = np.empty(shape + (3,), dtype="float32")
        y_pred[0, ..., 0] = -1.0
        y_pred[0, ..., 2] = 1.0   # width 2.0
        y_pred[1, ..., 0] = -2.0
        y_pred[1, ..., 2] = 2.0   # width 4.0
        y_pred[..., 1] = 0.0
        sw = np.ones((2, 4, 2), dtype="float32")
        sw[1, :, :] = 0.0

        m = SharpnessMetric(**low_high_indices)
        m.update_state(y_true, y_pred, sample_weight=sw)
        assert abs(float(m.result()) - 2.0) < 1e-6

    def test_3d_no_feature_axis(self, low_high_indices):
        # [B, H, Q] layout (tirex output, no feature axis).
        shape = (3, 5)  # [B, H]
        y_true = np.zeros(shape, dtype="float32")
        y_pred = _make_pred(-1.0, 0.0, 1.0, shape)  # -> [3, 5, 3], width 2.0
        m = SharpnessMetric(**low_high_indices)
        m.update_state(y_true, y_pred)
        assert abs(float(m.result()) - 2.0) < 1e-6
