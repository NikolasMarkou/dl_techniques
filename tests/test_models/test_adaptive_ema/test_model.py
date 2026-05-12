"""Tests for ``dl_techniques.models.adaptive_ema``.

Coverage floor for the adaptive EMA slope-filter model: initialization,
ctor validation, forward pass (2D and 3D inputs), quantile head shape,
soft/hard signal semantics, ``.keras`` serialization round-trip, gradient
flow through both threshold scalars, factory function, and the
zero-trainable-params guarantee for Mode A.
"""

import os
import tempfile
from typing import Any, Dict

import numpy as np
import pytest
import keras
from keras import ops
import tensorflow as tf

from dl_techniques.models.adaptive_ema import (
    AdaptiveEMASlopeFilterModel,
    create_adaptive_ema_slope_filter,
)


class TestAdaptiveEMASlopeFilterModel:
    """End-to-end test suite for ``AdaptiveEMASlopeFilterModel``."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        return {
            "ema_period": 10,
            "lookback_period": 5,
            "initial_upper_threshold": 1.5,
            "initial_lower_threshold": -1.5,
            "learnable_thresholds": False,
            "slope_softness": 1.0,
        }

    @pytest.fixture
    def learnable_config(self) -> Dict[str, Any]:
        return {
            "ema_period": 10,
            "lookback_period": 5,
            "initial_upper_threshold": 1.0,
            "initial_lower_threshold": -1.0,
            "learnable_thresholds": True,
            "slope_softness": 0.5,
        }

    @pytest.fixture
    def quantile_config(self) -> Dict[str, Any]:
        return {
            "ema_period": 10,
            "lookback_period": 5,
            "quantile_head_config": {"num_quantiles": 9},
        }

    @pytest.fixture
    def sample_2d(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return np.cumsum(rng.standard_normal((4, 64)), axis=1).astype("float32")

    @pytest.fixture
    def sample_3d(self) -> np.ndarray:
        rng = np.random.default_rng(1)
        return np.cumsum(
            rng.standard_normal((4, 64, 3)), axis=1
        ).astype("float32")

    # ------------------------------------------------------------------
    # Initialization + ctor validation
    # ------------------------------------------------------------------

    def test_initialization_basic(self, basic_config):
        model = AdaptiveEMASlopeFilterModel(**basic_config)
        assert model.ema_period == basic_config["ema_period"]
        assert model.lookback_period == basic_config["lookback_period"]
        assert (
            model.initial_upper_threshold
            == basic_config["initial_upper_threshold"]
        )
        assert (
            model.initial_lower_threshold
            == basic_config["initial_lower_threshold"]
        )
        assert model.learnable_thresholds is False
        assert model.slope_softness == basic_config["slope_softness"]
        assert model.quantile_head_config is None
        assert model.slope_featurizer is None
        assert model.quantile_head is None

    def test_initialization_with_quantile_head(self, quantile_config):
        model = AdaptiveEMASlopeFilterModel(**quantile_config)
        assert model.quantile_head is not None
        assert model.slope_featurizer is not None
        assert model.slope_featurizer.filters == model.slope_feature_dim
        assert model.slope_featurizer.kernel_size == (
            model.slope_feature_kernel,
        )

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_invalid_slope_softness_raises(self, bad):
        with pytest.raises(ValueError, match="slope_softness"):
            AdaptiveEMASlopeFilterModel(slope_softness=bad)

    def test_invalid_threshold_order_raises(self):
        with pytest.raises(
            ValueError, match="initial_upper_threshold"
        ):
            AdaptiveEMASlopeFilterModel(
                initial_upper_threshold=-1.0,
                initial_lower_threshold=1.0,
            )
        with pytest.raises(
            ValueError, match="initial_upper_threshold"
        ):
            AdaptiveEMASlopeFilterModel(
                initial_upper_threshold=0.0,
                initial_lower_threshold=0.0,
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def test_forward_pass_2d(self, basic_config, sample_2d):
        model = AdaptiveEMASlopeFilterModel(**basic_config)
        out = model(sample_2d, training=False)
        for key in (
            "ema", "slope", "signal_above", "signal_below", "signal_between",
        ):
            assert key in out, f"missing key: {key}"
            assert tuple(out[key].shape) == sample_2d.shape, (
                f"{key} shape {tuple(out[key].shape)} != {sample_2d.shape}"
            )

    def test_forward_pass_3d(self, basic_config, sample_3d):
        model = AdaptiveEMASlopeFilterModel(**basic_config)
        out = model(sample_3d, training=False)
        for key in (
            "ema", "slope", "signal_above", "signal_below", "signal_between",
        ):
            assert key in out
            assert tuple(out[key].shape) == sample_3d.shape

    def test_forward_pass_with_quantile_head(self, quantile_config, sample_2d):
        model = AdaptiveEMASlopeFilterModel(**quantile_config)
        out = model(sample_2d, training=False)
        assert "slope_quantiles" in out
        assert tuple(out["slope_quantiles"].shape) == (4, 64, 9)

    # ------------------------------------------------------------------
    # Signal semantics
    # ------------------------------------------------------------------

    def test_soft_signals_in_unit_interval(self, learnable_config, sample_2d):
        """Soft signals must live in [0, 1] per-channel.

        The sum constraint that holds for hard signals (above+below+between
        == 1) does NOT hold for the soft formulation: each signal is an
        independent sigmoid membership function. At the threshold boundary
        ``slope == upper`` for example we have ``above = 0.5`` and
        ``between ≈ 0.5``, so the sum can exceed 1. The per-channel [0, 1]
        bound is the primary contract; the hard partition is asserted
        separately by ``test_hard_signals_exact_binary``.
        """
        model = AdaptiveEMASlopeFilterModel(**learnable_config)
        out = model(sample_2d, training=True)
        for key in ("signal_above", "signal_below", "signal_between"):
            arr = ops.convert_to_numpy(out[key])
            assert np.all(arr >= 0.0), f"{key} below 0"
            assert np.all(arr <= 1.0), f"{key} above 1"

    def test_hard_signals_exact_binary(self, basic_config, sample_2d):
        model = AdaptiveEMASlopeFilterModel(**basic_config)
        out = model(sample_2d, training=False)
        for key in ("signal_above", "signal_below", "signal_between"):
            arr = ops.convert_to_numpy(out[key])
            unique = np.unique(arr)
            assert set(unique.tolist()).issubset({0.0, 1.0}), (
                f"{key} values not binary: {unique}"
            )
        total = (
            ops.convert_to_numpy(out["signal_above"])
            + ops.convert_to_numpy(out["signal_below"])
            + ops.convert_to_numpy(out["signal_between"])
        )
        assert np.allclose(total, 1.0), (
            f"hard signal partition violated: total range "
            f"[{total.min()}, {total.max()}]"
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_serialization_round_trip(self, learnable_config, sample_2d):
        model = AdaptiveEMASlopeFilterModel(**learnable_config)
        # Build the model.
        out_before = model(sample_2d, training=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adaptive_ema.keras")
            model.save(path)
            restored = keras.saving.load_model(path)
        out_after = restored(sample_2d, training=False)
        for key in out_before:
            a = ops.convert_to_numpy(out_before[key])
            b = ops.convert_to_numpy(out_after[key])
            assert np.allclose(a, b, atol=1e-5), (
                f"round-trip mismatch on key '{key}': "
                f"max diff = {np.abs(a - b).max()}"
            )

    # ------------------------------------------------------------------
    # Gradient flow
    # ------------------------------------------------------------------

    def test_gradient_flow_learnable_thresholds(
        self, learnable_config, sample_2d
    ):
        model = AdaptiveEMASlopeFilterModel(**learnable_config)
        x = tf.constant(sample_2d)
        with tf.GradientTape() as tape:
            out = model(x, training=True)
            loss = tf.reduce_mean(out["signal_between"])
        grads = tape.gradient(loss, [model.midpoint_var, model.log_half_range_var])
        assert grads[0] is not None, "no grad on midpoint_var"
        assert grads[1] is not None, "no grad on log_half_range_var"
        assert float(tf.reduce_sum(tf.abs(grads[0]))) > 0.0, (
            "midpoint_var grad is exactly zero"
        )
        assert float(tf.reduce_sum(tf.abs(grads[1]))) > 0.0, (
            "log_half_range_var grad is exactly zero"
        )

    # ------------------------------------------------------------------
    # Factory + trainable-surface promises
    # ------------------------------------------------------------------

    def test_factory_function(self, sample_2d):
        model = create_adaptive_ema_slope_filter(
            ema_period=10,
            lookback_period=5,
            initial_upper_threshold=1.5,
            initial_lower_threshold=-1.5,
        )
        assert isinstance(model, AdaptiveEMASlopeFilterModel)
        out = model(sample_2d, training=False)
        assert "signal_between" in out
        assert tuple(out["signal_between"].shape) == sample_2d.shape

    def test_zero_trainable_params_mode_a(self, basic_config, sample_2d):
        model = AdaptiveEMASlopeFilterModel(**basic_config)
        # Force build.
        _ = model(sample_2d, training=False)
        assert len(model.trainable_weights) == 0, (
            f"expected 0 trainable weights in Mode A, got "
            f"{len(model.trainable_weights)}: "
            f"{[w.name for w in model.trainable_weights]}"
        )

    # ------------------------------------------------------------------
    # I-18 — quantile head rejects multi-feature inputs (D-002).
    # ------------------------------------------------------------------

    def test_quantile_head_rejects_multifeature_input(self):
        """``call()`` raises ValueError on (B, T, F>1) when quantile head set."""
        model = AdaptiveEMASlopeFilterModel(
            ema_period=10,
            lookback_period=5,
            quantile_head_config={"num_quantiles": 5},
        )
        x = np.random.randn(2, 16, 3).astype(np.float32)
        with pytest.raises(ValueError, match="multi-feature inputs"):
            _ = model(ops.convert_to_tensor(x), training=False)

    def test_quantile_head_accepts_single_feature_3d(self):
        """``(B, T, 1)`` must still work with quantile head."""
        model = AdaptiveEMASlopeFilterModel(
            ema_period=10,
            lookback_period=5,
            quantile_head_config={"num_quantiles": 5},
        )
        x = np.random.randn(2, 16, 1).astype(np.float32)
        out = model(ops.convert_to_tensor(x), training=False)
        assert "slope_quantiles" in out
        assert tuple(out["slope_quantiles"].shape) == (2, 16, 5)
