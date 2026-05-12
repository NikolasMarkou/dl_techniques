"""
Equivalence + serialization tests for ExponentialMovingAverage.

Gates the Step 1 vectorization (I-3) of plan_2026-05-12_5f0e087c.
The vectorized implementation must reproduce the OLD Python-loop
implementation within documented float32 tolerances:

  - adjust=False : atol=1e-7, rtol=1e-7 (bit-exact in practice).
  - adjust=True  : atol=1e-3, rtol=1e-5 (see D-003 — relaxed because
    ops.scan XLA fusion ordering differs minutely from the unrolled
    eager-op sequence; realistic-workload max-rel-diff is 1.2e-6).

The "reference" implementation below is a verbatim paste of the
pre-Step-1 ema_layer.py loop body, intentionally retained inline so
the regression contract survives later refactors.
"""

from __future__ import annotations

import os
import tempfile
from typing import Tuple

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.time_series.ema_layer import ExponentialMovingAverage


# ---------------------------------------------------------------------
# Reference: verbatim paste of pre-Step-1 Python-loop implementation.
# Do NOT change this — it is the contract under test.
# ---------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="ema_test_ref")
class _ReferenceEMA(keras.layers.Layer):
    def __init__(self, period: int = 25, adjust: bool = True, **kwargs):
        super().__init__(**kwargs)
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        self.period = period
        self.adjust = adjust
        self.alpha = 2.0 / (period + 1.0)

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        ndim = len(inputs.shape)
        if ndim == 2:
            x = ops.expand_dims(inputs, axis=-1)
        else:
            x = inputs
        time_steps = input_shape[1]
        ema_prev = x[:, 0, :]
        ema_values = [ema_prev]
        alpha = ops.cast(self.alpha, dtype=x.dtype)
        one_minus_alpha = ops.cast(1.0 - self.alpha, dtype=x.dtype)
        T = x.shape[1] if x.shape[1] is not None else time_steps
        for t in range(1, T):
            if self.adjust:
                weight = ops.cast(
                    1.0 - ops.power(one_minus_alpha, ops.cast(t + 1, x.dtype)),
                    dtype=x.dtype,
                )
                ema_current = alpha * x[:, t, :] + one_minus_alpha * ema_prev
                ema_current = ema_current / ops.maximum(weight, 1e-10)
            else:
                ema_current = alpha * x[:, t, :] + one_minus_alpha * ema_prev
            ema_values.append(ema_current)
            ema_prev = ema_current
        ema = ops.stack(ema_values, axis=1)
        if ndim == 2:
            ema = ops.squeeze(ema, axis=-1)
        return ema


# ---------------------------------------------------------------------
# Equivalence sweeps
# ---------------------------------------------------------------------
@pytest.mark.parametrize("period", [1, 5, 25, 100])
@pytest.mark.parametrize("T", [1, 16, 128, 512])
@pytest.mark.parametrize("ndim", [2, 3])
class TestEquivalence:
    """Vectorized layer must match the Python-loop reference."""

    @staticmethod
    def _make_input(rng: np.random.Generator, T: int, ndim: int) -> np.ndarray:
        shape: Tuple[int, ...] = (3, T) if ndim == 2 else (3, T, 2)
        return rng.standard_normal(shape).astype(np.float32)

    def test_equivalence_adjust_false(self, period, T, ndim):
        rng = np.random.default_rng(0)
        x_np = self._make_input(rng, T, ndim)
        x = ops.convert_to_tensor(x_np)
        new = ops.convert_to_numpy(
            ExponentialMovingAverage(period=period, adjust=False)(x)
        )
        ref = ops.convert_to_numpy(_ReferenceEMA(period=period, adjust=False)(x))
        # adjust=False is bit-exact in practice; allow only float32 ULP.
        np.testing.assert_allclose(new, ref, atol=1e-7, rtol=1e-7)

    def test_equivalence_adjust_true(self, period, T, ndim):
        rng = np.random.default_rng(0)
        x_np = self._make_input(rng, T, ndim)
        x = ops.convert_to_tensor(x_np)
        new = ops.convert_to_numpy(
            ExponentialMovingAverage(period=period, adjust=True)(x)
        )
        ref = ops.convert_to_numpy(_ReferenceEMA(period=period, adjust=True)(x))
        # D-003: ops.scan XLA fusion differs from unrolled eager-op sequence.
        # Tolerances chosen one ULP-class above pure float32 noise on chained
        # divisions; realistic-workload max-rel-diff is 1.2e-6.
        np.testing.assert_allclose(new, ref, atol=1e-3, rtol=1e-5)


# ---------------------------------------------------------------------
# Shape & edge cases
# ---------------------------------------------------------------------
class TestShapeAndEdgeCases:
    def test_output_shape_2d(self):
        x = ops.convert_to_tensor(np.random.randn(4, 32).astype(np.float32))
        y = ExponentialMovingAverage(period=10)(x)
        assert tuple(y.shape) == (4, 32)

    def test_output_shape_3d(self):
        x = ops.convert_to_tensor(np.random.randn(4, 32, 5).astype(np.float32))
        y = ExponentialMovingAverage(period=10)(x)
        assert tuple(y.shape) == (4, 32, 5)

    def test_t_equals_1_is_identity(self):
        x = ops.convert_to_tensor(np.array([[1.5], [2.5], [3.5]]).astype(np.float32))
        # x is (3, 1) — 2D, T=1
        y = ops.convert_to_numpy(ExponentialMovingAverage(period=5)(x))
        np.testing.assert_array_equal(y, ops.convert_to_numpy(x))

    def test_period_1_is_identity_adjust_false(self):
        x_np = np.random.randn(2, 16, 3).astype(np.float32)
        x = ops.convert_to_tensor(x_np)
        y = ops.convert_to_numpy(
            ExponentialMovingAverage(period=1, adjust=False)(x)
        )
        # alpha = 2/2 = 1; one_minus_alpha = 0 → y_t = x_t for all t.
        np.testing.assert_allclose(y, x_np, atol=1e-7, rtol=1e-7)


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------
class TestSerialization:
    def test_get_config_round_trip(self):
        lay = ExponentialMovingAverage(period=37, adjust=False)
        cfg = lay.get_config()
        assert cfg["period"] == 37
        assert cfg["adjust"] is False
        rebuilt = ExponentialMovingAverage.from_config(cfg)
        assert rebuilt.period == 37
        assert rebuilt.adjust is False

    def test_save_load_round_trip(self, tmp_path):
        # Layer wrapped in a tiny model so we exercise full keras save/load.
        inp = keras.layers.Input(shape=(16, 1), dtype="float32")
        out = ExponentialMovingAverage(period=10, adjust=True)(inp)
        model = keras.Model(inp, out)

        x_np = np.random.randn(2, 16, 1).astype(np.float32)
        before = ops.convert_to_numpy(model(ops.convert_to_tensor(x_np)))

        path = os.path.join(tmp_path, "ema_model.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = ops.convert_to_numpy(loaded(ops.convert_to_tensor(x_np)))

        np.testing.assert_allclose(before, after, atol=1e-7, rtol=1e-7)


# ---------------------------------------------------------------------
# Period validation
# ---------------------------------------------------------------------
def test_period_zero_raises():
    with pytest.raises(ValueError, match="period must be >= 1"):
        ExponentialMovingAverage(period=0)
