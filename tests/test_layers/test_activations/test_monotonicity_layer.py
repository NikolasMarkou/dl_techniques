"""
Tests for MonotonicityLayer.

Focus: the layer advertises non-decreasing output along the monotonicity axis
for every supported method. Historically the ``sigmoid`` method did not actually
guarantee monotonicity (per-element deviation could exceed the adjacent target
spacing). These tests pin the contract for all methods, plus the standard
init / forward / serialization round-trip checks.
"""

import os
import numpy as np
import pytest
import keras

from dl_techniques.layers.activations.monotonicity_layer import MonotonicityLayer

# Methods that require an explicit value_range.
RANGE_METHODS = ["sigmoid", "normalized_softmax"]
# Methods that work without a value_range.
FREE_METHODS = ["cumulative_softplus", "exponential", "squared", "cumulative_exp"]
ALL_METHODS = FREE_METHODS + RANGE_METHODS


def _make_layer(method: str) -> MonotonicityLayer:
    if method in RANGE_METHODS:
        return MonotonicityLayer(method=method, axis=-1, value_range=(-5.0, 5.0))
    return MonotonicityLayer(method=method, axis=-1)


def _assert_non_decreasing(y: np.ndarray, method: str, n: int) -> None:
    """Adjacent differences must be >= 0 up to float32 precision at output scale.

    The unbounded cumulative/exp methods reach large magnitudes where float32
    rounding noise is non-trivial in absolute terms; the tolerance is scaled by
    the output magnitude (float32 has ~1e-7 relative precision).
    """
    diffs = np.diff(y, axis=-1)
    scale = float(np.max(np.abs(y))) + 1.0
    tol = -1e-5 * scale
    assert diffs.min() >= tol, (
        f"method={method} n={n} produced an inversion: "
        f"min adjacent diff = {diffs.min()} (tol={tol})"
    )


class TestMonotonicityContract:
    """Every advertised method must produce non-decreasing output along axis."""

    @pytest.mark.parametrize("method", ALL_METHODS)
    @pytest.mark.parametrize("n", [2, 5, 32])
    def test_non_decreasing(self, method: str, n: int) -> None:
        rng = np.random.default_rng(1234)
        layer = _make_layer(method)
        # Realistic prediction-scale inputs.
        x = rng.normal(0.0, 2.0, size=(8, n)).astype("float32")
        y = np.array(layer(keras.ops.convert_to_tensor(x)))
        _assert_non_decreasing(y, method, n)

    @pytest.mark.parametrize("n", [2, 5, 32])
    def test_sigmoid_non_decreasing_under_extreme_inputs(self, n: int) -> None:
        """Regression for the historical sigmoid-method monotonicity bug.

        The sigmoid method is bounded by value_range, so even extreme inputs do
        not blow up the output scale -- monotonicity must hold exactly here.
        """
        rng = np.random.default_rng(1234)
        layer = MonotonicityLayer(method="sigmoid", axis=-1, value_range=(-5.0, 5.0))
        x = (rng.normal(0.0, 5.0, size=(8, n)) * 20.0).astype("float32")
        y = np.array(layer(keras.ops.convert_to_tensor(x)))
        diffs = np.diff(y, axis=-1)
        assert diffs.min() >= -1e-5, (
            f"sigmoid n={n} inversion: min adjacent diff = {diffs.min()}"
        )

    def test_sigmoid_respects_value_range(self) -> None:
        layer = MonotonicityLayer(method="sigmoid", axis=-1, value_range=(-2.0, 2.0))
        rng = np.random.default_rng(7)
        x = (rng.normal(0.0, 5.0, size=(4, 16)) * 20.0).astype("float32")
        y = np.array(layer(keras.ops.convert_to_tensor(x)))
        assert y.min() >= -2.0 - 1e-5
        assert y.max() <= 2.0 + 1e-5


class TestMonotonicityBasics:

    def test_output_shape_preserved(self) -> None:
        layer = MonotonicityLayer(method="cumulative_softplus", axis=-1)
        x = keras.ops.convert_to_tensor(np.zeros((3, 7), dtype="float32"))
        y = layer(x)
        assert tuple(y.shape) == (3, 7)
        assert layer.compute_output_shape((None, 7)) == (None, 7)

    def test_build_idempotent(self) -> None:
        layer = MonotonicityLayer(method="cumulative_softplus", axis=-1)
        layer.build((None, 5))
        n1 = len(layer.weights)
        layer.build((None, 5))  # second build must be a no-op
        assert len(layer.weights) == n1

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError):
            MonotonicityLayer(method="not_a_method")

    def test_serialization_round_trip(self, tmp_path) -> None:
        inputs = keras.Input(shape=(8,))
        outputs = MonotonicityLayer(
            method="sigmoid", axis=-1, value_range=(-3.0, 3.0)
        )(inputs)
        model = keras.Model(inputs, outputs)

        x = np.random.default_rng(0).normal(size=(5, 8)).astype("float32")
        y_before = model(x)

        path = os.path.join(tmp_path, "monotonicity.keras")
        model.save(path)
        restored = keras.models.load_model(path)
        y_after = restored(x)

        np.testing.assert_allclose(
            np.array(y_before), np.array(y_after), rtol=1e-6, atol=1e-6
        )
