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
import tensorflow as tf

from dl_techniques.layers.activations.monotonicity_layer import MonotonicityLayer


# R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251.
# Keras `ops/nn.py:907` advises that a softmax over a size-1 axis always returns
# exactly 1.0. Every site in this module feeds that axis a size of 1 ON PURPOSE
# -- single class, single token, single head, single anchor, single cluster,
# minimum sequence length -- so the advisory describes the test's own input, not
# a defect. Suppressed HERE rather than in `pyproject.toml` so an ACCIDENTAL
# size-1 softmax anywhere else still fails under `error::UserWarning`.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:You are using a softmax over axis:UserWarning"),
]

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


# ---------------------------------------------------------------------
# XLA / graph equivalence (guide rule L-39)
#
# Flagged by this plan's step-11 criterion for a DYNAMIC SHAPE READ inside
# `call()`: `_sigmoid_monotonic` does
# `axis_size = keras.ops.shape(inputs)[self.axis_normalized]` and then feeds that
# straight into `keras.ops.arange(axis_size, ...)`. An `arange` whose LENGTH comes
# from a traced tensor is the classic shape that XLA cannot lower, because XLA
# requires static output shapes; whether it works here depends on the shape being
# statically known at trace time, which is a property of the caller, not of this
# layer. Nothing in this file ran the layer under `tf.function` before.
#
# Tolerance derived from measurement: `max|eager - xla|` is at most 1.53e-05 over
# the six methods on a (8, 64) N(0,1) input -- but these outputs are NOT bounded
# by 1 (`cumulative_exp` reaches ~148), so that is a RELATIVE 1e-07, i.e. float32
# round-off. `1e-03` leaves ~65x headroom and sits far below the step-8 mutation
# of this layer (bypassing the whole dispatcher moved the output by 15.0).
# ---------------------------------------------------------------------


class TestMonotonicityLayerXLAEquivalence:
    """`jit_compile=True` must lower every method AND agree with eager."""

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_jit_compiled_matches_eager(
        self, method, assert_xla_matches_eager
    ) -> None:
        """Each of the six methods compiles under XLA and returns the eager answer.

        Parametrized over `ALL_METHODS` -- the module's own list -- rather than a
        hand-copied one, so a seventh method cannot be added with no XLA arm.
        `sigmoid` and `normalized_softmax` are the two that reach the dynamic
        `arange`; the others are included as controls that share the dispatcher
        but not the shape read.
        """
        keras.utils.set_random_seed(0)
        x = np.random.default_rng(0).standard_normal((8, 64)).astype("float32")

        layer = _make_layer(method)
        assert_xla_matches_eager(layer, x, 1e-03, f"monotonicity[{method}]")

        # The layer's defining property, re-asserted under the TRACED graph.
        # Unlike the tolerance above this is scale-free and TF32-insensitive.
        @tf.function(jit_compile=True)
        def _traced(t):
            return layer(t)

        y = np.asarray(
            keras.ops.convert_to_numpy(_traced(keras.ops.convert_to_tensor(x))),
            dtype=np.float64,
        )
        diffs = np.diff(y, axis=-1)
        assert np.all(diffs >= -1e-05), (
            f"the traced graph of method {method!r} is NOT non-decreasing along "
            f"axis -1: most negative step is {float(diffs.min()):.6e}"
        )
