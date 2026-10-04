"""Tests for the HarMax harmonic-normalization layer.

Covers constructor validation, the ``softmax(-n * log d)`` oracle, rows that
sum to 1, any-rank inputs with a non-last axis, axis range rejection in both
``call`` and ``compute_output_shape``, the zero-distance epsilon floor,
gradient flow, a ``.keras`` value round trip, factory construction, and the
``ProbabilityOutput(probability_type="harmax")`` delegation.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.layers.activations.harmax import HarMax
from dl_techniques.layers.activations.factory import create_activation_layer
from dl_techniques.layers.activations.probability_output import ProbabilityOutput


def _harmax_reference(d: np.ndarray, n: float, epsilon: float) -> np.ndarray:
    """Independent numpy oracle: softmax(-n * log(max(d, eps))) over last axis."""
    d = np.asarray(d, dtype=np.float64)
    logits = -n * np.log(np.maximum(d, epsilon))
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class TestHarMaxInit:
    def test_defaults(self):
        layer = HarMax()
        assert layer.n == 1.0
        assert layer.epsilon == 1e-8
        assert layer.axis == -1

    @pytest.mark.parametrize("kwargs", [
        {"n": 0.0},
        {"n": -2.0},
        {"epsilon": 0.0},
        {"epsilon": -1e-8},
        {"axis": True},
        {"axis": 1.5},
        {"axis": "last"},
    ])
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            HarMax(**kwargs)


class TestHarMaxForward:
    def test_matches_softmax_oracle(self):
        rng = np.random.default_rng(0)
        d = np.abs(rng.standard_normal((4, 6))).astype("float32") + 0.05
        layer = HarMax(n=2.0)
        got = np.asarray(keras.ops.convert_to_numpy(layer(d)), dtype=np.float64)
        np.testing.assert_allclose(
            got, _harmax_reference(d, 2.0, 1e-8), atol=1e-6, rtol=0
        )

    def test_rows_sum_to_one(self):
        rng = np.random.default_rng(1)
        d = np.abs(rng.standard_normal((8, 10))).astype("float32") + 0.01
        got = np.asarray(keras.ops.convert_to_numpy(HarMax(n=3.0)(d)), dtype=np.float64)
        np.testing.assert_allclose(
            np.sum(got, axis=-1), np.ones(8), atol=1e-6, rtol=0
        )

    def test_smaller_distance_wins(self):
        d = np.array([[1.0, 2.0, 4.0]], dtype="float32")
        got = np.asarray(keras.ops.convert_to_numpy(HarMax(n=1.0)(d))).ravel()
        assert got[0] > got[1] > got[2]

    def test_larger_n_is_sharper(self):
        d = np.array([[1.0, 2.0]], dtype="float32")
        soft = np.asarray(keras.ops.convert_to_numpy(HarMax(n=1.0)(d))).ravel()
        sharp = np.asarray(keras.ops.convert_to_numpy(HarMax(n=8.0)(d))).ravel()
        assert sharp[0] > soft[0]

    def test_non_last_axis(self):
        rng = np.random.default_rng(2)
        d = np.abs(rng.standard_normal((3, 5, 4))).astype("float32") + 0.05
        layer = HarMax(n=1.5, axis=1)
        got = np.asarray(keras.ops.convert_to_numpy(layer(d)), dtype=np.float64)
        moved = np.moveaxis(d.astype(np.float64), 1, -1)
        expected = np.moveaxis(_harmax_reference(moved, 1.5, 1e-8), -1, 1)
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            np.sum(got, axis=1), np.ones((3, 4)), atol=1e-6, rtol=0
        )

    def test_zero_distances_are_finite(self):
        d = np.zeros((2, 4), dtype="float32")
        got = np.asarray(keras.ops.convert_to_numpy(HarMax()(d)))
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(
            np.sum(got, axis=-1), np.ones(2), atol=1e-6, rtol=0
        )

    def test_negative_inputs_clip_like_zeros(self):
        d_neg = np.full((2, 4), -3.0, dtype="float32")
        d_zero = np.zeros((2, 4), dtype="float32")
        layer = HarMax()
        got_neg = np.asarray(keras.ops.convert_to_numpy(layer(d_neg)))
        got_zero = np.asarray(keras.ops.convert_to_numpy(layer(d_zero)))
        np.testing.assert_allclose(got_neg, got_zero, atol=0.0, rtol=0)

    def test_out_of_range_axis_raises(self):
        layer = HarMax(axis=3)
        with pytest.raises(ValueError):
            layer(np.zeros((2, 4), dtype="float32"))

    def test_compute_output_shape_rejects_bad_axis(self):
        with pytest.raises(ValueError):
            HarMax(axis=2).compute_output_shape((None, 4))
        assert HarMax(axis=-1).compute_output_shape((None, 4, 7)) == (None, 4, 7)

    def test_gradient_flows(self):
        layer = HarMax(n=2.0)
        x = keras.ops.convert_to_tensor(
            np.abs(np.random.default_rng(3).standard_normal((4, 5))).astype("float32") + 0.1
        )
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = layer(x)
            loss = keras.ops.sum(y[:, 0])
        grad = keras.ops.convert_to_numpy(tape.gradient(loss, x))
        assert grad is not None
        assert np.all(np.isfinite(grad))
        assert np.max(np.abs(grad)) > 0


class TestHarMaxSerialization:
    def test_get_config_round_trip(self):
        layer = HarMax(n=2.5, epsilon=1e-6, axis=-2)
        rebuilt = HarMax.from_config(layer.get_config())
        assert rebuilt.n == 2.5
        assert rebuilt.epsilon == 1e-6
        assert rebuilt.axis == -2

    def test_keras_save_load_value_round_trip(self, tmp_path):
        path = str(tmp_path / "harmax.keras")
        inputs = keras.Input((6,))
        outputs = HarMax(n=2.0)(inputs)
        model = keras.Model(inputs, outputs)
        rng = np.random.default_rng(4)
        d = np.abs(rng.standard_normal((5, 6))).astype("float32") + 0.05
        before = model.predict(d, verbose=0)
        model.save(path)
        reloaded = keras.models.load_model(path)
        after = reloaded.predict(d, verbose=0)
        np.testing.assert_allclose(after, before, atol=1e-6, rtol=0)


class TestHarMaxWiring:
    def test_factory_builds_harmax(self):
        layer = create_activation_layer("harmax", n=2.0)
        assert isinstance(layer, HarMax)
        assert layer.n == 2.0

    def test_factory_rejects_bad_n(self):
        with pytest.raises(ValueError):
            create_activation_layer("harmax", n=0.0)

    def test_probability_output_delegates(self):
        rng = np.random.default_rng(5)
        d = np.abs(rng.standard_normal((4, 7))).astype("float32") + 0.05
        wrapper = ProbabilityOutput(
            probability_type="harmax", type_config={"n": 2.0}
        )
        assert isinstance(wrapper.strategy_layer, HarMax)
        got = np.asarray(keras.ops.convert_to_numpy(wrapper(d)), dtype=np.float64)
        assert got.shape == d.shape
        np.testing.assert_allclose(
            np.sum(got, axis=-1), np.ones(4), atol=1e-6, rtol=0
        )
        direct = np.asarray(
            keras.ops.convert_to_numpy(HarMax(n=2.0)(d)), dtype=np.float64
        )
        np.testing.assert_allclose(got, direct, atol=1e-6, rtol=0)
