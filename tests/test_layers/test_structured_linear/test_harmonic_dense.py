"""Tests for the HarmonicDense distance-based classifier head.

Covers constructor validation, ``compute_output_shape``, per-``output_mode``
shapes and values against an independent numpy oracle, the ``n=None ->
sqrt(dim)`` heuristic, the ``probs == softmax(logits)`` identity, any-rank
inputs, gradient flow to the kernel, and ``.keras`` value round trips.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.layers.structured_linear.harmonic_dense import HarmonicDense


def _sq_dist_reference(x: np.ndarray, w: np.ndarray, epsilon: float) -> np.ndarray:
    """Independent oracle for floored squared Euclidean distances."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return np.maximum(
        np.sum(x ** 2, axis=-1, keepdims=True)
        - 2.0 * (x @ w)
        + np.sum(w ** 2, axis=0, keepdims=True),
        epsilon,
    )


class TestHarmonicDenseInit:
    def test_defaults(self):
        layer = HarmonicDense(4)
        assert layer.units == 4
        assert layer.n is None
        assert layer.output_mode == "probs"
        assert layer.epsilon == 1e-8
        assert layer.kernel is None
        assert layer.effective_n is None

    @pytest.mark.parametrize("kwargs", [
        {"units": 0},
        {"units": -3},
        {"units": True},
        {"units": 4.0},
        {"output_mode": "softmax"},
        {"output_mode": "PROBS"},
        {"n": 0.0},
        {"n": -1.0},
        {"epsilon": 0.0},
        {"epsilon": -1e-8},
    ])
    def test_invalid_config_raises(self, kwargs):
        kwargs = dict(kwargs)
        units = kwargs.pop("units", 4)
        with pytest.raises(ValueError):
            HarmonicDense(units, **kwargs)

    def test_build_needs_known_last_dim(self):
        with pytest.raises(ValueError):
            HarmonicDense(4).build((None, None))

    def test_compute_output_shape(self):
        assert HarmonicDense(5).compute_output_shape((None, 8)) == (None, 5)
        assert HarmonicDense(5).compute_output_shape((2, 7, 8)) == (2, 7, 5)


class TestHarmonicDenseForward:
    def _layer(self, dim=6, units=4, seed=0, **kwargs):
        layer = HarmonicDense(units, **kwargs)
        layer.build((None, dim))
        w = np.random.default_rng(seed).standard_normal((dim, units))
        layer.set_weights([w.astype("float32")])
        return layer, w.astype(np.float64)

    def test_kernel_shape_and_heuristic(self):
        layer, _ = self._layer(dim=9, units=4)
        assert layer.kernel.shape == (9, 4)
        assert layer.effective_n == pytest.approx(3.0)

    def test_explicit_n_survives_build(self):
        layer, _ = self._layer(dim=9, units=4, n=2.0)
        assert layer.effective_n == 2.0

    @pytest.mark.parametrize("mode", ["probs", "logits", "distances"])
    def test_mode_shapes(self, mode):
        layer, _ = self._layer(output_mode=mode)
        x = np.random.default_rng(1).standard_normal((5, 6)).astype("float32")
        assert np.asarray(keras.ops.convert_to_numpy(layer(x))).shape == (5, 4)

    def test_distances_match_oracle(self):
        layer, w = self._layer(output_mode="distances")
        rng = np.random.default_rng(2)
        x = rng.standard_normal((5, 6)).astype("float32")
        got = np.asarray(keras.ops.convert_to_numpy(layer(x)), dtype=np.float64)
        np.testing.assert_allclose(
            got, np.sqrt(_sq_dist_reference(x, w, 1e-8)), atol=1e-6, rtol=0
        )

    def test_logits_match_oracle(self):
        layer, w = self._layer(output_mode="logits", n=2.0)
        rng = np.random.default_rng(3)
        x = rng.standard_normal((5, 6)).astype("float32")
        got = np.asarray(keras.ops.convert_to_numpy(layer(x)), dtype=np.float64)
        np.testing.assert_allclose(
            got,
            -0.5 * 2.0 * np.log(_sq_dist_reference(x, w, 1e-8)),
            atol=1e-6,
            rtol=0,
        )

    def test_probs_are_softmax_of_logits(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((5, 6)).astype("float32")
        probs_layer, w = self._layer(output_mode="probs", n=2.0)
        logits_layer, _ = self._layer(output_mode="logits", n=2.0, seed=0)
        # Same seed => identical kernel; compare probs against softmax(logits).
        logits_layer.set_weights(probs_layer.get_weights())
        p = np.asarray(
            keras.ops.convert_to_numpy(probs_layer(x)), dtype=np.float64
        )
        z = np.asarray(
            keras.ops.convert_to_numpy(logits_layer(x)), dtype=np.float64
        )
        shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        expected = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        np.testing.assert_allclose(p, expected, atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            np.sum(p, axis=-1), np.ones(5), atol=1e-6, rtol=0
        )

    def test_any_rank_input(self):
        layer, w = self._layer(output_mode="logits", n=1.0)
        rng = np.random.default_rng(5)
        x = rng.standard_normal((2, 7, 6)).astype("float32")
        got = np.asarray(keras.ops.convert_to_numpy(layer(x)), dtype=np.float64)
        assert got.shape == (2, 7, 4)
        flat_expected = -0.5 * 1.0 * np.log(
            _sq_dist_reference(x.reshape(-1, 6), w, 1e-8)
        )
        np.testing.assert_allclose(
            got.reshape(-1, 4), flat_expected, atol=1e-6, rtol=0
        )

    def test_gradient_reaches_kernel(self):
        layer, _ = self._layer(output_mode="logits")
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(6).standard_normal((4, 6)).astype("float32")
        )
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = keras.ops.mean(y)
        grads = tape.gradient(loss, layer.trainable_weights)
        assert len(grads) == 1
        grad = np.asarray(keras.ops.convert_to_numpy(grads[0]))
        assert np.all(np.isfinite(grad))
        assert np.max(np.abs(grad)) > 0


class TestHarmonicDenseSerialization:
    def test_get_config_preserves_none_n(self):
        layer = HarmonicDense(4)
        cfg = layer.get_config()
        assert cfg["n"] is None
        rebuilt = HarmonicDense.from_config(cfg)
        assert rebuilt.n is None
        assert rebuilt.units == 4
        assert rebuilt.output_mode == "probs"

    def test_get_config_round_trip(self):
        layer = HarmonicDense(
            3, n=2.0, output_mode="logits", kernel_initializer="he_normal"
        )
        rebuilt = HarmonicDense.from_config(layer.get_config())
        assert rebuilt.units == 3
        assert rebuilt.n == 2.0
        assert rebuilt.output_mode == "logits"

    @pytest.mark.parametrize("mode", ["probs", "logits", "distances"])
    def test_keras_save_load_value_round_trip(self, tmp_path, mode):
        path = str(tmp_path / f"harmonic_dense_{mode}.keras")
        inputs = keras.Input((6,))
        outputs = HarmonicDense(4, n=2.0, output_mode=mode)(inputs)
        model = keras.Model(inputs, outputs)
        x = np.random.default_rng(7).standard_normal((5, 6)).astype("float32")
        before = model.predict(x, verbose=0)
        model.save(path)
        reloaded = keras.models.load_model(path)
        after = reloaded.predict(x, verbose=0)
        np.testing.assert_allclose(after, before, atol=1e-6, rtol=0)
