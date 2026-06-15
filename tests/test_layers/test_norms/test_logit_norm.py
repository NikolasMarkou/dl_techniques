"""
Test suite for the LogitNorm layer.

LogitNorm contract:
- norm = sqrt(max(sum(x^2, axis), eps))   (epsilon is a floor, not an additive term)
- output = x / (norm * temperature)
- shape-preserving; temperature is a FIXED hyperparameter (not a trainable weight).
- stateless (no build / no weights).
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.norms.logit_norm import LogitNorm


@pytest.fixture
def logits_2d() -> np.ndarray:
    np.random.seed(7)
    return np.random.randn(8, 10).astype(np.float32)


class TestLogitNorm:
    def test_shape_preserving(self, logits_2d):
        y = LogitNorm()(ops.convert_to_tensor(logits_2d))
        assert tuple(y.shape) == logits_2d.shape

    def test_matches_formula(self, logits_2d):
        temperature = 0.04
        y = LogitNorm(temperature=temperature, epsilon=1e-12)(
            ops.convert_to_tensor(logits_2d)
        )
        norm = np.sqrt(np.maximum((logits_2d ** 2).sum(axis=-1, keepdims=True), 1e-12))
        expected = logits_2d / (norm * temperature)
        np.testing.assert_allclose(ops.convert_to_numpy(y), expected, atol=1e-4)

    def test_temperature_is_not_a_weight(self):
        layer = LogitNorm()
        layer(ops.convert_to_tensor(np.random.randn(2, 5).astype("float32")))
        assert len(layer.weights) == 0  # temperature is a fixed hyperparameter

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            LogitNorm(temperature=0.0)
        with pytest.raises(ValueError):
            LogitNorm(epsilon=-1.0)

    def test_config_roundtrip(self):
        layer = LogitNorm(temperature=0.1, axis=-1, epsilon=1e-6)
        rebuilt = LogitNorm.from_config(layer.get_config())
        assert rebuilt.get_config() == layer.get_config()

    def test_keras_model_roundtrip(self, logits_2d):
        x = ops.convert_to_tensor(logits_2d)
        inp = keras.Input(shape=(10,))
        out = LogitNorm(temperature=0.1)(inp)
        model = keras.Model(inp, out)
        path = os.path.join(tempfile.mkdtemp(), "logit_norm.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        np.testing.assert_allclose(
            ops.convert_to_numpy(model(x)),
            ops.convert_to_numpy(reloaded(x)),
            atol=1e-6,
        )

    def test_graph_trace(self):
        layer = LogitNorm()

        @tf.function(input_signature=[tf.TensorSpec([None, 10], tf.float32)])
        def fn(x):
            return layer(x)

        out = fn(tf.random.normal((4, 10)))
        assert tuple(out.shape) == (4, 10)
