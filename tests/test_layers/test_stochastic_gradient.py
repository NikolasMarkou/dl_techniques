"""Tests for the StochasticGradient regularization layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.stochastic_gradient import StochasticGradient

B, D = 4, 6


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, D)).astype("float32")


class TestStochasticGradient:

    def test_construction(self):
        layer = StochasticGradient(drop_path_rate=0.3)
        assert layer.drop_path_rate == 0.3

    @pytest.mark.parametrize("bad", [{"drop_path_rate": 1.0}, {"drop_path_rate": -0.1}])
    def test_invalid_rate_raises(self, bad):
        with pytest.raises(ValueError):
            StochasticGradient(**bad)

    def test_forward_is_identity_inference(self, sample):
        out = StochasticGradient(drop_path_rate=0.5)(sample, training=False)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(out), sample, atol=1e-6)

    def test_forward_identity_training(self, sample):
        # Forward value is always identity (only gradients are affected).
        out = StochasticGradient(drop_path_rate=0.5)(sample, training=True)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(out), sample, atol=1e-6)

    def test_compute_output_shape(self):
        assert StochasticGradient().compute_output_shape((B, D)) == (B, D)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(D,))
        out = StochasticGradient(drop_path_rate=0.4, name="sg")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample, training=False)
        path = os.path.join(tmp_path, "sg.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"StochasticGradient": StochasticGradient}
        )
        y1 = loaded(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1), atol=1e-6
        )
