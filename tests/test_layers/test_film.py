"""Tests for the configurable FiLMLayer (feature-wise linear modulation)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.film import FiLMLayer

B, H, W, C = 2, 4, 4, 8
S = 6  # style vector dim


@pytest.fixture
def sample_inputs():
    rng = np.random.default_rng(0)
    content = rng.standard_normal((B, H, W, C)).astype("float32")
    style = rng.standard_normal((B, S)).astype("float32")
    return content, style


class TestFiLMLayer:

    def test_construction(self):
        layer = FiLMLayer()
        assert layer.modulation_mode == "both"

    @pytest.mark.parametrize("bad", [
        {"projection_dropout": 1.0},
        {"modulation_mode": "bogus"},
        {"epsilon": 0.0},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            FiLMLayer(**bad)

    @pytest.mark.parametrize("mode", ["multiplicative", "additive", "both"])
    def test_forward_pass(self, sample_inputs, mode):
        content, style = sample_inputs
        layer = FiLMLayer(modulation_mode=mode)
        out = layer([content, style])
        assert tuple(out.shape) == (B, H, W, C)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_forward_with_layer_norm_and_dropout(self, sample_inputs):
        content, style = sample_inputs
        layer = FiLMLayer(use_layer_norm=True, projection_dropout=0.1)
        out = layer([content, style], training=True)
        assert tuple(out.shape) == (B, H, W, C)

    def test_compute_output_shape(self):
        layer = FiLMLayer()
        assert layer.compute_output_shape([(B, H, W, C), (B, S)]) == (B, H, W, C)

    def test_serialization_round_trip(self, sample_inputs, tmp_path):
        content, style = sample_inputs
        c_in = keras.Input(shape=(H, W, C), name="content")
        s_in = keras.Input(shape=(S,), name="style")
        out = FiLMLayer(name="film")([c_in, s_in])
        model = keras.Model([c_in, s_in], out)
        y0 = model([content, style])

        path = os.path.join(tmp_path, "film.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"FiLMLayer": FiLMLayer}
        )
        y1 = loaded([content, style])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = FiLMLayer(gamma_units=16, modulation_mode="multiplicative")
        rebuilt = FiLMLayer.from_config(layer.get_config())
        assert rebuilt.gamma_units == 16
        assert rebuilt.modulation_mode == "multiplicative"
