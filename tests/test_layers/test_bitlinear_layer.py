"""Tests for the BitLinear quantization-aware linear layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.bitlinear_layer import BitLinear

B, D, U = 3, 16, 8


@pytest.fixture
def sample_input():
    return np.random.default_rng(0).standard_normal((B, D)).astype("float32")


class TestBitLinear:

    def test_construction(self):
        layer = BitLinear(units=U)
        assert layer.units == U

    @pytest.mark.parametrize("bad", [
        {"units": 0},
        {"units": U, "weight_scale_method": "bogus"},
        {"units": U, "quantization_method": "bogus"},
        {"units": U, "ste_lambda": 0},
        {"units": U, "epsilon": 0},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            BitLinear(**bad)

    @pytest.mark.parametrize("use_input_norm", [False, True])
    @pytest.mark.parametrize("units", [U, D])  # units != input_dim and units == input_dim
    def test_forward_pass(self, sample_input, use_input_norm, units):
        layer = BitLinear(units=units, use_input_norm=use_input_norm)
        out = layer(sample_input)
        assert tuple(out.shape) == (B, units)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        layer = BitLinear(units=U)
        assert layer.compute_output_shape((B, D)) == (B, U)

    def test_compute_output_shape_matches_call(self, sample_input):
        layer = BitLinear(units=U)
        out = layer(sample_input)
        assert tuple(out.shape) == tuple(layer.compute_output_shape(sample_input.shape))

    @pytest.mark.parametrize("use_input_norm", [False, True])
    def test_serialization_round_trip(self, sample_input, use_input_norm, tmp_path):
        inp = keras.Input(shape=(D,))
        out = BitLinear(units=U, use_input_norm=use_input_norm, name="bl")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample_input)

        path = os.path.join(tmp_path, "bl.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"BitLinear": BitLinear}
        )
        y1 = loaded(sample_input)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = BitLinear(units=U, weight_bits=2, activation_bits=4, use_bias=False)
        rebuilt = BitLinear.from_config(layer.get_config())
        assert rebuilt.units == U
        assert rebuilt.use_bias is False
