"""Tests for the DepthwiseSeparableBlock layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.depthwise_separable_block import DepthwiseSeparableBlock

B, H, W, C = 2, 16, 16, 4
F = 8


@pytest.fixture
def sample_input():
    return np.random.default_rng(0).standard_normal((B, H, W, C)).astype("float32")


class TestDepthwiseSeparableBlock:

    def test_construction(self):
        layer = DepthwiseSeparableBlock(filters=F)
        assert layer.filters == F

    @pytest.mark.parametrize("bad", [
        {"filters": 0},
        {"filters": F, "stride": 0},
        {"filters": F, "block_id": -1},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            DepthwiseSeparableBlock(**bad)

    @pytest.mark.parametrize("stride,out_hw", [(1, 16), (2, 8)])
    def test_forward_pass(self, sample_input, stride, out_hw):
        layer = DepthwiseSeparableBlock(filters=F, stride=stride)
        out = layer(sample_input)
        assert tuple(out.shape) == (B, out_hw, out_hw, F)

    def test_compute_output_shape(self):
        layer = DepthwiseSeparableBlock(filters=F, stride=2)
        assert layer.compute_output_shape((B, H, W, C)) == (B, 8, 8, F)

    def test_compute_output_shape_matches_call(self, sample_input):
        layer = DepthwiseSeparableBlock(filters=F, stride=2)
        out = layer(sample_input)
        assert tuple(out.shape) == tuple(layer.compute_output_shape(sample_input.shape))

    def test_serialization_round_trip(self, sample_input, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = DepthwiseSeparableBlock(filters=F, stride=2, name="dsb")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample_input)

        path = os.path.join(tmp_path, "dsb.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"DepthwiseSeparableBlock": DepthwiseSeparableBlock}
        )
        y1 = loaded(sample_input)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = DepthwiseSeparableBlock(
            filters=F, normalization_type="layer_norm", activation_type="gelu"
        )
        rebuilt = DepthwiseSeparableBlock.from_config(layer.get_config())
        assert rebuilt.filters == F
        assert rebuilt.normalization_type == "layer_norm"
