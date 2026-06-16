"""Tests for the InvertedResidualBlock (MobileNetV2) layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.inverted_residual_block import InvertedResidualBlock

B, H, W, C = 2, 8, 8, 8


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, H, W, C)).astype("float32")


class TestInvertedResidualBlock:

    def test_construction(self):
        layer = InvertedResidualBlock(filters=C, block_id=1)
        assert layer.filters == C

    @pytest.mark.parametrize("stride,out_hw", [(1, H), (2, H // 2)])
    def test_forward_pass(self, sample, stride, out_hw):
        layer = InvertedResidualBlock(filters=C, stride=stride, block_id=0)
        out = layer(sample)
        assert tuple(out.shape) == (B, out_hw, out_hw, C)

    def test_get_config_round_trip(self):
        """Regression: forced name= used to collide with the stored config name."""
        layer = InvertedResidualBlock(filters=C, expansion_factor=4, block_id=5)
        config = layer.get_config()
        rebuilt = InvertedResidualBlock.from_config(config)
        assert rebuilt.filters == C
        assert rebuilt._block_id == 5

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = InvertedResidualBlock(filters=C, stride=1, block_id=7)(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "irb.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"InvertedResidualBlock": InvertedResidualBlock}
        )
        y1 = loaded(sample)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
