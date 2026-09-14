"""Tests for BottleneckBlock (relocated from test_standard_blocks.py's TestBottleneckBlock)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.bottleneck_block import BottleneckBlock

B = 2


def _roundtrip(layer, input_shape, data, name, tmp_path):
    inp = keras.Input(shape=input_shape)
    out = layer(inp)
    model = keras.Model(inp, out)
    y0 = model(data, training=False)
    path = os.path.join(tmp_path, f"{name}.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    y1 = loaded(data, training=False)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
        rtol=1e-5, atol=1e-5,
    )


class TestBottleneckBlock:
    def test_forward_and_shape(self):
        x = np.random.default_rng(0).standard_normal((B, 8, 8, 32)).astype("float32")
        layer = BottleneckBlock(filters=8)
        out = layer(x)
        assert tuple(out.shape) == (B, 8, 8, 32)

    def test_serialization(self, tmp_path):
        x = np.random.default_rng(0).standard_normal((B, 8, 8, 32)).astype("float32")
        _roundtrip(BottleneckBlock(filters=8, name="bottleneck"), (8, 8, 32), x, "bottleneck", tmp_path)


class TestInvalidArgs:
    @pytest.mark.parametrize("ctor", [
        lambda: BottleneckBlock(filters=0),
    ])
    def test_invalid_args_raise(self, ctor):
        with pytest.raises(ValueError):
            ctor()
