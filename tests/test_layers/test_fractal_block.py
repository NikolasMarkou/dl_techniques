"""Tests for the FractalBlock (FractalNet) layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.fractal_block import FractalBlock
from dl_techniques.layers.standard_blocks import ConvBlock

B, H, W, C = 2, 8, 8, 4
F = 8


def _block_config():
    return ConvBlock(filters=F, kernel_size=3).get_config()


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, H, W, C)).astype("float32")


class TestFractalBlock:

    def test_construction(self):
        layer = FractalBlock(block_config=_block_config(), depth=2)
        assert layer.depth == 2

    @pytest.mark.parametrize("bad", [
        {"depth": 0},
        {"drop_path_rate": 1.5},
    ])
    def test_invalid_args_raise(self, bad):
        kwargs = {"block_config": _block_config(), **bad}
        with pytest.raises(ValueError):
            FractalBlock(**kwargs)

    def test_invalid_block_config_raises(self):
        with pytest.raises(ValueError):
            FractalBlock(block_config="not-a-dict")

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_forward_pass(self, sample, depth):
        layer = FractalBlock(block_config=_block_config(), depth=depth)
        out = layer(sample)
        assert tuple(out.shape) == (B, H, W, F)

    def test_compute_output_shape(self):
        layer = FractalBlock(block_config=_block_config(), depth=2)
        assert layer.compute_output_shape((B, H, W, C)) == (B, H, W, F)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = FractalBlock(block_config=_block_config(), depth=2, name="fractal")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample, training=False)
        path = os.path.join(tmp_path, "fractal.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"FractalBlock": FractalBlock}
        )
        y1 = loaded(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
