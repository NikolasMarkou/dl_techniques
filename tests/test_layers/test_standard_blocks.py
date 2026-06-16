"""Tests for the standard_blocks layers (Conv/Dense/Residual/Basic/Bottleneck)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.standard_blocks import (
    ConvBlock,
    DenseBlock,
    ResidualDenseBlock,
    BasicBlock,
    BottleneckBlock,
)

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


class TestConvBlock:
    def test_forward_and_shape(self):
        x = np.random.default_rng(0).standard_normal((B, 8, 8, 4)).astype("float32")
        layer = ConvBlock(filters=8, kernel_size=3)
        out = layer(x)
        assert tuple(out.shape) == (B, 8, 8, 8)
        assert layer.compute_output_shape((B, 8, 8, 4)) == (B, 8, 8, 8)

    def test_serialization(self, tmp_path):
        x = np.random.default_rng(0).standard_normal((B, 8, 8, 4)).astype("float32")
        _roundtrip(ConvBlock(filters=8, kernel_size=3, name="conv"), (8, 8, 4), x, "conv", tmp_path)


class TestDenseBlock:
    def test_forward_and_shape(self):
        x = np.random.default_rng(0).standard_normal((B, 10)).astype("float32")
        layer = DenseBlock(units=8)
        out = layer(x)
        assert tuple(out.shape) == (B, 8)
        assert layer.compute_output_shape((B, 10)) == (B, 8)

    def test_serialization(self, tmp_path):
        x = np.random.default_rng(0).standard_normal((B, 10)).astype("float32")
        _roundtrip(DenseBlock(units=8, name="dense"), (10,), x, "dense", tmp_path)


class TestResidualDenseBlock:
    def test_forward_and_shape(self):
        x = np.random.default_rng(0).standard_normal((B, 10)).astype("float32")
        layer = ResidualDenseBlock(units=10)
        out = layer(x)
        assert tuple(out.shape) == (B, 10)

    def test_serialization(self, tmp_path):
        x = np.random.default_rng(0).standard_normal((B, 10)).astype("float32")
        _roundtrip(ResidualDenseBlock(units=10, name="resdense"), (10,), x, "resdense", tmp_path)


class TestBasicBlock:
    def test_forward_and_shape(self):
        x = np.random.default_rng(0).standard_normal((B, 8, 8, 8)).astype("float32")
        layer = BasicBlock(filters=8)
        out = layer(x)
        assert tuple(out.shape) == (B, 8, 8, 8)

    def test_serialization(self, tmp_path):
        x = np.random.default_rng(0).standard_normal((B, 8, 8, 8)).astype("float32")
        _roundtrip(BasicBlock(filters=8, name="basic"), (8, 8, 8), x, "basic", tmp_path)


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
        lambda: ConvBlock(filters=0, kernel_size=3),
        lambda: DenseBlock(units=0),
        lambda: BasicBlock(filters=0),
        lambda: BottleneckBlock(filters=0),
    ])
    def test_invalid_args_raise(self, ctor):
        with pytest.raises(ValueError):
            ctor()
