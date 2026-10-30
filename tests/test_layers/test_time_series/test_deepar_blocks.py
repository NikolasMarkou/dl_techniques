"""Tests for the DeepAR building blocks (the likelihood heads are tested in
tests/test_layers/test_heads/test_likelihood_heads.py)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.time_series.deepar_blocks import (
    ScaleLayer,
    DeepARCell,
)

B, T, D = 4, 6, 3


def _f32(*shape):
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).standard_normal(shape).astype("float32")
    )


class TestScaleLayer:
    def test_forward_and_inverse(self):
        layer = ScaleLayer(scale_per_sample=True)
        x = keras.ops.abs(_f32(B, T, D)) + 1.0
        scaled = layer(x)
        assert tuple(scaled.shape) == (B, T, D)

    def test_compute_output_shape(self):
        assert ScaleLayer().compute_output_shape((B, T, D)) == (B, T, D)

    def test_serialization(self, tmp_path):
        inp = keras.Input(shape=(T, D))
        out = ScaleLayer(name="scale")(inp)
        model = keras.Model(inp, out)
        x = np.abs(np.random.default_rng(0).standard_normal((B, T, D)).astype("float32")) + 1.0
        y0 = model(x)
        path = os.path.join(tmp_path, "scale.keras")
        model.save(path)
        loaded = keras.models.load_model(path, custom_objects={"ScaleLayer": ScaleLayer})
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(loaded(x)),
            rtol=1e-5, atol=1e-5,
        )


class TestDeepARCell:
    def test_construction(self):
        cell = DeepARCell(units=8)
        assert cell.state_size == 8

    def test_compute_output_shape(self):
        assert DeepARCell(units=8).compute_output_shape((B, D)) == (B, 8)

    def test_forward_via_rnn(self):
        rnn = keras.layers.RNN(DeepARCell(units=8))
        out = rnn(_f32(B, T, D))
        assert tuple(out.shape) == (B, 8)

    def test_serialization(self, tmp_path):
        inp = keras.Input(shape=(T, D))
        out = keras.layers.RNN(DeepARCell(units=8, name="cell"))(inp)
        model = keras.Model(inp, out)
        x = np.random.default_rng(0).standard_normal((B, T, D)).astype("float32")
        y0 = model(x)
        path = os.path.join(tmp_path, "deepar.keras")
        model.save(path)
        loaded = keras.models.load_model(path, custom_objects={"DeepARCell": DeepARCell})
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(loaded(x)),
            rtol=1e-5, atol=1e-5,
        )
