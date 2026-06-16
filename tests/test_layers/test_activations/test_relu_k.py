"""Tests for the ReLUK activation layer."""

import os
import tempfile

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.layers.activations.relu_k import ReLUK


def _x() -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.standard_normal((4, 8)).astype("float32")


class TestReLUK:

    def test_construction(self) -> None:
        assert ReLUK(k=2).k == 2

    def test_invalid_k_type(self) -> None:
        with pytest.raises(TypeError):
            ReLUK(k=2.5)

    def test_invalid_k_value(self) -> None:
        with pytest.raises(ValueError):
            ReLUK(k=0)

    def test_forward_pass(self) -> None:
        y = ops.convert_to_numpy(ReLUK(k=3)(_x()))
        assert y.shape == (4, 8)
        assert np.all(y >= 0.0)

    def test_compute_output_shape(self) -> None:
        layer = ReLUK(k=3)
        x = _x()
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)

    def test_serialization_round_trip(self) -> None:
        inp = keras.Input(shape=(8,))
        out = ReLUK(k=2)(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reluk.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-6
        )
