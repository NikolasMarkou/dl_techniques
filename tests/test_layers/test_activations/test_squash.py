"""Tests for the SquashLayer (capsule squashing) activation."""

import os
import tempfile

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.layers.activations.squash import SquashLayer


def _x() -> np.ndarray:
    rng = np.random.default_rng(4)
    return rng.standard_normal((4, 3, 8)).astype("float32")


class TestSquashLayer:

    def test_construction_default_epsilon(self) -> None:
        layer = SquashLayer()
        assert layer.axis == -1
        assert layer.epsilon > 0

    def test_invalid_axis(self) -> None:
        with pytest.raises(ValueError):
            SquashLayer(axis=1.5)

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError):
            SquashLayer(epsilon=-1.0)

    def test_forward_norm_below_one(self) -> None:
        y = ops.convert_to_numpy(SquashLayer()(_x()))
        assert y.shape == (4, 3, 8)
        norms = np.linalg.norm(y, axis=-1)
        assert np.all(norms <= 1.0 + 1e-5)

    def test_compute_output_shape(self) -> None:
        layer = SquashLayer()
        x = _x()
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)

    def test_serialization_round_trip(self) -> None:
        inp = keras.Input(shape=(3, 8))
        out = SquashLayer()(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "squash.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-6
        )
