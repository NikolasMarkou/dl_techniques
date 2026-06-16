"""Tests for the GoLU (Gompertz Linear Unit) activation layer."""

import os
import tempfile

import numpy as np
import keras
from keras import ops

from dl_techniques.layers.activations.golu import GoLU


def _x() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((4, 8)).astype("float32")


class TestGoLU:

    def test_construction_defaults(self) -> None:
        layer = GoLU()
        assert layer.alpha == 1.0 and layer.beta == 1.0 and layer.gamma == 1.0

    def test_construction_custom(self) -> None:
        layer = GoLU(alpha=2.0, beta=0.5, gamma=1.5)
        assert layer.alpha == 2.0 and layer.beta == 0.5 and layer.gamma == 1.5

    def test_forward_pass(self) -> None:
        layer = GoLU()
        y = layer(_x())
        assert tuple(y.shape) == (4, 8)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_compute_output_shape(self) -> None:
        layer = GoLU()
        x = _x()
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)

    def test_serialization_round_trip(self) -> None:
        inp = keras.Input(shape=(8,))
        out = GoLU(alpha=1.5, beta=0.7, gamma=1.2)(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "golu.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-6
        )
