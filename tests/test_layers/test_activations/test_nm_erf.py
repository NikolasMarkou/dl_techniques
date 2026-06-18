"""Tests for the NMErf (tanh sign carrier x Gaussian bump) activation layer."""

import os
import tempfile

import numpy as np
import keras
from keras import ops

from dl_techniques.layers.activations.nm_erf import NMErf, nm_erf
from dl_techniques.layers.activations import create_activation_layer


def _x() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((4, 8)).astype("float32")


def _x_spread() -> np.ndarray:
    # Spread including negatives, zero, the peak (~1.145) and a large value.
    return np.array(
        [-5.0, -2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 1.1447, 2.0, 5.0],
        dtype="float32",
    )


class TestNMErf:

    def test_construction_defaults(self) -> None:
        assert NMErf().m == 1.5

    def test_construction_custom(self) -> None:
        assert NMErf(m=0.5).m == 0.5

    def test_forward_pass(self) -> None:
        layer = NMErf()
        y = layer(_x())
        assert tuple(y.shape) == (4, 8)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_compute_output_shape(self) -> None:
        layer = NMErf()
        x = _x()
        assert tuple(layer.compute_output_shape(x.shape)) == (4, 8)
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)

    def test_formula_correctness(self) -> None:
        x = _x_spread()
        for m in (1.5, 0.5):
            ref = np.tanh(x) * np.exp(1.0 - (x ** 3 - m) ** 2)
            y = ops.convert_to_numpy(NMErf(m=m)(x))
            np.testing.assert_allclose(y, ref, atol=1e-6)

    def test_standalone_fn_matches_layer(self) -> None:
        x = _x_spread()
        for m in (1.5, 0.5):
            fn = ops.convert_to_numpy(nm_erf(x, m))
            layer = ops.convert_to_numpy(NMErf(m=m)(x))
            np.testing.assert_allclose(fn, layer, atol=1e-6)

    def test_factory_construction(self) -> None:
        layer = create_activation_layer("nm_erf", m=2.0)
        assert isinstance(layer, NMErf)
        assert layer.m == 2.0
        assert create_activation_layer("nm_erf").m == 1.5

    def test_serialization_round_trip(self) -> None:
        inp = keras.Input(shape=(8,))
        out = NMErf(m=0.7)(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nm_erf.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-6
        )
