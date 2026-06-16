"""Tests for the HardSigmoid and HardSwish activation layers."""

import os
import tempfile

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.layers.activations.hard_sigmoid import HardSigmoid
from dl_techniques.layers.activations.hard_swish import HardSwish


def _x() -> np.ndarray:
    rng = np.random.default_rng(1)
    return (rng.standard_normal((4, 8)) * 5.0).astype("float32")


@pytest.mark.parametrize("layer_cls", [HardSigmoid, HardSwish])
class TestHardActivations:

    def test_construction(self, layer_cls) -> None:
        layer = layer_cls()
        assert isinstance(layer, keras.layers.Layer)

    def test_forward_pass(self, layer_cls) -> None:
        layer = layer_cls()
        y = layer(_x())
        assert tuple(y.shape) == (4, 8)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_hard_sigmoid_range(self, layer_cls) -> None:
        if layer_cls is not HardSigmoid:
            pytest.skip("range check is HardSigmoid-specific")
        y = ops.convert_to_numpy(HardSigmoid()(_x()))
        assert y.min() >= 0.0 - 1e-6 and y.max() <= 1.0 + 1e-6

    def test_compute_output_shape(self, layer_cls) -> None:
        layer = layer_cls()
        x = _x()
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)

    def test_serialization_round_trip(self, layer_cls) -> None:
        inp = keras.Input(shape=(8,))
        out = layer_cls()(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "act.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-6
        )
