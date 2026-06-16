"""Tests for the PerceiverTransformerLayer block."""

import os
import tempfile

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.layers.transformers.perceiver_transformer import (
    PerceiverTransformerLayer,
)

B, M, N, DIM = 2, 6, 10, 16


def _q() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((B, M, DIM)).astype("float32")


def _kv() -> np.ndarray:
    return np.random.default_rng(1).standard_normal((B, N, DIM)).astype("float32")


class TestPerceiverTransformerLayer:

    def test_construction(self) -> None:
        layer = PerceiverTransformerLayer(dim=DIM, num_heads=4)
        assert layer.dim == DIM and layer.num_heads == 4

    @pytest.mark.parametrize("kwargs", [
        {"dim": 0, "num_heads": 4},
        {"dim": 16, "num_heads": 0},
        {"dim": 16, "num_heads": 5},      # not divisible
        {"dim": 16, "num_heads": 4, "mlp_ratio": 0.0},
        {"dim": 16, "num_heads": 4, "dropout_rate": 1.5},
    ])
    def test_invalid_construction(self, kwargs) -> None:
        with pytest.raises(ValueError):
            PerceiverTransformerLayer(**kwargs)

    def test_self_attention_forward(self) -> None:
        layer = PerceiverTransformerLayer(dim=DIM, num_heads=4)
        y = layer(_q())
        assert tuple(y.shape) == (B, M, DIM)

    def test_cross_attention_forward(self) -> None:
        layer = PerceiverTransformerLayer(dim=DIM, num_heads=4)
        y = layer(_q(), _kv())
        assert tuple(y.shape) == (B, M, DIM)

    def test_compute_output_shape(self) -> None:
        layer = PerceiverTransformerLayer(dim=DIM, num_heads=4)
        q = _q()
        assert tuple(layer.compute_output_shape(q.shape)) == tuple(layer(q).shape)

    def test_serialization_round_trip(self) -> None:
        inp = keras.Input(shape=(M, DIM))
        out = PerceiverTransformerLayer(dim=DIM, num_heads=4, dropout_rate=0.1)(inp)
        model = keras.Model(inp, out)
        x = _q()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "perceiver.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-5
        )
