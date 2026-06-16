"""Tests for the RouterLayer (dynamic conditional computation)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.transformers.transformer import TransformerLayer
from dl_techniques.layers.router import RouterLayer

B, SEQ, HID = 2, 12, 16


def _transformer():
    return TransformerLayer(hidden_size=HID, num_heads=2, intermediate_size=32)


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, SEQ, HID)).astype("float32")


class TestRouterLayer:

    def test_construction(self):
        layer = RouterLayer(transformer_layer=_transformer(), num_windows=4)
        assert layer.num_windows == 4

    def test_invalid_transformer_type(self):
        with pytest.raises(TypeError):
            RouterLayer(transformer_layer="not-a-transformer")

    @pytest.mark.parametrize("bad", [
        {"router_bottleneck_dim": 0},
        {"num_windows": 0},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            RouterLayer(transformer_layer=_transformer(), **bad)

    def test_forward_pass(self, sample):
        layer = RouterLayer(transformer_layer=_transformer(), router_bottleneck_dim=8, num_windows=4)
        out, logits = layer(sample)
        assert tuple(out.shape) == (B, SEQ, HID)
        assert tuple(logits.shape) == (B, 3)

    def test_compute_output_shape(self):
        layer = RouterLayer(transformer_layer=_transformer(), num_windows=4)
        out_shape, logits_shape = layer.compute_output_shape((B, SEQ, HID))
        assert out_shape == (B, SEQ, HID)
        assert logits_shape == (B, 3)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(SEQ, HID))
        out, logits = RouterLayer(
            transformer_layer=_transformer(), router_bottleneck_dim=8,
            num_windows=4, name="router",
        )(inp)
        model = keras.Model(inp, [out, logits])
        y0, l0 = model(sample)
        path = os.path.join(tmp_path, "router.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y1, l1 = loaded(sample)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(l0), keras.ops.convert_to_numpy(l1),
            rtol=1e-5, atol=1e-5,
        )
