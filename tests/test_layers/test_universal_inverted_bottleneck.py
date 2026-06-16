"""Tests for the UniversalInvertedBottleneck layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.universal_inverted_bottleneck import UniversalInvertedBottleneck

B, H, W, C = 2, 8, 8, 8


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, H, W, C)).astype("float32")


class TestUniversalInvertedBottleneck:

    def test_construction(self):
        layer = UniversalInvertedBottleneck(filters=C)
        assert layer.filters == C

    @pytest.mark.parametrize("bad", [
        {"filters": 0},
        {"filters": C, "stride": 0},
        {"filters": C, "kernel_size": 0},
        {"filters": C, "dropout_rate": 2.0},
        {"filters": C, "expanded_channels": -1},
        {"filters": C, "padding": "bogus"},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            UniversalInvertedBottleneck(**bad)

    @pytest.mark.parametrize("kw,out_hw", [
        ({"use_dw1": True, "use_dw2": False}, H),
        ({"use_dw1": True, "use_dw2": True}, H),
        ({"use_dw1": False, "use_dw2": False}, H),         # FFN variant
        ({"use_dw1": True, "stride": 2}, H // 2),
        ({"use_squeeze_excitation": True}, H),
    ])
    def test_forward_pass(self, sample, kw, out_hw):
        layer = UniversalInvertedBottleneck(filters=C, **kw)
        out = layer(sample)
        assert tuple(out.shape) == (B, out_hw, out_hw, C)

    def test_compute_output_shape(self):
        layer = UniversalInvertedBottleneck(filters=16, stride=2)
        assert layer.compute_output_shape((B, H, W, C)) == (B, 4, 4, 16)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = UniversalInvertedBottleneck(
            filters=C, use_dw1=True, use_squeeze_excitation=True, name="uib"
        )(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "uib.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"UniversalInvertedBottleneck": UniversalInvertedBottleneck}
        )
        y1 = loaded(sample)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = UniversalInvertedBottleneck(filters=C, expansion_factor=6, use_dw2=True)
        rebuilt = UniversalInvertedBottleneck.from_config(layer.get_config())
        assert rebuilt.filters == C and rebuilt.use_dw2 is True
