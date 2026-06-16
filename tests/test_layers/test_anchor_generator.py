"""Tests for the AnchorGenerator layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.anchor_generator import AnchorGenerator

B = 2
IMG = (64, 64)
STRIDES = [8, 16, 32]
TOTAL = 8 * 8 + 4 * 4 + 2 * 2  # 84


@pytest.fixture
def dummy_input():
    return np.zeros((B, *IMG, 3), dtype="float32")


class TestAnchorGenerator:

    def test_construction(self):
        layer = AnchorGenerator(input_image_shape=IMG, strides_config=STRIDES)
        assert layer.strides_config == STRIDES
        assert layer.total_anchor_points == TOTAL

    @pytest.mark.parametrize("bad", [
        {"input_image_shape": (64,)},
        {"input_image_shape": (0, 64)},
        {"input_image_shape": IMG, "strides_config": [8, -1]},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            AnchorGenerator(**bad)

    def test_forward_pass(self, dummy_input):
        layer = AnchorGenerator(input_image_shape=IMG, strides_config=STRIDES)
        anchors, strides = layer(dummy_input)
        assert tuple(anchors.shape) == (B, TOTAL, 2)
        assert tuple(strides.shape) == (B, TOTAL, 1)

    def test_compute_output_shape(self):
        layer = AnchorGenerator(input_image_shape=IMG, strides_config=STRIDES)
        a_shape, s_shape = layer.compute_output_shape((B, *IMG, 3))
        assert a_shape == (B, TOTAL, 2)
        assert s_shape == (B, TOTAL, 1)

    def test_serialization_round_trip(self, dummy_input, tmp_path):
        inp = keras.Input(shape=(*IMG, 3))
        anchors, strides = AnchorGenerator(
            input_image_shape=IMG, strides_config=STRIDES, name="anchors"
        )(inp)
        model = keras.Model(inp, [anchors, strides])
        a0, s0 = model(dummy_input)

        path = os.path.join(tmp_path, "anchors.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"AnchorGenerator": AnchorGenerator}
        )
        a1, s1 = loaded(dummy_input)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(a0), keras.ops.convert_to_numpy(a1), atol=1e-6
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(s0), keras.ops.convert_to_numpy(s1), atol=1e-6
        )

    def test_get_config_round_trip(self):
        layer = AnchorGenerator(input_image_shape=IMG, strides_config=STRIDES)
        rebuilt = AnchorGenerator.from_config(layer.get_config())
        assert rebuilt.input_image_shape == list(IMG) or rebuilt.input_image_shape == IMG
        assert rebuilt.strides_config == STRIDES
