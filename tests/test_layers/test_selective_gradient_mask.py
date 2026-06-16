"""Tests for the SelectiveGradientMask layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.selective_gradient_mask import SelectiveGradientMask

B, D = 4, 6


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((B, D)).astype("float32")
    mask = (rng.uniform(size=(B, D)) > 0.5).astype("float32")
    return signal, mask


class TestSelectiveGradientMask:

    def test_forward_is_identity(self, inputs):
        signal, mask = inputs
        out = SelectiveGradientMask()([signal, mask], training=True)
        # Forward pass reconstructs the signal exactly.
        np.testing.assert_allclose(keras.ops.convert_to_numpy(out), signal, atol=1e-6)

    def test_inference_returns_signal(self, inputs):
        signal, mask = inputs
        out = SelectiveGradientMask()([signal, mask], training=False)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(out), signal, atol=1e-6)

    def test_invalid_input_count(self):
        layer = SelectiveGradientMask()
        with pytest.raises(ValueError):
            layer([keras.ops.zeros((B, D))], training=True)

    def test_build_shape_mismatch(self):
        layer = SelectiveGradientMask()
        with pytest.raises(ValueError):
            layer.build([(B, D), (B, D + 1)])

    def test_compute_output_shape(self):
        assert SelectiveGradientMask().compute_output_shape([(B, D), (B, D)]) == (B, D)

    def test_serialization_round_trip(self, inputs, tmp_path):
        signal, mask = inputs
        sig_in = keras.Input(shape=(D,), name="signal")
        mask_in = keras.Input(shape=(D,), name="mask")
        out = SelectiveGradientMask(name="sgm")([sig_in, mask_in])
        model = keras.Model([sig_in, mask_in], out)
        y0 = model([signal, mask])
        path = os.path.join(tmp_path, "sgm.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"SelectiveGradientMask": SelectiveGradientMask}
        )
        y1 = loaded([signal, mask])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1), atol=1e-6
        )
