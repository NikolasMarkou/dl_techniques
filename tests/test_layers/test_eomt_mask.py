"""Tests for the EomtMask instance-segmentation prediction head."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.eomt_mask import EomtMask

B, Q, E = 2, 5, 16
HH, WW, PD = 4, 4, 16  # pixel feature map; PD must match mask_dim for dot product
NUM_CLASSES = 7


@pytest.fixture
def sample_inputs():
    rng = np.random.default_rng(0)
    query = rng.standard_normal((B, Q, E)).astype("float32")
    pixels = rng.standard_normal((B, HH, WW, PD)).astype("float32")
    return query, pixels


class TestEomtMask:

    def test_construction(self):
        layer = EomtMask(num_classes=NUM_CLASSES, mask_dim=PD)
        assert layer.num_classes == NUM_CLASSES

    @pytest.mark.parametrize("bad", [
        {"num_classes": 0, "mask_dim": PD},
        {"num_classes": NUM_CLASSES, "mask_dim": 0},
        {"num_classes": NUM_CLASSES, "mask_dim": PD, "mlp_dropout_rate": 2.0},
        {"num_classes": NUM_CLASSES, "mask_dim": PD, "mask_temperature": 0.0},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            EomtMask(**bad)

    def test_forward_pass(self, sample_inputs):
        query, pixels = sample_inputs
        layer = EomtMask(num_classes=NUM_CLASSES, mask_dim=PD, hidden_dims=[16])
        cls_out, mask_out = layer((query, pixels))
        assert tuple(cls_out.shape) == (B, Q, NUM_CLASSES)
        assert tuple(mask_out.shape) == (B, Q, HH, WW)

    def test_forward_learnable_temperature(self, sample_inputs):
        query, pixels = sample_inputs
        layer = EomtMask(
            num_classes=NUM_CLASSES, mask_dim=PD, hidden_dims=[16],
            learnable_temperature=True, use_class_norm=True, use_mask_norm=True,
        )
        cls_out, mask_out = layer((query, pixels))
        assert tuple(cls_out.shape) == (B, Q, NUM_CLASSES)

    def test_compute_output_shape(self):
        layer = EomtMask(num_classes=NUM_CLASSES, mask_dim=PD)
        cls_shape, mask_shape = layer.compute_output_shape(((B, Q, E), (B, HH, WW, PD)))
        assert cls_shape == (B, Q, NUM_CLASSES)
        assert mask_shape == (B, Q, HH, WW)

    def test_serialization_round_trip(self, sample_inputs, tmp_path):
        query, pixels = sample_inputs
        q_in = keras.Input(shape=(Q, E), name="query")
        p_in = keras.Input(shape=(HH, WW, PD), name="pixels")
        cls_out, mask_out = EomtMask(
            num_classes=NUM_CLASSES, mask_dim=PD, hidden_dims=[16], name="eomt"
        )((q_in, p_in))
        model = keras.Model([q_in, p_in], [cls_out, mask_out])
        c0, m0 = model([query, pixels])

        path = os.path.join(tmp_path, "eomt.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"EomtMask": EomtMask}
        )
        c1, m1 = loaded([query, pixels])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(c0), keras.ops.convert_to_numpy(c1),
            rtol=1e-5, atol=1e-5,
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(m0), keras.ops.convert_to_numpy(m1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = EomtMask(num_classes=NUM_CLASSES, mask_dim=PD, use_class_norm=True)
        rebuilt = EomtMask.from_config(layer.get_config())
        assert rebuilt.num_classes == NUM_CLASSES
        assert rebuilt.use_class_norm is True
