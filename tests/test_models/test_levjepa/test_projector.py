"""Tests for ``LeVJEPAProjector`` (shape, BN-reshape equivalence, round-trip)."""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.levjepa.projector import LeVJEPAProjector


class TestLeVJEPAProjectorForward:
    def test_forward_shape_default_output_dim(self):
        projector = LeVJEPAProjector(hidden_dim=64)
        x = keras.random.normal((3, 8, 32))  # (views, batch, dim)
        out = projector(x)
        assert out.shape == (3, 8, 32)

    def test_forward_shape_explicit_output_dim(self):
        projector = LeVJEPAProjector(hidden_dim=64, output_dim=17)
        x = keras.random.normal((3, 8, 32))
        out = projector(x)
        assert out.shape == (3, 8, 17)

    def test_forward_shape_rank2_input(self):
        projector = LeVJEPAProjector(hidden_dim=64)
        x = keras.random.normal((8, 32))
        out = projector(x)
        assert out.shape == (8, 32)

    def test_compute_output_shape_matches_call(self):
        projector = LeVJEPAProjector(hidden_dim=64, output_dim=17)
        input_shape = (3, 8, 32)
        assert projector.compute_output_shape(input_shape) == (3, 8, 17)


class TestLeVJEPAProjectorBatchNormReshapeEquivalence:
    """DECISION D-014: BatchNormalization(axis=-1) applied directly to a
    rank-3 tensor must be numerically identical to the reference's
    reshape-to-2D-then-back path -- prove it here, not just assert it."""

    def test_direct_rank3_matches_reshape_around(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 5, 8)).astype("float32")

        bn_direct = keras.layers.BatchNormalization(axis=-1)
        bn_direct.build((None, None, 8))
        out_direct = np.array(bn_direct(x, training=True))

        bn_reshaped = keras.layers.BatchNormalization(axis=-1)
        bn_reshaped.build((None, 8))
        x_flat = x.reshape(-1, 8)
        out_flat = np.array(bn_reshaped(x_flat, training=True))
        out_reshaped_back = out_flat.reshape(4, 5, 8)

        np.testing.assert_allclose(out_direct, out_reshaped_back, atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            np.array(bn_direct.moving_mean),
            np.array(bn_reshaped.moving_mean),
            atol=1e-6,
            rtol=0,
        )
        np.testing.assert_allclose(
            np.array(bn_direct.moving_variance),
            np.array(bn_reshaped.moving_variance),
            atol=1e-6,
            rtol=0,
        )


class TestLeVJEPAProjectorSerialization:
    def test_get_config_round_trip(self):
        projector = LeVJEPAProjector(hidden_dim=48, output_dim=17, name="proj")
        config = projector.get_config()
        restored = LeVJEPAProjector.from_config(config)

        assert restored.hidden_dim == 48
        assert restored.output_dim == 17

    def test_full_model_save_load_round_trip(self, tmp_path):
        inputs = keras.Input(shape=(32,))
        outputs = LeVJEPAProjector(hidden_dim=48)(inputs)
        model = keras.Model(inputs, outputs)

        x = keras.random.normal((5, 32))
        y_before = np.array(model(x, training=False))

        save_path = tmp_path / "projector.keras"
        model.save(save_path)
        loaded = keras.models.load_model(save_path)
        y_after = np.array(loaded(x, training=False))

        np.testing.assert_allclose(y_before, y_after, atol=1e-6, rtol=0)


class TestLeVJEPAProjectorValidation:
    def test_non_positive_hidden_dim_raises(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            LeVJEPAProjector(hidden_dim=0)

    def test_non_positive_output_dim_raises(self):
        with pytest.raises(ValueError, match="output_dim"):
            LeVJEPAProjector(hidden_dim=32, output_dim=0)
