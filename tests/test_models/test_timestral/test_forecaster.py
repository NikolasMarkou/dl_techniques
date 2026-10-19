"""Tests for TimestralForecaster inference path."""

import numpy as np
import pytest

from dl_techniques.models.time_series.timestral.config import TimestralConfig
from dl_techniques.models.time_series.timestral.model import TimestralModel
from dl_techniques.models.time_series.timestral.scaler import invert_affine_asinh


class TestTimestralForecaster:
    """Tests for the forecaster's block-autoregressive decode logic.

    These test the internal methods directly; `TimestralForecaster.from_pretrained`
    additionally needs a checkpoint directory on disk or the Hub.
    """

    @pytest.fixture
    def tiny_model(self):
        """Minimal model for testing decode logic."""
        config = TimestralConfig(
            patch_size=8,
            d_model=64,
            num_heads=4,
            mlp_hidden_dim=128,
            block_schedule=("time", "time_variate", "time", "time_variate"),
            quantile_knots=(0.1, 0.5, 0.9),
            block_size=4,
            context_length=128,
        )
        model = TimestralModel(config)
        model.build((None, 64, 1))
        return model

    def test_predict_block_context_only(self, tiny_model):
        """Test single block prediction without covariates."""
        r = 2  # packed rows
        seq_len = 32  # 4 patches of 8
        inputs = np.random.randn(r, seq_len).astype(np.float32)
        observed = np.ones((r, seq_len), dtype=bool)
        group_ids = np.zeros(r, dtype=np.int32)

        # Convert to Keras tensors
        import keras
        inputs_t = keras.ops.convert_to_tensor(inputs)
        observed_t = keras.ops.convert_to_tensor(observed)
        group_ids_t = keras.ops.convert_to_tensor(group_ids)

        preds = tiny_model.predict_block(inputs_t, observed_t, group_ids_t, horizon_patches=1)

        assert preds.shape == (r, 8, 3)  # 1 patch * 8 steps, 3 quantiles
        # Check monotonicity
        for i in range(r):
            for t in range(8):
                q = preds[i, t]
                assert np.all(np.diff(q) >= -1e-5)

    def test_predict_block_multiple_patches(self, tiny_model):
        """Test prediction with multiple horizon patches."""
        r = 1
        seq_len = 64  # 8 patches
        inputs = np.random.randn(r, seq_len).astype(np.float32)
        observed = np.ones((r, seq_len), dtype=bool)
        group_ids = np.zeros(r, dtype=np.int32)

        import keras
        inputs_t = keras.ops.convert_to_tensor(inputs)
        observed_t = keras.ops.convert_to_tensor(observed)
        group_ids_t = keras.ops.convert_to_tensor(group_ids)

        preds = tiny_model.predict_block(inputs_t, observed_t, group_ids_t, horizon_patches=2)

        assert preds.shape == (r, 16, 3)  # 2 patches * 8 steps, 3 quantiles

    def test_refine_block_single_patch(self, tiny_model):
        """Test _refine_block with single patch (no-op)."""
        r = 1
        seq_len = 32
        quantiles = np.random.randn(r, 8, 3).astype(np.float32)  # 1 patch
        context = np.random.randn(r, seq_len).astype(np.float32)
        mask = np.ones((r, seq_len), dtype=bool)
        target_rows = np.ones(r, dtype=bool)

        import keras
        quantiles_t = keras.ops.convert_to_tensor(quantiles)
        context_t = keras.ops.convert_to_tensor(context)
        mask_t = keras.ops.convert_to_tensor(mask)
        target_rows_t = keras.ops.convert_to_tensor(target_rows)

        refined = tiny_model._refine_block(quantiles_t, context_t, mask_t, target_rows_t)
        # Should return unchanged for single patch
        assert refined.shape == quantiles_t.shape

    def test_refine_block_multiple_patches(self, tiny_model):
        """Test _refine_block with multiple patches."""
        r = 1
        seq_len = 64
        quantiles = np.random.randn(r, 16, 3).astype(np.float32)  # 2 patches
        context = np.random.randn(r, seq_len).astype(np.float32)
        mask = np.ones((r, seq_len), dtype=bool)
        target_rows = np.ones(r, dtype=bool)

        import keras
        quantiles_t = keras.ops.convert_to_tensor(quantiles)
        context_t = keras.ops.convert_to_tensor(context)
        mask_t = keras.ops.convert_to_tensor(mask)
        target_rows_t = keras.ops.convert_to_tensor(target_rows)

        import keras
        from keras import ops

        refined = tiny_model._refine_block(quantiles_t, context_t, mask_t, target_rows_t)
        refined = np.asarray(refined)
        assert refined.shape == quantiles_t.shape

        # First horizon patch keeps the anchor frame: it must equal the manual
        # asinh inverse of (q - loc0) / scale0 with the context's last stats.
        _, loc, scale = tiny_model.scale(context_t, mask_t)
        loc0 = np.asarray(loc)[:, -1:]
        scale0 = np.asarray(scale)[:, -1:]
        sinh_norm = (quantiles - loc0[..., None]) / scale0[..., None]
        expected_first = np.asarray(
            invert_affine_asinh(sinh_norm[:, :8], loc0, scale0, use_asinh=True)
        )
        np.testing.assert_allclose(
            refined[:, :8], expected_first, rtol=1e-5, atol=1e-5
        )
        # Further patches use the advancing frame (finite, reordered output).
        assert np.isfinite(refined[:, 8:]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])