"""Tests for Timestral scaler (normalization)."""

import keras
import numpy as np
import pytest

from dl_techniques.models.time_series.timestral.scaler import (
    causal_patch_loc_scale,
    apply_affine_asinh_masked,
    invert_affine_asinh,
    clamp_nonfinite,
    denormalize_quantiles,
    advancing_block_frames,
    instance_norm_stats,
    welford_combine,
    cumulative_welford,
)


class TestScaler:
    """Test causal per-patch normalization and denormalization."""

    def test_causal_patch_loc_scale(self):
        """Test per-patch loc/scale computation."""
        data = keras.ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        mask = keras.ops.convert_to_tensor([[True]*8])
        loc, scale = causal_patch_loc_scale(data, mask, patch_size=4, minimum_scale=1e-6, stabilize_with_global=False)
        assert loc.shape == (1, 8)
        assert scale.shape == (1, 8)
        # Patch 0 (indices 0-3): mean=2.5, std=~1.29
        # Patch 1 (indices 4-7): mean=6.5, std=~1.29
        np.testing.assert_allclose(loc[0, 3], 2.5, rtol=0.1)
        np.testing.assert_allclose(loc[0, 7], 4.5, rtol=0.1)  # causal mean through t=7 over ALL data

    def test_apply_affine_asinh_masked(self):
        """Test asinh normalization."""
        data = keras.ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0]])
        loc = keras.ops.convert_to_tensor([[2.0, 2.0, 2.0, 2.0]])
        scale = keras.ops.convert_to_tensor([[1.0, 1.0, 1.0, 1.0]])
        mask = keras.ops.convert_to_tensor([[True, True, False, True]])

        normed = apply_affine_asinh_masked(data, loc, scale, mask, use_asinh=True)
        assert normed.shape == (1, 4)
        # Observed: asinh((x-2)/1)
        # Unobserved: 0
        np.testing.assert_allclose(normed[0, 2], 0.0)

    def test_invert_affine_asinh(self):
        """Test inverse of asinh normalization."""
        x = keras.ops.convert_to_tensor([[0.0, 1.0, -1.0]])
        loc = keras.ops.convert_to_tensor([[2.0, 2.0, 2.0]])
        scale = keras.ops.convert_to_tensor([[1.0, 1.0, 1.0]])

        inverted = invert_affine_asinh(x, loc, scale, use_asinh=True)
        # sinh(0)*1+2 = 2, sinh(1)*1+2 ≈ 3.35, sinh(-1)*1+2 ≈ 0.65
        np.testing.assert_allclose(inverted[0, 0], 2.0)
        np.testing.assert_allclose(inverted[0, 1], 2.0 + np.sinh(1.0), rtol=1e-5)

    def test_clamp_nonfinite(self):
        """Test clamping of infinities."""
        vals = keras.ops.convert_to_tensor([[1.0, np.inf, -np.inf, 3.0]])
        clamped = clamp_nonfinite(vals)
        assert clamped.shape == (1, 4)
        assert np.isfinite(clamped[0, 1])
        assert np.isfinite(clamped[0, 2])
        # Should clamp to max/min of finite values
        assert clamped[0, 1] == 3.0  # max finite
        assert clamped[0, 2] == 1.0  # min finite

    def test_denormalize_quantiles(self):
        """Test quantile denormalization."""
        preds = keras.ops.convert_to_tensor([[[0.0, 1.0, -1.0]]])  # (1, 1, 3)
        loc = keras.ops.convert_to_tensor([[2.0]])
        scale = keras.ops.convert_to_tensor([[1.0]])

        denormed = denormalize_quantiles(preds, loc, scale, use_asinh=True)
        assert denormed.shape == (1, 1, 3)
        np.testing.assert_allclose(denormed[0, 0, 0], 2.0)  # median -> 2

    def test_instance_norm_stats(self):
        """Test non-causal instance normalization stats."""
        values = keras.ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0]])
        mask = keras.ops.convert_to_tensor([[True]*4, [True]*4])
        loc, scale, is_binary = instance_norm_stats(values, mask, minimum_scale=1e-6)
        assert loc.shape == (2, 1)
        assert scale.shape == (2, 1)
        assert is_binary.shape == (2, 1)
        # Row 1 is indicator (all 0 or 1) -> loc=0, scale=1
        np.testing.assert_allclose(loc[1, 0], 0.0)
        np.testing.assert_allclose(scale[1, 0], 1.0)
        assert is_binary[1, 0]

    def test_welford_combine(self):
        """Test Welford state merging."""
        w_a = keras.ops.convert_to_tensor([2.0])
        mean_a = keras.ops.convert_to_tensor([1.0])
        m2_a = keras.ops.convert_to_tensor([2.0])  # variance * (n-1)
        w_b = keras.ops.convert_to_tensor([2.0])
        mean_b = keras.ops.convert_to_tensor([3.0])
        m2_b = keras.ops.convert_to_tensor([2.0])

        w, mean, m2 = welford_combine(w_a, mean_a, m2_a, w_b, mean_b, m2_b)
        assert w == 4.0
        assert mean == 2.0  # combined mean
        # m2 = m2_a + m2_b + (mean_b - mean_a)^2 * w_a * w_b / (w_a + w_b)
        # = 2 + 2 + (2)^2 * 2 * 2 / 4 = 4 + 16/4 = 8
        np.testing.assert_allclose(m2, 8.0)

    def test_cumulative_welford(self):
        """Test prefix scan of Welford states."""
        w = keras.ops.convert_to_tensor([[1.0, 1.0, 1.0, 1.0]])
        mean = keras.ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0]])
        m2 = keras.ops.convert_to_tensor([[0.0, 0.0, 0.0, 0.0]])

        w_cum, mean_cum, m2_cum = cumulative_welford(w, mean, m2)
        assert w_cum.shape == (1, 4)
        # Cumulative means: 1, 1.5, 2, 2.5
        np.testing.assert_allclose(mean_cum[0], [1.0, 1.5, 2.0, 2.5], rtol=1e-5)

    def test_advancing_block_frames(self):
        """Test advancing block frame computation."""
        r = 1
        context = keras.ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        mask = keras.ops.convert_to_tensor([[True]*8])
        sinh_median = keras.ops.convert_to_tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]])  # (1, 2, 4)

        locs, scales = advancing_block_frames(context, mask, sinh_median, minimum_scale=1e-6)
        assert locs.shape == (1, 2)
        assert scales.shape == (1, 2)
        # First patch uses context stats
        # Second patch includes first patch's median folded in


if __name__ == "__main__":
    pytest.main([__file__, "-v"])