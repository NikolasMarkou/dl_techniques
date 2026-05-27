"""Tests for the SimMIM-style MAE mask utilities."""

from __future__ import annotations

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.models.convnext_patch_vae_v2.mae_mask import (
    apply_mask_with_token,
    generate_patch_mask,
    upsample_mask_to_pixels,
)


class TestGeneratePatchMask:

    def test_zero_ratio_returns_all_zeros(self):
        m = generate_patch_mask(batch_size=4, hp=8, wp=8, ratio=0.0)
        m_np = np.array(m)
        assert m_np.shape == (4, 8, 8, 1)
        assert m_np.sum() == 0.0

    def test_full_ratio_returns_all_ones(self):
        m = generate_patch_mask(batch_size=4, hp=8, wp=8, ratio=1.0)
        m_np = np.array(m)
        assert m_np.sum() == 4 * 8 * 8

    @pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
    def test_per_sample_masked_count_matches_round(self, ratio):
        B, hp, wp = 6, 8, 8
        n = hp * wp
        n_expected = max(1, int(round(ratio * n)))
        m = generate_patch_mask(batch_size=B, hp=hp, wp=wp, ratio=ratio, seed=42)
        m_np = np.array(m)
        per_sample_counts = m_np.reshape(B, -1).sum(axis=1)
        # Each sample should have exactly n_expected masked patches.
        np.testing.assert_array_equal(per_sample_counts, np.full(B, n_expected))

    def test_seed_reproducibility(self):
        m1 = np.array(generate_patch_mask(4, 6, 6, 0.5, seed=123))
        m2 = np.array(generate_patch_mask(4, 6, 6, 0.5, seed=123))
        np.testing.assert_array_equal(m1, m2)

    def test_different_seeds_yield_different_masks(self):
        m1 = np.array(generate_patch_mask(4, 6, 6, 0.5, seed=123))
        m2 = np.array(generate_patch_mask(4, 6, 6, 0.5, seed=456))
        # Should differ on at least some entries.
        assert not np.array_equal(m1, m2)


class TestApplyMaskWithToken:

    def test_zero_mask_preserves_features(self):
        features = ops.convert_to_tensor(
            np.random.RandomState(0).randn(2, 4, 4, 8).astype("float32")
        )
        mask = ops.zeros((2, 4, 4, 1))
        token = ops.zeros((1, 1, 1, 8))
        out = apply_mask_with_token(features, mask, token)
        np.testing.assert_allclose(np.array(out), np.array(features), atol=1e-6)

    def test_full_mask_replaces_with_token(self):
        features = ops.convert_to_tensor(
            np.random.RandomState(0).randn(2, 4, 4, 8).astype("float32")
        )
        mask = ops.ones((2, 4, 4, 1))
        token_val = np.arange(8, dtype="float32").reshape(1, 1, 1, 8)
        token = ops.convert_to_tensor(token_val)
        out = np.array(apply_mask_with_token(features, mask, token))
        expected = np.broadcast_to(token_val, (2, 4, 4, 8))
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_partial_mask_is_per_position_blend(self):
        features = ops.ones((1, 2, 2, 3))
        token = ops.zeros((1, 1, 1, 3))
        mask = ops.convert_to_tensor(
            np.array([[[[1.0], [0.0]], [[0.0], [1.0]]]], dtype="float32")
        )
        out = np.array(apply_mask_with_token(features, mask, token))
        # Diagonal positions masked → 0; off-diagonal preserved → 1.
        assert out[0, 0, 0, 0] == 0.0
        assert out[0, 0, 1, 0] == 1.0
        assert out[0, 1, 0, 0] == 1.0
        assert out[0, 1, 1, 0] == 0.0


class TestUpsampleMaskToPixels:

    def test_shape_correct(self):
        mask = ops.zeros((2, 4, 4, 1))
        up = upsample_mask_to_pixels(mask, patch_size=8)
        assert tuple(up.shape) == (2, 32, 32, 1)

    def test_values_preserved_block_uniform(self):
        mask = ops.convert_to_tensor(
            np.array([[[[1.0], [0.0]], [[0.0], [1.0]]]], dtype="float32")
        )
        up = np.array(upsample_mask_to_pixels(mask, patch_size=2))
        assert up.shape == (1, 4, 4, 1)
        # Block (0,0) should be all 1s.
        np.testing.assert_array_equal(
            up[0, :2, :2, 0], np.ones((2, 2))
        )
        # Block (0,1) should be all 0s.
        np.testing.assert_array_equal(
            up[0, :2, 2:, 0], np.zeros((2, 2))
        )
        # Block (1,0) all 0s.
        np.testing.assert_array_equal(
            up[0, 2:, :2, 0], np.zeros((2, 2))
        )
        # Block (1,1) all 1s.
        np.testing.assert_array_equal(
            up[0, 2:, 2:, 0], np.ones((2, 2))
        )
