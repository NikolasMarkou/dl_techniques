"""Tests for Timestral categorical covariate encoding."""

import numpy as np
import pytest

from dl_techniques.models.time_series.timestral.covariate_encoding import target_encode


class TestCovariateEncoding:
    """Test target encoding of categorical covariates."""

    def test_basic_encoding(self):
        """Test basic target encoding."""
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        codes = np.array([0.0, 1.0, 0.0, 1.0, 0.0])  # two categories
        enc_past, enc_future = target_encode(target, codes)
        assert enc_past.shape == (5,)
        assert enc_future is None
        # Category 0 mean: (1+3+5)/3 = 3, smoothed: (1*3 + 9)/(1+3) = 3
        # Category 1 mean: (2+4)/2 = 3, smoothed: (1*3 + 6)/(1+2) = 3
        # Both map to 3 (plus smoothing)
        np.testing.assert_allclose(enc_past[0], enc_past[2], rtol=0.1)  # both category 0
        np.testing.assert_allclose(enc_past[1], enc_past[3], rtol=0.1)  # both category 1

    def test_encoding_with_nan_target(self):
        """Test encoding with NaN in target (ignored)."""
        target = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        codes = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        enc_past, _ = target_encode(target, codes)
        # Category 0: only 1 and 3 observed -> mean=2
        # Category 1: 4 and 5 -> mean=4.5
        assert not np.isnan(enc_past).any()

    def test_encoding_with_nan_codes(self):
        """Test encoding with NaN in codes (missing category)."""
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        codes = np.array([0.0, np.nan, 0.0, 1.0, np.nan])
        enc_past, _ = target_encode(target, codes)
        # NaN codes get their own category (last slot before unseen)
        assert not np.isnan(enc_past).any()

    def test_encoding_with_future(self):
        """Test encoding with future codes."""
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        past_codes = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        future_codes = np.array([0.0, 1.0, 2.0])  # 2 is unseen
        enc_past, enc_future = target_encode(target, past_codes, future_codes)
        assert enc_future is not None
        assert enc_future.shape == (3,)
        # Unseen category maps to overall mean (3.0)

    def test_smoothing(self):
        """Test smoothing parameter."""
        target = np.array([10.0, 20.0])  # mean=15
        codes = np.array([0.0, 1.0])
        # With smooth=1: category 0 -> (1*15 + 10)/(1+1) = 12.5
        enc_past, _ = target_encode(target, codes, smooth=1.0)
        # Category 0 appears once with value 10
        # Category 1 appears once with value 20
        # Both have count=1, sum=10/20, item_mean=15
        # Encoded: (15 + 10)/(1+1) = 12.5, (15 + 20)/(1+1) = 17.5

    def test_empty_categories(self):
        """Test handling of empty category list."""
        target = np.array([np.nan, np.nan])  # all NaN target
        codes = np.array([np.nan, np.nan])  # all NaN codes
        enc_past, _ = target_encode(target, codes)
        # Should not crash, item_mean=0
        assert enc_past.shape == (2,)

    def test_shape_mismatch_raises(self):
        """Test that shape mismatch raises error."""
        target = np.array([1.0, 2.0, 3.0])
        codes = np.array([0.0, 1.0])  # wrong length
        with pytest.raises(ValueError, match="must align"):
            target_encode(target, codes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])