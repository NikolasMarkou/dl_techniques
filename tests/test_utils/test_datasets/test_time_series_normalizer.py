import pytest
import numpy as np
from typing import List

# Import the classes to test
from dl_techniques.utils.datasets.time_series_normalizer import (
    TimeSeriesNormalizer,
    NormalizationMethod
)

class TestTimeSeriesNormalizer:
    """Comprehensive test suite for TimeSeriesNormalizer class."""

    # =====================================================================
    # Fixtures
    # =====================================================================

    @pytest.fixture
    def simple_data(self) -> np.ndarray:
        """Simple test data: [1, 2, 3, 4, 5]."""
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    @pytest.fixture
    def complex_data(self) -> np.ndarray:
        """Complex test data with normal distribution."""
        np.random.seed(42)
        return np.random.normal(100, 15, 50)

    @pytest.fixture
    def outlier_data(self) -> np.ndarray:
        """Data with outliers."""
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 2.5, 3.5])

    @pytest.fixture
    def zero_variance_data(self) -> np.ndarray:
        """Data with zero variance (all same values)."""
        return np.array([5.0, 5.0, 5.0, 5.0, 5.0])

    @pytest.fixture
    def data_with_nans(self) -> np.ndarray:
        """Data containing NaN values."""
        return np.array([1.0, 2.0, np.nan, 4.0, 5.0])

    @pytest.fixture
    def negative_data(self) -> np.ndarray:
        """Data with negative values."""
        return np.array([-5.0, -2.0, 0.0, 2.0, 5.0])

    @pytest.fixture
    def large_data(self) -> np.ndarray:
        """Large dataset for quantile method testing."""
        np.random.seed(42)
        return np.random.exponential(2, 1000)

    @pytest.fixture
    def multidimensional_data(self) -> np.ndarray:
        """2D array for testing shape preservation."""
        return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # =====================================================================
    # Initialization Tests
    # =====================================================================

    def test_initialization_default(self):
        """Test default initialization."""
        normalizer = TimeSeriesNormalizer()
        assert normalizer.method == NormalizationMethod.MINMAX
        assert normalizer.feature_range == (0.0, 1.0)
        assert normalizer.epsilon == 1e-8
        assert not normalizer.fitted

    def test_initialization_with_string_method(self):
        """Test initialization with string method name."""
        normalizer = TimeSeriesNormalizer(method='standard')
        assert normalizer.method == NormalizationMethod.STANDARD

    def test_initialization_with_enum_method(self):
        """Test initialization with enum method."""
        normalizer = TimeSeriesNormalizer(method=NormalizationMethod.ROBUST)
        assert normalizer.method == NormalizationMethod.ROBUST

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        normalizer = TimeSeriesNormalizer(
            method='minmax',
            feature_range=(-1.0, 1.0),
            epsilon=1e-6,
            n_quantiles=500
        )
        assert normalizer.feature_range == (-1.0, 1.0)
        assert normalizer.epsilon == 1e-6
        assert normalizer.n_quantiles == 500

    def test_initialization_invalid_method_string(self):
        """Test initialization with invalid string method raises error."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            TimeSeriesNormalizer(method='invalid_method')

    def test_initialization_invalid_method_type(self):
        """Test initialization with invalid method type raises error."""
        with pytest.raises(ValueError, match="Method must be a string or NormalizationMethod enum"):
            TimeSeriesNormalizer(method=123)

    # =====================================================================
    # Basic Functionality Tests
    # =====================================================================

    def test_fit_returns_self(self, simple_data):
        """Test that fit() returns self for method chaining."""
        normalizer = TimeSeriesNormalizer()
        result = normalizer.fit(simple_data)
        assert result is normalizer

    def test_fit_sets_fitted_flag(self, simple_data):
        """Test that fit() sets the fitted flag."""
        normalizer = TimeSeriesNormalizer()
        assert not normalizer.fitted
        normalizer.fit(simple_data)
        assert normalizer.fitted

    def test_transform_requires_fit(self, simple_data):
        """Test that transform() requires fit() to be called first."""
        normalizer = TimeSeriesNormalizer()
        with pytest.raises(ValueError, match="Normalizer must be fitted before transform"):
            normalizer.transform(simple_data)

    def test_inverse_transform_requires_fit(self, simple_data):
        """Test that inverse_transform() requires fit() to be called first."""
        normalizer = TimeSeriesNormalizer()
        with pytest.raises(ValueError, match="Normalizer must be fitted before inverse_transform"):
            normalizer.inverse_transform(simple_data)

    def test_fit_transform_convenience_method(self, simple_data):
        """Test fit_transform() convenience method."""
        normalizer = TimeSeriesNormalizer(method='minmax')
        transformed = normalizer.fit_transform(simple_data)

        assert normalizer.fitted
        assert transformed.shape == simple_data.shape
        assert np.min(transformed) == pytest.approx(0.0, abs=1e-6)
        assert np.max(transformed) == pytest.approx(1.0, abs=1e-6)

    def test_shape_preservation(self, multidimensional_data):
        """Test that transformation preserves input shape."""
        normalizer = TimeSeriesNormalizer(method='standard')
        transformed = normalizer.fit_transform(multidimensional_data)
        assert transformed.shape == multidimensional_data.shape

    # =====================================================================
    # Method-Specific Tests
    # =====================================================================

    def test_do_nothing_method(self, simple_data):
        """Test do_nothing method returns identical data."""
        normalizer = TimeSeriesNormalizer(method='do_nothing')
        transformed = normalizer.fit_transform(simple_data)
        np.testing.assert_array_equal(transformed, simple_data)

        # Test inverse transform
        reconstructed = normalizer.inverse_transform(transformed)
        np.testing.assert_array_equal(reconstructed, simple_data)

    def test_minmax_method(self, simple_data):
        """Test minmax normalization."""
        normalizer = TimeSeriesNormalizer(method='minmax')
        transformed = normalizer.fit_transform(simple_data)

        # Should be in range [0, 1]
        assert np.min(transformed) == pytest.approx(0.0, abs=1e-6)
        assert np.max(transformed) == pytest.approx(1.0, abs=1e-6)

        # Test with custom range
        normalizer_custom = TimeSeriesNormalizer(method='minmax', feature_range=(-1, 1))
        transformed_custom = normalizer_custom.fit_transform(simple_data)
        assert np.min(transformed_custom) == pytest.approx(-1.0, abs=1e-6)
        assert np.max(transformed_custom) == pytest.approx(1.0, abs=1e-6)

    def test_standard_method(self, simple_data):
        """Test standard (z-score) normalization."""
        normalizer = TimeSeriesNormalizer(method='standard')
        transformed = normalizer.fit_transform(simple_data)

        # Should have approximately zero mean and unit variance
        assert np.mean(transformed) == pytest.approx(0.0, abs=1e-6)
        assert np.std(transformed) == pytest.approx(1.0, abs=1e-6)

    def test_robust_method(self, outlier_data):
        """Test robust normalization with outlier data."""
        normalizer = TimeSeriesNormalizer(method='robust')
        transformed = normalizer.fit_transform(outlier_data)

        # Should handle outliers better than standard normalization
        assert not np.any(np.isinf(transformed))
        assert not np.any(np.isnan(transformed))

    def test_unit_vector_method(self, simple_data):
        """Test unit vector normalization."""
        normalizer = TimeSeriesNormalizer(method='unit_vector')
        transformed = normalizer.fit_transform(simple_data)

        # Check that it has unit norm
        norm = np.linalg.norm(transformed)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_max_abs_method(self, negative_data):
        """Test maximum absolute value normalization."""
        normalizer = TimeSeriesNormalizer(method='max_abs')
        transformed = normalizer.fit_transform(negative_data)

        # Should be in range [-1, 1]
        assert np.min(transformed) >= -1.0
        assert np.max(transformed) <= 1.0
        assert (np.max(np.abs(transformed)) == pytest.approx(1.0, abs=1e-6))

    def test_mad_method(self, outlier_data):
        """Test Median Absolute Deviation normalization."""
        normalizer = TimeSeriesNormalizer(method='mad')
        transformed = normalizer.fit_transform(outlier_data)

        # Should be robust to outliers
        assert not np.any(np.isinf(transformed))
        assert not np.any(np.isnan(transformed))

    def test_tanh_method(self, simple_data):
        """Test hyperbolic tangent normalization."""
        normalizer = TimeSeriesNormalizer(method='tanh')
        transformed = normalizer.fit_transform(simple_data)

        # Should be in range (-1, 1)
        assert np.all(transformed > -1.0)
        assert np.all(transformed < 1.0)

    def test_decimal_method(self, simple_data):
        """Test decimal scaling normalization."""
        large_data = simple_data * 1000  # Make data larger
        normalizer = TimeSeriesNormalizer(method='decimal')
        transformed = normalizer.fit_transform(large_data)

        # Should reduce magnitude
        assert np.max(np.abs(transformed)) < np.max(np.abs(large_data))

    def test_percent_change_method(self, simple_data):
        """Test percentage change normalization."""
        normalizer = TimeSeriesNormalizer(method='percent_change')
        transformed = normalizer.fit_transform(simple_data)

        # First value should be 0 (no change from itself)
        assert transformed[0] == pytest.approx(0.0, abs=1e-6)

    # =====================================================================
    # Parametrized Tests
    # =====================================================================

    @pytest.mark.parametrize("method", [
        'do_nothing', 'minmax', 'standard', 'robust', 'max_abs',
        'mad', 'tanh', 'decimal', 'percent_change'
    ])
    def test_perfect_inverse_methods(self, method, simple_data):
        """Test that methods with perfect inverse reconstruct data exactly."""
        normalizer = TimeSeriesNormalizer(method=method)

        if normalizer.supports_perfect_inverse:
            transformed = normalizer.fit_transform(simple_data)
            reconstructed = normalizer.inverse_transform(transformed)

            np.testing.assert_allclose(
                simple_data, reconstructed, rtol=1e-6, atol=1e-6,
                err_msg=f"Perfect reconstruction failed for method: {method}"
            )

    @pytest.mark.parametrize("method", list(NormalizationMethod))
    def test_all_methods_basic_functionality(self, method, complex_data):
        """Test that all methods can fit and transform without errors."""
        normalizer = TimeSeriesNormalizer(method=method)

        # Should not raise exceptions
        transformed = normalizer.fit_transform(complex_data)
        assert transformed.shape == complex_data.shape

        # Should not contain infinite values
        assert not np.any(np.isinf(transformed))

    def test_numerical_precision_bounds(self, simple_data):
        """Test that numerical precision is within acceptable bounds."""
        # Test that our tolerance expectations are reasonable
        normalizer = TimeSeriesNormalizer(method='minmax')
        transformed = normalizer.fit_transform(simple_data)

        # Check that we're within reasonable floating point precision
        min_val = np.min(transformed)
        max_val = np.max(transformed)

        # These should be very close to 0 and 1, but allow for floating point error
        assert abs(min_val - 0.0) < 1e-6, f"Min value {min_val} too far from 0.0"
        assert abs(max_val - 1.0) < 1e-6, f"Max value {max_val} too far from 1.0"

        # But they should also not be exactly equal (due to floating point)
        # This helps verify our tests are realistic
        assert min_val >= 0.0, "Min value should not be negative"
        assert max_val <= 1.0, "Max value should not exceed 1.0"

    # =====================================================================
    # Edge Case Tests
    # =====================================================================

    def test_empty_data_raises_error(self):
        """Test that empty data raises appropriate error."""
        normalizer = TimeSeriesNormalizer()
        empty_data = np.array([])

        with pytest.raises(ValueError, match="Cannot fit normalizer on empty data"):
            normalizer.fit(empty_data)

    def test_all_nan_data_raises_error(self):
        """Test that data with all NaN values raises error."""
        normalizer = TimeSeriesNormalizer()
        all_nan_data = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="Cannot fit normalizer on data containing only NaN values"):
            normalizer.fit(all_nan_data)

    def test_data_with_some_nans(self, data_with_nans):
        """Test handling of data with some NaN values."""
        normalizer = TimeSeriesNormalizer(method='standard')

        # Should fit successfully (uses nanmean, nanstd, etc.)
        transformed = normalizer.fit_transform(data_with_nans)

        # NaN positions should remain NaN
        nan_mask = np.isnan(data_with_nans)
        assert np.all(np.isnan(transformed[nan_mask]))

        # Non-NaN values should be transformed
        valid_mask = ~nan_mask
        assert not np.any(np.isnan(transformed[valid_mask]))

    def test_zero_variance_data_handling(self, zero_variance_data):
        """Test handling of data with zero variance."""
        normalizer = TimeSeriesNormalizer(method='standard')
        transformed = normalizer.fit_transform(zero_variance_data)

        # Should handle gracefully (returns zero-mean data)
        assert not np.any(np.isinf(transformed))
        assert not np.any(np.isnan(transformed))

    def test_single_value_data(self):
        """Test handling of single-value data."""
        single_value = np.array([5.0])
        normalizer = TimeSeriesNormalizer(method='minmax')
        transformed = normalizer.fit_transform(single_value)

        # Should handle gracefully
        assert transformed.shape == single_value.shape
        assert not np.any(np.isinf(transformed))

    # =====================================================================
    # Property and Utility Tests
    # =====================================================================

    def test_available_methods_property(self):
        """Test available_methods property returns all methods."""
        normalizer = TimeSeriesNormalizer()
        methods = normalizer.available_methods

        # Should include all enum values
        expected_methods = [method.value for method in NormalizationMethod]
        assert set(methods) == set(expected_methods)
        assert len(methods) == len(NormalizationMethod)

    def test_supports_perfect_inverse_property(self):
        """Test supports_perfect_inverse property."""
        # Test methods that support perfect inverse
        perfect_methods = ['do_nothing', 'minmax', 'standard', 'robust']
        for method in perfect_methods:
            normalizer = TimeSeriesNormalizer(method=method)
            assert normalizer.supports_perfect_inverse

        # Test methods that don't support perfect inverse
        imperfect_methods = ['unit_vector', 'quantile_uniform', 'power']
        for method in imperfect_methods:
            normalizer = TimeSeriesNormalizer(method=method)
            assert not normalizer.supports_perfect_inverse

    def test_get_method_info_class_method(self):
        """Test get_method_info class method."""
        info = TimeSeriesNormalizer.get_method_info()

        # Should be a dictionary with all methods
        assert isinstance(info, dict)
        assert len(info) == len(NormalizationMethod)

        # Each method should have required keys
        for method_name, method_info in info.items():
            assert 'description' in method_info
            assert 'output_range' in method_info
            assert 'perfect_inverse' in method_info
            assert 'robust_to_outliers' in method_info
            assert 'use_case' in method_info

    def test_get_statistics_requires_fit(self):
        """Test that get_statistics requires fitting first."""
        normalizer = TimeSeriesNormalizer()

        with pytest.raises(ValueError, match="Normalizer must be fitted to get statistics"):
            normalizer.get_statistics()

    def test_get_statistics_after_fit(self, simple_data):
        """Test get_statistics after fitting."""
        normalizer = TimeSeriesNormalizer(method='standard')
        normalizer.fit(simple_data)

        stats = normalizer.get_statistics()
        assert isinstance(stats, dict)
        assert stats['method'] == 'standard'
        assert stats['fitted'] is True
        assert 'mean_val' in stats
        assert 'std_val' in stats

    def test_repr_string(self):
        """Test string representation."""
        normalizer = TimeSeriesNormalizer(method='standard')
        repr_str = repr(normalizer)

        assert 'TimeSeriesNormalizer' in repr_str
        assert 'standard' in repr_str
        assert 'not fitted' in repr_str

    def test_summary_method(self, simple_data):
        """Test summary method."""
        normalizer = TimeSeriesNormalizer(method='standard')

        # Test before fitting
        summary_before = normalizer.summary()
        assert 'Not fitted' in summary_before

        # Test after fitting
        normalizer.fit(simple_data)
        summary_after = normalizer.summary()
        assert 'Fitted' in summary_after
        assert 'Fitted Parameters' in summary_after

    # =====================================================================
    # Quantile Method Tests
    # =====================================================================

    def test_quantile_uniform_method(self, large_data):
        """Test quantile uniform transformation."""
        normalizer = TimeSeriesNormalizer(method='quantile_uniform')
        transformed = normalizer.fit_transform(large_data)

        # Should be in range [0, 1]
        assert np.min(transformed) >= 0.0
        assert np.max(transformed) <= 1.0

    def test_quantile_normal_method(self, large_data):
        """Test quantile normal transformation."""
        normalizer = TimeSeriesNormalizer(method='quantile_normal')
        transformed = normalizer.fit_transform(large_data)

        # Should be approximately normally distributed
        # (just check it doesn't crash and produces reasonable output)
        assert not np.any(np.isinf(transformed))
        assert not np.any(np.isnan(transformed))

    # =====================================================================
    # Integration Tests
    # =====================================================================

    def test_multiple_transform_calls(self, simple_data, complex_data):
        """Test that normalizer can transform different datasets after fitting."""
        normalizer = TimeSeriesNormalizer(method='standard')
        normalizer.fit(simple_data)

        # Transform original data
        transformed1 = normalizer.transform(simple_data)

        # Transform different data with same normalizer
        transformed2 = normalizer.transform(complex_data)

        # Both should work
        assert transformed1.shape == simple_data.shape
        assert transformed2.shape == complex_data.shape

    def test_refit_changes_parameters(self, simple_data, complex_data):
        """Test that refitting changes the normalizer parameters."""
        normalizer = TimeSeriesNormalizer(method='standard')

        # Fit on first dataset
        normalizer.fit(simple_data)
        stats1 = normalizer.get_statistics()

        # Fit on second dataset
        normalizer.fit(complex_data)
        stats2 = normalizer.get_statistics()

        # Parameters should be different
        assert stats1['mean_val'] != stats2['mean_val']
        assert stats1['std_val'] != stats2['std_val']


# =====================================================================
# Test Fixtures and Utilities
# =====================================================================

@pytest.fixture
def all_normalization_methods() -> List[str]:
    """Fixture providing all normalization method names."""
    return [method.value for method in NormalizationMethod]


# =====================================================================
# Performance/Stress Tests (Optional)
# =====================================================================

class TestTimeSeriesNormalizerPerformance:
    """Performance and stress tests for TimeSeriesNormalizer."""

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Generate large dataset
        np.random.seed(42)
        large_data = np.random.normal(0, 1, 100000)

        normalizer = TimeSeriesNormalizer(method='standard')

        # Should complete without timeout (pytest default timeout handling)
        transformed = normalizer.fit_transform(large_data)
        assert transformed.shape == large_data.shape

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        # Create 3D array
        data_3d = np.random.normal(0, 1, (100, 50, 20))

        normalizer = TimeSeriesNormalizer(method='minmax')
        transformed = normalizer.fit_transform(data_3d)

        assert transformed.shape == data_3d.shape
        # Global min/max should be respected (with some tolerance for floating point)
        assert np.min(transformed) == pytest.approx(0.0, abs=1e-6)
        assert np.max(transformed) == pytest.approx(1.0, abs=1e-6)


# =====================================================================
# Run Tests
# =====================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])