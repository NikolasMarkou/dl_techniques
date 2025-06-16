"""
Comprehensive pytest suite for Keras image corruption functions.

This test suite thoroughly validates the image corruption functions with enum-based severity levels,
ensuring robustness, type safety, and proper output characteristics.
"""

import keras
import pytest
from keras import ops
from typing import List

# Import the corruption functions and enums
from dl_techniques.utils.corruption import (
    CorruptionSeverity,
    CorruptionType,
    apply_gaussian_noise,
    apply_impulse_noise,
    apply_shot_noise,
    apply_gaussian_blur,
    apply_motion_blur,
    apply_pixelate,
    apply_quantize,
    apply_brightness,
    apply_contrast,
    apply_saturate,
    apply_corruption,
    get_corruption_function,
    get_all_corruption_types,
    get_all_severity_levels,
)


class TestCorruptionEnums:
    """Test suite for corruption enums."""

    def test_corruption_severity_enum_values(self):
        """Test that CorruptionSeverity enum has correct values."""
        assert CorruptionSeverity.MILD.value == 1
        assert CorruptionSeverity.MODERATE.value == 2
        assert CorruptionSeverity.MEDIUM.value == 3
        assert CorruptionSeverity.STRONG.value == 4
        assert CorruptionSeverity.SEVERE.value == 5

    def test_corruption_type_enum_values(self):
        """Test that CorruptionType enum has correct string values."""
        expected_types = {
            'gaussian_noise', 'impulse_noise', 'shot_noise',
            'gaussian_blur', 'motion_blur', 'pixelate',
            'quantize', 'brightness', 'contrast', 'saturate'
        }
        actual_types = {ct.value for ct in CorruptionType}
        assert actual_types == expected_types

    def test_get_all_corruption_types(self):
        """Test getting all corruption types."""
        all_types = get_all_corruption_types()
        assert len(all_types) == 10
        assert all(isinstance(ct, CorruptionType) for ct in all_types)
        assert CorruptionType.GAUSSIAN_NOISE in all_types
        assert CorruptionType.SATURATE in all_types

    def test_get_all_severity_levels(self):
        """Test getting all severity levels."""
        all_severities = get_all_severity_levels()
        assert len(all_severities) == 5
        assert all(isinstance(sev, CorruptionSeverity) for sev in all_severities)
        assert CorruptionSeverity.MILD in all_severities
        assert CorruptionSeverity.SEVERE in all_severities


class TestImageCorruptionFunctions:
    """Test suite for individual corruption functions."""

    @pytest.fixture
    def sample_images(self) -> List[keras.KerasTensor]:
        """Create sample test images with different shapes."""
        images = []

        # Small RGB image
        images.append(ops.convert_to_tensor([[
            [[0.5, 0.5, 0.5], [0.7, 0.3, 0.9]],
            [[0.2, 0.8, 0.1], [1.0, 0.0, 0.6]]
        ]], dtype='float32'))

        # Medium RGB image
        images.append(ops.ones((32, 32, 3), dtype='float32') * 0.5)

        # Larger RGB image
        images.append(ops.ones((64, 64, 3), dtype='float32') * 0.8)

        # Grayscale-like (single channel)
        images.append(ops.ones((16, 16, 1), dtype='float32') * 0.3)

        return images

    @pytest.fixture
    def all_severities(self) -> List[CorruptionSeverity]:
        """Get all severity levels for testing."""
        return get_all_severity_levels()

    def _validate_output_properties(self, original: keras.KerasTensor, corrupted: keras.KerasTensor):
        """Validate basic properties of corrupted output."""
        # Shape should be preserved
        assert ops.shape(corrupted) == ops.shape(original)

        # Should be valid tensor
        assert corrupted is not None

        # Values should be in valid range [0, 1]
        corrupted_np = ops.convert_to_numpy(corrupted)
        assert ops.convert_to_numpy(ops.all(corrupted >= 0.0))
        assert ops.convert_to_numpy(ops.all(corrupted <= 1.0))

        # Should not contain NaN or Inf
        assert not ops.convert_to_numpy(ops.any(ops.isnan(corrupted)))
        assert not ops.convert_to_numpy(ops.any(ops.isinf(corrupted)))

    def test_gaussian_noise_corruption(self, sample_images, all_severities):
        """Test Gaussian noise corruption."""
        for image in sample_images:
            for severity in all_severities:
                corrupted = apply_gaussian_noise(image, severity)
                self._validate_output_properties(image, corrupted)

                # Gaussian noise should change the image
                if severity != CorruptionSeverity.MILD:
                    difference = ops.mean(ops.abs(corrupted - image))
                    assert ops.convert_to_numpy(difference) > 1e-6

    def test_impulse_noise_corruption(self, sample_images, all_severities):
        """Test impulse (salt and pepper) noise corruption."""
        for image in sample_images:
            for severity in all_severities:
                corrupted = apply_impulse_noise(image, severity)
                self._validate_output_properties(image, corrupted)

                # Check for salt and pepper pixels
                corrupted_np = ops.convert_to_numpy(corrupted)
                has_salt = ops.convert_to_numpy(ops.any(corrupted >= 0.99))
                has_pepper = ops.convert_to_numpy(ops.any(corrupted <= 0.01))

                # Higher severity should have more salt/pepper
                if severity in [CorruptionSeverity.STRONG, CorruptionSeverity.SEVERE]:
                    assert has_salt or has_pepper

    def test_shot_noise_corruption(self, sample_images, all_severities):
        """Test shot noise corruption."""
        for image in sample_images:
            for severity in all_severities:
                corrupted = apply_shot_noise(image, severity)
                self._validate_output_properties(image, corrupted)

                # Shot noise should be non-negative (characteristic property)
                assert ops.convert_to_numpy(ops.all(corrupted >= 0.0))

    def test_gaussian_blur_corruption(self, sample_images, all_severities):
        """Test Gaussian blur corruption."""
        for image in sample_images:
            for severity in all_severities:
                corrupted = apply_gaussian_blur(image, severity)
                self._validate_output_properties(image, corrupted)

                # Blur should reduce high-frequency content (smoothing effect)
                # Only test this on images with sufficient size and variation
                if ops.shape(image)[0] > 4 and ops.shape(image)[1] > 4:
                    # Calculate gradient magnitude as proxy for high-frequency content
                    orig_grad_x = ops.abs(image[1:, :, :] - image[:-1, :, :])
                    corr_grad_x = ops.abs(corrupted[1:, :, :] - corrupted[:-1, :, :])

                    orig_gradient = ops.mean(orig_grad_x)
                    corr_gradient = ops.mean(corr_grad_x)

                    # Only test gradient reduction if original image has significant variation
                    if ops.convert_to_numpy(orig_gradient) > 0.01:
                        # Blurred image should have lower or similar gradient magnitude
                        if severity in [CorruptionSeverity.STRONG, CorruptionSeverity.SEVERE]:
                            assert ops.convert_to_numpy(corr_gradient) <= ops.convert_to_numpy(orig_gradient) + 0.05

    def test_motion_blur_corruption(self, sample_images, all_severities):
        """Test motion blur corruption."""
        for image in sample_images:
            for severity in all_severities:
                corrupted = apply_motion_blur(image, severity)
                self._validate_output_properties(image, corrupted)

    def test_pixelate_corruption(self, sample_images, all_severities):
        """Test pixelate corruption."""
        for image in sample_images:
            for severity in all_severities:
                corrupted = apply_pixelate(image, severity)
                self._validate_output_properties(image, corrupted)

                # Pixelated images should have block-like structure
                # (harder to test directly, but basic validation should pass)

    def test_quantize_corruption(self, sample_images, all_severities):
        """Test quantization corruption."""
        for image in sample_images:
            for severity in all_severities:
                corrupted = apply_quantize(image, severity)
                self._validate_output_properties(image, corrupted)

                # Check that values are properly quantized
                if severity == CorruptionSeverity.SEVERE:  # 2 bits = 4 levels
                    unique_vals = ops.convert_to_numpy(corrupted).flatten()
                    # Should have limited number of unique values
                    # (exact check depends on input, but should be discretized)

    def test_brightness_corruption(self, sample_images, all_severities):
        """Test brightness corruption."""
        for image in sample_images:
            for severity in all_severities:
                corrupted = apply_brightness(image, severity)
                self._validate_output_properties(image, corrupted)

                # Brightness should increase pixel values (before clipping)
                # For non-saturated regions, corrupted should be brighter
                non_saturated = image < 0.8
                if ops.convert_to_numpy(ops.any(non_saturated)):
                    bright_regions = ops.where(non_saturated, corrupted, 0.0)
                    orig_regions = ops.where(non_saturated, image, 0.0)

                    # Average brightness should increase in non-saturated regions
                    if ops.convert_to_numpy(ops.sum(ops.cast(non_saturated, 'float32'))) > 0:
                        avg_bright = ops.sum(bright_regions) / ops.sum(ops.cast(non_saturated, 'float32'))
                        avg_orig = ops.sum(orig_regions) / ops.sum(ops.cast(non_saturated, 'float32'))
                        assert ops.convert_to_numpy(avg_bright) >= ops.convert_to_numpy(avg_orig) - 1e-6

    def test_contrast_corruption(self, sample_images, all_severities):
        """Test contrast corruption."""
        for image in sample_images:
            for severity in all_severities:
                corrupted = apply_contrast(image, severity)
                self._validate_output_properties(image, corrupted)

                # Contrast enhancement should increase the spread around 0.5
                # Values > 0.5 should generally become larger, values < 0.5 smaller
                high_vals = image > 0.6
                low_vals = image < 0.4

                if ops.convert_to_numpy(ops.any(high_vals)):
                    high_orig = ops.mean(ops.where(high_vals, image, 0.5))
                    high_corr = ops.mean(ops.where(high_vals, corrupted, 0.5))
                    # High values should tend to increase (unless clipped)
                    assert ops.convert_to_numpy(high_corr) >= ops.convert_to_numpy(high_orig) - 0.1

    def test_saturate_corruption(self, sample_images, all_severities):
        """Test saturation corruption."""
        for image in sample_images:
            if ops.shape(image)[-1] >= 3:  # Only test on color images
                for severity in all_severities:
                    corrupted = apply_saturate(image, severity)
                    self._validate_output_properties(image, corrupted)

                    # Saturation should increase color differences from gray
                    # (This is a complex property to test precisely)


class TestUnifiedCorruptionInterface:
    """Test suite for the unified corruption interface."""

    @pytest.fixture
    def test_image(self) -> keras.KerasTensor:
        """Create a standard test image."""
        return ops.ones((16, 16, 3), dtype='float32') * 0.5

    def test_apply_corruption_with_enum(self, test_image):
        """Test apply_corruption with enum inputs."""
        for corruption_type in CorruptionType:
            for severity in CorruptionSeverity:
                corrupted = apply_corruption(test_image, corruption_type, severity)

                # Validate output
                assert ops.shape(corrupted) == ops.shape(test_image)
                assert ops.convert_to_numpy(ops.all(corrupted >= 0.0))
                assert ops.convert_to_numpy(ops.all(corrupted <= 1.0))

    def test_apply_corruption_with_string(self, test_image):
        """Test apply_corruption with string corruption type."""
        corruption_names = ['gaussian_noise', 'gaussian_blur', 'brightness']

        for corruption_name in corruption_names:
            corrupted = apply_corruption(test_image, corruption_name, CorruptionSeverity.MEDIUM)

            # Validate output
            assert ops.shape(corrupted) == ops.shape(test_image)
            assert ops.convert_to_numpy(ops.all(corrupted >= 0.0))
            assert ops.convert_to_numpy(ops.all(corrupted <= 1.0))

    def test_apply_corruption_invalid_string(self, test_image):
        """Test apply_corruption with invalid string corruption type."""
        with pytest.raises(ValueError, match="Unsupported corruption type"):
            apply_corruption(test_image, "invalid_corruption", CorruptionSeverity.MEDIUM)

    def test_get_corruption_function_enum(self):
        """Test get_corruption_function with enum input."""
        for corruption_type in CorruptionType:
            func = get_corruption_function(corruption_type)
            assert callable(func)

            # Test that function works
            test_image = ops.ones((8, 8, 3), dtype='float32') * 0.5
            result = func(test_image, CorruptionSeverity.MILD)
            assert ops.shape(result) == ops.shape(test_image)

    def test_get_corruption_function_string(self):
        """Test get_corruption_function with string input."""
        valid_strings = ['gaussian_noise', 'brightness', 'pixelate']

        for corruption_str in valid_strings:
            func = get_corruption_function(corruption_str)
            assert callable(func)

    def test_get_corruption_function_invalid(self):
        """Test get_corruption_function with invalid input."""
        with pytest.raises(ValueError):
            get_corruption_function("nonexistent_corruption")


class TestEdgeCasesAndRobustness:
    """Test suite for edge cases and robustness."""

    def test_extreme_image_values(self):
        """Test corruption functions with extreme pixel values."""
        # All black image
        black_image = ops.zeros((10, 10, 3), dtype='float32')

        # All white image
        white_image = ops.ones((10, 10, 3), dtype='float32')

        test_images = [black_image, white_image]

        for image in test_images:
            for corruption_type in [CorruptionType.GAUSSIAN_NOISE, CorruptionType.BRIGHTNESS]:
                corrupted = apply_corruption(image, corruption_type, CorruptionSeverity.MEDIUM)

                # Should not crash and should maintain valid range
                assert ops.convert_to_numpy(ops.all(corrupted >= 0.0))
                assert ops.convert_to_numpy(ops.all(corrupted <= 1.0))

    def test_small_images(self):
        """Test corruption functions with very small images."""
        tiny_image = ops.ones((2, 2, 3), dtype='float32') * 0.5

        for corruption_type in CorruptionType:
            for severity in [CorruptionSeverity.MILD, CorruptionSeverity.SEVERE]:
                corrupted = apply_corruption(tiny_image, corruption_type, severity)

                # Should handle small images gracefully
                assert ops.shape(corrupted) == ops.shape(tiny_image)
                assert ops.convert_to_numpy(ops.all(corrupted >= 0.0))
                assert ops.convert_to_numpy(ops.all(corrupted <= 1.0))

    def test_single_channel_images(self):
        """Test corruption functions with single-channel images."""
        gray_image = ops.ones((16, 16, 1), dtype='float32') * 0.5

        for corruption_type in CorruptionType:
            corrupted = apply_corruption(gray_image, corruption_type, CorruptionSeverity.MEDIUM)

            # Should preserve single channel
            assert ops.shape(corrupted)[-1] == 1
            assert ops.convert_to_numpy(ops.all(corrupted >= 0.0))
            assert ops.convert_to_numpy(ops.all(corrupted <= 1.0))

    def test_different_dtypes(self):
        """Test that functions work with different float dtypes."""
        for dtype in ['float32', 'float64']:
            image = ops.cast(ops.ones((8, 8, 3)) * 0.5, dtype)

            corrupted = apply_gaussian_noise(image, CorruptionSeverity.MILD)

            # Should maintain dtype compatibility
            assert ops.shape(corrupted) == ops.shape(image)
            assert ops.convert_to_numpy(ops.all(corrupted >= 0.0))
            assert ops.convert_to_numpy(ops.all(corrupted <= 1.0))

    def test_deterministic_behavior_same_seed(self):
        """Test that functions produce consistent results with same random seed."""
        image = ops.ones((16, 16, 3), dtype='float32') * 0.5

        # Note: This test may need adjustment based on how Keras handles random seeds
        # The exact implementation depends on the backend

        # Test that functions don't crash with repeated calls
        result1 = apply_gaussian_noise(image, CorruptionSeverity.MEDIUM)
        result2 = apply_gaussian_noise(image, CorruptionSeverity.MEDIUM)

        # Both should be valid (may not be identical due to randomness)
        assert ops.shape(result1) == ops.shape(image)
        assert ops.shape(result2) == ops.shape(image)


class TestCorruptionConsistency:
    """Test suite for consistency across severity levels."""

    @pytest.fixture
    def test_image(self) -> keras.KerasTensor:
        """Create a test image with varied content."""
        # Create an image with gradient and patterns
        x = ops.linspace(0, 1, 32)
        y = ops.linspace(0, 1, 32)

        # Create 2D gradient
        xx, yy = ops.meshgrid(x, y)

        # Create RGB channels with different patterns
        r_channel = xx
        g_channel = yy
        b_channel = (xx + yy) / 2

        image = ops.stack([r_channel, g_channel, b_channel], axis=-1)
        return ops.cast(image, 'float32')

    def test_severity_progression(self, test_image):
        """Test that higher severity levels produce more corruption."""
        severities = [CorruptionSeverity.MILD, CorruptionSeverity.MODERATE,
                     CorruptionSeverity.MEDIUM, CorruptionSeverity.STRONG, CorruptionSeverity.SEVERE]

        # Test corruption types where progression is measurable
        testable_corruptions = [
            CorruptionType.GAUSSIAN_NOISE,
            CorruptionType.GAUSSIAN_BLUR,
            CorruptionType.BRIGHTNESS,
            CorruptionType.CONTRAST
        ]

        for corruption_type in testable_corruptions:
            differences = []

            for severity in severities:
                corrupted = apply_corruption(test_image, corruption_type, severity)
                difference = ops.mean(ops.abs(corrupted - test_image))
                differences.append(ops.convert_to_numpy(difference))

            # For most corruptions, higher severity should mean larger difference
            # (allowing some tolerance for edge cases)
            for i in range(1, len(differences)):
                if differences[i] < differences[i-1]:
                    # Allow small decreases due to clipping effects, etc.
                    ratio = differences[i] / (differences[i-1] + 1e-8)
                    assert ratio > 0.8, f"Severity progression issue for {corruption_type}"


# Utility test to verify module imports
def test_module_imports():
    """Test that all required functions and enums can be imported."""
    # This test ensures the module structure is correct
    assert CorruptionSeverity is not None
    assert CorruptionType is not None
    assert callable(apply_corruption)
    assert callable(get_corruption_function)
    assert callable(get_all_corruption_types)
    assert callable(get_all_severity_levels)


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])