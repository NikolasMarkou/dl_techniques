"""
Test suite for ShearletTransform implementation.

This module contains unit tests for the ShearletTransform layer using pytest.
Tests cover basic functionality, input validation, and numerical properties.
"""

import pytest
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.shearlet_transform import ShearletTransform


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(params=[(32, 32), (64, 64), (128, 128)])
def input_size(request) -> Tuple[int, int]:
    """Provide different input sizes for testing.

    Returns:
        Tuple[int, int]: Height and width of the input
    """
    return request.param


@pytest.fixture(params=[
    {'scales': 3, 'directions': 8},
    {'scales': 4, 'directions': 12},
    {'scales': 2, 'directions': 4}
])
def transform_configs(request) -> Dict[str, int]:
    """Provide different ShearletTransform configurations.

    Returns:
        Dict[str, int]: Configuration parameters
    """
    return request.param


@pytest.fixture
def default_transform() -> ShearletTransform:
    """Provide a default ShearletTransform instance.

    Returns:
        ShearletTransform: Default configured transform layer
    """
    return ShearletTransform(scales=3, directions=8)


@pytest.fixture
def sample_input(input_size: Tuple[int, int]) -> Tuple[tf.Tensor, tuple]:
    """Generate sample input tensors.

    Args:
        input_size: Tuple of (height, width)

    Returns:
        Tuple[tf.Tensor, tuple]: Sample input tensor and shape
    """
    h, w = input_size
    shape = (2, h, w, 1)

    # Create synthetic patterns
    x, y = np.meshgrid(
        np.linspace(-2, 2, w),
        np.linspace(-2, 2, h)
    )

    # Gaussian blob
    gaussian = np.exp(-(x ** 2 + y ** 2) / 0.5)

    # Add directional pattern
    pattern = gaussian + 0.5 * np.sin(2 * np.pi * (x + y))

    x = np.zeros(shape)
    x[0, :, :, 0] = pattern
    x[1, :, :, 0] = gaussian

    return tf.convert_to_tensor(x, dtype=tf.float32), shape


# ---------------------------------------------------------------------
# Basic Functionality Tests
# ---------------------------------------------------------------------

def test_output_shape(default_transform: ShearletTransform,
                      sample_input: Tuple[tf.Tensor, tuple]) -> None:
    """Test if output shape matches expected dimensions.

    Args:
        default_transform: ShearletTransform instance
        sample_input: Sample input tensor and shape
    """
    input_tensor, _ = sample_input
    output = default_transform(input_tensor)

    # Channels = low-pass filter + shearlets at each scale and direction
    # For each scale: 2 * directions + 1 shearlets (including the vertical one)
    expected_channels = 1 + default_transform.scales * (default_transform.directions + 1)
    assert output.shape[-1] == expected_channels, \
        f"Expected {expected_channels} output channels, got {output.shape[-1]}"


# ---------------------------------------------------------------------
# Numerical Property Tests
# ---------------------------------------------------------------------

def test_directional_sensitivity(default_transform: ShearletTransform,
                                 input_size: Tuple[int, int]) -> None:
    """Test if transform responds correctly to directional patterns.

    Args:
        default_transform: ShearletTransform instance
        input_size: Input dimensions
    """
    h, w = input_size

    # Create directional pattern with higher frequency to better test directionality
    x, y = np.meshgrid(
        np.linspace(-4, 4, w),
        np.linspace(-4, 4, h)
    )

    # Test patterns at more distinct angles
    angles = [0, 90]  # Using more distinct angles
    responses = []

    for angle in angles:
        # Create pattern at specific angle with windowing
        theta = np.radians(angle)
        # Add Gaussian windowing to reduce edge effects
        window = np.exp(-(x ** 2 + y ** 2) / 8.0)
        pattern = np.sin(4 * np.pi * (x * np.cos(theta) + y * np.sin(theta))) * window

        input_tensor = tf.convert_to_tensor(
            pattern[np.newaxis, :, :, np.newaxis],
            dtype=tf.float32
        )

        output = default_transform(input_tensor)

        # Compute response magnitude for each direction separately
        response = tf.reduce_max(tf.abs(output), axis=[1, 2])  # Max response per channel
        responses.append(response)

    # Compare responses at orthogonal angles
    correlation = tf.reduce_mean(tf.multiply(responses[0], responses[1])) / \
                  (tf.norm(responses[0]) * tf.norm(responses[1]))

    assert correlation < 0.85, \
        f"Responses too similar for orthogonal angles: correlation = {correlation:.3f}"


def test_scale_sensitivity(default_transform: ShearletTransform,
                           input_size: Tuple[int, int]) -> None:
    """Test if transform responds appropriately to different scales.

    Args:
        default_transform: ShearletTransform instance
        input_size: Input dimensions
    """
    h, w = input_size

    # Create patterns at different scales with better scale separation
    x, y = np.meshgrid(
        np.linspace(-4, 4, w),
        np.linspace(-4, 4, h)
    )

    # Use more distinct scales
    scales = [1, 4]  # Larger scale separation
    responses = []

    for scale in scales:
        # Create pattern with Gaussian windowing
        window = np.exp(-(x ** 2 + y ** 2) / 8.0)
        pattern = np.sin(2 * np.pi * scale * (x + y)) * window

        input_tensor = tf.convert_to_tensor(
            pattern[np.newaxis, :, :, np.newaxis],
            dtype=tf.float32
        )

        output = default_transform(input_tensor)

        # Use maximum response per channel instead of mean
        response = tf.reduce_max(tf.abs(output), axis=[1, 2])
        responses.append(response)

    # Compare responses at different scales
    correlation = tf.reduce_mean(tf.multiply(responses[0], responses[1])) / \
                  (tf.norm(responses[0]) * tf.norm(responses[1]))

    assert correlation < 0.80, \
        f"Responses too similar for distinct scales: correlation = {correlation:.3f}"

    # Additional check: verify energy distribution across scales
    energy_scale1 = tf.reduce_sum(tf.square(responses[0]))
    energy_scale2 = tf.reduce_sum(tf.square(responses[1]))
    energy_ratio = energy_scale1 / energy_scale2

    assert 0.1 < energy_ratio < 10.0, \
        f"Unexpected energy distribution between scales: ratio = {energy_ratio:.3f}"


# ---------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------

def test_numerical_stability(default_transform: ShearletTransform,
                             sample_input: Tuple[tf.Tensor, tuple]) -> None:
    """Test numerical stability with extreme input values.

    Args:
        default_transform: ShearletTransform instance
        sample_input: Sample input tensor and shape
    """
    input_tensor, _ = sample_input

    # Test with very small values
    small_input = input_tensor * 1e-10
    output_small = default_transform(small_input)
    assert not tf.reduce_any(tf.math.is_nan(output_small)), \
        "Transform produced NaN values for small input"

    # Test with very large values
    large_input = input_tensor * 1e10
    output_large = default_transform(large_input)
    assert not tf.reduce_any(tf.math.is_nan(output_large)), \
        "Transform produced NaN values for large input"


def test_mixed_precision(default_transform: ShearletTransform,
                         sample_input: Tuple[tf.Tensor, tuple]) -> None:
    """Test behavior with different precision inputs.

    Args:
        default_transform: ShearletTransform instance
        sample_input: Sample input tensor and shape
    """
    input_tensor, _ = sample_input

    # Test with float16 input
    input_fp16 = tf.cast(input_tensor, tf.float16)
    output_fp16 = default_transform(input_fp16)
    assert output_fp16.dtype == tf.float32, \
        "Expected float32 output for float16 input"

    # Test with float64 input
    input_fp64 = tf.cast(input_tensor, tf.float64)
    output_fp64 = default_transform(input_fp64)
    assert output_fp64.dtype == tf.float32, \
        "Expected float32 output for float64 input"


# ---------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__])
