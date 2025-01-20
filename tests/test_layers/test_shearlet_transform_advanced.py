"""
Test suite for ShearletTransform implementation.

This module contains unit tests for the ShearletTransform layer using pytest.
Tests cover basic functionality, input validation, and numerical properties.
"""

import pytest
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.shearlet_transform import ShearletTransform


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(params=[
    {'scales': 2, 'directions': 4},
    {'scales': 3, 'directions': 8},
    {'scales': 4, 'directions': 12}
])
def transform_config(request) -> Dict[str, int]:
    """Fixture providing different filter configurations.

    Returns:
        Dict[str, int]: Configuration parameters
    """
    return request.param


@pytest.fixture(params=[(32, 32), (64, 64), (128, 128)])
def input_shape(request) -> Tuple[int, int]:
    """Fixture providing test input shapes.

    Returns:
        Tuple[int, int]: (height, width) pair
    """
    return request.param


@pytest.fixture
def transform(transform_config: Dict[str, int]) -> ShearletTransform:
    """Fixture creating ShearletTransform instance.

    Args:
        transform_config: Configuration parameters

    Returns:
        ShearletTransform: Configured transform instance
    """
    return ShearletTransform(**transform_config)


@pytest.fixture
def built_transform(transform: ShearletTransform) -> ShearletTransform:
    """Fixture providing built transform with default size.

    Args:
        transform: Base transform instance

    Returns:
        ShearletTransform: Built transform instance
    """
    transform.build(
        input_shape=(1, 64, 64, 1)
    )

    return transform


# ---------------------------------------------------------------------
# Filter Creation Tests
# ---------------------------------------------------------------------

def test_filter_creation(transform_config: Dict[str, int], transform: ShearletTransform) -> None:
    """Test basic filter creation properties.

    Args:
        transform_config: Configuration parameters
        transform: Transform instance
    """
    transform.build((1, 64, 64, 1))
    filters = transform.filters

    # Check number of filters
    expected_filters = 1 + transform_config['scales'] * (transform_config['directions'] + 1)
    assert len(filters) == expected_filters, \
        f"Expected {expected_filters} filters, got {len(filters)}"

    # Check filter shapes
    for i, filter_kernel in enumerate(filters):
        assert filter_kernel.shape == (64, 64), \
            f"Filter {i} has wrong shape: {filter_kernel.shape}"

        # Check for complex values
        assert filter_kernel.dtype == tf.complex64, \
            f"Filter {i} has wrong dtype: {filter_kernel.dtype}"


def test_filter_size_adaptation(transform: ShearletTransform, input_shape: Tuple[int, int]) -> None:
    """Test if filters adapt to different input sizes.

    Args:
        transform: Transform instance
        input_shape: Input dimensions
    """
    transform.build((1, *input_shape, 1))

    for filter_kernel in transform.filters:
        assert filter_kernel.shape == input_shape, \
            f"Filter shape {filter_kernel.shape} doesn't match input {input_shape}"


# ---------------------------------------------------------------------
# Frequency Response Tests
# ---------------------------------------------------------------------

def test_frequency_coverage(built_transform: ShearletTransform) -> None:
    """Test if filters provide adequate frequency coverage.

    Args:
        built_transform: Built transform instance
    """
    # Stack magnitude responses and normalize
    responses = tf.stack([tf.abs(f) for f in built_transform.filters])
    responses = responses / (tf.reduce_max(responses) + 1e-10)  # Normalize with epsilon
    total_response = tf.reduce_sum(responses, axis=0)

    # Check if total response is approximately uniform
    # Allow for more variation since shearlets naturally have varying frequency coverage
    mean_response = tf.reduce_mean(total_response)
    std_response = tf.math.reduce_std(total_response)

    # Relaxed threshold since shearlet coverage is naturally non-uniform
    assert std_response / (mean_response + 1e-10) < 2.0, \
        "Frequency coverage is extremely uneven"

    # Check for severe coverage gaps
    min_response = tf.reduce_min(total_response)
    assert min_response > 1e-3, \
        f"Found severe coverage gap with minimum response {min_response}"


def test_low_pass_filter(built_transform: ShearletTransform) -> None:
    """Test properties of the low-pass filter.

    Args:
        built_transform: Built transform instance
    """
    low_pass = built_transform.filters[0]
    response = tf.abs(low_pass)

    # Normalize the response
    response = response / (tf.reduce_max(response) + 1e-10)

    # Check DC response (center point)
    dc_response = response[32, 32]
    assert tf.abs(dc_response - 1.0) < 0.2, \
        f"Low-pass DC response is {dc_response}, expected close to 1.0"

    # Check high-frequency attenuation
    edge_points = tf.concat([
        response[0:5, 0:5],
        response[-5:, 0:5],
        response[0:5, -5:],
        response[-5:, -5:]
    ], axis=0)
    edge_response = tf.reduce_mean(edge_points)
    assert edge_response < 0.3, \
        f"Low-pass high-frequency response is {edge_response}, expected <0.3"


# ---------------------------------------------------------------------
# Directional Selectivity Tests
# ---------------------------------------------------------------------

@pytest.fixture
def angle_patterns() -> List[Tuple[float, tf.Tensor]]:
    """Fixture providing test patterns at different angles.

    Returns:
        List[Tuple[float, tf.Tensor]]: List of (angle, pattern) pairs
    """
    x, y = np.meshgrid(
        np.linspace(-1, 1, 64),
        np.linspace(-1, 1, 64)
    )

    patterns = []
    for angle in [0, 45, 90, 135]:
        theta = np.radians(angle)
        pattern = np.sin(8 * np.pi * (x * np.cos(theta) + y * np.sin(theta)))
        pattern = pattern * np.exp(-(x ** 2 + y ** 2) / 0.5)  # Apply window

        tensor = tf.convert_to_tensor(
            pattern[np.newaxis, :, :, np.newaxis],
            dtype=tf.float32
        )
        patterns.append((angle, tensor))

    return patterns


def test_directional_response(built_transform: ShearletTransform,
                              angle_patterns: List[Tuple[float, tf.Tensor]]) -> None:
    """Test directional selectivity of the filters.

    Args:
        built_transform: Built transform instance
        angle_patterns: Test patterns at different angles
    """
    responses = []

    for angle, pattern in angle_patterns:
        output = built_transform(pattern)
        responses.append(tf.reduce_max(tf.abs(output), axis=[1, 2]))

    # Compare responses at orthogonal angles (0° vs 90°)
    correlation = tf.reduce_mean(
        tf.multiply(responses[0], responses[2])
    ) / (tf.norm(responses[0]) * tf.norm(responses[2]))

    assert correlation < 0.3, \
        f"High correlation {correlation} between orthogonal directions"


@pytest.fixture
def scale_patterns() -> List[Tuple[int, tf.Tensor]]:
    """Fixture providing test patterns at different scales.

    Returns:
        List[Tuple[int, tf.Tensor]]: List of (scale, pattern) pairs
    """
    x, y = np.meshgrid(
        np.linspace(-1, 1, 64),
        np.linspace(-1, 1, 64)
    )

    patterns = []
    for scale in [1, 4, 16]:
        pattern = np.sin(scale * np.pi * (x + y))
        pattern = pattern * np.exp(-(x ** 2 + y ** 2) / 0.5)

        tensor = tf.convert_to_tensor(
            pattern[np.newaxis, :, :, np.newaxis],
            dtype=tf.float32
        )
        patterns.append((scale, tensor))

    return patterns


def test_scale_separation(built_transform: ShearletTransform,
                          scale_patterns: List[Tuple[int, tf.Tensor]]) -> None:
    """Test scale selectivity of the filters.

    Args:
        built_transform: Built transform instance
        scale_patterns: Test patterns at different scales
    """
    responses = []

    for scale, pattern in scale_patterns:
        output = built_transform(pattern)
        responses.append(tf.reduce_max(tf.abs(output), axis=[1, 2]))

    # Check scale separation
    for i in range(len(responses) - 1):
        correlation = tf.reduce_mean(
            tf.multiply(responses[i], responses[i + 1])
        ) / (tf.norm(responses[i]) * tf.norm(responses[i + 1]))

        assert correlation < 0.5, \
            f"High correlation {correlation} between scales {scale_patterns[i][0]} and {scale_patterns[i + 1][0]}"


# ---------------------------------------------------------------------
# Frame Property Tests
# ---------------------------------------------------------------------

def test_shearlet_properties():
    """Test critical properties of the shearlet transform."""
    # Create test instance
    transform = ShearletTransform(scales=3, directions=8)
    transform.build((1, 64, 64, 1))

    # Test frequency coverage
    responses = tf.stack([tf.abs(f) for f in transform.filters])
    total_response = tf.reduce_sum(responses, axis=0)

    min_response = tf.reduce_min(total_response)
    assert min_response > 1e-3, "Coverage gap detected"

    # Test frame bounds
    responses_squared = tf.stack([tf.abs(f) ** 2 for f in transform.filters])
    total_energy = tf.reduce_sum(responses_squared, axis=0)

    min_energy = tf.reduce_min(total_energy)
    max_energy = tf.reduce_max(total_energy)
    frame_ratio = max_energy / (min_energy + 1e-6)

    assert frame_ratio < 4.0, "Frame bounds too large"

    # Test energy preservation
    mean_energy = tf.reduce_mean(total_energy)
    assert abs(mean_energy - 1.0) < 0.2, "Energy not preserved"


def test_frame_bounds(built_transform: ShearletTransform) -> None:
    """Test if filters satisfy frame bounds properties.

    Args:
        built_transform: Built transform instance

    Notes:
        Tests the following properties:
        1. Frame bounds ratio should be reasonably bounded (tight frame property)
        2. Energy preservation (Parseval-like property)
        3. Non-zero coverage across frequency domain
    """
    # Stack squared magnitude responses
    responses = tf.stack([tf.abs(f) ** 2 for f in built_transform.filters])
    total_energy = tf.reduce_sum(responses, axis=0)

    # Get frame bounds
    min_energy = tf.reduce_min(total_energy)
    max_energy = tf.reduce_max(total_energy)

    # Test 1: Check if frame is reasonably tight
    # Standard practice allows ratio up to 4.0 for shearlet frames
    assert max_energy / (min_energy + 1e-5) < 4.0, \
        f"Frame bounds ratio {max_energy / (min_energy + 1e-5)} too large"

    # Test 2: Check energy preservation (mean should be close to 1.0)
    mean_energy = tf.reduce_mean(total_energy)
    assert tf.abs(mean_energy - 1.0) < 0.2, \
        f"Mean energy {mean_energy} far from 1.0"

    # Test 3: Ensure non-zero coverage
    assert min_energy > 1e-3, \
        f"Minimum energy {min_energy} too close to zero"


# ---------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__])
