"""Tests for Haar Wavelet initialization components.

This module contains test cases for the HaarWaveletInitializer and
related functionality, ensuring proper wavelet properties and behavior.
"""

import pytest
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, List

# Import your module here
from dl_techniques.initializers.haar_wavelet_initializer import (
    HaarWaveletInitializer,
    create_haar_depthwise_conv2d
)


@pytest.fixture
def basic_shape() -> Tuple[int, int, int, int]:
    """Fixture for standard 2x2 kernel shape."""
    return (2, 2, 3, 4)


@pytest.fixture
def haar_initializer() -> HaarWaveletInitializer:
    """Fixture for default HaarWaveletInitializer."""
    return HaarWaveletInitializer(scale=1.0)


def test_haar_initializer_shape(basic_shape: Tuple[int, int, int, int],
                                haar_initializer: HaarWaveletInitializer):
    """Test if HaarWaveletInitializer produces correct output shape and type.

    Args:
        basic_shape: Standard kernel shape fixture
        haar_initializer: Default initializer fixture
    """
    weights = haar_initializer(basic_shape)

    assert isinstance(weights, tf.Tensor)
    assert weights.shape == basic_shape
    assert weights.dtype == tf.float32


@pytest.mark.parametrize("invalid_shape", [
    (3, 3, 3, 4),  # Invalid kernel size
    (2, 3, 3, 4),  # Mismatched dimensions
    (4, 4, 3, 4),  # Too large kernel
    (1, 1, 3, 4),  # Too small kernel
])
def test_haar_initializer_invalid_shapes(invalid_shape: Tuple[int, ...],
                                         haar_initializer: HaarWaveletInitializer):
    """Test if HaarWaveletInitializer raises appropriate errors for invalid shapes.

    Args:
        invalid_shape: Various invalid kernel shapes to test
        haar_initializer: Default initializer fixture
    """
    with pytest.raises(ValueError) as exc_info:
        haar_initializer(invalid_shape)
    assert "Haar wavelets require 2x2 kernels" in str(exc_info.value)


@pytest.mark.parametrize("scale,expected_max", [
    (1.0, 1 / np.sqrt(2)),  # Standard scale
    (2.0, 2 / np.sqrt(2)),  # Double scale
    (0.5, 0.5 / np.sqrt(2)),  # Half scale
])
def test_haar_initializer_scale(scale: float, expected_max: float):
    """Test if scaling works correctly in HaarWaveletInitializer.

    Args:
        scale: Scaling factor to test
        expected_max: Expected maximum absolute value after scaling
    """
    initializer = HaarWaveletInitializer(scale=scale)
    shape = (2, 2, 1, 4)
    weights = initializer(shape)
    weights_np = weights.numpy()

    assert np.isclose(np.max(np.abs(weights_np)), expected_max, rtol=1e-5)


def test_haar_initializer_orthogonality(haar_initializer: HaarWaveletInitializer):
    """Test if generated wavelets maintain orthogonality properties.

    Args:
        haar_initializer: Default initializer fixture
    """
    shape = (2, 2, 1, 4)
    weights = haar_initializer(shape).numpy()

    # Test orthogonality between different wavelet components
    for i in range(4):
        for j in range(i + 1, 4):
            w1 = weights[:, :, 0, i].flatten()
            w2 = weights[:, :, 0, j].flatten()
            dot_product = np.dot(w1, w2)
            assert np.isclose(dot_product, 0, atol=1e-6)


def test_create_haar_depthwise_conv2d():
    """Test if create_haar_depthwise_conv2d creates valid layer with correct output."""
    input_shape = (28, 28, 3)
    channel_multiplier = 4

    layer = create_haar_depthwise_conv2d(
        input_shape=input_shape,
        channel_multiplier=channel_multiplier,
        kernel_regularizer='l2',
        trainable=False
    )

    # Test forward pass
    dummy_input = tf.random.normal((1,) + input_shape)
    output = layer(dummy_input)

    # Check output shape (should be halved due to stride=2)
    expected_shape = (1, 14, 14, 3 * channel_multiplier)
    assert output.shape == expected_shape

    # Verify layer configuration
    assert not layer.trainable
    assert layer.padding == 'valid'
    assert layer.strides == (2, 2)


def test_haar_initializer_invalid_scale():
    """Test if HaarWaveletInitializer properly validates scale parameter."""
    with pytest.raises(ValueError) as exc_info:
        HaarWaveletInitializer(scale=-1.0)
    assert "Scale must be positive" in str(exc_info.value)

    with pytest.raises(ValueError):
        HaarWaveletInitializer(scale=0.0)


def test_haar_initializer_serialization():
    """Test if HaarWaveletInitializer can be properly serialized and deserialized."""
    original_initializer = HaarWaveletInitializer(scale=1.5, seed=42)
    config = original_initializer.get_config()

    # Verify config contents
    assert config['scale'] == 1.5
    assert config['seed'] == 42

    # Test reconstruction
    new_initializer = HaarWaveletInitializer(**config)

    # Compare outputs
    shape = (2, 2, 1, 4)
    original_weights = original_initializer(shape)
    new_weights = new_initializer(shape)

    np.testing.assert_array_equal(
        original_weights.numpy(),
        new_weights.numpy()
    )


def test_create_haar_depthwise_conv2d_invalid_params():
    """Test if create_haar_depthwise_conv2d validates parameters correctly."""
    with pytest.raises(ValueError):
        create_haar_depthwise_conv2d(
            input_shape=(28, 28),  # Invalid input shape
            channel_multiplier=4
        )

    with pytest.raises(ValueError):
        create_haar_depthwise_conv2d(
            input_shape=(28, 28, 3),
            channel_multiplier=-1  # Invalid multiplier
        )

    with pytest.raises(ValueError):
        create_haar_depthwise_conv2d(
            input_shape=(28, 28, 3),
            channel_multiplier=3,  # Invalid for non-trainable
            trainable=False
        )


if __name__ == '__main__':
    pytest.main([__file__])