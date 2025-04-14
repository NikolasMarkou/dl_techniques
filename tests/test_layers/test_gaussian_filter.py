"""
Test suite for Gaussian filter implementation.

This module provides comprehensive tests for:
- gaussian_kernel function
- depthwise_gaussian_kernel function
- GaussianFilter Keras layer
"""

import pytest
import numpy as np
import tensorflow as tf
from typing import Tuple, List

from dl_techniques.layers.gaussian_filter import (
    depthwise_gaussian_kernel,
    GaussianFilter
)


# Test fixtures
@pytest.fixture
def kernel_sizes() -> List[Tuple[int, int]]:
    """Common kernel sizes for testing."""
    return [(3, 3), (5, 5), (7, 7)]


@pytest.fixture
def sigma_values() -> List[Tuple[float, float]]:
    """Common sigma values for testing."""
    return [(1.0, 1.0), (2.0, 2.0), (0.5, 0.5)]


@pytest.fixture
def sample_image() -> tf.Tensor:
    """Generate a sample image for testing."""
    # Create a simple 4D tensor (batch_size=1, height=32, width=32, channels=3)
    return tf.random.uniform((1, 32, 32, 3))


# Test depthwise_gaussian_kernel function
def test_depthwise_kernel_shape():
    """Test if depthwise_gaussian_kernel produces correct shapes."""
    channels = 3
    kernel_size = (5, 5)
    nsig = (2.0, 2.0)
    kernel = depthwise_gaussian_kernel(channels, kernel_size, nsig)
    assert kernel.shape == (*kernel_size, channels, 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_depthwise_kernel_dtype(dtype):
    """Test if depthwise_gaussian_kernel respects dtype parameter."""
    kernel = depthwise_gaussian_kernel(dtype=dtype)
    assert kernel.dtype == dtype


def test_depthwise_kernel_channel_independence():
    """Test if depthwise_gaussian_kernel creates independent filters per channel."""
    channels = 3
    kernel = depthwise_gaussian_kernel(channels=channels)
    # Check if each channel has the same kernel
    for i in range(channels):
        assert np.allclose(kernel[:, :, 0, 0], kernel[:, :, i, 0])


# Test GaussianFilter layer
def test_gaussian_filter_initialization():
    """Test GaussianFilter initialization with various parameters."""
    layer = GaussianFilter(kernel_size=(5, 5), strides=(2, 2))
    assert layer._kernel_size == (5, 5)
    assert layer._strides == [1, 2, 2, 1]


def test_gaussian_filter_invalid_kernel_size():
    """Test if GaussianFilter handles invalid kernel sizes."""
    with pytest.raises(ValueError):
        GaussianFilter(kernel_size=(3,))


def test_gaussian_filter_auto_sigma():
    """Test automatic sigma calculation."""
    kernel_size = (5, 5)
    layer = GaussianFilter(kernel_size=kernel_size, sigma=-1)
    expected_sigma = ((kernel_size[0] - 1) / 2, (kernel_size[1] - 1) / 2)
    assert layer._sigma == expected_sigma


def test_gaussian_filter_output_shape(sample_image):
    """Test if GaussianFilter preserves input shape with 'SAME' padding."""
    layer = GaussianFilter()
    output = layer(sample_image)
    assert output.shape == sample_image.shape


def test_gaussian_filter_strided_output(sample_image):
    """Test if GaussianFilter handles strides correctly."""
    strides = (2, 2)
    layer = GaussianFilter(strides=strides)
    output = layer(sample_image)
    expected_shape = (
        sample_image.shape[0],
        sample_image.shape[1] // strides[0],
        sample_image.shape[2] // strides[1],
        sample_image.shape[3]
    )
    assert output.shape == expected_shape


def test_gaussian_filter_serialization():
    """Test if GaussianFilter can be properly serialized."""
    layer = GaussianFilter(kernel_size=(5, 5), strides=(2, 2), sigma=1.5)
    config = layer.get_config()

    # Recreate layer from config
    recreated_layer = GaussianFilter.from_config(config)

    # Compare kernel_size and strides
    assert recreated_layer._kernel_size == layer._kernel_size
    assert recreated_layer._strides == layer._strides

    # Compare sigma values - they may be converted to tuples
    if isinstance(layer._sigma, tuple):
        assert recreated_layer._sigma == layer._sigma
    else:
        # If sigma was a single float, it gets converted to a tuple
        assert recreated_layer._sigma == (layer._sigma, layer._sigma)


def test_gaussian_filter_training_inference():
    """Test if GaussianFilter behaves the same in training and inference modes."""
    layer = GaussianFilter()
    test_input = tf.random.uniform((1, 32, 32, 3))

    # Get outputs in both modes
    training_output = layer(test_input, training=True)
    inference_output = layer(test_input, training=False)

    # Should be identical since this layer doesn't behave differently in training
    assert tf.reduce_all(training_output == inference_output)


def test_gaussian_filter_model_integration():
    """Test if GaussianFilter can be integrated into a Keras model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        GaussianFilter(kernel_size=(5, 5)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')  # Add padding='same'
    ])

    # Test forward pass
    test_input = tf.random.uniform((1, 32, 32, 3))
    output = model(test_input)

    # Shape should be preserved due to 'same' padding
    assert output.shape == (1, 32, 32, 16)

    # Also test with 'valid' padding to ensure proper behavior
    model_valid = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        GaussianFilter(kernel_size=(5, 5)),
        tf.keras.layers.Conv2D(16, 3, padding='valid', activation='relu')
    ])

    output_valid = model_valid(test_input)
    # With 'valid' padding, spatial dimensions should be reduced
    expected_size = 30  # 32 - 2 due to Conv2D with kernel_size=3 and valid padding
    assert output_valid.shape == (1, expected_size, expected_size, 16)


@pytest.mark.parametrize("kernel_size,sigma", [
    ((3, 3), 1.0),
    ((5, 5), 2.0),
    ((7, 7), 3.0)
])
def test_gaussian_filter_different_configs(kernel_size, sigma):
    """Test GaussianFilter with different configurations."""
    layer = GaussianFilter(kernel_size=kernel_size, sigma=sigma)
    test_input = tf.random.uniform((1, 32, 32, 3))
    output = layer(test_input)

    assert output.shape == test_input.shape
    assert not tf.reduce_any(tf.math.is_nan(output))


def test_gaussian_filter_gradient():
    """Test if gradients can flow through the GaussianFilter layer."""
    layer = GaussianFilter()
    test_input = tf.random.uniform((1, 32, 32, 3))

    with tf.GradientTape() as tape:
        tape.watch(test_input)
        output = layer(test_input)
        loss = tf.reduce_mean(output)

    gradient = tape.gradient(loss, test_input)
    assert gradient is not None
    assert not tf.reduce_any(tf.math.is_nan(gradient))

if __name__ == "__main__":
    pytest.main([__file__])