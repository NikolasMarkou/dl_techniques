"""
Test suite for AdaptiveTemperatureSoftmax implementation.

This module provides comprehensive tests for:
- AdaptiveTemperatureSoftmax layer
- Layer behavior under different configurations
- Entropy calculation and temperature adaptation
- Serialization and deserialization
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any, Optional, List

from dl_techniques.layers.adaptive_softmax import AdaptiveTemperatureSoftmax


# Test fixtures
@pytest.fixture
def sample_logits() -> tf.Tensor:
    """Generate sample logits tensor."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 10))  # Batch of 2, 10 classes


@pytest.fixture
def sample_large_logits() -> tf.Tensor:
    """Generate sample logits tensor with more classes."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 100))  # Batch of 2, 100 classes


@pytest.fixture
def default_params() -> Dict[str, Any]:
    """Default parameters for AdaptiveTemperatureSoftmax."""
    return {
        "min_temp": 0.1,
        "max_temp": 1.0,
        "entropy_threshold": 0.5,
        "polynomial_coeffs": [-1.791, 4.917, -2.3, 0.481, -0.037]
    }


# AdaptiveTemperatureSoftmax tests
def test_initialization(default_params: Dict[str, Any]) -> None:
    """Test initialization of AdaptiveTemperatureSoftmax."""
    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)
    assert adaptive_softmax.min_temp == default_params["min_temp"]
    assert adaptive_softmax.max_temp == default_params["max_temp"]
    assert float(adaptive_softmax.entropy_threshold) == default_params["entropy_threshold"]
    assert adaptive_softmax.polynomial_coeffs == default_params["polynomial_coeffs"]


def test_output_shape(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test if output shape matches input shape."""
    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)
    outputs = adaptive_softmax(sample_logits)
    assert outputs.shape == sample_logits.shape


def test_compute_output_shape(default_params: Dict[str, Any]) -> None:
    """Test the compute_output_shape method."""
    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)
    input_shape = tf.TensorShape([None, 10])
    output_shape = adaptive_softmax.compute_output_shape(input_shape)
    assert output_shape.as_list() == input_shape.as_list()


def test_sum_to_one(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test if output probabilities sum to 1."""
    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)
    outputs = adaptive_softmax(sample_logits)
    row_sums = tf.reduce_sum(outputs, axis=-1)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_valid_probability_values(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test if output values are valid probabilities (between 0 and 1)."""
    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)
    outputs = adaptive_softmax(sample_logits)
    assert tf.reduce_all(tf.greater_equal(outputs, 0.0))
    assert tf.reduce_all(tf.less_equal(outputs, 1.0))


def test_temperature_adaptation(sample_logits: tf.Tensor, sample_large_logits: tf.Tensor,
                                default_params: Dict[str, Any]) -> None:
    """Test if temperature adaptation works differently for different input sizes."""
    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)

    # Process regular and large logits
    outputs_regular = adaptive_softmax(sample_logits)
    outputs_large = adaptive_softmax(sample_large_logits)

    # Get entropy for both outputs
    probs_regular = tf.nn.softmax(sample_logits, axis=-1)
    probs_large = tf.nn.softmax(sample_large_logits, axis=-1)

    eps = keras.backend.epsilon()
    entropy_regular = -tf.reduce_sum(
        probs_regular * tf.math.log(probs_regular + eps), axis=-1, keepdims=True
    )
    entropy_large = -tf.reduce_sum(
        probs_large * tf.math.log(probs_large + eps), axis=-1, keepdims=True
    )

    # The larger logits should have higher entropy on average
    assert tf.reduce_mean(entropy_large) > tf.reduce_mean(entropy_regular)

    # The adaptive softmax should produce sharper distributions for larger inputs
    # compared to standard softmax when exceeding the entropy threshold
    max_prob_adaptive_large = tf.reduce_max(outputs_large, axis=-1)
    max_prob_standard_large = tf.reduce_max(probs_large, axis=-1)

    # For samples with high entropy (above threshold), adaptive softmax should
    # produce sharper distributions (higher max probability)
    high_entropy_samples = tf.squeeze(entropy_large > default_params["entropy_threshold"])
    if tf.reduce_any(high_entropy_samples):
        adaptive_max = tf.boolean_mask(max_prob_adaptive_large, high_entropy_samples)
        standard_max = tf.boolean_mask(max_prob_standard_large, high_entropy_samples)
        assert tf.reduce_mean(adaptive_max) > tf.reduce_mean(standard_max)


def test_no_adaptation_below_threshold(default_params: Dict[str, Any]) -> None:
    """Test if no adaptation occurs below entropy threshold."""
    # Create logits that will produce low entropy (one value much higher than others)
    sharp_logits = tf.constant([[10.0, 0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)

    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)

    # Apply adaptive softmax
    adaptive_result = adaptive_softmax(sharp_logits)

    # Apply standard softmax
    standard_result = tf.nn.softmax(sharp_logits, axis=-1)

    # Results should be very close for low entropy inputs
    assert np.allclose(adaptive_result.numpy(), standard_result.numpy(), rtol=1e-5, atol=1e-5)


def test_temperature_bounds(sample_logits: tf.Tensor) -> None:
    """Test if temperature stays within bounds."""
    # Test with extreme bounds
    min_temp = 0.01
    max_temp = 0.5

    adaptive_softmax = AdaptiveTemperatureSoftmax(
        min_temp=min_temp,
        max_temp=max_temp,
        entropy_threshold=0.0  # Force adaptation for all inputs
    )

    # Compute initial probabilities and entropy
    probs = tf.nn.softmax(sample_logits, axis=-1)
    eps = keras.backend.epsilon()
    entropy = -tf.reduce_sum(
        probs * tf.math.log(probs + eps), axis=-1, keepdims=True
    )

    # Compute temperature directly
    temperature = adaptive_softmax.compute_temperature(entropy)

    # Check if temperature values are within bounds
    assert tf.reduce_all(tf.greater_equal(temperature, min_temp))
    assert tf.reduce_all(tf.less_equal(temperature, max_temp))


def test_invalid_temperature_values() -> None:
    """Test if invalid temperature values raise errors."""
    # Test min_temp <= 0
    with pytest.raises(ValueError):
        AdaptiveTemperatureSoftmax(min_temp=0.0, max_temp=1.0)

    with pytest.raises(ValueError):
        AdaptiveTemperatureSoftmax(min_temp=-0.1, max_temp=1.0)

    # Test min_temp > max_temp
    with pytest.raises(ValueError):
        AdaptiveTemperatureSoftmax(min_temp=0.8, max_temp=0.5)


def test_serialization(default_params: Dict[str, Any]) -> None:
    """Test serialization of AdaptiveTemperatureSoftmax."""
    original_layer = AdaptiveTemperatureSoftmax(**default_params)

    # Get config and recreate from config
    config = original_layer.get_config()
    restored_layer = AdaptiveTemperatureSoftmax.from_config(config)

    # Check if the key properties match
    assert restored_layer.min_temp == original_layer.min_temp
    assert restored_layer.max_temp == original_layer.max_temp
    assert float(restored_layer.entropy_threshold) == float(original_layer.entropy_threshold)
    assert restored_layer.polynomial_coeffs == original_layer.polynomial_coeffs
    assert restored_layer.eps == original_layer.eps

    # Check that restored layer's config has the same keys as original
    restored_config = restored_layer.get_config()
    assert set(restored_config.keys()) == set(config.keys())


def test_gradient_flow(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test gradient flow through the AdaptiveTemperatureSoftmax layer."""
    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)

    with tf.GradientTape() as tape:
        tape.watch(sample_logits)
        output = adaptive_softmax(sample_logits)
        loss = tf.reduce_mean(output)

    gradients = tape.gradient(loss, sample_logits)

    # Check if gradients exist
    assert gradients is not None
    # Check if gradients have same shape as input
    assert gradients.shape == sample_logits.shape
    # Check if gradients are finite
    assert tf.reduce_all(tf.math.is_finite(gradients))


def test_custom_polynomial_coeffs(sample_logits: tf.Tensor) -> None:
    """Test using custom polynomial coefficients."""
    custom_coeffs = [0.5, -1.0, 2.0]
    adaptive_softmax = AdaptiveTemperatureSoftmax(polynomial_coeffs=custom_coeffs)

    # Verify the coefficients were set properly
    assert adaptive_softmax.polynomial_coeffs == custom_coeffs

    # Apply the layer (should complete without errors)
    outputs = adaptive_softmax(sample_logits)
    assert outputs.shape == sample_logits.shape


def test_compute_entropy(default_params: Dict[str, Any]) -> None:
    """Test entropy computation directly."""
    # Create known probabilities with known entropy
    # For uniform distribution over 4 classes, entropy = log(4) = 1.386...
    probs = tf.constant([[0.25, 0.25, 0.25, 0.25]], dtype=tf.float32)

    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)
    entropy = adaptive_softmax.compute_entropy(probs)

    expected_entropy = np.log(4)
    assert np.isclose(entropy.numpy()[0][0], expected_entropy, rtol=1e-5, atol=1e-5)


def test_entropy_value_range(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test if entropy values are within expected range."""
    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)

    # Compute initial probabilities
    probs = tf.nn.softmax(sample_logits, axis=-1)

    # Compute entropy
    entropy = adaptive_softmax.compute_entropy(probs)

    # For a distribution over n classes, entropy should be between 0 and log(n)
    n = sample_logits.shape[-1]
    max_entropy = np.log(n)

    assert tf.reduce_all(tf.greater_equal(entropy, 0.0))
    assert tf.reduce_all(tf.less_equal(entropy, max_entropy + 1e-5))


def test_epsilon_handling(default_params: Dict[str, Any]) -> None:
    """Test if epsilon handling prevents numerical issues."""
    # Create zero probability (which would cause log(0) issue without epsilon)
    extreme_probs = tf.constant([[0.0, 0.0, 1.0, 0.0]], dtype=tf.float32)

    # Custom epsilon for testing
    test_eps = 1e-7
    adaptive_softmax = AdaptiveTemperatureSoftmax(eps=test_eps, **default_params)

    # This should not raise any numerical errors
    entropy = adaptive_softmax.compute_entropy(extreme_probs)

    # Entropy should be finite
    assert tf.reduce_all(tf.math.is_finite(entropy))


def test_model_integration(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test integrating the AdaptiveTemperatureSoftmax into a Keras model."""
    input_shape = sample_logits.shape[1:]
    inputs = keras.Input(shape=input_shape)
    adaptive_softmax_layer = AdaptiveTemperatureSoftmax(**default_params)
    outputs = adaptive_softmax_layer(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(sample_logits)
    assert result.shape == sample_logits.shape


def test_model_save_load(sample_logits: tf.Tensor, default_params: Dict[str, Any], tmp_path) -> None:
    """Test saving and loading a model with AdaptiveTemperatureSoftmax."""
    # Create a simple model with the layer
    input_shape = sample_logits.shape[1:]
    inputs = keras.Input(shape=input_shape)
    adaptive_softmax_layer = AdaptiveTemperatureSoftmax(**default_params)
    outputs = adaptive_softmax_layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Generate output before saving
    original_output = model(sample_logits).numpy()

    # Save model
    save_path = str(tmp_path / "model.weights.h5")
    model.save_weights(save_path)

    # Load model
    model.load_weights(save_path)

    # Generate output after loading
    loaded_output = model(sample_logits).numpy()

    # Outputs should be identical
    assert np.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch_size,seq_length,num_classes", [
    (1, 5, 10),  # Single item batch
    (4, 5, 10),  # Standard batch
    (2, 100, 50),  # Long sequence
    (3, 10, 1000)  # Large number of classes
])
def test_different_input_dimensions(batch_size: int, seq_length: int, num_classes: int,
                                    default_params: Dict[str, Any]) -> None:
    """Test layer with different input dimensions."""
    tf.random.set_seed(42)
    logits = tf.random.normal((batch_size, seq_length, num_classes))

    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)
    outputs = adaptive_softmax(logits)

    # Check shape
    assert outputs.shape == logits.shape

    # Check sum to 1 along class dimension
    row_sums = tf.reduce_sum(outputs, axis=-1)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_build(default_params: Dict[str, Any]) -> None:
    """Test the build method."""
    adaptive_softmax = AdaptiveTemperatureSoftmax(**default_params)
    input_shape = tf.TensorShape([None, 10])

    # Call build method
    adaptive_softmax.build(input_shape)

    # Verify poly_coeffs was initialized
    assert adaptive_softmax.polynomial_coeffs is not None


if __name__ == "__main__":
    pytest.main([__file__])