"""
Test suite for ThreshMax implementation.

This module provides comprehensive tests for:
- ThreshMax layer
- thresh_max functional interface
- Layer behavior under different configurations
- Sparse probability distribution generation
- Degenerate case handling (maximum entropy fallback)
- Serialization and deserialization
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any, Tuple

from dl_techniques.layers.activations.thresh_max import ThreshMax, thresh_max, create_thresh_max


# Test fixtures
@pytest.fixture
def sample_logits() -> tf.Tensor:
    """Generate sample logits tensor."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 10))  # Batch of 2, 10 classes


@pytest.fixture
def clear_winner_logits() -> tf.Tensor:
    """Generate logits with clear winner (should produce sparse output)."""
    return tf.constant([
        [1.0, 3.0, 0.5, -1.0],  # Class 1 is clear winner
        [-0.5, 2.5, 0.0, -2.0]  # Class 1 is clear winner
    ], dtype=tf.float32)


@pytest.fixture
def uniform_logits() -> tf.Tensor:
    """Generate uniform logits (should trigger fallback to standard softmax)."""
    return tf.constant([
        [2.0, 2.0, 2.0, 2.0],  # Perfectly uniform
        [1.5, 1.5, 1.5, 1.5]  # Perfectly uniform
    ], dtype=tf.float32)


@pytest.fixture
def low_confidence_logits() -> tf.Tensor:
    """Generate low confidence logits (should produce sparse filtering)."""
    return tf.constant([
        [0.1, 0.2, 0.1, 0.0],  # Low confidence, close to uniform
        [0.05, 0.15, 0.08, 0.02]  # Very low confidence
    ], dtype=tf.float32)


@pytest.fixture
def default_params() -> Dict[str, Any]:
    """Default parameters for ThreshMax."""
    return {
        "axis": -1,
        "epsilon": 1e-12
    }


# Basic functionality tests
def test_initialization(default_params: Dict[str, Any]) -> None:
    """Test initialization of ThreshMax."""
    thresh_max_layer = ThreshMax(**default_params)
    assert thresh_max_layer.axis == default_params["axis"]
    assert thresh_max_layer.epsilon == default_params["epsilon"]


def test_initialization_custom_params() -> None:
    """Test initialization with custom parameters."""
    custom_axis = 1
    custom_epsilon = 1e-10

    thresh_max_layer = ThreshMax(axis=custom_axis, epsilon=custom_epsilon)
    assert thresh_max_layer.axis == custom_axis
    assert thresh_max_layer.epsilon == custom_epsilon


def test_invalid_epsilon() -> None:
    """Test that invalid epsilon values raise errors."""
    # Test epsilon <= 0
    with pytest.raises(ValueError, match="epsilon must be positive"):
        ThreshMax(epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be positive"):
        ThreshMax(epsilon=-1e-6)


def test_output_shape(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test if output shape matches input shape."""
    thresh_max_layer = ThreshMax(**default_params)
    outputs = thresh_max_layer(sample_logits)
    assert outputs.shape == sample_logits.shape


def test_compute_output_shape(default_params: Dict[str, Any]) -> None:
    """Test the compute_output_shape method."""
    thresh_max_layer = ThreshMax(**default_params)
    input_shape = tf.TensorShape([None, 10])
    output_shape = thresh_max_layer.compute_output_shape(input_shape)
    assert output_shape.as_list() == input_shape.as_list()


def test_sum_to_one(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test if output probabilities sum to 1."""
    thresh_max_layer = ThreshMax(**default_params)
    outputs = thresh_max_layer(sample_logits)
    row_sums = tf.reduce_sum(outputs, axis=-1)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_valid_probability_values(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test if output values are valid probabilities (between 0 and 1)."""
    thresh_max_layer = ThreshMax(**default_params)
    outputs = thresh_max_layer(sample_logits)
    assert tf.reduce_all(tf.greater_equal(outputs, 0.0))
    assert tf.reduce_all(tf.less_equal(outputs, 1.0))


# Core ThreshMax behavior tests
def test_clear_winner_sparsity(clear_winner_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test that clear winners produce sparser distributions than standard softmax."""
    thresh_max_layer = ThreshMax(**default_params)

    # Get ThreshMax output
    thresh_max_output = thresh_max_layer(clear_winner_logits)

    # Get standard softmax output for comparison
    standard_softmax = tf.nn.softmax(clear_winner_logits, axis=-1)

    # ThreshMax should be sparser (higher maximum probability per row)
    thresh_max_peaks = tf.reduce_max(thresh_max_output, axis=-1)
    standard_peaks = tf.reduce_max(standard_softmax, axis=-1)

    # ThreshMax should have higher peak probabilities (more sparse)
    assert tf.reduce_all(tf.greater_equal(thresh_max_peaks, standard_peaks))

    # At least one should be strictly greater (unless already at 1.0)
    mask = tf.less(standard_peaks, 0.99)  # Not already peaked
    if tf.reduce_any(mask):
        masked_thresh = tf.boolean_mask(thresh_max_peaks, mask)
        masked_standard = tf.boolean_mask(standard_peaks, mask)
        assert tf.reduce_any(tf.greater(masked_thresh, masked_standard))


def test_uniform_fallback(uniform_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test that uniform logits fall back to standard softmax."""
    thresh_max_layer = ThreshMax(**default_params)

    # Get ThreshMax output
    thresh_max_output = thresh_max_layer(uniform_logits)

    # Get standard softmax output
    standard_softmax = tf.nn.softmax(uniform_logits, axis=-1)

    # For uniform inputs, outputs should be nearly identical
    assert np.allclose(thresh_max_output.numpy(), standard_softmax.numpy(), rtol=1e-5, atol=1e-5)


def test_low_confidence_filtering(low_confidence_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test that low confidence inputs are filtered appropriately."""
    thresh_max_layer = ThreshMax(**default_params)

    # Get ThreshMax output
    thresh_max_output = thresh_max_layer(low_confidence_logits)

    # Get standard softmax output for comparison
    standard_softmax = tf.nn.softmax(low_confidence_logits, axis=-1)

    # Outputs should be different (unless it's a degenerate case)
    # Check if any outputs are significantly different
    diff = tf.abs(thresh_max_output - standard_softmax)
    max_diff = tf.reduce_max(diff, axis=-1)

    # At least some differences should be non-trivial or it should be uniform fallback
    assert tf.reduce_any(tf.greater(max_diff, 1e-4)) or np.allclose(
        thresh_max_output.numpy(), standard_softmax.numpy(), rtol=1e-5, atol=1e-5
    )


def test_degenerate_case_detection() -> None:
    """Test explicit degenerate case detection."""
    thresh_max_layer = ThreshMax(epsilon=1e-6)

    # Create logits that will result in exactly uniform probabilities
    # This should trigger the degenerate case fallback
    uniform_logits = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)

    thresh_max_output = thresh_max_layer(uniform_logits)
    standard_softmax = tf.nn.softmax(uniform_logits, axis=-1)

    # Should be identical due to fallback
    assert np.allclose(thresh_max_output.numpy(), standard_softmax.numpy(), rtol=1e-5, atol=1e-5)

    # Should be exactly uniform
    expected_uniform = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
    assert np.allclose(thresh_max_output.numpy(), expected_uniform, rtol=1e-5, atol=1e-5)


# Functional interface tests
def test_functional_interface(clear_winner_logits: tf.Tensor) -> None:
    """Test the functional interface thresh_max."""
    # Test with default parameters
    output = thresh_max(clear_winner_logits)

    # Should have same shape
    assert output.shape == clear_winner_logits.shape

    # Should sum to 1
    row_sums = tf.reduce_sum(output, axis=-1)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_functional_interface_custom_params(clear_winner_logits: tf.Tensor) -> None:
    """Test functional interface with custom parameters."""
    output = thresh_max(clear_winner_logits, axis=-1, epsilon=1e-10)

    assert output.shape == clear_winner_logits.shape
    row_sums = tf.reduce_sum(output, axis=-1)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_functional_vs_layer_consistency(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test that functional interface produces same results as layer."""
    # Layer output
    thresh_max_layer = ThreshMax(**default_params)
    layer_output = thresh_max_layer(sample_logits)

    # Functional output
    func_output = thresh_max(sample_logits, **default_params)

    # Should be identical
    assert np.allclose(layer_output.numpy(), func_output.numpy(), rtol=1e-10, atol=1e-10)


# Factory function tests
def test_factory_function() -> None:
    """Test the create_thresh_max factory function."""
    layer = create_thresh_max(axis=1, epsilon=1e-10, name="test_thresh_max")

    assert layer.axis == 1
    assert layer.epsilon == 1e-10
    assert layer.name == "test_thresh_max"


# Serialization tests
def test_serialization(default_params: Dict[str, Any]) -> None:
    """Test serialization of ThreshMax."""
    original_layer = ThreshMax(**default_params)

    # Get config and recreate from config
    config = original_layer.get_config()
    restored_layer = ThreshMax.from_config(config)

    # Check if the key properties match
    assert restored_layer.axis == original_layer.axis
    assert restored_layer.epsilon == original_layer.epsilon

    # Check that restored layer's config has the same keys as original
    restored_config = restored_layer.get_config()
    assert set(restored_config.keys()) == set(config.keys())


def test_gradient_flow(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test gradient flow through the ThreshMax layer."""
    thresh_max_layer = ThreshMax(**default_params)

    with tf.GradientTape() as tape:
        tape.watch(sample_logits)
        output = thresh_max_layer(sample_logits)
        loss = tf.reduce_mean(output)

    gradients = tape.gradient(loss, sample_logits)

    # Check if gradients exist
    assert gradients is not None
    # Check if gradients have same shape as input
    assert gradients.shape == sample_logits.shape
    # Check if gradients are finite
    assert tf.reduce_all(tf.math.is_finite(gradients))


def test_model_integration(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test integrating the ThreshMax into a Keras model."""
    input_shape = sample_logits.shape[1:]
    inputs = keras.Input(shape=input_shape)
    thresh_max_layer = ThreshMax(**default_params)
    outputs = thresh_max_layer(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(sample_logits)
    assert result.shape == sample_logits.shape


def test_model_save_load(sample_logits: tf.Tensor, default_params: Dict[str, Any], tmp_path) -> None:
    """Test saving and loading a model with ThreshMax."""
    # Create a simple model with the layer
    input_shape = sample_logits.shape[1:]
    inputs = keras.Input(shape=input_shape)
    thresh_max_layer = ThreshMax(**default_params)
    outputs = thresh_max_layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Generate output before saving
    original_output = model(sample_logits).numpy()

    # Save model
    save_path = str(tmp_path / "model.keras")
    model.save(save_path, save_format="keras_v3")

    # Load model with custom objects
    loaded_model = keras.models.load_model(
        save_path,
        custom_objects={"ThreshMax": ThreshMax}
    )

    # Generate output after loading
    loaded_output = loaded_model(sample_logits).numpy()

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

    thresh_max_layer = ThreshMax(**default_params)
    outputs = thresh_max_layer(logits)

    # Check shape
    assert outputs.shape == logits.shape

    # Check sum to 1 along class dimension
    row_sums = tf.reduce_sum(outputs, axis=-1)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [-1, -2, 0, 1])
def test_different_axes(axis: int, default_params: Dict[str, Any]) -> None:
    """Test ThreshMax with different axis values."""
    # Create 3D tensor for testing different axes
    tf.random.set_seed(42)
    logits = tf.random.normal((2, 3, 4))  # batch, seq, classes

    thresh_max_layer = ThreshMax(axis=axis, epsilon=default_params["epsilon"])
    outputs = thresh_max_layer(logits)

    # Check shape
    assert outputs.shape == logits.shape

    # Check sum to 1 along specified axis
    row_sums = tf.reduce_sum(outputs, axis=axis)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_build(default_params: Dict[str, Any]) -> None:
    """Test the build method."""
    thresh_max_layer = ThreshMax(**default_params)
    input_shape = tf.TensorShape([None, 10])

    # Call build method
    thresh_max_layer.build(input_shape)

    # Verify layer is built
    assert thresh_max_layer.built


def test_extreme_values() -> None:
    """Test with extreme input values."""
    thresh_max_layer = ThreshMax()

    # Very large positive values
    large_logits = tf.constant([[100.0, -100.0, 50.0, -50.0]], dtype=tf.float32)
    large_output = thresh_max_layer(large_logits)

    # Should still be valid probabilities
    assert tf.reduce_all(tf.greater_equal(large_output, 0.0))
    assert tf.reduce_all(tf.less_equal(large_output, 1.0))
    assert np.allclose(tf.reduce_sum(large_output, axis=-1).numpy(), 1.0, rtol=1e-5, atol=1e-5)

    # Very small values (close to zero)
    small_logits = tf.constant([[1e-6, 2e-6, 1.5e-6, 0.5e-6]], dtype=tf.float32)
    small_output = thresh_max_layer(small_logits)

    # Should still be valid probabilities
    assert tf.reduce_all(tf.greater_equal(small_output, 0.0))
    assert tf.reduce_all(tf.less_equal(small_output, 1.0))
    assert np.allclose(tf.reduce_sum(small_output, axis=-1).numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_numerical_stability() -> None:
    """Test numerical stability with edge cases."""
    thresh_max_layer = ThreshMax(epsilon=1e-15)  # Very small epsilon

    # Test with values that might cause numerical issues
    edge_logits = tf.constant([
        [0.0, 0.0, 0.0],  # Uniform case
        [1e-15, 1e-15, 1e-15],  # Very small uniform
        [-1e10, 1e10, 0.0]  # Extreme range
    ], dtype=tf.float32)

    outputs = thresh_max_layer(edge_logits)

    # Should not produce NaN or Inf
    assert tf.reduce_all(tf.math.is_finite(outputs))

    # Should sum to 1
    row_sums = tf.reduce_sum(outputs, axis=-1)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-4, atol=1e-4)


def test_consistency_across_calls(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test that the layer produces consistent outputs across multiple calls."""
    thresh_max_layer = ThreshMax(**default_params)

    # Call multiple times with same input
    output1 = thresh_max_layer(sample_logits)
    output2 = thresh_max_layer(sample_logits)
    output3 = thresh_max_layer(sample_logits)

    # All outputs should be identical (deterministic)
    assert np.allclose(output1.numpy(), output2.numpy(), rtol=1e-10, atol=1e-10)
    assert np.allclose(output2.numpy(), output3.numpy(), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])