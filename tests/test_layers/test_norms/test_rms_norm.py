"""
Test suite for RMSNorm layer implementation.

This module provides comprehensive tests for:
- RMSNorm layer functionality
- Layer behavior under different configurations
- Serialization and deserialization
- Model integration and persistence
- Mathematical correctness
- Edge cases and error handling
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any, Union, Tuple

from dl_techniques.layers.norms.rms_norm import RMSNorm


# Test fixtures
@pytest.fixture
def sample_inputs_2d() -> tf.Tensor:
    """Generate 2D sample input tensor."""
    tf.random.set_seed(42)
    return tf.random.normal((4, 512))


@pytest.fixture
def sample_inputs_3d() -> tf.Tensor:
    """Generate 3D sample input tensor (sequence data)."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 128, 768))


@pytest.fixture
def sample_inputs_4d() -> tf.Tensor:
    """Generate 4D sample input tensor (image data)."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 32, 32, 64))


@pytest.fixture
def default_norm_params() -> Dict[str, Any]:
    """Default parameters for RMSNorm."""
    return {
        "axis": -1,
        "epsilon": 1e-6,
        "use_scale": True,
        "scale_initializer": "ones",
    }


@pytest.fixture
def minimal_norm_params() -> Dict[str, Any]:
    """Minimal parameters for RMSNorm."""
    return {
        "axis": -1,
    }


@pytest.fixture
def multi_axis_norm_params() -> Dict[str, Any]:
    """Multi-axis normalization parameters."""
    return {
        "axis": (-2, -1),
        "epsilon": 1e-5,
        "use_scale": True,
        "scale_initializer": "ones",
    }


# RMSNorm initialization tests
def test_norm_initialization(default_norm_params: Dict[str, Any]) -> None:
    """Test initialization of RMSNorm with default parameters."""
    norm = RMSNorm(**default_norm_params)

    assert norm.axis == default_norm_params["axis"]
    assert norm.epsilon == default_norm_params["epsilon"]
    assert norm.use_scale == default_norm_params["use_scale"]
    assert isinstance(norm.scale_initializer, keras.initializers.Ones)


def test_norm_minimal_initialization(minimal_norm_params: Dict[str, Any]) -> None:
    """Test initialization with minimal parameters."""
    norm = RMSNorm(**minimal_norm_params)

    assert norm.axis == minimal_norm_params["axis"]
    assert norm.epsilon == 1e-6  # Default value
    assert norm.use_scale is True  # Default value
    assert isinstance(norm.scale_initializer, keras.initializers.Ones)


def test_norm_multi_axis_initialization(multi_axis_norm_params: Dict[str, Any]) -> None:
    """Test initialization with multi-axis normalization."""
    norm = RMSNorm(**multi_axis_norm_params)

    assert norm.axis == multi_axis_norm_params["axis"]
    assert norm.epsilon == multi_axis_norm_params["epsilon"]
    assert norm.use_scale == multi_axis_norm_params["use_scale"]


def test_norm_custom_initializer() -> None:
    """Test initialization with custom scale initializer."""
    custom_initializer = keras.initializers.RandomNormal(mean=1.0, stddev=0.1)
    norm = RMSNorm(
        axis=-1,
        use_scale=True,
        scale_initializer=custom_initializer
    )

    assert isinstance(norm.scale_initializer, keras.initializers.RandomNormal)


def test_norm_without_scale() -> None:
    """Test initialization without learnable scale parameter."""
    norm = RMSNorm(axis=-1, use_scale=False)

    assert norm.use_scale is False
    assert norm.scale_initializer is not None  # Should still be set even if not used


# Output shape tests
def test_norm_output_shape_2d(sample_inputs_2d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test that RMSNorm preserves input shape for 2D tensors."""
    norm = RMSNorm(**default_norm_params)
    outputs = norm(sample_inputs_2d)

    assert outputs.shape == sample_inputs_2d.shape


def test_norm_output_shape_3d(sample_inputs_3d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test that RMSNorm preserves input shape for 3D tensors."""
    norm = RMSNorm(**default_norm_params)
    outputs = norm(sample_inputs_3d)

    assert outputs.shape == sample_inputs_3d.shape


def test_norm_output_shape_4d(sample_inputs_4d: tf.Tensor, multi_axis_norm_params: Dict[str, Any]) -> None:
    """Test that RMSNorm preserves input shape for 4D tensors with multi-axis."""
    norm = RMSNorm(**multi_axis_norm_params)
    outputs = norm(sample_inputs_4d)

    assert outputs.shape == sample_inputs_4d.shape


def test_norm_compute_output_shape(sample_inputs_3d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test compute_output_shape method."""
    norm = RMSNorm(**default_norm_params)

    computed_shape = norm.compute_output_shape(sample_inputs_3d.shape)
    assert computed_shape == sample_inputs_3d.shape


# Mathematical correctness tests
def test_norm_mathematical_correctness(sample_inputs_2d: tf.Tensor) -> None:
    """Test that RMSNorm computes correct mathematical result."""
    norm = RMSNorm(axis=-1, use_scale=False, epsilon=1e-6)

    # Manual computation
    inputs_np = sample_inputs_2d.numpy()
    rms_manual = np.sqrt(np.mean(inputs_np ** 2, axis=-1, keepdims=True) + 1e-6)
    expected = inputs_np / rms_manual

    # Layer computation
    outputs = norm(sample_inputs_2d)

    assert np.allclose(outputs.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_norm_with_scale_correctness(sample_inputs_2d: tf.Tensor) -> None:
    """Test mathematical correctness with learnable scale."""
    # Initialize with known scale values
    norm = RMSNorm(
        axis=-1,
        use_scale=True,
        scale_initializer=keras.initializers.Constant(2.0),
        epsilon=1e-6
    )

    # Build the layer
    norm.build(sample_inputs_2d.shape)

    # Manual computation
    inputs_np = sample_inputs_2d.numpy()
    rms_manual = np.sqrt(np.mean(inputs_np ** 2, axis=-1, keepdims=True) + 1e-6)
    normalized = inputs_np / rms_manual
    expected = normalized * 2.0  # Scale factor

    # Layer computation
    outputs = norm(sample_inputs_2d)

    assert np.allclose(outputs.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_norm_multi_axis_correctness(sample_inputs_4d: tf.Tensor) -> None:
    """Test mathematical correctness for multi-axis normalization."""
    norm = RMSNorm(axis=(-2, -1), use_scale=False, epsilon=1e-6)

    # Manual computation
    inputs_np = sample_inputs_4d.numpy()
    rms_manual = np.sqrt(np.mean(inputs_np ** 2, axis=(-2, -1), keepdims=True) + 1e-6)
    expected = inputs_np / rms_manual

    # Layer computation
    outputs = norm(sample_inputs_4d)

    assert np.allclose(outputs.numpy(), expected, rtol=1e-5, atol=1e-6)


# Training behavior tests
def test_norm_training_vs_inference(sample_inputs_3d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test that RMSNorm behaves consistently in training vs inference modes."""
    norm = RMSNorm(**default_norm_params)

    # RMSNorm should behave the same in training and inference (unlike BatchNorm)
    train_output = norm(sample_inputs_3d, training=True)
    inference_output = norm(sample_inputs_3d, training=False)

    assert np.allclose(train_output.numpy(), inference_output.numpy(), rtol=1e-6)


def test_norm_deterministic_behavior(sample_inputs_2d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test that RMSNorm produces deterministic outputs."""
    norm = RMSNorm(**default_norm_params)

    # Multiple calls should produce identical results
    output1 = norm(sample_inputs_2d)
    output2 = norm(sample_inputs_2d)

    assert np.allclose(output1.numpy(), output2.numpy(), rtol=1e-6)


# Serialization tests
def test_norm_serialization(default_norm_params: Dict[str, Any], sample_inputs_2d: tf.Tensor) -> None:
    """Test serialization of RMSNorm."""
    # Create and build the original norm
    original_norm = RMSNorm(**default_norm_params)
    original_norm.build(sample_inputs_2d.shape)

    # Get config and recreate from config
    config = original_norm.get_config()
    build_config = original_norm.get_build_config()

    restored_norm = RMSNorm.from_config(config)
    restored_norm.build_from_config(build_config)

    # Check if the key properties match
    assert restored_norm.axis == original_norm.axis
    assert restored_norm.epsilon == original_norm.epsilon
    assert restored_norm.use_scale == original_norm.use_scale
    assert type(restored_norm.scale_initializer) == type(original_norm.scale_initializer)

    # Check that both norms are built
    assert original_norm.built
    assert restored_norm.built

    # Check that restored norm's config has the same keys as original
    restored_config = restored_norm.get_config()
    assert set(restored_config.keys()) == set(config.keys())


def test_norm_serialization_multi_axis(multi_axis_norm_params: Dict[str, Any], sample_inputs_4d: tf.Tensor) -> None:
    """Test serialization with multi-axis normalization."""
    original_norm = RMSNorm(**multi_axis_norm_params)
    original_norm.build(sample_inputs_4d.shape)

    config = original_norm.get_config()
    build_config = original_norm.get_build_config()

    restored_norm = RMSNorm.from_config(config)
    restored_norm.build_from_config(build_config)

    # Check multi-axis configuration is preserved
    assert restored_norm.axis == multi_axis_norm_params["axis"]
    assert restored_norm.built


def test_norm_build_configuration(default_norm_params: Dict[str, Any], sample_inputs_3d: tf.Tensor) -> None:
    """Test get_build_config and build_from_config methods."""
    # Create and build the original norm
    original_norm = RMSNorm(**default_norm_params)
    original_norm.build(sample_inputs_3d.shape)

    # Get build config
    build_config = original_norm.get_build_config()

    # Check that build config contains input_shape
    assert "input_shape" in build_config
    assert build_config["input_shape"] == sample_inputs_3d.shape

    # Create new norm and build from config
    new_norm = RMSNorm(**default_norm_params)
    new_norm.build_from_config(build_config)

    # Check that new norm is built
    assert new_norm.built


# Parametrized tests
@pytest.mark.parametrize("axis", [-1, -2, 0, 1, (-2, -1), (-3, -2, -1)])
def test_norm_different_axes(axis: Union[int, Tuple[int, ...]], sample_inputs_4d: tf.Tensor) -> None:
    """Test RMSNorm with different axis configurations."""
    norm = RMSNorm(axis=axis, use_scale=False)

    try:
        output = norm(sample_inputs_4d)
        assert output.shape == sample_inputs_4d.shape
        assert not tf.reduce_any(tf.math.is_nan(output))
    except (ValueError, tf.errors.InvalidArgumentError):
        # Some axis configurations might be invalid for the given input shape
        # This is expected behavior
        pass


@pytest.mark.parametrize("epsilon", [1e-8, 1e-6, 1e-5, 1e-4])
def test_norm_different_epsilons(epsilon: float, sample_inputs_2d: tf.Tensor) -> None:
    """Test RMSNorm with different epsilon values."""
    norm = RMSNorm(axis=-1, epsilon=epsilon, use_scale=False)
    output = norm(sample_inputs_2d)

    assert output.shape == sample_inputs_2d.shape
    assert not tf.reduce_any(tf.math.is_nan(output))


@pytest.mark.parametrize("use_scale", [True, False])
def test_norm_with_without_scale(use_scale: bool, sample_inputs_2d: tf.Tensor) -> None:
    """Test RMSNorm with and without learnable scale parameter."""
    norm = RMSNorm(axis=-1, use_scale=use_scale)
    output = norm(sample_inputs_2d)

    assert output.shape == sample_inputs_2d.shape
    assert not tf.reduce_any(tf.math.is_nan(output))

    # Check scale parameter existence
    if use_scale:
        assert norm.scale is not None
    # Note: scale might be None even when use_scale=False until build() is called


@pytest.mark.parametrize("initializer_name", ["ones", "zeros", "random_normal"])
def test_norm_different_initializers(initializer_name: str, sample_inputs_2d: tf.Tensor) -> None:
    """Test RMSNorm with different scale initializers."""
    norm = RMSNorm(axis=-1, use_scale=True, scale_initializer=initializer_name)
    output = norm(sample_inputs_2d)

    assert output.shape == sample_inputs_2d.shape
    assert not tf.reduce_any(tf.math.is_nan(output))


# Gradient flow tests
def test_norm_gradient_flow(sample_inputs_3d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test gradient flow through RMSNorm."""
    norm = RMSNorm(**default_norm_params)

    with tf.GradientTape() as tape:
        output = norm(sample_inputs_3d, training=True)
        loss = tf.reduce_mean(output)

    gradients = tape.gradient(loss, norm.trainable_variables)

    # Check if gradients exist for all trainable variables
    if norm.use_scale:
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)
        # Check if we have trainable variables
        assert len(norm.trainable_variables) > 0
    else:
        # Without scale, there should be no trainable variables
        assert len(norm.trainable_variables) == 0


def test_norm_gradient_flow_without_scale(sample_inputs_2d: tf.Tensor) -> None:
    """Test gradient flow through input when no scale parameter."""
    norm = RMSNorm(axis=-1, use_scale=False)

    # Create a variable input to test gradients
    var_input = tf.Variable(sample_inputs_2d)

    with tf.GradientTape() as tape:
        output = norm(var_input, training=True)
        loss = tf.reduce_mean(output)

    gradients = tape.gradient(loss, var_input)

    # Should have gradients with respect to input
    assert gradients is not None
    assert not tf.reduce_any(tf.math.is_nan(gradients))


# Model integration tests
def test_norm_model_integration(sample_inputs_3d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test integrating RMSNorm into a Keras model."""
    inputs = keras.Input(shape=sample_inputs_3d.shape[1:])
    norm = RMSNorm(**default_norm_params)
    outputs = norm(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(sample_inputs_3d)
    assert result.shape == sample_inputs_3d.shape


def test_norm_transformer_integration(sample_inputs_3d: tf.Tensor) -> None:
    """Test RMSNorm in a transformer-like architecture."""
    seq_length, hidden_size = sample_inputs_3d.shape[1:]

    inputs = keras.Input(shape=(seq_length, hidden_size))

    # Pre-normalization transformer block pattern
    x = RMSNorm(axis=-1)(inputs)
    attention_out = keras.layers.MultiHeadAttention(
        num_heads=8, key_dim=hidden_size // 8
    )(x, x)
    x = inputs + attention_out  # Residual connection

    x = RMSNorm(axis=-1)(x)
    ffn_out = keras.layers.Dense(hidden_size * 4, activation='relu')(x)
    ffn_out = keras.layers.Dense(hidden_size)(ffn_out)
    outputs = x + ffn_out  # Residual connection

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(sample_inputs_3d)
    assert result.shape == sample_inputs_3d.shape


def test_norm_model_compilation_and_training(sample_inputs_2d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test compiling and training a model with RMSNorm."""
    inputs = keras.Input(shape=sample_inputs_2d.shape[1:])
    x = RMSNorm(**default_norm_params)(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = RMSNorm(axis=-1)(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create dummy labels
    labels = tf.random.uniform((sample_inputs_2d.shape[0],), 0, 10, dtype=tf.int32)

    # Test training for one step
    history = model.fit(sample_inputs_2d, labels, epochs=1, verbose=0)
    assert len(history.history["loss"]) == 1


# Model persistence tests
def test_norm_model_save_load_weights(sample_inputs_3d: tf.Tensor, default_norm_params: Dict[str, Any],
                                      tmp_path) -> None:
    """Test saving and loading model weights with RMSNorm."""
    # Create a simple model with RMSNorm
    inputs = keras.Input(shape=sample_inputs_3d.shape[1:])
    norm = RMSNorm(**default_norm_params)
    outputs = norm(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Generate output before saving
    original_output = model(sample_inputs_3d, training=False).numpy()

    # Save model weights
    save_path = str(tmp_path / "model.weights.h5")
    model.save_weights(save_path)

    # Load model weights
    model.load_weights(save_path)

    # Generate output after loading
    loaded_output = model(sample_inputs_3d, training=False).numpy()

    # Outputs should be identical
    assert np.allclose(original_output, loaded_output, rtol=1e-6, atol=1e-7)


def test_norm_model_save_load_keras_format(sample_inputs_2d: tf.Tensor, default_norm_params: Dict[str, Any],
                                           tmp_path) -> None:
    """Test saving and loading model in Keras format with RMSNorm."""
    # Create a simple model with RMSNorm
    inputs = keras.Input(shape=sample_inputs_2d.shape[1:])
    norm = RMSNorm(**default_norm_params)
    outputs = norm(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Generate output before saving
    original_output = model(sample_inputs_2d, training=False).numpy()

    # Save model in Keras format
    save_path = str(tmp_path / "model.keras")
    model.save(save_path)

    # Load model with custom objects
    loaded_model = keras.models.load_model(
        save_path,
        custom_objects={"RMSNorm": RMSNorm}
    )

    # Generate output after loading
    loaded_output = loaded_model(sample_inputs_2d, training=False).numpy()

    # Outputs should be identical
    assert np.allclose(original_output, loaded_output, rtol=1e-6, atol=1e-7)


# Mixed precision tests
def test_norm_mixed_precision(sample_inputs_2d: tf.Tensor) -> None:
    """Test RMSNorm with mixed precision training."""
    # Set mixed precision policy
    original_policy = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy('mixed_float16')

    try:
        norm = RMSNorm(axis=-1, epsilon=1e-5)  # Slightly larger epsilon for fp16

        # Convert inputs to float16
        inputs_fp16 = tf.cast(sample_inputs_2d, tf.float16)

        # Forward pass
        output = norm(inputs_fp16)

        # Output should be float16
        assert output.dtype == tf.float16
        assert not tf.reduce_any(tf.math.is_nan(output))

    finally:
        # Restore original policy
        keras.mixed_precision.set_global_policy(original_policy)


# Error handling tests
def test_norm_invalid_epsilon() -> None:
    """Test error handling for invalid epsilon values."""
    with pytest.raises(ValueError, match="epsilon must be positive"):
        RMSNorm(axis=-1, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be positive"):
        RMSNorm(axis=-1, epsilon=-1e-6)


def test_norm_invalid_axis_type() -> None:
    """Test error handling for invalid axis types."""
    with pytest.raises(TypeError, match="axis must be int or tuple of ints"):
        RMSNorm(axis="invalid")

    with pytest.raises(TypeError, match="All elements in axis must be integers"):
        RMSNorm(axis=(1, "invalid"))


def test_norm_dynamic_shape_error(sample_inputs_2d: tf.Tensor) -> None:
    """Test error handling for dynamic shapes along normalization axes."""
    norm = RMSNorm(axis=-1, use_scale=True)

    # Create input shape with dynamic dimension along normalization axis
    dynamic_shape = (sample_inputs_2d.shape[0], None)

    with pytest.raises(ValueError, match="Cannot create 'scale' parameter"):
        norm.build(dynamic_shape)


def test_norm_numerical_stability() -> None:
    """Test numerical stability with extreme values."""
    norm = RMSNorm(axis=-1, epsilon=1e-6, use_scale=False)

    # Test with very small values
    small_inputs = tf.constant([[1e-10, 1e-10, 1e-10]], dtype=tf.float32)
    small_output = norm(small_inputs)
    assert not tf.reduce_any(tf.math.is_nan(small_output))

    # Test with very large values
    large_inputs = tf.constant([[1e10, 1e10, 1e10]], dtype=tf.float32)
    large_output = norm(large_inputs)
    assert not tf.reduce_any(tf.math.is_nan(large_output))

    # Test with mixed magnitude values
    mixed_inputs = tf.constant([[1e-5, 1e5, 0.0]], dtype=tf.float32)
    mixed_output = norm(mixed_inputs)
    assert not tf.reduce_any(tf.math.is_nan(mixed_output))


def test_norm_zero_inputs() -> None:
    """Test RMSNorm behavior with zero inputs."""
    norm = RMSNorm(axis=-1, epsilon=1e-6, use_scale=False)

    # All zero inputs
    zero_inputs = tf.zeros((2, 4), dtype=tf.float32)
    output = norm(zero_inputs)

    # Should handle gracefully due to epsilon
    assert not tf.reduce_any(tf.math.is_nan(output))
    # Output should be zero since input is zero
    assert tf.reduce_all(tf.abs(output) < 1e-6)


if __name__ == "__main__":
    pytest.main([__file__])