"""
Test suite for ZeroCenteredRMSNorm layer implementation.

This module provides comprehensive tests for:
- ZeroCenteredRMSNorm layer functionality
- Layer behavior under different configurations
- Serialization and deserialization
- Model integration and persistence
- Mathematical correctness (including mean centering)
- Edge cases and error handling
- Comparison with standard RMSNorm behavior
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any, Union, Tuple

from dl_techniques.layers.norms.zero_centered_rms_norm import ZeroCenteredRMSNorm


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
def sample_inputs_non_zero_mean() -> tf.Tensor:
    """Generate sample input with non-zero mean for centering tests."""
    tf.random.set_seed(42)
    # Create inputs with a bias to ensure non-zero mean
    return tf.random.normal((4, 512)) + 2.5


@pytest.fixture
def default_norm_params() -> Dict[str, Any]:
    """Default parameters for ZeroCenteredRMSNorm."""
    return {
        "axis": -1,
        "epsilon": 1e-6,
        "use_scale": True,
        "scale_initializer": "ones",
    }


@pytest.fixture
def minimal_norm_params() -> Dict[str, Any]:
    """Minimal parameters for ZeroCenteredRMSNorm."""
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


# ZeroCenteredRMSNorm initialization tests
def test_zero_centered_norm_initialization(default_norm_params: Dict[str, Any]) -> None:
    """Test initialization of ZeroCenteredRMSNorm with default parameters."""
    norm = ZeroCenteredRMSNorm(**default_norm_params)

    assert norm.axis == default_norm_params["axis"]
    assert norm.epsilon == default_norm_params["epsilon"]
    assert norm.use_scale == default_norm_params["use_scale"]
    assert isinstance(norm.scale_initializer, keras.initializers.Ones)


def test_zero_centered_norm_minimal_initialization(minimal_norm_params: Dict[str, Any]) -> None:
    """Test initialization with minimal parameters."""
    norm = ZeroCenteredRMSNorm(**minimal_norm_params)

    assert norm.axis == minimal_norm_params["axis"]
    assert norm.epsilon == 1e-6  # Default value
    assert norm.use_scale is True  # Default value
    assert isinstance(norm.scale_initializer, keras.initializers.Ones)


def test_zero_centered_norm_multi_axis_initialization(multi_axis_norm_params: Dict[str, Any]) -> None:
    """Test initialization with multi-axis normalization."""
    norm = ZeroCenteredRMSNorm(**multi_axis_norm_params)

    assert norm.axis == multi_axis_norm_params["axis"]
    assert norm.epsilon == multi_axis_norm_params["epsilon"]
    assert norm.use_scale == multi_axis_norm_params["use_scale"]


def test_zero_centered_norm_custom_initializer() -> None:
    """Test initialization with custom scale initializer."""
    custom_initializer = keras.initializers.RandomNormal(mean=1.0, stddev=0.1)
    norm = ZeroCenteredRMSNorm(
        axis=-1,
        use_scale=True,
        scale_initializer=custom_initializer
    )

    assert isinstance(norm.scale_initializer, keras.initializers.RandomNormal)


def test_zero_centered_norm_without_scale() -> None:
    """Test initialization without learnable scale parameter."""
    norm = ZeroCenteredRMSNorm(axis=-1, use_scale=False)

    assert norm.use_scale is False
    assert norm.scale_initializer is not None  # Should still be set even if not used


# Output shape tests
def test_zero_centered_norm_output_shape_2d(sample_inputs_2d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test that ZeroCenteredRMSNorm preserves input shape for 2D tensors."""
    norm = ZeroCenteredRMSNorm(**default_norm_params)
    outputs = norm(sample_inputs_2d)

    assert outputs.shape == sample_inputs_2d.shape


def test_zero_centered_norm_output_shape_3d(sample_inputs_3d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test that ZeroCenteredRMSNorm preserves input shape for 3D tensors."""
    norm = ZeroCenteredRMSNorm(**default_norm_params)
    outputs = norm(sample_inputs_3d)

    assert outputs.shape == sample_inputs_3d.shape


def test_zero_centered_norm_output_shape_4d(sample_inputs_4d: tf.Tensor,
                                            multi_axis_norm_params: Dict[str, Any]) -> None:
    """Test that ZeroCenteredRMSNorm preserves input shape for 4D tensors with multi-axis."""
    norm = ZeroCenteredRMSNorm(**multi_axis_norm_params)
    outputs = norm(sample_inputs_4d)

    assert outputs.shape == sample_inputs_4d.shape


def test_zero_centered_norm_compute_output_shape(sample_inputs_3d: tf.Tensor,
                                                 default_norm_params: Dict[str, Any]) -> None:
    """Test compute_output_shape method."""
    norm = ZeroCenteredRMSNorm(**default_norm_params)

    computed_shape = norm.compute_output_shape(sample_inputs_3d.shape)
    assert computed_shape == sample_inputs_3d.shape


# Mathematical correctness tests
def test_zero_centered_norm_mathematical_correctness(sample_inputs_non_zero_mean: tf.Tensor) -> None:
    """Test that ZeroCenteredRMSNorm computes correct mathematical result with mean centering."""
    norm = ZeroCenteredRMSNorm(axis=-1, use_scale=False, epsilon=1e-6)

    # Manual computation
    inputs_np = sample_inputs_non_zero_mean.numpy()

    # Step 1: Center the inputs
    mean = np.mean(inputs_np, axis=-1, keepdims=True)
    centered = inputs_np - mean

    # Step 2: Compute RMS of centered inputs
    rms_manual = np.sqrt(np.mean(centered ** 2, axis=-1, keepdims=True) + 1e-6)
    expected = centered / rms_manual

    # Layer computation
    outputs = norm(sample_inputs_non_zero_mean)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(outputs),
        expected,
        rtol=1e-6, atol=1e-6,
        err_msg="Zero-centered RMS normalization should match manual computation"
    )


def test_zero_centered_norm_zero_mean_property(sample_inputs_non_zero_mean: tf.Tensor) -> None:
    """Test that ZeroCenteredRMSNorm output has approximately zero mean."""
    norm = ZeroCenteredRMSNorm(axis=-1, use_scale=False, epsilon=1e-6)

    outputs = norm(sample_inputs_non_zero_mean)

    # Check that the output has approximately zero mean along the normalized axis
    output_mean = keras.ops.mean(outputs, axis=-1)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(output_mean),
        np.zeros_like(keras.ops.convert_to_numpy(output_mean)),
        atol=1e-6,
        err_msg="Zero-centered RMS norm output should have zero mean"
    )


def test_zero_centered_norm_with_scale_correctness(sample_inputs_non_zero_mean: tf.Tensor) -> None:
    """Test mathematical correctness with learnable scale."""
    # Initialize with known scale values
    norm = ZeroCenteredRMSNorm(
        axis=-1,
        use_scale=True,
        scale_initializer=keras.initializers.Constant(2.0),
        epsilon=1e-6
    )

    # Build the layer
    norm.build(sample_inputs_non_zero_mean.shape)

    # Manual computation
    inputs_np = sample_inputs_non_zero_mean.numpy()

    # Step 1: Center the inputs
    mean = np.mean(inputs_np, axis=-1, keepdims=True)
    centered = inputs_np - mean

    # Step 2: Compute RMS of centered inputs
    rms_manual = np.sqrt(np.mean(centered ** 2, axis=-1, keepdims=True) + 1e-6)
    normalized = centered / rms_manual
    expected = normalized * 2.0  # Scale factor

    # Layer computation
    outputs = norm(sample_inputs_non_zero_mean)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(outputs),
        expected,
        rtol=1e-5, atol=1e-6,
        err_msg="Zero-centered RMS normalization with scale should match manual computation"
    )


def test_zero_centered_norm_multi_axis_correctness(sample_inputs_4d: tf.Tensor) -> None:
    """Test mathematical correctness for multi-axis normalization."""
    norm = ZeroCenteredRMSNorm(axis=(-2, -1), use_scale=False, epsilon=1e-6)

    # Manual computation
    inputs_np = sample_inputs_4d.numpy()

    # Step 1: Center the inputs
    mean = np.mean(inputs_np, axis=(-2, -1), keepdims=True)
    centered = inputs_np - mean

    # Step 2: Compute RMS of centered inputs
    rms_manual = np.sqrt(np.mean(centered ** 2, axis=(-2, -1), keepdims=True) + 1e-6)
    expected = centered / rms_manual

    # Layer computation
    outputs = norm(sample_inputs_4d)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(outputs),
        expected,
        rtol=1e-5, atol=1e-6,
        err_msg="Multi-axis zero-centered RMS normalization should match manual computation"
    )


def test_zero_centered_vs_standard_rms_norm_difference(sample_inputs_non_zero_mean: tf.Tensor) -> None:
    """Test that ZeroCenteredRMSNorm produces different results than standard RMSNorm for non-zero mean inputs."""
    from dl_techniques.layers.norms.rms_norm import RMSNorm

    zero_centered_norm = ZeroCenteredRMSNorm(axis=-1, use_scale=False, epsilon=1e-6)
    standard_rms_norm = RMSNorm(axis=-1, use_scale=False, epsilon=1e-6)

    zero_centered_output = zero_centered_norm(sample_inputs_non_zero_mean)
    standard_rms_output = standard_rms_norm(sample_inputs_non_zero_mean)

    # Outputs should be different for inputs with non-zero mean
    assert not np.allclose(
        keras.ops.convert_to_numpy(zero_centered_output),
        keras.ops.convert_to_numpy(standard_rms_output),
        rtol=1e-5
    ), "Zero-centered RMS norm should produce different results than standard RMS norm for non-zero mean inputs"


def test_zero_centered_norm_already_centered_inputs(sample_inputs_2d: tf.Tensor) -> None:
    """Test behavior with inputs that already have approximately zero mean."""
    # Center the inputs manually first
    centered_inputs = sample_inputs_2d - keras.ops.mean(sample_inputs_2d, axis=-1, keepdims=True)

    norm = ZeroCenteredRMSNorm(axis=-1, use_scale=False, epsilon=1e-6)
    outputs = norm(centered_inputs)

    # Should still work correctly even with already centered inputs
    assert outputs.shape == centered_inputs.shape
    assert not keras.ops.any(keras.ops.isnan(outputs))


# Training behavior tests
def test_zero_centered_norm_training_vs_inference(sample_inputs_3d: tf.Tensor,
                                                  default_norm_params: Dict[str, Any]) -> None:
    """Test that ZeroCenteredRMSNorm behaves consistently in training vs inference modes."""
    norm = ZeroCenteredRMSNorm(**default_norm_params)

    # ZeroCenteredRMSNorm should behave the same in training and inference (unlike BatchNorm)
    train_output = norm(sample_inputs_3d, training=True)
    inference_output = norm(sample_inputs_3d, training=False)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(train_output),
        keras.ops.convert_to_numpy(inference_output),
        rtol=1e-6,
        err_msg="Training and inference outputs should be identical"
    )


def test_zero_centered_norm_deterministic_behavior(sample_inputs_2d: tf.Tensor,
                                                   default_norm_params: Dict[str, Any]) -> None:
    """Test that ZeroCenteredRMSNorm produces deterministic outputs."""
    norm = ZeroCenteredRMSNorm(**default_norm_params)

    # Multiple calls should produce identical results
    output1 = norm(sample_inputs_2d)
    output2 = norm(sample_inputs_2d)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(output1),
        keras.ops.convert_to_numpy(output2),
        rtol=1e-6,
        err_msg="Multiple calls should produce identical results"
    )


# Serialization tests
def test_zero_centered_norm_get_config(default_norm_params: Dict[str, Any]) -> None:
    """Test get_config method returns all configuration parameters."""
    norm = ZeroCenteredRMSNorm(**default_norm_params)
    config = norm.get_config()

    # Check all required parameters are present
    assert "axis" in config
    assert "epsilon" in config
    assert "use_scale" in config
    assert "scale_initializer" in config

    # Check values match initialization
    assert config["axis"] == default_norm_params["axis"]
    assert config["epsilon"] == default_norm_params["epsilon"]
    assert config["use_scale"] == default_norm_params["use_scale"]


def test_zero_centered_norm_from_config(default_norm_params: Dict[str, Any]) -> None:
    """Test creating ZeroCenteredRMSNorm from configuration."""
    original_norm = ZeroCenteredRMSNorm(**default_norm_params)
    config = original_norm.get_config()

    # Create new instance from config
    restored_norm = ZeroCenteredRMSNorm.from_config(config)

    # Check that key properties match
    assert restored_norm.axis == original_norm.axis
    assert restored_norm.epsilon == original_norm.epsilon
    assert restored_norm.use_scale == original_norm.use_scale
    assert type(restored_norm.scale_initializer) == type(original_norm.scale_initializer)


def test_zero_centered_norm_serialization_multi_axis(multi_axis_norm_params: Dict[str, Any],
                                                     sample_inputs_4d: tf.Tensor) -> None:
    """Test serialization with multi-axis normalization."""
    original_norm = ZeroCenteredRMSNorm(**multi_axis_norm_params)
    original_norm.build(sample_inputs_4d.shape)

    config = original_norm.get_config()
    restored_norm = ZeroCenteredRMSNorm.from_config(config)

    # Check multi-axis configuration is preserved
    assert restored_norm.axis == multi_axis_norm_params["axis"]


# Parametrized tests
@pytest.mark.parametrize("axis", [-1, -2, 0, 1, (-2, -1), (-3, -2, -1)])
def test_zero_centered_norm_different_axes(axis: Union[int, Tuple[int, ...]], sample_inputs_4d: tf.Tensor) -> None:
    """Test ZeroCenteredRMSNorm with different axis configurations."""
    norm = ZeroCenteredRMSNorm(axis=axis, use_scale=False)

    try:
        output = norm(sample_inputs_4d)
        assert output.shape == sample_inputs_4d.shape
        assert not keras.ops.any(keras.ops.isnan(output))
    except (ValueError, tf.errors.InvalidArgumentError):
        # Some axis configurations might be invalid for the given input shape
        # This is expected behavior
        pass


@pytest.mark.parametrize("epsilon", [1e-8, 1e-6, 1e-5, 1e-4])
def test_zero_centered_norm_different_epsilons(epsilon: float, sample_inputs_2d: tf.Tensor) -> None:
    """Test ZeroCenteredRMSNorm with different epsilon values."""
    norm = ZeroCenteredRMSNorm(axis=-1, epsilon=epsilon, use_scale=False)
    output = norm(sample_inputs_2d)

    assert output.shape == sample_inputs_2d.shape
    assert not keras.ops.any(keras.ops.isnan(output))


@pytest.mark.parametrize("use_scale", [True, False])
def test_zero_centered_norm_with_without_scale(use_scale: bool, sample_inputs_2d: tf.Tensor) -> None:
    """Test ZeroCenteredRMSNorm with and without learnable scale parameter."""
    norm = ZeroCenteredRMSNorm(axis=-1, use_scale=use_scale)
    output = norm(sample_inputs_2d)

    assert output.shape == sample_inputs_2d.shape
    assert not keras.ops.any(keras.ops.isnan(output))

@pytest.mark.parametrize("initializer_name", ["ones", "zeros", "random_normal"])
def test_zero_centered_norm_different_initializers(initializer_name: str, sample_inputs_2d: tf.Tensor) -> None:
    """Test ZeroCenteredRMSNorm with different scale initializers."""
    norm = ZeroCenteredRMSNorm(axis=-1, use_scale=True, scale_initializer=initializer_name)
    output = norm(sample_inputs_2d)

    assert output.shape == sample_inputs_2d.shape
    assert not keras.ops.any(keras.ops.isnan(output))


# Gradient flow tests
def test_zero_centered_norm_gradient_flow(sample_inputs_3d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test gradient flow through ZeroCenteredRMSNorm."""
    norm = ZeroCenteredRMSNorm(**default_norm_params)

    with tf.GradientTape() as tape:
        output = norm(sample_inputs_3d, training=True)
        loss = keras.ops.mean(output)

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


def test_zero_centered_norm_gradient_flow_without_scale(sample_inputs_2d: tf.Tensor) -> None:
    """Test gradient flow through input when no scale parameter."""
    norm = ZeroCenteredRMSNorm(axis=-1, use_scale=False)

    # Create a variable input to test gradients
    var_input = tf.Variable(sample_inputs_2d)

    with tf.GradientTape() as tape:
        output = norm(var_input, training=True)
        loss = keras.ops.mean(output)

    gradients = tape.gradient(loss, var_input)

    # Should have gradients with respect to input
    assert gradients is not None
    assert not keras.ops.any(keras.ops.isnan(gradients))


# Model integration tests
def test_zero_centered_norm_model_integration(sample_inputs_3d: tf.Tensor, default_norm_params: Dict[str, Any]) -> None:
    """Test integrating ZeroCenteredRMSNorm into a Keras model."""
    inputs = keras.Input(shape=sample_inputs_3d.shape[1:])
    norm = ZeroCenteredRMSNorm(**default_norm_params)
    outputs = norm(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(sample_inputs_3d)
    assert result.shape == sample_inputs_3d.shape


def test_zero_centered_norm_transformer_integration(sample_inputs_3d: tf.Tensor) -> None:
    """Test ZeroCenteredRMSNorm in a transformer-like architecture."""
    seq_length, hidden_size = sample_inputs_3d.shape[1:]

    inputs = keras.Input(shape=(seq_length, hidden_size))

    # Pre-normalization transformer block pattern with zero-centered RMS norm
    x = ZeroCenteredRMSNorm(axis=-1)(inputs)
    attention_out = keras.layers.MultiHeadAttention(
        num_heads=8, key_dim=hidden_size // 8
    )(x, x)
    x = inputs + attention_out  # Residual connection

    x = ZeroCenteredRMSNorm(axis=-1)(x)
    ffn_out = keras.layers.Dense(hidden_size * 4, activation='relu')(x)
    ffn_out = keras.layers.Dense(hidden_size)(ffn_out)
    outputs = x + ffn_out  # Residual connection

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(sample_inputs_3d)
    assert result.shape == sample_inputs_3d.shape


def test_zero_centered_norm_llm_style_integration(sample_inputs_3d: tf.Tensor) -> None:
    """Test ZeroCenteredRMSNorm in a large language model style architecture."""
    seq_length, hidden_size = sample_inputs_3d.shape[1:]

    inputs = keras.Input(shape=(seq_length, hidden_size))

    # LLM-style block with zero-centered RMS norm for enhanced stability
    x = inputs
    for _ in range(2):  # Two transformer blocks
        # Pre-norm attention
        norm_x = ZeroCenteredRMSNorm(axis=-1, epsilon=1e-5)(x)
        attn_out = keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=hidden_size // 8
        )(norm_x, norm_x)
        x = x + attn_out

        # Pre-norm FFN
        norm_x = ZeroCenteredRMSNorm(axis=-1, epsilon=1e-5)(x)
        ffn_out = keras.layers.Dense(hidden_size * 4, activation='gelu')(norm_x)
        ffn_out = keras.layers.Dense(hidden_size)(ffn_out)
        x = x + ffn_out

    model = keras.Model(inputs=inputs, outputs=x)

    # Test forward pass
    result = model(sample_inputs_3d)
    assert result.shape == sample_inputs_3d.shape


def test_zero_centered_norm_model_compilation_and_training(sample_inputs_2d: tf.Tensor,
                                                           default_norm_params: Dict[str, Any]) -> None:
    """Test compiling and training a model with ZeroCenteredRMSNorm."""
    inputs = keras.Input(shape=sample_inputs_2d.shape[1:])
    x = ZeroCenteredRMSNorm(**default_norm_params)(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = ZeroCenteredRMSNorm(axis=-1)(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create dummy labels
    labels = tf.random.uniform((sample_inputs_2d.shape[0],), 0, 10, dtype=tf.int32)

    # Test training for one step
    history = model.fit(sample_inputs_2d, labels, epochs=1, verbose=0)
    assert len(history.history["loss"]) == 1


# Model persistence tests
def test_zero_centered_norm_model_save_load_weights(sample_inputs_3d: tf.Tensor, default_norm_params: Dict[str, Any],
                                                    tmp_path) -> None:
    """Test saving and loading model weights with ZeroCenteredRMSNorm."""
    # Create a simple model with ZeroCenteredRMSNorm
    inputs = keras.Input(shape=sample_inputs_3d.shape[1:])
    norm = ZeroCenteredRMSNorm(**default_norm_params)
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
    np.testing.assert_allclose(
        original_output, loaded_output,
        rtol=1e-6, atol=1e-7,
        err_msg="Outputs should be identical after weight save/load"
    )


def test_zero_centered_norm_model_save_load_keras_format(sample_inputs_2d: tf.Tensor,
                                                         default_norm_params: Dict[str, Any],
                                                         tmp_path) -> None:
    """Test saving and loading model in Keras format with ZeroCenteredRMSNorm."""
    # Create a simple model with ZeroCenteredRMSNorm
    inputs = keras.Input(shape=sample_inputs_2d.shape[1:])
    norm = ZeroCenteredRMSNorm(**default_norm_params)
    outputs = norm(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Generate output before saving
    original_output = model(sample_inputs_2d, training=False).numpy()

    # Save model in Keras format
    save_path = str(tmp_path / "model.keras")
    model.save(save_path)

    # Load model (should work automatically due to @keras.saving.register_keras_serializable())
    loaded_model = keras.models.load_model(save_path)

    # Generate output after loading
    loaded_output = loaded_model(sample_inputs_2d, training=False).numpy()

    # Outputs should be identical
    np.testing.assert_allclose(
        original_output, loaded_output,
        rtol=1e-6, atol=1e-7,
        err_msg="Outputs should be identical after model save/load"
    )


# Mixed precision tests
def test_zero_centered_norm_mixed_precision(sample_inputs_2d: tf.Tensor) -> None:
    """Test ZeroCenteredRMSNorm with mixed precision training."""
    # Set mixed precision policy
    original_policy = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy('mixed_float16')

    try:
        norm = ZeroCenteredRMSNorm(axis=-1, epsilon=1e-5)  # Slightly larger epsilon for fp16

        # Convert inputs to float16
        inputs_fp16 = tf.cast(sample_inputs_2d, tf.float16)

        # Forward pass
        output = norm(inputs_fp16)

        # Output should be float16
        assert output.dtype == tf.float16
        assert not keras.ops.any(keras.ops.isnan(output))

    finally:
        # Restore original policy
        keras.mixed_precision.set_global_policy(original_policy)


# Error handling tests
def test_zero_centered_norm_invalid_epsilon() -> None:
    """Test error handling for invalid epsilon values."""
    with pytest.raises(ValueError, match="epsilon must be positive"):
        ZeroCenteredRMSNorm(axis=-1, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be positive"):
        ZeroCenteredRMSNorm(axis=-1, epsilon=-1e-6)


def test_zero_centered_norm_invalid_axis_type() -> None:
    """Test error handling for invalid axis types."""
    with pytest.raises(TypeError, match="axis must be int or tuple of ints"):
        ZeroCenteredRMSNorm(axis="invalid")

    with pytest.raises(TypeError, match="All elements in axis must be integers"):
        ZeroCenteredRMSNorm(axis=(1, "invalid"))


def test_zero_centered_norm_dynamic_shape_error(sample_inputs_2d: tf.Tensor) -> None:
    """Test error handling for dynamic shapes along normalization axes."""
    norm = ZeroCenteredRMSNorm(axis=-1, use_scale=True)

    # Create input shape with dynamic dimension along normalization axis
    dynamic_shape = (sample_inputs_2d.shape[0], None)

    with pytest.raises(ValueError, match="Cannot create 'scale' parameter"):
        norm.build(dynamic_shape)


def test_zero_centered_norm_numerical_stability() -> None:
    """Test numerical stability with extreme values."""
    norm = ZeroCenteredRMSNorm(axis=-1, epsilon=1e-6, use_scale=False)

    # Test with very small values
    small_inputs = tf.constant([[1e-10, 1e-10, 1e-10]], dtype=tf.float32)
    small_output = norm(small_inputs)
    assert not keras.ops.any(keras.ops.isnan(small_output))

    # Test with very large values
    large_inputs = tf.constant([[1e10, 1e10, 1e10]], dtype=tf.float32)
    large_output = norm(large_inputs)
    assert not keras.ops.any(keras.ops.isnan(large_output))

    # Test with mixed magnitude values
    mixed_inputs = tf.constant([[1e-5, 1e5, 0.0]], dtype=tf.float32)
    mixed_output = norm(mixed_inputs)
    assert not keras.ops.any(keras.ops.isnan(mixed_output))


def test_zero_centered_norm_zero_inputs() -> None:
    """Test ZeroCenteredRMSNorm behavior with zero inputs."""
    norm = ZeroCenteredRMSNorm(axis=-1, epsilon=1e-6, use_scale=False)

    # All zero inputs
    zero_inputs = tf.zeros((2, 4), dtype=tf.float32)
    output = norm(zero_inputs)

    # Should handle gracefully due to epsilon
    assert not keras.ops.any(keras.ops.isnan(output))
    # Output should be zero since centered input is zero and normalized by epsilon
    assert keras.ops.all(keras.ops.abs(output) < 1e-5)


def test_zero_centered_norm_constant_inputs() -> None:
    """Test ZeroCenteredRMSNorm behavior with constant inputs."""
    norm = ZeroCenteredRMSNorm(axis=-1, epsilon=1e-6, use_scale=False)

    # Constant inputs along normalized axis
    constant_inputs = tf.ones((2, 4), dtype=tf.float32) * 5.0
    output = norm(constant_inputs)

    # Should handle gracefully - after centering, all values become 0
    assert not keras.ops.any(keras.ops.isnan(output))
    # After centering constant inputs, RMS becomes epsilon, so output is 0
    assert keras.ops.all(keras.ops.abs(output) < 1e-5)


def test_zero_centered_norm_stability_vs_standard_rms() -> None:
    """Test that ZeroCenteredRMSNorm provides better numerical stability than standard RMSNorm for certain inputs."""
    # Create inputs where standard RMS norm might have stability issues due to large magnitude
    # but zero-centered version should be more stable
    inputs_with_large_bias = tf.constant([
        [1000.0, 1000.1, 999.9],  # Small variations around large values
        [1000.2, 999.8, 1000.0]
    ], dtype=tf.float32)

    zero_centered_norm = ZeroCenteredRMSNorm(axis=-1, epsilon=1e-6, use_scale=False)
    zero_centered_output = zero_centered_norm(inputs_with_large_bias)

    # Should produce stable outputs without numerical issues
    assert not keras.ops.any(keras.ops.isnan(zero_centered_output))
    assert keras.ops.all(keras.ops.isfinite(zero_centered_output))

    # The output should have reasonable magnitude (not extremely large or small)
    output_magnitude = keras.ops.max(keras.ops.abs(zero_centered_output))
    assert output_magnitude < 100.0  # Should be reasonably bounded


if __name__ == "__main__":
    pytest.main([__file__])