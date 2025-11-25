"""
Test suite for QuantileHead implementation.

This module provides comprehensive tests for:
- QuantileHead layer initialization
- Output shape validation for both flatten and non-flatten modes
- Monotonicity enforcement
- Dropout behavior in training vs inference
- Flatten input functionality
- Serialization and deserialization
- Model integration and persistence
- Gradient flow
- Parameter validation
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any, Tuple


from dl_techniques.layers.time_series.quantile_head_fixed_io import QuantileHead


# Helper functions
def validate_quantile_outputs(quantiles: tf.Tensor, enforce_monotonicity: bool = False) -> None:
    """Validate that quantile outputs are reasonable."""
    # Check for NaN or Inf values
    assert not tf.reduce_any(tf.math.is_nan(quantiles)), "Quantiles contain NaN values"
    assert not tf.reduce_any(tf.math.is_inf(quantiles)), "Quantiles contain Inf values"

    # Check that outputs are not all zeros (which might indicate a problem)
    assert tf.reduce_any(tf.abs(quantiles) > 1e-6), "Quantiles are all zeros"

    # If monotonicity is enforced, verify that quantiles are non-decreasing
    if enforce_monotonicity and quantiles.shape[-1] > 1:
        # Check along the quantile dimension (last axis)
        for i in range(quantiles.shape[-1] - 1):
            # For each batch and time step, verify Q_i <= Q_{i+1}
            diff = quantiles[..., i + 1] - quantiles[..., i]
            # Allow small numerical errors
            assert tf.reduce_all(diff >= -1e-5), \
                f"Quantile crossing detected: Q_{i + 1} < Q_{i} at some positions"


def check_monotonicity(quantiles: np.ndarray, tolerance: float = 1e-5) -> bool:
    """Check if quantiles are monotonically non-decreasing."""
    if quantiles.shape[-1] <= 1:
        return True

    # Check all adjacent pairs
    for i in range(quantiles.shape[-1] - 1):
        diff = quantiles[..., i + 1] - quantiles[..., i]
        if np.any(diff < -tolerance):
            return False
    return True


# Test fixtures
@pytest.fixture
def sample_2d_inputs() -> tf.Tensor:
    """Generate sample 2D input tensor (batch, features)."""
    tf.random.set_seed(42)
    # Shape: (batch_size, features)
    return tf.random.normal((4, 128))


@pytest.fixture
def sample_3d_inputs() -> tf.Tensor:
    """Generate sample 3D input tensor (batch, seq, features) for flatten mode."""
    tf.random.set_seed(42)
    # Shape: (batch_size, sequence_length, features)
    return tf.random.normal((4, 24, 64))  # 24 time steps, 64 features


@pytest.fixture
def small_2d_inputs() -> tf.Tensor:
    """Generate smaller sample 2D input tensor for faster tests."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 32))


@pytest.fixture
def small_3d_inputs() -> tf.Tensor:
    """Generate smaller sample 3D input tensor for faster tests."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 10, 16))


@pytest.fixture
def default_quantile_head_params() -> Dict[str, Any]:
    """Default parameters for QuantileHead."""
    return {
        "num_quantiles": 3,
        "output_length": 12,
        "dropout_rate": 0.1,
        "use_bias": True,
        "flatten_input": False,
        "enforce_monotonicity": False,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
    }


@pytest.fixture
def flatten_quantile_head_params() -> Dict[str, Any]:
    """Parameters for QuantileHead with flatten_input=True."""
    return {
        "num_quantiles": 3,
        "output_length": 12,
        "dropout_rate": 0.1,
        "use_bias": True,
        "flatten_input": True,
        "enforce_monotonicity": False,
    }


@pytest.fixture
def minimal_quantile_head_params() -> Dict[str, Any]:
    """Minimal parameters for QuantileHead."""
    return {
        "num_quantiles": 3,
        "output_length": 6,
    }


@pytest.fixture
def monotonic_quantile_head_params() -> Dict[str, Any]:
    """Parameters for monotonic QuantileHead."""
    return {
        "num_quantiles": 5,
        "output_length": 12,
        "dropout_rate": 0.0,
        "enforce_monotonicity": True,
    }


# Basic initialization tests
def test_quantile_head_initialization(default_quantile_head_params: Dict[str, Any]) -> None:
    """Test initialization of QuantileHead."""
    head = QuantileHead(**default_quantile_head_params)

    assert head.num_quantiles == default_quantile_head_params["num_quantiles"]
    assert head.output_length == default_quantile_head_params["output_length"]
    assert head.dropout_rate == default_quantile_head_params["dropout_rate"]
    assert head.use_bias == default_quantile_head_params["use_bias"]
    assert head.flatten_input == default_quantile_head_params["flatten_input"]
    assert head.enforce_monotonicity == default_quantile_head_params["enforce_monotonicity"]


def test_quantile_head_minimal_initialization(minimal_quantile_head_params: Dict[str, Any]) -> None:
    """Test initialization with minimal parameters."""
    head = QuantileHead(**minimal_quantile_head_params)

    assert head.num_quantiles == minimal_quantile_head_params["num_quantiles"]
    assert head.output_length == minimal_quantile_head_params["output_length"]
    assert head.dropout_rate == 0.1  # default
    assert head.use_bias is True  # default
    assert head.flatten_input is False  # default
    assert head.enforce_monotonicity is False  # default


def test_quantile_head_flatten_initialization(flatten_quantile_head_params: Dict[str, Any]) -> None:
    """Test initialization with flatten_input=True."""
    head = QuantileHead(**flatten_quantile_head_params)

    assert head.flatten_input is True
    assert head.num_quantiles == flatten_quantile_head_params["num_quantiles"]
    assert head.output_length == flatten_quantile_head_params["output_length"]


# Output shape tests - Non-flatten mode
def test_quantile_head_output_shapes_2d(
        small_2d_inputs: tf.Tensor,
        minimal_quantile_head_params: Dict[str, Any]
) -> None:
    """Test QuantileHead output shapes with 2D input (non-flatten mode)."""
    head = QuantileHead(**minimal_quantile_head_params)
    outputs = head(small_2d_inputs)

    batch_size = small_2d_inputs.shape[0]
    expected_shape = (
        batch_size,
        minimal_quantile_head_params["output_length"],
        minimal_quantile_head_params["num_quantiles"]
    )

    assert outputs.shape == expected_shape
    validate_quantile_outputs(outputs, enforce_monotonicity=False)


def test_quantile_head_compute_output_shape(minimal_quantile_head_params: Dict[str, Any]) -> None:
    """Test QuantileHead compute_output_shape method."""
    head = QuantileHead(**minimal_quantile_head_params)

    input_shape = (4, 64)
    output_shape = head.compute_output_shape(input_shape)

    expected_shape = (
        4,
        minimal_quantile_head_params["output_length"],
        minimal_quantile_head_params["num_quantiles"]
    )
    assert output_shape == expected_shape


# Output shape tests - Flatten mode
def test_quantile_head_flatten_output_shapes(
        small_3d_inputs: tf.Tensor,
        flatten_quantile_head_params: Dict[str, Any]
) -> None:
    """Test QuantileHead output shapes with flatten_input=True."""
    head = QuantileHead(**flatten_quantile_head_params)
    outputs = head(small_3d_inputs)

    batch_size = small_3d_inputs.shape[0]
    expected_shape = (
        batch_size,
        flatten_quantile_head_params["output_length"],
        flatten_quantile_head_params["num_quantiles"]
    )

    assert outputs.shape == expected_shape
    validate_quantile_outputs(outputs, enforce_monotonicity=False)


def test_quantile_head_flatten_compute_output_shape(flatten_quantile_head_params: Dict[str, Any]) -> None:
    """Test compute_output_shape with flatten_input=True."""
    head = QuantileHead(**flatten_quantile_head_params)

    input_shape = (4, 24, 32)  # 3D input
    output_shape = head.compute_output_shape(input_shape)

    expected_shape = (
        4,
        flatten_quantile_head_params["output_length"],
        flatten_quantile_head_params["num_quantiles"]
    )
    assert output_shape == expected_shape


def test_quantile_head_flatten_vs_non_flatten_shapes(
        sample_2d_inputs: tf.Tensor,
        sample_3d_inputs: tf.Tensor
) -> None:
    """Test that both modes produce the same output shape structure."""
    # Non-flatten mode with 2D input
    head_2d = QuantileHead(num_quantiles=3, output_length=12, flatten_input=False)
    output_2d = head_2d(sample_2d_inputs)

    # Flatten mode with 3D input
    head_3d = QuantileHead(num_quantiles=3, output_length=12, flatten_input=True)
    output_3d = head_3d(sample_3d_inputs)

    # Both should produce 3D output with same last two dimensions
    assert len(output_2d.shape) == 3
    assert len(output_3d.shape) == 3
    assert output_2d.shape[-2:] == output_3d.shape[-2:]  # Same (output_length, num_quantiles)


# Monotonicity tests
def test_quantile_head_monotonicity_enforcement_2d(
        small_2d_inputs: tf.Tensor,
        monotonic_quantile_head_params: Dict[str, Any]
) -> None:
    """Test monotonicity enforcement with 2D input."""
    head = QuantileHead(**monotonic_quantile_head_params)
    outputs = head(small_2d_inputs, training=False)

    # Validate monotonicity
    validate_quantile_outputs(outputs, enforce_monotonicity=True)

    # Convert to numpy for detailed checking
    outputs_np = outputs.numpy()
    assert check_monotonicity(outputs_np), "Quantiles are not monotonic"


def test_quantile_head_monotonicity_enforcement_3d(
        small_3d_inputs: tf.Tensor,
        monotonic_quantile_head_params: Dict[str, Any]
) -> None:
    """Test monotonicity enforcement with 3D input (flatten mode)."""
    params = monotonic_quantile_head_params.copy()
    params["flatten_input"] = True

    head = QuantileHead(**params)
    outputs = head(small_3d_inputs, training=False)

    # Validate monotonicity
    validate_quantile_outputs(outputs, enforce_monotonicity=True)

    # Convert to numpy for detailed checking
    outputs_np = outputs.numpy()
    assert check_monotonicity(outputs_np), "Quantiles are not monotonic with flatten_input"


def test_quantile_head_monotonicity_with_random_inputs() -> None:
    """Test monotonicity with various random inputs."""
    head = QuantileHead(num_quantiles=7, output_length=10, enforce_monotonicity=True)

    # Test with multiple random seeds
    for seed in range(5):
        tf.random.set_seed(seed)
        inputs = tf.random.normal((3, 64))
        outputs = head(inputs, training=False)

        outputs_np = outputs.numpy()
        assert check_monotonicity(outputs_np), \
            f"Monotonicity violated with seed {seed}"


def test_quantile_head_monotonicity_extreme_values() -> None:
    """Test monotonicity with extreme input values."""
    head = QuantileHead(num_quantiles=5, output_length=8, enforce_monotonicity=True)

    # Test with very large values
    large_inputs = tf.random.normal((2, 32)) * 100.0
    large_outputs = head(large_inputs, training=False)
    assert check_monotonicity(large_outputs.numpy())

    # Test with very small values
    small_inputs = tf.random.normal((2, 32)) * 0.01
    small_outputs = head(small_inputs, training=False)
    assert check_monotonicity(small_outputs.numpy())


def test_quantile_head_without_monotonicity(small_2d_inputs: tf.Tensor) -> None:
    """Test that without monotonicity enforcement, quantiles can cross."""
    head = QuantileHead(num_quantiles=5, output_length=8, enforce_monotonicity=False)

    # Run to verify mechanism works
    outputs = head(small_2d_inputs, training=False)

    # Just verify outputs are valid (crossing is allowed)
    validate_quantile_outputs(outputs, enforce_monotonicity=False)


# Dropout tests
def test_quantile_head_dropout_training_vs_inference_2d(small_2d_inputs: tf.Tensor) -> None:
    """Test dropout behavior in training vs inference modes with 2D input."""
    head = QuantileHead(num_quantiles=3, output_length=6, dropout_rate=0.5)

    # Training mode - run multiple times, should get different outputs
    train_outputs_1 = head(small_2d_inputs, training=True)
    train_outputs_2 = head(small_2d_inputs, training=True)

    # Outputs should differ due to dropout randomness
    assert not np.allclose(
        train_outputs_1.numpy(),
        train_outputs_2.numpy(),
        rtol=1e-5,
        atol=1e-5
    ), "Training outputs should differ due to dropout"

    # Inference mode - should be deterministic
    inference_outputs_1 = head(small_2d_inputs, training=False)
    inference_outputs_2 = head(small_2d_inputs, training=False)

    # Outputs should be identical (no dropout)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(inference_outputs_1),
        keras.ops.convert_to_numpy(inference_outputs_2),
        rtol=1e-6, atol=1e-6,
        err_msg="Inference outputs should be identical"
    )


def test_quantile_head_dropout_training_vs_inference_3d(small_3d_inputs: tf.Tensor) -> None:
    """Test dropout behavior with flatten_input=True."""
    head = QuantileHead(
        num_quantiles=3,
        output_length=6,
        dropout_rate=0.5,
        flatten_input=True
    )

    # Training mode - run multiple times
    train_outputs_1 = head(small_3d_inputs, training=True)
    train_outputs_2 = head(small_3d_inputs, training=True)

    # Should differ due to dropout
    assert not np.allclose(
        train_outputs_1.numpy(),
        train_outputs_2.numpy(),
        rtol=1e-5,
        atol=1e-5
    ), "Training outputs should differ due to dropout (flatten mode)"


def test_quantile_head_zero_dropout(small_2d_inputs: tf.Tensor) -> None:
    """Test that zero dropout results in identical outputs."""
    head = QuantileHead(num_quantiles=3, output_length=6, dropout_rate=0.0)

    # Training and inference should give same results with no dropout
    train_output = head(small_2d_inputs, training=True)
    inference_output = head(small_2d_inputs, training=False)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(train_output),
        keras.ops.convert_to_numpy(inference_output),
        rtol=1e-6, atol=1e-6,
        err_msg="Zero dropout should produce identical outputs"
    )


# Flatten input functionality tests
def test_quantile_head_flatten_input_dimensions() -> None:
    """Test that flatten_input correctly handles 3D inputs."""
    head = QuantileHead(num_quantiles=3, output_length=6, flatten_input=True)

    # 3D input: (batch, seq, features)
    inputs = tf.random.normal((4, 10, 8))  # 10 timesteps, 8 features -> flattened to 80
    outputs = head(inputs)

    assert outputs.shape == (4, 6, 3)


def test_quantile_head_flatten_requires_3d_input() -> None:
    """Test that flatten_input=True requires 3D input."""
    head = QuantileHead(num_quantiles=3, output_length=6, flatten_input=True)

    # Should raise error with 2D input
    with pytest.raises(ValueError, match="flatten_input=True expects a 3D input"):
        head.build((4, 64))  # 2D shape


def test_quantile_head_non_flatten_accepts_2d() -> None:
    """Test that flatten_input=False accepts 2D input."""
    head = QuantileHead(num_quantiles=3, output_length=6, flatten_input=False)

    # Should work fine with 2D input
    inputs = tf.random.normal((4, 64))
    outputs = head(inputs)

    assert outputs.shape == (4, 6, 3)


def test_quantile_head_flatten_with_different_seq_lengths() -> None:
    """Test flatten mode with different sequence lengths."""
    for seq_len in [5, 10, 20, 50]:
        head = QuantileHead(num_quantiles=3, output_length=6, flatten_input=True)
        inputs = tf.random.normal((2, seq_len, 16))
        outputs = head(inputs)

        assert outputs.shape == (2, 6, 3)


# Serialization tests
def test_quantile_head_serialization(
        default_quantile_head_params: Dict[str, Any],
        small_2d_inputs: tf.Tensor
) -> None:
    """Test serialization of QuantileHead."""
    # Create and build original head
    original_head = QuantileHead(**default_quantile_head_params)
    original_head.build(small_2d_inputs.shape)

    # Get config and recreate from config
    config = original_head.get_config()
    restored_head = QuantileHead.from_config(config)

    # Check key properties match
    assert restored_head.num_quantiles == original_head.num_quantiles
    assert restored_head.output_length == original_head.output_length
    assert restored_head.dropout_rate == original_head.dropout_rate
    assert restored_head.use_bias == original_head.use_bias
    assert restored_head.flatten_input == original_head.flatten_input
    assert restored_head.enforce_monotonicity == original_head.enforce_monotonicity


def test_quantile_head_get_config(default_quantile_head_params: Dict[str, Any]) -> None:
    """Test that get_config returns all necessary parameters."""
    head = QuantileHead(**default_quantile_head_params)
    config = head.get_config()

    # Check that all important parameters are in config
    assert "num_quantiles" in config
    assert "output_length" in config
    assert "dropout_rate" in config
    assert "use_bias" in config
    assert "flatten_input" in config
    assert "enforce_monotonicity" in config
    assert "kernel_initializer" in config
    assert "bias_initializer" in config


def test_quantile_head_serialization_with_flatten(
        flatten_quantile_head_params: Dict[str, Any],
        small_3d_inputs: tf.Tensor
) -> None:
    """Test serialization with flatten_input=True."""
    original_head = QuantileHead(**flatten_quantile_head_params)
    original_head.build(small_3d_inputs.shape)

    config = original_head.get_config()
    restored_head = QuantileHead.from_config(config)

    assert restored_head.flatten_input is True
    assert restored_head.num_quantiles == original_head.num_quantiles


# Model integration tests
def test_quantile_head_model_integration_2d(
        small_2d_inputs: tf.Tensor,
        minimal_quantile_head_params: Dict[str, Any]
) -> None:
    """Test integrating QuantileHead into a Keras model with 2D input."""
    batch_size, features = small_2d_inputs.shape

    inputs = keras.Input(shape=(features,))
    head = QuantileHead(**minimal_quantile_head_params)
    outputs = head(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(small_2d_inputs)

    expected_shape = (
        batch_size,
        minimal_quantile_head_params["output_length"],
        minimal_quantile_head_params["num_quantiles"]
    )
    assert result.shape == expected_shape


def test_quantile_head_model_integration_3d(
        small_3d_inputs: tf.Tensor,
        flatten_quantile_head_params: Dict[str, Any]
) -> None:
    """Test integrating QuantileHead with flatten_input=True."""
    batch_size, seq_len, features = small_3d_inputs.shape

    inputs = keras.Input(shape=(seq_len, features))
    head = QuantileHead(**flatten_quantile_head_params)
    outputs = head(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(small_3d_inputs)

    expected_shape = (
        batch_size,
        flatten_quantile_head_params["output_length"],
        flatten_quantile_head_params["num_quantiles"]
    )
    assert result.shape == expected_shape


def test_quantile_head_in_sequential_model(small_2d_inputs: tf.Tensor) -> None:
    """Test QuantileHead in a Sequential model."""
    batch_size, features = small_2d_inputs.shape

    model = keras.Sequential([
        keras.layers.Input(shape=(features,)),
        keras.layers.Dense(64, activation="relu"),
        QuantileHead(num_quantiles=3, output_length=12)
    ])

    outputs = model(small_2d_inputs)
    assert outputs.shape == (batch_size, 12, 3)


def test_quantile_head_with_lstm_encoder(small_3d_inputs: tf.Tensor) -> None:
    """Test QuantileHead with LSTM encoder and flatten_input."""
    batch_size, seq_len, features = small_3d_inputs.shape

    # Build encoder-decoder with LSTM
    inputs = keras.Input(shape=(seq_len, features))

    # LSTM encoder - return sequences for flatten mode
    x = keras.layers.LSTM(32, return_sequences=True)(inputs)

    # Quantile head with flatten
    quantiles = QuantileHead(
        num_quantiles=5,
        output_length=6,
        flatten_input=True,
        enforce_monotonicity=True
    )(x)

    model = keras.Model(inputs=inputs, outputs=quantiles)

    outputs = model(small_3d_inputs)
    assert outputs.shape == (batch_size, 6, 5)

    # Verify monotonicity
    assert check_monotonicity(outputs.numpy())


def test_quantile_head_with_dense_encoder(small_3d_inputs: tf.Tensor) -> None:
    """Test QuantileHead with Dense encoder (no flatten)."""
    batch_size, seq_len, features = small_3d_inputs.shape

    # Build encoder that reduces to 2D
    inputs = keras.Input(shape=(seq_len, features))

    # Flatten manually
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)

    # Quantile head without flatten (expects 2D)
    quantiles = QuantileHead(
        num_quantiles=3,
        output_length=12,
        flatten_input=False
    )(x)

    model = keras.Model(inputs=inputs, outputs=quantiles)

    outputs = model(small_3d_inputs)
    assert outputs.shape == (batch_size, 12, 3)


def test_model_compilation_and_training(
        small_2d_inputs: tf.Tensor,
        minimal_quantile_head_params: Dict[str, Any]
) -> None:
    """Test compiling and training a model with QuantileHead."""
    batch_size, features = small_2d_inputs.shape

    inputs = keras.Input(shape=(features,))
    quantiles = QuantileHead(**minimal_quantile_head_params)(inputs)

    model = keras.Model(inputs=inputs, outputs=quantiles)
    model.compile(optimizer="adam", loss="mse")

    # Create dummy targets matching output shape
    targets = tf.random.normal((
        batch_size,
        minimal_quantile_head_params["output_length"],
        minimal_quantile_head_params["num_quantiles"]
    ))

    # Test training for one step
    history = model.fit(small_2d_inputs, targets, epochs=1, verbose=0)
    assert len(history.history["loss"]) == 1


def test_model_save_load_keras_format(
        small_2d_inputs: tf.Tensor,
        minimal_quantile_head_params: Dict[str, Any],
        tmp_path
) -> None:
    """Test saving and loading a model with QuantileHead in Keras format."""
    batch_size, features = small_2d_inputs.shape

    # Create a model with the quantile head
    inputs = keras.Input(shape=(features,))
    quantiles = QuantileHead(**minimal_quantile_head_params)(inputs)
    model = keras.Model(inputs=inputs, outputs=quantiles)

    # Generate output before saving
    original_output = model(small_2d_inputs, training=False)

    # Save model in Keras format
    save_path = str(tmp_path / "quantile_head_model.keras")
    model.save(save_path)

    # Load model
    loaded_model = keras.models.load_model(
        save_path,
        custom_objects={"QuantileHead": QuantileHead}
    )

    # Generate output after loading
    loaded_output = loaded_model(small_2d_inputs, training=False)

    # Outputs should be identical
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(original_output),
        keras.ops.convert_to_numpy(loaded_output),
        rtol=1e-5, atol=1e-5,
        err_msg="Model outputs should match after save/load"
    )


def test_model_save_load_with_flatten(
        small_3d_inputs: tf.Tensor,
        flatten_quantile_head_params: Dict[str, Any],
        tmp_path
) -> None:
    """Test saving and loading with flatten_input=True."""
    batch_size, seq_len, features = small_3d_inputs.shape

    inputs = keras.Input(shape=(seq_len, features))
    quantiles = QuantileHead(**flatten_quantile_head_params)(inputs)
    model = keras.Model(inputs=inputs, outputs=quantiles)

    original_output = model(small_3d_inputs, training=False)

    save_path = str(tmp_path / "quantile_head_flatten_model.keras")
    model.save(save_path)

    loaded_model = keras.models.load_model(
        save_path,
        custom_objects={"QuantileHead": QuantileHead}
    )

    loaded_output = loaded_model(small_3d_inputs, training=False)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(original_output),
        keras.ops.convert_to_numpy(loaded_output),
        rtol=1e-5, atol=1e-5,
        err_msg="Model outputs should match after save/load (flatten mode)"
    )


# Gradient flow tests
def test_quantile_head_gradient_flow(
        small_2d_inputs: tf.Tensor,
        default_quantile_head_params: Dict[str, Any]
) -> None:
    """Test gradient flow through QuantileHead."""
    head = QuantileHead(**default_quantile_head_params)

    with tf.GradientTape() as tape:
        outputs = head(small_2d_inputs, training=True)
        loss = tf.reduce_mean(outputs)

    gradients = tape.gradient(loss, head.trainable_variables)

    # Check if gradients exist for all trainable variables
    assert all(g is not None for g in gradients)

    # Check if we have trainable variables
    assert len(head.trainable_variables) > 0


def test_quantile_head_gradient_flow_with_monotonicity(small_2d_inputs: tf.Tensor) -> None:
    """Test gradient flow with monotonicity enforcement."""
    head = QuantileHead(num_quantiles=5, output_length=8, enforce_monotonicity=True)

    with tf.GradientTape() as tape:
        outputs = head(small_2d_inputs, training=True)
        loss = tf.reduce_mean(outputs)

    gradients = tape.gradient(loss, head.trainable_variables)

    # Check if gradients exist (softplus is differentiable)
    assert all(g is not None for g in gradients)
    assert len(gradients) > 0


def test_quantile_head_gradient_flow_with_flatten(small_3d_inputs: tf.Tensor) -> None:
    """Test gradient flow with flatten_input=True."""
    head = QuantileHead(num_quantiles=3, output_length=6, flatten_input=True)

    with tf.GradientTape() as tape:
        outputs = head(small_3d_inputs, training=True)
        loss = tf.reduce_mean(outputs)

    gradients = tape.gradient(loss, head.trainable_variables)

    assert all(g is not None for g in gradients)
    assert len(gradients) > 0


# Parametrized tests
@pytest.mark.parametrize("num_quantiles", [1, 3, 5, 7, 9])
def test_quantile_head_different_quantile_counts(
        num_quantiles: int,
        small_2d_inputs: tf.Tensor
) -> None:
    """Test QuantileHead with different numbers of quantiles."""
    head = QuantileHead(num_quantiles=num_quantiles, output_length=12)
    outputs = head(small_2d_inputs)

    batch_size = small_2d_inputs.shape[0]
    expected_shape = (batch_size, 12, num_quantiles)

    assert outputs.shape == expected_shape
    validate_quantile_outputs(outputs)


@pytest.mark.parametrize("output_length", [1, 6, 12, 24, 48])
def test_quantile_head_different_output_lengths(
        output_length: int,
        small_2d_inputs: tf.Tensor
) -> None:
    """Test QuantileHead with different forecast horizons."""
    head = QuantileHead(num_quantiles=3, output_length=output_length)
    outputs = head(small_2d_inputs)

    batch_size = small_2d_inputs.shape[0]
    expected_shape = (batch_size, output_length, 3)

    assert outputs.shape == expected_shape
    validate_quantile_outputs(outputs)


@pytest.mark.parametrize("dropout_rate", [0.0, 0.1, 0.3, 0.5])
def test_quantile_head_different_dropout_rates(
        dropout_rate: float,
        small_2d_inputs: tf.Tensor
) -> None:
    """Test QuantileHead with different dropout rates."""
    head = QuantileHead(num_quantiles=3, output_length=6, dropout_rate=dropout_rate)
    outputs = head(small_2d_inputs, training=False)

    # Should not raise errors and produce valid outputs
    validate_quantile_outputs(outputs)


@pytest.mark.parametrize("enforce_monotonicity", [True, False])
def test_quantile_head_monotonicity_parameter(
        enforce_monotonicity: bool,
        small_2d_inputs: tf.Tensor
) -> None:
    """Test QuantileHead with and without monotonicity."""
    head = QuantileHead(
        num_quantiles=5,
        output_length=8,
        enforce_monotonicity=enforce_monotonicity,
        dropout_rate=0.0
    )
    outputs = head(small_2d_inputs, training=False)

    validate_quantile_outputs(outputs, enforce_monotonicity=enforce_monotonicity)


@pytest.mark.parametrize("use_bias", [True, False])
def test_quantile_head_bias_parameter(
        use_bias: bool,
        small_2d_inputs: tf.Tensor
) -> None:
    """Test QuantileHead with and without bias."""
    head = QuantileHead(num_quantiles=3, output_length=6, use_bias=use_bias)
    outputs = head(small_2d_inputs)

    validate_quantile_outputs(outputs)

    # Check that bias is present or not in trainable variables
    var_names = [v.name for v in head.trainable_variables]
    has_bias = any("bias" in name for name in var_names)
    assert has_bias == use_bias


@pytest.mark.parametrize("flatten_input", [True, False])
def test_quantile_head_flatten_parameter(flatten_input: bool) -> None:
    """Test QuantileHead with flatten_input True/False."""
    head = QuantileHead(num_quantiles=3, output_length=6, flatten_input=flatten_input)

    if flatten_input:
        inputs = tf.random.normal((2, 10, 16))  # 3D
    else:
        inputs = tf.random.normal((2, 32))  # 2D

    outputs = head(inputs)
    assert outputs.shape == (2, 6, 3)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
def test_quantile_head_different_batch_sizes(batch_size: int) -> None:
    """Test QuantileHead with different batch sizes."""
    inputs = tf.random.normal((batch_size, 64))
    head = QuantileHead(num_quantiles=3, output_length=12)
    outputs = head(inputs)

    assert outputs.shape == (batch_size, 12, 3)


# Error handling tests
def test_invalid_num_quantiles():
    """Test error handling for invalid num_quantiles."""
    # Test zero quantiles
    with pytest.raises(ValueError, match="num_quantiles must be positive"):
        QuantileHead(num_quantiles=0, output_length=12)

    # Test negative quantiles
    with pytest.raises(ValueError, match="num_quantiles must be positive"):
        QuantileHead(num_quantiles=-1, output_length=12)


def test_invalid_output_length():
    """Test error handling for invalid output_length."""
    # Test zero output length
    with pytest.raises(ValueError, match="output_length must be positive"):
        QuantileHead(num_quantiles=3, output_length=0)

    # Test negative output length
    with pytest.raises(ValueError, match="output_length must be positive"):
        QuantileHead(num_quantiles=3, output_length=-1)


def test_invalid_dropout_rate():
    """Test error handling for invalid dropout_rate."""
    # Test negative dropout
    with pytest.raises(ValueError, match="dropout_rate must be between"):
        QuantileHead(num_quantiles=3, output_length=12, dropout_rate=-0.1)

    # Test dropout > 1
    with pytest.raises(ValueError, match="dropout_rate must be between"):
        QuantileHead(num_quantiles=3, output_length=12, dropout_rate=1.5)


def test_flatten_input_wrong_dimension():
    """Test error when flatten_input=True receives 2D input."""
    head = QuantileHead(num_quantiles=3, output_length=6, flatten_input=True)

    # Should raise error during build
    with pytest.raises(ValueError, match="flatten_input=True expects a 3D input"):
        head.build((4, 64))


def test_flatten_input_undefined_dimensions():
    """Test error when flatten_input=True has undefined dimensions."""
    head = QuantileHead(num_quantiles=3, output_length=6, flatten_input=True)

    # Should raise error with None dimensions
    with pytest.raises(ValueError, match="flatten_input=True requires both sequence length"):
        head.build((None, None, 32))

    with pytest.raises(ValueError, match="flatten_input=True requires both sequence length"):
        head.build((4, 10, None))


# Edge cases and special scenarios
def test_single_quantile(small_2d_inputs: tf.Tensor) -> None:
    """Test with single quantile (point prediction)."""
    head = QuantileHead(num_quantiles=1, output_length=12)
    outputs = head(small_2d_inputs)

    batch_size = small_2d_inputs.shape[0]
    assert outputs.shape == (batch_size, 12, 1)


def test_single_timestep_output(small_2d_inputs: tf.Tensor) -> None:
    """Test with single timestep output."""
    head = QuantileHead(num_quantiles=3, output_length=1)
    outputs = head(small_2d_inputs)

    batch_size = small_2d_inputs.shape[0]
    assert outputs.shape == (batch_size, 1, 3)


def test_single_quantile_with_monotonicity(small_2d_inputs: tf.Tensor) -> None:
    """Test single quantile with monotonicity (should work but not affect output)."""
    head = QuantileHead(num_quantiles=1, output_length=12, enforce_monotonicity=True)
    outputs = head(small_2d_inputs)

    # Should work without issues
    validate_quantile_outputs(outputs, enforce_monotonicity=True)


def test_very_long_output_horizon(small_2d_inputs: tf.Tensor) -> None:
    """Test with very long forecast horizon."""
    head = QuantileHead(num_quantiles=3, output_length=100)
    outputs = head(small_2d_inputs)

    batch_size = small_2d_inputs.shape[0]
    assert outputs.shape == (batch_size, 100, 3)


def test_very_high_dimensional_input() -> None:
    """Test with high-dimensional input."""
    high_dim_inputs = tf.random.normal((2, 512))
    head = QuantileHead(num_quantiles=3, output_length=12)
    outputs = head(high_dim_inputs)

    assert outputs.shape == (2, 12, 3)


def test_many_quantiles(small_2d_inputs: tf.Tensor) -> None:
    """Test with many quantiles."""
    head = QuantileHead(num_quantiles=99, output_length=12, enforce_monotonicity=True)
    outputs = head(small_2d_inputs, training=False)

    batch_size = small_2d_inputs.shape[0]
    assert outputs.shape == (batch_size, 12, 99)
    assert check_monotonicity(outputs.numpy())


def test_flatten_with_single_timestep() -> None:
    """Test flatten mode with sequence length of 1."""
    inputs = tf.random.normal((4, 1, 64))  # Seq len = 1
    head = QuantileHead(num_quantiles=3, output_length=6, flatten_input=True)
    outputs = head(inputs)

    assert outputs.shape == (4, 6, 3)


def test_flatten_with_long_sequence() -> None:
    """Test flatten mode with very long sequences."""
    inputs = tf.random.normal((2, 100, 32))  # Long sequence
    head = QuantileHead(num_quantiles=3, output_length=6, flatten_input=True)
    outputs = head(inputs)

    assert outputs.shape == (2, 6, 3)


# Numerical stability tests
def test_quantile_head_numerical_stability() -> None:
    """Test numerical stability with extreme inputs."""
    head = QuantileHead(num_quantiles=5, output_length=12, enforce_monotonicity=True)

    # Test with very large positive values
    large_inputs = tf.ones((2, 32)) * 1000.0
    large_outputs = head(large_inputs, training=False)
    validate_quantile_outputs(large_outputs, enforce_monotonicity=True)

    # Test with very large negative values
    neg_large_inputs = tf.ones((2, 32)) * -1000.0
    neg_large_outputs = head(neg_large_inputs, training=False)
    validate_quantile_outputs(neg_large_outputs, enforce_monotonicity=True)

    # Test with mixed extreme values
    mixed_inputs = tf.concat([
        tf.ones((1, 32)) * 1000.0,
        tf.ones((1, 32)) * -1000.0
    ], axis=0)
    mixed_outputs = head(mixed_inputs, training=False)
    validate_quantile_outputs(mixed_outputs, enforce_monotonicity=True)


def test_quantile_head_zero_input(small_2d_inputs: tf.Tensor) -> None:
    """Test behavior with zero input."""
    zero_inputs = tf.zeros_like(small_2d_inputs)
    head = QuantileHead(num_quantiles=3, output_length=12, enforce_monotonicity=True)
    outputs = head(zero_inputs, training=False)

    # Should produce valid outputs even with zero input
    assert outputs.shape == (zero_inputs.shape[0], 12, 3)


# Comparison tests
def test_monotonic_vs_non_monotonic_outputs(small_2d_inputs: tf.Tensor) -> None:
    """Compare outputs with and without monotonicity enforcement."""
    # Create two heads with same initialization
    tf.random.set_seed(42)
    head_mono = QuantileHead(
        num_quantiles=5,
        output_length=12,
        enforce_monotonicity=True,
        dropout_rate=0.0,
        kernel_initializer=keras.initializers.GlorotUniform(seed=42)
    )

    tf.random.set_seed(42)
    head_non_mono = QuantileHead(
        num_quantiles=5,
        output_length=12,
        enforce_monotonicity=False,
        dropout_rate=0.0,
        kernel_initializer=keras.initializers.GlorotUniform(seed=42)
    )

    # Get outputs
    outputs_mono = head_mono(small_2d_inputs, training=False)
    outputs_non_mono = head_non_mono(small_2d_inputs, training=False)

    # Monotonic output should satisfy monotonicity
    assert check_monotonicity(outputs_mono.numpy())

    # Outputs should generally be different due to transformation
    assert outputs_mono.shape == outputs_non_mono.shape


def test_flatten_vs_non_flatten_parameter_count() -> None:
    """Compare parameter counts between flatten and non-flatten modes."""
    # Non-flatten mode
    head_non_flatten = QuantileHead(
        num_quantiles=3,
        output_length=12,
        flatten_input=False
    )
    head_non_flatten.build((None, 64))  # 64 features

    # Flatten mode with equivalent total features
    head_flatten = QuantileHead(
        num_quantiles=3,
        output_length=12,
        flatten_input=True
    )
    head_flatten.build((None, 4, 16))  # 4*16 = 64 features

    # Both should have same number of output units
    params_non_flatten = head_non_flatten.count_params()
    params_flatten = head_flatten.count_params()

    # Should be identical since both project from 64 features to same output
    assert params_non_flatten == params_flatten


if __name__ == "__main__":
    pytest.main([__file__, "-v"])