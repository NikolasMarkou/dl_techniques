"""
Test suite for QuantileSequenceHead implementation.

This module provides comprehensive tests for:
- QuantileSequenceHead layer initialization
- Output shape validation
- Monotonicity enforcement
- Dropout behavior in training vs inference
- Serialization and deserialization
- Model integration and persistence
- Gradient flow
- Parameter validation
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any

from dl_techniques.layers.time_series.quantile_head_variable_io import QuantileSequenceHead


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
def sample_sequence_inputs() -> tf.Tensor:
    """Generate sample sequence input tensor."""
    tf.random.set_seed(42)
    # Shape: (batch_size, sequence_length, features)
    return tf.random.normal((4, 50, 128))  # 50 time steps, 128 features


@pytest.fixture
def small_sample_inputs() -> tf.Tensor:
    """Generate smaller sample input tensor for faster tests."""
    tf.random.set_seed(42)
    # Shape: (batch_size, sequence_length, features)
    return tf.random.normal((2, 10, 32))  # 10 time steps, 32 features


@pytest.fixture
def default_quantile_params() -> Dict[str, Any]:
    """Default parameters for QuantileSequenceHead."""
    return {
        "num_quantiles": 3,
        "dropout_rate": 0.1,
        "use_bias": True,
        "enforce_monotonicity": False,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": keras.regularizers.L2(0.001),
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
    }


@pytest.fixture
def minimal_quantile_params() -> Dict[str, Any]:
    """Minimal parameters for QuantileSequenceHead."""
    return {
        "num_quantiles": 3,
    }


@pytest.fixture
def monotonic_quantile_params() -> Dict[str, Any]:
    """Parameters for monotonic QuantileSequenceHead."""
    return {
        "num_quantiles": 5,
        "dropout_rate": 0.0,
        "enforce_monotonicity": True,
    }


# Basic initialization tests
def test_quantile_head_initialization(default_quantile_params: Dict[str, Any]) -> None:
    """Test initialization of QuantileSequenceHead."""
    head = QuantileSequenceHead(**default_quantile_params)

    assert head.num_quantiles == default_quantile_params["num_quantiles"]
    assert head.dropout_rate == default_quantile_params["dropout_rate"]
    assert head.use_bias == default_quantile_params["use_bias"]
    assert head.enforce_monotonicity == default_quantile_params["enforce_monotonicity"]


def test_quantile_head_minimal_initialization(minimal_quantile_params: Dict[str, Any]) -> None:
    """Test initialization with minimal parameters."""
    head = QuantileSequenceHead(**minimal_quantile_params)

    assert head.num_quantiles == minimal_quantile_params["num_quantiles"]
    assert head.dropout_rate == 0.1  # default
    assert head.use_bias is True  # default
    assert head.enforce_monotonicity is False  # default


def test_quantile_head_monotonic_initialization(monotonic_quantile_params: Dict[str, Any]) -> None:
    """Test initialization with monotonicity enforcement."""
    head = QuantileSequenceHead(**monotonic_quantile_params)

    assert head.num_quantiles == monotonic_quantile_params["num_quantiles"]
    assert head.enforce_monotonicity is True
    assert head.dropout_rate == 0.0


# Output shape tests
def test_quantile_head_output_shapes(
        small_sample_inputs: tf.Tensor,
        minimal_quantile_params: Dict[str, Any]
) -> None:
    """Test QuantileSequenceHead output shapes."""
    head = QuantileSequenceHead(**minimal_quantile_params)
    outputs = head(small_sample_inputs)

    batch_size, seq_len, features = small_sample_inputs.shape
    expected_shape = (batch_size, seq_len, minimal_quantile_params["num_quantiles"])

    assert outputs.shape == expected_shape
    validate_quantile_outputs(outputs, enforce_monotonicity=False)


def test_quantile_head_compute_output_shape(minimal_quantile_params: Dict[str, Any]) -> None:
    """Test QuantileSequenceHead compute_output_shape method."""
    head = QuantileSequenceHead(**minimal_quantile_params)

    input_shape = (4, 20, 64)
    output_shape = head.compute_output_shape(input_shape)

    expected_shape = (4, 20, minimal_quantile_params["num_quantiles"])
    assert output_shape == expected_shape


def test_quantile_head_sequence_preservation(
        sample_sequence_inputs: tf.Tensor,
        minimal_quantile_params: Dict[str, Any]
) -> None:
    """Test that sequence length is preserved (no temporal aggregation)."""
    head = QuantileSequenceHead(**minimal_quantile_params)
    outputs = head(sample_sequence_inputs)

    # Input and output sequence lengths should match
    assert outputs.shape[1] == sample_sequence_inputs.shape[1]


# Monotonicity tests
def test_quantile_head_monotonicity_enforcement(
        small_sample_inputs: tf.Tensor,
        monotonic_quantile_params: Dict[str, Any]
) -> None:
    """Test that monotonicity enforcement prevents quantile crossing."""
    head = QuantileSequenceHead(**monotonic_quantile_params)
    outputs = head(small_sample_inputs, training=False)

    # Validate monotonicity
    validate_quantile_outputs(outputs, enforce_monotonicity=True)

    # Convert to numpy for detailed checking
    outputs_np = outputs.numpy()
    assert check_monotonicity(outputs_np), "Quantiles are not monotonic"


def test_quantile_head_monotonicity_with_random_inputs() -> None:
    """Test monotonicity with various random inputs."""
    head = QuantileSequenceHead(num_quantiles=7, enforce_monotonicity=True)

    # Test with multiple random seeds
    for seed in range(5):
        tf.random.set_seed(seed)
        inputs = tf.random.normal((3, 15, 64))
        outputs = head(inputs, training=False)

        outputs_np = outputs.numpy()
        assert check_monotonicity(outputs_np), \
            f"Monotonicity violated with seed {seed}"


def test_quantile_head_monotonicity_extreme_values() -> None:
    """Test monotonicity with extreme input values."""
    head = QuantileSequenceHead(num_quantiles=5, enforce_monotonicity=True)

    # Test with very large values
    large_inputs = tf.random.normal((2, 10, 32)) * 100.0
    large_outputs = head(large_inputs, training=False)
    assert check_monotonicity(large_outputs.numpy())

    # Test with very small values
    small_inputs = tf.random.normal((2, 10, 32)) * 0.01
    small_outputs = head(small_inputs, training=False)
    assert check_monotonicity(small_outputs.numpy())


def test_quantile_head_without_monotonicity(small_sample_inputs: tf.Tensor) -> None:
    """Test that without monotonicity enforcement, quantiles can cross."""
    head = QuantileSequenceHead(num_quantiles=5, enforce_monotonicity=False)

    # Run multiple times to potentially observe crossing
    # (crossing is not guaranteed, but we test the mechanism works)
    outputs = head(small_sample_inputs, training=False)

    # Just verify outputs are valid (crossing is allowed)
    validate_quantile_outputs(outputs, enforce_monotonicity=False)


# Dropout tests
def test_quantile_head_dropout_training_vs_inference(
        small_sample_inputs: tf.Tensor
) -> None:
    """Test dropout behavior in training vs inference modes."""
    head = QuantileSequenceHead(num_quantiles=3, dropout_rate=0.5)

    # Training mode - run multiple times, should get different outputs
    train_outputs_1 = head(small_sample_inputs, training=True)
    train_outputs_2 = head(small_sample_inputs, training=True)

    # Outputs should differ due to dropout randomness
    assert not np.allclose(
        train_outputs_1.numpy(),
        train_outputs_2.numpy(),
        rtol=1e-5,
        atol=1e-5
    ), "Training outputs should differ due to dropout"

    # Inference mode - should be deterministic
    inference_outputs_1 = head(small_sample_inputs, training=False)
    inference_outputs_2 = head(small_sample_inputs, training=False)

    # Outputs should be identical (no dropout)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(inference_outputs_1),
        keras.ops.convert_to_numpy(inference_outputs_2),
        rtol=1e-6, atol=1e-6,
        err_msg="Inference outputs should be identical"
    )


def test_quantile_head_zero_dropout(small_sample_inputs: tf.Tensor) -> None:
    """Test that zero dropout results in identical outputs."""
    head = QuantileSequenceHead(num_quantiles=3, dropout_rate=0.0)

    # Training and inference should give same results with no dropout
    train_output = head(small_sample_inputs, training=True)
    inference_output = head(small_sample_inputs, training=False)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(train_output),
        keras.ops.convert_to_numpy(inference_output),
        rtol=1e-6, atol=1e-6,
        err_msg="Zero dropout should produce identical outputs"
    )


# Serialization tests
def test_quantile_head_serialization(
        default_quantile_params: Dict[str, Any],
        small_sample_inputs: tf.Tensor
) -> None:
    """Test serialization of QuantileSequenceHead."""
    # Create and build original head
    original_head = QuantileSequenceHead(**default_quantile_params)
    original_head.build(small_sample_inputs.shape)

    # Get config and recreate from config
    config = original_head.get_config()
    restored_head = QuantileSequenceHead.from_config(config)

    # Check key properties match
    assert restored_head.num_quantiles == original_head.num_quantiles
    assert restored_head.dropout_rate == original_head.dropout_rate
    assert restored_head.use_bias == original_head.use_bias
    assert restored_head.enforce_monotonicity == original_head.enforce_monotonicity


def test_quantile_head_get_config(default_quantile_params: Dict[str, Any]) -> None:
    """Test that get_config returns all necessary parameters."""
    head = QuantileSequenceHead(**default_quantile_params)
    config = head.get_config()

    # Check that all important parameters are in config
    assert "num_quantiles" in config
    assert "dropout_rate" in config
    assert "use_bias" in config
    assert "enforce_monotonicity" in config
    assert "kernel_initializer" in config
    assert "bias_initializer" in config
    assert "kernel_regularizer" in config


# Model integration tests
def test_quantile_head_model_integration(
        small_sample_inputs: tf.Tensor,
        minimal_quantile_params: Dict[str, Any]
) -> None:
    """Test integrating QuantileSequenceHead into a Keras model."""
    batch_size, seq_len, features = small_sample_inputs.shape

    inputs = keras.Input(shape=(seq_len, features))
    head = QuantileSequenceHead(**minimal_quantile_params)
    outputs = head(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(small_sample_inputs)

    expected_shape = (batch_size, seq_len, minimal_quantile_params["num_quantiles"])
    assert result.shape == expected_shape


def test_quantile_head_in_sequential_model(small_sample_inputs: tf.Tensor) -> None:
    """Test QuantileSequenceHead in a Sequential model."""
    batch_size, seq_len, features = small_sample_inputs.shape

    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, features)),
        keras.layers.Dense(64, activation="relu"),
        QuantileSequenceHead(num_quantiles=3)
    ])

    outputs = model(small_sample_inputs)
    assert outputs.shape == (batch_size, seq_len, 3)


def test_quantile_head_with_encoder(small_sample_inputs: tf.Tensor) -> None:
    """Test QuantileSequenceHead with an upstream encoder."""
    batch_size, seq_len, features = small_sample_inputs.shape

    # Build a simple encoder-decoder style model
    inputs = keras.Input(shape=(seq_len, features))

    # Simple encoder
    x = keras.layers.Dense(128, activation="relu")(inputs)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dense(64, activation="relu")(x)

    # Quantile head
    quantiles = QuantileSequenceHead(num_quantiles=5, enforce_monotonicity=True)(x)

    model = keras.Model(inputs=inputs, outputs=quantiles)

    outputs = model(small_sample_inputs)
    assert outputs.shape == (batch_size, seq_len, 5)

    # Verify monotonicity
    assert check_monotonicity(outputs.numpy())


def test_model_compilation_and_training(
        small_sample_inputs: tf.Tensor,
        minimal_quantile_params: Dict[str, Any]
) -> None:
    """Test compiling and training a model with QuantileSequenceHead."""
    batch_size, seq_len, features = small_sample_inputs.shape

    inputs = keras.Input(shape=(seq_len, features))
    quantiles = QuantileSequenceHead(**minimal_quantile_params)(inputs)

    model = keras.Model(inputs=inputs, outputs=quantiles)
    model.compile(optimizer="adam", loss="mse")

    # Create dummy targets matching output shape
    targets = tf.random.normal((batch_size, seq_len, minimal_quantile_params["num_quantiles"]))

    # Test training for one step
    history = model.fit(small_sample_inputs, targets, epochs=1, verbose=0)
    assert len(history.history["loss"]) == 1


def test_model_save_load_keras_format(
        small_sample_inputs: tf.Tensor,
        minimal_quantile_params: Dict[str, Any],
        tmp_path
) -> None:
    """Test saving and loading a model with QuantileSequenceHead in Keras format."""
    batch_size, seq_len, features = small_sample_inputs.shape

    # Create a model with the quantile head
    inputs = keras.Input(shape=(seq_len, features))
    quantiles = QuantileSequenceHead(**minimal_quantile_params)(inputs)
    model = keras.Model(inputs=inputs, outputs=quantiles)

    # Generate output before saving
    original_output = model(small_sample_inputs, training=False)

    # Save model in Keras format
    save_path = str(tmp_path / "quantile_head_model.keras")
    model.save(save_path)

    # Load model
    loaded_model = keras.models.load_model(
        save_path,
        custom_objects={"QuantileSequenceHead": QuantileSequenceHead}
    )

    # Generate output after loading
    loaded_output = loaded_model(small_sample_inputs, training=False)

    # Outputs should be identical
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(original_output),
        keras.ops.convert_to_numpy(loaded_output),
        rtol=1e-5, atol=1e-5,
        err_msg="Model outputs should match after save/load"
    )


# Gradient flow tests
def test_quantile_head_gradient_flow(
        small_sample_inputs: tf.Tensor,
        default_quantile_params: Dict[str, Any]
) -> None:
    """Test gradient flow through QuantileSequenceHead."""
    head = QuantileSequenceHead(**default_quantile_params)

    with tf.GradientTape() as tape:
        outputs = head(small_sample_inputs, training=True)
        loss = tf.reduce_mean(outputs)

    gradients = tape.gradient(loss, head.trainable_variables)

    # Check if gradients exist for all trainable variables
    assert all(g is not None for g in gradients)

    # Check if we have trainable variables
    assert len(head.trainable_variables) > 0


def test_quantile_head_gradient_flow_with_monotonicity(
        small_sample_inputs: tf.Tensor
) -> None:
    """Test gradient flow with monotonicity enforcement."""
    head = QuantileSequenceHead(num_quantiles=5, enforce_monotonicity=True)

    with tf.GradientTape() as tape:
        outputs = head(small_sample_inputs, training=True)
        loss = tf.reduce_mean(outputs)

    gradients = tape.gradient(loss, head.trainable_variables)

    # Check if gradients exist (softplus is differentiable)
    assert all(g is not None for g in gradients)
    assert len(gradients) > 0


# Parametrized tests
@pytest.mark.parametrize("num_quantiles", [1, 3, 5, 7, 9])
def test_quantile_head_different_quantile_counts(
        num_quantiles: int,
        small_sample_inputs: tf.Tensor
) -> None:
    """Test QuantileSequenceHead with different numbers of quantiles."""
    head = QuantileSequenceHead(num_quantiles=num_quantiles)
    outputs = head(small_sample_inputs)

    batch_size, seq_len, features = small_sample_inputs.shape
    expected_shape = (batch_size, seq_len, num_quantiles)

    assert outputs.shape == expected_shape
    validate_quantile_outputs(outputs)


@pytest.mark.parametrize("dropout_rate", [0.0, 0.1, 0.3, 0.5])
def test_quantile_head_different_dropout_rates(
        dropout_rate: float,
        small_sample_inputs: tf.Tensor
) -> None:
    """Test QuantileSequenceHead with different dropout rates."""
    head = QuantileSequenceHead(num_quantiles=3, dropout_rate=dropout_rate)
    outputs = head(small_sample_inputs, training=False)

    # Should not raise errors and produce valid outputs
    validate_quantile_outputs(outputs)


@pytest.mark.parametrize("enforce_monotonicity", [True, False])
def test_quantile_head_monotonicity_parameter(
        enforce_monotonicity: bool,
        small_sample_inputs: tf.Tensor
) -> None:
    """Test QuantileSequenceHead with and without monotonicity."""
    head = QuantileSequenceHead(
        num_quantiles=5,
        enforce_monotonicity=enforce_monotonicity,
        dropout_rate=0.0
    )
    outputs = head(small_sample_inputs, training=False)

    validate_quantile_outputs(outputs, enforce_monotonicity=enforce_monotonicity)


@pytest.mark.parametrize("use_bias", [True, False])
def test_quantile_head_bias_parameter(
        use_bias: bool,
        small_sample_inputs: tf.Tensor
) -> None:
    """Test QuantileSequenceHead with and without bias."""
    head = QuantileSequenceHead(num_quantiles=3, use_bias=use_bias)
    outputs = head(small_sample_inputs)

    validate_quantile_outputs(outputs)

    # Check that bias is present or not in trainable variables
    var_names = [v.name for v in head.trainable_variables]
    has_bias = any("bias" in name for name in var_names)
    assert has_bias == use_bias


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
def test_quantile_head_different_batch_sizes(batch_size: int) -> None:
    """Test QuantileSequenceHead with different batch sizes."""
    inputs = tf.random.normal((batch_size, 20, 64))
    head = QuantileSequenceHead(num_quantiles=3)
    outputs = head(inputs)

    assert outputs.shape == (batch_size, 20, 3)


@pytest.mark.parametrize("seq_len", [5, 10, 50, 100])
def test_quantile_head_different_sequence_lengths(seq_len: int) -> None:
    """Test QuantileSequenceHead with different sequence lengths."""
    inputs = tf.random.normal((4, seq_len, 32))
    head = QuantileSequenceHead(num_quantiles=3)
    outputs = head(inputs)

    assert outputs.shape == (4, seq_len, 3)


# Error handling tests
def test_invalid_num_quantiles():
    """Test error handling for invalid num_quantiles."""
    # Test zero quantiles
    with pytest.raises(ValueError, match="num_quantiles must be positive"):
        QuantileSequenceHead(num_quantiles=0)

    # Test negative quantiles
    with pytest.raises(ValueError, match="num_quantiles must be positive"):
        QuantileSequenceHead(num_quantiles=-1)


def test_invalid_dropout_rate():
    """Test error handling for invalid dropout_rate."""
    # Test negative dropout
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        QuantileSequenceHead(num_quantiles=3, dropout_rate=-0.1)

    # Test dropout > 1
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        QuantileSequenceHead(num_quantiles=3, dropout_rate=1.5)


def test_invalid_input_shape():
    """Test error handling for invalid input shapes."""
    head = QuantileSequenceHead(num_quantiles=3)

    # Test with 2D input (missing sequence dimension)
    with pytest.raises(ValueError, match="Expected 3D input shape"):
        head.compute_output_shape((4, 64))

    # Test with 4D input
    with pytest.raises(ValueError, match="Expected 3D input shape"):
        head.compute_output_shape((4, 10, 64, 3))


def test_undefined_feature_dimension():
    """Test error handling when feature dimension is undefined."""
    head = QuantileSequenceHead(num_quantiles=3)

    # Test with None feature dimension
    with pytest.raises(ValueError, match="feature dimension must be defined"):
        head.build((None, 10, None))


# Edge cases and special scenarios
def test_single_quantile(small_sample_inputs: tf.Tensor) -> None:
    """Test with single quantile (point prediction)."""
    head = QuantileSequenceHead(num_quantiles=1)
    outputs = head(small_sample_inputs)

    batch_size, seq_len, features = small_sample_inputs.shape
    assert outputs.shape == (batch_size, seq_len, 1)


def test_single_quantile_with_monotonicity(small_sample_inputs: tf.Tensor) -> None:
    """Test single quantile with monotonicity (should work but not affect output)."""
    head = QuantileSequenceHead(num_quantiles=1, enforce_monotonicity=True)
    outputs = head(small_sample_inputs)

    # Should work without issues
    validate_quantile_outputs(outputs, enforce_monotonicity=True)


def test_very_long_sequence() -> None:
    """Test with very long sequences."""
    long_inputs = tf.random.normal((2, 1000, 32))
    head = QuantileSequenceHead(num_quantiles=3)
    outputs = head(long_inputs)

    assert outputs.shape == (2, 1000, 3)


def test_very_high_dimensional_features() -> None:
    """Test with high-dimensional feature space."""
    high_dim_inputs = tf.random.normal((2, 10, 512))
    head = QuantileSequenceHead(num_quantiles=3)
    outputs = head(high_dim_inputs)

    assert outputs.shape == (2, 10, 3)


def test_many_quantiles() -> None:
    """Test with many quantiles."""
    inputs = tf.random.normal((2, 10, 32))
    head = QuantileSequenceHead(num_quantiles=99, enforce_monotonicity=True)
    outputs = head(inputs, training=False)

    assert outputs.shape == (2, 10, 99)
    assert check_monotonicity(outputs.numpy())


# Regularization tests
def test_quantile_head_with_kernel_regularizer(small_sample_inputs: tf.Tensor) -> None:
    """Test QuantileSequenceHead with kernel regularization."""
    head = QuantileSequenceHead(
        num_quantiles=3,
        kernel_regularizer=keras.regularizers.L2(0.01)
    )

    outputs = head(small_sample_inputs, training=True)

    # Check that regularization losses are added
    assert len(head.losses) > 0


def test_quantile_head_with_activity_regularizer(small_sample_inputs: tf.Tensor) -> None:
    """Test QuantileSequenceHead with activity regularization."""
    head = QuantileSequenceHead(
        num_quantiles=3,
        activity_regularizer=keras.regularizers.L1(0.01)
    )

    outputs = head(small_sample_inputs, training=True)

    # Check that regularization losses are added
    assert len(head.losses) > 0


# Constraint tests
def test_quantile_head_with_kernel_constraint(small_sample_inputs: tf.Tensor) -> None:
    """Test QuantileSequenceHead with kernel constraints."""
    head = QuantileSequenceHead(
        num_quantiles=3,
        kernel_constraint=keras.constraints.MaxNorm(2.0)
    )

    outputs = head(small_sample_inputs)

    # Should not raise errors
    validate_quantile_outputs(outputs)


# Numerical stability tests
def test_quantile_head_numerical_stability() -> None:
    """Test numerical stability with extreme inputs."""
    head = QuantileSequenceHead(num_quantiles=5, enforce_monotonicity=True)

    # Test with very large positive values
    large_inputs = tf.ones((2, 10, 32)) * 1000.0
    large_outputs = head(large_inputs, training=False)
    validate_quantile_outputs(large_outputs, enforce_monotonicity=True)

    # Test with very large negative values
    neg_large_inputs = tf.ones((2, 10, 32)) * -1000.0
    neg_large_outputs = head(neg_large_inputs, training=False)
    validate_quantile_outputs(neg_large_outputs, enforce_monotonicity=True)

    # Test with mixed extreme values
    mixed_inputs = tf.concat([
        tf.ones((1, 10, 32)) * 1000.0,
        tf.ones((1, 10, 32)) * -1000.0
    ], axis=0)
    mixed_outputs = head(mixed_inputs, training=False)
    validate_quantile_outputs(mixed_outputs, enforce_monotonicity=True)


def test_quantile_head_zero_input(small_sample_inputs: tf.Tensor) -> None:
    """Test behavior with zero input."""
    zero_inputs = tf.zeros_like(small_sample_inputs)
    head = QuantileSequenceHead(num_quantiles=3, enforce_monotonicity=True)
    outputs = head(zero_inputs, training=False)

    # Should produce valid outputs even with zero input
    assert outputs.shape == (
        zero_inputs.shape[0],
        zero_inputs.shape[1],
        3
    )


# Comparison tests
def test_monotonic_vs_non_monotonic_outputs(small_sample_inputs: tf.Tensor) -> None:
    """Compare outputs with and without monotonicity enforcement."""
    # Create two heads with same initialization
    tf.random.set_seed(42)
    head_mono = QuantileSequenceHead(
        num_quantiles=5,
        enforce_monotonicity=True,
        dropout_rate=0.0,
        kernel_initializer=keras.initializers.GlorotUniform(seed=42)
    )

    tf.random.set_seed(42)
    head_non_mono = QuantileSequenceHead(
        num_quantiles=5,
        enforce_monotonicity=False,
        dropout_rate=0.0,
        kernel_initializer=keras.initializers.GlorotUniform(seed=42)
    )

    # Get outputs
    outputs_mono = head_mono(small_sample_inputs, training=False)
    outputs_non_mono = head_non_mono(small_sample_inputs, training=False)

    # Monotonic output should satisfy monotonicity
    assert check_monotonicity(outputs_mono.numpy())

    # Outputs should generally be different due to transformation
    # (unless by chance the raw outputs are already monotonic)
    assert outputs_mono.shape == outputs_non_mono.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])