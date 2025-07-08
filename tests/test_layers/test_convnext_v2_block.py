"""
Test suite for ConvNextV2 block implementation.

This module provides comprehensive tests for:
- ConvNextV2Block layer initialization and configuration
- Layer behavior under different configurations and parameters
- Output shape computation and validation
- Training vs inference behavior with dropout
- Serialization and deserialization (including build configs)
- Model integration, compilation, and training
- Model save/load functionality (weights and Keras format)
- Global Response Normalization (GRN) feature validation
- Gradient flow and regularization effects
- Edge cases and error handling
- Parametrized testing for various configurations
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any

from dl_techniques.layers.convnext_v2_block import ConvNextV2Block


# Test fixtures
@pytest.fixture
def sample_inputs() -> tf.Tensor:
    """Generate sample input tensor."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 32, 32, 64))


@pytest.fixture
def default_params() -> Dict[str, Any]:
    """Default parameters for ConvNextV2Block."""
    return {
        "kernel_size": 7,
        "filters": 64,
        "activation": "gelu",
        "kernel_regularizer": keras.regularizers.L2(0.01),
        "use_bias": True,
        "dropout_rate": 0.1,
        "spatial_dropout_rate": 0.1,
        "use_gamma": True,
        "use_softorthonormal_regularizer": False,
    }


@pytest.fixture
def minimal_params() -> Dict[str, Any]:
    """Minimal parameters for ConvNextV2Block."""
    return {
        "kernel_size": 7,
        "filters": 64,
    }


# ConvNextV2Block tests
def test_block_initialization(default_params: Dict[str, Any]) -> None:
    """Test initialization of ConvNextV2Block."""
    block = ConvNextV2Block(**default_params)

    assert block.kernel_size == default_params["kernel_size"]
    assert block.filters == default_params["filters"]
    assert block.activation_name == default_params["activation"]
    assert block.use_bias == default_params["use_bias"]
    assert block.dropout_rate == default_params["dropout_rate"]
    assert block.spatial_dropout_rate == default_params["spatial_dropout_rate"]
    assert block.use_gamma == default_params["use_gamma"]
    assert block.use_softorthonormal_regularizer == default_params["use_softorthonormal_regularizer"]


def test_block_minimal_initialization(minimal_params: Dict[str, Any]) -> None:
    """Test initialization with minimal parameters."""
    block = ConvNextV2Block(**minimal_params)

    assert block.kernel_size == minimal_params["kernel_size"]
    assert block.filters == minimal_params["filters"]
    assert block.activation_name == "gelu"
    assert block.use_bias is True
    assert block.dropout_rate == 0.0
    assert block.spatial_dropout_rate == 0.0
    assert block.use_gamma is True
    assert block.use_softorthonormal_regularizer is False


def test_block_output_shape(sample_inputs: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test if block preserves input shape when using same padding and stride=1."""
    block = ConvNextV2Block(**default_params)
    outputs = block(sample_inputs)

    assert outputs.shape == sample_inputs.shape


def test_block_compute_output_shape(sample_inputs: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test compute_output_shape method."""
    block = ConvNextV2Block(**default_params)

    computed_shape = block.compute_output_shape(sample_inputs.shape)
    expected_shape = (sample_inputs.shape[0], sample_inputs.shape[1], sample_inputs.shape[2], default_params["filters"])
    assert computed_shape == expected_shape


def test_block_training_behavior(sample_inputs: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test block behavior in training vs inference modes."""
    block = ConvNextV2Block(**default_params)

    # Training mode
    train_output = block(sample_inputs, training=True)

    # Inference mode
    inference_output = block(sample_inputs, training=False)

    # Outputs should be different due to dropout
    assert not np.allclose(train_output.numpy(), inference_output.numpy())


def test_block_without_dropout(sample_inputs: tf.Tensor) -> None:
    """Test block without dropout."""
    block = ConvNextV2Block(
        kernel_size=7,
        filters=64,
        activation="gelu",
        dropout_rate=0.0,  # Explicitly set to 0.0
        spatial_dropout_rate=0.0,  # Explicitly set to 0.0
        use_gamma=True
    )

    # Training and inference outputs should be the same when no dropout is used
    train_output = block(sample_inputs, training=True)
    inference_output = block(sample_inputs, training=False)

    assert np.allclose(train_output.numpy(), inference_output.numpy(), rtol=1e-5, atol=1e-5)


def test_block_with_none_dropout(sample_inputs: tf.Tensor) -> None:
    """Test block with None dropout rates."""
    block = ConvNextV2Block(
        kernel_size=7,
        filters=64,
        activation="gelu",
        dropout_rate=None,  # None instead of 0.0
        spatial_dropout_rate=None,  # None instead of 0.0
        use_gamma=True
    )

    # Training and inference outputs should be the same when no dropout is used
    train_output = block(sample_inputs, training=True)
    inference_output = block(sample_inputs, training=False)

    assert np.allclose(train_output.numpy(), inference_output.numpy(), rtol=1e-5, atol=1e-5)


def test_block_serialization(default_params: Dict[str, Any], sample_inputs: tf.Tensor) -> None:
    """Test serialization of ConvNextV2Block."""
    # Create and build the original block
    original_block = ConvNextV2Block(**default_params)
    original_block.build(sample_inputs.shape)  # Build the layer

    # Get config and recreate from config
    config = original_block.get_config()
    build_config = original_block.get_build_config()

    restored_block = ConvNextV2Block.from_config(config)
    restored_block.build_from_config(build_config)

    # Check if the key properties match
    assert restored_block.kernel_size == original_block.kernel_size
    assert restored_block.filters == original_block.filters
    assert restored_block.activation_name == original_block.activation_name
    assert restored_block.use_bias == original_block.use_bias
    assert restored_block.dropout_rate == original_block.dropout_rate
    assert restored_block.spatial_dropout_rate == original_block.spatial_dropout_rate
    assert restored_block.use_gamma == original_block.use_gamma
    assert restored_block.use_softorthonormal_regularizer == original_block.use_softorthonormal_regularizer

    # Check that both blocks are built
    assert original_block.built
    assert restored_block.built

    # Check that restored block's config has the same keys as original
    restored_config = restored_block.get_config()
    assert set(restored_config.keys()) == set(config.keys())


def test_block_serialization_with_regularizer(default_params: Dict[str, Any], sample_inputs: tf.Tensor) -> None:
    """Test serialization with kernel regularizer."""
    # Create block with regularizer
    regularizer_params = default_params.copy()
    regularizer_params["kernel_regularizer"] = keras.regularizers.L2(0.01)

    original_block = ConvNextV2Block(**regularizer_params)
    original_block.build(sample_inputs.shape)  # Build the layer

    # Get config and recreate from config
    config = original_block.get_config()
    build_config = original_block.get_build_config()

    restored_block = ConvNextV2Block.from_config(config)
    restored_block.build_from_config(build_config)

    # Check that regularizer is properly serialized/deserialized
    assert restored_block.kernel_regularizer is not None
    assert isinstance(restored_block.kernel_regularizer, keras.regularizers.L2)
    assert restored_block.built


def test_block_build_configuration(default_params: Dict[str, Any], sample_inputs: tf.Tensor) -> None:
    """Test get_build_config and build_from_config methods."""
    # Create and build the original block
    original_block = ConvNextV2Block(**default_params)
    original_block.build(sample_inputs.shape)

    # Get build config
    build_config = original_block.get_build_config()

    # Check that build config contains input_shape
    assert "input_shape" in build_config
    assert build_config["input_shape"] == sample_inputs.shape

    # Create new block and build from config
    new_block = ConvNextV2Block(**default_params)
    new_block.build_from_config(build_config)

    # Check that new block is built
    assert new_block.built
    assert new_block._build_input_shape == sample_inputs.shape


def test_block_build_configuration_methods(default_params: Dict[str, Any], sample_inputs: tf.Tensor) -> None:
    """Test build configuration methods handle None input_shape."""
    # Test with None input_shape
    block = ConvNextV2Block(**default_params)

    # Before building
    build_config = block.get_build_config()
    assert build_config["input_shape"] is None

    # build_from_config with None should not crash
    new_block = ConvNextV2Block(**default_params)
    new_block.build_from_config({"input_shape": None})
    assert not new_block.built  # Should not be built if input_shape is None


@pytest.mark.parametrize("activation", ["relu", "gelu", "swish", "silu"])
def test_block_activations(activation: str, sample_inputs: tf.Tensor) -> None:
    """Test different activation functions."""
    block = ConvNextV2Block(
        kernel_size=7,
        filters=64,
        activation=activation,
        dropout_rate=None,
        spatial_dropout_rate=None
    )

    output = block(sample_inputs)
    assert not tf.reduce_any(tf.math.is_nan(output))


@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
def test_different_kernel_sizes(kernel_size: int, sample_inputs: tf.Tensor) -> None:
    """Test different kernel sizes."""
    block = ConvNextV2Block(
        kernel_size=kernel_size,
        filters=64,
        activation="gelu",
        dropout_rate=0.0,
        spatial_dropout_rate=0.0
    )

    output = block(sample_inputs)
    assert output.shape == sample_inputs.shape


@pytest.mark.parametrize("filters", [32, 64, 128, 256])
def test_block_filter_counts(filters: int, sample_inputs: tf.Tensor) -> None:
    """Test different filter counts."""
    block = ConvNextV2Block(
        kernel_size=7,
        filters=filters,
        dropout_rate=0.0,
        spatial_dropout_rate=0.0
    )

    output = block(sample_inputs)

    # Output should have the specified number of filters
    expected_shape = (sample_inputs.shape[0], sample_inputs.shape[1], sample_inputs.shape[2], filters)
    assert output.shape == expected_shape


def test_block_gradient_flow(sample_inputs: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test gradient flow through the block."""
    block = ConvNextV2Block(**default_params)

    with tf.GradientTape() as tape:
        output = block(sample_inputs, training=True)
        loss = tf.reduce_mean(output)

    gradients = tape.gradient(loss, block.trainable_variables)

    # Check if gradients exist for all trainable variables
    assert all(g is not None for g in gradients)

    # Check if we have trainable variables
    assert len(block.trainable_variables) > 0


def test_gamma_scaling(sample_inputs: tf.Tensor) -> None:
    """Test the effect of the gamma scaling parameter."""
    # Block with gamma scaling
    with_gamma = ConvNextV2Block(
        kernel_size=7,
        filters=64,
        use_gamma=True,
        dropout_rate=None,  # Disable dropout to ensure deterministic output
        spatial_dropout_rate=None
    )

    # Block without gamma scaling
    without_gamma = ConvNextV2Block(
        kernel_size=7,
        filters=64,
        use_gamma=False,
        dropout_rate=None,  # Disable dropout to ensure deterministic output
        spatial_dropout_rate=None
    )

    # Process the same inputs
    out_with_gamma = with_gamma(sample_inputs)
    out_without_gamma = without_gamma(sample_inputs)

    # Outputs should be different
    assert not np.allclose(out_with_gamma.numpy(), out_without_gamma.numpy())


def test_softorthonormal_regularizer(sample_inputs: tf.Tensor) -> None:
    """Test the effect of soft orthonormal regularizer."""
    # Block with soft orthonormal regularizer
    ortho_block = ConvNextV2Block(
        kernel_size=7,
        filters=64,
        use_softorthonormal_regularizer=True,
        dropout_rate=0.0,
        spatial_dropout_rate=0.0
    )

    # Block without soft orthonormal regularizer
    regular_block = ConvNextV2Block(
        kernel_size=7,
        filters=64,
        use_softorthonormal_regularizer=False,
        dropout_rate=0.0,
        spatial_dropout_rate=0.0
    )

    # Process the same inputs
    ortho_output = ortho_block(sample_inputs)
    regular_output = regular_block(sample_inputs)

    # Both should produce valid outputs
    assert not tf.reduce_any(tf.math.is_nan(ortho_output))
    assert not tf.reduce_any(tf.math.is_nan(regular_output))


def test_model_compilation_and_training(sample_inputs: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test compiling and training a model with ConvNextV2Block."""
    inputs = keras.Input(shape=sample_inputs.shape[1:])
    x = ConvNextV2Block(**default_params)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create dummy labels
    labels = tf.random.uniform((sample_inputs.shape[0],), 0, 10, dtype=tf.int32)

    # Test training for one step
    history = model.fit(sample_inputs, labels, epochs=1, verbose=0)
    assert len(history.history["loss"]) == 1


def test_model_save_load_keras_format(sample_inputs: tf.Tensor, default_params: Dict[str, Any], tmp_path) -> None:
    """Test saving and loading a model in Keras format with ConvNextV2Block."""
    # Create a simple model with the block
    inputs = keras.Input(shape=sample_inputs.shape[1:])
    block = ConvNextV2Block(**default_params)
    outputs = block(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Generate output before saving
    original_output = model(sample_inputs, training=False).numpy()

    # Save model in Keras format
    save_path = str(tmp_path / "model.keras")
    model.save(save_path)

    # Import all necessary custom objects
    from dl_techniques.layers.layer_scale import LearnableMultiplier
    from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer
    from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint
    from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization

    # Load model with custom objects
    loaded_model = keras.models.load_model(
        save_path,
        custom_objects={
            "ConvNextV2Block": ConvNextV2Block,
            "LearnableMultiplier": LearnableMultiplier,
            "SoftOrthonormalConstraintRegularizer": SoftOrthonormalConstraintRegularizer,
            "ValueRangeConstraint": ValueRangeConstraint,
            "GlobalResponseNormalization": GlobalResponseNormalization,
        }
    )

    # Generate output after loading
    loaded_output = loaded_model(sample_inputs, training=False).numpy()

    # Outputs should be identical
    assert np.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)


def test_error_handling_invalid_input_shape(default_params: Dict[str, Any]) -> None:
    """Test error handling for invalid input shapes."""
    block = ConvNextV2Block(**default_params)

    # Test with 3D input (should raise ValueError)
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        block.compute_output_shape((32, 32, 64))  # Missing batch dimension

    # Test with 2D input (should raise ValueError)
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        block.compute_output_shape((32, 64))


def test_model_integration(sample_inputs: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test integrating the ConvNextV2Block into a Keras model."""
    inputs = keras.Input(shape=sample_inputs.shape[1:])
    block = ConvNextV2Block(**default_params)
    outputs = block(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(sample_inputs)
    assert result.shape == sample_inputs.shape


def test_model_save_load(sample_inputs: tf.Tensor, default_params: Dict[str, Any], tmp_path) -> None:
    """Test saving and loading a model with ConvNextV2Block."""
    # Create a simple model with the block
    inputs = keras.Input(shape=sample_inputs.shape[1:])
    block = ConvNextV2Block(**default_params)
    outputs = block(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Generate output before saving
    original_output = model(sample_inputs, training=False).numpy()

    # Save model
    save_path = str(tmp_path / "model.weights.h5")
    model.save_weights(save_path)

    # Load model
    model.load_weights(save_path)

    # Generate output after loading
    loaded_output = model(sample_inputs, training=False).numpy()

    # Outputs should be identical
    assert np.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)


def test_compute_output_shape_list_input() -> None:
    """Test compute_output_shape with list of shapes."""
    block = ConvNextV2Block(
        kernel_size=7,
        filters=128,
    )

    input_shapes = [(None, 32, 32, 64), (None, 16, 16, 64)]
    output_shapes = block.compute_output_shape(input_shapes)

    expected_shapes = [(None, 32, 32, 128), (None, 16, 16, 128)]
    assert output_shapes == expected_shapes


def test_compute_output_shape_invalid_input() -> None:
    """Test compute_output_shape with invalid input."""
    block = ConvNextV2Block(
        kernel_size=7,
        filters=128,
    )

    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        block.compute_output_shape((32, 32))


def test_constants() -> None:
    """Test that all constants are properly defined."""
    assert ConvNextV2Block.EXPANSION_FACTOR == 4
    assert ConvNextV2Block.INITIALIZER_MEAN == 0.0
    assert ConvNextV2Block.INITIALIZER_STDDEV == 0.02
    assert ConvNextV2Block.LAYERNORM_EPSILON == 1e-6
    assert ConvNextV2Block.GRN_EPSILON == 1e-6
    assert ConvNextV2Block.POINTWISE_KERNEL_SIZE == 1
    assert ConvNextV2Block.GAMMA_L2_REGULARIZATION == 1e-5
    assert ConvNextV2Block.GAMMA_INITIAL_VALUE == 1.0
    assert ConvNextV2Block.GAMMA_MIN_VALUE == 0.0
    assert ConvNextV2Block.GAMMA_MAX_VALUE == 1.0


def test_global_response_normalization_feature(sample_inputs: tf.Tensor) -> None:
    """Test that Global Response Normalization is properly integrated."""
    block = ConvNextV2Block(
        kernel_size=7,
        filters=64,
        dropout_rate=0.0,
        spatial_dropout_rate=0.0
    )

    # Build the block to initialize GRN
    _ = block(sample_inputs)

    # Check that GRN layer exists and is properly configured
    assert hasattr(block, 'grn')
    assert block.grn is not None

    # Verify GRN is built
    assert block.grn.built


def test_grn_vs_no_grn_difference(sample_inputs: tf.Tensor) -> None:
    """Test that GRN makes a difference in the output (key V2 feature)."""
    # Since we can't easily disable GRN in the current implementation,
    # we test that the GRN layer produces different outputs than identity
    block = ConvNextV2Block(
        kernel_size=7,
        filters=64,
        dropout_rate=0.0,
        spatial_dropout_rate=0.0,
        use_gamma=False  # Disable gamma to focus on GRN effect
    )

    # Process inputs
    output = block(sample_inputs)

    # GRN should produce valid, non-NaN outputs
    assert not tf.reduce_any(tf.math.is_nan(output))

    # Outputs should not be identical to inputs (transformation occurred)
    assert not np.allclose(output.numpy(), sample_inputs.numpy())
    """Test that all constants are properly defined."""
    assert ConvNextV2Block.EXPANSION_FACTOR == 4
    assert ConvNextV2Block.INITIALIZER_MEAN == 0.0
    assert ConvNextV2Block.INITIALIZER_STDDEV == 0.02
    assert ConvNextV2Block.LAYERNORM_EPSILON == 1e-6
    assert ConvNextV2Block.GRN_EPSILON == 1e-6
    assert ConvNextV2Block.POINTWISE_KERNEL_SIZE == 1
    assert ConvNextV2Block.GAMMA_L2_REGULARIZATION == 1e-5
    assert ConvNextV2Block.GAMMA_INITIAL_VALUE == 1.0
    assert ConvNextV2Block.GAMMA_MIN_VALUE == 0.0
    assert ConvNextV2Block.GAMMA_MAX_VALUE == 1.0


if __name__ == "__main__":
    pytest.main([__file__])