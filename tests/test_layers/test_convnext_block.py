"""
Test suite for ConvNext block implementation.

This module provides comprehensive tests for:
- ConvNextBlock layer
- ConvNextConfig dataclass
- Layer behavior under different configurations
- Serialization and deserialization
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any, Optional

from dl_techniques.layers.convnext_block import (
    ConvNextBlock,
    ConvNextConfig
)


# Test fixtures
@pytest.fixture
def sample_inputs() -> tf.Tensor:
    """Generate sample input tensor."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 32, 32, 64))


@pytest.fixture
def block_config() -> ConvNextConfig:
    """Default configuration for ConvNextBlock."""
    return ConvNextConfig(
        kernel_size=7,
        filters=64,
        strides=(1, 1),
        activation="gelu",
        kernel_regularizer="l2",
        use_bias=True
    )


@pytest.fixture
def block_params() -> Dict[str, Any]:
    """Default parameters for ConvNextBlock."""
    return {
        "dropout_rate": 0.1,
        "spatial_dropout_rate": 0.1,
        "use_gamma": True,
    }


# ConvNextBlock tests
def test_block_initialization(block_config: ConvNextConfig, block_params: Dict[str, Any]) -> None:
    """Test initialization of ConvNextBlock."""
    block = ConvNextBlock(conv_config=block_config, **block_params)
    assert block.conv_config.kernel_size == block_config.kernel_size
    assert block.conv_config.filters == block_config.filters
    assert block.dropout_rate == block_params["dropout_rate"]
    assert block.spatial_dropout_rate == block_params["spatial_dropout_rate"]
    assert block.use_gamma == block_params["use_gamma"]


def test_block_output_shape(sample_inputs: tf.Tensor, block_config: ConvNextConfig,
                            block_params: Dict[str, Any]) -> None:
    """Test if block preserves input shape when using same padding and stride=1."""
    block = ConvNextBlock(conv_config=block_config, **block_params)
    outputs = block(sample_inputs)

    # With stride (1, 1), output shape should be the same as input
    assert outputs.shape == sample_inputs.shape


def test_block_output_shape_with_stride(sample_inputs: tf.Tensor, block_config: ConvNextConfig,
                                        block_params: Dict[str, Any]) -> None:
    """Test if block correctly changes output shape when using stride > 1."""
    # Create config with stride 2
    strided_config = ConvNextConfig(
        kernel_size=block_config.kernel_size,
        filters=block_config.filters,
        strides=(2, 2),  # Changed stride
        activation=block_config.activation,
        kernel_regularizer=block_config.kernel_regularizer,
        use_bias=block_config.use_bias
    )

    block = ConvNextBlock(conv_config=strided_config, **block_params)
    outputs = block(sample_inputs)

    # With stride (2, 2), output spatial dimensions should be halved
    expected_shape = (sample_inputs.shape[0],
                      sample_inputs.shape[1] // 2,
                      sample_inputs.shape[2] // 2,
                      sample_inputs.shape[3])
    assert outputs.shape == expected_shape


def test_block_training_behavior(sample_inputs: tf.Tensor, block_config: ConvNextConfig,
                                 block_params: Dict[str, Any]) -> None:
    """Test block behavior in training vs inference modes."""
    block = ConvNextBlock(conv_config=block_config, **block_params)

    # Training mode
    train_output = block(sample_inputs, training=True)

    # Inference mode
    inference_output = block(sample_inputs, training=False)

    # Outputs should be different due to dropout
    assert not np.allclose(train_output.numpy(), inference_output.numpy())


def test_block_without_dropout(sample_inputs: tf.Tensor, block_config: ConvNextConfig) -> None:
    """Test block without dropout."""
    block = ConvNextBlock(
        conv_config=block_config,
        dropout_rate=None,
        spatial_dropout_rate=None,
        use_gamma=True
    )

    # Training and inference outputs should be the same when no dropout is used
    train_output = block(sample_inputs, training=True)
    inference_output = block(sample_inputs, training=False)

    assert np.allclose(train_output.numpy(), inference_output.numpy(), rtol=1e-5, atol=1e-5)


def test_block_serialization(block_config: ConvNextConfig, block_params: Dict[str, Any]) -> None:
    """Test serialization of ConvNextBlock with more robust property comparison."""
    # Create and build the original block
    original_block = ConvNextBlock(conv_config=block_config, **block_params)

    # Get config and recreate from config
    config = original_block.get_config()
    restored_block = ConvNextBlock.from_config(config)


    # Check if the key properties match
    assert restored_block.dropout_rate == original_block.dropout_rate
    assert restored_block.spatial_dropout_rate == original_block.spatial_dropout_rate
    assert restored_block.use_gamma == original_block.use_gamma
    assert restored_block.use_softorthonormal_regularizer == original_block.use_softorthonormal_regularizer

    # Check that restored block's config has the same keys as original
    restored_config = restored_block.get_config()
    assert set(restored_config.keys()) == set(config.keys())

    # Check that the configs for ConvNextConfig match in key areas
    assert restored_block.conv_config.kernel_size == original_block.conv_config.kernel_size
    assert restored_block.conv_config.filters == original_block.conv_config.filters
    assert restored_block.conv_config.strides == original_block.conv_config.strides
    assert restored_block.conv_config.activation == original_block.conv_config.activation
    assert restored_block.conv_config.use_bias == original_block.conv_config.use_bias


@pytest.mark.parametrize("activation", ["relu", "gelu", "swish", "silu"])
def test_block_activations(activation: str, block_config: ConvNextConfig,
                           block_params: Dict[str, Any], sample_inputs: tf.Tensor) -> None:
    """Test different activation functions."""
    # Create config with different activation
    act_config = ConvNextConfig(
        kernel_size=block_config.kernel_size,
        filters=block_config.filters,
        strides=block_config.strides,
        activation=activation,  # Changed activation
        kernel_regularizer=block_config.kernel_regularizer,
        use_bias=block_config.use_bias
    )

    block = ConvNextBlock(conv_config=act_config, **block_params)
    output = block(sample_inputs)
    assert not tf.reduce_any(tf.math.is_nan(output))


def test_block_gradient_flow(sample_inputs: tf.Tensor, block_config: ConvNextConfig,
                             block_params: Dict[str, Any]) -> None:
    """Test gradient flow through the block."""
    block = ConvNextBlock(conv_config=block_config, **block_params)

    with tf.GradientTape() as tape:
        output = block(sample_inputs, training=True)
        loss = tf.reduce_mean(output)

    gradients = tape.gradient(loss, block.trainable_variables)

    # Check if gradients exist for all trainable variables
    assert all(g is not None for g in gradients)

    # Check if we have trainable variables
    assert len(block.trainable_variables) > 0


def test_gamma_scaling(sample_inputs: tf.Tensor, block_config: ConvNextConfig, block_params: Dict[str, Any]) -> None:
    """Test the effect of the gamma scaling parameter."""
    # Block with gamma scaling
    with_gamma = ConvNextBlock(
        conv_config=block_config,
        use_gamma=True,
        dropout_rate=None,  # Disable dropout to ensure deterministic output
        spatial_dropout_rate=None
    )

    # Block without gamma scaling
    without_gamma = ConvNextBlock(
        conv_config=block_config,
        use_gamma=False,
        dropout_rate=None,  # Disable dropout to ensure deterministic output
        spatial_dropout_rate=None
    )

    # Process the same inputs
    out_with_gamma = with_gamma(sample_inputs)
    out_without_gamma = without_gamma(sample_inputs)

    # Outputs should be different
    assert not np.allclose(out_with_gamma.numpy(), out_without_gamma.numpy())


def test_model_integration(sample_inputs: tf.Tensor, block_config: ConvNextConfig,
                           block_params: Dict[str, Any]) -> None:
    """Test integrating the ConvNextBlock into a Keras model."""
    inputs = keras.Input(shape=sample_inputs.shape[1:])
    block = ConvNextBlock(conv_config=block_config, **block_params)
    outputs = block(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(sample_inputs)
    assert result.shape == sample_inputs.shape


def test_model_save_load(sample_inputs: tf.Tensor, block_config: ConvNextConfig,
                         block_params: Dict[str, Any], tmp_path) -> None:
    """Test saving and loading a model with ConvNextBlock."""
    # Create a simple model with the block
    inputs = keras.Input(shape=sample_inputs.shape[1:])
    block = ConvNextBlock(conv_config=block_config, **block_params)
    outputs = block(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Generate output before saving
    original_output = model(sample_inputs).numpy()

    # Save model
    save_path = str(tmp_path / "convnext_model.keras")
    model.save(save_path)

    # Load model
    loaded_model = keras.models.load_model(save_path)

    # Generate output after loading
    loaded_output = loaded_model(sample_inputs).numpy()

    # Outputs should be identical
    assert np.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])