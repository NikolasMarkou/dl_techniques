"""
Test suite for Convolutional Transformer implementation.

This module provides comprehensive tests for:
- ConvolutionalTransformerBlock layer
- create_convolutional_transformer_model function
- Model integration and serialization
"""

import pytest
import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Any

from dl_techniques.layers.convolutional_transformer import (
    ConvolutionalTransformerBlock,
    create_convolutional_transformer_model
)


# Test fixtures
@pytest.fixture
def sample_inputs() -> tf.Tensor:
    """Generate sample input tensor."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 32, 32, 64))


@pytest.fixture
def block_params() -> Dict[str, Any]:
    """Default parameters for ConvolutionalTransformerBlock."""
    return {
        "dim": 64,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout_rate": 0.1,
        "attention_dropout": 0.1,
        "use_scale": True,
        "conv_kernel_size": 3,
        "activation": "gelu"
    }


@pytest.fixture
def model_params() -> Dict[str, Any]:
    """Default parameters for create_convolutional_transformer_model."""
    return {
        "input_shape": (32, 32, 3),
        "num_classes": 1000,
        "num_blocks": 2,
        "dim": 64,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout_rate": 0.1,
        "attention_dropout": 0.1,
        "activation": "gelu"
    }


# ConvolutionalTransformerBlock tests
def test_block_initialization(block_params: Dict[str, Any]) -> None:
    """Test initialization of ConvolutionalTransformerBlock."""
    block = ConvolutionalTransformerBlock(**block_params)
    assert block.dim == block_params["dim"]
    assert block.num_heads == block_params["num_heads"]
    assert block.mlp_ratio == block_params["mlp_ratio"]


def test_block_invalid_dimensions() -> None:
    """Test if block raises error for invalid dimensions."""
    with pytest.raises(ValueError):
        # dim not divisible by num_heads
        ConvolutionalTransformerBlock(dim=65, num_heads=8)


def test_block_output_shape(sample_inputs: tf.Tensor, block_params: Dict[str, Any]) -> None:
    """Test if block preserves input shape."""
    block = ConvolutionalTransformerBlock(**block_params)
    outputs = block(sample_inputs)
    assert outputs.shape == sample_inputs.shape


def test_block_training_behavior(sample_inputs: tf.Tensor, block_params: Dict[str, Any]) -> None:
    """Test block behavior in training vs inference modes."""
    block = ConvolutionalTransformerBlock(**block_params)

    # Training mode
    train_output = block(sample_inputs, training=True)

    # Inference mode
    inference_output = block(sample_inputs, training=False)

    # Outputs should be different due to dropout
    assert not np.allclose(train_output.numpy(), inference_output.numpy())


def test_block_serialization(block_params: Dict[str, Any]) -> None:
    """Test serialization of ConvolutionalTransformerBlock."""
    original_block = ConvolutionalTransformerBlock(**block_params)
    config = original_block.get_config()

    # Recreate from config
    restored_block = ConvolutionalTransformerBlock.from_config(config)

    # Check if configurations match
    assert restored_block.get_config() == config


@pytest.mark.parametrize("activation", ["relu", "gelu", "swish", tf.nn.relu])
def test_block_activations(activation, block_params: Dict[str, Any], sample_inputs: tf.Tensor) -> None:
    """Test different activation functions."""
    block_params["activation"] = activation
    block = ConvolutionalTransformerBlock(**block_params)
    output = block(sample_inputs)
    assert not tf.reduce_any(tf.math.is_nan(output))


def test_block_gradient_flow(sample_inputs: tf.Tensor, block_params: Dict[str, Any]) -> None:
    """Test gradient flow through the block."""
    block = ConvolutionalTransformerBlock(**block_params)

    with tf.GradientTape() as tape:
        output = block(sample_inputs, training=True)
        loss = tf.reduce_mean(output)

    gradients = tape.gradient(loss, block.trainable_variables)
    assert all(g is not None for g in gradients)


# Model tests
def test_model_creation(model_params: Dict[str, Any]) -> None:
    """Test creation of convolutional transformer model."""
    model = create_convolutional_transformer_model(**model_params)
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) > 0


def test_model_output_shape(model_params: Dict[str, Any]) -> None:
    """Test model output shape."""
    model = create_convolutional_transformer_model(**model_params)
    batch_size = 2
    inputs = tf.random.normal((batch_size, *model_params["input_shape"]))
    outputs = model(inputs)

    expected_shape = (batch_size, model_params["num_classes"])
    assert outputs.shape == expected_shape


def test_model_inference(model_params: Dict[str, Any]) -> None:
    """Test model inference."""
    model = create_convolutional_transformer_model(**model_params)

    # Test batch inference
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        inputs = tf.random.normal((batch_size, *model_params["input_shape"]))
        outputs = model(inputs, training=False)

        # Check output shape and values
        assert outputs.shape == (batch_size, model_params["num_classes"])
        assert tf.reduce_all(outputs >= 0.0) and tf.reduce_all(outputs <= 1.0)
        assert np.allclose(tf.reduce_sum(outputs, axis=-1), 1.0)


@pytest.mark.parametrize("num_blocks", [1, 2, 4])
def test_model_depth(num_blocks: int, model_params: Dict[str, Any]) -> None:
    """Test model with different numbers of transformer blocks."""
    model_params["num_blocks"] = num_blocks
    model = create_convolutional_transformer_model(**model_params)

    # Count transformer blocks
    transformer_blocks = [layer for layer in model.layers
                          if isinstance(layer, ConvolutionalTransformerBlock)]
    assert len(transformer_blocks) == num_blocks


def test_model_save_load(model_params: Dict[str, Any], tmp_path) -> None:
    """Test model serialization."""
    model = create_convolutional_transformer_model(**model_params)

    # Save model
    save_path = tmp_path / "transformer_model.keras"
    model.save(save_path)

    # Load model
    loaded_model = tf.keras.models.load_model(save_path)

    # Compare outputs
    test_input = tf.random.normal((1, *model_params["input_shape"]))
    original_output = model(test_input)
    loaded_output = loaded_model(test_input)
    assert np.allclose(original_output.numpy(), loaded_output.numpy())

if __name__ == "__main__":
    pytest.main([__file__])
