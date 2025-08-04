import keras
import pytest
import numpy as np
import tensorflow as tf
from dl_techniques.layers.attention.differential_attention import DifferentialMultiHeadAttention


@pytest.fixture
def create_test_data():
    """Create test data for attention layer tests."""
    batch_size = 4
    seq_length = 16
    dim = 64

    # Create input data
    x = np.random.normal(size=(batch_size, seq_length, dim)).astype(np.float32)

    # Create attention mask (1 = attend, 0 = ignore)
    mask = np.ones((batch_size, seq_length))
    # Mask out some positions for testing
    mask[:, -4:] = 0

    # Convert mask to TF attention mask format
    attention_mask = np.einsum('bi,bj->bij', mask, mask)[:, tf.newaxis, :, :]

    return x, attention_mask


def test_layer_initialization():
    """Test that the layer initializes correctly."""
    # Test with valid parameters
    layer = DifferentialMultiHeadAttention(
        dim=64,
        num_heads=8,
        head_dim=8,
        dropout=0.1,
        attention_dropout=0.1,
        lambda_init=0.8,
        kernel_regularizer=keras.regularizers.L2(1e-4)
    )

    assert layer.dim == 64
    assert layer.num_heads == 8
    assert layer.head_dim == 8
    assert layer.dropout_rate == 0.1
    assert layer.attention_dropout_rate == 0.1
    assert layer.lambda_init == 0.8

    # Test with invalid parameters
    with pytest.raises(ValueError):
        # dim not divisible by num_heads
        DifferentialMultiHeadAttention(
            dim=65,  # Not divisible by 8
            num_heads=8,
            head_dim=8
        )


def test_compute_output_shape():
    """Test the compute_output_shape method."""
    layer = DifferentialMultiHeadAttention(
        dim=64,
        num_heads=8,
        head_dim=8
    )

    # Test with TensorShape
    input_shape = tf.TensorShape([None, 16, 64])
    output_shape = layer.compute_output_shape(input_shape)
    assert output_shape == input_shape

    # Test with tuple
    input_shape = (None, 16, 64)
    output_shape = layer.compute_output_shape(input_shape)
    assert output_shape == input_shape

    # Test with list
    input_shape = [None, 16, 64]
    output_shape = layer.compute_output_shape(input_shape)
    assert tuple(output_shape) == tuple(input_shape)


def test_build_and_call(create_test_data):
    """Test building the layer and calling it."""
    x, mask = create_test_data

    layer = DifferentialMultiHeadAttention(
        dim=64,
        num_heads=8,
        head_dim=8,
        dropout=0.1,
        attention_dropout=0.1
    )

    # First call will build the layer
    output = layer(x, mask=mask, training=True)

    # Check output shape matches input shape
    assert output.shape == x.shape

    # Check that attributes are initialized after build
    assert layer.attention1 is not None
    assert layer.attention2 is not None
    assert layer.proj is not None
    assert layer.dropout_layer is not None
    assert layer.lambda_param is not None


def test_get_lambda():
    """Test the get_lambda method."""
    layer = DifferentialMultiHeadAttention(
        dim=64,
        num_heads=8,
        head_dim=8,
        lambda_init=0.8
    )

    # Build the layer
    input_shape = (4, 16, 64)
    layer.build(input_shape)

    # Test lambda for different layer indices
    lambda0 = layer.get_lambda(0)
    lambda5 = layer.get_lambda(5)
    lambda10 = layer.get_lambda(10)

    # Values should be in range [0.1, 0.9]
    assert 0.1 <= float(lambda0) <= 0.9
    assert 0.1 <= float(lambda5) <= 0.9
    assert 0.1 <= float(lambda10) <= 0.9

    # Lambda should increase with layer depth
    assert float(lambda0) <= float(lambda5) <= float(lambda10)


def test_serialization():
    """Test serialization and deserialization."""
    layer = DifferentialMultiHeadAttention(
        dim=64,
        num_heads=8,
        head_dim=8,
        dropout=0.1,
        attention_dropout=0.2,
        lambda_init=0.7,
        kernel_regularizer=keras.regularizers.L2(1e-4)
    )

    # Get config and recreate layer
    config = layer.get_config()
    new_layer = DifferentialMultiHeadAttention.from_config(config)

    # Check if configuration is preserved
    assert new_layer.dim == layer.dim
    assert new_layer.num_heads == layer.num_heads
    assert new_layer.head_dim == layer.head_dim
    assert new_layer.dropout_rate == layer.dropout_rate
    assert new_layer.attention_dropout_rate == layer.attention_dropout_rate
    assert new_layer.lambda_init == layer.lambda_init