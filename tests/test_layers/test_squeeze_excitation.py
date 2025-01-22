import pytest
import keras
import numpy as np
import tensorflow as tf
from typing import Tuple
from keras import layers, regularizers, initializers


# Import the layer (adjust the import path according to your project structure)
from dl_techniques.layers.squeeze_excitation import SqueezeExcitation


@pytest.fixture
def input_shape() -> Tuple[int, int, int, int]:
    """Fixture for standard input shape."""
    return (2, 32, 32, 64)  # (batch_size, height, width, channels)


@pytest.fixture
def sample_input(input_shape: Tuple[int, int, int, int]) -> tf.Tensor:
    """Fixture for creating sample input tensor."""
    return tf.random.normal(input_shape)


def test_initialization():
    """Test that the layer initializes correctly with valid parameters."""
    # Test initialization with default parameters
    layer = SqueezeExcitation()
    assert layer.reduction_ratio == 0.25
    assert isinstance(layer.kernel_initializer, initializers.GlorotNormal)
    assert layer.kernel_regularizer is None

    # Test initialization with custom parameters
    custom_layer = SqueezeExcitation(
        reduction_ratio=0.5,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.L2(0.01),
        activation='swish',
        use_bias=True
    )
    assert custom_layer.reduction_ratio == 0.5
    assert isinstance(custom_layer.kernel_initializer, initializers.HeNormal)
    assert isinstance(custom_layer.kernel_regularizer, regularizers.L2)
    assert custom_layer.use_bias is True


def test_invalid_reduction_ratio():
    """Test that the layer raises ValueError for invalid reduction ratio."""
    with pytest.raises(ValueError):
        SqueezeExcitation(reduction_ratio=0.0)

    with pytest.raises(ValueError):
        SqueezeExcitation(reduction_ratio=1.5)


def test_output_shape(sample_input: tf.Tensor):
    """Test that the output shape matches the input shape."""
    layer = SqueezeExcitation()
    output = layer(sample_input)

    # Check that output shape matches input shape
    assert output.shape == sample_input.shape

    # Check that output dtype matches input dtype
    assert output.dtype == sample_input.dtype


def test_forward_pass(sample_input: tf.Tensor):
    """Test the forward pass of the layer."""
    layer = SqueezeExcitation(reduction_ratio=0.5)
    output = layer(sample_input)

    # Check that values are in valid range (due to sigmoid activation)
    # The output should be element-wise multiplication of input with attention weights
    assert tf.reduce_min(output) >= tf.reduce_min(sample_input)
    assert tf.reduce_max(output) <= tf.reduce_max(sample_input)

    # Check that the output is not all zeros
    assert not tf.reduce_all(tf.equal(output, 0))


def test_serialization():
    """Test the serialization and deserialization of the layer."""
    original_layer = SqueezeExcitation(
        reduction_ratio=0.3,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.L2(0.01),
        activation='swish',
        use_bias=True
    )

    config = original_layer.get_config()
    new_layer = SqueezeExcitation.from_config(config)

    # Check that all configurations are preserved
    assert new_layer.reduction_ratio == original_layer.reduction_ratio
    assert isinstance(new_layer.kernel_initializer, type(original_layer.kernel_initializer))
    assert isinstance(new_layer.kernel_regularizer, type(original_layer.kernel_regularizer))
    assert new_layer.use_bias == original_layer.use_bias


def test_gradient_flow():
    """Test gradient flow through the layer."""
    layer = SqueezeExcitation(reduction_ratio=0.5)

    with tf.GradientTape() as tape:
        inputs = tf.random.normal((2, 16, 16, 32))
        tape.watch(inputs)
        outputs = layer(inputs)
        loss = tf.reduce_mean(outputs)

    # Calculate gradients
    grads = tape.gradient(loss, layer.trainable_variables)

    # Check that all layer variables receive gradients
    assert all(g is not None for g in grads)

    # Check gradient shapes match variable shapes
    for grad, var in zip(grads, layer.trainable_variables):
        assert grad.shape == var.shape


def test_model_integration():
    """Test integration with a complete model architecture."""
    # Create a simple CNN with SE blocks
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = SqueezeExcitation()(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = SqueezeExcitation()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile and check if model can be trained
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Test forward pass
    test_input = tf.random.normal((4, 32, 32, 3))
    output = model(test_input)

    assert output.shape == (4, 10)
    assert tf.reduce_all(tf.abs(tf.reduce_sum(output, axis=1) - 1.0) < 1e-6)


def test_numerical_stability():
    """Test numerical stability with extreme input values."""
    layer = SqueezeExcitation(reduction_ratio=0.1)

    # Test with very large values
    large_input = tf.random.normal((2, 16, 16, 32)) * 1e10
    large_output = layer(large_input)
    assert not tf.reduce_any(tf.math.is_inf(large_output))
    assert not tf.reduce_any(tf.math.is_nan(large_output))

    # Test with very small values
    small_input = tf.random.normal((2, 16, 16, 32)) * 1e-10
    small_output = layer(small_input)
    assert not tf.reduce_any(tf.math.is_inf(small_output))
    assert not tf.reduce_any(tf.math.is_nan(small_output))

    # Test with mixed values
    mixed_input = tf.concat([large_input, small_input], axis=0)
    mixed_output = layer(mixed_input)
    assert not tf.reduce_any(tf.math.is_inf(mixed_output))
    assert not tf.reduce_any(tf.math.is_nan(mixed_output))


if __name__ == "__main__":
    pytest.main([__file__])
