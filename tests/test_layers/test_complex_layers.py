"""
Tests for Complex-Valued Neural Network Layers
============================================

This module provides comprehensive tests for complex-valued neural network layers,
including initialization tests, shape verification, and numerical correctness checks.
"""

import pytest
import numpy as np
import tensorflow as tf
from typing import Tuple, List

from dl_techniques.layers.complex_layers import (
    ComplexLayer,
    ComplexConv2D,
    ComplexDense,
    ComplexReLU
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def random_complex_input() -> tf.Tensor:
    """Create random complex input tensor."""
    real = tf.random.normal((8, 32, 32, 3))
    imag = tf.random.normal((8, 32, 32, 3))
    return tf.complex(real, imag)


@pytest.fixture
def random_complex_dense_input() -> tf.Tensor:
    """Create random complex input tensor for dense layer."""
    real = tf.random.normal((8, 128))
    imag = tf.random.normal((8, 128))
    return tf.complex(real, imag)


# ---------------------------------------------------------------------
# Base Layer Tests
# ---------------------------------------------------------------------

class TestComplexLayer:
    """Tests for base ComplexLayer functionality."""

    def test_init_complex_weights(self) -> None:
        """Test complex weight initialization."""
        layer = ComplexLayer()
        shape = (3, 3, 64, 32)
        weights = layer._init_complex_weights(shape)

        # Check shape and dtype
        assert weights.shape == shape
        assert weights.dtype == tf.complex64

        # Check statistical properties
        magnitudes = tf.abs(weights)
        phases = tf.math.angle(weights)

        # Magnitude should follow Rayleigh distribution
        assert tf.reduce_mean(magnitudes) > 0
        assert tf.math.reduce_std(magnitudes) < 1.0

        # Phases should be uniformly distributed in [-π, π]
        assert tf.reduce_min(phases) >= -np.pi
        assert tf.reduce_max(phases) <= np.pi

    def test_epsilon_handling(self) -> None:
        """Test epsilon parameter handling."""
        custom_epsilon = 1e-5
        layer = ComplexLayer(epsilon=custom_epsilon)
        assert layer.epsilon == custom_epsilon

    def test_regularizer_attachment(self) -> None:
        """Test kernel regularizer attachment."""
        regularizer = tf.keras.regularizers.L2(0.01)
        layer = ComplexLayer(kernel_regularizer=regularizer)
        assert layer.kernel_regularizer == regularizer


# ---------------------------------------------------------------------
# Complex Convolution Tests
# ---------------------------------------------------------------------

class TestComplexConv2D:
    """Tests for ComplexConv2D layer."""

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = ComplexConv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='SAME'
        )
        assert layer.filters == 32
        assert layer.kernel_size == (3, 3)
        assert layer.strides == (1, 1)
        assert layer.padding == 'SAME'

    def test_build(self, random_complex_input: tf.Tensor) -> None:
        """Test layer building and weight creation."""
        layer = ComplexConv2D(filters=32, kernel_size=3)
        layer.build(random_complex_input.shape)

        # Check kernel shape
        assert layer.kernel.shape == (3, 3, 3, 32)
        assert layer.bias.shape == (32,)

        # Check dtypes
        assert layer.kernel.dtype == tf.complex64
        assert layer.bias.dtype == tf.complex64

    def test_forward_pass(self, random_complex_input: tf.Tensor) -> None:
        """Test forward pass computation."""
        layer = ComplexConv2D(filters=32, kernel_size=3)
        output = layer(random_complex_input)

        # Check output shape
        assert output.shape == (8, 32, 32, 32)
        assert output.dtype == tf.complex64

        # Check numerical properties
        assert not tf.reduce_any(tf.math.is_nan(tf.abs(output)))
        assert not tf.reduce_any(tf.math.is_inf(tf.abs(output)))

    def test_padding_modes(self, random_complex_input: tf.Tensor) -> None:
        """Test different padding modes."""
        # Test 'SAME' padding
        layer_same = ComplexConv2D(filters=32, kernel_size=3, padding='SAME')
        output_same = layer_same(random_complex_input)
        assert output_same.shape[1:3] == random_complex_input.shape[1:3]

        # Test 'VALID' padding
        layer_valid = ComplexConv2D(filters=32, kernel_size=3, padding='VALID')
        output_valid = layer_valid(random_complex_input)
        expected_shape = (
            random_complex_input.shape[0],  # batch
            random_complex_input.shape[1] - 2,  # height
            random_complex_input.shape[2] - 2,  # width
            32  # filters
        )
        assert output_valid.shape == expected_shape

    def test_strided_convolution(self, random_complex_input: tf.Tensor) -> None:
        """Test strided convolution."""
        # Test with strides=2
        layer = ComplexConv2D(filters=32, kernel_size=3, strides=2)
        output = layer(random_complex_input)

        expected_shape = (
            random_complex_input.shape[0],  # batch
            random_complex_input.shape[1] // 2,  # height
            random_complex_input.shape[2] // 2,  # width
            32  # filters
        )
        assert output.shape == expected_shape


# ---------------------------------------------------------------------
# Complex Dense Tests
# ---------------------------------------------------------------------

class TestComplexDense:
    """Tests for ComplexDense layer."""

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = ComplexDense(units=64)
        assert layer.units == 64

    def test_build(self, random_complex_dense_input: tf.Tensor) -> None:
        """Test layer building and weight creation."""
        layer = ComplexDense(units=64)
        layer.build(random_complex_dense_input.shape)

        # Check shapes
        assert layer.kernel.shape == (128, 64)
        assert layer.bias.shape == (64,)

        # Check dtypes
        assert layer.kernel.dtype == tf.complex64
        assert layer.bias.dtype == tf.complex64

    def test_forward_pass(self, random_complex_dense_input: tf.Tensor) -> None:
        """Test forward pass computation."""
        layer = ComplexDense(units=64)
        output = layer(random_complex_dense_input)

        # Check output properties
        assert output.shape == (8, 64)
        assert output.dtype == tf.complex64
        assert not tf.reduce_any(tf.math.is_nan(tf.abs(output)))
        assert not tf.reduce_any(tf.math.is_inf(tf.abs(output)))

    def test_weight_gradients(self, random_complex_dense_input: tf.Tensor) -> None:
        """Test gradient computation."""
        layer = ComplexDense(units=64)
        with tf.GradientTape() as tape:
            output = layer(random_complex_dense_input)
            loss = tf.reduce_mean(tf.abs(output))

        # Compute gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradient properties
        for grad in grads:
            assert grad is not None
            assert not tf.reduce_any(tf.math.is_nan(tf.abs(grad)))
            assert not tf.reduce_any(tf.math.is_inf(tf.abs(grad)))


# ---------------------------------------------------------------------
# Complex ReLU Tests
# ---------------------------------------------------------------------

class TestComplexReLU:
    """Tests for ComplexReLU activation."""

    def test_forward_pass(self, random_complex_input: tf.Tensor) -> None:
        """Test forward pass computation."""
        layer = ComplexReLU()
        output = layer(random_complex_input)

        # Check shape preservation
        assert output.shape == random_complex_input.shape
        assert output.dtype == tf.complex64

        # Check ReLU properties
        real_part = tf.math.real(output)
        imag_part = tf.math.imag(output)
        assert tf.reduce_all(real_part >= 0)
        assert tf.reduce_all(imag_part >= 0)

    def test_zero_input(self) -> None:
        """Test behavior with zero input."""
        layer = ComplexReLU()
        zero_input = tf.zeros((8, 32, 32, 3), dtype=tf.complex64)
        output = layer(zero_input)
        assert tf.reduce_all(tf.abs(output) == 0)

    def test_negative_input(self) -> None:
        """Test behavior with negative input."""
        layer = ComplexReLU()
        negative_input = tf.complex(
            -tf.ones((8, 32, 32, 3)),
            -tf.ones((8, 32, 32, 3))
        )
        output = layer(negative_input)
        assert tf.reduce_all(tf.abs(output) == 0)

    def test_gradient_flow(self) -> None:
        """Test gradient flow through activation."""
        layer = ComplexReLU()
        input_tensor = tf.complex(
            tf.random.normal((8, 32, 32, 3)),
            tf.random.normal((8, 32, 32, 3))
        )

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            output = layer(input_tensor)
            loss = tf.reduce_mean(tf.abs(output))

        gradient = tape.gradient(loss, input_tensor)
        assert gradient is not None
        assert not tf.reduce_any(tf.math.is_nan(tf.abs(gradient)))


# ---------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------

def test_layer_composition() -> None:
    """Test composition of complex layers."""
    # Create input
    input_tensor = tf.complex(
        tf.random.normal((8, 32, 32, 3)),
        tf.random.normal((8, 32, 32, 3))
    )

    # Create layer stack
    conv1 = ComplexConv2D(32, 3)
    relu1 = ComplexReLU()
    conv2 = ComplexConv2D(64, 3)
    relu2 = ComplexReLU()

    # Forward pass
    x = conv1(input_tensor)
    x = relu1(x)
    x = conv2(x)
    output = relu2(x)

    # Check final output
    assert output.shape == (8, 32, 32, 64)
    assert output.dtype == tf.complex64
    assert not tf.reduce_any(tf.math.is_nan(tf.abs(output)))
    assert not tf.reduce_any(tf.math.is_inf(tf.abs(output)))


def test_numerical_stability() -> None:
    """Test numerical stability with extreme values."""
    # Create input with large values
    large_input = tf.complex(
        1e5 * tf.random.normal((8, 32, 32, 3)),
        1e5 * tf.random.normal((8, 32, 32, 3))
    )

    # Test conv layer
    conv = ComplexConv2D(32, 3)
    conv_out = conv(large_input)
    assert not tf.reduce_any(tf.math.is_nan(tf.abs(conv_out)))
    assert not tf.reduce_any(tf.math.is_inf(tf.abs(conv_out)))

    # Test dense layer with scaled input
    dense = ComplexDense(64)
    dense_input = tf.reshape(large_input, (8, -1))[:, :128]  # Take first 128 features
    dense_out = dense(dense_input)
    assert not tf.reduce_any(tf.math.is_nan(tf.abs(dense_out)))
    assert not tf.reduce_any(tf.math.is_inf(tf.abs(dense_out)))


def test_training_loop() -> None:
    """Test full training loop with complex layers."""
    # Create synthetic dataset
    x_train = tf.complex(
        tf.random.normal((100, 32, 32, 3)),
        tf.random.normal((100, 32, 32, 3))
    )
    y_train = tf.random.uniform((100,), maxval=10, dtype=tf.int32)

    # Create model
    conv = ComplexConv2D(32, 3)
    relu = ComplexReLU()
    flatten = tf.keras.layers.Flatten()
    dense = ComplexDense(10)

    # Training loop
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for _ in range(2):  # Just test a couple iterations
        with tf.GradientTape() as tape:
            # Forward pass
            x = conv(x_train)
            x = relu(x)
            x = flatten(x)
            logits = tf.abs(dense(x))

            # Compute loss
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_train, logits, from_logits=True
                )
            )

        # Backward pass
        grads = tape.gradient(loss, [conv.trainable_variables,
                                     dense.trainable_variables])
        optimizer.apply_gradients(zip(grads[0], conv.trainable_variables))
        optimizer.apply_gradients(zip(grads[1], dense.trainable_variables))

        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)


if __name__ == '__main__':
    pytest.main([__file__])