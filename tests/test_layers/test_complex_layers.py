"""
Tests for Complex-Valued Neural Network Layers
============================================

This module provides comprehensive tests for complex-valued neural network layers,
including initialization tests, shape verification, and numerical correctness checks.
"""

import keras
import pytest
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Tuple, List, Optional

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


@dataclass
class ComplexModelConfig:
    """Configuration for complex model testing.

    Args:
        batch_size: Number of samples per batch
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        conv_filters: Number of filters in conv layer
        kernel_size: Size of conv kernel
        dense_units: Number of units in dense layer
        kernel_regularizer: Regularization factor for kernel
        kernel_initializer: Initializer for kernel weights
    """
    batch_size: int = 32
    input_shape: Tuple[int, ...] = (32, 32, 3)
    num_classes: int = 10
    learning_rate: float = 0.0001
    num_epochs: int = 10
    conv_filters: int = 32
    kernel_size: int = 3
    dense_units: int = 10
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None
    kernel_initializer: str = 'glorot_uniform'

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


def create_complex_model(config: ComplexModelConfig) -> keras.Model:
    """Create a model with complex layers.

    Args:
        config: Model configuration parameters

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        ComplexConv2D(
            filters=config.conv_filters,
            kernel_size=config.kernel_size,
            kernel_regularizer=config.kernel_regularizer,
            kernel_initializer=config.kernel_initializer,
        ),
        ComplexReLU(),
        keras.layers.Flatten(),
        ComplexDense(
            units=config.dense_units,
            kernel_regularizer=config.kernel_regularizer,
            kernel_initializer=config.kernel_initializer
        ),
        # Final layer to get real outputs
        keras.layers.Lambda(lambda x: tf.abs(x))
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


def generate_complex_data(
        config: ComplexModelConfig,
        num_samples: int = 100
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Generate synthetic complex data for testing.

    Args:
        config: Model configuration
        num_samples: Number of samples to generate

    Returns:
        Tuple of (x_train, y_train)
    """
    x_train = tf.complex(
        tf.random.normal((num_samples,) + config.input_shape),
        tf.random.normal((num_samples,) + config.input_shape)
    )
    y_train = tf.random.uniform(
        (num_samples,),
        maxval=config.num_classes,
        dtype=tf.int32
    )
    return x_train, y_train


@pytest.mark.integration
def test_complex_model_training() -> None:
    """Integration test for training complex model.

    Tests the full training loop with all complex layers integrated
    into a single model using the Keras API.
    """
    # Initialize configuration
    config = ComplexModelConfig(
        kernel_regularizer=keras.regularizers.L2(l2=0.01)
    )

    # Generate synthetic data
    x_train, y_train = generate_complex_data(config)

    # Create and compile model
    model = create_complex_model(config)

    # Train model
    history = model.fit(
        x_train,
        y_train,
        batch_size=config.batch_size,
        epochs=config.num_epochs,
        verbose=0
    )

    # Basic assertions to verify training
    assert history.history['loss'][-1] < history.history['loss'][0], \
       "Loss should decrease during training"
    assert not any(tf.math.is_nan(loss) for loss in history.history['loss']), \
        "Loss should not be NaN"
    assert not any(tf.math.is_inf(loss) for loss in history.history['loss']), \
        "Loss should not be infinite"

    # Test model save/load
    model.save("test_complex_model.keras")
    loaded_model = keras.models.load_model(
        "test_complex_model.keras"
    )
    #
    # # Verify loaded model predictions match original
    # original_pred = model.predict(x_train[:1])
    # loaded_pred = loaded_model.predict(x_train[:1])
    # np.testing.assert_allclose(
    #     original_pred,
    #     loaded_pred,
    #     rtol=1e-5,
    #     atol=1e-5
    # )


if __name__ == '__main__':
    pytest.main([__file__])