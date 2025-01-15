"""
Test suite for SRIP (Spectral Restricted Isometry Property) Regularizer.

This module provides comprehensive tests for:
- Configuration validation
- Regularizer initialization
- Numerical stability
- Training behavior
- Model integration
- Serialization/deserialization
"""
import os
import keras
import pytest
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict

from dl_techniques.regularizers.srip import SRIPRegularizer


# Fixtures
@pytest.fixture
def default_regularizer() -> SRIPRegularizer:
    """Create default regularizer instance."""
    return SRIPRegularizer(power_iterations=5)


@pytest.fixture
def sample_dense_weights() -> tf.Tensor:
    """Generate sample dense layer weights."""
    tf.random.set_seed(42)
    return tf.random.normal((64, 32))


@pytest.fixture
def sample_conv_weights() -> tf.Tensor:
    """Generate sample convolutional layer weights."""
    tf.random.set_seed(42)
    return tf.random.normal((3, 3, 32, 64))


@pytest.fixture
def test_model(default_regularizer) -> tf.keras.Model:
    """Create test model with SRIP regularizer."""
    return keras.Sequential([
        keras.layers.Conv2D(
            32, 3,
            kernel_regularizer=default_regularizer,
            padding='same',
            input_shape=(32, 32, 3)
        ),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(
            10,
            kernel_regularizer=default_regularizer,
            activation='softmax'  # Add softmax activation
        )
    ])


# Initialization Tests
def test_default_initialization():
    """Test initialization with default parameters."""
    regularizer = SRIPRegularizer()
    assert regularizer.lambda_init == 0.1
    assert regularizer.power_iterations == 2
    assert regularizer.epsilon == 1e-7
    assert isinstance(regularizer.lambda_schedule, dict)


@pytest.mark.parametrize("params", [
    {"lambda_init": 0.2},
    {"power_iterations": 3},
    {"epsilon": 1e-7},
    {"lambda_schedule": {10: 1e-3, 20: 1e-4}},
])
def test_custom_initialization(params):
    """Test initialization with custom parameters."""
    regularizer = SRIPRegularizer(**params)
    for key, value in params.items():
        assert getattr(regularizer, key) == value


@pytest.mark.parametrize("params,error", [
    ({"lambda_init": -1.0}, ValueError),
    ({"power_iterations": 0}, ValueError),
    ({"epsilon": 0.0}, ValueError)
])
def test_invalid_initialization(params, error):
    """Test initialization with invalid parameters."""
    with pytest.raises(error):
        SRIPRegularizer(**params)


# Numerical Stability Tests
def test_safe_normalize(default_regularizer, sample_dense_weights):
    """Test safe normalization operation."""
    normalized = default_regularizer._safe_normalize(sample_dense_weights)
    norms = tf.sqrt(tf.reduce_sum(tf.square(normalized), axis=0))
    assert tf.reduce_all(tf.abs(norms - 1.0) < 1e-2)


# Core Functionality Tests
def test_reshape_kernel(default_regularizer, sample_conv_weights):
    """Test kernel reshaping."""
    reshaped = default_regularizer._reshape_kernel(sample_conv_weights)
    expected_shape = (3 * 3 * 32, 64)
    assert reshaped.shape == expected_shape


def test_regularization_dense(default_regularizer, sample_dense_weights):
    """Test regularization on dense weights."""
    loss = default_regularizer(sample_dense_weights)
    assert not tf.math.is_nan(loss)
    assert loss.shape == ()
    assert loss > 0


def test_regularization_conv(default_regularizer, sample_conv_weights):
    """Test regularization on convolutional weights."""
    loss = default_regularizer(sample_conv_weights)
    assert not tf.math.is_nan(loss)
    assert loss.shape == ()
    assert loss > 0


@pytest.mark.parametrize("epoch,expected_lambda", [
    (0, 0.1),  # Initial lambda
    (20, 1e-3),  # First reduction
    (50, 1e-4),  # Second reduction
    (70, 1e-6),  # Third reduction
    (120, 0.0),  # Final value
])
def test_lambda_scheduling(default_regularizer, epoch, expected_lambda):
    """Test lambda scheduling."""
    default_regularizer.update_lambda(epoch)
    assert tf.abs(default_regularizer.current_lambda - expected_lambda) < 1e-7


# Model Integration Tests
def test_model_compilation(test_model):
    """Test model compilation with regularizer."""
    test_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    assert test_model.optimizer is not None
    assert test_model.loss is not None


def test_training_step(test_model):
    """Test single training step."""
    # Enable eager execution for the test
    tf.config.run_functions_eagerly(True)

    try:
        test_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate dummy data
        x = tf.random.normal((4, 32, 32, 3))
        y = tf.random.uniform((4,), maxval=10, dtype=tf.int32)

        # Run one training step
        history = test_model.fit(x, y, epochs=1, verbose=0)
        assert 'loss' in history.history
    finally:
        # Restore default execution mode
        tf.config.run_functions_eagerly(False)


def test_regularization_effect(test_model):
    """Test that regularization affects training."""
    # Enable eager execution for the test
    tf.config.run_functions_eagerly(True)
    try:
        test_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy'
        )

        # Generate dummy data
        x = tf.random.normal((4, 32, 32, 3))
        y = tf.random.uniform((4,), maxval=10, dtype=tf.int32)

        # Get initial weights
        initial_weights = test_model.get_weights()

        # Train for one epoch
        test_model.fit(x, y, epochs=1, verbose=0)

        # Check that weights have changed
        final_weights = test_model.get_weights()
        assert any(not np.allclose(i, f) for i, f in zip(initial_weights, final_weights))
    finally:
        # Restore default execution mode
        tf.config.run_functions_eagerly(False)


# Serialization Tests
def test_config_serialization(default_regularizer):
    """Test regularizer serialization/deserialization."""
    config = default_regularizer.get_config()
    recreated = SRIPRegularizer.from_config(config)

    assert recreated.lambda_init == default_regularizer.lambda_init
    assert recreated.power_iterations == default_regularizer.power_iterations
    assert recreated.epsilon == default_regularizer.epsilon
    assert recreated.lambda_schedule == default_regularizer.lambda_schedule


def test_model_serialization(test_model, tmp_path):
    """Test model serialization with regularizer."""
    # Save model
    save_path = os.path.join(tmp_path, "test_model.keras")
    test_model.save(save_path)

    # Load model
    loaded_model = keras.models.load_model(save_path)

    # Check regularizer presence
    assert isinstance(loaded_model.layers[0].kernel_regularizer, SRIPRegularizer)
    assert isinstance(loaded_model.layers[4].kernel_regularizer, SRIPRegularizer)


# Edge Case Tests
@pytest.mark.parametrize("weight_shape", [
    (1, 1),  # Minimal shape
    (1000, 10),  # Large shape
    (32, 32),  # Square shape
    (10, 100),  # Wide shape
    (100, 10),  # Tall shape
])
def test_various_weight_shapes(default_regularizer, weight_shape):
    """Test regularizer with various weight shapes."""
    weights = tf.random.normal(weight_shape)
    loss = default_regularizer(weights)
    assert not tf.math.is_nan(loss)
    assert loss.shape == ()


def test_near_zero_weights(default_regularizer):
    """Test behavior with near-zero weights."""
    weights = tf.random.normal((32, 32)) * 1e-10
    loss = default_regularizer(weights)
    assert not tf.math.is_nan(loss)
    assert loss > 0


def test_large_weights(default_regularizer):
    """Test behavior with large weights."""
    weights = tf.random.normal((32, 32)) * 1e10
    loss = default_regularizer(weights)
    assert not tf.math.is_nan(loss)
    assert tf.math.is_finite(loss)


if __name__ == "__main__":
    pytest.main([__file__])