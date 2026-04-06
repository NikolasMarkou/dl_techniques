"""
Tests for the L2_custom Regularizer implementation.

This test suite verifies the functionality of the L2_custom regularizer
including its support for negative l2 values (anti-regularization).
"""

import pytest
import numpy as np
import tensorflow as tf
from keras.api.layers import Dense
from keras.api.models import Sequential

from dl_techniques.regularizers.l2_custom import L2_custom, validate_float_arg


# Initialization Tests

def test_default_initialization():
    """Test proper initialization with default parameters."""
    reg = L2_custom()
    assert reg.l2 == 0.01


def test_custom_initialization():
    """Test initialization with custom l2 value."""
    reg = L2_custom(l2=0.1)
    assert reg.l2 == 0.1


def test_none_defaults_to_001():
    """Test that None l2 defaults to 0.01."""
    reg = L2_custom(l2=None)
    assert reg.l2 == 0.01


def test_negative_l2_allowed():
    """Test that negative l2 values are allowed (anti-regularization feature)."""
    reg = L2_custom(l2=-0.01)
    assert reg.l2 == -0.01


def test_zero_l2():
    """Test that zero l2 produces zero penalty."""
    reg = L2_custom(l2=0.0)
    weights = tf.random.normal((10, 10), seed=42)
    cost = reg(weights)
    assert abs(float(cost)) < 1e-10


# Validation Tests

def test_validate_float_arg_valid():
    """Test validation with valid float values."""
    assert validate_float_arg(1.0, "test") == 1.0
    assert validate_float_arg(0, "test") == 0.0
    assert validate_float_arg(-1.5, "test") == -1.5


@pytest.mark.parametrize("value", [
    float('inf'),
    float('-inf'),
    float('nan'),
    "string",
    [1.0],
    None,
])
def test_validate_float_arg_invalid(value):
    """Test validation rejects invalid values."""
    with pytest.raises(ValueError):
        validate_float_arg(value, "test")


# Mathematical Correctness Tests

def test_positive_l2_penalty():
    """Test that positive l2 produces expected penalty: l2 * sum(x^2)."""
    reg = L2_custom(l2=0.1)
    weights = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    cost = reg(weights)
    # Expected: 0.1 * (1 + 4 + 9 + 16) = 0.1 * 30 = 3.0
    np.testing.assert_allclose(float(cost), 3.0, atol=1e-6)


def test_negative_l2_reward():
    """Test that negative l2 produces negative penalty (encourages large weights)."""
    reg = L2_custom(l2=-0.1)
    weights = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    cost = reg(weights)
    # Expected: -0.1 * (1 + 4 + 9 + 16) = -0.1 * 30 = -3.0
    np.testing.assert_allclose(float(cost), -3.0, atol=1e-6)


def test_l2_scaling():
    """Test that penalty scales linearly with l2 coefficient."""
    weights = tf.random.normal((10, 10), seed=42)

    reg1 = L2_custom(l2=0.01)
    reg2 = L2_custom(l2=0.02)

    cost1 = reg1(weights)
    cost2 = reg2(weights)

    np.testing.assert_allclose(float(cost2), 2.0 * float(cost1), rtol=1e-5)


def test_penalty_proportional_to_weight_magnitude():
    """Test that larger weights produce larger absolute penalties."""
    reg = L2_custom(l2=0.01)

    small_weights = tf.constant([[0.1, 0.1]], dtype=tf.float32)
    large_weights = tf.constant([[10.0, 10.0]], dtype=tf.float32)

    cost_small = reg(small_weights)
    cost_large = reg(large_weights)

    assert abs(float(cost_large)) > abs(float(cost_small))


# Serialization Tests

def test_config_serialization():
    """Test configuration serialization and deserialization."""
    original = L2_custom(l2=0.05)
    config = original.get_config()

    restored = L2_custom.from_config(config)
    assert restored.l2 == original.l2


def test_negative_l2_serialization():
    """Test that negative l2 survives round-trip serialization."""
    original = L2_custom(l2=-0.001)
    config = original.get_config()
    restored = L2_custom(**config)
    assert restored.l2 == original.l2

    weights = tf.random.normal((5, 5), seed=42)
    np.testing.assert_allclose(
        original(weights).numpy(), restored(weights).numpy(), atol=1e-7
    )


# Keras Integration Tests

def test_keras_integration():
    """Test integration with Keras model."""
    reg = L2_custom(l2=0.01)
    model = Sequential([
        Dense(4, input_shape=(2,), kernel_regularizer=reg)
    ])
    model.compile(optimizer='adam', loss='mse')
    assert model.layers[0].kernel_regularizer is not None


def test_keras_integration_negative_l2():
    """Test Keras integration with negative l2 (anti-regularization)."""
    reg = L2_custom(l2=-0.0001)
    model = Sequential([
        Dense(4, input_shape=(2,), kernel_regularizer=reg)
    ])
    model.compile(optimizer='adam', loss='mse')
    assert model.layers[0].kernel_regularizer is not None


# Gradient Tests

def test_gradient_computation():
    """Test that gradients can be computed through the regularizer."""
    reg = L2_custom(l2=0.01)
    weights = tf.Variable(tf.random.normal((5, 5), seed=42))

    with tf.GradientTape() as tape:
        cost = reg(weights)

    gradients = tape.gradient(cost, weights)
    assert gradients is not None
    assert not tf.reduce_any(tf.math.is_nan(gradients))


def test_gradient_direction_positive_l2():
    """Test that positive l2 gradients push weights toward zero."""
    reg = L2_custom(l2=0.01)
    weights = tf.Variable(tf.constant([[1.0, 2.0]], dtype=tf.float32))

    with tf.GradientTape() as tape:
        cost = reg(weights)

    gradients = tape.gradient(cost, weights)
    # Gradient of l2 * sum(x^2) w.r.t. x is 2*l2*x
    # For positive weights and positive l2, gradient should be positive
    assert tf.reduce_all(gradients > 0)


def test_gradient_direction_negative_l2():
    """Test that negative l2 gradients push weights away from zero."""
    reg = L2_custom(l2=-0.01)
    weights = tf.Variable(tf.constant([[1.0, 2.0]], dtype=tf.float32))

    with tf.GradientTape() as tape:
        cost = reg(weights)

    gradients = tape.gradient(cost, weights)
    # For positive weights and negative l2, gradient should be negative
    # (encouraging weight growth)
    assert tf.reduce_all(gradients < 0)


# Edge Case Tests

def test_numerical_stability():
    """Test regularizer behavior with extreme values."""
    reg = L2_custom(l2=0.01)

    extreme_weights = tf.constant([
        [1e-10, 1e-10],
        [1e5, 1e5],
    ], dtype=tf.float32)

    cost = reg(extreme_weights)
    assert not tf.math.is_nan(cost)
    assert not tf.math.is_inf(cost)


def test_scalar_weight():
    """Test with a single scalar weight."""
    reg = L2_custom(l2=0.1)
    weights = tf.constant([3.0], dtype=tf.float32)
    cost = reg(weights)
    # Expected: 0.1 * 9.0 = 0.9
    np.testing.assert_allclose(float(cost), 0.9, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__])
