"""
Tests for the Binary Preference Regularizer implementation.

This test suite verifies the functionality of the BinaryPreferenceRegularizer
including its mathematical properties, integration with Keras, and edge cases.
"""

import pytest
import tensorflow as tf
from keras.api.layers import Dense
from keras.api.models import Sequential

from dl_techniques.regularizers.binary_preference import (
    BinaryPreferenceRegularizer,
    get_binary_regularizer
)


@pytest.fixture
def regularizer():
    """Fixture providing a default regularizer instance."""
    return BinaryPreferenceRegularizer(scale=1.0)


def test_regularizer_initialization():
    """Test proper initialization of the regularizer."""
    # Test default initialization
    reg = BinaryPreferenceRegularizer()
    assert reg.scale == 1.0

    # Test custom scale
    reg = BinaryPreferenceRegularizer(scale=2.0)
    assert reg.scale == 2.0


def test_binary_points_zero_cost(regularizer):
    """Test that binary values (0 and 1) produce zero cost."""
    # Create tensor with binary values
    binary_weights = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)
    cost = regularizer(binary_weights)

    # Cost should be very close to zero
    assert tf.abs(cost) < 1e-6


def test_midpoint_maximum_cost(regularizer):
    """Test that weights at 0.5 produce maximum cost."""
    # Create tensor with 0.5 values
    mid_weights = tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32)
    cost = regularizer(mid_weights)

    # Cost should be close to scale value (1.0 in this case)
    assert abs(float(cost) - 1.0) < 1e-6


def test_scaling():
    """Test that scaling factor properly affects the cost."""
    weights = tf.constant([[0.5]], dtype=tf.float32)

    # Test different scales
    reg1 = BinaryPreferenceRegularizer(scale=1.0)
    reg2 = BinaryPreferenceRegularizer(scale=2.0)

    cost1 = reg1(weights)
    cost2 = reg2(weights)

    # Cost should scale linearly
    assert abs(float(cost2) - 2.0 * float(cost1)) < 1e-6


def test_symmetry(regularizer):
    """Test that cost is symmetric around 0.5."""
    # Test pairs of points equidistant from 0.5
    points = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]

    for p1, p2 in points:
        weights1 = tf.constant([[p1]], dtype=tf.float32)
        weights2 = tf.constant([[p2]], dtype=tf.float32)

        cost1 = regularizer(weights1)
        cost2 = regularizer(weights2)

        assert abs(float(cost1) - float(cost2)) < 1e-6


def test_config_serialization():
    """Test configuration serialization and deserialization."""
    original_reg = BinaryPreferenceRegularizer(scale=2.0)
    config = original_reg.get_config()

    # Recreate from config
    new_reg = BinaryPreferenceRegularizer.from_config(config)

    assert new_reg.scale == original_reg.scale


def test_factory_function():
    """Test the get_binary_regularizer factory function."""
    # Test default parameters
    reg1 = get_binary_regularizer()
    assert isinstance(reg1, BinaryPreferenceRegularizer)
    assert reg1.scale == 1.0

    # Test custom scale
    reg2 = get_binary_regularizer(scale=2.0)
    assert reg2.scale == 2.0


def test_keras_integration():
    """Test integration with Keras model."""
    regularizer = get_binary_regularizer(scale=1.0)

    # Create simple model with regularizer
    model = Sequential([
        Dense(4, input_shape=(2,), kernel_regularizer=regularizer)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mse')

    # Should compile without errors
    assert model.layers[0].kernel_regularizer is not None


def test_numerical_stability():
    """Test regularizer behavior with extreme values."""
    regularizer = BinaryPreferenceRegularizer(scale=1.0)

    # Test very large and small values
    extreme_weights = tf.constant([
        [-1e5, 1e5],  # Very large values
        [1e-10, 1 - 1e-10],  # Very close to 0 and 1
        [0.5 - 1e-10, 0.5 + 1e-10]  # Very close to 0.5
    ], dtype=tf.float32)

    # Should not produce NaN or infinity
    cost = regularizer(extreme_weights)
    assert not tf.math.is_nan(cost)
    assert not tf.math.is_inf(cost)


def test_gradient_computation():
    """Test that gradients can be computed through the regularizer."""
    regularizer = BinaryPreferenceRegularizer(scale=1.0)
    weights = tf.Variable([[0.3, 0.7]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        cost = regularizer(weights)

    # Should be able to compute gradients
    gradients = tape.gradient(cost, weights)
    assert gradients is not None
    assert not tf.reduce_any(tf.math.is_nan(gradients))


if __name__ == '__main__':
    pytest.main([__file__])