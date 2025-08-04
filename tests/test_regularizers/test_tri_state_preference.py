"""
Tests for the Tri-State Preference Regularizer implementation.

This test suite verifies the functionality of the TriStatePreferenceRegularizer
including its mathematical properties, integration with Keras, and edge cases.
The regularizer should encourage weights to converge to -1, 0, or 1.
"""

import pytest
import tensorflow as tf
from keras.api.layers import Dense
from keras.api.models import Sequential
from dl_techniques.regularizers.tri_state_preference import TriStatePreferenceRegularizer


@pytest.fixture
def regularizer():
    """Fixture providing a default regularizer instance."""
    return TriStatePreferenceRegularizer(scale=1.0)


def test_regularizer_initialization():
    """Test proper initialization of the regularizer."""
    # Test default initialization
    reg = TriStatePreferenceRegularizer()
    assert reg.scale == 1.0
    assert abs(reg.base_coefficient - 32.0 / 4.5) < 1e-6

    # Test custom scale
    reg = TriStatePreferenceRegularizer(scale=2.0)
    assert reg.scale == 2.0


def test_stable_points_zero_cost(regularizer):
    """Test that stable points (-1, 0, 1) produce zero cost."""
    # Create tensor with stable points
    stable_weights = tf.constant([[-1.0, 0.0, 1.0]], dtype=tf.float32)
    cost = regularizer(stable_weights)

    # Cost should be very close to zero
    assert tf.abs(cost) < 1e-6


def test_local_maxima(regularizer):
    """Test that weights at ±0.5 produce local maxima of 1.0."""
    # Create tensor with values at local maxima
    max_weights = tf.constant([[-0.5, 0.5]], dtype=tf.float32)
    cost = regularizer(max_weights)

    # Cost should be close to scale value (1.0 in this case)
    assert abs(float(cost) - 1.0) < 1e-6


def test_multiplier():
    """Test that multiplier factor properly affects the cost."""
    weights = tf.constant([[0.5]], dtype=tf.float32)

    # Test different scales
    reg1 = TriStatePreferenceRegularizer(multiplier=1.0)
    reg2 = TriStatePreferenceRegularizer(multiplier=2.0)

    cost1 = reg1(weights)
    cost2 = reg2(weights)

    # Cost should scale linearly
    assert abs(float(cost2) - 2.0 * float(cost1)) < 1e-6


def test_symmetry(regularizer):
    """Test that cost function is symmetric around x=0."""
    # Test pairs of points symmetric around 0
    points = [
        (-0.8, 0.8),
        (-0.5, 0.5),
        (-0.3, 0.3),
        (-0.1, 0.1)
    ]

    for p1, p2 in points:
        weights1 = tf.constant([[p1]], dtype=tf.float32)
        weights2 = tf.constant([[p2]], dtype=tf.float32)

        cost1 = regularizer(weights1)
        cost2 = regularizer(weights2)

        assert abs(float(cost1) - float(cost2)) < 1e-6


def test_monotonicity_regions():
    """Test monotonicity in regions between stable points and beyond."""
    regularizer = TriStatePreferenceRegularizer()

    # Test increasing cost from -1 to -0.5
    x1 = tf.constant([[-0.9]], dtype=tf.float32)
    x2 = tf.constant([[-0.7]], dtype=tf.float32)
    assert float(regularizer(x1)) < float(regularizer(x2))

    # Test decreasing cost from -0.5 to 0
    x3 = tf.constant([[-0.4]], dtype=tf.float32)
    x4 = tf.constant([[-0.2]], dtype=tf.float32)
    assert float(regularizer(x3)) > float(regularizer(x4))

    # Test increasing cost beyond |1|
    x5 = tf.constant([[1.0]], dtype=tf.float32)
    x6 = tf.constant([[1.5]], dtype=tf.float32)
    assert float(regularizer(x5)) < float(regularizer(x6))


def test_config_serialization():
    """Test configuration serialization and deserialization."""
    original_reg = TriStatePreferenceRegularizer(scale=2.0)
    config = original_reg.get_config()

    # Recreate from config
    new_reg = TriStatePreferenceRegularizer.from_config(config)

    assert new_reg.scale == original_reg.scale
    assert new_reg.base_coefficient == original_reg.base_coefficient



def test_keras_integration():
    """Test integration with Keras model."""
    regularizer = TriStatePreferenceRegularizer(scale=1.0)

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
    regularizer = TriStatePreferenceRegularizer(scale=1.0)

    # Test very large and small values
    extreme_weights = tf.constant([
        [-1e5, 1e5],  # Very large values
        [-1 + 1e-10, 1 - 1e-10],  # Very close to -1 and 1
        [-1e-10, 1e-10],  # Very close to 0
        [-0.5 - 1e-10, 0.5 + 1e-10]  # Very close to local maxima
    ], dtype=tf.float32)

    # Should not produce NaN or infinity
    cost = regularizer(extreme_weights)
    assert not tf.math.is_nan(cost)
    assert not tf.math.is_inf(cost)


def test_gradient_computation():
    """Test that gradients can be computed through the regularizer."""
    regularizer = TriStatePreferenceRegularizer(scale=1.0)
    weights = tf.Variable([[-0.3, 0.0, 0.7]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        cost = regularizer(weights)

    # Should be able to compute gradients
    gradients = tape.gradient(cost, weights)
    assert gradients is not None
    assert not tf.reduce_any(tf.math.is_nan(gradients))


if __name__ == '__main__':
    pytest.main([__file__])