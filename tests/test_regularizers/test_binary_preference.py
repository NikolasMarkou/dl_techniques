"""
Tests for the Binary Preference Regularizer implementation.

This test suite verifies the functionality of the BinaryPreferenceRegularizer
including its mathematical properties, integration with Keras, and edge cases.
"""

import pytest
import tensorflow as tf
from keras.api.layers import Dense, Input
from keras.api.models import Sequential

from dl_techniques.regularizers.binary_preference import (
    BinaryPreferenceRegularizer
)


@pytest.fixture
def regularizer():
    """Fixture providing a default regularizer instance."""
    return BinaryPreferenceRegularizer(multiplier=1.0)


def test_regularizer_initialization():
    """Test proper initialization of the regularizer."""
    # Test default initialization
    reg = BinaryPreferenceRegularizer()
    assert reg.multiplier == 1.0

    # Test custom scale
    reg = BinaryPreferenceRegularizer(multiplier=2.0)
    assert reg.multiplier == 2.0


def test_binary_points_zero_cost(regularizer):
    """Test that binary values (0 and 1) produce zero cost."""
    # Create tensor with binary values
    binary_weights = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)
    cost = regularizer(binary_weights)

    # Cost should be very close to zero
    assert tf.abs(cost) < 1e-6


@pytest.mark.parametrize(
    "reduction, expected",
    [
        # Derivation (from the module docstring's published formula):
        #   L(w) = m * (w - low)^2 * (w - high)^2 / h^4,  h = (high - low) / 2
        # At the midpoint w = (low + high) / 2 each of the two differences is
        # +/- h, so the numerator is h^2 * h^2 = h^4 and the per-element penalty
        # is EXACTLY m. The barrier height is the multiplier itself,
        # independent of low/high.
        # For the default low=0, high=1: h = 0.5, h^4 = 0.0625; at w = 0.5
        #   (0.5)^2 * (-0.5)^2 / 0.0625 = 0.25 * 0.25 / 0.0625 = 1.0 per element.
        # The tensor is 2x2, so N = 4 elements.
        #   reduction="sum"  -> m * N = 1.0 * 4 = 4.0
        ("sum", 4.0),
        #   reduction="mean" -> m       = 1.0   (what the old test really meant;
        #                                        `mean` was the implicit default)
        ("mean", 1.0),
    ],
)
def test_midpoint_maximum_cost(reduction, expected):
    """Pin the midpoint barrier under BOTH reductions.

    The default reduction changed from an implicit `mean` to `sum`, so a
    "cost at the maximum == 1.0" assertion now scales with the element count.
    """
    reg = BinaryPreferenceRegularizer(multiplier=1.0, reduction=reduction)
    mid_weights = tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32)
    assert int(tf.size(mid_weights)) == 4

    cost = float(reg(mid_weights))
    assert abs(cost - expected) < 1e-6


@pytest.mark.parametrize(
    "low, high, multiplier",
    [
        # Symmetric kernel targets: h = (1 - (-1)) / 2 = 1, h^4 = 1.
        # Midpoint w = 0: (0 - (-1))^2 * (0 - 1)^2 / 1 = 1 * 1 / 1 = 1
        # per element, times m = 2.0 -> 2.0.
        (-1.0, 1.0, 2.0),
        # Narrow gate targets: h = (0.75 - 0.25) / 2 = 0.25, h^4 = 0.00390625.
        # Midpoint w = 0.5: (0.25)^2 * (-0.25)^2 = 0.00390625, and dividing by
        # h^4 gives exactly 1 per element, times m = 0.5 -> 0.5.
        (0.25, 0.75, 0.5),
    ],
)
def test_barrier_height_equals_multiplier_for_any_targets(low, high, multiplier):
    """The barrier height is `multiplier`, independent of `low` and `high`.

    The h^4 divisor exists precisely to cancel the gap width: at the midpoint
    the numerator is always h^4, so L(midpoint) = m for every (low, high).
    Checked at two non-default triples so a residual gap-width dependence
    could not hide behind the canonical low=0, high=1 case. `mean` is used so
    the value read is the per-element barrier rather than N times it.
    """
    reg = BinaryPreferenceRegularizer(
        multiplier=multiplier, low=low, high=high, reduction="mean"
    )
    midpoint = (low + high) / 2.0
    weights = tf.constant([[midpoint, midpoint]], dtype=tf.float32)

    assert abs(float(reg(weights)) - multiplier) < 1e-6


def test_scaling():
    """Test that scaling factor properly affects the cost."""
    weights = tf.constant([[0.5]], dtype=tf.float32)

    # Test different scales
    reg1 = BinaryPreferenceRegularizer(multiplier=1.0)
    reg2 = BinaryPreferenceRegularizer(multiplier=2.0)

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
    original_reg = BinaryPreferenceRegularizer(multiplier=2.0)
    config = original_reg.get_config()

    # Recreate from config
    new_reg = BinaryPreferenceRegularizer.from_config(config)

    assert new_reg.multiplier == original_reg.multiplier


def test_keras_integration():
    """Test integration with Keras model."""
    regularizer = BinaryPreferenceRegularizer(scale=1.0)

    # Create simple model with regularizer
    model = Sequential([
        Input(shape=(2,)),
        Dense(4, kernel_regularizer=regularizer)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mse')

    # Should compile without errors
    assert model.layers[0].kernel_regularizer is not None


def test_numerical_stability():
    """Test regularizer behavior with extreme values."""
    regularizer = BinaryPreferenceRegularizer(multiplier=1.0)

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
    regularizer = BinaryPreferenceRegularizer(multiplier=1.0)
    weights = tf.Variable([[0.3, 0.7]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        cost = regularizer(weights)

    # Should be able to compute gradients
    gradients = tape.gradient(cost, weights)
    assert gradients is not None
    assert not tf.reduce_any(tf.math.is_nan(gradients))


if __name__ == '__main__':
    pytest.main([__file__])