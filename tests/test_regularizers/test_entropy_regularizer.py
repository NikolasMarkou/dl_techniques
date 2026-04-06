"""
Tests for the Entropy Regularizer implementation.

This test suite verifies the functionality of the EntropyRegularizer
including its mathematical properties, integration with Keras, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
from keras.api.layers import Dense
from keras.api.models import Sequential

from dl_techniques.regularizers.entropy_regularizer import (
    EntropyRegularizer,
    create_entropy_regularizer,
    DEFAULT_ENTROPY_STRENGTH,
    DEFAULT_TARGET_ENTROPY,
    DEFAULT_ENTROPY_EPSILON,
    ENTROPY_LOW,
    ENTROPY_MEDIUM,
    ENTROPY_HIGH,
)


@pytest.fixture
def regularizer():
    """Fixture providing a default regularizer instance."""
    return EntropyRegularizer()


# Initialization Tests

def test_default_initialization():
    """Test proper initialization with default parameters."""
    reg = EntropyRegularizer()
    assert reg.strength == DEFAULT_ENTROPY_STRENGTH
    assert reg.target_entropy == DEFAULT_TARGET_ENTROPY
    assert reg.axis == -1
    assert reg.epsilon == DEFAULT_ENTROPY_EPSILON


def test_custom_initialization():
    """Test initialization with custom parameters."""
    reg = EntropyRegularizer(strength=0.05, target_entropy=0.3, axis=0, epsilon=1e-8)
    assert reg.strength == 0.05
    assert reg.target_entropy == 0.3
    assert reg.axis == 0
    assert reg.epsilon == 1e-8


@pytest.mark.parametrize("params,error_match", [
    ({"strength": -0.1}, "strength must be non-negative"),
    ({"target_entropy": -0.1}, "target_entropy must be in"),
    ({"target_entropy": 1.5}, "target_entropy must be in"),
    ({"epsilon": 0.0}, "epsilon must be positive"),
    ({"epsilon": -1e-10}, "epsilon must be positive"),
])
def test_invalid_initialization(params, error_match):
    """Test that invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match=error_match):
        EntropyRegularizer(**params)


# Mathematical Property Tests

def test_uniform_weights_high_entropy(regularizer):
    """Test that uniform weights produce high normalized entropy."""
    # Uniform weights should have entropy near 1.0
    uniform_weights = tf.ones((1, 100), dtype=tf.float32)
    cost = regularizer(uniform_weights)
    # With default target_entropy=0.7 and uniform weights (entropy~1.0),
    # the penalty should be non-zero
    assert not tf.math.is_nan(cost)
    assert cost >= 0


def test_concentrated_weights_low_entropy():
    """Test that concentrated weights produce low normalized entropy."""
    # One dominant weight, rest near zero -> low entropy
    reg_low = EntropyRegularizer(strength=1.0, target_entropy=0.0)
    reg_high = EntropyRegularizer(strength=1.0, target_entropy=1.0)

    # Create weights where one value dominates
    weights = tf.constant([[10.0, 0.01, 0.01, 0.01, 0.01]], dtype=tf.float32)

    cost_low_target = reg_low(weights)
    cost_high_target = reg_high(weights)

    # Concentrated weights should have lower penalty with low target
    assert cost_low_target < cost_high_target


def test_strength_scaling():
    """Test that strength parameter scales the penalty linearly."""
    weights = tf.random.normal((10, 10), seed=42)

    reg1 = EntropyRegularizer(strength=0.01, target_entropy=0.5)
    reg2 = EntropyRegularizer(strength=0.02, target_entropy=0.5)

    cost1 = reg1(weights)
    cost2 = reg2(weights)

    # Cost should scale linearly with strength
    np.testing.assert_allclose(float(cost2), 2.0 * float(cost1), rtol=1e-5)


def test_zero_strength_zero_cost():
    """Test that zero strength produces zero cost."""
    reg = EntropyRegularizer(strength=0.0, target_entropy=0.5)
    weights = tf.random.normal((10, 10), seed=42)
    cost = reg(weights)
    assert abs(float(cost)) < 1e-10


def test_penalty_is_non_negative(regularizer):
    """Test that penalty is always non-negative."""
    for seed in range(5):
        weights = tf.random.normal((20, 20), seed=seed)
        cost = regularizer(weights)
        assert float(cost) >= 0.0


# Serialization Tests

def test_config_serialization():
    """Test configuration serialization and deserialization."""
    original = EntropyRegularizer(
        strength=0.05, target_entropy=0.3, axis=0, epsilon=1e-8
    )
    config = original.get_config()

    restored = EntropyRegularizer.from_config(config)
    assert restored.strength == original.strength
    assert restored.target_entropy == original.target_entropy
    assert restored.axis == original.axis
    assert restored.epsilon == original.epsilon


def test_serialization_roundtrip_identical_output():
    """Test that round-trip serialization produces identical loss."""
    original = EntropyRegularizer(strength=0.05, target_entropy=0.4)
    config = original.get_config()
    restored = EntropyRegularizer(**config)

    weights = tf.random.normal((10, 10), seed=42)
    original_loss = original(weights)
    restored_loss = restored(weights)
    np.testing.assert_allclose(
        original_loss.numpy(), restored_loss.numpy(), atol=1e-7,
        err_msg="Round-trip serialization must produce identical loss"
    )


# Keras Integration Tests

def test_keras_integration():
    """Test integration with Keras model."""
    regularizer = EntropyRegularizer(strength=0.01)

    model = Sequential([
        Dense(4, input_shape=(2,), kernel_regularizer=regularizer)
    ])
    model.compile(optimizer='adam', loss='mse')

    assert model.layers[0].kernel_regularizer is not None


# Gradient Tests

def test_gradient_computation():
    """Test that gradients can be computed through the regularizer."""
    regularizer = EntropyRegularizer(strength=0.01)
    weights = tf.Variable(tf.random.normal((5, 5), seed=42))

    with tf.GradientTape() as tape:
        cost = regularizer(weights)

    gradients = tape.gradient(cost, weights)
    assert gradients is not None
    assert not tf.reduce_any(tf.math.is_nan(gradients))


# Edge Case Tests

def test_numerical_stability():
    """Test regularizer behavior with extreme values."""
    regularizer = EntropyRegularizer(strength=0.01)

    extreme_weights = tf.constant([
        [1e-10, 1e-10, 1e-10],
        [1e5, 1e5, 1e5],
    ], dtype=tf.float32)

    cost = regularizer(extreme_weights)
    assert not tf.math.is_nan(cost)
    assert not tf.math.is_inf(cost)


def test_small_weight_vector():
    """Test with a small but valid weight vector."""
    regularizer = EntropyRegularizer(strength=0.01)
    # Two weights is the minimum for meaningful entropy
    weights = tf.constant([[1.0, 2.0]], dtype=tf.float32)
    cost = regularizer(weights)
    assert not tf.math.is_nan(cost)
    assert not tf.math.is_inf(cost)


# Factory Function Tests

def test_factory_default():
    """Test factory with default parameters."""
    reg = create_entropy_regularizer()
    assert reg.target_entropy == DEFAULT_TARGET_ENTROPY
    assert reg.strength == DEFAULT_ENTROPY_STRENGTH


@pytest.mark.parametrize("mode,expected_target", [
    ("low", ENTROPY_LOW),
    ("medium", ENTROPY_MEDIUM),
    ("high", ENTROPY_HIGH),
])
def test_factory_modes(mode, expected_target):
    """Test factory with predefined modes."""
    reg = create_entropy_regularizer(mode=mode)
    assert reg.target_entropy == expected_target


def test_factory_custom_target():
    """Test factory with custom target_entropy."""
    reg = create_entropy_regularizer(target_entropy=0.6)
    assert reg.target_entropy == 0.6


def test_factory_target_overrides_mode():
    """Test that explicit target_entropy overrides mode."""
    reg = create_entropy_regularizer(target_entropy=0.6, mode="low")
    assert reg.target_entropy == 0.6


def test_factory_invalid_mode():
    """Test factory with invalid mode."""
    with pytest.raises(ValueError, match="Invalid mode"):
        create_entropy_regularizer(mode="invalid")


def test_factory_invalid_strength():
    """Test factory with invalid strength."""
    with pytest.raises(ValueError, match="strength must be non-negative"):
        create_entropy_regularizer(strength=-0.1)


if __name__ == '__main__':
    pytest.main([__file__])
