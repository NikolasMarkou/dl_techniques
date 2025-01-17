import keras
import pytest
import numpy as np
import tensorflow as tf
from typing import Type, Tuple

from dl_techniques.layers.mish import Mish, ScaledMish


@pytest.fixture
def input_data() -> Tuple[tf.Tensor, ...]:
    """Fixture providing various input tensors for testing."""
    return (
        tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32),  # Regular cases
        tf.constant([-1e5, 1e5], dtype=tf.float32),  # Large values
        tf.constant([-1e-5, 1e-5], dtype=tf.float32),  # Small values
        tf.constant([np.nan], dtype=tf.float32),  # NaN
        tf.constant([np.inf, -np.inf], dtype=tf.float32),  # Infinity
    )


class TestMish:
    """Test suite for Mish activation layer."""

    def test_initialization(self):
        """Test proper initialization of Mish layer."""
        layer = Mish()
        assert isinstance(layer, keras.layers.Layer)

    @pytest.mark.parametrize("shape", [
        (1,), (2, 3), (2, 3, 4), (2, 3, 4, 5)
    ])
    def test_shape_inference(self, shape):
        """Test if output shape matches input shape for various dimensions."""
        layer = Mish()
        inputs = tf.random.normal(shape)
        outputs = layer(inputs)
        assert outputs.shape == inputs.shape

    def test_regular_values(self, input_data):
        """Test activation for regular input values."""
        layer = Mish()
        regular_inputs = input_data[0]
        outputs = layer(regular_inputs)

        # Test specific known values
        expected = regular_inputs * tf.math.tanh(tf.math.softplus(regular_inputs))
        tf.debugging.assert_near(outputs, expected, rtol=1e-5)

    def test_numerical_stability(self, input_data):
        """Test activation behavior with extreme values."""
        layer = Mish()
        large_inputs = input_data[1]
        small_inputs = input_data[2]

        # Test large values
        large_outputs = layer(large_inputs)
        assert tf.reduce_all(tf.math.is_finite(large_outputs))

        # Test small values
        small_outputs = layer(small_inputs)
        assert tf.reduce_all(tf.math.is_finite(small_outputs))

    def test_gradient(self):
        """Test if gradients are computed correctly."""
        layer = Mish()
        x = tf.Variable([1.0, 2.0, 3.0])

        with tf.GradientTape() as tape:
            y = layer(x)

        gradients = tape.gradient(y, x)
        assert gradients is not None
        assert tf.reduce_all(tf.math.is_finite(gradients))

    def test_serialization(self):
        """Test serialization and deserialization."""
        layer = Mish()
        config = layer.get_config()
        new_layer = Mish.from_config(config)
        assert isinstance(new_layer, Mish)

    def test_invalid_inputs(self, input_data):
        """Test behavior with invalid inputs."""
        layer = Mish()
        nan_inputs = input_data[3]
        inf_inputs = input_data[4]

        # NaN should propagate
        nan_outputs = layer(nan_inputs)
        assert tf.math.is_nan(nan_outputs[0])


class TestScaledMish:
    """Test suite for ScaledMish activation layer."""

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_initialization(self, alpha):
        """Test proper initialization with different alpha values."""
        layer = ScaledMish(alpha=alpha)
        assert isinstance(layer, keras.layers.Layer)
        assert layer._alpha == alpha

    def test_invalid_alpha(self):
        """Test if invalid alpha values raise appropriate errors."""
        with pytest.raises(ValueError):
            ScaledMish(alpha=0.0)
        with pytest.raises(ValueError):
            ScaledMish(alpha=-1.0)

    @pytest.mark.parametrize("shape", [
        (1,), (2, 3), (2, 3, 4), (2, 3, 4, 5)
    ])
    def test_shape_inference(self, shape):
        """Test if output shape matches input shape for various dimensions."""
        layer = ScaledMish()
        inputs = tf.random.normal(shape)
        outputs = layer(inputs)
        assert outputs.shape == inputs.shape

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_bounds(self, alpha):
        """Test if outputs are properly bounded by alpha."""
        layer = ScaledMish(alpha=alpha)
        large_inputs = tf.constant([-1e5, 1e5], dtype=tf.float32)
        outputs = layer(large_inputs)

        assert tf.reduce_all(tf.abs(outputs) <= alpha * 1.01)  # Allow for small numerical errors

    def test_gradient(self):
        """Test if gradients are computed correctly."""
        layer = ScaledMish(alpha=1.0)
        x = tf.Variable([1.0, 2.0, 3.0])

        with tf.GradientTape() as tape:
            y = layer(x)

        gradients = tape.gradient(y, x)
        assert gradients is not None
        assert tf.reduce_all(tf.math.is_finite(gradients))

    def test_serialization(self):
        """Test serialization and deserialization."""
        layer = ScaledMish(alpha=2.0)
        config = layer.get_config()
        new_layer = ScaledMish.from_config(config)
        assert isinstance(new_layer, ScaledMish)
        assert new_layer._alpha == 2.0

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_numerical_stability(self, alpha, input_data):
        """Test activation behavior with extreme values."""
        layer = ScaledMish(alpha=alpha)
        large_inputs = input_data[1]
        small_inputs = input_data[2]

        # Test large values
        large_outputs = layer(large_inputs)
        assert tf.reduce_all(tf.math.is_finite(large_outputs))

        # Test small values
        small_outputs = layer(small_inputs)
        assert tf.reduce_all(tf.math.is_finite(small_outputs))


def test_compare_with_mish():
    """Test that ScaledMish with alpha=1 approximates Mish near zero."""
    inputs = tf.constant([-0.1, 0.0, 0.1], dtype=tf.float32)
    mish_layer = Mish()
    scaled_mish_layer = ScaledMish(alpha=1.0)

    mish_outputs = mish_layer(inputs)
    scaled_mish_outputs = scaled_mish_layer(inputs)

    # Should be very close for small values
    tf.debugging.assert_near(mish_outputs, scaled_mish_outputs, rtol=1e-2)


# Integration test
def test_in_model():
    """Test both activation layers in a simple model."""
    model = keras.Sequential([
        keras.layers.Dense(10),
        Mish(),
        keras.layers.Dense(10),
        ScaledMish(alpha=1.0),
        keras.layers.Dense(1)
    ])

    # Test forward pass
    inputs = tf.random.normal((32, 10))
    outputs = model(inputs)
    assert outputs.shape == (32, 1)

    # Test training step
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = tf.reduce_mean(outputs ** 2)

    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in gradients)
    assert all(tf.reduce_all(tf.math.is_finite(g)) for g in gradients)