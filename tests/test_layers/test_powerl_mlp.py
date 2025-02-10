"""Test suite for PowerMLP implementation."""

import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.powel_mlp import (
    ModelConfig,
    ReLUK,
    BasisFunction,
    PowerMLPLayer,
    PowerMLP,
)


@pytest.fixture
def model_config():
    """Fixture providing default model configuration."""
    return ModelConfig(
        hidden_units=[64, 32, 10],
        k=3,
        weight_decay=0.0001,
        output_activation="softmax"
    )


@pytest.fixture
def sample_input():
    """Fixture providing sample input tensor."""
    return tf.random.normal((32, 784))


def test_relu_k():
    """Test ReLU-k activation function."""
    layer = ReLUK(k=2)
    x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = layer(x)

    expected = tf.constant([0.0, 0.0, 0.0, 1.0, 4.0])
    tf.debugging.assert_near(y, expected)


def test_basis_function():
    """Test basis function layer."""
    layer = BasisFunction()
    x = tf.constant([0.0])
    y = layer(x)

    # At x=0, output should be 0
    tf.debugging.assert_near(y, tf.constant([0.0]))


def test_power_mlp_layer():
    """Test PowerMLP layer."""
    layer = PowerMLPLayer(units=32, k=3)
    x = tf.random.normal((16, 64))
    y = layer(x)

    assert y.shape == (16, 32)


def test_power_mlp_model(model_config, sample_input):
    """Test full PowerMLP model."""
    model = PowerMLP(model_config)
    y = model(sample_input)

    assert y.shape == (32, 10)
    tf.debugging.assert_less_equal(tf.reduce_max(y), 1.0)
    tf.debugging.assert_greater_equal(tf.reduce_min(y), 0.0)


def test_model_training(model_config):
    """Test model training."""
    # Create synthetic dataset
    x_train = tf.random.normal((100, 784))
    y_train = tf.random.uniform((100,), maxval=10, dtype=tf.int32)

    # Create and compile model
    model = PowerMLP(model_config)
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    # Train for one epoch
    history = model.fit(x_train, y_train, epochs=1)

    assert "loss" in history.history
    assert "accuracy" in history.history


if __name__ == "__main__":
    pytest.main([__file__])