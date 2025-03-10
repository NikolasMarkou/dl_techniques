"""
Test suite for CoShNet implementation.

This module contains basic pytest tests to verify the functionality
of the CoShNet model, including initialization, forward pass,
configuration, serialization, and factory function.
"""

import os
import pytest
import numpy as np
import tensorflow as tf
import keras
from keras.api.regularizers import L2

# Import the CoShNet implementation
from dl_techniques.models.coshnet import (
    CoShNet,
    CoShNetConfig,
)

# Constants for tests
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
BATCH_SIZE = 4


@pytest.fixture
def config():
    """Fixture to create a basic CoShNetConfig."""
    return CoShNetConfig(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        conv_filters=[16, 32],
        dense_units=[100, 50],
        dropout_rate=0.1,
        kernel_regularizer=L2(1e-4),
        kernel_initializer=keras.initializers.HeNormal()
    )


@pytest.fixture
def model(config):
    """Fixture to create a CoShNet model with the default config."""
    return CoShNet(config)


@pytest.fixture
def dummy_input():
    """Fixture to create a dummy input tensor."""
    return tf.random.normal((BATCH_SIZE, *INPUT_SHAPE))


def test_model_initialization(config):
    """Test that the model initializes correctly."""
    # Arrange & Act
    model = CoShNet(config)

    # Assert
    assert isinstance(model, CoShNet)
    assert model.config.num_classes == NUM_CLASSES
    assert model.config.conv_filters == [16, 32]
    assert model.config.dense_units == [100, 50]
    assert len(model.conv_layers) == len(config.conv_filters)
    assert len(model.dense_layers) == len(config.dense_units)
    assert isinstance(model.kernel_regularizer, L2)


def test_model_forward_pass(model, dummy_input):
    """Test the forward pass of the model."""
    # Arrange - done through fixtures

    # Act
    output = model(dummy_input, training=True)

    # Assert
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    # Check output is probability distribution
    assert tf.reduce_all(output >= 0)
    assert tf.reduce_all(output <= 1)
    # Sum across classes should be close to 1 (softmax output)
    assert np.allclose(tf.reduce_sum(output, axis=1).numpy(), 1.0, atol=1e-5)


def test_config_serialization(config):
    """Test that the configuration can be serialized and deserialized."""
    # Arrange - done through fixture

    # Act
    config_dict = config.to_dict()
    restored_config = CoShNetConfig(**config_dict)

    # Assert
    assert restored_config.input_shape == config.input_shape
    assert restored_config.num_classes == config.num_classes
    assert restored_config.conv_filters == config.conv_filters
    assert restored_config.dense_units == config.dense_units
    assert restored_config.dropout_rate == config.dropout_rate
    # Note: We don't compare regularizer and initializer objects directly


def test_model_save_load(model, dummy_input, tmp_path):
    """Test saving and loading the model."""
    # Arrange
    save_path = os.path.join(tmp_path, "test_model.keras")

    # Act - Save the model
    model.save_model(save_path)

    # Get original predictions
    original_output = model(dummy_input).numpy()

    # Load the model
    loaded_model = CoShNet.load_model(save_path)
    loaded_output = loaded_model(dummy_input).numpy()

    # Assert
    assert os.path.exists(save_path)
    assert isinstance(loaded_model, CoShNet)
    # Check if the model produces the same output after loading
    assert np.allclose(original_output, loaded_output, atol=1e-5)


