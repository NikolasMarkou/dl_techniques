import pytest
import numpy as np
import tensorflow as tf
from itertools import product
from typing import Tuple, Dict, Any
from dl_techniques.layers.band_rms_norm import BandRMSNorm


def test_initialization():
    """Test layer initialization with default and custom parameters."""
    # Test default initialization
    layer = BandRMSNorm()
    assert layer.max_band_width == 0.2
    assert layer.axis == -1
    assert layer.epsilon == 1e-7

    # Test custom initialization
    custom_layer = BandRMSNorm(
        max_band_width=0.1,
        axis=1,
        epsilon=1e-6
    )
    assert custom_layer.max_band_width == 0.1
    assert custom_layer.axis == 1
    assert custom_layer.epsilon == 1e-6


def test_input_validation():
    """Test parameter validation during initialization."""
    # Test invalid max_band_width
    with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
        BandRMSNorm(max_band_width=1.5)
    with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
        BandRMSNorm(max_band_width=-0.1)

    # Test invalid epsilon
    with pytest.raises(ValueError, match="epsilon must be positive"):
        BandRMSNorm(epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        BandRMSNorm(epsilon=-1e-7)


def test_shape_handling():
    """Test layer behavior with different input shapes."""
    test_shapes = [
        (2, 32),  # 2D input
        (3, 16, 8),  # 3D input
        (4, 10, 20, 3)  # 4D input
    ]

    for shape in test_shapes:
        # Create new layer for each test to avoid shape conflicts
        layer = BandRMSNorm()
        inputs = tf.random.normal(shape)
        outputs = layer(inputs)

        # Check shape preservation
        assert outputs.shape == inputs.shape, f"Shape mismatch for input shape {shape}"

        # Check parameter shapes
        assert len(layer.band_param.shape) == len(shape)
        axis = layer.axis if layer.axis >= 0 else len(shape) + layer.axis
        for i in range(len(shape)):
            if i == axis:
                assert layer.band_param.shape[i] == shape[i]
            else:
                assert layer.band_param.shape[i] == 1


def test_normalization_bounds():
    """Test that normalized outputs respect the band constraints."""
    layer = BandRMSNorm(max_band_width=0.2)
    inputs = tf.random.normal((32, 16))
    outputs = layer(inputs)

    # Compute RMS of outputs along the normalization axis
    output_rms = tf.sqrt(tf.reduce_mean(tf.square(outputs), axis=-1))

    # Check if RMS values are within the expected band
    min_allowed = 1.0 - layer.max_band_width
    max_allowed = 1.0

    tf.debugging.assert_greater_equal(output_rms, min_allowed - 1e-5)
    tf.debugging.assert_less_equal(output_rms, max_allowed + 1e-03)


def test_training():
    """Test that the layer is trainable and gradients flow properly."""
    layer = BandRMSNorm()
    inputs = tf.random.normal((16, 8))

    with tf.GradientTape() as tape:
        outputs = layer(inputs)
        loss = tf.reduce_mean(tf.square(outputs))

    # Check if gradients exist and are not None
    grads = tape.gradient(loss, layer.trainable_variables)
    assert len(grads) == 1  # Should have one trainable variable (band_param)
    assert grads[0] is not None  # Gradient should exist

    # Check if band_param is updated during training
    initial_param = tf.identity(layer.band_param)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    optimizer.apply_gradients(zip(grads, layer.trainable_variables))

    # Parameter should have changed after optimization step
    assert not tf.reduce_all(tf.equal(initial_param, layer.band_param))


def test_serialization_and_deserialization():
    """Test layer serialization, deserialization, and config preservation."""
    original_layer = BandRMSNorm(
        max_band_width=0.15,
        axis=1,
        epsilon=1e-6
    )

    # Initialize the layer
    inputs = tf.random.normal((4, 8, 16))
    original_layer(inputs)

    # Get config and create new layer
    config = original_layer.get_config()
    new_layer = BandRMSNorm.from_config(config)

    # Verify config preservation
    assert new_layer.max_band_width == original_layer.max_band_width
    assert new_layer.axis == original_layer.axis
    assert new_layer.epsilon == original_layer.epsilon

    # Compare outputs
    new_outputs = new_layer(inputs)
    original_outputs = original_layer(inputs)
    tf.debugging.assert_near(new_outputs, original_outputs, atol=1e-5)


def test_model_integration():
    """Test layer integration within a Keras model with training."""
    # Create a simple model with RMSNorm
    inputs = tf.keras.Input(shape=(20, 10))
    # Flatten the input for proper dense layer processing
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(16)(x)
    x = BandRMSNorm(max_band_width=0.1)(x)
    x = tf.keras.layers.Dense(8)(x)
    x = BandRMSNorm(max_band_width=0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile and train
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )

    # Generate dummy data
    x_train = tf.random.normal((100, 20, 10))
    y_train = tf.random.normal((100, 1))

    # Train for a few steps
    history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Verify training occurred
    assert len(history.history['loss']) == 5
    assert history.history['loss'][0] > history.history['loss'][-1]

    # Test prediction shape
    predictions = model.predict(x_train[:2], verbose=0)
    assert predictions.shape == (2, 1)


def test_dynamic_batch_size():
    """Test layer behavior with varying batch sizes and dynamic shapes."""
    layer = BandRMSNorm()

    # Test with different batch sizes
    batch_sizes = [1, 4, 16, 32]
    base_shape = (None, 10)  # Dynamic batch size

    # Create model with dynamic batch size
    inputs = tf.keras.Input(shape=base_shape[1:])
    outputs = layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    for batch_size in batch_sizes:
        # Test with current batch size
        inputs = tf.random.normal((batch_size,) + base_shape[1:])
        outputs = model(inputs)

        # Verify shape handling
        assert outputs.shape[0] == batch_size
        assert outputs.shape[1:] == inputs.shape[1:]

        # Verify normalization for each batch item
        for i in range(batch_size):
            batch_output = outputs[i]
            rms = tf.sqrt(tf.reduce_mean(tf.square(batch_output)))
            tf.debugging.assert_near(rms, 1.0, atol=0.2)  # Within band width


@pytest.mark.parametrize(
    "input_shape, max_band_width",
    [
        ((32, 16, 64), 0.2),  # 3D input
        ((8, 16, 32, 3), 0.3),  # 4D input (like images)
        ((16, 8, 8, 16, 4), 0.1),  # 5D input
    ]
)
def test_higher_dimensional_inputs(
        input_shape: Tuple[int, ...],
        max_band_width: float
) -> None:
    """Test that layer works correctly with higher dimensional inputs.

    Args:
        input_shape: Shape of input tensor
        max_band_width: Maximum allowed deviation from unit norm
    """
    layer = BandRMSNorm(max_band_width=max_band_width)
    inputs = tf.random.normal(input_shape)
    outputs = layer(inputs)

    # Verify output shape
    assert outputs.shape == input_shape

    # Check normalization bounds
    output_rms = tf.sqrt(tf.reduce_mean(tf.square(outputs), axis=-1))
    min_allowed = 1.0 - max_band_width
    max_allowed = 1.0

    tf.debugging.assert_greater_equal(output_rms, min_allowed - 1e-5)
    tf.debugging.assert_less_equal(output_rms, max_allowed + 1e-5)


@pytest.mark.parametrize(
    "config",
    [
        {"max_band_width": 0},  # Zero band width
        {"max_band_width": 1.0},  # Max band width too large
        {"max_band_width": -0.1},  # Negative band width
        {"epsilon": 0},  # Zero epsilon
        {"epsilon": -1e-7},  # Negative epsilon
    ]
)
def test_invalid_configurations(config: Dict[str, Any]) -> None:
    """Test that layer correctly handles invalid configurations.

    Args:
        config: Dictionary of invalid configuration parameters
    """
    with pytest.raises(ValueError):
        _ = BandRMSNorm(**config)


if __name__ == '__main__':
    pytest.main([__file__])
