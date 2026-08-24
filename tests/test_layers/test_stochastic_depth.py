"""
Test suite for StochasticDepth implementation.

This module provides comprehensive tests for:
- StochasticDepth layer initialization
- Training and inference behavior
- Model integration
- Serialization/deserialization
"""

import keras
import pytest
import numpy as np
import tensorflow as tf


from dl_techniques.layers.stochastic_depth import StochasticDepth


# Test fixtures
@pytest.fixture
def sample_input() -> tf.Tensor:
    """Generate sample input tensor."""
    tf.random.set_seed(42)
    return tf.random.normal((4, 16, 16, 32))


@pytest.fixture
def test_model(drop_path_rate: float = 0.5) -> tf.keras.Model:
    """Create a test model with StochasticDepth layer."""
    inputs = tf.keras.Input(shape=(16, 16, 32))
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
    residual = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    residual = StochasticDepth(drop_path_rate=drop_path_rate)(residual)
    outputs = tf.keras.layers.Add()([x, residual])
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Initialization tests
def test_initialization_valid():
    """Test initialization with valid drop_path_rate."""
    valid_rates = [0.0, 0.3, 0.5, 0.7, 0.9999]
    for rate in valid_rates:
        layer = StochasticDepth(drop_path_rate=rate)
        assert layer.drop_path_rate == rate


def test_initialization_invalid():
    """Test initialization with invalid drop_path_rate."""
    invalid_rates = [-0.1, 1.0, 1.1, 2.0]
    for rate in invalid_rates:
        with pytest.raises(ValueError):
            StochasticDepth(drop_path_rate=rate)


# Layer behavior tests
def test_output_shape(sample_input):
    """Test that output shape matches input shape."""
    layer = StochasticDepth(drop_path_rate=0.5)
    output = layer(sample_input)
    assert output.shape == sample_input.shape


def test_training_behavior(sample_input):
    """Test layer behavior in training mode."""
    layer = StochasticDepth(drop_path_rate=0.5)

    # Multiple forward passes should give different results in training
    outputs_training = [
        layer(sample_input, training=True) for _ in range(5)
    ]

    # Check that outputs are different (stochastic behavior)
    outputs_are_different = False
    for i in range(len(outputs_training) - 1):
        if not np.allclose(outputs_training[i], outputs_training[i + 1]):
            outputs_are_different = True
            break
    assert outputs_are_different


def test_inference_behavior(sample_input):
    """Test layer behavior in inference mode."""
    layer = StochasticDepth(drop_path_rate=0.5)

    # Multiple forward passes should be identical in inference mode
    outputs_inference = [
        layer(sample_input, training=False) for _ in range(5)
    ]

    # Check that all outputs are identical
    for i in range(len(outputs_inference) - 1):
        assert np.allclose(outputs_inference[i], outputs_inference[i + 1])


def test_zero_drop_rate(sample_input):
    """Test behavior with drop_path_rate=0."""
    layer = StochasticDepth(drop_path_rate=0.0)
    output = layer(sample_input, training=True)
    assert np.allclose(output.numpy(), sample_input.numpy())


def test_full_drop_rate(sample_input):
    """Test behavior with drop_path_rate=0.99999."""
    # Note: Using value very close to 1 since Keras dropout doesn't accept rate=1.0
    layer = StochasticDepth(drop_path_rate=0.99999)
    output = layer(sample_input, training=True)
    assert np.allclose(output.numpy(), np.zeros_like(output), atol=1e-6)


def test_drop_decision_is_per_sample():
    """Test that the drop decision is drawn independently per sample.

    The distinguishing observable between the implemented per-sample mask
    (``noise_shape = (batch, 1, ..., 1)``) and a batch-wide/scalar mask
    (``noise_shape = (1, 1, ..., 1)``) is a MIXED forward pass: some rows of
    the batch fully zeroed while others survive, in the SAME call. Under a
    batch-wide mask every row shares one draw, so a mixed pass has probability
    exactly 0 and this test fails deterministically.

    The layer draws from ``keras.random.uniform`` without a seed, so a single
    pass is a coin flip; the multi-trial structure below makes a false failure
    negligible. With batch=16, rate=0.5, a trial fails to be mixed only if all
    16 samples are dropped or all 16 are kept:
        P(trial not mixed) = 2 * 0.5**16 = 3.0518e-05
    and over 20 independent trials
        P(no trial is mixed) = (3.0518e-05)**20 ~= 5e-91
    i.e. negligible. (``keras.utils.set_random_seed`` is deliberately NOT
    relied upon here: it does not reproduce a later ``keras.random.*`` draw.)
    """
    batch_size = 16
    rate = 0.5
    trials = 20
    layer = StochasticDepth(drop_path_rate=rate)

    # All-ones input: no zero entries, so "row is all zeros" unambiguously
    # means "this sample's path was dropped", never "the data was zero".
    inputs = tf.ones((batch_size, 4, 4, 8), dtype=tf.float32)
    reduce_axes = (1, 2, 3)

    saw_mixed_batch = False
    for _ in range(trials):
        output = layer(inputs, training=True).numpy()
        row_is_dropped = np.all(output == 0.0, axis=reduce_axes)
        if row_is_dropped.any() and not row_is_dropped.all():
            saw_mixed_batch = True
            # Surviving rows are scaled by 1 / (1 - rate); free to check here
            # since the input is exactly 1.0 everywhere.
            kept = output[~row_is_dropped]
            assert np.allclose(kept, 1.0 / (1.0 - rate), atol=1e-6)
            break

    assert saw_mixed_batch, (
        f"In {trials} training passes with batch={batch_size} and "
        f"drop_path_rate={rate}, no forward pass ever zeroed SOME samples "
        f"while keeping OTHERS. That is impossible for a per-sample mask "
        f"(P ~= 5e-91) and certain for a batch-wide/scalar mask."
    )


# Integration tests
def test_model_integration():
    """Test integration with Keras model."""
    inputs = tf.keras.Input(shape=(16, 16, 32))
    x = StochasticDepth(drop_path_rate=0.5)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    assert isinstance(model.layers[0], tf.keras.layers.InputLayer)
    assert isinstance(model.layers[1], StochasticDepth)

    # Test forward pass
    test_input = tf.random.normal((1, 16, 16, 32))
    output = model(test_input)
    assert output.shape == test_input.shape


def test_training_loop():
    """Test layer in training loop."""
    # Create model
    inputs = tf.keras.Input(shape=(16, 16, 32))
    x = StochasticDepth(drop_path_rate=0.5)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    # Compile model
    model.compile(optimizer='adam', loss='mse')

    # Generate dummy data
    x = tf.random.normal((32, 16, 16, 32))
    y = tf.random.normal((32, 16, 16, 32))

    # Train for one epoch.
    #
    # R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251. `StochasticDepth`
    # owns NO weights, so this one-layer model has none either and Keras says so
    # (`backend/tensorflow/trainer.py:82`). That is exactly the property under
    # test -- the layer must survive a real `fit()` without contributing a
    # parameter -- so the advisory is ASSERTED here rather than suppressed: if
    # `StochasticDepth` ever grows a weight, this goes red instead of silently
    # passing.
    with pytest.warns(UserWarning, match="does not have any trainable weights"):
        history = model.fit(x, y, epochs=1, verbose=0)
    assert 'loss' in history.history


# Serialization tests
def test_serialization():
    """Test layer serialization and deserialization."""
    original_layer = StochasticDepth(drop_path_rate=0.5)
    config = original_layer.get_config()
    recreated_layer = StochasticDepth.from_config(config)

    assert original_layer.drop_path_rate == recreated_layer.drop_path_rate


def test_model_serialization(tmp_path, test_model):
    """Test model serialization with StochasticDepth layer."""

    # Save and load model
    save_path = tmp_path / "test_model.keras"
    test_model.save(save_path)
    loaded_model = keras.models.load_model(save_path)

    # Compare architectures
    assert len(test_model.layers) == len(loaded_model.layers)
    assert isinstance(loaded_model.layers[3], StochasticDepth)


@pytest.mark.parametrize("input_shape", [
    (1, 16, 16, 32),
    (4, 32, 32, 64),
    (8, 8, 8, 128)
])
def test_different_input_shapes(input_shape):
    """Test layer with different input shapes."""
    layer = StochasticDepth(drop_path_rate=0.5)
    input_tensor = tf.random.normal(input_shape)
    output = layer(input_tensor)
    assert output.shape == input_shape


@pytest.mark.parametrize("drop_path_rate", [0.0, 0.3, 0.7, 0.999])
def test_different_drop_rates(drop_path_rate, sample_input):
    """Test layer with different drop rates."""
    layer = StochasticDepth(drop_path_rate=drop_path_rate)
    output = layer(sample_input, training=True)
    assert output.shape == sample_input.shape


def test_learning_phase():
    """Test that layer respects Keras learning phase."""
    layer = StochasticDepth(drop_path_rate=0.5)
    input_tensor = tf.random.normal((4, 16, 16, 32))

    # Test with explicit training flag
    outputs_training = [layer(input_tensor, training=True) for _ in range(5)]
    outputs_inference = [layer(input_tensor, training=False) for _ in range(5)]

    # Verify training outputs are different
    training_outputs_different = not all(
        np.allclose(outputs_training[0], out) for out in outputs_training[1:]
    )

    # Verify inference outputs are same
    inference_outputs_same = all(
        np.allclose(outputs_inference[0], out) for out in outputs_inference[1:]
    )

    assert training_outputs_different
    assert inference_outputs_same

if __name__ == "__main__":
    pytest.main([__file__, "-v"])