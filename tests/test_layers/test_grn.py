import pytest
import numpy as np
import keras
import tensorflow as tf
from typing import Tuple, Any, List
from dataclasses import dataclass
import json
import tempfile
import os

# Import the layer to test
from dl_techniques.layers.global_response_norm import GlobalResponseNormalization  # Update with correct import


@dataclass
class TestCase:
    """Data class for test cases"""
    input_shape: Tuple[int, ...]
    eps: float
    gamma_init: Any
    beta_init: Any
    expected_error: Any = None
    description: str = ""


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility"""
    np.random.seed(42)
    tf.random.set_seed(42)


@pytest.fixture
def valid_test_cases() -> List[TestCase]:
    """Generate valid test cases"""
    return [
        TestCase(
            input_shape=(2, 16, 16, 32),
            eps=1e-6,
            gamma_init='ones',
            beta_init='zeros',
            description="Basic case with default initializers"
        ),
        TestCase(
            input_shape=(1, 32, 32, 64),
            eps=1e-5,
            gamma_init=keras.initializers.Ones(),
            beta_init=keras.initializers.Zeros(),
            description="Single batch with custom initializers"
        ),
        TestCase(
            input_shape=(4, 8, 8, 16),
            eps=1e-4,
            gamma_init=keras.initializers.RandomNormal(seed=42),
            beta_init=keras.initializers.RandomNormal(seed=42),
            description="Random normal initializers"
        )
    ]


@pytest.fixture
def invalid_test_cases() -> List[TestCase]:
    """Generate invalid test cases"""
    return [
        TestCase(
            input_shape=(16, 16, 32),  # Missing batch dimension
            eps=1e-6,
            gamma_init='ones',
            beta_init='zeros',
            expected_error=ValueError,
            description="Invalid 3D input"
        ),
        TestCase(
            input_shape=(2, 16, 16, None),  # None channels
            eps=1e-6,
            gamma_init='ones',
            beta_init='zeros',
            expected_error=ValueError,
            description="Undefined channels"
        ),
        TestCase(
            input_shape=(2, 16, 16, 32),
            eps=-1e-6,  # Negative epsilon
            gamma_init='ones',
            beta_init='zeros',
            expected_error=ValueError,
            description="Negative epsilon"
        )
    ]


def test_layer_creation():
    """Test basic layer creation"""
    layer = GlobalResponseNormalization()
    assert isinstance(layer, keras.layers.Layer)
    assert layer.eps == 1e-6


def test_build_weights(random_seed):
    """Test weight creation during build"""
    layer = GlobalResponseNormalization()
    input_shape = (2, 16, 16, 32)
    layer.build(input_shape)

    assert layer.built
    assert layer.gamma is not None
    assert layer.beta is not None
    assert layer.gamma.shape == (1, 1, 1, 32)
    assert layer.beta.shape == (1, 1, 1, 32)


def test_valid_cases(valid_test_cases, random_seed):
    """Test layer with valid test cases"""
    for test_case in valid_test_cases:
        layer = GlobalResponseNormalization(
            eps=test_case.eps,
            gamma_initializer=test_case.gamma_init,
            beta_initializer=test_case.beta_init
        )

        inputs = np.random.randn(*test_case.input_shape).astype(np.float32)
        output = layer(inputs)

        assert output.shape == test_case.input_shape
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


def test_serialization(random_seed):
    """Test serialization and deserialization"""
    layer = GlobalResponseNormalization(
        eps=1e-5,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )

    # Test get_config and from_config
    config = layer.get_config()
    restored_layer = GlobalResponseNormalization.from_config(config)

    assert layer.eps == restored_layer.eps
    assert isinstance(restored_layer.gamma_initializer, keras.initializers.Initializer)
    assert isinstance(restored_layer.beta_initializer, keras.initializers.Initializer)


def test_model_integration(random_seed):
    """Test integration with Keras model"""
    input_shape = (16, 16, 32)
    inputs = keras.layers.Input(shape=input_shape)
    grn = GlobalResponseNormalization()(inputs)
    model = keras.Model(inputs, grn)

    # Test forward pass
    test_input = np.random.randn(2, *input_shape).astype(np.float32)
    output = model.predict(test_input)

    assert output.shape == (2, *input_shape)

    # Test save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.keras")
        model.save(model_path)
        loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model.predict(test_input)
        np.testing.assert_allclose(
            output,
            loaded_output,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Loaded model output should match original"
        )


def test_gradient_flow(random_seed):
    """Test gradient flow through the layer"""
    layer = GlobalResponseNormalization()
    input_shape = (2, 8, 8, 16)

    with tf.GradientTape() as tape:
        inputs = tf.random.normal(input_shape)
        tape.watch(inputs)
        outputs = layer(inputs)
        loss = tf.reduce_mean(outputs)

    gradients = tape.gradient(loss, inputs)
    assert gradients is not None
    assert not np.any(np.isnan(gradients))
    assert not np.any(np.isinf(gradients))


def test_training_loop(random_seed):
    """Test layer in training loop"""
    model = keras.Sequential([
        keras.layers.Input(shape=(16, 16, 32)),
        GlobalResponseNormalization(),
        keras.layers.Conv2D(64, 3, padding='same'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # Generate dummy data
    x = np.random.randn(32, 16, 16, 32).astype(np.float32)
    y = np.random.randn(32, 10).astype(np.float32)

    # Test training
    history = model.fit(x, y, epochs=2, batch_size=16, verbose=0)
    assert len(history.history['loss']) == 2
    assert all(not np.isnan(loss) for loss in history.history['loss'])


def test_batch_independence(random_seed):
    """Test that processing of each batch element is independent"""
    layer = GlobalResponseNormalization()

    # Create two different batches
    input1 = np.ones((1, 4, 4, 3), dtype=np.float32)
    input2 = np.zeros((1, 4, 4, 3), dtype=np.float32)

    # Process separately
    output1 = layer(input1)
    output2 = layer(input2)

    # Process as batch
    combined_input = np.concatenate([input1, input2], axis=0)
    combined_output = layer(combined_input)

    # Check that results match
    np.testing.assert_allclose(
        output1,
        combined_output[0:1],
        rtol=1e-5,
        atol=1e-5,
        err_msg="Batch processing should be independent"
    )
    np.testing.assert_allclose(
        output2,
        combined_output[1:2],
        rtol=1e-5,
        atol=1e-5,
        err_msg="Batch processing should be independent"
    )


def test_dtype_handling():
    """Test handling of different data types"""
    layer = GlobalResponseNormalization()
    input_shape = (2, 4, 4, 3)

    # Test float32
    inputs_f32 = np.random.randn(*input_shape).astype(np.float32)
    output_f32 = layer(inputs_f32)
    assert output_f32.dtype == tf.float32

    # Test float64
    inputs_f64 = np.random.randn(*input_shape).astype(np.float64)
    output_f64 = layer(inputs_f64)
    assert output_f64.dtype == tf.float32  # Should convert to float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
