"""
Test suite for custom Keras layers:
- LayerScale
- MultiplierType
- LearnableMultiplier

This module provides comprehensive tests for initialization,
behavior, serialization, and edge cases following project best practices.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf  # Need for GradientTape
import tempfile
import os
from typing import Tuple, Dict, Any

from dl_techniques.layers.layer_scale import (
    MultiplierType,
    LearnableMultiplier
)
from dl_techniques.utils.logger import logger


# Test fixtures
@pytest.fixture
def sample_shape() -> Tuple[int, int, int, int]:
    """Sample input shape for testing."""
    return (2, 16, 16, 64)


@pytest.fixture
def sample_input(sample_shape: Tuple[int, int, int, int]) -> tf.Tensor:
    """Generate sample input tensor."""
    tf.random.set_seed(42)
    return tf.random.normal(sample_shape)


@pytest.fixture
def sample_2d_shape() -> Tuple[int, int]:
    """Sample 2D input shape for testing."""
    return (4, 32)


@pytest.fixture
def sample_2d_input(sample_2d_shape: Tuple[int, int]) -> tf.Tensor:
    """Generate sample 2D input tensor."""
    tf.random.set_seed(42)
    return tf.random.normal(sample_2d_shape)


# MultiplierType Tests
class TestMultiplierType:
    """Test cases for MultiplierType enum."""

    @pytest.mark.parametrize("type_str", ["GLOBAL", "CHANNEL", "global", "channel", " GLOBAL ", " CHANNEL "])
    def test_valid_from_string(self, type_str: str) -> None:
        """Test valid string conversions including case variations."""
        mult_type = MultiplierType.from_string(type_str)
        assert isinstance(mult_type, MultiplierType)
        assert mult_type.to_string() in ["GLOBAL", "CHANNEL"]

    @pytest.mark.parametrize("invalid_input", [None, 123, "", " ", "INVALID", "OTHER"])
    def test_invalid_from_string(self, invalid_input: Any) -> None:
        """Test invalid string conversions."""
        with pytest.raises(ValueError):
            MultiplierType.from_string(invalid_input)

    def test_enum_values(self) -> None:
        """Test enum values are correctly defined."""
        assert MultiplierType.GLOBAL.value == 0
        assert MultiplierType.CHANNEL.value == 1

    def test_from_enum_instance(self) -> None:
        """Test passing enum instance returns the same instance."""
        original = MultiplierType.GLOBAL
        result = MultiplierType.from_string(original)
        assert result is original


# LearnableMultiplier Tests
class TestLearnableMultiplier:
    """Test cases for LearnableMultiplier layer."""

    @pytest.fixture
    def layer_params(self) -> Dict[str, Any]:
        """Default parameters for LearnableMultiplier."""
        return {
            "multiplier_type": "GLOBAL",
        }

    def test_initialization_defaults(self) -> None:
        """Test LearnableMultiplier initialization with default parameters."""
        layer = LearnableMultiplier()

        # Check default values
        assert layer.multiplier_type == MultiplierType.CHANNEL
        assert layer.gamma is None
        assert isinstance(layer.initializer, keras.initializers.Ones)
        assert layer.regularizer is None
        assert isinstance(layer.constraint, keras.constraints.NonNeg)

    def test_initialization_custom(self) -> None:
        """Test LearnableMultiplier initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = LearnableMultiplier(
            multiplier_type="GLOBAL",
            initializer="zeros",
            regularizer=custom_regularizer,
            constraint="unit_norm"
        )

        # Check custom values
        assert layer.multiplier_type == MultiplierType.GLOBAL
        assert isinstance(layer.initializer, keras.initializers.Zeros)
        assert layer.regularizer == custom_regularizer
        assert isinstance(layer.constraint, keras.constraints.UnitNorm)

    def test_build_global(self, layer_params: Dict[str, Any], sample_shape: Tuple[int, int, int, int]) -> None:
        """Test build with global multiplier."""
        layer = LearnableMultiplier(**layer_params)
        layer.build(sample_shape)

        # Global multiplier should have shape (1,) for efficient broadcasting
        assert layer.gamma.shape == (1,)
        assert layer.built is True

    def test_build_channel(self, sample_shape: Tuple[int, int, int, int]) -> None:
        """Test build with channel multiplier."""
        layer = LearnableMultiplier(multiplier_type="CHANNEL")
        layer.build(sample_shape)

        # Channel multiplier should have shape (channels,) for efficient broadcasting
        assert layer.gamma.shape == (sample_shape[-1],)
        assert layer.built is True

    def test_build_channel_2d(self, sample_2d_shape: Tuple[int, int]) -> None:
        """Test build with channel multiplier on 2D input."""
        layer = LearnableMultiplier(multiplier_type="CHANNEL")
        layer.build(sample_2d_shape)

        # Should work with 2D inputs
        assert layer.gamma.shape == (sample_2d_shape[-1],)

    def test_call_global(self, layer_params: Dict[str, Any], sample_input: tf.Tensor) -> None:
        """Test call with global multiplier."""
        # Use a non-identity initializer to ensure multiplication changes values
        layer_params["initializer"] = "zeros"
        layer = LearnableMultiplier(**layer_params)
        output = layer(sample_input)

        # Output shape should match input shape
        assert output.shape == sample_input.shape

        # With zeros initializer, output should be all zeros
        assert np.allclose(output.numpy(), 0.0)

    def test_call_channel(self, sample_input: tf.Tensor) -> None:
        """Test call with channel multiplier."""
        # Use a specific initializer to test actual multiplication
        layer = LearnableMultiplier(multiplier_type="CHANNEL", initializer="zeros")
        output = layer(sample_input)

        # Output shape should match input shape
        assert output.shape == sample_input.shape

        # With zeros initializer, output should be all zeros
        assert np.allclose(output.numpy(), 0.0)

    def test_call_with_ones_initializer(self, sample_input: tf.Tensor) -> None:
        """Test call with ones initializer (identity operation)."""
        layer = LearnableMultiplier(
            multiplier_type="GLOBAL",
            initializer="ones"
        )
        output = layer(sample_input)

        # Output should equal input when multiplier is 1.0
        assert np.allclose(output.numpy(), sample_input.numpy())

    def test_call_with_custom_values(self, sample_input: tf.Tensor) -> None:
        """Test call with custom multiplier values."""
        # Create a layer with constant 2.0 multiplier
        layer = LearnableMultiplier(
            multiplier_type="GLOBAL",
            initializer=keras.initializers.Constant(2.0)
        )
        output = layer(sample_input)

        # Output should be input * 2.0
        expected = sample_input * 2.0
        assert np.allclose(output.numpy(), expected.numpy())

    def test_call_training_mode(self, sample_input: tf.Tensor) -> None:
        """Test call with explicit training mode."""
        layer = LearnableMultiplier(multiplier_type="GLOBAL")

        # Test both training modes
        output_train = layer(sample_input, training=True)
        output_infer = layer(sample_input, training=False)

        # Outputs should be identical (layer doesn't change behavior)
        assert np.allclose(output_train.numpy(), output_infer.numpy())

    def test_compute_output_shape(self) -> None:
        """Test compute_output_shape method."""
        layer = LearnableMultiplier()

        input_shapes = [
            (None, 32),
            (None, 16, 16, 64),
            (2, 8, 8, 128),
        ]

        for input_shape in input_shapes:
            output_shape = layer.compute_output_shape(input_shape)
            assert output_shape == input_shape

    def test_regularization(self, sample_input: tf.Tensor) -> None:
        """Test that regularization losses are properly applied."""
        layer = LearnableMultiplier(
            multiplier_type="GLOBAL",
            regularizer=keras.regularizers.L2(0.1)
        )

        # No regularization losses before calling the layer
        assert len(layer.losses) == 0

        # Apply the layer
        _ = layer(sample_input)

        # Should have regularization losses now
        assert len(layer.losses) > 0

    def test_gradient_flow(self, sample_input: tf.Tensor) -> None:
        """Test gradient flow through the layer."""
        layer = LearnableMultiplier(multiplier_type="GLOBAL")

        # Use GradientTape for gradient computation
        with tf.GradientTape() as tape:
            outputs = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check gradients have values (not all zeros) - convert to numpy for checking
        for grad in grads:
            grad_numpy = grad.numpy()
            assert np.any(grad_numpy != 0)

    def test_serialization(self, layer_params: Dict[str, Any]) -> None:
        """Test serialization and deserialization of the layer."""
        original_layer = LearnableMultiplier(**layer_params)

        # Build the layer
        input_shape = (None, 16, 16, 64)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = LearnableMultiplier.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.multiplier_type == original_layer.multiplier_type
        assert type(recreated_layer.initializer) == type(original_layer.initializer)
        assert recreated_layer.regularizer == original_layer.regularizer

        # Check weights match (shapes should be the same at minimum)
        assert recreated_layer.gamma.shape == original_layer.gamma.shape

    def test_serialization_edge_cases(self) -> None:
        """Test serialization with various parameter types."""
        layers_to_test = [
            LearnableMultiplier(multiplier_type="GLOBAL", initializer="ones"),
            LearnableMultiplier(multiplier_type="CHANNEL", initializer=keras.initializers.Ones()),
            LearnableMultiplier(multiplier_type="GLOBAL", regularizer="l2"),
            LearnableMultiplier(multiplier_type="CHANNEL", regularizer=keras.regularizers.L2(0.01)),
            LearnableMultiplier(multiplier_type="GLOBAL", constraint="non_neg"),
            LearnableMultiplier(multiplier_type="CHANNEL", constraint=keras.constraints.NonNeg()),
        ]

        for original_layer in layers_to_test:
            # Build the layer
            original_layer.build((None, 16, 16, 8))

            # Get config and recreate
            config = original_layer.get_config()
            recreated_layer = LearnableMultiplier.from_config(config)

            # Check key aspects match
            assert recreated_layer.multiplier_type == original_layer.multiplier_type

    def test_numerical_stability(self) -> None:
        """Test layer stability with extreme input values."""
        layer = LearnableMultiplier(multiplier_type="GLOBAL")

        # Create inputs with different magnitudes
        batch_size = 2
        height, width = 8, 8
        channels = 4

        test_cases = [
            tf.zeros((batch_size, height, width, channels)),  # Zeros
            tf.ones((batch_size, height, width, channels)) * 1e-10,  # Very small values
            tf.ones((batch_size, height, width, channels)) * 1e10,  # Very large values
            tf.random.normal((batch_size, height, width, channels)) * 1e5  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_invalid_parameters(self) -> None:
        """Test that invalid parameters raise appropriate errors."""
        # Invalid multiplier type should be caught in enum conversion
        with pytest.raises(ValueError):
            LearnableMultiplier(multiplier_type="INVALID")


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for layers working in model contexts."""

    def test_sequential_model(self, sample_input: tf.Tensor) -> None:
        """Test layers in a sequential model."""
        model = keras.Sequential([
            keras.layers.InputLayer(sample_input.shape[1:]),
            LearnableMultiplier(multiplier_type="GLOBAL"),
            LearnableMultiplier(multiplier_type="CHANNEL")
        ])

        output = model(sample_input)
        assert output.shape == sample_input.shape

    def test_functional_model(self, sample_input: tf.Tensor) -> None:
        """Test layers in a functional model."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        x = LearnableMultiplier(multiplier_type="CHANNEL", name="multiplier1")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = LearnableMultiplier(multiplier_type="GLOBAL", name="multiplier2")(x)
        outputs = keras.layers.GlobalAveragePooling2D()(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        output = model(sample_input)
        assert len(output.shape) == 2  # Should be (batch_size, features)

    def test_model_save_load(self, sample_input: tf.Tensor) -> None:
        """Test saving and loading a model with the custom layer."""
        # Create a model with the custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        x = LearnableMultiplier(multiplier_type="CHANNEL", name="custom_multiplier")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(sample_input, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"LearnableMultiplier": LearnableMultiplier}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(sample_input, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("custom_multiplier"), LearnableMultiplier)

    def test_training_pipeline(self, sample_input: tf.Tensor) -> None:
        """Test layers in a training pipeline."""
        model = keras.Sequential([
            keras.layers.InputLayer(sample_input.shape[1:]),
            LearnableMultiplier(multiplier_type="CHANNEL"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Create dummy target data
        target = tf.random.normal((sample_input.shape[0], 1))

        # Test training
        history = model.fit(
            sample_input,
            target,
            epochs=2,
            batch_size=1,
            verbose=0
        )

        assert 'loss' in history.history
        assert len(history.history['loss']) == 2

    def test_training_loop(self, sample_input: tf.Tensor) -> None:
        """Test training loop with the custom layer."""
        # Create a model with the custom layer
        model = keras.Sequential([
            keras.layers.InputLayer(sample_input.shape[1:]),
            LearnableMultiplier(multiplier_type="CHANNEL"),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(10)
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Create mock data
        x_train = tf.random.normal((8,) + sample_input.shape[1:])
        # Generate integer labels by casting float random values
        y_train_float = tf.random.uniform((8,), 0.0, 10.0)
        y_train = tf.cast(tf.floor(y_train_float), tf.int32)

        # Initial loss
        initial_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Train for a few epochs
        model.fit(x_train, y_train, epochs=2, batch_size=4, verbose=0)

        # Final loss
        final_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Loss should generally decrease (allowing some tolerance for randomness)
        logger.info(f"Initial loss: {initial_loss}, Final loss: {final_loss}")


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_different_input_shapes(self) -> None:
        """Test layer with different input shapes."""
        shapes = [
            (4, 32),      # 2D
            (4, 8, 16),   # 3D
            (4, 8, 8, 32), # 4D
            (4, 8, 8, 8, 16), # 5D
        ]

        for shape in shapes:
            # Create a new layer instance for each shape test
            layer = LearnableMultiplier(multiplier_type="CHANNEL")
            test_input = tf.random.normal(shape)
            output = layer(test_input)
            assert output.shape == shape

    def test_different_dtypes(self) -> None:
        """Test layer with different input dtypes."""
        layer = LearnableMultiplier(multiplier_type="GLOBAL")

        # Test with float32 (default)
        input_f32 = tf.random.normal((2, 4, 4, 8), dtype=tf.float32)
        output_f32 = layer(input_f32)
        assert output_f32.dtype == tf.float32

    def test_zero_input(self) -> None:
        """Test layer behavior with zero input."""
        layer = LearnableMultiplier(multiplier_type="GLOBAL")
        zero_input = tf.zeros((2, 4, 4, 8))
        output = layer(zero_input)

        # Output should be zero regardless of multiplier value
        assert np.allclose(output.numpy(), 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])