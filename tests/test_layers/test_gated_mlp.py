"""
Tests for GatedMLP implementation.

This module provides comprehensive tests for the GatedMLP layer,
ensuring it initializes, processes data, and serializes correctly.
"""

import os
import tempfile
import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops

from dl_techniques.layers.gated_mlp import GatedMLP


class TestGatedMLP:
    """Test suite for the GatedMLP layer implementation."""

    @pytest.fixture
    def input_tensor_2d(self):
        """Create a 2D input tensor with channels last format."""
        return tf.random.normal([4, 32, 32, 64])

    @pytest.fixture
    def input_tensor_channels_first(self):
        """Create a 2D input tensor with channels first format."""
        return tf.random.normal([4, 64, 32, 32])

    def test_initialization_defaults(self):
        """Test GatedMLP initialization with default parameters."""
        # Default initialization
        layer = GatedMLP(filters=128)

        # Check default values
        assert layer.filters == 128
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.attention_activation == "relu"
        assert layer.output_activation == "linear"
        assert layer.data_format == "channels_last"  # Default in most TF installations

    def test_initialization_custom(self):
        """Test GatedMLP initialization with custom parameters."""
        # Custom initialization
        custom_kernel_initializer = keras.initializers.HeNormal()
        custom_bias_initializer = keras.initializers.Ones()
        custom_kernel_regularizer = keras.regularizers.L2(1e-4)
        custom_bias_regularizer = keras.regularizers.L1(1e-5)

        layer = GatedMLP(
            filters=256,
            use_bias=False,
            kernel_initializer=custom_kernel_initializer,
            bias_initializer=custom_bias_initializer,
            kernel_regularizer=custom_kernel_regularizer,
            bias_regularizer=custom_bias_regularizer,
            attention_activation="gelu",
            output_activation="swish",
            data_format="channels_first",
            name="custom_gmlp"
        )

        # Check custom values
        assert layer.filters == 256
        assert layer.use_bias is False
        assert layer.kernel_initializer == custom_kernel_initializer
        assert layer.bias_initializer == custom_bias_initializer
        assert layer.kernel_regularizer == custom_kernel_regularizer
        assert layer.bias_regularizer == custom_bias_regularizer
        assert layer.attention_activation == "gelu"
        assert layer.output_activation == "swish"
        assert layer.data_format == "channels_first"
        assert layer.name == "custom_gmlp"

    def test_invalid_data_format(self):
        """Test that an invalid data_format raises a ValueError."""
        with pytest.raises(ValueError, match="data_format must be"):
            GatedMLP(filters=64, data_format="invalid_format")

    def test_build_sublayers(self, input_tensor_2d):
        """Test that all sublayers are properly built."""
        layer = GatedMLP(filters=128)

        # Build the layer
        output = layer(input_tensor_2d)

        # Verify sublayers are created
        assert layer.conv_gate is not None
        assert layer.conv_up is not None
        assert layer.conv_down is not None
        assert layer.attention_activation_fn is not None
        assert layer.output_activation_fn is not None

        # Check that Conv2D layers have the correct configuration
        for conv_layer in [layer.conv_gate, layer.conv_up, layer.conv_down]:
            assert conv_layer.filters == 128
            assert conv_layer.kernel_size == (1, 1)
            assert conv_layer.strides == (1, 1)
            assert conv_layer.padding == "same"
            assert conv_layer.data_format == "channels_last"
            assert conv_layer.use_bias == True

    def test_output_shape_channels_last(self, input_tensor_2d):
        """Test output shape with channels_last format."""
        filters_to_test = [32, 64, 128]

        for filters in filters_to_test:
            layer = GatedMLP(filters=filters, data_format="channels_last")
            output = layer(input_tensor_2d)

            # Expected shape: [batch, height, width, filters]
            expected_shape = (input_tensor_2d.shape[0], input_tensor_2d.shape[1],
                              input_tensor_2d.shape[2], filters)
            assert output.shape == expected_shape

            # Test compute_output_shape method separately
            computed_shape = layer.compute_output_shape(input_tensor_2d.shape)
            assert computed_shape == expected_shape

    def test_forward_pass_values(self, input_tensor_2d):
        """Test that forward pass produces expected values."""
        layer = GatedMLP(filters=64)
        output = layer(input_tensor_2d)

        # Check output values are finite
        assert np.all(np.isfinite(output.numpy()))

        # Test a specific layer configuration with linear activations for deterministic output
        deterministic_layer = GatedMLP(
            filters=32,
            kernel_initializer="ones",
            bias_initializer="zeros",
            attention_activation="linear",
            output_activation="linear"
        )

        # Use a controlled input
        controlled_input = tf.ones([1, 2, 2, 2])
        result = deterministic_layer(controlled_input)

        # With linear activations and ones initialization, we can predict the output pattern
        # Output should be non-zero with this setup
        assert np.any(result.numpy() != 0)

    def test_different_activations(self, input_tensor_2d):
        """Test layer with different activation functions."""
        activations = ["relu", "gelu", "swish", "linear"]

        for act in activations:
            layer = GatedMLP(
                filters=64,
                attention_activation=act,
                output_activation=act
            )
            output = layer(input_tensor_2d)

            # Check output is valid
            assert output.shape == (input_tensor_2d.shape[0], input_tensor_2d.shape[1],
                                    input_tensor_2d.shape[2], 64)
            assert np.all(np.isfinite(output.numpy()))

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = GatedMLP(
            filters=128,
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="ones",
            kernel_regularizer=keras.regularizers.L2(1e-4),
            bias_regularizer=keras.regularizers.L1(1e-5),
            attention_activation="gelu",
            output_activation="relu",
            data_format="channels_last"
        )

        # Get the config
        config = original_layer.get_config()

        # Recreate the layer
        recreated_layer = GatedMLP.from_config(config)

        # Check configuration
        assert recreated_layer.filters == original_layer.filters
        assert recreated_layer.use_bias == original_layer.use_bias
        assert keras.initializers.serialize(recreated_layer.kernel_initializer) == \
               keras.initializers.serialize(original_layer.kernel_initializer)
        assert keras.initializers.serialize(recreated_layer.bias_initializer) == \
               keras.initializers.serialize(original_layer.bias_initializer)
        assert keras.regularizers.serialize(recreated_layer.kernel_regularizer) == \
               keras.regularizers.serialize(original_layer.kernel_regularizer)
        assert keras.regularizers.serialize(recreated_layer.bias_regularizer) == \
               keras.regularizers.serialize(original_layer.bias_regularizer)
        assert recreated_layer.attention_activation == original_layer.attention_activation
        assert recreated_layer.output_activation == original_layer.output_activation
        assert recreated_layer.data_format == original_layer.data_format

    def test_model_integration(self, input_tensor_2d):
        """Test GatedMLP in a model context."""
        # Create a simple model with GatedMLP
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        x = GatedMLP(filters=32)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = GatedMLP(filters=64)(x)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        # Test forward pass
        y_pred = model(input_tensor_2d, training=False)
        assert y_pred.shape == (input_tensor_2d.shape[0], 10)

    def test_training_inference_modes(self, input_tensor_2d):
        """Test layer behavior in training and inference modes."""
        # Use BatchNormalization to observe different behavior between training and inference
        model = keras.Sequential([
            layers.InputLayer(input_tensor_2d.shape[1:]),
            GatedMLP(filters=64),
            layers.BatchNormalization(),
            GatedMLP(filters=32)
        ])

        # Get outputs in different modes
        training_output = model(input_tensor_2d, training=True)
        inference_output = model(input_tensor_2d, training=False)

        # The outputs should be different due to BatchNormalization
        # We can't guarantee exact differences, but they shouldn't be identical
        assert not np.allclose(training_output.numpy(), inference_output.numpy())

    def test_gradient_flow(self, input_tensor_2d):
        """Test gradient flow through the layer."""
        layer = GatedMLP(filters=64)

        # Watch the variables
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor_2d)
            outputs = layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check gradients have values (not all zeros)
        assert all(np.any(g.numpy() != 0) for g in grads)

        # Check gradients are finite
        assert all(np.all(np.isfinite(g.numpy())) for g in grads)

    def test_model_save_load(self, input_tensor_2d):
        """Test saving and loading a model with GatedMLP layer."""
        # Create a model with GatedMLP
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        x = GatedMLP(filters=32, name="gmlp1")(inputs)
        x = layers.BatchNormalization()(x)
        x = GatedMLP(filters=64, name="gmlp2")(x)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor_2d)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "gmlp_model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"GatedMLP": GatedMLP}  # Explicitly register custom layer
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor_2d)

            # Predictions should be the same
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("gmlp1"), GatedMLP)
            assert isinstance(loaded_model.get_layer("gmlp2"), GatedMLP)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = GatedMLP(filters=16)

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

    def test_training_loop(self, input_tensor_2d):
        """Test training loop with GatedMLP layer."""
        # Create a model with GatedMLP
        model = keras.Sequential([
            layers.InputLayer(input_tensor_2d.shape[1:]),
            GatedMLP(32),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            GatedMLP(64),
            layers.GlobalAveragePooling2D(),
            layers.Dense(10)
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Create mock data
        x_train = tf.random.normal([32] + list(input_tensor_2d.shape[1:]))
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)

        # Initial loss
        initial_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Train for a few epochs
        model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)

        # Final loss
        final_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Loss should decrease
        assert final_loss < initial_loss

    def test_layer_with_regularization(self, input_tensor_2d):
        """Test that regularization losses are properly applied."""
        # Create layer with regularization
        layer = GatedMLP(
            filters=32,
            kernel_regularizer=keras.regularizers.L2(0.1),  # Strong regularization for testing
            bias_regularizer=keras.regularizers.L1(0.1)
        )

        # No regularization losses before calling the layer
        assert len(layer.losses) == 0

        # Apply the layer
        _ = layer(input_tensor_2d)

        # Should have regularization losses now
        assert len(layer.losses) > 0

        # Test in model context
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        x = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        # Model should have regularization losses
        assert len(model.losses) > 0

    def test_layer_in_functional_api(self, input_tensor_2d):
        """Test GatedMLP in a functional API model."""
        # Create model using functional API
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])

        # First branch
        x1 = GatedMLP(32, attention_activation="relu")(inputs)

        # Second branch
        x2 = GatedMLP(32, attention_activation="gelu")(inputs)

        # Merge branches
        merged = layers.Add()([x1, x2])

        # Final GatedMLP layer
        output = GatedMLP(64, output_activation="swish")(merged)

        model = keras.Model(inputs=inputs, outputs=output)

        # Test forward pass
        result = model(input_tensor_2d)

        # Check output shape
        assert result.shape == (input_tensor_2d.shape[0], input_tensor_2d.shape[1],
                                input_tensor_2d.shape[2], 64)

if __name__ == '__main__':
    pytest.main([__file__])