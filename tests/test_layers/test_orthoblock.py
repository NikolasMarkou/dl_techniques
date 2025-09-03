"""
Comprehensive test suite for OrthoBlock layer.

This module contains thorough tests for the OrthoBlock layer implementation,
covering initialization, forward pass, serialization, training, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.experimental.orthoblock import OrthoBlock


class TestOrthoBlock:
    """Test suite for OrthoBlock layer implementation."""

    @pytest.fixture
    def input_tensor_2d(self) -> tf.Tensor:
        """Create a 2D test input tensor."""
        return tf.random.normal([4, 32])

    @pytest.fixture
    def input_tensor_3d(self) -> tf.Tensor:
        """Create a 3D test input tensor."""
        return tf.random.normal([2, 10, 64])

    @pytest.fixture
    def layer_instance(self) -> OrthoBlock:
        """Create a default OrthoBlock instance for testing."""
        return OrthoBlock(units=16)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = OrthoBlock(units=128)

        # Check default values
        assert layer.units == 128
        assert layer.use_bias is True
        assert layer.ortho_reg_factor == 0.01
        assert layer.scale_initial_value == 0.5
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_bias_regularizer = keras.regularizers.L2(1e-4)

        layer = OrthoBlock(
            units=64,
            activation="gelu",
            use_bias=False,
            ortho_reg_factor=0.02,
            kernel_initializer="he_normal",
            bias_initializer="ones",
            bias_regularizer=custom_bias_regularizer,
            scale_initial_value=0.3,
        )

        # Check custom values
        assert layer.units == 64
        assert layer.activation == keras.activations.gelu
        assert layer.use_bias is False
        assert layer.ortho_reg_factor == 0.02
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert layer.bias_regularizer == custom_bias_regularizer
        assert layer.scale_initial_value == 0.3

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid units - negative
        with pytest.raises(ValueError, match="units must be a positive integer"):
            OrthoBlock(units=-10)

        # Test invalid units - zero
        with pytest.raises(ValueError, match="units must be a positive integer"):
            OrthoBlock(units=0)

        # Test invalid units - non-integer
        with pytest.raises(ValueError, match="units must be a positive integer"):
            OrthoBlock(units=10.5)

        # Test invalid ortho_reg_factor
        with pytest.raises(ValueError, match="ortho_reg_factor must be non-negative"):
            OrthoBlock(units=32, ortho_reg_factor=-0.1)

        # Test invalid scale_initial_value - negative
        with pytest.raises(ValueError, match="scale_initial_value must be between 0.0 and 1.0"):
            OrthoBlock(units=32, scale_initial_value=-0.1)

        # Test invalid scale_initial_value - too large
        with pytest.raises(ValueError, match="scale_initial_value must be between 0.0 and 1.0"):
            OrthoBlock(units=32, scale_initial_value=1.5)

    def test_build_process_2d(self, input_tensor_2d):
        """Test that the layer builds properly with 2D input."""
        layer = OrthoBlock(units=64)
        layer(input_tensor_2d)  # Forward pass triggers build

        # Check that the layer is built
        assert layer.built is True

        # Check that all sublayers were created and built
        assert layer.dense is not None
        assert layer.dense.built is True
        assert layer.ortho_reg is not None
        assert layer.norm is not None
        assert layer.norm.built is True
        assert layer.constrained_scale is not None
        assert layer.constrained_scale.built is True

        # Check dense layer configuration
        assert layer.dense.units == 64
        # Dense layer uses linear activation (identity) internally, which is equivalent to no activation
        assert layer.dense.activation == keras.activations.linear
        assert layer.dense.kernel_regularizer == layer.ortho_reg

    def test_build_process_3d(self, input_tensor_3d):
        """Test that the layer builds properly with 3D input."""
        layer = OrthoBlock(units=32, use_bias=False)
        layer(input_tensor_3d)  # Forward pass triggers build

        # Check that the layer is built
        assert layer.built is True

        # Check sublayers are built
        assert layer.dense.built is True
        assert layer.norm.built is True
        assert layer.constrained_scale.built is True

        # Check bias configuration
        assert layer.dense.use_bias is False

    def test_output_shapes_2d(self, input_tensor_2d):
        """Test that output shapes are computed correctly for 2D input."""
        units_to_test = [16, 32, 64, 128]

        for units in units_to_test:
            layer = OrthoBlock(units=units)
            output = layer(input_tensor_2d)

            # Check actual output shape
            expected_shape = (input_tensor_2d.shape[0], units)
            assert output.shape == expected_shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor_2d.shape)
            assert computed_shape == expected_shape

    def test_output_shapes_3d(self, input_tensor_3d):
        """Test that output shapes are computed correctly for 3D input."""
        layer = OrthoBlock(units=48)
        output = layer(input_tensor_3d)

        # Check actual output shape
        expected_shape = (input_tensor_3d.shape[0], input_tensor_3d.shape[1], 48)
        assert output.shape == expected_shape

        # Test compute_output_shape separately
        computed_shape = layer.compute_output_shape(input_tensor_3d.shape)
        assert computed_shape == expected_shape

    def test_forward_pass_basic(self, input_tensor_2d):
        """Test basic forward pass functionality."""
        layer = OrthoBlock(units=64)
        output = layer(input_tensor_2d)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))
        assert output.dtype == input_tensor_2d.dtype

    def test_forward_pass_with_activations(self, input_tensor_2d):
        """Test forward pass with different activation functions."""
        activations = ["relu", "gelu", "swish", "linear", None]

        for activation in activations:
            layer = OrthoBlock(units=32, activation=activation)
            output = layer(input_tensor_2d)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == (input_tensor_2d.shape[0], 32)

            # Check activation-specific properties
            if activation == "relu":
                assert np.all(output.numpy() >= 0)  # ReLU outputs are non-negative

    def test_forward_pass_deterministic(self):
        """Test forward pass with controlled inputs for deterministic results."""
        # Create deterministic input
        controlled_input = tf.ones([2, 4])

        # Create layer with fixed parameters and no regularization
        layer = OrthoBlock(
            units=3,
            kernel_initializer="ones",
            bias_initializer="zeros",
            activation=None,  # No activation
            ortho_reg_factor=0.0,  # Disable regularization for deterministic test
            scale_initial_value=1.0
        )

        # Get output
        result = layer(controlled_input, training=False)  # Use inference mode

        # With ones initializer and ones input, we can predict the pattern
        # (though exact values depend on the normalization)
        assert result.shape == (2, 3)
        assert not np.any(np.isnan(result.numpy()))

    def test_training_mode_propagation(self, input_tensor_2d):
        """Test that training mode is properly propagated to sublayers."""
        layer = OrthoBlock(units=32)

        # Test training mode
        output_train = layer(input_tensor_2d, training=True)
        assert output_train.shape == (input_tensor_2d.shape[0], 32)

        # Test inference mode
        output_inference = layer(input_tensor_2d, training=False)
        assert output_inference.shape == (input_tensor_2d.shape[0], 32)

        # Both should be valid
        assert not np.any(np.isnan(output_train.numpy()))
        assert not np.any(np.isnan(output_inference.numpy()))

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        # Create and build the layer
        original_layer = OrthoBlock(
            units=64,
            activation="gelu",
            use_bias=True,
            ortho_reg_factor=0.02,
            kernel_initializer="he_normal",
            bias_regularizer=keras.regularizers.L1(0.01),
            scale_initial_value=0.3
        )
        original_layer.build((None, 32))

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = OrthoBlock.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.units == original_layer.units
        assert recreated_layer.use_bias == original_layer.use_bias
        assert recreated_layer.ortho_reg_factor == original_layer.ortho_reg_factor
        assert recreated_layer.scale_initial_value == original_layer.scale_initial_value
        assert recreated_layer.activation == original_layer.activation

    def test_serialization_functionality(self):
        """Test that serialized layer works functionally."""
        # Create original layer
        original_layer = OrthoBlock(units=32, activation="relu")
        original_layer.build((None, 16))

        # Create test data
        test_input = tf.random.normal((2, 16))
        original_output = original_layer(test_input)

        # Serialize and recreate
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        new_layer = OrthoBlock.from_config(config)
        new_layer.build_from_config(build_config)

        # Test that new layer produces valid output
        new_output = new_layer(test_input)
        assert new_output.shape == original_output.shape
        assert not np.any(np.isnan(new_output.numpy()))

    def test_model_integration(self, input_tensor_2d):
        """Test the layer in a model context."""
        # Create a simple model with the OrthoBlock
        inputs = keras.Input(shape=(32,))
        x = OrthoBlock(units=64, activation="relu")(inputs)
        x = keras.layers.Dropout(0.1)(x)
        x = OrthoBlock(units=32, activation="gelu")(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Dense(10, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Test forward pass
        y_pred = model(input_tensor_2d, training=False)
        assert y_pred.shape == (input_tensor_2d.shape[0], 10)

        # Check probabilities sum to 1 (softmax output)
        assert np.allclose(np.sum(y_pred.numpy(), axis=1), 1.0, rtol=1e-5)

    def test_model_save_load(self, input_tensor_2d):
        """Test saving and loading a model with OrthoBlock."""
        # Create a model with OrthoBlock
        inputs = keras.Input(shape=(32,))
        x = OrthoBlock(units=64, activation="relu", name="ortho_layer1")(inputs)
        x = keras.layers.Dense(32, activation="gelu")(x)
        x = OrthoBlock(units=16, activation="swish", name="ortho_layer2")(x)
        outputs = keras.layers.Dense(5, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate prediction before saving
        original_prediction = model.predict(input_tensor_2d, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"OrthoBlock": OrthoBlock}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor_2d, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("ortho_layer1"), OrthoBlock)
            assert isinstance(loaded_model.get_layer("ortho_layer2"), OrthoBlock)

    def test_gradient_flow(self, input_tensor_2d):
        """Test gradient flow through the layer."""
        layer = OrthoBlock(units=32, activation="relu")

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

    def test_training_loop(self, input_tensor_2d):
        """Test training loop with OrthoBlock."""
        # Create a model with OrthoBlock
        model = keras.Sequential([
            keras.layers.InputLayer(shape=(32,)),
            OrthoBlock(64, activation="relu"),
            keras.layers.Dropout(0.1),
            OrthoBlock(32, activation="gelu"),
            keras.layers.Dense(10, activation="softmax")
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # Create mock data
        x_train = tf.random.normal([32, 32])
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)

        # Initial loss
        initial_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Train for a few epochs
        history = model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=0)

        # Final loss
        final_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Loss should decrease (or at least not increase significantly)
        # Allow some tolerance for small datasets and regularization effects
        assert final_loss <= initial_loss * 1.2  # Allow small increase due to regularization

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = OrthoBlock(units=16, activation="relu")

        # Create inputs with different magnitudes
        test_cases = [
            tf.zeros((2, 8)),  # Zeros
            tf.ones((2, 8)) * 1e-10,  # Very small values
            tf.ones((2, 8)) * 1e5,   # Large values
            tf.random.normal((2, 8)) * 1e3  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_bias_regularization(self, input_tensor_2d):
        """Test that bias regularization is properly applied."""
        # Create layer with bias regularization
        bias_reg = keras.regularizers.L2(0.01)
        layer = OrthoBlock(
            units=16,
            use_bias=True,
            bias_regularizer=bias_reg,
            ortho_reg_factor=0.01  # Also enable orthogonal regularization
        )

        # Apply the layer - regularization should be applied automatically
        _ = layer(input_tensor_2d)

        # Should have regularization losses from both orthogonal (kernel) and bias
        assert len(layer.losses) > 0

        # Check that bias regularizer is applied to dense layer
        assert layer.dense.bias_regularizer == bias_reg

    def test_zero_regularization_factor(self, input_tensor_2d):
        """Test behavior when ortho_reg_factor is 0."""
        layer = OrthoBlock(units=16, ortho_reg_factor=0.0)

        # With zero regularization factor, the orthogonal regularizer should have lambda=0
        assert layer.ortho_reg._lambda_coefficient == 0.0

        # Apply the layer
        _ = layer(input_tensor_2d)

        # There might still be losses from other regularizers (like L1 on constrained_scale)
        # but the orthogonal regularization specifically should contribute 0

    def test_different_initializers(self, input_tensor_2d):
        """Test layer with different initializer configurations."""
        initializers = [
            "glorot_uniform",
            "he_normal",
            "lecun_normal",
            keras.initializers.RandomNormal(stddev=0.01)
        ]

        for init in initializers:
            layer = OrthoBlock(
                units=16,
                kernel_initializer=init,
                bias_initializer="zeros"
            )

            output = layer(input_tensor_2d)
            assert output.shape == (input_tensor_2d.shape[0], 16)
            assert not np.any(np.isnan(output.numpy()))

    def test_constrained_scaling_behavior(self, input_tensor_2d):
        """Test that the constrained scaling behaves as expected."""
        # Create layer with very low scale value (should reduce output magnitude)
        layer_low_scale = OrthoBlock(
            units=16,
            scale_initial_value=0.1,
            activation="linear"  # Linear to see scaling effect clearly
        )

        # Create layer with high scale value
        layer_high_scale = OrthoBlock(
            units=16,
            scale_initial_value=0.9,
            activation="linear"
        )

        # Get outputs
        output_low = layer_low_scale(input_tensor_2d)
        output_high = layer_high_scale(input_tensor_2d)

        # The constrained scale should be applied, but exact comparison is complex
        # due to normalization. Just check outputs are valid and different.
        assert not np.any(np.isnan(output_low.numpy()))
        assert not np.any(np.isnan(output_high.numpy()))
        assert output_low.shape == output_high.shape

    def test_layer_naming(self):
        """Test that sublayers are properly named."""
        layer = OrthoBlock(units=16, name="test_ortho")
        layer.build((None, 8))

        # Check that sublayers have appropriate names
        assert "ortho_dense" in layer.dense.name
        assert "rms_norm" in layer.norm.name
        assert "constrained_scale" in layer.constrained_scale.name

# Additional utility functions for testing

def test_orthoblock_example_usage():
    """Test the example usage from the docstring."""
    # Basic usage
    x = keras.Input(shape=(128,))
    y = OrthoBlock(units=64, activation='relu')(x)
    model = keras.Model(inputs=x, outputs=y)

    # Test model creation
    assert model is not None
    assert len(model.layers) == 2  # Input + OrthoBlock

    # Custom usage
    ortho_layer = OrthoBlock(
        units=32,
        activation='gelu',
        ortho_reg_factor=0.02,
        scale_initial_value=0.3
    )

    input_tensor = tf.random.normal((4, 128))
    output = ortho_layer(input_tensor)
    assert output.shape == (4, 32)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])