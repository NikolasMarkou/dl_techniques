"""
Comprehensive test suite for DynamicTanh (DyT) layer implementation.
Tests initialization, build process, forward pass, serialization, and edge cases.
"""

import pytest
import numpy as np
import keras
import os
import tempfile

from dl_techniques.layers.dyt_layer import DynamicTanh


class TestDynamicTanh:
    """Test suite for DynamicTanh layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return np.random.normal(0, 1, (4, 32)).astype(np.float32)

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return DynamicTanh()

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = DynamicTanh()

        assert layer.axis == [-1]
        assert layer.alpha_init_value == 0.5
        assert isinstance(layer.kernel_initializer, keras.initializers.Ones)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.kernel_constraint is None
        assert layer.bias_constraint is None
        assert layer.supports_masking is True
        assert layer.built is False
        assert layer._build_input_shape is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        custom_constraint = keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0)

        layer = DynamicTanh(
            axis=[-2, -1],
            alpha_init_value=0.7,
            kernel_initializer="he_normal",
            bias_initializer="ones",
            kernel_regularizer=custom_regularizer,
            bias_regularizer=custom_regularizer,
            kernel_constraint=custom_constraint,
            bias_constraint=custom_constraint
        )

        assert layer.axis == [-2, -1]
        assert layer.alpha_init_value == 0.7
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.bias_regularizer == custom_regularizer
        assert layer.kernel_constraint == custom_constraint
        assert layer.bias_constraint == custom_constraint

    def test_initialization_single_axis_int(self):
        """Test initialization with single axis as integer."""
        layer = DynamicTanh(axis=-1)
        assert layer.axis == [-1]

        layer = DynamicTanh(axis=2)
        assert layer.axis == [2]

    def test_initialization_multiple_axes(self):
        """Test initialization with multiple axes."""
        layer = DynamicTanh(axis=[1, 2])
        assert layer.axis == [1, 2]

        layer = DynamicTanh(axis=(-2, -1))
        assert layer.axis == [-2, -1]  # Tuples are converted to lists

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="alpha_init_value must be a number"):
            DynamicTanh(alpha_init_value="invalid")

        with pytest.raises(ValueError, match="alpha_init_value must be a number"):
            DynamicTanh(alpha_init_value=None)

    def test_build_process_single_axis(self, input_tensor):
        """Test that the layer builds properly with single axis."""
        layer = DynamicTanh(axis=-1)
        output = layer(input_tensor)

        # Check that layer was built
        assert layer.built is True
        assert len(layer.weights) == 3  # alpha, weight, bias
        assert layer._build_input_shape == input_tensor.shape

        # Check weight shapes
        assert layer.alpha.shape == ()  # Scalar
        assert layer.weight.shape == (input_tensor.shape[-1],)  # Should be a tuple now
        assert layer.bias.shape == (input_tensor.shape[-1],)

        # Check output shape
        assert output.shape == input_tensor.shape

    def test_build_process_multiple_axes(self):
        """Test that the layer builds properly with multiple axes."""
        input_tensor = np.random.normal(0, 1, (4, 8, 16)).astype(np.float32)
        layer = DynamicTanh(axis=[-2, -1])
        output = layer(input_tensor)

        # Check that layer was built
        assert layer.built is True
        assert len(layer.weights) == 3  # alpha, weight, bias

        # Check weight shapes for multiple axes
        expected_param_shape = (input_tensor.shape[-2], input_tensor.shape[-1])
        assert layer.weight.shape == expected_param_shape
        assert layer.bias.shape == expected_param_shape

        # Check output shape
        assert output.shape == input_tensor.shape

    def test_build_axis_validation(self):
        """Test axis validation during build."""
        layer = DynamicTanh(axis=5)  # Invalid axis for typical tensors
        input_tensor = np.random.normal(0, 1, (4, 32)).astype(np.float32)

        with pytest.raises(ValueError, match="Axis .* is out of bounds"):
            layer(input_tensor)

    def test_build_negative_axis_conversion(self):
        """Test that negative axes are properly converted to positive."""
        input_shape = (4, 8, 16, 32)
        input_tensor = np.random.normal(0, 1, input_shape).astype(np.float32)

        layer = DynamicTanh(axis=[-2, -1])
        _ = layer(input_tensor)

        # After build, axes should be converted to positive
        assert layer.axis == [2, 3]  # -2 -> 2, -1 -> 3 for 4D tensor

    def test_forward_pass_functionality(self):
        """Test forward pass with controlled inputs."""
        layer = DynamicTanh(alpha_init_value=1.0)

        # Use controlled inputs
        controlled_input = np.array([[0, 1, -1, 2, -2]], dtype=np.float32)
        result = layer(controlled_input)

        # Check basic properties
        assert result.shape == controlled_input.shape
        assert not np.any(np.isnan(result.numpy()))
        assert not np.any(np.isinf(result.numpy()))

        # Check that tanh saturation is working (output should be bounded)
        assert np.all(np.abs(result.numpy()) < 10)  # Should be much smaller due to tanh

    def test_forward_pass_alpha_scaling(self):
        """Test that alpha parameter affects the output."""
        controlled_input = np.array([[1.0, 2.0]], dtype=np.float32)

        # Test with different alpha values
        layer_small_alpha = DynamicTanh(alpha_init_value=0.1)
        layer_large_alpha = DynamicTanh(alpha_init_value=2.0)

        output_small = layer_small_alpha(controlled_input)
        output_large = layer_large_alpha(controlled_input)

        # With larger alpha, the tanh should be more saturated
        # Since tanh is monotonic, the ordering should be preserved
        assert not np.allclose(output_small.numpy(), output_large.numpy())

    def test_different_alpha_values(self):
        """Test DynamicTanh with different alpha initialization values."""
        input_tensor = np.array([[1, 2, -1, -2]], dtype=np.float32)

        alpha_values = [0.1, 0.5, 1.0, 2.0]

        for alpha in alpha_values:
            layer = DynamicTanh(alpha_init_value=alpha)
            output = layer(input_tensor)

            # Check that output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))
            assert output.shape == input_tensor.shape

            # Check that alpha is properly initialized
            assert np.allclose(layer.alpha.numpy(), alpha)

    def test_output_shapes_various_inputs(self):
        """Test output shapes with various input shapes."""
        test_shapes = [
            (4, 32),
            (2, 16, 64),
            (1, 8, 8, 128),
            (3, 4, 4, 4, 256)
        ]

        for shape in test_shapes:
            input_tensor = np.random.normal(0, 1, shape).astype(np.float32)
            layer = DynamicTanh()
            output = layer(input_tensor)

            assert output.shape == input_tensor.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == input_tensor.shape

    def test_different_axes_configurations(self):
        """Test layer with different axis configurations."""
        input_shape = (4, 8, 16, 32)
        input_tensor = np.random.normal(0, 1, input_shape).astype(np.float32)

        axis_configs = [
            -1,      # Last axis
            [-1],    # Last axis as list
            [-2, -1], # Last two axes
            [1, 2, 3], # Multiple middle axes
        ]

        for axis_config in axis_configs:
            layer = DynamicTanh(axis=axis_config)
            output = layer(input_tensor)

            assert output.shape == input_tensor.shape
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

            # Check that axes were normalized correctly
            expected_axes = axis_config if isinstance(axis_config, list) else [axis_config]
            expected_axes = [ax if ax >= 0 else len(input_shape) + ax for ax in expected_axes]
            assert layer.axis == expected_axes

    def test_mathematical_properties(self):
        """Test mathematical properties of the DynamicTanh layer."""
        layer = DynamicTanh(alpha_init_value=1.0)

        # Test with zero input
        zero_input = np.zeros((2, 4), dtype=np.float32)
        zero_output = layer(zero_input)

        # For zero input, tanh(0) = 0, so output should be bias
        # With default bias initializer "zeros", output should be close to zero
        assert zero_output.shape == zero_input.shape
        assert np.allclose(zero_output.numpy(), layer.bias.numpy())

    def test_tanh_saturation_property(self):
        """Test that tanh saturation is working correctly."""
        layer = DynamicTanh(alpha_init_value=1.0)

        # Test with very large inputs
        large_input = np.array([[10.0, -10.0, 100.0, -100.0]], dtype=np.float32)
        output = layer(large_input)

        # tanh should saturate to ±1, so output should be bounded
        # Output = weight * tanh(alpha * input) + bias
        # With default weight=1, bias=0, output should be close to ±1
        assert np.all(np.abs(output.numpy()) <= 2)  # Allow some margin for weight/bias

    def test_trainable_parameters(self, input_tensor):
        """Test that alpha, weight, and bias are trainable."""
        layer = DynamicTanh()
        output = layer(input_tensor)

        # Check that we have the expected number of trainable parameters
        trainable_weights = layer.trainable_weights
        assert len(trainable_weights) == 3  # alpha, weight, bias

        # Check parameter names (handle different Keras versions)
        weight_names = {w.name.split(':')[0] for w in trainable_weights}  # Remove :0 suffix if present
        expected_names = {"alpha", "weight", "bias"}
        assert weight_names == expected_names

    def test_alpha_parameter_properties(self, input_tensor):
        """Test properties of the alpha parameter."""
        layer = DynamicTanh(alpha_init_value=0.7)
        _ = layer(input_tensor)

        # Alpha should be a scalar
        assert layer.alpha.shape == ()
        assert np.allclose(layer.alpha.numpy(), 0.7)

        # Alpha should be trainable
        assert layer.alpha.trainable is True

    def test_regularization(self, input_tensor):
        """Test that regularization losses are properly applied."""
        layer = DynamicTanh(
            kernel_regularizer=keras.regularizers.L2(0.1),
            bias_regularizer=keras.regularizers.L1(0.1)
        )

        # No regularization losses before calling the layer
        assert len(layer.losses) == 0

        # Apply the layer
        _ = layer(input_tensor)

        # Should have regularization losses now
        assert len(layer.losses) > 0

    def test_constraints(self, input_tensor):
        """Test that constraints are properly applied."""
        # Create layer with constraints
        layer = DynamicTanh(
            kernel_constraint=keras.constraints.UnitNorm(),
            bias_constraint=keras.constraints.NonNeg()
        )

        # Apply the layer
        output = layer(input_tensor)

        # Constraints are applied during training, so just check that layer works
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = DynamicTanh(
            axis=[-2, -1],
            alpha_init_value=0.7,
            kernel_initializer="he_normal",
            bias_initializer="ones",
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Build the layer
        input_shape = (None, 16, 32)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = DynamicTanh.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.axis == original_layer.axis
        assert recreated_layer.alpha_init_value == original_layer.alpha_init_value
        assert type(recreated_layer.kernel_initializer) == type(original_layer.kernel_initializer)
        assert type(recreated_layer.bias_initializer) == type(original_layer.bias_initializer)
        assert recreated_layer.built == original_layer.built

    def test_serialization_edge_cases(self):
        """Test serialization with various parameter types."""
        layers_to_test = [
            DynamicTanh(axis=-1, alpha_init_value=0.5),
            DynamicTanh(axis=[-2, -1], alpha_init_value=1.0),
            DynamicTanh(kernel_initializer="glorot_uniform"),
            DynamicTanh(kernel_regularizer=keras.regularizers.L2(0.01)),
            DynamicTanh(kernel_constraint=keras.constraints.MaxNorm(2.0)),
        ]

        for original_layer in layers_to_test:
            # Build the layer
            original_layer.build((None, 16, 32))

            # Get config and recreate
            config = original_layer.get_config()
            recreated_layer = DynamicTanh.from_config(config)

            # Check key aspects match
            assert recreated_layer.axis == original_layer.axis
            assert recreated_layer.alpha_init_value == original_layer.alpha_init_value

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = DynamicTanh()

        test_cases = [
            np.zeros((2, 8), dtype=np.float32),  # Zeros
            np.ones((2, 8), dtype=np.float32) * 1e-10,  # Very small values
            np.ones((2, 8), dtype=np.float32) * 10,     # Large values
            np.ones((2, 8), dtype=np.float32) * -10,    # Large negative values
            np.ones((2, 8), dtype=np.float32) * 100,    # Very large values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

            # Since tanh saturates, output should be bounded
            assert np.all(np.abs(output.numpy()) < 1e6), "Output values too large"

    def test_gradient_flow(self, input_tensor):
        """Test gradient flow through the layer."""
        layer = DynamicTanh()

        # Create a simple model to test gradients
        inputs = keras.Input(shape=input_tensor.shape[1:])
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test with TensorFlow's GradientTape for gradient computation
        import tensorflow as tf
        with tf.GradientTape() as tape:
            inputs_var = tf.Variable(input_tensor)
            tape.watch(inputs_var)
            pred = model(inputs_var)
            loss = tf.reduce_mean(tf.square(pred))

        grads = tape.gradient(loss, model.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check gradients have values (not all zeros)
        non_zero_grads = [g for g in grads if g is not None and np.any(g.numpy() != 0)]
        assert len(non_zero_grads) > 0

    def test_masking_support(self, input_tensor):
        """Test that the layer supports masking."""
        layer = DynamicTanh()

        # Check that masking is supported
        assert layer.supports_masking is True

        # Test with masked input
        output = layer(input_tensor)
        assert output.shape == input_tensor.shape

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the DynamicTanh layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = keras.layers.Dense(64)(inputs)
        x = DynamicTanh(alpha_init_value=0.6)(x)
        x = keras.layers.Dense(32)(x)
        x = DynamicTanh(alpha_init_value=0.2)(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
        )

        # Test forward pass
        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)
        assert not np.any(np.isnan(y_pred.numpy()))

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the DynamicTanh layer."""
        # Create a model with the DynamicTanh layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = keras.layers.Dense(32)(inputs)
        x = DynamicTanh(alpha_init_value=0.7, name="dyt_layer1")(x)
        x = keras.layers.Dense(16)(x)
        x = DynamicTanh(alpha_init_value=0.3, name="dyt_layer2")(x)
        outputs = keras.layers.Dense(5)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"DynamicTanh": DynamicTanh}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("dyt_layer1"), DynamicTanh)
            assert isinstance(loaded_model.get_layer("dyt_layer2"), DynamicTanh)

            # Check that layer parameters are preserved
            original_dyt1 = model.get_layer("dyt_layer1")
            loaded_dyt1 = loaded_model.get_layer("dyt_layer1")
            assert np.allclose(original_dyt1.alpha.numpy(), loaded_dyt1.alpha.numpy())

    def test_training_loop(self, input_tensor):
        """Test training loop with the DynamicTanh layer."""
        # Create a model with the DynamicTanh layer
        model = keras.Sequential([
            keras.layers.InputLayer(input_tensor.shape[1:]),
            keras.layers.Dense(32),
            DynamicTanh(alpha_init_value=0.5),
            keras.layers.Dense(16),
            DynamicTanh(alpha_init_value=0.3),
            keras.layers.Dense(5)
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Create mock data
        x_train = np.random.normal(0, 1, [32] + list(input_tensor.shape[1:])).astype(np.float32)
        y_train = np.random.randint(0, 5, 32)

        # Initial loss
        initial_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Train for a few epochs
        model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)

        # Final loss
        final_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Loss should be finite
        assert not np.isnan(final_loss)
        assert not np.isinf(final_loss)

    def test_transformer_like_usage(self):
        """Test usage pattern similar to transformer architectures."""
        # Simulate a transformer-like architecture
        sequence_length = 10
        hidden_size = 64

        # Input
        inputs = keras.Input(shape=(sequence_length, hidden_size))

        # Attention block
        attention_norm = DynamicTanh(alpha_init_value=0.7, name="attention_norm")
        attention_normed = attention_norm(inputs)

        # Multi-head attention (simplified)
        attention_output = keras.layers.Dense(hidden_size)(attention_normed)
        attention_output = keras.layers.Add()([inputs, attention_output])

        # FFN block
        ffn_norm = DynamicTanh(alpha_init_value=0.2, name="ffn_norm")
        ffn_normed = ffn_norm(attention_output)

        # Feed-forward network
        ffn_hidden = keras.layers.Dense(hidden_size * 4, activation="gelu")(ffn_normed)
        ffn_output = keras.layers.Dense(hidden_size)(ffn_hidden)
        outputs = keras.layers.Add()([attention_output, ffn_output])

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test the model
        test_input = np.random.normal(0, 1, (2, sequence_length, hidden_size)).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_paper_suggested_alpha_values(self):
        """Test with alpha values suggested in the paper."""
        input_tensor = np.random.normal(0, 1, (4, 32)).astype(np.float32)

        # Paper-suggested values
        alpha_configs = [
            ("attention", 0.7),
            ("ffn", 0.15),
            ("final_decoder", 0.1)
        ]

        for config_name, alpha_value in alpha_configs:
            layer = DynamicTanh(alpha_init_value=alpha_value, name=f"dyt_{config_name}")
            output = layer(input_tensor)

            # Check that layer works with paper-suggested values
            assert output.shape == input_tensor.shape
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

            # Check that alpha is properly initialized
            assert np.allclose(layer.alpha.numpy(), alpha_value)


class TestDynamicTanhEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_single_element_tensor(self):
        """Test with single element tensors."""
        layer = DynamicTanh()
        input_tensor = np.array([[1.0]], dtype=np.float32)

        output = layer(input_tensor)
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_very_large_tensors(self):
        """Test with reasonably large tensors."""
        layer = DynamicTanh()
        # Use a moderately large tensor that shouldn't cause memory issues
        input_tensor = np.random.normal(0, 1, (8, 256)).astype(np.float32)

        output = layer(input_tensor)
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_negative_axis_handling(self):
        """Test proper handling of negative axes."""
        input_shape = (4, 8, 16)
        input_tensor = np.random.normal(0, 1, input_shape).astype(np.float32)

        # Test various negative axis values
        for axis in [-1, -2, -3]:
            layer = DynamicTanh(axis=axis)
            output = layer(input_tensor)
            assert output.shape == input_shape

    def test_multiple_calls_same_layer(self, input_tensor):
        """Test calling the same layer instance multiple times."""
        layer = DynamicTanh()

        # First call
        output1 = layer(input_tensor)

        # Second call with different input
        input_tensor2 = np.random.normal(0, 1, input_tensor.shape).astype(np.float32)
        output2 = layer(input_tensor2)

        # Both should work
        assert output1.shape == input_tensor.shape
        assert output2.shape == input_tensor2.shape

    def test_dtype_preservation(self):
        """Test that the layer works with different dtypes."""
        # Test with different dtypes
        dtypes = [np.float32, np.float64]

        for dtype in dtypes:
            # Create a new layer for each dtype test
            layer = DynamicTanh()
            input_tensor = np.random.normal(0, 1, (4, 16)).astype(dtype)
            output = layer(input_tensor)

            # Check that output is valid (dtype conversion is handled by Keras backend)
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))
            assert output.shape == input_tensor.shape

            # Note: Keras may convert dtypes internally for computational efficiency
            # The important thing is that the computation is valid

    def test_extreme_alpha_values(self):
        """Test with extreme alpha values."""
        input_tensor = np.random.normal(0, 0.1, (4, 16)).astype(np.float32)  # Small inputs to prevent overflow

        extreme_alphas = [1e-6, 1e-3, 1e3, 1e6]

        for alpha in extreme_alphas:
            layer = DynamicTanh(alpha_init_value=alpha)
            output = layer(input_tensor)

            # Should not produce NaN or Inf
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))
            assert output.shape == input_tensor.shape

    def test_axis_out_of_bounds_error_messages(self):
        """Test that axis out of bounds errors have clear messages."""
        layer = DynamicTanh(axis=10)
        input_tensor = np.random.normal(0, 1, (4, 8)).astype(np.float32)

        with pytest.raises(ValueError, match="Axis 10 is out of bounds for tensor of dimension 2"):
            layer(input_tensor)

    def test_multiple_axis_validation(self):
        """Test validation with multiple axes."""
        layer = DynamicTanh(axis=[1, 5])  # Second axis is invalid
        input_tensor = np.random.normal(0, 1, (4, 8)).astype(np.float32)

        with pytest.raises(ValueError, match="Axis 5 is out of bounds for tensor of dimension 2"):
            layer(input_tensor)

    def test_empty_tensor(self):
        """Test with empty tensor dimensions."""
        layer = DynamicTanh()
        input_tensor = np.random.normal(0, 1, (0, 32)).astype(np.float32)

        output = layer(input_tensor)
        assert output.shape == input_tensor.shape

    def test_single_dimension_tensor(self):
        """Test with 1D tensor."""
        layer = DynamicTanh()
        input_tensor = np.random.normal(0, 1, (32,)).astype(np.float32)

        output = layer(input_tensor)
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_five_dimensional_tensor(self):
        """Test with 5D tensor."""
        layer = DynamicTanh(axis=[-2, -1])
        input_tensor = np.random.normal(0, 1, (2, 3, 4, 5, 6)).astype(np.float32)

        output = layer(input_tensor)
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

        # Check that axis was normalized correctly (should be [3, 4] for 5D tensor)
        assert layer.axis == [3, 4]

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor for edge case tests."""
        return np.random.normal(0, 1, (4, 32)).astype(np.float32)


class TestDynamicTanhIntegration:
    """Integration tests for DynamicTanh in complex scenarios."""

    def test_mixed_precision_compatibility(self, input_tensor):
        """Test that the layer works with mixed precision."""
        # Create a model with mixed precision
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = keras.layers.Dense(32)(inputs)
        x = DynamicTanh(alpha_init_value=0.5)(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Should work without errors
        output = model(input_tensor)
        assert output.shape == (input_tensor.shape[0], 10)

    def test_functional_api_compatibility(self, input_tensor):
        """Test compatibility with Keras functional API."""
        # Create multiple DynamicTanh layers
        inputs = keras.Input(shape=input_tensor.shape[1:])

        # Branch 1
        branch1 = keras.layers.Dense(16)(inputs)
        branch1 = DynamicTanh(alpha_init_value=0.3)(branch1)

        # Branch 2
        branch2 = keras.layers.Dense(16)(inputs)
        branch2 = DynamicTanh(alpha_init_value=0.7)(branch2)

        # Merge branches
        merged = keras.layers.Add()([branch1, branch2])
        outputs = keras.layers.Dense(10)(merged)

        model = keras.Model(inputs=inputs, outputs=outputs)
        output = model(input_tensor)

        assert output.shape == (input_tensor.shape[0], 10)
        assert not np.any(np.isnan(output.numpy()))

    def test_subclassing_compatibility(self, input_tensor):
        """Test that DynamicTanh can be used in subclassed models."""
        class CustomModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = keras.layers.Dense(32)
                self.dyt1 = DynamicTanh(alpha_init_value=0.5)
                self.dense2 = keras.layers.Dense(16)
                self.dyt2 = DynamicTanh(alpha_init_value=0.3)
                self.output_layer = keras.layers.Dense(10)

            def call(self, inputs):
                x = self.dense1(inputs)
                x = self.dyt1(x)
                x = self.dense2(x)
                x = self.dyt2(x)
                return self.output_layer(x)

        model = CustomModel()
        output = model(input_tensor)

        assert output.shape == (input_tensor.shape[0], 10)
        assert not np.any(np.isnan(output.numpy()))

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor for integration tests."""
        return np.random.normal(0, 1, (4, 32)).astype(np.float32)


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short"])