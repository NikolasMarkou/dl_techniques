"""
Test suite for SwinMLP layer.

This file should be placed in tests/test_layers/test_swin_mlp.py
"""
import pytest
import numpy as np
import keras
import tempfile
import os

from dl_techniques.layers.ffn.swin_mlp import SwinMLP


class TestSwinMLP:
    """Comprehensive test suite for SwinMLP layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([8, 16, 128])  # batch, seq_len, features

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return SwinMLP(hidden_dim=256)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = SwinMLP(hidden_dim=128)

        # Check default values
        assert layer.hidden_dim == 128
        assert layer.out_dim is None
        assert layer.act_layer == "gelu"
        assert layer.dropout_rate == 0.0
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.activity_regularizer is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = SwinMLP(
            hidden_dim=256,
            out_dim=64,
            act_layer="relu",
            dropout_rate=0.2,
            kernel_initializer="he_normal",
            bias_initializer="ones",
            kernel_regularizer=custom_regularizer,
            bias_regularizer=custom_regularizer,
            activity_regularizer=custom_regularizer,
        )

        # Check custom values
        assert layer.hidden_dim == 256
        assert layer.out_dim == 64
        assert layer.act_layer == "relu"
        assert layer.dropout_rate == 0.2
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.bias_regularizer == custom_regularizer
        assert layer.activity_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test negative hidden_dim
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            SwinMLP(hidden_dim=-10)

        # Test zero hidden_dim
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            SwinMLP(hidden_dim=0)

        # Test invalid dropout rate
        with pytest.raises(ValueError, match="drop rate must be between 0.0 and 1.0"):
            SwinMLP(hidden_dim=64, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="drop rate must be between 0.0 and 1.0"):
            SwinMLP(hidden_dim=64, dropout_rate=1.5)

    def test_build_process(self, input_tensor):
        """Test that the layer builds properly."""
        layer = SwinMLP(hidden_dim=256, dropout_rate=0.1)
        layer(input_tensor)  # Forward pass triggers build

        # Check that weights were created
        assert layer.built is True
        assert len(layer.weights) > 0
        assert hasattr(layer, "fc1")
        assert hasattr(layer, "fc2")
        assert layer.fc1 is not None
        assert layer.fc2 is not None

        # Check sublayers were built
        assert layer.fc1.built is True
        assert layer.fc2.built is True

        # Check dropout layers were created
        assert layer.drop1 is not None
        assert layer.drop2 is not None

    def test_build_process_no_dropout(self, input_tensor):
        """Test build process without dropout."""
        layer = SwinMLP(hidden_dim=256, dropout_rate=0.0)
        layer(input_tensor)

        # Check that dropout layers were not created
        assert layer.drop1 is None
        assert layer.drop2 is None

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        test_cases = [
            (128, None),  # No output dim specified
            (256, 64),  # Custom output dim
            (512, 32),  # Another custom output dim
        ]

        for hidden_dim, out_dim in test_cases:
            layer = SwinMLP(hidden_dim=hidden_dim, out_dim=out_dim)
            output = layer(input_tensor)

            # Check output shape
            expected_shape = list(input_tensor.shape)
            if out_dim is not None:
                expected_shape[-1] = out_dim
            expected_shape = tuple(expected_shape)

            assert output.shape == expected_shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == expected_shape

    def test_forward_pass(self, input_tensor):
        """Test that forward pass produces expected values."""
        layer = SwinMLP(hidden_dim=256)
        output = layer(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Test with controlled inputs for deterministic output
        controlled_input = keras.ops.ones([2, 4, 8])
        deterministic_layer = SwinMLP(
            hidden_dim=16,
            out_dim=4,
            kernel_initializer="ones",
            bias_initializer="zeros",
            act_layer="linear"
        )
        result = deterministic_layer(controlled_input)

        # Check output shape
        assert result.shape == (2, 4, 4)
        assert not np.any(np.isnan(result.numpy()))

    def test_different_activations(self, input_tensor):
        """Test layer with different activation functions."""
        activations = ["relu", "gelu", "swish", "tanh", "linear"]

        for act in activations:
            layer = SwinMLP(hidden_dim=128, act_layer=act)
            output = layer(input_tensor)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == input_tensor.shape  # Same shape since out_dim is None

    def test_dropout_behavior(self, input_tensor):
        """Test dropout behavior during training vs inference."""
        layer = SwinMLP(hidden_dim=128, dropout_rate=0.5)

        # Test inference mode (no dropout)
        output_inference = layer(input_tensor, training=False)

        # Test training mode (with dropout)
        output_training = layer(input_tensor, training=True)

        # Both should have same shape
        assert output_inference.shape == output_training.shape
        assert output_inference.shape == input_tensor.shape

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = SwinMLP(
            hidden_dim=256,
            out_dim=64,
            act_layer="relu",
            dropout_rate=0.1,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Build the layer
        input_shape = (None, 16, 128)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = SwinMLP.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.hidden_dim == original_layer.hidden_dim
        assert recreated_layer.out_dim == original_layer.out_dim
        assert recreated_layer.act_layer == original_layer.act_layer
        assert recreated_layer.dropout_rate == original_layer.dropout_rate

        # Check weights match (shapes should be the same)
        assert len(recreated_layer.weights) == len(original_layer.weights)
        for w1, w2 in zip(original_layer.weights, recreated_layer.weights):
            assert w1.shape == w2.shape

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = SwinMLP(hidden_dim=256)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = SwinMLP(hidden_dim=128, out_dim=64)(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
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

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the custom layer."""
        # Create a model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = SwinMLP(hidden_dim=128, name="swin_mlp1")(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = SwinMLP(hidden_dim=64, name="swin_mlp2")(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
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
                custom_objects={"SwinMLP": SwinMLP}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("swin_mlp1"), SwinMLP)
            assert isinstance(loaded_model.get_layer("swin_mlp2"), SwinMLP)

    def test_regularization(self, input_tensor):
        """Test that regularization losses are properly applied."""
        # Create layer with regularization
        layer = SwinMLP(
            hidden_dim=128,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01)
        )

        # No regularization losses before calling the layer
        assert len(layer.losses) == 0

        # Apply the layer
        _ = layer(input_tensor)

        # Should have regularization losses now
        assert len(layer.losses) > 0

    def test_shape_handling(self):
        """Test shape handling with different input formats."""
        layer = SwinMLP(hidden_dim=64, out_dim=32)

        # Test with tuple shape
        tuple_shape = (None, 16, 128)
        output_shape = layer.compute_output_shape(tuple_shape)
        assert output_shape == (None, 16, 32)

        # Test with list shape
        list_shape = [None, 16, 128]
        output_shape = layer.compute_output_shape(list_shape)
        assert output_shape == (None, 16, 32)

        # Test without out_dim (should preserve input shape)
        layer_no_out = SwinMLP(hidden_dim=64)
        output_shape = layer_no_out.compute_output_shape(tuple_shape)
        assert output_shape == tuple_shape

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = SwinMLP(hidden_dim=32)

        # Create inputs with different magnitudes
        batch_size = 4
        seq_len = 8
        features = 16

        test_cases = [
            keras.ops.zeros((batch_size, seq_len, features)),  # Zeros
            keras.ops.ones((batch_size, seq_len, features)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, seq_len, features)) * 1e5,  # Large values
            keras.random.normal((batch_size, seq_len, features)) * 100  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_different_input_shapes(self):
        """Test layer with different input shapes."""
        layer = SwinMLP(hidden_dim=64)

        # Test different input shapes
        test_shapes = [
            (2, 32),  # 2D input
            (4, 16, 32),  # 3D input
            (2, 8, 16, 32),  # 4D input
        ]

        for shape in test_shapes:
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Output should have same shape as input (no out_dim specified)
            assert output.shape == test_input.shape

    def test_training_vs_inference_mode(self, input_tensor):
        """Test behavior differences between training and inference modes."""
        layer = SwinMLP(hidden_dim=128, dropout_rate=0.3)

        # Test multiple calls in inference mode - should be consistent
        output1 = layer(input_tensor, training=False)
        output2 = layer(input_tensor, training=False)

        # In inference mode, outputs should be identical (no dropout)
        assert np.allclose(output1.numpy(), output2.numpy())

        # Test training mode - outputs may differ due to dropout
        output_train1 = layer(input_tensor, training=True)
        output_train2 = layer(input_tensor, training=True)

        # Outputs should have the same shape
        assert output_train1.shape == output_train2.shape == input_tensor.shape

    def test_layer_weights_structure(self, input_tensor):
        """Test that layer weights have expected structure."""
        layer = SwinMLP(hidden_dim=256, out_dim=64)
        layer(input_tensor)  # Build the layer

        # Should have weights from two dense layers
        # Each dense layer has kernel + bias = 4 total weights
        assert len(layer.weights) == 4

        # Check weight shapes
        input_dim = input_tensor.shape[-1]
        fc1_kernel_shape = (input_dim, 256)
        fc1_bias_shape = (256,)
        fc2_kernel_shape = (256, 64)
        fc2_bias_shape = (64,)

        expected_shapes = [fc1_kernel_shape, fc1_bias_shape, fc2_kernel_shape, fc2_bias_shape]
        actual_shapes = [w.shape for w in layer.weights]

        assert actual_shapes == expected_shapes


class TestSwinMLPEdgeCases:
    """Test edge cases and error handling for SwinMLP."""

    def test_very_small_hidden_dim(self):
        """Test with very small hidden dimension."""
        layer = SwinMLP(hidden_dim=1, out_dim=1)
        test_input = keras.random.normal([2, 3, 4])

        output = layer(test_input)
        assert output.shape == (2, 3, 1)
        assert not np.any(np.isnan(output.numpy()))

    def test_large_hidden_dim(self):
        """Test with large hidden dimension."""
        layer = SwinMLP(hidden_dim=2048)
        test_input = keras.random.normal([2, 4, 8])

        output = layer(test_input)
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_custom_activation_callable(self):
        """Test with custom activation function as callable."""

        def custom_activation(x):
            return keras.ops.relu(x) * 0.5

        layer = SwinMLP(hidden_dim=64, act_layer=custom_activation)
        test_input = keras.random.normal([2, 4, 8])

        output = layer(test_input)
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))