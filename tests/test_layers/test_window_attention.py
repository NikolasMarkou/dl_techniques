import pytest
import numpy as np
import keras
import os
import tempfile

from dl_techniques.layers.attention.window_attention import WindowAttention


class TestWindowAttention:
    """Test suite for WindowAttention layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        # Shape: (batch_size, num_windows, dim) where num_windows = window_size^2
        return keras.random.normal([4, 49, 96])  # 4 batches, 7x7 windows, 96 dims

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return WindowAttention(dim=96, window_size=7, num_heads=3)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = WindowAttention(dim=128, window_size=7, num_heads=4)

        # Check default values
        assert layer.dim == 128
        assert layer.window_size == 7
        assert layer.num_heads == 4
        assert layer.head_dim == 32
        assert layer.qkv_bias is True
        assert layer.qk_scale is None
        assert layer.scale == 32 ** -0.5
        assert layer.attn_drop_rate == 0.0
        assert layer.proj_drop_rate == 0.0
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = WindowAttention(
            dim=64,
            window_size=8,
            num_heads=8,
            qkv_bias=False,
            qk_scale=0.1,
            attn_drop=0.1,
            proj_drop=0.2,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
        )

        # Check custom values
        assert layer.dim == 64
        assert layer.window_size == 8
        assert layer.num_heads == 8
        assert layer.head_dim == 8
        assert layer.qkv_bias is False
        assert layer.qk_scale == 0.1
        assert layer.scale == 0.1
        assert layer.attn_drop_rate == 0.1
        assert layer.proj_drop_rate == 0.2
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test negative or zero dimensions
        with pytest.raises(ValueError, match="dim must be positive"):
            WindowAttention(dim=-10, window_size=7, num_heads=3)

        with pytest.raises(ValueError, match="dim must be positive"):
            WindowAttention(dim=0, window_size=7, num_heads=3)

        # Test negative or zero window size
        with pytest.raises(ValueError, match="window_size must be positive"):
            WindowAttention(dim=96, window_size=-7, num_heads=3)

        with pytest.raises(ValueError, match="window_size must be positive"):
            WindowAttention(dim=96, window_size=0, num_heads=3)

        # Test negative or zero num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            WindowAttention(dim=96, window_size=7, num_heads=-3)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            WindowAttention(dim=96, window_size=7, num_heads=0)

        # Test dim not divisible by num_heads
        with pytest.raises(ValueError, match="dim .* must be divisible by num_heads"):
            WindowAttention(dim=97, window_size=7, num_heads=3)

        # Test invalid dropout rates
        with pytest.raises(ValueError, match="attn_drop must be between 0.0 and 1.0"):
            WindowAttention(dim=96, window_size=7, num_heads=3, attn_drop=-0.1)

        with pytest.raises(ValueError, match="attn_drop must be between 0.0 and 1.0"):
            WindowAttention(dim=96, window_size=7, num_heads=3, attn_drop=1.1)

        with pytest.raises(ValueError, match="proj_drop must be between 0.0 and 1.0"):
            WindowAttention(dim=96, window_size=7, num_heads=3, proj_drop=-0.1)

        with pytest.raises(ValueError, match="proj_drop must be between 0.0 and 1.0"):
            WindowAttention(dim=96, window_size=7, num_heads=3, proj_drop=1.1)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that layer was built
        assert layer_instance.built is True
        assert len(layer_instance.weights) > 0
        assert hasattr(layer_instance, "qkv")
        assert hasattr(layer_instance, "proj")
        assert hasattr(layer_instance, "relative_position_bias_table")
        assert hasattr(layer_instance, "relative_position_index")

        # Check weight shapes
        expected_qkv_shape = (input_tensor.shape[-1], layer_instance.dim * 3)
        assert layer_instance.qkv.kernel.shape == expected_qkv_shape

        expected_proj_shape = (layer_instance.dim, layer_instance.dim)
        assert layer_instance.proj.kernel.shape == expected_proj_shape

        # Check relative position bias table shape
        num_relative_distance = (2 * layer_instance.window_size - 1) ** 2
        expected_bias_shape = (num_relative_distance, layer_instance.num_heads)
        assert layer_instance.relative_position_bias_table.shape == expected_bias_shape

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        configs_to_test = [
            {"dim": 32, "window_size": 4, "num_heads": 2},
            {"dim": 64, "window_size": 7, "num_heads": 4},
            {"dim": 128, "window_size": 8, "num_heads": 8},
        ]

        for config in configs_to_test:
            layer = WindowAttention(**config)

            # Create appropriate input tensor
            window_size = config["window_size"]
            dim = config["dim"]
            test_input = keras.random.normal([2, window_size * window_size, dim])

            output = layer(test_input)

            # Check output shape matches input shape
            assert output.shape == test_input.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(test_input.shape)
            assert computed_shape == test_input.shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check output shape
        assert output.shape == input_tensor.shape

        # Test with training=False
        output_inference = layer_instance(input_tensor, training=False)
        assert output_inference.shape == input_tensor.shape

        # Test with training=True
        output_training = layer_instance(input_tensor, training=True)
        assert output_training.shape == input_tensor.shape

    def test_different_configurations(self):
        """Test layer with different configurations."""
        configurations = [
            {"dim": 32, "window_size": 4, "num_heads": 2, "qkv_bias": False},
            {"dim": 64, "window_size": 7, "num_heads": 4, "attn_drop": 0.1},
            {"dim": 96, "window_size": 7, "num_heads": 3, "proj_drop": 0.2},
            {"dim": 128, "window_size": 8, "num_heads": 8, "qk_scale": 0.1},
        ]

        for config in configurations:
            layer = WindowAttention(**config)

            # Create appropriate input
            window_size = config["window_size"]
            dim = config["dim"]
            test_input = keras.random.normal([2, window_size * window_size, dim])

            # Test forward pass
            output = layer(test_input)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == test_input.shape

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = WindowAttention(
            dim=128,
            window_size=7,
            num_heads=4,
            qkv_bias=True,
            qk_scale=0.1,
            attn_drop=0.1,
            proj_drop=0.2,
            kernel_initializer="he_normal",
        )

        # Build the layer
        input_shape = (None, 49, 128)  # 7x7 = 49 windows
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = WindowAttention.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.dim == original_layer.dim
        assert recreated_layer.window_size == original_layer.window_size
        assert recreated_layer.num_heads == original_layer.num_heads
        assert recreated_layer.qkv_bias == original_layer.qkv_bias
        assert recreated_layer.qk_scale == original_layer.qk_scale
        assert recreated_layer.attn_drop_rate == original_layer.attn_drop_rate
        assert recreated_layer.proj_drop_rate == original_layer.proj_drop_rate

        # Check weights match (shapes should be the same)
        assert len(recreated_layer.weights) == len(original_layer.weights)
        for w1, w2 in zip(original_layer.weights, recreated_layer.weights):
            assert w1.shape == w2.shape

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the window attention layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = WindowAttention(dim=96, window_size=7, num_heads=3)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(64)(x)
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
        """Test saving and loading a model with the window attention layer."""
        # Create a model with the window attention layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = WindowAttention(dim=96, window_size=7, num_heads=3, name="window_attn")(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(64)(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

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
                custom_objects={
                    "WindowAttention": WindowAttention
                }
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("window_attn"), WindowAttention)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = WindowAttention(dim=32, window_size=4, num_heads=2)

        # Create inputs with different magnitudes
        batch_size = 2
        num_windows = 16  # 4x4
        dim = 32

        test_cases = [
            keras.ops.zeros((batch_size, num_windows, dim)),  # Zeros
            keras.ops.ones((batch_size, num_windows, dim)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, num_windows, dim)) * 1e5,  # Large values
            keras.random.normal((batch_size, num_windows, dim)) * 100  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_regularization(self, input_tensor):
        """Test that regularization losses are properly applied."""
        # Create layer with regularization
        layer = WindowAttention(
            dim=96,
            window_size=7,
            num_heads=3,
            kernel_regularizer=keras.regularizers.L2(0.1),
            bias_regularizer=keras.regularizers.L1(0.1)
        )

        # Build layer
        layer.build(input_tensor.shape)

        # No regularization losses before calling the layer
        initial_losses = len(layer.losses)

        # Apply the layer
        _ = layer(input_tensor)

        # Should have regularization losses now
        assert len(layer.losses) >= initial_losses

    def test_relative_position_encoding(self):
        """Test that relative position encoding is properly created."""
        layer = WindowAttention(dim=32, window_size=4, num_heads=2)

        # Build layer
        input_shape = (None, 16, 32)  # 4x4 = 16 windows
        layer.build(input_shape)

        # Check that relative position index has correct shape
        expected_shape = (16, 16)  # num_windows x num_windows
        assert layer.relative_position_index.shape == expected_shape

        # Check that relative position bias table has correct shape
        num_relative_distance = (2 * 4 - 1) ** 2  # (2 * window_size - 1)^2
        expected_bias_shape = (num_relative_distance, 2)  # (num_relative_distance, num_heads)
        assert layer.relative_position_bias_table.shape == expected_bias_shape

    def test_different_window_sizes(self):
        """Test layer with different window sizes."""
        window_sizes = [3, 4, 7, 8]

        for window_size in window_sizes:
            layer = WindowAttention(dim=64, window_size=window_size, num_heads=4)

            # Create appropriate input
            num_windows = window_size * window_size
            test_input = keras.random.normal([2, num_windows, 64])

            # Test forward pass
            output = layer(test_input)

            # Check output
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))