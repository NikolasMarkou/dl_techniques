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
        """Create a test input tensor for a full window."""
        # Shape: (batch_size, num_tokens, dim) where num_tokens = window_size^2
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
        assert layer.attn_dropout_rate == 0.0
        assert layer.proj_dropout_rate == 0.0
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
            attn_dropout_rate=0.1,
            proj_dropout_rate=0.2,
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
        assert layer.attn_dropout_rate == 0.1
        assert layer.proj_dropout_rate == 0.2
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

        # Test invalid dropout rates - using correct parameter names
        with pytest.raises(ValueError, match="attn_dropout_rate must be between 0.0 and 1.0"):
            WindowAttention(dim=96, window_size=7, num_heads=3, attn_dropout_rate=-0.1)

        with pytest.raises(ValueError, match="attn_dropout_rate must be between 0.0 and 1.0"):
            WindowAttention(dim=96, window_size=7, num_heads=3, attn_dropout_rate=1.1)

        with pytest.raises(ValueError, match="proj_dropout_rate must be between 0.0 and 1.0"):
            WindowAttention(dim=96, window_size=7, num_heads=3, proj_dropout_rate=-0.1)

        with pytest.raises(ValueError, match="proj_dropout_rate must be between 0.0 and 1.0"):
            WindowAttention(dim=96, window_size=7, num_heads=3, proj_dropout_rate=1.1)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        output = layer_instance(input_tensor)

        assert layer_instance.built is True
        assert len(layer_instance.weights) > 0
        assert hasattr(layer_instance, "qkv")
        assert hasattr(layer_instance, "proj")
        assert hasattr(layer_instance, "relative_position_bias_table")
        assert hasattr(layer_instance, "relative_position_index")

        expected_qkv_shape = (input_tensor.shape[-1], layer_instance.dim * 3)
        assert layer_instance.qkv.kernel.shape == expected_qkv_shape

        expected_proj_shape = (layer_instance.dim, layer_instance.dim)
        assert layer_instance.proj.kernel.shape == expected_proj_shape

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
            window_size = config["window_size"]
            dim = config["dim"]
            test_input = keras.random.normal([2, window_size * window_size, dim])
            output = layer(test_input)
            assert output.shape == test_input.shape
            computed_shape = layer.compute_output_shape(test_input.shape)
            assert computed_shape == test_input.shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))
        assert output.shape == input_tensor.shape

        output_inference = layer_instance(input_tensor, training=False)
        assert output_inference.shape == input_tensor.shape

        output_training = layer_instance(input_tensor, training=True)
        assert output_training.shape == input_tensor.shape

    # --- NEW TESTS FOR PADDING FUNCTIONALITY ---

    @pytest.mark.parametrize(
        "dim, window_size, num_heads, seq_len",
        [
            (32, 4, 2, 16),  # Full window, no padding
            (32, 4, 2, 10),  # Partial window, requires padding
            (32, 4, 2, 1),   # Minimal window, requires padding
            (96, 7, 3, 30),  # Another partial window case
        ]
    )
    def test_padding_and_unpadding(self, dim, window_size, num_heads, seq_len):
        """Test that padding and unpadding works for various sequence lengths."""
        layer = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        input_tensor = keras.random.normal((4, seq_len, dim))

        # Forward pass
        output = layer(input_tensor)

        # CRITICAL: Output shape must match the original input shape, not the padded one.
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_forward_pass_with_too_long_sequence(self):
        """Test that an error is raised if the input sequence is too long."""
        layer = WindowAttention(dim=32, window_size=4, num_heads=2)
        # Window area is 4*4=16, input length is 17
        invalid_input = keras.random.normal((2, 17, 32))

        with pytest.raises(ValueError, match="Input sequence length .* cannot be greater than"):
            layer(invalid_input)

    def test_attention_mask_integration(self):
        """Test that user-provided attention mask integrates with padding mask."""
        batch_size = 2
        dim = 32
        window_size = 5
        num_heads = 4
        N_target = window_size * window_size  # 25
        N_actual = 20  # A partial sequence

        layer = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        input_tensor = keras.random.normal((batch_size, N_actual, dim))

        # Create a user mask for the original sequence length
        user_mask = np.ones((batch_size, N_actual), dtype="int32")
        user_mask[:, -5:] = 0  # Mask out the last 5 tokens

        # Call the layer with the user mask
        output = layer(input_tensor, attention_mask=keras.ops.convert_to_tensor(user_mask))

        # The output shape should still match the original input
        assert output.shape == input_tensor.shape

        # Test with a full window and a user mask (no internal padding mask)
        full_input = keras.random.normal((batch_size, N_target, dim))
        full_user_mask = np.ones((batch_size, N_target), dtype="int32")
        full_user_mask[:, -3:] = 0 # Mask last 3 tokens
        full_output = layer(full_input, attention_mask=keras.ops.convert_to_tensor(full_user_mask))
        assert full_output.shape == full_input.shape

    def test_gradient_flow_with_padding(self):
        """Ensure gradients flow correctly when padding is active."""
        dim, window_size, num_heads, seq_len = 32, 4, 2, 10
        layer = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)

        # Using tf.Variable to track gradients
        import tensorflow as tf
        input_tensor = tf.Variable(keras.random.normal((2, seq_len, dim)))

        with tf.GradientTape() as tape:
            output = layer(input_tensor)
            # A simple loss function to get a scalar value for gradient computation
            # FIX: Changed `reduce_sum` to `sum`
            loss = keras.ops.sum(output)

        # Compute gradients with respect to all trainable variables in the layer
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check that gradients were computed for all trainable variables
        assert len(grads) == len(layer.trainable_variables)
        for grad, var in zip(grads, layer.trainable_variables):
            assert grad is not None, f"Gradient is None for variable: {var.name}"
            # Check that gradients are not all zero, indicating they flowed correctly
            assert keras.ops.any(keras.ops.not_equal(grad, 0)), f"Gradient is all zeros for variable: {var.name}"

    # --- END OF NEW TESTS ---

    def test_different_configurations(self):
        """Test layer with different configurations."""
        configurations = [
            {"dim": 32, "window_size": 4, "num_heads": 2, "qkv_bias": False},
            {"dim": 64, "window_size": 7, "num_heads": 4, "attn_dropout_rate": 0.1},
            {"dim": 96, "window_size": 7, "num_heads": 3, "proj_dropout_rate": 0.2},
            {"dim": 128, "window_size": 8, "num_heads": 8, "qk_scale": 0.1},
        ]

        for config in configurations:
            layer = WindowAttention(**config)
            window_size = config["window_size"]
            dim = config["dim"]
            test_input = keras.random.normal([2, window_size * window_size, dim])
            output = layer(test_input)
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
            attn_dropout_rate=0.1,
            proj_dropout_rate=0.2,
            kernel_initializer="he_normal",
        )
        input_shape = (None, 49, 128)
        original_layer.build(input_shape)
        config = original_layer.get_config()
        recreated_layer = WindowAttention.from_config(config)
        recreated_layer.build(input_shape)

        assert recreated_layer.dim == original_layer.dim
        assert recreated_layer.window_size == original_layer.window_size
        assert recreated_layer.num_heads == original_layer.num_heads
        assert recreated_layer.qkv_bias == original_layer.qkv_bias
        assert recreated_layer.qk_scale == original_layer.qk_scale
        assert recreated_layer.attn_dropout_rate == original_layer.attn_dropout_rate
        assert recreated_layer.proj_dropout_rate == original_layer.proj_dropout_rate
        assert len(recreated_layer.weights) == len(original_layer.weights)
        for w1, w2 in zip(original_layer.weights, recreated_layer.weights):
            assert w1.shape == w2.shape

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = WindowAttention(dim=96, window_size=7, num_heads=3)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(64)(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the window attention layer."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = WindowAttention(dim=96, window_size=7, num_heads=3, name="window_attn")(inputs)
        x = keras.layers.LayerNormalization()(x)
        outputs = keras.layers.Dense(10)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)
            assert isinstance(loaded_model.get_layer("window_attn"), WindowAttention)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = WindowAttention(dim=32, window_size=4, num_heads=2)
        batch_size, num_tokens, dim = 2, 16, 32
        test_cases = [
            keras.ops.zeros((batch_size, num_tokens, dim)),
            keras.ops.ones((batch_size, num_tokens, dim)) * 1e-10,
            keras.ops.ones((batch_size, num_tokens, dim)) * 1e5,
            keras.random.normal((batch_size, num_tokens, dim)) * 100,
        ]
        for test_input in test_cases:
            output = layer(test_input)
            assert not np.any(np.isnan(output.numpy())), "NaN values detected"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected"

    def test_regularization(self, input_tensor):
        """Test that regularization losses are properly applied."""
        layer = WindowAttention(
            dim=96,
            window_size=7,
            num_heads=3,
            kernel_regularizer=keras.regularizers.L2(0.1),
            bias_regularizer=keras.regularizers.L1(0.1)
        )

        # Call the layer to compute the losses
        _ = layer(input_tensor)

        # FIX: Check that losses exist and their computed value is > 0
        assert len(layer.losses) > 0
        total_regularization_loss = sum(layer.losses)
        assert total_regularization_loss > 0.0


    def test_relative_position_encoding(self):
        """Test that relative position encoding is properly created."""
        layer = WindowAttention(dim=32, window_size=4, num_heads=2)
        input_shape = (None, 16, 32)
        layer.build(input_shape)

        expected_shape = (16, 16)
        assert layer.relative_position_index.shape == expected_shape

        num_relative_distance = (2 * 4 - 1) ** 2
        expected_bias_shape = (num_relative_distance, 2)
        assert layer.relative_position_bias_table.shape == expected_bias_shape

    def test_different_window_sizes(self):
        """Test layer with different window sizes."""
        for window_size in [3, 4, 7, 8]:
            layer = WindowAttention(dim=64, window_size=window_size, num_heads=4)
            num_tokens = window_size * window_size
            test_input = keras.random.normal([2, num_tokens, 64])
            output = layer(test_input)
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])