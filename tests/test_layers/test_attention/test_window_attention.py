import pytest
import numpy as np
import keras
import os
import tempfile
import tensorflow as tf

from dl_techniques.layers.attention.window_attention import WindowAttention, SingleWindowAttention


def build_transformer_block(inputs, dim, window_size, num_heads, mlp_ratio=4.0):
    """Helper function to build a standard Transformer block for testing."""
    x1 = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    attn_out = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)(x1)
    x = keras.layers.Add()([inputs, attn_out])
    x2 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_out = keras.layers.Dense(int(dim * mlp_ratio), activation="gelu")(x2)
    mlp_out = keras.layers.Dense(dim)(mlp_out)
    return keras.layers.Add()([x, mlp_out])


class TestWindowAttention:
    """Test suite for the refactored WindowAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor with an arbitrary sequence length."""
        return keras.random.normal([4, 50, 96])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return WindowAttention(dim=96, window_size=7, num_heads=3)

    def test_initialization_defaults(self):
        """Test initialization with default parameters on the inner attention layer."""
        layer = WindowAttention(dim=128, window_size=7, num_heads=4)
        inner_attn = layer.attention
        assert isinstance(inner_attn, SingleWindowAttention)
        assert inner_attn.dim == 128
        assert inner_attn.window_size == 7
        assert inner_attn.num_heads == 4
        assert inner_attn.head_dim == 32
        assert inner_attn.qkv_bias is True
        assert inner_attn.qk_scale is None
        assert inner_attn.scale == 32 ** -0.5
        assert inner_attn.dropout_rate == 0.0
        assert isinstance(inner_attn.kernel_initializer, keras.initializers.GlorotUniform)

    def test_initialization_custom(self):
        """Test initialization with custom parameters passed to the inner layer."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        layer = WindowAttention(
            dim=64, window_size=8, num_heads=8, qkv_bias=False, qk_scale=0.1,
            dropout_rate=0.1, kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
        )
        inner_attn = layer.attention
        assert inner_attn.dim == 64
        assert inner_attn.window_size == 8
        assert inner_attn.num_heads == 8
        assert inner_attn.head_dim == 8
        assert inner_attn.qkv_bias is False
        assert inner_attn.qk_scale == 0.1
        assert inner_attn.scale == 0.1
        assert inner_attn.dropout_rate == 0.1
        assert isinstance(inner_attn.kernel_initializer, keras.initializers.HeNormal)
        assert inner_attn.kernel_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors during initialization."""
        with pytest.raises(ValueError, match="dim must be positive"):
            WindowAttention(dim=-10, window_size=7, num_heads=3)
        with pytest.raises(ValueError, match="window_size must be positive"):
            WindowAttention(dim=96, window_size=0, num_heads=3)
        with pytest.raises(ValueError, match="num_heads must be positive"):
            WindowAttention(dim=96, window_size=7, num_heads=-3)
        with pytest.raises(ValueError, match="dim .* must be divisible by num_heads"):
            WindowAttention(dim=97, window_size=7, num_heads=3)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer and its sub-layer build properly."""
        output = layer_instance(input_tensor)
        inner_attn = layer_instance.attention
        assert layer_instance.built is True
        assert inner_attn.built is True
        assert len(inner_attn.weights) > 0
        expected_qkv_shape = (input_tensor.shape[-1], inner_attn.dim * 3)
        assert inner_attn.qkv.kernel.shape == expected_qkv_shape
        expected_proj_shape = (inner_attn.dim, inner_attn.dim)
        assert inner_attn.proj.kernel.shape == expected_proj_shape
        num_relative_distance = (2 * inner_attn.window_size - 1) ** 2
        expected_bias_shape = (num_relative_distance, inner_attn.num_heads)
        assert inner_attn.relative_position_bias_table.shape == expected_bias_shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test forward pass for training and inference modes."""
        output = layer_instance(input_tensor)
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))
        assert output.shape == input_tensor.shape
        output_inference = layer_instance(input_tensor, training=False)
        assert output_inference.shape == input_tensor.shape
        output_training = layer_instance(input_tensor, training=True)
        assert output_training.shape == input_tensor.shape

    @pytest.mark.parametrize(
        "dim, window_size, num_heads, seq_len",
        [
            (32, 4, 2, 16),  # Perfect grid, perfect window fit
            (32, 4, 2, 15),  # Imperfect grid
            (32, 7, 2, 100),  # Grid needs padding for windowing
            (32, 4, 2, 1),  # Single token
            (96, 7, 3, 30),  # Original test case
            (64, 4, 4, 64),  # Perfect grid (8x8), perfect window fit
            (64, 7, 4, 81),  # Perfect grid (9x9), imperfect window fit
            (128, 8, 8, 200),  # Large non-square sequence
            (32, 8, 2, 50),  # Window size equals grid size (8x8)
            (32, 4, 2, 2),  # Very small sequence
        ]
    )
    def test_arbitrary_sequence_lengths(self, dim, window_size, num_heads, seq_len):
        """Test that padding and unpadding works for various sequence lengths."""
        layer = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        input_tensor = keras.random.normal((4, seq_len, dim))
        output = layer(input_tensor)
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize(
        "config, seq_len",
        [
            ({"dim": 32, "window_size": 4, "num_heads": 2}, 17),
            ({"dim": 60, "window_size": 5, "num_heads": 5, "qkv_bias": False}, 99),
            ({"dim": 128, "window_size": 8, "num_heads": 16, "dropout_rate": 0.1}, 250),
        ]
    )
    def test_comprehensive_configurations(self, config, seq_len):
        """Test various layer configurations with different sequence lengths."""
        layer = WindowAttention(**config)
        input_tensor = keras.random.normal((2, seq_len, config["dim"]))
        output = layer(input_tensor, training=True)
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_attention_mask_integration(self):
        """Test that a user-provided attention mask works correctly."""
        layer = WindowAttention(dim=32, window_size=5, num_heads=4)
        input_tensor = keras.random.normal((2, 55, 32))
        user_mask = np.ones((2, 55), dtype="int32")
        user_mask[:, -5:] = 0
        output = layer(input_tensor, attention_mask=keras.ops.convert_to_tensor(user_mask))
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_gradient_flow(self):
        """Ensure gradients flow correctly for the whole layer."""
        layer = WindowAttention(dim=32, window_size=4, num_heads=2)
        input_tensor = tf.Variable(keras.random.normal((2, 10, 32)))
        with tf.GradientTape() as tape:
            output = layer(input_tensor)
            loss = keras.ops.sum(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == len(layer.trainable_variables)
        for grad in grads:
            assert grad is not None
            assert keras.ops.any(keras.ops.not_equal(grad, 0))

    @pytest.mark.parametrize(
        "config",
        [
            {"dim": 128, "window_size": 7, "num_heads": 4, "qkv_bias": True, "qk_scale": 0.1,
             "dropout_rate": 0.1, "kernel_initializer": "he_normal"},
            {"dim": 64, "window_size": 8, "num_heads": 8, "qkv_bias": False},
            {"dim": 32, "window_size": 4, "num_heads": 2, "proj_bias": False},
        ]
    )
    def test_serialization_comprehensive(self, config):
        """Test serialization and deserialization with various configurations."""
        original_layer = WindowAttention(**config)
        input_shape = (None, 50, config["dim"])
        original_layer.build(input_shape)

        layer_config = original_layer.get_config()
        recreated_layer = WindowAttention.from_config(layer_config)
        recreated_layer.build(input_shape)

        assert recreated_layer.get_config() == layer_config
        assert len(recreated_layer.weights) == len(original_layer.weights)

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

    def test_transformer_block_integration(self):
        """Test the layer inside a full Transformer block."""
        dim, window_size, num_heads, seq_len = 64, 4, 4, 50
        inputs = keras.Input(shape=(seq_len, dim))
        outputs = build_transformer_block(inputs, dim, window_size, num_heads)
        model = keras.Model(inputs=inputs, outputs=outputs)
        test_data = keras.random.normal((2, seq_len, dim))
        output_data = model(test_data)
        assert output_data.shape == test_data.shape
        assert not np.any(np.isnan(output_data.numpy()))

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = WindowAttention(dim=32, window_size=4, num_heads=2)
        test_cases = [
            keras.ops.zeros((2, 15, 32)),
            keras.ops.ones((2, 15, 32)) * 1e-10,
            keras.ops.ones((2, 15, 32)) * 1e5,
            keras.random.normal((2, 15, 32)) * 100,
        ]
        for test_input in test_cases:
            output = layer(test_input)
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

    def test_regularization(self, input_tensor):
        """Test that regularization losses from the inner layer are collected."""
        layer = WindowAttention(
            dim=96, window_size=7, num_heads=3,
            kernel_regularizer=keras.regularizers.L2(0.1),
            bias_regularizer=keras.regularizers.L1(0.1)
        )
        _ = layer(input_tensor)
        assert len(layer.losses) > 0
        assert keras.ops.sum(layer.losses).numpy() > 0.0

    def test_relative_position_encoding_attributes(self):
        """Test that relative position encoding attributes exist on the inner layer."""
        layer = WindowAttention(dim=32, window_size=4, num_heads=2)
        layer.build((None, 10, 32))
        inner_attn = layer.attention
        window_area = 4 * 4
        assert inner_attn.relative_position_index.shape == (window_area, window_area)
        num_relative_distance = (2 * 4 - 1) ** 2
        assert inner_attn.relative_position_bias_table.shape == (num_relative_distance, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
