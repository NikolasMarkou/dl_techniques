import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

# Import the new layers
from dl_techniques.layers.attention.window_attention_zigzag import (
    SingleWindowZigZagAttention,
    WindowZigZagAttention,
)

# Import the standard WindowAttention for comparison
from dl_techniques.layers.attention.window_attention import WindowAttention


def build_transformer_block(
    inputs, dim, window_size, num_heads, mlp_ratio=4.0
):
    """Helper function to build a Transformer block with Zigzag Attention."""
    x1 = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    attn_out = WindowZigZagAttention(
        dim=dim, window_size=window_size, num_heads=num_heads
    )(x1)
    x = keras.layers.Add()([inputs, attn_out])
    x2 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_out = keras.layers.Dense(int(dim * mlp_ratio), activation="gelu")(x2)
    mlp_out = keras.layers.Dense(dim)(mlp_out)
    return keras.layers.Add()([x, mlp_out])


class TestWindowZigZagAttention:
    """Test suite for the refactored WindowZigZagAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor with an arbitrary sequence length."""
        return keras.random.normal([4, 50, 96])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return WindowZigZagAttention(dim=96, window_size=7, num_heads=3)

    def test_initialization_defaults(self):
        """Test initialization with default parameters on the inner attention layer."""
        layer = WindowZigZagAttention(dim=128, window_size=7, num_heads=4)
        inner_attn = layer.attention
        assert isinstance(inner_attn, SingleWindowZigZagAttention)
        assert inner_attn.dim == 128
        assert inner_attn.window_size == 7
        assert inner_attn.num_heads == 4
        assert inner_attn.head_dim == 32
        assert inner_attn.qkv_bias is True
        assert inner_attn.scale == 32**-0.5
        assert inner_attn.use_hierarchical_routing is False
        assert inner_attn.use_adaptive_softmax is False
        assert isinstance(
            inner_attn.kernel_initializer, keras.initializers.GlorotUniform
        )

    def test_initialization_custom_and_advanced(self):
        """Test initialization with custom and advanced normalization parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        adaptive_config = {"min_temp": 0.2, "max_temp": 2.0}
        layer = WindowZigZagAttention(
            dim=64,
            window_size=8,
            num_heads=8,
            qkv_bias=False,
            qk_scale=0.1,
            attn_dropout_rate=0.1,
            proj_dropout_rate=0.2,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
            use_adaptive_softmax=True,
            adaptive_softmax_config=adaptive_config,
        )
        inner_attn = layer.attention
        assert inner_attn.dim == 64
        assert inner_attn.num_heads == 8
        assert inner_attn.qkv_bias is False
        assert inner_attn.scale == 0.1
        assert inner_attn.use_adaptive_softmax is True
        assert inner_attn.use_hierarchical_routing is False
        assert inner_attn.adaptive_softmax_config["min_temp"] == 0.2
        assert isinstance(
            inner_attn.kernel_initializer, keras.initializers.HeNormal
        )
        assert inner_attn.kernel_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="dim must be positive"):
            WindowZigZagAttention(dim=-10, window_size=7, num_heads=3)
        with pytest.raises(ValueError, match="window_size must be positive"):
            WindowZigZagAttention(dim=96, window_size=0, num_heads=3)
        with pytest.raises(ValueError, match="num_heads must be positive"):
            WindowZigZagAttention(dim=96, window_size=7, num_heads=-3)
        with pytest.raises(
            ValueError, match="dim .* must be divisible by num_heads"
        ):
            WindowZigZagAttention(dim=97, window_size=7, num_heads=3)
        with pytest.raises(
            ValueError,
            match="Only one of `use_adaptive_softmax` or `use_hierarchical_routing`",
        ):
            WindowZigZagAttention(
                dim=96,
                window_size=7,
                num_heads=3,
                use_adaptive_softmax=True,
                use_hierarchical_routing=True,
            )
        with pytest.raises(ValueError, match="max_temp .* must be > min_temp"):
            WindowZigZagAttention(
                dim=96,
                window_size=7,
                num_heads=3,
                use_adaptive_softmax=True,
                adaptive_softmax_config={"min_temp": 2.0, "max_temp": 1.0},
            )

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer and its sub-layer build properly."""
        _ = layer_instance(input_tensor)
        inner_attn = layer_instance.attention
        assert layer_instance.built is True
        assert inner_attn.built is True
        assert len(inner_attn.weights) > 0
        expected_qkv_shape = (input_tensor.shape[-1], inner_attn.dim * 3)
        assert inner_attn.qkv.kernel.shape == expected_qkv_shape
        num_relative_distance = (2 * inner_attn.window_size - 1) ** 2
        expected_bias_shape = (num_relative_distance, inner_attn.num_heads)
        assert inner_attn.relative_position_bias_table.shape == expected_bias_shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test forward pass for training and inference modes."""
        output = layer_instance(input_tensor)
        assert not np.any(np.isnan(output.numpy()))
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
            (32, 4, 2, 2),  # Very small sequence
        ],
    )
    def test_arbitrary_sequence_lengths(
        self, dim, window_size, num_heads, seq_len
    ):
        """Test that padding and unpadding works for various sequence lengths."""
        layer = WindowZigZagAttention(
            dim=dim, window_size=window_size, num_heads=num_heads
        )
        input_tensor = keras.random.normal((4, seq_len, dim))
        output = layer(input_tensor)
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize(
        "config, seq_len",
        [
            ({"dim": 32, "window_size": 4, "num_heads": 2}, 17),
            (
                {
                    "dim": 60,
                    "window_size": 5,
                    "num_heads": 5,
                    "use_hierarchical_routing": True,
                },
                99,
            ),
            (
                {
                    "dim": 128,
                    "window_size": 8,
                    "num_heads": 16,
                    "use_adaptive_softmax": True,
                },
                250,
            ),
        ],
    )
    def test_comprehensive_configurations(self, config, seq_len):
        """Test various layer configurations with different sequence lengths."""
        layer = WindowZigZagAttention(**config)
        input_tensor = keras.random.normal((2, seq_len, config["dim"]))
        output = layer(input_tensor, training=True)
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_attention_mask_integration(self):
        """Test that a user-provided attention mask works correctly."""
        layer = WindowZigZagAttention(dim=32, window_size=5, num_heads=4)
        input_tensor = keras.random.normal((2, 55, 32))
        user_mask = np.ones((2, 55), dtype="int32")
        user_mask[:, -5:] = 0
        output = layer(
            input_tensor, attention_mask=keras.ops.convert_to_tensor(user_mask)
        )
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_gradient_flow(self):
        """Ensure gradients flow correctly for the whole layer."""
        layer = WindowZigZagAttention(dim=32, window_size=4, num_heads=2)
        input_tensor = tf.Variable(keras.random.normal((2, 10, 32)))
        with tf.GradientTape() as tape:
            output = layer(input_tensor)
            loss = keras.ops.sum(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == len(layer.trainable_variables)
        for grad in grads:
            assert grad is not None
            assert keras.ops.any(keras.ops.not_equal(grad, 0))

    def test_serialization_comprehensive(self):
        """Test serialization and deserialization with advanced configurations."""
        config = {
            "dim": 128,
            "window_size": 7,
            "num_heads": 4,
            "qkv_bias": True,
            "qk_scale": 0.1,
            "attn_dropout_rate": 0.1,
            "proj_dropout_rate": 0.2,
            "use_adaptive_softmax": True,
            "adaptive_softmax_config": {"min_temp": 0.5},
            "kernel_initializer": "he_normal",
        }
        original_layer = WindowZigZagAttention(**config)
        input_shape = (None, 50, config["dim"])
        original_layer.build(input_shape)

        layer_config = original_layer.get_config()
        recreated_layer = WindowZigZagAttention.from_config(layer_config)
        recreated_layer.build(input_shape)

        assert recreated_layer.get_config() == layer_config
        assert len(recreated_layer.weights) == len(original_layer.weights)

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the zigzag attention layer."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = WindowZigZagAttention(
            dim=96,
            window_size=7,
            num_heads=3,
            name="zigzag_attn",
            use_hierarchical_routing=True,
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=x)
        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)
            assert np.allclose(
                original_prediction, loaded_prediction, rtol=1e-5
            )
            assert isinstance(
                loaded_model.get_layer("zigzag_attn"), WindowZigZagAttention
            )

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

    # --- ZIGZAG SPECIFIC TESTS ---

    @staticmethod
    def test_generate_zigzag_coords():
        """Test the static zigzag coordinate generation method."""
        coords_3x3 = SingleWindowZigZagAttention._generate_zigzag_coords(3)
        expected_3x3 = [
            (0, 0),
            (0, 1),
            (1, 0),
            (2, 0),
            (1, 1),
            (0, 2),
            (1, 2),
            (2, 1),
            (2, 2),
        ]
        assert coords_3x3 == expected_3x3
        assert len(coords_3x3) == 9

    def test_relative_position_index_is_different_from_standard(self):
        """Crucially, test that the zigzag RPB index differs from raster-scan."""
        dim, window_size, num_heads = 32, 4, 2
        input_shape = (None, 10, dim)

        zigzag_layer = WindowZigZagAttention(
            dim=dim, window_size=window_size, num_heads=num_heads
        )
        standard_layer = WindowAttention(
            dim=dim, window_size=window_size, num_heads=num_heads
        )

        zigzag_layer.build(input_shape)
        standard_layer.build(input_shape)

        zigzag_index = zigzag_layer.attention.relative_position_index
        standard_index = standard_layer.attention.relative_position_index

        assert zigzag_index.shape == standard_index.shape
        are_equal = np.array_equal(zigzag_index.numpy(), standard_index.numpy())
        assert not are_equal, (
            "Zigzag relative position index should not match the standard one."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])