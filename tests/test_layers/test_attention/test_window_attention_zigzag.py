import pytest
import numpy as np
import keras
import os
import tempfile

# Import the new WindowZigZagAttention layer
from dl_techniques.layers.attention.window_attention_zigzag import WindowZigZagAttention
# Import the original for comparison
from dl_techniques.layers.attention.window_attention import WindowAttention


class TestWindowZigZagAttention:
    """Test suite for WindowZigZagAttention layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor for a full window."""
        return keras.random.normal([4, 49, 96])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return WindowZigZagAttention(dim=96, window_size=7, num_heads=3)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = WindowZigZagAttention(dim=128, window_size=7, num_heads=4)
        assert layer.dim == 128
        assert layer.window_size == 7
        assert layer.num_heads == 4
        assert layer.head_dim == 32
        assert layer.qkv_bias is True
        assert layer.scale == 32 ** -0.5

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="dim .* must be divisible by num_heads"):
            WindowZigZagAttention(dim=97, window_size=7, num_heads=3)
        with pytest.raises(ValueError, match="dim must be positive"):
            WindowZigZagAttention(dim=0, window_size=7, num_heads=3)
        with pytest.raises(ValueError, match="window_size must be positive"):
            WindowZigZagAttention(dim=96, window_size=0, num_heads=3)
        with pytest.raises(ValueError, match="num_heads must be positive"):
            WindowZigZagAttention(dim=96, window_size=7, num_heads=0)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        output = layer_instance(input_tensor)
        assert layer_instance.built is True
        assert len(layer_instance.weights) > 0
        assert hasattr(layer_instance, "relative_position_bias_table")
        assert hasattr(layer_instance, "relative_position_index")

        expected_qkv_shape = (input_tensor.shape[-1], layer_instance.dim * 3)
        assert layer_instance.qkv.kernel.shape == expected_qkv_shape
        num_relative_distance = (2 * layer_instance.window_size - 1) ** 2
        expected_bias_shape = (num_relative_distance, layer_instance.num_heads)
        assert layer_instance.relative_position_bias_table.shape == expected_bias_shape

    def test_output_shapes(self):
        """Test that output shapes are computed correctly."""
        layer = WindowZigZagAttention(dim=64, window_size=7, num_heads=4)
        test_input = keras.random.normal([2, 49, 64])
        output = layer(test_input)
        assert output.shape == test_input.shape
        computed_shape = layer.compute_output_shape(test_input.shape)
        assert computed_shape == test_input.shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)
        assert not np.any(np.isnan(output.numpy()))
        assert output.shape == input_tensor.shape

    # --- ZIGZAG SPECIFIC TESTS ---

    def test_generate_zigzag_coords(self):
        """Test the static zigzag coordinate generation method."""
        coords_3x3 = WindowZigZagAttention._generate_zigzag_coords(3)
        expected_3x3 = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (1, 2), (2, 1), (2, 2)]
        assert coords_3x3 == expected_3x3
        assert len(coords_3x3) == 9

    def test_relative_position_index_is_different_from_standard(self):
        """Crucially, test that the zigzag relative position index is different from raster-scan."""
        dim, window_size, num_heads = 32, 4, 2
        input_shape = (None, window_size**2, dim)

        zigzag_layer = WindowZigZagAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        standard_layer = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)

        zigzag_layer.build(input_shape)
        standard_layer.build(input_shape)

        zigzag_index = zigzag_layer.relative_position_index
        standard_index = standard_layer.relative_position_index

        assert zigzag_index.shape == standard_index.shape

        # FIX: Use keras.ops.all and keras.ops.equal for backend-agnostic tensor comparison
        are_equal = keras.ops.all(keras.ops.equal(zigzag_index, standard_index))
        assert not are_equal, "Zigzag relative position index should not match the standard one."

    # --- PADDING AND MASKING TESTS ---

    @pytest.mark.parametrize(
        "dim, window_size, num_heads, seq_len",
        [
            (32, 4, 2, 16),
            (32, 4, 2, 10),
        ]
    )
    def test_padding_and_unpadding(self, dim, window_size, num_heads, seq_len):
        layer = WindowZigZagAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        input_tensor = keras.random.normal((4, seq_len, dim))
        output = layer(input_tensor)
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_forward_pass_with_too_long_sequence(self):
        layer = WindowZigZagAttention(dim=32, window_size=4, num_heads=2)
        invalid_input = keras.random.normal((2, 17, 32))
        with pytest.raises(ValueError, match="Input sequence length .* cannot be greater than"):
            layer(invalid_input)

    def test_attention_mask_integration(self):
        layer = WindowZigZagAttention(dim=32, window_size=5, num_heads=4)
        input_tensor = keras.random.normal((2, 20, 32))
        user_mask = np.ones((2, 20), dtype="int32")
        user_mask[:, -5:] = 0
        output = layer(input_tensor, attention_mask=keras.ops.convert_to_tensor(user_mask))
        assert output.shape == input_tensor.shape

    def test_gradient_flow_with_padding(self):
        import tensorflow as tf
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

    # --- MODEL AND SERIALIZATION TESTS ---

    def test_serialization(self):
        original_layer = WindowZigZagAttention(dim=128, window_size=7, num_heads=4, qk_scale=0.1)
        input_shape = (None, 49, 128)
        original_layer.build(input_shape)
        config = original_layer.get_config()
        recreated_layer = WindowZigZagAttention.from_config(config)
        recreated_layer.build(input_shape)
        assert recreated_layer.dim == original_layer.dim
        assert len(recreated_layer.weights) == len(original_layer.weights)

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = WindowZigZagAttention(dim=96, window_size=7, num_heads=3)(inputs)
        x = keras.layers.LayerNormalization()(x)
        # FIX: Add pooling layer to reduce sequence dimension before final Dense layer
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        y_pred = model(input_tensor)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the zigzag attention layer."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = WindowZigZagAttention(dim=96, window_size=7, num_heads=3, name="zigzag_attn")(inputs)
        model = keras.Model(inputs=inputs, outputs=x)
        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)
            assert isinstance(loaded_model.get_layer("zigzag_attn"), WindowZigZagAttention)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])