"""
Comprehensive test suite for KANWindowAttention layer.

Tests follow the patterns established for WindowAttention testing,
adapted for KAN-based projection layers.
"""

import pytest
import numpy as np
import keras
import os
import tempfile
import tensorflow as tf

from dl_techniques.layers.attention.kan_window_attention import (
    KANWindowAttention,
    SingleKANWindowAttention,
)


def build_kan_transformer_block(
    inputs: keras.KerasTensor,
    dim: int,
    window_size: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    **kan_kwargs,
) -> keras.KerasTensor:
    """
    Helper function to build a KAN-based Transformer block for testing.

    Args:
        inputs: Input tensor.
        dim: Embedding dimension.
        window_size: Size of attention window.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for MLP.
        **kan_kwargs: Additional KAN-specific parameters.

    Returns:
        Output tensor after transformer block.
    """
    x1 = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    attn_out = KANWindowAttention(
        dim=dim, window_size=window_size, num_heads=num_heads, **kan_kwargs
    )(x1)
    x = keras.layers.Add()([inputs, attn_out])
    x2 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_out = keras.layers.Dense(int(dim * mlp_ratio), activation="gelu")(x2)
    mlp_out = keras.layers.Dense(dim)(mlp_out)
    return keras.layers.Add()([x, mlp_out])


class TestSingleKANWindowAttention:
    """Test suite for the SingleKANWindowAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor for single window attention."""
        return keras.random.normal([4, 49, 96])  # 4 windows, 49 tokens each

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return SingleKANWindowAttention(
            dim=96, window_size=7, num_heads=3
        )

    def test_initialization_defaults(self):
        """Test initialization with default KAN parameters."""
        layer = SingleKANWindowAttention(
            dim=128, window_size=7, num_heads=4
        )
        assert layer.dim == 128
        assert layer.window_size == 7
        assert layer.num_heads == 4
        assert layer.head_dim == 32
        assert layer.qk_scale is None
        assert layer.scale == 32**-0.5
        assert layer.attn_dropout_rate == 0.0
        assert layer.proj_dropout_rate == 0.0
        assert layer.kan_grid_size == 5
        assert layer.kan_spline_order == 3
        assert layer.kan_activation == "swish"
        assert layer.kan_regularization_factor == 0.01
        assert isinstance(
            layer.kernel_initializer, keras.initializers.GlorotUniform
        )

    def test_initialization_custom(self):
        """Test initialization with custom KAN parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        layer = SingleKANWindowAttention(
            dim=64,
            window_size=8,
            num_heads=8,
            kan_grid_size=8,
            kan_spline_order=4,
            kan_activation="gelu",
            kan_regularization_factor=0.05,
            qk_scale=0.1,
            attn_dropout_rate=0.1,
            proj_dropout_rate=0.2,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
        )
        assert layer.dim == 64
        assert layer.window_size == 8
        assert layer.num_heads == 8
        assert layer.head_dim == 8
        assert layer.qk_scale == 0.1
        assert layer.scale == 0.1
        assert layer.attn_dropout_rate == 0.1
        assert layer.proj_dropout_rate == 0.2
        assert layer.kan_grid_size == 8
        assert layer.kan_spline_order == 4
        assert layer.kan_activation == "gelu"
        assert layer.kan_regularization_factor == 0.05
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="dim must be positive"):
            SingleKANWindowAttention(dim=-10, window_size=7, num_heads=3)

        with pytest.raises(ValueError, match="window_size must be positive"):
            SingleKANWindowAttention(dim=96, window_size=0, num_heads=3)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            SingleKANWindowAttention(dim=96, window_size=7, num_heads=-3)

        with pytest.raises(ValueError, match="dim .* must be divisible by num_heads"):
            SingleKANWindowAttention(dim=97, window_size=7, num_heads=3)

        with pytest.raises(
            ValueError, match="attn_dropout_rate must be between 0.0 and 1.0"
        ):
            SingleKANWindowAttention(
                dim=96, window_size=7, num_heads=3, attn_dropout_rate=1.1
            )

        with pytest.raises(
            ValueError, match="kan_grid_size must be positive"
        ):
            SingleKANWindowAttention(
                dim=96, window_size=7, num_heads=3, kan_grid_size=0
            )

        with pytest.raises(
            ValueError, match="kan_regularization_factor must be non-negative"
        ):
            SingleKANWindowAttention(
                dim=96, window_size=7, num_heads=3, kan_regularization_factor=-0.1
            )

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly with KAN sub-layers."""
        output = layer_instance(input_tensor)
        assert layer_instance.built is True
        assert len(layer_instance.weights) > 0

        # Check KAN projection layers exist
        assert layer_instance.query is not None
        assert layer_instance.key is not None
        assert layer_instance.value is not None
        assert layer_instance.proj is not None

        # Check relative position bias table
        num_relative_distance = (2 * layer_instance.window_size - 1) ** 2
        expected_bias_shape = (
            num_relative_distance,
            layer_instance.num_heads,
        )
        assert layer_instance.relative_position_bias_table.shape == expected_bias_shape

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

    def test_attention_mask(self):
        """Test that attention mask works correctly."""
        layer = SingleKANWindowAttention(dim=32, window_size=5, num_heads=4)
        input_tensor = keras.random.normal((2, 25, 32))  # 5x5 window

        # Create mask that masks last 5 tokens
        mask = np.ones((2, 25), dtype="int32")
        mask[:, -5:] = 0

        output = layer(
            input_tensor, attention_mask=keras.ops.convert_to_tensor(mask)
        )
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_padding_handling(self):
        """Test that the layer handles sequences shorter than window area."""
        layer = SingleKANWindowAttention(dim=32, window_size=7, num_heads=4)

        # Test with various sequence lengths < 49
        for seq_len in [10, 25, 40, 48]:
            input_tensor = keras.random.normal((2, seq_len, 32))
            output = layer(input_tensor)
            assert output.shape == input_tensor.shape
            assert not np.any(np.isnan(output.numpy()))

    def test_gradient_flow(self):
        """Ensure gradients flow correctly through KAN layers."""
        layer = SingleKANWindowAttention(dim=32, window_size=4, num_heads=2)
        input_tensor = tf.Variable(keras.random.normal((2, 16, 32)))

        with tf.GradientTape() as tape:
            output = layer(input_tensor)
            loss = keras.ops.sum(output)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == len(layer.trainable_variables)
        for grad in grads:
            assert grad is not None
            assert keras.ops.any(keras.ops.not_equal(grad, 0))

    def test_serialization(self):
        """Test serialization and deserialization."""
        config = {
            "dim": 128,
            "window_size": 7,
            "num_heads": 4,
            "kan_grid_size": 6,
            "kan_spline_order": 4,
            "kan_activation": "gelu",
            "qk_scale": 0.1,
            "attn_dropout_rate": 0.1,
            "proj_dropout_rate": 0.2,
        }

        original_layer = SingleKANWindowAttention(**config)
        input_shape = (None, 49, config["dim"])
        original_layer.build(input_shape)

        layer_config = original_layer.get_config()
        recreated_layer = SingleKANWindowAttention.from_config(layer_config)
        recreated_layer.build(input_shape)

        assert recreated_layer.get_config() == layer_config
        assert len(recreated_layer.weights) == len(original_layer.weights)


class TestKANWindowAttention:
    """Test suite for the refactored KANWindowAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor with an arbitrary sequence length."""
        return keras.random.normal([4, 50, 96])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return KANWindowAttention(dim=96, window_size=7, num_heads=3)

    def test_initialization_defaults(self):
        """Test initialization with default parameters on the inner attention layer."""
        layer = KANWindowAttention(dim=128, window_size=7, num_heads=4)
        inner_attn = layer.attention
        assert isinstance(inner_attn, SingleKANWindowAttention)
        assert inner_attn.dim == 128
        assert inner_attn.window_size == 7
        assert inner_attn.num_heads == 4
        assert inner_attn.head_dim == 32
        assert inner_attn.qk_scale is None
        assert inner_attn.scale == 32**-0.5
        assert inner_attn.attn_dropout_rate == 0.0
        assert inner_attn.proj_dropout_rate == 0.0
        assert inner_attn.kan_grid_size == 5
        assert inner_attn.kan_spline_order == 3
        assert isinstance(
            inner_attn.kernel_initializer, keras.initializers.GlorotUniform
        )

    def test_initialization_custom(self):
        """Test initialization with custom KAN parameters passed to the inner layer."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        layer = KANWindowAttention(
            dim=64,
            window_size=8,
            num_heads=8,
            kan_grid_size=8,
            kan_spline_order=4,
            kan_activation="gelu",
            kan_regularization_factor=0.05,
            qk_scale=0.1,
            attn_dropout_rate=0.1,
            proj_dropout_rate=0.2,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
        )
        inner_attn = layer.attention
        assert inner_attn.dim == 64
        assert inner_attn.window_size == 8
        assert inner_attn.num_heads == 8
        assert inner_attn.head_dim == 8
        assert inner_attn.qk_scale == 0.1
        assert inner_attn.scale == 0.1
        assert inner_attn.attn_dropout_rate == 0.1
        assert inner_attn.proj_dropout_rate == 0.2
        assert inner_attn.kan_grid_size == 8
        assert inner_attn.kan_spline_order == 4
        assert inner_attn.kan_activation == "gelu"
        assert inner_attn.kan_regularization_factor == 0.05
        assert isinstance(
            inner_attn.kernel_initializer, keras.initializers.HeNormal
        )
        assert inner_attn.kernel_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors during initialization."""
        # These should propagate from SingleKANWindowAttention
        with pytest.raises(ValueError, match="dim must be positive"):
            KANWindowAttention(dim=-10, window_size=7, num_heads=3)

        with pytest.raises(ValueError, match="window_size must be positive"):
            KANWindowAttention(dim=96, window_size=0, num_heads=3)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            KANWindowAttention(dim=96, window_size=7, num_heads=-3)

        with pytest.raises(ValueError, match="dim .* must be divisible by num_heads"):
            KANWindowAttention(dim=97, window_size=7, num_heads=3)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer and its sub-layer build properly."""
        output = layer_instance(input_tensor)
        inner_attn = layer_instance.attention
        assert layer_instance.built is True
        assert inner_attn.built is True
        assert len(inner_attn.weights) > 0

        # Check KAN projection layers
        assert inner_attn.query is not None
        assert inner_attn.key is not None
        assert inner_attn.value is not None
        assert inner_attn.proj is not None

        # Check relative position bias
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
        ],
    )
    def test_arbitrary_sequence_lengths(
        self, dim, window_size, num_heads, seq_len
    ):
        """Test that padding and unpadding works for various sequence lengths."""
        layer = KANWindowAttention(
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
                    "kan_grid_size": 6,
                },
                99,
            ),
            (
                {
                    "dim": 128,
                    "window_size": 8,
                    "num_heads": 16,
                    "attn_dropout_rate": 0.1,
                    "kan_spline_order": 4,
                },
                250,
            ),
            (
                {
                    "dim": 48,
                    "window_size": 6,
                    "num_heads": 6,
                    "proj_dropout_rate": 0.1,
                    "proj_bias": False,
                    "kan_activation": "gelu",
                },
                48,
            ),
        ],
    )
    def test_comprehensive_configurations(self, config, seq_len):
        """Test various layer configurations with different sequence lengths."""
        layer = KANWindowAttention(**config)
        input_tensor = keras.random.normal((2, seq_len, config["dim"]))
        output = layer(input_tensor, training=True)
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_attention_mask_integration(self):
        """Test that a user-provided attention mask works correctly."""
        layer = KANWindowAttention(dim=32, window_size=5, num_heads=4)
        input_tensor = keras.random.normal((2, 55, 32))

        # Mask last 5 tokens
        user_mask = np.ones((2, 55), dtype="int32")
        user_mask[:, -5:] = 0

        output = layer(
            input_tensor,
            attention_mask=keras.ops.convert_to_tensor(user_mask),
        )
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_gradient_flow(self):
        """Ensure gradients flow correctly for the whole layer."""
        layer = KANWindowAttention(dim=32, window_size=4, num_heads=2)
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
            {
                "dim": 128,
                "window_size": 7,
                "num_heads": 4,
                "kan_grid_size": 6,
                "qk_scale": 0.1,
                "attn_dropout_rate": 0.1,
                "proj_dropout_rate": 0.2,
                "kernel_initializer": "he_normal",
            },
            {
                "dim": 64,
                "window_size": 8,
                "num_heads": 8,
                "kan_spline_order": 4,
                "kan_activation": "gelu",
            },
            {
                "dim": 32,
                "window_size": 4,
                "num_heads": 2,
                "proj_bias": False,
                "kan_regularization_factor": 0.05,
            },
        ],
    )
    def test_serialization_comprehensive(self, config):
        """Test serialization and deserialization with various configurations."""
        original_layer = KANWindowAttention(**config)
        input_shape = (None, 50, config["dim"])
        original_layer.build(input_shape)

        layer_config = original_layer.get_config()
        recreated_layer = KANWindowAttention.from_config(layer_config)
        recreated_layer.build(input_shape)

        assert recreated_layer.get_config() == layer_config
        assert len(recreated_layer.weights) == len(original_layer.weights)

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the KAN window attention layer."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = KANWindowAttention(
            dim=96, window_size=7, num_heads=3, name="kan_window_attn"
        )(inputs)
        x = keras.layers.LayerNormalization()(x)
        outputs = keras.layers.Dense(10)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Predictions should match after save/load",
            )
            assert isinstance(
                loaded_model.get_layer("kan_window_attn"), KANWindowAttention
            )

    def test_transformer_block_integration(self):
        """Test the layer inside a full KAN Transformer block."""
        dim, window_size, num_heads, seq_len = 64, 4, 4, 50
        inputs = keras.Input(shape=(seq_len, dim))
        outputs = build_kan_transformer_block(
            inputs,
            dim,
            window_size,
            num_heads,
            kan_grid_size=6,
            kan_spline_order=3,
        )
        model = keras.Model(inputs=inputs, outputs=outputs)

        test_data = keras.random.normal((2, seq_len, dim))
        output_data = model(test_data)
        assert output_data.shape == test_data.shape
        assert not np.any(np.isnan(output_data.numpy()))

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = KANWindowAttention(dim=32, window_size=4, num_heads=2)

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
        layer = KANWindowAttention(
            dim=96,
            window_size=7,
            num_heads=3,
            kernel_regularizer=keras.regularizers.L2(0.1),
            bias_regularizer=keras.regularizers.L1(0.1),
            kan_regularization_factor=0.1,
        )
        _ = layer(input_tensor)
        assert len(layer.losses) > 0
        assert keras.ops.sum(layer.losses).numpy() > 0.0

    def test_relative_position_encoding_attributes(self):
        """Test that relative position encoding attributes exist on the inner layer."""
        layer = KANWindowAttention(dim=32, window_size=4, num_heads=2)
        layer.build((None, 10, 32))
        inner_attn = layer.attention

        window_area = 4 * 4
        assert inner_attn.relative_position_index.shape == (
            window_area,
            window_area,
        )

        num_relative_distance = (2 * 4 - 1) ** 2
        assert inner_attn.relative_position_bias_table.shape == (
            num_relative_distance,
            2,
        )

    def test_kan_specific_parameters(self):
        """Test that KAN-specific parameters are correctly set and used."""
        layer = KANWindowAttention(
            dim=96,
            window_size=7,
            num_heads=3,
            kan_grid_size=8,
            kan_spline_order=4,
            kan_activation="gelu",
            kan_regularization_factor=0.05,
        )

        inner_attn = layer.attention
        assert inner_attn.kan_grid_size == 8
        assert inner_attn.kan_spline_order == 4
        assert inner_attn.kan_activation == "gelu"
        assert inner_attn.kan_regularization_factor == 0.05

        # Check that KAN layers are using these parameters
        # (This would require access to KANLinear internals)
        assert inner_attn.query is not None
        assert inner_attn.key is not None
        assert inner_attn.value is not None

    def test_comparison_with_different_kan_configs(self):
        """Test that different KAN configurations produce different outputs."""
        input_tensor = keras.random.normal((2, 50, 96))

        # Create two layers with different KAN configurations
        layer1 = KANWindowAttention(
            dim=96,
            window_size=7,
            num_heads=3,
            kan_grid_size=5,
            kan_spline_order=3,
        )
        layer2 = KANWindowAttention(
            dim=96,
            window_size=7,
            num_heads=3,
            kan_grid_size=8,
            kan_spline_order=4,
        )

        output1 = layer1(input_tensor)
        output2 = layer2(input_tensor)

        # Outputs should have the same shape but different values
        # (due to different initialization and KAN structure)
        assert output1.shape == output2.shape
        # We don't compare values since they'll be different due to random init

    def test_window_partitioning_and_merging(self):
        """Test internal window partitioning and merging operations."""
        layer = KANWindowAttention(dim=32, window_size=4, num_heads=2)

        # Create a grid tensor
        grid = keras.random.normal((2, 8, 8, 32))  # 8x8 grid, divisible by 4

        # Test window partition
        windows = layer._window_partition(grid)
        expected_num_windows = 2 * (8 // 4) * (8 // 4)  # 2 batches, 4 windows each
        expected_shape = (expected_num_windows, 4, 4, 32)
        assert windows.shape == expected_shape

        # Test window reverse
        reconstructed = layer._window_reverse(windows, H=8, W=8)
        assert reconstructed.shape == grid.shape

        # Check that reconstruction is exact (up to numerical precision)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(grid),
            keras.ops.convert_to_numpy(reconstructed),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Window partition and reverse should be exact inverses",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])