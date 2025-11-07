import pytest
import numpy as np
import keras
import os
import tempfile
import tensorflow as tf


from dl_techniques.layers.attention.window_attention import (
    WindowAttention,
    create_grid_window_attention,
    create_zigzag_window_attention,
    create_kan_key_window_attention,
    create_adaptive_softmax_window_attention,
)
from dl_techniques.layers.attention.single_window_attention import SingleWindowAttention


def build_transformer_block(inputs, dim, window_size, num_heads, **kwargs):
    """Helper function to build a Transformer block with the unified WindowAttention."""
    x1 = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    # Pass all additional kwargs to the unified layer
    attn_out = WindowAttention(
        dim=dim, window_size=window_size, num_heads=num_heads, **kwargs
    )(x1)
    x = keras.layers.Add()([inputs, attn_out])
    x2 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_out = keras.layers.Dense(int(dim * 4.0), activation="gelu")(x2)
    mlp_out = keras.layers.Dense(dim)(mlp_out)
    return keras.layers.Add()([x, mlp_out])


# Shared configurations for forward pass tests
common_configs = [
    # Grid mode variations
    {"partition_mode": "grid", "attention_mode": "linear", "normalization": "softmax",
     "use_relative_position_bias": True},
    {"partition_mode": "grid", "attention_mode": "kan_key", "normalization": "softmax",
     "use_relative_position_bias": True},
    {"partition_mode": "grid", "attention_mode": "linear", "normalization": "adaptive_softmax",
     "use_relative_position_bias": False},
    # Zigzag mode variations
    {"partition_mode": "zigzag", "attention_mode": "linear", "normalization": "softmax",
     "use_relative_position_bias": False},
    {"partition_mode": "zigzag", "attention_mode": "kan_key", "normalization": "hierarchical_routing",
     "use_relative_position_bias": False},
    {"partition_mode": "zigzag", "attention_mode": "linear", "normalization": "softmax",
     "use_relative_position_bias": True},
]


class TestWindowAttention:
    """Comprehensive test suite for the unified WindowAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a standard test input tensor."""
        return keras.random.normal([4, 50, 96])

    # 1. Initialization and Configuration Tests
    # =================================================================

    def test_initialization_defaults(self):
        """Test layer initializes with grid mode and standard linear attention."""
        layer = WindowAttention(dim=96, window_size=7, num_heads=4)
        assert layer.partition_mode == "grid"

        inner_attn = layer.attention
        assert isinstance(inner_attn, SingleWindowAttention)
        assert inner_attn.attention_mode == "linear"
        assert inner_attn.normalization == "softmax"
        assert inner_attn.use_relative_position_bias is True
        assert hasattr(inner_attn, "qkv")
        assert not hasattr(inner_attn, "query")

    def test_initialization_zigzag_mode(self):
        """Test initialization with zigzag partition mode."""
        layer = WindowAttention(dim=96, window_size=7, num_heads=4, partition_mode="zigzag")
        assert layer.partition_mode == "zigzag"
        # Check that it prepares attributes for zigzag indices
        assert hasattr(layer, "zigzag_indices")

    def test_initialization_kan_mode(self):
        """Test initialization with KAN-based attention."""
        layer = WindowAttention(dim=96, window_size=7, num_heads=4, attention_mode="kan_key")
        inner_attn = layer.attention
        assert inner_attn.attention_mode == "kan_key"
        assert hasattr(inner_attn, "query")
        assert hasattr(inner_attn, "key")  # KANLinear
        assert hasattr(inner_attn, "value")
        assert not hasattr(inner_attn, "qkv")
        assert inner_attn.kan_grid_size == 5  # Default value

    @pytest.mark.parametrize("norm_mode", ["adaptive_softmax", "hierarchical_routing"])
    def test_initialization_advanced_normalization(self, norm_mode):
        """Test initialization with advanced normalization schemes."""
        layer = WindowAttention(dim=96, window_size=7, num_heads=4, normalization=norm_mode)
        inner_attn = layer.attention
        assert inner_attn.normalization == norm_mode
        if norm_mode == "adaptive_softmax":
            assert hasattr(inner_attn, "adaptive_softmax")
        elif norm_mode == "hierarchical_routing":
            assert hasattr(inner_attn, "hierarchical_routing")

    def test_initialization_no_relative_bias(self):
        """Test explicit disabling of relative position bias."""
        layer = WindowAttention(dim=96, window_size=7, num_heads=4, use_relative_position_bias=False)
        assert layer.use_relative_position_bias is False
        assert layer.attention.use_relative_position_bias is False

    # 2. Build and Weight Creation Tests
    # =================================================================

    @pytest.mark.parametrize(
        "config, expected_attrs, forbidden_attrs",
        [
            # Standard grid mode: expects relative bias table, no zigzag indices
            (
                    {"partition_mode": "grid", "use_relative_position_bias": True, "window_size": 7},
                    ["relative_position_bias_table"],
                    ["zigzag_indices"],
            ),
            # Grid mode without bias, different window size
            (
                    {"partition_mode": "grid", "use_relative_position_bias": False, "window_size": 4},
                    [],
                    ["relative_position_bias_table", "zigzag_indices"],
            ),
            # Zigzag mode with bias
            (
                    {"partition_mode": "zigzag", "use_relative_position_bias": True, "window_size": 7,
                     "input_shape": (4, 10, 96)},
                    ["zigzag_indices", "inverse_zigzag_indices", "relative_position_bias_table"],
                    [],
            ),
            # Zigzag mode without bias, different window size
            (
                    {"partition_mode": "zigzag", "use_relative_position_bias": False, "window_size": 6,
                     "input_shape": (4, 10, 96)},
                    ["zigzag_indices", "inverse_zigzag_indices"],
                    ["relative_position_bias_table"],
            ),
        ],
    )
    def test_build_process_combinations(self, config, expected_attrs, forbidden_attrs):
        """Test that build creates the correct weights/attributes for each mode and window size."""
        input_shape = config.pop("input_shape", (4, 50, 96))
        window_size = config.pop("window_size")
        layer = WindowAttention(dim=96, window_size=window_size, num_heads=4, **config)
        layer.build(input_shape)

        assert layer.built

        inner_attn = layer.attention
        for attr in expected_attrs:
            if attr.endswith("_table"):  # It's a weight on the inner layer
                assert hasattr(inner_attn, attr)
            else:  # It's an attribute on the outer layer
                assert hasattr(layer, attr)
                assert getattr(layer, attr) is not None

        for attr in forbidden_attrs:
            if attr.endswith("_table"):
                assert not hasattr(inner_attn, attr)
            else:
                assert not hasattr(layer, attr) or getattr(layer, attr) is None

    # 3. Forward Pass and Functional Correctness Tests
    # =================================================================

    @pytest.mark.parametrize("config", common_configs)
    @pytest.mark.parametrize("window_size", [4, 7])
    @pytest.mark.parametrize("seq_len", [49, 50, 64])
    def test_forward_pass_combinations(self, config, window_size, seq_len):
        """Test forward pass for a matrix of configurations, window sizes, and sequence lengths."""
        dim, num_heads = 96, 4
        input_tensor = keras.random.normal([4, seq_len, dim])
        layer = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads, **config)

        # Test training and inference
        output_train = layer(input_tensor, training=True)
        output_infer = layer(input_tensor, training=False)

        assert output_train.shape == input_tensor.shape
        assert output_infer.shape == input_tensor.shape
        assert not np.any(np.isnan(output_train.numpy()))
        assert not np.any(np.isnan(output_infer.numpy()))

    @pytest.mark.parametrize("partition_mode", ["grid", "zigzag"])
    @pytest.mark.parametrize("window_size", [4, 5, 8])
    @pytest.mark.parametrize("seq_len", [55, 60])
    def test_attention_mask_integration(self, partition_mode, window_size, seq_len):
        """Test that the attention mask is correctly applied across modes, window sizes, and lengths."""
        dim, num_heads = 32, 4
        layer = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads, partition_mode=partition_mode)
        input_data = keras.random.normal((2, seq_len, dim))

        mask = np.ones((2, seq_len), dtype="int32")
        mask[:, -10:] = 0  # Mask out the last 10 tokens

        output = layer(input_data, attention_mask=keras.ops.convert_to_tensor(mask))
        assert output.shape == input_data.shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize("partition_mode", ["grid", "zigzag"])
    @pytest.mark.parametrize("seq_len", [1, 15, 16, 63, 64, 100])
    @pytest.mark.parametrize("window_size", [3, 4, 8])
    def test_arbitrary_shapes_and_windows(self, partition_mode, seq_len, window_size):
        """Test robustness to various sequence lengths and window sizes."""
        dim, num_heads = 32, 2
        layer = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads, partition_mode=partition_mode)
        input_data = keras.random.normal((4, seq_len, dim))
        output = layer(input_data)
        assert output.shape == input_data.shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize(
        "config",
        [
            {"attention_mode": "linear", "use_relative_position_bias": True},
            {"attention_mode": "kan_key", "use_relative_position_bias": False},
            {"normalization": "adaptive_softmax"},
            {"normalization": "hierarchical_routing"},
        ]
    )
    @pytest.mark.parametrize("window_size", [3, 4])
    def test_gradient_flow(self, config, window_size):
        """Ensure gradients flow correctly for all trainable modes and window sizes."""
        layer = WindowAttention(dim=32, window_size=window_size, num_heads=2, **config)
        input_data = tf.Variable(keras.random.normal((2, 10, 32)))

        with tf.GradientTape() as tape:
            output = layer(input_data)
            loss = keras.ops.sum(output)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == len(layer.trainable_variables)
        for grad in grads:
            assert grad is not None
            assert keras.ops.any(keras.ops.not_equal(grad, 0))

    # 4. Keras Ecosystem Integration Tests
    # =================================================================

    @pytest.mark.parametrize(
        "config",
        [
            # Grid mode with KAN and regularization, different window sizes
            {
                "partition_mode": "grid", "attention_mode": "kan_key", "window_size": 8,
                "kan_grid_size": 8, "kernel_regularizer": "l2"
            },
            # Zigzag mode with adaptive softmax and different window size
            {
                "partition_mode": "zigzag", "normalization": "adaptive_softmax", "window_size": 5,
                "use_relative_position_bias": False,
                "adaptive_softmax_config": {"min_temp": 0.1}
            },
            # Standard default config with a non-default window size
            {"window_size": 6},
        ]
    )
    def test_serialization_comprehensive(self, config):
        """Test get_config and from_config for complex configurations and window sizes."""
        base_config = {"dim": 64, "num_heads": 4}
        full_config = {**base_config, **config}

        layer = WindowAttention(**full_config)
        input_shape = (None, 20, 64)
        layer.build(input_shape)

        config_dict = layer.get_config()
        recreated_layer = WindowAttention.from_config(config_dict)
        recreated_layer.build(input_shape)

        assert recreated_layer.get_config() == config_dict
        assert len(recreated_layer.weights) == len(layer.weights)

    @pytest.mark.parametrize(
        "config",
        [
            {"partition_mode": "grid", "name": "grid_attn", "window_size": 7, "num_heads": 3},
            {"partition_mode": "zigzag", "attention_mode": "kan_key", "name": "zigzag_kan_attn", "window_size": 5,
             "num_heads": 4},
            {"partition_mode": "grid", "normalization": "adaptive_softmax", "name": "grid_adaptive",
             "window_size": 4, "num_heads": 2}
        ]
    )
    def test_model_save_load(self, config, input_tensor):
        """Test saving and loading a model containing the layer with various configs."""
        dim = 96
        inputs = keras.Input(shape=input_tensor.shape[1:])
        outputs = WindowAttention(dim=dim, **config)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.keras")
            model.save(path)
            loaded_model = keras.models.load_model(path)
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            assert np.allclose(original_prediction, loaded_prediction, atol=1e-6)
            assert isinstance(loaded_model.get_layer(config["name"]), WindowAttention)

    # 5. Factory Function Tests
    # =================================================================

    def test_factory_grid_window_attention(self):
        """Test the grid attention factory function and kwarg passthrough."""
        layer = create_grid_window_attention(
            dim=96, window_size=7, num_heads=4, dropout_rate=0.1, qkv_bias=False
        )
        assert isinstance(layer, WindowAttention)
        assert layer.partition_mode == "grid"
        assert layer.use_relative_position_bias is True  # Default for grid
        assert layer.dropout_rate == 0.1
        assert layer.qkv_bias is False

    def test_factory_zigzag_window_attention(self):
        """Test the zigzag attention factory function and kwarg passthrough."""
        layer = create_zigzag_window_attention(
            dim=96, window_size=7, num_heads=4, proj_bias=False
        )
        assert isinstance(layer, WindowAttention)
        assert layer.partition_mode == "zigzag"
        assert layer.use_relative_position_bias is False  # Default for zigzag
        assert layer.proj_bias is False

    def test_factory_kan_key_window_attention(self):
        """Test the KAN key attention factory function and kwarg passthrough."""
        # Test with default grid mode and KAN params
        layer_grid = create_kan_key_window_attention(
            dim=96, window_size=7, num_heads=4, kan_grid_size=10
        )
        assert layer_grid.attention.attention_mode == "kan_key"
        assert layer_grid.partition_mode == "grid"
        assert layer_grid.kan_grid_size == 10

        # Test with zigzag mode and KAN params
        layer_zigzag = create_kan_key_window_attention(
            dim=96, window_size=7, num_heads=4, partition_mode="zigzag", kan_spline_order=2
        )
        assert layer_zigzag.attention.attention_mode == "kan_key"
        assert layer_zigzag.partition_mode == "zigzag"
        assert layer_zigzag.kan_spline_order == 2

    def test_factory_adaptive_softmax_attention(self):
        """Test the adaptive softmax attention factory function and kwarg passthrough."""
        config = {"min_temp": 0.5}
        layer = create_adaptive_softmax_window_attention(
            dim=96, window_size=7, num_heads=4, adaptive_softmax_config=config
        )
        assert layer.attention.normalization == "adaptive_softmax"
        assert layer.partition_mode == "grid"  # Default partition mode
        assert layer.adaptive_softmax_config == config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])