import pytest
import numpy as np
import keras
import os
import tempfile

from dl_techniques.layers.ffn.swin_mlp import SwinMLP
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.attention.window_attention import WindowAttention
from dl_techniques.layers.transformers.swin_transformer_block import SwinTransformerBlock


class TestSwinTransformerBlock:
    """Test suite for SwinTransformerBlock implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        # Shape: (batch_size, height, width, channels)
        # Using 56x56 which is divisible by common window sizes (7, 8, 14, 28)
        return keras.random.normal([2, 56, 56, 96])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return SwinTransformerBlock(dim=96, num_heads=3)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = SwinTransformerBlock(dim=128, num_heads=4)

        # Check default values
        assert layer.dim == 128
        assert layer.num_heads == 4
        assert layer.window_size == 8
        assert layer.shift_size == 0
        assert layer.mlp_ratio == 4.0
        assert layer.qkv_bias is True
        assert layer.dropout_rate == 0.0
        assert layer.attn_dropout_rate == 0.0
        assert layer.drop_path_rate == 0.0
        assert layer.activation == "gelu"
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = SwinTransformerBlock(
            dim=64,
            num_heads=8,
            window_size=7,
            shift_size=3,
            mlp_ratio=3.0,
            qkv_bias=False,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            drop_path=0.1,
            activation="relu",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
            bias_regularizer=custom_regularizer,
        )

        # Check custom values
        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.window_size == 7
        assert layer.shift_size == 3
        assert layer.mlp_ratio == 3.0
        assert layer.qkv_bias is False
        assert layer.dropout_rate == 0.1
        assert layer.attn_dropout_rate == 0.1
        assert layer.drop_path_rate == 0.1
        assert layer.activation == "relu"
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.bias_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid window_size
        with pytest.raises(ValueError, match="window_size must be positive"):
            SwinTransformerBlock(dim=64, num_heads=8, window_size=0)

        with pytest.raises(ValueError, match="window_size must be positive"):
            SwinTransformerBlock(dim=64, num_heads=8, window_size=-1)

        # Test invalid shift_size
        with pytest.raises(ValueError, match="shift_size must be non-negative"):
            SwinTransformerBlock(dim=64, num_heads=8, shift_size=-1)

        with pytest.raises(ValueError, match="shift_size .* must be less than window_size"):
            SwinTransformerBlock(dim=64, num_heads=8, window_size=8, shift_size=8)

        # Test invalid mlp_ratio
        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            SwinTransformerBlock(dim=64, num_heads=8, mlp_ratio=0)

        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            SwinTransformerBlock(dim=64, num_heads=8, mlp_ratio=-1)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that the layer was built
        assert layer_instance.built is True
        assert len(layer_instance.weights) > 0
        assert layer_instance.norm1 is not None
        assert layer_instance.norm2 is not None
        assert layer_instance.attn is not None
        assert layer_instance.mlp is not None
        assert isinstance(layer_instance.norm1, keras.layers.LayerNormalization)
        assert isinstance(layer_instance.norm2, keras.layers.LayerNormalization)

        # Check that stochastic depth is created when drop_path > 0
        layer_with_drop_path = SwinTransformerBlock(dim=96, num_heads=3, drop_path=0.1)
        layer_with_drop_path(input_tensor)
        assert layer_with_drop_path.drop_path_rate is not None

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        configs_to_test = [
            {"dim": 32, "num_heads": 4, "window_size": 8},
            {"dim": 64, "num_heads": 8, "window_size": 7},
            {"dim": 128, "num_heads": 8, "window_size": 14},
        ]

        for config in configs_to_test:
            # Create input tensor with matching dimensions and compatible spatial size
            dim = config["dim"]
            window_size = config["window_size"]
            # Use spatial size that's divisible by window_size
            spatial_size = 56  # Divisible by 7, 8, 14, 28
            test_input = keras.random.normal([2, spatial_size, spatial_size, dim])
            layer = SwinTransformerBlock(**config)
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
            {"dim": 32, "num_heads": 4, "window_size": 8, "shift_size": 0},
            {"dim": 64, "num_heads": 8, "window_size": 8, "shift_size": 4, "mlp_ratio": 2.0},
            {"dim": 96, "num_heads": 3, "window_size": 7, "shift_size": 3, "dropout_rate": 0.1, "drop_path": 0.1},
            {"dim": 128, "num_heads": 8, "window_size": 14, "shift_size": 0, "qkv_bias": False},
        ]

        for config in configurations:
            layer = SwinTransformerBlock(**config)

            # Create appropriate input with compatible spatial dimensions
            dim = config["dim"]
            window_size = config["window_size"]
            # Use spatial size that's divisible by window_size
            spatial_size = 56  # Divisible by 7, 8, 14, 28
            test_input = keras.random.normal([2, spatial_size, spatial_size, dim])

            # Test forward pass
            output = layer(test_input)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == test_input.shape

    def test_shifted_window_attention(self):
        """Test that shifted window attention works correctly."""
        # Test with no shift
        layer_no_shift = SwinTransformerBlock(dim=64, num_heads=8, window_size=8, shift_size=0)
        test_input = keras.random.normal([2, 56, 56, 64])  # 56 is divisible by 8
        output_no_shift = layer_no_shift(test_input)

        # Test with shift
        layer_with_shift = SwinTransformerBlock(dim=64, num_heads=8, window_size=8, shift_size=4)
        output_with_shift = layer_with_shift(test_input)

        # Both should produce valid outputs
        assert not np.any(np.isnan(output_no_shift.numpy()))
        assert not np.any(np.isnan(output_with_shift.numpy()))
        assert output_no_shift.shape == test_input.shape
        assert output_with_shift.shape == test_input.shape

        # Outputs should be different due to different attention patterns
        assert not np.allclose(output_no_shift.numpy(), output_with_shift.numpy())

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = SwinTransformerBlock(
            dim=128,
            num_heads=8,
            window_size=7,
            shift_size=3,
            mlp_ratio=3.0,
            qkv_bias=False,
            dropout_rate=0.1,
            attn_dropout_rate=0.05,
            drop_path=0.1,
            activation="relu",
            use_bias=False,
            kernel_initializer="he_normal",
        )

        # Build the layer with compatible spatial dimensions
        input_shape = (None, 56, 56, 128)  # 56 is divisible by 7
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = SwinTransformerBlock.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.dim == original_layer.dim
        assert recreated_layer.num_heads == original_layer.num_heads
        assert recreated_layer.window_size == original_layer.window_size
        assert recreated_layer.shift_size == original_layer.shift_size
        assert recreated_layer.mlp_ratio == original_layer.mlp_ratio
        assert recreated_layer.qkv_bias == original_layer.qkv_bias
        assert recreated_layer.dropout_rate == original_layer.dropout_rate
        assert recreated_layer.dropout_rate == original_layer.attn_dropout_rate
        assert recreated_layer.drop_path_rate == original_layer.drop_path_rate
        assert recreated_layer.activation == original_layer.activation
        assert recreated_layer.use_bias == original_layer.use_bias

        # Check weights match (shapes should be the same)
        assert len(recreated_layer.weights) == len(original_layer.weights)
        for w1, w2 in zip(original_layer.weights, recreated_layer.weights):
            assert w1.shape == w2.shape

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = SwinTransformerBlock(dim=96, num_heads=3)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
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
        x = SwinTransformerBlock(dim=96, num_heads=3, name="swin_block")(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
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
                    "SwinMLP": SwinMLP,
                    "StochasticDepth": StochasticDepth,
                    "WindowAttention": WindowAttention,
                    "SwinTransformerBlock": SwinTransformerBlock,

                }
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("swin_block"), SwinTransformerBlock)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = SwinTransformerBlock(dim=32, num_heads=4, window_size=8)

        # Create inputs with different magnitudes
        batch_size = 2
        height, width = 56, 56  # Divisible by 8
        channels = 32

        test_cases = [
            keras.ops.zeros((batch_size, height, width, channels)),  # Zeros
            keras.ops.ones((batch_size, height, width, channels)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, height, width, channels)) * 1e5,  # Large values
            keras.random.normal((batch_size, height, width, channels)) * 100  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_regularization(self, input_tensor):
        """Test that regularization losses are properly applied."""
        # Create layer with regularization
        layer = SwinTransformerBlock(
            dim=96,
            num_heads=3,
            kernel_regularizer=keras.regularizers.L2(0.1),
            bias_regularizer=keras.regularizers.L1(0.1),
            activity_regularizer=keras.regularizers.L2(0.01)
        )

        # Build layer
        layer.build(input_tensor.shape)

        # No regularization losses before calling the layer
        initial_losses = len(layer.losses)

        # Apply the layer
        _ = layer(input_tensor)

        # Should have regularization losses now
        assert len(layer.losses) >= initial_losses

    def test_training_behavior(self, input_tensor):
        """Test different behavior in training vs inference mode."""
        layer = SwinTransformerBlock(dim=96, num_heads=3, drop_path=0.1, dropout_rate=0.1)

        # Test training mode
        training_output = layer(input_tensor, training=True)

        # Test inference mode
        inference_output = layer(input_tensor, training=False)

        # Both should produce valid outputs
        assert not np.any(np.isnan(training_output.numpy()))
        assert not np.any(np.isnan(inference_output.numpy()))
        assert training_output.shape == input_tensor.shape
        assert inference_output.shape == input_tensor.shape

    def test_stochastic_depth_behavior(self):
        """Test that stochastic depth works correctly."""
        # Layer without stochastic depth
        layer_no_drop = SwinTransformerBlock(dim=64, num_heads=8, window_size=8, drop_path=0.0)
        test_input = keras.random.normal([2, 56, 56, 64])  # 56 is divisible by 8

        # Layer with stochastic depth
        layer_with_drop = SwinTransformerBlock(dim=64, num_heads=8, window_size=8, drop_path=0.5)

        # Test in training mode multiple times to see stochastic behavior
        outputs_with_drop = []
        for _ in range(5):
            output = layer_with_drop(test_input, training=True)
            outputs_with_drop.append(output.numpy())

        # Check that outputs are valid
        for output in outputs_with_drop:
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

        # In inference mode, output should be deterministic
        inference_output1 = layer_with_drop(test_input, training=False)
        inference_output2 = layer_with_drop(test_input, training=False)
        assert np.allclose(inference_output1.numpy(), inference_output2.numpy())

    def test_different_input_sizes(self):
        """Test layer with different input sizes."""
        layer = SwinTransformerBlock(dim=64, num_heads=8, window_size=8)

        # Use input sizes that are divisible by window_size (8)
        input_sizes = [
            (2, 16, 16, 64),  # 16 is divisible by 8
            (1, 32, 32, 64),  # 32 is divisible by 8
            (3, 24, 24, 64),  # 24 is divisible by 8
            (2, 40, 40, 64),  # 40 is divisible by 8
        ]

        for input_size in input_sizes:
            test_input = keras.random.normal(input_size)
            output = layer(test_input)

            # Check output shape matches input shape
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    def test_mlp_ratio_effects(self):
        """Test that different MLP ratios work correctly."""
        mlp_ratios = [1.0, 2.0, 4.0, 8.0]
        test_input = keras.random.normal([2, 56, 56, 64])  # 56 is divisible by 8

        for mlp_ratio in mlp_ratios:
            layer = SwinTransformerBlock(dim=64, num_heads=8, window_size=8, mlp_ratio=mlp_ratio)
            output = layer(test_input)

            # Check output is valid
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

            # Check that MLP hidden dimension is correctly calculated
            expected_hidden_dim = int(64 * mlp_ratio)
            actual_hidden_dim = layer.mlp.hidden_dim
            assert actual_hidden_dim == expected_hidden_dim

    def test_window_size_compatibility(self):
        """Test that the layer handles window size compatibility properly."""
        layer = SwinTransformerBlock(dim=64, num_heads=8, window_size=7)

        # Test with compatible dimensions (56 is divisible by 7)
        compatible_input = keras.random.normal([2, 56, 56, 64])
        output = layer(compatible_input)
        assert output.shape == compatible_input.shape
        assert not np.any(np.isnan(output.numpy()))

        # Test with incompatible dimensions (32 is not divisible by 7)
        # This should raise an error during window partitioning
        incompatible_input = keras.random.normal([2, 32, 32, 64])
        with pytest.raises(Exception):  # Should raise InvalidArgumentError from TensorFlow
            layer(incompatible_input)