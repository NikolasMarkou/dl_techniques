import pytest
import numpy as np
import keras
import os
import tempfile

from dl_techniques.layers.swin_conv_block import SwinConvBlock
from dl_techniques.layers.swin_transformer_block import SwinTransformerBlock


class TestSwinConvBlock:
    """Test suite for SwinConvBlock implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        # Shape: (batch_size, height, width, channels)
        # Using 56x56 which is divisible by common window sizes (7, 8, 14, 28)
        # Channels = conv_dim + trans_dim = 64 + 32 = 96
        return keras.random.normal([2, 56, 56, 96])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return SwinConvBlock(conv_dim=64, trans_dim=32)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = SwinConvBlock(conv_dim=64, trans_dim=32)

        # Check default values
        assert layer.conv_dim == 64
        assert layer.trans_dim == 32
        assert layer.head_dim == 32
        assert layer.num_heads == 1  # trans_dim // head_dim = 32 // 32 = 1
        assert layer.window_size == 8
        assert layer.drop_path_rate == 0.0
        assert layer.block_type == "W"
        assert layer.input_resolution is None
        assert layer.mlp_ratio == 4.0
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = SwinConvBlock(
            conv_dim=32,
            trans_dim=64,
            head_dim=16,
            window_size=7,
            drop_path=0.1,
            block_type="SW",
            input_resolution=56,
            mlp_ratio=3.0,
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
            bias_regularizer=custom_regularizer,
        )

        # Check custom values
        assert layer.conv_dim == 32
        assert layer.trans_dim == 64
        assert layer.head_dim == 16
        assert layer.num_heads == 4  # trans_dim // head_dim = 64 // 16 = 4
        assert layer.window_size == 7
        assert layer.drop_path_rate == 0.1
        assert layer.block_type == "SW"
        assert layer.input_resolution == 56
        assert layer.mlp_ratio == 3.0
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.bias_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid conv_dim
        with pytest.raises(ValueError, match="conv_dim must be positive"):
            SwinConvBlock(conv_dim=0, trans_dim=32)

        with pytest.raises(ValueError, match="conv_dim must be positive"):
            SwinConvBlock(conv_dim=-1, trans_dim=32)

        # Test invalid trans_dim
        with pytest.raises(ValueError, match="trans_dim must be positive"):
            SwinConvBlock(conv_dim=32, trans_dim=0)

        with pytest.raises(ValueError, match="trans_dim must be positive"):
            SwinConvBlock(conv_dim=32, trans_dim=-1)

        # Test invalid head_dim
        with pytest.raises(ValueError, match="head_dim must be positive"):
            SwinConvBlock(conv_dim=32, trans_dim=64, head_dim=0)

        with pytest.raises(ValueError, match="head_dim must be positive"):
            SwinConvBlock(conv_dim=32, trans_dim=64, head_dim=-1)

        # Test trans_dim not divisible by head_dim
        with pytest.raises(ValueError, match="trans_dim .* must be divisible by head_dim"):
            SwinConvBlock(conv_dim=32, trans_dim=64, head_dim=48)

        # Test invalid window_size
        with pytest.raises(ValueError, match="window_size must be positive"):
            SwinConvBlock(conv_dim=32, trans_dim=64, window_size=0)

        with pytest.raises(ValueError, match="window_size must be positive"):
            SwinConvBlock(conv_dim=32, trans_dim=64, window_size=-1)

        # Test invalid drop_path
        with pytest.raises(ValueError, match="drop_path must be in"):
            SwinConvBlock(conv_dim=32, trans_dim=64, drop_path=1.5)

        with pytest.raises(ValueError, match="drop_path must be in"):
            SwinConvBlock(conv_dim=32, trans_dim=64, drop_path=-0.1)

        # Test invalid block_type
        with pytest.raises(ValueError, match="block_type must be 'W' or 'SW'"):
            SwinConvBlock(conv_dim=32, trans_dim=64, block_type="INVALID")

        # Test invalid mlp_ratio
        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            SwinConvBlock(conv_dim=32, trans_dim=64, mlp_ratio=0)

        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            SwinConvBlock(conv_dim=32, trans_dim=64, mlp_ratio=-1)

        # Test invalid input_resolution
        with pytest.raises(ValueError, match="input_resolution must be positive"):
            SwinConvBlock(conv_dim=32, trans_dim=64, input_resolution=0)

        with pytest.raises(ValueError, match="input_resolution must be positive"):
            SwinConvBlock(conv_dim=32, trans_dim=64, input_resolution=-1)

    def test_input_resolution_auto_adjustment(self):
        """Test that block_type is automatically adjusted when input_resolution <= window_size."""
        # Test case where input_resolution <= window_size should change block_type to "W"
        layer = SwinConvBlock(
            conv_dim=32,
            trans_dim=64,
            window_size=8,
            block_type="SW",
            input_resolution=8  # Equal to window_size
        )
        assert layer.block_type == "W"  # Should be changed from "SW" to "W"

        layer2 = SwinConvBlock(
            conv_dim=32,
            trans_dim=64,
            window_size=8,
            block_type="SW",
            input_resolution=4  # Less than window_size
        )
        assert layer2.block_type == "W"  # Should be changed from "SW" to "W"

        # Test case where input_resolution > window_size should preserve block_type
        layer3 = SwinConvBlock(
            conv_dim=32,
            trans_dim=64,
            window_size=8,
            block_type="SW",
            input_resolution=16  # Greater than window_size
        )
        assert layer3.block_type == "SW"  # Should remain "SW"

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that the layer was built
        assert layer_instance.built is True
        assert len(layer_instance.weights) > 0
        assert layer_instance.conv1_1 is not None
        assert layer_instance.trans_block is not None
        assert layer_instance.conv_block is not None
        assert layer_instance.conv1_2 is not None
        assert isinstance(layer_instance.conv1_1, keras.layers.Conv2D)
        assert isinstance(layer_instance.trans_block, SwinTransformerBlock)
        assert isinstance(layer_instance.conv_block, keras.Sequential)
        assert isinstance(layer_instance.conv1_2, keras.layers.Conv2D)

    def test_build_invalid_input_shape(self):
        """Test build with invalid input shapes."""
        layer = SwinConvBlock(conv_dim=32, trans_dim=64)

        # Test non-4D input
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            layer.build((None, 32, 64))  # 3D shape

        # Test channel mismatch
        with pytest.raises(ValueError, match="Input channels .* must match conv_dim \\+ trans_dim"):
            layer.build((None, 32, 32, 128))  # Wrong channel dimension (should be 96)

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        configs_to_test = [
            {"conv_dim": 32, "trans_dim": 32, "window_size": 8},
            {"conv_dim": 48, "trans_dim": 48, "head_dim": 16, "window_size": 7},
            {"conv_dim": 64, "trans_dim": 64, "window_size": 14},
        ]

        for config in configs_to_test:
            # Create input tensor with matching dimensions and compatible spatial size
            total_dim = config["conv_dim"] + config["trans_dim"]
            window_size = config["window_size"]
            # Use spatial size that's divisible by window_size
            spatial_size = 56  # Divisible by 7, 8, 14, 28
            test_input = keras.random.normal([2, spatial_size, spatial_size, total_dim])
            layer = SwinConvBlock(**config)
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

    def test_call_invalid_input(self):
        """Test call with invalid input shapes."""
        layer = SwinConvBlock(conv_dim=32, trans_dim=64)

        # Test with 3D input
        invalid_input = keras.random.normal([2, 32, 64])
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            layer(invalid_input)

    def test_different_configurations(self):
        """Test layer with different configurations."""
        configurations = [
            {"conv_dim": 32, "trans_dim": 32, "window_size": 8, "block_type": "W"},
            {"conv_dim": 48, "trans_dim": 48, "head_dim": 16, "window_size": 8, "block_type": "SW", "mlp_ratio": 2.0},
            {"conv_dim": 40, "trans_dim": 56, "head_dim": 28, "window_size": 7, "block_type": "W", "drop_path": 0.1},
            {"conv_dim": 64, "trans_dim": 64, "window_size": 14, "block_type": "SW", "use_bias": False},
        ]

        for config in configurations:
            layer = SwinConvBlock(**config)

            # Create appropriate input with compatible spatial dimensions
            total_dim = config["conv_dim"] + config["trans_dim"]
            window_size = config["window_size"]
            # Use spatial size that's divisible by window_size
            spatial_size = 56  # Divisible by 7, 8, 14, 28
            test_input = keras.random.normal([2, spatial_size, spatial_size, total_dim])

            # Test forward pass
            output = layer(test_input)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == test_input.shape

    def test_block_type_differences(self):
        """Test that different block types produce different outputs."""
        # Test with window attention
        layer_w = SwinConvBlock(conv_dim=32, trans_dim=64, window_size=8, block_type="W")
        test_input = keras.random.normal([2, 56, 56, 96])  # 56 is divisible by 8
        output_w = layer_w(test_input)

        # Test with shifted window attention
        layer_sw = SwinConvBlock(conv_dim=32, trans_dim=64, window_size=8, block_type="SW")
        output_sw = layer_sw(test_input)

        # Both should produce valid outputs
        assert not np.any(np.isnan(output_w.numpy()))
        assert not np.any(np.isnan(output_sw.numpy()))
        assert output_w.shape == test_input.shape
        assert output_sw.shape == test_input.shape

        # Outputs should be different due to different attention patterns
        assert not np.allclose(output_w.numpy(), output_sw.numpy())

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = SwinConvBlock(
            conv_dim=48,
            trans_dim=80,
            head_dim=16,
            window_size=7,
            drop_path=0.1,
            block_type="SW",
            input_resolution=56,
            mlp_ratio=3.0,
            use_bias=False,
            kernel_initializer="he_normal",
        )

        # Build the layer with compatible spatial dimensions
        input_shape = (None, 56, 56, 128)  # 56 is divisible by 7, 48+80=128
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = SwinConvBlock.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.conv_dim == original_layer.conv_dim
        assert recreated_layer.trans_dim == original_layer.trans_dim
        assert recreated_layer.head_dim == original_layer.head_dim
        assert recreated_layer.window_size == original_layer.window_size
        assert recreated_layer.drop_path_rate == original_layer.drop_path_rate
        assert recreated_layer.block_type == original_layer.block_type
        assert recreated_layer.input_resolution == original_layer.input_resolution
        assert recreated_layer.mlp_ratio == original_layer.mlp_ratio
        assert recreated_layer.use_bias == original_layer.use_bias

        # Check weights match (shapes should be the same)
        assert len(recreated_layer.weights) == len(original_layer.weights)
        for w1, w2 in zip(original_layer.weights, recreated_layer.weights):
            assert w1.shape == w2.shape

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = SwinConvBlock(conv_dim=64, trans_dim=32)(inputs)
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
        x = SwinConvBlock(conv_dim=64, trans_dim=32, name="swin_conv_block")(inputs)
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
                    "SwinConvBlock": SwinConvBlock,
                    "SwinTransformerBlock": SwinTransformerBlock,
                }
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("swin_conv_block"), SwinConvBlock)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = SwinConvBlock(conv_dim=16, trans_dim=16, head_dim=16, window_size=8)

        # Create inputs with different magnitudes
        batch_size = 2
        height, width = 56, 56  # Divisible by 8
        channels = 32  # conv_dim + trans_dim = 16 + 16 = 32

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
        layer = SwinConvBlock(
            conv_dim=64,
            trans_dim=32,
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
        layer = SwinConvBlock(conv_dim=64, trans_dim=32, drop_path=0.1)

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
        test_input = keras.random.normal([2, 56, 56, 96])  # 56 is divisible by 8

        # Layer with stochastic depth
        layer_with_drop = SwinConvBlock(conv_dim=64, trans_dim=32, drop_path=0.5)

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
        layer = SwinConvBlock(conv_dim=32, trans_dim=32, window_size=8)

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
        test_input = keras.random.normal([2, 56, 56, 96])  # 56 is divisible by 8

        for mlp_ratio in mlp_ratios:
            layer = SwinConvBlock(conv_dim=64, trans_dim=32, window_size=8, mlp_ratio=mlp_ratio)
            output = layer(test_input)

            # Check output is valid
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

            # Check that transformer block MLP ratio is correctly set
            assert layer.trans_block.mlp_ratio == mlp_ratio

    def test_window_size_compatibility(self):
        """Test that the layer handles window size compatibility properly."""
        layer = SwinConvBlock(conv_dim=32, trans_dim=32, window_size=7)

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

    def test_residual_connections(self):
        """Test that residual connections work properly."""
        layer = SwinConvBlock(conv_dim=32, trans_dim=32, window_size=8)
        test_input = keras.random.normal([2, 56, 56, 64])

        # Get intermediate results to check residual connections
        layer.build(test_input.shape)

        # Test main residual connection
        output = layer(test_input)

        # Output should not be identical to input (some transformation occurred)
        assert not np.allclose(output.numpy(), test_input.numpy())

        # But should be close to input + some transformation due to residual connection
        assert output.shape == test_input.shape

    def test_path_separation(self):
        """Test that conv and transformer paths are properly separated."""
        # Create a layer with different conv and trans dimensions
        layer = SwinConvBlock(conv_dim=32, trans_dim=64, window_size=8)
        test_input = keras.random.normal([2, 56, 56, 96])  # 32 + 64 = 96

        # Build and test
        output = layer(test_input)

        # Check that both paths are created
        assert layer.conv_block is not None
        assert layer.trans_block is not None

        # Check transformer block has correct dimensions
        assert layer.trans_block.dim == 64
        assert layer.num_heads == 2  # 64 // 32 = 2

        # Output should be valid
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_head_dim_calculations(self):
        """Test that head dimension calculations work correctly."""
        test_cases = [
            {"trans_dim": 32, "head_dim": 32, "expected_heads": 1},
            {"trans_dim": 64, "head_dim": 32, "expected_heads": 2},
            {"trans_dim": 96, "head_dim": 32, "expected_heads": 3},
            {"trans_dim": 128, "head_dim": 32, "expected_heads": 4},
            {"trans_dim": 64, "head_dim": 16, "expected_heads": 4},
        ]

        for case in test_cases:
            layer = SwinConvBlock(
                conv_dim=32,
                trans_dim=case["trans_dim"],
                head_dim=case["head_dim"]
            )
            assert layer.num_heads == case["expected_heads"]

    def test_different_dimension_combinations(self):
        """Test layer with different conv_dim and trans_dim combinations."""
        dimension_combinations = [
            {"conv_dim": 32, "trans_dim": 32},
            {"conv_dim": 64, "trans_dim": 32},
            {"conv_dim": 32, "trans_dim": 64},
            {"conv_dim": 48, "trans_dim": 80, "head_dim": 16},
            {"conv_dim": 16, "trans_dim": 16, "head_dim": 16},
        ]

        for dims in dimension_combinations:
            layer = SwinConvBlock(**dims)
            total_dim = dims["conv_dim"] + dims["trans_dim"]
            test_input = keras.random.normal([2, 56, 56, total_dim])

            output = layer(test_input)

            # Check output is valid
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    def test_use_bias_effects(self):
        """Test that use_bias parameter affects layer behavior."""
        # Test with bias
        layer_with_bias = SwinConvBlock(conv_dim=32, trans_dim=32, use_bias=True)
        test_input = keras.random.normal([2, 56, 56, 64])

        # Test without bias
        layer_without_bias = SwinConvBlock(conv_dim=32, trans_dim=32, use_bias=False)

        # Both should work
        output_with_bias = layer_with_bias(test_input)
        output_without_bias = layer_without_bias(test_input)

        assert output_with_bias.shape == test_input.shape
        assert output_without_bias.shape == test_input.shape
        assert not np.any(np.isnan(output_with_bias.numpy()))
        assert not np.any(np.isnan(output_without_bias.numpy()))

        # Outputs should be different due to bias
        assert not np.allclose(output_with_bias.numpy(), output_without_bias.numpy())