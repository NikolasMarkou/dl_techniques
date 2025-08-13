import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any, Tuple
import tensorflow as tf

# Import the YOLOv12 building blocks
from dl_techniques.layers.yolo12 import (
    ConvBlock,
    AreaAttention,
    AttentionBlock,
    Bottleneck,
    C3k2Block,
    A2C2fBlock
)


class TestConvBlock:
    """Comprehensive test suite for ConvBlock layer."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing."""
        return keras.random.normal([4, 32, 32, 64])  # batch, height, width, channels

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'filters': 128,
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same'
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration with all parameters."""
        return {
            'filters': 256,
            'kernel_size': 5,
            'strides': 2,
            'padding': 'valid',
            'groups': 4,
            'activation': False,
            'use_bias': True,
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': keras.regularizers.L2(1e-4)
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = ConvBlock(filters=64)

        # Check stored configuration
        assert layer.filters == 64
        assert layer.kernel_size == 3
        assert layer.strides == 1
        assert layer.padding == "same"
        assert layer.groups == 1
        assert layer.activation is True
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that sub-layers are created but not built
        assert layer.conv is not None
        assert layer.bn is not None
        assert layer.act is not None
        assert not layer.conv.built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with custom parameters."""
        layer = ConvBlock(**custom_layer_config)

        assert layer.filters == 256
        assert layer.kernel_size == 5
        assert layer.strides == 2
        assert layer.padding == "valid"
        assert layer.groups == 4
        assert layer.activation is False
        assert layer.use_bias is True
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)

        # When activation=False, act should be None
        assert layer.act is None

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid filters values
        with pytest.raises(ValueError, match="filters must be positive"):
            ConvBlock(filters=0)

        with pytest.raises(ValueError, match="filters must be positive"):
            ConvBlock(filters=-10)

        # Test invalid kernel_size values
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            ConvBlock(filters=64, kernel_size=0)

        # Test invalid strides values
        with pytest.raises(ValueError, match="strides must be positive"):
            ConvBlock(filters=64, strides=-1)

        # Test invalid groups values
        with pytest.raises(ValueError, match="groups must be positive"):
            ConvBlock(filters=64, groups=0)

        # Test invalid padding values
        with pytest.raises(ValueError, match="padding must be 'same' or 'valid'"):
            ConvBlock(filters=64, padding="invalid")

    def test_build_process(self, sample_input, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = ConvBlock(**layer_config)

        # Layer should not be built initially
        assert not layer.built

        # Forward pass triggers build
        output = layer(sample_input)

        # Check that layer is now built
        assert layer.built

        # Check that all sub-layers are built
        assert layer.conv.built
        assert layer.bn.built

        # Verify output shape
        expected_shape = layer.conv.compute_output_shape(sample_input.shape)
        assert output.shape == expected_shape

    def test_forward_pass_with_activation(self, sample_input, layer_config):
        """Test forward pass with activation enabled."""
        layer = ConvBlock(**layer_config)
        output = layer(sample_input)

        # Basic sanity checks
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

        # With SiLU activation, output should have some positive values
        assert keras.ops.any(keras.ops.greater(output, 0))

    def test_forward_pass_without_activation(self, sample_input):
        """Test forward pass with activation disabled."""
        layer = ConvBlock(filters=128, activation=False)
        output = layer(sample_input)

        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_compute_output_shape(self):
        """Test output shape computation."""
        layer = ConvBlock(filters=64, kernel_size=3, strides=2, padding='same')

        input_shape = (None, 32, 32, 128)
        expected_shape = (None, 16, 16, 64)  # Height and width halved due to stride=2

        computed_shape = layer.compute_output_shape(input_shape)
        assert computed_shape == expected_shape

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = ConvBlock(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )

    def test_gradient_flow(self, sample_input, layer_config):
        """Test that gradients flow properly through the layer."""
        layer = ConvBlock(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) == len(layer.trainable_variables)
        assert all(g is not None for g in gradients)

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, sample_input, layer_config, training):
        """Test layer behavior in different training modes."""
        layer = ConvBlock(**layer_config)
        output = layer(sample_input, training=training)

        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))


class TestAreaAttention:
    """Comprehensive test suite for AreaAttention layer."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing."""
        return keras.random.normal([2, 16, 16, 256])  # batch, height, width, channels

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'dim': 256,
            'num_heads': 8,
            'area': 1
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration."""
        return {
            'dim': 512,
            'num_heads': 16,
            'area': 4,
            'kernel_initializer': 'glorot_uniform'
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = AreaAttention(dim=256)

        assert layer.dim == 256
        assert layer.num_heads == 8
        assert layer.area == 1
        assert layer.head_dim == 32  # 256 // 8
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)

        # Check sub-layers are created
        assert layer.qk_conv is not None
        assert layer.v_conv is not None
        assert layer.pe_conv is not None
        assert layer.proj_conv is not None

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid dim
        with pytest.raises(ValueError, match="dim must be positive"):
            AreaAttention(dim=0)

        # Test invalid num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            AreaAttention(dim=256, num_heads=0)

        # Test dim not divisible by num_heads
        with pytest.raises(ValueError, match="dim .* must be divisible by num_heads"):
            AreaAttention(dim=256, num_heads=7)

        # Test invalid area
        with pytest.raises(ValueError, match="area must be positive"):
            AreaAttention(dim=256, area=0)

    def test_forward_pass_global_attention(self, sample_input):
        """Test forward pass with global attention (area=1)."""
        layer = AreaAttention(dim=256, num_heads=8, area=1)
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_forward_pass_area_attention(self):
        """Test forward pass with area-based attention."""
        # Use input size that's divisible by area
        sample_input = keras.random.normal([2, 8, 8, 256])  # 64 total positions
        layer = AreaAttention(dim=256, num_heads=8, area=4)  # 16 positions per area

        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_compute_output_shape(self):
        """Test output shape computation."""
        layer = AreaAttention(dim=128, num_heads=4)

        input_shape = (None, 32, 32, 256)
        expected_shape = (None, 32, 32, 128)

        computed_shape = layer.compute_output_shape(input_shape)
        assert computed_shape == expected_shape

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = AreaAttention(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )


class TestAttentionBlock:
    """Comprehensive test suite for AttentionBlock layer."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing."""
        return keras.random.normal([2, 16, 16, 256])

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'dim': 256,
            'num_heads': 8,
            'mlp_ratio': 1.2,
            'area': 1
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = AttentionBlock(dim=256)

        assert layer.dim == 256
        assert layer.num_heads == 8
        assert layer.mlp_ratio == 1.2
        assert layer.area == 1
        assert layer.mlp_hidden_dim == int(256 * 1.2)

        # Check sub-layers are created
        assert layer.attn is not None
        assert layer.mlp1 is not None
        assert layer.mlp2 is not None

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="dim must be positive"):
            AttentionBlock(dim=0)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            AttentionBlock(dim=256, num_heads=0)

        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            AttentionBlock(dim=256, mlp_ratio=0)

        with pytest.raises(ValueError, match="area must be positive"):
            AttentionBlock(dim=256, area=0)

    def test_forward_pass_residual_connections(self, sample_input, layer_config):
        """Test that residual connections work properly."""
        layer = AttentionBlock(**layer_config)

        # Ensure input matches expected dimension
        if sample_input.shape[-1] != layer_config['dim']:
            sample_input = keras.random.normal([2, 16, 16, layer_config['dim']])

        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

        # Output should be different from input due to transformations
        diff = keras.ops.mean(keras.ops.abs(output - sample_input))
        assert keras.ops.convert_to_numpy(diff) > 1e-6

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Ensure input matches expected dimension
        if sample_input.shape[-1] != layer_config['dim']:
            sample_input = keras.random.normal([2, 16, 16, layer_config['dim']])

        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = AttentionBlock(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )


class TestBottleneck:
    """Comprehensive test suite for Bottleneck layer."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing."""
        return keras.random.normal([4, 32, 32, 64])

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'filters': 64,
            'shortcut': True
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = Bottleneck(filters=128)

        assert layer.filters == 128
        assert layer.shortcut is True
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)

        # Check sub-layers are created
        assert layer.cv1 is not None
        assert layer.cv2 is not None

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="filters must be positive"):
            Bottleneck(filters=0)

        with pytest.raises(ValueError, match="filters must be positive"):
            Bottleneck(filters=-10)

    def test_forward_pass_with_shortcut(self, sample_input, layer_config):
        """Test forward pass with shortcut connection."""
        layer = Bottleneck(**layer_config)
        output = layer(sample_input)

        # Should have same shape as input
        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_forward_pass_without_shortcut(self, sample_input):
        """Test forward pass without shortcut connection."""
        layer = Bottleneck(filters=64, shortcut=False)
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_shortcut_dimension_mismatch(self):
        """Test behavior when input channels don't match output filters."""
        sample_input = keras.random.normal([4, 32, 32, 32])  # 32 input channels
        layer = Bottleneck(filters=64, shortcut=True)  # 64 output channels

        output = layer(sample_input)

        # Should work but no shortcut applied due to dimension mismatch
        expected_shape = (4, 32, 32, 64)
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = Bottleneck(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )


class TestC3k2Block:
    """Comprehensive test suite for C3k2Block layer."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing."""
        return keras.random.normal([4, 32, 32, 128])

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'filters': 256,
            'n': 2,
            'shortcut': True
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = C3k2Block(filters=128)

        assert layer.filters == 128
        assert layer.n == 1
        assert layer.shortcut is True
        assert layer.hidden_filters == 64  # filters // 2

        # Check sub-layers are created
        assert layer.cv1 is not None
        assert layer.cv2 is not None
        assert layer.cv3 is not None
        assert len(layer.bottlenecks) == 1

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="filters must be positive"):
            C3k2Block(filters=0)

        with pytest.raises(ValueError, match="n must be non-negative"):
            C3k2Block(filters=128, n=-1)

    def test_forward_pass_multiple_bottlenecks(self, layer_config):
        """Test forward pass with multiple bottleneck layers."""
        sample_input = keras.random.normal([4, 32, 32, 128])
        layer = C3k2Block(**layer_config)
        output = layer(sample_input)

        expected_shape = (4, 32, 32, layer_config['filters'])
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_forward_pass_zero_bottlenecks(self):
        """Test forward pass with no bottleneck layers."""
        sample_input = keras.random.normal([4, 32, 32, 128])
        layer = C3k2Block(filters=256, n=0)
        output = layer(sample_input)

        expected_shape = (4, 32, 32, 256)
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_compute_output_shape(self):
        """Test output shape computation."""
        layer = C3k2Block(filters=256, n=3)

        input_shape = (None, 32, 32, 128)
        expected_shape = (None, 32, 32, 256)

        computed_shape = layer.compute_output_shape(input_shape)
        assert computed_shape == expected_shape

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = C3k2Block(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )


class TestA2C2fBlock:
    """Comprehensive test suite for A2C2fBlock layer."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing."""
        return keras.random.normal([2, 16, 16, 256])

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'filters': 512,
            'n': 2,
            'area': 1
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = A2C2fBlock(filters=256)

        assert layer.filters == 256
        assert layer.n == 1
        assert layer.area == 1
        assert layer.hidden_filters == 128  # filters // 2

        # Check sub-layers are created
        assert layer.cv1 is not None
        assert layer.cv2 is not None
        assert len(layer.attention_first_blocks) == 1
        assert len(layer.attention_second_blocks) == 1  # Two attention blocks per iteration

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="filters must be positive"):
            A2C2fBlock(filters=0)

        with pytest.raises(ValueError, match="n must be non-negative"):
            A2C2fBlock(filters=256, n=-1)

        with pytest.raises(ValueError, match="area must be positive"):
            A2C2fBlock(filters=256, area=0)

    def test_forward_pass_progressive_features(self, layer_config, sample_input):
        """Test forward pass with progressive feature extraction."""
        layer = A2C2fBlock(**layer_config)
        output = layer(sample_input)

        expected_shape = (sample_input.shape[0], sample_input.shape[1],
                          sample_input.shape[2], layer_config['filters'])
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_forward_pass_zero_attention_pairs(self):
        """Test forward pass with no attention block pairs."""
        sample_input = keras.random.normal([2, 16, 16, 256])
        layer = A2C2fBlock(filters=512, n=0)
        output = layer(sample_input)

        expected_shape = (2, 16, 16, 512)
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_attention_head_calculation(self):
        """Test that attention heads are calculated correctly."""
        # Test with different hidden dimensions
        test_cases = [
            (64, max(1, 32 // 32)),  # Should be 1
            (128, max(1, 64 // 32)),  # Should be 2
            (256, max(1, 128 // 32)),  # Should be 4
            (512, max(1, 256 // 32)),  # Should be 8
        ]

        for filters, expected_heads in test_cases:
            layer = A2C2fBlock(filters=filters)

            # Build layer to access attention blocks
            sample_input = keras.random.normal([1, 8, 8, 128])
            layer(sample_input)

            # Check first attention block's num_heads
            actual_heads = layer.attention_first_blocks[0].num_heads
            assert actual_heads == expected_heads

    def test_compute_output_shape(self):
        """Test output shape computation."""
        layer = A2C2fBlock(filters=512, n=3, area=4)

        input_shape = (None, 16, 16, 256)
        expected_shape = (None, 16, 16, 512)

        computed_shape = layer.compute_output_shape(input_shape)
        assert computed_shape == expected_shape

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = A2C2fBlock(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )


class TestYOLOv12Integration:
    """Test integration of multiple YOLOv12 components in complex models."""

    def test_complete_yolo_block_chain(self):
        """Test a complete chain of YOLOv12 blocks working together."""
        sample_input = keras.random.normal([2, 64, 64, 128])

        # Create a chain of different blocks
        inputs = keras.Input(shape=sample_input.shape[1:])

        # Start with ConvBlock
        x = ConvBlock(filters=256, kernel_size=3)(inputs)

        # Add C3k2Block for feature extraction
        x = C3k2Block(filters=512, n=3)(x)

        # Add A2C2fBlock for attention-based feature fusion
        x = A2C2fBlock(filters=512, n=2, area=4)(x)

        # Add Bottleneck for residual processing
        x = Bottleneck(filters=512, shortcut=True)(x)

        # Final ConvBlock
        outputs = ConvBlock(filters=256, kernel_size=1)(x)

        model = keras.Model(inputs, outputs)

        # Test forward pass
        prediction = model(sample_input)
        expected_shape = (2, 64, 64, 256)
        assert prediction.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(prediction))

        # Test gradient flow through entire chain
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = model(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in gradients)

    def test_multi_scale_attention_integration(self):
        """Test integration of attention blocks with different areas."""
        sample_input = keras.random.normal([2, 32, 32, 256])

        inputs = keras.Input(shape=sample_input.shape[1:])

        # Multi-scale attention processing
        x1 = AttentionBlock(dim=256, area=1)(inputs)  # Global attention
        x2 = AttentionBlock(dim=256, area=4)(inputs)  # Local attention

        # Combine features
        x = keras.layers.Add()([x1, x2])

        # Process with A2C2fBlock
        outputs = A2C2fBlock(filters=512, n=2, area=2)(x)

        model = keras.Model(inputs, outputs)
        prediction = model(sample_input)

        expected_shape = (2, 32, 32, 512)
        assert prediction.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(prediction))

    def test_model_serialization_integration(self):
        """Test serialization of a complex model with all YOLOv12 components."""
        sample_input = keras.random.normal([2, 32, 32, 64])

        inputs = keras.Input(shape=sample_input.shape[1:])

        # Build complex architecture
        x = ConvBlock(filters=128)(inputs)
        x = C3k2Block(filters=256, n=2)(x)
        x = A2C2fBlock(filters=512, n=1, area=2)(x)
        x = Bottleneck(filters=512)(x)
        outputs = ConvBlock(filters=256, activation=False)(x)

        model = keras.Model(inputs, outputs)
        original_prediction = model(sample_input)

        # Test serialization cycle
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'complex_yolo_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Complex model predictions differ after serialization cycle"
            )


class TestYOLOv12EdgeCases:
    """Test edge cases and boundary conditions for YOLOv12 blocks."""

    def test_minimal_dimensions(self):
        """Test with minimal viable dimensions."""
        # Test ConvBlock with minimal filters
        conv_layer = ConvBlock(filters=1)
        test_input = keras.random.normal([1, 8, 8, 3])
        output = conv_layer(test_input)
        assert output.shape == (1, 8, 8, 1)
        assert not keras.ops.any(keras.ops.isnan(output))

        # Test AreaAttention with minimal dimensions
        attn_layer = AreaAttention(dim=8, num_heads=1)
        test_input = keras.random.normal([1, 4, 4, 8])
        output = attn_layer(test_input)
        assert output.shape == (1, 4, 4, 8)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_large_dimensions(self):
        """Test with large dimensions."""
        # Test with large feature dimensions
        layer = C3k2Block(filters=1024, n=5)
        test_input = keras.random.normal([1, 16, 16, 512])
        output = layer(test_input)
        assert output.shape == (1, 16, 16, 1024)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_single_pixel_input(self):
        """Test with 1x1 spatial dimensions."""
        test_input = keras.random.normal([4, 1, 1, 256])

        # Test different blocks with 1x1 input
        conv_output = ConvBlock(filters=128)(test_input)
        assert conv_output.shape == (4, 1, 1, 128)

        bottleneck_output = Bottleneck(filters=256)(test_input)
        assert bottleneck_output.shape == (4, 1, 1, 256)

    def test_extreme_aspect_ratios(self):
        """Test with extreme aspect ratios."""
        # Very wide input
        wide_input = keras.random.normal([2, 4, 64, 128])
        conv_output = ConvBlock(filters=256)(wide_input)
        assert conv_output.shape == (2, 4, 64, 256)

        # Very tall input
        tall_input = keras.random.normal([2, 64, 4, 128])
        c3k2_output = C3k2Block(filters=256)(tall_input)
        assert c3k2_output.shape == (2, 64, 4, 256)

    def test_different_batch_sizes(self):
        """Test with different batch sizes including batch_size=1."""
        layer = AttentionBlock(dim=128)

        batch_sizes = [1, 2, 8, 16]
        for batch_size in batch_sizes:
            test_input = keras.random.normal([batch_size, 16, 16, 128])
            output = layer(test_input)
            assert output.shape == (batch_size, 16, 16, 128)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_layer_reuse(self):
        """Test that the same layer instance can be reused with different inputs."""
        layer = ConvBlock(filters=64)

        # Use same layer with different inputs
        input1 = keras.random.normal([2, 32, 32, 128])
        input2 = keras.random.normal([3, 16, 16, 128])

        output1 = layer(input1)
        output2 = layer(input2)

        assert output1.shape == (2, 32, 32, 64)
        assert output2.shape == (3, 16, 16, 64)

        # Both outputs should be valid
        assert not keras.ops.any(keras.ops.isnan(output1))
        assert not keras.ops.any(keras.ops.isnan(output2))

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme input values."""
        layer = AreaAttention(dim=64, num_heads=4)

        test_cases = [
            keras.ops.zeros((2, 8, 8, 64)),  # All zeros
            keras.ops.ones((2, 8, 8, 64)) * 1e-10,  # Very small values
            keras.ops.ones((2, 8, 8, 64)) * 1e5,  # Large values
            keras.random.normal((2, 8, 8, 64)) * 100,  # Large random
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for numerical stability
            assert not keras.ops.any(keras.ops.isnan(output)), "NaN detected"
            assert not keras.ops.any(keras.ops.isinf(output)), "Inf detected"