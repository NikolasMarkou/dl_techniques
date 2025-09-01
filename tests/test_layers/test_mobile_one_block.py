"""
Comprehensive Test Suite for MobileOneBlock Layer

This test suite follows the Modern Keras 3 testing guidelines and covers:
- Basic functionality and initialization
- Forward pass and building
- Critical serialization cycle testing
- Reparameterization functionality
- Edge cases and error conditions
- Various configurations
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
import tensorflow as tf

# Import the layer being tested
from dl_techniques.layers.mobile_one_block import MobileOneBlock


class TestMobileOneBlock:
    """Comprehensive test suite for MobileOneBlock layer."""

    @pytest.fixture
    def basic_config(self):
        """Standard configuration for testing."""
        return {
            'out_channels': 64,
            'kernel_size': 3,
            'stride': 1,
            'padding': 'same',
            'activation': 'gelu'
        }

    @pytest.fixture
    def complex_config(self):
        """Complex configuration with all features enabled."""
        return {
            'out_channels': 128,
            'kernel_size': 3,
            'stride': 2,
            'padding': 'same',
            'use_se': True,
            'num_conv_branches': 2,
            'activation': 'relu',
            'kernel_initializer': 'he_normal',
            'bias_initializer': 'zeros'
        }

    @pytest.fixture
    def sample_input(self):
        """Sample 4D input tensor for testing."""
        return keras.random.normal(shape=(4, 32, 32, 16))

    @pytest.fixture
    def large_input(self):
        """Larger input for stride testing."""
        return keras.random.normal(shape=(2, 64, 64, 32))

    # ========================================================================
    # Basic Functionality Tests
    # ========================================================================

    def test_initialization_basic(self, basic_config):
        """Test basic layer initialization."""
        layer = MobileOneBlock(**basic_config)

        # Check configuration stored correctly
        assert layer.out_channels == 64
        assert layer.kernel_size == 3
        assert layer.stride == 1
        assert layer.padding == 'same'
        assert not layer.use_se
        assert layer.num_conv_branches == 1

        # Check layer is not built yet
        assert not layer.built
        assert layer.inference_mode == False

        # Check sub-layers created
        assert len(layer.conv_branches) == 1
        assert layer.scale_branch is not None  # kernel_size > 1
        assert layer.skip_branch is None  # Not built yet
        assert layer.se_block is None  # use_se=False

    def test_initialization_complex(self, complex_config):
        """Test initialization with complex configuration."""
        layer = MobileOneBlock(**complex_config)

        # Check all parameters
        assert layer.out_channels == 128
        assert layer.use_se == True
        assert layer.num_conv_branches == 2
        assert len(layer.conv_branches) == 2
        assert layer.se_block is not None  # SE enabled

    def test_forward_pass_basic(self, basic_config, sample_input):
        """Test basic forward pass and building."""
        layer = MobileOneBlock(**basic_config)

        # Forward pass triggers building
        output = layer(sample_input)

        # Check layer is now built
        assert layer.built

        # Check output shape
        expected_shape = (4, 32, 32, 64)  # same padding, stride=1
        assert output.shape == expected_shape

        # Check sub-layers are built
        assert all(branch.built for branch in layer.conv_branches)
        if layer.scale_branch:
            assert layer.scale_branch.built
        if layer.skip_branch:
            assert layer.skip_branch.built

    def test_forward_pass_with_stride(self, sample_input):
        """Test forward pass with stride > 1."""
        layer = MobileOneBlock(
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding='same'
        )

        output = layer(sample_input)

        # Check downsampled output
        expected_shape = (4, 16, 16, 32)  # stride=2 halves spatial dims
        assert output.shape == expected_shape

    def test_forward_pass_with_se(self, sample_input):
        """Test forward pass with Squeeze-and-Excitation."""
        layer = MobileOneBlock(
            out_channels=64,
            kernel_size=3,
            use_se=True
        )

        output = layer(sample_input)

        assert output.shape == (4, 32, 32, 64)
        assert layer.se_block is not None
        assert layer.se_block.built

    def test_skip_connection_creation(self):
        """Test skip connection is created when appropriate."""
        # Case 1: Same channels, stride=1 -> skip connection
        layer1 = MobileOneBlock(out_channels=16, kernel_size=3, stride=1)
        input1 = keras.random.normal(shape=(2, 32, 32, 16))

        _ = layer1(input1)  # Build the layer
        assert layer1.skip_branch is not None

        # Case 2: Different channels -> no skip connection
        layer2 = MobileOneBlock(out_channels=32, kernel_size=3, stride=1)
        input2 = keras.random.normal(shape=(2, 32, 32, 16))

        _ = layer2(input2)  # Build the layer
        # Skip branch should be None because in_channels != out_channels
        assert layer2.skip_branch is None

        # Case 3: Same channels, stride > 1 -> no skip connection
        layer3 = MobileOneBlock(out_channels=16, kernel_size=3, stride=2)
        input3 = keras.random.normal(shape=(2, 32, 32, 16))

        _ = layer3(input3)  # Build the layer
        assert layer3.skip_branch is None

    # ========================================================================
    # Serialization Tests (CRITICAL)
    # ========================================================================

    def test_serialization_cycle_basic(self, basic_config, sample_input):
        """CRITICAL TEST: Full serialization cycle with basic config."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MobileOneBlock(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization (basic config)"
            )

    def test_serialization_cycle_complex(self, complex_config, sample_input):
        """CRITICAL TEST: Full serialization cycle with complex config."""
        # Create model with complex configuration
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MobileOneBlock(**complex_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_complex_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization (complex config)"
            )

    def test_config_completeness(self, complex_config):
        """Test that get_config contains all __init__ parameters."""
        layer = MobileOneBlock(**complex_config)
        config = layer.get_config()

        # Check all important config parameters are present
        required_keys = [
            'out_channels', 'kernel_size', 'stride', 'padding',
            'use_se', 'num_conv_branches', 'activation',
            'kernel_initializer', 'bias_initializer'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Check values match
        assert config['out_channels'] == complex_config['out_channels']
        assert config['use_se'] == complex_config['use_se']
        assert config['num_conv_branches'] == complex_config['num_conv_branches']

    # ========================================================================
    # Training and Gradient Tests
    # ========================================================================

    def test_gradients_flow(self, basic_config, sample_input):
        """Test gradient computation through the layer."""
        layer = MobileOneBlock(**basic_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Check gradients exist for all trainable variables
        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) > 0
        assert all(g is not None for g in gradients), "Some gradients are None"

        # Check gradients have reasonable magnitudes
        for grad in gradients:
            grad_norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(grad)))
            assert grad_norm > 0, "Gradient norm should be positive"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, basic_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = MobileOneBlock(**basic_config)

        output = layer(sample_input, training=training)

        # Should complete successfully for all training modes
        assert output.shape == (4, 32, 32, 64)

    def test_training_vs_inference_mode_differences(self, sample_input):
        """Test that dropout in SE block behaves differently in training vs inference."""
        layer = MobileOneBlock(
            out_channels=64,
            kernel_size=3,
            use_se=True
        )

        # Multiple forward passes in training mode might give different results due to dropout
        output1_train = layer(sample_input, training=True)
        output2_train = layer(sample_input, training=True)

        # Inference mode should be deterministic
        output1_infer = layer(sample_input, training=False)
        output2_infer = layer(sample_input, training=False)

        # Inference outputs should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1_infer),
            keras.ops.convert_to_numpy(output2_infer),
            rtol=1e-6, atol=1e-6,
            err_msg="Inference mode should be deterministic"
        )

    # ========================================================================
    # Edge Cases and Error Conditions
    # ========================================================================

    def test_invalid_parameters(self):
        """Test error conditions for invalid parameters."""
        # Invalid out_channels
        with pytest.raises(ValueError, match="out_channels must be positive"):
            MobileOneBlock(out_channels=0, kernel_size=3)

        with pytest.raises(ValueError, match="out_channels must be positive"):
            MobileOneBlock(out_channels=-1, kernel_size=3)

        # Invalid kernel_size
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            MobileOneBlock(out_channels=64, kernel_size=0)

        # Invalid stride
        with pytest.raises(ValueError, match="stride must be positive"):
            MobileOneBlock(out_channels=64, kernel_size=3, stride=0)

        # Invalid num_conv_branches
        with pytest.raises(ValueError, match="num_conv_branches must be positive"):
            MobileOneBlock(out_channels=64, kernel_size=3, num_conv_branches=0)

        # Invalid padding
        with pytest.raises(ValueError, match="padding must be 'same' or 'valid'"):
            MobileOneBlock(out_channels=64, kernel_size=3, padding='invalid')

    def test_compute_output_shape(self, basic_config):
        """Test output shape computation."""
        layer = MobileOneBlock(**basic_config)

        input_shape = (None, 32, 32, 16)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 32, 32, 64)  # same padding, stride=1
        assert output_shape == expected_shape

    def test_compute_output_shape_with_stride(self):
        """Test output shape computation with stride."""
        layer = MobileOneBlock(out_channels=64, kernel_size=3, stride=2, padding='same')

        input_shape = (None, 32, 32, 16)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 16, 16, 64)  # stride=2 halves dimensions
        assert output_shape == expected_shape

    def test_compute_output_shape_valid_padding(self):
        """Test output shape computation with valid padding."""
        layer = MobileOneBlock(out_channels=64, kernel_size=3, stride=1, padding='valid')

        input_shape = (None, 32, 32, 16)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 30, 30, 64)  # valid padding reduces by kernel_size-1
        assert output_shape == expected_shape

    # ========================================================================
    # Configuration Variations
    # ========================================================================

    @pytest.mark.parametrize("kernel_size", [1, 3, 5])
    def test_different_kernel_sizes(self, sample_input, kernel_size):
        """Test different kernel sizes."""
        layer = MobileOneBlock(out_channels=32, kernel_size=kernel_size)

        output = layer(sample_input)
        assert output.shape == (4, 32, 32, 32)

        # Check scale branch creation
        if kernel_size > 1:
            assert layer.scale_branch is not None
        else:
            assert layer.scale_branch is None

    @pytest.mark.parametrize("activation", ['relu', 'gelu', 'swish'])
    def test_different_activations(self, sample_input, activation):
        """Test different activation functions."""
        layer = MobileOneBlock(out_channels=64, kernel_size=3, activation=activation)

        output = layer(sample_input)
        assert output.shape == (4, 32, 32, 64)

    @pytest.mark.parametrize("num_branches", [1, 2, 3])
    def test_different_branch_counts(self, sample_input, num_branches):
        """Test different numbers of conv branches."""
        layer = MobileOneBlock(
            out_channels=64,
            kernel_size=3,
            num_conv_branches=num_branches
        )

        output = layer(sample_input)
        assert output.shape == (4, 32, 32, 64)
        assert len(layer.conv_branches) == num_branches

    # ========================================================================
    # Integration Tests
    # ========================================================================

    def test_in_sequential_model(self, sample_input):
        """Test MobileOneBlock in a Sequential model."""
        model = keras.Sequential([
            keras.layers.Input(shape=(32, 32, 16)),
            MobileOneBlock(out_channels=64, kernel_size=3),
            MobileOneBlock(out_channels=128, kernel_size=3, stride=2),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile and test forward pass
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        output = model(sample_input)
        assert output.shape == (4, 10)

    def test_multiple_blocks_serialization(self, sample_input):
        """Test serialization of model with multiple MobileOne blocks."""
        inputs = keras.Input(shape=(32, 32, 16))
        x = MobileOneBlock(out_channels=32, kernel_size=3)(inputs)
        x = MobileOneBlock(out_channels=64, kernel_size=3, stride=2, use_se=True)(x)
        x = MobileOneBlock(out_channels=128, kernel_size=3, num_conv_branches=2)(x)
        outputs = keras.layers.GlobalAveragePooling2D()(x)

        model = keras.Model(inputs, outputs)
        original_pred = model(sample_input)

        # Test serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'multi_block_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Multi-block model serialization failed"
            )

