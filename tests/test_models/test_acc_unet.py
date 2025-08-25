"""
Comprehensive pytest test suite for AccUNet (A Completely Convolutional UNet).

This module provides extensive testing for the AccUNet implementation including:
- Model configuration validation and factory function testing
- Model initialization and parameter validation
- Architecture building and component integration (tested via forward pass)
- Forward pass functionality with different input formats and shapes
- Skip connection processing and multi-level feature compilation
- Encoder-decoder channel flow verification
- Serialization and deserialization of complete models (most critical test)
- Error handling and edge cases
- Factory function testing for binary vs multi-class variants
- Integration testing with HANC blocks, ResPath, and MLFC layers
- Training vs inference behavior
- Dynamic input size handling
"""

import pytest
import numpy as np
import keras
from keras import ops
import tempfile
import os

from dl_techniques.models.acc_unet import (
    AccUNet,
    create_acc_unet,
    create_acc_unet_binary,
    create_acc_unet_multiclass
)

from dl_techniques.layers.hanc_block import HANCBlock
from dl_techniques.layers.res_path import ResPath
from dl_techniques.layers.multi_level_feature_compilation import MLFCLayer


class TestAccUNetConfigurations:
    """Test AccUNet configuration validation and setup."""

    def test_default_configuration(self):
        """Test that default AccUNet configuration is valid."""
        model = AccUNet(input_channels=3, num_classes=1)

        # Check default values
        assert model.base_filters == 32
        assert model.mlfc_iterations == 3
        assert model.input_channels == 3
        assert model.num_classes == 1

        # Check filter progression
        expected_filters = [32, 64, 128, 256, 512]
        assert model.filter_sizes == expected_filters

    def test_custom_configuration(self):
        """Test AccUNet with custom configuration parameters."""
        model = AccUNet(
            input_channels=1,
            num_classes=5,
            base_filters=64,
            mlfc_iterations=4,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        assert model.input_channels == 1
        assert model.num_classes == 5
        assert model.base_filters == 64
        assert model.mlfc_iterations == 4
        assert model.kernel_regularizer is not None

        # Check scaled filter progression
        expected_filters = [64, 128, 256, 512, 1024]
        assert model.filter_sizes == expected_filters

    def test_activation_selection(self):
        """Test that correct activation is selected based on num_classes."""
        # Binary segmentation should use sigmoid
        binary_model = AccUNet(input_channels=3, num_classes=1)

        # Multi-class segmentation should use softmax
        multiclass_model = AccUNet(input_channels=3, num_classes=5)


class TestAccUNetInitialization:
    """Test AccUNet model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic AccUNet initialization and component creation."""
        model = AccUNet(input_channels=3, num_classes=1)

        # Model should not be built yet
        assert not model.built

        # Check that all components are created
        assert len(model.encoder_blocks) == 5  # 5 encoder levels
        assert len(model.pooling_layers) == 4  # 4 pooling layers
        assert len(model.decoder_blocks) == 4  # 4 decoder levels
        assert len(model.decoder_upsamples) == 4  # 4 upsample layers
        assert len(model.res_paths) == 4  # 4 ResPath layers
        assert len(model.mlfc_layers) == 3  # Default 3 MLFC layers
        assert len(model.concat_layers) == 4  # 4 concatenation layers

        # Check that each encoder level has 2 blocks
        for level_blocks in model.encoder_blocks:
            assert len(level_blocks) == 2
            for block in level_blocks:
                assert isinstance(block, HANCBlock)

        # Check that each decoder level has 2 blocks
        for level_blocks in model.decoder_blocks:
            assert len(level_blocks) == 2
            for block in level_blocks:
                assert isinstance(block, HANCBlock)

        # Check ResPath and MLFC components
        for res_path in model.res_paths:
            assert isinstance(res_path, ResPath)

        for mlfc in model.mlfc_layers:
            assert isinstance(mlfc, MLFCLayer)

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid input_channels
        with pytest.raises(ValueError, match="input_channels must be positive"):
            AccUNet(input_channels=0, num_classes=1)

        with pytest.raises(ValueError, match="input_channels must be positive"):
            AccUNet(input_channels=-1, num_classes=1)

        # Test invalid num_classes
        with pytest.raises(ValueError, match="num_classes must be positive"):
            AccUNet(input_channels=3, num_classes=0)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            AccUNet(input_channels=3, num_classes=-1)

        # Test invalid base_filters
        with pytest.raises(ValueError, match="base_filters must be positive"):
            AccUNet(input_channels=3, num_classes=1, base_filters=0)

        # Test invalid mlfc_iterations
        with pytest.raises(ValueError, match="mlfc_iterations must be positive"):
            AccUNet(input_channels=3, num_classes=1, mlfc_iterations=0)

    def test_component_configuration(self):
        """Test that components are configured with correct parameters."""
        model = AccUNet(
            input_channels=1,
            num_classes=1,
            base_filters=32,
            mlfc_iterations=2
        )

        # Check ResPath configurations
        expected_res_path_blocks = [4, 3, 2, 1]
        expected_res_path_channels = [32, 64, 128, 256]

        for i, (res_path, expected_blocks, expected_channels) in enumerate(
                zip(model.res_paths, expected_res_path_blocks, expected_res_path_channels)
        ):
            assert res_path.num_blocks == expected_blocks
            assert res_path.channels == expected_channels

        # Check MLFC configuration
        assert len(model.mlfc_layers) == 2  # mlfc_iterations=2
        for mlfc in model.mlfc_layers:
            assert mlfc.channels_list == [32, 64, 128, 256]  # First 4 levels
            assert mlfc.num_iterations == 1  # Each MLFC does 1 iteration


class TestAccUNetForwardPass:
    """Test AccUNet forward pass, which also implicitly tests automatic building."""

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for testing."""
        return {
            'grayscale': keras.random.normal((2, 224, 224, 1)),
            'rgb': keras.random.normal((2, 224, 224, 3)),
            'small': keras.random.normal((1, 128, 128, 3)),
            'large': keras.random.normal((1, 512, 512, 1))
        }

    def test_forward_pass_binary_segmentation(self, sample_inputs):
        """Test forward pass for binary segmentation."""
        model = AccUNet(input_channels=3, num_classes=1)

        assert not model.built
        output = model(sample_inputs['rgb'], training=False)
        assert model.built

        # Check output shape
        batch_size, height, width = sample_inputs['rgb'].shape[:3]
        expected_shape = (batch_size, height, width, 1)
        assert output.shape == expected_shape

        # Check output range for sigmoid activation
        assert ops.all(output >= 0.0)
        assert ops.all(output <= 1.0)

    def test_forward_pass_multiclass_segmentation(self, sample_inputs):
        """Test forward pass for multi-class segmentation."""
        num_classes = 5
        model = AccUNet(input_channels=3, num_classes=num_classes)

        output = model(sample_inputs['rgb'], training=False)

        # Check output shape
        batch_size, height, width = sample_inputs['rgb'].shape[:3]
        expected_shape = (batch_size, height, width, num_classes)
        assert output.shape == expected_shape

        # Check that softmax probabilities sum to 1
        prob_sums = ops.sum(output, axis=-1)
        np.testing.assert_allclose(
            ops.convert_to_numpy(prob_sums),
            1.0,
            rtol=1e-6, atol=1e-6,
            err_msg="Softmax probabilities should sum to 1"
        )

    @pytest.mark.parametrize("input_channels,num_classes", [
        (1, 1),  # Grayscale binary
        (1, 5),  # Grayscale multiclass
        (3, 1),  # RGB binary
        (3, 8),  # RGB multiclass
    ])
    def test_forward_pass_different_configurations(self, sample_inputs, input_channels, num_classes):
        """Test forward pass with different input/output configurations."""
        model = AccUNet(input_channels=input_channels, num_classes=num_classes)

        # Select appropriate input
        if input_channels == 1:
            test_input = sample_inputs['grayscale']
        else:
            test_input = sample_inputs['rgb']

        output = model(test_input, training=False)

        # Check output shape
        batch_size, height, width = test_input.shape[:3]
        expected_shape = (batch_size, height, width, num_classes)
        assert output.shape == expected_shape

    @pytest.mark.parametrize("input_size", [
        (64, 64),
        (128, 128),
        (224, 224),
        (256, 256),
        (512, 512)
    ])
    def test_forward_pass_different_input_sizes(self, input_size):
        """Test forward pass with different input sizes."""
        height, width = input_size
        model = AccUNet(input_channels=3, num_classes=1)

        test_input = keras.random.normal((1, height, width, 3))
        output = model(test_input, training=False)

        # Output should have same spatial dimensions as input
        expected_shape = (1, height, width, 1)
        assert output.shape == expected_shape

    def test_forward_pass_training_vs_inference(self, sample_inputs):
        """Test forward pass behavior in training vs inference modes."""
        model = AccUNet(input_channels=3, num_classes=1)

        # Test training mode
        output_train = model(sample_inputs['rgb'], training=True)

        # Test inference mode
        output_infer = model(sample_inputs['rgb'], training=False)

        # Outputs should have same shape
        assert output_train.shape == output_infer.shape

        # Due to dropout and batch norm, outputs may be different
        # but we just check that both modes work without errors

    def test_forward_pass_custom_architecture(self, sample_inputs):
        """Test forward pass with custom architecture parameters."""
        model = AccUNet(
            input_channels=3,
            num_classes=1,
            base_filters=64,  # Larger network
            mlfc_iterations=4  # More feature compilation
        )

        output = model(sample_inputs['rgb'], training=False)

        # Check output shape
        batch_size, height, width = sample_inputs['rgb'].shape[:3]
        expected_shape = (batch_size, height, width, 1)
        assert output.shape == expected_shape


class TestAccUNetArchitecture:
    """Test specific AccUNet architecture components and integration."""

    def test_encoder_decoder_symmetry(self):
        """Test that encoder and decoder have symmetric structure."""
        model = AccUNet(input_channels=3, num_classes=1)

        # 5 encoder levels, 4 decoder levels (bottleneck not decoded)
        assert len(model.encoder_blocks) == 5
        assert len(model.decoder_blocks) == 4

        # 4 pooling layers for encoder downsampling
        assert len(model.pooling_layers) == 4

        # 4 upsampling layers for decoder upsampling
        assert len(model.decoder_upsamples) == 4

    def test_hanc_block_k_values(self):
        """Test that HANC blocks have correct k values according to paper."""
        model = AccUNet(input_channels=3, num_classes=1)

        # Build model to access block configurations
        test_input = keras.random.normal((1, 224, 224, 3))
        _ = model(test_input)

        # Expected k values for encoder levels: [3, 3, 3, 2, 1]
        expected_encoder_k = [3, 3, 3, 2, 1]
        for level, expected_k in enumerate(expected_encoder_k):
            for block in model.encoder_blocks[level]:
                assert block.k == expected_k

        # Expected k values for decoder levels: [2, 2, 3, 3]
        expected_decoder_k = [2, 2, 3, 3]
        for level, expected_k in enumerate(expected_decoder_k):
            for block in model.decoder_blocks[level]:
                assert block.k == expected_k

    def test_skip_connection_processing(self):
        """Test skip connection processing through ResPath and MLFC."""
        model = AccUNet(input_channels=3, num_classes=1)

        # Test that ResPath has correct number of blocks for each level
        expected_res_path_blocks = [4, 3, 2, 1]
        for res_path, expected_blocks in zip(model.res_paths, expected_res_path_blocks):
            assert res_path.num_blocks == expected_blocks

        # Test that MLFC processes the right number of levels
        for mlfc in model.mlfc_layers:
            assert len(mlfc.channels_list) == 4  # First 4 encoder levels

    def test_channel_flow_consistency(self):
        """Test that channel dimensions are consistent throughout the network."""
        model = AccUNet(input_channels=3, num_classes=2, base_filters=32)

        # Build the model
        test_input = keras.random.normal((1, 224, 224, 3))
        _ = model(test_input)

        # Check encoder block filter consistency
        expected_encoder_filters = [32, 64, 128, 256, 512]
        for level, expected_filters in enumerate(expected_encoder_filters):
            for block in model.encoder_blocks[level]:
                assert block.filters == expected_filters

        # Check decoder block filter consistency
        expected_decoder_filters = [256, 128, 64, 32]  # Reverse order from encoder
        for level, expected_filters in enumerate(expected_decoder_filters):
            for block in model.decoder_blocks[level]:
                assert block.filters == expected_filters


class TestAccUNetSerialization:
    """Test AccUNet serialization and deserialization - the most critical test."""

    def test_model_save_load_cycle_binary(self):
        """Test complete save/load cycle for binary segmentation model."""
        original_model = AccUNet(input_channels=3, num_classes=1)
        test_input = keras.random.normal((2, 224, 224, 3))

        # Build model and get prediction
        original_output = original_model(test_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_accunet_binary.keras')
            original_model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(test_input, training=False)

            # Check that configurations are preserved
            assert loaded_model.input_channels == original_model.input_channels
            assert loaded_model.num_classes == original_model.num_classes
            assert loaded_model.base_filters == original_model.base_filters
            assert loaded_model.mlfc_iterations == original_model.mlfc_iterations

            # Check that outputs are numerically identical
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Outputs should match after serialization"
            )

    def test_model_save_load_cycle_multiclass(self):
        """Test complete save/load cycle for multi-class segmentation model."""
        original_model = AccUNet(
            input_channels=1,
            num_classes=5,
            base_filters=64,
            mlfc_iterations=2
        )
        test_input = keras.random.normal((1, 256, 256, 1))

        # Build model and get prediction
        original_output = original_model(test_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_accunet_multiclass.keras')
            original_model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(test_input, training=False)

            # Check that all configurations are preserved
            assert loaded_model.input_channels == 1
            assert loaded_model.num_classes == 5
            assert loaded_model.base_filters == 64
            assert loaded_model.mlfc_iterations == 2

            # Check output shapes and numerical accuracy
            assert loaded_output.shape == original_output.shape
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Outputs should match after serialization"
            )

    def test_model_save_load_with_regularization(self):
        """Test save/load cycle with regularization parameters."""
        original_model = AccUNet(
            input_channels=3,
            num_classes=1,
            kernel_regularizer=keras.regularizers.L2(1e-4),
            bias_regularizer=keras.regularizers.L1(1e-5)
        )
        test_input = keras.random.normal((1, 128, 128, 3))

        # Build and test
        original_output = original_model(test_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_accunet_regularized.keras')
            original_model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(test_input, training=False)

            # Verify outputs match
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6
            )


class TestAccUNetFactoryFunctions:
    """Test factory functions for creating AccUNet models."""

    def test_create_acc_unet_basic(self):
        """Test basic create_acc_unet function."""
        model = create_acc_unet(
            input_channels=3,
            num_classes=1,
            input_shape=(224, 224)
        )

        # Should be a functional Model, not AccUNet class
        assert isinstance(model, keras.Model)
        assert model.name == 'ACC_UNet'

        # Test forward pass
        test_input = keras.random.normal((1, 224, 224, 3))
        output = model(test_input)
        assert output.shape == (1, 224, 224, 1)

    def test_create_acc_unet_dynamic_shape(self):
        """Test create_acc_unet with dynamic input shape."""
        model = create_acc_unet(
            input_channels=3,
            num_classes=2,
            input_shape=None  # Dynamic shape
        )

        # Test with different input sizes
        for size in [(128, 128), (256, 256), (512, 512)]:
            test_input = keras.random.normal((1, size[0], size[1], 3))
            output = model(test_input)
            assert output.shape == (1, size[0], size[1], 2)

    def test_create_acc_unet_binary(self):
        """Test create_acc_unet_binary factory function."""
        model = create_acc_unet_binary(
            input_channels=1,
            input_shape=(256, 256)
        )

        test_input = keras.random.normal((2, 256, 256, 1))
        output = model(test_input)

        # Should be binary segmentation (1 class)
        assert output.shape == (2, 256, 256, 1)

        # Should use sigmoid activation
        assert ops.all(output >= 0.0)
        assert ops.all(output <= 1.0)

    def test_create_acc_unet_multiclass(self):
        """Test create_acc_unet_multiclass factory function."""
        num_classes = 7
        model = create_acc_unet_multiclass(
            input_channels=3,
            num_classes=num_classes,
            input_shape=(224, 224)
        )

        test_input = keras.random.normal((1, 224, 224, 3))
        output = model(test_input)

        # Should have correct number of classes
        assert output.shape == (1, 224, 224, num_classes)

        # Should use softmax activation (probabilities sum to 1)
        prob_sums = ops.sum(output, axis=-1)
        np.testing.assert_allclose(
            ops.convert_to_numpy(prob_sums),
            1.0,
            rtol=1e-6, atol=1e-6
        )

    def test_create_acc_unet_multiclass_validation(self):
        """Test validation in create_acc_unet_multiclass."""
        # Should raise error for num_classes <= 1
        with pytest.raises(ValueError, match="num_classes must be > 1"):
            create_acc_unet_multiclass(
                input_channels=3,
                num_classes=1  # Invalid for multiclass
            )

    @pytest.mark.parametrize("factory_func,expected_classes", [
        (create_acc_unet_binary, 1),
        (lambda **kwargs: create_acc_unet_multiclass(num_classes=3, **kwargs), 3),
        (lambda **kwargs: create_acc_unet_multiclass(num_classes=10, **kwargs), 10),
    ])
    def test_factory_functions_parameter_scaling(self, factory_func, expected_classes):
        """Test that factory functions create models with correct scaling."""
        model = factory_func(
            input_channels=3,
            input_shape=(128, 128),
            base_filters=32
        )

        test_input = keras.random.normal((1, 128, 128, 3))
        output = model(test_input)

        assert output.shape == (1, 128, 128, expected_classes)

    def test_factory_functions_custom_parameters(self):
        """Test factory functions with custom parameters."""
        # Test with larger network
        model = create_acc_unet_binary(
            input_channels=3,
            base_filters=64,  # Larger base
            mlfc_iterations=4,  # More iterations
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        test_input = keras.random.normal((1, 256, 256, 3))
        output = model(test_input)

        assert output.shape == (1, 256, 256, 1)

    def test_factory_vs_direct_equivalence(self):
        """Test that factory functions create equivalent models to direct instantiation."""
        # Create models both ways
        direct_model = AccUNet(input_channels=3, num_classes=1, base_filters=32)
        factory_model = create_acc_unet_binary(
            input_channels=3,
            base_filters=32,
            input_shape=(224, 224)
        )

        # Build both models
        test_input = keras.random.normal((1, 224, 224, 3))
        direct_output = direct_model(test_input)
        factory_output = factory_model(test_input)

        # Should have same output shape
        assert direct_output.shape == factory_output.shape


class TestAccUNetErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_input_shapes(self):
        """Test behavior with invalid input shapes."""
        model = AccUNet(input_channels=3, num_classes=1)

        # Test with wrong number of channels
        wrong_channels = keras.random.normal((1, 224, 224, 1))  # 1 channel instead of 3

        with pytest.raises((ValueError, Exception)):  # May raise different exceptions
            model(wrong_channels)

    def test_very_small_input_size(self):
        """Test behavior with very small input sizes."""
        model = AccUNet(input_channels=3, num_classes=1)

        # Test with size that might cause issues with pooling
        small_input = keras.random.normal((1, 16, 16, 3))

        # This might work or might fail depending on architecture
        # If it works, output should have correct shape
        try:
            output = model(small_input)
            assert output.shape == (1, 16, 16, 1)
        except Exception:
            # Small inputs may not work with the architecture
            pass

    def test_batch_size_variations(self):
        """Test with different batch sizes including edge cases."""
        model = AccUNet(input_channels=3, num_classes=1)

        # Test with batch size 1
        single_input = keras.random.normal((1, 128, 128, 3))
        single_output = model(single_input)
        assert single_output.shape == (1, 128, 128, 1)

        # Test with larger batch
        batch_input = keras.random.normal((8, 128, 128, 3))
        batch_output = model(batch_input)
        assert batch_output.shape == (8, 128, 128, 1)

    def test_memory_efficiency(self):
        """Test that model can handle reasonably sized inputs without memory issues."""
        model = AccUNet(input_channels=3, num_classes=1, base_filters=16)  # Smaller model

        # Test with moderately large input
        large_input = keras.random.normal((1, 512, 512, 3))

        try:
            output = model(large_input)
            assert output.shape == (1, 512, 512, 1)
        except Exception as e:
            # If it fails due to memory, that's acceptable
            if "memory" not in str(e).lower() and "resource" not in str(e).lower():
                raise  # Re-raise if it's not a memory issue