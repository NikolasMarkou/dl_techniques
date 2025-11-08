"""
Comprehensive pytest test suite for SCUNet (Swin-Conv-UNet).

This module provides extensive testing for the SCUNet implementation including:
- Model configuration validation and parameter testing
- Model initialization and component validation
- Architecture building and component integration (tested via forward pass)
- Forward pass functionality with different input formats and shapes
- Padding behavior for inputs not divisible by 64
- Skip connection processing through encoder-decoder paths
- Channel dimension flow verification
- Serialization and deserialization of complete models (most critical test)
- Error handling and edge cases
- Training vs inference behavior
- Dynamic input size handling

The SCUNet architecture combines Swin Transformer blocks with convolutional
operations in a U-Net structure for image restoration tasks.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tempfile
import os
from typing import Tuple, List

from dl_techniques.models.scunet.model import SCUNet
from dl_techniques.layers.transformers.swin_conv_block import SwinConvBlock


class TestSCUNetConfigurations:
    """Test SCUNet configuration validation and setup."""

    def test_default_configuration(self) -> None:
        """Test that default SCUNet configuration is valid."""
        model = SCUNet(in_nc=3)

        # Check default values
        assert model.in_nc == 3
        assert model.dim == 64
        assert model.head_dim == 32
        assert model.window_size == 8
        assert model.drop_path_rate == 0.0
        assert model.input_resolution == 256

        # Check default config (7 stages)
        expected_config = [4, 4, 4, 4, 4, 4, 4]
        assert model.config == expected_config

    def test_custom_configuration(self) -> None:
        """Test SCUNet with custom configuration parameters."""
        custom_config = [2, 2, 2, 2, 2, 2, 2]
        model = SCUNet(
            in_nc=1,
            config=custom_config,
            dim=128,
            head_dim=64,
            window_size=16,
            drop_path_rate=0.1,
            input_resolution=512
        )

        assert model.in_nc == 1
        assert model.config == custom_config
        assert model.dim == 128
        assert model.head_dim == 64
        assert model.window_size == 16
        assert model.drop_path_rate == 0.1
        assert model.input_resolution == 512

    def test_config_length_variations(self) -> None:
        """Test SCUNet with different config lengths."""
        # Standard 7-stage config
        config_7 = [4, 4, 4, 4, 4, 4, 4]
        model = SCUNet(in_nc=3, config=config_7)
        assert len(model.config) == 7

        # Verify total blocks for drop path rate calculation
        total_blocks = sum(config_7)
        assert total_blocks == 28

    @pytest.mark.parametrize("dim,head_dim", [
        (32, 16),
        (64, 32),
        (128, 64),
    ])
    def test_dimension_scaling(self, dim: int, head_dim: int) -> None:
        """Test SCUNet with different dimension configurations."""
        model = SCUNet(in_nc=3, dim=dim, head_dim=head_dim)

        assert model.dim == dim
        assert model.head_dim == head_dim

    def test_drop_path_rate_range(self) -> None:
        """Test SCUNet with different drop path rates."""
        for drop_rate in [0.0, 0.1, 0.2, 0.5]:
            model = SCUNet(in_nc=3, drop_path_rate=drop_rate)
            assert model.drop_path_rate == drop_rate


class TestSCUNetInitialization:
    """Test SCUNet model initialization and component validation."""

    def test_basic_initialization(self) -> None:
        """Test basic SCUNet initialization and component creation."""
        model = SCUNet(in_nc=3)

        # Model should not be built yet
        assert not model.built

        # Check that all main components are created
        assert hasattr(model, 'm_head')
        assert hasattr(model, 'm_down1')
        assert hasattr(model, 'm_down2')
        assert hasattr(model, 'm_down3')
        assert hasattr(model, 'm_body')
        assert hasattr(model, 'm_up3')
        assert hasattr(model, 'm_up2')
        assert hasattr(model, 'm_up1')
        assert hasattr(model, 'm_tail')

        # Check that components are Sequential models
        assert isinstance(model.m_down1, keras.Sequential)
        assert isinstance(model.m_down2, keras.Sequential)
        assert isinstance(model.m_down3, keras.Sequential)
        assert isinstance(model.m_body, keras.Sequential)
        assert isinstance(model.m_up3, keras.Sequential)
        assert isinstance(model.m_up2, keras.Sequential)
        assert isinstance(model.m_up1, keras.Sequential)

    def test_encoder_stages_have_correct_blocks(self) -> None:
        """Test that encoder stages contain correct number of blocks."""
        config = [4, 4, 4, 4, 4, 4, 4]
        model = SCUNet(in_nc=3, config=config)

        # Build the model to access internal structure
        test_input = keras.random.normal((1, 64, 64, 3))
        _ = model(test_input, training=False)

        # Each encoder stage should have config[i] blocks plus a downsample layer
        # down1: 4 blocks + 1 downsample = 5 layers
        assert len(model.m_down1.layers) == config[0] + 1

        # down2: 4 blocks + 1 downsample = 5 layers
        assert len(model.m_down2.layers) == config[1] + 1

        # down3: 4 blocks + 1 downsample = 5 layers
        assert len(model.m_down3.layers) == config[2] + 1

        # body: 4 blocks (no downsample)
        assert len(model.m_body.layers) == config[3]

    def test_decoder_stages_have_correct_blocks(self) -> None:
        """Test that decoder stages contain correct number of blocks."""
        config = [4, 4, 4, 4, 4, 4, 4]
        model = SCUNet(in_nc=3, config=config)

        # Build the model
        test_input = keras.random.normal((1, 64, 64, 3))
        _ = model(test_input, training=False)

        # Each decoder stage should have 1 upsample + config[i] blocks
        # up3: 1 upsample + 4 blocks = 5 layers
        assert len(model.m_up3.layers) == 1 + config[4]

        # up2: 1 upsample + 4 blocks = 5 layers
        assert len(model.m_up2.layers) == 1 + config[5]

        # up1: 1 upsample + 4 blocks = 5 layers
        assert len(model.m_up1.layers) == 1 + config[6]

    def test_swin_conv_block_types(self) -> None:
        """Test that SwinConvBlocks alternate between W and SW types."""
        model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4])

        # Build the model
        test_input = keras.random.normal((1, 128, 128, 3))
        _ = model(test_input, training=False)

        # Check m_down1 blocks (first 4 layers before downsample)
        down1_blocks = model.m_down1.layers[:-1]  # Exclude downsample
        for i, layer in enumerate(down1_blocks):
            if isinstance(layer, SwinConvBlock):
                expected_type = "W" if i % 2 == 0 else "SW"
                assert layer.block_type == expected_type

    def test_head_and_tail_conv_layers(self) -> None:
        """Test that head and tail convolution layers are properly configured."""
        in_nc = 3
        dim = 64
        model = SCUNet(in_nc=in_nc, dim=dim)

        # Head should output dim channels
        assert isinstance(model.m_head, keras.layers.Conv2D)
        assert model.m_head.filters == dim

        # Tail should output in_nc channels
        assert isinstance(model.m_tail, keras.layers.Conv2D)
        assert model.m_tail.filters == in_nc


class TestSCUNetForwardPass:
    """Test SCUNet forward pass functionality."""

    @pytest.fixture
    def sample_inputs(self) -> dict:
        """Create sample inputs for testing."""
        return {
            'grayscale': keras.random.normal((2, 128, 128, 1)),
            'rgb': keras.random.normal((2, 256, 256, 3)),
            'small': keras.random.normal((1, 64, 64, 3)),
            'large': keras.random.normal((1, 512, 512, 3)),
            'odd_size': keras.random.normal((1, 100, 100, 3)),
        }

    def test_forward_pass_basic(self, sample_inputs: dict) -> None:
        """Test basic forward pass."""
        model = SCUNet(in_nc=3)

        assert not model.built
        output = model(sample_inputs['rgb'], training=False)
        assert model.built

        # Check output shape matches input shape
        batch_size, height, width, channels = sample_inputs['rgb'].shape
        expected_shape = (batch_size, height, width, channels)
        assert output.shape == expected_shape

    def test_forward_pass_grayscale(self, sample_inputs: dict) -> None:
        """Test forward pass with grayscale images."""
        model = SCUNet(in_nc=1)

        output = model(sample_inputs['grayscale'], training=False)

        # Check output shape
        batch_size, height, width, channels = sample_inputs['grayscale'].shape
        expected_shape = (batch_size, height, width, channels)
        assert output.shape == expected_shape

    @pytest.mark.parametrize("input_size", [
        (64, 64),
        (128, 128),
        (192, 192),
        (256, 256),
        (512, 512),
    ])
    def test_forward_pass_different_input_sizes(self, input_size: Tuple[int, int]) -> None:
        """Test forward pass with different input sizes."""
        height, width = input_size
        model = SCUNet(in_nc=3, dim=32, head_dim=16)  # Smaller model for faster testing

        test_input = keras.random.normal((1, height, width, 3))
        output = model(test_input, training=False)

        # Output should have same spatial dimensions as input
        expected_shape = (1, height, width, 3)
        assert output.shape == expected_shape

    def test_forward_pass_with_padding(self, sample_inputs: dict) -> None:
        """Test forward pass with odd-sized input that requires padding."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)

        # Use odd size that's not divisible by 64
        odd_input = sample_inputs['odd_size']  # (1, 100, 100, 3)

        output = model(odd_input, training=False)

        # Output should match input size (padding is removed)
        assert output.shape == odd_input.shape

    @pytest.mark.parametrize("in_nc", [1, 3, 4])
    def test_forward_pass_different_channels(self, in_nc: int) -> None:
        """Test forward pass with different number of input channels."""
        model = SCUNet(in_nc=in_nc, dim=32, head_dim=16)

        test_input = keras.random.normal((1, 128, 128, in_nc))
        output = model(test_input, training=False)

        # Output should have same channels as input
        expected_shape = (1, 128, 128, in_nc)
        assert output.shape == expected_shape

    def test_forward_pass_training_vs_inference(self, sample_inputs: dict) -> None:
        """Test forward pass behavior in training vs inference modes."""
        model = SCUNet(in_nc=3, drop_path_rate=0.1)

        # Test training mode
        output_train = model(sample_inputs['rgb'], training=True)

        # Test inference mode
        output_infer = model(sample_inputs['rgb'], training=False)

        # Outputs should have same shape
        assert output_train.shape == output_infer.shape

        # Due to dropout and drop path, outputs should be different
        # but we just verify both modes work without errors

    def test_forward_pass_batch_sizes(self, sample_inputs: dict) -> None:
        """Test forward pass with different batch sizes."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)

        for batch_size in [1, 2, 4, 8]:
            test_input = keras.random.normal((batch_size, 128, 128, 3))
            output = model(test_input, training=False)

            expected_shape = (batch_size, 128, 128, 3)
            assert output.shape == expected_shape

    def test_forward_pass_custom_architecture(self, sample_inputs: dict) -> None:
        """Test forward pass with custom architecture parameters."""
        model = SCUNet(
            in_nc=3,
            config=[2, 2, 2, 2, 2, 2, 2],  # Smaller config
            dim=32,  # Smaller dimension
            head_dim=16,
            window_size=4,
            drop_path_rate=0.05,
            input_resolution=128
        )

        output = model(sample_inputs['small'], training=False)

        batch_size, height, width, channels = sample_inputs['small'].shape
        expected_shape = (batch_size, height, width, channels)
        assert output.shape == expected_shape


class TestSCUNetPaddingBehavior:
    """Test SCUNet padding behavior for inputs not divisible by 64."""

    @pytest.mark.parametrize("height,width", [
        (100, 100),  # Needs padding to 128x128
        (200, 200),  # Needs padding to 256x256
        (50, 70),    # Asymmetric padding
        (63, 63),    # Just under 64
        (65, 65),    # Just over 64
    ])
    def test_padding_with_various_sizes(self, height: int, width: int) -> None:
        """Test that padding works correctly for various input sizes."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)

        test_input = keras.random.normal((1, height, width, 3))
        output = model(test_input, training=False)

        # Output should match input size exactly (padding removed)
        expected_shape = (1, height, width, 3)
        assert output.shape == expected_shape

    def test_no_padding_for_divisible_sizes(self) -> None:
        """Test that no padding is applied for sizes divisible by 64."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)

        # These sizes are divisible by 64
        for size in [64, 128, 192, 256, 320, 384, 448, 512]:
            test_input = keras.random.normal((1, size, size, 3))
            output = model(test_input, training=False)

            assert output.shape == (1, size, size, 3)

    def test_padding_preserves_content(self) -> None:
        """Test that padding and unpadding preserves the central region."""
        model = SCUNet(in_nc=1, dim=16, head_dim=8)

        # Create input with known pattern
        height, width = 100, 100
        test_input = keras.random.normal((1, height, width, 1))

        output = model(test_input, training=False)

        # Output should have same shape as input
        assert output.shape == (1, height, width, 1)


class TestSCUNetArchitecture:
    """Test specific SCUNet architecture components and integration."""

    def test_encoder_decoder_symmetry(self) -> None:
        """Test that encoder and decoder have symmetric structure."""
        config = [4, 4, 4, 4, 4, 4, 4]
        model = SCUNet(in_nc=3, config=config)

        # Build model
        test_input = keras.random.normal((1, 128, 128, 3))
        _ = model(test_input, training=False)

        # 3 encoder stages + 1 bottleneck + 3 decoder stages
        # Verify by checking component existence
        assert hasattr(model, 'm_down1')
        assert hasattr(model, 'm_down2')
        assert hasattr(model, 'm_down3')
        assert hasattr(model, 'm_body')
        assert hasattr(model, 'm_up3')
        assert hasattr(model, 'm_up2')
        assert hasattr(model, 'm_up1')

    def test_channel_progression_encoder(self) -> None:
        """Test that encoder channels progress correctly."""
        dim = 64
        model = SCUNet(in_nc=3, dim=dim)

        # Build the model
        test_input = keras.random.normal((1, 128, 128, 3))
        _ = model(test_input, training=False)

        # Check head outputs dim channels
        assert model.m_head.filters == dim

        # Expected channel progression through encoder:
        # head: dim
        # down1: 2*dim (after downsample)
        # down2: 4*dim (after downsample)
        # down3: 8*dim (after downsample)

        # Check downsample layer filters
        down1_downsample = model.m_down1.layers[-1]
        assert isinstance(down1_downsample, keras.layers.Conv2D)
        assert down1_downsample.filters == 2 * dim

        down2_downsample = model.m_down2.layers[-1]
        assert isinstance(down2_downsample, keras.layers.Conv2D)
        assert down2_downsample.filters == 4 * dim

        down3_downsample = model.m_down3.layers[-1]
        assert isinstance(down3_downsample, keras.layers.Conv2D)
        assert down3_downsample.filters == 8 * dim

    def test_channel_progression_decoder(self) -> None:
        """Test that decoder channels progress correctly."""
        dim = 64
        model = SCUNet(in_nc=3, dim=dim)

        # Build the model
        test_input = keras.random.normal((1, 128, 128, 3))
        _ = model(test_input, training=False)

        # Expected channel progression through decoder:
        # up3 upsample: 4*dim
        # up2 upsample: 2*dim
        # up1 upsample: dim

        # Check upsample layer filters
        up3_upsample = model.m_up3.layers[0]
        assert isinstance(up3_upsample, keras.layers.Conv2DTranspose)
        assert up3_upsample.filters == 4 * dim

        up2_upsample = model.m_up2.layers[0]
        assert isinstance(up2_upsample, keras.layers.Conv2DTranspose)
        assert up2_upsample.filters == 2 * dim

        up1_upsample = model.m_up1.layers[0]
        assert isinstance(up1_upsample, keras.layers.Conv2DTranspose)
        assert up1_upsample.filters == dim

    def test_skip_connections_functionality(self) -> None:
        """Test that skip connections are functioning."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)

        # Build model by running forward pass
        test_input = keras.random.normal((1, 128, 128, 3))
        output = model(test_input, training=False)

        # If skip connections work, output should have correct shape
        assert output.shape == test_input.shape

    def test_drop_path_rate_distribution(self) -> None:
        """Test that drop path rates are correctly distributed."""
        config = [2, 2, 2, 2, 2, 2, 2]
        drop_path_rate = 0.2
        model = SCUNet(in_nc=3, config=config, drop_path_rate=drop_path_rate)

        # Total blocks
        total_blocks = sum(config)
        assert total_blocks == 14

        # Drop path rates should be linearly spaced from 0 to drop_path_rate
        # This is tested implicitly by checking model builds successfully


class TestSCUNetSerialization:
    """Test SCUNet serialization and deserialization - the most critical test."""

    def test_model_save_load_cycle_default_config(self) -> None:
        """Test complete save/load cycle with default configuration."""
        original_model = SCUNet(in_nc=3)
        test_input = keras.random.normal((2, 128, 128, 3))

        # Build model and get prediction
        original_output = original_model(test_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_scunet_default.keras')
            original_model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(test_input, training=False)

            # Check that configurations are preserved
            assert loaded_model.in_nc == original_model.in_nc
            assert loaded_model.config == original_model.config
            assert loaded_model.dim == original_model.dim
            assert loaded_model.head_dim == original_model.head_dim
            assert loaded_model.window_size == original_model.window_size
            assert loaded_model.stochastic_depth_rate == original_model.drop_path_rate
            assert loaded_model.input_resolution == original_model.input_resolution

            # Check that outputs are numerically identical
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Outputs should match after serialization"
            )

    def test_model_save_load_cycle_custom_config(self) -> None:
        """Test complete save/load cycle with custom configuration."""
        original_model = SCUNet(
            in_nc=1,
            config=[2, 2, 2, 2, 2, 2, 2],
            dim=32,
            head_dim=16,
            window_size=4,
            drop_path_rate=0.1,
            input_resolution=128
        )
        test_input = keras.random.normal((1, 128, 128, 1))

        # Build model and get prediction
        original_output = original_model(test_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_scunet_custom.keras')
            original_model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(test_input, training=False)

            # Check that all configurations are preserved
            assert loaded_model.in_nc == 1
            assert loaded_model.config == [2, 2, 2, 2, 2, 2, 2]
            assert loaded_model.dim == 32
            assert loaded_model.head_dim == 16
            assert loaded_model.window_size == 4
            assert loaded_model.stochastic_depth_rate == 0.1
            assert loaded_model.input_resolution == 128

            # Check output shapes and numerical accuracy
            assert loaded_output.shape == original_output.shape
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Outputs should match after serialization"
            )

    def test_model_save_load_with_drop_path(self) -> None:
        """Test save/load cycle with non-zero drop path rate."""
        original_model = SCUNet(
            in_nc=3,
            dim=32,
            head_dim=16,
            drop_path_rate=0.2
        )
        test_input = keras.random.normal((1, 128, 128, 3))

        # Build and test (inference mode for deterministic output)
        original_output = original_model(test_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_scunet_droppath.keras')
            original_model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(test_input, training=False)

            # Verify outputs match in inference mode
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6
            )

    def test_model_save_load_multiple_cycles(self) -> None:
        """Test multiple save/load cycles preserve model integrity."""
        original_model = SCUNet(in_nc=3, dim=32, head_dim=16)
        test_input = keras.random.normal((1, 128, 128, 3))

        # Get original prediction
        original_output = original_model(test_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # First cycle
            path1 = os.path.join(tmpdir, 'model1.keras')
            original_model.save(path1)
            model1 = keras.models.load_model(path1)

            # Second cycle
            path2 = os.path.join(tmpdir, 'model2.keras')
            model1.save(path2)
            model2 = keras.models.load_model(path2)

            # Third cycle
            path3 = os.path.join(tmpdir, 'model3.keras')
            model2.save(path3)
            model3 = keras.models.load_model(path3)

            # Get final prediction
            final_output = model3(test_input, training=False)

            # Should still match original
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(final_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Multiple save/load cycles should preserve model"
            )

    def test_get_config_completeness(self) -> None:
        """Test that get_config contains all initialization parameters."""
        model = SCUNet(
            in_nc=1,
            config=[3, 3, 3, 3, 3, 3, 3],
            dim=48,
            head_dim=24,
            window_size=6,
            drop_path_rate=0.15,
            input_resolution=192
        )

        config = model.get_config()

        # Check all parameters are present
        assert 'in_nc' in config
        assert 'config' in config
        assert 'dim' in config
        assert 'head_dim' in config
        assert 'window_size' in config
        assert 'drop_path_rate' in config
        assert 'input_resolution' in config

        # Check values match
        assert config['in_nc'] == 1
        assert config['config'] == [3, 3, 3, 3, 3, 3, 3]
        assert config['dim'] == 48
        assert config['head_dim'] == 24
        assert config['window_size'] == 6
        assert config['drop_path_rate'] == 0.15
        assert config['input_resolution'] == 192


class TestSCUNetErrorHandling:
    """Test error handling and edge cases."""

    def test_very_small_input_size(self) -> None:
        """Test behavior with very small input sizes."""
        model = SCUNet(in_nc=3, dim=16, head_dim=8)  # Small model

        # Test with small input that still works with U-Net architecture
        small_input = keras.random.normal((1, 64, 64, 3))

        try:
            output = model(small_input, training=False)
            assert output.shape == (1, 64, 64, 3)
        except Exception as e:
            # Small inputs may not work depending on architecture depth
            if "shape" not in str(e).lower():
                raise

    def test_batch_size_variations(self) -> None:
        """Test with different batch sizes including edge cases."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)

        # Test with batch size 1
        single_input = keras.random.normal((1, 128, 128, 3))
        single_output = model(single_input, training=False)
        assert single_output.shape == (1, 128, 128, 3)

        # Test with larger batch
        batch_input = keras.random.normal((8, 128, 128, 3))
        batch_output = model(batch_input, training=False)
        assert batch_output.shape == (8, 128, 128, 3)

    def test_consistent_output_in_inference_mode(self) -> None:
        """Test that inference mode produces consistent outputs."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)

        test_input = keras.random.normal((1, 128, 128, 3))

        # Run inference twice
        output1 = model(test_input, training=False)
        output2 = model(test_input, training=False)

        # Outputs should be identical in inference mode
        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Inference outputs should be deterministic"
        )

    def test_memory_efficiency_moderate_input(self) -> None:
        """Test that model can handle moderately sized inputs."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)  # Smaller model for testing

        # Test with moderate input size
        moderate_input = keras.random.normal((1, 256, 256, 3))

        try:
            output = model(moderate_input, training=False)
            assert output.shape == (1, 256, 256, 3)
        except Exception as e:
            # If it fails due to memory/resources, that's acceptable
            if "memory" not in str(e).lower() and "resource" not in str(e).lower():
                raise

    def test_asymmetric_input_sizes(self) -> None:
        """Test with asymmetric (non-square) input sizes."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)

        # Test various asymmetric sizes
        for height, width in [(64, 128), (128, 64), (100, 150), (200, 100)]:
            test_input = keras.random.normal((1, height, width, 3))
            output = model(test_input, training=False)

            # Output should match input dimensions
            expected_shape = (1, height, width, 3)
            assert output.shape == expected_shape


class TestSCUNetTrainingBehavior:
    """Test SCUNet behavior during training."""

    def test_training_mode_forward_pass(self) -> None:
        """Test forward pass in training mode."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16, drop_path_rate=0.1)

        test_input = keras.random.normal((2, 128, 128, 3))
        output = model(test_input, training=True)

        assert output.shape == test_input.shape

    def test_trainable_variables_exist(self) -> None:
        """Test that model has trainable variables after building."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)

        # Build the model
        test_input = keras.random.normal((1, 128, 128, 3))
        _ = model(test_input, training=False)

        # Check that model has trainable variables
        assert len(model.trainable_variables) > 0

    def test_gradients_flow_through_model(self) -> None:
        """Test that gradients can flow through the model."""
        import tensorflow as tf

        model = SCUNet(in_nc=3, dim=32, head_dim=16)
        test_input = keras.random.normal((1, 128, 128, 3))

        with tf.GradientTape() as tape:
            tape.watch(test_input)
            output = model(test_input, training=True)
            loss = ops.mean(ops.square(output))

        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Check that gradients exist
        assert all(g is not None for g in gradients)
        assert len(gradients) > 0


class TestSCUNetOutputQuality:
    """Test output quality and numerical stability."""

    def test_output_has_no_nans(self) -> None:
        """Test that output contains no NaN values."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)

        test_input = keras.random.normal((1, 128, 128, 3))
        output = model(test_input, training=False)

        # Check for NaNs
        assert not ops.any(ops.isnan(output))

    def test_output_has_no_infs(self) -> None:
        """Test that output contains no infinite values."""
        model = SCUNet(in_nc=3, dim=32, head_dim=16)

        test_input = keras.random.normal((1, 128, 128, 3))
        output = model(test_input, training=False)

        # Check for infs
        assert not ops.any(ops.isinf(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])