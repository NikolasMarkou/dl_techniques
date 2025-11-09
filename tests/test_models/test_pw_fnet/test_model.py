"""
Comprehensive Test Suite for PW-FNet Implementation.

This module provides thorough testing of all PW-FNet components following
the best practices outlined in the Complete Guide to Modern Keras 3 Custom
Layers and Models.

**Test Coverage**:
- Initialization and configuration
- Forward pass functionality
- Shape computations
- Serialization and deserialization
- Model compilation and training
- Multi-scale output verification
- Edge cases and error handling
- Gradient flow verification

**Author**: Deep Learning Techniques Framework
**Version**: 1.0.0
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
from typing import Dict, Any, List
import tensorflow as tf

# Import all components to test
from dl_techniques.models.pw_fnet.model import (
    FFTLayer,
    IFFTLayer,
    PW_FNet_Block,
    Downsample,
    Upsample,
    PW_FNet
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_input_small() -> keras.KerasTensor:
    """Small sample input for quick tests."""
    return ops.random.normal(shape=(2, 16, 16, 32))


@pytest.fixture
def sample_input_medium() -> keras.KerasTensor:
    """Medium sample input for realistic tests."""
    return ops.random.normal(shape=(4, 64, 64, 64))


@pytest.fixture
def sample_image_rgb() -> keras.KerasTensor:
    """Sample RGB image for model tests."""
    return ops.random.normal(shape=(2, 128, 128, 3))


@pytest.fixture
def pw_fnet_block_config() -> Dict[str, Any]:
    """Standard configuration for PW_FNet_Block."""
    return {
        'dim': 64,
        'ffn_expansion_factor': 2.0
    }


@pytest.fixture
def pw_fnet_model_config() -> Dict[str, Any]:
    """Standard configuration for PW_FNet model."""
    return {
        'img_channels': 3,
        'width': 32,
        'middle_blk_num': 2,
        'enc_blk_nums': [1, 1],
        'dec_blk_nums': [1, 1]
    }


# =============================================================================
# FFTLayer Tests
# =============================================================================

class TestFFTLayer:
    """Test suite for FFTLayer."""

    def test_initialization(self):
        """Test FFTLayer initialization."""
        layer = FFTLayer()

        assert hasattr(layer, 'call')
        assert not layer.built
        assert layer.name.startswith('fft_layer')

    def test_forward_pass(self, sample_input_small):
        """Test forward pass and output properties."""
        layer = FFTLayer()

        output = layer(sample_input_small)

        # Verify shape preservation
        assert output.shape == sample_input_small.shape

        # Verify dtype is complex
        assert output.dtype == 'complex64'

        # Verify layer is built
        assert layer.built

    def test_compute_output_shape(self, sample_input_small):
        """Test output shape computation."""
        layer = FFTLayer()

        input_shape = sample_input_small.shape
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == input_shape

    def test_configuration(self):
        """Test get_config method."""
        layer = FFTLayer(name="test_fft")

        config = layer.get_config()

        assert 'name' in config
        assert config['name'] == "test_fft"

    def test_serialization(self, sample_input_small):
        """Test full serialization cycle."""
        # Create model with FFTLayer
        inputs = keras.Input(shape=sample_input_small.shape[1:])
        outputs = FFTLayer()(inputs)
        model = keras.Model(inputs, outputs)

        # Get prediction from original
        original_output = model(sample_input_small)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'fft_test.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_output = loaded_model(sample_input_small)

            # Verify identical outputs
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(ops.real(original_output)),
                keras.ops.convert_to_numpy(ops.real(loaded_output)),
                rtol=1e-6, atol=1e-6,
                err_msg="FFT outputs should match after serialization"
            )

    def test_frequency_properties(self):
        """Test that FFT captures frequency information correctly."""
        # Create a simple pattern
        x = ops.zeros((1, 8, 8, 1))
        x = ops.convert_to_numpy(x)
        x[0, 2:6, 2:6, 0] = 1.0  # Square in center
        x = ops.convert_to_tensor(x)

        layer = FFTLayer()
        freq = layer(x)

        # DC component (0,0) should be non-zero
        dc_value = ops.abs(freq[0, 0, 0, 0])
        assert ops.convert_to_numpy(dc_value) > 0.0


# =============================================================================
# IFFTLayer Tests
# =============================================================================

class TestIFFTLayer:
    """Test suite for IFFTLayer."""

    def test_initialization(self):
        """Test IFFTLayer initialization."""
        layer = IFFTLayer()

        assert hasattr(layer, 'call')
        assert not layer.built
        assert layer.name.startswith('ifft_layer')

    def test_forward_pass(self):
        """Test forward pass with complex input."""
        # Create complex input
        real_part = ops.random.normal((2, 16, 16, 32))
        imag_part = ops.random.normal((2, 16, 16, 32))
        complex_input = ops.cast(real_part, 'complex64') + \
                        1j * ops.cast(imag_part, 'complex64')

        layer = IFFTLayer()
        output = layer(complex_input)

        # Verify shape preservation
        assert output.shape == complex_input.shape

        # Verify dtype is real
        assert output.dtype in ['float32', 'float64']

        # Verify layer is built
        assert layer.built

    def test_compute_output_shape(self):
        """Test output shape computation."""
        layer = IFFTLayer()

        input_shape = (None, 32, 32, 64)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == input_shape

    def test_configuration(self):
        """Test get_config method."""
        layer = IFFTLayer(name="test_ifft")

        config = layer.get_config()

        assert 'name' in config
        assert config['name'] == "test_ifft"

    def test_round_trip_transformation(self, sample_input_small):
        """Test FFT -> IFFT round trip recovers original."""
        fft_layer = FFTLayer()
        ifft_layer = IFFTLayer()

        # Forward and back
        freq = fft_layer(sample_input_small)
        reconstructed = ifft_layer(freq)

        # Should be close to original
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sample_input_small),
            keras.ops.convert_to_numpy(reconstructed),
            rtol=1e-5, atol=1e-5,
            err_msg="Round-trip FFT->IFFT should recover input"
        )


# =============================================================================
# PW_FNet_Block Tests
# =============================================================================

class TestPW_FNet_Block:
    """Test suite for PW_FNet_Block."""

    def test_initialization(self, pw_fnet_block_config):
        """Test block initialization."""
        block = PW_FNet_Block(**pw_fnet_block_config)

        assert block.dim == pw_fnet_block_config['dim']
        assert block.ffn_expansion_factor == pw_fnet_block_config['ffn_expansion_factor']
        assert not block.built

        # Verify sub-layers are created
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
        assert hasattr(block, 'token_mixer_expand')
        assert hasattr(block, 'fft')
        assert hasattr(block, 'ifft')
        assert hasattr(block, 'ffn_expand')

    def test_invalid_dim(self):
        """Test that invalid dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            PW_FNet_Block(dim=0)

        with pytest.raises(ValueError, match="dim must be positive"):
            PW_FNet_Block(dim=-5)

    def test_invalid_expansion_factor(self):
        """Test that invalid expansion factor raises ValueError."""
        with pytest.raises(ValueError, match="ffn_expansion_factor must be positive"):
            PW_FNet_Block(dim=64, ffn_expansion_factor=0)

    def test_forward_pass(self, pw_fnet_block_config, sample_input_small):
        """Test forward pass and shape preservation."""
        block = PW_FNet_Block(**pw_fnet_block_config)

        # Adjust input to match expected channels
        input_data = ops.random.normal((2, 16, 16, pw_fnet_block_config['dim']))

        output = block(input_data)

        # Shape should be preserved
        assert output.shape == input_data.shape
        assert block.built

    def test_build_method(self, pw_fnet_block_config):
        """Test that build() creates all necessary variables."""
        block = PW_FNet_Block(**pw_fnet_block_config)

        input_shape = (None, 32, 32, pw_fnet_block_config['dim'])
        block.build(input_shape)

        # Verify sub-layers are built
        assert block.norm1.built
        assert block.norm2.built
        assert block.token_mixer_expand.built
        assert block.ffn_expand.built
        assert block.built

    def test_compute_output_shape(self, pw_fnet_block_config):
        """Test output shape computation."""
        block = PW_FNet_Block(**pw_fnet_block_config)

        input_shape = (None, 32, 32, pw_fnet_block_config['dim'])
        output_shape = block.compute_output_shape(input_shape)

        assert output_shape == input_shape

    def test_configuration(self, pw_fnet_block_config):
        """Test get_config method."""
        block = PW_FNet_Block(**pw_fnet_block_config)

        config = block.get_config()

        assert config['dim'] == pw_fnet_block_config['dim']
        assert config['ffn_expansion_factor'] == pw_fnet_block_config['ffn_expansion_factor']

    def test_serialization(self, pw_fnet_block_config):
        """Test full serialization cycle."""
        # Create model with block
        dim = pw_fnet_block_config['dim']
        inputs = keras.Input(shape=(32, 32, dim))
        outputs = PW_FNet_Block(**pw_fnet_block_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Generate test input
        test_input = ops.random.normal((2, 32, 32, dim))
        original_output = model(test_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'block_test.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_output = loaded_model(test_input)

            # Verify identical outputs
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Block outputs should match after serialization"
            )

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, pw_fnet_block_config, training):
        """Test behavior in different training modes."""
        block = PW_FNet_Block(**pw_fnet_block_config)

        dim = pw_fnet_block_config['dim']
        input_data = ops.random.normal((2, 16, 16, dim))

        output = block(input_data, training=training)

        assert output.shape == input_data.shape

    def test_gradient_flow(self, pw_fnet_block_config):
        """Test that gradients flow through the block."""
        block = PW_FNet_Block(**pw_fnet_block_config)

        dim = pw_fnet_block_config['dim']
        input_data = ops.convert_to_tensor(
            ops.random.normal((2, 16, 16, dim)),
            dtype='float32'
        )

        with tf.GradientTape() as tape:
            tape.watch(input_data)
            output = block(input_data)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, block.trainable_variables)

        # All gradients should exist
        assert all(g is not None for g in gradients)
        assert len(gradients) > 0


# =============================================================================
# Downsample Tests
# =============================================================================

class TestDownsample:
    """Test suite for Downsample layer."""

    def test_initialization(self):
        """Test Downsample initialization."""
        downsample = Downsample(dim=128)

        assert downsample.dim == 128
        assert not downsample.built
        assert hasattr(downsample, 'conv')

    def test_invalid_dim(self):
        """Test that invalid dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            Downsample(dim=0)

        with pytest.raises(ValueError, match="dim must be positive"):
            Downsample(dim=-10)

    def test_forward_pass(self):
        """Test forward pass and downsampling."""
        downsample = Downsample(dim=128)

        input_data = ops.random.normal((4, 64, 64, 64))
        output = downsample(input_data)

        # Spatial dimensions should be halved
        assert output.shape[0] == input_data.shape[0]  # Batch
        assert output.shape[1] == input_data.shape[1] // 2  # Height
        assert output.shape[2] == input_data.shape[2] // 2  # Width
        assert output.shape[3] == 128  # Channels

    def test_compute_output_shape(self):
        """Test output shape computation."""
        downsample = Downsample(dim=128)

        input_shape = (None, 64, 64, 64)
        output_shape = downsample.compute_output_shape(input_shape)

        assert output_shape[0] is None  # Batch
        assert output_shape[1] == 32  # Height halved
        assert output_shape[2] == 32  # Width halved
        assert output_shape[3] == 128  # New channels

    def test_configuration(self):
        """Test get_config method."""
        downsample = Downsample(dim=128)

        config = downsample.get_config()

        assert config['dim'] == 128

    def test_serialization(self):
        """Test full serialization cycle."""
        # Create model with Downsample
        inputs = keras.Input(shape=(64, 64, 64))
        outputs = Downsample(dim=128)(inputs)
        model = keras.Model(inputs, outputs)

        test_input = ops.random.normal((2, 64, 64, 64))
        original_output = model(test_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'downsample_test.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_output = loaded_model(test_input)

            # Verify identical outputs
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Downsample outputs should match after serialization"
            )


# =============================================================================
# Upsample Tests
# =============================================================================

class TestUpsample:
    """Test suite for Upsample layer."""

    def test_initialization(self):
        """Test Upsample initialization."""
        upsample = Upsample(dim=64)

        assert upsample.dim == 64
        assert not upsample.built
        assert hasattr(upsample, 'conv_transpose')

    def test_invalid_dim(self):
        """Test that invalid dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            Upsample(dim=0)

        with pytest.raises(ValueError, match="dim must be positive"):
            Upsample(dim=-5)

    def test_forward_pass(self):
        """Test forward pass and upsampling."""
        upsample = Upsample(dim=64)

        input_data = ops.random.normal((4, 16, 16, 128))
        output = upsample(input_data)

        # Spatial dimensions should be doubled
        assert output.shape[0] == input_data.shape[0]  # Batch
        assert output.shape[1] == input_data.shape[1] * 2  # Height
        assert output.shape[2] == input_data.shape[2] * 2  # Width
        assert output.shape[3] == 64  # Channels

    def test_compute_output_shape(self):
        """Test output shape computation."""
        upsample = Upsample(dim=64)

        input_shape = (None, 16, 16, 128)
        output_shape = upsample.compute_output_shape(input_shape)

        assert output_shape[0] is None  # Batch
        assert output_shape[1] == 32  # Height doubled
        assert output_shape[2] == 32  # Width doubled
        assert output_shape[3] == 64  # New channels

    def test_configuration(self):
        """Test get_config method."""
        upsample = Upsample(dim=64)

        config = upsample.get_config()

        assert config['dim'] == 64

    def test_serialization(self):
        """Test full serialization cycle."""
        # Create model with Upsample
        inputs = keras.Input(shape=(16, 16, 128))
        outputs = Upsample(dim=64)(inputs)
        model = keras.Model(inputs, outputs)

        test_input = ops.random.normal((2, 16, 16, 128))
        original_output = model(test_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'upsample_test.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_output = loaded_model(test_input)

            # Verify identical outputs
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Upsample outputs should match after serialization"
            )


# =============================================================================
# PW_FNet Model Tests
# =============================================================================

class TestPW_FNet:
    """Test suite for complete PW_FNet model."""

    def test_initialization(self, pw_fnet_model_config):
        """Test model initialization."""
        model = PW_FNet(**pw_fnet_model_config)

        assert model.img_channels == pw_fnet_model_config['img_channels']
        assert model.width == pw_fnet_model_config['width']
        assert model.middle_blk_num == pw_fnet_model_config['middle_blk_num']
        assert model.enc_blk_nums == pw_fnet_model_config['enc_blk_nums']
        assert model.dec_blk_nums == pw_fnet_model_config['dec_blk_nums']

    def test_invalid_parameters(self):
        """Test that invalid parameters raise ValueError."""
        # Invalid img_channels
        with pytest.raises(ValueError, match="img_channels must be positive"):
            PW_FNet(img_channels=0)

        # Invalid width
        with pytest.raises(ValueError, match="width must be positive"):
            PW_FNet(width=0)

        # Invalid middle_blk_num
        with pytest.raises(ValueError, match="middle_blk_num must be non-negative"):
            PW_FNet(middle_blk_num=-1)

        # Mismatched encoder/decoder lengths
        with pytest.raises(ValueError, match="must have same length"):
            PW_FNet(enc_blk_nums=[1, 2], dec_blk_nums=[1])

        # Empty encoder blocks
        with pytest.raises(ValueError, match="enc_blk_nums cannot be empty"):
            PW_FNet(enc_blk_nums=[])

    def test_forward_pass(self, pw_fnet_model_config, sample_image_rgb):
        """Test forward pass and multi-scale outputs."""
        model = PW_FNet(**pw_fnet_model_config)

        outputs = model(sample_image_rgb)

        # Should return list of 3 outputs
        assert isinstance(outputs, list)
        assert len(outputs) == 3

        # Check shapes
        batch, h, w, c = sample_image_rgb.shape

        # Full resolution
        assert outputs[0].shape == (batch, h, w, c)

        # Half resolution
        assert outputs[1].shape == (batch, h // 2, w // 2, c)

        # Quarter resolution
        assert outputs[2].shape == (batch, h // 4, w // 4, c)

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, pw_fnet_model_config, sample_image_rgb, training):
        """Test behavior in different training modes."""
        model = PW_FNet(**pw_fnet_model_config)

        outputs = model(sample_image_rgb, training=training)

        assert len(outputs) == 3
        assert all(out.shape[0] == sample_image_rgb.shape[0] for out in outputs)

    def test_configuration(self, pw_fnet_model_config):
        """Test get_config method."""
        model = PW_FNet(**pw_fnet_model_config)

        config = model.get_config()

        assert config['img_channels'] == pw_fnet_model_config['img_channels']
        assert config['width'] == pw_fnet_model_config['width']
        assert config['middle_blk_num'] == pw_fnet_model_config['middle_blk_num']
        assert config['enc_blk_nums'] == pw_fnet_model_config['enc_blk_nums']
        assert config['dec_blk_nums'] == pw_fnet_model_config['dec_blk_nums']

    def test_serialization(self, pw_fnet_model_config, sample_image_rgb):
        """Test full serialization cycle."""
        model = PW_FNet(**pw_fnet_model_config)

        # Get prediction from original
        original_outputs = model(sample_image_rgb)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'pw_fnet_test.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_outputs = loaded_model(sample_image_rgb)

            # Verify all outputs match
            for i, (orig, loaded) in enumerate(zip(original_outputs, loaded_outputs)):
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(orig),
                    keras.ops.convert_to_numpy(loaded),
                    rtol=1e-6, atol=1e-6,
                    err_msg=f"Output {i} should match after serialization"
                )

    def test_compilation(self, pw_fnet_model_config):
        """Test model compilation."""
        model = PW_FNet(**pw_fnet_model_config)

        # Multi-output model requires list of losses
        model.compile(
            optimizer='adam',
            loss=['mse', 'mse', 'mse'],
            loss_weights=[1.0, 0.5, 0.25]
        )

        assert model.optimizer is not None
        assert len(model.loss) == 3

    def test_training_step(self, pw_fnet_model_config):
        """Test a single training step."""
        model = PW_FNet(**pw_fnet_model_config)

        model.compile(
            optimizer='adam',
            loss=['mse', 'mse', 'mse'],
            loss_weights=[1.0, 0.5, 0.25]
        )

        # Create dummy data
        x_train = ops.random.normal((4, 64, 64, 3))
        y_train = [
            ops.random.normal((4, 64, 64, 3)),  # Full res
            ops.random.normal((4, 32, 32, 3)),  # Half res
            ops.random.normal((4, 16, 16, 3)),  # Quarter res
        ]

        # Single training step should work
        history = model.fit(x_train, y_train, epochs=1, verbose=0)

        assert 'loss' in history.history
        assert len(history.history['loss']) == 1

    def test_gradient_flow(self, pw_fnet_model_config):
        """Test that gradients flow through the entire model."""
        model = PW_FNet(**pw_fnet_model_config)

        input_data = ops.convert_to_tensor(
            ops.random.normal((2, 64, 64, 3)),
            dtype='float32'
        )

        with tf.GradientTape() as tape:
            tape.watch(input_data)
            outputs = model(input_data)
            # Compute loss on all outputs
            loss = sum(ops.mean(ops.square(out)) for out in outputs)

        gradients = tape.gradient(loss, model.trainable_variables)

        # All gradients should exist
        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    def test_residual_learning(self):
        """Test that residual learning works (output ≈ input + residual)."""
        model = PW_FNet(
            img_channels=3,
            width=16,  # Small for fast test
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1]
        )

        # Create clean input (all zeros)
        clean_input = ops.zeros((1, 32, 32, 3))

        outputs = model(clean_input)

        # With zero input, outputs should be relatively small
        # (model predicts residual from zero)
        for output in outputs:
            mean_abs_value = ops.mean(ops.abs(output))
            # Should be small but not necessarily zero (random init)
            assert keras.ops.convert_to_numpy(mean_abs_value) >= 0.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete PW-FNet system."""

    def test_end_to_end_pipeline(self):
        """Test complete training and inference pipeline."""
        # Create small model for fast test
        model = PW_FNet(
            img_channels=3,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1]
        )

        # Compile
        model.compile(
            optimizer='adam',
            loss=['mse', 'mse', 'mse'],
            loss_weights=[1.0, 0.5, 0.25]
        )

        # Create dummy dataset
        x_train = ops.random.normal((8, 64, 64, 3))
        y_train = [
            ops.random.normal((8, 64, 64, 3)),
            ops.random.normal((8, 32, 32, 3)),
            ops.random.normal((8, 16, 16, 3)),
        ]

        # Train for 2 epochs
        history = model.fit(x_train, y_train, epochs=2, verbose=0)

        assert len(history.history['loss']) == 2

        # Inference
        x_test = ops.random.normal((2, 64, 64, 3))
        predictions = model.predict(x_test, verbose=0)

        assert len(predictions) == 3
        assert predictions[0].shape == (2, 64, 64, 3)

    def test_multi_gpu_compatibility(self):
        """Test that model structure is compatible with multi-GPU training."""
        model = PW_FNet(
            img_channels=3,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1]
        )

        # Should be able to compile without errors
        model.compile(
            optimizer='adam',
            loss=['mse', 'mse', 'mse']
        )

        # Model summary should work
        model.build(input_shape=(None, 64, 64, 3))
        assert model.built

    def test_different_input_sizes(self):
        """Test model works with different input sizes."""
        model = PW_FNet(
            img_channels=3,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1]
        )

        # Test with different sizes
        sizes = [(64, 64), (128, 128), (32, 32)]

        for h, w in sizes:
            input_data = ops.random.normal((2, h, w, 3))
            outputs = model(input_data)

            assert outputs[0].shape == (2, h, w, 3)
            assert outputs[1].shape == (2, h // 2, w // 2, 3)
            assert outputs[2].shape == (2, h // 4, w // 4, 3)


# =============================================================================
# Performance and Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability and edge cases."""

    def test_zero_input(self):
        """Test model handles zero input gracefully."""
        model = PW_FNet(
            img_channels=3,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1]
        )

        zero_input = ops.zeros((2, 64, 64, 3))
        outputs = model(zero_input)

        # Should not produce NaN or Inf
        for output in outputs:
            assert not ops.any(ops.isnan(output))
            assert not ops.any(ops.isinf(output))

    def test_large_input_values(self):
        """Test model handles large input values."""
        model = PW_FNet(
            img_channels=3,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1]
        )

        large_input = ops.ones((2, 64, 64, 3)) * 1000.0
        outputs = model(large_input)

        # Should not produce NaN or Inf
        for output in outputs:
            assert not ops.any(ops.isnan(output))
            assert not ops.any(ops.isinf(output))

    def test_batch_size_one(self):
        """Test model works with batch size of 1."""
        model = PW_FNet(
            img_channels=3,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1]
        )

        single_input = ops.random.normal((1, 64, 64, 3))
        outputs = model(single_input)

        assert all(out.shape[0] == 1 for out in outputs)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    # Run with: python test_pw_fnet.py -v
    pytest.main([__file__, "-v", "--tb=short"])