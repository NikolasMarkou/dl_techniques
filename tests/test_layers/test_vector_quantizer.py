"""
Comprehensive test suite for VQ-VAE implementation.

This module provides thorough testing of the VectorQuantizer layer

Tests cover:
- Initialization and configuration
- Forward pass and building
- Serialization/deserialization cycles
- Gradient flow
- Training modes
- Edge cases and error conditions
- EMA updates
- Codebook operations
- Model training and evaluation

"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Any


from dl_techniques.layers.vector_quantizer import VectorQuantizer


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def quantizer_config() -> Dict[str, Any]:
    """Standard configuration for VectorQuantizer testing."""
    return {
        'num_embeddings': 64,
        'embedding_dim': 32,
        'commitment_cost': 0.25,
        'initializer': 'uniform',
        'use_ema': False,
        'ema_decay': 0.99,
        'epsilon': 1e-5,
    }


@pytest.fixture
def quantizer_ema_config() -> Dict[str, Any]:
    """Configuration for VectorQuantizer with EMA."""
    return {
        'num_embeddings': 64,
        'embedding_dim': 32,
        'commitment_cost': 0.25,
        'initializer': 'uniform',
        'use_ema': True,
        'ema_decay': 0.99,
        'epsilon': 1e-5,
    }


@pytest.fixture
def sample_input_2d() -> keras.KerasTensor:
    """Sample 2D input for testing (batch, height, width, channels)."""
    return ops.cast(
        keras.random.normal(shape=(4, 8, 8, 32)),
        dtype='float32'
    )


@pytest.fixture
def sample_input_1d() -> keras.KerasTensor:
    """Sample 1D input for testing (batch, sequence_length, channels)."""
    return ops.cast(
        keras.random.normal(shape=(4, 16, 32)),
        dtype='float32'
    )


@pytest.fixture
def simple_encoder() -> keras.Model:
    """Simple encoder for VQVAEModel testing."""
    return keras.Sequential([
        keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'),
        keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        keras.layers.Conv2D(32, 3, padding='same'),  # Output embedding_dim=32
    ], name='encoder')


@pytest.fixture
def simple_decoder() -> keras.Model:
    """Simple decoder for VQVAEModel testing."""
    return keras.Sequential([
        keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
        keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
        keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid'),
    ], name='decoder')


@pytest.fixture
def vqvae_config(simple_encoder, simple_decoder) -> Dict[str, Any]:
    """Configuration for VQVAEModel testing."""
    return {
        'encoder': simple_encoder,
        'decoder': simple_decoder,
        'num_embeddings': 128,
        'embedding_dim': 32,
        'commitment_cost': 0.25,
        'use_ema': False,
        'ema_decay': 0.99,
        'reconstruction_loss_weight': 1.0,
        'quantizer_initializer': 'uniform',
    }


@pytest.fixture
def sample_images() -> keras.KerasTensor:
    """Sample images for VQVAEModel testing."""
    return ops.cast(
        keras.random.uniform(shape=(8, 32, 32, 3), minval=0.0, maxval=1.0),
        dtype='float32'
    )


# ============================================================================
# VectorQuantizer Layer Tests
# ============================================================================

class TestVectorQuantizer:
    """Comprehensive test suite for VectorQuantizer layer."""

    def test_initialization(self, quantizer_config):
        """Test layer initialization with valid configuration."""
        layer = VectorQuantizer(**quantizer_config)

        # Check attributes are stored
        assert layer.num_embeddings == quantizer_config['num_embeddings']
        assert layer.embedding_dim == quantizer_config['embedding_dim']
        assert layer.commitment_cost == quantizer_config['commitment_cost']
        assert layer.use_ema == quantizer_config['use_ema']
        assert layer.ema_decay == quantizer_config['ema_decay']
        assert layer.epsilon == quantizer_config['epsilon']

        # Layer should not be built yet
        assert not layer.built
        assert layer.embeddings is None

    def test_initialization_with_ema(self, quantizer_ema_config):
        """Test layer initialization with EMA enabled."""
        layer = VectorQuantizer(**quantizer_ema_config)

        assert layer.use_ema is True
        assert layer.ema_decay == quantizer_ema_config['ema_decay']
        assert not layer.built

    def test_initialization_invalid_num_embeddings(self):
        """Test initialization with invalid num_embeddings."""
        with pytest.raises(ValueError, match="num_embeddings must be positive"):
            VectorQuantizer(
                num_embeddings=0,
                embedding_dim=32,
            )

        with pytest.raises(ValueError, match="num_embeddings must be positive"):
            VectorQuantizer(
                num_embeddings=-10,
                embedding_dim=32,
            )

    def test_initialization_invalid_embedding_dim(self):
        """Test initialization with invalid embedding_dim."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            VectorQuantizer(
                num_embeddings=64,
                embedding_dim=0,
            )

        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            VectorQuantizer(
                num_embeddings=64,
                embedding_dim=-5,
            )

    def test_initialization_invalid_commitment_cost(self):
        """Test initialization with invalid commitment_cost."""
        with pytest.raises(ValueError, match="commitment_cost must be non-negative"):
            VectorQuantizer(
                num_embeddings=64,
                embedding_dim=32,
                commitment_cost=-0.5,
            )

    def test_initialization_invalid_ema_decay(self):
        """Test initialization with invalid ema_decay when using EMA."""
        with pytest.raises(ValueError, match="ema_decay must be in"):
            VectorQuantizer(
                num_embeddings=64,
                embedding_dim=32,
                use_ema=True,
                ema_decay=0.0,
            )

        with pytest.raises(ValueError, match="ema_decay must be in"):
            VectorQuantizer(
                num_embeddings=64,
                embedding_dim=32,
                use_ema=True,
                ema_decay=1.0,
            )

    def test_build(self, quantizer_config, sample_input_2d):
        """Test layer building creates proper weights."""
        layer = VectorQuantizer(**quantizer_config)

        # Build the layer
        layer.build(sample_input_2d.shape)

        # Check layer is built
        assert layer.built

        # Check embeddings are created
        assert layer.embeddings is not None
        assert layer.embeddings.shape == (
            quantizer_config['num_embeddings'],
            quantizer_config['embedding_dim']
        )

        # Check that embeddings are trainable when not using EMA
        assert layer.embeddings.trainable is True

        # EMA variables should not exist
        assert layer.ema_cluster_size is None
        assert layer.ema_embeddings is None

    def test_build_with_ema(self, quantizer_ema_config, sample_input_2d):
        """Test layer building with EMA creates proper weights."""
        layer = VectorQuantizer(**quantizer_ema_config)

        # Build the layer
        layer.build(sample_input_2d.shape)

        # Check layer is built
        assert layer.built

        # Check embeddings are created and not trainable
        assert layer.embeddings is not None
        assert layer.embeddings.trainable is False

        # Check EMA variables are created
        assert layer.ema_cluster_size is not None
        assert layer.ema_embeddings is not None

        assert layer.ema_cluster_size.shape == (quantizer_ema_config['num_embeddings'],)
        assert layer.ema_embeddings.shape == (
            quantizer_ema_config['num_embeddings'],
            quantizer_ema_config['embedding_dim']
        )

        # EMA variables should not be trainable
        assert layer.ema_cluster_size.trainable is False
        assert layer.ema_embeddings.trainable is False

    def test_build_invalid_input_shape(self, quantizer_config):
        """Test build with mismatched embedding dimension."""
        layer = VectorQuantizer(**quantizer_config)

        # Input with wrong embedding dimension (64 instead of 32)
        invalid_shape = (4, 8, 8, 64)

        with pytest.raises(ValueError, match="must match embedding_dim"):
            layer.build(invalid_shape)

    def test_forward_pass_2d(self, quantizer_config, sample_input_2d):
        """Test forward pass with 2D spatial input."""
        layer = VectorQuantizer(**quantizer_config)

        output = layer(sample_input_2d)

        # Check layer is built
        assert layer.built

        # Check output shape matches input shape
        assert output.shape == sample_input_2d.shape

        # Check output is a tensor
        assert ops.is_tensor(output)

        # Check that losses were added
        assert len(layer.losses) == 2  # codebook + commitment

    def test_forward_pass_1d(self, quantizer_config, sample_input_1d):
        """Test forward pass with 1D sequential input."""
        layer = VectorQuantizer(**quantizer_config)

        output = layer(sample_input_1d)

        # Check output shape matches input shape
        assert output.shape == sample_input_1d.shape

        # Check that losses were added
        assert len(layer.losses) == 2

    def test_forward_pass_training_false(self, quantizer_config, sample_input_2d):
        """Test forward pass with training=False."""
        layer = VectorQuantizer(**quantizer_config)

        output = layer(sample_input_2d, training=False)

        assert output.shape == sample_input_2d.shape

    def test_forward_pass_training_true(self, quantizer_config, sample_input_2d):
        """Test forward pass with training=True."""
        layer = VectorQuantizer(**quantizer_config)

        output = layer(sample_input_2d, training=True)

        assert output.shape == sample_input_2d.shape

    def test_losses_are_added(self, quantizer_config, sample_input_2d):
        """Test that quantization losses are added to the layer."""
        layer = VectorQuantizer(**quantizer_config)

        # First call
        _ = layer(sample_input_2d)

        # Should have 2 losses: codebook + commitment
        assert len(layer.losses) == 2

        # Losses should be tensors
        for loss in layer.losses:
            # Convert to numpy to check it's a scalar
            loss_value = ops.convert_to_numpy(loss)
            assert loss_value.shape == ()  # Scalar
            assert loss_value >= 0  # Non-negative

    def test_ema_updates_during_training(self, quantizer_ema_config, sample_input_2d):
        """Test that EMA variables are updated during training."""
        layer = VectorQuantizer(**quantizer_ema_config)

        # Initial call to build
        _ = layer(sample_input_2d, training=True)

        # Store initial EMA values
        initial_cluster_size = ops.convert_to_numpy(layer.ema_cluster_size).copy()
        initial_embeddings = ops.convert_to_numpy(layer.ema_embeddings).copy()

        # Second call with different data
        new_input = ops.cast(
            keras.random.normal(shape=(4, 8, 8, 32)),
            dtype='float32'
        )
        _ = layer(new_input, training=True)

        # EMA values should have changed
        new_cluster_size = ops.convert_to_numpy(layer.ema_cluster_size)
        new_embeddings = ops.convert_to_numpy(layer.ema_embeddings)

        # Check that at least some values changed
        assert not np.allclose(initial_cluster_size, new_cluster_size, rtol=1e-6)
        assert not np.allclose(initial_embeddings, new_embeddings, rtol=1e-6)

    def test_ema_no_updates_during_inference(self, quantizer_ema_config, sample_input_2d):
        """Test that EMA variables are not updated during inference."""
        layer = VectorQuantizer(**quantizer_ema_config)

        # Initial call to build
        _ = layer(sample_input_2d, training=True)

        # Store EMA values after initial training call
        initial_cluster_size = ops.convert_to_numpy(layer.ema_cluster_size).copy()
        initial_embeddings = ops.convert_to_numpy(layer.ema_embeddings).copy()

        # Call with training=False
        new_input = ops.cast(
            keras.random.normal(shape=(4, 8, 8, 32)),
            dtype='float32'
        )
        _ = layer(new_input, training=False)

        # EMA values should not have changed
        new_cluster_size = ops.convert_to_numpy(layer.ema_cluster_size)
        new_embeddings = ops.convert_to_numpy(layer.ema_embeddings)

        np.testing.assert_allclose(
            initial_cluster_size, new_cluster_size,
            rtol=1e-6, atol=1e-6,
            err_msg="EMA cluster size changed during inference"
        )
        np.testing.assert_allclose(
            initial_embeddings, new_embeddings,
            rtol=1e-6, atol=1e-6,
            err_msg="EMA embeddings changed during inference"
        )

    def test_get_codebook_indices(self, quantizer_config, sample_input_2d):
        """Test getting discrete codebook indices."""
        layer = VectorQuantizer(**quantizer_config)

        # Get indices
        indices = layer.get_codebook_indices(sample_input_2d)

        # Check shape (should remove last dimension)
        expected_shape = sample_input_2d.shape[:-1]
        assert indices.shape == expected_shape

        # Check indices are integers
        indices_np = ops.convert_to_numpy(indices)
        assert indices_np.dtype in [np.int32, np.int64]

        # Check indices are in valid range
        assert np.all(indices_np >= 0)
        assert np.all(indices_np < quantizer_config['num_embeddings'])

    def test_quantize_from_indices(self, quantizer_config, sample_input_2d):
        """Test converting indices back to embeddings."""
        layer = VectorQuantizer(**quantizer_config)

        # Build the layer first
        _ = layer(sample_input_2d)

        # Get indices
        indices = layer.get_codebook_indices(sample_input_2d)

        # Quantize from indices
        quantized = layer.quantize_from_indices(indices)

        # Check shape
        assert quantized.shape == sample_input_2d.shape

        # Check that it's the same as going through the full quantization
        quantized_direct = layer(sample_input_2d)

        np.testing.assert_allclose(
            ops.convert_to_numpy(quantized),
            ops.convert_to_numpy(quantized_direct),
            rtol=1e-5, atol=1e-5,
            err_msg="Quantized from indices should match direct quantization"
        )

    def test_indices_roundtrip(self, quantizer_config, sample_input_2d):
        """Test that indices -> quantize -> indices gives same result."""
        layer = VectorQuantizer(**quantizer_config)

        # Get indices from input
        indices_1 = layer.get_codebook_indices(sample_input_2d)

        # Convert to embeddings
        quantized = layer.quantize_from_indices(indices_1)

        # Get indices again
        indices_2 = layer.get_codebook_indices(quantized)

        # Should be identical
        np.testing.assert_array_equal(
            ops.convert_to_numpy(indices_1),
            ops.convert_to_numpy(indices_2),
            err_msg="Indices should be identical after roundtrip"
        )

    def test_gradient_flow(self, quantizer_config, sample_input_2d):
        """Test that gradients flow through the layer."""
        layer = VectorQuantizer(**quantizer_config)

        # Use TensorFlow's GradientTape for gradient computation
        sample_input_var = tf.Variable(sample_input_2d)

        with tf.GradientTape() as tape:
            output = layer(sample_input_var, training=True)
            # Add layer losses to ensure gradients for embeddings
            loss = ops.mean(ops.square(output)) + ops.sum(layer.losses)

        # Get gradients
        gradients = tape.gradient(loss, [sample_input_var] + layer.trainable_variables)

        # Check that gradients exist and are not None
        assert all(g is not None for g in gradients)

        # Check that gradients are not all zeros
        for g in gradients:
            g_np = ops.convert_to_numpy(g)
            assert not np.allclose(g_np, 0.0)

    def test_compute_output_shape(self, quantizer_config):
        """Test compute_output_shape method."""
        layer = VectorQuantizer(**quantizer_config)

        input_shape = (None, 8, 8, 32)
        output_shape = layer.compute_output_shape(input_shape)

        # Output shape should be same as input shape
        assert output_shape == input_shape

    def test_config_completeness(self, quantizer_config):
        """Test that get_config contains all initialization parameters."""
        layer = VectorQuantizer(**quantizer_config)

        config = layer.get_config()

        # Check all important parameters are in config
        assert 'num_embeddings' in config
        assert 'embedding_dim' in config
        assert 'commitment_cost' in config
        assert 'initializer' in config
        assert 'use_ema' in config
        assert 'ema_decay' in config
        assert 'epsilon' in config

        # Check values match
        assert config['num_embeddings'] == quantizer_config['num_embeddings']
        assert config['embedding_dim'] == quantizer_config['embedding_dim']
        assert config['commitment_cost'] == quantizer_config['commitment_cost']
        assert config['use_ema'] == quantizer_config['use_ema']
        assert config['ema_decay'] == quantizer_config['ema_decay']
        assert config['epsilon'] == quantizer_config['epsilon']

    def test_serialization_cycle(self, quantizer_config, sample_input_2d):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        outputs = VectorQuantizer(**quantizer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_2d, training=False)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_2d, training=False)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions differ after serialization"
            )

    def test_serialization_with_ema(self, quantizer_ema_config, sample_input_2d):
        """Test serialization cycle with EMA enabled."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        outputs = VectorQuantizer(**quantizer_ema_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Train for a few steps to update EMA
        for _ in range(5):
            _ = model(sample_input_2d, training=True)

        # Get prediction
        original_pred = model(sample_input_2d, training=False)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model_ema.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_2d, training=False)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions differ after serialization with EMA"
            )

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, quantizer_config, sample_input_2d, training):
        """Test behavior in different training modes."""
        layer = VectorQuantizer(**quantizer_config)

        output = layer(sample_input_2d, training=training)

        # Output shape should be correct regardless of training mode
        assert output.shape == sample_input_2d.shape

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])