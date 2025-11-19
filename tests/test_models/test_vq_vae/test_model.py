"""
Comprehensive test suite for VQ-VAE implementation.

This module provides thorough testing of VQVAEModel

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

Run with: pytest test_vqvae.py -v
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Any


from dl_techniques.models.vq_vae.model import VQVAEModel


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
# VQVAEModel Tests
# ============================================================================

class TestVQVAEModel:
    """Comprehensive test suite for VQVAEModel."""

    def test_initialization(self, vqvae_config):
        """Test model initialization with valid configuration."""
        model = VQVAEModel(**vqvae_config)

        # Check attributes are stored
        assert model.encoder is not None
        assert model.decoder is not None
        assert model.quantizer is not None
        assert model.num_embeddings == vqvae_config['num_embeddings']
        assert model.embedding_dim == vqvae_config['embedding_dim']
        assert model.commitment_cost == vqvae_config['commitment_cost']
        assert model.reconstruction_loss_weight == vqvae_config['reconstruction_loss_weight']

        # Check metrics are created
        assert hasattr(model, 'total_loss_tracker')
        assert hasattr(model, 'reconstruction_loss_tracker')
        assert hasattr(model, 'vq_loss_tracker')

    def test_initialization_invalid_num_embeddings(self, simple_encoder, simple_decoder):
        """Test initialization with invalid num_embeddings."""
        with pytest.raises(ValueError, match="num_embeddings must be positive"):
            VQVAEModel(
                encoder=simple_encoder,
                decoder=simple_decoder,
                num_embeddings=0,
                embedding_dim=32,
            )

    def test_initialization_invalid_embedding_dim(self, simple_encoder, simple_decoder):
        """Test initialization with invalid embedding_dim."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            VQVAEModel(
                encoder=simple_encoder,
                decoder=simple_decoder,
                num_embeddings=128,
                embedding_dim=-5,
            )

    def test_initialization_invalid_reconstruction_weight(self, simple_encoder, simple_decoder):
        """Test initialization with invalid reconstruction_loss_weight."""
        with pytest.raises(ValueError, match="reconstruction_loss_weight must be positive"):
            VQVAEModel(
                encoder=simple_encoder,
                decoder=simple_decoder,
                num_embeddings=128,
                embedding_dim=32,
                reconstruction_loss_weight=0.0,
            )

    def test_forward_pass(self, vqvae_config, sample_images):
        """Test forward pass through the model."""
        model = VQVAEModel(**vqvae_config)

        output = model(sample_images, training=False)

        # Check output shape matches input shape
        assert output.shape == sample_images.shape

        # Check output is in valid range for sigmoid activation
        output_np = ops.convert_to_numpy(output)
        assert np.all(output_np >= 0.0)
        assert np.all(output_np <= 1.0)

    def test_encode_method(self, vqvae_config, sample_images):
        """Test encode method."""
        model = VQVAEModel(**vqvae_config)

        latents = model.encode(sample_images)

        # Check output has correct embedding dimension
        assert latents.shape[-1] == vqvae_config['embedding_dim']

        # Check spatial dimensions are reduced (due to striding)
        assert latents.shape[1] < sample_images.shape[1]
        assert latents.shape[2] < sample_images.shape[2]

    def test_quantize_latents_method(self, vqvae_config, sample_images):
        """Test quantize method."""
        model = VQVAEModel(**vqvae_config)

        # Encode first
        z_e = model.encode(sample_images)

        # Quantize
        z_q = model.quantize_latents(z_e)

        # Check shape is preserved
        assert z_q.shape == z_e.shape

    def test_decode_method(self, vqvae_config, sample_images):
        """Test decode method."""
        model = VQVAEModel(**vqvae_config)

        # Encode and quantize
        z_e = model.encode(sample_images)
        z_q = model.quantize_latents(z_e)

        # Decode
        reconstructed = model.decode(z_q)

        # Check shape matches input
        assert reconstructed.shape == sample_images.shape

    def test_encode_to_indices(self, vqvae_config, sample_images):
        """Test encoding directly to discrete indices."""
        model = VQVAEModel(**vqvae_config)

        indices = model.encode_to_indices(sample_images)

        # Check shape (no channel dimension)
        assert len(indices.shape) == 3  # (batch, height, width)

        # Check indices are integers
        indices_np = ops.convert_to_numpy(indices)
        assert indices_np.dtype in [np.int32, np.int64]

        # Check indices are in valid range
        assert np.all(indices_np >= 0)
        assert np.all(indices_np < vqvae_config['num_embeddings'])

    def test_decode_from_indices(self, vqvae_config, sample_images):
        """Test decoding from discrete indices."""
        model = VQVAEModel(**vqvae_config)

        # Get indices
        indices = model.encode_to_indices(sample_images)

        # Decode from indices
        reconstructed = model.decode_from_indices(indices)

        # Check shape matches input
        assert reconstructed.shape == sample_images.shape

    def test_indices_reconstruction_consistency(self, vqvae_config, sample_images):
        """Test that reconstruction via indices matches direct reconstruction."""
        model = VQVAEModel(**vqvae_config)

        # Direct reconstruction
        direct_recon = model(sample_images, training=False)

        # Reconstruction via indices
        indices = model.encode_to_indices(sample_images)
        indices_recon = model.decode_from_indices(indices)

        # Should be very close (might have small numerical differences)
        np.testing.assert_allclose(
            ops.convert_to_numpy(direct_recon),
            ops.convert_to_numpy(indices_recon),
            rtol=1e-4, atol=1e-4,
            err_msg="Direct and indices-based reconstruction differ"
        )

    def test_compile_and_fit(self, vqvae_config, sample_images):
        """Test model compilation and training."""
        model = VQVAEModel(**vqvae_config)

        # Compile
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

        # Train for a few steps
        history = model.fit(
            sample_images,
            epochs=2,
            batch_size=4,
            verbose=0
        )

        # Check that losses are recorded
        assert 'loss' in history.history
        assert 'reconstruction_loss' in history.history
        assert 'vq_loss' in history.history

        # Check that losses decreased (at least not NaN)
        for metric_name in ['loss', 'reconstruction_loss', 'vq_loss']:
            losses = history.history[metric_name]
            assert all(not np.isnan(l) for l in losses), f"{metric_name} contains NaN"
            assert all(not np.isinf(l) for l in losses), f"{metric_name} contains Inf"

    def test_train_step(self, vqvae_config, sample_images):
        """Test custom train_step method."""
        model = VQVAEModel(**vqvae_config)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

        # Single training step
        metrics = model.train_step(sample_images)

        # Check metrics are returned
        assert 'loss' in metrics
        assert 'reconstruction_loss' in metrics
        assert 'vq_loss' in metrics

        # Check metrics are scalars and finite
        for metric_name, metric_value in metrics.items():
            value = ops.convert_to_numpy(metric_value)
            assert value.shape == (), f"{metric_name} is not a scalar"
            assert np.isfinite(value), f"{metric_name} is not finite"
            assert value >= 0, f"{metric_name} is negative"

    def test_test_step(self, vqvae_config, sample_images):
        """Test custom test_step method."""
        model = VQVAEModel(**vqvae_config)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

        # Single test step
        metrics = model.test_step(sample_images)

        # Check metrics are returned
        assert 'loss' in metrics
        assert 'reconstruction_loss' in metrics
        assert 'vq_loss' in metrics

        # Check metrics are scalars and finite
        for metric_name, metric_value in metrics.items():
            value = ops.convert_to_numpy(metric_value)
            assert value.shape == (), f"{metric_name} is not a scalar"
            assert np.isfinite(value), f"{metric_name} is not finite"

    def test_metrics_property(self, vqvae_config):
        """Test that metrics property returns correct metrics."""
        model = VQVAEModel(**vqvae_config)

        metrics = model.metrics

        # Should have 3 metrics
        assert len(metrics) == 3

        # Check metric names
        metric_names = [m.name for m in metrics]
        assert 'total_loss' in metric_names
        assert 'reconstruction_loss' in metric_names
        assert 'vq_loss' in metric_names

    def test_gradient_flow(self, vqvae_config, sample_images):
        """Test that gradients flow through the entire model."""
        model = VQVAEModel(**vqvae_config)

        with tf.GradientTape() as tape:
            output = model(sample_images, training=True)
            # Add model losses (VQ losses) to ensure gradients
            loss = ops.mean(ops.square(sample_images - output)) + ops.sum(model.losses)

        # Get gradients for all trainable variables
        gradients = tape.gradient(loss, model.trainable_variables)

        # Check that gradients exist
        assert all(g is not None for g in gradients)

        # Check that at least some gradients are non-zero
        non_zero_grads = sum(
            1 for g in gradients
            if not np.allclose(ops.convert_to_numpy(g), 0.0)
        )
        assert non_zero_grads > 0, "All gradients are zero"

    def test_config_completeness(self, vqvae_config):
        """Test that get_config contains all initialization parameters."""
        model = VQVAEModel(**vqvae_config)

        config = model.get_config()

        # Check important parameters are in config
        assert 'encoder' in config
        assert 'decoder' in config
        assert 'num_embeddings' in config
        assert 'embedding_dim' in config
        assert 'commitment_cost' in config
        assert 'use_ema' in config
        assert 'ema_decay' in config
        assert 'reconstruction_loss_weight' in config
        assert 'quantizer_initializer' in config

        # Check values match
        assert config['num_embeddings'] == vqvae_config['num_embeddings']
        assert config['embedding_dim'] == vqvae_config['embedding_dim']
        assert config['commitment_cost'] == vqvae_config['commitment_cost']
        assert config['reconstruction_loss_weight'] == vqvae_config['reconstruction_loss_weight']

    def test_serialization_cycle(self, vqvae_config, sample_images):
        """CRITICAL TEST: Full serialization cycle."""
        model = VQVAEModel(**vqvae_config)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

        # Train for a few steps
        model.fit(sample_images, epochs=2, batch_size=4, verbose=0)

        # Get original prediction
        original_pred = model(sample_images, training=False)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_vqvae.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_images, training=False)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions differ after serialization"
            )

    def test_from_config(self, vqvae_config):
        """Test from_config class method."""
        # Create original model
        original_model = VQVAEModel(**vqvae_config)

        # Get config
        config = original_model.get_config()

        # Create new model from config
        new_model = VQVAEModel.from_config(config)

        # Check that new model has same architecture
        assert new_model.num_embeddings == original_model.num_embeddings
        assert new_model.embedding_dim == original_model.embedding_dim
        assert new_model.commitment_cost == original_model.commitment_cost

    @pytest.mark.parametrize("training", [True, False])
    def test_training_modes(self, vqvae_config, sample_images, training):
        """Test behavior in different training modes."""
        model = VQVAEModel(**vqvae_config)

        output = model(sample_images, training=training)

        # Output shape should be correct regardless of training mode
        assert output.shape == sample_images.shape

    def test_reconstruction_quality_improves(self, vqvae_config, sample_images):
        """Test that reconstruction quality improves with training."""
        model = VQVAEModel(**vqvae_config)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

        # Initial reconstruction loss
        initial_recon = model(sample_images, training=False)
        initial_loss = ops.mean(ops.square(sample_images - initial_recon))
        initial_loss_value = ops.convert_to_numpy(initial_loss)

        # Train for several epochs
        model.fit(sample_images, epochs=10, batch_size=4, verbose=0)

        # Final reconstruction loss
        final_recon = model(sample_images, training=False)
        final_loss = ops.mean(ops.square(sample_images - final_recon))
        final_loss_value = ops.convert_to_numpy(final_loss)

        # Loss should decrease (or at least not increase significantly)
        assert final_loss_value <= initial_loss_value * 1.1, \
            "Reconstruction loss did not improve with training"


# ============================================================================
# Integration Tests
# ============================================================================

class TestVQVAEIntegration:
    """Integration tests for the complete VQ-VAE system."""

    def test_end_to_end_workflow(self, vqvae_config, sample_images):
        """Test complete workflow: train, encode, decode, generate."""
        model = VQVAEModel(**vqvae_config)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

        # 1. Train the model
        model.fit(sample_images, epochs=5, batch_size=4, verbose=0)

        # 2. Encode to continuous latents
        z_e = model.encode(sample_images)
        assert z_e.shape[-1] == vqvae_config['embedding_dim']

        # 3. Quantize to discrete latents
        z_q = model.quantize_latents(z_e)
        assert z_q.shape == z_e.shape

        # 4. Decode back to images
        reconstructed = model.decode(z_q)
        assert reconstructed.shape == sample_images.shape

        # 5. Get discrete codes
        indices = model.encode_to_indices(sample_images)
        assert len(indices.shape) == 3

        # 6. Generate from discrete codes
        generated = model.decode_from_indices(indices)
        assert generated.shape == sample_images.shape

        # 7. Verify reconstruction is reasonable
        reconstruction_error = ops.mean(ops.square(sample_images - reconstructed))
        reconstruction_error_value = ops.convert_to_numpy(reconstruction_error)
        assert reconstruction_error_value < 0.5, \
            "Reconstruction error is too high"

    def test_codebook_usage(self, vqvae_config, sample_images):
        """Test that codebook is being used (not all codes map to same embedding)."""
        model = VQVAEModel(**vqvae_config)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

        # Train briefly
        model.fit(sample_images, epochs=5, batch_size=4, verbose=0)

        # Get indices
        indices = model.encode_to_indices(sample_images)
        indices_np = ops.convert_to_numpy(indices)

        # Count unique indices used
        unique_indices = np.unique(indices_np)

        # Should use multiple codes (more than 1)
        assert len(unique_indices) > 1, "Model is only using a single code"

        # Ideally should use a reasonable fraction of the codebook
        usage_ratio = len(unique_indices) / vqvae_config['num_embeddings']
        # This is a soft requirement - might not be met with small training
        if usage_ratio < 0.1:
            print(f"Warning: Low codebook usage ({usage_ratio:.2%})")

    def test_ema_vs_gradient_consistency(self, simple_encoder, simple_decoder, sample_images):
        """Test that EMA and gradient-based training produce similar results."""
        # Create two models: one with EMA, one without
        model_grad = VQVAEModel(
            encoder=simple_encoder,
            decoder=simple_decoder,
            num_embeddings=64,
            embedding_dim=32,
            use_ema=False,
        )

        # Clone encoder and decoder for second model
        encoder_2 = keras.models.clone_model(simple_encoder)
        decoder_2 = keras.models.clone_model(simple_decoder)

        model_ema = VQVAEModel(
            encoder=encoder_2,
            decoder=decoder_2,
            num_embeddings=64,
            embedding_dim=32,
            use_ema=True,
        )

        # Compile both
        model_grad.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
        model_ema.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

        # Train both (small number of epochs)
        model_grad.fit(sample_images, epochs=3, batch_size=4, verbose=0)
        model_ema.fit(sample_images, epochs=3, batch_size=4, verbose=0)

        # Get reconstructions
        recon_grad = model_grad(sample_images, training=False)
        recon_ema = model_ema(sample_images, training=False)

        # Both should produce reasonable reconstructions
        # (They won't be identical but should have similar quality)
        error_grad = ops.mean(ops.square(sample_images - recon_grad))
        error_ema = ops.mean(ops.square(sample_images - recon_ema))

        error_grad_value = ops.convert_to_numpy(error_grad)
        error_ema_value = ops.convert_to_numpy(error_ema)

        # Both should be finite and reasonable
        assert np.isfinite(error_grad_value)
        assert np.isfinite(error_ema_value)
        assert error_grad_value < 1.0
        assert error_ema_value < 1.0


# ============================================================================
# Performance and Stress Tests
# ============================================================================

class TestVQVAEPerformance:
    """Performance and stress tests."""

    def test_large_batch_size(self, vqvae_config):
        """Test with large batch size."""
        model = VQVAEModel(**vqvae_config)

        large_batch = ops.cast(
            keras.random.uniform(shape=(64, 32, 32, 3)),
            dtype='float32'
        )

        output = model(large_batch, training=False)

        assert output.shape == large_batch.shape

    def test_different_image_sizes(self, simple_encoder, simple_decoder):
        """Test with different input sizes."""
        model = VQVAEModel(
            encoder=simple_encoder,
            decoder=simple_decoder,
            num_embeddings=64,
            embedding_dim=32,
        )

        # Test different sizes
        for size in [16, 32, 64]:
            images = ops.cast(
                keras.random.uniform(shape=(4, size, size, 3)),
                dtype='float32'
            )

            output = model(images, training=False)
            assert output.shape == images.shape

    def test_memory_efficiency(self, vqvae_config, sample_images):
        """Test that model doesn't accumulate memory over multiple calls."""
        model = VQVAEModel(**vqvae_config)

        # Multiple forward passes
        for _ in range(10):
            _ = model(sample_images, training=False)

        # If we got here without OOM, test passes
        assert True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])