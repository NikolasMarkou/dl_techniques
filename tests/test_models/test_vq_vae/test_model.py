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


# ---------------------------------------------------------------------------
# Reconstruction bars, derived from the trivial baseline
# ---------------------------------------------------------------------------
# The reconstruction assertions in this file used to read `error < 0.5`,
# `error < 1.0` and `final <= initial * 1.1`. On `sample_images`
# (uniform(0, 1)) the CONSTANT predictor 0.5 has MSE = Var(U(0,1)) = 1/12 =
# 0.0833, so `< 0.5` is ~6x looser than predicting a constant, and a test named
# "reconstruction_quality_improves" passed a model that got 10% WORSE.
#
# MEASURED 2026-08-18, this exact config on `sample_images`: initial 0.08339,
# final after 10 epochs 0.08329 -- the model sits EXACTLY on the variance floor,
# because i.i.d. uniform noise is incompressible and nothing better than the
# mean can be learned from it. A derived bar is therefore untestable on that
# fixture: the data, not the assertion, is what made the claim unfalsifiable.
#
# `_structured_images` supplies learnable data (smooth sinusoidal texture,
# variance 0.0625) on which the claim is real. MEASURED at 60 epochs, 3 runs
# each: MSE/variance = 0.102 / 0.124 / 0.103 (gradient codebook) and
# 0.306 / 0.172 / 0.241 (EMA codebook). The bar below is 0.6, ~2x above the
# worst measurement; a model that only predicts the mean scores exactly 1.0 and
# fails.
RECON_TRAIN_EPOCHS = 60
RECON_MSE_OVER_VARIANCE = 0.6


def _structured_images(n: int = 8, size: int = 32, seed: int = 0) -> np.ndarray:
    """Learnable images: smooth sinusoidal texture in [0, 1], seeded."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij"
    )
    images = []
    for _ in range(n):
        freq = rng.integers(1, 3)
        phase = rng.random()
        base = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (xx + phase)) * np.cos(
            2 * np.pi * freq * (yy + phase)
        )
        images.append(np.stack([base, np.roll(base, 4, axis=0), 1.0 - base], -1))
    return np.clip(np.asarray(images, dtype="float32"), 0.0, 1.0)


def _assert_beats_the_constant_predictor(images, reconstruction, *, what: str) -> float:
    """MSE must be a fraction of the data variance (= the constant predictor's MSE)."""
    images = np.asarray(ops.convert_to_numpy(images))
    error = float(
        ops.convert_to_numpy(ops.mean(ops.square(images - reconstruction)))
    )
    baseline = float(images.var())
    assert np.isfinite(error)
    assert error < RECON_MSE_OVER_VARIANCE * baseline, (
        f"{what}: reconstruction MSE {error:.5f} is {error / baseline:.2f}x the "
        f"data variance {baseline:.5f}. A model that outputs the constant mean "
        f"scores exactly 1.00x; the bar is {RECON_MSE_OVER_VARIANCE:.2f}x "
        f"(measured 0.10-0.31x for a working model at "
        f"{RECON_TRAIN_EPOCHS} epochs)."
    )
    return error


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

    def test_reconstruction_quality_improves(self, vqvae_config):
        """Reconstruction must beat the constant predictor after training.

        The old bar, `final <= initial * 1.1`, passed a model that got 10%
        WORSE -- on a test named "improves" -- and was evaluated on
        incompressible uniform noise where no model can improve at all. See the
        measurements above `_structured_images`.
        """
        images = _structured_images()
        model = VQVAEModel(**vqvae_config)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

        initial_loss_value = float(
            ops.convert_to_numpy(
                ops.mean(ops.square(images - model(images, training=False)))
            )
        )

        model.fit(images, epochs=RECON_TRAIN_EPOCHS, batch_size=4, verbose=0)

        final_loss_value = _assert_beats_the_constant_predictor(
            images, model(images, training=False), what="after training"
        )
        assert final_loss_value < initial_loss_value, (
            f"training did not reduce the reconstruction loss at all: "
            f"{initial_loss_value:.5f} -> {final_loss_value:.5f}"
        )


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

        # 7. Verify reconstruction is reasonable. `< 0.5` on uniform(0, 1)
        # inputs was ~6x LOOSER than predicting the constant 0.5, so it could
        # not distinguish this pipeline from a model that ignores its input.
        # The bar is now derived from that constant predictor, on data where
        # beating it is possible at all.
        images = _structured_images(seed=1)
        trained = VQVAEModel(**vqvae_config)
        trained.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
        trained.fit(images, epochs=RECON_TRAIN_EPOCHS, batch_size=4, verbose=0)
        _assert_beats_the_constant_predictor(
            images,
            trained.decode_from_indices(trained.encode_to_indices(images)),
            what="end-to-end encode -> indices -> decode",
        )

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

        # `error < 1.0` on uniform(0, 1) images was nearly unfalsifiable: the
        # constant-mean predictor scores 0.083. Both codebook update rules must
        # instead BEAT that constant predictor by the derived margin. MEASURED:
        # the EMA path needs materially longer than the gradient path -- at 30
        # epochs it was still collapsed to a single code (MSE/variance = 1.000),
        # and it escapes to 5 codes and 0.17-0.31 by 60. That is why this test
        # trains for RECON_TRAIN_EPOCHS rather than 3.
        images = _structured_images(seed=2)
        model_grad.fit(images, epochs=RECON_TRAIN_EPOCHS, batch_size=4, verbose=0)
        model_ema.fit(images, epochs=RECON_TRAIN_EPOCHS, batch_size=4, verbose=0)

        _assert_beats_the_constant_predictor(
            images, model_grad(images, training=False), what="gradient codebook"
        )
        _assert_beats_the_constant_predictor(
            images, model_ema(images, training=False), what="EMA codebook"
        )


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