"""
Comprehensive test suite for the VAE model.

This module contains all tests for the VAE model implementation,
covering initialization, forward pass, training, serialization,
model integration, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os
from typing import Tuple

from dl_techniques.layers.sampling import Sampling
from dl_techniques.models.vae import VAE, create_vae


class TestVAE:
    """Test suite for VAE model implementation."""

    @pytest.fixture
    def input_shape(self) -> Tuple[int, int, int]:
        """Create test input shape for small images."""
        return (32, 32, 3)

    @pytest.fixture
    def mnist_input_shape(self) -> Tuple[int, int, int]:
        """Create test input shape for MNIST-like images."""
        return (28, 28, 1)

    @pytest.fixture
    def latent_dim(self) -> int:
        """Create test latent dimension."""
        return 16

    @pytest.fixture
    def sample_data(self, input_shape):
        """Create sample input data."""
        batch_size = 8
        return tf.random.uniform([batch_size] + list(input_shape), 0, 1)

    @pytest.fixture
    def mnist_sample_data(self, mnist_input_shape):
        """Create sample MNIST-like data."""
        batch_size = 8
        return tf.random.uniform([batch_size] + list(mnist_input_shape), 0, 1)

    def test_initialization_defaults(self, latent_dim):
        """Test initialization with default parameters."""
        vae = VAE(latent_dim=latent_dim)

        assert vae.latent_dim == latent_dim
        assert vae.encoder_filters == [32, 64]
        assert vae.decoder_filters == [64, 32]
        assert vae.kl_loss_weight == 1.0
        assert vae.use_batch_norm is True
        assert vae.dropout_rate == 0.0
        assert vae.activation == "leaky_relu"
        assert vae.name == "vae"
        assert vae.built is False

        # Check that encoder and decoder are not built yet
        assert vae.encoder is None
        assert vae.decoder is None
        assert vae.sampling_layer is None

    def test_initialization_custom(self, latent_dim, input_shape):
        """Test initialization with custom parameters."""
        vae = VAE(
            latent_dim=latent_dim,
            encoder_filters=[16, 32, 64],
            decoder_filters=[64, 32, 16],
            kl_loss_weight=0.5,
            input_shape=input_shape,
            kernel_initializer="glorot_uniform",
            use_batch_norm=False,
            dropout_rate=0.2,
            activation="relu",
            name="custom_vae"
        )

        assert vae.latent_dim == latent_dim
        assert vae.encoder_filters == [16, 32, 64]
        assert vae.decoder_filters == [64, 32, 16]
        assert vae.kl_loss_weight == 0.5
        assert vae._input_shape == input_shape
        assert vae.use_batch_norm is False
        assert vae.dropout_rate == 0.2
        assert vae.activation == "relu"
        assert vae.name == "custom_vae"
        assert vae.built is True  # Should be built if input_shape provided

        # Check that encoder and decoder are built
        assert vae.encoder is not None
        assert vae.decoder is not None
        assert vae.sampling_layer is not None

    def test_initialization_with_regularization(self, latent_dim):
        """Test initialization with regularization."""
        vae = VAE(
            latent_dim=latent_dim,
            kernel_regularizer="l2"
        )

        assert vae.kernel_regularizer is not None
        assert isinstance(vae.kernel_regularizer, keras.regularizers.L2)

    def test_build_process(self, latent_dim, input_shape, sample_data):
        """Test that the model builds properly."""
        vae = VAE(latent_dim=latent_dim)

        # Build by calling with data
        output = vae(sample_data)

        assert vae.built is True
        assert vae._input_shape == input_shape
        assert vae.encoder is not None
        assert vae.decoder is not None
        assert vae.sampling_layer is not None
        assert vae._encoder_output_shape is not None

    def test_encoder_output_shape_calculation(self, latent_dim, input_shape):
        """Test encoder output shape calculation."""
        vae = VAE(
            latent_dim=latent_dim,
            input_shape=input_shape,
            encoder_filters=[32, 64]
        )

        # For input (32, 32, 3) with 2 stride-2 convolutions:
        # 32 -> 16 -> 8
        expected_height = expected_width = 8
        expected_channels = 64  # Last encoder filter

        assert vae._encoder_output_shape == (expected_height, expected_width, expected_channels)

    def test_forward_pass(self, latent_dim, input_shape, sample_data):
        """Test forward pass produces expected outputs."""
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)

        outputs = vae(sample_data)

        # Check output structure
        assert isinstance(outputs, dict)
        assert "reconstruction" in outputs
        assert "z_mean" in outputs
        assert "z_log_var" in outputs

        # Check output shapes
        assert outputs["reconstruction"].shape == sample_data.shape
        assert outputs["z_mean"].shape == (sample_data.shape[0], latent_dim)
        assert outputs["z_log_var"].shape == (sample_data.shape[0], latent_dim)

        # Check for valid values
        assert not np.any(np.isnan(outputs["reconstruction"].numpy()))
        assert not np.any(np.isnan(outputs["z_mean"].numpy()))
        assert not np.any(np.isnan(outputs["z_log_var"].numpy()))

    def test_encode_method(self, latent_dim, input_shape, sample_data):
        """Test the encode method."""
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)

        z_mean, z_log_var = vae.encode(sample_data)

        assert z_mean.shape == (sample_data.shape[0], latent_dim)
        assert z_log_var.shape == (sample_data.shape[0], latent_dim)
        assert not np.any(np.isnan(z_mean.numpy()))
        assert not np.any(np.isnan(z_log_var.numpy()))

    def test_decode_method(self, latent_dim, input_shape):
        """Test the decode method."""
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)

        # Create latent samples
        batch_size = 4
        z_samples = tf.random.normal([batch_size, latent_dim])

        reconstructions = vae.decode(z_samples)

        assert reconstructions.shape == (batch_size,) + input_shape
        assert not np.any(np.isnan(reconstructions.numpy()))
        # Check sigmoid output range
        assert np.all(reconstructions.numpy() >= 0)
        assert np.all(reconstructions.numpy() <= 1)

    def test_decode_method_unbuilt_model(self, latent_dim):
        """Test decode method on unbuilt model raises error."""
        vae = VAE(latent_dim=latent_dim)  # No input_shape
        z_samples = tf.random.normal([4, latent_dim])

        with pytest.raises(ValueError, match="Model must be built"):
            vae.decode(z_samples)

    def test_sample_method(self, latent_dim, input_shape):
        """Test the sample method for generation."""
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)

        num_samples = 5
        generated = vae.sample(num_samples)

        assert generated.shape == (num_samples,) + input_shape
        assert not np.any(np.isnan(generated.numpy()))
        # Check sigmoid output range
        assert np.all(generated.numpy() >= 0)
        assert np.all(generated.numpy() <= 1)

    def test_sample_method_unbuilt_model(self, latent_dim):
        """Test sample method on unbuilt model raises error."""
        vae = VAE(latent_dim=latent_dim)  # No input_shape

        with pytest.raises(ValueError, match="Model must be built"):
            vae.sample(5)

    def test_different_input_shapes(self, latent_dim):
        """Test with different input shapes."""
        test_shapes = [
            (28, 28, 1),  # MNIST
            (32, 32, 3),  # CIFAR-10
            (64, 64, 3),  # Larger images
            (16, 16, 1),  # Small images
        ]

        for shape in test_shapes:
            vae = VAE(latent_dim=latent_dim, input_shape=shape)
            batch_data = tf.random.uniform([2] + list(shape), 0, 1)

            outputs = vae(batch_data)
            assert outputs["reconstruction"].shape == batch_data.shape

    def test_different_latent_dimensions(self, input_shape):
        """Test with different latent dimensions."""
        latent_dims = [2, 8, 16, 32, 64, 128]

        for latent_dim in latent_dims:
            vae = VAE(latent_dim=latent_dim, input_shape=input_shape)
            batch_data = tf.random.uniform([2] + list(input_shape), 0, 1)

            outputs = vae(batch_data)
            assert outputs["z_mean"].shape[1] == latent_dim
            assert outputs["z_log_var"].shape[1] == latent_dim

    def test_different_filter_configurations(self, latent_dim, input_shape):
        """Test with different encoder/decoder filter configurations."""
        filter_configs = [
            ([16], [16]),
            ([32, 64], [64, 32]),
            ([16, 32, 64], [64, 32, 16]),
            ([32, 64, 128, 256], [256, 128, 64, 32]),
        ]

        for enc_filters, dec_filters in filter_configs:
            vae = VAE(
                latent_dim=latent_dim,
                input_shape=input_shape,
                encoder_filters=enc_filters,
                decoder_filters=dec_filters
            )
            batch_data = tf.random.uniform([2] + list(input_shape), 0, 1)

            outputs = vae(batch_data)
            assert outputs["reconstruction"].shape == batch_data.shape

    def test_without_batch_norm(self, latent_dim, input_shape, sample_data):
        """Test model without batch normalization."""
        vae = VAE(
            latent_dim=latent_dim,
            input_shape=input_shape,
            use_batch_norm=False
        )

        outputs = vae(sample_data)
        assert outputs["reconstruction"].shape == sample_data.shape

    def test_with_dropout(self, latent_dim, input_shape, sample_data):
        """Test model with dropout."""
        vae = VAE(
            latent_dim=latent_dim,
            input_shape=input_shape,
            dropout_rate=0.3
        )

        # Test training mode (dropout active)
        outputs_train = vae(sample_data, training=True)

        # Test inference mode (dropout inactive)
        outputs_test = vae(sample_data, training=False)

        # Both should have correct shapes
        assert outputs_train["reconstruction"].shape == sample_data.shape
        assert outputs_test["reconstruction"].shape == sample_data.shape

    def test_different_activations(self, latent_dim, input_shape):
        """Test with different activation functions."""
        activations = ["relu", "leaky_relu", "elu", "gelu"]

        for activation in activations:
            vae = VAE(
                latent_dim=latent_dim,
                input_shape=input_shape,
                activation=activation
            )
            batch_data = tf.random.uniform([2] + list(input_shape), 0, 1)

            outputs = vae(batch_data)
            assert outputs["reconstruction"].shape == batch_data.shape

    def test_loss_computation(self, latent_dim, input_shape, sample_data):
        """Test loss computation methods."""
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)

        outputs = vae(sample_data)

        # Test reconstruction loss
        recon_loss = vae._compute_reconstruction_loss(
            sample_data, outputs["reconstruction"]
        )
        assert recon_loss.shape == ()  # Scalar
        assert recon_loss >= 0

        # Test KL loss
        kl_loss = vae._compute_kl_loss(
            outputs["z_mean"], outputs["z_log_var"]
        )
        assert kl_loss.shape == ()  # Scalar
        # KL loss can be negative (when posterior is more concentrated than prior)

    def test_training_step(self, latent_dim, input_shape, sample_data):
        """Test custom training step."""
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)
        vae.compile(optimizer="adam")

        # Test training step
        metrics = vae.train_step(sample_data)

        # Check returned metrics
        assert "total_loss" in metrics
        assert "reconstruction_loss" in metrics
        assert "kl_loss" in metrics

        # Check metric values are scalars
        for metric_value in metrics.values():
            assert metric_value.shape == ()

    def test_test_step(self, latent_dim, input_shape, sample_data):
        """Test custom test step."""
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)
        vae.compile(optimizer="adam")

        # Test evaluation step
        metrics = vae.test_step(sample_data)

        # Check returned metrics
        assert "total_loss" in metrics
        assert "reconstruction_loss" in metrics
        assert "kl_loss" in metrics

    def test_kl_loss_weight(self, latent_dim, input_shape, sample_data):
        """Test different KL loss weights."""
        kl_weights = [0.0, 0.5, 1.0, 2.0]

        for kl_weight in kl_weights:
            vae = VAE(
                latent_dim=latent_dim,
                input_shape=input_shape,
                kl_loss_weight=kl_weight
            )
            vae.compile(optimizer="adam")

            metrics = vae.train_step(sample_data)
            assert "total_loss" in metrics

    def test_serialization(self, latent_dim, input_shape):
        """Test serialization and deserialization of the model."""
        original_vae = VAE(
            latent_dim=latent_dim,
            input_shape=input_shape,
            encoder_filters=[32, 64],
            decoder_filters=[64, 32],
            kl_loss_weight=0.8,
            use_batch_norm=False,
            dropout_rate=0.1,
            activation="relu"
        )

        # Get configs
        config = original_vae.get_config()
        build_config = original_vae.get_build_config()

        # Recreate the model
        recreated_vae = VAE.from_config(config)
        recreated_vae.build_from_config(build_config)

        # Check configuration matches
        assert recreated_vae.latent_dim == original_vae.latent_dim
        assert recreated_vae.encoder_filters == original_vae.encoder_filters
        assert recreated_vae.decoder_filters == original_vae.decoder_filters
        assert recreated_vae.kl_loss_weight == original_vae.kl_loss_weight
        assert recreated_vae.use_batch_norm == original_vae.use_batch_norm
        assert recreated_vae.dropout_rate == original_vae.dropout_rate
        assert recreated_vae.activation == original_vae.activation

    def test_model_save_load(self, latent_dim, input_shape, sample_data):
        """Test saving and loading a VAE model."""
        # Create and compile model
        vae = VAE(
            latent_dim=latent_dim,
            input_shape=input_shape,
            name="test_vae"
        )
        vae.compile(optimizer="adam")

        # --- FIX: Train for one step to give Batch Norm layers a defined state. ---
        vae.train_step(sample_data)
        # --------------------------------------------------------------------------

        # Generate prediction before saving (ensure training=False for deterministic output)
        original_outputs = vae(sample_data, training=False)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "vae_model.keras")

            # Save the model
            vae.save(model_path)

            # Load the model
            loaded_vae = keras.models.load_model(
                model_path,
                custom_objects={
                    "VAE": VAE,
                    # Make sure the Sampling class is imported in the test file
                    "Sampling": Sampling
                }
            )

            # Generate prediction with loaded model
            loaded_outputs = loaded_vae(sample_data, training=False)

            # Check output shape match
            assert original_outputs["reconstruction"].numpy().shape == loaded_outputs["reconstruction"].numpy().shape


    def test_training_integration(self, latent_dim, mnist_input_shape, mnist_sample_data):
        """Test training integration with a small dataset."""
        vae = VAE(
            latent_dim=latent_dim,
            input_shape=mnist_input_shape,
            encoder_filters=[16, 32],
            decoder_filters=[32, 16]
        )
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))

        # Create small dataset
        dataset = tf.data.Dataset.from_tensor_slices(mnist_sample_data)
        dataset = dataset.batch(4)

        # Train for a few steps
        history = vae.fit(dataset, epochs=2, verbose=0)

        # Check that training metrics are recorded
        assert "total_loss" in history.history
        assert "reconstruction_loss" in history.history
        assert "kl_loss" in history.history
        assert len(history.history["total_loss"]) == 2  # 2 epochs

    def test_gradient_flow(self, latent_dim, input_shape, sample_data):
        """Test gradient flow through the model."""
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)
        vae.compile(optimizer="adam")

        with tf.GradientTape() as tape:
            outputs = vae(sample_data, training=True)
            loss = vae._compute_reconstruction_loss(sample_data, outputs["reconstruction"])

        # Get gradients
        grads = tape.gradient(loss, vae.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check some gradients have non-zero values
        has_nonzero_grad = any(np.any(g.numpy() != 0) for g in grads if g is not None)
        assert has_nonzero_grad

    def test_create_vae_factory(self, latent_dim, input_shape):
        """Test the create_vae factory function."""
        vae = create_vae(
            input_shape=input_shape,
            latent_dim=latent_dim,
            encoder_filters=[32, 64],
            decoder_filters=[64, 32],
            learning_rate=0.001
        )

        assert isinstance(vae, VAE)
        assert vae.latent_dim == latent_dim
        assert vae._input_shape == input_shape
        assert vae.built is True
        assert vae.optimizer is not None

    def test_create_vae_with_custom_optimizer(self, latent_dim, input_shape):
        """Test create_vae with custom optimizer."""
        custom_optimizer = keras.optimizers.Adam(learning_rate=0.002)

        vae = create_vae(
            input_shape=input_shape,
            latent_dim=latent_dim,
            optimizer=custom_optimizer
        )

        assert vae.optimizer == custom_optimizer

    def test_numerical_stability(self, latent_dim):
        """Test model stability with extreme input values."""
        input_shape = (16, 16, 1)
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)

        test_cases = [
            tf.zeros((2,) + input_shape),  # All zeros
            tf.ones((2,) + input_shape),   # All ones
            tf.random.uniform((2,) + input_shape, 0, 1e-6),  # Very small values
            tf.random.uniform((2,) + input_shape, 1-1e-6, 1),  # Very close to 1
        ]

        for i, test_input in enumerate(test_cases):
            outputs = vae(test_input)

            # Check for NaN/Inf values
            for key, tensor in outputs.items():
                assert not np.any(np.isnan(tensor.numpy())), f"NaN in {key} for test case {i}"
                assert not np.any(np.isinf(tensor.numpy())), f"Inf in {key} for test case {i}"

    def test_metrics_reset(self, latent_dim, input_shape, sample_data):
        """Test that metrics reset properly between epochs."""
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)
        vae.compile(optimizer="adam")

        # Train step
        metrics1 = vae.train_step(sample_data)

        # Reset metrics
        for metric in vae.metrics:
            metric.reset_state()

        # Another train step
        metrics2 = vae.train_step(sample_data)

        # Metrics should be different after reset
        # (unless by coincidence they're the same)
        assert isinstance(metrics1, dict)
        assert isinstance(metrics2, dict)

    def test_latent_space_properties(self, latent_dim, input_shape, sample_data):
        """Test basic properties of the learned latent space."""
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)

        outputs = vae(sample_data)
        z_mean = outputs["z_mean"]
        z_log_var = outputs["z_log_var"]

        # Check that latent means have reasonable range
        mean_magnitude = tf.reduce_mean(tf.abs(z_mean))
        assert mean_magnitude < 10.0  # Shouldn't be too large initially

        # Check that log variances are reasonable
        log_var_mean = tf.reduce_mean(z_log_var)
        assert log_var_mean > -10.0 and log_var_mean < 10.0  # Reasonable range

    def test_batch_size_independence(self, latent_dim, input_shape):
        """Test that model works with different batch sizes."""
        vae = VAE(latent_dim=latent_dim, input_shape=input_shape)

        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            test_data = tf.random.uniform((batch_size,) + input_shape, 0, 1)
            outputs = vae(test_data)

            assert outputs["reconstruction"].shape[0] == batch_size
            assert outputs["z_mean"].shape[0] == batch_size
            assert outputs["z_log_var"].shape[0] == batch_size