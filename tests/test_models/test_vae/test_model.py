import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vae.model import VAE, create_vae, create_vae_from_config


class TestVAEInitialization:
    """Test VAE model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic VAE initialization with default parameters."""
        input_shape = (32, 32, 3)
        vae = VAE(latent_dim=64, input_shape=input_shape)

        assert vae.latent_dim == 64
        assert vae._input_shape == input_shape
        assert vae.depths == 2
        assert vae.steps_per_depth == 1
        assert vae.filters == [32, 64]
        assert vae.kl_loss_weight == 0.01
        assert vae.use_batch_norm is True
        assert vae.dropout_rate == 0.0
        assert vae.activation == "leaky_relu"
        assert vae.final_activation == "sigmoid"

    def test_custom_initialization(self):
        """Test VAE initialization with custom parameters."""
        custom_filters = [16, 32, 64, 128]
        input_shape = (64, 64, 3)
        vae = VAE(
            latent_dim=128,
            input_shape=input_shape,
            depths=4,
            steps_per_depth=3,
            filters=custom_filters,
            kl_loss_weight=0.001,
            dropout_rate=0.2,
            activation="relu",
            use_batch_norm=False,
            final_activation="tanh",
        )

        assert vae.latent_dim == 128
        assert vae._input_shape == input_shape
        assert vae.depths == 4
        assert vae.steps_per_depth == 3
        assert vae.filters == custom_filters
        assert vae.kl_loss_weight == 0.001
        assert vae.dropout_rate == 0.2
        assert vae.activation == "relu"
        assert vae.use_batch_norm is False
        assert vae.final_activation == "tanh"

    def test_invalid_parameters(self):
        """Test VAE initialization with invalid parameters raises errors."""
        input_shape = (32, 32, 3)

        # Test negative latent_dim
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            VAE(latent_dim=-10, input_shape=input_shape)

        # Test zero latent_dim
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            VAE(latent_dim=0, input_shape=input_shape)

        # Test negative depths
        with pytest.raises(ValueError, match="depths must be positive"):
            VAE(latent_dim=64, input_shape=input_shape, depths=-1)

        # Test negative steps_per_depth
        with pytest.raises(ValueError, match="steps_per_depth must be positive"):
            VAE(latent_dim=64, input_shape=input_shape, steps_per_depth=0)

        # Test invalid dropout_rate
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            VAE(latent_dim=64, input_shape=input_shape, dropout_rate=1.5)

        # Test mismatched filters length
        with pytest.raises(ValueError, match="Filters array length"):
            VAE(latent_dim=64, input_shape=input_shape, depths=3, filters=[32, 64])

        # Test invalid input shape
        with pytest.raises(ValueError, match="input_shape must be 3D"):
            VAE(latent_dim=64, input_shape=(32, 32))

        # Test too small input dimensions
        with pytest.raises(ValueError, match="Input dimensions must be at least 8x8"):
            VAE(latent_dim=64, input_shape=(4, 4, 3))

    def test_initializer_and_regularizer_handling(self):
        """Test proper handling of initializers and regularizers."""
        input_shape = (32, 32, 3)

        # Test with string initializer
        vae1 = VAE(
            latent_dim=64, input_shape=input_shape, kernel_initializer="glorot_normal"
        )
        assert vae1.kernel_initializer == "glorot_normal"

        # Test with initializer object
        init = keras.initializers.HeNormal()
        vae2 = VAE(latent_dim=64, input_shape=input_shape, kernel_initializer=init)
        assert vae2.kernel_initializer is init

        # Test with regularizer object
        reg = keras.regularizers.L1(0.01)
        vae3 = VAE(latent_dim=64, input_shape=input_shape, kernel_regularizer=reg)
        assert vae3.kernel_regularizer is reg


class TestVAEVariants:
    """Test VAE model variants functionality."""

    def test_variant_creation(self):
        """Test creating VAE models from variants."""
        input_shape = (28, 28, 1)

        variants_to_test = ["micro", "small", "medium", "large", "xlarge"]

        for variant in variants_to_test:
            vae = VAE.from_variant(variant, input_shape=input_shape)

            assert isinstance(vae, VAE)
            assert vae._input_shape == input_shape

            # Check that configuration matches variant
            variant_config = VAE.MODEL_VARIANTS[variant]
            assert vae.depths == variant_config["depths"]
            assert vae.steps_per_depth == variant_config["steps_per_depth"]
            assert vae.filters == variant_config["filters"]
            assert vae.kl_loss_weight == variant_config["kl_loss_weight"]
            assert vae.latent_dim == variant_config["default_latent_dim"]

    def test_variant_with_custom_latent_dim(self):
        """Test variant creation with custom latent dimension."""
        input_shape = (32, 32, 3)
        custom_latent_dim = 256

        vae = VAE.from_variant(
            "medium", input_shape=input_shape, latent_dim=custom_latent_dim
        )

        assert vae.latent_dim == custom_latent_dim
        # Other parameters should match variant
        variant_config = VAE.MODEL_VARIANTS["medium"]
        assert vae.depths == variant_config["depths"]
        assert vae.filters == variant_config["filters"]

    def test_invalid_variant(self):
        """Test that invalid variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            VAE.from_variant("invalid_variant", input_shape=(32, 32, 3))

    def test_variant_functional_test(self):
        """Test that variant models are functional."""
        input_shape = (32, 32, 3)
        vae = VAE.from_variant("small", input_shape=input_shape)

        # Test forward pass
        test_input = keras.random.normal((2,) + input_shape)
        outputs = vae(test_input, training=False)

        assert isinstance(outputs, dict)
        assert "reconstruction" in outputs
        assert outputs["reconstruction"].shape == test_input.shape


class TestVAEForwardPass:
    """Test VAE forward pass functionality."""

    @pytest.fixture
    def vae_model(self) -> VAE:
        """Create a VAE for testing."""
        return VAE(latent_dim=64, input_shape=(32, 32, 3))

    def test_forward_pass_shapes(self, vae_model):
        """Test forward pass produces correct output shapes."""
        batch_size = 4
        input_shape = (32, 32, 3)
        test_input = keras.random.normal((batch_size,) + input_shape)

        outputs = vae_model(test_input, training=False)

        assert isinstance(outputs, dict)
        assert "reconstruction" in outputs
        assert "z" in outputs
        assert "z_mean" in outputs
        assert "z_log_var" in outputs

        assert outputs["reconstruction"].shape == test_input.shape
        assert outputs["z"].shape == (batch_size, 64)
        assert outputs["z_mean"].shape == (batch_size, 64)
        assert outputs["z_log_var"].shape == (batch_size, 64)

    def test_forward_pass_training_mode(self, vae_model):
        """Test forward pass in training mode."""
        test_input = keras.random.normal((2, 32, 32, 3))

        outputs_train = vae_model(test_input, training=True)
        outputs_eval = vae_model(test_input, training=False)

        # Both should have same structure
        assert set(outputs_train.keys()) == set(outputs_eval.keys())
        assert outputs_train["reconstruction"].shape == outputs_eval["reconstruction"].shape

    def test_encode_method(self, vae_model):
        """Test encode method functionality."""
        test_input = keras.random.normal((3, 32, 32, 3))

        z_mean, z_log_var = vae_model.encode(test_input)

        assert z_mean.shape == (3, 64)
        assert z_log_var.shape == (3, 64)

    def test_decode_method(self, vae_model):
        """Test decode method functionality."""
        test_z = keras.random.normal((3, 64))

        reconstruction = vae_model.decode(test_z)

        assert reconstruction.shape == (3, 32, 32, 3)

    def test_sample_method(self, vae_model):
        """Test sample method functionality."""
        num_samples = 5
        samples = vae_model.sample(num_samples)

        assert samples.shape == (num_samples, 32, 32, 3)

    def test_different_input_shapes(self):
        """Test VAE with different input shapes."""
        test_shapes = [
            (28, 28, 1),  # MNIST-like
            (64, 64, 3),  # Medium resolution
            (128, 128, 1),  # High resolution grayscale
        ]

        for input_shape in test_shapes:
            vae = VAE(latent_dim=64, input_shape=input_shape)
            test_input = keras.random.normal((2,) + input_shape)

            outputs = vae(test_input, training=False)
            assert outputs["reconstruction"].shape == test_input.shape


class TestVAETraining:
    """Test VAE training functionality."""

    @pytest.fixture
    def compiled_vae(self) -> VAE:
        """Create a compiled VAE for testing."""
        vae = VAE(latent_dim=64, input_shape=(28, 28, 1), kl_loss_weight=0.001)
        vae.compile(optimizer="adam")
        return vae

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Create sample training data."""
        return np.random.rand(16, 28, 28, 1).astype(np.float32)

    def test_train_step_basic(self, compiled_vae, sample_data):
        """Test basic train step functionality."""
        losses = compiled_vae.train_step(sample_data)

        assert isinstance(losses, dict)
        assert "total_loss" in losses
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses

        # All losses should be positive
        assert keras.ops.convert_to_numpy(losses["total_loss"]) > 0
        assert keras.ops.convert_to_numpy(losses["reconstruction_loss"]) > 0
        assert keras.ops.convert_to_numpy(losses["kl_loss"]) >= 0

    def test_test_step_basic(self, compiled_vae, sample_data):
        """Test basic test step functionality."""
        losses = compiled_vae.test_step(sample_data)

        assert isinstance(losses, dict)
        assert "total_loss" in losses
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses

    def test_train_step_with_tuple_data(self, compiled_vae, sample_data):
        """Test train step with tuple data format."""
        # Create fake labels (should be ignored)
        fake_labels = np.random.randint(0, 10, (16,))
        tuple_data = (sample_data, fake_labels)

        losses = compiled_vae.train_step(tuple_data)

        assert isinstance(losses, dict)
        assert "total_loss" in losses

    def test_metrics_tracking(self, compiled_vae, sample_data):
        """Test that metrics are properly tracked."""
        # Reset metrics
        for metric in compiled_vae.metrics:
            metric.reset_state()

        # Run train step
        compiled_vae.train_step(sample_data)

        # Check metrics are updated
        assert compiled_vae.total_loss_tracker.count > 0
        assert compiled_vae.reconstruction_loss_tracker.count > 0
        assert compiled_vae.kl_loss_tracker.count > 0

    def test_fit_method(self, compiled_vae, sample_data):
        """Test that fit method works correctly."""
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(sample_data).batch(8)

        # Train for a few steps
        history = compiled_vae.fit(dataset, epochs=2, verbose=0)

        assert "total_loss" in history.history
        assert "reconstruction_loss" in history.history
        assert "kl_loss" in history.history

        # Check that losses are recorded
        assert len(history.history["total_loss"]) == 2

    def test_gradient_flow(self, compiled_vae, sample_data):
        """Test that gradients flow properly through the network."""
        with tf.GradientTape() as tape:
            outputs = compiled_vae(sample_data, training=True)
            reconstruction_loss = compiled_vae._compute_reconstruction_loss(
                sample_data, outputs["reconstruction"]
            )
            kl_loss = compiled_vae._compute_kl_loss(
                outputs["z_mean"], outputs["z_log_var"]
            )
            total_loss = reconstruction_loss + compiled_vae.kl_loss_weight * kl_loss

        gradients = tape.gradient(total_loss, compiled_vae.trainable_weights)

        # Check that we have gradients for all trainable weights
        assert len(gradients) == len(compiled_vae.trainable_weights)

        # Check that most gradients are not None
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > 0


class TestVAELossFunctions:
    """Test VAE loss function computations."""

    @pytest.fixture
    def vae_model(self) -> VAE:
        """Create a VAE for testing."""
        return VAE(latent_dim=64, input_shape=(32, 32, 3))

    def test_reconstruction_loss_computation(self, vae_model):
        """Test reconstruction loss computation."""
        batch_size = 4
        y_true = keras.random.uniform((batch_size, 32, 32, 3), 0, 1)
        y_pred = keras.random.uniform((batch_size, 32, 32, 3), 0, 1)

        loss = vae_model._compute_reconstruction_loss(y_true, y_pred)

        assert isinstance(loss, tf.Tensor)
        assert loss.shape == ()  # Scalar
        assert keras.ops.convert_to_numpy(loss) >= 0

    def test_kl_loss_computation(self, vae_model):
        """Test KL divergence loss computation."""
        batch_size = 4
        latent_dim = 64

        z_mean = keras.random.normal((batch_size, latent_dim))
        z_log_var = keras.random.normal((batch_size, latent_dim))

        kl_loss = vae_model._compute_kl_loss(z_mean, z_log_var)

        assert isinstance(kl_loss, tf.Tensor)
        assert kl_loss.shape == ()  # Scalar
        assert keras.ops.convert_to_numpy(kl_loss) >= 0

    def test_reconstruction_loss_shape_mismatch(self, vae_model):
        """Test that reconstruction loss raises error on shape mismatch."""
        y_true = keras.random.uniform((4, 32, 32, 3), 0, 1)
        y_pred = keras.random.uniform((4, 16, 16, 3), 0, 1)  # Different shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            vae_model._compute_reconstruction_loss(y_true, y_pred)

    def test_kl_loss_numerical_stability(self, vae_model):
        """Test KL loss numerical stability with extreme values."""
        batch_size = 4
        latent_dim = 64

        # Test with extreme log variance values
        z_mean = keras.ops.zeros((batch_size, latent_dim))
        z_log_var = keras.ops.full((batch_size, latent_dim), 100.0)  # Very large

        kl_loss = vae_model._compute_kl_loss(z_mean, z_log_var)

        # Should not be NaN or infinite
        assert keras.ops.isfinite(kl_loss)

    def test_reconstruction_loss_numerical_stability(self, vae_model):
        """Test reconstruction loss numerical stability with extreme values."""
        batch_size = 4

        # Test with extreme predictions
        y_true = keras.ops.ones((batch_size, 32, 32, 3))
        y_pred = keras.ops.zeros((batch_size, 32, 32, 3))  # All zeros

        loss = vae_model._compute_reconstruction_loss(y_true, y_pred)

        # Should not be NaN or infinite
        assert keras.ops.isfinite(loss)


class TestVAESerialization:
    """Test VAE serialization and deserialization."""

    def test_get_config_basic(self):
        """Test basic get_config functionality."""
        vae = VAE(
            latent_dim=128,
            input_shape=(64, 64, 3),
            depths=3,
            steps_per_depth=2,
            filters=[32, 64, 128],
            kl_loss_weight=0.005,
        )

        config = vae.get_config()

        assert isinstance(config, dict)
        assert config["latent_dim"] == 128
        assert config["input_shape"] == (64, 64, 3)
        assert config["depths"] == 3
        assert config["steps_per_depth"] == 2
        assert config["filters"] == [32, 64, 128]
        assert config["kl_loss_weight"] == 0.005

    def test_from_config_basic(self):
        """Test basic from_config functionality."""
        original_vae = VAE(
            latent_dim=128,
            input_shape=(32, 32, 1),
            depths=2,
            steps_per_depth=1,
            filters=[32, 64],
        )

        config = original_vae.get_config()
        reconstructed_vae = VAE.from_config(config)

        assert reconstructed_vae.latent_dim == original_vae.latent_dim
        assert reconstructed_vae._input_shape == original_vae._input_shape
        assert reconstructed_vae.depths == original_vae.depths
        assert reconstructed_vae.steps_per_depth == original_vae.steps_per_depth
        assert reconstructed_vae.filters == original_vae.filters

    def test_config_with_complex_objects(self):
        """Test config with initializers and regularizers."""
        vae = VAE(
            latent_dim=64,
            input_shape=(32, 32, 3),
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=keras.regularizers.L2(0.01),
        )

        config = vae.get_config()
        reconstructed_vae = VAE.from_config(config)

        # Check that objects were properly serialized/deserialized
        assert isinstance(
            reconstructed_vae.kernel_initializer, keras.initializers.HeNormal
        )
        assert isinstance(reconstructed_vae.kernel_regularizer, keras.regularizers.L2)

    def test_model_save_load(self):
        """Test saving and loading complete model deterministically."""
        # Create and compile model
        vae = VAE(latent_dim=64, input_shape=(28, 28, 1))
        vae.compile(optimizer="adam")

        # Create test data and train briefly to get non-random weights
        test_data = np.random.rand(16, 28, 28, 1).astype(np.float32)
        vae.fit(test_data, epochs=1, verbose=0)

        test_input = test_data[:4]

        # Use deterministic parts of the model for comparison
        original_z_mean, _ = vae.encode(test_input)
        original_reconstruction = vae.decode(original_z_mean)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_vae.keras")
            vae.save(model_path)

            loaded_vae = keras.models.load_model(model_path)

            # Test that loaded model produces same deterministic output
            loaded_z_mean, _ = loaded_vae.encode(test_input)
            loaded_reconstruction = loaded_vae.decode(loaded_z_mean)

            # Check that encoder outputs are identical
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_z_mean),
                keras.ops.convert_to_numpy(loaded_z_mean),
                rtol=1e-6,
                atol=1e-6,
            )

            # Check that decoder outputs are identical
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_reconstruction),
                keras.ops.convert_to_numpy(loaded_reconstruction),
                rtol=1e-6,
                atol=1e-6,
            )


class TestVAEEdgeCases:
    """Test VAE edge cases and error handling."""

    def test_small_input_sizes(self):
        """Test VAE with very small input sizes."""
        # Test with minimum possible size after downsampling
        vae = VAE(latent_dim=32, input_shape=(8, 8, 1), depths=2, filters=[32, 64])

        test_input = keras.random.normal((2, 8, 8, 1))
        outputs = vae(test_input, training=False)

        assert outputs["reconstruction"].shape == test_input.shape

    def test_large_latent_dimensions(self):
        """Test VAE with large latent dimensions."""
        vae = VAE(latent_dim=1024, input_shape=(32, 32, 3))

        test_input = keras.random.normal((2, 32, 32, 3))
        outputs = vae(test_input, training=False)

        assert outputs["z"].shape == (2, 1024)

    def test_single_sample_batch(self):
        """Test VAE with batch size of 1."""
        vae = VAE(latent_dim=64, input_shape=(32, 32, 3))

        test_input = keras.random.normal((1, 32, 32, 3))
        outputs = vae(test_input, training=False)

        assert outputs["reconstruction"].shape == (1, 32, 32, 3)

    def test_grayscale_and_rgb(self):
        """Test VAE with both grayscale and RGB inputs."""
        # Grayscale
        vae_gray = VAE(latent_dim=64, input_shape=(28, 28, 1))
        test_gray = keras.random.normal((2, 28, 28, 1))
        outputs_gray = vae_gray(test_gray, training=False)
        assert outputs_gray["reconstruction"].shape == test_gray.shape

        # RGB
        vae_rgb = VAE(latent_dim=64, input_shape=(32, 32, 3))
        test_rgb = keras.random.normal((2, 32, 32, 3))
        outputs_rgb = vae_rgb(test_rgb, training=False)
        assert outputs_rgb["reconstruction"].shape == test_rgb.shape

    def test_different_batch_sizes(self):
        """Test VAE with different batch sizes."""
        vae = VAE(latent_dim=64, input_shape=(32, 32, 3))

        batch_sizes = [1, 4, 8, 16, 32]

        for batch_size in batch_sizes:
            test_input = keras.random.normal((batch_size, 32, 32, 3))
            outputs = vae(test_input, training=False)
            assert outputs["reconstruction"].shape[0] == batch_size


class TestCreateVAEFactory:
    """Test the create_vae factory function."""

    def test_create_vae_basic(self):
        """Test basic create_vae functionality."""
        vae = create_vae(input_shape=(28, 28, 1), latent_dim=64)

        assert isinstance(vae, VAE)
        assert vae.latent_dim == 64
        assert vae._input_shape == (28, 28, 1)
        assert vae.compiled_loss is not None  # Should be compiled

    def test_create_vae_with_custom_optimizer(self):
        """Test create_vae with custom optimizer."""
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        vae = create_vae(input_shape=(32, 32, 3), latent_dim=128, optimizer=optimizer)

        assert vae.optimizer is optimizer

    def test_create_vae_overrides_defaults(self):
        """Test that create_vae properly overrides default parameters."""
        vae = create_vae(
            input_shape=(64, 64, 3),
            latent_dim=256,
            variant="large",
            learning_rate=0.0005,
        )

        assert vae.latent_dim == 256
        # Should use large variant configuration
        variant_config = VAE.MODEL_VARIANTS["large"]
        assert vae.depths == variant_config["depths"]
        assert vae.filters == variant_config["filters"]

    def test_create_vae_functional_test(self):
        """Test that created VAE is functional."""
        vae = create_vae(input_shape=(32, 32, 1), latent_dim=64, variant="small")

        # Test forward pass
        test_input = keras.random.normal((4, 32, 32, 1))
        outputs = vae(test_input, training=False)

        assert outputs["reconstruction"].shape == test_input.shape
        assert outputs["z"].shape == (4, 64)

        # Test encoding/decoding
        z_mean, z_log_var = vae.encode(test_input)
        assert z_mean.shape == (4, 64)

        samples = vae.sample(num_samples=3)
        assert samples.shape == (3, 32, 32, 1)

    def test_create_vae_validation_passes(self):
        """Test that create_vae's internal validation works."""
        # This should not raise any errors
        vae = create_vae(
            input_shape=(64, 64, 3),
            latent_dim=128,
            variant="medium",
            learning_rate=0.001,
        )

        # Model should be ready for training
        assert vae.compiled_loss is not None

    def test_create_vae_from_config(self):
        """Test create_vae_from_config function."""
        config = {
            "latent_dim": 128,
            "input_shape": (64, 64, 3),
            "depths": 3,
            "filters": [32, 64, 128],
            "kl_loss_weight": 0.01,
        }

        vae = create_vae_from_config(config)

        assert isinstance(vae, VAE)
        assert vae.latent_dim == 128
        assert vae._input_shape == (64, 64, 3)
        assert vae.depths == 3
        assert vae.filters == [32, 64, 128]
        assert vae.compiled_loss is not None


class TestVAEIntegration:
    """Integration tests for VAE components working together."""

    def test_end_to_end_training_workflow(self):
        """Test complete training workflow."""
        # Create model
        vae = create_vae(input_shape=(28, 28, 1), latent_dim=64, variant="small")

        # Create synthetic dataset
        x_train = np.random.rand(100, 28, 28, 1).astype(np.float32)
        x_val = np.random.rand(20, 28, 28, 1).astype(np.float32)

        # Train model
        history = vae.fit(
            x_train, validation_data=x_val, epochs=3, batch_size=16, verbose=0
        )

        # Check training history
        assert len(history.history["total_loss"]) == 3
        assert len(history.history["val_total_loss"]) == 3

        # Test evaluation
        eval_results = vae.evaluate(x_val, verbose=0)
        assert isinstance(eval_results, list)

        # Test generation
        samples = vae.sample(num_samples=5)
        assert samples.shape == (5, 28, 28, 1)

    def test_encode_decode_consistency(self):
        """Test that encode/decode operations are consistent."""
        vae = create_vae(input_shape=(32, 32, 3), latent_dim=128, variant="small")

        # Test data
        test_input = keras.random.normal((4, 32, 32, 3))

        # Forward pass through model
        outputs = vae(test_input, training=False)

        # Manual encode/decode
        z_mean, z_log_var = vae.encode(test_input)

        # Results should be consistent
        assert outputs["z_mean"].shape == z_mean.shape
        assert outputs["z_log_var"].shape == z_log_var.shape

        # Mean and log_var should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(outputs["z_mean"]),
            keras.ops.convert_to_numpy(z_mean),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(outputs["z_log_var"]),
            keras.ops.convert_to_numpy(z_log_var),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_loss_components_integration(self):
        """Test that loss components integrate correctly during training."""
        vae = create_vae(input_shape=(28, 28, 1), latent_dim=32, variant="micro")

        # Test data
        test_data = np.random.rand(16, 28, 28, 1).astype(np.float32)

        # Manual loss computation
        outputs = vae(test_data, training=True)
        recon_loss = vae._compute_reconstruction_loss(
            test_data, outputs["reconstruction"]
        )
        kl_loss = vae._compute_kl_loss(outputs["z_mean"], outputs["z_log_var"])

        # Train step loss computation
        train_losses = vae.train_step(test_data)

        # Should be approximately equal (within numerical precision)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(train_losses["reconstruction_loss"]),
            keras.ops.convert_to_numpy(recon_loss),
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(train_losses["kl_loss"]),
            keras.ops.convert_to_numpy(kl_loss),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_shape_consistency_across_variants(self):
        """Test that all variants maintain shape consistency."""
        input_shape = (32, 32, 3)
        latent_dim = 64
        test_input = keras.random.normal((4,) + input_shape)

        for variant in ["micro", "small", "medium", "large"]:  # Skip xlarge for speed
            vae = VAE.from_variant(
                variant, input_shape=input_shape, latent_dim=latent_dim
            )

            outputs = vae(test_input, training=False)

            # All variants should maintain input/output shape consistency
            assert outputs["reconstruction"].shape == test_input.shape
            assert outputs["z"].shape == (4, latent_dim)
            assert outputs["z_mean"].shape == (4, latent_dim)
            assert outputs["z_log_var"].shape == (4, latent_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])