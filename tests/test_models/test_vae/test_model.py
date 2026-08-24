import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.sampling import Sampling, HypersphereSampling, VMFSampling
from dl_techniques.models.vae.model import VAE, create_vae, create_vae_from_config

SAMPLING_MODES = [
    "gaussian",
    "hypersphere",
    "vmf",
]


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

    def test_create_vae_returns_a_compiled_model(self):
        """``create_vae``'s remaining contract: it compiles. It no longer self-tests.

        The self-test it used to run lives at
        ``TestVAESamplingTypes::test_create_vae_output_shapes``.
        """
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


class TestVAESamplingTypes:
    """Test the swappable ``sampling_type`` knob across both modes."""

    INPUT_SHAPE = (16, 16, 1)
    LATENT_DIM = 8
    BATCH = 2

    def _expected_log_var_dim(self, mode: str) -> int:
        return 1 if mode in ("hypersphere", "vmf") else self.LATENT_DIM

    def test_invalid_sampling_type(self):
        """An unknown sampling_type must raise ValueError."""
        with pytest.raises(ValueError, match="sampling_type must be one of"):
            VAE(latent_dim=self.LATENT_DIM, input_shape=self.INPUT_SHAPE,
                sampling_type="not_a_mode")

    def test_default_sampling_type_is_gaussian(self):
        """The default mode is the unchanged gaussian baseline."""
        vae = VAE(latent_dim=self.LATENT_DIM, input_shape=self.INPUT_SHAPE)
        assert vae.sampling_type == "gaussian"

    @pytest.mark.parametrize("mode", SAMPLING_MODES)
    def test_build_and_forward_shapes(self, mode):
        """Build + forward pass yields the right output dict and shapes."""
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type=mode,
        )
        x = keras.random.normal((self.BATCH,) + self.INPUT_SHAPE)
        outputs = vae(x, training=False)

        assert set(outputs.keys()) == {"z", "z_mean", "z_log_var", "reconstruction"}
        assert outputs["reconstruction"].shape == x.shape
        # z is [B, latent_dim] in ALL modes (HypersphereSampling emits [B, D]).
        assert outputs["z"].shape == (self.BATCH, self.LATENT_DIM)
        assert outputs["z_mean"].shape == (self.BATCH, self.LATENT_DIM)
        assert outputs["z_log_var"].shape == (
            self.BATCH,
            self._expected_log_var_dim(mode),
        )

    @pytest.mark.parametrize("mode", SAMPLING_MODES)
    def test_train_step_loss_finite(self, mode):
        """A single train step produces a finite (non-NaN/Inf) total_loss."""
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type=mode,
            kl_loss_weight=0.01,
        )
        vae.compile(optimizer="adam")
        data = np.random.rand(4, *self.INPUT_SHAPE).astype(np.float32)

        losses = vae.train_step(data)
        total = keras.ops.convert_to_numpy(losses["total_loss"])
        kl = keras.ops.convert_to_numpy(losses["kl_loss"])

        assert np.isfinite(total), f"{mode}: total_loss not finite ({total})"
        assert np.isfinite(kl), f"{mode}: kl_loss not finite ({kl})"

    @pytest.mark.parametrize("mode", SAMPLING_MODES)
    def test_get_config_carries_sampling_type(self, mode):
        """get_config exposes sampling_type and from_config rebuilds it."""
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type=mode,
        )
        config = vae.get_config()
        assert config["sampling_type"] == mode

        rebuilt = VAE.from_config(config)
        assert rebuilt.sampling_type == mode
        assert rebuilt.latent_dim == self.LATENT_DIM

    @pytest.mark.parametrize("mode", SAMPLING_MODES)
    def test_save_load_roundtrip(self, mode):
        """Full .save()/load_model() round-trip; decoder + encoder still work.

        Compares the DETERMINISTIC encode() mean before/after reload (NOT the
        stochastic reconstruction, since sampling layers are stochastic by
        design).
        """
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type=mode,
        )
        vae.compile(optimizer="adam")
        data = np.random.rand(4, *self.INPUT_SHAPE).astype(np.float32)
        vae.fit(data, epochs=1, verbose=0)

        test_input = data[:self.BATCH]
        original_z_mean, _ = vae.encode(test_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, f"{mode}.keras")
            vae.save(model_path)

            loaded = keras.models.load_model(
                model_path,
                custom_objects={
                    "VAE": VAE,
                    "Sampling": Sampling,
                    "HypersphereSampling": HypersphereSampling,
                    "VMFSampling": VMFSampling,
                },
            )

            assert loaded.sampling_type == mode

            # Deterministic encoder mean must match exactly.
            loaded_z_mean, _ = loaded.encode(test_input)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_z_mean),
                keras.ops.convert_to_numpy(loaded_z_mean),
                rtol=1e-6,
                atol=1e-6,
            )

            # Decoder is usable via the "vae_sampling" extraction.
            z = keras.random.normal((self.BATCH, self.LATENT_DIM))
            recon = loaded.decode(z)
            assert recon.shape == (self.BATCH,) + self.INPUT_SHAPE

            # Encoder is usable.
            enc_mean, enc_log_var = loaded.encode(test_input)
            assert enc_mean.shape == (self.BATCH, self.LATENT_DIM)
            assert enc_log_var.shape == (
                self.BATCH,
                self._expected_log_var_dim(mode),
            )

    def test_sample_prior_gaussian_is_standard_normal(self):
        """gaussian mode: _sample_prior draws ~ N(0, I) (mean~0, std~1)."""
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type="gaussian",
        )
        z = keras.ops.convert_to_numpy(vae._sample_prior(20000))
        assert z.shape == (20000, self.LATENT_DIM)
        # Loose tolerances: a finite Monte-Carlo draw, not exact moments.
        assert abs(float(z.mean())) < 0.05
        assert abs(float(z.std()) - 1.0) < 0.05

    def test_sample_prior_hypersphere_lives_on_radius_shell(self):
        """hypersphere mode: every prior latent has L2 norm == layer radius."""
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type="hypersphere",
        )
        radius = vae.get_layer("vae_sampling").radius
        z = keras.ops.convert_to_numpy(vae._sample_prior(512))
        assert z.shape == (512, self.LATENT_DIM)
        norms = np.linalg.norm(z, axis=-1)
        np.testing.assert_allclose(
            norms, np.full_like(norms, radius), atol=1e-4
        )

    def test_sample_prior_vmf_lives_on_unit_sphere(self):
        """vmf mode: every prior latent is uniform on the UNIT sphere (||z|| == 1).

        The vMF prior (kappa = 0) is exactly the uniform distribution on
        S^{D-1}; _sample_prior draws via Marsaglia at radius 1.0 and reads NO
        .radius attribute (VMFSampling has none).
        """
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type="vmf",
        )
        z = keras.ops.convert_to_numpy(vae._sample_prior(512))
        assert z.shape == (512, self.LATENT_DIM)
        norms = np.linalg.norm(z, axis=-1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4)

    def test_vmf_save_load_roundtrip_deterministic_mu(self):
        """vmf .save()/load_model() round-trip; deterministic encode() mu matches.

        Compares the DETERMINISTIC encoder mean (NOT the stochastic VMFSampling
        output). custom_objects must include VMFSampling (plus the gaussian /
        hypersphere samplers) for reload to resolve the registered layers.
        """
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type="vmf",
        )
        vae.compile(optimizer="adam")
        data = np.random.rand(4, *self.INPUT_SHAPE).astype(np.float32)
        # LESSONS: call/fit once before save so all layers are built.
        vae.fit(data, epochs=1, verbose=0)

        test_input = data[:self.BATCH]
        original_z_mean, original_kappa = vae.encode(test_input)
        # The vmf "z_log_var" slot is the strictly-positive concentration kappa.
        assert original_kappa.shape == (self.BATCH, 1)
        assert float(keras.ops.convert_to_numpy(original_kappa).min()) > 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "vmf.keras")
            vae.save(model_path)

            loaded = keras.models.load_model(
                model_path,
                custom_objects={
                    "VAE": VAE,
                    "Sampling": Sampling,
                    "HypersphereSampling": HypersphereSampling,
                    "VMFSampling": VMFSampling,
                },
            )

            assert loaded.sampling_type == "vmf"

            loaded_z_mean, _ = loaded.encode(test_input)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_z_mean),
                keras.ops.convert_to_numpy(loaded_z_mean),
                rtol=1e-4,
                atol=1e-4,
            )

    @pytest.mark.parametrize("mode", SAMPLING_MODES)
    def test_sample_decodes_prior_to_image_shape(self, mode):
        """sample() still returns decoded images of the input shape (all modes)."""
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type=mode,
        )
        samples = vae.sample(num_samples=3)
        assert samples.shape == (3,) + self.INPUT_SHAPE

    @pytest.mark.parametrize("mode", SAMPLING_MODES)
    def test_create_vae_output_shapes(self, mode):
        """The shape contract ``create_vae`` used to ``assert`` on, as a real test.

        Until step 19 the factory ran a random forward pass and ``assert``-ed on
        these three shapes ITSELF (audit rule R-051, routed SEVERE). That was
        void under ``python -O``, and the ``keras.random.uniform`` draw shifted
        the global seed stream for every later caller. The check moved here, and
        it got stronger on the way: it now runs for all three sampling types
        rather than for whichever one the caller happened to request.

        ``hypersphere`` emits a single scalar radius log-variance ``[B, 1]`` and
        ``vmf`` a single scalar concentration kappa ``[B, 1]``; ``gaussian``
        keeps the full ``[B, latent_dim]`` log-variance.
        """
        vae = create_vae(
            input_shape=self.INPUT_SHAPE,
            latent_dim=self.LATENT_DIM,
            variant="micro",
            sampling_type=mode,
        )
        assert vae.sampling_type == mode

        batch = keras.random.uniform((2,) + self.INPUT_SHAPE)
        out = vae(batch, training=False)
        assert tuple(out["reconstruction"].shape) == (2,) + self.INPUT_SHAPE
        assert tuple(out["z_mean"].shape) == (2, self.LATENT_DIM)
        expected_log_var_dim = 1 if mode in ("hypersphere", "vmf") else self.LATENT_DIM
        assert tuple(out["z_log_var"].shape) == (2, expected_log_var_dim)

    def test_create_vae_runs_no_forward_pass(self):
        """The factory must BUILD a model, never RUN one.

        This is the RED proof for the removal: restore the self-test forward
        pass in ``create_vae`` and the counter below reads 1 instead of 0.

        A first draft of this test tried to prove the point via the global seed
        stream -- seed, build, draw, and compare against the same sequence
        without the factory's ``keras.random.uniform`` call. It FAILED, and the
        instrument was the reason: weight initialization draws from that same
        global stream, so BOTH arms shift it and the comparison could never
        isolate the factory's own draw. Counting invocations is unconfounded.
        """
        calls = []
        original_call = VAE.call

        def counting_call(self, *args, **kwargs):
            calls.append(1)
            return original_call(self, *args, **kwargs)

        VAE.call = counting_call
        try:
            create_vae(
                input_shape=self.INPUT_SHAPE,
                latent_dim=self.LATENT_DIM,
                variant="micro",
            )
        finally:
            VAE.call = original_call

        assert calls == [], (
            f"create_vae invoked the model {len(calls)} time(s). The factory "
            "builds and compiles; it does not run. Its old self-test forward "
            "pass moved to test_create_vae_output_shapes."
        )

    def test_legacy_hypersphere_faithful_alias_maps_to_hypersphere(self):
        """Back-compat: the deprecated 'hypersphere_faithful' value constructs
        successfully and is normalized to 'hypersphere' (so old configs /
        checkpoints whose stored sampling_type is the legacy name still load)."""
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type="hypersphere_faithful",
        )
        assert vae.sampling_type == "hypersphere"
        # The normalized value is what get_config round-trips.
        assert vae.get_config()["sampling_type"] == "hypersphere"

    def test_removed_hypersphere_controlled_raises(self):
        """The dropped 'hypersphere_controlled' mode now raises ValueError."""
        with pytest.raises(ValueError, match="hypersphere_controlled.*removed"):
            VAE(
                latent_dim=self.LATENT_DIM,
                input_shape=self.INPUT_SHAPE,
                sampling_type="hypersphere_controlled",
            )

    @pytest.mark.parametrize("mode", SAMPLING_MODES)
    def test_kl_weight_variable_inits_from_kl_loss_weight(self, mode):
        """The schedulable kl_weight tf.Variable starts == the ctor kl_loss_weight
        (so with no warmup callback attached, behavior is identical to before)."""
        klw = 0.007
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type=mode,
            kl_loss_weight=klw,
        )
        assert vae.kl_loss_weight == klw  # python-float source of truth
        assert float(keras.ops.convert_to_numpy(vae.kl_weight)) == pytest.approx(klw)

    def test_kl_loss_weight_roundtrips_with_kl_weight_variable(self):
        """Adding the kl_weight variable must not break get_config/from_config
        round-trip of the python-float kl_loss_weight."""
        klw = 0.003
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=self.INPUT_SHAPE,
            sampling_type="vmf",
            kl_loss_weight=klw,
        )
        cfg = vae.get_config()
        assert cfg["kl_loss_weight"] == klw
        rebuilt = VAE.from_config(cfg)
        assert rebuilt.kl_loss_weight == klw
        # The rebuilt model's variable also re-inits from the round-tripped float.
        assert float(keras.ops.convert_to_numpy(rebuilt.kl_weight)) == pytest.approx(klw)

    def test_vmf_initial_kappa_is_high(self):
        """vmf encoder_kappa head starts at kappa ~= softplus(12) ~= 12 (the
        zeros-kernel + Constant(12) bias init), NOT softplus(0) ~= 0.69. This is
        the posterior-collapse cure (D-007): z is informative from step 0."""
        vae = VAE(
            latent_dim=self.LATENT_DIM,
            input_shape=(8, 8, 1),
            sampling_type="vmf",
        )
        x = keras.random.normal((4, 8, 8, 1))
        out = vae(x, training=False)
        kappa = keras.ops.convert_to_numpy(out["z_log_var"])
        assert kappa.shape == (4, 1)
        # softplus(12) ~= 12.000006; the zeros kernel makes init independent of x.
        assert float(kappa.min()) > 8.0, f"init kappa too low: {kappa.ravel()}"
        assert float(kappa.mean()) == pytest.approx(12.0, abs=1.0)


class TestKLWarmupCallback:
    """The train_vae KL-warmup callback ramps model.kl_weight 0 -> target."""

    def _build_model(self, kl_loss_weight=0.02):
        return VAE(
            latent_dim=4,
            input_shape=(8, 8, 1),
            sampling_type="vmf",
            kl_loss_weight=kl_loss_weight,
        )

    def test_warmup_ramps_then_saturates(self):
        from train.vae.train_vae import KLWarmupCallback

        target = 0.02
        N = 5
        model = self._build_model(kl_loss_weight=target)
        cb = KLWarmupCallback(target=target, warmup_epochs=N)
        cb.set_model(model)

        seen = []
        for epoch in range(0, N + 3):
            cb.on_epoch_begin(epoch)
            seen.append(float(keras.ops.convert_to_numpy(model.kl_weight)))

        # epoch 0 -> 0; linear ramp; saturates at target from epoch N onward.
        assert seen[0] == pytest.approx(0.0)
        assert seen[1] == pytest.approx(target * 1 / N)
        assert seen[N - 1] == pytest.approx(target * (N - 1) / N)
        assert seen[N] == pytest.approx(target)
        assert seen[N + 2] == pytest.approx(target)  # clamped, no overshoot
        # Monotone non-decreasing.
        assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))

    def test_warmup_rejects_nonpositive_epochs(self):
        from train.vae.train_vae import KLWarmupCallback

        with pytest.raises(ValueError):
            KLWarmupCallback(target=0.01, warmup_epochs=0)


class TestVAEEffectiveBeta:
    """F-53: `kl_loss_weight` is `beta / prod(input_shape)`, and that is PINNED.

    The reconstruction term is a MEAN over pixels while the gaussian KL is a SUM
    over latents, so the `beta` of the standard sum-over-pixels ELBO is
    `kl_loss_weight * prod(input_shape)`. This was undocumented, which made a
    nominal `kl_loss_weight` mean different things at different resolutions:
    MEASURED over the shipped `MODEL_VARIANTS`, `micro`/`small` optimize
    beta 7.84 at (28, 28, 1) and 30.72 at (32, 32, 3) -- a 3.92x swing in
    regularization strength from the input resolution alone.

    The arithmetic is deliberately UNCHANGED (see the D-028 anchor in
    `_compute_reconstruction_loss`); what these tests pin is that the convention
    is what the docs now say it is, so a later "cleanup" of either reduction is
    caught here rather than silently re-scaling every shipped preset.
    """

    def test_reconstruction_is_a_mean_over_pixels_not_a_sum(self):
        vae = VAE(latent_dim=4, input_shape=(8, 8, 1), kl_loss_weight=0.01)
        rng = np.random.default_rng(0)
        y_true = rng.uniform(size=(4, 8, 8, 1)).astype("float32")
        y_pred = np.clip(y_true + 0.1, 1e-3, 1 - 1e-3).astype("float32")

        got = float(keras.ops.convert_to_numpy(
            vae._compute_reconstruction_loss(y_true, y_pred)))
        per_pixel_mean = float(np.mean(keras.ops.convert_to_numpy(
            keras.losses.binary_crossentropy(
                y_true.reshape(4, -1),
                np.clip(y_pred.reshape(4, -1), 1e-7, 1 - 1e-7)))))

        assert got == pytest.approx(per_pixel_mean, rel=1e-5)
        # And it is emphatically NOT the sum over the 64 pixels, which is the
        # reduction the standard ELBO uses and the one `effective_kl_beta`
        # converts to. `rel=0.5` is deliberately loose: this arm only has to
        # separate a mean from a 64x-larger sum, not measure either.
        assert got != pytest.approx(64.0 * per_pixel_mean, rel=0.5)

    def test_kl_is_a_sum_over_the_latent_axis_for_gaussian(self):
        vae = VAE(latent_dim=6, input_shape=(8, 8, 1), sampling_type="gaussian")
        z_mean = np.zeros((4, 6), dtype="float32")
        z_log_var = np.zeros((4, 6), dtype="float32")
        z_mean[:, :] = 1.0
        got = float(keras.ops.convert_to_numpy(
            vae._compute_kl_loss(z_mean, z_log_var)))
        # -0.5 * sum_j (1 + 0 - 1 - 1) = 0.5 * latent_dim
        assert got == pytest.approx(0.5 * 6, rel=1e-5)

    @pytest.mark.parametrize("shape", [(28, 28, 1), (32, 32, 3)])
    def test_effective_beta_is_the_weight_times_the_pixel_count(self, shape):
        vae = VAE(latent_dim=8, input_shape=shape, kl_loss_weight=0.01)
        assert vae.effective_kl_beta == pytest.approx(
            0.01 * int(np.prod(shape)))

    def test_the_same_nominal_weight_is_a_different_beta_per_resolution(self):
        """The finding itself, as an assertion."""
        mnist = VAE(latent_dim=8, input_shape=(28, 28, 1), kl_loss_weight=0.01)
        cifar = VAE(latent_dim=8, input_shape=(32, 32, 3), kl_loss_weight=0.01)
        assert mnist.kl_loss_weight == cifar.kl_loss_weight
        assert cifar.effective_kl_beta / mnist.effective_kl_beta == (
            pytest.approx(3072.0 / 784.0))

    def test_shipped_variant_effective_betas_are_the_measured_ones(self):
        """Any re-tune of `MODEL_VARIANTS` must update these numbers on purpose."""
        expected_at_mnist = {
            "micro": 7.84, "small": 7.84,
            "medium": 3.92, "large": 3.92,
            "xlarge": 0.784,
        }
        assert set(expected_at_mnist) == set(VAE.MODEL_VARIANTS)
        for variant, beta in expected_at_mnist.items():
            weight = VAE.MODEL_VARIANTS[variant]["kl_loss_weight"]
            assert weight * 784 == pytest.approx(beta, rel=1e-6), (
                f"variant {variant!r} now optimizes an effective beta of "
                f"{weight * 784:.4g} at (28, 28, 1), not {beta}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])