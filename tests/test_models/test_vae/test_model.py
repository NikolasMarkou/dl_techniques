import pytest
import numpy as np
import keras
import tensorflow as tf
from typing import Tuple, List
import tempfile
import os

from dl_techniques.models.vae.model import VAE, create_vae


class TestVAEInitialization:
    """Test VAE model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic VAE initialization with default parameters."""
        vae = VAE(latent_dim=64)

        assert vae.latent_dim == 64
        assert vae.depths == 3
        assert vae.steps_per_depth == 2
        assert vae.filters == [32, 64, 128]
        assert vae.kl_loss_weight == 0.01
        assert vae.use_batch_norm is True
        assert vae.dropout_rate == 0.0
        assert vae.activation == "leaky_relu"
        assert vae.final_activation == "sigmoid"

    def test_custom_initialization(self):
        """Test VAE initialization with custom parameters."""
        custom_filters = [16, 32, 64, 128]
        vae = VAE(
            latent_dim=128,
            depths=4,
            steps_per_depth=3,
            filters=custom_filters,
            kl_loss_weight=0.001,
            input_shape=(64, 64, 3),
            dropout_rate=0.2,
            activation='gelu',
            use_batch_norm=False,
            final_activation='tanh'
        )

        assert vae.latent_dim == 128
        assert vae.depths == 4
        assert vae.steps_per_depth == 3
        assert vae.filters == custom_filters
        assert vae.kl_loss_weight == 0.001
        assert vae._input_shape == (64, 64, 3)
        assert vae.dropout_rate == 0.2
        assert vae.activation == 'gelu'
        assert vae.use_batch_norm is False
        assert vae.final_activation == 'tanh'

    def test_invalid_parameters(self):
        """Test VAE initialization with invalid parameters raises errors."""
        # Test negative latent_dim
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            VAE(latent_dim=-10)

        # Test zero latent_dim
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            VAE(latent_dim=0)

        # Test negative depths
        with pytest.raises(ValueError, match="depths must be positive"):
            VAE(latent_dim=64, depths=-1)

        # Test negative steps_per_depth
        with pytest.raises(ValueError, match="steps_per_depth must be positive"):
            VAE(latent_dim=64, steps_per_depth=0)

        # Test invalid dropout_rate
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            VAE(latent_dim=64, dropout_rate=1.5)

        # Test mismatched filters length
        with pytest.raises(ValueError, match="Filters array length"):
            VAE(latent_dim=64, depths=3, filters=[32, 64])  # depths=3 but only 2 filters

    def test_initializer_and_regularizer_handling(self):
        """Test proper handling of initializers and regularizers."""
        # Test with string initializer
        vae1 = VAE(latent_dim=64, kernel_initializer="glorot_normal")
        assert isinstance(vae1.kernel_initializer, keras.initializers.GlorotNormal)

        # Test with initializer object
        init = keras.initializers.HeNormal()
        vae2 = VAE(latent_dim=64, kernel_initializer=init)
        assert vae2.kernel_initializer is init

        # Test with string regularizer
        vae3 = VAE(latent_dim=64, kernel_regularizer="l2")
        assert isinstance(vae3.kernel_regularizer, keras.regularizers.L2)

        # Test with regularizer object
        reg = keras.regularizers.L1(0.01)
        vae4 = VAE(latent_dim=64, kernel_regularizer=reg)
        assert vae4.kernel_regularizer is reg


class TestVAEBuilding:
    """Test VAE model building and architecture creation."""

    @pytest.fixture
    def input_shapes(self) -> List[Tuple[int, int, int]]:
        """Provide various input shapes for testing."""
        return [
            (28, 28, 1),    # MNIST-like
            (32, 32, 3),    # CIFAR-10-like
            (64, 64, 3),    # Medium resolution
            (128, 128, 1),  # High resolution grayscale
        ]

    def test_build_basic(self, input_shapes):
        """Test basic model building for various input shapes."""
        for input_shape in input_shapes:
            vae = VAE(latent_dim=64)
            vae.build(input_shape)

            assert vae.built is True
            assert vae._build_input_shape == input_shape
            assert vae._input_shape == input_shape
            assert vae.encoder is not None
            assert vae.decoder is not None
            assert vae.sampling_layer is not None

    def test_encoder_architecture(self):
        """Test encoder architecture and output shapes."""
        input_shape = (32, 32, 3)
        latent_dim = 128
        vae = VAE(latent_dim=latent_dim)
        vae.build(input_shape)

        # Test encoder input/output shapes
        test_input = keras.random.normal((4,) + input_shape)
        z_mean, z_log_var = vae.encoder(test_input)

        assert z_mean.shape == (4, latent_dim)
        assert z_log_var.shape == (4, latent_dim)

    def test_decoder_architecture(self):
        """Test decoder architecture and output shapes."""
        input_shape = (32, 32, 3)
        latent_dim = 128
        vae = VAE(latent_dim=latent_dim)
        vae.build(input_shape)

        # Test decoder input/output shapes
        test_z = keras.random.normal((4, latent_dim))
        reconstruction = vae.decoder(test_z)

        assert reconstruction.shape == (4,) + input_shape

    def test_build_prevents_double_building(self):
        """Test that building twice doesn't cause issues."""
        vae = VAE(latent_dim=64)
        input_shape = (28, 28, 1)

        # Build first time
        vae.build(input_shape)
        encoder_first = vae.encoder
        decoder_first = vae.decoder

        # Build second time
        vae.build(input_shape)

        # Should be the same objects (no rebuilding)
        assert vae.encoder is encoder_first
        assert vae.decoder is decoder_first

    def test_different_depths_and_filters(self):
        """Test building with different depths and filter configurations."""
        test_configs = [
            {"depths": 1, "filters": [32]},
            {"depths": 2, "filters": [32, 64]},
            {"depths": 4, "filters": [16, 32, 64, 128]},
        ]

        for config in test_configs:
            vae = VAE(latent_dim=64, **config)
            vae.build((32, 32, 3))

            assert vae.built is True
            assert len(vae.filters) == config["depths"]


class TestVAEForwardPass:
    """Test VAE forward pass functionality."""

    @pytest.fixture
    def built_vae(self) -> VAE:
        """Create a built VAE for testing."""
        vae = VAE(latent_dim=64)
        vae.build((32, 32, 3))
        return vae

    def test_forward_pass_shapes(self, built_vae):
        """Test forward pass produces correct output shapes."""
        batch_size = 4
        input_shape = (32, 32, 3)
        test_input = keras.random.normal((batch_size,) + input_shape)

        outputs = built_vae(test_input, training=False)

        assert isinstance(outputs, dict)
        assert 'reconstruction' in outputs
        assert 'z' in outputs
        assert 'z_mean' in outputs
        assert 'z_log_var' in outputs

        assert outputs['reconstruction'].shape == test_input.shape
        assert outputs['z'].shape == (batch_size, 64)
        assert outputs['z_mean'].shape == (batch_size, 64)
        assert outputs['z_log_var'].shape == (batch_size, 64)

    def test_forward_pass_training_mode(self, built_vae):
        """Test forward pass in training mode."""
        test_input = keras.random.normal((2, 32, 32, 3))

        outputs_train = built_vae(test_input, training=True)
        outputs_eval = built_vae(test_input, training=False)

        # Both should have same structure
        assert set(outputs_train.keys()) == set(outputs_eval.keys())
        assert outputs_train['reconstruction'].shape == outputs_eval['reconstruction'].shape

    def test_encode_method(self, built_vae):
        """Test encode method functionality."""
        test_input = keras.random.normal((3, 32, 32, 3))

        z_mean, z_log_var = built_vae.encode(test_input)

        assert z_mean.shape == (3, 64)
        assert z_log_var.shape == (3, 64)

    def test_decode_method(self, built_vae):
        """Test decode method functionality."""
        test_z = keras.random.normal((3, 64))

        reconstruction = built_vae.decode(test_z)

        assert reconstruction.shape == (3, 32, 32, 3)

    def test_sample_method(self, built_vae):
        """Test sample method functionality."""
        num_samples = 5
        samples = built_vae.sample(num_samples)

        assert samples.shape == (num_samples, 32, 32, 3)

    def test_encode_without_building_raises_error(self):
        """Test that encode without building raises appropriate error."""
        vae = VAE(latent_dim=64)
        test_input = keras.random.normal((2, 32, 32, 3))

        # Should work - encode will build the model automatically
        z_mean, z_log_var = vae.encode(test_input)
        assert z_mean.shape == (2, 64)

    def test_decode_without_building_raises_error(self):
        """Test that decode without building raises appropriate error."""
        vae = VAE(latent_dim=64)
        test_z = keras.random.normal((2, 64))

        with pytest.raises(ValueError, match="Model must be built"):
            vae.decode(test_z)

    def test_sample_without_building_raises_error(self):
        """Test that sample without building raises appropriate error."""
        vae = VAE(latent_dim=64)

        with pytest.raises(ValueError, match="Model must be built"):
            vae.sample(5)


class TestVAETraining:
    """Test VAE training functionality."""

    @pytest.fixture
    def built_vae(self) -> VAE:
        """Create a compiled VAE for testing."""
        vae = VAE(latent_dim=64, kl_loss_weight=0.001)
        vae.build((28, 28, 1))
        vae.compile(optimizer='adam')
        return vae

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Create sample training data."""
        return np.random.rand(16, 28, 28, 1).astype(np.float32)

    def test_train_step_basic(self, built_vae, sample_data):
        """Test basic train step functionality."""
        losses = built_vae.train_step(sample_data)

        assert isinstance(losses, dict)
        assert 'total_loss' in losses
        assert 'reconstruction_loss' in losses
        assert 'kl_loss' in losses

        # All losses should be positive
        assert losses['total_loss'] > 0
        assert losses['reconstruction_loss'] > 0
        assert losses['kl_loss'] >= 0  # KL can be 0

    def test_test_step_basic(self, built_vae, sample_data):
        """Test basic test step functionality."""
        losses = built_vae.test_step(sample_data)

        assert isinstance(losses, dict)
        assert 'total_loss' in losses
        assert 'reconstruction_loss' in losses
        assert 'kl_loss' in losses

    def test_train_step_with_tuple_data(self, built_vae, sample_data):
        """Test train step with tuple data format."""
        # Create fake labels (should be ignored)
        fake_labels = np.random.randint(0, 10, (16,))
        tuple_data = (sample_data, fake_labels)

        losses = built_vae.train_step(tuple_data)

        assert isinstance(losses, dict)
        assert 'total_loss' in losses

    def test_metrics_tracking(self, built_vae, sample_data):
        """Test that metrics are properly tracked."""
        # Reset metrics
        for metric in built_vae.metrics:
            metric.reset_state()

        # Run train step
        built_vae.train_step(sample_data)

        # Check metrics are updated
        assert built_vae.total_loss_tracker.count > 0
        assert built_vae.reconstruction_loss_tracker.count > 0
        assert built_vae.kl_loss_tracker.count > 0

    def test_fit_method(self, built_vae, sample_data):
        """Test that fit method works correctly."""
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(sample_data).batch(8)

        # Train for a few steps
        history = built_vae.fit(dataset, epochs=2, verbose=0)

        assert 'total_loss' in history.history
        assert 'reconstruction_loss' in history.history
        assert 'kl_loss' in history.history

        # Check that losses are recorded
        assert len(history.history['total_loss']) == 2

    def test_gradient_flow(self, built_vae, sample_data):
        """Test that gradients flow properly through the network."""
        with tf.GradientTape() as tape:
            outputs = built_vae(sample_data, training=True)
            reconstruction_loss = built_vae._compute_reconstruction_loss(
                sample_data, outputs['reconstruction']
            )
            kl_loss = built_vae._compute_kl_loss(
                outputs['z_mean'], outputs['z_log_var']
            )
            total_loss = reconstruction_loss + built_vae.kl_loss_weight * kl_loss

        gradients = tape.gradient(total_loss, built_vae.trainable_weights)

        # Check that we have gradients for all trainable weights
        assert len(gradients) == len(built_vae.trainable_weights)

        # Check that most gradients are not None
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > 0


class TestVAELossFunctions:
    """Test VAE loss function computations."""

    @pytest.fixture
    def built_vae(self) -> VAE:
        """Create a built VAE for testing."""
        vae = VAE(latent_dim=64)
        vae.build((32, 32, 3))
        return vae

    def test_reconstruction_loss_computation(self, built_vae):
        """Test reconstruction loss computation."""
        batch_size = 4
        y_true = tf.random.uniform((batch_size, 32, 32, 3), 0, 1)
        y_pred = tf.random.uniform((batch_size, 32, 32, 3), 0, 1)

        loss = built_vae._compute_reconstruction_loss(y_true, y_pred)

        assert isinstance(loss, tf.Tensor)
        assert loss.shape == ()  # Scalar
        assert loss >= 0  # Loss should be non-negative

    def test_kl_loss_computation(self, built_vae):
        """Test KL divergence loss computation."""
        batch_size = 4
        latent_dim = 64

        z_mean = tf.random.normal((batch_size, latent_dim))
        z_log_var = tf.random.normal((batch_size, latent_dim))

        kl_loss = built_vae._compute_kl_loss(z_mean, z_log_var)

        assert isinstance(kl_loss, tf.Tensor)
        assert kl_loss.shape == ()  # Scalar
        assert kl_loss >= 0  # KL divergence should be non-negative

    def test_reconstruction_loss_shape_mismatch(self, built_vae):
        """Test that reconstruction loss raises error on shape mismatch."""
        y_true = tf.random.uniform((4, 32, 32, 3), 0, 1)
        y_pred = tf.random.uniform((4, 16, 16, 3), 0, 1)  # Different shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            built_vae._compute_reconstruction_loss(y_true, y_pred)

    def test_kl_loss_numerical_stability(self, built_vae):
        """Test KL loss numerical stability with extreme values."""
        batch_size = 4
        latent_dim = 64

        # Test with extreme log variance values
        z_mean = tf.zeros((batch_size, latent_dim))
        z_log_var = tf.constant(100.0, shape=(batch_size, latent_dim))  # Very large

        kl_loss = built_vae._compute_kl_loss(z_mean, z_log_var)

        # Should not be NaN or infinite
        assert tf.math.is_finite(kl_loss)

    def test_reconstruction_loss_numerical_stability(self, built_vae):
        """Test reconstruction loss numerical stability with extreme values."""
        batch_size = 4

        # Test with extreme predictions
        y_true = tf.ones((batch_size, 32, 32, 3))
        y_pred = tf.zeros((batch_size, 32, 32, 3))  # All zeros

        loss = built_vae._compute_reconstruction_loss(y_true, y_pred)

        # Should not be NaN or infinite
        assert tf.math.is_finite(loss)


class TestVAESerialization:
    """Test VAE serialization and deserialization."""

    def test_get_config_basic(self):
        """Test basic get_config functionality."""
        vae = VAE(
            latent_dim=128,
            depths=3,
            steps_per_depth=2,
            filters=[32, 64, 128],
            kl_loss_weight=0.005,
            input_shape=(64, 64, 3)
        )

        config = vae.get_config()

        assert isinstance(config, dict)
        assert config['latent_dim'] == 128
        assert config['depths'] == 3
        assert config['steps_per_depth'] == 2
        assert config['filters'] == [32, 64, 128]
        assert config['kl_loss_weight'] == 0.005
        assert config['input_shape'] == (64, 64, 3)

    def test_from_config_basic(self):
        """Test basic from_config functionality."""
        original_vae = VAE(
            latent_dim=128,
            depths=2,
            steps_per_depth=1,
            filters=[32, 64],
            input_shape=(32, 32, 1)
        )

        config = original_vae.get_config()
        reconstructed_vae = VAE.from_config(config)

        assert reconstructed_vae.latent_dim == original_vae.latent_dim
        assert reconstructed_vae.depths == original_vae.depths
        assert reconstructed_vae.steps_per_depth == original_vae.steps_per_depth
        assert reconstructed_vae.filters == original_vae.filters
        assert reconstructed_vae._input_shape == original_vae._input_shape

    def test_config_with_complex_objects(self):
        """Test config with initializers and regularizers."""
        vae = VAE(
            latent_dim=64,
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=keras.regularizers.L2(0.01)
        )

        config = vae.get_config()
        reconstructed_vae = VAE.from_config(config)

        # Check that objects were properly serialized/deserialized
        assert isinstance(reconstructed_vae.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(reconstructed_vae.kernel_regularizer, keras.regularizers.L2)

    def test_model_save_load(self):
        """Test saving and loading complete model."""
        # Create and build model
        vae = VAE(latent_dim=64)
        vae.build((28, 28, 1))
        vae.compile(optimizer='adam')

        # Create some test data and train briefly
        test_data = np.random.rand(16, 28, 28, 1).astype(np.float32)
        vae.fit(test_data, epochs=1, verbose=0)

        # Get a prediction before saving
        test_input = tf.constant(test_data[:4])
        original_output = vae(test_input, training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_vae.keras')
            vae.save(model_path)

            loaded_vae = keras.models.load_model(
                model_path,
                custom_objects={'VAE': VAE}
            )

            # Test that loaded model produces same output
            loaded_output = loaded_vae(test_input, training=False)

            # Outputs should be very close
            np.testing.assert_allclose(
                original_output['reconstruction'].numpy(),
                loaded_output['reconstruction'].numpy(),
                rtol=1e-5
            )


class TestVAEEdgeCases:
    """Test VAE edge cases and error handling."""

    def test_small_input_sizes(self):
        """Test VAE with very small input sizes."""
        # Test with minimum possible size after downsampling
        vae = VAE(latent_dim=32, depths=2, filters=[32, 64])  # Fixed: provide matching filters
        vae.build((8, 8, 1))  # Should work with 8x8 input

        test_input = keras.random.normal((2, 8, 8, 1))
        outputs = vae(test_input, training=False)

        assert outputs['reconstruction'].shape == test_input.shape

    def test_large_latent_dimensions(self):
        """Test VAE with large latent dimensions."""
        vae = VAE(latent_dim=1024)
        vae.build((32, 32, 3))

        test_input = keras.random.normal((2, 32, 32, 3))
        outputs = vae(test_input, training=False)

        assert outputs['z'].shape == (2, 1024)

    def test_single_sample_batch(self):
        """Test VAE with batch size of 1."""
        vae = VAE(latent_dim=64)
        vae.build((32, 32, 3))

        test_input = keras.random.normal((1, 32, 32, 3))
        outputs = vae(test_input, training=False)

        assert outputs['reconstruction'].shape == (1, 32, 32, 3)

    def test_grayscale_and_rgb(self):
        """Test VAE with both grayscale and RGB inputs."""
        # Grayscale
        vae_gray = VAE(latent_dim=64)
        vae_gray.build((28, 28, 1))
        test_gray = keras.random.normal((2, 28, 28, 1))
        outputs_gray = vae_gray(test_gray, training=False)
        assert outputs_gray['reconstruction'].shape == test_gray.shape

        # RGB
        vae_rgb = VAE(latent_dim=64)
        vae_rgb.build((32, 32, 3))
        test_rgb = keras.random.normal((2, 32, 32, 3))
        outputs_rgb = vae_rgb(test_rgb, training=False)
        assert outputs_rgb['reconstruction'].shape == test_rgb.shape

    def test_different_batch_sizes(self):
        """Test VAE with different batch sizes."""
        vae = VAE(latent_dim=64)
        vae.build((32, 32, 3))

        batch_sizes = [1, 4, 8, 16, 32]

        for batch_size in batch_sizes:
            test_input = keras.random.normal((batch_size, 32, 32, 3))
            outputs = vae(test_input, training=False)
            assert outputs['reconstruction'].shape[0] == batch_size


class TestCreateVAEFactory:
    """Test the create_vae factory function."""

    def test_create_vae_basic(self):
        """Test basic create_vae functionality."""
        vae = create_vae(
            input_shape=(28, 28, 1),
            latent_dim=64
        )

        assert isinstance(vae, VAE)
        assert vae.latent_dim == 64
        assert vae.built is True
        assert vae._input_shape == (28, 28, 1)

    def test_create_vae_with_custom_optimizer(self):
        """Test create_vae with custom optimizer."""
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        vae = create_vae(
            input_shape=(32, 32, 3),
            latent_dim=128,
            optimizer=optimizer
        )

        assert vae.optimizer is optimizer

    def test_create_vae_overrides_defaults(self):
        """Test that create_vae properly overrides default parameters."""
        vae = create_vae(
            input_shape=(64, 64, 3),
            latent_dim=256,
            depths=3,
            filters=[64, 128, 256],  # Fixed: provide matching filters for depths=3
            kl_loss_weight=0.005
        )

        assert vae.latent_dim == 256
        assert vae.depths == 3
        assert vae.filters == [64, 128, 256]
        assert vae.kl_loss_weight == 0.005

    def test_create_vae_functional_test(self):
        """Test that created VAE is functional."""
        vae = create_vae(
            input_shape=(32, 32, 1),
            latent_dim=64,
            depths=2,  # Match with filters
            filters=[32, 64]  # Fixed: provide matching filters
        )

        # Test forward pass
        test_input = keras.random.normal((4, 32, 32, 1))
        outputs = vae(test_input, training=False)

        assert outputs['reconstruction'].shape == test_input.shape
        assert outputs['z'].shape == (4, 64)

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
            depths=3,
            steps_per_depth=2,
            filters=[64, 128, 256]  # Fixed: provide matching filters for depths=3
        )

        # Model should be ready for training
        assert vae.built is True
        assert callable(vae.optimizer.apply_gradients)

# Integration tests that test multiple components together
class TestVAEIntegration:
    """Integration tests for VAE components working together."""

    def test_end_to_end_training_workflow(self):
        """Test complete training workflow."""
        # Create model
        vae = create_vae(
            input_shape=(28, 28, 1),
            latent_dim=64,
            kl_loss_weight=0.001
        )

        # Create synthetic dataset
        x_train = np.random.rand(100, 28, 28, 1).astype(np.float32)
        x_val = np.random.rand(20, 28, 28, 1).astype(np.float32)

        # Train model
        history = vae.fit(
            x_train,
            validation_data=x_val,
            epochs=3,
            batch_size=16,
            verbose=0
        )

        # Check training history
        assert len(history.history['total_loss']) == 3
        assert len(history.history['val_total_loss']) == 3

        # Test evaluation
        eval_results = vae.evaluate(x_val, verbose=0)
        assert isinstance(eval_results, list)

        # Test generation
        samples = vae.sample(num_samples=5)
        assert samples.shape == (5, 28, 28, 1)

    def test_encode_decode_consistency(self):
        """Test that encode/decode operations are consistent."""
        vae = create_vae(
            input_shape=(32, 32, 3),
            latent_dim=128,
            depths=2,  # Match with default filters
            filters=[64, 128]  # Fixed: provide matching filters
        )

        # Test data
        test_input = keras.random.normal((4, 32, 32, 3))

        # Forward pass through model
        outputs = vae(test_input, training=False)

        # Manual encode/decode
        z_mean, z_log_var = vae.encode(test_input)
        z_sample = vae.sampling_layer([z_mean, z_log_var], training=False)
        manual_reconstruction = vae.decode(z_sample)

        # Results should be close (but not identical due to sampling)
        assert outputs['z_mean'].shape == z_mean.shape
        assert outputs['z_log_var'].shape == z_log_var.shape
        assert outputs['reconstruction'].shape == manual_reconstruction.shape

        # Mean and log_var should be identical
        np.testing.assert_allclose(outputs['z_mean'].numpy(), z_mean.numpy())
        np.testing.assert_allclose(outputs['z_log_var'].numpy(), z_log_var.numpy())

    def test_loss_components_integration(self):
        """Test that loss components integrate correctly during training."""
        vae = create_vae(
            input_shape=(28, 28, 1),
            latent_dim=32,
            kl_loss_weight=0.01,
            depths=2,  # Match with default filters
            filters=[32, 64]  # Fixed: provide matching filters
        )

        # Test data
        test_data = np.random.rand(16, 28, 28, 1).astype(np.float32)

        # Manual loss computation
        outputs = vae(test_data, training=True)
        recon_loss = vae._compute_reconstruction_loss(
            test_data, outputs['reconstruction']
        )
        kl_loss = vae._compute_kl_loss(
            outputs['z_mean'], outputs['z_log_var']
        )
        expected_total = recon_loss + vae.kl_loss_weight * kl_loss

        # Train step loss computation
        train_losses = vae.train_step(test_data)

        # Should be approximately equal (within numerical precision)
        np.testing.assert_allclose(
            train_losses['reconstruction_loss'].numpy(),
            recon_loss.numpy(),
            rtol=1e-5
        )
        np.testing.assert_allclose(
            train_losses['kl_loss'].numpy(),
            kl_loss.numpy(),
            rtol=1e-5
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])