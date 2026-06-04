"""
Comprehensive test suite for the Sampling layer.

This module contains all tests for the Sampling layer implementation,
covering initialization, forward pass, serialization, model integration,
and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os
from typing import Tuple, List

from dl_techniques.layers.sampling import (
    Sampling,
    HypersphereSampling,
    VMFSampling,
    create_sampling_layer,
    create_sampling_from_config,
    validate_sampling_config,
    get_sampling_info,
    SAMPLING_REGISTRY,
    vmf_kl_divergence,
)


class TestSampling:
    """Test suite for Sampling layer implementation."""

    @pytest.fixture
    def input_shapes(self) -> Tuple[Tuple, Tuple]:
        """Create test input shapes for 2D tensors."""
        return ((None, 10), (None, 10))

    @pytest.fixture
    def input_shapes_3d(self) -> Tuple[Tuple, Tuple]:
        """Create test input shapes for 3D tensors."""
        return ((None, 5, 8), (None, 5, 8))

    @pytest.fixture
    def input_tensors(self):
        """Create test input tensors."""
        batch_size = 4
        latent_dim = 10
        z_mean = tf.random.normal([batch_size, latent_dim])
        z_log_var = tf.random.normal([batch_size, latent_dim])
        return z_mean, z_log_var

    @pytest.fixture
    def input_tensors_3d(self):
        """Create test input tensors with 3D shape."""
        batch_size = 4
        height = 5
        width = 8
        z_mean = tf.random.normal([batch_size, height, width])
        z_log_var = tf.random.normal([batch_size, height, width])
        return z_mean, z_log_var

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = Sampling()

        assert layer.seed is None
        assert layer.built is False

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = Sampling(seed=42, name="test_sampling")

        assert layer.seed == 42
        assert layer.name == "test_sampling"

    def test_initialization_with_kwargs(self):
        """Test initialization with additional kwargs."""
        layer = Sampling(
            seed=123,
            name="custom_sampling",
            trainable=False,
            dtype="float32"
        )

        assert layer.seed == 123
        assert layer.name == "custom_sampling"
        assert layer.trainable is False
        assert layer.dtype == "float32"

    def test_build_process(self, input_shapes):
        """Test that the layer builds properly."""
        layer = Sampling()
        layer.build(input_shapes)

        assert layer.built is True

    def test_build_with_3d_tensors(self, input_shapes_3d):
        """Test build with 3D input tensors."""
        layer = Sampling()
        layer.build(input_shapes_3d)

        assert layer.built is True

    def test_build_invalid_num_inputs(self):
        """Test build with wrong number of inputs."""
        layer = Sampling()

        # Test with only one input
        with pytest.raises(ValueError, match="expects exactly 2 inputs"):
            layer.build(((None, 10),))

        # Test with three inputs
        with pytest.raises(ValueError, match="expects exactly 2 inputs"):
            layer.build(((None, 10), (None, 10), (None, 10)))

    def test_build_mismatched_shapes(self):
        """Test build with mismatched input shapes."""
        layer = Sampling()

        # Test with different latent dimensions
        with pytest.raises(ValueError, match="must have the same shape"):
            layer.build(((None, 10), (None, 5)))

        # Test with different spatial dimensions for 3D tensors
        with pytest.raises(ValueError, match="must have the same shape"):
            layer.build(((None, 5, 8), (None, 4, 8)))

    def test_build_different_num_dimensions(self):
        """Test build with different number of dimensions."""
        layer = Sampling()

        # Test 2D vs 3D tensors
        with pytest.raises(ValueError, match="same number of dimensions"):
            layer.build(((None, 10), (None, 5, 2)))

    def test_forward_pass_2d(self, input_tensors):
        """Test forward pass with 2D tensors."""
        layer = Sampling(seed=42)
        z_mean, z_log_var = input_tensors

        # Test forward pass
        output = layer([z_mean, z_log_var])

        # Check output properties
        assert output.shape == z_mean.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_3d(self, input_tensors_3d):
        """Test forward pass with 3D tensors."""
        layer = Sampling(seed=42)
        z_mean, z_log_var = input_tensors_3d

        # Test forward pass
        output = layer([z_mean, z_log_var])

        # Check output properties
        assert output.shape == z_mean.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_call_invalid_num_inputs(self, input_tensors):
        """Test call with wrong number of inputs."""
        layer = Sampling()
        z_mean, _ = input_tensors

        # Test with only one input - error occurs during build
        with pytest.raises(ValueError, match="expects exactly 2 inputs"):
            layer([z_mean])

        # Test with three inputs - error occurs during build
        with pytest.raises(ValueError, match="expects exactly 2 inputs"):
            layer([z_mean, z_mean, z_mean])

    def test_call_invalid_num_inputs_after_build(self, input_tensors, input_shapes):
        """Test call with wrong number of inputs after layer is already built."""
        layer = Sampling()
        z_mean, z_log_var = input_tensors

        # Build the layer first with correct inputs
        layer.build(input_shapes)

        # Now test call-time validation with wrong number of inputs
        with pytest.raises(ValueError, match="call expects exactly 2 inputs"):
            layer([z_mean])

        with pytest.raises(ValueError, match="call expects exactly 2 inputs"):
            layer([z_mean, z_log_var, z_mean])

    def test_deterministic_with_seed(self, input_tensors):
        """Test that same seed produces same output."""
        z_mean, z_log_var = input_tensors

        layer1 = Sampling(seed=42)
        layer2 = Sampling(seed=42)

        output1 = layer1([z_mean, z_log_var])
        output2 = layer2([z_mean, z_log_var])

        # Should be identical with same seed
        np.testing.assert_array_equal(output1.numpy(), output2.numpy())

    def test_different_seeds_different_outputs(self, input_tensors):
        """Test that different seeds produce different outputs."""
        z_mean, z_log_var = input_tensors

        layer1 = Sampling(seed=42)
        layer2 = Sampling(seed=123)

        output1 = layer1([z_mean, z_log_var])
        output2 = layer2([z_mean, z_log_var])

        # Should be different with different seeds
        assert not np.allclose(output1.numpy(), output2.numpy())

    def test_no_seed_different_outputs(self, input_tensors):
        """Test that no seed produces different outputs on multiple calls."""
        z_mean, z_log_var = input_tensors
        layer = Sampling()  # No seed

        output1 = layer([z_mean, z_log_var])
        output2 = layer([z_mean, z_log_var])

        # Should be different without seed
        assert not np.allclose(output1.numpy(), output2.numpy())

    def test_output_shapes_2d(self, input_shapes, input_tensors):
        """Test that output shapes are computed correctly for 2D tensors."""
        layer = Sampling()
        z_mean, z_log_var = input_tensors

        # Test compute_output_shape
        computed_shape = layer.compute_output_shape(input_shapes)
        assert computed_shape == input_shapes[0]

        # Test actual output shape
        output = layer([z_mean, z_log_var])
        assert output.shape == z_mean.shape

    def test_output_shapes_3d(self, input_shapes_3d, input_tensors_3d):
        """Test that output shapes are computed correctly for 3D tensors."""
        layer = Sampling()
        z_mean, z_log_var = input_tensors_3d

        # Test compute_output_shape
        computed_shape = layer.compute_output_shape(input_shapes_3d)
        assert computed_shape == input_shapes_3d[0]

        # Test actual output shape
        output = layer([z_mean, z_log_var])
        assert output.shape == z_mean.shape

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        layer = Sampling(seed=42)
        latent_dim = 5

        for batch_size in [1, 4, 16, 32]:
            z_mean = tf.random.normal([batch_size, latent_dim])
            z_log_var = tf.random.normal([batch_size, latent_dim])

            output = layer([z_mean, z_log_var])
            assert output.shape == (batch_size, latent_dim)

    def test_different_latent_dimensions(self):
        """Test with different latent dimensions."""
        layer = Sampling(seed=42)
        batch_size = 4

        for latent_dim in [1, 2, 10, 50, 128]:
            z_mean = tf.random.normal([batch_size, latent_dim])
            z_log_var = tf.random.normal([batch_size, latent_dim])

            output = layer([z_mean, z_log_var])
            assert output.shape == (batch_size, latent_dim)

    def test_serialization(self, input_shapes):
        """Test serialization and deserialization of the layer."""
        original_layer = Sampling(seed=42, name="test_sampling")
        original_layer.build(input_shapes)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = Sampling.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.seed == original_layer.seed
        assert recreated_layer.name == original_layer.name
        assert recreated_layer.built == original_layer.built

    def test_serialization_no_seed(self, input_shapes):
        """Test serialization with no seed specified."""
        original_layer = Sampling(name="no_seed_sampling")
        original_layer.build(input_shapes)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = Sampling.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.seed is None
        assert recreated_layer.name == original_layer.name

    def test_model_integration(self, input_tensors):
        """Test the layer in a model context."""
        z_mean, z_log_var = input_tensors
        latent_dim = z_mean.shape[-1]

        # Create a simple model with the sampling layer
        z_mean_input = keras.Input(shape=(latent_dim,), name="z_mean")
        z_log_var_input = keras.Input(shape=(latent_dim,), name="z_log_var")

        z_sample = Sampling(seed=42)([z_mean_input, z_log_var_input])
        decoder_output = keras.layers.Dense(10, activation="sigmoid")(z_sample)

        model = keras.Model(
            inputs=[z_mean_input, z_log_var_input],
            outputs=decoder_output
        )

        # Test model compilation
        model.compile(optimizer="adam", loss="mse")

        # Test prediction
        prediction = model.predict([z_mean, z_log_var], verbose=0)
        assert prediction.shape == (z_mean.shape[0], 10)

    def test_vae_encoder_decoder_integration(self):
        """Test the layer in a complete VAE-like architecture."""
        input_dim = 784  # MNIST-like input
        latent_dim = 20
        batch_size = 8

        # Create inputs
        inputs = keras.Input(shape=(input_dim,))

        # Encoder
        h = keras.layers.Dense(256, activation="relu")(inputs)
        h = keras.layers.Dense(128, activation="relu")(h)
        z_mean = keras.layers.Dense(latent_dim, name="z_mean")(h)
        z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(h)

        # Sampling
        z = Sampling(seed=42)([z_mean, z_log_var])

        # Decoder
        h_decoded = keras.layers.Dense(128, activation="relu")(z)
        h_decoded = keras.layers.Dense(256, activation="relu")(h_decoded)
        outputs = keras.layers.Dense(input_dim, activation="sigmoid")(h_decoded)

        # Create VAE model
        vae = keras.Model(inputs, outputs)
        vae.compile(optimizer="adam", loss="binary_crossentropy")

        # Test with random data
        x_test = tf.random.normal([batch_size, input_dim])
        reconstruction = vae.predict(x_test, verbose=0)

        assert reconstruction.shape == (batch_size, input_dim)
        assert np.all(reconstruction >= 0) and np.all(reconstruction <= 1)

    def test_model_save_load(self, input_tensors):
        """Test saving and loading a model with the custom layer."""
        z_mean, z_log_var = input_tensors
        latent_dim = z_mean.shape[-1]

        # Create a model with the custom layer
        z_mean_input = keras.Input(shape=(latent_dim,), name="z_mean")
        z_log_var_input = keras.Input(shape=(latent_dim,), name="z_log_var")

        z_sample = Sampling(seed=42, name="sampling_layer")([z_mean_input, z_log_var_input])
        outputs = keras.layers.Dense(5, activation="linear")(z_sample)

        model = keras.Model(
            inputs=[z_mean_input, z_log_var_input],
            outputs=outputs
        )

        # Generate a prediction before saving
        original_prediction = model.predict([z_mean, z_log_var], verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"Sampling": Sampling}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict([z_mean, z_log_var], verbose=0)

            # Check predictions match (with same seed they should be identical)
            np.testing.assert_array_equal(original_prediction, loaded_prediction)

            # Check layer type is preserved
            sampling_layer = loaded_model.get_layer("sampling_layer")
            assert isinstance(sampling_layer, Sampling)
            assert sampling_layer.seed == 42

    def test_gradient_flow(self, input_tensors):
        """Test gradient flow through the layer."""
        z_mean, z_log_var = input_tensors
        layer = Sampling()

        # Watch the variables
        with tf.GradientTape() as tape:
            z_mean_var = tf.Variable(z_mean)
            z_log_var_var = tf.Variable(z_log_var)

            outputs = layer([z_mean_var, z_log_var_var])
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, [z_mean_var, z_log_var_var])

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check gradients have values (not all zeros)
        assert all(np.any(g.numpy() != 0) for g in grads)

    def test_training_inference_consistency(self, input_tensors):
        """Test that layer behaves consistently in training and inference modes."""
        z_mean, z_log_var = input_tensors
        layer = Sampling(seed=42)

        # Test in training mode
        training_output = layer([z_mean, z_log_var], training=True)

        # Reset layer and test in inference mode
        layer_inference = Sampling(seed=42)
        inference_output = layer_inference([z_mean, z_log_var], training=False)

        # With the same seed, outputs should be identical
        np.testing.assert_array_equal(training_output.numpy(), inference_output.numpy())

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = Sampling(seed=42)
        batch_size = 2
        latent_dim = 4

        test_cases = [
            # Very small values
            (tf.ones((batch_size, latent_dim)) * 1e-10,
             tf.ones((batch_size, latent_dim)) * -20),  # Very small log_var
            # Very large mean values
            (tf.ones((batch_size, latent_dim)) * 1e5,
             tf.ones((batch_size, latent_dim)) * 1),    # Large mean, normal log_var
            # Very large log_var values
            (tf.zeros((batch_size, latent_dim)),
             tf.ones((batch_size, latent_dim)) * 10),   # Normal mean, large log_var
            # Mixed extreme values
            (tf.random.normal((batch_size, latent_dim)) * 1e3,
             tf.random.normal((batch_size, latent_dim)) * 5)
        ]

        for i, (z_mean_test, z_log_var_test) in enumerate(test_cases):
            output = layer([z_mean_test, z_log_var_test])

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), f"NaN values detected in test case {i}"
            assert not np.any(np.isinf(output.numpy())), f"Inf values detected in test case {i}"

    def test_reparameterization_properties(self):
        """Test mathematical properties of the reparameterization trick."""
        batch_size = 4
        latent_dim = 10

        # Use controlled inputs with bounded z_log_var to ensure
        # statistical convergence with finite samples
        tf.random.set_seed(123)
        z_mean = tf.random.normal([batch_size, latent_dim])
        z_log_var = tf.clip_by_value(
            tf.random.normal([batch_size, latent_dim]), -2.0, 2.0
        )

        # Use a layer without seed for true randomness
        layer = Sampling()

        # Generate multiple samples
        num_samples = 5000
        samples = []
        for _ in range(num_samples):
            sample = layer([z_mean, z_log_var])
            samples.append(sample.numpy())

        samples = np.array(samples)  # Shape: (num_samples, batch_size, latent_dim)

        # Check sample statistics (approximate due to finite sampling)
        sample_mean = np.mean(samples, axis=0)
        sample_var = np.var(samples, axis=0)
        expected_var = np.exp(z_log_var.numpy())

        # Mean should be close to z_mean (with some tolerance due to sampling)
        np.testing.assert_allclose(sample_mean, z_mean.numpy(), atol=0.3)

        # Variance should be close to exp(z_log_var) (with some tolerance)
        np.testing.assert_allclose(sample_var, expected_var, rtol=0.3)

    def test_zero_variance_case(self):
        """Test behavior when log variance is very negative (near zero variance)."""
        batch_size = 4
        latent_dim = 5

        layer = Sampling(seed=42)

        z_mean = tf.random.normal([batch_size, latent_dim])
        z_log_var = tf.ones([batch_size, latent_dim]) * -20  # Very small variance

        output = layer([z_mean, z_log_var])

        # Output should be close to the mean when variance is very small
        np.testing.assert_allclose(output.numpy(), z_mean.numpy(), atol=1e-4)

    def test_list_inputs(self, input_tensors):
        """Test that the layer works with list inputs as well as tuples."""
        z_mean, z_log_var = input_tensors
        layer = Sampling(seed=42)

        # Test with list input
        output_list = layer([z_mean, z_log_var])

        # Test with tuple input
        output_tuple = layer((z_mean, z_log_var))

        # Should produce the same result
        np.testing.assert_array_equal(output_list.numpy(), output_tuple.numpy())

    def test_multiple_calls_same_layer(self, input_tensors):
        """Test multiple calls to the same layer instance."""
        z_mean, z_log_var = input_tensors
        layer = Sampling(seed=42)

        # Multiple calls should work without issues
        for _ in range(5):
            output = layer([z_mean, z_log_var])
            assert output.shape == z_mean.shape
            assert not np.any(np.isnan(output.numpy()))


class TestHypersphereSampling:
    """Test suite for HypersphereSampling thin-shell sampler implementation."""

    @pytest.fixture
    def input_shapes(self) -> Tuple[Tuple, Tuple]:
        """Create test input shapes: z_mean [B, D], z_log_var [B, 1]."""
        return ((None, 10), (None, 1))

    @pytest.fixture
    def input_tensors(self):
        """Create test input tensors: z_mean [B, D], z_log_var [B, 1]."""
        batch_size = 4
        latent_dim = 10
        z_mean = tf.random.normal([batch_size, latent_dim])
        z_log_var = tf.random.normal([batch_size, 1])
        return z_mean, z_log_var

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        layer = HypersphereSampling()

        assert layer.radius == 1.0
        assert layer.shell_thickness == 0.1
        assert layer.seed is None
        assert layer.built is False

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        layer = HypersphereSampling(
            radius=2.0, shell_thickness=0.25, seed=42, name="test_hsphere"
        )

        assert layer.radius == 2.0
        assert layer.shell_thickness == 0.25
        assert layer.seed == 42
        assert layer.name == "test_hsphere"

    def test_invalid_radius(self):
        """Test that non-positive radius raises ValueError."""
        with pytest.raises(ValueError, match="radius must be > 0"):
            HypersphereSampling(radius=0)

        with pytest.raises(ValueError, match="radius must be > 0"):
            HypersphereSampling(radius=-1)

    def test_invalid_shell_thickness(self):
        """Test that non-positive shell_thickness raises ValueError."""
        with pytest.raises(ValueError, match="shell_thickness must be > 0"):
            HypersphereSampling(shell_thickness=0)

        with pytest.raises(ValueError, match="shell_thickness must be > 0"):
            HypersphereSampling(shell_thickness=-0.1)

    def test_invalid_seed(self):
        """Test that a non-int, non-None seed raises TypeError."""
        with pytest.raises(TypeError, match="seed must be an integer or None"):
            HypersphereSampling(seed=1.5)

    def test_build_validation(self):
        """Test build-time shape validation."""
        # Wrong number of inputs
        with pytest.raises(ValueError, match="expects exactly 2 inputs"):
            HypersphereSampling().build(((None, 10),))

        # Rank-1 inputs (not at least 2D)
        with pytest.raises(ValueError, match="at least 2D"):
            HypersphereSampling().build(((10,), (1,)))

        # z_log_var last dim != 1
        with pytest.raises(ValueError, match="last dimension must be exactly 1"):
            HypersphereSampling().build(((None, 5), (None, 2)))

    def test_output_shape(self):
        """Test forward pass output shape and compute_output_shape."""
        layer = HypersphereSampling(seed=42)
        z_mean = tf.random.normal([8, 5])
        z_log_var = tf.random.normal([8, 1])

        output = layer([z_mean, z_log_var])

        assert output.shape == (8, 5)
        assert np.all(np.isfinite(output.numpy()))

        computed_shape = layer.compute_output_shape(((8, 5), (8, 1)))
        assert computed_shape == (8, 5)

    def test_shell_radius(self):
        """Thin, strictly-positive shell -> ``||z|| > 0`` and mean ~ radius.

        With the default ``shell_thickness=0.1`` and ``z_log_var=0`` the
        per-sample radius is ``radius * (1 + 0.1 * eta)`` floored at
        ``radius * 0.05``. The core fix is that ``||z||`` is ALWAYS strictly
        positive (no antipode flips / no origin samples); the mean stays close
        to ``radius`` (loose: within ``0.3 * radius``).
        """
        radius = 2.0
        layer = HypersphereSampling(radius=radius, seed=42)  # default thickness 0.1

        n = 2000
        latent_dim = 6
        z_mean = tf.random.normal([n, latent_dim])
        z_log_var = tf.zeros([n, 1])  # log_var = 0 -> shell std = 0.1 * radius

        output = layer([z_mean, z_log_var]).numpy()

        norms = np.linalg.norm(output, axis=-1)
        assert not np.any(np.isnan(output))
        # Core fix: strictly positive radius (no zero / negative / origin).
        assert np.min(norms) > 0.0
        # Mean concentrates near the configured radius (loose tolerance).
        np.testing.assert_allclose(np.mean(norms), radius, atol=0.3 * radius)

    def test_radius_strictly_positive(self):
        """Large positive z_log_var must NOT produce zero/negative radii.

        This directly locks the bug fix: under the old
        ``radius + exp(0.5*lv)*eta`` shell a large ``log_var`` made the shell
        std blow past the radius, flipping ~10% of samples to a negative radius
        and dropping ~15% at/near the origin. With the thin, floored,
        multiplicative shell ``||z||`` is always > 0 regardless of ``log_var``.
        """
        radius = 1.0
        layer = HypersphereSampling(radius=radius, seed=11)

        n = 2000
        latent_dim = 8
        z_mean = tf.random.normal([n, latent_dim])
        z_log_var = tf.ones([n, 1]) * 5.0  # large variance -> stress the shell

        output = layer([z_mean, z_log_var]).numpy()

        norms = np.linalg.norm(output, axis=-1)
        assert np.all(np.isfinite(output))
        assert np.min(norms) > 0.0

    def test_degenerate_direction_e0(self):
        """Robust finiteness + norm~radius for the degenerate (zero-mean) case.

        The exact ``g == 0`` -> ``e_0`` branch is a measure-zero path (g =
        z_mean + eps with eps ~ N(0, I) is a.s. non-zero) and is covered by the
        Step-1 empirical check. Here we assert the observable robust property:
        with ``z_mean = 0`` the output stays finite (no NaN) across many draws
        and concentrates at ``radius`` (since normalize(eps) is a unit vector
        and the shell is thin), exercising the same normalize-with-e_0 code path
        without faking an exact-zero direction.
        """
        radius = 3.0
        layer = HypersphereSampling(radius=radius, seed=7)

        n = 1000
        latent_dim = 4
        z_mean = tf.zeros([n, latent_dim])
        z_log_var = tf.ones([n, 1]) * -20.0

        output = layer([z_mean, z_log_var]).numpy()

        assert np.all(np.isfinite(output))
        norms = np.linalg.norm(output, axis=-1)
        np.testing.assert_allclose(np.mean(norms), radius, atol=0.1)

    def test_deterministic_with_seed(self, input_tensors):
        """Test that same seed produces identical output."""
        z_mean, z_log_var = input_tensors

        layer1 = HypersphereSampling(seed=42)
        layer2 = HypersphereSampling(seed=42)

        output1 = layer1([z_mean, z_log_var])
        output2 = layer2([z_mean, z_log_var])

        np.testing.assert_array_equal(output1.numpy(), output2.numpy())

    def test_gradient_flow(self, input_tensors):
        """Test gradient flow through the layer to both inputs."""
        z_mean, z_log_var = input_tensors
        layer = HypersphereSampling(seed=42)

        with tf.GradientTape() as tape:
            z_mean_var = tf.Variable(z_mean)
            z_log_var_var = tf.Variable(z_log_var)

            outputs = layer([z_mean_var, z_log_var_var])
            loss = tf.reduce_sum(tf.square(outputs))

        grads = tape.gradient(loss, [z_mean_var, z_log_var_var])

        # Both gradients exist and are not None
        assert all(g is not None for g in grads)

        # Both gradients have values (not all zeros)
        assert all(np.any(g.numpy() != 0) for g in grads)

    def test_serialization(self, input_shapes):
        """Test serialization and deserialization of the layer."""
        original_layer = HypersphereSampling(
            radius=2.0, shell_thickness=0.2, seed=42, name="test_hsphere"
        )
        original_layer.build(input_shapes)

        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        assert config["radius"] == 2.0
        assert config["shell_thickness"] == 0.2
        assert config["seed"] == 42

        recreated_layer = HypersphereSampling.from_config(config)
        recreated_layer.build_from_config(build_config)

        assert recreated_layer.radius == original_layer.radius
        assert recreated_layer.shell_thickness == original_layer.shell_thickness
        assert recreated_layer.seed == original_layer.seed
        assert recreated_layer.name == original_layer.name
        assert recreated_layer.built == original_layer.built

    def test_model_save_load(self, tmp_path):
        """Test saving and loading a model with the custom layer."""
        batch_size = 4
        latent_dim = 10
        z_mean = tf.random.normal([batch_size, latent_dim])
        z_log_var = tf.random.normal([batch_size, 1])

        # Create a model with the custom layer
        z_mean_input = keras.Input(shape=(latent_dim,), name="z_mean")
        z_log_var_input = keras.Input(shape=(1,), name="z_log_var")

        z_sample = HypersphereSampling(
            radius=2.0, seed=42, name="hsphere_layer"
        )([z_mean_input, z_log_var_input])
        outputs = keras.layers.Dense(5, activation="linear")(z_sample)

        model = keras.Model(
            inputs=[z_mean_input, z_log_var_input],
            outputs=outputs
        )

        original_prediction = model.predict([z_mean, z_log_var], verbose=0)

        model_path = os.path.join(str(tmp_path), "model.keras")
        model.save(model_path)

        loaded_model = keras.models.load_model(
            model_path,
            custom_objects={"HypersphereSampling": HypersphereSampling}
        )

        loaded_prediction = loaded_model.predict([z_mean, z_log_var], verbose=0)

        # With the same seed, predictions should be identical
        np.testing.assert_array_equal(original_prediction, loaded_prediction)

        hsphere_layer = loaded_model.get_layer("hsphere_layer")
        assert isinstance(hsphere_layer, HypersphereSampling)
        assert hsphere_layer.radius == 2.0
        assert hsphere_layer.seed == 42


class TestSamplingFactory:
    """Test suite for the inline registry-driven sampling factory."""

    def test_create_gaussian(self):
        """create_sampling_layer('gaussian') returns a Sampling layer."""
        layer = create_sampling_layer("gaussian")

        assert isinstance(layer, Sampling)

    def test_create_hypersphere_with_radius(self):
        """create_sampling_layer('hypersphere', radius=2.0) honors radius."""
        layer = create_sampling_layer("hypersphere", radius=2.0)

        assert isinstance(layer, HypersphereSampling)
        assert layer.radius == 2.0

    def test_create_unknown_type_raises(self):
        """Unknown type raises ValueError listing the available types."""
        with pytest.raises(ValueError) as exc_info:
            create_sampling_layer("nope")

        message = str(exc_info.value)
        assert "gaussian" in message
        assert "hypersphere" in message

    def test_create_hypersphere_zero_radius_raises(self):
        """radius=0 is rejected by validate_sampling_config (radius > 0)."""
        with pytest.raises(ValueError):
            create_sampling_layer("hypersphere", radius=0)

    def test_create_gaussian_seed_passthrough(self):
        """seed kwarg flows through to the constructed Sampling layer."""
        layer = create_sampling_layer("gaussian", seed=42)

        assert isinstance(layer, Sampling)
        assert layer.seed == 42

    def test_create_name_injection(self):
        """name= is injected into the constructed layer."""
        layer = create_sampling_layer("gaussian", name="my_sampler")

        assert layer.name == "my_sampler"

    def test_create_from_config_hypersphere(self):
        """create_sampling_from_config builds from a dict with 'type'."""
        layer = create_sampling_from_config(
            {"type": "hypersphere", "radius": 3.0, "seed": 7}
        )

        assert isinstance(layer, HypersphereSampling)
        assert layer.radius == 3.0
        assert layer.seed == 7

    def test_create_from_config_missing_type_raises(self):
        """A config dict without a 'type' key raises ValueError."""
        with pytest.raises(ValueError, match="'type'"):
            create_sampling_from_config({"radius": 2.0})

    def test_get_sampling_info_keys(self):
        """get_sampling_info returns metadata for all registered types."""
        info = get_sampling_info()

        assert set(info.keys()) == {"gaussian", "hypersphere", "vmf"}
        assert info["gaussian"]["class"] is Sampling
        assert info["hypersphere"]["class"] is HypersphereSampling
        assert info["vmf"]["class"] is VMFSampling

    def test_get_sampling_info_is_shallow_copy(self):
        """Mutating the returned dict does NOT mutate the module registry."""
        info = get_sampling_info()
        info.pop("gaussian")

        # The module-level registry must be unaffected.
        assert "gaussian" in SAMPLING_REGISTRY
        assert "gaussian" in get_sampling_info()

    def test_validate_sampling_config_ok(self):
        """A valid config validates without raising and returns None."""
        assert validate_sampling_config("hypersphere", radius=1.0) is None

    def test_validate_sampling_config_unknown_raises(self):
        """validate_sampling_config rejects an unknown type."""
        with pytest.raises(ValueError, match="Unknown sampling type"):
            validate_sampling_config("nope")


class TestVMFSampling:
    """Test suite for the VMFSampling Wood-rejection sampler (SC2/SC3)."""

    @pytest.fixture
    def input_tensors(self):
        """Create test input tensors: z_mean [B, D], kappa [B, 1]."""
        batch_size = 4
        latent_dim = 8
        z_mean = tf.random.normal([batch_size, latent_dim])
        kappa = tf.ones([batch_size, 1]) * 10.0
        return z_mean, kappa

    def test_init_and_validation(self):
        """Default ctor ok; bad rejection_oversample / seed types raise."""
        layer = VMFSampling()
        assert layer.rejection_oversample == 32
        assert layer.seed is None
        assert layer.built is False

        layer2 = VMFSampling(rejection_oversample=16, seed=42, name="vmf")
        assert layer2.rejection_oversample == 16
        assert layer2.seed == 42
        assert layer2.name == "vmf"

        with pytest.raises(ValueError, match="rejection_oversample must be"):
            VMFSampling(rejection_oversample=0)
        with pytest.raises(ValueError, match="rejection_oversample must be"):
            VMFSampling(rejection_oversample=-4)
        with pytest.raises(ValueError, match="rejection_oversample must be"):
            VMFSampling(rejection_oversample=8.0)
        with pytest.raises(TypeError, match="seed must be an integer or None"):
            VMFSampling(seed=1.5)

    def test_build_validation(self):
        """kappa last-dim != 1 raises; D < 2 (z_mean last-dim 1) raises."""
        with pytest.raises(ValueError, match="expects exactly 2 inputs"):
            VMFSampling().build(((None, 8),))

        with pytest.raises(ValueError, match="last dimension must be exactly 1"):
            VMFSampling().build(((None, 8), (None, 2)))

        with pytest.raises(ValueError, match="latent_dim.*>= 2"):
            VMFSampling().build(((None, 1), (None, 1)))

    def test_output_shape(self):
        """Output shape == z_mean shape; compute_output_shape mirrors it."""
        layer = VMFSampling(seed=42)
        z_mean = tf.random.normal([8, 5])
        kappa = tf.ones([8, 1]) * 5.0

        output = layer([z_mean, kappa])
        assert output.shape == (8, 5)
        assert np.all(np.isfinite(output.numpy()))

        assert layer.compute_output_shape(((8, 5), (8, 1))) == (8, 5)

    def test_unit_sphere_output(self):
        """SC2: ||z|| within 1e-5 of 1.0 for all rows, multiple kappa and D."""
        for D in (2, 8, 16):
            for k in (1.0, 10.0, 100.0):
                layer = VMFSampling(seed=123)
                n = 256
                z_mean = tf.random.normal([n, D])
                kappa = tf.ones([n, 1]) * k
                out = layer([z_mean, kappa]).numpy()
                norms = np.linalg.norm(out, axis=-1)
                assert np.all(np.isfinite(out)), f"non-finite at D={D}, k={k}"
                np.testing.assert_allclose(
                    norms, 1.0, atol=1e-5,
                    err_msg=f"off-sphere at D={D}, k={k}",
                )

    def test_high_kappa_concentrates_near_mean(self):
        """SC2: kappa=1000 -> mean(z . mu_hat) > 0.99."""
        D = 8
        n = 512
        layer = VMFSampling(seed=7)
        z_mean = tf.random.normal([n, D])
        kappa = tf.ones([n, 1]) * 1000.0

        out = layer([z_mean, kappa]).numpy()
        mu_hat = z_mean.numpy()
        mu_hat = mu_hat / np.linalg.norm(mu_hat, axis=-1, keepdims=True)
        align = np.sum(out * mu_hat, axis=-1)
        assert np.mean(align) > 0.99, f"mean alignment {np.mean(align):.4f}"

    def test_low_kappa_spreads(self):
        """Low kappa (~0.1) -> small mean resultant length (not concentrated)."""
        D = 8
        n = 2000
        layer = VMFSampling(seed=3)
        z_mean = tf.random.normal([n, D])
        kappa = tf.ones([n, 1]) * 0.1

        out = layer([z_mean, kappa]).numpy()
        mu_hat = z_mean.numpy()
        mu_hat = mu_hat / np.linalg.norm(mu_hat, axis=-1, keepdims=True)
        align = np.sum(out * mu_hat, axis=-1)
        # Near-uniform: mean alignment should be far from 1 (loose sanity bound).
        assert np.mean(align) < 0.5, f"mean alignment {np.mean(align):.4f}"

    def test_sampler_mean_matches_bessel_ratio(self):
        """Strong cross-check: empirical mean(z . mu_hat) ~ A_m(k) = ratio.

        Ties the sampler to the KL's Bessel ratio: for vMF(mu, k) on S^{m-1}
        the expected resultant E[z . mu] = A_{m/2}(k) = I_{m/2}(k)/I_{m/2-1}(k),
        which is exactly ``_bessel_ratio_cf(k, m/2)``.
        """
        from dl_techniques.layers.sampling import _bessel_ratio_cf

        for m, k in [(2, 5.0), (4, 10.0), (8, 20.0)]:
            n = 8000
            layer = VMFSampling(seed=99)
            # Fixed mu_hat = e_0.
            z_mean = np.zeros((n, m), dtype="float32")
            z_mean[:, 0] = 1.0
            kappa = np.ones((n, 1), dtype="float32") * k
            out = layer([tf.constant(z_mean), tf.constant(kappa)]).numpy()
            emp = float(np.mean(out[:, 0]))  # z . e_0
            ref = float(_bessel_ratio_cf(tf.constant([[k]]), m / 2.0).numpy()[0, 0])
            assert abs(emp - ref) < 0.03, (
                f"m={m}, k={k}: empirical {emp:.4f} vs A_m {ref:.4f}"
            )

    def test_gradient_flow(self, input_tensors):
        """SC3: gradients to BOTH z_mean and kappa are not-None and non-zero."""
        z_mean, kappa = input_tensors
        layer = VMFSampling(seed=42)

        with tf.GradientTape() as tape:
            z_mean_var = tf.Variable(z_mean)
            kappa_var = tf.Variable(kappa)
            outputs = layer([z_mean_var, kappa_var])
            loss = tf.reduce_sum(tf.square(outputs))

        grads = tape.gradient(loss, [z_mean_var, kappa_var])

        assert all(g is not None for g in grads)
        assert all(np.any(g.numpy() != 0) for g in grads)

    def test_get_config(self):
        """get_config / from_config round-trip preserves all parameters."""
        original = VMFSampling(
            rejection_oversample=16, seed=42, name="vmf_test"
        )
        original.build(((None, 8), (None, 1)))

        config = original.get_config()
        assert config["rejection_oversample"] == 16
        assert config["seed"] == 42

        recreated = VMFSampling.from_config(config)
        assert recreated.rejection_oversample == 16
        assert recreated.seed == 42
        assert recreated.name == "vmf_test"

    def test_model_save_load(self, tmp_path):
        """seed=42 determinism: saved+reloaded model gives identical output."""
        batch_size = 4
        latent_dim = 8
        z_mean = tf.random.normal([batch_size, latent_dim])
        kappa = tf.ones([batch_size, 1]) * 10.0

        z_mean_input = keras.Input(shape=(latent_dim,), name="z_mean")
        kappa_input = keras.Input(shape=(1,), name="kappa")

        z_sample = VMFSampling(seed=42, name="vmf_layer")(
            [z_mean_input, kappa_input]
        )
        outputs = keras.layers.Dense(5, activation="linear")(z_sample)

        model = keras.Model(
            inputs=[z_mean_input, kappa_input], outputs=outputs
        )

        original_prediction = model.predict([z_mean, kappa], verbose=0)

        model_path = os.path.join(str(tmp_path), "model.keras")
        model.save(model_path)

        loaded_model = keras.models.load_model(
            model_path, custom_objects={"VMFSampling": VMFSampling}
        )
        loaded_prediction = loaded_model.predict([z_mean, kappa], verbose=0)

        # Stochastic by design but seed-reproducible -> identical predictions.
        np.testing.assert_array_equal(original_prediction, loaded_prediction)

        vmf_layer = loaded_model.get_layer("vmf_layer")
        assert isinstance(vmf_layer, VMFSampling)
        assert vmf_layer.seed == 42

    def test_factory_create_vmf(self):
        """create_sampling_layer('vmf') returns a VMFSampling layer."""
        layer = create_sampling_layer("vmf")
        assert isinstance(layer, VMFSampling)

    def test_factory_seed_passthrough(self):
        """seed + rejection_oversample flow through the factory."""
        layer = create_sampling_layer("vmf", seed=42, rejection_oversample=16)
        assert isinstance(layer, VMFSampling)
        assert layer.seed == 42
        assert layer.rejection_oversample == 16


class TestVMFNumerics:
    """Gate tests for the vMF KL numerics helpers (SC1).

    Validates ``vmf_kl_divergence`` against a ``scipy.special.ive``-based
    reference KL across an (m, kappa) grid spanning even AND odd latent dims.
    This is a HARD gate: it MUST pass before any vMF training.
    """

    @staticmethod
    def _scipy_vmf_kl(m: int, k: float) -> float:
        """Reference KL(vMF(mu, k) || Uniform(S^{m-1})) via scipy."""
        import math as _math
        from scipy.special import ive, gammaln

        nu = m / 2.0
        # A_{m/2}(k) = I_{m/2}(k) / I_{m/2 - 1}(k); ive is exp(-|k|)-scaled,
        # so the scaling cancels in the ratio.
        A = ive(nu, k) / ive(nu - 1.0, k)
        # log I_{m/2 - 1}(k) = log(ive(nu-1, k)) + k.
        logI = _math.log(ive(nu - 1.0, k)) + k
        logCm = (
            (m / 2.0 - 1.0) * _math.log(k)
            - (m / 2.0) * _math.log(2.0 * _math.pi)
            - logI
        )
        logCm0 = (
            -_math.log(2.0)
            - (m / 2.0) * _math.log(_math.pi)
            + gammaln(m / 2.0)
        )
        return k * A + logCm - logCm0

    def test_vmf_kl_vs_scipy(self):
        """vmf_kl_divergence matches the scipy reference to < 1e-3 abserr.

        Grid: m in {2,3,5,8,15,16,17,32,33} (even AND odd) x
        kappa in {0.5, 1.0, 10.0, 50.0}. The orchestrator measured ~6e-6; the
        1e-3 gate is a safe margin (SC1).
        """
        dims = [2, 3, 5, 8, 15, 16, 17, 32, 33]
        kappas = [0.5, 1.0, 10.0, 50.0]

        max_abserr = 0.0
        worst = None
        for m in dims:
            for k in kappas:
                ref = self._scipy_vmf_kl(m, k)
                got = vmf_kl_divergence(
                    np.array([[k]], dtype=np.float32), m
                )
                got_val = float(np.asarray(got).reshape(-1)[0])

                # KL of any non-uniform vMF posterior must be non-negative.
                assert got_val >= -1e-4, (
                    f"negative KL at m={m}, kappa={k}: {got_val}"
                )
                assert np.isfinite(got_val), (
                    f"non-finite KL at m={m}, kappa={k}: {got_val}"
                )

                abserr = abs(got_val - ref)
                if abserr > max_abserr:
                    max_abserr = abserr
                    worst = (m, k, got_val, ref)

        print(
            f"\n[vmf_kl_vs_scipy] max abserr={max_abserr:.3e} "
            f"(worst m,kappa,got,ref={worst})"
        )
        assert max_abserr < 1e-3, (
            f"vMF KL abserr {max_abserr:.3e} exceeds 1e-3 gate; worst={worst}"
        )