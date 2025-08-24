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

from dl_techniques.layers.sampling import Sampling


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

    def test_reparameterization_properties(self, input_tensors):
        """Test mathematical properties of the reparameterization trick."""
        z_mean, z_log_var = input_tensors

        # Use a layer without seed for true randomness
        layer = Sampling()

        # Generate multiple samples
        num_samples = 1000
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
        np.testing.assert_allclose(sample_var, expected_var, rtol=0.2)

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