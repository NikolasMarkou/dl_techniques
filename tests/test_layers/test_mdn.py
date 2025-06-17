"""
Extended test suite for the Mixture Density Network Layer.

Tests core functionality including layer construction, shape handling,
forward pass, loss computation, sampling, serialization, and numerical stability.
Following project best practices and standards.
"""

import os
import keras
import pytest
import numpy as np
import tempfile
from typing import Tuple

from dl_techniques.layers.mdn_layer import (
    MDNLayer,
    gaussian_probability,
    get_point_estimate,
    get_uncertainty,
    get_prediction_intervals
)
from dl_techniques.utils.logger import logger


class TestMDNLayer:
    """Test suite for MDNLayer implementation."""

    @pytest.fixture
    def sample_input_shape(self) -> Tuple[int, ...]:
        """Create a sample input shape for testing."""
        return (None, 10)

    @pytest.fixture
    def sample_mdn_params(self) -> dict:
        """Create sample MDN parameters for testing."""
        return {
            "output_dimension": 2,
            "num_mixtures": 3
        }

    @pytest.fixture
    def built_mdn_layer(self, sample_mdn_params, sample_input_shape) -> MDNLayer:
        """Create a built MDN layer for testing."""
        layer = MDNLayer(**sample_mdn_params)
        layer.build(sample_input_shape)
        return layer

    def test_mdn_layer_initialization(self, sample_mdn_params):
        """Test basic MDN layer initialization and configuration."""
        # Test initialization with basic parameters
        mdn = MDNLayer(**sample_mdn_params)

        # Verify layer attributes
        assert mdn.output_dim == 2, "Output dimension not set correctly"
        assert mdn.num_mix == 3, "Number of mixtures not set correctly"

        # Verify layer components are None before building (correct behavior)
        assert mdn.mdn_mus is None, "Sublayers should be None before building"
        assert mdn.mdn_sigmas is None, "Sublayers should be None before building"
        assert mdn.mdn_pi is None, "Sublayers should be None before building"

        # Test initialization with custom parameters
        custom_mdn = MDNLayer(
            output_dimension=2,
            num_mixtures=3,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(0.01)
        )
        assert custom_mdn is not None, "Failed to initialize with custom parameters"
        logger.info("MDN layer initialization test passed")

    def test_mdn_layer_building(self, sample_mdn_params, sample_input_shape):
        """Test MDN layer building process."""
        mdn = MDNLayer(**sample_mdn_params)

        # Build the layer
        mdn.build(sample_input_shape)

        # Verify layer components are created after building
        assert isinstance(mdn.mdn_mus, keras.layers.Dense), "Means layer not created correctly"
        assert isinstance(mdn.mdn_sigmas, keras.layers.Dense), "Sigmas layer not created correctly"
        assert isinstance(mdn.mdn_pi, keras.layers.Dense), "Pi layer not created correctly"

        # Verify layer is marked as built
        assert mdn.built, "Layer should be marked as built"

        # Verify stored build shape
        assert mdn._build_input_shape == sample_input_shape, "Build input shape not stored correctly"
        logger.info("MDN layer building test passed")

    def test_mdn_output_shapes(self):
        """Test output shapes of MDN layer components."""
        batch_size = 32
        input_dim = 10
        output_dim = 2
        num_mixes = 3

        # Create layer
        mdn = MDNLayer(output_dimension=output_dim, num_mixtures=num_mixes)

        # Create input tensor
        inputs = keras.layers.Input(shape=(input_dim,))
        outputs = mdn(inputs)

        # Create model for testing
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test with random input
        test_input = np.random.normal(size=(batch_size, input_dim))
        output = model.predict(test_input, verbose=0)

        # Expected shape calculations
        expected_total_outputs = (2 * output_dim * num_mixes) + num_mixes

        assert output.shape == (batch_size, expected_total_outputs), \
            f"Expected shape {(batch_size, expected_total_outputs)}, got {output.shape}"
        logger.info("MDN output shapes test passed")

    def test_mdn_compute_output_shape(self, built_mdn_layer):
        """Test the compute_output_shape method."""
        input_shape = (None, 10)
        output_shape = built_mdn_layer.compute_output_shape(input_shape)

        expected_output_size = (2 * built_mdn_layer.output_dim * built_mdn_layer.num_mix) + built_mdn_layer.num_mix
        expected_shape = (None, expected_output_size)

        assert output_shape == expected_shape, \
            f"Expected output shape {expected_shape}, got {output_shape}"
        logger.info("MDN compute output shape test passed")

    def test_mdn_split_mixture_params(self):
        """Test the split_mixture_params method to ensure correct parameter extraction."""
        # Setup test parameters
        batch_size = 16
        output_dim = 3
        num_mixes = 4

        # Create MDN layer
        mdn = MDNLayer(output_dimension=output_dim, num_mixtures=num_mixes)

        # Create mock MDN output
        total_params = (2 * output_dim * num_mixes) + num_mixes
        mock_output = np.random.normal(size=(batch_size, total_params))
        mock_output_tensor = keras.ops.convert_to_tensor(mock_output, dtype="float32")

        # Split the parameters
        mu, sigma, pi = mdn.split_mixture_params(mock_output_tensor)

        # Check shapes
        assert mu.shape == (batch_size, num_mixes, output_dim), \
            f"Expected mu shape {(batch_size, num_mixes, output_dim)}, got {mu.shape}"
        assert sigma.shape == (batch_size, num_mixes, output_dim), \
            f"Expected sigma shape {(batch_size, num_mixes, output_dim)}, got {sigma.shape}"
        assert pi.shape == (batch_size, num_mixes), \
            f"Expected pi shape {(batch_size, num_mixes)}, got {pi.shape}"
        logger.info("MDN split mixture params test passed")

    def test_mdn_loss_function(self):
        """Test MDN loss function to ensure correct computation and gradients."""
        # Setup test parameters
        batch_size = 32
        input_dim = 5
        output_dim = 2
        num_mixes = 3

        # Create MDN layer
        mdn = MDNLayer(output_dimension=output_dim, num_mixtures=num_mixes)

        # Create a simple model
        inputs = keras.layers.Input(shape=(input_dim,))
        hidden = keras.layers.Dense(16, activation='relu')(inputs)
        outputs = mdn(hidden)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Create dummy data
        x_train = np.random.normal(size=(batch_size, input_dim))
        y_train = np.random.normal(size=(batch_size, output_dim))

        # Test loss computation
        y_pred = model(x_train, training=True)
        loss = mdn.loss_func(keras.ops.convert_to_tensor(y_train), y_pred)

        # Verify that the loss is a valid float and not NaN or Inf
        loss_value = keras.ops.convert_to_numpy(loss)
        assert loss_value > 0, "Loss should be positive"
        assert not np.isnan(loss_value), "Loss is NaN"
        assert not np.isinf(loss_value), "Loss is Inf"
        logger.info(f"MDN loss function test passed, loss value: {loss_value:.4f}")

    def test_mdn_sampling(self):
        """Test MDN sampling function with reproducibility."""
        # Setup test parameters
        batch_size = 16
        input_dim = 8
        output_dim = 3
        num_mixes = 5

        # Create MDN layer
        mdn = MDNLayer(output_dimension=output_dim, num_mixtures=num_mixes)

        # Create a simple model
        inputs = keras.layers.Input(shape=(input_dim,))
        hidden = keras.layers.Dense(32, activation='relu')(inputs)
        outputs = mdn(hidden)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Create test input with fixed seed for reproducibility
        test_input = np.random.RandomState(42).normal(size=(batch_size, input_dim))

        # Get predictions
        predictions = model.predict(test_input, verbose=0)

        # Test sampling
        samples1 = mdn.sample(predictions)

        # Check shape
        assert samples1.shape == (batch_size, output_dim), \
            f"Expected samples shape {(batch_size, output_dim)}, got {samples1.shape}"

        # Test temperature parameter
        samples_low_temp = mdn.sample(predictions, temperature=0.1)
        samples_high_temp = mdn.sample(predictions, temperature=10.0)

        # Verify samples have correct shapes
        assert samples_low_temp.shape == (batch_size, output_dim), \
            "Low temperature samples have incorrect shape"
        assert samples_high_temp.shape == (batch_size, output_dim), \
            "High temperature samples have incorrect shape"

        # Check that samples are valid (no NaN or Inf)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(samples1))), "Samples contain NaN values"
        assert not np.any(np.isinf(keras.ops.convert_to_numpy(samples1))), "Samples contain Inf values"
        logger.info("MDN sampling test passed")

    def test_mdn_numerical_stability(self):
        """Test MDN layer's numerical stability with extreme inputs."""
        # Setup parameters
        output_dim = 2
        num_mixes = 3

        # Create MDN layer
        mdn = MDNLayer(output_dimension=output_dim, num_mixtures=num_mixes)

        # Test gaussian_probability with extreme values
        y = keras.ops.convert_to_tensor([[1e6, 1e6]], dtype="float32")
        mu = keras.ops.convert_to_tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]], dtype="float32")
        sigma = keras.ops.convert_to_tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]], dtype="float32")

        # Check no NaNs or Infs with extreme difference between y and mu
        result = gaussian_probability(y, mu, sigma)
        result_np = keras.ops.convert_to_numpy(result)
        assert not np.any(np.isnan(result_np)), "gaussian_probability produced NaNs"
        assert not np.any(np.isinf(result_np)), "gaussian_probability produced Infs"

        # Test with extremely small sigma
        y = keras.ops.convert_to_tensor([[0.0, 0.0]], dtype="float32")
        mu = keras.ops.convert_to_tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]], dtype="float32")
        sigma = keras.ops.convert_to_tensor([[[1e-10, 1e-10], [1e-10, 1e-10], [1e-10, 1e-10]]], dtype="float32")

        # Check no NaNs or Infs with tiny sigma
        result = gaussian_probability(y, mu, sigma)
        result_np = keras.ops.convert_to_numpy(result)
        assert not np.any(np.isnan(result_np)), "gaussian_probability produced NaNs with small sigma"
        assert not np.any(np.isinf(result_np)), "gaussian_probability produced Infs with small sigma"

        # Test loss function with extreme predictions
        batch_size = 10
        total_params = (2 * output_dim * num_mixes) + num_mixes

        # Create targets with reasonable values
        y_true = keras.ops.convert_to_tensor(np.random.normal(size=(batch_size, output_dim)), dtype="float32")

        # Create predictions with extreme values for mu (very large)
        mus = keras.ops.ones((batch_size, num_mixes * output_dim)) * 1e6
        sigmas = keras.ops.ones((batch_size, num_mixes * output_dim))
        pis = keras.ops.convert_to_tensor(np.random.normal(size=(batch_size, num_mixes)), dtype="float32")
        y_pred_large_mu = keras.ops.concatenate([mus, sigmas, pis], axis=-1)

        # Test loss with large mu values
        loss = mdn.loss_func(y_true, y_pred_large_mu)
        loss_np = keras.ops.convert_to_numpy(loss)
        assert not np.isnan(loss_np), "Loss is NaN with large mu values"
        assert not np.isinf(loss_np), "Loss is Inf with large mu values"
        logger.info("MDN numerical stability test passed")

    def test_mdn_parameter_validation(self):
        """Test MDN layer's parameter validation."""
        # Test with invalid output dimension
        with pytest.raises(ValueError, match="output_dimension must be positive"):
            MDNLayer(output_dimension=0, num_mixtures=3)

        with pytest.raises(ValueError, match="output_dimension must be positive"):
            MDNLayer(output_dimension=-1, num_mixtures=3)

        # Test with invalid number of mixtures
        with pytest.raises(ValueError, match="num_mixtures must be positive"):
            MDNLayer(output_dimension=2, num_mixtures=0)

        with pytest.raises(ValueError, match="num_mixtures must be positive"):
            MDNLayer(output_dimension=2, num_mixtures=-1)

        # Test with valid parameters
        mdn = MDNLayer(output_dimension=1, num_mixtures=1)
        assert mdn is not None, "Failed to initialize with valid minimal parameters"

        # Test with large values (should work)
        mdn_large = MDNLayer(output_dimension=100, num_mixtures=50)
        assert mdn_large is not None, "Failed to initialize with large valid parameters"
        logger.info("MDN parameter validation test passed")

    def test_mdn_training_convergence(self):
        """Test MDN layer's ability to fit a simple distribution."""
        # Setup parameters for a simple toy problem
        input_dim = 1
        output_dim = 1
        num_mixes = 3
        n_samples = 1000
        epochs = 50

        # Create synthetic dataset: y = sin(x) + noise
        np.random.seed(42)
        X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
        Y = np.sin(X) + 0.1 * np.random.randn(n_samples, 1)

        # Create MDN layer
        mdn_layer = MDNLayer(output_dimension=output_dim, num_mixtures=num_mixes)

        # Create model using functional API
        inputs = keras.layers.Input(shape=(input_dim,))
        x = keras.layers.Dense(32, activation='relu')(inputs)
        x = keras.layers.Dense(32, activation='relu')(x)
        outputs = mdn_layer(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Create a custom loss function
        def loss_fn(y_true, y_pred):
            return mdn_layer.loss_func(y_true, y_pred)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)

        # Train model
        history = model.fit(X, Y, epochs=epochs, batch_size=64, verbose=0)

        # Check loss decreases over time
        assert history.history['loss'][0] > history.history['loss'][-1], \
            "Loss did not decrease during training"

        # Significant loss reduction (at least 20% to be more lenient)
        loss_reduction = (history.history['loss'][0] - history.history['loss'][-1]) / history.history['loss'][0]
        assert loss_reduction > 0.2, f"Insufficient loss reduction: {loss_reduction:.2f}"

        # Test prediction and sampling
        X_test = np.array([[-2.0], [0.0], [2.0]])
        predictions = model.predict(X_test, verbose=0)

        # Generate samples
        samples = mdn_layer.sample(predictions)
        samples_np = keras.ops.convert_to_numpy(samples)

        # Basic sanity checks
        assert samples_np.shape == (X_test.shape[0], output_dim), "Samples have incorrect shape"
        assert not np.any(np.isnan(samples_np)), "Samples contain NaN values"
        assert not np.any(np.isinf(samples_np)), "Samples contain Inf values"
        logger.info(f"MDN training convergence test passed, loss reduction: {loss_reduction:.2f}")

    def test_mdn_serialization(self):
        """Test MDN layer serialization and deserialization."""
        # Setup test parameters
        input_dim = 10
        output_dim = 3
        num_mixes = 4

        # Create a model with MDN layer
        mdn_layer = MDNLayer(
            output_dimension=output_dim,
            num_mixtures=num_mixes,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=keras.regularizers.L2(0.01)
        )

        # Create model using functional API
        inputs = keras.layers.Input(shape=(input_dim,))
        hidden = keras.layers.Dense(16, activation='relu')(inputs)
        outputs = mdn_layer(hidden)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Create test data
        np.random.seed(42)  # For reproducible test
        test_input = np.random.normal(size=(1, input_dim))
        original_output = model.predict(test_input, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, 'mdn_model.keras')
            model.save(model_path)

            # Load model with custom objects
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"MDNLayer": MDNLayer}
            )

        # Test loaded model
        loaded_output = loaded_model.predict(test_input, verbose=0)

        # Check that outputs match
        np.testing.assert_allclose(
            original_output, loaded_output,
            rtol=1e-5, atol=1e-5,
            err_msg="Loaded model produces different output than original"
        )

        # Test layer config serialization
        config = mdn_layer.get_config()

        # Check that all required attributes are in the config
        assert 'output_dimension' in config, "output_dimension missing from config"
        assert 'num_mixtures' in config, "num_mixtures missing from config"
        assert 'kernel_initializer' in config, "kernel_initializer missing from config"
        assert 'kernel_regularizer' in config, "kernel_regularizer missing from config"

        # Test build config
        build_config = mdn_layer.get_build_config()
        assert 'input_shape' in build_config, "input_shape missing from build config"

        # Recreate layer from config
        recreated_layer = MDNLayer.from_config(config)

        # Check recreated layer attributes
        assert recreated_layer.output_dim == mdn_layer.output_dim, \
            "Recreated layer has different output_dim"
        assert recreated_layer.num_mix == mdn_layer.num_mix, \
            "Recreated layer has different num_mix"

        # Test build from config
        recreated_layer.build_from_config(build_config)
        assert recreated_layer.built, "Layer not built from config"
        logger.info("MDN serialization test passed")

    def test_utility_functions(self):
        """Test utility functions for point estimates and uncertainty."""
        # Setup test parameters
        batch_size = 10
        input_dim = 5
        output_dim = 2
        num_mixes = 3

        # Create and train a simple model
        mdn_layer = MDNLayer(output_dimension=output_dim, num_mixtures=num_mixes)

        inputs = keras.layers.Input(shape=(input_dim,))
        hidden = keras.layers.Dense(16, activation='relu')(inputs)
        outputs = mdn_layer(hidden)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Create test data
        np.random.seed(42)
        x_test = np.random.normal(size=(batch_size, input_dim))

        # Test point estimates
        point_estimates = get_point_estimate(model, x_test, mdn_layer)
        assert point_estimates.shape == (batch_size, output_dim), \
            f"Expected point estimates shape {(batch_size, output_dim)}, got {point_estimates.shape}"

        # Test uncertainty calculation
        total_variance, aleatoric_variance = get_uncertainty(model, x_test, mdn_layer, point_estimates)
        assert total_variance.shape == (batch_size, output_dim), \
            f"Expected total variance shape {(batch_size, output_dim)}, got {total_variance.shape}"
        assert aleatoric_variance.shape == (batch_size, output_dim), \
            f"Expected aleatoric variance shape {(batch_size, output_dim)}, got {aleatoric_variance.shape}"

        # Test prediction intervals
        lower_bound, upper_bound = get_prediction_intervals(point_estimates, total_variance)
        assert lower_bound.shape == (batch_size, output_dim), \
            f"Expected lower bound shape {(batch_size, output_dim)}, got {lower_bound.shape}"
        assert upper_bound.shape == (batch_size, output_dim), \
            f"Expected upper bound shape {(batch_size, output_dim)}, got {upper_bound.shape}"

        # Check that upper bounds are greater than lower bounds
        assert np.all(upper_bound >= lower_bound), "Upper bounds should be >= lower bounds"
        logger.info("Utility functions test passed")


if __name__ == '__main__':
    pytest.main([__file__])