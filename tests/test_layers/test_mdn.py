"""
Extended test suite for the Mixture Density Network Layer.

Tests core functionality including layer construction, shape handling,
forward pass, loss computation, sampling, serialization, and numerical stability.
"""

import os
import keras
import pytest
import numpy as np
import tensorflow as tf
import tempfile

from dl_techniques.layers.mdn_layer import (
    MDNLayer,
    gaussian_probability
)


def test_mdn_layer_initialization():
    """Test basic MDN layer initialization and configuration."""
    # Test initialization with basic parameters
    mdn = MDNLayer(output_dimension=2, num_mixtures=3)

    # Verify layer attributes
    assert mdn.output_dim == 2, "Output dimension not set correctly"
    assert mdn.num_mix == 3, "Number of mixtures not set correctly"

    # Verify layer components
    assert isinstance(mdn.mdn_mus, keras.layers.Dense), "Means layer not initialized correctly"
    assert isinstance(mdn.mdn_sigmas, keras.layers.Dense), "Sigmas layer not initialized correctly"
    assert isinstance(mdn.mdn_pi, keras.layers.Dense), "Pi layer not initialized correctly"

    # Test initialization with custom parameters
    custom_mdn = MDNLayer(
        output_dimension=2,
        num_mixtures=3,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.L2(0.01)
    )
    assert custom_mdn is not None, "Failed to initialize with custom parameters"


def test_mdn_output_shapes():
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
    output = model.predict(test_input)

    # Expected shape calculations
    expected_total_outputs = (2 * output_dim * num_mixes) + num_mixes

    assert output.shape == (batch_size, expected_total_outputs), \
        f"Expected shape {(batch_size, expected_total_outputs)}, got {output.shape}"


def test_mdn_split_mixture_params():
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
    mock_output_tensor = tf.convert_to_tensor(mock_output, dtype=tf.float32)

    # Split the parameters
    mu, sigma, pi = mdn.split_mixture_params(mock_output_tensor)

    # Check shapes
    assert mu.shape == (batch_size, num_mixes, output_dim), \
        f"Expected mu shape {(batch_size, num_mixes, output_dim)}, got {mu.shape}"
    assert sigma.shape == (batch_size, num_mixes, output_dim), \
        f"Expected sigma shape {(batch_size, num_mixes, output_dim)}, got {sigma.shape}"
    assert pi.shape == (batch_size, num_mixes), \
        f"Expected pi shape {(batch_size, num_mixes)}, got {pi.shape}"

    # Test with non-standard shaped input (should still work through reshaping)
    reshaped_output = tf.reshape(mock_output_tensor, [batch_size // 2, -1])
    mu2, sigma2, pi2 = mdn.split_mixture_params(reshaped_output)

    # Verify that the tensors were properly reshaped
    assert mu2.shape[1:] == (num_mixes, output_dim), \
        f"Failed to handle reshaping of non-standard input"


def test_mdn_loss_function():
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

    # Compile with the MDN loss function
    model.compile(optimizer='adam', loss=mdn.loss_func)

    # Create dummy data
    x_train = np.random.normal(size=(batch_size, input_dim))
    y_train = np.random.normal(size=(batch_size, output_dim))

    # Check that training doesn't error out
    history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0)

    # Verify that the loss is a valid float and not NaN or Inf
    assert history.history['loss'][0] > 0, "Loss should be positive"
    assert not np.isnan(history.history['loss'][0]), "Loss is NaN"
    assert not np.isinf(history.history['loss'][0]), "Loss is Inf"

    # Test gradients using GradientTape
    with tf.GradientTape() as tape:
        y_pred = model(x_train, training=True)
        loss = mdn.loss_func(tf.convert_to_tensor(y_train), y_pred)

    # Get gradients with respect to model variables
    grads = tape.gradient(loss, model.trainable_variables)

    # Check that gradients are not None, NaN, or Inf
    for g in grads:
        assert g is not None, "Gradient is None"
        assert not tf.reduce_any(tf.math.is_nan(g)), "Gradient contains NaN values"
        assert not tf.reduce_any(tf.math.is_inf(g)), "Gradient contains Inf values"


def test_mdn_sampling():
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

    # Create test input with fixed seed for reproducibility of the test itself
    np.random.seed(42)
    test_input = np.random.normal(size=(batch_size, input_dim))

    # Get predictions
    predictions = model.predict(test_input)

    # Test sampling
    samples1 = mdn.sample(predictions)

    # Check shape
    assert samples1.shape == (batch_size, output_dim), \
        f"Expected samples shape {(batch_size, output_dim)}, got {samples1.shape}"

    # Test reproducibility with seed
    tf.random.set_seed(123)  # Reset global seed before each sampling
    samples2 = mdn.sample(predictions, seed=42)

    tf.random.set_seed(123)  # Reset global seed again
    samples3 = mdn.sample(predictions, seed=42)

    # Same seed should give same samples
    np.testing.assert_allclose(samples2, samples3, rtol=1e-5, atol=1e-5,
                               err_msg="Sampling with same seed produced different results")

    # Different seeds should (very likely) give different samples
    tf.random.set_seed(123)  # Reset global seed again
    samples4 = mdn.sample(predictions, seed=100)

    # This is a probabilistic test, but the chances of identical samples with different seeds is negligible
    # We use a loose comparison because the test should still pass in edge cases
    different_samples = np.any(np.abs(samples2 - samples4) > 1e-5)
    assert different_samples, "Sampling with different seeds produced identical results"

    # Test temperature parameter
    tf.random.set_seed(123)
    samples_low_temp = mdn.sample(predictions, temp=0.1, seed=42)

    tf.random.set_seed(123)
    samples_high_temp = mdn.sample(predictions, temp=10.0, seed=42)

    # Different temperatures should produce different results
    # This is a probabilistic test, but with such extreme temperatures it should be reliable
    different_temps = np.any(np.abs(samples_low_temp - samples_high_temp) > 1e-5)
    assert different_temps, "Sampling with different temperatures produced identical results"


def test_mdn_numerical_stability():
    """Test MDN layer's numerical stability with extreme inputs."""
    # Setup parameters
    output_dim = 2
    num_mixes = 3

    # Create MDN layer
    mdn = MDNLayer(output_dimension=output_dim, num_mixtures=num_mixes)

    # Test gaussian_probability with extreme values
    y = tf.constant([[1e6, 1e6]], dtype=tf.float32)
    mu = tf.constant([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]], dtype=tf.float32)
    sigma = tf.constant([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]], dtype=tf.float32)

    # Check no NaNs or Infs with extreme difference between y and mu
    result = gaussian_probability(y, mu, sigma)
    assert not tf.reduce_any(tf.math.is_nan(result)), "gaussian_probability produced NaNs"
    assert not tf.reduce_any(tf.math.is_inf(result)), "gaussian_probability produced Infs"

    # Test with extremely small sigma
    y = tf.constant([[0.0, 0.0]], dtype=tf.float32)
    mu = tf.constant([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]], dtype=tf.float32)
    sigma = tf.constant([[[1e-10, 1e-10], [1e-10, 1e-10], [1e-10, 1e-10]]], dtype=tf.float32)

    # Check no NaNs or Infs with tiny sigma
    result = gaussian_probability(y, mu, sigma)
    assert not tf.reduce_any(tf.math.is_nan(result)), "gaussian_probability produced NaNs with small sigma"
    assert not tf.reduce_any(tf.math.is_inf(result)), "gaussian_probability produced Infs with small sigma"

    # Test loss function with extreme predictions
    batch_size = 10
    total_params = (2 * output_dim * num_mixes) + num_mixes

    # Create targets with reasonable values
    y_true = tf.random.normal((batch_size, output_dim), mean=0.0, stddev=1.0)

    # Create predictions with extreme values for mu (very large)
    mus = tf.ones((batch_size, num_mixes * output_dim)) * 1e6
    sigmas = tf.ones((batch_size, num_mixes * output_dim))
    pis = tf.random.normal((batch_size, num_mixes))
    y_pred_large_mu = tf.concat([mus, sigmas, pis], axis=-1)

    # Test loss with large mu values
    loss = mdn.loss_func(y_true, y_pred_large_mu)
    assert not tf.math.is_nan(loss), "Loss is NaN with large mu values"
    assert not tf.math.is_inf(loss), "Loss is Inf with large mu values"

    # Create predictions with very small sigma values
    mus = tf.random.normal((batch_size, num_mixes * output_dim))
    sigmas = tf.ones((batch_size, num_mixes * output_dim)) * 1e-6
    pis = tf.random.normal((batch_size, num_mixes))
    y_pred_small_sigma = tf.concat([mus, sigmas, pis], axis=-1)

    # Test loss with small sigma values
    loss = mdn.loss_func(y_true, y_pred_small_sigma)
    assert not tf.math.is_nan(loss), "Loss is NaN with small sigma values"
    assert not tf.math.is_inf(loss), "Loss is Inf with small sigma values"


def test_mdn_parameter_validation():
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


def test_mdn_training_convergence():
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

    # Create and compile model - FIXING THE WARNING BY USING INPUT LAYER
    mdn_layer = MDNLayer(output_dimension=output_dim, num_mixtures=num_mixes)

    # Use functional API instead of Sequential to avoid the warning
    inputs = keras.layers.Input(shape=(input_dim,))
    x = keras.layers.Dense(32, activation='relu')(inputs)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = mdn_layer(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Create a custom loss function that doesn't need to be serialized
    def loss_fn(y_true, y_pred):
        return mdn_layer.loss_func(y_true, y_pred)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)

    # Train model
    history = model.fit(X, Y, epochs=epochs, batch_size=64, verbose=0)

    # Check loss decreases over time
    assert history.history['loss'][0] > history.history['loss'][-1], \
        "Loss did not decrease during training"

    # Significant loss reduction (at least 30%)
    loss_reduction = (history.history['loss'][0] - history.history['loss'][-1]) / history.history['loss'][0]
    assert loss_reduction > 0.3, f"Insufficient loss reduction: {loss_reduction:.2f}"

    # Test prediction and sampling
    X_test = np.array([[-2.0], [0.0], [2.0]])
    predictions = model.predict(X_test)

    # Generate multiple samples
    n_samples_per_point = 100
    samples = np.zeros((X_test.shape[0], n_samples_per_point))

    for i in range(n_samples_per_point):
        batch_samples = mdn_layer.sample(predictions).numpy()
        samples[:, i] = batch_samples.flatten()

    # For each test point, calculate mean and std of samples
    sample_means = np.mean(samples, axis=1)

    # Expected values (sine function at test points)
    expected_values = np.sin(X_test).flatten()

    # Check that means are close to the expected values (within reasonable bounds for a stochastic model)
    np.testing.assert_allclose(
        sample_means, expected_values,
        rtol=0.3, atol=0.3,  # Allow some divergence due to the stochastic nature
        err_msg="MDN failed to learn a simple sine function distribution"
    )


def test_mdn_serialization():
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

    # Compile model with the serializable loss
    model.compile(optimizer='adam', loss=None)

    # Create test data
    np.random.seed(42)  # For reproducible test
    test_input = np.random.normal(size=(1, input_dim))
    original_output = model.predict(test_input)

    # Save and load model
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, 'mdn_model.keras')
        model.save(model_path)

        # Load model
        loaded_model = keras.models.load_model(
            model_path
        )

    # Test loaded model
    loaded_output = loaded_model.predict(test_input)

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

    # Recreate layer from config
    recreated_layer = MDNLayer.from_config(config)

    # Check recreated layer attributes
    assert recreated_layer.output_dim == mdn_layer.output_dim, \
        "Recreated layer has different output_dim"
    assert recreated_layer.num_mix == mdn_layer.num_mix, \
        "Recreated layer has different num_mix"

if __name__ == '__main__':
    pytest.main([__file__])