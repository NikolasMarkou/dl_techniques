"""
Basic test suite for the Mixture Density Network Layer.

Tests core functionality including layer construction, shape handling,
forward pass, loss computation, and sampling.
"""

import keras
import pytest
import numpy as np
import tensorflow as tf


from dl_techniques.layers.mdn import (
    MDN,
    get_mixture_loss_func,
    get_mixture_sampling_func,
    gaussian_probability,
    softmax
)


def test_mdn_layer_initialization():
    """Test basic MDN layer initialization and configuration."""
    # Test initialization with basic parameters
    mdn = MDN(output_dimension=2, num_mixtures=3)

    # Verify layer attributes
    assert mdn.output_dim == 2, "Output dimension not set correctly"
    assert mdn.num_mix == 3, "Number of mixtures not set correctly"

    # Verify layer components
    assert isinstance(mdn.mdn_mus, keras.layers.Dense), "Means layer not initialized correctly"
    assert isinstance(mdn.mdn_sigmas, keras.layers.Dense), "Sigmas layer not initialized correctly"
    assert isinstance(mdn.mdn_pi, keras.layers.Dense), "Pi layer not initialized correctly"

    # Test initialization with custom parameters
    custom_mdn = MDN(
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
    mdn = MDN(output_dimension=output_dim, num_mixtures=num_mixes)

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


def test_mdn_loss_computation():
    """Test MDN loss function computation."""
    output_dim = 2
    num_mixes = 3
    batch_size = 16

    # Get loss function
    loss_func = get_mixture_loss_func(output_dim, num_mixes)

    # Create sample predictions and targets
    y_true = np.random.normal(size=(batch_size, output_dim))

    # Create mock MDN outputs (mus, sigmas, pis concatenated)
    mus = np.random.normal(size=(batch_size, num_mixes * output_dim))
    sigmas = np.abs(np.random.normal(size=(batch_size, num_mixes * output_dim))) + 1.0
    pis = np.random.normal(size=(batch_size, num_mixes))
    y_pred = np.concatenate([mus, sigmas, pis], axis=-1)

    # Compute loss
    loss = loss_func(
        tf.convert_to_tensor(y_true, dtype=tf.float32),
        tf.convert_to_tensor(y_pred, dtype=tf.float32)
    )

    # Verify loss properties
    assert not tf.math.is_nan(loss), "Loss computation resulted in NaN"
    assert not tf.math.is_inf(loss), "Loss computation resulted in Inf"
    assert loss.numpy() > 0, "Loss should be positive"


def test_mdn_sampling():
    """Test MDN sampling function with various batch sizes and dimensions."""
    test_configs = [
        {'output_dim': 2, 'num_mixes': 3, 'batch_size': 1},  # Single sample
        {'output_dim': 2, 'num_mixes': 3, 'batch_size': 16},  # Normal batch
        {'output_dim': 5, 'num_mixes': 10, 'batch_size': 32},  # Larger dimensions
    ]

    for config in test_configs:
        output_dim = config['output_dim']
        num_mixes = config['num_mixes']
        batch_size = config['batch_size']

        # Get sampling function
        sampling_func = get_mixture_sampling_func(output_dim, num_mixes)

        # Create mock MDN outputs with controlled values
        mus = np.random.normal(size=(batch_size, num_mixes * output_dim))
        sigmas = np.abs(np.random.normal(size=(batch_size, num_mixes * output_dim))) + 1.0
        pis = np.zeros((batch_size, num_mixes))  # Initialize with zeros
        pis[:, 0] = 1.0  # Set highest probability to first component

        # Concatenate parameters
        y_pred = np.concatenate([mus, sigmas, pis], axis=-1)
        y_pred_tensor = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        # Generate multiple samples to test randomness
        n_samples = 5
        all_samples = [sampling_func(y_pred_tensor) for _ in range(n_samples)]

        # Verify sample properties
        for samples in all_samples:
            # Check shape
            assert samples.shape == (batch_size, output_dim), \
                f"Expected shape {(batch_size, output_dim)}, got {samples.shape}"

            # Check numerical stability
            assert not tf.reduce_any(tf.math.is_nan(samples)), \
                "Sampling produced NaN values"
            assert not tf.reduce_any(tf.math.is_inf(samples)), \
                "Sampling produced Inf values"

            # Check value ranges
            assert tf.reduce_all(tf.math.is_finite(samples)), \
                "Sampling produced non-finite values"

        # Test reproducibility with fixed seed
        tf.random.set_seed(42)
        sample1 = sampling_func(y_pred_tensor)
        tf.random.set_seed(42)
        sample2 = sampling_func(y_pred_tensor)
        assert tf.reduce_all(tf.equal(sample1, sample2)), \
            "Sampling is not reproducible with fixed seed"


def test_mdn_training():
    """Test MDN layer in a simple training loop."""
    # Model parameters
    input_dim = 5
    output_dim = 2
    num_mixes = 3
    hidden_size = 32
    batch_size = 64
    epochs = 2

    # Create simple model
    model = keras.Sequential([
        keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_dim,)),
        MDN(output_dimension=output_dim, num_mixtures=num_mixes)
    ])

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=get_mixture_loss_func(output_dim, num_mixes)
    )

    # Generate dummy training data
    X_train = np.random.normal(size=(batch_size, input_dim))
    y_train = np.random.normal(size=(batch_size, output_dim))

    # Train model
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0)

    # Verify training
    assert len(history.history['loss']) == epochs, "Training did not complete all epochs"
    assert history.history['loss'][-1] < history.history['loss'][0], \
        "Loss did not decrease during training"


if __name__ == '__main__':
    pytest.main([__file__])
