import keras
import pytest
import numpy as np
import tensorflow as tf
from typing import Generator, Any


# Import your RBF layer implementation
from dl_techniques.layers.radial_basis_function import RBFLayer


@pytest.fixture
def sample_input() -> Generator[tuple[tf.Tensor, tuple[int, int]], Any, None]:
    """
    Fixture providing sample input data for testing.

    Returns:
        Generator yielding a tuple of (input tensor, input shape)
    """
    batch_size, input_dim = 32, 10
    input_data = tf.random.normal((batch_size, input_dim))
    yield input_data, (batch_size, input_dim)


def test_rbf_layer_initialization() -> None:
    """
    Test RBF layer initialization with valid parameters.
    """
    units, gamma = 15, 0.5
    layer = RBFLayer(units=units, gamma=gamma)

    assert layer.units == units
    assert layer.gamma == gamma
    assert isinstance(layer.initializer, keras.initializers.Initializer)


def test_invalid_initialization() -> None:
    """
    Test RBF layer initialization with invalid parameters.
    """
    with pytest.raises(ValueError, match="Number of units must be positive"):
        RBFLayer(units=0)

    with pytest.raises(ValueError, match="Gamma must be positive"):
        RBFLayer(units=5, gamma=0.0)


def test_build_and_output_shape(sample_input: tuple[tf.Tensor, tuple[int, int]]) -> None:
    """
    Test layer building and output shape computation.

    Args:
        sample_input: Tuple of (input tensor, input shape)
    """
    input_tensor, (batch_size, input_dim) = sample_input
    units = 20
    layer = RBFLayer(units=units)

    # Build the layer
    layer.build((batch_size, input_dim))

    # Check weights shapes
    assert layer.centers.shape == (units, input_dim)
    assert layer.widths.shape == (units,)

    # Check output shape computation
    output_shape = layer.compute_output_shape((batch_size, input_dim))
    assert output_shape == (batch_size, units)


def test_forward_pass(sample_input: tuple[tf.Tensor, tuple[int, int]]) -> None:
    """
    Test forward pass of the RBF layer.

    Args:
        sample_input: Tuple of (input tensor, input shape)
    """
    input_tensor, _ = sample_input
    units = 15
    layer = RBFLayer(units=units)

    # Perform forward pass
    output = layer(input_tensor)

    # Check output properties
    assert output.shape[1] == units
    assert tf.reduce_all(output >= 0)  # RBF outputs should be non-negative
    assert tf.reduce_all(output <= 1)  # RBF outputs should be bounded by 1


def test_serialization() -> None:
    """
    Test layer serialization and deserialization.
    """
    original_layer = RBFLayer(
        units=10,
        gamma=0.7,
        initializer='glorot_uniform'
    )

    # Get config and create new layer
    config = original_layer.get_config()
    new_layer = RBFLayer(**config)

    # Check if configurations match
    assert config['units'] == new_layer.units
    assert config['gamma'] == new_layer.gamma
    assert config['initializer'] == keras.initializers.serialize(new_layer.initializer)


def test_gradient_flow() -> None:
    """
    Test if gradients flow properly through the layer during training.
    Verifies both centers and widths are being updated.
    """
    batch_size, input_dim, units = 16, 8, 10
    layer = RBFLayer(units=units)

    with tf.GradientTape() as tape:
        inputs = tf.random.normal((batch_size, input_dim))
        initial_centers = tf.identity(layer.centers) if layer.centers is not None else None
        initial_widths = tf.identity(layer.widths) if layer.widths is not None else None

        # Forward pass
        outputs = layer(inputs)
        loss = tf.reduce_mean(outputs)

    # Calculate gradients
    grads = tape.gradient(loss, [layer.centers, layer.widths])

    # Check if gradients exist and are not zero
    assert all(g is not None and tf.reduce_any(tf.not_equal(g, 0)) for g in grads)


def test_numerical_stability() -> None:
    """
    Test numerical stability with extreme input values and varying gamma values.
    """
    batch_size, input_dim, units = 16, 4, 5
    layer = RBFLayer(units=units, gamma=1e-3)  # Small gamma

    # Test with large input values
    large_inputs = tf.random.normal((batch_size, input_dim)) * 1e3
    large_outputs = layer(large_inputs)

    # Test with very small input values
    small_inputs = tf.random.normal((batch_size, input_dim)) * 1e-3
    small_outputs = layer(small_inputs)

    # Check for NaN or Inf values
    assert not tf.reduce_any(tf.math.is_nan(large_outputs))
    assert not tf.reduce_any(tf.math.is_inf(large_outputs))
    assert not tf.reduce_any(tf.math.is_nan(small_outputs))
    assert not tf.reduce_any(tf.math.is_inf(small_outputs))


def test_batch_independence() -> None:
    """
    Test that samples in a batch are processed independently.
    """
    batch_size, input_dim, units = 4, 3, 5
    layer = RBFLayer(units=units)

    # Create two identical batches except for one sample
    inputs1 = tf.random.normal((batch_size, input_dim))
    inputs2 = tf.identity(inputs1)

    # Modify one sample in the second batch
    modified_index = 2
    inputs2 = tf.tensor_scatter_nd_update(
        inputs2,
        [[modified_index, 0]],
        [inputs2[modified_index, 0] + 1.0]
    )

    outputs1 = layer(inputs1)
    outputs2 = layer(inputs2)

    # Check that only the modified sample's output changed
    unmodified_samples = tf.reduce_all(
        tf.equal(
            outputs1[tf.range(batch_size) != modified_index],
            outputs2[tf.range(batch_size) != modified_index]
        )
    )
    modified_sample_changed = tf.reduce_any(
        tf.not_equal(
            outputs1[modified_index],
            outputs2[modified_index]
        )
    )

    assert unmodified_samples
    assert modified_sample_changed


def test_center_adaptation() -> None:
    """
    Test if centers adapt to the input distribution during training.
    """
    batch_size, input_dim, units = 32, 4, 8
    layer = RBFLayer(units=units)
    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    # Generate clustered data
    n_clusters = 3
    cluster_centers = tf.random.uniform((n_clusters, input_dim), -2, 2)

    for _ in range(50):  # Training iterations
        with tf.GradientTape() as tape:
            # Generate samples around cluster centers
            cluster_idx = tf.random.uniform(
                (batch_size,), 0, n_clusters, dtype=tf.int32
            )
            inputs = tf.gather(cluster_centers, cluster_idx) + \
                     tf.random.normal((batch_size, input_dim), 0, 0.1)

            outputs = layer(inputs)
            # Loss that encourages specialization to clusters
            loss = -tf.reduce_mean(tf.reduce_max(outputs, axis=1))

        grads = tape.gradient(loss, layer.trainable_variables)
        optimizer.apply_gradients(zip(grads, layer.trainable_variables))

    # Check if centers have adapted to be near cluster centers
    min_distances = tf.reduce_min(
        tf.reduce_sum(
            tf.square(
                tf.expand_dims(layer.centers, 1) - \
                tf.expand_dims(cluster_centers, 0)
            ),
            axis=-1
        ),
        axis=1
    )

    # Assert that at least some centers are close to cluster centers
    assert tf.reduce_any(min_distances < 1.0)

if __name__ == '__main__':
    pytest.main([__file__])
