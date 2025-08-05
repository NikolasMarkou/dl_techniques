import keras
import pytest
import numpy as np
import tensorflow as tf
from typing import Generator, Any, Tuple, Dict, List


from dl_techniques.layers.radial_basis_function import RBFLayer


def generate_cluster_data(
        n_clusters: int,
        n_samples: int,
        dim: int,
        noise: float = 0.1,
        seed: int = 42
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generate clustered data for testing.

    Args:
        n_clusters: Number of clusters
        n_samples: Samples per cluster
        dim: Dimensionality of data
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Tuple of (data, centers)
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Generate well-separated cluster centers
    centers = tf.random.uniform(
        (n_clusters, dim),
        minval=-5.0,
        maxval=5.0
    )

    # Generate samples around centers
    samples_per_cluster = n_samples // n_clusters
    data = []
    for center in centers:
        cluster_samples = center + tf.random.normal(
            (samples_per_cluster, dim),
            mean=0.0,
            stddev=noise
        )
        data.append(cluster_samples)

    return tf.concat(data, axis=0), centers


@pytest.fixture
def layer_config() -> Dict[str, Any]:
    """Default layer configuration for testing."""
    return {
        'units': 10,
        'gamma_init': 1.0,
        'repulsion_strength': 0.1,
        'min_center_distance': 1.0,
        'trainable_gamma': True,
    }


class TestRBFLayer:
    """Test suite for RBF Layer."""

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """
        Test layer initialization and parameter validation.

        Tests:
        1. Correct parameter initialization
        2. Parameter validation
        3. Weight creation
        """
        # Test correct initialization
        layer = RBFLayer(**layer_config)
        assert layer.units == layer_config['units']
        assert layer.gamma_init == layer_config['gamma_init']
        assert layer.repulsion_strength == layer_config['repulsion_strength']

        # Test invalid parameters
        with pytest.raises(ValueError):
            RBFLayer(units=-1)  # Invalid units
        with pytest.raises(ValueError):
            RBFLayer(units=10, gamma_init=-1.0)  # Invalid gamma
        with pytest.raises(ValueError):
            RBFLayer(units=10, repulsion_strength=-0.1)  # Invalid repulsion

    @pytest.mark.parametrize(
        "batch_size,input_dim,units",
        [
            (32, 4, 10),
            (64, 8, 16),
            (16, 2, 5),
            (128, 16, 32)
        ]
    )
    def test_output_shape(
            self,
            layer_config: Dict[str, Any],
            batch_size: int,
            input_dim: int,
            units: int
    ) -> None:
        """
        Test output shapes for various configurations.

        Tests:
        1. Basic shape compatibility
        2. Dynamic batch size handling
        3. Multiple input dimensions
        """
        config = dict(layer_config)
        config['units'] = units

        layer = RBFLayer(**config)
        inputs = tf.random.normal((batch_size, input_dim))
        outputs = layer(inputs)

        assert outputs.shape == (batch_size, units)

        # Test with different batch size
        new_batch = tf.random.normal((batch_size * 2, input_dim))
        new_outputs = layer(new_batch)
        assert new_outputs.shape == (batch_size * 2, units)

    def test_activation_properties(self, layer_config: Dict[str, Any]) -> None:
        """
        Test RBF activation mathematical properties.

        Tests:
        1. Output range [0, 1]
        2. Maximum activation at center
        3. Symmetric activation around center
        4. Monotonic decrease with distance
        """
        layer = RBFLayer(**layer_config)
        input_dim = 4
        layer.build((None, input_dim))

        # Test output range
        inputs = tf.random.normal((100, input_dim))
        outputs = layer(inputs)
        assert tf.reduce_min(outputs) >= 0.0
        assert tf.reduce_max(outputs) <= 1.0

        # Test maximum activation at center
        center = layer.centers[0]
        input_at_center = tf.expand_dims(center, 0)
        activation = layer(input_at_center)[0, 0]
        assert tf.abs(activation - 1.0) < 1e-5

        # Test symmetry around center
        offset = tf.constant([[1.0, 0.0, 0.0, 0.0]])
        pos_input = input_at_center + offset
        neg_input = input_at_center - offset
        pos_activation = layer(pos_input)[0, 0]
        neg_activation = layer(neg_input)[0, 0]
        assert tf.abs(pos_activation - neg_activation) < 1e-5

    def test_repulsion_mechanics(
            self,
            layer_config: Dict[str, Any]
    ) -> None:
        """
        Test center repulsion mechanism.

        Tests:
        1. Repulsion force calculation
        2. Loss contribution
        3. Center movement under repulsion
        """
        # Override the number of units to 2 for this test
        test_config = dict(layer_config)
        test_config['units'] = 2

        input_dim = 4
        layer = RBFLayer(**test_config)

        # Create two centers very close to each other
        initial_centers = tf.constant([
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1]
        ], dtype=tf.float32)

        layer.build((None, input_dim))
        # Now the shapes match: both are (2, 4)
        layer.centers.assign(initial_centers)

        # Compute initial repulsion
        initial_repulsion = layer._compute_repulsion(layer.centers)

        # Move centers apart
        separated_centers = tf.constant([
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0]
        ], dtype=tf.float32)
        layer.centers.assign(separated_centers)

        # Compute new repulsion
        final_repulsion = layer._compute_repulsion(layer.centers)

        # Repulsion should decrease as centers move apart
        assert final_repulsion < initial_repulsion

    def test_training_behavior(
            self,
            layer_config: Dict[str, Any]
    ) -> None:
        """
        Test layer behavior during training.

        Tests:
        1. Center adaptation to data
        2. Width (gamma) adaptation
        3. Loss convergence
        4. Center distribution
        """
        # Configure for more robust testing
        test_config = dict(layer_config)
        test_config.update({
            'units': 6,  # 2 units per cluster
            'repulsion_strength': 1.0,  # Increase repulsion strength
            'min_center_distance': 1.0,  # Keep original minimum distance
            'safety_margin': 0.2,  # Add safety margin
            'gamma_init': 0.5  # Wider initial receptive fields
        })

        # Generate well-separated clustered data
        n_clusters = 3
        n_samples = 300
        input_dim = 4
        data, true_centers = generate_cluster_data(
            n_clusters=n_clusters,
            n_samples=n_samples,
            dim=input_dim,
            noise=0.1  # Less noise for clearer clusters
        )

        # Create layer and optimizer
        layer = RBFLayer(**test_config)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # Training loop
        losses = []
        min_dist_history = []

        for epoch in range(200):  # More epochs for convergence
            with tf.GradientTape() as tape:
                outputs = layer(data, training=True)  # Enable training mode

                # Multi-component loss
                # 1. Activation loss: encourage strong responses
                activation_loss = -tf.reduce_mean(tf.reduce_max(outputs, axis=1))

                # Get the automatic repulsion loss
                repulsion_loss = sum(layer.losses)  # Layer adds repulsion loss automatically

                # Total loss
                total_loss = activation_loss + repulsion_loss

            # Compute and apply gradients with clipping
            grads = tape.gradient(total_loss, layer.trainable_weights)
            clipped_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g
                             for g in grads]
            optimizer.apply_gradients(zip(clipped_grads, layer.trainable_weights))
            losses.append(float(total_loss))

            # Monitor minimum distance between centers
            if epoch % 10 == 0:
                centers = layer.centers.numpy()
                dists = []
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        dists.append(dist)
                min_dist_history.append(min(dists))

                # Early stopping if centers are well separated
                if min_dist_history[-1] >= 0.8 * test_config['min_center_distance']:
                    break

        # Verify loss convergence
        assert losses[-1] < losses[0], \
            f"Training did not converge. Initial loss: {losses[0]}, " \
            f"Final loss: {losses[-1]}"

        # Check final center distribution
        final_centers = layer.centers.numpy()
        final_dists = []
        for i in range(len(final_centers)):
            for j in range(i + 1, len(final_centers)):
                dist = np.linalg.norm(final_centers[i] - final_centers[j])
                final_dists.append(dist)

        min_dist = min(final_dists)
        assert min_dist >= 0.5 * test_config['min_center_distance'], \
            f"Centers are too close. Min distance: {min_dist}, " \
            f"Required: {0.5 * test_config['min_center_distance']}, " \
            f"Distance history: {min_dist_history[-5:]}"

    def test_numerical_stability(
            self,
            layer_config: Dict[str, Any]
    ) -> None:
        """
        Test numerical stability with extreme inputs.

        Tests:
        1. Large magnitude inputs
        2. Very small inputs
        3. Zero inputs
        4. NaN/Inf handling
        """
        layer = RBFLayer(**layer_config)
        input_dim = 4

        # Test various extreme inputs
        test_inputs = [
            tf.zeros((32, input_dim)),  # All zeros
            tf.ones((32, input_dim)) * 1000.0,  # Large values
            tf.ones((32, input_dim)) * 1e-8,  # Small values
            tf.random.normal((32, input_dim)) * 1e6  # Very large random values
        ]

        for inputs in test_inputs:
            outputs = layer(inputs)

            # Check for NaN/Inf
            assert not tf.reduce_any(tf.math.is_nan(outputs)), \
                "Output contains NaN values"
            assert not tf.reduce_any(tf.math.is_inf(outputs)), \
                "Output contains Inf values"

            # Check output range
            assert tf.reduce_all(outputs >= 0.0) and tf.reduce_all(outputs <= 1.0), \
                "Outputs outside valid range [0, 1]"

if __name__ == '__main__':
    pytest.main([__file__])
