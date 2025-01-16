"""Tests for ClusteringLoss class."""

import pytest
import numpy as np
import tensorflow as tf
from dl_techniques.losses.clustering_loss import \
    ClusteringLoss, \
    ClusteringMetrics, \
    ClusteringMetricsCallback


@pytest.fixture
def random_seed():
    """Fixture for consistent random numbers."""
    return 42


@pytest.fixture
def loss_fn():
    """Fixture to create a default ClusteringLoss instance."""
    return ClusteringLoss()


class TestClusteringLoss:
    """Test suite for DifferentiableKMeansLayer."""

    def test_initialization(self):
        """Test clustering loss initialization with default and custom parameters."""
        # Test default initialization
        loss = ClusteringLoss()
        assert loss.distance_weight == 1.0
        assert loss.distribution_weight == 0.5
        assert loss.name == 'clustering_loss'

        # Test custom initialization
        custom_loss = ClusteringLoss(
            distance_weight=2.0,
            distribution_weight=0.3,
            name='custom_loss'
        )
        assert custom_loss.distance_weight == 2.0
        assert custom_loss.distribution_weight == 0.3
        assert custom_loss.name == 'custom_loss'

    def test_perfect_prediction(self, loss_fn):
        """Test loss computation when predictions match targets exactly."""
        y_true = tf.constant([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=tf.float32)
        y_pred = y_true

        loss = loss_fn(y_true, y_pred)

        # For perfect predictions with uniform distribution,
        # loss should be very close to 0
        assert loss.numpy() >= 0
        assert loss.numpy() < 0.01

    def test_completely_wrong_prediction(self, loss_fn):
        """Test loss computation when predictions are completely wrong."""
        y_true = tf.constant([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0]
        ], dtype=tf.float32)

        y_pred = tf.constant([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ], dtype=tf.float32)

        # Get baseline loss for perfect predictions
        baseline_loss = loss_fn(y_true, y_true)

        # Get loss for wrong predictions
        wrong_loss = loss_fn(y_true, y_pred)

        # Wrong predictions should have significantly higher loss
        assert wrong_loss > baseline_loss * 2

    def test_uniform_distribution(self, loss_fn):
        """Test loss computation with uniform cluster distribution."""
        y_true = tf.constant([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1]
        ], dtype=tf.float32)

        y_pred = y_true

        loss = loss_fn(y_true, y_pred)

        # For uniform distribution with perfect predictions,
        # loss should be very small
        assert loss.numpy() < 0.01

    @pytest.mark.parametrize("distance_scale", [0.5, 1.0, 2.0])
    def test_weight_scaling(self, distance_scale):
        """Test that loss scales properly with weights."""
        y_true = tf.constant([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=tf.float32)

        y_pred = tf.constant([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ], dtype=tf.float32)

        # Create two loss functions with scaled weights
        loss1 = ClusteringLoss(
            distance_weight=1.0,
            distribution_weight=1.0
        )
        loss2 = ClusteringLoss(
            distance_weight=distance_scale,
            distribution_weight=distance_scale
        )

        value1 = loss1(y_true, y_pred).numpy()
        value2 = loss2(y_true, y_pred).numpy()

        # Check that loss scales approximately with weights
        expected_ratio = distance_scale
        actual_ratio = value2 / value1
        assert np.abs(actual_ratio - expected_ratio) < 0.1

    @pytest.mark.parametrize("batch_size,n_clusters", [
        (1, 2),
        (4, 3),
        (10, 5),
        (100, 10)
    ])
    def test_different_shapes(self, batch_size, n_clusters):
        """Test loss computation with different batch sizes and number of clusters."""
        loss_fn = ClusteringLoss()

        # Create random predictions and targets
        y_true = tf.random.uniform((batch_size, n_clusters))
        y_true = y_true / tf.reduce_sum(y_true, axis=1, keepdims=True)

        y_pred = tf.random.uniform((batch_size, n_clusters))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=1, keepdims=True)

        loss = loss_fn(y_true, y_pred)

        assert loss.shape == ()  # Loss should be a scalar
        assert not tf.math.is_nan(loss)  # Loss should not be NaN
        assert not tf.math.is_inf(loss)  # Loss should not be infinite

    @pytest.mark.parametrize("invalid_weight", [-1.0, 0.0, float('inf'), float('nan')])
    def test_invalid_weights(self, invalid_weight):
        """Test that initialization fails with invalid weights."""
        with pytest.raises(ValueError):
            ClusteringLoss(distance_weight=invalid_weight)

        with pytest.raises(ValueError):
            ClusteringLoss(distribution_weight=invalid_weight)

    def test_gradient_computation(self):
        """Test that gradients can be computed through the loss."""
        loss_fn = ClusteringLoss()

        # Create trainable predictions
        y_pred = tf.Variable(tf.random.uniform((4, 3)))
        y_true = tf.constant([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ], dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss = loss_fn(y_true, y_pred)

        gradients = tape.gradient(loss, y_pred)

        assert gradients is not None
        assert not tf.reduce_any(tf.math.is_nan(gradients))
        assert not tf.reduce_any(tf.math.is_inf(gradients))

    def test_different_batch_sizes(self):
        """Test that loss works with different batch sizes."""
        loss_fn = ClusteringLoss()

        for batch_size in [1, 4, 16]:
            y_true = tf.random.uniform((batch_size, 3))
            y_true = y_true / tf.reduce_sum(y_true, axis=1, keepdims=True)

            y_pred = tf.random.uniform((batch_size, 3))
            y_pred = y_pred / tf.reduce_sum(y_pred, axis=1, keepdims=True)

            loss = loss_fn(y_true, y_pred)

            assert loss.shape == ()  # Loss should be a scalar
            assert not tf.math.is_nan(loss)
            assert not tf.math.is_inf(loss)

    def test_imbalanced_clusters(self, loss_fn):
        """Test loss behavior with imbalanced cluster distributions.

        This test verifies that the distribution penalty correctly penalizes
        imbalanced cluster assignments while still maintaining reasonable
        distance metrics.
        """
        # Create highly imbalanced predictions
        y_true = tf.constant([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],  # Only one sample for second cluster
            [0, 0, 1],  # Only one sample for third cluster
        ], dtype=tf.float32)

        # Perfect predictions but imbalanced distribution
        y_pred = y_true
        imbalanced_loss = loss_fn(y_true, y_pred)

        # Create balanced predictions for comparison
        y_true_balanced = tf.constant([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=tf.float32)

        y_pred_balanced = y_true_balanced
        balanced_loss = loss_fn(y_true_balanced, y_pred_balanced)

        # Verify both losses are non-negative
        assert imbalanced_loss >= 0
        assert balanced_loss >= 0

        # Verify imbalanced distribution has non-zero loss
        assert imbalanced_loss > 0

        # Loss should be reasonable (not extremely large)
        assert imbalanced_loss < 1.0

    def test_noisy_predictions(self, loss_fn, random_seed):
        """Test loss behavior with noisy predictions.

        This test verifies that the loss function handles noisy predictions
        gracefully and maintains a reasonable gradient signal.
        """
        tf.random.set_seed(random_seed)

        # Create base predictions
        y_true = tf.constant([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=tf.float32)

        # Generate predictions with different noise levels
        noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        losses = []

        for noise in noise_levels:
            # Add noise to predictions
            noise_tensor = tf.random.uniform(y_true.shape, maxval=noise)
            y_pred = y_true * (1 - noise) + noise_tensor

            # Normalize predictions to sum to 1
            y_pred = y_pred / tf.reduce_sum(y_pred, axis=1, keepdims=True)

            loss = loss_fn(y_true, y_pred)
            losses.append(float(loss.numpy()))

        # Verify losses are non-negative
        assert all(loss >= 0 for loss in losses)

        # Verify that high noise leads to higher loss
        assert losses[0] < losses[-1]

        # Verify losses remain finite and reasonable
        assert all(loss < 10.0 for loss in losses)

        # Verify loss changes are not too extreme between noise levels
        loss_differences = [abs(losses[i + 1] - losses[i]) for i in range(len(losses) - 1)]
        assert all(diff < 2.0 for diff in loss_differences)

    def test_gradient_behavior(self):
        """Test gradient behavior in different scenarios.

        This test verifies that gradients have reasonable magnitudes and
        directions for various prediction scenarios.
        """
        loss_fn = ClusteringLoss(distance_weight=1.0, distribution_weight=1.0)

        y_true = tf.constant([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=tf.float32)

        # Test cases with different prediction patterns
        test_cases = {
            'near_perfect': tf.constant([
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
            ], dtype=tf.float32),
            'low_confidence': tf.constant([
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.3, 0.3, 0.4],
            ], dtype=tf.float32),
        }

        gradients = {}
        losses = {}
        for case_name, y_pred in test_cases.items():
            y_pred = tf.Variable(y_pred)
            with tf.GradientTape() as tape:
                loss = loss_fn(y_true, y_pred)
                losses[case_name] = loss.numpy()
            grad = tape.gradient(loss, y_pred)
            gradients[case_name] = grad.numpy()

        # Verify loss ordering
        assert losses['near_perfect'] < losses['low_confidence']

        # Verify gradients exist and are reasonable
        for case_name, grad in gradients.items():
            assert not tf.reduce_any(tf.math.is_nan(grad))
            assert not tf.reduce_any(tf.math.is_inf(grad))

            # Verify gradient magnitudes are reasonable
            grad_norm = tf.norm(grad).numpy()
            assert grad_norm > 0
            assert grad_norm < 10.0

    def test_numerical_stability(self):
        """Test numerical stability with extreme values.

        This test verifies that the loss function remains stable with very
        small and very large logit values, and with various numerical patterns.
        """
        loss_fn = ClusteringLoss()

        # Test various numerical edge cases
        test_cases = {
            'very_small_values': {
                'pred': tf.constant([
                    [1e-7, 1 - 2e-7, 1e-7],
                    [1e-7, 1e-7, 1 - 2e-7],
                ], dtype=tf.float32),
                'true': tf.constant([
                    [0, 1, 0],
                    [0, 0, 1],
                ], dtype=tf.float32)
            },
            'very_large_differences': {
                'pred': tf.constant([
                    [1 - 1e-12, 1e-12, 1e-12],
                    [1e-12, 1 - 1e-12, 1e-12],
                ], dtype=tf.float32),
                'true': tf.constant([
                    [1, 0, 0],
                    [0, 1, 0],
                ], dtype=tf.float32)
            }
        }

        for case_name, data in test_cases.items():
            loss = loss_fn(data['true'], data['pred'])

            # Verify loss is finite and reasonable
            assert not tf.math.is_nan(loss)
            assert not tf.math.is_inf(loss)
            assert loss >= 0
            assert loss < 10

            # Verify gradients exist and are reasonable
            y_pred = tf.Variable(data['pred'])
            with tf.GradientTape() as tape:
                loss = loss_fn(data['true'], y_pred)
            gradients = tape.gradient(loss, y_pred)

            assert not tf.reduce_any(tf.math.is_nan(gradients))
            assert not tf.reduce_any(tf.math.is_inf(gradients))
            assert tf.reduce_all(tf.abs(gradients) < 100.0)

    def test_weight_interaction(self):
        """Test interaction between distance and distribution weights.

        This test verifies that the distance and distribution components
        interact appropriately under different weight combinations.
        """
        y_true = tf.constant([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=tf.float32)

        y_pred = tf.constant([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ], dtype=tf.float32)

        # Test different weight combinations
        weight_combinations = [
            (1.0, 1.0),  # Equal weights
            (2.0, 1.0),  # Distance emphasized
            (1.0, 2.0),  # Distribution emphasized
        ]

        losses = []
        for distance_weight, distribution_weight in weight_combinations:
            loss_fn = ClusteringLoss(
                distance_weight=distance_weight,
                distribution_weight=distribution_weight
            )
            loss = loss_fn(y_true, y_pred)
            losses.append(float(loss.numpy()))

        # Verify basic properties
        assert all(loss > 0 for loss in losses)
        assert all(not np.isnan(loss) for loss in losses)
        assert all(not np.isinf(loss) for loss in losses)
        assert all(loss < 10.0 for loss in losses)

        # Verify weights influence the loss
        assert abs(losses[0] - losses[1]) > 1e-5  # Different weights should give different losses
        assert abs(losses[0] - losses[2]) > 1e-5

if __name__ == "__main__":
    pytest.main([__file__])