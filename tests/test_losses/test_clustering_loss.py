"""Tests for ClusteringLoss class."""

import keras
import pytest
import numpy as np
import tensorflow as tf
from tests.test_losses.keras_mask_probe import (
    assert_a_per_sample_mask_is_left_to_keras,
    assert_refuses_a_keras_mask,
    assert_refuses_per_timestep_weights,
)
from dl_techniques.losses.clustering_loss import ClusteringLoss


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

    def test_from_config_round_trips_the_emitted_config(self):
        """``get_config`` emits ``reduction``; ``from_config`` must accept it."""
        original = ClusteringLoss(distance_weight=2.0, distribution_weight=0.3)
        config = original.get_config()
        assert "reduction" in config
        restored = ClusteringLoss.from_config(config)
        assert restored.distance_weight == 2.0
        assert restored.distribution_weight == 0.3
        assert restored.reduction == original.reduction

    def test_reduction_is_honoured(self):
        """The accepted ``reduction`` reaches the base class instead of being dropped."""
        assert ClusteringLoss(reduction="sum").reduction == "sum"

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

        assert loss.shape == ()  # the default reduction turns call()'s (batch,) rows into a scalar
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

            assert loss.shape == ()  # the default reduction turns call()'s (batch,) rows into a scalar
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


# ---------------------------------------------------------------------
# per-sample contract (plan-2026-11-11T223825-64d1b401, step 3)
# ---------------------------------------------------------------------

BATCH = 4
CLUSTERS = 5
ROW_ATOL = 1e-7


def _cluster_batch():
    """Dirichlet rows for ``y_true`` and sharper (alpha 0.5) Dirichlet rows for ``y_pred``, float64.

    The same draws as the golden fixture ``cluster_pin`` recorded at HEAD b4831ec33.
    """
    rng = np.random.default_rng(1234)
    y_true = rng.dirichlet(np.ones(CLUSTERS), size=BATCH)
    y_pred = rng.dirichlet(np.ones(CLUSTERS) * 0.5, size=BATCH)
    return y_true, y_pred


def _distance_rows64(y_true, y_pred):
    """``d_i = mean_k (p_ik - q_ik)^2``, float64, independent of the loss code."""
    return ((y_pred - y_true) ** 2).mean(axis=-1)


def _balance64(y_pred):
    """``c = mean_k (mean_b p_bk - 1/K)^2``, float64: the batch balance penalty."""
    return ((y_pred.mean(axis=0) - 1.0 / y_pred.shape[-1]) ** 2).mean()


def _rows64(y_true, y_pred, wd=1.0, wdi=0.5):
    """Reference rows ``wd * d_i + wdi * c`` in float64."""
    return wd * _distance_rows64(y_true, y_pred) + wdi * _balance64(y_pred)


def _old_scalar64(y_true, y_pred, wd=1.0, wdi=0.5):
    """The scalar the loss returned before this plan, evaluated in float64."""
    return wd * _distance_rows64(y_true, y_pred).mean() + wdi * _balance64(y_pred)


def _f32(x):
    return tf.constant(np.asarray(x, dtype="float32"))


def _returns_per_row(loss_obj):
    """True when ``call`` returns ``(B,)`` rows equal to the float64 reference."""
    y_true, y_pred = _cluster_batch()
    out = loss_obj.call(_f32(y_true), _f32(y_pred))
    if tuple(out.shape) != (BATCH,):
        return False
    return bool(np.abs(out.numpy() - _rows64(y_true, y_pred)).max() < ROW_ATOL)


def _balance_is_one_constant_for_every_row(loss_obj, wd=1.0, wdi=0.5):
    """The D-003 rule as a predicate: ``row_i - wd * d_i`` is the SAME batch value in every row.

    It is also required to equal the float64 batch penalty ``wdi * c``, so a per-row
    attribution of the balance term (whose mean is ``c`` but whose rows differ) fails.
    """
    y_true, y_pred = _cluster_batch()
    rows = loss_obj.call(_f32(y_true), _f32(y_pred)).numpy().astype("float64")
    balance = rows - wd * _distance_rows64(y_true, y_pred)
    same = np.abs(balance - balance[0]).max() < ROW_ATOL
    right = abs(balance[0] - wdi * _balance64(y_pred)) < ROW_ATOL
    return bool(same and right)


class _ScalarClustering(ClusteringLoss):
    """The pre-fix behaviour: one batch scalar, reduced inside ``call``."""

    def call(self, y_true, y_pred):
        return keras.ops.mean(super().call(y_true, y_pred))


class _AlignmentClustering(ClusteringLoss):
    """Option (c) of D-003: rows whose mean is ``c`` but which differ row to row.

    ``r_i = mean_j((f_j - 1/K)(p_ij - 1/K))`` has ``mean_i r_i == c``, so the default
    value is unchanged and only the balance guard can tell it from the shipped rule.
    """

    def call(self, y_true, y_pred):
        d = keras.ops.mean(keras.ops.square(y_pred - y_true), axis=-1)
        u = 1.0 / y_pred.shape[-1]
        f = keras.ops.mean(y_pred, axis=0)
        r = keras.ops.mean((f - u) * (y_pred - u), axis=-1)
        return self.distance_weight * d + self.distribution_weight * r


class TestClusteringLossIsPerSample:
    """``call`` returns one value per row; the balance term is a batch constant in each."""

    def test_call_returns_one_value_per_sample(self):
        y_true, y_pred = _cluster_batch()
        out = ClusteringLoss().call(_f32(y_true), _f32(y_pred))
        assert tuple(out.shape) == (BATCH,)

    def test_rows_equal_the_float64_reference(self):
        y_true, y_pred = _cluster_batch()
        for wd, wdi in [(1.0, 0.5), (2.0, 0.3), (0.5, 3.0)]:
            out = ClusteringLoss(distance_weight=wd, distribution_weight=wdi).call(
                _f32(y_true), _f32(y_pred)
            )
            ref = _rows64(y_true, y_pred, wd, wdi)
            assert np.abs(out.numpy() - ref).max() < ROW_ATOL, (wd, wdi)

    def test_default_value_is_the_old_scalar(self):
        """Invariant: the default reduction value does not move (measured 1.5e-8)."""
        y_true, y_pred = _cluster_batch()
        value = float(ClusteringLoss()(_f32(y_true), _f32(y_pred)))
        assert abs(value - _old_scalar64(y_true, y_pred)) < ROW_ATOL

    def test_default_gradient_is_the_old_scalar_gradient(self):
        y_true, y_pred = _cluster_batch()
        p = tf.Variable(_f32(y_pred))
        with tf.GradientTape() as tape:
            value = ClusteringLoss()(_f32(y_true), p)
        new = tape.gradient(value, p).numpy()

        p2 = tf.Variable(_f32(y_pred))
        with tf.GradientTape() as tape:
            d = tf.reduce_mean(tf.square(p2 - _f32(y_true)))
            c = tf.reduce_mean(
                tf.square(tf.reduce_mean(p2, axis=0) - 1.0 / CLUSTERS)
            )
            old_value = 1.0 * d + 0.5 * c
        old = tape.gradient(old_value, p2).numpy()
        assert np.abs(new - old).max() < 1e-6

    def test_reduction_none_returns_the_rows(self):
        """``reduction`` is live: 'none' was a shape-() no-op before this fix."""
        y_true, y_pred = _cluster_batch()
        out = ClusteringLoss(reduction="none")(_f32(y_true), _f32(y_pred))
        assert tuple(out.shape) == (BATCH,)
        assert np.abs(np.asarray(out) - _rows64(y_true, y_pred)).max() < ROW_ATOL

    def test_reduction_sum_is_batch_times_the_default(self):
        y_true, y_pred = _cluster_batch()
        default = float(ClusteringLoss()(_f32(y_true), _f32(y_pred)))
        summed = float(ClusteringLoss(reduction="sum")(_f32(y_true), _f32(y_pred)))
        assert abs(summed - BATCH * default) < 1e-6
        assert abs(summed - default) > 0.1  # it was equal to the default before

    @pytest.mark.parametrize("reduction", ["none", "sum", "sum_over_batch_size"])
    def test_config_round_trip_keeps_a_live_reduction(self, reduction):
        original = ClusteringLoss(
            distance_weight=2.0, distribution_weight=0.3, reduction=reduction
        )
        restored = ClusteringLoss.from_config(original.get_config())
        assert restored.reduction == reduction
        y_true, y_pred = _cluster_batch()
        a = np.asarray(original(_f32(y_true), _f32(y_pred)))
        b = np.asarray(restored(_f32(y_true), _f32(y_pred)))
        assert a.shape == b.shape == ((BATCH,) if reduction == "none" else ())
        assert np.allclose(a, b, atol=0.0)

    def test_weighted_value_scales_the_balance_term_by_the_mean_weight(self):
        """``loss(w) = wd * sum(d * w)/B + wdi * c * mean(w)``; drop-a-row form for w=[1,1,1,0]."""
        y_true, y_pred = _cluster_batch()
        w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
        weighted = float(ClusteringLoss()(_f32(y_true), _f32(y_pred), sample_weight=w))
        d = _distance_rows64(y_true, y_pred)
        c = _balance64(y_pred)
        assert abs(weighted - (np.sum(d * w) / BATCH + 0.5 * c * w.mean())) < ROW_ATOL
        # Equivalent: loss(w) * B/(B-1) == wd * mean(d[:3]) + wdi * c_full.
        assert abs(weighted * BATCH / (BATCH - 1) - (d[:3].mean() + 0.5 * c)) < ROW_ATOL

    def test_a_zero_sample_weight_does_not_drop_the_row_s_share_of_the_balance_penalty(self):
        """BATCH-COUPLED, pinned: zeroing a row is NOT the loss of the batch without it.

        ``call`` never sees ``sample_weight``, so the balance penalty ``c`` is the one of
        the FULL batch. The zero-weighted value matches the reference built with ``c_full``
        and misses ``loss(batch[:3])`` by the measured gap (about 2.9e-3 here).
        """
        y_true, y_pred = _cluster_batch()
        w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
        weighted = float(ClusteringLoss()(_f32(y_true), _f32(y_pred), sample_weight=w))
        dropped = float(ClusteringLoss()(_f32(y_true[:3]), _f32(y_pred[:3])))
        d = _distance_rows64(y_true, y_pred)
        with_full_c = d[:3].mean() + 0.5 * _balance64(y_pred)
        with_kept_c = d[:3].mean() + 0.5 * _balance64(y_pred[:3])

        assert abs(weighted * BATCH / (BATCH - 1) - with_full_c) < ROW_ATOL
        assert abs(dropped - with_kept_c) < ROW_ATOL
        gap = abs(weighted * BATCH / (BATCH - 1) - dropped)
        assert gap > 1e-3, gap

    def test_a_zero_weighted_row_still_moves_the_balance_gradient(self):
        """The coupling in gradient form: only the row's own DISTANCE share is removed."""
        y_true, y_pred = _cluster_batch()
        w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
        p = tf.Variable(_f32(y_pred))
        with tf.GradientTape() as tape:
            value = ClusteringLoss()(_f32(y_true), p, sample_weight=w)
        grad = tape.gradient(value, p).numpy()
        # Balance gradient reaches the zero-weighted row through mean_b(p).
        assert np.abs(grad[3]).max() > 1e-4

    # DECISION plan-2026-11-11T223825-64d1b401/D-003 (test): the guard for the broadcast
    # balance term in ClusteringLoss.call. It fails when each row carries a different
    # share of the balance penalty (the alignment decomposition of option (c), signed
    # rows) or when the penalty stops being the batch value c.
    def test_the_balance_penalty_is_one_batch_constant_in_every_row(self):
        assert _balance_is_one_constant_for_every_row(ClusteringLoss())
        assert _balance_is_one_constant_for_every_row(
            ClusteringLoss(distance_weight=2.0, distribution_weight=0.3), wd=2.0, wdi=0.3
        )

    def test_the_guard_rejects_a_per_row_attribution_with_the_same_mean(self):
        y_true, y_pred = _cluster_batch()
        alt = _AlignmentClustering()
        same_default = float(alt(_f32(y_true), _f32(y_pred)))
        assert abs(same_default - _old_scalar64(y_true, y_pred)) < 1e-6
        assert not _balance_is_one_constant_for_every_row(alt)
        assert not _returns_per_row(alt)

    def test_the_per_row_tests_reject_the_scalar_returning_class(self):
        """ANTI-VACUITY: the pre-fix scalar fails the shape and reference checks."""
        assert _returns_per_row(ClusteringLoss())
        assert not _returns_per_row(_ScalarClustering())
        y_true, y_pred = _cluster_batch()
        assert tuple(_ScalarClustering(reduction="none")(_f32(y_true), _f32(y_pred)).shape) == ()

    @pytest.mark.parametrize("dtype", ["float64", "float16"])
    def test_a_direct_call_in_another_float_dtype_is_finite_per_row(self, dtype):
        """The ``1/K`` constant follows the input dtype (a float32 cast raised on float64)."""
        y_true, y_pred = _cluster_batch()
        out = ClusteringLoss().call(
            tf.constant(y_true.astype(dtype)), tf.constant(y_pred.astype(dtype))
        )
        assert tuple(out.shape) == (BATCH,)
        assert out.dtype == tf.as_dtype(dtype)
        values = out.numpy().astype("float64")
        assert np.isfinite(values).all()
        atol = 1e-12 if dtype == "float64" else 2e-3
        assert np.abs(values - _rows64(y_true, y_pred)).max() < atol

    def test_float16_inputs_through_the_loss_object_match_float32(self):
        """Through ``__call__`` the inputs are cast to float32 first, as under mixed_float16."""
        y_true, y_pred = _cluster_batch()
        base = float(ClusteringLoss()(_f32(y_true), _f32(y_pred)))
        half = float(
            ClusteringLoss()(
                tf.constant(y_true.astype("float16")), tf.constant(y_pred.astype("float16"))
            )
        )
        assert abs(base - half) < 2e-3

    def test_the_docstrings_say_one_value_per_sample_and_batch_coupled(self):
        doc = " ".join((ClusteringLoss.__doc__ or "").split())
        assert "batch-coupled" in doc
        assert "one value per sample" in " ".join(
            (ClusteringLoss.call.__doc__ or "").split()
        ).lower()


def _cluster_batch_of(batch):
    """`_cluster_batch()` tiled to ``batch`` rows as float32 (5 makes batch == CLUSTERS)."""
    y_true, y_pred = _cluster_batch()
    index = np.arange(batch) % BATCH
    return _f32(y_true[index]), _f32(y_pred[index])


class TestClusteringLossRefusesWhatItCannotCombine:
    """A Keras mask of rank 2 or more and a per-timestep weight cannot join ``(batch,)`` rows.

    Keras multiplies the mask of ``y_pred`` into the rows. A ``(batch, K)`` mask against
    ``(batch,)`` rows is a shape error, and with ``batch == K == 5`` it broadcasts and
    silently mis-weights the rows. The loss gives such a mask no meaning, so ``__call__``
    refuses it with a ``ValueError``; a rank-1 per-sample mask is a 0/1 sample weight
    and is left to Keras (plan-2026-11-11T223825-64d1b401, D-014).
    """

    @pytest.mark.parametrize("batch", [4, 5])
    @pytest.mark.parametrize("extra", [(CLUSTERS,), (CLUSTERS, 2)])
    def test_a_keras_mask_on_y_pred_raises_a_clear_error(self, batch, extra):
        y_true, y_pred = _cluster_batch_of(batch)
        assert_refuses_a_keras_mask(ClusteringLoss, y_true, y_pred, (batch,) + extra)

    @pytest.mark.parametrize("batch", [4, 5])
    def test_a_per_timestep_sample_weight_raises_a_clear_error(self, batch):
        y_true, y_pred = _cluster_batch_of(batch)
        assert_refuses_per_timestep_weights(ClusteringLoss, y_true, y_pred, (batch, CLUSTERS))

    def test_a_per_sample_mask_and_a_column_weight_still_work(self):
        y_true, y_pred = _cluster_batch_of(4)
        assert_a_per_sample_mask_is_left_to_keras(ClusteringLoss, y_true, y_pred)
        w = np.array([1.0, 0.5, 2.0, 0.0], "float32")
        flat = float(ClusteringLoss()(y_true, y_pred, sample_weight=w))
        column = float(ClusteringLoss()(y_true, y_pred, sample_weight=w[:, None]))
        assert abs(flat - column) < 1e-7

    def test_the_docstring_states_the_mask_rule(self):
        doc = " ".join((ClusteringLoss.__doc__ or "").split())
        assert "Keras mask" in doc and "ValueError" in doc


if __name__ == "__main__":
    pytest.main([__file__])