"""
Test suite for Deep Supervision Weight Scheduling Module.

This module contains comprehensive tests for all deep supervision scheduling
functions and the schedule builder. Tests cover:
- Correct weight generation at various training progress points
- Proper normalization (weights sum to 1.0)
- Parameter validation and error handling
- Edge cases and boundary conditions
- Schedule builder configuration
- Weight order inversion
"""

import numpy as np
import pytest

import keras

from dl_techniques.optimization.deep_supervision import (
    schedule_builder,
    constant_equal_schedule,
    constant_low_to_high_schedule,
    constant_high_to_low_schedule,
    linear_low_to_high_schedule,
    non_linear_low_to_high_schedule,
    custom_sigmoid_low_to_high_schedule,
    scale_by_scale_low_to_high_schedule,
    cosine_annealing_schedule,
    curriculum_schedule,
    step_wise_schedule,
    _normalize_weights,
    ScheduleType
)


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def assert_weights_normalized(weights: np.ndarray, tol: float = 1e-6) -> None:
    """Assert that weights sum to 1.0 within tolerance.

    Args:
        weights: Weight array to check.
        tol: Tolerance for sum comparison.

    Raises:
        AssertionError: If weights don't sum to 1.0 within tolerance.
    """
    weight_sum = np.sum(weights)
    np.testing.assert_allclose(
        weight_sum,
        1.0,
        rtol=tol,
        atol=tol,
        err_msg=f"Weights should sum to 1.0, got {weight_sum}"
    )


def assert_weights_match(
        actual: np.ndarray,
        expected: np.ndarray,
        rtol: float = 1e-2,
        atol: float = 1e-2
) -> None:
    """Assert two weight arrays match within tolerance.

    Args:
        actual: Actual weights from schedule function.
        expected: Expected weights.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        AssertionError: If weights don't match within tolerance.
    """
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(actual),
        keras.ops.convert_to_numpy(expected),
        rtol=rtol,
        atol=atol,
        err_msg="Weights should match"
    )


# ---------------------------------------------------------------------
# Tests for _normalize_weights
# ---------------------------------------------------------------------

class TestNormalizeWeights:
    """Test suite for the _normalize_weights helper function."""

    def test_normalize_positive_weights(self) -> None:
        """Test normalization of positive weights."""
        weights = np.array([1.0, 2.0, 3.0])
        normalized = _normalize_weights(weights)

        expected = np.array([1 / 6, 2 / 6, 3 / 6])
        assert_weights_match(normalized, expected)
        assert_weights_normalized(normalized)

    def test_normalize_zero_weights(self) -> None:
        """Test normalization when all weights are zero."""
        weights = np.zeros(5)
        normalized = _normalize_weights(weights)

        expected = np.ones(5) / 5  # Should return uniform weights
        assert_weights_match(normalized, expected)
        assert_weights_normalized(normalized)

    def test_normalize_near_zero_weights(self) -> None:
        """Test normalization when weights sum to near-zero."""
        weights = np.array([1e-10, 1e-10, 1e-10])
        normalized = _normalize_weights(weights)

        expected = np.ones(3) / 3
        assert_weights_match(normalized, expected)
        assert_weights_normalized(normalized)

    def test_normalize_single_weight(self) -> None:
        """Test normalization with single weight."""
        weights = np.array([5.0])
        normalized = _normalize_weights(weights)

        expected = np.array([1.0])
        assert_weights_match(normalized, expected)


# ---------------------------------------------------------------------
# Tests for constant_equal_schedule
# ---------------------------------------------------------------------

class TestConstantEqualSchedule:
    """Test suite for constant_equal_schedule."""

    def test_equal_weights_start(self) -> None:
        """Test equal weights at training start."""
        weights = constant_equal_schedule(0.0, 5)
        expected = np.ones(5) / 5

        assert_weights_match(weights, expected)
        assert_weights_normalized(weights)

    def test_equal_weights_middle(self) -> None:
        """Test equal weights at training middle."""
        weights = constant_equal_schedule(0.5, 5)
        expected = np.ones(5) / 5

        assert_weights_match(weights, expected)
        assert_weights_normalized(weights)

    def test_equal_weights_end(self) -> None:
        """Test equal weights at training end."""
        weights = constant_equal_schedule(1.0, 5)
        expected = np.ones(5) / 5

        assert_weights_match(weights, expected)
        assert_weights_normalized(weights)

    def test_different_output_counts(self) -> None:
        """Test equal weights with different output counts."""
        for n in [2, 3, 4, 7, 10]:
            weights = constant_equal_schedule(0.5, n)
            expected = np.ones(n) / n

            assert_weights_match(weights, expected)
            assert_weights_normalized(weights)


# ---------------------------------------------------------------------
# Tests for constant_low_to_high_schedule
# ---------------------------------------------------------------------

class TestConstantLowToHighSchedule:
    """Test suite for constant_low_to_high_schedule."""

    def test_weights_favor_shallow(self) -> None:
        """Test that weights favor shallower (lower index) outputs."""
        weights = constant_low_to_high_schedule(0.5, 5)

        # Weights should decrease from index 0 to 4
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1], \
                f"Weight at index {i} should be greater than at {i + 1}"

        assert_weights_normalized(weights)

    def test_weights_constant_over_progress(self) -> None:
        """Test that weights remain constant throughout training."""
        weights_start = constant_low_to_high_schedule(0.0, 5)
        weights_mid = constant_low_to_high_schedule(0.5, 5)
        weights_end = constant_low_to_high_schedule(1.0, 5)

        assert_weights_match(weights_start, weights_mid)
        assert_weights_match(weights_mid, weights_end)

    def test_specific_values(self) -> None:
        """Test specific weight values for 5 outputs."""
        weights = constant_low_to_high_schedule(0.0, 5)
        # weights = [5, 4, 3, 2, 1] / 15 = [0.333, 0.267, 0.2, 0.133, 0.067]
        expected = np.array([5, 4, 3, 2, 1], dtype=np.float64) / 15

        assert_weights_match(weights, expected)


# ---------------------------------------------------------------------
# Tests for constant_high_to_low_schedule
# ---------------------------------------------------------------------

class TestConstantHighToLowSchedule:
    """Test suite for constant_high_to_low_schedule."""

    def test_weights_favor_deep(self) -> None:
        """Test that weights favor deeper (higher index) outputs."""
        weights = constant_high_to_low_schedule(0.5, 5)

        # Weights should increase from index 0 to 4
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1], \
                f"Weight at index {i} should be less than at {i + 1}"

        assert_weights_normalized(weights)

    def test_weights_constant_over_progress(self) -> None:
        """Test that weights remain constant throughout training."""
        weights_start = constant_high_to_low_schedule(0.0, 5)
        weights_mid = constant_high_to_low_schedule(0.5, 5)
        weights_end = constant_high_to_low_schedule(1.0, 5)

        assert_weights_match(weights_start, weights_mid)
        assert_weights_match(weights_mid, weights_end)

    def test_specific_values(self) -> None:
        """Test specific weight values for 5 outputs."""
        weights = constant_high_to_low_schedule(0.0, 5)
        # weights = [1, 2, 3, 4, 5] / 15 = [0.067, 0.133, 0.2, 0.267, 0.333]
        expected = np.array([1, 2, 3, 4, 5], dtype=np.float64) / 15

        assert_weights_match(weights, expected)


# ---------------------------------------------------------------------
# Tests for linear_low_to_high_schedule
# ---------------------------------------------------------------------

class TestLinearLowToHighSchedule:
    """Test suite for linear_low_to_high_schedule."""

    def test_start_favors_deep(self) -> None:
        """Test that start of training favors deep layers."""
        weights = linear_low_to_high_schedule(0.0, 5)
        expected = constant_high_to_low_schedule(0.0, 5)

        assert_weights_match(weights, expected)

    def test_end_favors_shallow(self) -> None:
        """Test that end of training favors shallow layers."""
        weights = linear_low_to_high_schedule(1.0, 5)
        expected = constant_low_to_high_schedule(0.0, 5)

        assert_weights_match(weights, expected)

    def test_middle_equal_weights(self) -> None:
        """Test that middle of training has equal weights."""
        weights = linear_low_to_high_schedule(0.5, 5)
        expected = np.ones(5) / 5

        assert_weights_match(weights, expected, atol=1e-5)

    def test_progress_at_25_percent(self) -> None:
        """Test weights at 25% progress."""
        weights = linear_low_to_high_schedule(0.25, 5)

        # Should be 75% toward deep, 25% toward shallow
        deep_weights = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        shallow_weights = np.array([5, 4, 3, 2, 1], dtype=np.float64)
        expected = (0.75 * deep_weights + 0.25 * shallow_weights)
        expected = expected / np.sum(expected)

        assert_weights_match(weights, expected)
        assert_weights_normalized(weights)

    def test_monotonic_transition(self) -> None:
        """Test that first output weight increases monotonically."""
        prev_weight_0 = 0.0
        for progress in np.linspace(0.0, 1.0, 11):
            weights = linear_low_to_high_schedule(progress, 5)
            assert weights[0] >= prev_weight_0, \
                f"Weight[0] should increase monotonically"
            prev_weight_0 = weights[0]


# ---------------------------------------------------------------------
# Tests for non_linear_low_to_high_schedule
# ---------------------------------------------------------------------

class TestNonLinearLowToHighSchedule:
    """Test suite for non_linear_low_to_high_schedule."""

    def test_start_favors_deep(self) -> None:
        """Test that start of training favors deep layers."""
        weights = non_linear_low_to_high_schedule(0.0, 5)
        expected = constant_high_to_low_schedule(0.0, 5)

        assert_weights_match(weights, expected)

    def test_end_favors_shallow(self) -> None:
        """Test that end of training favors shallow layers."""
        weights = non_linear_low_to_high_schedule(1.0, 5)
        expected = constant_low_to_high_schedule(0.0, 5)

        assert_weights_match(weights, expected)

    def test_50_percent_still_biased_deep(self) -> None:
        """Test that at 50% progress, still biased toward deep layers."""
        weights_50 = non_linear_low_to_high_schedule(0.5, 5)
        weights_linear_50 = linear_low_to_high_schedule(0.5, 5)

        # At 50%, non-linear uses factor 0.25, so should favor deep more
        # Output 4 (deepest) should have higher weight in non-linear
        assert weights_50[4] > weights_linear_50[4], \
            "Non-linear should favor deep layers more at 50%"

        assert_weights_normalized(weights_50)

    def test_quadratic_progression(self) -> None:
        """Test quadratic nature of progression."""
        weights_50 = non_linear_low_to_high_schedule(0.5, 5)
        weights_25_linear = linear_low_to_high_schedule(0.25, 5)

        # At 50% with quadratic, factor is 0.25, same as 25% linear
        assert_weights_match(weights_50, weights_25_linear, atol=1e-5)


# ---------------------------------------------------------------------
# Tests for custom_sigmoid_low_to_high_schedule
# ---------------------------------------------------------------------

class TestCustomSigmoidLowToHighSchedule:
    """Test suite for custom_sigmoid_low_to_high_schedule."""

    def test_before_transition_point(self) -> None:
        """Test weights before transition point favor deep layers."""
        weights = custom_sigmoid_low_to_high_schedule(
            0.2, 5, k=10.0, x0=0.5, transition_point=0.25
        )
        expected = constant_high_to_low_schedule(0.0, 5)

        assert_weights_match(weights, expected)

    def test_at_transition_point(self) -> None:
        """Test weights at transition point."""
        weights = custom_sigmoid_low_to_high_schedule(
            0.25, 5, k=10.0, x0=0.5, transition_point=0.25
        )
        expected = constant_high_to_low_schedule(0.0, 5)

        assert_weights_match(weights, expected)

    def test_after_full_transition(self) -> None:
        """Test weights after full transition favor shallow layers."""
        weights = custom_sigmoid_low_to_high_schedule(
            1.0, 5, k=10.0, x0=0.5, transition_point=0.25
        )
        expected = constant_low_to_high_schedule(0.0, 5)

        assert_weights_match(weights, expected, atol=1e-2)

    def test_different_k_values(self) -> None:
        """Test different steepness values."""
        weights_sharp = custom_sigmoid_low_to_high_schedule(
            0.625, 5, k=20.0, x0=0.5, transition_point=0.25
        )
        weights_gradual = custom_sigmoid_low_to_high_schedule(
            0.625, 5, k=5.0, x0=0.5, transition_point=0.25
        )

        # Both should be normalized
        assert_weights_normalized(weights_sharp)
        assert_weights_normalized(weights_gradual)

        # Sharper transition should be closer to extremes
        equal_weights = np.ones(5) / 5
        dist_sharp = np.linalg.norm(weights_sharp - equal_weights)
        dist_gradual = np.linalg.norm(weights_gradual - equal_weights)
        assert dist_sharp >= dist_gradual, \
            "Sharper k should deviate more from equal weights"

    def test_invalid_parameters(self) -> None:
        """Test that invalid parameters raise errors through schedule_builder."""
        # Invalid k
        with pytest.raises(ValueError, match="k must be a positive number"):
            config = {
                "type": "custom_sigmoid_low_to_high",
                "config": {"k": -1.0}
            }
            schedule_builder(config, 5)

        # Invalid x0
        with pytest.raises(ValueError, match="x0 must be in range"):
            config = {
                "type": "custom_sigmoid_low_to_high",
                "config": {"x0": 1.5}
            }
            schedule_builder(config, 5)

        # Invalid transition_point
        with pytest.raises(ValueError, match="transition_point must be in range"):
            config = {
                "type": "custom_sigmoid_low_to_high",
                "config": {"transition_point": -0.1}
            }
            schedule_builder(config, 5)


# ---------------------------------------------------------------------
# Tests for scale_by_scale_low_to_high_schedule
# ---------------------------------------------------------------------

class TestScaleByScaleLowToHighSchedule:
    """Test suite for scale_by_scale_low_to_high_schedule."""

    def test_one_hot_encoding(self) -> None:
        """Test that exactly one output is active at any time."""
        for progress in np.linspace(0.0, 0.99, 20):
            weights = scale_by_scale_low_to_high_schedule(progress, 5)

            # Exactly one weight should be 1.0, others 0.0
            assert np.sum(weights == 1.0) == 1, \
                f"Exactly one weight should be 1.0 at progress {progress}"
            assert np.sum(weights == 0.0) == 4, \
                f"Four weights should be 0.0 at progress {progress}"

            assert_weights_normalized(weights)

    def test_start_deepest_layer(self) -> None:
        """Test that training starts with deepest layer."""
        weights = scale_by_scale_low_to_high_schedule(0.0, 5)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

        assert_weights_match(weights, expected)

    def test_end_shallowest_layer(self) -> None:
        """Test that training ends with shallowest layer."""
        weights = scale_by_scale_low_to_high_schedule(1.0, 5)
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        assert_weights_match(weights, expected)

    def test_progression_through_scales(self) -> None:
        """Test progression through all scales."""
        # At 10%, should be on scale 4 (deepest)
        weights_10 = scale_by_scale_low_to_high_schedule(0.1, 5)
        assert weights_10[4] == 1.0

        # At 30%, should be on scale 3
        weights_30 = scale_by_scale_low_to_high_schedule(0.3, 5)
        assert weights_30[3] == 1.0

        # At 50%, should be on scale 2
        weights_50 = scale_by_scale_low_to_high_schedule(0.5, 5)
        assert weights_50[2] == 1.0

        # At 70%, should be on scale 1
        weights_70 = scale_by_scale_low_to_high_schedule(0.7, 5)
        assert weights_70[1] == 1.0

        # At 90%, should be on scale 0 (shallowest)
        weights_90 = scale_by_scale_low_to_high_schedule(0.9, 5)
        assert weights_90[0] == 1.0


# ---------------------------------------------------------------------
# Tests for cosine_annealing_schedule
# ---------------------------------------------------------------------

class TestCosineAnnealingSchedule:
    """Test suite for cosine_annealing_schedule."""

    def test_weights_normalized(self) -> None:
        """Test that weights are always normalized."""
        for progress in np.linspace(0.0, 1.0, 20):
            weights = cosine_annealing_schedule(progress, 5, frequency=3.0)
            assert_weights_normalized(weights)

    def test_oscillation_occurs(self) -> None:
        """Test that weights oscillate during training."""
        weights_0 = cosine_annealing_schedule(0.0, 5, frequency=3.0)
        weights_17 = cosine_annealing_schedule(0.167, 5, frequency=3.0)  # 1/6 of cycle

        # Weights should be different due to oscillation
        assert not np.allclose(weights_0, weights_17), \
            "Weights should oscillate"

    def test_annealing_reduces_amplitude(self) -> None:
        """Test that amplitude decreases over time."""
        weights_start = cosine_annealing_schedule(0.0, 5, frequency=3.0, final_ratio=0.5)
        weights_end = cosine_annealing_schedule(1.0, 5, frequency=3.0, final_ratio=0.5)

        # Calculate variance as measure of amplitude
        var_start = np.var(weights_start)
        var_end = np.var(weights_end)

        assert var_end < var_start, \
            "Variance should decrease due to annealing"

    def test_final_ratio_zero(self) -> None:
        """Test with final_ratio=0 converges to equal weights."""
        weights = cosine_annealing_schedule(1.0, 5, frequency=3.0, final_ratio=0.0)
        expected = np.ones(5) / 5

        assert_weights_match(weights, expected, atol=1e-5)

    def test_different_frequencies(self) -> None:
        """Test different oscillation frequencies."""
        weights_low_freq = cosine_annealing_schedule(0.5, 5, frequency=1.0)
        weights_high_freq = cosine_annealing_schedule(0.5, 5, frequency=10.0)

        # Both should be normalized
        assert_weights_normalized(weights_low_freq)
        assert_weights_normalized(weights_high_freq)

    def test_invalid_parameters(self) -> None:
        """Test that invalid parameters raise errors through schedule_builder."""
        # Invalid frequency
        with pytest.raises(ValueError, match="frequency must be a positive number"):
            config = {
                "type": "cosine_annealing",
                "config": {"frequency": -1.0}
            }
            schedule_builder(config, 5)

        # Invalid final_ratio
        with pytest.raises(ValueError, match="final_ratio must be in range"):
            config = {
                "type": "cosine_annealing",
                "config": {"final_ratio": 1.5}
            }
            schedule_builder(config, 5)


# ---------------------------------------------------------------------
# Tests for curriculum_schedule
# ---------------------------------------------------------------------

class TestCurriculumSchedule:
    """Test suite for curriculum_schedule."""

    def test_start_one_active(self) -> None:
        """Test that training starts with only deepest layer active."""
        weights = curriculum_schedule(0.0, 5, max_active_outputs=5, activation_strategy='linear')

        # Only deepest (index 4) should be active
        assert weights[4] == 1.0
        assert np.sum(weights[:4]) == 0.0
        assert_weights_normalized(weights)

    def test_end_all_active(self) -> None:
        """Test that training ends with all layers active."""
        weights = curriculum_schedule(1.0, 5, max_active_outputs=5, activation_strategy='linear')
        expected = np.ones(5) / 5

        assert_weights_match(weights, expected)

    def test_progressive_activation_linear(self) -> None:
        """Test progressive activation with linear strategy."""
        # At 40%, should have 2 active (5 * 0.4 = 2)
        weights_40 = curriculum_schedule(0.4, 5, max_active_outputs=5, activation_strategy='linear')
        assert np.sum(weights_40 > 0) == 2, "Should have 2 active outputs at 40%"
        assert weights_40[4] > 0 and weights_40[3] > 0, "Should activate from deep"

        # At 60%, should have 3 active (5 * 0.6 = 3)
        weights_60 = curriculum_schedule(0.6, 5, max_active_outputs=5, activation_strategy='linear')
        assert np.sum(weights_60 > 0) == 3, "Should have 3 active outputs at 60%"

    def test_progressive_activation_exp(self) -> None:
        """Test progressive activation with exponential strategy."""
        # At 50%, should have 1-2 active (5 * 0.5^2 = 1.25)
        weights = curriculum_schedule(0.5, 5, max_active_outputs=5, activation_strategy='exp')
        active_count = np.sum(weights > 0)
        assert active_count <= 2, "Should have fewer active with exp strategy at 50%"
        assert_weights_normalized(weights)

    def test_max_active_outputs_limits(self) -> None:
        """Test that max_active_outputs limits active outputs."""
        weights = curriculum_schedule(1.0, 5, max_active_outputs=3, activation_strategy='linear')
        active_count = np.sum(weights > 0)
        assert active_count == 3, "Should have at most 3 active outputs"
        assert_weights_normalized(weights)

    def test_equal_weight_distribution(self) -> None:
        """Test that active outputs receive equal weights."""
        weights = curriculum_schedule(0.6, 5, max_active_outputs=5, activation_strategy='linear')
        active_weights = weights[weights > 0]

        # All active weights should be equal
        assert np.allclose(active_weights, active_weights[0]), \
            "Active outputs should have equal weights"

    def test_invalid_activation_strategy(self) -> None:
        """Test that invalid activation strategy raises error."""
        with pytest.raises(ValueError, match="activation_strategy must be"):
            config = {
                "type": "curriculum",
                "config": {"activation_strategy": "invalid"}
            }
            schedule_builder(config, 5)


# ---------------------------------------------------------------------
# Tests for step_wise_schedule
# ---------------------------------------------------------------------

class TestStepWiseSchedule:
    """Test suite for step_wise_schedule."""

    def test_before_threshold_linear(self) -> None:
        """Test linear progression before threshold."""
        # At 25% with threshold 0.5, should be 50% through linear progression
        weights = step_wise_schedule(0.25, 5, threshold=0.5)
        expected = linear_low_to_high_schedule(0.5, 5)

        assert_weights_match(weights, expected)

    def test_at_threshold_switch(self) -> None:
        """Test hard switch at threshold."""
        weights = step_wise_schedule(0.5, 5, threshold=0.5)
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        assert_weights_match(weights, expected)

    def test_after_threshold_stays_switched(self) -> None:
        """Test that weights stay on shallowest after threshold."""
        weights_50 = step_wise_schedule(0.5, 5, threshold=0.5)
        weights_75 = step_wise_schedule(0.75, 5, threshold=0.5)
        weights_100 = step_wise_schedule(1.0, 5, threshold=0.5)

        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert_weights_match(weights_50, expected)
        assert_weights_match(weights_75, expected)
        assert_weights_match(weights_100, expected)

    def test_different_thresholds(self) -> None:
        """Test with different threshold values."""
        # Early threshold (0.3)
        weights_early = step_wise_schedule(0.5, 5, threshold=0.3)
        expected_early = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert_weights_match(weights_early, expected_early)

        # Late threshold (0.9)
        weights_late = step_wise_schedule(0.5, 5, threshold=0.9)
        # Should still be in linear phase at 50% (50/90 ≈ 0.556)
        expected_late = linear_low_to_high_schedule(0.5 / 0.9, 5)
        assert_weights_match(weights_late, expected_late)

    def test_threshold_zero(self) -> None:
        """Test edge case with threshold=0."""
        weights = step_wise_schedule(0.0, 5, threshold=0.0)
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        assert_weights_match(weights, expected)

    def test_threshold_one(self) -> None:
        """Test edge case with threshold=1."""
        weights = step_wise_schedule(0.5, 5, threshold=1.0)
        # Should be linear throughout
        expected = linear_low_to_high_schedule(0.5, 5)

        assert_weights_match(weights, expected)

    def test_invalid_threshold(self) -> None:
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError, match="threshold must be in range"):
            config = {
                "type": "step_wise",
                "config": {"threshold": 1.5}
            }
            schedule_builder(config, 5)


# ---------------------------------------------------------------------
# Tests for schedule_builder
# ---------------------------------------------------------------------

class TestScheduleBuilder:
    """Test suite for schedule_builder function."""

    def test_build_all_schedule_types(self) -> None:
        """Test building all available schedule types."""
        schedule_types = [
            "constant_equal",
            "constant_low_to_high",
            "constant_high_to_low",
            "linear_low_to_high",
            "non_linear_low_to_high",
            "custom_sigmoid_low_to_high",
            "scale_by_scale_low_to_high",
            "cosine_annealing",
            "curriculum",
            "step_wise"
        ]

        for schedule_type in schedule_types:
            config = {"type": schedule_type}
            scheduler = schedule_builder(config, 5)

            # Test that scheduler is callable and returns valid weights
            weights = scheduler(0.5)
            assert isinstance(weights, np.ndarray)
            assert len(weights) == 5
            assert_weights_normalized(weights)

    def test_invalid_config_type(self) -> None:
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="config must be a dictionary"):
            schedule_builder("not a dict", 5)

    def test_invalid_no_outputs(self) -> None:
        """Test that invalid no_outputs raises ValueError."""
        config = {"type": "constant_equal"}

        with pytest.raises(ValueError, match="no_outputs must be a positive integer"):
            schedule_builder(config, 0)

        with pytest.raises(ValueError, match="no_outputs must be a positive integer"):
            schedule_builder(config, -1)

    def test_invalid_invert_order_type(self) -> None:
        """Test that invalid invert_order type raises TypeError."""
        config = {"type": "constant_equal"}

        with pytest.raises(TypeError, match="invert_order must be a boolean"):
            schedule_builder(config, 5, invert_order="True")

    def test_missing_type_in_config(self) -> None:
        """Test that missing 'type' key raises TypeError."""
        config = {"config": {}}

        with pytest.raises(TypeError, match="'type' must be a string"):
            schedule_builder(config, 5)

    def test_unknown_schedule_type(self) -> None:
        """Test that unknown schedule type raises ValueError."""
        config = {"type": "unknown_schedule"}

        with pytest.raises(ValueError, match="Unknown deep supervision schedule type"):
            schedule_builder(config, 5)

    def test_invalid_config_params_type(self) -> None:
        """Test that invalid config params type raises TypeError."""
        config = {
            "type": "constant_equal",
            "config": "not a dict"
        }

        with pytest.raises(TypeError, match="'config' must be a dictionary"):
            schedule_builder(config, 5)

    def test_invert_order_functionality(self) -> None:
        """Test that invert_order reverses the weight array."""
        config = {"type": "constant_low_to_high"}

        scheduler_normal = schedule_builder(config, 5, invert_order=False)
        scheduler_inverted = schedule_builder(config, 5, invert_order=True)

        weights_normal = scheduler_normal(0.5)
        weights_inverted = scheduler_inverted(0.5)

        # Inverted should be reverse of normal
        assert_weights_match(weights_inverted, weights_normal[::-1])

    def test_schedule_with_params(self) -> None:
        """Test building schedule with custom parameters."""
        config = {
            "type": "custom_sigmoid_low_to_high",
            "config": {
                "k": 15.0,
                "x0": 0.6,
                "transition_point": 0.3
            }
        }

        scheduler = schedule_builder(config, 5)
        weights = scheduler(0.5)

        assert isinstance(weights, np.ndarray)
        assert_weights_normalized(weights)

    def test_scheduler_is_stateless(self) -> None:
        """Test that scheduler is stateless (same input = same output)."""
        config = {"type": "linear_low_to_high"}
        scheduler = schedule_builder(config, 5)

        weights_1 = scheduler(0.5)
        weights_2 = scheduler(0.5)
        weights_3 = scheduler(0.5)

        assert_weights_match(weights_1, weights_2)
        assert_weights_match(weights_2, weights_3)


# ---------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------

class TestScheduleIntegration:
    """Integration tests for schedule behavior."""

    def test_all_schedules_produce_valid_weights_range(self) -> None:
        """Test that all schedules produce valid weights across progress range."""
        configs = [
            {"type": "constant_equal"},
            {"type": "constant_low_to_high"},
            {"type": "constant_high_to_low"},
            {"type": "linear_low_to_high"},
            {"type": "non_linear_low_to_high"},
            {"type": "custom_sigmoid_low_to_high"},
            {"type": "scale_by_scale_low_to_high"},
            {"type": "cosine_annealing"},
            {"type": "curriculum"},
            {"type": "step_wise"},
        ]

        for config in configs:
            scheduler = schedule_builder(config, 5)

            for progress in np.linspace(0.0, 1.0, 21):
                weights = scheduler(progress)

                # Check basic properties
                assert len(weights) == 5
                assert np.all(weights >= 0), \
                    f"All weights should be non-negative for {config['type']}"
                assert_weights_normalized(weights)

    def test_edge_case_single_output(self) -> None:
        """Test schedules with single output."""
        config = {"type": "linear_low_to_high"}
        scheduler = schedule_builder(config, 1)

        weights = scheduler(0.5)
        expected = np.array([1.0])

        assert_weights_match(weights, expected)

    def test_edge_case_two_outputs(self) -> None:
        """Test schedules with two outputs."""
        config = {"type": "linear_low_to_high"}
        scheduler = schedule_builder(config, 2)

        # At start, should favor deep (index 1)
        weights_start = scheduler(0.0)
        assert weights_start[1] > weights_start[0]

        # At end, should favor shallow (index 0)
        weights_end = scheduler(1.0)
        assert weights_end[0] > weights_end[1]

        assert_weights_normalized(weights_start)
        assert_weights_normalized(weights_end)

    def test_schedule_consistency_across_calls(self) -> None:
        """Test that schedules are consistent across multiple calls."""
        config = {"type": "cosine_annealing", "config": {"frequency": 3.0}}
        scheduler = schedule_builder(config, 5)

        # Call multiple times and ensure consistency
        for _ in range(10):
            weights_1 = scheduler(0.33)
            weights_2 = scheduler(0.33)
            assert_weights_match(weights_1, weights_2)


# ---------------------------------------------------------------------
# Parameterized Tests
# ---------------------------------------------------------------------

class TestParameterizedSchedules:
    """Parameterized tests across multiple schedules."""

    @pytest.mark.parametrize("no_outputs", [2, 3, 5, 7, 10])
    def test_different_output_counts(self, no_outputs: int) -> None:
        """Test schedules with different output counts."""
        config = {"type": "linear_low_to_high"}
        scheduler = schedule_builder(config, no_outputs)

        weights = scheduler(0.5)
        assert len(weights) == no_outputs
        assert_weights_normalized(weights)

    @pytest.mark.parametrize("progress", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_different_progress_values(self, progress: float) -> None:
        """Test schedules at different progress points."""
        config = {"type": "non_linear_low_to_high"}
        scheduler = schedule_builder(config, 5)

        weights = scheduler(progress)
        assert_weights_normalized(weights)

    @pytest.mark.parametrize("schedule_type", [e.value for e in ScheduleType])
    def test_all_schedule_types_enum(self, schedule_type: str) -> None:
        """Test all schedule types from enum."""
        config = {"type": schedule_type}
        scheduler = schedule_builder(config, 5)

        weights = scheduler(0.5)
        assert isinstance(weights, np.ndarray)
        assert_weights_normalized(weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])