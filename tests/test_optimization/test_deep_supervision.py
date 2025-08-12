import pytest
import numpy as np
from typing import Dict, Any

from dl_techniques.optimization.deep_supervision import (
    schedule_builder,
    ScheduleType,
)
from dl_techniques.utils.constants import TYPE_STR, CONFIG_STR


class TestScheduleBuilderValidation:
    """Tests for input validation in schedule_builder function."""

    def test_invalid_config_type(self):
        """Test that non-dict config raises TypeError."""
        with pytest.raises(TypeError, match="config must be a dictionary"):
            schedule_builder("not_a_dict", 5)

        with pytest.raises(TypeError, match="config must be a dictionary"):
            schedule_builder(["not", "dict"], 5)

        with pytest.raises(TypeError, match="config must be a dictionary"):
            schedule_builder(None, 5)

    def test_invalid_no_outputs(self):
        """Test that invalid no_outputs raises ValueError."""
        config = {TYPE_STR: ScheduleType.CONSTANT_EQUAL}

        # Test zero outputs
        with pytest.raises(ValueError, match="no_outputs must be a positive integer"):
            schedule_builder(config, 0)

        # Test negative outputs
        with pytest.raises(ValueError, match="no_outputs must be a positive integer"):
            schedule_builder(config, -1)

        # Test non-integer type
        with pytest.raises(ValueError, match="no_outputs must be a positive integer"):
            schedule_builder(config, 3.5)

    def test_invalid_invert_order_type(self):
        """Test that non-bool invert_order raises TypeError."""
        config = {TYPE_STR: ScheduleType.CONSTANT_EQUAL}

        with pytest.raises(TypeError, match="invert_order must be a boolean"):
            schedule_builder(config, 5, invert_order="true")

        with pytest.raises(TypeError, match="invert_order must be a boolean"):
            schedule_builder(config, 5, invert_order=1)

    def test_missing_schedule_type(self):
        """Test that missing schedule type raises ValueError."""
        with pytest.raises(ValueError, match="schedule_type cannot be None"):
            schedule_builder({}, 5)

        with pytest.raises(ValueError, match="schedule_type cannot be None"):
            schedule_builder({CONFIG_STR: {}}, 5)

    def test_invalid_schedule_type(self):
        """Test that invalid schedule type raises ValueError."""
        config = {TYPE_STR: "invalid_schedule_type"}
        with pytest.raises(ValueError, match="Unknown deep supervision schedule_type"):
            schedule_builder(config, 5)

    def test_non_string_schedule_type(self):
        """Test that non-string schedule type raises TypeError."""
        config = {TYPE_STR: 123}
        with pytest.raises(TypeError, match="schedule_type must be a string"):
            schedule_builder(config, 5)

    def test_invalid_config_params_type(self):
        """Test that non-dict config params raises TypeError."""
        config = {
            TYPE_STR: ScheduleType.CONSTANT_EQUAL,
            CONFIG_STR: "not_a_dict"
        }
        with pytest.raises(TypeError, match="'config' must be a dictionary"):
            schedule_builder(config, 5)


class TestConstantSchedules:
    """Tests for constant weighting schedules."""

    @pytest.fixture
    def base_config(self) -> Dict[str, Any]:
        """Create a base configuration for testing."""
        return {CONFIG_STR: {}}

    def test_constant_equal_schedule(self, base_config):
        """Test constant equal schedule output."""
        config = {TYPE_STR: ScheduleType.CONSTANT_EQUAL, **base_config}
        schedule_fn = schedule_builder(config, 5)

        # Test at different training percentages
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weights = schedule_fn(progress)

            # Check shape and sum
            assert weights.shape == (5,)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)

            # Check equal weights
            expected_weight = 1.0 / 5.0
            assert np.allclose(weights, expected_weight, atol=1e-10)

    def test_constant_low_to_high_schedule(self, base_config):
        """Test constant low to high schedule output."""
        config = {TYPE_STR: ScheduleType.CONSTANT_LOW_TO_HIGH, **base_config}
        schedule_fn = schedule_builder(config, 5)

        # Test at different training percentages
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weights = schedule_fn(progress)

            # Check shape and sum
            assert weights.shape == (5,)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)

            # Check increasing weights (output 0 < output 1 < ... < output 4)
            assert np.all(np.diff(weights) > 0)

            # Verify expected pattern: weights should be [1, 2, 3, 4, 5] normalized
            expected = np.array([1, 2, 3, 4, 5], dtype=np.float64)
            expected = expected / np.sum(expected)
            assert np.allclose(weights, expected, atol=1e-10)

    def test_constant_high_to_low_schedule(self, base_config):
        """Test constant high to low schedule output."""
        config = {TYPE_STR: ScheduleType.CONSTANT_HIGH_TO_LOW, **base_config}
        schedule_fn = schedule_builder(config, 5)

        # Test at different training percentages
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weights = schedule_fn(progress)

            # Check shape and sum
            assert weights.shape == (5,)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)

            # Check decreasing weights (output 0 > output 1 > ... > output 4)
            assert np.all(np.diff(weights) < 0)

            # Verify expected pattern: weights should be [5, 4, 3, 2, 1] normalized
            expected = np.array([5, 4, 3, 2, 1], dtype=np.float64)
            expected = expected / np.sum(expected)
            assert np.allclose(weights, expected, atol=1e-10)


class TestProgressiveSchedules:
    """Tests for schedules that change with training progress.

    Key indexing convention (from module docstring):
    - Output 0: Final inference output (highest resolution, shallowest)
    - Output (n-1): Deepest scale in the network (lowest resolution, deepest)

    Schedule naming convention:
    - "low_to_high" means transitioning from low indices (shallow) to high indices (deep)
    - This corresponds to transitioning from high resolution to low resolution focus
    """

    @pytest.fixture
    def base_config(self) -> Dict[str, Any]:
        """Create a base configuration for testing."""
        return {CONFIG_STR: {}}

    def test_linear_low_to_high_schedule(self, base_config):
        """Test linear low to high schedule transition."""
        config = {TYPE_STR: ScheduleType.LINEAR_LOW_TO_HIGH, **base_config}
        schedule_fn = schedule_builder(config, 3)

        # Test at start of training (should favor shallow layers - lower indices)
        weights_start = schedule_fn(0.0)
        assert weights_start.shape == (3,)
        assert np.isclose(np.sum(weights_start), 1.0, atol=1e-10)
        # At start: should be like constant_high_to_low [3, 2, 1] normalized
        # This favors index 0 (shallow layer)
        assert weights_start[0] > weights_start[1] > weights_start[2]

        # Test at end of training (should favor deeper layers - higher indices)
        weights_end = schedule_fn(1.0)
        assert weights_end.shape == (3,)
        assert np.isclose(np.sum(weights_end), 1.0, atol=1e-10)
        # At end: should be like constant_low_to_high [1, 2, 3] normalized
        # This favors index 2 (deep layer)
        assert weights_end[0] < weights_end[1] < weights_end[2]

        # Test middle of training
        weights_mid = schedule_fn(0.5)
        assert weights_mid.shape == (3,)
        assert np.isclose(np.sum(weights_mid), 1.0, atol=1e-10)

        # Verify transition direction
        # Shallow layer (index 0) should decrease from start to end
        assert weights_start[0] > weights_end[0]
        # Deep layer (index 2) should increase from start to end
        assert weights_start[2] < weights_end[2]

        # Verify all weights are non-negative
        assert np.all(weights_start >= 0)
        assert np.all(weights_mid >= 0)
        assert np.all(weights_end >= 0)

    def test_non_linear_low_to_high_schedule(self, base_config):
        """Test non-linear (quadratic) low to high schedule."""
        config = {TYPE_STR: ScheduleType.NON_LINEAR_LOW_TO_HIGH, **base_config}
        schedule_fn = schedule_builder(config, 4)

        # Test different progress points
        progress_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        for progress in progress_points:
            weights = schedule_fn(progress)
            assert weights.shape == (4,)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
            assert np.all(weights >= 0)

        # Verify quadratic behavior - should be different from linear
        weights_linear_mid = np.array([2.5, 2.5, 2.5, 2.5]) / 10.0  # What linear would give
        weights_quad_mid = schedule_fn(0.5)
        # Quadratic should be different from linear at midpoint
        assert not np.allclose(weights_linear_mid, weights_quad_mid, atol=1e-3)

    def test_scale_by_scale_schedule(self, base_config):
        """Test scale by scale schedule (one active output at a time)."""
        config = {TYPE_STR: ScheduleType.SCALE_BY_SCALE_LOW_TO_HIGH, **base_config}
        schedule_fn = schedule_builder(config, 4)

        # Test at different progress points
        test_cases = [
            (0.0, 0),  # Start: first scale active
            (0.24, 0),  # Still first scale
            (0.25, 1),  # Second scale active
            (0.49, 1),  # Still second scale
            (0.5, 2),  # Third scale active
            (0.75, 3),  # Fourth scale active
            (1.0, 3),  # End: last scale active
        ]

        for progress, expected_active in test_cases:
            weights = schedule_fn(progress)
            assert weights.shape == (4,)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)

            # Only one weight should be 1.0, others should be 0.0
            assert np.sum(weights == 1.0) == 1
            assert np.sum(weights == 0.0) == 3
            assert weights[expected_active] == 1.0


class TestCustomizableSchedules:
    """Tests for schedules with custom parameters."""

    def test_custom_sigmoid_default_params(self):
        """Test custom sigmoid schedule with default parameters."""
        config = {
            TYPE_STR: ScheduleType.CUSTOM_SIGMOID_LOW_TO_HIGH,
            CONFIG_STR: {}  # Use default parameters
        }
        schedule_fn = schedule_builder(config, 5)

        # Test at different progress points
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weights = schedule_fn(progress)
            assert weights.shape == (5,)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
            assert np.all(weights >= 0)

    def test_custom_sigmoid_with_params(self):
        """Test custom sigmoid schedule with custom parameters."""
        config = {
            TYPE_STR: ScheduleType.CUSTOM_SIGMOID_LOW_TO_HIGH,
            CONFIG_STR: {
                'k': 15.0,
                'x0': 0.6,
                'transition_point': 0.3
            }
        }
        schedule_fn = schedule_builder(config, 5)

        # Test function runs with custom parameters
        weights = schedule_fn(0.5)
        assert weights.shape == (5,)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
        assert np.all(weights >= 0)

        # Test before transition point (should be more like initial state)
        weights_early = schedule_fn(0.1)  # Before transition_point=0.3
        weights_late = schedule_fn(0.8)  # Well after transition_point

        # "low_to_high" means transition from low indices (shallow) to high indices (deep)
        # Early should favor shallow layers (low indices) more than late
        # Late should favor deeper layers (high indices) more than early
        assert weights_early[0] > weights_late[0]  # First (shallowest) layer decreases
        assert weights_early[-1] < weights_late[-1]  # Last (deepest) layer increases

    def test_cosine_annealing_schedule(self):
        """Test cosine annealing schedule."""
        config = {
            TYPE_STR: ScheduleType.COSINE_ANNEALING,
            CONFIG_STR: {
                'frequency': 2.0,
                'final_ratio': 0.5
            }
        }
        schedule_fn = schedule_builder(config, 4)

        # Test at different progress points
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weights = schedule_fn(progress)
            assert weights.shape == (4,)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
            assert np.all(weights >= 0)

        # Test that weights oscillate (cosine behavior)
        weights_0 = schedule_fn(0.0)
        weights_quarter = schedule_fn(0.25)
        weights_half = schedule_fn(0.5)

        # Due to cosine oscillation, weights should vary
        assert not np.allclose(weights_0, weights_quarter, atol=1e-3)
        assert not np.allclose(weights_quarter, weights_half, atol=1e-3)

    def test_curriculum_schedule_default_params(self):
        """Test curriculum schedule with default parameters."""
        config = {
            TYPE_STR: ScheduleType.CURRICULUM,
            CONFIG_STR: {}
        }
        schedule_fn = schedule_builder(config, 5)

        # Test at different progress points
        weights_start = schedule_fn(0.0)
        weights_mid = schedule_fn(0.5)
        weights_end = schedule_fn(1.0)

        # Check basic properties
        for weights in [weights_start, weights_mid, weights_end]:
            assert weights.shape == (5,)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
            assert np.all(weights >= 0)

        # At start, fewer outputs should be active
        active_start = np.sum(weights_start > 0)
        active_end = np.sum(weights_end > 0)
        assert active_start <= active_end

    def test_curriculum_with_params(self):
        """Test curriculum schedule with custom parameters."""
        config = {
            TYPE_STR: ScheduleType.CURRICULUM,
            CONFIG_STR: {
                'max_active_outputs': 3,
                'activation_strategy': 'exp'
            }
        }
        schedule_fn = schedule_builder(config, 5)

        # Verify function runs with custom parameters
        weights = schedule_fn(0.5)
        assert weights.shape == (5,)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
        assert np.all(weights >= 0)

        # At most 3 outputs should be active (non-zero)
        active_count = np.sum(weights > 0)
        assert active_count <= 3

        # Test exponential vs linear strategy difference
        config_linear = {
            TYPE_STR: ScheduleType.CURRICULUM,
            CONFIG_STR: {
                'max_active_outputs': 3,
                'activation_strategy': 'linear'
            }
        }
        schedule_fn_linear = schedule_builder(config_linear, 5)
        weights_linear = schedule_fn_linear(0.5)

        # Test early training behavior - exponential should activate fewer outputs
        # than linear at the same early progress point
        progress_early = 0.3
        weights_exp_early = schedule_fn(progress_early)
        weights_linear_early = schedule_fn_linear(progress_early)

        active_exp = np.sum(weights_exp_early > 0)
        active_linear = np.sum(weights_linear_early > 0)
        # Exponential should activate fewer or equal outputs early in training
        # (due to quadratic vs linear activation progression)
        assert active_exp <= active_linear


class TestInvertOrder:
    """Tests for the invert_order parameter."""

    def test_invert_order_false(self):
        """Test that invert_order=False gives default behavior."""
        config = {TYPE_STR: ScheduleType.CONSTANT_LOW_TO_HIGH, CONFIG_STR: {}}
        schedule_fn = schedule_builder(config, 4, invert_order=False)

        weights = schedule_fn(0.5)
        # Should be increasing: [1, 2, 3, 4] normalized
        expected = np.array([1, 2, 3, 4], dtype=np.float64)
        expected = expected / np.sum(expected)
        assert np.allclose(weights, expected, atol=1e-10)

    def test_invert_order_true(self):
        """Test that invert_order=True inverts the weight array."""
        config = {TYPE_STR: ScheduleType.CONSTANT_LOW_TO_HIGH, CONFIG_STR: {}}

        schedule_fn_normal = schedule_builder(config, 4, invert_order=False)
        schedule_fn_inverted = schedule_builder(config, 4, invert_order=True)

        weights_normal = schedule_fn_normal(0.5)
        weights_inverted = schedule_fn_inverted(0.5)

        # Inverted should be the reverse of normal
        assert np.allclose(weights_normal, weights_inverted[::-1], atol=1e-10)

        # Both should sum to 1.0
        assert np.isclose(np.sum(weights_normal), 1.0, atol=1e-10)
        assert np.isclose(np.sum(weights_inverted), 1.0, atol=1e-10)

    def test_invert_order_with_different_schedules(self):
        """Test invert_order with different schedule types."""
        schedule_types = [
            ScheduleType.CONSTANT_EQUAL,
            ScheduleType.LINEAR_LOW_TO_HIGH,
            ScheduleType.SCALE_BY_SCALE_LOW_TO_HIGH
        ]

        for schedule_type in schedule_types:
            config = {TYPE_STR: schedule_type, CONFIG_STR: {}}

            schedule_normal = schedule_builder(config, 3, invert_order=False)
            schedule_inverted = schedule_builder(config, 3, invert_order=True)

            progress = 0.6  # Test at 60% progress
            weights_normal = schedule_normal(progress)
            weights_inverted = schedule_inverted(progress)

            # For CONSTANT_EQUAL, inversion should make no difference
            if schedule_type == ScheduleType.CONSTANT_EQUAL:
                assert np.allclose(weights_normal, weights_inverted, atol=1e-10)
            else:
                # For others, should be inverted
                assert np.allclose(weights_normal, weights_inverted[::-1], atol=1e-10)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_output(self):
        """Test with single output."""
        config = {TYPE_STR: ScheduleType.LINEAR_LOW_TO_HIGH, CONFIG_STR: {}}
        schedule_fn = schedule_builder(config, 1)

        weights = schedule_fn(0.5)
        assert weights.shape == (1,)
        assert np.isclose(weights[0], 1.0, atol=1e-10)

    def test_large_number_of_outputs(self):
        """Test with large number of outputs."""
        config = {TYPE_STR: ScheduleType.CONSTANT_EQUAL, CONFIG_STR: {}}
        schedule_fn = schedule_builder(config, 100)

        weights = schedule_fn(0.5)
        assert weights.shape == (100,)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
        assert np.allclose(weights, 0.01, atol=1e-10)  # Each should be 1/100

    def test_progress_boundary_values(self):
        """Test with boundary progress values."""
        config = {TYPE_STR: ScheduleType.LINEAR_LOW_TO_HIGH, CONFIG_STR: {}}
        schedule_fn = schedule_builder(config, 3)

        # Test exact boundary values
        for progress in [0.0, 1.0]:
            weights = schedule_fn(progress)
            assert weights.shape == (3,)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
            assert np.all(weights >= 0)

        # Test slightly outside boundaries (should still work)
        for progress in [-0.1, 1.1]:
            weights = schedule_fn(progress)
            assert weights.shape == (3,)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)


class TestAllScheduleTypes:
    """Test all schedule types for basic functionality."""

    def test_all_schedule_types_basic_functionality(self):
        """Test that all schedule types run successfully."""
        no_outputs = 4
        test_progress_points = [0.0, 0.25, 0.5, 0.75, 1.0]

        for schedule_type in ScheduleType:
            config = {TYPE_STR: schedule_type, CONFIG_STR: {}}
            schedule_fn = schedule_builder(config, no_outputs)

            for progress in test_progress_points:
                weights = schedule_fn(progress)

                # Check basic properties for all schedules
                assert weights.shape == (no_outputs,)
                assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
                assert np.all(weights >= 0)
                assert not np.any(np.isnan(weights))
                assert not np.any(np.isinf(weights))

    def test_schedule_types_with_inversion(self):
        """Test all schedule types work with invert_order=True."""
        no_outputs = 3

        for schedule_type in ScheduleType:
            config = {TYPE_STR: schedule_type, CONFIG_STR: {}}

            try:
                schedule_fn = schedule_builder(config, no_outputs, invert_order=True)
                weights = schedule_fn(0.5)

                assert weights.shape == (no_outputs,)
                assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
                assert np.all(weights >= 0)

            except Exception as e:
                pytest.fail(f"Schedule type {schedule_type} failed with invert_order=True: {e}")


class TestIntegration:
    """Integration tests simulating real usage scenarios.

    Note: The naming convention can be confusing:
    - "linear_low_to_high" means transition from low indices to high indices
    - Low indices = shallow layers = high resolution outputs (index 0)
    - High indices = deep layers = low resolution outputs (index n-1)
    """

    def test_training_simulation(self):
        """Simulate a training scenario with progress updates."""
        config = {TYPE_STR: ScheduleType.LINEAR_LOW_TO_HIGH, CONFIG_STR: {}}
        schedule_fn = schedule_builder(config, 5)

        # Simulate 100 training steps
        num_steps = 100
        weights_history = []

        for step in range(num_steps + 1):
            progress = step / num_steps
            weights = schedule_fn(progress)
            weights_history.append(weights.copy())

            # Verify properties at each step
            assert weights.shape == (5,)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
            assert np.all(weights >= 0)

        # Verify smooth transition
        weights_history = np.array(weights_history)

        # "linear_low_to_high" transitions from low indices (shallow) to high indices (deep)
        # First output (shallowest, index 0) should decrease over time
        first_output_weights = weights_history[:, 0]
        assert first_output_weights[0] > first_output_weights[-1]

        # Last output (deepest, index -1) should increase over time
        last_output_weights = weights_history[:, -1]
        assert last_output_weights[0] < last_output_weights[-1]

    def test_multi_scale_unet_scenario(self):
        """Test scenario mimicking multi-scale U-Net training."""
        # U-Net with 4 scales: final output + 3 deep supervision outputs
        num_scales = 4

        configs_to_test = [
            {
                TYPE_STR: ScheduleType.LINEAR_LOW_TO_HIGH,
                CONFIG_STR: {},
                "description": "Linear transition from deep to shallow"
            },
            {
                TYPE_STR: ScheduleType.CUSTOM_SIGMOID_LOW_TO_HIGH,
                CONFIG_STR: {"k": 8.0, "x0": 0.6, "transition_point": 0.2},
                "description": "Sigmoid transition starting at 20% training"
            },
            {
                TYPE_STR: ScheduleType.CURRICULUM,
                CONFIG_STR: {"max_active_outputs": 2, "activation_strategy": "linear"},
                "description": "Curriculum learning with max 2 active outputs"
            }
        ]

        training_phases = [0.0, 0.1, 0.3, 0.6, 0.8, 1.0]

        for config_info in configs_to_test:
            config = {k: v for k, v in config_info.items() if k != "description"}
            schedule_fn = schedule_builder(config, num_scales)

            phase_weights = []
            for phase in training_phases:
                weights = schedule_fn(phase)
                phase_weights.append(weights)

                # Basic validation
                assert weights.shape == (num_scales,)
                assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
                assert np.all(weights >= 0)

            # Store for potential debugging
            print(f"\nTested: {config_info['description']}")
            print(f"Early weights: {phase_weights[0]}")
            print(f"Late weights: {phase_weights[-1]}")