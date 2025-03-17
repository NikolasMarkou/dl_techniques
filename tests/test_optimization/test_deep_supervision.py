import pytest
import numpy as np
from typing import Dict, Any
from dl_techniques.optimization.deep_supervision import (
    schedule_builder,
    ScheduleType,
    TYPE_STR,
    CONFIG_STR)

@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Create a base configuration for testing."""
    return {
        TYPE_STR: ScheduleType.CONSTANT_EQUAL,
        CONFIG_STR: {}
    }


def test_schedule_builder_validation():
    """Test input validation in schedule_builder."""
    # Test invalid config type
    with pytest.raises(ValueError, match="config must be a dictionary"):
        schedule_builder("not_a_dict", 5)

    # Test invalid number of outputs
    with pytest.raises(ValueError, match="no_outputs must be a positive integer"):
        schedule_builder({TYPE_STR: ScheduleType.CONSTANT_EQUAL}, 0)

    # Test missing schedule type
    with pytest.raises(ValueError, match="schedule_type cannot be None"):
        schedule_builder({}, 5)

    # Test invalid schedule type
    with pytest.raises(ValueError, match="Unknown deep supervision schedule_type"):
        schedule_builder({TYPE_STR: "invalid_schedule_type"}, 5)


def test_constant_equal_schedule():
    """Test constant equal schedule output."""
    config = {TYPE_STR: ScheduleType.CONSTANT_EQUAL}
    schedule_fn = schedule_builder(config, 5)

    # Test at different training percentages
    for percentage in [0.0, 0.5, 1.0]:
        weights = schedule_fn(percentage)

        # Check shape and sum
        assert weights.shape == (5,)
        assert np.isclose(np.sum(weights), 1.0)

        # Check equal weights
        assert np.allclose(weights, 0.2)


def test_constant_low_to_high_schedule():
    """Test constant low to high schedule output."""
    config = {TYPE_STR: ScheduleType.CONSTANT_LOW_TO_HIGH}
    schedule_fn = schedule_builder(config, 5)

    # Test at different training percentages
    for percentage in [0.0, 0.5, 1.0]:
        weights = schedule_fn(percentage)

        # Check shape and sum
        assert weights.shape == (5,)
        assert np.isclose(np.sum(weights), 1.0)

        # Check increasing weights
        assert np.all(np.diff(weights) > 0)


def test_custom_sigmoid_with_params():
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

    # Just verify function runs with custom parameters
    weights = schedule_fn(0.5)
    assert weights.shape == (5,)
    assert np.isclose(np.sum(weights), 1.0)


def test_curriculum_with_params():
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
    assert np.isclose(np.sum(weights), 1.0)


def test_all_schedule_types():
    """Test all schedule types run successfully."""
    no_outputs = 4

    for schedule_type in ScheduleType:
        config = {TYPE_STR: schedule_type}
        schedule_fn = schedule_builder(config, no_outputs)

        for percentage in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weights = schedule_fn(percentage)

            # Check shape, sum and non-negative values
            assert weights.shape == (no_outputs,)
            assert np.isclose(np.sum(weights), 1.0)
            assert np.all(weights >= 0)