import keras
import pytest
import numpy as np
from typing import Dict, Any

from dl_techniques.optimization.optimizer import (
    WarmupSchedule,
    schedule_builder,
    optimizer_builder,
    ScheduleType,
    OptimizerType,
    CONFIG_STR,
    TYPE_STR
)


class TestScheduleBuilder:
    """Tests for the schedule_builder function."""

    @pytest.fixture
    def basic_schedule_config(self) -> Dict[str, Any]:
        """Create a basic schedule configuration for testing."""
        return {
            TYPE_STR: ScheduleType.EXPONENTIAL_DECAY,
            "warmup_steps": 100,
            "warmup_start_lr": 1e-6,
            CONFIG_STR: {
                "learning_rate": 0.001,
                "decay_steps": 1000,
                "decay_rate": 0.96
            }
        }

    def test_schedule_builder_validation(self):
        """Test input validation in schedule_builder."""
        # Test invalid config type
        with pytest.raises(ValueError, match="config must be a dictionary"):
            schedule_builder("not_a_dict")

        # Test missing schedule type
        with pytest.raises(ValueError, match="schedule_type cannot be None"):
            schedule_builder({})

        # Test invalid schedule type
        with pytest.raises(ValueError, match="Unknown learning_rate schedule_type"):
            schedule_builder({TYPE_STR: "invalid_schedule_type"})

        # Test invalid warmup_steps
        config = {TYPE_STR: ScheduleType.EXPONENTIAL_DECAY, "warmup_steps": None}
        with pytest.raises(ValueError, match="warmup_steps must be specified"):
            schedule_builder(config)

    def test_exponential_decay_schedule(self, basic_schedule_config):
        """Test exponential decay schedule creation."""
        basic_schedule_config[TYPE_STR] = ScheduleType.EXPONENTIAL_DECAY
        schedule = schedule_builder(basic_schedule_config)

        # Test the schedule is the correct type
        assert isinstance(schedule, WarmupSchedule)

        # Just verify the schedule values are not None and decrease over time
        lr_at_zero = schedule(0).numpy()
        lr_at_warmup = schedule(1000).numpy()
        lr_after_decay = schedule(20000).numpy()

        # Values should decrease as training progresses
        assert lr_at_warmup > lr_at_zero
        assert lr_at_warmup > lr_after_decay
        assert lr_at_zero is not None

    def test_cosine_decay_schedule(self, basic_schedule_config):
        """Test cosine decay schedule creation."""
        basic_schedule_config[TYPE_STR] = ScheduleType.COSINE_DECAY
        basic_schedule_config[CONFIG_STR]["alpha"] = 0.1
        schedule = schedule_builder(basic_schedule_config)

        # Test the schedule is the correct type
        assert isinstance(schedule, WarmupSchedule)

        # Testing after warmup (should follow cosine decay)
        lr_start = schedule(100).numpy()
        lr_middle = schedule(600).numpy()
        lr_end = schedule(1100).numpy()

        # Verify cosine decay behavior - values should decrease
        assert lr_middle < lr_start
        assert lr_end < lr_middle

    def test_cosine_decay_restarts_schedule(self, basic_schedule_config):
        """Test cosine decay with restarts schedule creation."""
        basic_schedule_config[TYPE_STR] = ScheduleType.COSINE_DECAY_RESTARTS
        basic_schedule_config[CONFIG_STR].update({
            "t_mul": 2.0,
            "m_mul": 0.9,
            "alpha": 0.2
        })
        schedule = schedule_builder(basic_schedule_config)

        # Test the schedule is the correct type
        assert isinstance(schedule, WarmupSchedule)

        # Testing the restart behavior
        lr_cycle1_end = schedule(1100).numpy()
        lr_cycle2_start = schedule(1101).numpy()

        # Verify restart behavior - LR should jump back up after cycle
        assert lr_cycle2_start > lr_cycle1_end


class TestOptimizerBuilder:
    """Tests for the optimizer_builder function."""

    @pytest.fixture
    def basic_optimizer_config(self) -> Dict[str, Any]:
        """Create a basic optimizer configuration for testing."""
        return {
            "type": OptimizerType.ADAM,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "schedule": {
                TYPE_STR: ScheduleType.COSINE_DECAY,
                "warmup_steps": 100,
                CONFIG_STR: {
                    "learning_rate": 0.001,
                    "decay_steps": 1000,
                    "alpha": 0.0001
                }
            },
            "gradient_clipping_by_norm": 1.0
        }

    def test_optimizer_builder_validation(self):
        """Test input validation in optimizer_builder."""
        # Test invalid config type
        with pytest.raises(ValueError, match="config must be a dictionary"):
            optimizer_builder("not_a_dict")

        # Test invalid optimizer type
        invalid_config = {
            "type": "invalid_optimizer",
            "schedule": {
                TYPE_STR: ScheduleType.COSINE_DECAY,
                CONFIG_STR: {"learning_rate": 0.001, "decay_steps": 1000}
            }
        }
        with pytest.raises(ValueError, match="Unknown optimizer_type"):
            optimizer_builder(invalid_config)

    def test_adam_optimizer(self, basic_optimizer_config):
        """Test Adam optimizer creation."""
        basic_optimizer_config["type"] = OptimizerType.ADAM
        optimizer, schedule = optimizer_builder(basic_optimizer_config)

        # Verify optimizer type
        assert isinstance(optimizer, keras.optimizers.Adam)

        # Test whether optimizer is created successfully
        # In Keras 3.8.0, attributes like _name are not directly accessible
        config = optimizer.get_config()
        assert config.get('name').startswith('Adam')
        assert config.get('beta_1') == 0.9
        assert config.get('beta_2') == 0.999
        assert config.get('global_clipnorm') == 1.0

    def test_rmsprop_optimizer(self, basic_optimizer_config):
        """Test RMSprop optimizer creation."""
        basic_optimizer_config["type"] = OptimizerType.RMSPROP
        basic_optimizer_config.update({
            "rho": 0.95,
            "momentum": 0.1,
            "centered": True
        })
        optimizer, schedule = optimizer_builder(basic_optimizer_config)

        # Verify optimizer type
        assert isinstance(optimizer, keras.optimizers.RMSprop)

        # Test whether optimizer is created successfully
        config = optimizer.get_config()
        assert config.get('name').startswith('RMSprop')
        assert config.get('rho') == 0.95
        assert config.get('momentum') == 0.1
        assert config.get('centered') == True
        assert config.get('global_clipnorm') == 1.0

    def test_adadelta_optimizer(self, basic_optimizer_config):
        """Test Adadelta optimizer creation."""
        basic_optimizer_config["type"] = OptimizerType.ADADELTA
        basic_optimizer_config.update({
            "rho": 0.95,
            "epsilon": 1e-8,
            "gradient_clipping_by_value": 0.5  # Using clipvalue instead of clipnorm
        })
        # Remove global clipnorm
        basic_optimizer_config.pop("gradient_clipping_by_norm")

        optimizer, schedule = optimizer_builder(basic_optimizer_config)

        # Verify optimizer type
        assert isinstance(optimizer, keras.optimizers.Adadelta)

        # Test whether optimizer is created successfully
        config = optimizer.get_config()
        assert config.get('name').startswith('Adadelta')
        assert config.get('rho') == 0.95
        assert config.get('epsilon') == 1e-8
        assert config.get('clipvalue') == 0.5
        assert config.get('global_clipnorm') is None

    def test_default_parameters(self):
        """Test that default parameters are used correctly."""
        minimal_config = {
            "schedule": {
                TYPE_STR: ScheduleType.EXPONENTIAL_DECAY,
                CONFIG_STR: {
                    "learning_rate": 0.001,
                    "decay_steps": 1000,
                    "decay_rate": 0.96
                }
            }
        }
        optimizer, _ = optimizer_builder(minimal_config)

        # Verify that default optimizer (RMSprop) is used
        assert isinstance(optimizer, keras.optimizers.RMSprop)

        # Verify default parameters using get_config()
        config = optimizer.get_config()
        assert config.get('rho') == 0.9
        assert config.get('momentum') == 0.0
        assert config.get('epsilon') == 1e-7
        assert config.get('centered') == False
        assert config.get('clipvalue') is None
        assert config.get('clipnorm') is None
        assert config.get('global_clipnorm') is None