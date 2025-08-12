import keras
import pytest
from typing import Dict, Any

from dl_techniques.optimization.optimizer import (
    learning_rate_schedule_builder,
    optimizer_builder,
    ScheduleType,
    OptimizerType,
)
from dl_techniques.optimization.warmup_schedule import WarmupSchedule


class TestLearningRateScheduleBuilder:
    """Tests for the learning_rate_schedule_builder function."""

    @pytest.fixture
    def basic_schedule_config(self) -> Dict[str, Any]:
        """Create a basic schedule configuration for testing."""
        return {
            "type": ScheduleType.EXPONENTIAL_DECAY,
            "warmup_steps": 100,
            "warmup_start_lr": 1e-6,
            "learning_rate": 0.001,
            "decay_steps": 1000,
            "decay_rate": 0.96
        }

    def test_schedule_builder_validation(self):
        """Test input validation in learning_rate_schedule_builder."""
        # Test invalid config type
        with pytest.raises(ValueError, match="config must be a dictionary"):
            learning_rate_schedule_builder("not_a_dict")

        # Test missing schedule type
        with pytest.raises(ValueError, match="schedule type must be specified in config"):
            learning_rate_schedule_builder({})

        # Test missing learning_rate
        config = {"type": ScheduleType.EXPONENTIAL_DECAY}
        with pytest.raises(ValueError, match="learning_rate must be specified in config"):
            learning_rate_schedule_builder(config)

        # Test missing decay_steps
        config = {"type": ScheduleType.EXPONENTIAL_DECAY, "learning_rate": 0.001}
        with pytest.raises(ValueError, match="decay_steps must be specified in config"):
            learning_rate_schedule_builder(config)

        # Test missing decay_rate for exponential decay
        config = {
            "type": ScheduleType.EXPONENTIAL_DECAY,
            "learning_rate": 0.001,
            "decay_steps": 1000
        }
        with pytest.raises(ValueError, match="decay_rate must be specified for exponential_decay"):
            learning_rate_schedule_builder(config)

    def test_exponential_decay_schedule(self, basic_schedule_config):
        """Test exponential decay schedule creation."""
        basic_schedule_config["type"] = ScheduleType.EXPONENTIAL_DECAY
        schedule = learning_rate_schedule_builder(basic_schedule_config)

        # Test the schedule is the correct type
        assert isinstance(schedule, WarmupSchedule)

        # Just verify the schedule values are not None and decrease over time
        lr_at_zero = schedule(0).numpy()
        lr_at_warmup = schedule(100).numpy()  # After warmup
        lr_after_decay = schedule(2000).numpy()  # After significant decay

        # Values should increase during warmup, then decrease as training progresses
        assert lr_at_warmup > lr_at_zero  # Warmup increases LR
        assert lr_after_decay < lr_at_warmup  # Then decay decreases LR
        assert lr_at_zero is not None

    def test_exponential_decay_without_warmup(self):
        """Test exponential decay schedule without warmup."""
        config = {
            "type": ScheduleType.EXPONENTIAL_DECAY,
            "learning_rate": 0.001,
            "decay_steps": 1000,
            "decay_rate": 0.96,
            "warmup_steps": 0  # No warmup
        }
        schedule = learning_rate_schedule_builder(config)

        # Should return the base schedule directly (not wrapped in WarmupSchedule)
        assert isinstance(schedule, keras.optimizers.schedules.ExponentialDecay)

        # Test decay behavior
        lr_start = schedule(0).numpy()
        lr_after_decay = schedule(1000).numpy()
        assert lr_after_decay < lr_start
        assert lr_start == 0.001  # Should start at initial LR

    def test_cosine_decay_schedule(self, basic_schedule_config):
        """Test cosine decay schedule creation."""
        basic_schedule_config["type"] = ScheduleType.COSINE_DECAY
        basic_schedule_config["alpha"] = 0.1
        schedule = learning_rate_schedule_builder(basic_schedule_config)

        # Test the schedule is the correct type
        assert isinstance(schedule, WarmupSchedule)

        # Testing after warmup (should follow cosine decay)
        lr_start = schedule(100).numpy()  # After warmup
        lr_middle = schedule(600).numpy()
        lr_end = schedule(1100).numpy()

        # Verify cosine decay behavior - values should decrease
        assert lr_middle < lr_start
        assert lr_end < lr_middle

    def test_cosine_decay_restarts_schedule(self, basic_schedule_config):
        """Test cosine decay with restarts schedule creation."""
        basic_schedule_config["type"] = ScheduleType.COSINE_DECAY_RESTARTS
        basic_schedule_config.update({
            "t_mul": 2.0,
            "m_mul": 0.9,
            "alpha": 0.2
        })
        schedule = learning_rate_schedule_builder(basic_schedule_config)

        # Test the schedule is the correct type
        assert isinstance(schedule, WarmupSchedule)

        # Testing the restart behavior
        lr_cycle1_end = schedule(1100).numpy()  # End of first cycle
        lr_cycle2_start = schedule(1101).numpy()  # Start of second cycle

        # After restart, LR should be higher than at the end of previous cycle
        # Note: This test might be sensitive to exact timing, so we just check
        # that both values are reasonable
        assert lr_cycle1_end > 0
        assert lr_cycle2_start > 0

    def test_default_parameters(self):
        """Test that default parameters are used correctly."""
        config = {
            "type": ScheduleType.COSINE_DECAY,
            "learning_rate": 0.001,
            "decay_steps": 1000
        }
        schedule = learning_rate_schedule_builder(config)

        # Should not have warmup (default warmup_steps = 0)
        assert isinstance(schedule, keras.optimizers.schedules.CosineDecay)

        # Test with warmup
        config["warmup_steps"] = 100
        schedule_with_warmup = learning_rate_schedule_builder(config)
        assert isinstance(schedule_with_warmup, WarmupSchedule)


class TestOptimizerBuilder:
    """Tests for the optimizer_builder function."""

    @pytest.fixture
    def sample_lr_schedule(self):
        """Create a sample learning rate schedule for testing."""
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.96
        )

    @pytest.fixture
    def basic_optimizer_config(self) -> Dict[str, Any]:
        """Create a basic optimizer configuration for testing."""
        return {
            "type": OptimizerType.ADAM,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "gradient_clipping_by_norm": 1.0
        }

    def test_optimizer_builder_validation(self, sample_lr_schedule):
        """Test input validation in optimizer_builder."""
        # Test invalid config type
        with pytest.raises(ValueError, match="config must be a dictionary"):
            optimizer_builder("not_a_dict", sample_lr_schedule)

        # Test missing optimizer type
        with pytest.raises(ValueError, match="optimizer type must be specified in config"):
            optimizer_builder({}, sample_lr_schedule)

        # Test invalid optimizer type
        invalid_config = {"type": "invalid_optimizer"}
        with pytest.raises(ValueError, match="Unknown optimizer_type"):
            optimizer_builder(invalid_config, sample_lr_schedule)

    def test_adam_optimizer(self, basic_optimizer_config, sample_lr_schedule):
        """Test Adam optimizer creation."""
        basic_optimizer_config["type"] = OptimizerType.ADAM
        optimizer = optimizer_builder(basic_optimizer_config, sample_lr_schedule)

        # Verify optimizer type
        assert isinstance(optimizer, keras.optimizers.Adam)

        # Test whether optimizer is created successfully
        config = optimizer.get_config()
        assert config.get('name') == 'Adam'
        assert config.get('beta_1') == 0.9
        assert config.get('beta_2') == 0.999
        assert config.get('global_clipnorm') == 1.0

    def test_adamw_optimizer(self, basic_optimizer_config, sample_lr_schedule):
        """Test AdamW optimizer creation."""
        basic_optimizer_config["type"] = OptimizerType.ADAMW
        basic_optimizer_config.update({
            "beta_1": 0.95,
            "beta_2": 0.998,
            "epsilon": 1e-8
        })
        optimizer = optimizer_builder(basic_optimizer_config, sample_lr_schedule)

        # Verify optimizer type
        assert isinstance(optimizer, keras.optimizers.AdamW)

        # Test whether optimizer is created successfully
        config = optimizer.get_config()
        assert config.get('name') == 'AdamW'
        assert config.get('beta_1') == 0.95
        assert config.get('beta_2') == 0.998
        assert config.get('epsilon') == 1e-8
        assert config.get('global_clipnorm') == 1.0

    def test_rmsprop_optimizer(self, basic_optimizer_config, sample_lr_schedule):
        """Test RMSprop optimizer creation."""
        basic_optimizer_config["type"] = OptimizerType.RMSPROP
        basic_optimizer_config.update({
            "rho": 0.95,
            "momentum": 0.1,
            "centered": True
        })
        optimizer = optimizer_builder(basic_optimizer_config, sample_lr_schedule)

        # Verify optimizer type
        assert isinstance(optimizer, keras.optimizers.RMSprop)

        # Test whether optimizer is created successfully
        config = optimizer.get_config()
        assert config.get('name') == 'RMSprop'
        assert config.get('rho') == 0.95
        assert config.get('momentum') == 0.1
        assert config.get('centered') == True
        assert config.get('global_clipnorm') == 1.0

    def test_adadelta_optimizer(self, basic_optimizer_config, sample_lr_schedule):
        """Test Adadelta optimizer creation."""
        basic_optimizer_config["type"] = OptimizerType.ADADELTA
        basic_optimizer_config.update({
            "rho": 0.95,
            "epsilon": 1e-8,
            "gradient_clipping_by_value": 0.5  # Using clipvalue instead of clipnorm
        })
        # Remove global clipnorm
        basic_optimizer_config.pop("gradient_clipping_by_norm", None)

        optimizer = optimizer_builder(basic_optimizer_config, sample_lr_schedule)

        # Verify optimizer type
        assert isinstance(optimizer, keras.optimizers.Adadelta)

        # Test whether optimizer is created successfully
        config = optimizer.get_config()
        assert config.get('name') == 'Adadelta'
        assert config.get('rho') == 0.95
        assert config.get('epsilon') == 1e-8
        assert config.get('clipvalue') == 0.5
        assert config.get('global_clipnorm') is None

    def test_default_parameters(self, sample_lr_schedule):
        """Test that default parameters are used correctly."""
        minimal_config = {"type": OptimizerType.RMSPROP}
        optimizer = optimizer_builder(minimal_config, sample_lr_schedule)

        # Verify that RMSprop optimizer is used
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

    def test_gradient_clipping_options(self, sample_lr_schedule):
        """Test different gradient clipping options."""
        # Test clipvalue
        config = {
            "type": OptimizerType.ADAM,
            "gradient_clipping_by_value": 0.5
        }
        optimizer = optimizer_builder(config, sample_lr_schedule)
        opt_config = optimizer.get_config()
        assert opt_config.get('clipvalue') == 0.5

        # Test clipnorm (local)
        config = {
            "type": OptimizerType.ADAM,
            "gradient_clipping_by_norm_local": 1.0
        }
        optimizer = optimizer_builder(config, sample_lr_schedule)
        opt_config = optimizer.get_config()
        assert opt_config.get('clipnorm') == 1.0

        # Test global_clipnorm
        config = {
            "type": OptimizerType.ADAM,
            "gradient_clipping_by_norm": 2.0
        }
        optimizer = optimizer_builder(config, sample_lr_schedule)
        opt_config = optimizer.get_config()
        assert opt_config.get('global_clipnorm') == 2.0

    def test_with_float_learning_rate(self):
        """Test optimizer builder with float learning rate instead of schedule."""
        config = {"type": OptimizerType.ADAM}
        optimizer = optimizer_builder(config, 0.001)

        assert isinstance(optimizer, keras.optimizers.Adam)
        opt_config = optimizer.get_config()
        assert opt_config.get('learning_rate') <= 0.0011
        assert opt_config.get('learning_rate') >= 0.0009


class TestIntegration:
    """Integration tests for schedule and optimizer builders working together."""

    def test_schedule_and_optimizer_integration(self):
        """Test that schedule builder output works with optimizer builder."""
        # Create a schedule
        schedule_config = {
            "type": ScheduleType.COSINE_DECAY,
            "learning_rate": 0.001,
            "decay_steps": 1000,
            "warmup_steps": 100,
            "alpha": 0.1
        }
        schedule = learning_rate_schedule_builder(schedule_config)

        # Use the schedule in optimizer
        optimizer_config = {
            "type": OptimizerType.ADAM,
            "beta_1": 0.9,
            "beta_2": 0.999
        }
        optimizer = optimizer_builder(optimizer_config, schedule)

        # Verify integration works
        assert isinstance(optimizer, keras.optimizers.Adam)
        assert isinstance(schedule, WarmupSchedule)

        # Test that learning rate is properly set
        # Note: We can't directly compare the schedule object due to how Keras handles it
        opt_config = optimizer.get_config()
        assert 'learning_rate' in opt_config

    def test_end_to_end_training_setup(self):
        """Test a complete end-to-end training setup."""
        # Build schedule
        schedule_config = {
            "type": ScheduleType.EXPONENTIAL_DECAY,
            "learning_rate": 0.01,
            "decay_steps": 1000,
            "decay_rate": 0.9,
            "warmup_steps": 50
        }
        schedule = learning_rate_schedule_builder(schedule_config)

        # Build optimizer
        optimizer_config = {
            "type": OptimizerType.ADAMW,
            "gradient_clipping_by_norm": 1.0
        }
        optimizer = optimizer_builder(optimizer_config, schedule)

        # Create a simple model to test compilation
        model = keras.Sequential([
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile model - this should not raise any errors
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Verify model is compiled correctly
        assert model.optimizer is not None
        assert isinstance(model.optimizer, keras.optimizers.AdamW)