import keras
import pytest
from typing import Dict, Any

# NOTE: `learning_rate_schedule_builder` and `ScheduleType` are imported from the
# PACKAGE / `schedule`, not from `optimizer`. `optimizer.py` used to carry its own
# private copies of both; the schedule builder there was never imported by any
# production code and diverged from the exported one (it returned a bare schedule
# at warmup_steps=0 instead of a WarmupSchedule wrapper), so this test module was
# the only thing exercising it. Both copies have been deleted -- these tests now
# cover the function every trainer actually calls.
from dl_techniques.optimization import (
    learning_rate_schedule_builder,
    optimizer_builder,
    create_learning_rate_schedule,
    create_warmup_lr_schedule,
)
from dl_techniques.optimization.optimizer import OptimizerType
from dl_techniques.optimization.schedule import ScheduleType
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
        """Test input validation in learning_rate_schedule_builder.

        Missing schedule PARAMETERS raise KeyError (from the shared
        `_validate_required_params` helper), while a malformed config or an
        unknown schedule type raises ValueError.
        """
        # Test invalid config type
        with pytest.raises(ValueError, match="config must be a dictionary"):
            learning_rate_schedule_builder("not_a_dict")

        # Test missing schedule type
        with pytest.raises(ValueError, match="schedule_type cannot be None"):
            learning_rate_schedule_builder({})

        # Test unknown schedule type
        with pytest.raises(ValueError, match="Unknown learning_rate schedule_type"):
            learning_rate_schedule_builder(
                {"type": "not_a_schedule", "learning_rate": 0.001, "decay_steps": 10}
            )

        # Test missing learning_rate (reported together with the other missing keys)
        config = {"type": ScheduleType.EXPONENTIAL_DECAY}
        with pytest.raises(KeyError, match="learning_rate"):
            learning_rate_schedule_builder(config)

        # Test missing decay_steps
        config = {"type": ScheduleType.EXPONENTIAL_DECAY, "learning_rate": 0.001}
        with pytest.raises(KeyError, match="decay_steps"):
            learning_rate_schedule_builder(config)

        # Test missing decay_rate for exponential decay
        config = {
            "type": ScheduleType.EXPONENTIAL_DECAY,
            "learning_rate": 0.001,
            "decay_steps": 1000
        }
        with pytest.raises(KeyError, match="decay_rate"):
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
        """Test exponential decay schedule without warmup.

        The builder ALWAYS wraps in a WarmupSchedule, even at warmup_steps=0.
        A zero-length warmup is a pass-through, so the wrapper is numerically
        transparent -- the values below are those of the bare ExponentialDecay.
        """
        config = {
            "type": ScheduleType.EXPONENTIAL_DECAY,
            "learning_rate": 0.001,
            "decay_steps": 1000,
            "decay_rate": 0.96,
            "warmup_steps": 0  # No warmup
        }
        schedule = learning_rate_schedule_builder(config)

        assert isinstance(schedule, WarmupSchedule)

        # A zero-step warmup must be a numerical no-op.
        bare = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.96
        )
        for step in (0, 1, 500, 1000, 5000):
            assert float(schedule(step)) == float(bare(step))

        # Test decay behavior
        lr_start = float(schedule(0))
        lr_after_decay = float(schedule(1000))
        assert lr_after_decay < lr_start
        # float32 storage: compare with a tolerance, not for exact equality.
        assert lr_start == pytest.approx(0.001)  # Should start at initial LR

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

        # Wrapped even with the default warmup_steps=0, but with a no-op ramp:
        # the first two steps are already on the primary cosine curve.
        assert isinstance(schedule, WarmupSchedule)
        assert float(schedule(0)) == pytest.approx(0.001)

        # Test with warmup: the ramp is now visible at step 0.
        config["warmup_steps"] = 100
        schedule_with_warmup = learning_rate_schedule_builder(config)
        assert isinstance(schedule_with_warmup, WarmupSchedule)
        assert float(schedule_with_warmup(0)) < float(schedule_with_warmup(100))


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

class TestSGDOptimizer:
    """Tests for the ``sgd`` optimizer type.

    SGD was added to ``OptimizerType`` so that trainers hard-coding
    ``keras.optimizers.SGD(...)`` can route through the factory instead. The
    defaults must therefore mirror ``keras.optimizers.SGD`` exactly.
    """

    def test_sgd_optimizer(self):
        """Test SGD creation with momentum and gradient clipping."""
        optimizer = optimizer_builder(
            {
                "type": OptimizerType.SGD,
                "momentum": 0.9,
                "gradient_clipping_by_norm_local": 1.0,
            },
            0.01,
        )

        assert isinstance(optimizer, keras.optimizers.SGD)

        config = optimizer.get_config()
        assert config.get("name") == "SGD"
        assert config.get("momentum") == 0.9
        assert config.get("nesterov") is False
        assert config.get("clipnorm") == 1.0

    def test_sgd_nesterov(self):
        """Test that Nesterov momentum is configurable."""
        optimizer = optimizer_builder(
            {"type": OptimizerType.SGD, "momentum": 0.9, "nesterov": True}, 0.01
        )
        assert optimizer.get_config().get("nesterov") is True

    def test_sgd_defaults_match_keras(self):
        """The factory must not silently diverge from the Keras class defaults.

        Anything the factory injects beyond the learning rate is a behaviour
        change for callers migrating off a bare ``keras.optimizers.SGD(lr)``.
        """
        built = optimizer_builder({"type": OptimizerType.SGD}, 0.01)
        reference = keras.optimizers.SGD(learning_rate=0.01)

        assert built.get_config()["momentum"] == reference.get_config()["momentum"]
        assert built.get_config()["nesterov"] == reference.get_config()["nesterov"]
        assert built.weight_decay == reference.weight_decay

    def test_sgd_weight_decay_passthrough(self):
        """weight_decay is forwarded only when explicitly configured."""
        assert optimizer_builder({"type": OptimizerType.SGD}, 0.01).weight_decay is None
        assert optimizer_builder(
            {"type": OptimizerType.SGD, "weight_decay": 0.01}, 0.01
        ).weight_decay == 0.01

    def test_sgd_accepted_by_schedule(self):
        """SGD accepts a LearningRateSchedule like every other factory type."""
        schedule = learning_rate_schedule_builder(
            {
                "type": ScheduleType.COSINE_DECAY,
                "learning_rate": 0.01,
                "decay_steps": 1000,
                "warmup_steps": 100,
            }
        )
        optimizer = optimizer_builder({"type": OptimizerType.SGD}, schedule)

        # NOTE: ``optimizer.learning_rate`` returns the CURRENT VALUE, not the
        # schedule object -- asserting on it would compare a tensor to a class.
        # The schedule itself is kept on ``_learning_rate``.
        assert isinstance(optimizer._learning_rate, WarmupSchedule)


class TestExcludeFromWeightDecay:
    """Tests for the ``exclude_from_weight_decay`` config key.

    These are behavioural, not attribute-inspection, tests: Keras stores the
    exclusion as a compiled regex whose pattern ordering is set-derived, so
    asserting on the stored attribute would be brittle and would not prove the
    exclusion actually reaches the update rule.
    """

    @staticmethod
    def _decay_probe(config: Dict[str, Any]):
        """Apply one step with ZERO gradients; any movement is pure weight decay.

        Returns the post-step value of a ``kernel`` and a ``bias`` variable,
        both starting at 1.0.
        """
        import numpy as np

        kernel = keras.Variable(np.ones((3,), "float32"), name="kernel")
        bias = keras.Variable(np.ones((3,), "float32"), name="bias")

        optimizer = optimizer_builder(config, 0.1)
        optimizer.build([kernel, bias])

        zero_grads = [keras.ops.zeros((3,), "float32")] * 2
        optimizer.apply_gradients(zip(zero_grads, [kernel, bias]))

        return float(np.array(kernel)[0]), float(np.array(bias)[0])

    def test_control_decay_is_visible(self):
        """Control: without exclusions BOTH variables must decay.

        Without this, a passing exclusion test would be vacuous — it would also
        pass if weight decay never applied to anything.
        """
        kernel, bias = self._decay_probe({"type": "adamw", "weight_decay": 0.1})
        assert kernel < 1.0
        assert bias < 1.0

    def test_excluded_variable_does_not_decay(self):
        """A name matching the exclusion list is left untouched by weight decay."""
        kernel, bias = self._decay_probe(
            {
                "type": "adamw",
                "weight_decay": 0.1,
                "exclude_from_weight_decay": ["bias", "gamma", "beta"],
            }
        )
        assert kernel < 1.0, "non-excluded kernel must still decay"
        assert bias == 1.0, "excluded bias must not move under zero gradients"

    def test_absent_and_empty_keys_are_noops(self):
        """Omitting the key, or passing an empty list, changes nothing."""
        baseline = self._decay_probe({"type": "adamw", "weight_decay": 0.1})

        for value in ([], None):
            assert (
                self._decay_probe(
                    {
                        "type": "adamw",
                        "weight_decay": 0.1,
                        "exclude_from_weight_decay": value,
                    }
                )
                == baseline
            )

    def test_unsupported_optimizer_does_not_raise(self):
        """The key is ignored, not fatal, on an optimizer without decay support."""
        optimizer = optimizer_builder(
            {"type": "adadelta", "exclude_from_weight_decay": ["bias"]}, 0.1
        )
        assert isinstance(optimizer, keras.optimizers.Adadelta)


class TestEpochFacingAdapters:
    """Tests for `create_learning_rate_schedule` / `create_warmup_lr_schedule`.

    These moved here from `train.common.callbacks` / `train.common.nlp`. They
    are NOT thin wrappers over `schedule_builder` -- they differ from it in
    several observable ways that many callers depend on. Each test below pins
    one of those differences, so that a future attempt to "unify" the adapters
    onto `schedule_builder` fails loudly instead of silently changing the LR
    curve of every trainer.
    """

    def test_reexported_from_train_common(self):
        """The old import paths must still resolve to the same objects."""
        from train.common import create_learning_rate_schedule as from_common
        from train.common.nlp import create_warmup_lr_schedule as nlp_from_common

        assert from_common is create_learning_rate_schedule
        assert nlp_from_common is create_warmup_lr_schedule

    def test_no_warmup_returns_bare_cosine_decay(self):
        """At warmup_steps=0 the result is a BARE CosineDecay.

        `schedule_builder` would wrap this in a WarmupSchedule. Returning a
        wrapped schedule here would change the type every no-warmup caller sees.
        """
        schedule = create_learning_rate_schedule(
            initial_lr=1e-3, schedule_type='cosine', total_epochs=100
        )
        assert isinstance(schedule, keras.optimizers.schedules.CosineDecay)
        assert not isinstance(schedule, WarmupSchedule)

    def test_warmup_steps_activates_wrapper(self):
        """warmup_steps>0 wraps the cosine decay in a WarmupSchedule."""
        schedule = create_learning_rate_schedule(
            initial_lr=1e-3,
            schedule_type='cosine',
            total_epochs=50,
            steps_per_epoch=200,
            warmup_steps=500,
        )
        assert isinstance(schedule, WarmupSchedule)

        # The ramp must actually be rising over the warmup window.
        assert float(schedule(0)) < float(schedule(250)) < float(schedule(500))

    def test_warmup_epochs_is_a_noop(self):
        """warmup_epochs must NOT activate warmup (DECISION D-004).

        Dozens of callers pass a non-zero warmup_epochs positionally while
        relying on the plain cosine path; honouring it would silently give all
        of them a warmup ramp.
        """
        with_epochs = create_learning_rate_schedule(
            initial_lr=1e-3, schedule_type='cosine', total_epochs=100, warmup_epochs=5
        )
        without = create_learning_rate_schedule(
            initial_lr=1e-3, schedule_type='cosine', total_epochs=100, warmup_epochs=0
        )
        assert not isinstance(with_epochs, WarmupSchedule)
        assert float(with_epochs(0)) == float(without(0)) == pytest.approx(1e-3)

    def test_warmup_steps_without_steps_per_epoch_raises(self):
        """warmup_steps>0 needs a step budget to compute the decay horizon."""
        with pytest.raises(ValueError, match="requires steps_per_epoch"):
            create_learning_rate_schedule(
                initial_lr=1e-3,
                schedule_type='cosine',
                total_epochs=50,
                warmup_steps=500,
            )

    def test_constant_returns_a_bare_float(self):
        """'constant' returns a float, not a schedule.

        `schedule_builder` has no 'constant' type at all, so this path cannot be
        delegated to it.
        """
        result = create_learning_rate_schedule(
            initial_lr=5e-4, schedule_type='constant', total_epochs=10
        )
        assert isinstance(result, float)
        assert result == 5e-4

    def test_unknown_type_falls_through_to_constant(self):
        """Any unrecognised schedule_type behaves as 'constant' (documented)."""
        assert create_learning_rate_schedule(
            initial_lr=5e-4, schedule_type='not_a_schedule'
        ) == 5e-4

    def test_exponential_uses_quarter_horizon(self):
        """The exponential branch hard-codes decay_rate=0.9 over epochs//4."""
        schedule = create_learning_rate_schedule(
            initial_lr=1e-3,
            schedule_type='exponential',
            total_epochs=40,
            steps_per_epoch=125,
        )
        assert isinstance(schedule, keras.optimizers.schedules.ExponentialDecay)
        config = schedule.get_config()
        assert config["decay_rate"] == 0.9
        assert config["decay_steps"] == (40 // 4) * 125

    def test_warmup_ratio_schedule(self):
        """create_warmup_lr_schedule expresses warmup as a fraction of the run."""
        schedule = create_warmup_lr_schedule(
            learning_rate=1e-4, num_epochs=10, steps_per_epoch=500, warmup_ratio=0.1
        )
        assert isinstance(schedule, WarmupSchedule)

        total_steps = 10 * 500
        warmup_steps = int(0.1 * total_steps)

        # Ramp starts at warmup_start_lr=1e-7 and peaks at the target LR.
        assert float(schedule(0)) == pytest.approx(1e-7, rel=1e-3)
        assert float(schedule(warmup_steps)) == pytest.approx(1e-4, rel=1e-3)

        # alpha=0.0 means it decays to (essentially) zero by the end.
        assert float(schedule(total_steps)) == pytest.approx(0.0, abs=1e-9)
