import pytest
import keras
import tensorflow as tf
import numpy as np
from typing import Dict, Any

from dl_techniques.optimization.warmup_schedule import WarmupSchedule


class TestWarmupScheduleInitialization:
    """Tests for WarmupSchedule initialization and validation."""

    def test_valid_initialization(self):
        """Test valid initialization parameters."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=100,
            warmup_start_lr=1e-6,
            primary_schedule=primary_schedule
        )

        assert warmup_schedule.warmup_steps == 100
        assert warmup_schedule.warmup_start_lr == 1e-6
        assert warmup_schedule.primary_schedule == primary_schedule

    def test_default_warmup_start_lr(self):
        """Test default warmup_start_lr value."""
        primary_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=1000
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=100,
            primary_schedule=primary_schedule
        )

        assert warmup_schedule.warmup_start_lr == 1e-8

    def test_zero_warmup_steps(self):
        """Test initialization with zero warmup steps."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=0,
            primary_schedule=primary_schedule
        )

        assert warmup_schedule.warmup_steps == 0

    def test_invalid_warmup_steps(self):
        """Test that negative warmup_steps raises ValueError."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

        with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
            WarmupSchedule(
                warmup_steps=-1,
                primary_schedule=primary_schedule
            )

    def test_invalid_warmup_start_lr(self):
        """Test that negative warmup_start_lr raises ValueError."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

        with pytest.raises(ValueError, match="warmup_start_lr must be >= 0.0"):
            WarmupSchedule(
                warmup_steps=100,
                warmup_start_lr=-1e-6,
                primary_schedule=primary_schedule
            )

    def test_none_primary_schedule(self):
        """Test that None primary_schedule raises ValueError."""
        with pytest.raises(ValueError, match="primary_schedule cannot be None"):
            WarmupSchedule(
                warmup_steps=100,
                warmup_start_lr=1e-6,
                primary_schedule=None
            )


class TestWarmupScheduleBehavior:
    """Tests for WarmupSchedule learning rate behavior."""

    @pytest.fixture
    def sample_primary_schedule(self):
        """Create a sample primary schedule for testing."""
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

    @pytest.fixture
    def warmup_schedule(self, sample_primary_schedule):
        """Create a sample warmup schedule for testing."""
        return WarmupSchedule(
            warmup_steps=100,
            warmup_start_lr=1e-6,
            primary_schedule=sample_primary_schedule
        )

    def test_no_warmup_behavior(self, sample_primary_schedule):
        """Test behavior when warmup_steps=0 (no warmup)."""
        warmup_schedule = WarmupSchedule(
            warmup_steps=0,
            primary_schedule=sample_primary_schedule
        )

        # With no warmup, should return primary schedule values directly
        for step in [0, 10, 100, 1000]:
            warmup_lr = warmup_schedule(step).numpy()
            primary_lr = sample_primary_schedule(step).numpy()
            assert np.isclose(warmup_lr, primary_lr, atol=1e-10)

    def test_warmup_phase_linear_increase(self, warmup_schedule, sample_primary_schedule):
        """Test linear increase during warmup phase."""
        warmup_steps = 100
        warmup_start_lr = 1e-6
        target_lr = sample_primary_schedule(0).numpy()  # Initial LR of primary schedule

        # Test at different points during warmup
        test_steps = [0, 25, 50, 75, 99]

        for step in test_steps:
            lr = warmup_schedule(step).numpy()
            expected_progress = step / warmup_steps
            expected_lr = warmup_start_lr + (target_lr - warmup_start_lr) * expected_progress

            assert np.isclose(lr, expected_lr, atol=1e-8)

    def test_warmup_boundary_conditions(self, warmup_schedule, sample_primary_schedule):
        """Test learning rate at warmup boundaries."""
        warmup_start_lr = 1e-6
        target_lr = sample_primary_schedule(0).numpy()

        # At step 0, should equal warmup_start_lr
        lr_at_start = warmup_schedule(0).numpy()
        assert np.isclose(lr_at_start, warmup_start_lr, atol=1e-10)

        # At step warmup_steps-1, should be very close to target_lr but not quite there
        lr_at_end_warmup = warmup_schedule(99).numpy()
        expected_at_99 = warmup_start_lr + (target_lr - warmup_start_lr) * (99 / 100)
        assert np.isclose(lr_at_end_warmup, expected_at_99, atol=1e-8)

    def test_post_warmup_phase(self, warmup_schedule, sample_primary_schedule):
        """Test behavior after warmup phase."""
        warmup_steps = 100

        # Test steps after warmup
        test_steps = [100, 150, 200, 500, 1000]

        for step in test_steps:
            warmup_lr = warmup_schedule(step).numpy()
            # Primary schedule should receive adjusted step (step - warmup_steps)
            adjusted_step = step - warmup_steps
            expected_lr = sample_primary_schedule(adjusted_step).numpy()

            assert np.isclose(warmup_lr, expected_lr, atol=1e-8)

    def test_different_primary_schedules(self):
        """Test warmup with different types of primary schedules."""
        warmup_steps = 50
        warmup_start_lr = 1e-5

        # Test with CosineDecay
        cosine_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.01,
            decay_steps=1000,
            alpha=0.1
        )

        warmup_cosine = WarmupSchedule(
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            primary_schedule=cosine_schedule
        )

        # Test warmup phase
        lr_mid_warmup = warmup_cosine(25).numpy()
        target_lr = cosine_schedule(0).numpy()
        expected_mid = warmup_start_lr + (target_lr - warmup_start_lr) * 0.5
        assert np.isclose(lr_mid_warmup, expected_mid, atol=1e-8)

        # Test post-warmup phase
        lr_post_warmup = warmup_cosine(100).numpy()
        expected_post = cosine_schedule(100 - warmup_steps).numpy()
        assert np.isclose(lr_post_warmup, expected_post, atol=1e-8)

        # Test with CosineDecayRestarts
        cosine_restarts_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.001,
            first_decay_steps=500,
            t_mul=2.0,
            m_mul=0.9
        )

        warmup_restarts = WarmupSchedule(
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            primary_schedule=cosine_restarts_schedule
        )

        # Test that it works without errors
        lr_restarts = warmup_restarts(200).numpy()
        assert lr_restarts > 0  # Should be a positive learning rate

    def test_tensor_input_compatibility(self, warmup_schedule):
        """Test that the schedule works with tensor inputs."""
        # Test with TensorFlow tensor input
        step_tensor = tf.constant(50, dtype=tf.int64)
        lr = warmup_schedule(step_tensor)

        assert isinstance(lr, tf.Tensor)
        assert lr.numpy() > 0

        # Test with different tensor dtypes
        step_float = tf.constant(75.0, dtype=tf.float32)
        lr_float = warmup_schedule(step_float)

        assert isinstance(lr_float, tf.Tensor)
        assert lr_float.numpy() > 0

    def test_large_warmup_steps(self):
        """Test behavior with large warmup_steps."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000,
            decay_rate=0.95
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=5000,  # Large warmup
            warmup_start_lr=1e-7,
            primary_schedule=primary_schedule
        )

        # Test mid-warmup
        lr_mid = warmup_schedule(2500).numpy()
        target_lr = primary_schedule(0).numpy()
        expected_mid = 1e-7 + (target_lr - 1e-7) * 0.5
        assert np.isclose(lr_mid, expected_mid, atol=1e-8)

        # Test post-warmup
        lr_post = warmup_schedule(7500).numpy()
        expected_post = primary_schedule(2500).numpy()  # 7500 - 5000
        assert np.isclose(lr_post, expected_post, atol=1e-8)


class TestWarmupScheduleSerialization:
    """Tests for WarmupSchedule serialization and deserialization."""

    def test_get_config(self):
        """Test get_config method."""
        primary_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            alpha=0.1
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=100,
            warmup_start_lr=1e-6,
            primary_schedule=primary_schedule
        )

        config = warmup_schedule.get_config()

        # Check that all required keys are present
        required_keys = ['warmup_steps', 'warmup_start_lr', 'primary_schedule']
        for key in required_keys:
            assert key in config

        # Check values
        assert config['warmup_steps'] == 100
        assert config['warmup_start_lr'] == 1e-6
        assert isinstance(config['primary_schedule'], dict)

    def test_from_config(self):
        """Test from_config class method."""
        original_primary = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.005,
            decay_steps=2000,
            decay_rate=0.8
        )

        original_schedule = WarmupSchedule(
            warmup_steps=200,
            warmup_start_lr=5e-6,
            primary_schedule=original_primary
        )

        # Get config and recreate
        config = original_schedule.get_config()
        recreated_schedule = WarmupSchedule.from_config(config)

        # Check that parameters match
        assert recreated_schedule.warmup_steps == original_schedule.warmup_steps
        assert recreated_schedule.warmup_start_lr == original_schedule.warmup_start_lr
        assert isinstance(recreated_schedule.primary_schedule,
                          type(original_schedule.primary_schedule))

    def test_from_config_missing_keys(self):
        """Test from_config with missing required keys."""
        incomplete_config = {
            'warmup_steps': 100,
            'warmup_start_lr': 1e-6
            # Missing 'primary_schedule'
        }

        with pytest.raises(KeyError, match="Missing required configuration keys"):
            WarmupSchedule.from_config(incomplete_config)

    def test_serialization_roundtrip(self):
        """Test complete serialization/deserialization roundtrip."""
        primary_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.002,
            decay_steps=1500,
            alpha=0.01
        )

        original_schedule = WarmupSchedule(
            warmup_steps=150,
            warmup_start_lr=2e-6,
            primary_schedule=primary_schedule
        )

        # Serialize and deserialize
        config = original_schedule.get_config()
        restored_schedule = WarmupSchedule.from_config(config)

        # Test that both schedules produce the same learning rates
        test_steps = [0, 75, 150, 300, 600, 1000]

        for step in test_steps:
            original_lr = original_schedule(step).numpy()
            restored_lr = restored_schedule(step).numpy()
            assert np.isclose(original_lr, restored_lr, atol=1e-10)

    def test_keras_serialization_compatibility(self):
        """Test compatibility with Keras serialization utilities."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.003,
            decay_steps=800,
            decay_rate=0.95
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=80,
            warmup_start_lr=3e-6,
            primary_schedule=primary_schedule
        )

        # Test with Keras serialization
        serialized = keras.optimizers.schedules.serialize(warmup_schedule)
        deserialized = keras.optimizers.schedules.deserialize(serialized)

        # Verify they produce the same results
        test_steps = [0, 40, 80, 160, 400]
        for step in test_steps:
            original_lr = warmup_schedule(step).numpy()
            deserialized_lr = deserialized(step).numpy()
            assert np.isclose(original_lr, deserialized_lr, atol=1e-10)


class TestWarmupScheduleEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_warmup_steps(self):
        """Test with very small warmup_steps."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=1,  # Only 1 warmup step
            warmup_start_lr=1e-6,
            primary_schedule=primary_schedule
        )

        # At step 0, should be warmup_start_lr
        lr_0 = warmup_schedule(0).numpy()
        assert np.isclose(lr_0, 1e-6, atol=1e-10)

        # At step 1 and beyond, should use primary schedule with adjusted step
        lr_1 = warmup_schedule(1).numpy()
        expected_1 = primary_schedule(0).numpy()  # 1 - 1 = 0
        assert np.isclose(lr_1, expected_1, atol=1e-8)

    def test_very_large_warmup_start_lr(self):
        """Test with warmup_start_lr larger than target learning rate."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

        # Warmup start LR is larger than target LR
        warmup_schedule = WarmupSchedule(
            warmup_steps=100,
            warmup_start_lr=0.01,  # Larger than primary's initial LR (0.001)
            primary_schedule=primary_schedule
        )

        # Should still work - warmup will decrease from 0.01 to 0.001
        lr_0 = warmup_schedule(0).numpy()
        assert np.isclose(lr_0, 0.01, atol=1e-10)

        lr_50 = warmup_schedule(50).numpy()
        target_lr = primary_schedule(0).numpy()
        expected_50 = 0.01 + (target_lr - 0.01) * 0.5
        assert np.isclose(lr_50, expected_50, atol=1e-8)

    def test_zero_warmup_start_lr(self):
        """Test with zero warmup_start_lr."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=100,
            warmup_start_lr=0.0,  # Start from zero
            primary_schedule=primary_schedule
        )

        # At step 0, should be 0.0
        lr_0 = warmup_schedule(0).numpy()
        assert np.isclose(lr_0, 0.0, atol=1e-10)

        # At step 50, should be halfway to target
        lr_50 = warmup_schedule(50).numpy()
        target_lr = primary_schedule(0).numpy()
        expected_50 = 0.0 + target_lr * 0.5
        assert np.isclose(lr_50, expected_50, atol=1e-8)

    def test_equal_warmup_start_and_target_lr(self):
        """Test when warmup_start_lr equals target learning rate."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

        target_lr = primary_schedule(0).numpy()
        warmup_schedule = WarmupSchedule(
            warmup_steps=100,
            warmup_start_lr=target_lr,  # Same as target
            primary_schedule=primary_schedule
        )

        # During warmup, learning rate should remain constant at target_lr
        for step in [0, 25, 50, 75, 99]:
            lr = warmup_schedule(step).numpy()
            assert np.isclose(lr, target_lr, atol=1e-8)

    def test_float_step_input(self):
        """Test behavior with float step inputs."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=100,
            warmup_start_lr=1e-6,
            primary_schedule=primary_schedule
        )

        # Test with float steps
        float_steps = [0.0, 25.5, 50.0, 99.9, 100.1, 200.7]

        for step in float_steps:
            lr = warmup_schedule(step)
            assert isinstance(lr, tf.Tensor)
            assert lr.numpy() > 0


class TestWarmupScheduleIntegration:
    """Integration tests with optimizers and training scenarios."""

    def test_with_adam_optimizer(self):
        """Test integration with Adam optimizer."""
        primary_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=1000
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=100,
            warmup_start_lr=1e-6,
            primary_schedule=primary_schedule
        )

        # Create optimizer with warmup schedule
        optimizer = keras.optimizers.Adam(learning_rate=warmup_schedule)

        # Verify that the optimizer was created successfully
        assert optimizer is not None

        # Verify that the learning rate at step 0 matches our warmup_start_lr
        # The optimizer starts at step 0, so we should get the warmup start rate
        current_lr = float(optimizer.learning_rate.numpy())
        expected_lr = float(warmup_schedule(0).numpy())

        assert np.isclose(current_lr, expected_lr, atol=1e-8)
        assert np.isclose(current_lr, 1e-6, atol=1e-8)  # Should match warmup_start_lr

    def test_with_different_optimizers(self):
        """Test with different optimizer types."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=500,
            decay_rate=0.8
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=50,
            primary_schedule=primary_schedule
        )

        optimizers_to_test = [
            keras.optimizers.Adam,
            keras.optimizers.SGD,
            keras.optimizers.RMSprop,
            keras.optimizers.AdamW
        ]

        for optimizer_class in optimizers_to_test:
            optimizer = optimizer_class(learning_rate=warmup_schedule)

            # Verify that the optimizer was created successfully
            assert optimizer is not None

            # Verify that the learning rate makes sense
            current_lr = float(optimizer.learning_rate.numpy())
            expected_lr = float(warmup_schedule(0).numpy())  # At step 0

            assert np.isclose(current_lr, expected_lr, atol=1e-8)
            # Should be close to default warmup_start_lr (1e-8)
            assert np.isclose(current_lr, 1e-8, atol=1e-10)

    def test_training_simulation(self):
        """Test a complete training simulation scenario."""
        # Create a more realistic schedule
        primary_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.01,
            decay_steps=1000,
            alpha=0.001
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=100,
            warmup_start_lr=1e-5,
            primary_schedule=primary_schedule
        )

        # Simulate training steps
        learning_rates = []
        steps = list(range(0, 1200, 10))  # Every 10 steps up to 1200

        for step in steps:
            lr = warmup_schedule(step).numpy()
            learning_rates.append(lr)

        # Verify expected behavior
        # Early steps should show warmup increase
        assert learning_rates[1] > learning_rates[0]  # Step 10 > Step 0
        assert learning_rates[5] > learning_rates[1]  # Step 50 > Step 10

        # After warmup, should follow cosine decay pattern
        post_warmup_start = learning_rates[10]  # Around step 100
        post_warmup_later = learning_rates[50]  # Around step 500
        post_warmup_end = learning_rates[-1]  # Around step 1200

        # Should decrease after warmup due to cosine decay
        assert post_warmup_start > post_warmup_later > post_warmup_end

    def test_repr_string(self):
        """Test string representation of the schedule."""
        primary_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

        warmup_schedule = WarmupSchedule(
            warmup_steps=100,
            warmup_start_lr=1e-6,
            primary_schedule=primary_schedule
        )

        repr_str = repr(warmup_schedule)

        # Check that representation contains key information
        assert "WarmupSchedule" in repr_str
        assert "warmup_steps=100" in repr_str
        assert "warmup_start_lr=1e-06" in repr_str
        assert "ExponentialDecay" in repr_str