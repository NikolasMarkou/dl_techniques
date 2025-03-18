"""
# ---------------------------------------------------------------------
# Warmup Learning Rate Schedule
# ---------------------------------------------------------------------
#
# This module implements a learning rate schedule with a linear warmup phase
# followed by a primary schedule. During warmup, the learning rate linearly
# increases from a small initial value to the target rate, which helps with
# training stability in the early phases.
#
# The schedule tracks steps internally to ensure correct warmup behavior
# regardless of how step values are provided to the scheduler.
#
# Usage Example:
#   primary_schedule = keras.optimizers.schedules.CosineDecay(
#       initial_learning_rate=0.001,
#       decay_steps=10000
#   )
#   lr_schedule = WarmupSchedule(
#       warmup_steps=1000,
#       warmup_start_lr=1e-6,
#       primary_schedule=primary_schedule
#   )
#   optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
"""
import keras
import tensorflow as tf
from typing import Dict, Optional, Union, Any, ClassVar

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup followed by primary schedule.

    This schedule implements a warmup period for the first N steps where the 
    learning rate linearly increases from a small initial value to the target 
    learning rate, followed by the primary learning rate schedule.

    During warmup, the learning rate follows:
        lr = warmup_start_lr + (primary_lr - warmup_start_lr) * (step / warmup_steps)

    After warmup, the learning rate is determined by the primary schedule.
    """

    def __init__(
            self,
            warmup_steps: int,
            warmup_start_lr: float = 1e-8,
            primary_schedule: Optional[keras.optimizers.schedules.LearningRateSchedule] = None
    ):
        """Initialize the warmup schedule.

        Args:
            warmup_steps: Number of warmup steps. Must be >= 0.
            warmup_start_lr: Starting learning rate for warmup. Must be >= 0.0.
            primary_schedule: Main schedule to use after warmup. Required.

        Raises:
            ValueError: If warmup_steps < 0, warmup_start_lr < 0, or primary_schedule is None.
        """
        super().__init__()

        # Validate input parameters
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        if warmup_start_lr < 0.0:
            raise ValueError("warmup_start_lr must be >= 0.0")

        if primary_schedule is None:
            raise ValueError("primary_schedule must be defined")

        # Store configuration
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.int32)
        self.warmup_start_lr = tf.constant(warmup_start_lr, dtype=tf.float32)
        self.primary_schedule = primary_schedule

        # Initialize step counter for internal tracking
        self._steps_called = tf.Variable(0, trainable=False, dtype=tf.int32, name="warmup_steps_counter")

        # Log initialization
        logger.info(f"Warmup schedule initialized with {warmup_steps} steps")

    def __call__(self, step: Union[int, tf.Tensor]) -> tf.Tensor:
        """Calculate the learning rate for the current step.

        Args:
            step: Current training step index (used by primary schedule,
                 but internal counter is used for warmup calculation)

        Returns:
            Learning rate tensor for the current step
        """
        # Increment internal step counter
        self._steps_called.assign_add(1)
        current_step = tf.cast(self._steps_called, tf.float32)

        # Get the target learning rate from primary schedule
        primary_lr = self.primary_schedule(step)

        # Handle the case where warmup_steps is 0
        if self.warmup_steps == 0:
            return primary_lr

        # Calculate warmup progress as a fraction (0.0 to 1.0)
        warmup_fraction = tf.minimum(
            current_step / tf.cast(self.warmup_steps, tf.float32),
            1.0
        )

        # Linear interpolation between warmup_start_lr and primary_lr
        warmup_lr = self.warmup_start_lr + (primary_lr - self.warmup_start_lr) * warmup_fraction

        # Use warmup_lr during warmup phase, then switch to primary_lr
        return tf.cond(
            current_step <= tf.cast(self.warmup_steps, tf.float32),
            lambda: warmup_lr,
            lambda: primary_lr
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration of the schedule.

        Returns:
            Dictionary containing the configuration of this schedule.
        """
        return {
            'warmup_steps': int(self.warmup_steps.numpy()),
            'warmup_start_lr': float(self.warmup_start_lr.numpy()),
            'primary_schedule': keras.optimizers.schedules.serialize(self.primary_schedule)
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WarmupSchedule':
        """Create schedule from configuration dictionary.

        Args:
            config: Dictionary containing configuration parameters.

        Returns:
            An instance of WarmupSchedule initialized with the given config.
        """
        primary_schedule_config = config.pop('primary_schedule')
        config['primary_schedule'] = keras.optimizers.schedules.deserialize(primary_schedule_config)
        return cls(**config)