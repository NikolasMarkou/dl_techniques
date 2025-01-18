import keras
import tensorflow as tf
from typing import Dict, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule with warmup period.

    This schedule implements a warmup period for the first N steps where the learning rate
    linearly increases from a small initial value to the target learning rate, followed
    by the primary learning rate schedule.
    """

    def __init__(
            self,
            warmup_steps: int,
            warmup_start_lr: float = 1e-8,
            primary_schedule: Optional[keras.optimizers.schedules.LearningRateSchedule] = None
    ):
        """Initialize the warmup schedule.

        Args:
            warmup_steps: Number of warmup steps
            warmup_start_lr: Starting learning rate for warmup
            primary_schedule: Main schedule to use after warmup (if None, uses constant rate)
        """
        super().__init__()

        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        if warmup_start_lr < 0.0:
            raise ValueError("warmup_start_lr must be >= 0.0")

        if primary_schedule is None:
            raise ValueError("primary_schedule must be defined")

        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.primary_schedule = primary_schedule

        # Count of steps called
        self.steps_called = tf.Variable(0, trainable=False, dtype=tf.int32)

        # Log initial setup
        logger.info(f"Warmup schedule initialized with {warmup_steps} steps")

    def __call__(self, step: Union[int, tf.Tensor]) -> tf.Tensor:
        """Calculate the learning rate for the current step.

        Args:
            step: Current training step (not used for warmup calculation)

        Returns:
            Learning rate for the current step
        """
        # Increment step counter
        self.steps_called.assign_add(1)

        current_step = tf.cast(self.steps_called, tf.float32)
        current_step_percentage = current_step / (self.warmup_steps + 1e-7)

        # Primary schedule phase
        primary_lr = self.primary_schedule(step)

        # Calculate warmup rate of change as linear interpolation between points
        warmup_rate_at_step = (
            tf.maximum(
                x=0.0,
                y=(primary_lr * current_step_percentage) + self.warmup_start_lr * (1.0 - current_step_percentage)
            )
        )

        # Use warmup_lr during warmup phase, then switch to primary_lr
        return tf.cond(
            current_step <= self.warmup_steps,
            lambda: warmup_rate_at_step,
            lambda: primary_lr
        )

    def get_config(self) -> Dict:
        """Return configuration of the schedule."""
        return {
            'warmup_steps': self.warmup_steps,
            'warmup_start_lr': self.warmup_start_lr,
            'primary_schedule': keras.optimizers.schedules.serialize(self.primary_schedule)
        }

    @classmethod
    def from_config(cls, config: Dict) -> 'WarmupSchedule':
        """Create schedule from configuration dictionary."""
        return cls(**config)

# ---------------------------------------------------------------------
