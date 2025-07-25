"""
Warmup Learning Rate Schedule Implementation for Deep Learning Techniques.

This module implements a learning rate schedule with a linear warmup phase
followed by a primary schedule. During warmup, the learning rate linearly
increases from a small initial value to the target rate, which helps stabilize
training in the early phases by preventing large gradient updates that could
destabilize the model.

The WarmupSchedule class acts as a wrapper around any Keras learning rate
schedule, adding a warmup period at the beginning of training. The schedule
tracks steps internally to ensure correct warmup behavior regardless of how
step values are provided to the scheduler.

Key Features:
- Linear warmup from configurable starting learning rate
- Seamless transition to primary schedule after warmup
- Internal step tracking for consistent behavior
- Full serialization support for model saving/loading
- Compatible with all Keras optimizers and schedules

Mathematical Behavior:
    During warmup (step <= warmup_steps):
        lr = warmup_start_lr + (primary_lr - warmup_start_lr) * (step / warmup_steps)

    After warmup (step > warmup_steps):
        lr = primary_schedule(step)

Usage Example:
    >>> primary_schedule = keras.optimizers.schedules.CosineDecay(
    ...     initial_learning_rate=0.001,
    ...     decay_steps=10000
    ... )
    >>> warmup_lr_schedule = WarmupSchedule(
    ...     warmup_steps=1000,
    ...     warmup_start_lr=1e-6,
    ...     primary_schedule=primary_schedule
    ... )
    >>> optimizer = keras.optimizers.Adam(learning_rate=warmup_lr_schedule)
"""

import keras
import tensorflow as tf
from typing import Dict, Optional, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# main classes
# ---------------------------------------------------------------------


class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup followed by primary schedule.

    This schedule implements a warmup period for the first N steps where the
    learning rate linearly increases from a small initial value to the target
    learning rate, followed by the primary learning rate schedule. The warmup
    helps prevent training instability in early epochs by gradually increasing
    the learning rate instead of starting with the full rate immediately.

    The schedule maintains an internal step counter to ensure consistent warmup
    behavior regardless of the step values passed to the __call__ method. This
    is important because different training loops may pass different step values.

    Attributes:
        warmup_steps: Number of steps over which to perform warmup.
        warmup_start_lr: Initial learning rate at the start of warmup.
        primary_schedule: The main learning rate schedule to use after warmup.
    """

    def __init__(
            self,
            warmup_steps: int,
            warmup_start_lr: float = 1e-8,
            primary_schedule: Optional[keras.optimizers.schedules.LearningRateSchedule] = None
    ) -> None:
        """Initialize the warmup learning rate schedule.

        Args:
            warmup_steps: Number of warmup steps. Must be >= 0. If 0, no warmup
                is applied and the primary schedule is used immediately.
            warmup_start_lr: Starting learning rate for warmup phase. Must be >= 0.0.
                This should typically be much smaller than the target learning rate.
            primary_schedule: Main learning rate schedule to use after warmup.
                Cannot be None - this is the schedule that determines the target
                learning rate and post-warmup behavior.

        Raises:
            ValueError: If warmup_steps < 0, warmup_start_lr < 0, or
                       primary_schedule is None.

        Example:
            >>> cosine_schedule = keras.optimizers.schedules.CosineDecay(
            ...     initial_learning_rate=0.001, decay_steps=10000
            ... )
            >>> warmup_schedule = WarmupSchedule(
            ...     warmup_steps=1000,
            ...     warmup_start_lr=1e-6,
            ...     primary_schedule=cosine_schedule
            ... )
        """
        super().__init__()

        # Validate input parameters with descriptive error messages
        if warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be >= 0, got {warmup_steps}. "
                "Use 0 for no warmup period."
            )

        if warmup_start_lr < 0.0:
            raise ValueError(
                f"warmup_start_lr must be >= 0.0, got {warmup_start_lr}. "
                "This should be a small positive learning rate for the warmup start."
            )

        if primary_schedule is None:
            raise ValueError(
                "primary_schedule cannot be None. Must provide a valid "
                "LearningRateSchedule instance to use after warmup."
            )

        # Store configuration as TensorFlow constants for efficiency
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.int32, name="warmup_steps")
        self.warmup_start_lr = tf.constant(warmup_start_lr, dtype=tf.float32, name="warmup_start_lr")
        self.primary_schedule = primary_schedule

        # Initialize internal step counter for tracking warmup progress
        # This counter is independent of the step parameter passed to __call__
        self._internal_step_counter = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int32,
            name="warmup_internal_step_counter"
        )

        # Log initialization with configuration details
        logger.info(
            f"WarmupSchedule initialized: warmup_steps={warmup_steps}, "
            f"warmup_start_lr={warmup_start_lr}, "
            f"primary_schedule={primary_schedule.__class__.__name__}"
        )

    def __call__(self, step: Union[int, tf.Tensor]) -> tf.Tensor:
        """Calculate the learning rate for the current step.

        This method implements the core warmup logic by using an internal step
        counter to track warmup progress, while still passing the provided step
        to the primary schedule for consistency.

        Args:
            step: Current training step index. This is passed to the primary
                schedule but the internal counter is used for warmup calculations
                to ensure consistent behavior.

        Returns:
            Learning rate tensor for the current step. During warmup, this is
            a linear interpolation between warmup_start_lr and the primary
            schedule's rate. After warmup, this is the primary schedule's rate.

        Note:
            The internal step counter is incremented on each call, ensuring
            consistent warmup behavior regardless of the step parameter values.
        """
        # Increment internal step counter for warmup tracking
        self._internal_step_counter.assign_add(1)
        current_internal_step = tf.cast(self._internal_step_counter, tf.float32)

        # Get the target learning rate from primary schedule
        primary_learning_rate = self.primary_schedule(step)

        # Handle the no-warmup case efficiently
        if self.warmup_steps == 0:
            return primary_learning_rate

        # Calculate warmup progress as a fraction (0.0 to 1.0)
        warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = tf.minimum(current_internal_step / warmup_steps_float, 1.0)

        # Linear interpolation between warmup_start_lr and primary_lr
        learning_rate_range = primary_learning_rate - self.warmup_start_lr
        warmup_learning_rate = self.warmup_start_lr + learning_rate_range * warmup_progress

        # Use warmup learning rate during warmup phase, primary rate afterward
        return tf.cond(
            current_internal_step <= warmup_steps_float,
            lambda: warmup_learning_rate,
            lambda: primary_learning_rate
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization.

        Returns:
            Dictionary containing all configuration parameters needed to
            recreate this schedule instance. The primary_schedule is
            serialized using Keras serialization utilities.

        Example:
            >>> config = warmup_schedule.get_config()
            >>> restored_schedule = WarmupSchedule.from_config(config)
        """
        return {
            'warmup_steps': int(self.warmup_steps.numpy()),
            'warmup_start_lr': float(self.warmup_start_lr.numpy()),
            'primary_schedule': keras.optimizers.schedules.serialize(self.primary_schedule)
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WarmupSchedule':
        """Create WarmupSchedule instance from configuration dictionary.

        This method enables proper deserialization of saved models that use
        WarmupSchedule learning rate schedules.

        Args:
            config: Configuration dictionary created by get_config(). Must
                contain 'warmup_steps', 'warmup_start_lr', and 'primary_schedule'.

        Returns:
            New WarmupSchedule instance configured with the provided parameters.

        Raises:
            KeyError: If required configuration keys are missing.

        Example:
            >>> config = existing_schedule.get_config()
            >>> new_schedule = WarmupSchedule.from_config(config)
        """
        # Validate required configuration keys
        required_keys = ['warmup_steps', 'warmup_start_lr', 'primary_schedule']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(
                f"Missing required configuration keys: {missing_keys}. "
                f"Required keys: {required_keys}"
            )

        # Extract and deserialize primary schedule
        primary_schedule_config = config.pop('primary_schedule')
        config['primary_schedule'] = keras.optimizers.schedules.deserialize(primary_schedule_config)

        return cls(**config)

    def __repr__(self) -> str:
        """Return string representation of the schedule for debugging.

        Returns:
            String representation showing key configuration parameters.
        """
        return (
            f"WarmupSchedule("
            f"warmup_steps={int(self.warmup_steps.numpy())}, "
            f"warmup_start_lr={float(self.warmup_start_lr.numpy())}, "
            f"primary_schedule={self.primary_schedule.__class__.__name__})"
        )