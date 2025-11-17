"""
Warmup Learning Rate Schedule Implementation for Deep Learning Techniques.

This module implements a learning rate schedule with a linear warmup phase
followed by a primary schedule. During warmup, the learning rate linearly
increases from a small initial value to the target rate, which helps stabilize
training in the early phases by preventing large gradient updates that could
destabilize the model.

The WarmupSchedule class acts as a wrapper around any Keras learning rate
schedule, adding a warmup period at the beginning of training. The schedule
uses the step parameter directly to avoid device placement issues with internal
counters.

Key Features:
- Linear warmup from configurable starting learning rate
- Seamless transition to primary schedule after warmup
- Device-agnostic implementation
- Full serialization support for model saving/loading
- Compatible with all Keras optimizers and schedules

Mathematical Behavior:
    During warmup (step <= warmup_steps):
        lr = warmup_start_lr + (primary_lr - warmup_start_lr) * (step / warmup_steps)

    After warmup (step > warmup_steps):
        lr = primary_schedule(step)

Usage Example:
    >>> # Direct usage
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
    >>>
    >>> # Via schedule_builder (recommended)
    >>> config = {
    ...     "type": "cosine_decay",
    ...     "warmup_steps": 1000,
    ...     "warmup_start_lr": 1e-6,
    ...     "learning_rate": 0.001,
    ...     "decay_steps": 10000,
    ...     "alpha": 0.0001
    ... }
    >>> lr_schedule = schedule_builder(config)
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

@keras.saving.register_keras_serializable()
class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup followed by primary schedule.

    This schedule implements a warmup period for the first N steps where the
    learning rate linearly increases from a small initial value to the target
    learning rate, followed by the primary learning rate schedule. The warmup
    helps prevent training instability in early epochs by gradually increasing
    the learning rate instead of starting with the full rate immediately.

    The schedule uses the step parameter directly to determine warmup progress,
    avoiding device placement issues with internal variables.

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

        # Store configuration - keep warmup_steps as Python int for boolean checks
        self.warmup_steps = warmup_steps  # Python int for boolean checks
        self.warmup_start_lr = warmup_start_lr  # Keep as Python float
        self.primary_schedule = primary_schedule

        # Log initialization with configuration details
        logger.info(
            f"WarmupSchedule initialized: warmup_steps={warmup_steps}, "
            f"warmup_start_lr={warmup_start_lr}, "
            f"primary_schedule={primary_schedule.__class__.__name__}"
        )

    def __call__(self, step: Union[int, tf.Tensor]) -> tf.Tensor:
        """Calculate the learning rate for the current step with corrected logic."""
        # Handle the no-warmup case efficiently
        if self.warmup_steps == 0:
            return self.primary_schedule(step)

        # Cast all values to float32 for graph-safe calculations
        step_float = tf.cast(step, tf.float32)
        warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)

        # Determine a FIXED target learning rate
        # This is the learning rate the warmup should ramp up to.
        # It's the initial_learning_rate of the primary schedule (i.e., its value at step 0).
        target_learning_rate = self.primary_schedule(0)

        # Logic for the warmup phase
        def warmup_fn():
            warmup_progress = step_float / warmup_steps_float
            # Correct linear interpolation to the fixed target
            return self.warmup_start_lr + (target_learning_rate - self.warmup_start_lr) * warmup_progress

        # Logic for the post-warmup phase
        def primary_fn():
            # Adjust the step for the primary schedule
            # The primary schedule should start its decay *after* the warmup is complete.
            # So, we pass it a step count that starts from 0 at the end of warmup.
            return self.primary_schedule(step - self.warmup_steps)

        # Use TensorFlow conditional to choose between warmup and primary rate
        return tf.cond(
            step_float < warmup_steps_float,
            true_fn=warmup_fn,
            false_fn=primary_fn
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
            'warmup_steps': self.warmup_steps,
            'warmup_start_lr': self.warmup_start_lr,
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
            f"warmup_steps={self.warmup_steps}, "
            f"warmup_start_lr={self.warmup_start_lr}, "
            f"primary_schedule={self.primary_schedule.__class__.__name__})"
        )

# ---------------------------------------------------------------------
