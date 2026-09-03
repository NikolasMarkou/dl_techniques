"""``FlatSchedule``: a constant-returning learning rate schedule.

``optimization/schedule.py`` is deliberately NOT modified for this plan (see
plan.md's Step 6 spec, `[SOFT]` constraint) -- `WarmupSchedule.primary_schedule`
is REQUIRED (cannot be ``None``, see its own docstring), so a trainer wanting
"linear warmup then flat, no decay" needs SOME
``keras.optimizers.schedules.LearningRateSchedule`` that just returns a fixed
value at every step, to hand to ``WarmupSchedule`` as its primary schedule.
This is that schedule, local to this trainer since no other trainer in this
plan needs it.
"""

from typing import Any, Dict, Union

import keras
import tensorflow as tf

# ---------------------------------------------------------------------


class FlatSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """A ``LearningRateSchedule`` that always returns the same learning rate.

    :param learning_rate: The constant learning rate to return at every step.
        Must be positive.
    :type learning_rate: float

    Example:

    .. code-block:: python

        from dl_techniques.optimization import WarmupSchedule
        from train.levjepa.schedule import FlatSchedule

        schedule = WarmupSchedule(
            warmup_steps=100,
            primary_schedule=FlatSchedule(learning_rate=3e-4),
        )
    """

    def __init__(self, learning_rate: float) -> None:
        super().__init__()
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        self.learning_rate = float(learning_rate)

    def __call__(self, step: Union[int, tf.Tensor]) -> tf.Tensor:
        del step
        return tf.constant(self.learning_rate, dtype=tf.float32)

    def get_config(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate}


# ---------------------------------------------------------------------
