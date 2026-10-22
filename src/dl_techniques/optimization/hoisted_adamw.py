"""
AdamW whose decoupled weight decay calls the learning-rate schedule once per step.

Keras 3.8's ``AdamW`` reads ``self.learning_rate`` inside its decoupled weight
decay once per decayed variable (``_apply_weight_decay``), and with a
``LearningRateSchedule`` every read calls the schedule again. In an eager step
that is arithmetic repeated per variable; in a traced step (``fit`` under
``tf.function`` and XLA) it is graph built per variable: CliffordCLIP ``nano``
(691 variables) traces one copy of ``WarmupSchedule`` per decayed variable,
each with its ``tf.cond``, in its first training step.

This optimizer reads the schedule once for the whole decay:

    lr  = schedule(iterations)              # once per step
    p  -= p * weight_decay * lr             # every decayed variable

which is stock Keras' own expression in stock Keras' order, so the result
equals ``keras.optimizers.AdamW``. The Adam update is stock: Keras already
passes it one learning rate for every variable. The schedule is therefore
called twice per step (decay and update, both at the same ``iterations``)
where stock AdamW calls it ``1 + n_decayed`` times.

The measured effect on CliffordCLIP's first training step is in
``src/train/cliffordnet/README.md`` (Gotcha 5), its one caller.

The decay is applied with a plain ``assign``, not through the TensorFlow
backend's ``tf.distribute`` merge call: this optimizer is for single-device
training, like ``LRMultiplierAdamW``.
"""

from typing import Any, Dict, List, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


# DECISION plan-2026-10-14T062616-81938e81/D-280: a subclass that reads the
# schedule once for AdamW's decoupled decay, not a custom train_step
# (forbidden: stock fit only) and not a cheaper schedule (a where-based
# WarmupSchedule cut 13 s of 264 s, inside the noise, D-244). Do NOT re-read
# self.learning_rate inside the per-variable loop: that re-traces the schedule
# once per variable, which is the cost this class removes. The Adam scalars
# are NOT hoisted too: that cut a further 14 s but the ship rule takes the
# simpler of two qualifying arms (D-280). Equality with stock AdamW and the
# call count are held by tests/test_optimization/test_hoisted_adamw.py.
@register_dl_technique("dl_techniques.optimization.hoisted_adamw")
class HoistedAdamW(keras.optimizers.AdamW):
    """AdamW that evaluates its schedule once per step for the weight decay.

    Interface contract: every argument is :class:`keras.optimizers.AdamW`'s,
    and the variable layout (one ``momentum`` and one ``velocity`` slot per
    trainable variable) is AdamW's, so weights, slots and configs move
    between the two. ``exclude_from_weight_decay`` is honoured. It runs
    wrapped in ``keras.optimizers.LossScaleOptimizer``. Only
    ``_apply_weight_decay`` differs from AdamW; nothing is raised beyond
    AdamW's own errors.

    Args:
        **kwargs: Passed to :class:`keras.optimizers.AdamW`.

    Example:
        >>> schedule = learning_rate_schedule_builder({...})
        >>> optimizer = HoistedAdamW(learning_rate=schedule, weight_decay=0.1)
    """

    def _apply_weight_decay(self, variables: List[keras.Variable]) -> None:
        """Decoupled weight decay with one schedule call for all variables."""
        if self.weight_decay is None:
            return
        learning_rate = self.learning_rate
        by_dtype: Dict[str, Tuple[Any, Any]] = {}
        for variable in variables:
            if not self._use_weight_decay(variable):
                continue
            dtype = variable.dtype
            if dtype not in by_dtype:
                by_dtype[dtype] = (
                    keras.ops.cast(learning_rate, dtype),
                    keras.ops.cast(self.weight_decay, dtype),
                )
            lr, wd = by_dtype[dtype]
            variable.assign(variable - variable * wd * lr)

# ---------------------------------------------------------------------
