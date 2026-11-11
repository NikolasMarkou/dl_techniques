"""
``EMAShadowCallback`` -- checkpoint-only EMA-shadow driver.

Drives a model-owned "shadow" set of weights that tracks an exponential
moving average of the LIVE trainable weights but never participates in the
forward pass or the loss -- distinct from
``dl_techniques.models.vision.depth_anything.teacher_ema.TeacherEMACallback``,
which drives a second network that DOES feed the loss every step. This
callback reuses that module's decay schedules (``cosine_ema_schedule`` /
``linear_ema_schedule``) rather than re-deriving the math, and mirrors its
duck-typed model-method contract (here ``update_ema_shadow(decay=...)`` instead of ``update_teacher_ema``).

Step-skip semantics port the PyTorch reference's ``WeightEMA`` callback
(``callbacks.py``): updates fire only every ``update_every`` steps, at or
after ``start_step``. Unlike the reference, this port does NOT compound
``decay ** elapsed_steps`` when steps were skipped -- see the ``DECISION`` note on ``EMAShadowCallback`` for why
that reference detail is not reproduced.

References:
    - Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role
      models. (EMA teacher/shadow lineage.)
    - LeVJEPA PyTorch reference, ``callbacks.py::WeightEMA`` (pasted
      transcript; no public arXiv id).
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import keras

# ------------------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.models.vision.depth_anything.teacher_ema import (
    cosine_ema_schedule,
    linear_ema_schedule,
)

# ------------------------------------------------------------------------------

# Re-exported for convenience -- callers of EMAShadowCallback should not need
# a separate import from teacher_ema.py to build a schedule.
__all__ = ["EMAShadowCallback", "cosine_ema_schedule", "linear_ema_schedule"]

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.callbacks.ema_shadow_callback")
class EMAShadowCallback(keras.callbacks.Callback):
    """Drive ``model.update_ema_shadow(decay=...)`` on a step schedule.

    On each qualifying ``on_train_batch_end`` (``current_step >= start_step``
    and ``current_step % update_every == 0``), resolves a decay value and
    calls ``self.model.update_ema_shadow(decay=<resolved_decay>)``. The model
    must expose that method; if it is missing, the callback logs a single
    warning and disables itself (mirrors ``TeacherEMACallback``).

    ``current_step`` is the GLOBAL 0-based optimizer step, read from
    ``model.optimizer.iterations`` (post-increment at batch end, so
    ``iterations - 1``). The callback keeps no step counter of its own: a new
    instance attached to a resumed run (``initial_epoch > 0``, restored
    optimizer) continues the schedule where the optimizer stands, and
    ``start_step`` / ``update_every`` are global-step quantities. The decay
    schedule is called with that same global step, so
    ``cosine_ema_schedule(..., total_steps=T)`` ramps over ``T`` optimizer
    steps, not over ``T`` qualifying updates. A model compiled with
    ``steps_per_execution > 1`` is refused in ``on_train_begin``: Keras then
    calls ``on_train_batch_end`` once per execution, so the per-batch
    schedule could not be honoured.

    This callback does NOT snapshot or initialize the shadow at
    ``on_train_begin`` -- see the ``DECISION`` note below. It only decides
    WHEN to call ``update_ema_shadow`` and with what decay; the model owns
    the shadow's storage and is responsible for lazily initializing it (e.g.
    copying the live weights) on its own first call, since only the model
    knows the shadow's actual variables.

    :param decay: Resolved decay value(s), either a flat ``float`` (constant
        every qualifying step) or a ``Callable[[int], float]`` schedule such
        as :func:`cosine_ema_schedule` / :func:`linear_ema_schedule`, called
        with the global 0-based optimizer step of the qualifying update. See
        the ``DECISION`` note for why both forms are supported.
    :type decay: Union[float, Callable[[int], float]]
    :param update_every: Apply an update only every ``update_every`` global
        training steps. Must be positive. Defaults to ``1`` (every step),
        matching ``TeacherEMACallback``'s implicit behavior.
    :type update_every: int
    :param start_step: Global optimizer step (0-indexed, counting every
        batch, not just qualifying ones) before which no update fires.
        Defaults to ``0``.
    :type start_step: int

    Example:

    .. code-block:: python

        from dl_techniques.callbacks.ema_shadow_callback import (
            EMAShadowCallback, cosine_ema_schedule,
        )

        callback = EMAShadowCallback(decay=0.9999, update_every=32)
        # or, with a schedule:
        callback = EMAShadowCallback(
            decay=cosine_ema_schedule(0.5, 0.9999, total_steps=10_000),
            update_every=1,
            start_step=100,
        )
        model.fit(dataset, callbacks=[callback])
    """

    # DECISION plan-2026-09-03T113223-2a714a91/D-015
    # The PyTorch reference's `WeightEMA` takes a flat `decay: float` only
    # (no schedule). `TeacherEMACallback` (this repo's closest sibling, per
    # D-006) takes ONLY a `Callable[[int], float]` schedule. This class
    # accepts EITHER: a bare `float` is normalized once in `__init__` into
    # a constant-returning callable, so `call time` always deals with one
    # shape (`Callable[[int], float]`). WHAT NOT TO DO: do not special-case
    # `isinstance(self.decay, float)` inside `on_train_batch_end` -- that
    # duplicates the branch on every batch for no benefit; normalize once.
    # Reasoning: the reference's actual shipped default (`decay=0.9999`,
    # flat) is what this plan's own trainer will most likely use, but
    # `cosine_ema_schedule`/`linear_ema_schedule` already exist and are
    # reused (not re-derived) precisely so a caller CAN ramp the shadow decay
    # exactly like `TeacherEMACallback` does, without a second math
    # implementation. Supporting both is the union of "match the reference
    # default" and "match the house schedule pattern", not speculative
    # generality -- both call sites (a flat default and a ramped variant)
    # are real, differing only in which one the training script picks.
    # See decisions.md D-015.
    #
    # DECISION plan-2026-09-03T113223-2a714a91/D-016
    # The PyTorch reference's `WeightEMA` compounds `decay ** elapsed_steps`
    # when `current_step - self._last_ema_step > update_every` (i.e. after a
    # gap, e.g. resumed training or skipped batches), so a longer gap decays
    # the shadow harder toward the live weights. This port does NOT
    # reproduce that compounding: `decay` (or `decay_schedule(step)`) is
    # applied as-is on every qualifying step, with the global optimizer step
    # as the schedule argument (D-018 replaced the earlier qualifying-update
    # count, which restarted at 0 on resume).
    # WHAT NOT TO DO: do not add `decay ** elapsed_steps` "for fidelity" --
    # this callback has no way to observe a genuine step gap (Keras does not
    # skip `on_train_batch_end` calls the way a resumed-Lightning-trainer's
    # `global_step` can jump), so `elapsed_steps` would always be exactly
    # `update_every` in ordinary use, making the compounding a no-op that
    # only adds a footgun if `update_every` is later changed mid-training.
    # See decisions.md D-016.
    #
    # DECISION plan-2026-11-11T131349-13b28b37/D-018
    # The step comes from `model.optimizer.iterations`, never from a counter
    # on this object. A Python counter restarts at 0 on every new instance,
    # so a resumed run (restored optimizer, `initial_epoch > 0`) replayed the
    # schedule from its 0.5 start and re-applied `start_step`; it also
    # carried over, unreset, across two `fit` calls on one instance.
    # `iterations` is already incremented when `on_train_batch_end` runs
    # (batch 0 of a fresh run sees 1), hence `- 1` for the 0-based step
    # (pinned against a real fit in test_ema_shadow_callback.py). WHAT NOT TO
    # DO: do not reintroduce `_step` / `_update_count`, and do not "support"
    # `steps_per_execution > 1` by multiplying: Keras fires the batch-end hook
    # once per execution, so `update_every` and the schedule would silently
    # coarsen. It is refused in `on_train_begin` instead. See decisions.md
    # D-018.

    def __init__(
        self,
        decay: Union[float, Callable[[int], float]],
        update_every: int = 1,
        start_step: int = 0,
    ) -> None:
        super().__init__()
        if update_every <= 0:
            raise ValueError(f"update_every must be positive, got {update_every}")
        if start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {start_step}")

        if callable(decay):
            self.decay_schedule: Callable[[int], float] = decay
        else:
            decay_value = float(decay)
            self.decay_schedule = lambda _count, _value=decay_value: _value

        self.update_every = int(update_every)
        self.start_step = int(start_step)
        self._disabled: bool = False

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        steps_per_execution = getattr(self.model, "steps_per_execution", 1)
        if steps_per_execution != 1:
            raise ValueError(
                f"EMAShadowCallback needs steps_per_execution=1, got "
                f"{steps_per_execution}: Keras then calls on_train_batch_end once "
                f"per execution, so the per-step EMA schedule would be miscounted."
            )

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        if self._disabled:
            return

        # `iterations` is post-increment here: the first batch of a fresh run
        # sees 1. See the D-018 anchor above.
        current_step = int(keras.ops.convert_to_numpy(self.model.optimizer.iterations)) - 1
        if current_step < self.start_step:
            return
        if current_step % self.update_every != 0:
            return

        update_fn = getattr(self.model, "update_ema_shadow", None)
        if update_fn is None:
            logger.warning(
                "EMAShadowCallback: model has no update_ema_shadow(decay); "
                "disabling callback."
            )
            self._disabled = True
            return

        update_fn(decay=float(self.decay_schedule(current_step)))

    def get_config(self) -> dict:
        """Return the non-callable constructor arguments.

        ``decay`` is omitted (a schedule is a closure, not data), so the dict
        cannot rebuild the callback on its own; it is for introspection and
        logging. Callbacks are not part of a saved ``.keras`` model config.

        :return: Dictionary with ``update_every`` and ``start_step``.
        :rtype: dict
        """
        return {
            "update_every": self.update_every,
            "start_step": self.start_step,
        }


# ---------------------------------------------------------------------
