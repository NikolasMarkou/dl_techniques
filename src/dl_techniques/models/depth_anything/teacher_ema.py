"""Teacher EMA decay schedules + Keras callback for DepthAnything.

This module provides on-step EMA decay schedules (cosine and linear, with
warmup → asymptote) and a small ``keras.callbacks.Callback`` that drives
``DepthAnything.update_teacher_ema`` once per training batch.

The schedules follow the standard Mean-Teacher / DINO recipe: a low decay at
the start (so the teacher absorbs early student updates quickly) and a high
asymptotic decay (so the teacher becomes a slowly-moving average).

Mathematical forms
------------------
Let ``t`` be the (post-warmup) step index in ``[0, T]``.

* **Cosine schedule** —
  ``decay(t) = end - (end - start) * 0.5 * (1 + cos(pi * min(t, T) / T))``
  Smooth s-curve from ``start`` to ``end``.

* **Linear schedule** —
  ``decay(t) = start + (end - start) * min(t / T, 1.0)``
  Constant-rate ramp from ``start`` to ``end``.

Both schedules clamp at ``end`` once ``t >= T`` so multi-epoch training past
``total_steps`` does not over-shoot.

References
----------
- Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role models.
- Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision
  Transformers (DINO). *ICCV*.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import keras

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


def cosine_ema_schedule(
    decay_start: float,
    decay_end: float,
    total_steps: int,
) -> Callable[[int], float]:
    """Return a callable ``step -> decay`` implementing a cosine ramp.

    :param decay_start: Decay value at step 0. Typical: ``0.5``.
    :param decay_end: Asymptotic decay value at step ``total_steps``. Typical: ``0.999``.
    :param total_steps: Number of steps over which to ramp.
    :returns: Function ``f(step) -> float``. Clamps at ``decay_end`` for ``step >= total_steps``.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")

    def _schedule(step: int) -> float:
        t = max(0, min(int(step), int(total_steps)))
        progress = t / float(total_steps)
        return float(
            decay_end - (decay_end - decay_start) * 0.5 * (1.0 + math.cos(math.pi * progress))
        )

    return _schedule

# ---------------------------------------------------------------------


def linear_ema_schedule(
    decay_start: float,
    decay_end: float,
    total_steps: int,
) -> Callable[[int], float]:
    """Return a callable ``step -> decay`` implementing a linear ramp.

    :param decay_start: Decay value at step 0.
    :param decay_end: Decay value at step ``total_steps``.
    :param total_steps: Number of steps over which to ramp.
    :returns: Function ``f(step) -> float``. Clamps at ``decay_end`` for ``step >= total_steps``.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")

    def _schedule(step: int) -> float:
        t = max(0, min(int(step), int(total_steps)))
        progress = t / float(total_steps)
        return float(decay_start + (decay_end - decay_start) * progress)

    return _schedule

# ---------------------------------------------------------------------


class TeacherEMACallback(keras.callbacks.Callback):
    """Drive ``DepthAnything.update_teacher_ema`` once per train batch.

    On each ``on_train_batch_end``, advances an internal step counter; once
    the counter exceeds ``warmup_steps``, calls
    ``self.model.update_teacher_ema(decay=schedule(step - warmup_steps))``.

    The model must expose ``update_teacher_ema(decay: float)``. If the method
    is missing, the callback logs a single warning and disables itself.

    :param schedule: Callable ``step -> decay`` (e.g. ``cosine_ema_schedule(...)``).
    :param warmup_steps: Number of initial training steps during which the
        teacher is *not* updated. Defaults to ``0``.
    :param log_every: If ``> 0``, log the current decay every ``log_every``
        applied steps (post-warmup). Defaults to ``0`` (no logging).
    """

    def __init__(
        self,
        schedule: Callable[[int], float],
        warmup_steps: int = 0,
        log_every: int = 0,
    ) -> None:
        super().__init__()
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.schedule = schedule
        self.warmup_steps = int(warmup_steps)
        self.log_every = int(log_every)
        self._step: int = 0
        self._disabled: bool = False
        self._post_warmup_count: int = 0

    @property
    def step(self) -> int:
        """Total batches seen (pre + post warmup)."""
        return self._step

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        if self._disabled:
            return
        self._step += 1
        if self._step <= self.warmup_steps:
            return
        # Post-warmup step index, 0-based.
        post = self._step - self.warmup_steps - 1
        decay = float(self.schedule(post))
        update_fn = getattr(self.model, "update_teacher_ema", None)
        if update_fn is None:
            logger.warning(
                "TeacherEMACallback: model has no update_teacher_ema(decay); "
                "disabling callback."
            )
            self._disabled = True
            return
        update_fn(decay=decay)
        self._post_warmup_count += 1
        if self.log_every > 0 and self._post_warmup_count % self.log_every == 0:
            logger.info(
                f"TeacherEMACallback: step={self._step} post={post} decay={decay:.6f}"
            )

# ---------------------------------------------------------------------
