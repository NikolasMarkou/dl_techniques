"""
NoiseSigmaCurriculumCallback — curriculum learning for additive-Gaussian-noise
denoising. Progressively WIDENS the per-image noise-sigma sampling range across
training epochs so the model first sees low noise and gradually faces an
increasingly large spread of noise levels.

Context: bias-free denoisers generalize across
noise levels, but training converges faster and more stably when the noise
difficulty is ramped (curriculum). This callback owns a ``keras.Variable``
(``sigma_max_var``) that the tf.data noise-injection function reads when sampling
each patch's sigma from ``[sigma_min, sigma_max_var]``. On ``on_epoch_begin`` the
callback computes a scheduled ``sigma_max`` (linear / cosine / exp interpolation
from ``sigma_max_start`` to ``sigma_max_end`` over ``total_epochs``) and
``.assign``s it to the variable. A captured
``tf.Variable`` read inside ``tf.data.map`` reflects per-epoch ``.assign``
(checked by a spike before this callback was written).

Schedule math mirrors ``TemperatureAnnealingCallback``. An optional second
variable (``sigma_min_var``) can be scheduled the same way if a non-zero,
widening lower bound is desired; by default only the upper bound is ramped.

Serialization: ``get_config`` returns only the scalar schedule parameters — NOT
the live ``keras.Variable`` (which the trainer re-supplies at construction). A
callback reconstructed via ``from_config`` has ``sigma_max_var=None`` and is a
no-op until a variable is attached; this keeps the callback round-trippable
without binding it to a specific training graph.
"""

from typing import Optional

import keras

# ------------------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------------------

from dl_techniques.callbacks._shared import SCHEDULES, interpolate_schedule
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ------------------------------------------------------------------------------

@register_dl_technique("dl_techniques.callbacks.noise_sigma_curriculum")
class NoiseSigmaCurriculumCallback(keras.callbacks.Callback):
    """Widen the noise-sigma sampling range across epochs (noise curriculum).

    Args:
        sigma_max_var: A ``keras.Variable`` (or ``tf.Variable``) that the tf.data
            noise function reads as the upper bound of the per-image sigma sampling
            range. May be ``None`` (e.g. after ``from_config``); the callback is a
            no-op until a variable is attached. Not serialized.
        sigma_max_start: Upper-bound sigma at epoch 0 (narrow range / low noise).
        sigma_max_end: Upper-bound sigma at the final epoch (wide range / high noise).
        total_epochs: Schedule length in epochs. Must be >= 1.
        schedule: One of ``'linear'``, ``'cosine'``, ``'exp'``.
        sigma_min_var: Optional second ``keras.Variable`` for the lower bound. If
            provided, it is scheduled from ``sigma_min_start`` to ``sigma_min_end``
            with the same schedule. Not serialized.
        sigma_min_start: Lower-bound sigma at epoch 0 (used only if ``sigma_min_var``).
        sigma_min_end: Lower-bound sigma at the final epoch (used only if ``sigma_min_var``).

    Note:
        For ``schedule='exp'`` all scheduled endpoints must be strictly positive
        (log-space interpolation); the ``sigma_min_*`` endpoints count only when
        ``sigma_min_var`` is given. For ``'linear'``/``'cosine'`` they must be >= 0.
        With ``sigma_min_var`` the lower endpoints must not exceed the upper ones.
        Both are checked at construction, not at the first epoch.
    """

    SCHEDULES = SCHEDULES

    def __init__(
        self,
        sigma_max_var=None,
        sigma_max_start: float = 0.05,
        sigma_max_end: float = 0.5,
        total_epochs: int = 50,
        schedule: str = "linear",
        sigma_min_var=None,
        sigma_min_start: float = 0.0,
        sigma_min_end: float = 0.0,
    ) -> None:
        super().__init__()
        if schedule not in self.SCHEDULES:
            raise ValueError(
                f"schedule must be one of {sorted(self.SCHEDULES)}; got {schedule!r}."
            )
        if total_epochs < 1:
            raise ValueError("total_epochs must be >= 1.")
        for name, val in (
            ("sigma_max_start", sigma_max_start),
            ("sigma_max_end", sigma_max_end),
            ("sigma_min_start", sigma_min_start),
            ("sigma_min_end", sigma_min_end),
        ):
            if val < 0:
                raise ValueError(f"{name} must be >= 0; got {val}.")
        if schedule == "exp" and (sigma_max_start <= 0 or sigma_max_end <= 0):
            raise ValueError(
                "schedule='exp' requires sigma_max_start and sigma_max_end > 0."
            )
        # DECISION plan-2026-11-11T131349-13b28b37/D-031: the lower bound is validated
        # HERE, at construction, and only when ``sigma_min_var`` is attached (otherwise
        # sigma_min_* is never scheduled). Do NOT defer it to ``on_epoch_begin``: the exp
        # schedule took ``math.log(0)`` inside ``fit`` ("math domain error") and a min above
        # the max produced an empty sampling range part-way through a run. Comparing the two
        # endpoint pairs is enough: both bounds use the same fraction, so under all three
        # schedules ``max - min`` is monotone between its endpoints and cannot cross inside.
        if sigma_min_var is not None:
            if schedule == "exp" and (sigma_min_start <= 0 or sigma_min_end <= 0):
                raise ValueError(
                    "schedule='exp' with sigma_min_var requires sigma_min_start and "
                    f"sigma_min_end > 0; got {sigma_min_start} and {sigma_min_end}."
                )
            if sigma_min_start > sigma_max_start or sigma_min_end > sigma_max_end:
                raise ValueError(
                    "sigma_min must not exceed sigma_max at either endpoint; got "
                    f"sigma_min ({sigma_min_start} -> {sigma_min_end}) against "
                    f"sigma_max ({sigma_max_start} -> {sigma_max_end})."
                )

        self.sigma_max_var = sigma_max_var
        self.sigma_max_start = float(sigma_max_start)
        self.sigma_max_end = float(sigma_max_end)
        self.total_epochs = int(total_epochs)
        self.schedule = schedule
        self.sigma_min_var = sigma_min_var
        self.sigma_min_start = float(sigma_min_start)
        self.sigma_min_end = float(sigma_min_end)

    # ------------------------------------------------------------------
    def _interp(self, start: float, end: float, epoch: int) -> float:
        """Interpolate start -> end at ``epoch`` per the configured schedule."""
        return interpolate_schedule(
            self.schedule, start, end, epoch, self.total_epochs
        )

    def sigma_max_at(self, epoch: int) -> float:
        """Public helper: scheduled upper-bound sigma at ``epoch``."""
        return self._interp(self.sigma_max_start, self.sigma_max_end, epoch)

    def sigma_min_at(self, epoch: int) -> float:
        """Public helper: scheduled lower-bound sigma at ``epoch``."""
        return self._interp(self.sigma_min_start, self.sigma_min_end, epoch)

    # ------------------------------------------------------------------
    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        if self.sigma_max_var is None:
            logger.warning(
                "NoiseSigmaCurriculumCallback: sigma_max_var is None; "
                "noise curriculum is a no-op this run."
            )
            return
        smax = self.sigma_max_at(epoch)
        self.sigma_max_var.assign(smax)
        msg = f"NoiseSigmaCurriculum: epoch {epoch} -> sigma_max={smax:.4f}"
        if self.sigma_min_var is not None:
            smin = self.sigma_min_at(epoch)
            self.sigma_min_var.assign(smin)
            msg += f", sigma_min={smin:.4f}"
        logger.info(msg)

    def get_config(self) -> dict:
        # Live Variables are intentionally NOT serialized (re-supplied by trainer).
        return {
            "sigma_max_start": self.sigma_max_start,
            "sigma_max_end": self.sigma_max_end,
            "total_epochs": self.total_epochs,
            "schedule": self.schedule,
            "sigma_min_start": self.sigma_min_start,
            "sigma_min_end": self.sigma_min_end,
        }

    @classmethod
    def from_config(cls, config: dict) -> "NoiseSigmaCurriculumCallback":
        # keras.callbacks.Callback provides no default from_config; reconstruct with
        # no live Variable attached (the trainer re-supplies it at construction).
        return cls(**config)

# ------------------------------------------------------------------------------

