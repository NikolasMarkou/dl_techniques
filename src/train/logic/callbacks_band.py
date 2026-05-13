"""
Partial-convergence training callback for the LearnableNeuralCircuit
faithfulness/attribution experiments (E1, E3).

Plan: plan_2026-05-13_798d3a60.

The headline metric for the E1/E3 experiments is **hard-extraction Δ at
non-saturation** — i.e. ``soft_val_acc - hard_val_acc`` measured at a
checkpoint where the model is partially converged (val_accuracy in some
band like ``[0.70, 0.95]``) rather than at full saturation.

``StopOnAccuracyBand`` watches a monitor metric (typically ``val_accuracy``)
and sets ``model.stop_training = True`` the first epoch that the metric
falls inside the configured ``[low, high]`` band. Behaviour at the edges and
when the metric overshoots without ever sampling the band is controlled by
``mode``.
"""

from __future__ import annotations

from typing import Optional

import keras

from dl_techniques.utils.logger import logger


class StopOnAccuracyBand(keras.callbacks.Callback):
    """Stop training the first epoch where ``monitor`` enters ``[low, high]``.

    Parameters
    ----------
    monitor : str
        Name of the logs key to watch (e.g. ``"val_accuracy"``).
    low : float
        Lower (inclusive) bound of the non-saturation band.
    high : float
        Upper (inclusive) bound of the non-saturation band.
    mode : {"enter"}
        Currently only ``"enter"`` is supported: fire once when the monitor
        first satisfies ``low <= value <= high``. Reserved for future
        ``"window"`` semantics (require N consecutive epochs in band).
    verbose : int
        ``0`` = silent, ``1`` = log on band entry.

    Notes
    -----
    - The callback fires exactly once per training run. After firing,
      ``self.fired`` is ``True`` and ``self.band_epoch`` / ``self.band_value``
      record the entry point. Subsequent epochs (if any) are ignored.
    - If the monitor is missing from the epoch logs, the callback warns once
      and otherwise behaves as a no-op for that epoch.
    - If the metric jumps over the band (e.g. epoch 4 value = 0.45, epoch 5
      value = 0.99), no band sample is recorded — caller can detect this via
      ``self.fired is False`` at end of training and record an honest
      negative per the plan's Pre-Mortem scenarios.
    """

    def __init__(
        self,
        monitor: str = "val_accuracy",
        low: float = 0.70,
        high: float = 0.95,
        mode: str = "enter",
        verbose: int = 1,
    ) -> None:
        super().__init__()
        if mode != "enter":
            raise ValueError(
                f"StopOnAccuracyBand only supports mode='enter' (got {mode!r})"
            )
        if not (low <= high):
            raise ValueError(
                f"low ({low}) must be <= high ({high})"
            )
        self.monitor = monitor
        self.low = float(low)
        self.high = float(high)
        self.mode = mode
        self.verbose = int(verbose)
        self.fired: bool = False
        self.band_epoch: Optional[int] = None
        self.band_value: Optional[float] = None
        self._warned_missing: bool = False

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if self.fired:
            return
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            if not self._warned_missing:
                logger.warning(
                    f"StopOnAccuracyBand: monitor {self.monitor!r} not found "
                    f"in epoch logs (keys: {sorted(logs.keys())}); "
                    "callback will be a no-op until it appears."
                )
                self._warned_missing = True
            return
        value = float(value)
        if self.low <= value <= self.high:
            self.fired = True
            self.band_epoch = int(epoch)
            self.band_value = value
            if self.verbose:
                logger.info(
                    f"StopOnAccuracyBand: epoch {epoch} {self.monitor}={value:.4f} "
                    f"entered band [{self.low:.4f}, {self.high:.4f}] — "
                    "stopping training."
                )
            # Tell Keras to halt at end of this epoch.
            self.model.stop_training = True
