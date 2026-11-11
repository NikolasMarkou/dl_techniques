"""
HarmonicExponentAnnealingCallback — anneals per-level harmonic exponents.

Hierarchical harmonic heads route softly when their exponents are small
and sharply when they are large. Starting soft keeps every cluster
reachable early in training (no premature hard routing); hardening on a
schedule recovers sharp, interpretable nearest-prototype decisions by the
end. This callback moves each targeted layer's per-level ``n`` from
``n_init`` to ``n_final`` over ``total_epochs`` on a linear, cosine, or
exponential (log-space geometric) schedule.

Targeted layers expose ``num_levels`` (int) and
``set_n_per_level(values)`` — matches the contract of
``dl_techniques.layers.structured_linear.hierarchical_harmonic.HierarchicalHarmonicHead``.
``n_init`` / ``n_final`` are each a scalar (broadcast to all levels) or one
value per level.
"""

from typing import List, Optional, Sequence, Union

import keras
import numpy as np

# ------------------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------------------

from dl_techniques.callbacks._shared import SCHEDULES, interpolate_schedule, walk_layers
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ------------------------------------------------------------------------------


@register_dl_technique("dl_techniques.callbacks.harmonic_exponent_annealing")
class HarmonicExponentAnnealingCallback(keras.callbacks.Callback):
    """Anneal per-level harmonic exponents across epochs.

    :param schedule: One of ``'linear'``, ``'cosine'``, ``'exp'``.
    :param n_init: Starting exponent(s) at epoch 0: scalar or per-level.
    :param n_final: Final exponent(s): scalar or per-level.
    :param total_epochs: Schedule length. Must be >= 1.
    :param layer_names: If provided, only these layer names are touched.
        Otherwise all layers exposing ``set_n_per_level`` are annealed.
        A run in which no layer matches logs one WARNING in ``on_train_begin``.

    ``n_init`` / ``n_final`` may be numpy arrays; they are stored as plain lists of
    floats so ``get_config`` stays JSON-serializable. The exponent is overwritten at
    every epoch start, so any move the optimizer made to a trainable exponent during
    the previous epoch is discarded; freezing it is deferred
    (plan-2026-11-11T131349-13b28b37/D-023).
    """

    SCHEDULES = SCHEDULES

    def __init__(
        self,
        schedule: str = "cosine",
        n_init: Union[float, Sequence[float]] = 1.0,
        n_final: Union[float, Sequence[float]] = 8.0,
        total_epochs: int = 50,
        layer_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if schedule not in self.SCHEDULES:
            raise ValueError(
                f"schedule must be one of {sorted(self.SCHEDULES)}; got {schedule!r}."
            )
        # A numpy array becomes a list BEFORE validation: ``float(array)`` raised a
        # TypeError for length > 1, and a length-1 array reached get_config as an ndarray.
        if isinstance(n_init, np.ndarray):
            n_init = n_init.tolist()
        if isinstance(n_final, np.ndarray):
            n_final = n_final.tolist()
        for name, value in (("n_init", n_init), ("n_final", n_final)):
            vals = list(value) if isinstance(value, (list, tuple)) else [value]
            if any(float(v) <= 0 for v in vals):
                raise ValueError(f"{name} values must be positive, got {value!r}.")
        if total_epochs < 1:
            raise ValueError("total_epochs must be >= 1.")
        self.schedule = schedule
        self.n_init = n_init
        self.n_final = n_final
        self.total_epochs = int(total_epochs)
        self.layer_names = layer_names


    @staticmethod
    def _as_list(
        value: Union[float, Sequence[float]], depth: int, name: str
    ) -> List[float]:
        """Broadcast a scalar or validate a per-level sequence.

        :param value: Scalar or sequence of exponents.
        :type value: Union[float, Sequence[float]]
        :param depth: Expected per-level length.
        :type depth: int
        :param name: Argument name for error messages.
        :type name: str
        :return: One float per level.
        :rtype: List[float]
        :raises ValueError: If a sequence does not match ``depth``.
        """
        if isinstance(value, (list, tuple)):
            if len(value) != depth:
                raise ValueError(
                    f"{name} has length {len(value)} but the layer has "
                    f"{depth} levels."
                )
            return [float(v) for v in value]
        return [float(value)] * depth

    def _n_at(self, epoch: int, init: float, final: float) -> float:
        """Interpolate one exponent at ``epoch`` under the schedule.

        :param epoch: Zero-based epoch index.
        :type epoch: int
        :param init: Starting value.
        :type init: float
        :param final: Final value.
        :type final: float
        :return: Interpolated exponent.
        :rtype: float
        """
        return interpolate_schedule(
            self.schedule, init, final, epoch, self.total_epochs
        )

    def _iter_target_layers(self):
        if self.model is None:
            return
        for layer in walk_layers(self.model):
            if self.layer_names is not None and layer.name not in self.layer_names:
                continue
            if not callable(getattr(layer, "set_n_per_level", None)):
                continue
            yield layer

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        # Counted once per run: a layer_names typo is otherwise a silent dead knob.
        if not any(True for _ in self._iter_target_layers()):
            logger.warning(
                "HarmonicExponentAnnealingCallback: no layer matched "
                f"(layer_names={self.layer_names!r}); the schedule changes nothing."
            )

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        touched = 0
        for layer in self._iter_target_layers():
            depth = int(getattr(layer, "num_levels", 1))
            inits = self._as_list(self.n_init, depth, "n_init")
            finals = self._as_list(self.n_final, depth, "n_final")
            layer.set_n_per_level(
                [self._n_at(epoch, a, b) for a, b in zip(inits, finals)]
            )
            touched += 1
        if touched > 0:
            logger.debug(
                f"HarmonicExponentAnnealingCallback: epoch {epoch} applied "
                f"to {touched} layer(s)."
            )

    def get_config(self) -> dict:
        return {
            "schedule": self.schedule,
            "n_init": self.n_init,
            "n_final": self.n_final,
            "total_epochs": self.total_epochs,
            "layer_names": self.layer_names,
        }

# ------------------------------------------------------------------------------
