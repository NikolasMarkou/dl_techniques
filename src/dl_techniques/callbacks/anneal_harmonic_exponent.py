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

import math
from typing import List, Optional, Sequence, Union

import keras

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.callbacks.harmonic_exponent_annealing")
class HarmonicExponentAnnealingCallback(keras.callbacks.Callback):
    """Anneal per-level harmonic exponents across epochs.

    :param schedule: One of ``'linear'``, ``'cosine'``, ``'exp'``.
    :param n_init: Starting exponent(s) at epoch 0: scalar or per-level.
    :param n_final: Final exponent(s): scalar or per-level.
    :param total_epochs: Schedule length. Must be >= 1.
    :param layer_names: If provided, only these layer names are touched.
        Otherwise all layers exposing ``set_n_per_level`` are annealed.
    """

    SCHEDULES = frozenset({"linear", "cosine", "exp"})

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

    # ------------------------------------------------------------------
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
        if self.total_epochs == 1:
            return final
        frac = min(max(epoch, 0), self.total_epochs - 1) / (self.total_epochs - 1)
        if self.schedule == "linear":
            return init + (final - init) * frac
        if self.schedule == "cosine":
            cos = 0.5 * (1.0 + math.cos(math.pi * frac))
            return final + (init - final) * cos
        log_init = math.log(init)
        log_final = math.log(final)
        return math.exp(log_init + (log_final - log_init) * frac)

    def _iter_target_layers(self):
        if self.model is None:
            return
        seen = set()
        stack = list(self.model.layers)
        while stack:
            layer = stack.pop()
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            if hasattr(layer, "_layers"):
                stack.extend(getattr(layer, "_layers"))
            elif hasattr(layer, "layers"):
                stack.extend(getattr(layer, "layers"))
            if self.layer_names is not None and layer.name not in self.layer_names:
                continue
            if not callable(getattr(layer, "set_n_per_level", None)):
                continue
            yield layer

    # ------------------------------------------------------------------
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
