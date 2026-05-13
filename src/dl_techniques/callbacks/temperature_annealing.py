"""
TemperatureAnnealingCallback — anneals the ``temperature`` parameter on
DARTS-style soft-selection layers across training epochs.

M2 (plan_2026-05-13_3a2f1d23): provides cosine / linear / exponential
schedules that lower the temperature from ``t_init`` to ``t_final`` over
``total_epochs``. Compatible with both the legacy ``softplus_temperature=
False`` parameterization (raw value == temperature) and the canonical
``softplus_temperature=True`` parameterization (raw == log(exp(t) - 1)).

Targeted layers must expose a ``temperature`` attribute (a trainable
``keras.Variable``) and a ``softplus_temperature`` boolean attribute —
matches the contract of ``dl_techniques.layers.logic.*`` operators.
"""

import math
from typing import List, Optional

import keras

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class TemperatureAnnealingCallback(keras.callbacks.Callback):
    """Anneal temperature on soft-selection layers across epochs.

    :param schedule: One of ``'cosine'``, ``'linear'``, ``'exp'``.
    :param t_init: Starting temperature (epoch 0).
    :param t_final: Final temperature (epoch ``total_epochs - 1``).
    :param total_epochs: Schedule length. Must be >= 1.
    :param layer_names: If provided, only these layer names are touched.
        Otherwise all layers with a ``temperature`` Variable are annealed.
    """

    SCHEDULES = frozenset({"cosine", "linear", "exp"})

    def __init__(
        self,
        schedule: str = "cosine",
        t_init: float = 5.0,
        t_final: float = 0.1,
        total_epochs: int = 50,
        layer_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if schedule not in self.SCHEDULES:
            raise ValueError(
                f"schedule must be one of {sorted(self.SCHEDULES)}; got {schedule!r}."
            )
        if t_init <= 0 or t_final <= 0:
            raise ValueError("t_init and t_final must be positive.")
        if total_epochs < 1:
            raise ValueError("total_epochs must be >= 1.")
        self.schedule = schedule
        self.t_init = float(t_init)
        self.t_final = float(t_final)
        self.total_epochs = int(total_epochs)
        self.layer_names = layer_names

    # ------------------------------------------------------------------
    def _temperature_at(self, epoch: int) -> float:
        if self.total_epochs == 1:
            return self.t_final
        frac = min(max(epoch, 0), self.total_epochs - 1) / (self.total_epochs - 1)
        if self.schedule == "linear":
            return self.t_init + (self.t_final - self.t_init) * frac
        if self.schedule == "cosine":
            cos = 0.5 * (1.0 + math.cos(math.pi * frac))
            return self.t_final + (self.t_init - self.t_final) * cos
        # exp: geometric interpolation in log-space.
        log_init = math.log(self.t_init)
        log_final = math.log(self.t_final)
        return math.exp(log_init + (log_final - log_init) * frac)

    def _iter_target_layers(self):
        if self.model is None:
            return
        # Walk all (possibly nested) sub-layers.
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
            if getattr(layer, "temperature", None) is None:
                continue
            yield layer

    def _set_layer_temperature(self, layer, value: float) -> None:
        """Assign the temperature, honoring softplus_temperature mode."""
        soft = bool(getattr(layer, "softplus_temperature", False))
        if soft:
            # softplus_inv(y) = log(exp(y) - 1).
            raw = math.log(math.expm1(value))
        else:
            raw = value
        layer.temperature.assign(raw)

    # ------------------------------------------------------------------
    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        t = self._temperature_at(epoch)
        touched = 0
        for layer in self._iter_target_layers():
            self._set_layer_temperature(layer, t)
            touched += 1
        if touched > 0:
            logger.debug(
                f"TemperatureAnnealingCallback: epoch {epoch} -> t={t:.4f} "
                f"applied to {touched} layer(s)."
            )

    def get_config(self) -> dict:
        return {
            "schedule": self.schedule,
            "t_init": self.t_init,
            "t_final": self.t_final,
            "total_epochs": self.total_epochs,
            "layer_names": self.layer_names,
        }
