"""
AdamW with a per-variable learning-rate multiplier.

Keras has no parameter groups, so a recipe such as Depth Anything's (arXiv
2401.10891, section 4.1: "the base learning rate of the pre-trained encoder is
set as 5e-6, while the randomly initialized decoder uses a 10x larger learning
rate") cannot be written with one stock optimizer, and two optimizers would
need a custom ``train_step``. Keras 3's ``Adam.update_step(gradient, variable,
learning_rate)`` runs once per variable, so scaling ``learning_rate`` there
gives each variable its own rate under stock ``compile``/``fit``.

Update rule, for a variable whose multiplier is ``k`` (``k = 1`` for every
variable the table does not match):

    p -= (k * lr) * weight_decay * p                     # decoupled weight decay
    p -= (k * lr) * m_hat / (sqrt(v_hat) + epsilon)      # Adam step

which is PyTorch's AdamW over a parameter group whose ``lr`` is ``k * lr``: the
group rate scales the weight decay as well as the step.

The multiplier table maps a variable path, or a path prefix that ends at a
``/`` boundary, to a positive multiplier; the longest matching key wins.
Variable paths start with the model's own name (``depth_anything/...``, or
``depth_anything_1/...`` for a second instance), so a caller usually builds
the table from ``layer.trainable_variables`` paths rather than typing them.

Weight decay is applied per variable with a plain ``assign`` (the generic
``BaseOptimizer`` form), not through the TensorFlow backend's
``tf.distribute`` merge call: this optimizer is for single-device training.
"""

import math
import numbers
from typing import Any, Dict, List, Mapping, Optional

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.optimization.lr_multiplier_adamw")
class LRMultiplierAdamW(keras.optimizers.AdamW):
    """AdamW whose learning rate is scaled per variable by a path-keyed table.

    Interface contract: ``lr_multipliers`` maps a variable path, or a path
    prefix ending at a ``/`` boundary, to a finite multiplier ``> 0``; the
    longest matching key wins and an unmatched variable gets ``1.0``. Every
    other argument is :class:`keras.optimizers.AdamW`'s. The table is resolved
    once per variable in :meth:`build`; :meth:`multiplier_for` answers for any
    path. ``get_config`` stores the table, so ``from_config`` rebuilds the same
    optimizer. Raises ``ValueError`` at construction for an empty or non-string
    key, or a multiplier that is not a finite number ``> 0``.

    Args:
        lr_multipliers: Mapping of variable path (or ``/``-bounded path
            prefix) to learning-rate multiplier. ``None`` or empty means every
            variable trains at the base rate, exactly as ``AdamW``.
        **kwargs: Passed to :class:`keras.optimizers.AdamW`.

    Example:
        >>> decoder_paths = {v.path: 10.0 for v in model.decoder.trainable_variables}
        >>> optimizer = LRMultiplierAdamW(lr_multipliers=decoder_paths, learning_rate=5e-6)
    """

    def __init__(
            self,
            lr_multipliers: Optional[Mapping[str, float]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        table: Dict[str, float] = {}
        for key, value in dict(lr_multipliers or {}).items():
            if not isinstance(key, str) or not key.strip("/"):
                raise ValueError(
                    f"lr_multipliers keys must be non-empty variable paths, got {key!r}")
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(
                    f"lr_multipliers[{key!r}] must be a number, got {value!r}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"lr_multipliers[{key!r}] must be finite and > 0, got {value!r}")
            table[key.rstrip("/")] = float(value)
        self.lr_multipliers = table
        self._variable_multipliers: List[float] = []

    def multiplier_for(self, path: str) -> float:
        """The multiplier a variable with this path trains at.

        Args:
            path: A variable path, e.g. ``"depth_anything/depth_head/refinenet1/out_conv/kernel"``.

        Returns:
            The value of the longest key equal to ``path`` or a ``/``-bounded
            prefix of it, else ``1.0``.
        """
        best_key = None
        for key in self.lr_multipliers:
            if path == key or path.startswith(key + "/"):
                if best_key is None or len(key) > len(best_key):
                    best_key = key
        return 1.0 if best_key is None else self.lr_multipliers[best_key]

    def build(self, var_list: List[keras.Variable]) -> None:
        """Create the Adam slots and resolve each variable's multiplier.

        Args:
            var_list: The trainable variables, in the order the optimizer
                indexes them.
        """
        if self.built:
            return
        super().build(var_list)
        self._variable_multipliers = [self.multiplier_for(v.path) for v in var_list]

    def _multiplier(self, variable: Any) -> float:
        # The TensorFlow backend hands update_step the raw tf.Variable, which
        # has no Keras path; the optimizer's own index maps either form.
        return self._variable_multipliers[self._get_variable_index(variable)]

    def update_step(self, gradient: Any, variable: Any, learning_rate: Any) -> None:
        """Adam's step at ``multiplier * learning_rate`` for this variable."""
        k = self._multiplier(variable)
        if k != 1.0:
            learning_rate = keras.ops.multiply(
                keras.ops.cast(learning_rate, variable.dtype), k)
        super().update_step(gradient, variable, learning_rate)

    def _apply_weight_decay(self, variables: List[keras.Variable]) -> None:
        """Decoupled weight decay at each variable's own rate, ``k * lr * wd``."""
        if self.weight_decay is None:
            return
        for variable in variables:
            if self._use_weight_decay(variable):
                lr = keras.ops.cast(self.learning_rate, variable.dtype)
                wd = keras.ops.cast(self.weight_decay, variable.dtype)
                k = self._multiplier(variable)
                variable.assign(variable - variable * wd * (lr * k))

    def get_config(self) -> Dict[str, Any]:
        """The AdamW config plus the multiplier table."""
        config = super().get_config()
        config["lr_multipliers"] = dict(self.lr_multipliers)
        return config

# ---------------------------------------------------------------------
