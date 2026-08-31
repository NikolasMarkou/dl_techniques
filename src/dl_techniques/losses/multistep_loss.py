"""
Multistep (h-steps-ahead) loss functions for forecasting models.

Conventional estimation minimises the ONE-step-ahead error. That is rarely the
error anybody is paid for: supply chains order against a lead time, budgets are
planned by quarter, and a model tuned on `t+1` is not the model that minimises
the decision-relevant loss at `t+h`. Multistep losses close that gap by scoring
the errors made `h` steps ahead of every in-sample forecast origin.

This module implements the four estimators from Svetunkov's ADAM framework. Let
``e_{t+j|t}`` be the j-step-ahead error produced from origin ``t``. Averaging
over origins (a training minibatch IS a sample of forecast origins):

    MSEh  = mean_t e_{t+h|t}^2                        # step h only
    TMSE  = sum_{j=1..h} mean_t e_{t+j|t}^2           # trace
    GTMSE = sum_{j=1..h} log( mean_t e_{t+j|t}^2 )    # geometric trace
    MSCE  = mean_t ( sum_{j=1..h} e_{t+j|t} )^2       # cumulative error

Choosing between them
---------------------
    MSEh   Strongest shrinkage, simplest interpretation. See the WARNING below
           before using it on a direct multi-horizon model.
    TMSE   Balances every horizon, but the arithmetic sum is dominated by the
           longer horizons, whose errors are simply larger.
    GTMSE  The log puts every horizon's MSE on a comparable scale, so short and
           long horizons are minimised with similar force. Mildest shrinkage,
           least biased on small samples. A good default.
    MSCE   Scores the CUMULATIVE error over the lead time, which is the quantity
           an inventory decision actually depends on.

WARNING -- MSEh on a direct multi-horizon model
-----------------------------------------------
In ADAM these losses constrain a RECURSIVE model, so the error at step h still
moves every shared parameter. Most forecasters in this repository are DIRECT:
they emit the whole ``[B, H, F]`` block from per-step output heads in a single
pass. Under ``aggregation="mseh"`` such a model receives EXACTLY ZERO gradient
on every horizon column other than ``h`` -- those heads never train at all.
There is no shape symptom and no warning at runtime; the model simply ships with
untrained outputs. Use ``"mseh"`` only when step ``h`` is the only step that will
ever be read. This is pinned by
``tests/test_losses/test_the_mseh_starves_other_horizons.py``.

A note on shrinkage
-------------------
The theoretical result motivating these losses -- that minimising h-step errors
SHRINKS a model's smoothing parameters toward zero, making it less reactive to
noise -- is proven for PURE ADDITIVE ETS and ARIMA models, and it arises through
RECURSIVE error accumulation. On a direct multi-horizon model these losses are a
principled horizon-axis reweighting that aligns the objective with the decision,
but there is no smoothing parameter to shrink and no shrinkage should be claimed.

References
----------
-   Svetunkov, I., Kourentzes, N., & Killick, R. (2023). "Multi-step Estimators
    and Shrinkage Effect in Time Series Models". *Computational Statistics*.
    DOI: 10.1007/s00180-023-01377-x
-   Svetunkov, I. (2023). *Forecasting and Analytics with the Augmented Dynamic
    Adaptive Model (ADAM)*, section 11.3 "Multistep losses".
    https://openforecast.org/adam/multistepLosses.html
"""

import keras
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: The four ADAM multistep aggregations, in the order the monograph presents them.
MULTISTEP_AGGREGATIONS = ("mseh", "tmse", "gtmse", "msce")

#: Keyword arguments `create_multistep_loss` will forward. Anything else RAISES,
#: per the house factory rule: never filter-and-drop a caller's keyword.
_FACTORY_KEYS = frozenset(
    {"h", "error_power", "epsilon", "name", "reduction", "dtype"}
)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.multistep_loss")
class MultistepLoss(keras.losses.Loss):
    """Multistep (h-steps-ahead) forecasting loss.

    Implements the ADAM multistep estimators ``MSEh``, ``TMSE``, ``GTMSE`` and
    ``MSCE`` over the horizon axis of a multi-horizon forecast. See the module
    docstring for the formulas, the selection guidance, and the ``mseh``
    zero-gradient warning.

    Shape contract:
        Axis 1 is the horizon, matching ``MASELoss`` and the ``[B, H, F]``
        ``Forecast.point`` contract of ``models/time_series/``. Accepts
        ``(batch, horizon)`` and ``(batch, horizon, features)``; any trailing
        feature axes are averaged. A grid of forecast origins ``[B, T, H]``
        should be flattened by the caller to ``[B * T, H]``.

        Quantile outputs (``[B, H, F, Q]``) are NOT supported: a pinball
        objective is a different loss, not a reduction of this one.

    Batch sensitivity:
        ``gtmse`` takes the logarithm of a BATCH-level mean error, so its value
        and its gradient depend on the batch size. This is inherent to the
        estimator (the batch stands in for the sample of forecast origins), not
        an implementation artifact. ``mseh``, ``tmse`` and ``msce`` decompose
        exactly per sample and carry no such dependence.

    Args:
        aggregation: One of ``"mseh"``, ``"tmse"``, ``"gtmse"``, ``"msce"``.
            Defaults to ``"tmse"``.
        h: Horizon to evaluate over. For ``"tmse"``, ``"gtmse"`` and ``"msce"``
            the loss uses steps ``1..h``; for ``"mseh"`` it uses step ``h``
            ALONE. ``None`` (the default) means the full horizon axis, i.e.
            ``h = H`` for the aggregations and the LAST step for ``"mseh"``.
        error_power: Exponent applied to the absolute error. ``2.0`` (default)
            gives the MSE family above; ``1.0`` gives ADAM's MAE analogues
            (``MAEh`` / ``TMAE`` / ``MACE``), which the monograph defines by
            analogy. Only ``2.0`` is covered by the numeric reference tests.
        epsilon: Floor applied to the per-step mean error before the logarithm
            in ``"gtmse"``, guarding ``log(0)`` on a perfectly fitted step.
            Unused by the other three aggregations. Defaults to ``1e-8``.
        name: Name of the loss. Defaults to ``"multistep_loss"``.
        **kwargs: Forwarded to ``keras.losses.Loss`` (``reduction``, ``dtype``).

    Raises:
        ValueError: If ``aggregation`` is not one of the four names, if ``h`` is
            not ``None`` or a positive integer, or if ``error_power`` or
            ``epsilon`` is not strictly positive.

    Example:
        >>> import keras
        >>> from dl_techniques.losses.multistep_loss import MultistepLoss
        >>> # Optimise the whole 12-step horizon, all steps weighted comparably.
        >>> model.compile(optimizer="adam", loss=MultistepLoss("gtmse", h=12))
        >>> # Optimise the cumulative error over a 4-week lead time.
        >>> model.compile(optimizer="adam", loss=MultistepLoss("msce", h=4))
    """

    def __init__(
        self,
        aggregation: str = "tmse",
        h: Optional[int] = None,
        error_power: float = 2.0,
        epsilon: float = 1e-8,
        name: str = "multistep_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        if aggregation not in MULTISTEP_AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation {aggregation!r}. "
                f"Expected one of {sorted(MULTISTEP_AGGREGATIONS)}."
            )
        if h is not None and (not isinstance(h, int) or isinstance(h, bool) or h < 1):
            raise ValueError(f"h must be None or a positive integer, got {h!r}.")
        if error_power <= 0.0:
            raise ValueError(f"error_power must be > 0, got {error_power}.")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}.")

        self.aggregation = aggregation
        self.h = h
        self.error_power = float(error_power)
        self.epsilon = float(epsilon)

    # -----------------------------------------------------------------

    def _magnitude(self, error: keras.KerasTensor) -> keras.KerasTensor:
        """Raise the absolute error to ``error_power``.

        ``error_power == 2.0`` is special-cased to ``square`` so the MSE family
        is computed exactly rather than through ``exp(2 * log|e|)``, which is
        neither exact nor defined at ``e == 0``.

        Args:
            error: Error tensor of any shape.

        Returns:
            The element-wise error magnitude, same shape as ``error``.
        """
        if self.error_power == 2.0:
            return keras.ops.square(error)
        return keras.ops.power(keras.ops.abs(error), self.error_power)

    @staticmethod
    def _mean_over_features(values: keras.KerasTensor) -> keras.KerasTensor:
        """Average every axis after the horizon axis, keeping ``(batch, horizon)``.

        Args:
            values: Tensor of shape ``(batch, horizon, ...)``.

        Returns:
            Tensor of shape ``(batch, horizon)``.
        """
        rank = len(values.shape)
        if rank == 2:
            return values
        return keras.ops.mean(values, axis=tuple(range(2, rank)))

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute the multistep loss.

        Args:
            y_true: Ground truth, shape ``(batch, horizon)`` or
                ``(batch, horizon, features)``.
            y_pred: Forecast, same shape as ``y_true``.

        Returns:
            Per-sample loss of shape ``(batch,)``. Reducing this vector with the
            mean recovers the exact estimator -- including for ``"gtmse"``, whose
            per-sample form is a surrogate constructed so that its batch mean and
            its gradient both match the exact batch-global expression.

        Raises:
            ValueError: If the inputs are rank < 2, or if ``h`` exceeds the
                statically known horizon length.
        """
        y_true = keras.ops.cast(y_true, y_pred.dtype)

        rank = len(y_pred.shape)
        if rank < 2:
            raise ValueError(
                "MultistepLoss needs a horizon axis: expected inputs of shape "
                f"(batch, horizon[, features]), got rank {rank} with shape "
                f"{y_pred.shape}."
            )

        horizon = y_pred.shape[1]
        if self.h is not None and horizon is not None and self.h > horizon:
            raise ValueError(
                f"h={self.h} exceeds the horizon axis length {horizon} of "
                f"y_pred with shape {y_pred.shape}."
            )

        # Signed error. Only MSCE depends on the sign surviving to the
        # aggregation, but keeping one definition avoids two error tensors.
        error = y_pred - y_true

        if self.aggregation == "msce":
            return self._msce(error)

        # Per-step, per-sample error magnitude: (batch, horizon).
        per_step = self._mean_over_features(self._magnitude(error))
        if self.h is not None:
            per_step = per_step[:, : self.h] if self.aggregation != "mseh" else per_step

        if self.aggregation == "mseh":
            index = (self.h - 1) if self.h is not None else -1
            return per_step[:, index]

        if self.aggregation == "tmse":
            return keras.ops.sum(per_step, axis=1)

        return self._gtmse(per_step)

    def _msce(self, error: keras.KerasTensor) -> keras.KerasTensor:
        """Mean squared CUMULATIVE error, per sample.

        The errors are summed along the horizon FIRST and the magnitude is taken
        afterwards, so over- and under-forecasts cancel exactly as they do in a
        stock position accumulated over a lead time. Taking the magnitude first
        would silently turn this into TMSE.

        Args:
            error: Signed error of shape ``(batch, horizon, ...)``.

        Returns:
            Tensor of shape ``(batch,)``.
        """
        if self.h is not None:
            error = error[:, : self.h]
        cumulative = keras.ops.sum(error, axis=1)
        magnitude = self._magnitude(cumulative)
        rank = len(magnitude.shape)
        if rank == 1:
            return magnitude
        return keras.ops.mean(magnitude, axis=tuple(range(1, rank)))

    def _gtmse(self, per_step: keras.KerasTensor) -> keras.KerasTensor:
        """Geometric trace MSE, as an exact per-sample surrogate.

        ``GTMSE = sum_j log(M_j)`` where ``M_j = mean_i e_ij^2`` is a BATCH
        statistic, so it has no per-sample decomposition. Returning the scalar
        broadcast across the batch is not an option: Keras multiplies
        ``values * sample_weight`` BEFORE reducing, so a constant vector charges
        every row the batch aggregate and makes ``reduction=`` a dead knob.

        This returns the first-order expansion of ``log`` about the detached
        batch mean::

            L_i = sum_j [ log(sg(M_j)) + e_ij^2 / sg(M_j) - 1 ]

        Two exact identities hold, and both are pinned by
        ``tests/test_losses/test_the_gtmse_surrogate_matches_the_exact_form.py``:

        -   VALUE: ``mean_i L_i == sum_j log(M_j)``, because
            ``mean_i e_ij^2 / M_j == 1`` by construction of ``M_j``.
        -   GRADIENT: ``d/dtheta mean_i L_i == sum_j (1/M_j) dM_j/dtheta``,
            which is exactly ``d/dtheta sum_j log(M_j)``.

        The detachment's scope is narrower than it looks, and was MEASURED
        rather than assumed: because ``mean_i e_ij^2 / M_j == 1`` identically,
        the attached form collapses to the same function, so removing
        ``stop_gradient`` changes neither the value nor the unweighted gradient
        (1.19e-07, float32 noise). It matters only once ``sample_weight`` breaks
        that cancellation -- there the two differ by 1.49e-01, and only the
        detached form still reads as "GTMSE over the weighted rows, linearised
        at the full-batch ``M_j``".

        Do NOT "simplify" this to ``sum_j log(e_ij^2 + eps)``. That decomposes
        trivially but is a DIFFERENT objective -- the log of a per-sample error
        rather than of a mean, separated by a Jensen gap -- and it makes
        ``epsilon`` load-bearing wherever a sample is fitted exactly.

        Args:
            per_step: Per-sample, per-step error magnitude, shape
                ``(batch, horizon)``.

        Returns:
            Tensor of shape ``(batch,)``.
        """
        batch_mean = keras.ops.mean(per_step, axis=0, keepdims=True)
        batch_mean = keras.ops.maximum(batch_mean, self.epsilon)
        batch_mean = keras.ops.stop_gradient(batch_mean)

        terms = keras.ops.log(batch_mean) + (per_step / batch_mean) - 1.0
        return keras.ops.sum(terms, axis=1)

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration for serialization.

        Returns:
            The base ``keras.losses.Loss`` config plus this loss's four knobs.
        """
        config = super().get_config()
        config.update(
            {
                "aggregation": self.aggregation,
                "h": self.h,
                "error_power": self.error_power,
                "epsilon": self.epsilon,
            }
        )
        return config


# ---------------------------------------------------------------------


def create_multistep_loss(name: str, **kwargs: Any) -> MultistepLoss:
    """Build a :class:`MultistepLoss` by aggregation name.

    Unknown names AND unknown keyword arguments both raise ``ValueError``,
    per the house factory rule in ``src/dl_techniques/CLAUDE.md``: a factory
    that silently drops a keyword it does not recognise has previously shipped
    dead knobs repo-wide (``dropout=`` against a ``dropout_rate`` parameter,
    ``qkv_bias=`` against ``use_bias``), each invisible at every shape check.

    Args:
        name: One of ``"mseh"``, ``"tmse"``, ``"gtmse"``, ``"msce"``.
            Case-insensitive.
        **kwargs: Any of ``h``, ``error_power``, ``epsilon``, ``name``
            (the loss's own name), ``reduction``, ``dtype``.

    Returns:
        A configured :class:`MultistepLoss`.

    Raises:
        ValueError: If ``name`` is not a known aggregation, or if any keyword
            argument is not one this factory forwards.

    Example:
        >>> loss = create_multistep_loss("gtmse", h=12)
        >>> create_multistep_loss("mse")
        Traceback (most recent call last):
        ValueError: Unknown multistep loss 'mse'. ...
    """
    key = name.lower() if isinstance(name, str) else name
    if key not in MULTISTEP_AGGREGATIONS:
        raise ValueError(
            f"Unknown multistep loss {name!r}. "
            f"Expected one of {sorted(MULTISTEP_AGGREGATIONS)}."
        )

    unknown = sorted(set(kwargs) - _FACTORY_KEYS)
    if unknown:
        raise ValueError(
            f"create_multistep_loss got unexpected keyword argument(s) {unknown}. "
            f"Accepted keywords are {sorted(_FACTORY_KEYS)}."
        )

    return MultistepLoss(aggregation=key, **kwargs)

# ---------------------------------------------------------------------
