"""
Adaptive EMA Slope Filter Model.

This module implements ``AdaptiveEMASlopeFilterModel`` — a ``keras.Model`` (not
a layer) for time-series trading signal generation based on the slope of an
exponential moving average. The model emits a dict containing the EMA, the
slope, three trading signals (``signal_above``, ``signal_below``,
``signal_between``), and optionally a probabilistic ``slope_quantiles`` head.

Threshold parameterization
--------------------------
When ``learnable_thresholds=True`` the upper/lower bounds are parameterized by
two raw scalar weights — ``midpoint_var`` and ``log_half_range_var`` — such
that

    upper = midpoint_var + softplus(log_half_range_var)
    lower = midpoint_var - softplus(log_half_range_var)

This is injective in the raw weights and produces a strictly positive band
width by construction.

Trainable surface
-----------------
* ``learnable_thresholds=False`` and ``quantile_head_config=None`` → zero
  trainable parameters.
* ``learnable_thresholds=True`` → 2 trainable scalars (``midpoint_var``,
  ``log_half_range_var``).
* ``quantile_head_config`` set → a small causal Conv1D featurizer
  (``slope_feature_dim`` filters, kernel ``slope_feature_kernel``, GELU
  activation) precedes the ``QuantileSequenceHead`` so the head sees a learned
  representation of the slope window instead of a scalar.

References
----------
* LeBeau, C. (1992). *Computer Analysis of the Futures Markets.* McGraw-Hill —
  origin of the EMA-slope filtering heuristic used as the rule-based path.
* Bollinger, J. (2001). *Bollinger on Bollinger Bands.* McGraw-Hill — slope as
  a trend/volatility regime indicator.
* Koenker, R. & Bassett, G. (1978). "Regression Quantiles." *Econometrica*
  46(1): 33-50 — quantile-loss formulation used by ``QuantileSequenceHead``.
"""

import math
import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.time_series.ema_layer import ExponentialMovingAverage
from dl_techniques.layers.time_series.quantile_head_variable_io import QuantileSequenceHead
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


def _inverse_softplus(x: float) -> float:
    """Return the inverse of ``softplus(y) = log(1 + exp(y))`` at ``x``.

    ``softplus`` is bijective from ``R → (0, +inf)``. This helper is used to
    initialize ``log_half_range_var`` so that the initial band width matches
    ``(initial_upper_threshold - initial_lower_threshold) / 2``.

    :param x: Target softplus output. Must be strictly positive.
    :raises ValueError: If ``x <= 0``.
    :return: The unique ``y`` such that ``softplus(y) == x``.
    """
    if x <= 0:
        raise ValueError(f"inverse_softplus requires x > 0, got {x}")
    return math.log(math.expm1(x))


@keras.saving.register_keras_serializable()
class AdaptiveEMASlopeFilterModel(keras.Model):
    """
    ``keras.Model`` for adaptive EMA slope filtering with optional learnable
    thresholds and an optional probabilistic slope-quantile head.

    Architecture overview
    ---------------------
    .. code-block:: text

        Input: price = (batch, time)  or  (batch, time, features)
                            │
                            ▼
               ┌────────────────────────────┐
               │  ExponentialMovingAverage  │
               └─────────────┬──────────────┘
                             │
                             ▼  EMA
                  slope_t = EMA_t − EMA_{t-L}
                             │
            ┌────────────────┴────────────────┐
            │                                 │
            ▼                                 ▼
       (threshold path)               (optional quantile path)
       upper = m + softplus(r)        Conv1D causal featurizer
       lower = m - softplus(r)               │
            │                                 ▼
            ▼                          QuantileSequenceHead → (B,T,K)
       signal_{above,below,between}

    Trainable surface
    -----------------
    * ``learnable_thresholds=False`` + ``quantile_head_config=None`` ⇒ **zero**
      trainable parameters. The model is a pure rule.
    * ``learnable_thresholds=True`` ⇒ **2** trainable scalars (``midpoint_var``,
      ``log_half_range_var``).
    * ``quantile_head_config`` set ⇒ additional parameters from the Conv1D
      featurizer + the ``QuantileSequenceHead`` projection.

    Signal semantics
    ----------------
    * ``training=True``: soft signals via sigmoid, all in ``[0, 1]``.
    * ``training=False`` (inference): hard 0/1 signals, partition exact
      (``above + below + between == 1``).

    Hard- vs soft-mode mutual exclusivity
    -------------------------------------
    The two threshold modes — rule-based hard thresholds (``output_mode``
    style ``above``/``below``/``between`` signals) and the learned soft
    ``slope_quantiles`` head — are produced from the *same* slope tensor and
    are mutually independent at the dict level (both can be active in one
    forward pass). However they answer different questions and should not be
    used together in a downstream classifier without an explicit fusion
    policy: the hard signals are deterministic membership indicators of a
    threshold band; ``slope_quantiles`` is a distributional forecast of the
    slope itself. Treat them as alternatives, not as a single conditioned
    output.

    :param ema_period: Period for the underlying ``ExponentialMovingAverage``.
        Must be ``>= 1``.
    :param lookback_period: Number of bars back used in the slope:
        ``slope_t = EMA_t - EMA_{t-L}``. Must be ``>= 1``.
    :param initial_upper_threshold: Initial upper slope threshold. Must be
        strictly greater than ``initial_lower_threshold``.
    :param initial_lower_threshold: Initial lower slope threshold.
    :param learnable_thresholds: If True, the two raw scalars
        (``midpoint_var``, ``log_half_range_var``) are trainable.
    :param adjust_ema: Pass-through to the EMA layer's ``adjust`` flag.
    :param slope_softness: Temperature ``T`` for the sigmoid-based soft signals.
        Smaller → harder, larger → flatter. Must be ``> 0``.
    :param quantile_head_config: Optional dict configuring a
        ``QuantileSequenceHead``. Required key: ``num_quantiles``. Optional:
        ``dropout_rate``, ``enforce_monotonicity``, ``use_bias``.
    :param slope_feature_dim: Number of filters in the causal Conv1D
        featurizer preceding the quantile head. Ignored when the head is
        disabled. Must be ``> 0``.
    :param slope_feature_kernel: Kernel size of the causal Conv1D featurizer.
        Ignored when the head is disabled. Must be ``> 0``.
    """

    def __init__(
        self,
        ema_period: int = 25,
        lookback_period: int = 25,
        initial_upper_threshold: float = 15.0,
        initial_lower_threshold: float = -15.0,
        learnable_thresholds: bool = False,
        adjust_ema: bool = True,
        slope_softness: float = 1.0,
        quantile_head_config: Optional[Dict[str, Any]] = None,
        slope_feature_dim: int = 16,
        slope_feature_kernel: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if ema_period < 1:
            raise ValueError(f"ema_period must be >= 1, got {ema_period}")
        if lookback_period < 1:
            raise ValueError(f"lookback_period must be >= 1, got {lookback_period}")
        if initial_upper_threshold <= initial_lower_threshold:
            raise ValueError(
                f"initial_upper_threshold ({initial_upper_threshold}) must be "
                f"strictly greater than initial_lower_threshold "
                f"({initial_lower_threshold})"
            )
        if slope_softness <= 0:
            raise ValueError(
                f"slope_softness must be > 0, got {slope_softness}"
            )
        if slope_feature_dim <= 0:
            raise ValueError(
                f"slope_feature_dim must be > 0, got {slope_feature_dim}"
            )
        if slope_feature_kernel <= 0:
            raise ValueError(
                f"slope_feature_kernel must be > 0, got {slope_feature_kernel}"
            )

        # I-19: surface the easy-to-misconfigure case where the user asked
        # for learnable thresholds but did NOT attach a quantile head — the
        # model then has only 2 trainable scalars and no head to project
        # them through, which usually indicates a misconfigured experiment.
        if learnable_thresholds and quantile_head_config is None:
            logger.warning(
                "AdaptiveEMASlopeFilterModel: learnable_thresholds=True "
                "with quantile_head_config=None gives a model with only 2 "
                "trainable scalars (midpoint_var, log_half_range_var). "
                "If you intended to learn a distribution over slopes, pass "
                "quantile_head_config={'num_quantiles': K}."
            )

        self.ema_period = ema_period
        self.lookback_period = lookback_period
        self.initial_upper_threshold = initial_upper_threshold
        self.initial_lower_threshold = initial_lower_threshold
        self.learnable_thresholds = learnable_thresholds
        self.adjust_ema = adjust_ema
        self.slope_softness = float(slope_softness)
        self.quantile_head_config = quantile_head_config
        self.slope_feature_dim = slope_feature_dim
        self.slope_feature_kernel = slope_feature_kernel

        # EMA layer
        self.ema_layer = ExponentialMovingAverage(
            period=ema_period,
            adjust=adjust_ema,
            name="ema",
        )

        # Optional quantile head with a learned causal Conv1D featurizer
        # (I-2b: the head is no longer fed a raw scalar slope).
        if quantile_head_config is not None:
            if "num_quantiles" not in quantile_head_config:
                raise ValueError(
                    "quantile_head_config must contain 'num_quantiles' key"
                )

            self.slope_featurizer = keras.layers.Conv1D(
                filters=slope_feature_dim,
                kernel_size=slope_feature_kernel,
                padding="causal",
                activation="gelu",
                name="slope_featurizer",
            )
            self.quantile_head = QuantileSequenceHead(
                num_quantiles=quantile_head_config["num_quantiles"],
                dropout_rate=quantile_head_config.get("dropout_rate", 0.1),
                enforce_monotonicity=quantile_head_config.get(
                    "enforce_monotonicity", True
                ),
                use_bias=quantile_head_config.get("use_bias", True),
                name="slope_quantile_head",
            )
        else:
            self.slope_featurizer = None
            self.quantile_head = None

        self.midpoint_var: Optional[keras.Variable] = None
        self.log_half_range_var: Optional[keras.Variable] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the two threshold scalars; sub-layers self-build on first call."""
        midpoint_init = (
            self.initial_upper_threshold + self.initial_lower_threshold
        ) / 2.0
        half_range_init = (
            self.initial_upper_threshold - self.initial_lower_threshold
        ) / 2.0
        log_half_range_init = _inverse_softplus(half_range_init)

        # I-10: threshold weights stored in float32 even under mixed precision.
        self.midpoint_var = self.add_weight(
            name="midpoint_var",
            shape=(),
            initializer=keras.initializers.Constant(midpoint_init),
            trainable=self.learnable_thresholds,
            dtype="float32",
        )
        self.log_half_range_var = self.add_weight(
            name="log_half_range_var",
            shape=(),
            initializer=keras.initializers.Constant(log_half_range_init),
            trainable=self.learnable_thresholds,
            dtype="float32",
        )

        # I-9: do NOT manually build sub-layers — Keras builds them lazily on
        # first call. ``super().build`` marks this model as built.
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Compute EMA, slope, threshold signals, and optionally slope quantiles."""
        # Compute EMA.
        ema = self.ema_layer(inputs)

        # I-8: static-shape slope shift. ``...`` ellipsis handles both
        # ``(B, T)`` and ``(B, T, F)`` cases.
        L = self.lookback_period
        ema_lagged = ops.concatenate(
            [ops.zeros_like(ema[:, :L, ...]), ema[:, :-L, ...]],
            axis=1,
        )
        slope = ema - ema_lagged

        # I-5: upper/lower from midpoint ± softplus(log_half_range). Cast to
        # the slope's compute dtype (I-10 runtime cast).
        midpoint = ops.cast(self.midpoint_var, slope.dtype)
        half_range = ops.softplus(ops.cast(self.log_half_range_var, slope.dtype))
        upper = midpoint + half_range
        lower = midpoint - half_range

        # I-6, I-7: sigmoid-based soft signals, partition relaxed to soft
        # membership functions. Temperature ``T`` (``slope_softness``) is a
        # configurable scalar > 0.
        T = ops.cast(self.slope_softness, slope.dtype)
        if training:
            signal_above = ops.sigmoid((slope - upper) / T)
            signal_below = ops.sigmoid((lower - slope) / T)
            signal_between = (
                ops.sigmoid((slope - lower) / T)
                * ops.sigmoid((upper - slope) / T)
            )
        else:
            # Hard inference signals — exact partition.
            signal_above = ops.cast(slope > upper, dtype=slope.dtype)
            signal_below = ops.cast(slope < lower, dtype=slope.dtype)
            signal_between = ops.cast(
                ops.logical_and(slope >= lower, slope <= upper),
                dtype=slope.dtype,
            )

        outputs: Dict[str, keras.KerasTensor] = {
            "ema": ema,
            "slope": slope,
            "signal_above": signal_above,
            "signal_below": signal_below,
            "signal_between": signal_between,
            "upper_threshold": upper,
            "lower_threshold": lower,
        }

        # Optional quantile head — Conv1D causal featurization first (I-2b).
        if self.quantile_head is not None:
            ndim = len(inputs.shape)
            # DECISION plan_2026-05-12_5f0e087c/D-002:
            # Reject multi-feature inputs when a quantile head is attached
            # rather than silently mixing features through the Conv1D
            # featurizer. Documented as L-7 in README §11.
            if (
                ndim == 3
                and inputs.shape[-1] is not None
                and inputs.shape[-1] > 1
            ):
                raise ValueError(
                    "AdaptiveEMASlopeFilterModel with quantile_head_config "
                    "set does not support multi-feature inputs "
                    f"(got inputs.shape[-1]={inputs.shape[-1]}). "
                    "Either drop the quantile head or reduce inputs to a "
                    "single feature channel."
                )
            if ndim == 2:
                slope_3d = ops.expand_dims(slope, axis=-1)
            else:
                slope_3d = slope
            slope_features = self.slope_featurizer(slope_3d, training=training)
            outputs["slope_quantiles"] = self.quantile_head(
                slope_features, training=training
            )

        return outputs

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Static-shape map of the dict output (I-14).

        :param input_shape: Shape of ``inputs``, either ``(B, T)`` or
            ``(B, T, F)``.
        :return: Dict mirroring the keys returned by :meth:`call`.
            ``ema``, ``slope``, and the three signal tensors take the
            full input shape. ``upper_threshold`` and ``lower_threshold``
            are scalars. ``slope_quantiles`` (if the head is enabled) is
            ``(B, T, K)`` where ``K = num_quantiles``.
        """
        shapes: Dict[str, Tuple[Optional[int], ...]] = {
            "ema": tuple(input_shape),
            "slope": tuple(input_shape),
            "signal_above": tuple(input_shape),
            "signal_below": tuple(input_shape),
            "signal_between": tuple(input_shape),
            "upper_threshold": (),
            "lower_threshold": (),
        }
        if self.quantile_head_config is not None:
            B = input_shape[0]
            T = input_shape[1] if len(input_shape) >= 2 else None
            K = self.quantile_head_config["num_quantiles"]
            shapes["slope_quantiles"] = (B, T, K)
        return shapes

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            "ema_period": self.ema_period,
            "lookback_period": self.lookback_period,
            "initial_upper_threshold": self.initial_upper_threshold,
            "initial_lower_threshold": self.initial_lower_threshold,
            "learnable_thresholds": self.learnable_thresholds,
            "adjust_ema": self.adjust_ema,
            "slope_softness": self.slope_softness,
            "quantile_head_config": self.quantile_head_config,
            "slope_feature_dim": self.slope_feature_dim,
            "slope_feature_kernel": self.slope_feature_kernel,
        })
        return config


# ---------------------------------------------------------------------


def create_adaptive_ema_slope_filter(
    ema_period: int = 25,
    lookback_period: int = 25,
    initial_upper_threshold: float = 15.0,
    initial_lower_threshold: float = -15.0,
    learnable_thresholds: bool = False,
    adjust_ema: bool = True,
    slope_softness: float = 1.0,
    quantile_head_config: Optional[Dict[str, Any]] = None,
    slope_feature_dim: int = 16,
    slope_feature_kernel: int = 5,
    **kwargs: Any,
) -> AdaptiveEMASlopeFilterModel:
    """Factory function for :class:`AdaptiveEMASlopeFilterModel`.

    Mirrors the factory style used by other ``dl_techniques`` model packages
    (e.g. ``create_vit``, ``create_tirex_by_variant``). Defaults match the
    model constructor. The returned instance is unbuilt — Keras builds it
    lazily on the first call. ``**kwargs`` are forwarded to ``keras.Model``
    (e.g. ``name``).

    :return: An unbuilt :class:`AdaptiveEMASlopeFilterModel` instance.
    """
    return AdaptiveEMASlopeFilterModel(
        ema_period=ema_period,
        lookback_period=lookback_period,
        initial_upper_threshold=initial_upper_threshold,
        initial_lower_threshold=initial_lower_threshold,
        learnable_thresholds=learnable_thresholds,
        adjust_ema=adjust_ema,
        slope_softness=slope_softness,
        quantile_head_config=quantile_head_config,
        slope_feature_dim=slope_feature_dim,
        slope_feature_kernel=slope_feature_kernel,
        **kwargs,
    )

# ---------------------------------------------------------------------
