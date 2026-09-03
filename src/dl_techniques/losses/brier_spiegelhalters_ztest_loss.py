"""
Calibration losses and metrics: the Brier score and Spiegelhalter's Z-test.

Calibration is the statistical consistency between a model's predicted
probabilities and the long-run frequency of the outcomes they predict. A
well-calibrated model that says "40%" is right 40% of the time. This module
provides a proper scoring rule (the Brier score), the two batch-level
calibration statistics most often confused with each other, and streaming
metrics for both.

Binary classification only:
    Every quantity here assumes a single Bernoulli outcome per sample. The
    variance term `p(1-p)` is the Bernoulli variance, and a flat sum across a
    class axis would pool independent calibration questions into one number.
    The loss classes therefore RAISE when `y_pred`'s last axis is statically
    known and wider than one column. For multi-class calibration use
    `dl_techniques.metrics.brier_score.CategoricalBrierScore`.

Two statistics, and why the distinction matters:
    Let `oᵢ ∈ {0, 1}` be the observed outcome and `pᵢ` the predicted
    probability for sample `i` of a batch of `N`.

    1.  Calibration-in-the-large (the calibration-intercept z-test) is the
        standardized net bias:

            Z_cil = Σᵢ (oᵢ - pᵢ) / sqrt( Σᵢ pᵢ(1 - pᵢ) )

    2.  Spiegelhalter's Z (1986) is derived from the Murphy / variance
        decomposition of the Brier score rather than from raw residuals. With
        `B = (1/N) Σ (pᵢ - oᵢ)²`, the per-sample expectation and variance
        under the null hypothesis of perfect calibration are

            E[(pᵢ - oᵢ)²] = pᵢ(1 - pᵢ)
            Var[(pᵢ - oᵢ)²] = pᵢ(1 - pᵢ)(1 - 2pᵢ)²

        and since `(pᵢ - oᵢ)² - pᵢ(1 - pᵢ) = (oᵢ - pᵢ)(1 - 2pᵢ)`, the statistic
        is

            Z_sh = Σᵢ (oᵢ - pᵢ)(1 - 2pᵢ) / sqrt( Σᵢ pᵢ(1 - pᵢ)(1 - 2pᵢ)² )

    The `(1 - 2pᵢ)` weight is the entire content of Spiegelhalter's test. Both
    statistics are standard normal under the null, but they are NOT the same
    test: `Z_cil` sees only the net bias, one scalar degree of freedom, and is
    blind to a model that is over-confident on one half of the probability
    range and under-confident on the other. Only `Z_sh` is supported by the
    1986 citation. This module implements both and names them honestly; the
    default is `"spiegelhalter"`.

    Versions of this module before 2026-09-02 computed `Z_cil` under the name
    `SpiegelhalterZLoss`. That behaviour is still reachable as
    `statistic="calibration_in_the_large"`, and a config saved before the
    change deserializes into it (see `from_config`).

Using a calibration statistic as a loss:
    Three properties of `Z` govern how it may be used, and the naive
    `Loss = Z²` gets all three wrong.

    -   The null value of `Z²` is ONE, not zero. Under perfect calibration
        `Z ~ N(0, 1)`, so `E[Z²] = 1`. Minimizing `Z² → 0` asks `Σ pᵢ` to
        match the REALIZED label count `Σ oᵢ` in every minibatch, which is
        fitting the sampling noise of the batch and demands information the
        model cannot have. `chance_corrected=True` optimizes
        `relu(Z² - 1)` instead, which is zero for a calibrated model.
    -   `Z²` scales linearly with the batch size. For a systematic bias `b` in
        probability units and mean variance `v̄`, `Z² ≈ N b² / v̄`, so moving
        from batch 32 to batch 256 multiplies the term's effective weight by
        eight. `normalize_by_n=True` divides by `N`.
    -   A pure calibration statistic is NOT a proper scoring rule. It is
        globally minimized by the constant predictor `pᵢ = ȳ`, which has zero
        discrimination, and `E[Z²]` can be reduced by inflating the denominator
        `Σ p(1-p)`, i.e. by pushing predictions toward 0.5. A calibration term
        must therefore be a SMALL-WEIGHT REGULARIZER anchored to a proper
        scoring rule, never a co-equal objective. `CombinedCalibrationLoss`
        provides exactly that anchoring; `SpiegelhalterZLoss` used alone does
        not, and its docstring says so.

    If you only need calibrated probabilities, post-hoc temperature scaling on
    a held-out split is monotone (so it cannot damage discrimination), costs
    almost nothing, and is a stronger baseline than any in-training penalty.

Per-sample decomposition:
    `Z²` is a batch-global scalar, but `keras.losses.Loss.call()` must return
    one value per sample: Keras multiplies `call()`'s output by `sample_weight`
    BEFORE reducing, so a scalar return silently yields
    `whole_batch_loss * mean(sample_weight)` and makes `reduction=` a dead
    knob. With `cᵢ = (oᵢ - pᵢ)wᵢ`, `num = Σ cᵢ` and `den = Σ vᵢwᵢ²`, this
    module returns

        z2_vec[i] = N · cᵢ · num / den        mean(z2_vec) = num²/den = Z²

    which is an algebraic identity, so the value AND the gradient are exact
    while a zero-weighted row genuinely drops out. The chance-corrected and
    batch-normalized forms decompose the same way.

Numerical stability:
    Predictions are clipped to `[1e-6, 1 - 1e-6]` BEFORE the variance is
    formed, so `den` carries a floor proportional to `N` rather than an
    absolute `1e-7`. Without the clip, a saturated sigmoid drives
    `den → 0` and `Z² → 1e7 · num²`. Note that `Z_sh` is separately degenerate
    at `pᵢ = 0.5`, where the weight `(1 - 2pᵢ)` zeroes both the numerator and
    the denominator; the same floor keeps that case finite and equal to zero.

References:
    - Spiegelhalter, D. J. (1986). "Probabilistic prediction in patient
      management and clinical trials." Statistics in Medicine, 5(5), 421-433.
      (The `(1 - 2p)`-weighted statistic, `statistic="spiegelhalter"`.)
    - Brier, G. W. (1950). "Verification of forecasts expressed in terms of
      probability." Monthly Weather Review, 78(1), 1-3.
    - Murphy, A. H. (1973). "A new vector partition of the probability score."
      Journal of Applied Meteorology, 12(4), 595-600. (The decomposition
      Spiegelhalter's Z is derived from.)
    - Cox, D. R. (1958). "Two further applications of a model for binary
      regression." Biometrika, 45(3/4), 562-565. (Calibration-in-the-large.)
    - Guo, C., et al. (2017). "On Calibration of Modern Neural Networks."
      (Temperature scaling, the post-hoc alternative.)
"""

from typing import Any, Dict, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

#: Probabilities are clipped into ``[PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON]``
#: before any variance term is formed. This is what gives the Z denominator a
#: floor proportional to the batch size instead of an absolute one: at the clip
#: boundary ``p(1-p) ~= PROBABILITY_EPSILON``.
PROBABILITY_EPSILON: float = 1e-6

#: Spiegelhalter's Z: residuals weighted by ``(1 - 2p)``.
STATISTIC_SPIEGELHALTER: str = "spiegelhalter"

#: Calibration-in-the-large / calibration-intercept z-test: unweighted residuals.
STATISTIC_CALIBRATION_IN_THE_LARGE: str = "calibration_in_the_large"

VALID_STATISTICS: Tuple[str, ...] = (
    STATISTIC_SPIEGELHALTER,
    STATISTIC_CALIBRATION_IN_THE_LARGE,
)

VALID_BASE_LOSSES: Tuple[str, ...] = ("brier", "bce")

# ---------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------


def _accumulation_dtype(*dtypes: Any) -> str:
    """Return a never-narrowing dtype for the calibration accumulation.

    A sum over a batch under ``mixed_float16`` loses precision fast, so the
    statistic is formed in float32 -- but "run this reduction in float32"
    NARROWS a float64 input, so the rule is ``max(inputs, float32)`` rather
    than a hard-coded literal. The loss's own ``dtype`` is not sufficient on
    its own: ``keras.losses.Loss`` takes its dtype from ``backend.floatx()``,
    which a float64 GLOBAL POLICY does not change.

    Args:
        *dtypes: Candidate dtypes -- the loss dtype and the input dtypes.

    Returns:
        ``"float64"`` if any candidate is float64, otherwise ``"float32"``.
    """
    for dtype in dtypes:
        if dtype is None:
            continue
        # `getattr(d, "name", None) or str(d)`, not `keras.backend.standardize_dtype`:
        # a Keras-2 residue banned across all of `src/`. Do NOT reduce it to a bare
        # `str(d)` -- a `tf.DType` stringifies as "<dtype: 'float32'>". D-007.
        if (getattr(dtype, "name", None) or str(dtype)) == "float64":
            return "float64"
    return "float32"


def _row_reduce_sum(x: Any) -> Any:
    """Collapse every axis except the batch axis with a sum, giving ``(batch,)``."""
    rank = len(x.shape)
    if rank <= 1:
        return x
    return keras.ops.sum(x, axis=tuple(range(1, rank)))


def _row_reduce_mean(x: Any) -> Any:
    """Collapse every axis except the batch axis with a mean, giving ``(batch,)``."""
    rank = len(x.shape)
    if rank <= 1:
        return x
    return keras.ops.mean(x, axis=tuple(range(1, rank)))


def _require_binary(y_pred: Any) -> None:
    """Reject a statically-known multi-column prediction tensor.

    The Bernoulli variance ``p(1-p)`` and a flat sum over a class axis do not
    describe a multi-class problem; summing across classes would pool
    independent calibration questions into a single statistic.

    Args:
        y_pred: The prediction tensor.

    Raises:
        ValueError: If the last axis is statically known and wider than one.
    """
    shape = getattr(y_pred, "shape", None)
    if shape is None or len(shape) < 2:
        return
    last = shape[-1]
    if last is not None and int(last) > 1:
        raise ValueError(
            "the calibration losses in brier_spiegelhalters_ztest_loss are BINARY "
            f"ONLY, but y_pred has a last axis of width {int(last)} (shape "
            f"{tuple(shape)}). The Bernoulli variance p(1-p) and the flat sum used "
            "by the Z statistic would pool independent per-class calibration "
            "questions into one number. Pass a single column, or use "
            "dl_techniques.metrics.brier_score.CategoricalBrierScore for the "
            "multi-class Brier score."
        )


def _to_probabilities(y_pred: Any, from_logits: bool, dtype: str, clip: bool) -> Any:
    """Cast, optionally sigmoid, and optionally clip predictions into (0, 1)."""
    y_pred = keras.ops.cast(y_pred, dtype)
    if from_logits:
        y_pred = keras.ops.sigmoid(y_pred)
    if clip:
        y_pred = keras.ops.clip(
            y_pred, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
        )
    return y_pred


def _z_statistic_terms(
    y_true: Any, probabilities: Any, statistic: str
) -> Tuple[Any, Any, Any, Any]:
    """Compute the pieces of a calibration z-statistic.

    Args:
        y_true: Observed outcomes, cast to the accumulation dtype.
        probabilities: Clipped predicted probabilities, same dtype.
        statistic: One of :data:`VALID_STATISTICS`.

    Returns:
        A tuple ``(contributions, numerator, denominator, batch_size)`` where
        ``contributions`` has shape ``(batch,)`` and sums to ``numerator``, and
        ``denominator`` already carries its stability floor.
    """
    residuals = y_true - probabilities
    variances = probabilities * (1.0 - probabilities)

    if statistic == STATISTIC_SPIEGELHALTER:
        weights = 1.0 - 2.0 * probabilities
    else:
        weights = keras.ops.ones_like(probabilities)

    contributions = _row_reduce_sum(residuals * weights)
    variance_terms = _row_reduce_sum(variances * keras.ops.square(weights))

    numerator = keras.ops.sum(contributions)
    denominator = keras.ops.sum(variance_terms)

    batch_size = keras.ops.cast(
        keras.ops.shape(contributions)[0], probabilities.dtype
    )

    # A floor PROPORTIONAL to the batch, matching the variance the clip itself
    # guarantees per element, rather than an absolute epsilon that a saturated
    # batch dwarfs. keras.config.epsilon() keeps a zero-length batch finite.
    denominator = (
        denominator + batch_size * PROBABILITY_EPSILON + keras.config.epsilon()
    )

    return contributions, numerator, denominator, batch_size


def calibration_z_per_sample(
    y_true: Any,
    probabilities: Any,
    statistic: str = STATISTIC_SPIEGELHALTER,
    use_squared: bool = True,
    chance_corrected: bool = True,
    normalize_by_n: bool = True,
) -> Any:
    """Return a per-sample vector whose MEAN is the requested calibration term.

    The batch-global statistic is attributed to rows by an exact algebraic
    identity, so both the value and the gradient match the scalar form while
    ``sample_weight`` keeps selecting rows.

    Args:
        y_true: Observed outcomes, already cast to the accumulation dtype.
        probabilities: Clipped predicted probabilities, same dtype.
        statistic: One of :data:`VALID_STATISTICS`.
        use_squared: If ``True`` the term is ``Z**2``; otherwise ``|Z|``.
        chance_corrected: If ``True`` the term is ``relu(Z**2 - 1)``, which is
            zero under perfect calibration. Requires ``use_squared``.
        normalize_by_n: If ``True`` the term is divided by the batch size,
            making it invariant to batch size for a fixed systematic bias.

    Returns:
        A tensor of shape ``(batch,)``.
    """
    contributions, numerator, denominator, batch_size = _z_statistic_terms(
        y_true, probabilities, statistic
    )

    if use_squared:
        # mean_i( N * c_i * num/den ) == num^2 / den == Z^2, exactly.
        per_sample = batch_size * contributions * (numerator / denominator)
        if chance_corrected:
            z_squared = keras.ops.square(numerator) / denominator
            # The gate is a constant w.r.t. the parameters, which is precisely
            # the subgradient relu(.) would supply.
            gate = keras.ops.cast(z_squared > 1.0, per_sample.dtype)
            per_sample = gate * (per_sample - 1.0)
    else:
        # mean_i( N * c_i * sign(num)/sqrt(den) ) == |num|/sqrt(den) == |Z|.
        scale = keras.ops.sign(numerator) / keras.ops.sqrt(denominator)
        per_sample = batch_size * contributions * scale

    if normalize_by_n:
        per_sample = per_sample / batch_size

    return per_sample

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.brier_spiegelhalters_ztest_loss")
class BrierScoreLoss(keras.losses.Loss):
    """Brier score loss: the mean squared error between probabilities and outcomes.

    B = (1/N) * Σ(pᵢ - oᵢ)²

    The Brier score is a PROPER scoring rule -- it is minimized in expectation
    by the true conditional probability -- which is what makes it a legitimate
    standalone objective, unlike the calibration statistics in this module.

    On binary targets this is exactly `keras.losses.MeanSquaredError`; the only
    thing this class adds is the `from_logits` toggle. Prefer the stock Keras
    loss when your model already outputs probabilities. The single-column
    convention is deliberate: a perfectly inverted prediction scores 1.0 here,
    not the 2.0 of the two-column multi-class convention. For the multi-class
    Brier score use `dl_techniques.metrics.brier_score.CategoricalBrierScore`.

    Args:
        from_logits: Whether the predictions are logits (not passed through a
            sigmoid). Default is False.
        reduction: Type of reduction to apply to the loss. Default is
            'sum_over_batch_size'.
        name: Optional name for the loss function.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:
        >>> model.compile(
        ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        ...     loss=BrierScoreLoss(),
        ...     metrics=["accuracy"],
        ... )
    """

    def __init__(
        self,
        from_logits: bool = False,
        reduction: str = "sum_over_batch_size",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BrierScoreLoss.

        Args:
            from_logits: Whether model outputs raw logits without sigmoid.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            reduction=reduction, name=name or "brier_score_loss", **kwargs
        )
        self.from_logits = bool(from_logits)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Compute the per-sample Brier score.

        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_pred: Predicted probabilities or logits.

        Returns:
            A tensor of shape ``(batch,)``.
        """
        dtype = _accumulation_dtype(self.dtype, y_pred.dtype, y_true.dtype)
        y_pred = _to_probabilities(y_pred, self.from_logits, dtype, clip=False)
        y_true = keras.ops.cast(y_true, dtype)
        return _row_reduce_mean(keras.ops.square(y_pred - y_true))

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({"from_logits": self.from_logits})
        return config

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.brier_spiegelhalters_ztest_loss")
class SpiegelhalterZLoss(keras.losses.Loss):
    """Batch-level calibration penalty built on a z-statistic.

    Implements both statistics described in the module docstring:

    - `statistic="spiegelhalter"` (default): Spiegelhalter's (1986) Z, whose
      residuals carry the `(1 - 2p)` weight from the Murphy decomposition of
      the Brier score.
    - `statistic="calibration_in_the_large"`: the unweighted calibration-
      intercept z-test. This is what versions of this class before 2026-09-02
      computed under the Spiegelhalter name.

    **This class is not a proper scoring rule and must not be used alone.**
    Every pure calibration statistic is globally minimized by the constant
    predictor `pᵢ = ȳ`, which has no discrimination whatsoever, and `E[Z²]`
    can additionally be reduced by inflating the denominator `Σ p(1-p)` --
    an entropy-maximizing pressure toward `p = 0.5` that degrades sharpness
    exactly when the term is loudest. Use `CombinedCalibrationLoss`, which
    anchors this penalty to a proper scoring rule at a small weight, or add it
    to your own base loss with a `lambda` of order 1e-2.

    The defaults remove the two scaling defects of the naive `Loss = Z²`:
    `chance_corrected=True` subtracts the `E[Z²] = 1` null floor, and
    `normalize_by_n=True` removes the linear batch-size dependence.
    `call()` returns one value per sample whose mean is the requested term.

    Binary classification only, and the statistic is only meaningful for
    reasonably large batches (>= ~128 samples).

    Args:
        statistic: Which statistic to compute; one of
            ``("spiegelhalter", "calibration_in_the_large")``. Default is
            "spiegelhalter".
        chance_corrected: Whether to penalize `relu(Z² - 1)` rather than `Z²`.
            Under perfect calibration `E[Z²] = 1`, so the uncorrected form has
            an unreachable floor and its residual gradient pressure is pure
            minibatch label noise. Requires `use_squared=True`. Default is True.
        normalize_by_n: Whether to divide the term by the batch size. `Z² ≈
            N b²/v̄` for a systematic bias `b`, so without this the term's
            effective weight is proportional to the batch size. Default is True.
        use_squared: Whether to use `Z²` instead of `|Z|`. `|Z|` has a
            backend-defined subgradient at 0 and a constant gradient magnitude,
            so it chatters around the optimum; it is retained for
            compatibility. Default is True.
        from_logits: Whether the predictions are logits (not passed through a
            sigmoid). Default is False.
        reduction: Type of reduction to apply to the loss. Default is
            'sum_over_batch_size'.
        name: Optional name for the loss function.
        **kwargs: Additional keyword arguments passed to the parent class.

    Raises:
        ValueError: If `statistic` is not a recognized value, or if
            `chance_corrected` is requested without `use_squared` (the null
            value of `|Z|` is `sqrt(2/pi)`, not 1, so the correction does not
            apply).

    Example:
        >>> penalty = SpiegelhalterZLoss()          # correct statistic, sane scaling
        >>> legacy = SpiegelhalterZLoss(            # the pre-2026-09-02 behaviour
        ...     statistic="calibration_in_the_large",
        ...     chance_corrected=False,
        ...     normalize_by_n=False,
        ... )
    """

    def __init__(
        self,
        statistic: str = STATISTIC_SPIEGELHALTER,
        chance_corrected: bool = True,
        normalize_by_n: bool = True,
        use_squared: bool = True,
        from_logits: bool = False,
        reduction: str = "sum_over_batch_size",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SpiegelhalterZLoss.

        Args:
            statistic: Which calibration statistic to compute.
            chance_corrected: Whether to subtract the `E[Z²] = 1` null floor.
            normalize_by_n: Whether to divide by the batch size.
            use_squared: Whether to use `Z²` instead of `|Z|`.
            from_logits: Whether model outputs raw logits without sigmoid.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: On an unknown `statistic`, or `chance_corrected`
                without `use_squared`.
        """
        if statistic not in VALID_STATISTICS:
            raise ValueError(
                f"statistic must be one of {VALID_STATISTICS}, got {statistic!r}"
            )
        if chance_corrected and not use_squared:
            raise ValueError(
                "chance_corrected=True requires use_squared=True: the chance "
                "floor E[Z**2] = 1 is a property of the SQUARED statistic, "
                "whereas E[|Z|] = sqrt(2/pi) ~= 0.7979. Got "
                f"use_squared={use_squared!r}."
            )

        super().__init__(
            reduction=reduction, name=name or "spiegelhalter_z_loss", **kwargs
        )
        self.statistic = statistic
        self.chance_corrected = bool(chance_corrected)
        self.normalize_by_n = bool(normalize_by_n)
        self.use_squared = bool(use_squared)
        self.from_logits = bool(from_logits)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Compute the per-sample calibration penalty.

        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_pred: Predicted probabilities or logits.

        Returns:
            A tensor of shape ``(batch,)`` whose mean is the batch-level term.

        Raises:
            ValueError: If `y_pred` has a statically-known last axis wider
                than one column.
        """
        _require_binary(y_pred)
        dtype = _accumulation_dtype(self.dtype, y_pred.dtype, y_true.dtype)
        probabilities = _to_probabilities(
            y_pred, self.from_logits, dtype, clip=True
        )
        y_true = keras.ops.cast(y_true, dtype)

        return calibration_z_per_sample(
            y_true,
            probabilities,
            statistic=self.statistic,
            use_squared=self.use_squared,
            chance_corrected=self.chance_corrected,
            normalize_by_n=self.normalize_by_n,
        )

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update(
            {
                "statistic": self.statistic,
                "chance_corrected": self.chance_corrected,
                "normalize_by_n": self.normalize_by_n,
                "use_squared": self.use_squared,
                "from_logits": self.from_logits,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpiegelhalterZLoss":
        """Rebuild from a config, mapping pre-2026-09-02 configs to old numerics.

        A config written before the statistic was corrected carries neither
        `statistic` nor the two scaling flags. Constructing it with today's
        defaults would silently change the objective a saved model was trained
        against, so such a config is restored to the OLD behaviour and the
        substitution is logged.

        Args:
            config: The configuration dictionary.

        Returns:
            A configured `SpiegelhalterZLoss`.
        """
        config = dict(config)
        if "statistic" not in config:
            logger.warning(
                "SpiegelhalterZLoss.from_config received a config with no "
                "'statistic' key, i.e. one written before 2026-09-02, when this "
                "class computed the CALIBRATION-IN-THE-LARGE statistic under the "
                "Spiegelhalter name. Restoring the original numerics "
                "(statistic='calibration_in_the_large', chance_corrected=False, "
                "normalize_by_n=False) so the checkpoint behaves as saved. Today's "
                "defaults compute a DIFFERENT, corrected objective -- construct a "
                "fresh instance to adopt it. Note that call() now also returns one "
                "value per sample, so sample_weight selects rows instead of "
                "scaling the batch aggregate."
            )
            config.setdefault("statistic", STATISTIC_CALIBRATION_IN_THE_LARGE)
            config.setdefault("chance_corrected", False)
            config.setdefault("normalize_by_n", False)
        elif config.get("chance_corrected") and not config.get("use_squared", True):
            logger.warning(
                "SpiegelhalterZLoss.from_config received chance_corrected=True "
                "with use_squared=False, which the constructor rejects. Setting "
                "chance_corrected=False so the config still loads."
            )
            config["chance_corrected"] = False
        return cls(**config)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.brier_spiegelhalters_ztest_loss")
class CombinedCalibrationLoss(keras.losses.Loss):
    """A proper scoring rule anchored by a small-weight calibration penalty.

    Loss = base(y, p) + lambda_cal * calibration_penalty(y, p)

    The base term (Brier or binary cross-entropy) is a proper scoring rule, so
    the objective's population minimizer remains `P(y=1|x)`. The calibration
    penalty is a REGULARIZER: it constrains a single batch-level degree of
    freedom and, on its own, would be minimized by a constant predictor. Keep
    `lambda_cal` small -- 1e-2 to 1e-1. There is no value of `lambda_cal` at
    which the calibration term becomes a co-equal objective without damaging
    discrimination.

    Both terms are per-sample, so `sample_weight` selects rows for the whole
    loss.

    Legacy mode:
        Passing `alpha` (rather than leaving it None) restores the pre-2026-09-02
        objective `alpha * Brier + (1 - alpha) * Z_cil²`. That blend is retained
        only so existing checkpoints reproduce their numerics; it is unsound as
        a training objective, because `Brier ∈ [0, 1]` while `Z² ∈ [0, O(N)]`
        -- at N=256 an `alpha` of 0.5 is not a 50/50 blend, it is "ignore the
        Brier term". A warning is logged on construction.

    Binary classification only.

    Args:
        lambda_cal: Weight of the calibration penalty. Must be >= 0. Default
            is 0.05.
        base: Which proper scoring rule anchors the objective; one of
            ``("brier", "bce")``. Default is "brier".
        statistic: Which calibration statistic to penalize; see
            `SpiegelhalterZLoss`. Default is "spiegelhalter".
        chance_corrected: Whether to penalize `relu(Z² - 1)`. Default is True.
        normalize_by_n: Whether to divide the penalty by the batch size.
            Default is True.
        use_squared_z: Whether the penalty uses `Z²` rather than `|Z|`.
            Default is True.
        alpha: Legacy blend weight for the Brier component. `None` (default)
            selects the anchored-regularizer form above; a float in `[0, 1]`
            selects the legacy blend.
        from_logits: Whether the predictions are logits (not passed through a
            sigmoid). Default is False.
        reduction: Type of reduction to apply to the loss. Default is
            'sum_over_batch_size'.
        name: Optional name for the loss function.
        **kwargs: Additional keyword arguments passed to the parent class.

    Raises:
        ValueError: If `lambda_cal` is negative, `base` is unrecognized, or
            `alpha` is outside the range [0, 1].

    Example:
        >>> model.compile(
        ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        ...     loss=CombinedCalibrationLoss(lambda_cal=0.05, base="bce"),
        ...     metrics=[SpiegelhalterZMetric()],
        ... )
    """

    def __init__(
        self,
        lambda_cal: float = 0.05,
        base: str = "brier",
        statistic: str = STATISTIC_SPIEGELHALTER,
        chance_corrected: bool = True,
        normalize_by_n: bool = True,
        use_squared_z: bool = True,
        alpha: Optional[float] = None,
        from_logits: bool = False,
        reduction: str = "sum_over_batch_size",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the CombinedCalibrationLoss.

        Args:
            lambda_cal: Weight of the calibration penalty.
            base: Which proper scoring rule anchors the objective.
            statistic: Which calibration statistic to penalize.
            chance_corrected: Whether to penalize `relu(Z² - 1)`.
            normalize_by_n: Whether to divide the penalty by the batch size.
            use_squared_z: Whether the penalty uses `Z²` rather than `|Z|`.
            alpha: Legacy blend weight; `None` selects the anchored form.
            from_logits: Whether model outputs raw logits without sigmoid.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: On a negative `lambda_cal`, an unknown `base`, or an
                `alpha` outside [0, 1].
        """
        if lambda_cal < 0.0:
            raise ValueError(f"lambda_cal must be >= 0, got {lambda_cal}")
        if base not in VALID_BASE_LOSSES:
            raise ValueError(
                f"base must be one of {VALID_BASE_LOSSES}, got {base!r}"
            )
        if alpha is not None and (alpha < 0.0 or alpha > 1.0):
            raise ValueError(f"alpha must be in the range [0, 1], got {alpha}")

        super().__init__(
            reduction=reduction, name=name or "combined_calibration_loss", **kwargs
        )
        self.lambda_cal = float(lambda_cal)
        self.base = base
        self.statistic = statistic
        self.chance_corrected = bool(chance_corrected)
        self.normalize_by_n = bool(normalize_by_n)
        self.use_squared_z = bool(use_squared_z)
        self.alpha = None if alpha is None else float(alpha)
        self.from_logits = bool(from_logits)

        if self.alpha is not None:
            logger.warning(
                "CombinedCalibrationLoss was constructed with alpha=%s, selecting "
                "the LEGACY objective alpha * Brier + (1 - alpha) * Z_cil**2. That "
                "blend is retained for checkpoint compatibility only: Brier is in "
                "[0, 1] while Z**2 is in [0, O(N)], so alpha does not weight the "
                "two terms comparably, and a pure calibration statistic is not a "
                "proper scoring rule. Prefer alpha=None with a small lambda_cal.",
                self.alpha,
            )

        # Component losses. In legacy mode the Z component is pinned to the
        # pre-2026-09-02 configuration so the blend reproduces its numerics.
        legacy = self.alpha is not None
        self.brier_loss = BrierScoreLoss(
            from_logits=from_logits, reduction="none"
        )
        self.z_loss = SpiegelhalterZLoss(
            statistic=(
                STATISTIC_CALIBRATION_IN_THE_LARGE if legacy else self.statistic
            ),
            chance_corrected=False if legacy else self.chance_corrected,
            normalize_by_n=False if legacy else self.normalize_by_n,
            use_squared=self.use_squared_z,
            from_logits=from_logits,
            reduction="none",
        )

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Compute the per-sample combined loss.

        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_pred: Predicted probabilities or logits.

        Returns:
            A tensor of shape ``(batch,)``.

        Raises:
            ValueError: If `y_pred` has a statically-known last axis wider
                than one column.
        """
        _require_binary(y_pred)

        # `.call()`, never `__call__`: the latter would re-run dtype conversion,
        # rank squeezing and reduction inside this loss's own call.
        penalty = self.z_loss.call(y_true, y_pred)

        if self.alpha is not None:
            brier = self.brier_loss.call(y_true, y_pred)
            return self.alpha * brier + (1.0 - self.alpha) * penalty

        if self.base == "brier":
            base_term = self.brier_loss.call(y_true, y_pred)
        else:
            probabilities = _to_probabilities(
                y_pred,
                self.from_logits,
                _accumulation_dtype(self.dtype, y_pred.dtype, y_true.dtype),
                clip=True,
            )
            targets = keras.ops.cast(y_true, probabilities.dtype)
            base_term = _row_reduce_mean(
                -(
                    targets * keras.ops.log(probabilities)
                    + (1.0 - targets) * keras.ops.log(1.0 - probabilities)
                )
            )

        return base_term + self.lambda_cal * penalty

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update(
            {
                "lambda_cal": self.lambda_cal,
                "base": self.base,
                "statistic": self.statistic,
                "chance_corrected": self.chance_corrected,
                "normalize_by_n": self.normalize_by_n,
                "use_squared_z": self.use_squared_z,
                "alpha": self.alpha,
                "from_logits": self.from_logits,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CombinedCalibrationLoss":
        """Rebuild from a config, mapping pre-2026-09-02 configs to old numerics.

        A legacy config carries `alpha` and `use_squared_z` but not
        `lambda_cal`. Such a config selects legacy blend mode, which reproduces
        the objective the checkpoint was trained against.

        Args:
            config: The configuration dictionary.

        Returns:
            A configured `CombinedCalibrationLoss`.
        """
        config = dict(config)
        if "lambda_cal" not in config:
            logger.warning(
                "CombinedCalibrationLoss.from_config received a config with no "
                "'lambda_cal' key, i.e. one written before 2026-09-02. Restoring "
                "the legacy blend alpha * Brier + (1 - alpha) * Z_cil**2 so the "
                "checkpoint behaves as saved. Today's default is an anchored "
                "regularizer with a corrected statistic -- construct a fresh "
                "instance to adopt it."
            )
            config.setdefault("alpha", 0.5)
        return cls(**config)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.brier_spiegelhalters_ztest_loss")
class BrierScoreMetric(keras.metrics.Metric):
    """Streaming Brier score for monitoring calibration during training.

    Mean squared difference between predicted probabilities and outcomes,
    accumulated across batches. Lower is better.

    `dl_techniques.metrics.brier_score.BrierScore` is the canonical
    implementation of the same quantity and is the one to reach for in new
    code; this class exists so that a model compiled against this module keeps
    working, and the two are held to agree by
    `tests/test_losses/test_brier_spiegelhalters_ztest_loss.py`. The only
    difference is the default of `from_logits` (False here, True there).

    Args:
        name: String name of the metric instance.
        from_logits: Whether predictions are logits (not passed through
            sigmoid). Default is False.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        name: str = "brier_score",
        from_logits: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the BrierScoreMetric.

        Args:
            name: Name of the metric.
            from_logits: Whether predictions are logits.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.from_logits = bool(from_logits)
        self.total_score = self.add_weight(name="total_score", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None,
    ) -> None:
        """Update the metric state.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.
            sample_weight: Optional sample weights, broadcastable to the shape
                of the per-element scores.
        """
        dtype = self.dtype or "float32"
        y_pred = _to_probabilities(y_pred, self.from_logits, dtype, clip=False)
        y_true = keras.ops.cast(y_true, dtype)
        scores = keras.ops.square(y_pred - y_true)

        if sample_weight is None:
            self.total_score.assign_add(keras.ops.sum(scores))
            self.count.assign_add(
                keras.ops.cast(keras.ops.size(scores), dtype)
            )
            return

        # A weighted MEAN needs a weighted DENOMINATOR. Accumulating a weighted
        # numerator over a raw element count is not a mean of anything.
        sample_weight = keras.ops.cast(sample_weight, dtype)
        while len(sample_weight.shape) < len(scores.shape):
            sample_weight = keras.ops.expand_dims(sample_weight, axis=-1)
        weights = keras.ops.broadcast_to(sample_weight, keras.ops.shape(scores))
        self.total_score.assign_add(keras.ops.sum(scores * weights))
        self.count.assign_add(keras.ops.sum(weights))

    def result(self) -> Any:
        """Compute the final metric result.

        Returns:
            The accumulated Brier score, or 0.0 when nothing was accumulated.
        """
        return self.total_score / (self.count + keras.config.epsilon())

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.total_score.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Get metric configuration for serialization.

        Returns:
            Dictionary containing the metric configuration.
        """
        config = super().get_config()
        config.update({"from_logits": self.from_logits})
        return config

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.brier_spiegelhalters_ztest_loss")
class SpiegelhalterZMetric(keras.metrics.Metric):
    """Streaming calibration z-statistic for monitoring during training.

    Under the null hypothesis of perfect calibration the statistic is standard
    normal, so `|Z| > 1.96` rejects calibration at p < 0.05 and values near 0
    indicate good calibration. Note that the SIGN is informative: positive Z
    means the outcomes exceeded the predictions.

    Defaults to Spiegelhalter's (1986) `(1 - 2p)`-weighted statistic;
    `statistic="calibration_in_the_large"` gives the unweighted
    calibration-intercept test, which is what this class computed before
    2026-09-02.

    Sample weights enter the numerator linearly and the variance QUADRATICALLY
    (`Σ wᵢ² pᵢ(1-pᵢ)wₛₕ²`), which is what keeps the ratio standard normal;
    weighting the variance linearly, as this class previously did, leaves a
    number that is not a z-score of anything.

    Args:
        name: String name of the metric instance.
        from_logits: Whether predictions are logits (not passed through
            sigmoid). Default is False.
        statistic: Which statistic to compute; one of
            ``("spiegelhalter", "calibration_in_the_large")``.
        **kwargs: Additional keyword arguments passed to the parent class.

    Raises:
        ValueError: If `statistic` is not a recognized value.
    """

    def __init__(
        self,
        name: str = "spiegelhalter_z",
        from_logits: bool = False,
        statistic: str = STATISTIC_SPIEGELHALTER,
        **kwargs: Any,
    ) -> None:
        """Initialize the SpiegelhalterZMetric.

        Args:
            name: Name of the metric.
            from_logits: Whether predictions are logits.
            statistic: Which calibration statistic to compute.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: On an unknown `statistic`.
        """
        if statistic not in VALID_STATISTICS:
            raise ValueError(
                f"statistic must be one of {VALID_STATISTICS}, got {statistic!r}"
            )
        super().__init__(name=name, **kwargs)
        self.from_logits = bool(from_logits)
        self.statistic = statistic
        self.residual_sum = self.add_weight(name="residual_sum", initializer="zeros")
        self.variance_sum = self.add_weight(name="variance_sum", initializer="zeros")

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None,
    ) -> None:
        """Update the metric state.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.
            sample_weight: Optional sample weights, broadcastable to the shape
                of the per-element contributions.
        """
        dtype = self.dtype or "float32"
        probabilities = _to_probabilities(y_pred, self.from_logits, dtype, clip=True)
        y_true = keras.ops.cast(y_true, dtype)

        if self.statistic == STATISTIC_SPIEGELHALTER:
            weights = 1.0 - 2.0 * probabilities
        else:
            weights = keras.ops.ones_like(probabilities)

        contributions = (y_true - probabilities) * weights
        variances = (
            probabilities * (1.0 - probabilities) * keras.ops.square(weights)
        )

        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, dtype)
            while len(sample_weight.shape) < len(contributions.shape):
                sample_weight = keras.ops.expand_dims(sample_weight, axis=-1)
            contributions = contributions * sample_weight
            # Quadratic, not linear: Var[Σ wᵢ rᵢ] = Σ wᵢ² Var[rᵢ].
            variances = variances * keras.ops.square(sample_weight)

        self.residual_sum.assign_add(keras.ops.sum(contributions))
        self.variance_sum.assign_add(keras.ops.sum(variances))

    def result(self) -> Any:
        """Compute the final metric result.

        Returns:
            The calibration z-statistic, or 0.0 when nothing was accumulated.
        """
        denominator = keras.ops.sqrt(
            keras.ops.maximum(self.variance_sum, keras.config.epsilon())
        )
        return self.residual_sum / denominator

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.residual_sum.assign(0.0)
        self.variance_sum.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Get metric configuration for serialization.

        Returns:
            Dictionary containing the metric configuration.
        """
        config = super().get_config()
        config.update(
            {"from_logits": self.from_logits, "statistic": self.statistic}
        )
        return config

# ---------------------------------------------------------------------
