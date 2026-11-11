"""
Cross-entropy with a per-sample confidence penalty and an outlier-exposure path.

This loss replaced ``GoodhartAwareLoss`` (``goodhart_loss.py``) outright. The old
name asserted a claim the audit of that class refuted: a loss that only ever
sees ``(y_true, y_pred)`` cannot tell strong evidence from flimsy evidence on a
labeled row, so it cannot "penalize confidence on flimsy evidence". What it can
do, and what each part of this loss was measured to do, is listed below.

What each part does:

1.  **Cross-entropy, optionally label-smoothed.** Computed from the SAME
    log-probabilities as the penalty, so ``epsilon`` governs every term. It is
    computed explicitly, ``-sum(y * log p)``, and NOT with
    ``keras.losses.categorical_crossentropy``: on the TensorFlow backend that
    fused op returns the gradient ``softmax - y`` for ANY label mass, so an
    all-zero target row (value ``0``) still pushed gradient into the logits
    (measured: ``|grad| = 0.125`` summed over four such rows) and a row of mass
    0.5 got a gradient off by 3.1e-2.
2.  **Confidence penalty** ``-entropy_weight * H(p)`` (Pereyra et al., 2017). For
    a hard label its optimum caps the logit margin at about
    ``1 / entropy_weight`` nats (10 at 0.1, i.e. ``p_y ~ 0.9996``, so the
    default barely binds). It is input-agnostic: it lowers confidence
    everywhere, not on weak evidence specifically. It is NOT a proper scoring
    rule: the population minimizer is not ``p(y|x)``, so raw outputs are biased
    toward uniform. Its verified benefit is generalization (accuracy, NLL after
    recalibration), not calibrated probabilities.
3.  **No-evidence rows** (outlier exposure, Hendrycks et al., 2019). A row whose
    target is the uniform vector ``1/K`` makes the cross-entropy pull ``p``
    toward uniform on THAT input. This is the only part that lowers confidence
    where the input carries no class evidence, and it needs the caller to
    supply such inputs (noise, another dataset, corrupted samples). Weight them
    with ``sample_weight`` (0.5 is a sane start).
4.  **Ignored rows.** An all-zero target row contributes exactly ``0`` to every
    term, including label smoothing and the penalty, in value AND gradient.

What this loss does NOT do: it never sees ``X``, it cannot tell strong from weak
evidence on labeled rows, and it does not produce calibrated probabilities.
Calibration is a held-out quantity; fit a temperature on validation logits after
training (a post-hoc fit, outside this module) and divide the logits by it at
inference.

Removed with the rename: ``prior_weight`` (batch-marginal KL: measured to raise
test ECE 3.4x at batch 32 and 18x at batch 8, inert at batch 256, redundant under
supervised labels), ``class_prior`` and ``mi_weight``. They are not in the
signature at all, so an old ``GoodhartAwareLoss`` config raises ``TypeError``
instead of loading into a different objective, and a ``.keras`` archive compiled
with the old loss no longer loads with ``compile=True`` (it loads with
``compile=False``).

References:
    -   Pereyra, G., Tucker, G., Chorowski, J., Kaiser, L., & Hinton, G.
        (2017). "Regularizing Neural Networks by Penalizing Confident Output
        Distributions." arXiv:1701.06548.
    -   Hendrycks, D., Mazeika, M., & Dietterich, T. (2019). "Deep Anomaly
        Detection with Outlier Exposure." ICLR 2019. arXiv:1812.04606.
"""

import warnings
import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.dtype_policy import stability_floor
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.validation import non_negative_float

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.confidence_penalty_loss")
class ConfidencePenaltyLoss(keras.losses.Loss):
    """
    Cross-entropy with a per-sample confidence penalty.

    The per-row loss is

    ``L_i = CE(y'_i, p_i) - entropy_weight * H(p_i) * [sum(y_i) > 0]``

    with ``y'_i = (1 - label_smoothing) * y_i + sum(y_i) * label_smoothing / K``
    and ``p_i = softmax(y_pred_i)`` (or the clipped, renormalized ``y_pred_i`` when
    ``from_logits=False``).

    **Targets** have shape ``(batch_size, num_classes)``, matching ``y_pred``.
    Three kinds of row are told apart by their mass ``sum(y_i)``:

    - one-hot, smoothed or soft (mixup) rows are labeled rows;
    - the uniform row ``1/K`` is a no-evidence (outlier exposure) row;
    - the all-zero row is IGNORED: value exactly ``0`` and gradient exactly ``0``,
      including under label smoothing and the confidence penalty.

    Rows of any other mass are labeled rows scored against the target as given:
    the cross-entropy is exact for them, and the penalty is applied at full
    weight to every row with ``mass > 0`` whatever its mass. Integer class
    indices of shape ``(batch_size,)`` or ``(batch_size, 1)`` are rejected with a
    ``ValueError``; use ``keras.ops.one_hot`` first.

    **Ignored means zero weight, not dropped.** The reduction still divides by
    the batch size, so ``loss(5 ignored of 16)`` equals
    ``loss(the 11 valid rows) * 11 / 16``, exactly the
    ``loss(w=[1, ..., 1, 0]) * B / (B - 1) == loss(batch[:-1])`` identity of a
    zero ``sample_weight``. Nothing couples the rows, so that identity is exact.
    A NaN or inf logit in an ignored row is NOT ignored (the mask multiplies, and
    ``0 * nan`` is ``nan``); any finite value is.

    **Outlier exposure.** Append uniform-target rows for inputs that carry no
    class evidence and give them ``sample_weight=0.5``. The loss never sees ``X``.

    **Label smoothing identity.** ``CE(y', p) = (1 - e) * CE(y, p) + e * H(u, p)``
    for ``e = label_smoothing`` and uniform ``u``, NOT ``CE + e * H(u, p)``.

    The confidence penalty is the uniform divergence up to a constant:
    ``KL(p || u) = log K - H(p)``. Label smoothing adds ``e * KL(u || p)`` up to a
    constant, so the two regularizers use the same divergence in opposite
    directions (Pereyra et al., 2017, section 3).
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        entropy_weight: float = 0.1,
        *,
        from_logits: bool = True,
        epsilon: float = 1e-8,
        name: str = "confidence_penalty_loss",
        reduction: str = "sum_over_batch_size",
        dtype: Optional[str] = None,
    ) -> None:
        """
        Initializes the ConfidencePenaltyLoss.

        Unknown keyword arguments raise ``TypeError``; in particular the removed
        ``prior_weight``, ``class_prior`` and ``mi_weight`` are not accepted, not
        even at an inert value.

        :param label_smoothing: Label smoothing factor, in ``[0, 1)``. Accepts any
            real number type (NumPy scalars included); ``bool``, ``nan`` and
            ``inf`` are rejected. Defaults to 0.0.
        :type label_smoothing: float
        :param entropy_weight: Weight of the confidence penalty, non-negative and
            finite. Typical range ``[0.001, 0.5]``; above 0.5 a ``UserWarning`` is
            raised because the hard-label logit margin is then capped below 2
            nats. Defaults to 0.1.
        :type entropy_weight: float
        :param from_logits: Whether ``y_pred`` holds logits (default) or
            probabilities.
        :type from_logits: bool
        :param epsilon: Clip floor for probabilities on the ``from_logits=False``
            path, in ``(0, 0.1)``. It governs the cross-entropy as well as the
            penalty (one producer of log-probabilities). **The effective floor is
            ``max(epsilon, smallest normal of y_pred.dtype)``**: 1e-8 is below
            float16's smallest normal (6.1e-05), so on a float16 ``y_pred`` the
            floor is lifted to that value. float32, float64 and bfloat16 (which has
            float32's exponent range) floor at exactly ``epsilon``. Defaults to
            1e-8.
        :type epsilon: float
        :param name: Name of the loss.
        :type name: str
        :param reduction: Keras reduction. Defaults to ``"sum_over_batch_size"``.
        :type reduction: str
        :param dtype: Optional dtype for the computation; ``None`` follows
            ``keras.backend.floatx()``.
        :type dtype: Optional[str]
        :raises ValueError: If a numeric argument is not a finite non-negative real
            or is outside its range.
        :raises TypeError: On any unknown keyword argument.
        """
        # DECISION plan-2026-11-11T131349-13b28b37/D-041
        # The stored value is `float(value)`, not the value as passed. The
        # validator accepts NumPy scalars (the old `isinstance(x, (int, float))`
        # refused `np.float32(0.1)` and `np.int64(1)`), and a `np.float32` left
        # in the config makes plain `json.dumps(loss.get_config())` raise
        # "Object of type float32 is not JSON serializable". `bool` is refused
        # on purpose (`True` passed the old check and meant 1.0) and so are
        # nan/inf (`inf` produced a loss of -inf). The rule lives in
        # `utils.validation.non_negative_float` (it replaced a private twin of
        # `positive_float` here): do not re-spell it in this file, do not store
        # the argument verbatim and do not relax the bool check. See
        # decisions.md D-041.
        label_smoothing = non_negative_float(
            label_smoothing, "label_smoothing", "ConfidencePenaltyLoss"
        )
        if label_smoothing >= 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}."
            )
        entropy_weight = non_negative_float(
            entropy_weight, "entropy_weight", "ConfidencePenaltyLoss"
        )
        epsilon = non_negative_float(epsilon, "epsilon", "ConfidencePenaltyLoss")
        if not 0.0 < epsilon < 0.1:
            raise ValueError(f"epsilon must be in (0, 0.1), got {epsilon}.")
        if entropy_weight > 0.5:
            warnings.warn(
                f"entropy_weight={entropy_weight} caps the hard-label logit "
                f"margin near {1.0 / entropy_weight:.2f} nats and may dominate "
                "training. Consider values in [0.001, 0.5].",
                UserWarning,
                stacklevel=2,
            )

        super().__init__(name=name, reduction=reduction, dtype=dtype)

        self.label_smoothing = label_smoothing
        self.entropy_weight = entropy_weight
        self.from_logits = bool(from_logits)
        self.epsilon = epsilon

        # DECISION plan-2026-11-11T131349-13b28b37/D-036
        # Keep the user's dtype ARGUMENT, not `self.dtype`. Keras 3's
        # Loss.get_config() emits only {name, reduction}, and `self.dtype`
        # resolves None to the live global policy -- emitting it would pin that
        # policy into every saved config. Do not "simplify" to self.dtype.
        # See decisions.md D-036.
        self._dtype_arg = dtype

    # -- the single producer of log p for every term ------------------
    def _log_probabilities(self, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Produce log-probabilities, the single source of every term.

        On the ``from_logits=True`` path this is ``log_softmax``, exact for
        saturated logits where ``softmax -> clip -> log`` would floor the entropy
        at ``(K - 1) * epsilon * |log epsilon|`` and zero its gradient. On the
        ``from_logits=False`` path the probabilities are clipped away from zero
        and RENORMALIZED so each row still sums to one.

        :param y_pred: Logits or probabilities, shape ``(batch_size, num_classes)``.
        :type y_pred: keras.KerasTensor
        :return: Log-probabilities of the same shape.
        :rtype: keras.KerasTensor
        """
        if self.from_logits:
            return keras.ops.log_softmax(y_pred, axis=-1)

        # DECISION plan-2026-11-11T131349-13b28b37/D-037
        # The clip floor is `stability_floor(dtype, epsilon)` =
        # max(epsilon, finfo(dtype).tiny), never the bare literal. `epsilon`
        # (1e-8) is BELOW float16's smallest normal (6.1e-5), so a bare
        # clip(y, epsilon, 1) floors to 0.0 and log() gives -inf on any float16
        # y_pred -- reached via `dtype=` on this loss or a direct call, not via
        # the global policy. Do NOT call `np.finfo(dtype).tiny` here: it raises
        # for bfloat16 ("data type ... not inexact"), which the previous class
        # did; `stability_floor` resolves bfloat16 through ml_dtypes. Do NOT
        # reduce the dtype name to `str(dtype)`: a `tf.DType` stringifies as
        # "<dtype: 'float32'>". See decisions.md D-037.
        floor = stability_floor(y_pred.dtype, self.epsilon)
        probs = keras.ops.clip(y_pred, floor, 1.0)
        probs = probs / keras.ops.sum(probs, axis=-1, keepdims=True)
        return keras.ops.log(probs)

    @staticmethod
    def _check_target_shape(
        y_true: keras.KerasTensor, y_pred: keras.KerasTensor
    ) -> None:
        """Refuse targets that are not row-aligned with the predictions.

        :raises ValueError: If ``y_true`` and ``y_pred`` differ in rank or in any
            statically known dimension.
        """
        # DECISION plan-2026-11-11T131349-13b28b37/D-040
        # The explicit cross-entropy `-sum(y * log p)` BROADCASTS, so integer
        # class indices of shape (B,) or (B, 1) against (B, K) log-probabilities
        # would return a plausible wrong number. Keras's own cross-entropy used to
        # raise here and this loss must keep refusing. Do not drop this check as
        # "Keras validates shapes": the fused op is no longer called. See
        # decisions.md D-040.
        true_shape = tuple(y_true.shape)
        pred_shape = tuple(y_pred.shape)
        if len(pred_shape) < 2 or len(true_shape) != len(pred_shape) or any(
            a is not None and b is not None and a != b
            for a, b in zip(true_shape, pred_shape)
        ):
            raise ValueError(
                f"y_true must be a (batch, num_classes) tensor with the shape of "
                f"y_pred (one-hot, smoothed, soft or all-zero rows; class indices "
                f"are not accepted), got y_true shape {true_shape} and y_pred "
                f"shape {pred_shape}."
            )

    def components(
        self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Per-sample building blocks of the loss.

        :param y_true: Targets, shape ``(batch_size, num_classes)``.
        :type y_true: keras.KerasTensor
        :param y_pred: Logits or probabilities, same shape.
        :type y_pred: keras.KerasTensor
        :return: ``(cross_entropy, entropy, has_target)``, each of shape
            ``(batch_size,)``. ``cross_entropy`` is exactly ``0`` on an all-zero
            row; ``has_target`` is ``1`` for rows with mass above zero and ``0``
            for ignored rows; ``entropy`` is the entropy of every row's
            prediction (callers mask it with ``has_target``).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        :raises ValueError: If ``y_true`` is not row-aligned with ``y_pred``.
        """
        self._check_target_shape(y_true, y_pred)
        y_true = keras.ops.cast(y_true, y_pred.dtype)
        log_probs = self._log_probabilities(y_pred)

        mass = keras.ops.sum(y_true, axis=-1, keepdims=True)
        has_target = keras.ops.cast(mass > 0, y_pred.dtype)
        if self.label_smoothing > 0:
            num_classes = keras.ops.cast(keras.ops.shape(y_true)[-1], y_pred.dtype)
            # Mass-preserving smoothing; an all-zero row has mass 0 and stays 0.
            y_true = (
                y_true * (1.0 - self.label_smoothing)
                + mass * self.label_smoothing / num_classes
            )

        # DECISION plan-2026-11-11T131349-13b28b37/D-039
        # The cross-entropy is `-sum(y * log p)` over the shared log-probabilities,
        # NOT `keras.losses.categorical_crossentropy`. On the TensorFlow backend
        # that fused op's gradient is `softmax - y` for ANY label mass: an
        # all-zero row has value 0 but gradient `softmax` (measured 0.125 summed
        # over four rows, even at entropy_weight=0 and label_smoothing=0), and a
        # row of mass 0.5 is off by 3.1e-2. Do not swap it back to save three
        # lines: the masked-row gradient tests in test_confidence_penalty_loss.py
        # go red. See decisions.md D-039.
        cross_entropy = -keras.ops.sum(y_true * log_probs, axis=-1)
        entropy = -keras.ops.sum(keras.ops.exp(log_probs) * log_probs, axis=-1)
        return cross_entropy, entropy, keras.ops.squeeze(has_target, axis=-1)

    def call(
        self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute the per-sample loss.

        :param y_true: Targets, shape ``(batch_size, num_classes)``.
        :type y_true: keras.KerasTensor
        :param y_pred: Logits or probabilities, same shape.
        :type y_pred: keras.KerasTensor
        :return: Per-sample loss of shape ``(batch_size,)``; exactly ``0`` for
            all-zero target rows. The penalty rides inside the per-row vector, so
            ``sample_weight`` weights it row by row.
        :rtype: keras.KerasTensor
        """
        cross_entropy, entropy, has_target = self.components(y_true, y_pred)
        if self.entropy_weight > 0:
            return cross_entropy - self.entropy_weight * entropy * has_target
        return cross_entropy

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration dictionary for serialization.

        ``super().get_config()`` returns only ``{name, reduction}`` on Keras 3, so
        ``dtype`` is emitted explicitly, and it is the caller's argument, not the
        resolved ``self.dtype``. Every number is a plain Python float.

        :return: The configuration of the loss.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "label_smoothing": self.label_smoothing,
                "entropy_weight": self.entropy_weight,
                "from_logits": self.from_logits,
                "epsilon": self.epsilon,
                "dtype": self._dtype_arg,
            }
        )
        return config


def analyze_confidence_penalty_loss(
    loss_fn: ConfidencePenaltyLoss,
    y_true: Any,
    y_pred: Any,
    sample_weight: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Break a ``ConfidencePenaltyLoss`` down into its components, for debugging.

    ``total_loss`` IS the live loss: it is ``loss_fn(y_true, y_pred,
    sample_weight=sample_weight)`` under the loss's own ``reduction``, so it equals
    what ``fit`` minimizes for the same inputs. The components are unweighted
    means over the rows that carry a target, and the reduction divides by the full
    batch size, so with no ``sample_weight`` and the default reduction
    ``(cross_entropy + entropy_term_weighted) * valid_rows / batch_size ==
    total_loss``, where ``valid_rows = batch_size - ignored_rows``.

    :param loss_fn: The loss instance whose settings are analyzed.
    :type loss_fn: ConfidencePenaltyLoss
    :param y_true: Targets, shape ``(batch_size, num_classes)``.
    :param y_pred: Logits or probabilities, same shape.
    :param sample_weight: Optional per-sample weights, passed to the live loss for
        ``total_loss`` only.
    :return: A dictionary with ``total_loss``, ``cross_entropy`` (mean over labeled
        rows, label smoothing included), ``entropy_unweighted`` (mean
        ``H(p)`` over labeled rows), ``entropy_term_weighted``
        (``-entropy_weight * entropy_unweighted``), ``ignored_rows`` (count of
        all-zero target rows), the magnitudes ``ce_magnitude``,
        ``entropy_magnitude``, ``total_magnitude``, and the bounded shares
        ``ce_share`` and ``entropy_share`` (in ``[0, 1]``, summing to one, except
        that both are ``0.0`` when every magnitude is exactly zero, which an
        all-ignored batch reaches).
    :rtype: Dict[str, Any]
    :raises ValueError: If ``loss_fn.reduction`` is ``"none"`` (there is no scalar
        live loss to report) or ``y_true`` is not row-aligned with ``y_pred``.
    """
    if loss_fn.reduction is None or str(loss_fn.reduction).lower() == "none":
        raise ValueError(
            "analyze_confidence_penalty_loss needs a scalar live loss; "
            "reduction='none' has none."
        )
    # Same conversion Loss.__call__ applies, so the components see the dtype the
    # live loss sees.
    y_true = keras.ops.convert_to_tensor(y_true, dtype=loss_fn.dtype)
    y_pred = keras.ops.convert_to_tensor(y_pred, dtype=loss_fn.dtype)

    cross_entropy, entropy, has_target = loss_fn.components(y_true, y_pred)
    to_np = keras.ops.convert_to_numpy
    cross_entropy = to_np(cross_entropy)
    entropy = to_np(entropy)
    has_target = to_np(has_target).astype(bool)

    valid = max(int(has_target.sum()), 1)
    ce_mean = float(cross_entropy[has_target].sum() / valid)
    entropy_mean = float(entropy[has_target].sum() / valid)
    entropy_weighted = -loss_fn.entropy_weight * entropy_mean
    total = float(to_np(loss_fn(y_true, y_pred, sample_weight=sample_weight)))

    # DECISION plan-2026-11-11T131349-13b28b37/D-038
    # Do NOT restore `component / total_loss * 100`. The total is a SIGNED sum of
    # signed terms, so that ratio is unbounded: at a negative total it reads
    # ce = -100% with entropy = +200%. Divide magnitudes by the sum of
    # magnitudes instead -- every share is then in [0, 1] and they sum to one.
    # The degenerate all-zero case (every row ignored) gives 0.0 shares, not 1/2:
    # 0/0 is not a share. See decisions.md D-038.
    magnitudes = {
        "ce_magnitude": abs(ce_mean),
        "entropy_magnitude": abs(entropy_weighted),
    }
    total_magnitude = sum(magnitudes.values())
    shares = {
        "ce_share": magnitudes["ce_magnitude"] / total_magnitude
        if total_magnitude > 0.0
        else 0.0,
        "entropy_share": magnitudes["entropy_magnitude"] / total_magnitude
        if total_magnitude > 0.0
        else 0.0,
    }
    return {
        "total_loss": total,
        "cross_entropy": ce_mean,
        "entropy_unweighted": entropy_mean,
        "entropy_term_weighted": entropy_weighted,
        "ignored_rows": int((~has_target).sum()),
        **magnitudes,
        "total_magnitude": total_magnitude,
        **shares,
    }

# ---------------------------------------------------------------------
