"""
An information-theoretic loss to promote robust generalization.

This loss function is designed to address the ML equivalent of Goodhart's Law,
where a model, in optimizing a simple metric like cross-entropy, learns to
exploit statistical shortcuts or "spurious correlations" in the training
data rather than learning the underlying causal features. The objective is to
create a more holistic training signal that encourages robustness and better
generalization by combining the standard task loss with information-theoretic
regularizers.

Conceptual Overview:
    The core idea is that a robust model should not only be accurate but also
    well-calibrated (not overconfident) and information-efficient (it should
    compress the input, retaining only the information essential for the task).
    By explicitly penalizing overconfidence and information redundancy, this
    loss function guides the model away from brittle, shortcut-based solutions
    towards more generalizable representations.

Architectural Design:
    The total loss is a weighted composite of three distinct components:
    1.  Categorical Cross-Entropy: The standard supervised loss that drives
        the model to make accurate predictions. This is the primary "task"
        component.
    2.  Entropy Regularization: This term penalizes overconfident predictions
        by maximizing the Shannon entropy of the model's output distribution for
        each sample. A higher entropy corresponds to a less confident, more
        uniform prediction. This discourages the model from collapsing its
        predictions based on flimsy evidence from a single spurious feature.
    3.  Mutual Information Regularization: Based on the Information Bottleneck
        principle, this term penalizes the mutual information between the raw
        input `X` and the model's prediction `Ŷ`. It encourages the model to
        learn a compressed internal representation, forcing it to "forget"
        information from the input that is not strictly necessary for the
        prediction task. This compression is hypothesized to discard noisy or
        spurious features, retaining only the robust, generalizable ones.

Mathematical Formulation:
    The total loss is a linear combination of the three components:

    L = L_CE - λ * H(p(Ŷ|X)) + β * I(X; Ŷ)

    Where:
    -   `L_CE` is the standard categorical cross-entropy loss.
    -   `H(p(Ŷ|X))` is the conditional entropy of the prediction `Ŷ` given the
        input `X`, averaged over the batch. The loss *maximizes* this entropy
        (by minimizing its negative) to penalize confidence. `λ` is its weight.
    -   `I(X; Ŷ)` is the mutual information between the input and the prediction.
        The loss *minimizes* this term to encourage compression. `β` is its
        weight. The mutual information is practically approximated over a
        batch using the identity `I(X; Ŷ) = H(Ŷ) - H(Ŷ|X)`, where `H(Ŷ)` is the
        entropy of the marginal prediction distribution (the average prediction
        across the batch).

References:
    -   Pereyra, G., et al. (2017). "Regularizing Neural Networks by Penalizing
        Confident Output Distributions." (For the entropy regularization term).
    -   Tishby, N., Pereira, F. C., & Bialek, W. (2000). "The Information
        Bottleneck Method." (For the mutual information term).
"""

import keras
import warnings
import numpy as np
from typing import Any, Dict, Optional, Sequence
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.losses.goodhart_loss")
class GoodhartAwareLoss(keras.losses.Loss):
    """
    An information-theoretic loss combining cross-entropy with regularization.

    This loss function augments standard cross-entropy with a per-sample
    confidence penalty and an optional, off-by-default anti-collapse term that
    keeps the batch-marginal prediction distribution close to a class prior.

    The total loss is calculated as:
    ``L = CE(y, y_pred) - entropy_weight * H(y_pred) + prior_weight * KL(prior || mean_p)``
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        entropy_weight: float = 0.1,
        prior_weight: float = 0.0,
        class_prior: Optional[Sequence[float]] = None,
        from_logits: bool = True,
        epsilon: float = 1e-8,
        name: str = 'goodhart_aware_loss',
        reduction: str = 'sum_over_batch_size',
        dtype: Optional[str] = None,
        mi_weight: Optional[float] = None
    ) -> None:
        """
        Initializes the GoodhartAwareLoss.

        :param label_smoothing: Factor for label smoothing, a regularization
            technique applied to the cross-entropy component. Must be in the
            range [0, 1). Defaults to 0.0 (no smoothing).
        :type label_smoothing: float
        :param entropy_weight: Weight (:math:`\\lambda`) for the confidence
            penalty. Controls how much the model is encouraged to maintain
            uncertainty. Higher values combat overconfidence more strongly.
            Typical range: [0.001, 0.5]. Defaults to 0.1.
        :type entropy_weight: float
        :param prior_weight: Weight for the anti-collapse term
            :math:`KL(prior \\| \\bar{p})`, where :math:`\\bar{p}` is the mean
            prediction over the batch. Must be non-negative. Defaults to 0.0,
            which switches the term off entirely. Any positive value makes the
            loss batch-coupled; see :meth:`get_config` for the consequences.
        :type prior_weight: float
        :param class_prior: Target marginal distribution over the classes for
            the anti-collapse term. Entries must be finite and strictly
            positive; the sequence is normalized to sum to one if it does not
            already. Its length must equal the number of classes. Defaults to
            ``None``, meaning the uniform prior ``1 / num_classes``.
        :type class_prior: Optional[Sequence[float]]
        :param from_logits: Whether `y_pred` is a tensor of logits or
            probabilities. Set to ``True`` (default) if your model does not have
            a final softmax activation.
        :type from_logits: bool
        :param epsilon: A small constant for numerical stability on the
            ``from_logits=False`` path, where probabilities are clipped and
            renormalized. Should be much smaller than 1/num_classes.
            Defaults to 1e-8.
        :type epsilon: float
        :param name: String name for the loss function.
        :type name: str
        :param reduction: Type of reduction to apply to loss.
        :type reduction: str
        :param dtype: Optional dtype (or dtype policy name) for the loss
            computation. ``None`` (default) follows the global Keras policy.
        :type dtype: Optional[str]
        :param mi_weight: **Removed.** Passing anything other than ``None``
            raises. See the ``:raises:`` entry below.
        :type mi_weight: Optional[float]
        :raises ValueError: If any parameter is outside its valid range, or if
            ``mi_weight`` is passed at all -- the term it weighted was added
            with the wrong sign and has been removed, not reweighted.
        """
        # --- Parameter Validation (before super().__init__ so no half-built
        # --- object can escape a raise) ---
        # DECISION plan-2026-09-02T081011-9b26b501/D-001
        # Do NOT silently drop or silently accept mi_weight: an archived config
        # would then deserialize into a DIFFERENT objective. Do not flip its
        # sign and keep it either -- the term is not I(X;Y-hat) under any sign.
        if mi_weight is not None:
            raise ValueError(
                "mi_weight has been removed from GoodhartAwareLoss "
                f"(got mi_weight={mi_weight}). The term it weighted was "
                "H(mean_p) - mean H(p_i) added with a POSITIVE sign, so it was "
                "maximized at the accurate, confident classifier and minimized "
                "at marginal collapse -- the opposite of the intended effect. "
                "Use prior_weight (with the optional class_prior) for a "
                "correctly-signed anti-collapse term."
            )
        if not (0 <= label_smoothing < 1):
            raise ValueError(
                f"label_smoothing must be in the range [0, 1), "
                f"but got {label_smoothing}."
            )
        if not (isinstance(entropy_weight, (int, float)) and entropy_weight >= 0):
            raise ValueError(
                f"Entropy weight must be a non-negative number, "
                f"but got {entropy_weight}."
            )
        if not (isinstance(prior_weight, (int, float)) and prior_weight >= 0):
            raise ValueError(
                f"Prior weight must be a non-negative number, "
                f"but got {prior_weight}."
            )
        if not (0 < epsilon < 0.1):
            raise ValueError(
                f"Epsilon must be a small positive number in (0, 0.1), "
                f"but got {epsilon}."
            )

        normalized_prior = None
        if class_prior is not None:
            prior_array = np.asarray(class_prior, dtype="float64").reshape(-1)
            if prior_array.size == 0:
                raise ValueError("class_prior must be a non-empty sequence.")
            if not np.all(np.isfinite(prior_array)):
                raise ValueError(
                    f"class_prior entries must all be finite, but got {class_prior}."
                )
            if not np.all(prior_array > 0):
                raise ValueError(
                    f"class_prior entries must all be strictly positive, "
                    f"but got {class_prior}."
                )
            normalized_prior = prior_array / float(prior_array.sum())

        if entropy_weight > 0.5:
            warnings.warn(
                f"High entropy_weight ({entropy_weight}) may dominate training. "
                "Consider values in [0.001, 0.5].", UserWarning
            )

        super().__init__(name=name, reduction=reduction, dtype=dtype)

        self.label_smoothing = float(label_smoothing)
        self.entropy_weight = float(entropy_weight)
        self.prior_weight = float(prior_weight)
        self.class_prior = None if class_prior is None else list(class_prior)
        self.from_logits = bool(from_logits)
        self.epsilon = float(epsilon)

        # DECISION plan-2026-09-02T081011-9b26b501/D-004
        # Keep the user's dtype ARGUMENT, not `self.dtype`. Keras 3's
        # Loss.get_config() emits only {name, reduction}, and `self.dtype`
        # resolves None to the live global policy -- emitting it would pin that
        # policy into every saved config. Do not "simplify" to self.dtype.
        self._dtype_arg = dtype
        # Normalized prior kept separately so get_config() can round-trip the
        # sequence the caller actually passed.
        self._normalized_prior = normalized_prior

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Computes the Goodhart-aware loss for a batch.

        :param y_true: Ground truth labels, shape (batch_size, num_classes).
        :type y_true: keras.KerasTensor
        :param y_pred: Predicted logits or probabilities, shape
            (batch_size, num_classes).
        :type y_pred: keras.KerasTensor
        :return: A tensor of shape (batch_size,) holding the per-sample loss.
            At ``prior_weight > 0`` a single batch-level scalar is added to
            every entry, so the returned vector no longer decomposes per row.
        :rtype: keras.KerasTensor
        """
        y_true = keras.ops.cast(y_true, dtype=y_pred.dtype)

        # --- Component 1: Standard Cross-Entropy Loss ---
        # This component drives task accuracy and incorporates label smoothing.
        per_sample_loss = keras.losses.categorical_crossentropy(
            y_true=y_true,
            y_pred=y_pred,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing
        )

        # Every regularizer is derived from ONE producer of log p, so the
        # conditional entropy cannot be computed two different ways.
        log_probs = self._log_probabilities(y_pred)

        # --- Component 2: Confidence Penalty (per sample) ---
        # Maximizing H(p_i) is minimizing -H(p_i). It rides INSIDE the
        # (batch,) vector so sample_weight weights it row-wise.
        if self.entropy_weight > 0:
            per_sample_loss = per_sample_loss - (
                self.entropy_weight * self._conditional_entropy(log_probs)
            )

        # --- Component 3: Anti-collapse term (batch level) ---
        if self.prior_weight > 0:
            per_sample_loss = per_sample_loss + (
                self.prior_weight * self._prior_matching_regularization(log_probs)
            )

        return per_sample_loss

    def _log_probabilities(
        self,
        y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Produces log-probabilities, the single source of every regularizer.

        On the ``from_logits=True`` path this is ``log_softmax``, exact for
        saturated logits where ``softmax -> clip -> log`` floored the entropy
        at ``(K-1) * epsilon * |log epsilon|`` and zeroed its gradient. On the
        ``from_logits=False`` path probabilities are clipped away from zero and
        RENORMALIZED so each row still sums to one.

        :param y_pred: Predicted logits or probabilities, shape
            (batch_size, num_classes).
        :type y_pred: keras.KerasTensor
        :return: Log-probabilities of the same shape as `y_pred`.
        :rtype: keras.KerasTensor
        """
        if self.from_logits:
            return keras.ops.log_softmax(y_pred, axis=-1)

        # DECISION plan-2026-09-02T081011-9b26b501/D-005
        # `epsilon` defaults to 1e-8, BELOW float16's smallest normal (6.1e-5),
        # so a bare clip(y, self.epsilon, ...) floors to 0.0 under
        # mixed_float16 and log() returns -inf. Do not restore the literal.
        dtype = keras.backend.standardize_dtype(y_pred.dtype)
        floor = max(self.epsilon, float(np.finfo(dtype).tiny))
        probs = keras.ops.clip(y_pred, floor, 1.0)
        probs = probs / keras.ops.sum(probs, axis=-1, keepdims=True)
        return keras.ops.log(probs)

    def _conditional_entropy(
        self,
        log_probs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Computes the Shannon entropy of each row, :math:`H(\\hat{Y}|X=x_i)`.

        :param log_probs: Log-probabilities, shape (batch_size, num_classes).
        :type log_probs: keras.KerasTensor
        :return: Per-sample entropy, shape (batch_size,).
        :rtype: keras.KerasTensor
        """
        probs = keras.ops.exp(log_probs)
        return -keras.ops.sum(probs * log_probs, axis=-1)

    def _prior_matching_regularization(
        self,
        log_probs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Computes the anti-collapse term :math:`KL(q \\| \\bar{p})`.

        :math:`\\bar{p}` is the batch-marginal prediction and :math:`q` is
        `class_prior` (uniform when unset). The term is zero when the marginal
        matches the prior and grows without bound as the marginal collapses
        onto a strict subset of the classes, so a COLLAPSED marginal scores
        strictly higher -- the sign the removed `mi_weight` term had backwards.
        The value is irreducibly batch-level: it is computed over every row
        regardless of `sample_weight`.

        :param log_probs: Log-probabilities, shape (batch_size, num_classes).
        :type log_probs: keras.KerasTensor
        :return: A scalar tensor.
        :rtype: keras.KerasTensor
        :raises ValueError: If `class_prior`'s length does not match the
            statically known number of classes.
        """
        compute_dtype = log_probs.dtype
        batch_size = keras.ops.cast(keras.ops.shape(log_probs)[0], compute_dtype)

        # log of the batch-marginal, computed in log space so an underflowed
        # softmax column cannot become log(0).
        log_mean_probs = keras.ops.logsumexp(
            log_probs, axis=0
        ) - keras.ops.log(batch_size)

        if self._normalized_prior is None:
            # Uniform q: KL = -log(K) - mean_k log(mean_p_k).
            num_classes = keras.ops.cast(
                keras.ops.shape(log_probs)[-1], compute_dtype
            )
            return -keras.ops.log(num_classes) - keras.ops.mean(log_mean_probs)

        static_classes = log_probs.shape[-1]
        if static_classes is not None and static_classes != self._normalized_prior.size:
            raise ValueError(
                f"class_prior has length {self._normalized_prior.size} but "
                f"y_pred has {static_classes} classes."
            )
        prior = keras.ops.convert_to_tensor(
            self._normalized_prior, dtype=compute_dtype
        )
        log_prior = keras.ops.convert_to_tensor(
            np.log(self._normalized_prior), dtype=compute_dtype
        )
        return keras.ops.sum(prior * (log_prior - log_mean_probs))

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration dictionary for serialization.

        ``super().get_config()`` returns only ``{name, reduction}`` on Keras 3,
        so ``dtype`` is emitted explicitly -- and it is the caller's argument,
        not the resolved ``self.dtype``, so a config saved under one global
        policy does not pin that policy on reload.

        Note that at ``prior_weight > 0`` this loss is batch-coupled: the
        anti-collapse term is one scalar computed over ALL rows and added to
        every entry of the returned vector, so the loss does NOT decompose per
        row and zeroing a row's ``sample_weight`` is NOT equivalent to dropping
        that row. At the default ``prior_weight = 0.0`` it decomposes exactly.

        :return: A dictionary of the loss function's configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'label_smoothing': self.label_smoothing,
            'entropy_weight': self.entropy_weight,
            'prior_weight': self.prior_weight,
            'class_prior': self.class_prior,
            'from_logits': self.from_logits,
            'epsilon': self.epsilon,
            'dtype': self._dtype_arg,
        })
        return config


def analyze_loss_components(
    loss_fn: GoodhartAwareLoss,
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor
) -> Dict[str, float]:
    """
    Analyzes individual components of the GoodhartAwareLoss for debugging.

    :param loss_fn: An instance of the GoodhartAwareLoss function.
    :type loss_fn: GoodhartAwareLoss
    :param y_true: Ground truth labels.
    :type y_true: keras.KerasTensor
    :param y_pred: Predicted logits or probabilities.
    :type y_pred: keras.KerasTensor
    :return: A dictionary containing, for each component, its unweighted value,
        its weighted value, the absolute magnitude of that weighted value
        (``*_magnitude``) and its bounded share of the total magnitude
        (``*_share``, in [0, 1], summing to one). ``total_loss`` is the same
        number the live loss returns for the same inputs.
    :rtype: Dict[str, float]
    """
    y_true = keras.ops.cast(y_true, dtype=y_pred.dtype)

    # Calculate individual components using the loss function's settings
    ce_loss = keras.losses.categorical_crossentropy(
        y_true=y_true,
        y_pred=y_pred,
        from_logits=loss_fn.from_logits,
        label_smoothing=loss_fn.label_smoothing
    )

    # Same producer of log p that call() uses, so the diagnostic cannot drift
    # away from the live loss.
    log_probs = loss_fn._log_probabilities(y_pred)

    entropy_term_unweighted = -keras.ops.mean(loss_fn._conditional_entropy(log_probs))
    prior_term_unweighted = loss_fn._prior_matching_regularization(log_probs)

    # Compute weighted contributions
    entropy_term_weighted = loss_fn.entropy_weight * entropy_term_unweighted
    prior_term_weighted = loss_fn.prior_weight * prior_term_unweighted
    total_loss = keras.ops.mean(ce_loss + entropy_term_weighted + prior_term_weighted)

    # Convert tensors to Python floats for easy inspection
    results = {
        'total_loss': float(keras.ops.convert_to_numpy(total_loss)),
        'cross_entropy': float(keras.ops.convert_to_numpy(keras.ops.mean(ce_loss))),
        'entropy_term_unweighted': float(keras.ops.convert_to_numpy(entropy_term_unweighted)),
        'prior_term_unweighted': float(keras.ops.convert_to_numpy(prior_term_unweighted)),
        'entropy_term_weighted': float(keras.ops.convert_to_numpy(entropy_term_weighted)),
        'prior_term_weighted': float(keras.ops.convert_to_numpy(prior_term_weighted)),
        'label_smoothing': loss_fn.label_smoothing,
        'entropy_weight': loss_fn.entropy_weight,
        'prior_weight': loss_fn.prior_weight
    }

    # DECISION plan-2026-09-02T081011-9b26b501/D-006
    # Do NOT restore `component / total_loss * 100`. The total is a SIGNED sum
    # of signed terms, so that ratio is unbounded: at a negative total it reads
    # ce = -100% with entropy = +200%. Divide magnitudes by the sum of
    # magnitudes instead -- every share is then in [0, 1] and they sum to one.
    magnitudes = {
        'ce_magnitude': abs(results['cross_entropy']),
        'entropy_magnitude': abs(results['entropy_term_weighted']),
        'prior_magnitude': abs(results['prior_term_weighted']),
    }
    results.update(magnitudes)

    total_magnitude = sum(magnitudes.values())
    results['total_magnitude'] = total_magnitude
    if total_magnitude > 0.0:
        results.update({
            'ce_share': magnitudes['ce_magnitude'] / total_magnitude,
            'entropy_share': magnitudes['entropy_magnitude'] / total_magnitude,
            'prior_share': magnitudes['prior_magnitude'] / total_magnitude,
        })
    else:
        # Every component is exactly zero; no component has a share of the
        # total, and 0/0 is not 1/3.
        results.update({
            'ce_share': 0.0,
            'entropy_share': 0.0,
            'prior_share': 0.0,
        })
    return results

# ---------------------------------------------------------------------
