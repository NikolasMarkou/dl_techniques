"""Post-hoc temperature scaling of a classifier's logits.

A classifier trained with cross-entropy, or with a confidence penalty such as
:class:`~dl_techniques.losses.confidence_penalty_loss.ConfidencePenaltyLoss`,
is rarely calibrated: the loss sees training labels only, and calibration is a
property of held-out data. :func:`fit_temperature` closes that gap after
training. It finds the single scalar ``T > 0`` that minimises the negative
log-likelihood of ``softmax(logits / T)`` on a labelled set::

    probs = softmax(test_logits / T)

Dividing every logit by one positive scalar never changes the argmax, so
accuracy is untouched; only the confidence moves. ``T > 1`` softens an
overconfident model, ``T < 1`` sharpens an underconfident one.

Use VALIDATION data, never training data. On a memorised training set the
optimum is degenerate (``T`` runs to the bottom of the search range) and the
fitted value says nothing about unseen inputs. The same holds for a validation
set that also chose the checkpoint: hold out a split used for neither.

The result is a number, not a verdict. Check it with
:func:`~dl_techniques.analyzer.calibration_metrics.compute_ece` on a third
split, before and after dividing by ``T`` (``compute_ece`` takes probabilities,
so pass ``softmax(logits / T)``).

The search is a golden-section search over ``log(1 / T)`` on ``[-5, 5]``, so
``T`` lies in ``[exp(-5), exp(5)]``, about ``[0.0067, 148]``. The NLL is
convex in ``1 / T``, so the search converges to the global optimum. When the
optimum is the edge of that range (a perfectly separable or an always-wrong
set, where the true optimum is outside it) the edge is returned and a warning
says so.
"""

from typing import Any

import numpy as np

from dl_techniques.utils.conformal_classification import _check_logits_labels
from dl_techniques.utils.logger import logger
from dl_techniques.utils.validation import positive_int

# ---------------------------------------------------------------------

_LOG_BETA_RANGE = 5.0
"""Search range of ``log(1 / T)``: ``[-5, 5]``, so ``T`` is in ``[e^-5, e^5]``."""

_EDGE_TOLERANCE = 1e-6
"""A ``log(1 / T)`` within this of the range limit counts as sitting on the edge."""

_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0

# ---------------------------------------------------------------------


def _validated(logits: Any, labels: Any):
    """Return ``(logits float64 (N, C), labels int64 (N,))`` or raise ``ValueError``."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    # DECISION plan-2026-11-11T131349-13b28b37/D-043
    # One-hot (or any 2-D) labels are REFUSED, never flattened or argmax-ed. The
    # first version of this fit read labels as ``reshape(-1).astype(int)``, which
    # turns an (N, C) one-hot array into N * C garbage class indices and fails
    # (or worse, fits) far from the cause. Do not add a "helpful" argmax here:
    # soft or smoothed labels are not class indices and would be silently wrong.
    if labels.ndim == 2:
        raise ValueError(
            "labels must be integer class indices of shape (N,), got shape "
            f"{labels.shape}: one-hot or soft labels are not accepted, pass "
            "labels.argmax(axis=1) for one-hot labels"
        )
    if labels.dtype.kind not in "iu":
        raise ValueError(
            f"labels must be integer class indices, got dtype {labels.dtype}"
        )
    _check_logits_labels(logits, labels)
    if logits.shape[0] < 1:
        raise ValueError("need at least one validation sample, got none")
    if logits.shape[1] < 2:
        raise ValueError(
            f"need at least 2 classes, got logits of shape {logits.shape}"
        )
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite (found NaN or inf)")
    return logits, labels.astype(np.int64)


def fit_temperature(logits: Any, labels: Any, iters: int = 100) -> float:
    """Fit the temperature ``T > 0`` that minimises held-out NLL.

    Fit on VALIDATION data, never on training data (see the module docstring),
    then divide logits by ``T`` at inference: ``softmax(logits / T)``. Accuracy
    is unchanged; check calibration with
    :func:`~dl_techniques.analyzer.calibration_metrics.compute_ece`. Pairs with
    :class:`~dl_techniques.losses.confidence_penalty_loss.ConfidencePenaltyLoss`,
    which does not produce calibrated probabilities on its own.

    Interface contract:

    * ``logits``: array-like ``(N, C)`` of finite numbers, ``N >= 1``,
      ``C >= 2``; read as float64.
    * ``labels``: array-like ``(N,)`` of integer class indices in ``[0, C)``.
      One-hot, soft, float and bool labels are refused.
    * ``iters``: positive ``int``, golden-section iterations (the bracket
      shrinks by 0.618 per iteration; 100 is far past float precision).
    * Returns a Python ``float`` in ``[exp(-5), exp(5)]``.
    * Raises ``ValueError`` for every input above that violates its rule.
    * Logs a ``logger.warning`` and still returns the edge value when the
      optimum lies at the edge of the range: the validation set is perfectly
      separable (``T`` at the lower edge) or always wrong (upper edge), and the
      returned ``T`` is then a bound, not an estimate.

    :param logits: ``(N, C)`` validation logits.
    :param labels: ``(N,)`` integer validation labels.
    :param iters: golden-section iterations.
    :return: the fitted temperature.
    """
    iters = positive_int(iters, "iters", "fit_temperature")
    z, y = _validated(logits, labels)
    rows = np.arange(y.shape[0])

    def nll(log_beta: float) -> float:
        # Max-shifted log-sum-exp in float64: stable for any logit scale.
        s = z * np.exp(log_beta)
        s = s - s.max(axis=1, keepdims=True)
        return float(-(s[rows, y] - np.log(np.exp(s).sum(axis=1))).mean())

    lo, hi = -_LOG_BETA_RANGE, _LOG_BETA_RANGE
    a, b = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(iters):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - _GOLDEN * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + _GOLDEN * (hi - lo)
            fb = nll(b)
    log_beta = (lo + hi) / 2.0
    temperature = float(np.exp(-log_beta))

    # DECISION plan-2026-11-11T131349-13b28b37/D-043
    # The edge is returned AND announced. For a perfectly separable or an
    # always-wrong validation set (and, before the finite check, NaN logits) the
    # NLL has no interior minimum and the search ends on the range limit with
    # no sign of it: exp(-5) = 0.0067 looks like a fitted value. Do not raise
    # (a bound is still the right answer to apply) and do not drop the warning.
    if abs(log_beta) >= _LOG_BETA_RANGE - _EDGE_TOLERANCE:
        side = "lower (separable data)" if log_beta > 0 else "upper (always-wrong data)"
        logger.warning(
            f"fit_temperature: the optimum sits at the {side} edge of the "
            f"search range, T = {temperature:.5g} in "
            f"[{np.exp(-_LOG_BETA_RANGE):.3g}, {np.exp(_LOG_BETA_RANGE):.3g}]. "
            "The validation set is degenerate (perfectly separable or always "
            "wrong, or it is training data), so T is a bound, not an estimate. "
            "Fit on a held-out validation set."
        )
    return temperature
