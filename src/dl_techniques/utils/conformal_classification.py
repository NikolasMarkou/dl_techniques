"""Split-conformal prediction sets for a classifier (Adaptive Prediction Sets).

Turns the logits of any trained classifier into a prediction set that contains
the true class with probability at least ``1 - alpha``, without touching the
model. The score is APS (Romano, Sesia and Candes, 2020;
``research/conformal_prediction.md``): for a sample with class probabilities
``p`` sorted in descending order, the score of label ``y`` at sorted position
``k`` is the mass of the strictly more likely classes plus ``u`` times its own
probability::

    E(x, y, u) = sum_{j < k} p_(j) + u * p_(k)

With ``randomized=True``, ``u`` is drawn uniformly per sample (one draw, reused for
every candidate label of that sample), which gives exact coverage. With
``randomized=False``, ``u = 1`` (inclusive mass): deterministic and still valid; its
scores are the larger ones, so it calibrates to a larger ``tau`` and over-covers a
little. The deterministic rule is the one the in-graph
:class:`~dl_techniques.layers.statistics.conformal.prediction_set.ConformalPredictionSet`
runs, so the layer and :func:`predict_sets` agree exactly.

The score of a label is non-decreasing in its sorted position, so the set is
``{y : E(x, y, u) <= tau}``.

Guarantees and caveats:

* **Marginal coverage.** ``P(y in set) >= 1 - alpha`` on average over the
  population, not for every input or every class. :func:`evaluate_sets` reports
  the class-conditional minimum so the gap is visible.
* **Exchangeability and a fixed model.** The calibration data and the data the sets
  are used on must be exchangeable, and the model must not have been chosen using
  the calibration data. A checkpoint picked on the same validation set that is then
  used to calibrate weakens the guarantee.
* **Small calibration sets.** When ``ceil((n + 1) * (1 - alpha)) > n`` no finite
  threshold can guarantee coverage, so ``tau`` is ``+inf`` and every label is in
  every set (see :func:`~dl_techniques.utils.conformal_denoiser_intervals.conformal_quantile`).
* **Ties.** Equal probabilities are ordered by class index (stable sort), in the
  calibration and in the prediction alike.
* **Empty sets** are allowed. Forcing the top label in would only raise coverage,
  but it would also change what the in-graph layer computes, so it is not done.

References:
    - Romano, Sesia and Candes, 2020. Classification with Valid and Adaptive
      Coverage. NeurIPS. (https://arxiv.org/abs/2006.02544)
    - Angelopoulos and Bates, 2021. A Gentle Introduction to Conformal
      Prediction and Distribution-Free Uncertainty Quantification.
      (https://arxiv.org/abs/2107.07511)
"""

from typing import Dict, Optional

import numpy as np

from dl_techniques.utils.conformal_denoiser_intervals import conformal_quantile

# ---------------------------------------------------------------------


def _check_logits_labels(logits: np.ndarray, labels: Optional[np.ndarray]) -> None:
    if logits.ndim != 2:
        raise ValueError(f"logits must be (N, C), got shape {logits.shape}")
    if labels is None:
        return
    if labels.shape != (logits.shape[0],):
        raise ValueError(
            f"labels must be ({logits.shape[0]},), got shape {labels.shape}"
        )
    if labels.size and (labels.min() < 0 or labels.max() >= logits.shape[1]):
        raise ValueError(
            f"labels must be in [0, {logits.shape[1]}), got range "
            f"[{labels.min()}, {labels.max()}]"
        )


def softmax(logits: np.ndarray) -> np.ndarray:
    """Max-shifted softmax over the last axis, in float64."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _draw_u(n: int, randomized: bool, seed: Optional[int]) -> np.ndarray:
    if not randomized:
        return np.ones(n, dtype=np.float64)
    return np.random.default_rng(seed).random(n)


def label_scores(logits: np.ndarray, u: np.ndarray) -> np.ndarray:
    """APS score of every label for every sample, ``(N, C)`` in class order.

    :param logits: ``(N, C)`` logits.
    :param u: ``(N,)`` per-sample randomization in ``[0, 1]``.
    """
    probs = softmax(logits)
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_p = np.take_along_axis(probs, order, axis=1)
    below = np.cumsum(sorted_p, axis=1) - sorted_p
    scores_sorted = below + u[:, None] * sorted_p
    scores = np.empty_like(scores_sorted)
    np.put_along_axis(scores, order, scores_sorted, axis=1)
    return scores


def aps_scores(
        logits: np.ndarray,
        labels: np.ndarray,
        randomized: bool = True,
        seed: Optional[int] = None,
) -> np.ndarray:
    """APS conformity score of each sample's true label, ``(N,)``."""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    _check_logits_labels(logits, labels)
    u = _draw_u(logits.shape[0], randomized, seed)
    return label_scores(logits, u)[np.arange(logits.shape[0]), labels]


def calibrate_aps(
        logits: np.ndarray,
        labels: np.ndarray,
        alpha: float,
        randomized: bool = True,
        seed: Optional[int] = None,
) -> float:
    """Threshold ``tau`` giving coverage ``>= 1 - alpha`` (``+inf`` if ``n`` is too small)."""
    return conformal_quantile(aps_scores(logits, labels, randomized, seed), alpha)


def predict_sets(
        logits: np.ndarray,
        tau: float,
        randomized: bool = True,
        seed: Optional[int] = None,
) -> np.ndarray:
    """Boolean ``(N, C)`` membership: ``True`` where the label is in the set."""
    logits = np.asarray(logits)
    _check_logits_labels(logits, None)
    u = _draw_u(logits.shape[0], randomized, seed)
    return label_scores(logits, u) <= tau


def evaluate_sets(
        logits: np.ndarray,
        labels: np.ndarray,
        sets: np.ndarray,
) -> Dict[str, float]:
    """Coverage and size statistics of ``sets`` against the true ``labels``.

    Returns marginal ``coverage``, ``mean_size``, ``median_size``, ``empty_rate``,
    ``size_given_covered``, the class-conditional ``class_coverage_min`` and
    ``class_coverage_p10`` (over classes with at least one sample) and the top-1
    ``accuracy`` for reference.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    _check_logits_labels(logits, labels)
    if sets.shape != logits.shape:
        raise ValueError(f"sets must be {logits.shape}, got {sets.shape}")
    n = logits.shape[0]
    covered = sets[np.arange(n), labels]
    size = sets.sum(axis=1)

    present = np.unique(labels)
    per_class = np.array([covered[labels == c].mean() for c in present])
    return {
        "coverage": float(covered.mean()),
        "mean_size": float(size.mean()),
        "median_size": float(np.median(size)),
        "empty_rate": float((size == 0).mean()),
        "size_given_covered": float(size[covered].mean()) if covered.any() else float("nan"),
        "class_coverage_min": float(per_class.min()),
        "class_coverage_p10": float(np.percentile(per_class, 10)),
        "accuracy": float((logits.argmax(axis=1) == labels).mean()),
    }


def repeated_split_coverage(
        logits: np.ndarray,
        labels: np.ndarray,
        alpha: float,
        repeats: int = 20,
        seed: int = 0,
        randomized: bool = True,
) -> Dict[str, float]:
    """Empirical check of the guarantee over random 50/50 calibrate/evaluate splits.

    Each repeat permutes the samples, calibrates ``tau`` on the first half and
    measures coverage and set size on the second. Returns the mean, minimum and
    maximum coverage, the mean set size and empty-set rate over the repeats, and the
    ``n_calibration`` / ``n_evaluation`` of one split.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    _check_logits_labels(logits, labels)
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    n = logits.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 samples to split, got {n}")
    rng = np.random.default_rng(seed)
    half = n // 2
    cov, size, empty = [], [], []
    for r in range(repeats):
        perm = rng.permutation(n)
        cal, ev = perm[:half], perm[half:]
        tau = calibrate_aps(logits[cal], labels[cal], alpha, randomized, seed=seed + 2 * r)
        sets = predict_sets(logits[ev], tau, randomized, seed=seed + 2 * r + 1)
        stats = evaluate_sets(logits[ev], labels[ev], sets)
        cov.append(stats["coverage"])
        size.append(stats["mean_size"])
        empty.append(stats["empty_rate"])
    return {
        "coverage_mean": float(np.mean(cov)),
        "coverage_min": float(np.min(cov)),
        "coverage_max": float(np.max(cov)),
        "mean_size": float(np.mean(size)),
        "empty_rate": float(np.mean(empty)),
        "n_calibration": half,
        "n_evaluation": n - half,
        "repeats": repeats,
    }
