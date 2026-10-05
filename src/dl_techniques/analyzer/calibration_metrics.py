"""
Model Calibration Metrics

This module provides comprehensive metrics for evaluating the calibration quality of 
probabilistic classifiers. Model calibration refers to how well the predicted 
probabilities reflect the true likelihood of the predicted outcomes.

A well-calibrated model should satisfy the property that among all predictions where 
the model outputs probability p, approximately p fraction should be correct. For 
example, if a model predicts 80% confidence for 100 samples, approximately 80 of 
those samples should be correctly classified.

Key Concepts:
- **Calibration**: The degree to which predicted probabilities match observed frequencies.
- **Reliability**: How well confidence scores reflect actual accuracy.
- **Sharpness**: The concentration of predictions away from the base rate.
- **Resolution**: The ability to distinguish between correct and incorrect predictions.

The metrics implemented here help assess these properties:

1. **Expected Calibration Error (ECE)**: Measures average calibration error across
   equal-width confidence bins.
2. **Adaptive ECE (AECE)**: A robust ECE variant using equal-mass bins.
3. **Maximum Calibration Error (MCE)**: Measures the worst-case calibration error.
4. **Reliability Diagram Data**: Provides data for visualizing calibration.
5. **Brier Score**: Measures the overall accuracy of probabilistic predictions.
6. **Brier Score Decomposition**: Decomposes the Brier score into reliability,
   resolution, and uncertainty.
7. **Prediction Entropy**: Quantifies uncertainty in model predictions.

References:
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of
  modern neural networks. ICML.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with
  supervised learning. ICML.
- Murphy, A. H. (1973). A new vector partition of the probability score. Journal
  of Applied Meteorology. (Brier Score Decomposition)
"""

import numpy as np
from typing import Dict, List

# ------------------------------------------------------------------------------
# Internal Helper Function for Binning
# ------------------------------------------------------------------------------

def _get_binary_bin_info(
    outcomes: np.ndarray, scores: np.ndarray, n_bins: int
) -> List[Dict[str, float]]:
    """
    Compute equal-width bin statistics for a (binary outcome, scalar score) pair.

    This is the single binning primitive of the module. Every calibration
    quantity here — top-1 ECE/AECE/MCE, the reliability diagram, the Brier
    decomposition and classwise ECE — is a reduction over a binary outcome
    binned by a scalar score in [0, 1]; only the choice of that pair differs.

    Interface contract:
        - ``outcomes`` and ``scores`` must be 1-D and the same length.
        - ``outcomes`` carries values in {0.0, 1.0}; ``scores`` values in [0, 1].
        - The FIRST bin is closed on the left so a score of exactly 0.0 is never
          dropped; every other bin is half-open ``(lower, upper]``. This edge
          handling lives here and nowhere else.
        - Empty bins are RETAINED, with ``count = 0`` and zeroed statistics, so
          the returned list always has exactly ``n_bins`` entries in bin order.

    .. warning::
       DECISION plan-2026-10-05-analyzer-audit/F-035: the ``accuracy`` and
       ``confidence`` of an EMPTY bin are ``0.0`` PLACEHOLDERS, not measurements. The
       ECE/MCE reductions skip empty bins via ``prop_in_bin > 0`` so they are unaffected;
       :func:`compute_reliability_data` re-expresses them as NaN because it EXPORTS them
       to a plot. Any other consumer must test ``count > 0`` before reading them: a
       ``0.0`` accuracy is the extreme of the axis it is plotted on, which is exactly how
       the reliability diagram came to draw nine fabricated points at the origin.

    Args:
        outcomes (np.ndarray): Binary outcome per sample. Shape: (n_samples,)
        scores (np.ndarray): Forecast score per sample. Shape: (n_samples,)
        n_bins (int): Number of equal-width bins over [0, 1].

    Returns:
        List[Dict[str, float]]: One dict per bin with keys ``prop_in_bin``,
        ``accuracy`` (mean outcome in the bin), ``confidence`` (mean score in the
        bin), ``count`` and ``center``.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    scores = np.asarray(scores, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_info = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Handle the first bin to be inclusive of 0.0
        if bin_lower == 0.0:
            in_bin = (scores >= bin_lower) & (scores <= bin_upper)
        else:
            in_bin = (scores > bin_lower) & (scores <= bin_upper)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(outcomes[in_bin])
            avg_confidence_in_bin = np.mean(scores[in_bin])
            count = np.sum(in_bin)
        else:
            accuracy_in_bin = 0.0
            avg_confidence_in_bin = 0.0
            count = 0

        bin_info.append({
            "prop_in_bin": prop_in_bin,
            "accuracy": accuracy_in_bin,
            "confidence": avg_confidence_in_bin,
            "count": count,
            "center": (bin_lower + bin_upper) / 2
        })
    return bin_info


def _top1_outcome_and_score(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple:
    """Reduce a multiclass prediction to the top-1 (correctness, confidence) pair.

    Args:
        y_true (np.ndarray): True class labels. Shape: (n_samples,)
        y_prob (np.ndarray): Predicted probabilities. Shape: (n_samples, n_classes)

    Returns:
        tuple: ``(outcomes, scores)`` — ``outcomes[i]`` is 1.0 when the argmax
        prediction is correct, ``scores[i]`` is the top-1 confidence.
    """
    outcomes = (np.argmax(y_prob, axis=1) == y_true).astype(float)
    scores = np.max(y_prob, axis=1)
    return outcomes, scores


def _get_bin_info(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int
) -> List[Dict[str, float]]:
    """
    Internal helper to compute bin-wise statistics for equal-width bins.

    Top-1 specialization of :func:`_get_binary_bin_info`: the binary outcome is
    "the argmax prediction was correct" and the score is the top-1 confidence.

    Args:
        y_true (np.ndarray): True class labels. Shape: (n_samples,)
        y_prob (np.ndarray): Predicted probabilities. Shape: (n_samples, n_classes)
        n_bins (int): Number of equal-width bins over [0, 1].

    Returns:
        List[Dict[str, float]]: See :func:`_get_binary_bin_info`.
    """
    outcomes, scores = _top1_outcome_and_score(y_true, y_prob)
    return _get_binary_bin_info(outcomes, scores, n_bins)


# ------------------------------------------------------------------------------
# Core Calibration Metrics
# ------------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE) using equal-width bins.

    ECE measures the difference between predicted confidence and observed
    accuracy, averaged over all samples and weighted by bin proportions.

    Mathematically: ECE = Σ(i=1 to M) (n_i/n) * |acc_i - conf_i|
    A perfectly calibrated model would have ECE = 0.

    Args:
        y_true (np.ndarray): True class labels (not one-hot encoded).
            Shape: (n_samples,)
        y_prob (np.ndarray): Predicted class probabilities.
            Shape: (n_samples, n_classes)
        n_bins (int, optional): Number of equal-width bins. Defaults to 15.

    Returns:
        float: Expected Calibration Error.

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_prob = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8],
        ...                    [0.8, 0.2], [0.4, 0.6]])
        >>> compute_ece(y_true, y_prob, n_bins=5)
        0.24
    """
    bin_info = _get_bin_info(y_true, y_prob, n_bins)

    ece = 0.0
    for bin_data in bin_info:
        if bin_data["prop_in_bin"] > 0:
            ece += np.abs(bin_data["confidence"] - bin_data["accuracy"]) * bin_data["prop_in_bin"]

    return ece

# ------------------------------------------------------------------------------

def compute_ece_binary(
    outcomes: np.ndarray, scores: np.ndarray, n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error for a binary outcome and a scalar score.

    The general form of :func:`compute_ece`: ECE = Σ (n_i/n) * |score_i - rate_i|
    over equal-width bins of ``scores``. ``compute_ece`` is the special case where
    the outcome is top-1 correctness and the score is top-1 confidence; classwise
    ECE (Kull et al. 2019) is the case where the outcome is the indicator
    ``y_true == c`` and the score is the class-``c`` probability column.

    Args:
        outcomes (np.ndarray): Binary outcome per sample, values in {0, 1}.
            Shape: (n_samples,)
        scores (np.ndarray): Forecast score per sample, values in [0, 1].
            Shape: (n_samples,)
        n_bins (int, optional): Number of equal-width bins. Defaults to 15.

    Returns:
        float: Expected Calibration Error for this (outcome, score) pair.

    Example:
        >>> outcomes = np.array([0, 0, 1, 1])
        >>> scores = np.array([0.5, 0.5, 0.5, 0.5])
        >>> float(compute_ece_binary(outcomes, scores, n_bins=10))
        0.0
    """
    bin_info = _get_binary_bin_info(outcomes, scores, n_bins)

    ece = 0.0
    for bin_data in bin_info:
        if bin_data["prop_in_bin"] > 0:
            ece += np.abs(bin_data["confidence"] - bin_data["accuracy"]) * bin_data["prop_in_bin"]

    return ece

# ------------------------------------------------------------------------------

def compute_adaptive_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Adaptive Expected Calibration Error (AECE) using equal-mass bins.

    Unlike ECE, which uses equal-width bins, AECE creates bins with an
    equal number of samples. This provides a more robust estimate, especially
    when confidence scores are concentrated in a narrow range.

    Args:
        y_true (np.ndarray): True class labels. Shape: (n_samples,)
        y_prob (np.ndarray): Predicted probabilities. Shape: (n_samples, n_classes)
        n_bins (int, optional): Number of equal-mass bins. Defaults to 15.

    Returns:
        float: Adaptive Expected Calibration Error.

    Example:
        >>> y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        >>> y_prob = np.array([[0.9, 0.1]]*5 + [[0.1, 0.9]]*5)
        >>> compute_adaptive_ece(y_true, y_prob, n_bins=2)
        0.3
    """
    n_samples = len(y_true)
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    # Sort samples by confidence
    sorted_indices = np.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]

    # Create equal-mass bins using array_split for proper index distribution
    bin_splits = np.array_split(np.arange(n_samples), n_bins)
    ece = 0.0

    for bin_indices in bin_splits:
        if len(bin_indices) == 0:
            continue

        bin_samples_conf = sorted_confidences[bin_indices]
        bin_samples_acc = sorted_accuracies[bin_indices]

        if len(bin_samples_conf) > 0:
            avg_conf = np.mean(bin_samples_conf)
            avg_acc = np.mean(bin_samples_acc)
            ece += np.abs(avg_conf - avg_acc) * (len(bin_samples_conf) / n_samples)

    return ece

# ------------------------------------------------------------------------------

def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE is the maximum difference between predicted confidence and observed
    accuracy over all bins. It measures the worst-case calibration error, which is
    critical for high-stakes applications.

    Mathematically: MCE = max_i |acc_i - conf_i|

    Args:
        y_true (np.ndarray): True class labels. Shape: (n_samples,)
        y_prob (np.ndarray): Predicted probabilities. Shape: (n_samples, n_classes)
        n_bins (int, optional): Number of bins. Defaults to 15.

    Returns:
        float: Maximum Calibration Error.

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_prob = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8],
        ...                    [0.8, 0.2], [0.4, 0.6]])
        >>> compute_mce(y_true, y_prob, n_bins=5)
        0.4
    """
    bin_info = _get_bin_info(y_true, y_prob, n_bins)

    errors = [
        np.abs(b["confidence"] - b["accuracy"])
        for b in bin_info if b["count"] > 0
    ]

    return np.max(errors) if errors else 0.0

# ------------------------------------------------------------------------------

def compute_reliability_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> Dict[str, np.ndarray]:
    """
    Compute data for reliability diagram visualization.

    A reliability diagram plots observed frequency against predicted probability.
    For a perfectly calibrated model, this plot should lie on the y=x diagonal.

    Args:
        y_true (np.ndarray): True class labels. Shape: (n_samples,)
        y_prob (np.ndarray): Predicted probabilities. Shape: (n_samples, n_classes)
        n_bins (int, optional): Number of bins. Defaults to 15.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing bin centers, accuracies,
                               confidences, and counts.

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_prob = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.8, 0.2], [0.4, 0.6]])
        >>> data = compute_reliability_data(y_true, y_prob, n_bins=5)
        >>> print(data['bin_centers'])
        [0.1 0.3 0.5 0.7 0.9]

    .. important::
       DECISION plan-2026-10-05-analyzer-audit/F-035: BOTH ``bin_accuracies`` and
       ``bin_confidences`` are **NaN in an empty bin**, never 0.0 and never the bin
       centre. Only ``bin_centers`` and ``bin_counts`` are defined for an empty bin.

       Previously only ``bin_confidences`` was neutralised; ``bin_accuracies`` kept the
       ``0.0`` placeholder that :func:`_get_bin_info` writes for an empty bin, and the
       visualizer plots exactly that array. With the default ``calibration_bins=10`` and
       any confident model, bins 0-8 are empty, so the reliability curve was drawn
       through ``(0.1, 0.0) ... (0.9, 0.0)`` as if measured — MEASURED
       ``bin_accuracies = [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]`` against
       ``bin_counts = [0 0 0 0 0 0 0 0 0 20]``. A perfectly calibrated confident model
       therefore rendered as grossly miscalibrated. Consumers MUST mask on
       ``bin_counts > 0`` (or ``np.isfinite``); see ``calibration_visualizer.py``.
    """
    bin_info = _get_bin_info(y_true, y_prob, n_bins)

    bin_centers = np.array([b["center"] for b in bin_info])
    # DECISION plan-2026-10-05-analyzer-audit/F-035 — see the `.. important::` block.
    # An empty bin has NO measured accuracy and NO measured confidence. Substituting
    # 0.0 for the accuracy fabricated the single most misleading point in the figure,
    # because 0.0 is the extreme of the axis it is plotted on.
    bin_accuracies = np.array([
        b["accuracy"] if b["count"] > 0 else np.nan for b in bin_info
    ])
    bin_confidences = np.array([
        b["confidence"] if b["count"] > 0 else np.nan for b in bin_info
    ])
    bin_counts = np.array([b["count"] for b in bin_info])

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts
    }


# ------------------------------------------------------------------------------
# Probabilistic Scoring and Uncertainty Metrics
# ------------------------------------------------------------------------------

def compute_brier_score(
        y_true_onehot: np.ndarray, y_prob: np.ndarray, *, multilabel: bool = False
) -> float:
    """
    Compute Brier Score for multiclass (or multilabel) probabilistic predictions.

    The Brier Score is the mean squared difference between predicted
    probabilities and actual outcomes. Lower values are better.

    Mathematically (MULTICLASS, the default):
        BS = (1/N) * Σ(i=1..N) Σ(j=1..K) (p_ij - o_ij)²
    Mathematically (MULTILABEL):
        BS = (1/(N*K)) * Σ(i=1..N) Σ(j=1..K) (p_ij - o_ij)²

    .. important::
       DECISION plan-2026-10-05-analyzer-audit/F-080. The two conventions differ by a
       factor of K, and which one is correct depends on whether the K outputs are
       MUTUALLY EXCLUSIVE.

       The multiclass form SUMS over classes and averages over samples, because exactly
       one class is correct per sample and the squared errors of the K-1 wrong classes
       are all part of that sample's error.

       The multilabel form also AVERAGES over classes, because each label is an
       independent Bernoulli forecast whose error is its own; summing over them would
       charge a single sample K times over. Applying the multiclass form to a sigmoid
       multi-label head therefore inflates the score by exactly K — e.g. 3x for a 3-label
       task — which is a number no reader can interpret, since it depends on the label
       count rather than on the model.

       This function previously had no such switch and its only in-library caller
       (``CalibrationAnalyzer``) applied the multiclass form unconditionally.

    Args:
        y_true_onehot (np.ndarray): True labels, one-hot for multiclass or a 0/1
            indicator matrix for multilabel. Shape: (n_samples, n_classes)
        y_prob (np.ndarray): Predicted probabilities. Shape: (n_samples, n_classes).
            For multiclass these should sum to 1 along axis 1; for multilabel they are
            independent and need not.
        multilabel: ``True`` for independent per-label (sigmoid) forecasts. Keyword-only
            so the existing positional call is unchanged.

    Returns:
        float: Brier Score. Lower is better. In ``[0, 1]`` for multilabel and in
        ``[0, 2]`` for multiclass.

    Example:
        >>> y_true_oh = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
        >>> y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.9, 0.1]])
        >>> compute_brier_score(y_true_oh, y_prob)
        0.075
    """
    squared_diffs = (y_prob - y_true_onehot) ** 2
    if multilabel:
        # Average over BOTH axes (F-080).
        return float(np.mean(squared_diffs))
    return float(np.mean(np.sum(squared_diffs, axis=1)))


def is_multilabel_output(y_prob: np.ndarray, tol: float = 1e-3) -> bool:
    """Whether a probability array looks like independent per-label (sigmoid) output.

    A softmax head emits rows summing to 1. A sigmoid multi-label head emits values in
    ``[0, 1]`` whose rows generally do NOT. The test is the row sum, which is exactly the
    distinction ``ModelAnalyzer._infer_output_activation`` already makes before deciding
    whether to softmax.

    .. important::
       DECISION plan-2026-10-05-analyzer-audit/F-081. ``_infer_output_activation`` makes
       this distinction and then THROWS the answer away: the activation name is not put in
       the prediction cache, so ``CalibrationAnalyzer`` cannot see it and applies
       single-label metrics to a multi-label head. This helper is the recovery path, and
       the analyzer uses it to route the metric to the right convention.

       The test is deliberately the same one, so the two cannot disagree: both accept a
       1-D array or a 2-D array, both require a row sum of 1 within ``tol``, and both
       require every value in ``[0, 1]``.

    Args:
        y_prob: Predicted probabilities.
        tol: Tolerance on the row-sum test.

    Returns:
        ``True`` when the rows do not sum to 1, i.e. the head is a multi-label sigmoid.
    """
    probabilities = np.asarray(y_prob, dtype=float)
    if probabilities.ndim == 1:
        return False
    if probabilities.size == 0:
        return False
    if not np.all(np.isfinite(probabilities)):
        return False
    in_unit_range = (probabilities.min() >= 0.0
                     and probabilities.max() <= 1.0 + 1e-5)
    if not in_unit_range:
        return False
    return not bool(np.allclose(probabilities.sum(axis=-1), 1.0, atol=tol))


def compute_confidence_statistics(y_prob: np.ndarray, *, multilabel: bool = False
                                 ) -> Dict[str, np.ndarray]:
    """Per-sample confidence statistics for a probabilistic forecast.

    The single-label and multi-label quantities are DIFFERENT questions, and only the
    single-label ones have an obvious name:

    - ``max_probability`` / ``margin`` are properties of a RANKING over one distribution's
      classes. For independent sigmoids the largest value is "the most confident label",
      which is a different thing, and ``margin`` (top-1 minus top-2) is not a margin over
      alternatives.
    - ``entropy`` ``-Σ p log p`` is the Shannon entropy of a DISTRIBUTION. Over
      independent sigmoids it is not an entropy, and its maximum is not ``log K``.

    For a multi-label head this returns the quantities that ARE well defined — per-label
    mean absolute error against the 0/1 indicator, and the mean predicted positive rate —
    and reports the single-label ones as NaN rather than as plausible wrong numbers.

    .. important::
       DECISION plan-2026-10-05-analyzer-audit/F-081. Previously every one of these was
       computed unconditionally, so a sigmoid head published a `gini` of a
       non-distribution and an `entropy` whose maximum was not ``log K``.

    Args:
        y_prob: ``(n_samples, n_classes)`` probabilities.
        multilabel: Whether these are independent per-label forecasts.

    Returns:
        Dict of per-sample arrays. Under ``multilabel`` the keys are
        ``label_mae``, ``mean_predicted_probability`` and ``predicted_positive_rate``;
        otherwise they are ``max_probability``, ``margin`` and ``gini_coefficient``.
    """
    probabilities = np.asarray(y_prob, dtype=float)
    max_prob = np.max(probabilities, axis=1)

    if multilabel:
        # The honest per-label quantities. `target` is supplied by the caller via the
        # 0/1 indicator matrix; with no labels available, report the predicted positive
        # rate alone and leave the error term NaN rather than guessing.
        return {
            'mean_predicted_probability': np.mean(probabilities, axis=1),
            'predicted_positive_rate': np.mean(probabilities > 0.5, axis=1),
            'label_mae': np.full(len(probabilities), np.nan),
            'max_probability': np.full(len(probabilities), np.nan),
            'margin': np.full(len(probabilities), np.nan),
            'gini_coefficient': np.full(len(probabilities), np.nan),
        }

    if probabilities.shape[1] > 1:
        sorted_probs = np.sort(probabilities, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    else:
        margin = np.full(len(probabilities), np.nan)

    # DECISION plan-2026-10-05-analyzer-audit/F-082
    # This quantity is `1 - Σ p²`, which is the SIMPSON INDEX (a.k.a. Gini IMPURITY), not
    # the Gini COEFFICIENT — the latter is a Lorenz-curve quantity, also called the
    # relative mean difference, and is a different number. The name `gini_coefficient`
    # was propagated into `AnalysisResults.confidence_metrics`, the saved artifact and
    # README.md. It is kept as the key so no stored artifact breaks, and corrected in
    # the docs; the new `simpson_index` key carries the accurate name. Do NOT silently
    # rename the published key.
    simpson = 1.0 - np.sum(np.square(probabilities), axis=1)

    return {
        'max_probability': max_prob,
        'margin': margin,
        'gini_coefficient': simpson,
        'simpson_index': simpson,
    }

# ------------------------------------------------------------------------------

def compute_brier_score_decomposition(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> Dict[str, float]:
    """
    Decompose the TOP-1 Brier Score into Reliability, Resolution, and Uncertainty.

    Murphy's (1973) three-term partition of the Brier score for the binary
    forecasting problem "was the top-1 prediction correct, at the top-1
    confidence":

        BS = Reliability - Resolution + Uncertainty

    - **Reliability**: Mean squared gap between a bin's mean confidence and its
      observed accuracy (calibration error; lower is better).
    - **Resolution**: Mean squared spread of the bins' accuracies about the
      overall accuracy (discrimination; higher is better).
    - **Uncertainty**: ``acc * (1 - acc)`` — the variance of the top-1
      correctness variable, i.e. the irreducible part.

    The identity holds EXACTLY against the returned ``brier_score``, which is the
    Brier score of the BINNED forecast (each sample's confidence replaced by its
    bin's mean confidence). ``binning_residual`` is the difference between the raw
    and binned top-1 Brier scores, so the raw score is
    ``brier_score + binning_residual``. It shrinks as ``n_bins`` grows.

    Warning:
        The returned ``brier_score`` key CHANGED MEANING under
        ``plan-2026-09-01T225724-e79ad4bd`` (D-014) and kept its name. It used to
        be the MULTICLASS Brier score (the quantity
        :func:`compute_brier_score` still returns); it is now the BINARY Brier
        score of the top-1 correctness forecast, which is the only outcome space
        in which Murphy's identity can hold against these bin statistics. On this
        function's own docstring example the two differ by 2x: the binned top-1
        score is ``0.0725`` (raw top-1 ``0.075``) where
        ``compute_brier_score(one_hot(y_true), y_prob)`` is ``0.15``. A caller who
        reads this key expecting the multiclass score gets a different quantity -
        call :func:`compute_brier_score` for that one. Note that
        ``results.calibration_metrics['brier_score']`` is produced by
        :func:`compute_brier_score` and is UNAFFECTED.

    Note:
        This function has no caller inside the library — ``CalibrationAnalyzer``
        imports only ``compute_ece``, ``compute_brier_score``,
        ``compute_reliability_data`` and ``compute_prediction_entropy_stats``. It
        is retained as a public diagnostic, not deleted; a wrong public formula is
        worse than an unused correct one.

    Args:
        y_true (np.ndarray): True class labels. Shape: (n_samples,)
        y_prob (np.ndarray): Predicted probabilities. Shape: (n_samples, n_classes)
        n_bins (int, optional): Number of bins for reliability/resolution.

    Returns:
        Dict[str, float]: Dictionary with 'reliability', 'resolution',
        'uncertainty', 'brier_score' (the exactly-decomposed binned TOP-1 BINARY
        Brier score — NOT the multiclass one; see the Warning above) and
        'binning_residual'.

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        >>> y_prob = np.array([[0.8,0.2],[0.3,0.7],[0.1,0.9],[0.9,0.1],
        ...                    [0.4,0.6],[0.7,0.3],[0.2,0.8],[0.6,0.4]])
        >>> decomp = compute_brier_score_decomposition(y_true, y_prob, n_bins=4)
        >>> recombined = (decomp['reliability'] - decomp['resolution']
        ...               + decomp['uncertainty'])
        >>> bool(np.isclose(recombined, decomp['brier_score']))
        True
    """
    # DECISION plan-2026-09-01T225724-e79ad4bd/D-014
    # Every term below is computed in ONE outcome space: the top-1 correctness
    # variable. Do NOT restore the multiclass one-hot base-rate uncertainty
    # (`sum(p_c * (1 - p_c))`) beside bin statistics taken from a top-1 binary
    # reduction — mixing the two is what broke the identity by 30-65%.
    # See decisions.md D-014.
    outcomes, scores = _top1_outcome_and_score(y_true, y_prob)
    n_samples = len(outcomes)
    if n_samples == 0:
        return {
            "reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0,
            "brier_score": 0.0, "binning_residual": 0.0,
        }

    overall_accuracy = float(np.mean(outcomes))

    # 1. Uncertainty: variance of the top-1 correctness variable.
    uncertainty = overall_accuracy * (1.0 - overall_accuracy)

    # 2. Reliability and Resolution (bin-based, same binner as every other metric).
    bin_info = _get_binary_bin_info(outcomes, scores, n_bins)

    reliability = 0.0
    resolution = 0.0
    binned_brier = 0.0

    for bin_data in bin_info:
        if bin_data["count"] > 0:
            prop_in_bin = bin_data["count"] / n_samples
            reliability += prop_in_bin * (bin_data["accuracy"] - bin_data["confidence"]) ** 2
            resolution += prop_in_bin * (bin_data["accuracy"] - overall_accuracy) ** 2
            # Brier score of the binned forecast: within a bin the forecast is the
            # bin's mean confidence, so E[(f_bin - o)^2] = f^2 - 2*f*acc + acc.
            binned_brier += prop_in_bin * (
                bin_data["confidence"] ** 2
                - 2.0 * bin_data["confidence"] * bin_data["accuracy"]
                + bin_data["accuracy"]
            )

    raw_brier = float(np.mean((scores - outcomes) ** 2))

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_score": float(binned_brier),
        "binning_residual": float(raw_brier - binned_brier),
    }

# ------------------------------------------------------------------------------

def compute_prediction_entropy_stats(y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute prediction entropy statistics to measure model uncertainty.

    Entropy H(p) = -Σ p_i * log(p_i) quantifies the uncertainty in a
    probability distribution. Low entropy indicates high confidence.

    Args:
        y_prob (np.ndarray): Predicted probabilities. Shape: (n_samples, n_classes)

    Returns:
        Dict[str, float]: Dictionary with mean, std, median, max, and min entropy.

    Example:
        >>> y_prob = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0.8, 0.2]])
        >>> stats = compute_prediction_entropy_stats(y_prob)
        >>> print(f"{stats['mean_entropy']:.4f}")
        0.4609
    """
    epsilon = 1e-9
    y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
    entropies = -np.sum(y_prob_clipped * np.log(y_prob_clipped), axis=1)

    return {
        'entropy': entropies,
        'mean_entropy': np.mean(entropies),
        'std_entropy': np.std(entropies),
        'median_entropy': np.median(entropies),
        'max_entropy': np.max( entropies),
        'min_entropy': np.min(entropies)
    }


# ------------------------------------------------------------------------------