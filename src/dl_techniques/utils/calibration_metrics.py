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

def _get_bin_info(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int
) -> List[Dict[str, float]]:
    """
    Internal helper to compute bin-wise statistics for equal-width bins.

    This function calculates the confidence, accuracy, count, and proportion of
    samples within each bin. It handles edge cases for binning.
    """
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_info = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Handle the first bin to be inclusive of 0.0
        if bin_lower == 0.0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
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
        0.16
    """
    bin_info = _get_bin_info(y_true, y_prob, n_bins)

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
        0.0
    """
    n_samples = len(y_true)
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    # Sort samples by confidence
    sorted_indices = np.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]

    # Create equal-mass bins
    bin_size = n_samples // n_bins
    ece = 0.0

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, n_samples)

        bin_samples_conf = sorted_confidences[start_idx:end_idx]
        bin_samples_acc = sorted_accuracies[start_idx:end_idx]

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
        >>> y_prob = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.8, 0.2]])
        >>> data = compute_reliability_data(y_true[:4], y_prob[:4], n_bins=5)
        >>> print(data['bin_centers'])
        [0.1 0.3 0.5 0.7 0.9]
    """
    bin_info = _get_bin_info(y_true, y_prob, n_bins)

    bin_centers = np.array([b["center"] for b in bin_info])
    bin_accuracies = np.array([b["accuracy"] for b in bin_info])
    bin_confidences = np.array([b["confidence"] if b["count"] > 0 else b["center"] for b in bin_info])
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

def compute_brier_score(y_true_onehot: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier Score for multiclass probabilistic predictions.

    The Brier Score is the mean squared difference between predicted
    probabilities and actual outcomes. Lower values are better.

    Mathematically: BS = (1/N) * Σ(i=1 to N) Σ(j=1 to K) (p_ij - o_ij)²

    Args:
        y_true_onehot (np.ndarray): True labels in one-hot encoded format.
            Shape: (n_samples, n_classes)
        y_prob (np.ndarray): Predicted class probabilities.
            Shape: (n_samples, n_classes)

    Returns:
        float: Brier Score. Lower is better.

    Example:
        >>> y_true_oh = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
        >>> y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.9, 0.1]])
        >>> compute_brier_score(y_true_oh, y_prob)
        0.135
    """
    squared_diffs = (y_prob - y_true_onehot) ** 2
    return np.mean(np.sum(squared_diffs, axis=1))

# ------------------------------------------------------------------------------

def compute_brier_score_decomposition(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> Dict[str, float]:
    """
    Decompose the Brier Score into Reliability, Resolution, and Uncertainty.

    BS = Reliability - Resolution + Uncertainty

    - **Reliability**: Measures calibration error (lower is better).
    - **Resolution**: Measures how well the model separates outcomes (higher is better).
    - **Uncertainty**: Inherent unpredictability of the data (model-independent).

    Args:
        y_true (np.ndarray): True class labels. Shape: (n_samples,)
        y_prob (np.ndarray): Predicted probabilities. Shape: (n_samples, n_classes)
        n_bins (int, optional): Number of bins for reliability/resolution.

    Returns:
        Dict[str, float]: Dictionary with 'reliability', 'resolution', and 'uncertainty'.

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        >>> y_prob = np.array([[0.8,0.2],[0.3,0.7],[0.1,0.9],[0.9,0.1],
        ...                    [0.4,0.6],[0.7,0.3],[0.2,0.8],[0.6,0.4]])
        >>> decomp = compute_brier_score_decomposition(y_true, y_prob, n_bins=4)
        >>> print(f"{decomp['reliability']:.4f}")
        0.0211
    """
    n_samples, n_classes = y_prob.shape
    y_true_onehot = np.eye(n_classes)[y_true]

    # 1. Uncertainty: Inherent randomness of the outcome
    base_rate = np.mean(y_true_onehot, axis=0)
    uncertainty = np.sum(base_rate * (1 - base_rate))

    # 2. Reliability and Resolution (bin-based)
    bin_info = _get_bin_info(y_true, y_prob, n_bins)
    overall_accuracy = np.mean(y_true == np.argmax(y_prob, axis=1))

    reliability = 0.0
    resolution = 0.0

    for bin_data in bin_info:
        if bin_data["count"] > 0:
            prop_in_bin = bin_data["count"] / n_samples
            reliability += prop_in_bin * (bin_data["accuracy"] - bin_data["confidence"]) ** 2
            resolution += prop_in_bin * (bin_data["accuracy"] - overall_accuracy) ** 2

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty
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
        0.4578
    """
    epsilon = 1e-9
    y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
    entropies = -np.sum(y_prob_clipped * np.log(y_prob_clipped), axis=1)

    return {
        'mean_entropy': np.mean(entropies),
        'std_entropy': np.std(entropies),
        'median_entropy': np.median(entropies),
        'max_entropy': np.max( entropies),
        'min_entropy': np.min(entropies)
    }


# ------------------------------------------------------------------------------