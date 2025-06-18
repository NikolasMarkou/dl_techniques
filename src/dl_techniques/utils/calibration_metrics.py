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
- **Calibration**: The degree to which predicted probabilities match observed frequencies
- **Reliability**: How well confidence scores reflect actual accuracy
- **Sharpness**: The concentration of predictions away from the base rate
- **Resolution**: The ability to distinguish between correct and incorrect predictions

The metrics implemented here help assess these properties:

1. **Expected Calibration Error (ECE)**: Measures average calibration error across 
   confidence bins
2. **Reliability Diagram Data**: Provides data for visualizing calibration through 
   reliability plots
3. **Brier Score**: Measures the accuracy of probabilistic predictions
4. **Prediction Entropy**: Quantifies uncertainty in model predictions

References:
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of 
  modern neural networks. ICML.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with 
  supervised learning. ICML.
"""

import numpy as np
from typing import Dict


# ------------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).

    The Expected Calibration Error measures the difference between predicted 
    confidence and observed accuracy across different confidence levels. It is 
    calculated by:

    1. Binning predictions by their confidence scores
    2. Computing the accuracy within each bin
    3. Computing the average confidence within each bin
    4. Weighing the absolute difference by the proportion of samples in each bin

    Mathematically:
    ECE = Σ(i=1 to M) (n_i/n) * |acc_i - conf_i|

    Where:
    - M is the number of bins
    - n_i is the number of samples in bin i
    - n is the total number of samples
    - acc_i is the accuracy of samples in bin i
    - conf_i is the average confidence of samples in bin i

    A perfectly calibrated model would have ECE = 0.

    Args:
        y_true (np.ndarray): True class labels (not one-hot encoded). 
            Shape: (n_samples,)
        y_prob (np.ndarray): Predicted class probabilities. 
            Shape: (n_samples, n_classes)
        n_bins (int, optional): Number of equal-width bins for confidence 
            discretization. Defaults to 15.

    Returns:
        float: Expected Calibration Error value between 0 and 1, where 0 
            indicates perfect calibration.

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_prob = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], 
        ...                    [0.8, 0.2], [0.4, 0.6]])
        >>> ece = compute_ece(y_true, y_prob)
        >>> print(f"ECE: {ece:.4f}")
    """
    # Extract predicted class (highest probability) and confidence (max probability)
    y_pred = np.argmax(y_prob, axis=1)  # Predicted class labels
    confidences = np.max(y_prob, axis=1)  # Confidence scores (max probability)

    # Compute binary accuracy for each prediction
    accuracies = (y_pred == y_true).astype(float)

    # Create equal-width bins from 0 to 1
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]  # Lower bounds of each bin
    bin_uppers = bin_boundaries[1:]  # Upper bounds of each bin

    ece = 0.0

    # Iterate through each confidence bin
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples whose confidence falls in current bin
        # Note: Using (confidence > lower] and (confidence <= upper] for right-inclusive bins
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        # Calculate proportion of total samples in this bin
        prop_in_bin = in_bin.mean()

        # Only process bins that contain samples
        if prop_in_bin > 0:
            # Calculate average accuracy within this bin
            accuracy_in_bin = accuracies[in_bin].mean()

            # Calculate average confidence within this bin
            avg_confidence_in_bin = confidences[in_bin].mean()

            # Add weighted calibration error for this bin
            # |confidence - accuracy| weighted by bin proportion
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


# ------------------------------------------------------------------------------

def compute_reliability_data(y_true: np.ndarray, y_prob: np.ndarray,
                             n_bins: int = 15) -> Dict[str, np.ndarray]:
    """
    Compute data for reliability diagram visualization.

    A reliability diagram (also called calibration plot) is a graphical method 
    for assessing the calibration of a probabilistic classifier. It plots the 
    observed frequency of positive outcomes against the predicted probability.

    For a perfectly calibrated classifier, the reliability diagram should lie 
    on the diagonal line y=x, meaning that among predictions with confidence p, 
    the fraction of correct predictions should be p.

    The function computes:
    - Bin centers: Midpoints of confidence intervals
    - Bin accuracies: Observed accuracy within each confidence bin
    - Bin confidences: Average predicted confidence within each bin
    - Bin counts: Number of samples in each bin

    Args:
        y_true (np.ndarray): True class labels (not one-hot encoded). 
            Shape: (n_samples,)
        y_prob (np.ndarray): Predicted class probabilities. 
            Shape: (n_samples, n_classes)
        n_bins (int, optional): Number of equal-width bins for confidence 
            discretization. Defaults to 15.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'bin_centers': Center points of each confidence bin
            - 'bin_accuracies': Observed accuracy in each bin
            - 'bin_confidences': Average confidence in each bin  
            - 'bin_counts': Number of samples in each bin

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_prob = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], 
        ...                    [0.8, 0.2], [0.4, 0.6]])
        >>> reliability_data = compute_reliability_data(y_true, y_prob)
        >>> print("Bin centers:", reliability_data['bin_centers'])
        >>> print("Bin accuracies:", reliability_data['bin_accuracies'])
    """
    # Extract predictions and confidences
    y_pred = np.argmax(y_prob, axis=1)  # Predicted class labels
    confidences = np.max(y_prob, axis=1)  # Confidence scores
    accuracies = (y_pred == y_true).astype(float)  # Binary accuracy

    # Create bin boundaries and compute bin centers
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2  # Midpoint of each bin

    # Initialize lists to store bin statistics
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    # Process each confidence bin
    for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Identify samples in current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum()

        if bin_size > 0:
            # Compute statistics for non-empty bins
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(bin_size)
        else:
            # Handle empty bins by setting accuracy to 0 and confidence to bin center
            bin_accuracies.append(0.0)
            bin_confidences.append(bin_centers[bin_idx])
            bin_counts.append(0)

    return {
        'bin_centers': bin_centers,
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts)
    }


# ------------------------------------------------------------------------------

def compute_brier_score(y_true_onehot: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier Score for probabilistic predictions.

    The Brier Score is a scoring rule that measures the accuracy of probabilistic 
    predictions. It is the mean squared difference between predicted probabilities 
    and the actual binary outcomes (expressed as 0 or 1).

    Mathematically, for multiclass problems:
    BS = (1/N) * Σ(i=1 to N) Σ(j=1 to K) (p_ij - o_ij)²

    Where:
    - N is the number of samples
    - K is the number of classes
    - p_ij is the predicted probability of class j for sample i
    - o_ij is 1 if class j is the true class for sample i, 0 otherwise

    Properties:
    - Range: [0, 2] for binary classification, [0, 2] for multiclass
    - Lower values indicate better calibration
    - BS = 0 indicates perfect predictions
    - BS can be decomposed into reliability, resolution, and uncertainty components

    The Brier Score rewards both calibration and sharpness:
    - Calibration: Predicted probabilities should match observed frequencies
    - Sharpness: Predictions should be confident (away from uniform distribution)

    Args:
        y_true_onehot (np.ndarray): True labels in one-hot encoded format. 
            Shape: (n_samples, n_classes)
        y_prob (np.ndarray): Predicted class probabilities. 
            Shape: (n_samples, n_classes)

    Returns:
        float: Brier Score value. Lower values indicate better predictions.

    Example:
        >>> y_true_oh = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
        >>> y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.9, 0.1]])
        >>> bs = compute_brier_score(y_true_oh, y_prob)
        >>> print(f"Brier Score: {bs:.4f}")
    """
    # Compute squared differences between predictions and true labels
    # Shape: (n_samples, n_classes)
    squared_diffs = (y_prob - y_true_onehot) ** 2

    # Sum over classes for each sample, then average over samples
    # This gives the mean squared error across all probability predictions
    brier_score = np.mean(np.sum(squared_diffs, axis=1))

    return brier_score


# ------------------------------------------------------------------------------

def compute_prediction_entropy_stats(y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute prediction entropy statistics to measure model uncertainty.

    Entropy quantifies the uncertainty in probability distributions. For 
    probabilistic predictions, high entropy indicates the model is uncertain 
    (probabilities are spread across classes), while low entropy indicates 
    confidence (probability mass concentrated on few classes).

    The entropy of a probability distribution p is calculated as:
    H(p) = -Σ(i=1 to K) p_i * log(p_i)

    Where:
    - K is the number of classes
    - p_i is the probability of class i
    - log is typically the natural logarithm

    Properties:
    - Range: [0, log(K)] where K is the number of classes
    - H = 0 indicates complete certainty (probability 1 for one class)
    - H = log(K) indicates maximum uncertainty (uniform distribution)
    - Higher entropy suggests the model is less confident in its predictions

    This function computes comprehensive statistics of the entropy distribution
    across all predictions, providing insights into:
    - Overall model confidence (mean entropy)
    - Consistency of confidence (standard deviation)
    - Distribution characteristics (median, min, max)

    Args:
        y_prob (np.ndarray): Predicted class probabilities. 
            Shape: (n_samples, n_classes)

    Returns:
        Dict[str, float]: Dictionary containing entropy statistics:
            - 'mean_entropy': Average entropy across all predictions
            - 'std_entropy': Standard deviation of entropies
            - 'median_entropy': Median entropy value
            - 'max_entropy': Maximum entropy observed
            - 'min_entropy': Minimum entropy observed

    Example:
        >>> y_prob = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0.8, 0.2]])
        >>> entropy_stats = compute_prediction_entropy_stats(y_prob)
        >>> print(f"Mean entropy: {entropy_stats['mean_entropy']:.4f}")
        >>> print(f"Std entropy: {entropy_stats['std_entropy']:.4f}")
    """
    # Add small epsilon to avoid log(0) which would result in -inf
    # This is a common numerical stability technique
    epsilon = 1e-8
    y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)

    # Compute entropy for each prediction using the formula H = -Σ p_i * log(p_i)
    # Shape: (n_samples,)
    entropies = -np.sum(y_prob_clipped * np.log(y_prob_clipped), axis=1)

    # Compute comprehensive statistics of the entropy distribution
    return {
        'mean_entropy': np.mean(entropies),  # Average uncertainty
        'std_entropy': np.std(entropies),  # Variability in uncertainty
        'median_entropy': np.median(entropies),  # Median uncertainty
        'max_entropy': np.max(entropies),  # Highest uncertainty
        'min_entropy': np.min(entropies)  # Lowest uncertainty
    }

# ------------------------------------------------------------------------------
