"""
Assess the reliability and uncertainty of model predictions.

This analyzer evaluates how well a model's predicted probabilities align with
the true likelihood of outcomes. Modern neural networks, particularly when
trained with techniques that encourage low-entropy outputs (e.g., cross-
entropy loss), often become overconfident. This means the model might assign
a high probability (e.g., 99%) to a prediction that is incorrect. For
high-stakes applications, understanding and quantifying this discrepancy
between confidence and accuracy is critical. This module provides the tools
to measure this gap.

Architecture
------------
The analyzer operates as a post-processing component. It does not run the
model but consumes pre-computed predictions (probabilities) and true labels.
Its primary role is to delegate these data to a suite of specialized metric
functions, each designed to probe a different aspect of probabilistic
prediction quality. The results are then structured into two distinct
categories:
-   **Calibration Metrics**: Focus on the reliability of the probabilities. It
    answers the question: "When the model says it's P% confident, is it
    correct P% of the time?" Key metrics include ECE and Brier score.
-   **Confidence Metrics**: Characterize the model's internal sense of
    certainty, irrespective of correctness. It answers: "How certain is the
    model in its predictions?" Key metrics include prediction entropy and
    the distribution of maximum probabilities.

Foundational Mathematics
------------------------
The core of the analysis rests on established metrics from statistics and
information theory to quantify the quality of probabilistic forecasts.

-   **Expected Calibration Error (ECE)**: This is the primary metric for
    miscalibration. It measures the average gap between a model's prediction
    confidence and its actual accuracy. The calculation involves partitioning
    predictions into `M` bins based on their confidence scores. For each bin
    `B_m`, the average confidence `conf(B_m)` and accuracy `acc(B_m)` are
    computed. The ECE is the weighted average of their absolute difference:
    ECE = Σ_{m=1 to M} (|B_m|/n) * |acc(B_m) - conf(B_m)|
    A perfectly calibrated model has an ECE of 0.

-   **Brier Score**: This is a "proper scoring rule" that measures both
    calibration and resolution (the model's ability to distinguish outcomes).
    It is the mean squared error between the predicted probability vector `p`
    and the one-hot encoded true label vector `o`:
    BS = (1/N) * Σ_{i=1 to N} Σ_{j=1 to K} (p_ij - o_ij)²
    A lower Brier score is better, indicating predictions that are both
    accurate and well-calibrated.

-   **Shannon Entropy**: This metric is used to quantify the uncertainty of an
    individual prediction. For a probability distribution `p` over `K`
    classes, the entropy is:
    H(p) = -Σ_{i=1 to K} p_i * log(p_i)
    A low entropy value corresponds to a "peaked," high-confidence
    prediction, while a high entropy value indicates an uncertain prediction
    with probabilities spread across multiple classes.

References
----------
1.  Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On calibration
    of modern neural networks." ICML.
2.  Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting good
    probabilities with supervised learning." ICML.
3.  Brier, G. W. (1950). "Verification of forecasts expressed in terms of
    probability." Monthly Weather Review.

"""

import keras
import numpy as np
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .base import BaseAnalyzer
from ..data_types import AnalysisResults, DataInput
from ..calibration_metrics import (
    compute_ece,
    compute_brier_score,
    compute_reliability_data,
    compute_prediction_entropy_stats
)

# ---------------------------------------------------------------------


class CalibrationAnalyzer(BaseAnalyzer):
    """Analyzes model confidence and calibration."""

    def requires_data(self) -> bool:
        """Calibration analysis requires input data."""
        return True

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Analyze model confidence and calibration with consolidated metric storage.
        """
        logger.info("Analyzing confidence and calibration...")

        if cache is None:
            raise ValueError("Prediction cache is required for calibration analysis")

        for model_name in self.models:
            if model_name not in cache:
                continue

            model_cache = cache[model_name]
            # This variable now correctly holds probabilities, not logits.
            y_pred_proba = model_cache.get('predictions', None)

            if y_pred_proba is None:
                logger.warning(f"Could not find predictions for {model_name}: attempting to get logits")
                y_pred_logits = model_cache.get('logits', None)
                if y_pred_logits is None:
                    y_pred_proba = keras.ops.softmax(y_pred_logits, axis=-1)
                else:
                    logger.warning(f"Could not find logits for {model_name}")

            if y_pred_proba is None:
                logger.warning(f"Skipping calibration analysis for {model_name}: No predictions / logits available.")
                continue

            y_true = model_cache['y_data']

            # Convert to class indices if needed
            y_true = np.asarray(y_true)
            try:
                if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                    y_true_idx = np.argmax(y_true, axis=-1)
                else:
                    y_true_idx = y_true.flatten().astype(int)
            except (ValueError, TypeError) as e:
                logger.error(f"Cannot convert y_true to integer indices for {model_name}: {e}")
                continue

            # Compute calibration-specific metrics
            ece = compute_ece(y_true_idx, y_pred_proba, self.config.calibration_bins)
            reliability_data = compute_reliability_data(y_true_idx, y_pred_proba, self.config.calibration_bins)

            # Brier score requires one-hot encoded true labels. Convert if necessary.
            y_true_one_hot = y_true
            num_classes = y_pred_proba.shape[1]
            if len(y_true.shape) == 1 or y_true.shape[1] == 1:
                 y_true_one_hot = np.zeros((y_true_idx.size, num_classes))
                 y_true_one_hot[np.arange(y_true_idx.size), y_true_idx] = 1
            brier_score = compute_brier_score(y_true_one_hot, y_pred_proba)

            # Compute per-class ECE
            n_classes = y_pred_proba.shape[1]
            per_class_ece = []
            per_class_bins = max(2, self.config.calibration_bins // 2)

            for c in range(n_classes):
                class_mask = y_true_idx == c
                if np.any(class_mask):
                    class_ece = compute_ece(y_true_idx[class_mask], y_pred_proba[class_mask],
                                          per_class_bins)
                    per_class_ece.append(class_ece)
                else:
                    per_class_ece.append(0.0)

            # Store only calibration-specific metrics (no entropy here)
            results.calibration_metrics[model_name] = {
                'ece': ece,
                'brier_score': brier_score,
                'per_class_ece': per_class_ece,
            }

            # Store reliability data separately (for plotting)
            results.reliability_data[model_name] = reliability_data

            # Consolidate ALL confidence-related metrics including entropy
            confidence_metrics = self._compute_confidence_metrics(y_pred_proba)
            entropy_stats = compute_prediction_entropy_stats(y_pred_proba)

            # Combine all confidence-related metrics into one place
            all_confidence_metrics = {**confidence_metrics, **entropy_stats}
            results.confidence_metrics[model_name] = all_confidence_metrics

    def _compute_confidence_metrics(self, probabilities: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute various confidence metrics (excluding entropy which comes from entropy_stats)."""
        max_prob = np.max(probabilities, axis=1)

        # Handle single-class case for margin and gini
        if probabilities.shape[1] > 1:
            # Sort probabilities to find the top two for margin calculation
            sorted_probs = np.sort(probabilities, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            gini = 1 - np.sum(sorted_probs**2, axis=1)
        else:
            # Margin and Gini are not well-defined for a single class output
            margin = np.full(probabilities.shape[0], np.nan)
            gini = np.zeros(probabilities.shape[0])

        return {
            'max_probability': max_prob,
            'margin': margin,
            'gini_coefficient': gini
        }

# ---------------------------------------------------------------------
