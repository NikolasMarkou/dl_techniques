"""
Calibration Analysis Module

Analyzes model calibration and confidence metrics.
"""

import keras
import numpy as np
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.calibration_metrics import (
    compute_ece,
    compute_brier_score,
    compute_reliability_data,
    compute_prediction_entropy_stats
)
from .base import BaseAnalyzer
from ..data_types import AnalysisResults, DataInput

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

        FIXED: Eliminates redundancy by storing all confidence-related metrics
        (including entropy) in results.confidence_metrics, while keeping only
        calibration-specific metrics in results.calibration_metrics.
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

            # Manually calculate per-sample entropy if it's missing, ensuring visualizers work.
            if 'entropy' not in entropy_stats:
                # Add a small epsilon to prevent log(0) for probabilities of 0.
                epsilon = 1e-9
                per_sample_entropy = -np.sum(y_pred_proba * np.log2(y_pred_proba + epsilon), axis=1)
                entropy_stats['entropy'] = per_sample_entropy

            # Ensure mean_entropy is present for summary tables.
            if 'entropy' in entropy_stats and 'mean_entropy' not in entropy_stats:
                entropy_stats['mean_entropy'] = float(np.mean(entropy_stats['entropy']))

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