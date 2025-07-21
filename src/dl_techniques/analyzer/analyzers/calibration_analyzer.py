"""
Calibration Analysis Module
============================================================================

Analyzes model calibration and confidence metrics.
"""

import numpy as np
from typing import Dict, Any, Optional
from .base import BaseAnalyzer
from ..data_types import AnalysisResults, DataInput
from dl_techniques.utils.logger import logger
from dl_techniques.utils.calibration_metrics import (
    compute_ece,
    compute_brier_score,
    compute_reliability_data,
    compute_prediction_entropy_stats
)


class CalibrationAnalyzer(BaseAnalyzer):
    """Analyzes model confidence and calibration."""

    def requires_data(self) -> bool:
        """Calibration analysis requires input data."""
        return True

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Analyze model confidence and calibration in a unified way."""
        logger.info("Analyzing confidence and calibration...")

        if cache is None:
            raise ValueError("Prediction cache is required for calibration analysis")

        for model_name in self.models:
            if model_name not in cache:
                continue

            model_cache = cache[model_name]
            y_pred_proba = model_cache.get('predictions') # Use .get() for safety

            # This check prevents the crash
            if y_pred_proba is None:
                logger.warning(f"Skipping calibration analysis for {model_name}: No predictions available.")
                continue

            y_true = model_cache['y_data']

            # Convert to class indices if needed - handle different data types
            y_true = np.asarray(y_true)  # Ensure numpy array
            try:
                if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                    y_true_idx = np.argmax(y_true, axis=1)
                else:
                    y_true_idx = y_true.flatten().astype(int)
            except (ValueError, TypeError) as e:
                logger.error(f"Cannot convert y_true to integer indices for {model_name}: {e}")
                continue

            # Compute calibration metrics
            ece = compute_ece(y_true_idx, y_pred_proba, self.config.calibration_bins)
            reliability_data = compute_reliability_data(y_true_idx, y_pred_proba, self.config.calibration_bins)
            brier_score = compute_brier_score(model_cache['y_data'], y_pred_proba)
            entropy_stats = compute_prediction_entropy_stats(y_pred_proba)

            # Compute per-class ECE with validated bins
            n_classes = y_pred_proba.shape[1]
            per_class_ece = []
            per_class_bins = max(2, self.config.calibration_bins // 2)  # Ensure minimum 2 bins

            for c in range(n_classes):
                class_mask = y_true_idx == c
                if np.any(class_mask):
                    class_ece = compute_ece(y_true_idx[class_mask], y_pred_proba[class_mask],
                                          per_class_bins)
                    per_class_ece.append(class_ece)
                else:
                    per_class_ece.append(0.0)

            results.calibration_metrics[model_name] = {
                'ece': ece,
                'brier_score': brier_score,
                'per_class_ece': per_class_ece,
                **entropy_stats
            }

            results.reliability_data[model_name] = reliability_data

            # Compute confidence metrics
            results.confidence_metrics[model_name] = self._compute_confidence_metrics(y_pred_proba)

    def _compute_confidence_metrics(self, probabilities: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute various confidence metrics."""
        max_prob = np.max(probabilities, axis=1)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)

        # Handle single-class case
        if probabilities.shape[1] > 1:
            sorted_probs = np.sort(probabilities, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            gini = 1 - np.sum(sorted_probs**2, axis=1)
        else:
            # Margin and Gini are not well-defined for single class
            margin = np.full(probabilities.shape[0], np.nan)
            gini = np.zeros(probabilities.shape[0])

        return {
            'max_probability': max_prob,
            'entropy': entropy,
            'margin': margin,
            'gini_coefficient': gini
        }