"""
Training Dynamics Analysis Module

Analyzes training history to understand how models learned.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from .base import BaseAnalyzer
from ..data_types import AnalysisResults, DataInput, TrainingMetrics
from ..constants import (
    CONVERGENCE_THRESHOLD, TRAINING_STABILITY_WINDOW, OVERFITTING_ANALYSIS_FRACTION,
    LOSS_PATTERNS, VAL_LOSS_PATTERNS, ACC_PATTERNS, VAL_ACC_PATTERNS
)
from ..utils import find_metric_in_history, smooth_curve
from dl_techniques.utils.logger import logger


class TrainingDynamicsAnalyzer(BaseAnalyzer):
    """Analyzes training dynamics from history."""

    def requires_data(self) -> bool:
        """Training dynamics analysis doesn't require input data."""
        return False

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Analyze training history to understand how models learned."""
        logger.info("Analyzing training dynamics...")

        if not results.training_history:
            logger.warning("No training history available for analysis")
            return

        # Initialize training metrics container
        results.training_metrics = TrainingMetrics()

        for model_name, history in results.training_history.items():
            if not history:
                logger.warning(f"No training history available for {model_name}")
                continue

            # Compute quantitative metrics
            self._compute_training_metrics(model_name, history, results.training_metrics)

            # Apply smoothing if requested
            if self.config.smooth_training_curves:
                self._smooth_training_curves(model_name, history, results.training_metrics)

    def _compute_training_metrics(
            self,
            model_name: str,
            history: Dict[str, List[float]],
            metrics: TrainingMetrics
    ) -> None:
        """Compute quantitative metrics from training history with robust length handling."""
        # Extract metrics using flexible pattern matching
        train_loss = find_metric_in_history(history, LOSS_PATTERNS)
        val_loss = find_metric_in_history(history, VAL_LOSS_PATTERNS)
        train_acc = find_metric_in_history(history, ACC_PATTERNS)
        val_acc = find_metric_in_history(history, VAL_ACC_PATTERNS)

        # Epochs to convergence (95% of max validation accuracy)
        if val_acc is not None and len(val_acc) > 0:
            max_val_acc = max(val_acc)
            threshold = CONVERGENCE_THRESHOLD * max_val_acc
            epochs_to_conv = next((i for i, acc in enumerate(val_acc) if acc >= threshold), len(val_acc))
            metrics.epochs_to_convergence[model_name] = epochs_to_conv

        # Training stability score (lower is more stable)
        if val_loss and len(val_loss) > TRAINING_STABILITY_WINDOW:
            recent_losses = val_loss[-TRAINING_STABILITY_WINDOW:]
            stability_score = np.std(recent_losses)
            metrics.training_stability_score[model_name] = stability_score

        # Overfitting index with robust length handling
        if train_loss and val_loss:
            # Use the minimum length to ensure we're comparing the same epochs
            min_length = min(len(train_loss), len(val_loss))

            if min_length < 3:  # Need at least 3 epochs for meaningful analysis
                logger.warning(f"Insufficient epochs ({min_length}) for overfitting analysis of {model_name}")
                metrics.overfitting_index[model_name] = 0.0
                metrics.final_gap[model_name] = 0.0
            else:
                # Truncate both to same length
                train_loss_aligned = train_loss[:min_length]
                val_loss_aligned = val_loss[:min_length]

                # Calculate final third based on aligned length
                final_third_start = int(min_length * (1 - OVERFITTING_ANALYSIS_FRACTION))
                final_third_start = max(final_third_start, 1)  # Ensure at least 1 epoch in final third

                train_final = np.mean(train_loss_aligned[final_third_start:])
                val_final = np.mean(val_loss_aligned[final_third_start:])
                overfitting_index = val_final - train_final

                metrics.overfitting_index[model_name] = overfitting_index

                # Final gap using aligned losses
                metrics.final_gap[model_name] = val_loss_aligned[-1] - train_loss_aligned[-1]

                # Log if we had to truncate
                if len(train_loss) != len(val_loss):
                    logger.info(f"Aligned train/val losses for {model_name}: "
                              f"train={len(train_loss)} val={len(val_loss)} → {min_length} epochs")

        # Peak performance
        if val_acc is not None and len(val_acc) > 0:
            # Explicitly align val_acc and val_loss to prevent IndexErrors
            val_loss_safe = val_loss if val_loss is not None else []
            min_len = min(len(val_acc), len(val_loss_safe))
            val_acc_aligned = val_acc[:min_len]
            val_loss_aligned = val_loss_safe[:min_len]

            if val_acc_aligned:
                best_epoch = np.argmax(val_acc_aligned)
                peak_performance = {
                    'epoch': best_epoch,
                    'val_accuracy': val_acc_aligned[best_epoch]
                }

                # Add validation loss at best epoch if available
                if val_loss_aligned:
                    peak_performance['val_loss'] = val_loss_aligned[best_epoch]
                else:
                    peak_performance['val_loss'] = None

                metrics.peak_performance[model_name] = peak_performance

        # Log warning if no metrics found
        if not any([train_loss, val_loss, train_acc, val_acc]):
            logger.warning(
                f"No recognized training metrics found for {model_name}. Available keys: {list(history.keys())}")

    def _smooth_training_curves(self, model_name: str, history: Dict[str, List[float]],
                                metrics: TrainingMetrics) -> None:
        """Apply smoothing to training curves for cleaner visualization."""
        smoothed = {}

        for metric_name, values in history.items():
            if isinstance(values, list) and len(values) > self.config.smoothing_window:
                smoothed_values = smooth_curve(np.array(values), self.config.smoothing_window)
                smoothed[metric_name] = smoothed_values
            else:
                smoothed[metric_name] = values

        metrics.smoothed_curves[model_name] = smoothed