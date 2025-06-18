"""
Model Calibration Analysis Module.

This module provides comprehensive tools for analyzing and visualizing the calibration
effectiveness of machine learning models. Model calibration refers to how well the
predicted probabilities of a model reflect the true likelihood of the predicted outcomes.

Mathematical Foundations:

1. **Expected Calibration Error (ECE)**:
   Formula: ECE = Σ(i=1 to M) (n_i/n) × |acc_i - conf_i|

   Where:
   - M = number of confidence bins (default: 10)
   - n_i = number of samples in bin i
   - n = total number of samples
   - acc_i = accuracy of predictions in bin i
   - conf_i = average confidence (max probability) in bin i

   Interpretation:
   - Measures the weighted average of absolute differences between confidence and accuracy
   - Range: [0, 1], where 0 indicates perfect calibration
   - Penalizes both overconfidence (conf > acc) and underconfidence (conf < acc)
   - Weights bins by sample count, giving more importance to frequently occurring confidence levels

2. **Brier Score**:
   Formula: BS = (1/N) × Σ(i=1 to N) Σ(j=1 to K) (y_ij - p_ij)²

   Where:
   - N = number of samples
   - K = number of classes
   - y_ij = 1 if sample i belongs to class j, 0 otherwise (one-hot encoding)
   - p_ij = predicted probability that sample i belongs to class j

   Interpretation:
   - Measures the mean squared difference between predicted probabilities and true outcomes
   - Range: [0, 2] for binary classification, [0, 1] for well-calibrated multi-class
   - Lower scores indicate better calibration and accuracy combined
   - Decomposable into reliability, resolution, and uncertainty components
   - Sensitive to both calibration quality and discriminative ability

3. **Prediction Entropy**:
   Formula: H(p_i) = -Σ(j=1 to K) p_ij × log₂(p_ij)

   Where:
   - p_ij = predicted probability for sample i and class j
   - K = number of classes
   - log₂ denotes logarithm base 2 (bits of information)

   Statistics Computed:
   - Mean Entropy: (1/N) × Σ(i=1 to N) H(p_i)
   - Standard Deviation of Entropy: std(H(p_i))
   - Median Entropy: median(H(p_i))

   Interpretation:
   - Measures uncertainty/randomness in predictions
   - Range: [0, log₂(K)], where 0 = completely certain, log₂(K) = maximum uncertainty
   - High entropy: model is uncertain (probabilities spread across classes)
   - Low entropy: model is confident (probability concentrated on few classes)
   - Well-calibrated models should show entropy patterns consistent with true uncertainty

4. **Reliability Diagram Components**:
   For each confidence bin i:
   - Bin boundaries: [i/M, (i+1)/M] where M is number of bins
   - Bin center: (i + 0.5)/M
   - Bin accuracy: acc_i = (# correct predictions in bin i) / (# samples in bin i)
   - Bin confidence: conf_i = average of max probabilities in bin i
   - Bin count: n_i = number of samples in bin i

   Perfect Calibration Line:
   - y = x (45-degree line)
   - Points on this line indicate perfect calibration
   - Deviations above: underconfidence
   - Deviations below: overconfidence

Calibration Quality Indicators:

- **Well-Calibrated Model**:
  * ECE ≈ 0, Brier Score is low
  * Reliability diagram points lie close to perfect calibration line
  * Prediction entropy reflects true uncertainty in the data

- **Overconfident Model**:
  * ECE > 0, points in reliability diagram below perfect line
  * Low entropy but high error rate
  * High confidence in incorrect predictions

- **Underconfident Model**:
  * ECE > 0, points in reliability diagram above perfect line
  * High entropy even for easy predictions
  * Low confidence in correct predictions

Example:
    >>> import keras
    >>> from pathlib import Path
    >>>
    >>> # Load your trained models
    >>> models = {
    ...     'baseline_model': keras.models.load_model('baseline.keras'),
    ...     'calibrated_model': keras.models.load_model('calibrated.keras')
    ... }
    >>>
    >>> # Initialize analyzer with 15 bins for finer calibration analysis
    >>> analyzer = CalibrationAnalyzer(models, calibration_bins=15)
    >>>
    >>> # Analyze calibration (y_test should be one-hot encoded)
    >>> metrics = analyzer.analyze_calibration(x_test, y_test)
    >>>
    >>> # Expected output format:
    >>> # {
    >>> #     'baseline_model': {
    >>> #         'ece': 0.0234,  # 2.34% calibration error
    >>> #         'brier_score': 0.1456,  # Lower is better
    >>> #         'mean_entropy': 1.234,  # Average prediction uncertainty
    >>> #         'std_entropy': 0.456,   # Variability in uncertainty
    >>> #         'median_entropy': 1.123  # Median prediction uncertainty
    >>> #     },
    >>> #     'calibrated_model': {...}
    >>> # }
    >>>
    >>> # Generate comprehensive visualizations
    >>> output_dir = Path('calibration_analysis')
    >>> analyzer.plot_reliability_diagrams(output_dir)
    >>> analyzer.plot_calibration_metrics_comparison(output_dir)
    >>> analyzer.save_calibration_analysis(output_dir)
    >>>
    >>> # Identify best calibrated model
    >>> best_model = analyzer.get_best_calibrated_model('ece')
    >>> print(f"Best calibrated model: {best_model}")

Practical Applications:
    - Medical diagnosis: Ensuring probability predictions reflect true risk levels
    - Financial modeling: Risk assessment accuracy for investment decisions
    - Autonomous systems: Reliable confidence estimates for safety-critical decisions
    - A/B testing: Comparing calibration quality of different model architectures

Notes:
    - Input data should be properly preprocessed and normalized
    - Target data (y_test) must be one-hot encoded for multi-class problems
    - Larger bin counts (15-20) provide finer calibration analysis but require more data
    - ECE and Brier Score: lower values indicate better calibration
    - Entropy statistics should be interpreted relative to the number of classes and data complexity
    - Consider both calibration metrics and standard accuracy when evaluating models
"""

import keras
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Optional

# ------------------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------------------

from .logger import logger
from .calibration_metrics import (
    compute_ece,
    compute_brier_score,
    compute_reliability_data,
    compute_prediction_entropy_stats
)

# ------------------------------------------------------------------------------


class CalibrationAnalyzer:
    """Analyzer for model calibration effectiveness and reliability.

    This class provides methods to analyze and visualize model calibration
    using various metrics such as Expected Calibration Error (ECE) and
    Brier Score.
    """

    def __init__(
            self,
            models: Dict[str, keras.Model],
            calibration_bins: int = 10,
            plot_format: str = 'png'
    ) -> None:
        """Initialize the calibration analyzer.

        Args:
            models: Dictionary of trained models where keys are model names
                and values are Keras models
            calibration_bins: Number of bins to use for calibration analysis.
                Defaults to 10
            plot_format: Format for saving plots ('png', 'pdf', 'svg', etc.).
                Defaults to 'png'
        """
        self.models = models
        self.calibration_bins = calibration_bins
        self.plot_format = plot_format
        self.calibration_metrics: Dict[str, Dict[str, float]] = {}
        self.reliability_data: Dict[str, Dict[str, np.ndarray]] = {}

    def analyze_calibration(
            self,
            x_test: np.ndarray,
            y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Analyze calibration for all models.

        Args:
            x_test: Test input data of shape (n_samples, ...)
            y_test: Test target data (one-hot encoded) of shape (n_samples, n_classes)

        Returns:
            Dictionary containing calibration metrics for each model.
            Each model's metrics include: 'ece', 'brier_score', 'mean_entropy',
            'std_entropy', and 'median_entropy'
        """
        logger.info("Analyzing calibration effectiveness...")

        y_true_classes = np.argmax(y_test, axis=1)

        for name, model in self.models.items():
            logger.info(f"Computing calibration metrics for {name} model...")

            # Get predictions
            y_pred_proba = model.predict(x_test, verbose=0)

            # Compute calibration metrics
            ece = compute_ece(y_true_classes, y_pred_proba, self.calibration_bins)
            reliability_data = compute_reliability_data(
                y_true_classes,
                y_pred_proba,
                self.calibration_bins
            )
            brier_score = compute_brier_score(y_test, y_pred_proba)
            entropy_stats = compute_prediction_entropy_stats(y_pred_proba)

            self.calibration_metrics[name] = {
                'ece': ece,
                'brier_score': brier_score,
                'mean_entropy': entropy_stats['mean_entropy'],
                'std_entropy': entropy_stats['std_entropy'],
                'median_entropy': entropy_stats['median_entropy']
            }

            self.reliability_data[name] = reliability_data

            logger.info(f"Calibration metrics for {name}:")
            logger.info(f"  ECE: {ece:.4f}")
            logger.info(f"  Brier Score: {brier_score:.4f}")
            logger.info(f"  Mean Entropy: {entropy_stats['mean_entropy']:.4f}")

        return self.calibration_metrics

    def plot_reliability_diagrams(self, output_dir: Path) -> None:
        """Plot reliability diagrams for all models.

        Creates individual reliability diagrams for each model and a
        comparison plot showing all models together.

        Args:
            output_dir: Directory to save plots. Will be created if it doesn't exist
        """
        if not self.reliability_data:
            logger.warning("No reliability data available for plotting")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Individual reliability diagrams
        for model_name, reliability_data in self.reliability_data.items():
            plt.figure(figsize=(8, 8))

            # Plot perfect calibration line
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')

            # Plot model calibration
            bin_centers = reliability_data['bin_centers']
            bin_accuracies = reliability_data['bin_accuracies']
            bin_counts = reliability_data['bin_counts']

            # Use bin counts for marker sizes
            sizes = 100 * bin_counts / np.max(bin_counts + 1)

            scatter = plt.scatter(
                bin_centers,
                bin_accuracies,
                s=sizes,
                alpha=0.7,
                label=f'{model_name.replace("_", " ").title()} Calibration',
                color='red'
            )

            plt.xlabel('Confidence', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title(f'Reliability Diagram - {model_name.replace("_", " ").title()}', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])

            # Add colorbar for bin counts
            cbar = plt.colorbar(scatter)
            cbar.set_label('Number of Samples', rotation=270, labelpad=15)

            plt.tight_layout()
            plt.savefig(output_dir / f'reliability_{model_name}.{self.plot_format}',
                        dpi=300, bbox_inches='tight')
            plt.close()

        # Comparison reliability diagram
        plt.figure(figsize=(10, 8))

        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Perfect Calibration')

        colors = ['blue', 'red', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']

        for i, (model_name, reliability_data) in enumerate(self.reliability_data.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            plt.plot(
                reliability_data['bin_centers'],
                reliability_data['bin_accuracies'],
                marker=marker,
                markersize=8,
                linewidth=2,
                label=model_name.replace('_', ' ').title(),
                color=color,
                alpha=0.8
            )

        plt.xlabel('Confidence', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Calibration Comparison: Loss Functions', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / f'calibration_comparison.{self.plot_format}',
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved reliability diagrams to {output_dir}")

    def plot_calibration_metrics_comparison(self, output_dir: Path) -> None:
        """Plot comparison of calibration metrics across models.

        Creates a single figure with bar charts comparing ECE, Brier Score,
        and mean entropy across all models.

        Args:
            output_dir: Directory to save plots. Will be created if it doesn't exist
        """
        if not self.calibration_metrics:
            logger.warning("No calibration metrics available for plotting")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_to_plot = ['ece', 'brier_score', 'mean_entropy']
        metric_labels = ['Expected Calibration Error', 'Brier Score', 'Mean Prediction Entropy']

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        models = list(self.calibration_metrics.keys())
        model_names = [m.replace('_', ' ').title() for m in models]

        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[i]

            values = [self.calibration_metrics[model][metric] for model in models]

            bars = ax.bar(model_names, values, alpha=0.7)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_ylabel(label, fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f'{value:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        plt.suptitle('Calibration Metrics Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'calibration_metrics_comparison.{self.plot_format}',
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved calibration metrics comparison plot to {output_dir}")

    def save_calibration_analysis(self, output_dir: Path) -> None:
        """Save calibration analysis results to a text file.

        Args:
            output_dir: Directory to save results. Will be created if it doesn't exist
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save calibration statistics
        with open(output_dir / 'calibration_statistics.txt', 'w') as f:
            f.write("Calibration Analysis Results\n")
            f.write("=" * 50 + "\n\n")

            for model_name, metrics in self.calibration_metrics.items():
                f.write(f"{model_name.upper().replace('_', ' ')} MODEL:\n")
                f.write("-" * 30 + "\n")
                for metric, value in metrics.items():
                    f.write(f"{metric.replace('_', ' ').title()}: {value:.6f}\n")
                f.write("\n")

        logger.info(f"Saved calibration analysis to {output_dir}")

    def get_best_calibrated_model(self, metric: str = 'ece') -> Optional[str]:
        """Get the name of the best calibrated model based on a specific metric.

        Args:
            metric: Metric to use for comparison. Options: 'ece', 'brier_score'.
                Lower values indicate better calibration. Defaults to 'ece'

        Returns:
            Name of the best calibrated model, or None if no metrics available

        Raises:
            ValueError: If the specified metric is not available
        """
        if not self.calibration_metrics:
            logger.warning("No calibration metrics available")
            return None

        if metric not in ['ece', 'brier_score']:
            raise ValueError(f"Metric '{metric}' not supported. Use 'ece' or 'brier_score'")

        # Find model with lowest metric value (better calibration)
        best_model = min(
            self.calibration_metrics.items(),
            key=lambda x: x[1][metric]
        )

        logger.info(f"Best calibrated model by {metric}: {best_model[0]} "
                    f"({metric}={best_model[1][metric]:.4f})")

        return best_model[0]

    def get_calibration_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of all calibration metrics.

        Returns:
            Copy of calibration metrics dictionary
        """
        return self.calibration_metrics.copy()

# ------------------------------------------------------------------------------
