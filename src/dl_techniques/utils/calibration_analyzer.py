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

        Creates bar charts comparing ECE, Brier Score, and mean entropy
        across all models.

        Args:
            output_dir: Directory to save plots. Will be created if it doesn't exist
        """
        if not self.calibration_metrics:
            logger.warning("No calibration metrics available for plotting")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_to_plot = ['ece', 'brier_score', 'mean_entropy']
        metric_labels = ['Expected Calibration Error', 'Brier Score', 'Mean Prediction Entropy']

        for metric, label in zip(metrics_to_plot, metric_labels):
            plt.figure(figsize=(10, 6))

            models = list(self.calibration_metrics.keys())
            values = [self.calibration_metrics[model][metric] for model in models]

            bars = plt.bar([m.replace('_', ' ').title() for m in models], values)
            plt.title(f'{label} Comparison')
            plt.ylabel(label)
            plt.xticks(rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{value:.4f}',
                    ha='center',
                    va='bottom'
                )

            plt.tight_layout()
            plt.savefig(output_dir / f'{metric}_comparison.{self.plot_format}',
                        dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Saved calibration metrics plots to {output_dir}")

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
