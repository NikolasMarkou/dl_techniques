"""
Enhanced Calibration Visualization Module
Creates visualizations for calibration analysis results with centralized legend management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from typing import List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseVisualizer
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

class CalibrationVisualizer(BaseVisualizer):
    """Creates calibration analysis visualizations with centralized legend."""

    def create_visualizations(self) -> None:
        """Create unified confidence and calibration visualizations with single legend."""
        if not self.results.calibration_metrics:
            return

        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Reliability Diagram
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_reliability_diagram(ax1)

        # 2. Confidence Distributions (Violin Plot)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_confidence_distribution(ax2)

        # 3. Per-Class ECE
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_per_class_ece(ax3)

        # 4. Uncertainty Landscape
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_uncertainty_landscape(ax4)

        # Add single figure-level legend
        models_with_data = self._get_models_with_data()
        if models_with_data:
            self._create_figure_legend(fig, title="Models", specific_models=models_with_data)

        plt.suptitle('Confidence and Calibration Analysis', fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.85)  # Adjusted for legend

        if self.config.save_plots:
            self._save_figure(fig, 'confidence_calibration_analysis')
        plt.close(fig)

    def _get_models_with_data(self) -> List[str]:
        """Get models that have calibration/confidence data."""
        models_with_data = []
        for model_name in self.model_order:
            has_calibration = model_name in self.results.calibration_metrics
            has_confidence = model_name in self.results.confidence_metrics
            has_reliability = model_name in self.results.reliability_data

            if has_calibration or has_confidence or has_reliability:
                models_with_data.append(model_name)
        return models_with_data

    def _plot_reliability_diagram(self, ax) -> None:
        """Plot reliability diagram with confidence intervals."""
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, label='Perfect Calibration')

        # Use consistent model ordering
        for model_name in self._sort_models_consistently(list(self.results.reliability_data.keys())):
            rel_data = self.results.reliability_data[model_name]
            color = self._get_model_color(model_name)

            # Plot main line
            ax.plot(rel_data['bin_centers'], rel_data['bin_accuracies'],
                    'o-', color=color, linewidth=2, markersize=5)

            # Add shaded confidence region if we have sample counts
            if 'bin_counts' in rel_data:
                # Simple confidence interval based on binomial proportion
                counts = rel_data['bin_counts']
                props = rel_data['bin_accuracies']
                se = np.sqrt(props * (1 - props) / (counts + 1))

                ax.fill_between(rel_data['bin_centers'],
                                props - 1.96 * se, props + 1.96 * se,
                                alpha=0.2, color=color)

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagrams with 95% CI')
        # REMOVED: Individual legend - will use figure-level legend
        ax.grid(True, alpha=0.2)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    def _plot_confidence_distribution(self, ax) -> None:
        """Plot confidence distributions as a vertical violin plot."""
        confidence_data = []

        # Access confidence metrics from the correct location with consistent ordering
        for model_name in self._sort_models_consistently(list(self.results.confidence_metrics.keys())):
            metrics = self.results.confidence_metrics[model_name]
            # Safety check for required keys
            if 'max_probability' not in metrics:
                logger.warning(f"Missing 'max_probability' key for model {model_name}")
                continue

            for conf in metrics['max_probability']:
                confidence_data.append({
                    'Model': model_name,
                    'Confidence': conf
                })

        if not confidence_data:
            ax.text(0.5, 0.5, 'No confidence data available', ha='center', va='center')
            ax.set_title('Confidence Score Distributions')
            ax.axis('off')
            return

        df = pd.DataFrame(confidence_data)
        model_order = self._sort_models_consistently(list(df['Model'].unique()))

        # Create vertical violin plot to better show distributions
        parts = ax.violinplot(
            [df[df['Model'] == m]['Confidence'].values for m in model_order],
            positions=range(len(model_order)),
            showmeans=True,
            showmedians=True
        )

        # Style the violin plots with model-specific colors
        for i, model in enumerate(model_order):
            color = self._get_model_color(model)
            parts['bodies'][i].set_facecolor(color)
            parts['bodies'][i].set_edgecolor('black')
            parts['bodies'][i].set_alpha(0.7)

        # Style the internal components of the violin plot
        for partname in ['cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars']:
            if partname in parts:
                parts[partname].set_color('black')
                parts[partname].set_linewidth(1.5)
                parts[partname].set_alpha(0.8)

        # Configure axis labels and annotations
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels([f'M{i+1}' for i in range(len(model_order))])
        ax.set_xlabel('Models')
        ax.set_ylabel('Confidence (Max Probability)')
        ax.set_title('Confidence Score Distributions')
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean confidence annotations for quick reference
        for i, model in enumerate(model_order):
            mean_conf = df[df['Model'] == model]['Confidence'].mean()
            # Position text annotation within the plot area for better visibility
            ax.text(i, ax.get_ylim()[1] * 0.98, f'{mean_conf:.3f}',
                    ha='center', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Add explanatory note for model abbreviations
        model_mapping_str = ", ".join([f"M{i+1}={name}" for i, name in enumerate(model_order)])
        ax.text(
            0.02, 0.02,
            f'Model mapping: {model_mapping_str}',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=7,
            style='italic', alpha=0.7
        )

    def _plot_per_class_ece(self, ax) -> None:
        """Plot per-class Expected Calibration Error."""
        ece_data = []

        # Use consistent model ordering
        for model_name in self._sort_models_consistently(list(self.results.calibration_metrics.keys())):
            metrics = self.results.calibration_metrics[model_name]
            if 'per_class_ece' in metrics:
                for class_idx, ece in enumerate(metrics['per_class_ece']):
                    ece_data.append({
                        'Model': model_name,
                        'Class': str(class_idx),
                        'ECE': ece
                    })

        if ece_data:
            df = pd.DataFrame(ece_data)

            # Sort models to match color order
            model_order = self._sort_models_consistently(list(df['Model'].unique()))
            n_models = len(model_order)
            n_classes = len(df['Class'].unique())

            x = np.arange(n_classes)
            width = 0.8 / n_models

            for i, model in enumerate(model_order):
                model_data = df[df['Model'] == model]
                color = self._get_model_color(model)
                ax.bar(x + i * width, model_data['ECE'], width,
                       alpha=0.8, color=color)

            ax.set_xlabel('Class')
            ax.set_ylabel('Expected Calibration Error')
            ax.set_title('Per-Class Calibration Error')
            ax.set_xticks(x + width * (n_models - 1) / 2)
            ax.set_xticklabels([str(i) for i in range(n_classes)])
            # REMOVED: Individual legend - will use figure-level legend
            ax.grid(True, alpha=0.3, axis='y')

    def _plot_uncertainty_landscape(self, ax) -> None:
        """Plot uncertainty landscape with density contours for each model."""
        # Sort models for consistent ordering
        model_order = self._sort_models_consistently(list(self.results.confidence_metrics.keys()))

        # Track successful contour plots for legend
        successful_models = []

        # Plot contours for each model
        for model_name in model_order:
            # UPDATED: Access metrics from confidence_metrics
            metrics = self.results.confidence_metrics[model_name]

            # Safety checks for required keys
            if 'max_probability' not in metrics or 'entropy' not in metrics:
                logger.warning(f"Missing confidence metrics keys for model {model_name}")
                continue

            confidence = metrics['max_probability']
            entropy = metrics['entropy']
            color = self._get_model_color(model_name)

            if len(confidence) < 10:  # Skip if too few points
                continue

            try:
                # Create 2D KDE
                xy = np.vstack([confidence, entropy])
                kde = gaussian_kde(xy)

                # Create grid for contour plot
                conf_min, conf_max = confidence.min(), confidence.max()
                ent_min, ent_max = entropy.min(), entropy.max()

                # Add some padding
                conf_range = conf_max - conf_min
                ent_range = ent_max - ent_min
                conf_min -= 0.05 * conf_range
                conf_max += 0.05 * conf_range
                ent_min -= 0.05 * ent_range
                ent_max += 0.05 * ent_range

                # Create meshgrid
                xx = np.linspace(conf_min, conf_max, 100)
                yy = np.linspace(ent_min, ent_max, 100)
                X, Y = np.meshgrid(xx, yy)

                # Evaluate KDE on grid
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = kde(positions).reshape(X.shape)

                # Plot contours
                contours = ax.contour(X, Y, Z, levels=5, colors=[color],
                                      alpha=0.8, linewidths=2)
                ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

                # Plot filled contours with transparency
                ax.contourf(X, Y, Z, levels=5, colors=[color], alpha=0.1)

                successful_models.append(model_name)

            except Exception as e:
                logger.warning(f"Could not create density contours for {model_name}: {e}")
                # Fallback: plot a simple scatter with low alpha
                ax.scatter(confidence, entropy, color=color, alpha=0.1,
                           s=20)

        ax.set_xlabel('Confidence (Max Probability)')
        ax.set_ylabel('Entropy')
        ax.set_title('Uncertainty Landscape (Density Contours)')
        # REMOVED: Individual legend - will use figure-level legend
        ax.grid(True, alpha=0.1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)