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

from dl_techniques.utils.logger import logger
from .base import BaseVisualizer

# Figure Layout Constants
FIGURE_SIZE = (14, 10)
GRID_HSPACE = 0.3
GRID_WSPACE = 0.3
SUBPLOT_TOP = 0.93
SUBPLOT_BOTTOM = 0.1
SUBPLOT_LEFT = 0.1
SUBPLOT_RIGHT = 0.92

# Text Styling Constants
TITLE_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 8
CONTOUR_LABEL_FONT_SIZE = 8

# Plot Styling Constants
LINE_WIDTH_STANDARD = 2
LINE_WIDTH_MEDIUM = 1.5
MARKER_SIZE_STANDARD = 5
SCATTER_SIZE_FALLBACK = 20

# Alpha Constants
ALPHA_PERFECT_CALIBRATION = 0.2
ALPHA_CONFIDENCE_FILL = 0.2
ALPHA_GRID_LIGHT = 0.2
ALPHA_GRID_STANDARD = 0.3
ALPHA_GRID_MINIMAL = 0.1
ALPHA_VIOLIN_BODY = 0.7
ALPHA_VIOLIN_INTERNAL = 0.8
ALPHA_BAR_STANDARD = 0.8
ALPHA_CONTOUR_LINE = 0.8
ALPHA_CONTOUR_FILL = 0.1
ALPHA_SCATTER_FALLBACK = 0.1
ALPHA_ANNOTATION_BOX = 0.7

# Reliability Diagram Constants
CONFIDENCE_INTERVAL_MULTIPLIER = 1.96
AXIS_LIMIT_MIN = 0
AXIS_LIMIT_MAX = 1
BINOMIAL_EPSILON = 1

# Text Positioning Constants
MODEL_NAME_TRUNCATE_LENGTH = 8
MODEL_NAME_ELLIPSIS = '...'
ANNOTATION_Y_POSITION_FACTOR = 0.98
ANNOTATION_X_CENTER = 0.5
ANNOTATION_Y_CENTER = 0.5

# Bar Chart Constants
BAR_WIDTH_FACTOR = 0.8

# Uncertainty Landscape Constants
KDE_GRID_RESOLUTION = 100
DENSITY_CONTOUR_LEVELS = 5
PADDING_FACTOR = 0.05
MIN_POINTS_FOR_KDE = 10

# ---------------------------------------------------------------------

class CalibrationVisualizer(BaseVisualizer):
    """Creates calibration analysis visualizations with centralized legend."""

    def create_visualizations(self) -> None:
        """Create unified confidence and calibration visualizations with single legend."""
        if not self.results.calibration_metrics:
            return

        fig = plt.figure(figsize=FIGURE_SIZE)
        gs = plt.GridSpec(2, 2, figure=fig, hspace=GRID_HSPACE, wspace=GRID_WSPACE)

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

        plt.suptitle('Confidence and Calibration Analysis',
                    fontsize=TITLE_FONT_SIZE, fontweight='bold')
        fig.subplots_adjust(top=SUBPLOT_TOP, bottom=SUBPLOT_BOTTOM,
                           left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT)

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
        ax.plot([AXIS_LIMIT_MIN, AXIS_LIMIT_MAX], [AXIS_LIMIT_MIN, AXIS_LIMIT_MAX],
               'k--', alpha=ALPHA_PERFECT_CALIBRATION, label='Perfect Calibration')

        # Use consistent model ordering
        for model_name in self._sort_models_consistently(list(self.results.reliability_data.keys())):
            rel_data = self.results.reliability_data[model_name]
            color = self._get_model_color(model_name)

            # Plot main line
            ax.plot(rel_data['bin_centers'], rel_data['bin_accuracies'],
                    'o-', color=color, linewidth=LINE_WIDTH_STANDARD,
                    markersize=MARKER_SIZE_STANDARD)

            # Add shaded confidence region if we have sample counts
            if 'bin_counts' in rel_data:
                # Simple confidence interval based on binomial proportion
                counts = rel_data['bin_counts']
                props = rel_data['bin_accuracies']
                se = np.sqrt(props * (1 - props) / (counts + BINOMIAL_EPSILON))

                ax.fill_between(rel_data['bin_centers'],
                                props - CONFIDENCE_INTERVAL_MULTIPLIER * se,
                                props + CONFIDENCE_INTERVAL_MULTIPLIER * se,
                                alpha=ALPHA_CONFIDENCE_FILL, color=color)

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagrams with 95% CI')
        # REMOVED: Individual legend - will use figure-level legend
        ax.grid(True, alpha=ALPHA_GRID_LIGHT)
        ax.set_xlim([AXIS_LIMIT_MIN, AXIS_LIMIT_MAX])
        ax.set_ylim([AXIS_LIMIT_MIN, AXIS_LIMIT_MAX])

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
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER,
                   'No confidence data available', ha='center', va='center')
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
            parts['bodies'][i].set_alpha(ALPHA_VIOLIN_BODY)

        # Style the internal components of the violin plot
        for partname in ['cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars']:
            if partname in parts:
                parts[partname].set_color('black')
                parts[partname].set_linewidth(LINE_WIDTH_MEDIUM)
                parts[partname].set_alpha(ALPHA_VIOLIN_INTERNAL)

        # Configure axis labels and annotations
        ax.set_xticks(range(len(model_order)))
        # Use truncated model names directly instead of M1, M2 abbreviations
        truncated_names = [
            name[:MODEL_NAME_TRUNCATE_LENGTH] + MODEL_NAME_ELLIPSIS
            if len(name) > MODEL_NAME_TRUNCATE_LENGTH else name
            for name in model_order
        ]
        ax.set_xticklabels(truncated_names, rotation=45, ha='right')
        ax.set_xlabel('Models')
        ax.set_ylabel('Confidence (Max Probability)')
        ax.set_title('Confidence Score Distributions')
        ax.grid(True, alpha=ALPHA_GRID_STANDARD, axis='y')

        # Add mean confidence annotations for quick reference
        for i, model in enumerate(model_order):
            mean_conf = df[df['Model'] == model]['Confidence'].mean()
            # Position text annotation within the plot area for better visibility
            ax.text(i, ax.get_ylim()[1] * ANNOTATION_Y_POSITION_FACTOR, f'{mean_conf:.3f}',
                    ha='center', va='top', fontsize=ANNOTATION_FONT_SIZE,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             alpha=ALPHA_ANNOTATION_BOX))

        # Removed model mapping annotation

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
            width = BAR_WIDTH_FACTOR / n_models

            for i, model in enumerate(model_order):
                model_data = df[df['Model'] == model]
                color = self._get_model_color(model)
                ax.bar(x + i * width, model_data['ECE'], width,
                       alpha=ALPHA_BAR_STANDARD, color=color)

            ax.set_xlabel('Class')
            ax.set_ylabel('Expected Calibration Error')
            ax.set_title('Per-Class Calibration Error')
            ax.set_xticks(x + width * (n_models - 1) / 2)
            ax.set_xticklabels([str(i) for i in range(n_classes)])
            # REMOVED: Individual legend - will use figure-level legend
            ax.grid(True, alpha=ALPHA_GRID_STANDARD, axis='y')

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

            if len(confidence) < MIN_POINTS_FOR_KDE:  # Skip if too few points
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
                conf_min -= PADDING_FACTOR * conf_range
                conf_max += PADDING_FACTOR * conf_range
                ent_min -= PADDING_FACTOR * ent_range
                ent_max += PADDING_FACTOR * ent_range

                # Create meshgrid
                xx = np.linspace(conf_min, conf_max, KDE_GRID_RESOLUTION)
                yy = np.linspace(ent_min, ent_max, KDE_GRID_RESOLUTION)
                X, Y = np.meshgrid(xx, yy)

                # Evaluate KDE on grid
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = kde(positions).reshape(X.shape)

                # Plot contours
                contours = ax.contour(X, Y, Z, levels=DENSITY_CONTOUR_LEVELS, colors=[color],
                                      alpha=ALPHA_CONTOUR_LINE, linewidths=LINE_WIDTH_STANDARD)
                ax.clabel(contours, inline=True, fontsize=CONTOUR_LABEL_FONT_SIZE, fmt='%.2f')

                # Plot filled contours with transparency
                ax.contourf(X, Y, Z, levels=DENSITY_CONTOUR_LEVELS, colors=[color],
                           alpha=ALPHA_CONTOUR_FILL)

                successful_models.append(model_name)

            except Exception as e:
                logger.warning(f"Could not create density contours for {model_name}: {e}")
                # Fallback: plot a simple scatter with low alpha
                ax.scatter(confidence, entropy, color=color, alpha=ALPHA_SCATTER_FALLBACK,
                           s=SCATTER_SIZE_FALLBACK)

        ax.set_xlabel('Confidence (Max Probability)')
        ax.set_ylabel('Entropy')
        ax.set_title('Uncertainty Landscape (Density Contours)')
        # REMOVED: Individual legend - will use figure-level legend
        ax.grid(True, alpha=ALPHA_GRID_MINIMAL)
        ax.set_xlim(AXIS_LIMIT_MIN, AXIS_LIMIT_MAX)
        ax.set_ylim(AXIS_LIMIT_MIN, None)