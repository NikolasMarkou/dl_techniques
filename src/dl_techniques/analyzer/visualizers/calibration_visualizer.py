"""
Calibration Visualization Module
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
# DECISION plan-2026-10-05-analyzer-audit/F-036
# An ad-hoc +1 continuity correction in the normal approximation, NOT a Bayesian or
# Wilson prior. Documented at the call site, which is why the figure no longer claims a
# "95% CI". Do NOT rename this to something implying a real interval.
BINOMIAL_EPSILON = 1
# Alpha for the measured in-bin mean confidence overlay (F-035).
ALPHA_MEASURED_CONFIDENCE = 0.7

# "No data" panel message (F-039)
NO_DATA_MESSAGE_FONT_SIZE = 12

# Text Positioning Constants
# DECISION plan-2026-10-05-analyzer-audit/F-037: these two were declared here and never
# referenced in this file; the identical pair is USED in `weight_visualizer.py` and
# `information_flow_visualizer.py:330` re-implemented the truncation inline as a literal.
# Do NOT reintroduce a third copy of the number.
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

            # DECISION plan-2026-10-05-analyzer-audit/F-035
            # Plot bin CENTERS on x and MASK the empty bins, rather than plotting the
            # `0.0` placeholder `compute_reliability_data` used to publish for an
            # empty bin's accuracy. With the default 10 bins and a confident model,
            # bins 0-8 are empty, so the curve previously dragged along y=0 for nine
            # tenths of the axis and a perfectly calibrated confident model rendered
            # as grossly miscalibrated. `calibration_metrics` now returns NaN there;
            # this is the consumer side of that contract, and BOTH arrays are masked.
            centers = np.asarray(rel_data['bin_centers'], dtype=float)
            accuracies = np.asarray(rel_data['bin_accuracies'], dtype=float)
            confidences = np.asarray(
                rel_data.get('bin_confidences', np.full_like(accuracies, np.nan)),
                dtype=float)

            if 'bin_counts' in rel_data:
                occupied = np.asarray(rel_data['bin_counts']) > 0
            else:
                occupied = np.isfinite(accuracies)
            # Never trust a finite accuracy in a bin with no samples, whatever a
            # hand-edited artifact claims.
            occupied &= np.isfinite(accuracies)

            if not occupied.any():
                logger.warning(
                    f"Model '{model_name}' has no occupied reliability bin; its "
                    f"reliability curve is not drawn.")
                continue

            ax.plot(centers[occupied], accuracies[occupied],
                    'o-', color=color, linewidth=LINE_WIDTH_STANDARD,
                    markersize=MARKER_SIZE_STANDARD)

            # DECISION plan-2026-10-05-analyzer-audit/F-036
            # The band is drawn only over OCCUPIED bins. For an empty bin the old code
            # evaluated `sqrt(0 * (1 - 0) / (0 + 1)) = 0` and drew a ZERO-HEIGHT band at
            # y=0, reinforcing the phantom point F-035 removed from the line.
            #
            # This remains a normal-approximation interval on the OBSERVED ACCURACY with
            # an ad-hoc `+1` continuity correction — NOT a Wilson interval, and not an
            # interval on the confidence-vs-accuracy gap the diagram is really about.
            # Both limits are documented rather than fixed: for a bin with p=0 and n=5 the
            # true Wilson interval is [0, 0.49] where this draws [0, 0]. The title no
            # longer claims "95% CI" because the correction is not one. Do NOT restore
            # that title without replacing the estimator.
            occupied_counts = np.asarray(
                rel_data.get('bin_counts', np.ones_like(centers)),
                dtype=float)[occupied]
            occupied_props = accuracies[occupied]
            se = np.sqrt(
                occupied_props * (1.0 - occupied_props)
                / (occupied_counts + BINOMIAL_EPSILON))

            ax.fill_between(centers[occupied],
                            occupied_props - CONFIDENCE_INTERVAL_MULTIPLIER * se,
                            occupied_props + CONFIDENCE_INTERVAL_MULTIPLIER * se,
                            alpha=ALPHA_CONFIDENCE_FILL, color=color)

            # Plot the MEASURED in-bin confidence too when it differs from the centre.
            # The band above is the interval on the observed accuracy; this is the
            # model's actual mean confidence in each bin, which is the quantity the
            # "predicted probability" axis claims to show.
            finite_conf = occupied & np.isfinite(confidences)
            if finite_conf.any() and not np.allclose(
                    confidences[finite_conf], centers[finite_conf], atol=1e-12):
                ax.plot(centers[finite_conf], confidences[finite_conf],
                        'x', color=color, markersize=MARKER_SIZE_STANDARD * 0.8,
                        alpha=ALPHA_MEASURED_CONFIDENCE)

        ax.set_xlabel('Predicted probability (bin centre; × = measured in-bin mean)')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagrams (occupied bins only)')
        # REMOVED: Individual legend - will use figure-level legend
        ax.grid(True, alpha=ALPHA_GRID_LIGHT)
        ax.set_xlim([AXIS_LIMIT_MIN, AXIS_LIMIT_MAX])
        ax.set_ylim([AXIS_LIMIT_MIN, AXIS_LIMIT_MAX])

    def _plot_confidence_distribution(self, ax) -> None:
        """Plot confidence distributions as a vertical violin plot."""
        confidence_data = []

        model_order_source = self._sort_models_consistently(
            list(self.results.calibration_metrics.keys())
        )

        for model_name in model_order_source:
            # Safely access confidence metrics and check for required keys
            if (model_name not in self.results.confidence_metrics or
                'max_probability' not in self.results.confidence_metrics[model_name]):
                logger.warning(f"Missing 'max_probability' data for model {model_name}, skipping confidence plot.")
                continue

            metrics = self.results.confidence_metrics[model_name]
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

        # Remove the x-axis labels and ticks entirely.
        # The main figure legend will identify the models.
        ax.set_xticks([])
        ax.set_xlabel('') # Clear the x-axis label

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
            all_classes = sorted(df['Class'].unique(), key=lambda c: int(c))

            # DECISION plan-2026-10-05-analyzer-audit/F-038
            # Index EACH MODEL's bars by CLASS LABEL, not by its own row order. `x` was
            # built from the UNION of classes while `model_data['ECE']` was that model's
            # own (shorter) series, so two models with different class counts — K=3 and
            # K=5 — raised `ValueError: shape mismatch` from `ax.bar`. That propagated
            # out of `create_visualizations`, which `ModelAnalyzer._create_visualizations`
            # calls with no try, so the whole `analyze()` aborted BEFORE
            # `save_results()` and every metric was lost to a plotting bug. A model with
            # fewer classes now simply contributes bars for the classes it has, and the
            # union axis is shared so the models remain comparable.
            x = np.arange(len(all_classes), dtype=float)
            width = BAR_WIDTH_FACTOR / n_models

            for i, model in enumerate(model_order):
                model_data = df[df['Model'] == model].set_index('Class')
                # Reindex onto the shared class axis; a class this model does not
                # report becomes NaN, which `bar` renders as absent rather than 0.0 —
                # a missing class is not a zero-error class.
                values = model_data['ECE'].reindex(all_classes)
                color = self._get_model_color(model)
                ax.bar(x + i * width, values.to_numpy(dtype=float), width,
                       alpha=ALPHA_BAR_STANDARD, color=color)

            ax.set_xlabel('Class')
            ax.set_ylabel('Expected Calibration Error')
            ax.set_title('Per-Class Calibration Error')
            ax.set_xticks(x + width * (n_models - 1) / 2)
            ax.set_xticklabels(all_classes)
            # REMOVED: Individual legend - will use figure-level legend
            ax.grid(True, alpha=ALPHA_GRID_STANDARD, axis='y')
        else:
            # DECISION plan-2026-10-05-analyzer-audit/F-039
            # This branch had NO `else`, so a run where no model carried
            # `per_class_ece` left the axes completely bare — no title, no message —
            # unlike every sibling panel. An axis with nothing on it reads as a
            # rendering failure rather than as "not computed".
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER,
                    'No per-class calibration data available',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=NO_DATA_MESSAGE_FONT_SIZE)
            ax.set_title('Per-Class Calibration Error')
            ax.axis('off')

    def _plot_uncertainty_landscape(self, ax) -> None:
        """Plot uncertainty landscape with density contours for each model."""
        # Iterate over models with calibration metrics as the source of truth
        model_order = self._sort_models_consistently(list(self.results.calibration_metrics.keys()))

        # Track successful contour plots for legend
        successful_models = []

        # Plot contours for each model
        for model_name in model_order:
            # access confidence metrics and required keys
            if (model_name not in self.results.confidence_metrics or
                'max_probability' not in self.results.confidence_metrics[model_name] or
                'entropy' not in self.results.confidence_metrics[model_name]):
                logger.warning(f"Missing confidence/entropy data for model {model_name}, skipping uncertainty plot.")
                continue

            metrics = self.results.confidence_metrics[model_name]
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