"""
Calibration Visualization Module
Creates visualizations for calibration analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from .base import BaseVisualizer
from dl_techniques.utils.logger import logger


class CalibrationVisualizer(BaseVisualizer):
    """Creates calibration analysis visualizations."""

    def create_visualizations(self) -> None:
        """Create unified confidence and calibration visualizations."""
        if not self.results.calibration_metrics:
            return

        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Reliability Diagram
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_reliability_diagram(ax1)

        # 2. Confidence Distributions (Raincloud)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_confidence_raincloud(ax2)

        # 3. Per-Class ECE
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_per_class_ece(ax3)

        # 4. Uncertainty Landscape
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_uncertainty_landscape(ax4)

        plt.suptitle('Confidence and Calibration Analysis', fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'confidence_calibration_analysis')
        plt.close(fig)

    def _plot_reliability_diagram(self, ax) -> None:
        """Plot reliability diagram with confidence intervals."""
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')

        for model_name, rel_data in self.results.reliability_data.items():
            color = self.model_colors.get(model_name, '#333333')

            # Plot main line
            ax.plot(rel_data['bin_centers'], rel_data['bin_accuracies'],
                    'o-', color=color, label=model_name, linewidth=2, markersize=8)

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
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    def _plot_confidence_raincloud(self, ax) -> None:
        """Plot confidence distributions as raincloud plot."""
        confidence_data = []

        # UPDATED: Access confidence metrics from the correct location
        for model_name, metrics in self.results.confidence_metrics.items():
            # Safety check for required keys
            if 'max_probability' not in metrics:
                logger.warning(f"Missing 'max_probability' key for model {model_name}")
                continue

            for conf in metrics['max_probability']:
                confidence_data.append({
                    'Model': model_name,
                    'Confidence': conf
                })

        if confidence_data:
            df = pd.DataFrame(confidence_data)

            # Sort models to match color order
            model_order = sorted(df['Model'].unique())

            # Create violin plot with quartiles using matplotlib
            parts = ax.violinplot([df[df['Model'] == m]['Confidence'].values
                                   for m in model_order],
                                  positions=range(len(model_order)),
                                  vert=False, showmeans=False, showmedians=True,
                                  showextrema=True)

            # Color the violin plots and add to legend
            legend_elements = []
            for i, model in enumerate(model_order):
                color = self.model_colors.get(model, '#333333')
                parts['bodies'][i].set_facecolor(color)
                parts['bodies'][i].set_alpha(0.7)
                # Create legend entry
                legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=model))

            # Color other violin parts
            for partname in ['cmeans', 'cmaxes', 'cmins', 'cbars', 'cmedians', 'cquantiles']:
                if partname in parts:
                    parts[partname].set_color('black')
                    parts[partname].set_alpha(0.8)

            # Add model labels
            ax.set_yticks(range(len(model_order)))
            ax.set_yticklabels(model_order)

            # Add summary statistics as text
            for i, model in enumerate(model_order):
                model_data = df[df['Model'] == model]['Confidence']
                mean_conf = model_data.mean()
                ax.text(mean_conf, i, f'{mean_conf:.3f}',
                        ha='center', va='bottom', fontsize=9)

            ax.set_xlabel('Confidence (Max Probability)')
            ax.set_ylabel('')
            ax.set_title('Confidence Score Distributions')
            ax.grid(True, alpha=0.3, axis='x')
            # Add legend
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_per_class_ece(self, ax) -> None:
        """Plot per-class Expected Calibration Error."""
        ece_data = []

        for model_name, metrics in self.results.calibration_metrics.items():
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
            model_order = sorted(df['Model'].unique())
            n_models = len(model_order)
            n_classes = len(df['Class'].unique())

            x = np.arange(n_classes)
            width = 0.8 / n_models

            for i, model in enumerate(model_order):
                model_data = df[df['Model'] == model]
                color = self.model_colors.get(model, '#333333')
                ax.bar(x + i * width, model_data['ECE'], width,
                       label=model, alpha=0.8, color=color)

            ax.set_xlabel('Class')
            ax.set_ylabel('Expected Calibration Error')
            ax.set_title('Per-Class Calibration Error')
            ax.set_xticks(x + width * (n_models - 1) / 2)
            ax.set_xticklabels([str(i) for i in range(n_classes)])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')

    def _plot_uncertainty_landscape(self, ax) -> None:
        """Plot uncertainty landscape with density contours for each model."""
        # Sort models for consistent ordering
        model_order = sorted(self.results.confidence_metrics.keys())

        # Plot contours for each model
        legend_elements = []
        for model_name in model_order:
            # UPDATED: Access metrics from confidence_metrics
            metrics = self.results.confidence_metrics[model_name]

            # Safety checks for required keys
            if 'max_probability' not in metrics or 'entropy' not in metrics:
                logger.warning(f"Missing confidence metrics keys for model {model_name}")
                continue

            confidence = metrics['max_probability']
            entropy = metrics['entropy']
            color = self.model_colors.get(model_name, '#333333')

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
                ax.contourf(X, Y, Z, levels=5, colors=[color], alpha=0.2)

                # Create legend element
                legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3, label=model_name))

            except Exception as e:
                logger.warning(f"Could not create density contours for {model_name}: {e}")
                # Fallback: plot a simple scatter with low alpha
                scatter = ax.scatter(confidence, entropy, color=color, alpha=0.3,
                           s=20, label=model_name)
                legend_elements.append(scatter)

        ax.set_xlabel('Confidence (Max Probability)')
        ax.set_ylabel('Entropy')
        ax.set_title('Uncertainty Landscape (Density Contours)')
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)