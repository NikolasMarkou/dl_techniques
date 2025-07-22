"""
Summary Dashboard Visualization Module

Creates comprehensive summary dashboard with key insights across all analyses.
Provides an integrated view of model performance, training dynamics, calibration,
and confidence metrics in a single dashboard layout.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .base import BaseVisualizer
from ..utils import find_model_metric, truncate_model_name
from dl_techniques.utils.logger import logger


class SummaryVisualizer(BaseVisualizer):
    """
    Creates comprehensive summary dashboard visualization.

    This visualizer generates a 2x2 grid dashboard containing:
    1. Performance metrics table with training insights
    2. Model similarity plot based on weight PCA
    3. Confidence distribution profiles across models
    4. Calibration performance landscape comparison

    The dashboard adapts based on available data (e.g., training history presence).
    """

    def create_visualizations(self) -> None:
        """
        Create a comprehensive summary dashboard with training insights.

        Generates a 2x2 subplot layout containing key model analysis results.
        The layout dynamically adapts based on available analysis data.
        """
        # Create figure with optimized layout for dashboard presentation
        fig: Figure = plt.figure(figsize=(16, 10))
        gs = plt.GridSpec(
            2, 2,
            figure=fig,
            hspace=0.35,
            wspace=0.25,
            height_ratios=[1, 1],
            width_ratios=[1.2, 1]
        )

        # Performance metrics table (top-left, wider for readability)
        ax1: Axes = fig.add_subplot(gs[0, 0])
        self._plot_performance_table(ax1)

        # Model similarity analysis (top-right)
        ax2: Axes = fig.add_subplot(gs[0, 1])
        self._plot_model_similarity(ax2)

        # Confidence distribution profiles (bottom-left)
        ax3: Axes = fig.add_subplot(gs[1, 0])
        self._plot_confidence_profile_summary(ax3)

        # Calibration performance comparison (bottom-right)
        ax4: Axes = fig.add_subplot(gs[1, 1])
        self._plot_calibration_performance_summary(ax4)

        # Configure overall dashboard styling
        plt.suptitle(
            'Model Analysis Summary Dashboard',
            fontsize=18,
            fontweight='bold'
        )
        fig.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.96)

        # Save and cleanup
        if self.config.save_plots:
            self._save_figure(fig, 'summary_dashboard')
        plt.close(fig)

    def _plot_performance_table(self, ax: Axes) -> None:
        """
        Create a comprehensive performance table including training metrics.

        Args:
            ax: Matplotlib axes object for the table visualization.

        Notes:
            Table structure adapts based on training data availability:
            - With training data: Final Acc, Best Acc, Loss, ECE, Brier, Conv Speed, Overfit
            - Without training data: Accuracy, Loss, ECE, Brier Score, Mean Entropy
        """
        # Prepare table data structure
        table_data: List[List[str]] = []

        # Determine table headers based on available training metrics
        if (self.results.training_metrics and
            self.results.training_metrics.peak_performance):
            headers = [
                'Model', 'Final Acc', 'Best Acc', 'Loss',
                'ECE', 'Brier', 'Conv Speed', 'Overfit'
            ]
        else:
            headers = [
                'Model', 'Accuracy', 'Loss', 'ECE',
                'Brier Score', 'Mean Entropy'
            ]

        # Process each model's metrics
        for model_name in sorted(self.results.model_metrics.keys()):
            # truncate model name to never overflow from the table
            row_data: List[str] = [truncate_model_name(model_name)]
            model_metrics: Dict[str, Any] = self.results.model_metrics.get(
                model_name, {}
            )

            if (self.results.training_metrics and
                self.results.training_metrics.peak_performance):
                # Enhanced table with training insights
                self._add_training_metrics_to_row(
                    row_data, model_name, model_metrics
                )
            else:
                # Standard table without training data
                self._add_standard_metrics_to_row(
                    row_data, model_name, model_metrics
                )

            table_data.append(row_data)

        # Create and style the performance table
        self._create_styled_table(ax, table_data, headers)

    def _add_training_metrics_to_row(
        self,
        row_data: List[str],
        model_name: str,
        model_metrics: Dict[str, Any]
    ) -> None:
        """
        Add training-specific metrics to table row.

        Args:
            row_data: List to append metric values to.
            model_name: Name of the current model.
            model_metrics: Dictionary containing model performance metrics.
        """
        # Final accuracy from evaluation
        accuracy_keys = [
            'accuracy', 'compile_metrics', 'val_accuracy',
            'categorical_accuracy', 'sparse_categorical_accuracy'
        ]
        final_acc = find_model_metric(model_metrics, accuracy_keys, 0.0)
        row_data.append(f'{final_acc:.3f}')

        # Best accuracy from training history
        peak_performance = self.results.training_metrics.peak_performance.get(
            model_name, {}
        )
        best_acc = peak_performance.get('val_accuracy', final_acc)
        row_data.append(f'{best_acc:.3f}')

        # Loss value
        loss = find_model_metric(model_metrics, ['loss'], 0.0)
        row_data.append(f'{loss:.3f}')

        # Calibration metrics
        calibration_metrics = self.results.calibration_metrics.get(
            model_name, {}
        )
        ece = calibration_metrics.get('ece', 0.0)
        brier = calibration_metrics.get('brier_score', 0.0)
        row_data.extend([f'{ece:.3f}', f'{brier:.3f}'])

        # Training efficiency metrics
        conv_speed = self.results.training_metrics.epochs_to_convergence.get(
            model_name, 0
        )
        overfit = self.results.training_metrics.overfitting_index.get(
            model_name, 0.0
        )
        row_data.extend([f'{conv_speed}', f'{overfit:+.3f}'])

    def _add_standard_metrics_to_row(
        self,
        row_data: List[str],
        model_name: str,
        model_metrics: Dict[str, Any]
    ) -> None:
        """
        Add standard metrics to table row (without training data).

        Args:
            row_data: List to append metric values to.
            model_name: Name of the current model.
            model_metrics: Dictionary containing model performance metrics.
        """
        # Accuracy metric
        accuracy_keys = [
            'accuracy', 'compile_metrics', 'val_accuracy',
            'categorical_accuracy', 'sparse_categorical_accuracy'
        ]
        acc = find_model_metric(model_metrics, accuracy_keys, 0.0)
        row_data.append(f'{acc:.3f}')

        # Loss metric
        loss = find_model_metric(model_metrics, ['loss'], 0.0)
        row_data.append(f'{loss:.3f}')

        # Calibration metrics
        calibration_metrics = self.results.calibration_metrics.get(
            model_name, {}
        )
        ece = calibration_metrics.get('ece', 0.0)
        brier = calibration_metrics.get('brier_score', 0.0)
        row_data.extend([f'{ece:.3f}', f'{brier:.3f}'])

        # Confidence entropy from confidence metrics
        confidence_metrics = self.results.confidence_metrics.get(
            model_name, {}
        )
        entropy = confidence_metrics.get('mean_entropy', 0.0)
        row_data.append(f'{entropy:.3f}')

    def _create_styled_table(
        self,
        ax: Axes,
        table_data: List[List[str]],
        headers: List[str]
    ) -> None:
        """
        Create and style the performance metrics table.

        Args:
            ax: Matplotlib axes object for the table.
            table_data: 2D list containing table cell data.
            headers: List of column header strings.
        """
        # Create the table with proper positioning
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        # Configure table typography
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Style header row with consistent formatting
        for i in range(len(headers)):
            header_cell = table[(0, i)]
            header_cell.set_facecolor('#E8E8E8')
            header_cell.set_text_props(weight='bold')
            header_cell.set_height(0.08)

        # Style data rows with model-specific colors
        for i, row_data in enumerate(table_data, 1):
            model_name = row_data[0]
            model_color = self.model_colors.get(model_name, '#F5F5F5')
            light_color = self._lighten_color(model_color, 0.8)

            # Apply styling to each cell in the row
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor(light_color)
                cell.set_height(0.08)

                # Configure text properties based on column type
                if j == 0:  # Model name column
                    cell.set_text_props(weight='bold', fontsize=9)
                else:  # Metric columns
                    cell.set_text_props(fontsize=9)

        # Remove axes for clean table presentation
        ax.axis('off')

    def _plot_calibration_performance_summary(self, ax: Axes) -> None:
        """
        Plot calibration performance comparison using fixed thresholds.

        Args:
            ax: Matplotlib axes object for the calibration plot.

        Notes:
            Uses established calibration thresholds:
            - ECE < 0.05 (5%) indicates good calibration
            - Brier < 0.15 indicates good probabilistic performance
        """
        if not self.results.calibration_metrics:
            self._plot_no_data_message(
                ax,
                'No calibration data available',
                'Calibration Performance'
            )
            return

        # Extract calibration metrics for all models
        models = sorted(self.results.calibration_metrics.keys())
        ece_values: List[float] = []
        brier_values: List[float] = []

        for model_name in models:
            metrics = self.results.calibration_metrics[model_name]
            ece_values.append(metrics.get('ece', 0.0))
            brier_values.append(metrics.get('brier_score', 0.0))

        # Create scatter plot for ECE vs Brier Score comparison
        for i, model in enumerate(models):
            color = self.model_colors.get(model, '#333333')
            ax.scatter(
                ece_values[i], brier_values[i],
                s=150, color=color, alpha=0.8,
                edgecolors='black', linewidth=1.5,
                label=model
            )

        # Add reference lines for calibration quality assessment
        if ece_values and brier_values:
            self._add_calibration_reference_lines(ax)
            self._add_calibration_quadrant_labels(ax)

        # Configure plot aesthetics and labels
        ax.set_xlabel('Expected Calibration Error (ECE) - Lower is Better')
        ax.set_ylabel('Brier Score - Lower is Better')
        ax.set_title('Calibration Performance Landscape')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set adaptive axis limits with margin for visibility
        if ece_values and brier_values:
            ax.set_xlim(left=0, right=max(ece_values) * 1.1)
            ax.set_ylim(bottom=0, top=max(brier_values) * 1.1)
        else:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

    def _add_calibration_reference_lines(self, ax: Axes) -> None:
        """
        Add reference lines for calibration quality thresholds.

        Args:
            ax: Matplotlib axes object to add reference lines to.
        """
        # Established thresholds for good calibration performance
        GOOD_ECE_THRESHOLD = 0.05  # 5% ECE threshold
        GOOD_BRIER_THRESHOLD = 0.15  # Reasonable Brier score threshold

        # Add visual reference lines for performance benchmarks
        ax.axvline(
            GOOD_ECE_THRESHOLD,
            color='darkgreen',
            linestyle='--',
            alpha=0.7,
            label=f'Good ECE (<{GOOD_ECE_THRESHOLD})'
        )
        ax.axhline(
            GOOD_BRIER_THRESHOLD,
            color='darkgreen',
            linestyle='--',
            alpha=0.7,
            label=f'Good Brier (<{GOOD_BRIER_THRESHOLD})'
        )

    def _add_calibration_quadrant_labels(self, ax: Axes) -> None:
        """
        Add interpretative labels for calibration performance quadrants.

        Args:
            ax: Matplotlib axes object to add quadrant labels to.
        """
        GOOD_ECE_THRESHOLD = 0.05
        GOOD_BRIER_THRESHOLD = 0.15

        # Good calibration and certainty quadrant
        ax.text(
            GOOD_ECE_THRESHOLD / 2, GOOD_BRIER_THRESHOLD / 2,
            'Good Calibration\nGood Certainty',
            ha='center', va='center', color='darkgreen',
            alpha=0.8, fontsize=9, weight='bold'
        )

        # Poor calibration and certainty quadrant
        ax.text(
            GOOD_ECE_THRESHOLD * 2, GOOD_BRIER_THRESHOLD * 2,
            'Poor Calibration\nPoor Certainty',
            ha='center', va='center', color='darkred',
            alpha=0.8, fontsize=9, weight='bold'
        )

    def _plot_model_similarity(self, ax: Axes) -> None:
        """
        Plot model similarity based on weight space PCA analysis.

        Args:
            ax: Matplotlib axes object for the similarity plot.

        Notes:
            Visualizes models in 2D PCA space of concatenated weight statistics.
            Distance between points indicates similarity in weight distributions.
        """
        if not self.results.weight_pca:
            self._plot_no_data_message(
                ax,
                'No weight PCA data available',
                'Model Similarity (Weight Space)'
            )
            return

        # Extract PCA results
        components = self.results.weight_pca['components']
        labels = self.results.weight_pca['labels']
        explained_var = self.results.weight_pca['explained_variance']

        # Validate PCA component dimensionality
        if len(components) == 0 or len(components[0]) < 2:
            self._plot_no_data_message(
                ax,
                'Insufficient PCA components for visualization',
                'Model Similarity (Weight Space)'
            )
            return

        # Create scatter plot with model-specific colors
        for i, (label, comp) in enumerate(zip(labels, components)):
            color = self.model_colors.get(label, '#333333')
            ax.scatter(
                comp[0], comp[1], c=[color], label=label,
                s=200, alpha=0.8, edgecolors='black', linewidth=2
            )

            # Add connecting lines to origin for reference
            ax.plot(
                [0, comp[0]], [0, comp[1]],
                '--', color=color, alpha=0.3
            )

        # Mark origin point
        ax.scatter(0, 0, c='black', s=50, marker='x')

        # Configure plot labels and styling
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax.set_ylabel(
            f'PC2 ({explained_var[1]:.1%})' if len(explained_var) > 1 else 'PC2'
        )
        ax.set_title('Model Similarity (Concatenated Weight Statistics)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    def _plot_confidence_profile_summary(self, ax: Axes) -> None:
        """
        Plot confidence distribution summary across models.

        Args:
            ax: Matplotlib axes object for the confidence profile plot.

        Notes:
            Uses violin plots to show confidence distribution shapes.
            Includes mean annotations for quick comparison.
        """
        # Prepare confidence data for visualization
        confidence_data: List[Dict[str, Any]] = []

        for model_name, metrics in self.results.confidence_metrics.items():
            # Validate presence of required confidence metrics
            if 'max_probability' not in metrics:
                logger.warning(
                    f"Missing 'max_probability' key for model {model_name}"
                )
                continue

            # Extract confidence values for this model
            for conf in metrics['max_probability']:
                confidence_data.append({
                    'Model': model_name,
                    'Confidence': conf
                })

        if confidence_data:
            self._create_confidence_violin_plot(ax, confidence_data)
        else:
            self._plot_no_data_message(
                ax,
                'No confidence data available',
                'Confidence Distribution Profiles'
            )

    def _create_confidence_violin_plot(
        self,
        ax: Axes,
        confidence_data: List[Dict[str, Any]]
    ) -> None:
        """
        Create violin plot for confidence distributions.

        Args:
            ax: Matplotlib axes object for the violin plot.
            confidence_data: List of dictionaries containing model confidence data.
        """
        df = pd.DataFrame(confidence_data)
        model_order = sorted(df['Model'].unique())

        # Create violin plot with distribution statistics
        parts = ax.violinplot(
            [df[df['Model'] == m]['Confidence'].values for m in model_order],
            positions=range(len(model_order)),
            showmeans=True,
            showmedians=True
        )

        # Style violin plots with model-specific colors
        legend_elements: List[patches.Rectangle] = []
        for i, model in enumerate(model_order):
            color = self.model_colors.get(model, '#333333')
            parts['bodies'][i].set_facecolor(color)
            parts['bodies'][i].set_alpha(0.6)

            # Create legend entries for better visualization
            legend_elements.append(
                patches.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=color, alpha=0.6,
                    edgecolor='black', linewidth=1,
                    label=model
                )
            )

        # Style violin plot components
        for partname in ['cmeans', 'cmaxes', 'cmins', 'cbars', 'cmedians']:
            if partname in parts:
                parts[partname].set_color('black')
                parts[partname].set_alpha(0.8)

        # Configure axis labels and annotations
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels([f'M{i+1}' for i in range(len(model_order))], fontsize=9)

        # Add mean confidence annotations for quick reference
        for i, model in enumerate(model_order):
            model_data = df[df['Model'] == model]['Confidence']
            mean_conf = model_data.mean()
            ax.text(
                i, mean_conf, f'{mean_conf:.3f}',
                ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
            )

        # Configure plot styling and labels
        ax.set_ylabel('Confidence (Max Probability)')
        ax.set_xlabel('Models')
        ax.set_title('Confidence Distribution Profiles')
        ax.grid(True, alpha=0.3, axis='y')

        # DO NOT ADD LEGEND
        # # Add conditional legend for detailed view
        # if self.config.verbose:
        #     ax.legend(
        #         handles=legend_elements,
        #         bbox_to_anchor=(1.05, 1),
        #         loc='upper left',
        #         fontsize=8
        #     )

        # Add explanatory note for model abbreviations
        ax.text(
            0.02, 0.02,
            'M1, M2, etc. correspond to models in legend',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=7,
            style='italic', alpha=0.7
        )

    def _plot_no_data_message(
        self,
        ax: Axes,
        message: str,
        title: str
    ) -> None:
        """
        Display a standardized no-data message on an axes.

        Args:
            ax: Matplotlib axes object to display message on.
            message: Text message to display.
            title: Title for the subplot.
        """
        ax.text(
            0.5, 0.5, message,
            ha='center', va='center',
            transform=ax.transAxes, fontsize=12
        )
        ax.set_title(title)
        ax.axis('off')