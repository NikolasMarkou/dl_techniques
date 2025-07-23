# visualizers/training_dynamics_visualizer.py

"""
Training Dynamics Visualization Module

Creates visualizations for training dynamics analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseVisualizer
from ..constants import LOSS_PATTERNS, VAL_LOSS_PATTERNS, ACC_PATTERNS, VAL_ACC_PATTERNS
from ..utils import find_metric_in_history, smooth_curve

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

# Figure Layout Constants
FIGURE_SIZE = (16, 12)
GRID_HSPACE = 0.35
GRID_WSPACE = 0.3
GRID_HEIGHT_RATIOS = [1, 1, 0.8]
SUBPLOT_TOP = 0.94
SUBPLOT_BOTTOM = 0.05
SUBPLOT_LEFT = 0.08
SUBPLOT_RIGHT = 0.96

# Plot Styling Constants
LINE_WIDTH_STANDARD = 2
LINE_WIDTH_THICK = 2.5
LINE_ALPHA_STANDARD = 0.8
GRID_ALPHA = 0.3
FILL_ALPHA = 0.1
REFERENCE_LINE_ALPHA = 0.5
REFERENCE_LINE_WIDTH = 1
MARKER_SIZE = 100
MARKER_EDGE_WIDTH = 1
Y_AXIS_LIMIT_MAX = 1.05
XLIM_PADDING = 1

# Text Styling Constants
TITLE_FONT_SIZE = 18
LEGEND_FONT_SIZE = 8
ANNOTATION_FONT_SIZE = 9
TABLE_FONT_SIZE = 9
TABLE_HEADER_FONT_SIZE = 9
SUBTITLE_FONT_SIZE = 12

# Table Styling Constants
TABLE_SCALE_X = 1
TABLE_SCALE_Y = 1.5
TABLE_CELL_HEIGHT = 0.08
TABLE_HEADER_COLOR = '#E8E8E8'
TABLE_DEFAULT_COLOR = '#F5F5F5'
COLOR_LIGHTEN_FACTOR = 0.8

# Annotation Constants
ANNOTATION_OFFSET_X = 5
ANNOTATION_OFFSET_Y = 5
ANNOTATION_TEXT_X = 0.02
ANNOTATION_TEXT_Y = 0.98

# ---------------------------------------------------------------------

class TrainingDynamicsVisualizer(BaseVisualizer):
    """Creates training dynamics visualizations."""

    def create_visualizations(self) -> None:
        """Create comprehensive training dynamics visualizations."""
        if not self.results.training_history:
            return

        fig = plt.figure(figsize=FIGURE_SIZE)
        gs = plt.GridSpec(3, 2, figure=fig, hspace=GRID_HSPACE, wspace=GRID_WSPACE,
                         height_ratios=GRID_HEIGHT_RATIOS)

        # 1. Loss curves (train and validation)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_loss_curves(ax1)

        # 2. Accuracy curves (train and validation)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_accuracy_curves(ax2)

        # 3. Overfitting analysis
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_overfitting_analysis(ax3)

        # 4. Best epoch performance
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_best_epoch_performance(ax4)

        # 5. Training summary table
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_training_summary_table(ax5)

        plt.suptitle('Training Dynamics Analysis', fontsize=TITLE_FONT_SIZE, fontweight='bold')
        fig.subplots_adjust(top=SUBPLOT_TOP, bottom=SUBPLOT_BOTTOM,
                           left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT)

        if self.config.save_plots:
            self._save_figure(fig, 'training_dynamics')
        plt.close(fig)

    def _get_metric_data(self, model_name: str, metric_patterns: list,
                        exclude_prefixes: list = None) -> tuple:
        """
        Robustly get metric data from either original or smoothed curves.

        This addresses the ambiguity identified in the code review by creating
        a single, consistent method for retrieving metrics regardless of smoothing.

        Args:
            model_name: Name of the model
            metric_patterns: List of possible metric name patterns
            exclude_prefixes: Prefixes to exclude from matching

        Returns:
            Tuple of (metric_data, epochs) or (None, None) if not found
        """
        # First try smoothed curves if available and smoothing is enabled
        if (self.config.smooth_training_curves and
            self.results.training_metrics and
            model_name in self.results.training_metrics.smoothed_curves):

            smoothed_curves = self.results.training_metrics.smoothed_curves[model_name]
            metric_data = find_metric_in_history(smoothed_curves, metric_patterns, exclude_prefixes)

            if metric_data is not None and len(metric_data) > 0:
                return metric_data, range(len(metric_data))

        # Fallback to original history
        if model_name in self.results.training_history:
            original_history = self.results.training_history[model_name]
            metric_data = find_metric_in_history(original_history, metric_patterns, exclude_prefixes)

            if metric_data is not None and len(metric_data) > 0:
                return metric_data, range(len(metric_data))

        return None, None

    def _plot_loss_curves(self, ax) -> None:
        """Plot training and validation loss curves with robust metric retrieval."""
        for model_name in sorted(self.results.training_history.keys()):
            color = self.model_colors.get(model_name, '#333333')

            # Plot training loss - using robust method
            train_loss, epochs = self._get_metric_data(
                model_name, LOSS_PATTERNS, exclude_prefixes=['val_']
            )
            if train_loss is not None:
                ax.plot(epochs, train_loss, '-', color=color,
                       linewidth=LINE_WIDTH_STANDARD, label=f'{model_name} (train)',
                       alpha=LINE_ALPHA_STANDARD)

            # Plot validation loss - using robust method
            val_loss, epochs = self._get_metric_data(
                model_name, VAL_LOSS_PATTERNS
            )
            if val_loss is not None:
                ax.plot(epochs, val_loss, '--', color=color,
                       linewidth=LINE_WIDTH_STANDARD, label=f'{model_name} (val)',
                       alpha=LINE_ALPHA_STANDARD)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss Evolution')
        # DO NOT ADD LEGEND
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.set_yscale('log')  # Log scale often better for loss

    def _plot_accuracy_curves(self, ax) -> None:
        """Plot training and validation accuracy curves with robust metric retrieval."""
        for model_name in sorted(self.results.training_history.keys()):
            color = self.model_colors.get(model_name, '#333333')

            # Plot training accuracy - using robust method
            train_acc, epochs = self._get_metric_data(
                model_name, ACC_PATTERNS, exclude_prefixes=['val_']
            )
            if train_acc is not None:
                ax.plot(epochs, train_acc, '-', color=color,
                       linewidth=LINE_WIDTH_STANDARD, label=f'{model_name} (train)',
                       alpha=LINE_ALPHA_STANDARD)

            # Plot validation accuracy - using robust method
            val_acc, epochs = self._get_metric_data(
                model_name, VAL_ACC_PATTERNS
            )
            if val_acc is not None:
                ax.plot(epochs, val_acc, '--', color=color,
                       linewidth=LINE_WIDTH_STANDARD, label=f'{model_name} (val)',
                       alpha=LINE_ALPHA_STANDARD)

                # Mark best epoch
                if (self.results.training_metrics and
                    model_name in self.results.training_metrics.peak_performance):
                    best_epoch = self.results.training_metrics.peak_performance[model_name]['epoch']
                    best_acc = self.results.training_metrics.peak_performance[model_name]['val_accuracy']
                    ax.scatter(best_epoch, best_acc, color=color, s=MARKER_SIZE,
                             marker='*', edgecolor='black', linewidth=MARKER_EDGE_WIDTH, zorder=5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.set_ylim(0, Y_AXIS_LIMIT_MAX)

    def _plot_overfitting_analysis(self, ax) -> None:
        """Plot dedicated overfitting analysis with robust metric retrieval."""
        for model_name in sorted(self.results.training_history.keys()):
            color = self.model_colors.get(model_name, '#333333')

            # Get training and validation loss using robust method
            train_loss, _ = self._get_metric_data(
                model_name, LOSS_PATTERNS, exclude_prefixes=['val_']
            )
            val_loss, epochs = self._get_metric_data(
                model_name, VAL_LOSS_PATTERNS
            )

            if (train_loss is not None and val_loss is not None and
                    len(train_loss) > 0 and len(val_loss) == len(train_loss)):

                # Calculate gap over time
                gap = np.array(val_loss) - np.array(train_loss)

                # Apply additional smoothing to gap if requested
                if self.config.smooth_training_curves:
                    gap_smooth = smooth_curve(gap, self.config.smoothing_window)
                    ax.plot(epochs, gap_smooth, '-', color=color,
                           linewidth=LINE_WIDTH_THICK, label=model_name)
                else:
                    ax.plot(epochs, gap, '-', color=color,
                           linewidth=LINE_WIDTH_STANDARD, label=model_name,
                           alpha=LINE_ALPHA_STANDARD)

                # Add shaded region for positive gap (overfitting)
                ax.fill_between(epochs, 0, gap, where=(gap > 0),
                               color=color, alpha=FILL_ALPHA)

        # Add reference line at 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=REFERENCE_LINE_ALPHA,
                  linewidth=REFERENCE_LINE_WIDTH)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss - Training Loss')
        ax.set_title('Overfitting Analysis (Gap Evolution)')
        # DO NOT ADD LEGEND
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=GRID_ALPHA)

        # Add annotation
        ax.text(ANNOTATION_TEXT_X, ANNOTATION_TEXT_Y, 'Above 0 = Overfitting\nBelow 0 = Underfitting',
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=LINE_ALPHA_STANDARD),
               fontsize=ANNOTATION_FONT_SIZE)

    def _plot_best_epoch_performance(self, ax) -> None:
        """Plot best epoch performance comparison."""
        if not self.results.training_metrics:
            ax.text(0.5, 0.5, 'No training metrics available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Best Epoch Performance')
            ax.axis('off')
            return

        peak_data = self.results.training_metrics.peak_performance

        if not peak_data:
            ax.text(0.5, 0.5, 'No peak performance data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Best Epoch Performance')
            ax.axis('off')
            return

        models = sorted(peak_data.keys())
        best_accs = [peak_data[m]['val_accuracy'] for m in models]
        best_epochs = [peak_data[m]['epoch'] for m in models]

        # Create scatter plot instead of confusing bar chart
        for i, model in enumerate(models):
            color = self.model_colors.get(model, '#333333')
            ax.scatter(best_epochs[i], best_accs[i],
                      s=MARKER_SIZE, color=color, alpha=LINE_ALPHA_STANDARD,
                      edgecolors='black', linewidth=MARKER_EDGE_WIDTH,
                      label=model)

            # NO NEED FOR ANNOTATION
            # # Add model name as annotation
            # ax.annotate(model,
            #            (best_epochs[i], best_accs[i]),
            #            xytext=(ANNOTATION_OFFSET_X, ANNOTATION_OFFSET_Y),
            #            textcoords='offset points',
            #            fontsize=ANNOTATION_FONT_SIZE,
            #            bbox=dict(boxstyle='round,pad=0.3',
            #                    facecolor=color,
            #                    alpha=0.3))

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Best Validation Accuracy')
        ax.set_title('Peak Performance: Accuracy vs Convergence Speed')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=GRID_ALPHA)

        # Set reasonable axis limits
        if best_epochs:
            ax.set_xlim(-XLIM_PADDING, max(best_epochs) + XLIM_PADDING)
        ax.set_ylim(0, Y_AXIS_LIMIT_MAX)

    def _plot_training_summary_table(self, ax) -> None:
        """Create comprehensive training summary table."""
        if not self.results.training_metrics:
            ax.text(0.5, 0.5, 'No training metrics available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Metrics Summary')
            ax.axis('off')
            return

        # Prepare data
        table_data = []
        headers = ['Model', 'Final Acc', 'Best Acc', 'Best Epoch', 'Conv. Speed',
                  'Stability', 'Overfit Index', 'Final Gap']

        for model_name in sorted(self.results.training_history.keys()):
            row = [model_name]

            # Final accuracy - using robust method
            val_acc, _ = self._get_metric_data(model_name, VAL_ACC_PATTERNS)
            final_acc = val_acc[-1] if val_acc is not None and len(val_acc) > 0 else 0.0
            row.append(f'{final_acc:.3f}')

            # Best accuracy and epoch
            peak = self.results.training_metrics.peak_performance.get(model_name, {})
            best_acc = peak.get('val_accuracy', 0.0)
            best_epoch = peak.get('epoch', 0)
            row.append(f'{best_acc:.3f}')
            row.append(f'{best_epoch}')

            # Convergence speed
            conv_speed = self.results.training_metrics.epochs_to_convergence.get(model_name, 0)
            row.append(f'{conv_speed}')

            # Stability score
            stability = self.results.training_metrics.training_stability_score.get(model_name, 0.0)
            row.append(f'{stability:.3f}')

            # Overfitting index
            overfit = self.results.training_metrics.overfitting_index.get(model_name, 0.0)
            row.append(f'{overfit:.3f}')

            # Final gap
            final_gap = self.results.training_metrics.final_gap.get(model_name, 0.0)
            row.append(f'{final_gap:.3f}')

            table_data.append(row)

        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(TABLE_FONT_SIZE)
        table.scale(TABLE_SCALE_X, TABLE_SCALE_Y)

        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor(TABLE_HEADER_COLOR)
            table[(0, i)].set_text_props(weight='bold')
            table[(0, i)].set_height(TABLE_CELL_HEIGHT)

        # Color model rows
        for i, row_data in enumerate(table_data, 1):
            model_name = row_data[0]
            color = self.model_colors.get(model_name, TABLE_DEFAULT_COLOR)
            light_color = self._lighten_color(color, COLOR_LIGHTEN_FACTOR)

            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor(light_color)
                cell.set_height(TABLE_CELL_HEIGHT)

                if j == 0:  # Model name
                    cell.set_text_props(weight='bold', fontsize=TABLE_HEADER_FONT_SIZE)
                else:
                    cell.set_text_props(fontsize=TABLE_FONT_SIZE)

        ax.axis('off')
        ax.set_title('Training Metrics Summary', fontsize=SUBTITLE_FONT_SIZE,
                    fontweight='bold', pad=10)