"""
Training Dynamics Visualization Module
============================================================================

Creates visualizations for training dynamics analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from .base import BaseVisualizer
from ..constants import LOSS_PATTERNS, VAL_LOSS_PATTERNS, ACC_PATTERNS, VAL_ACC_PATTERNS
from ..utils import find_metric_in_history, smooth_curve


class TrainingDynamicsVisualizer(BaseVisualizer):
    """Creates training dynamics visualizations."""

    def create_visualizations(self) -> None:
        """Create comprehensive training dynamics visualizations."""
        if not self.results.training_history:
            return

        fig = plt.figure(figsize=(16, 12))
        gs = plt.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                         height_ratios=[1, 1, 0.8])

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

        plt.suptitle('Training Dynamics Analysis', fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.94, bottom=0.05, left=0.08, right=0.96)

        if self.config.save_plots:
            self._save_figure(fig, 'training_dynamics')
        plt.close(fig)

    def _plot_loss_curves(self, ax) -> None:
        """Plot training and validation loss curves."""
        for model_name in sorted(self.results.training_history.keys()):
            history = self.results.training_history[model_name]
            color = self.model_colors.get(model_name, '#333333')

            # Use smoothed curves if available
            if self.config.smooth_training_curves and model_name in self.results.training_metrics.smoothed_curves:
                curves = self.results.training_metrics.smoothed_curves[model_name]
            else:
                curves = history

            # Plot training loss using flexible pattern matching
            if curves is history:
                train_loss = find_metric_in_history(curves, LOSS_PATTERNS, exclude_prefixes=['val_'])
            else:
                train_loss = curves.get('loss', find_metric_in_history(curves, LOSS_PATTERNS, exclude_prefixes=['val_']))

            if train_loss is not None and len(train_loss) > 0:
                epochs = range(len(train_loss))
                ax.plot(epochs, train_loss, '-', color=color,
                       linewidth=2, label=f'{model_name} (train)', alpha=0.8)

            # Plot validation loss using flexible pattern matching
            if curves is history:
                val_loss = find_metric_in_history(curves, VAL_LOSS_PATTERNS)
            else:
                val_loss = curves.get('val_loss', find_metric_in_history(curves, VAL_LOSS_PATTERNS))

            if val_loss is not None and len(val_loss) > 0:
                epochs = range(len(val_loss))
                ax.plot(epochs, val_loss, '--', color=color,
                       linewidth=2, label=f'{model_name} (val)', alpha=0.8)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)  # Enable legend
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale often better for loss

    def _plot_accuracy_curves(self, ax) -> None:
        """Plot training and validation accuracy curves."""
        for model_name in sorted(self.results.training_history.keys()):
            history = self.results.training_history[model_name]
            color = self.model_colors.get(model_name, '#333333')

            # Use smoothed curves if available
            if self.config.smooth_training_curves and model_name in self.results.training_metrics.smoothed_curves:
                curves = self.results.training_metrics.smoothed_curves[model_name]
            else:
                curves = history

            # Plot training accuracy using flexible pattern matching
            if curves is history:
                train_acc = find_metric_in_history(curves, ACC_PATTERNS, exclude_prefixes=['val_'])
            else:
                train_acc = curves.get('accuracy', find_metric_in_history(curves, ACC_PATTERNS, exclude_prefixes=['val_']))

            if train_acc is not None and len(train_acc) > 0:
                epochs = range(len(train_acc))
                ax.plot(epochs, train_acc, '-', color=color,
                       linewidth=2, label=f'{model_name} (train)', alpha=0.8)

            # Plot validation accuracy using flexible pattern matching
            if curves is history:
                val_acc = find_metric_in_history(curves, VAL_ACC_PATTERNS)
            else:
                val_acc = curves.get('val_accuracy', find_metric_in_history(curves, VAL_ACC_PATTERNS))

            if val_acc is not None and len(val_acc) > 0:
                epochs = range(len(val_acc))
                ax.plot(epochs, val_acc, '--', color=color,
                       linewidth=2, label=f'{model_name} (val)', alpha=0.8)

                # Mark best epoch
                if model_name in self.results.training_metrics.peak_performance:
                    best_epoch = self.results.training_metrics.peak_performance[model_name]['epoch']
                    best_acc = self.results.training_metrics.peak_performance[model_name]['val_accuracy']
                    ax.scatter(best_epoch, best_acc, color=color, s=100,
                             marker='*', edgecolor='black', linewidth=1, zorder=5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    def _plot_overfitting_analysis(self, ax) -> None:
        """Plot dedicated overfitting analysis."""
        overfitting_data = []

        for model_name in sorted(self.results.training_history.keys()):
            history = self.results.training_history[model_name]

            train_loss = find_metric_in_history(history, LOSS_PATTERNS, exclude_prefixes=['val_'])
            val_loss = find_metric_in_history(history, VAL_LOSS_PATTERNS)

            if (train_loss is not None and val_loss is not None and
                    len(train_loss) > 0 and 0 < len(val_loss) == len(train_loss)):
                # Calculate gap over time
                gap = np.array(val_loss) - np.array(train_loss)

                color = self.model_colors.get(model_name, '#333333')
                epochs = range(len(gap))

                # Apply smoothing to gap if requested
                if self.config.smooth_training_curves:
                    gap_smooth = smooth_curve(gap, self.config.smoothing_window)
                    ax.plot(epochs, gap_smooth, '-', color=color,
                           linewidth=2.5, label=model_name)
                else:
                    ax.plot(epochs, gap, '-', color=color,
                           linewidth=2, label=model_name, alpha=0.8)

                # Add shaded region for positive gap (overfitting)
                ax.fill_between(epochs, 0, gap, where=(gap > 0),
                               color=color, alpha=0.1)

        # Add reference line at 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss - Training Loss')
        ax.set_title('Overfitting Analysis (Gap Evolution)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Enable legend
        ax.grid(True, alpha=0.3)

        # Add annotation
        ax.text(0.02, 0.98, 'Above 0 = Overfitting\nBelow 0 = Underfitting',
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
               fontsize=9)

    def _plot_best_epoch_performance(self, ax) -> None:
        """Plot best epoch performance comparison."""
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
                      s=200, color=color, alpha=0.8,
                      edgecolors='black', linewidth=2,
                      label=model)

            # Add model name as annotation
            ax.annotate(model,
                       (best_epochs[i], best_accs[i]),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor=color,
                               alpha=0.3))

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Best Validation Accuracy')
        ax.set_title('Peak Performance: Accuracy vs Convergence Speed')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # Set reasonable axis limits
        if best_epochs:
            ax.set_xlim(-1, max(best_epochs) + 1)
        ax.set_ylim(0, 1.05)

    def _plot_training_summary_table(self, ax) -> None:
        """Create comprehensive training summary table."""
        # Prepare data
        table_data = []
        headers = ['Model', 'Final Acc', 'Best Acc', 'Best Epoch', 'Conv. Speed',
                  'Stability', 'Overfit Index', 'Final Gap']

        for model_name in sorted(self.results.training_history.keys()):
            row = [model_name]

            # Final accuracy
            history = self.results.training_history.get(model_name, {})
            val_acc = find_metric_in_history(history, VAL_ACC_PATTERNS)
            final_acc = val_acc[-1] if val_acc and len(val_acc) > 0 else 0.0
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
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
            table[(0, i)].set_height(0.08)

        # Color model rows
        for i, row_data in enumerate(table_data, 1):
            model_name = row_data[0]
            color = self.model_colors.get(model_name, '#F5F5F5')
            light_color = self._lighten_color(color, 0.8)

            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor(light_color)
                cell.set_height(0.08)

                if j == 0:  # Model name
                    cell.set_text_props(weight='bold', fontsize=9)
                else:
                    cell.set_text_props(fontsize=9)

        ax.axis('off')
        ax.set_title('Training Metrics Summary', fontsize=12, fontweight='bold', pad=10)