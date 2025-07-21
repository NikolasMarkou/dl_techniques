"""
Summary Dashboard Visualization Module
============================================================================

Creates summary dashboard with key insights across all analyses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .base import BaseVisualizer
from dl_techniques.utils.logger import logger


class SummaryVisualizer(BaseVisualizer):
    """Creates summary dashboard visualization."""

    def create_visualizations(self) -> None:
        """Create a summary dashboard with training insights."""
        fig = plt.figure(figsize=(16, 10))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                         height_ratios=[1, 1], width_ratios=[1.2, 1])

        # 1. Performance Table (with training metrics)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_performance_table(ax1)

        # 2. Model Similarity (unchanged)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_model_similarity(ax2)

        # 3. Confidence Distribution Profiles (unchanged)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_confidence_profile_summary(ax3)

        # 4. Calibration Performance Comparison (unchanged)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_calibration_performance_summary(ax4)

        plt.suptitle('Model Analysis Summary Dashboard', fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.96)

        if self.config.save_plots:
            self._save_figure(fig, 'summary_dashboard')
        plt.close(fig)

    def _plot_performance_table(self, ax) -> None:
        """Create performance table including training metrics."""
        # Prepare data for the table
        table_data = []

        # Adjust headers based on whether we have training data
        if self.results.training_metrics and self.results.training_metrics.peak_performance:
            headers = ['Model', 'Final Acc', 'Best Acc', 'Loss', 'ECE', 'Brier', 'Conv Speed', 'Overfit']
        else:
            headers = ['Model', 'Accuracy', 'Loss', 'ECE', 'Brier Score', 'Mean Entropy']

        for model_name in sorted(self.results.model_metrics.keys()):
            row_data = [model_name]

            # Get model metrics
            model_metrics = self.results.model_metrics.get(model_name, {})

            if self.results.training_metrics and self.results.training_metrics.peak_performance:
                # Include training insights
                # Final accuracy
                acc = model_metrics.get('accuracy')
                if acc is None:
                    acc = model_metrics.get('compile_metrics')
                if acc is None:
                    acc = model_metrics.get('val_accuracy')
                if acc is None:
                    acc = 0.0
                row_data.append(f'{acc:.3f}')

                # Best accuracy from training
                peak = self.results.training_metrics.peak_performance.get(model_name, {})
                best_acc = peak.get('val_accuracy', acc)
                row_data.append(f'{best_acc:.3f}')

                # Loss
                loss = model_metrics.get('loss', 0.0)
                row_data.append(f'{loss:.3f}')

                # ECE
                ece = self.results.calibration_metrics.get(model_name, {}).get('ece', 0.0)
                row_data.append(f'{ece:.3f}')

                # Brier Score
                brier = self.results.calibration_metrics.get(model_name, {}).get('brier_score', 0.0)
                row_data.append(f'{brier:.3f}')

                # Convergence speed
                conv_speed = self.results.training_metrics.epochs_to_convergence.get(model_name, 0)
                row_data.append(f'{conv_speed}')

                # Overfitting index
                overfit = self.results.training_metrics.overfitting_index.get(model_name, 0.0)
                row_data.append(f'{overfit:+.3f}')  # Show sign

            else:
                # Original table without training data
                # Accuracy
                acc = model_metrics.get('accuracy')
                if acc is None:
                    acc = model_metrics.get('compile_metrics')
                if acc is None:
                    acc = model_metrics.get('val_accuracy')
                if acc is None:
                    acc = 0.0
                row_data.append(f'{acc:.3f}')

                # Loss
                loss = model_metrics.get('loss', 0.0)
                row_data.append(f'{loss:.3f}')

                # ECE
                ece = self.results.calibration_metrics.get(model_name, {}).get('ece', 0.0)
                row_data.append(f'{ece:.3f}')

                # Brier Score
                brier = self.results.calibration_metrics.get(model_name, {}).get('brier_score', 0.0)
                row_data.append(f'{brier:.3f}')

                # Mean Entropy
                entropy = self.results.calibration_metrics.get(model_name, {}).get('mean_entropy', 0.0)
                row_data.append(f'{entropy:.3f}')

            table_data.append(row_data)

        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Color the header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
            table[(0, i)].set_height(0.08)

        # Color the model rows
        for i, row_data in enumerate(table_data, 1):
            model_name = row_data[0]
            color = self.model_colors.get(model_name, '#F5F5F5')

            # Apply light version of model color to the entire row
            light_color = self._lighten_color(color, 0.8)
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor(light_color)
                cell.set_height(0.08)

                # Ensure text fits properly
                if j == 0:  # Model name column
                    cell.set_text_props(weight='bold', fontsize=9)
                else:  # Metric columns
                    cell.set_text_props(fontsize=9)

        # Remove axis
        ax.axis('off')

    def _plot_calibration_performance_summary(self, ax) -> None:
        """Plot calibration performance comparison."""
        if not self.results.calibration_metrics:
            ax.text(0.5, 0.5, 'No calibration data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Calibration Performance')
            ax.axis('off')
            return

        # Prepare data for plotting
        models = sorted(self.results.calibration_metrics.keys())
        ece_values = []
        brier_values = []

        for model_name in models:
            metrics = self.results.calibration_metrics[model_name]
            ece_values.append(metrics.get('ece', 0.0))
            brier_values.append(metrics.get('brier_score', 0.0))

        # Create a scatter plot showing ECE vs Brier Score
        for i, model in enumerate(models):
            color = self.model_colors.get(model, '#333333')
            ax.scatter(ece_values[i], brier_values[i],
                      s=200, color=color, alpha=0.8,
                      edgecolors='black', linewidth=2,
                      label=model)

        # Add reference lines
        if ece_values and brier_values:
            # Reference line where ECE equals Brier Score
            max_val = max(max(ece_values), max(brier_values))
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5,
                   label='ECE = Brier Score')

            # Add quadrants for interpretation
            mean_ece = np.mean(ece_values)
            mean_brier = np.mean(brier_values)

            ax.axvline(mean_ece, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(mean_brier, color='gray', linestyle=':', alpha=0.5)

            # Add quadrant labels
            ax.text(0.02, 0.98, 'Well Calibrated\nLow Uncertainty',
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                   fontsize=8)

            ax.text(0.98, 0.02, 'Poorly Calibrated\nHigh Uncertainty',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                   fontsize=8)

        ax.set_xlabel('Expected Calibration Error (ECE)')
        ax.set_ylabel('Brier Score')
        ax.set_title('Calibration Performance Landscape')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # Set axis limits to start from 0
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    def _plot_model_similarity(self, ax) -> None:
        """Plot model similarity based on weight PCA.

        Note: PCA is performed on concatenated weight statistics from all layers
        to ensure fair comparison between models with identical layer structures.
        """
        if not self.results.weight_pca:
            ax.text(0.5, 0.5, 'No weight PCA data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Similarity (Weight Space)')
            ax.axis('off')
            return

        components = self.results.weight_pca['components']
        labels = self.results.weight_pca['labels']
        explained_var = self.results.weight_pca['explained_variance']

        # Validate PCA components
        if len(components) == 0 or len(components[0]) < 2:
            ax.text(0.5, 0.5, 'Insufficient PCA components for visualization',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Similarity (Weight Space)')
            ax.axis('off')
            return

        # Create scatter plot using consistent colors
        for i, (label, comp) in enumerate(zip(labels, components)):
            color = self.model_colors.get(label, '#333333')
            ax.scatter(comp[0], comp[1], c=[color], label=label,
                      s=200, alpha=0.8, edgecolors='black', linewidth=2)

            # Add connecting lines to origin
            ax.plot([0, comp[0]], [0, comp[1]], '--', color=color, alpha=0.3)

        # Add origin
        ax.scatter(0, 0, c='black', s=50, marker='x')

        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})' if len(explained_var) > 1 else 'PC2')
        ax.set_title('Model Similarity (Concatenated Weight Statistics)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    def _plot_confidence_profile_summary(self, ax) -> None:
        """Plot confidence distribution summary."""
        confidence_data = []

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
            model_order = sorted(df['Model'].unique())

            # Create violin plot
            parts = ax.violinplot([df[df['Model'] == m]['Confidence'].values
                                  for m in model_order],
                                 positions=range(len(model_order)),
                                 showmeans=True, showmedians=True)

            # Customize colors using consistent palette
            for i, model in enumerate(model_order):
                color = self.model_colors.get(model, '#333333')
                parts['bodies'][i].set_facecolor(color)
                parts['bodies'][i].set_alpha(0.6)

            ax.set_xticks(range(len(model_order)))
            ax.set_xticklabels([])  # Remove model names from x-axis
            ax.set_ylabel('Confidence (Max Probability)')
            ax.set_title('Confidence Distribution Profiles')
            ax.grid(True, alpha=0.3, axis='y')