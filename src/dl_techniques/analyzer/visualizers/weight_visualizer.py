"""
Weight Visualization Module
============================================================================

Creates visualizations for weight analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from .base import BaseVisualizer
from ..constants import WEIGHT_HEALTH_L2_NORMALIZER, WEIGHT_HEALTH_SPARSITY_THRESHOLD


class WeightVisualizer(BaseVisualizer):
    """Creates weight analysis visualizations."""

    def create_visualizations(self) -> None:
        """Create unified Weight Learning Journey visualization."""
        if not self.results.weight_stats:
            return

        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 1, figure=fig, hspace=0.4, height_ratios=[1, 1])

        # Top: Weight magnitude evolution through layers
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_weight_learning_evolution(ax1)

        # Bottom: Weight health heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_weight_health_heatmap(ax2)

        plt.suptitle('Weight Learning Journey', fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'weight_learning_journey')
        plt.close(fig)

    def _plot_weight_learning_evolution(self, ax) -> None:
        """Plot how weight magnitudes evolve through network layers."""
        evolution_data = {}

        for model_name, weight_stats in self.results.weight_stats.items():
            layer_indices = []
            l2_norms = []
            mean_abs_weights = []

            # Sort layers by name to get consistent ordering
            sorted_layers = sorted(weight_stats.items(), key=lambda x: x[0])

            for idx, (layer_name, stats) in enumerate(sorted_layers):
                layer_indices.append(idx)
                l2_norms.append(stats['norms']['l2'])
                mean_abs_weights.append(abs(stats['basic']['mean']))

            evolution_data[model_name] = {
                'indices': layer_indices,
                'l2_norms': l2_norms,
                'mean_abs_weights': mean_abs_weights
            }

        if evolution_data:
            # Plot L2 norm evolution
            for model_name, data in evolution_data.items():
                color = self.model_colors.get(model_name, '#333333')
                ax.plot(data['indices'], data['l2_norms'], 'o-',
                        label=f'{model_name}', color=color, linewidth=2, markersize=6)

            ax.set_xlabel('Layer Index')
            ax.set_ylabel('L2 Norm')
            ax.set_title('Weight Magnitude Evolution Through Network Depth', fontsize=12, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            # Add insight text
            ax.text(0.02, 0.98,
                    'Higher values indicate larger weight magnitudes\nSteep changes suggest learning transitions',
                    transform=ax.transAxes, ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))
        else:
            ax.text(0.5, 0.5, 'No weight evolution data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Magnitude Evolution')
            ax.axis('off')

    def _plot_weight_health_heatmap(self, ax) -> None:
        """Create a comprehensive weight health heatmap."""
        if not self.results.weight_stats:
            ax.text(0.5, 0.5, 'No weight statistics available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Health Heatmap')
            ax.axis('off')
            return

        # Prepare health metrics - each model uses its own layer sequence
        health_metrics = []
        models = sorted(self.results.weight_stats.keys())
        max_layers = 12  # Show first 12 layers for each model

        for model_name in models:
            weight_stats = self.results.weight_stats[model_name]
            model_health = []

            # Get this model's layers in order (first 12)
            model_layers = sorted(list(weight_stats.keys()))[:max_layers]

            for layer_idx in range(max_layers):
                if layer_idx < len(model_layers):
                    layer_name = model_layers[layer_idx]
                    stats = weight_stats[layer_name]

                    # Calculate health score (0-1, higher is better)
                    # Normalize L2 norm (smaller is often better, but not too small)
                    l2_norm = stats['norms']['l2']
                    norm_health = 1.0 / (1.0 + l2_norm / WEIGHT_HEALTH_L2_NORMALIZER)

                    # Sparsity (moderate sparsity is ok, too much is bad)
                    sparsity = stats['distribution']['zero_fraction']
                    sparsity_health = 1.0 - min(sparsity, WEIGHT_HEALTH_SPARSITY_THRESHOLD)

                    # Weight distribution (closer to normal is better)
                    weight_std = stats['basic']['std']
                    weight_mean = abs(stats['basic']['mean'])
                    dist_health = min(weight_std / (weight_mean + 1e-6), 1.0)  # Ratio of std to mean

                    # Combined health score
                    health_score = (norm_health + sparsity_health + dist_health) / 3.0
                    model_health.append(health_score)
                else:
                    # Model has fewer layers than max_layers
                    model_health.append(0.0)

            health_metrics.append(model_health)

        if health_metrics:
            # Create heatmap
            health_array = np.array(health_metrics)

            im = ax.imshow(health_array, cmap='RdYlGn', aspect='auto',
                           vmin=0, vmax=1, interpolation='nearest')

            # Set labels
            ax.set_title('Weight Health Across Layers', fontsize=12, fontweight='bold')
            ax.set_xlabel('Layer Position')
            ax.set_ylabel('Model')

            # Set ticks
            ax.set_xticks(range(max_layers))
            ax.set_xticklabels([f'L{i}' for i in range(max_layers)], fontsize=9)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models, fontsize=9)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Health Score', rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=9)

            # Add value annotations for better readability
            for i in range(len(models)):
                for j in range(max_layers):
                    value = health_array[i, j]
                    text_color = 'white' if value < 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color=text_color, fontsize=8, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for health heatmap',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Health Heatmap')
            ax.axis('off')