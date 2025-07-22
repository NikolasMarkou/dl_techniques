"""
Weight Visualization Module

Creates visualizations for weight analysis results.
"""

import numpy as np
from typing import List
import matplotlib.pyplot as plt

from .base import BaseVisualizer
from ..constants import WEIGHT_HEALTH_L2_NORMALIZER, WEIGHT_HEALTH_SPARSITY_THRESHOLD
from dl_techniques.utils.logger import logger


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

    def _get_layer_order(self, model_name: str) -> List[str]:
        """Get the correct layer order for a model, using explicit ordering when available."""
        # Use explicit layer ordering from AnalysisResults
        if (hasattr(self.results, 'weight_stats_layer_order') and
            self.results.weight_stats_layer_order and
            model_name in self.results.weight_stats_layer_order):
            return self.results.weight_stats_layer_order[model_name]

        # Fallback to insertion order (which should be correct for Python 3.7+)
        if model_name in self.results.weight_stats:
            logger.warning(f"Using insertion order for {model_name} weights - "
                          "explicit ordering not available")
            return list(self.results.weight_stats[model_name].keys())

        return []

    def _plot_weight_learning_evolution(self, ax) -> None:
        """Plot how weight magnitudes evolve through network layers."""
        evolution_data = {}

        for model_name, weight_stats in self.results.weight_stats.items():
            layer_indices = []
            l2_norms = []
            mean_abs_weights = []

            # Use explicit layer ordering to ensure correct network depth representation
            layer_order = self._get_layer_order(model_name)

            for idx, layer_name in enumerate(layer_order):
                if layer_name in weight_stats:
                    stats = weight_stats[layer_name]
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
                if data['indices']:  # Check if we have data
                    color = self.model_colors.get(model_name, '#333333')
                    ax.plot(data['indices'], data['l2_norms'], 'o-',
                           label=f'{model_name}', color=color, linewidth=2, markersize=6)

            ax.set_xlabel('Layer Index (Network Depth)')
            ax.set_ylabel('L2 Norm')
            ax.set_title('Weight Magnitude Evolution Through Network Depth', fontsize=12, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            # Add insight text
            ax.text(0.02, 0.98, 'Higher values indicate larger weight magnitudes\nSteep changes suggest learning transitions',
                   transform=ax.transAxes, ha='left', va='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))
        else:
            ax.text(0.5, 0.5, 'No weight evolution data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Magnitude Evolution')
            ax.axis('off')

    def _plot_weight_health_heatmap(self, ax) -> None:
        """
        Create a comprehensive weight health heatmap with improved readability.
        """
        if not self.results.weight_stats:
            ax.text(0.5, 0.5, 'No weight statistics available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Health Heatmap')
            ax.axis('off')
            return

        # Prepare health metrics
        health_metrics = []
        models = sorted(self.results.weight_stats.keys())
        max_layers = getattr(self.config, 'max_layers_heatmap', 12)

        actual_max_layers = 0
        for model_name in models:
            layer_order = self._get_layer_order(model_name)
            actual_max_layers = max(actual_max_layers, len(layer_order))

        display_layers = min(max_layers, actual_max_layers)
        if actual_max_layers == 0:
            ax.text(0.5, 0.5, 'No layer ordering information available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Health Heatmap')
            ax.axis('off')
            return

        for model_name in models:
            weight_stats = self.results.weight_stats[model_name]
            model_health = []
            layer_order = self._get_layer_order(model_name)

            for layer_idx in range(display_layers):
                if layer_idx < len(layer_order):
                    layer_name = layer_order[layer_idx]
                    if layer_name in weight_stats:
                        stats = weight_stats[layer_name]
                        l2_norm = stats['norms']['l2']
                        norm_health = 1.0 / (1.0 + l2_norm / WEIGHT_HEALTH_L2_NORMALIZER)
                        sparsity = stats['distribution']['zero_fraction']
                        sparsity_health = 1.0 - min(sparsity, WEIGHT_HEALTH_SPARSITY_THRESHOLD)
                        weight_std = stats['basic']['std']
                        weight_mean = abs(stats['basic']['mean'])
                        dist_health = min(weight_std / (weight_mean + 1e-6), 1.0)
                        health_score = (norm_health + sparsity_health + dist_health) / 3.0
                        model_health.append(health_score)
                    else:
                        model_health.append(np.nan)
                else:
                    model_health.append(np.nan)
            health_metrics.append(model_health)

        if health_metrics and display_layers > 0:
            health_array = np.array(health_metrics)
            custom_cmap = plt.cm.get_cmap('RdYlGn').copy()
            custom_cmap.set_bad(color='lightgray')
            im = ax.imshow(health_array, cmap=custom_cmap, aspect='auto',
                           vmin=0, vmax=1, interpolation='nearest')

            ax.set_title('Weight Health Across Layers (Network Order)',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Layer Position in Network')
            ax.set_ylabel('Model')

            # Use abbreviated model names for y-axis readability
            model_abbreviations = [f'M{i+1}' for i in range(len(models))]
            ax.set_xticks(range(display_layers))
            ax.set_xticklabels([f'L{i}' for i in range(display_layers)], fontsize=9)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(model_abbreviations, fontsize=9)

            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Health Score', rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=9)

            for i in range(len(models)):
                for j in range(display_layers):
                    if j < len(health_array[i]):
                        value = health_array[i, j]
                        if np.isnan(value):
                            ax.text(j, i, 'N/A', ha='center', va='center',
                                   color='black', fontsize=8, fontweight='bold')
                        else:
                            text_color = 'white' if value < 0.5 else 'black'
                            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                                   color=text_color, fontsize=8, fontweight='bold')

            # Add an improved explanatory note with model mappings
            model_mapping_str = ", ".join([f"M{i+1}={name}" for i, name in enumerate(models)])
            full_note = (
                f"Note: Layers ordered by network depth. Gray cells (N/A) indicate models with fewer layers.\n"
                f"Models: {model_mapping_str}"
            )
            ax.text(0.0, -0.25, full_note,
                   transform=ax.transAxes, ha='left', va='top', fontsize=8,
                   style='italic', alpha=0.8, wrap=True)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for health heatmap',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Health Heatmap')
            ax.axis('off')