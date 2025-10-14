"""
Enhanced Weight Visualization Module

Creates visualizations for weight analysis results with centralized legend management.
"""

import numpy as np
from typing import List
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .base import BaseVisualizer
from ..constants import WEIGHT_HEALTH_L2_NORMALIZER, WEIGHT_HEALTH_SPARSITY_THRESHOLD

# ---------------------------------------------------------------------

# Figure Layout Constants
FIGURE_SIZE = (14, 10)
GRID_HSPACE = 0.4
GRID_HEIGHT_RATIOS = [1, 1]
SUBPLOT_TOP = 0.93
SUBPLOT_BOTTOM = 0.1
SUBPLOT_LEFT = 0.1
SUBPLOT_RIGHT = 0.92

# Text Styling Constants
TITLE_FONT_SIZE = 16
SUBTITLE_FONT_SIZE = 12
LABEL_FONT_SIZE = 9
ANNOTATION_FONT_SIZE = 8
CELL_TEXT_FONT_SIZE = 8

# Plot Styling Constants
LINE_WIDTH_STANDARD = 2
MARKER_SIZE_STANDARD = 6
GRID_ALPHA = 0.3
BBOX_ALPHA_LIGHT = 0.3
BBOX_ALPHA_STANDARD = 0.8

# Heatmap Constants
DEFAULT_MAX_LAYERS = 12
HEATMAP_VMIN = 0
HEATMAP_VMAX = 1
HEATMAP_INTERPOLATION = 'nearest'
HEATMAP_ASPECT = 'auto'
COLORBAR_SHRINK = 0.8
COLORBAR_ROTATION = 270
COLORBAR_LABELPAD = 20

# Health Analysis Constants
HEALTH_SCORE_COMPONENTS = 3.0
EPSILON_SMALL = 1e-6
TEXT_COLOR_THRESHOLD = 0.5

# Text and Layout Constants
MODEL_NAME_TRUNCATE_LENGTH = 8
MODEL_NAME_ELLIPSIS = '...'
ANNOTATION_X_LEFT = 0.02
ANNOTATION_Y_TOP = 0.98
ANNOTATION_X_CENTER = 0.5
ANNOTATION_Y_CENTER = 0.5
EXPLANATORY_NOTE_Y_OFFSET = -0.25

# Color Constants
BBOX_COLOR_LIGHT = 'lightblue'
BBOX_COLOR_GRAY = 'lightgray'
TEXT_COLOR_WHITE = 'white'
TEXT_COLOR_BLACK = 'black'

# ---------------------------------------------------------------------

class WeightVisualizer(BaseVisualizer):
    """Creates weight analysis visualizations with centralized legend."""

    def create_visualizations(self) -> None:
        """Create unified Weight Learning Journey visualization with single legend."""
        if not self.results.weight_stats:
            return

        fig = plt.figure(figsize=FIGURE_SIZE)
        gs = plt.GridSpec(2, 1, figure=fig, hspace=GRID_HSPACE,
                         height_ratios=GRID_HEIGHT_RATIOS)

        # Top: Weight magnitude evolution through layers
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_weight_learning_evolution(ax1)

        # Bottom: Weight health heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_weight_health_heatmap(ax2)

        # Add single figure-level legend
        models_with_data = self._get_models_with_data()
        if models_with_data:
            self._create_figure_legend(fig, title="Models", specific_models=models_with_data)

        plt.suptitle('Weight Learning Journey', fontsize=TITLE_FONT_SIZE, fontweight='bold')
        fig.subplots_adjust(top=SUBPLOT_TOP, bottom=SUBPLOT_BOTTOM,
                           left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT)

        if self.config.save_plots:
            self._save_figure(fig, 'weight_learning_journey')
        plt.close(fig)

    def _get_models_with_data(self) -> List[str]:
        """Get models that have weight statistics data."""
        models_with_data = []
        for model_name in self.model_order:
            if model_name in self.results.weight_stats and self.results.weight_stats[model_name]:
                models_with_data.append(model_name)
        return models_with_data

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

        # Use consistent model ordering
        for model_name in self._sort_models_consistently(list(self.results.weight_stats.keys())):
            weight_stats = self.results.weight_stats[model_name]
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
            # Plot L2 norm evolution with consistent colors and ordering
            for model_name in self._sort_models_consistently(list(evolution_data.keys())):
                data = evolution_data[model_name]
                if data['indices']:  # Check if we have data
                    color = self._get_model_color(model_name)
                    ax.plot(data['indices'], data['l2_norms'], 'o-',
                           color=color, linewidth=LINE_WIDTH_STANDARD,
                           markersize=MARKER_SIZE_STANDARD)

            ax.set_xlabel('Layer Index (Network Depth)')
            ax.set_ylabel('L2 Norm')
            ax.set_title('Weight Magnitude Evolution Through Network Depth',
                        fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
            # REMOVED: Individual legend - will use figure-level legend
            ax.grid(True, alpha=GRID_ALPHA)

            # Add insight text
            ax.text(ANNOTATION_X_LEFT, ANNOTATION_Y_TOP,
                   'Higher values indicate larger weight magnitudes\nSteep changes suggest learning transitions',
                   transform=ax.transAxes, ha='left', va='top',
                   fontsize=LABEL_FONT_SIZE,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=BBOX_COLOR_LIGHT,
                            alpha=BBOX_ALPHA_LIGHT))
        else:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER,
                   'No weight evolution data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Magnitude Evolution')
            ax.axis('off')

    def _plot_weight_health_heatmap(self, ax) -> None:
        """
        Create a comprehensive weight health heatmap with improved readability.
        """
        if not self.results.weight_stats:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER,
                   'No weight statistics available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Health Heatmap')
            ax.axis('off')
            return

        # Prepare health metrics with consistent model ordering
        health_metrics = []
        models = self._sort_models_consistently(list(self.results.weight_stats.keys()))
        max_layers = getattr(self.config, 'max_layers_heatmap', DEFAULT_MAX_LAYERS)

        actual_max_layers = 0
        for model_name in models:
            layer_order = self._get_layer_order(model_name)
            actual_max_layers = max(actual_max_layers, len(layer_order))

        display_layers = min(max_layers, actual_max_layers)
        if actual_max_layers == 0:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER,
                   'No layer ordering information available',
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
                        dist_health = min(weight_std / (weight_mean + EPSILON_SMALL), 1.0)
                        health_score = (norm_health + sparsity_health + dist_health) / HEALTH_SCORE_COMPONENTS
                        model_health.append(health_score)
                    else:
                        model_health.append(np.nan)
                else:
                    model_health.append(np.nan)
            health_metrics.append(model_health)

        if health_metrics and display_layers > 0:
            health_array = np.array(health_metrics)
            custom_cmap = plt.cm.get_cmap('RdYlGn').copy()
            custom_cmap.set_bad(color=BBOX_COLOR_GRAY)
            im = ax.imshow(health_array, cmap=custom_cmap, aspect=HEATMAP_ASPECT,
                           vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX,
                           interpolation=HEATMAP_INTERPOLATION)

            ax.set_title('Weight Health Across Layers (Network Order)',
                        fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
            ax.set_xlabel('Layer Position in Network')
            ax.set_ylabel('Model')

            # Use truncated model names for y-axis readability with consistent ordering
            truncated_names = [
                name[:MODEL_NAME_TRUNCATE_LENGTH] + MODEL_NAME_ELLIPSIS
                if len(name) > MODEL_NAME_TRUNCATE_LENGTH else name
                for name in models
            ]
            ax.set_xticks(range(display_layers))
            ax.set_xticklabels([f'L{i}' for i in range(display_layers)],
                              fontsize=LABEL_FONT_SIZE)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(truncated_names, fontsize=LABEL_FONT_SIZE)

            cbar = plt.colorbar(im, ax=ax, shrink=COLORBAR_SHRINK)
            cbar.set_label('Health Score', rotation=COLORBAR_ROTATION,
                          labelpad=COLORBAR_LABELPAD)
            cbar.ax.tick_params(labelsize=LABEL_FONT_SIZE)

            for i in range(len(models)):
                for j in range(display_layers):
                    if j < len(health_array[i]):
                        value = health_array[i, j]
                        if np.isnan(value):
                            ax.text(j, i, 'N/A', ha='center', va='center',
                                   color=TEXT_COLOR_BLACK, fontsize=CELL_TEXT_FONT_SIZE,
                                   fontweight='bold')
                        else:
                            text_color = (TEXT_COLOR_WHITE if value < TEXT_COLOR_THRESHOLD
                                        else TEXT_COLOR_BLACK)
                            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                                   color=text_color, fontsize=CELL_TEXT_FONT_SIZE,
                                   fontweight='bold')

            # Add an improved explanatory note without model mappings
            full_note = (
                "Note: Layers ordered by network depth. Gray cells (N/A) indicate models with fewer layers.\n"
                "Model names are truncated for display purposes."
            )
            ax.text(0.0, EXPLANATORY_NOTE_Y_OFFSET, full_note,
                   transform=ax.transAxes, ha='left', va='top',
                   fontsize=ANNOTATION_FONT_SIZE,
                   style='italic', alpha=BBOX_ALPHA_STANDARD, wrap=True)
        else:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER,
                   'Insufficient data for health heatmap',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Health Heatmap')
            ax.axis('off')

# ---------------------------------------------------------------------