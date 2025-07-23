"""
Information Flow Visualization Module

Creates visualizations for information flow analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseVisualizer
from ..constants import ACTIVATION_MAGNITUDE_NORMALIZER, LAYER_SPECIALIZATION_MAX_RANK
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

# Figure Layout Constants
FIGURE_SIZE = (14, 10)
GRID_HSPACE = 0.3
GRID_WSPACE = 0.3
SUBPLOT_TOP = 0.93
SUBPLOT_BOTTOM = 0.1
SUBPLOT_LEFT = 0.1
SUBPLOT_RIGHT = 0.95

# Text Styling Constants
TITLE_FONT_SIZE = 16
SUBTITLE_FONT_SIZE = 12
SMALL_TITLE_FONT_SIZE = 10
LABEL_FONT_SIZE = 8
COLORBAR_FONT_SIZE = 7

# Plot Styling Constants
LINE_WIDTH_STANDARD = 2
MARKER_SIZE_SMALL = 6
MARKER_SIZE_LARGE = 8
ALPHA_FILL = 0.2
ALPHA_STANDARD = 0.8
ALPHA_GRID = 0.3

# Subplot Layout Constants
SUB_WSPACE_WIDE = 0.4
SUB_WSPACE_NARROW = 0.1
SUB_HSPACE_STANDARD = 0.4
SUB_HSPACE_NARROW = 0.1
HEIGHT_RATIOS_EQUAL = [1, 1]
HEIGHT_RATIOS_BOTTOM_HEAVY = [1, 1.5]

# Heatmap Constants
HEATMAP_VMIN = 0
HEATMAP_VMAX = 1.0
COLORBAR_SHRINK = 0.6
INTERPOLATION_METHOD = 'nearest'

# Specialization Analysis Constants
BAR_LABEL_OFFSET = 0.02
AXIS_LIMIT_MIN = 0
AXIS_LIMIT_MAX = 1
SATURATION_THRESHOLD = 0.9
BALANCE_SCORE_MULTIPLIER = 2
SPECIALIZATION_SCORE_COMPONENTS = 3.0

# Health Metrics Constants
ANNOTATION_X_CENTER = 0.5
ANNOTATION_Y_CENTER = 0.5
TITLE_PAD = 20

# ---------------------------------------------------------------------

class InformationFlowVisualizer(BaseVisualizer):
    """Creates information flow visualizations."""

    def create_visualizations(self) -> None:
        """Create information flow visualizations."""
        if not self.results.information_flow:
            return

        fig = plt.figure(figsize=FIGURE_SIZE)
        gs = plt.GridSpec(2, 2, figure=fig, hspace=GRID_HSPACE, wspace=GRID_WSPACE)

        # Top row: Flow overview
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_activation_flow_overview(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_effective_rank_evolution(ax2)

        # Bottom row: Actionable insights
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_activation_health_dashboard(ax3)

        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_layer_specialization_analysis(ax4)

        plt.suptitle('Information Flow and Activation Analysis',
                    fontsize=TITLE_FONT_SIZE, fontweight='bold')
        fig.subplots_adjust(top=SUBPLOT_TOP, bottom=SUBPLOT_BOTTOM,
                           left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT)

        if self.config.save_plots:
            self._save_figure(fig, 'information_flow_analysis')
        plt.close(fig)

    def _get_ordered_layer_analysis(self, model_name: str) -> list:
        """
        Get layer analysis in consistent order for reliable visualization.

        Since information flow analysis preserves the order from the extraction model
        which is built from model.layers in order, we can rely on the iteration order.
        However, we make this explicit for robustness.

        Args:
            model_name: Name of the model

        Returns:
            List of (layer_name, analysis) tuples in network order
        """
        if model_name not in self.results.information_flow:
            return []

        layer_analysis = self.results.information_flow[model_name]

        # The information flow analyzer creates layers in the order they appear in the model
        # We can rely on this order, but we make it explicit here for clarity
        return list(layer_analysis.items())

    def _plot_activation_flow_overview(self, ax) -> None:
        """Plot activation statistics evolution through layers."""
        for model_name in sorted(self.results.information_flow.keys()):
            ordered_layers = self._get_ordered_layer_analysis(model_name)

            if not ordered_layers:
                continue

            means = []
            stds = []
            layer_positions = []

            for i, (layer_name, analysis) in enumerate(ordered_layers):
                means.append(analysis['mean_activation'])
                stds.append(analysis['std_activation'])
                layer_positions.append(i)

            if means:  # Check if we have data
                # Plot mean with std as shaded region
                means = np.array(means)
                stds = np.array(stds)

                color = self.model_colors.get(model_name, '#333333')
                line = ax.plot(layer_positions, means, 'o-', label=f'{model_name}',
                               linewidth=LINE_WIDTH_STANDARD, markersize=MARKER_SIZE_SMALL,
                               color=color)
                ax.fill_between(layer_positions, means - stds, means + stds,
                                alpha=ALPHA_FILL, color=color)

        ax.set_xlabel('Layer Depth (Network Order)')
        ax.set_ylabel('Activation Statistics')
        ax.set_title('Activation Mean ± Std Evolution')
        # DONT SHOE LEGEND HERE
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=ALPHA_GRID)

    def _plot_effective_rank_evolution(self, ax) -> None:
        """Plot effective rank evolution through network."""
        for model_name in sorted(self.results.information_flow.keys()):
            ordered_layers = self._get_ordered_layer_analysis(model_name)

            if not ordered_layers:
                continue

            ranks = []
            positions = []

            for i, (layer_name, analysis) in enumerate(ordered_layers):
                if 'effective_rank' in analysis and analysis['effective_rank'] > 0:
                    ranks.append(analysis['effective_rank'])
                    positions.append(i)

            if ranks:
                color = self.model_colors.get(model_name, '#333333')
                ax.plot(positions, ranks, 'o-', label=model_name,
                        linewidth=LINE_WIDTH_STANDARD, markersize=MARKER_SIZE_LARGE,
                        color=color)

        ax.set_xlabel('Layer Depth (Network Order)')
        ax.set_ylabel('Effective Rank')
        ax.set_title('Information Dimensionality Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=ALPHA_GRID)

    def _plot_activation_health_dashboard(self, ax) -> None:
        """Create an activation health dashboard showing model health metrics."""
        if not self.results.information_flow:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER, 'No activation data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Activation Health Dashboard')
            ax.axis('off')
            return

        # Prepare health metrics data with consistent ordering
        health_data = []

        for model_name in sorted(self.results.information_flow.keys()):
            ordered_layers = self._get_ordered_layer_analysis(model_name)

            for i, (layer_name, analysis) in enumerate(ordered_layers):
                # Calculate health metrics
                sparsity = analysis.get('sparsity', 0.0)
                positive_ratio = analysis.get('positive_ratio', 0.5)
                mean_activation = abs(analysis.get('mean_activation', 0.0))

                # Health indicators
                dead_neurons = sparsity  # High sparsity indicates dead neurons
                saturation = 1.0 - positive_ratio if positive_ratio > SATURATION_THRESHOLD else 0.0
                activation_magnitude = min(mean_activation,
                                           ACTIVATION_MAGNITUDE_NORMALIZER) / ACTIVATION_MAGNITUDE_NORMALIZER

                # Use layer index for consistent ordering instead of truncated layer name
                layer_label = f"L{i}"  # This ensures consistent ordering

                health_data.append({
                    'Model': model_name,
                    'Layer': layer_label,
                    'Layer_Index': i,  # Keep index for sorting
                    'Dead Neurons': dead_neurons,
                    'Saturation': saturation,
                    'Activation Level': activation_magnitude
                })

        if health_data:
            df = pd.DataFrame(health_data)

            # Sort by layer index to ensure correct order
            df = df.sort_values(['Model', 'Layer_Index'])

            # Create heatmap data
            models = sorted(df['Model'].unique())
            max_layers = getattr(self.config, 'max_layers_info_flow', 8)

            # Get unique layers and limit based on config, maintaining order
            unique_layers = sorted(df['Layer_Index'].unique())[:max_layers]
            layer_labels = [f"L{i}" for i in unique_layers]

            # Create separate heatmaps for each metric
            metrics = ['Dead Neurons', 'Saturation', 'Activation Level']

            # Create subplots within the main axis
            gs_sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(),
                                             wspace=SUB_WSPACE_WIDE, hspace=SUB_HSPACE_NARROW)

            for idx, metric in enumerate(metrics):
                ax_sub = plt.subplot(gs_sub[0, idx])

                # Filter data for layers within our limit
                df_filtered = df[df['Layer_Index'].isin(unique_layers)]

                # Create pivot table for heatmap using layer index for proper ordering
                heatmap_data = df_filtered.pivot_table(
                    values=metric,
                    index='Model',
                    columns='Layer_Index',
                    fill_value=0
                )

                # Ensure we have the right models in the right order
                heatmap_data = heatmap_data.reindex(models, fill_value=0)

                # Ensure columns are in the right order
                heatmap_data = heatmap_data.reindex(columns=unique_layers, fill_value=0)

                # Choose colormap based on metric
                if metric == 'Dead Neurons':
                    cmap = 'Reds'
                    vmax = HEATMAP_VMAX
                elif metric == 'Saturation':
                    cmap = 'Oranges'
                    vmax = HEATMAP_VMAX
                else:  # Activation Level
                    cmap = 'Greens'
                    vmax = HEATMAP_VMAX

                # Create heatmap
                im = ax_sub.imshow(heatmap_data.values, cmap=cmap, aspect='auto',
                                   vmin=HEATMAP_VMIN, vmax=vmax,
                                   interpolation=INTERPOLATION_METHOD)

                # Set labels
                ax_sub.set_title(metric, fontsize=SMALL_TITLE_FONT_SIZE, fontweight='bold')
                ax_sub.set_xticks(range(len(unique_layers)))
                ax_sub.set_xticklabels(layer_labels, rotation=45, ha='right',
                                      fontsize=LABEL_FONT_SIZE)
                ax_sub.set_yticks(range(len(heatmap_data.index)))

                # Only show model names on the leftmost heatmap
                if idx == 0:
                    ax_sub.set_yticklabels(heatmap_data.index, fontsize=LABEL_FONT_SIZE)
                else:
                    ax_sub.set_yticklabels([])

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax_sub, shrink=COLORBAR_SHRINK)
                cbar.ax.tick_params(labelsize=COLORBAR_FONT_SIZE)

            ax.axis('off')  # Hide the parent axis
            ax.set_title('Activation Health Dashboard (Network Order)',
                        fontsize=SUBTITLE_FONT_SIZE, fontweight='bold', pad=TITLE_PAD)
        else:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER,
                   'Insufficient data for health analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Activation Health Dashboard')
            ax.axis('off')

    def _plot_layer_specialization_analysis(self, ax) -> None:
        """Analyze how specialized each model's layers have become."""
        if not self.results.information_flow:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER, 'No activation data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Layer Specialization Analysis')
            ax.axis('off')
            return

        # Calculate specialization metrics for each model with consistent ordering
        specialization_data = []

        for model_name in sorted(self.results.information_flow.keys()):
            ordered_layers = self._get_ordered_layer_analysis(model_name)

            if not ordered_layers:
                continue

            total_specialization = 0
            layer_count = 0
            max_layers = getattr(self.config, 'max_layers_info_flow', 10)
            layer_specializations = []

            for i, (layer_name, analysis) in enumerate(ordered_layers[:max_layers]):
                # Specialization indicators:
                # 1. Low sparsity (neurons are active)
                # 2. Balanced positive ratio (not all saturated)
                # 3. Good effective rank (diverse representations)

                sparsity = analysis.get('sparsity', 1.0)
                positive_ratio = analysis.get('positive_ratio', 0.5)
                effective_rank = analysis.get('effective_rank', 1.0)

                # Calculate specialization score (0-1, higher is better)
                activation_health = 1.0 - sparsity
                balance_score = 1.0 - abs(positive_ratio - 0.5) * BALANCE_SCORE_MULTIPLIER
                rank_score = min(effective_rank / LAYER_SPECIALIZATION_MAX_RANK, 1.0) if effective_rank > 0 else 0.0

                # Combined specialization score
                layer_spec = (activation_health + balance_score + rank_score) / SPECIALIZATION_SCORE_COMPONENTS
                layer_specializations.append(layer_spec)

                total_specialization += layer_spec
                layer_count += 1

            if layer_count > 0:
                avg_specialization = total_specialization / layer_count
                specialization_data.append({
                    'Model': model_name,
                    'Average Specialization': avg_specialization,
                    'Layer Specializations': layer_specializations
                })

        if specialization_data:
            # Create two sub-visualizations
            gs_sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(),
                                             hspace=SUB_HSPACE_STANDARD,
                                             height_ratios=HEIGHT_RATIOS_BOTTOM_HEAVY)

            # Top: Overall specialization comparison
            ax_top = plt.subplot(gs_sub[0, 0])
            models = [d['Model'] for d in specialization_data]
            avg_specs = [d['Average Specialization'] for d in specialization_data]

            bars = ax_top.bar(range(len(models)), avg_specs, alpha=ALPHA_STANDARD)

            # Color bars with model colors
            for i, model in enumerate(models):
                color = self.model_colors.get(model, '#333333')
                bars[i].set_facecolor(color)

            # DONT SHOW THIS
            # ax_top.set_title('Overall Model Specialization', fontsize=SMALL_TITLE_FONT_SIZE, fontweight='bold')
            ax_top.set_ylabel('Specialization Score')
            ax_top.set_xticks(range(len(models)))
            ax_top.set_xticklabels([])  # Remove model names from x-axis
            ax_top.grid(True, alpha=ALPHA_GRID, axis='y')
            ax_top.set_ylim(AXIS_LIMIT_MIN, AXIS_LIMIT_MAX)

            # Add value labels on bars
            for i, v in enumerate(avg_specs):
                ax_top.text(i, v + BAR_LABEL_OFFSET, f'{v:.2f}', ha='center', va='bottom',
                           fontsize=LABEL_FONT_SIZE)

            # Bottom: Layer-by-layer specialization evolution
            ax_bottom = plt.subplot(gs_sub[1, 0])

            for data in specialization_data:
                model_name = data['Model']
                layer_specs = data['Layer Specializations']
                color = self.model_colors.get(model_name, '#333333')

                x_positions = range(len(layer_specs))
                ax_bottom.plot(x_positions, layer_specs, 'o-',
                               label=model_name, color=color,
                               linewidth=LINE_WIDTH_STANDARD, markersize=MARKER_SIZE_SMALL)

            ax_bottom.set_title('Layer-wise Specialization Evolution (Network Order)',
                               fontsize=SMALL_TITLE_FONT_SIZE, fontweight='bold')
            ax_bottom.set_xlabel('Layer Index (Network Depth)')
            ax_bottom.set_ylabel('Specialization Score')
            # DONT SHOW LEGEND
            # ax_bottom.legend(fontsize=LABEL_FONT_SIZE, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_bottom.grid(True, alpha=ALPHA_GRID)
            ax_bottom.set_ylim(AXIS_LIMIT_MIN, AXIS_LIMIT_MAX)

            ax.axis('off')  # Hide parent axis
            ax.set_title('Layer Specialization Analysis', fontsize=SUBTITLE_FONT_SIZE,
                        fontweight='bold', pad=TITLE_PAD)
        else:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER,
                   'Insufficient data for specialization analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Layer Specialization Analysis')
            ax.axis('off')

# ---------------------------------------------------------------------
