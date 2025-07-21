"""
Information Flow Visualization Module
============================================================================

Creates visualizations for information flow analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from .base import BaseVisualizer
from ..constants import ACTIVATION_MAGNITUDE_NORMALIZER, LAYER_SPECIALIZATION_MAX_RANK


class InformationFlowVisualizer(BaseVisualizer):
    """Creates information flow visualizations."""

    def create_visualizations(self) -> None:
        """Create information flow visualizations."""
        if not self.results.information_flow:
            return

        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

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

        plt.suptitle('Information Flow and Activation Analysis', fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'information_flow_analysis')
        plt.close(fig)

    def _plot_activation_flow_overview(self, ax) -> None:
        """Plot activation statistics evolution through layers."""
        for model_name in sorted(self.results.information_flow.keys()):
            layer_analysis = self.results.information_flow[model_name]
            means = []
            stds = []
            layer_positions = []

            for i, (layer_name, analysis) in enumerate(layer_analysis.items()):
                means.append(analysis['mean_activation'])
                stds.append(analysis['std_activation'])
                layer_positions.append(i)

            # Plot mean with std as shaded region
            means = np.array(means)
            stds = np.array(stds)

            color = self.model_colors.get(model_name, '#333333')
            line = ax.plot(layer_positions, means, 'o-', label=f'{model_name}',
                          linewidth=2, markersize=6, color=color)
            ax.fill_between(layer_positions, means - stds, means + stds,
                           alpha=0.2, color=color)

        ax.set_xlabel('Layer Depth')
        ax.set_ylabel('Activation Statistics')
        ax.set_title('Activation Mean ± Std Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_effective_rank_evolution(self, ax) -> None:
        """Plot effective rank evolution through network."""
        for model_name in sorted(self.results.information_flow.keys()):
            layer_analysis = self.results.information_flow[model_name]
            ranks = []
            positions = []

            for i, (layer_name, analysis) in enumerate(layer_analysis.items()):
                if 'effective_rank' in analysis and analysis['effective_rank'] > 0:
                    ranks.append(analysis['effective_rank'])
                    positions.append(i)

            if ranks:
                color = self.model_colors.get(model_name, '#333333')
                ax.plot(positions, ranks, 'o-', label=model_name,
                       linewidth=2, markersize=8, color=color)

        ax.set_xlabel('Layer Depth')
        ax.set_ylabel('Effective Rank')
        ax.set_title('Information Dimensionality Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_activation_health_dashboard(self, ax) -> None:
        """Create an activation health dashboard showing model health metrics."""
        if not self.results.information_flow:
            ax.text(0.5, 0.5, 'No activation data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Activation Health Dashboard')
            ax.axis('off')
            return

        # Prepare health metrics data
        health_data = []

        for model_name in sorted(self.results.information_flow.keys()):
            layer_analysis = self.results.information_flow[model_name]

            for layer_name, analysis in layer_analysis.items():
                # Calculate health metrics
                sparsity = analysis.get('sparsity', 0.0)
                positive_ratio = analysis.get('positive_ratio', 0.5)
                mean_activation = abs(analysis.get('mean_activation', 0.0))

                # Health indicators
                dead_neurons = sparsity  # High sparsity indicates dead neurons
                saturation = 1.0 - positive_ratio if positive_ratio > 0.9 else 0.0  # Very high positive ratio suggests saturation
                activation_magnitude = min(mean_activation, ACTIVATION_MAGNITUDE_NORMALIZER) / ACTIVATION_MAGNITUDE_NORMALIZER

                health_data.append({
                    'Model': model_name,
                    'Layer': layer_name.split('_')[0] if '_' in layer_name else layer_name[:8],
                    'Dead Neurons': dead_neurons,
                    'Saturation': saturation,
                    'Activation Level': activation_magnitude
                })

        if health_data:
            df = pd.DataFrame(health_data)

            # Create heatmap data
            models = sorted(df['Model'].unique())
            layers = df['Layer'].unique()[:self.config.max_layers_info_flow]  # Use configurable limit

            # Create separate heatmaps for each metric
            metrics = ['Dead Neurons', 'Saturation', 'Activation Level']

            # Create subplots within the main axis
            gs_sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(),
                                           wspace=0.4, hspace=0.1)

            for idx, metric in enumerate(metrics):
                ax_sub = plt.subplot(gs_sub[0, idx])

                # Create pivot table for heatmap
                df_filtered = df[df['Layer'].isin(layers)]
                heatmap_data = df_filtered.pivot_table(values=metric, index='Model', columns='Layer', fill_value=0)

                # Ensure we have the right models in the right order
                heatmap_data = heatmap_data.reindex(models, fill_value=0)

                # Choose colormap based on metric
                if metric == 'Dead Neurons':
                    cmap = 'Reds'  # Red = bad (more dead neurons)
                    vmax = 1.0
                elif metric == 'Saturation':
                    cmap = 'Oranges'  # Orange = bad (more saturation)
                    vmax = 1.0
                else:  # Activation Level
                    cmap = 'Greens'  # Green = good (healthy activation)
                    vmax = 1.0

                # Create heatmap
                im = ax_sub.imshow(heatmap_data.values, cmap=cmap, aspect='auto',
                                  vmin=0, vmax=vmax, interpolation='nearest')

                # Set labels
                ax_sub.set_title(metric, fontsize=10, fontweight='bold')
                ax_sub.set_xticks(range(len(heatmap_data.columns)))
                ax_sub.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=8)
                ax_sub.set_yticks(range(len(heatmap_data.index)))

                # Only show model names on the leftmost heatmap
                if idx == 0:
                    ax_sub.set_yticklabels(heatmap_data.index, fontsize=8)
                else:
                    ax_sub.set_yticklabels([])

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax_sub, shrink=0.6)
                cbar.ax.tick_params(labelsize=7)

            ax.axis('off')  # Hide the parent axis
            ax.set_title('Activation Health Dashboard', fontsize=12, fontweight='bold', pad=20)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for health analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Activation Health Dashboard')
            ax.axis('off')

    def _plot_layer_specialization_analysis(self, ax) -> None:
        """Analyze how specialized each model's layers have become."""
        if not self.results.information_flow:
            ax.text(0.5, 0.5, 'No activation data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Layer Specialization Analysis')
            ax.axis('off')
            return

        # Calculate specialization metrics for each model
        specialization_data = []

        for model_name in sorted(self.results.information_flow.keys()):
            layer_analysis = self.results.information_flow[model_name]

            total_specialization = 0
            layer_count = 0
            layer_specializations = []

            for layer_name, analysis in layer_analysis.items():
                # Specialization indicators:
                # 1. Low sparsity (neurons are active)
                # 2. Balanced positive ratio (not all saturated)
                # 3. Good effective rank (diverse representations)

                sparsity = analysis.get('sparsity', 1.0)
                positive_ratio = analysis.get('positive_ratio', 0.5)
                effective_rank = analysis.get('effective_rank', 1.0)

                # Calculate specialization score (0-1, higher is better)
                # Low sparsity is good (1 - sparsity)
                activation_health = 1.0 - sparsity

                # Balanced activation is good (closer to 0.5 is better)
                balance_score = 1.0 - abs(positive_ratio - 0.5) * 2

                # Higher effective rank is good (normalize by a reasonable max)
                rank_score = min(effective_rank / LAYER_SPECIALIZATION_MAX_RANK, 1.0) if effective_rank > 0 else 0.0

                # Combined specialization score
                layer_spec = (activation_health + balance_score + rank_score) / 3.0
                layer_specializations.append(layer_spec)

                total_specialization += layer_spec
                layer_count += 1

            if layer_count > 0:
                avg_specialization = total_specialization / layer_count
                specialization_data.append({
                    'Model': model_name,
                    'Average Specialization': avg_specialization,
                    'Layer Specializations': layer_specializations[:self.config.max_layers_specialization]  # Use configurable limit
                })

        if specialization_data:
            # Create two sub-visualizations
            gs_sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(),
                                           hspace=0.4, height_ratios=[1, 1.5])

            # Top: Overall specialization comparison
            ax_top = plt.subplot(gs_sub[0, 0])
            models = [d['Model'] for d in specialization_data]
            avg_specs = [d['Average Specialization'] for d in specialization_data]

            bars = ax_top.bar(range(len(models)), avg_specs, alpha=0.8)

            # Color bars with model colors
            for i, model in enumerate(models):
                color = self.model_colors.get(model, '#333333')
                bars[i].set_facecolor(color)

            ax_top.set_title('Overall Model Specialization', fontsize=10, fontweight='bold')
            ax_top.set_ylabel('Specialization Score')
            ax_top.set_xticks(range(len(models)))
            ax_top.set_xticklabels([])  # Remove model names from x-axis
            ax_top.grid(True, alpha=0.3, axis='y')
            ax_top.set_ylim(0, 1)

            # Add value labels on bars
            for i, v in enumerate(avg_specs):
                ax_top.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

            # Bottom: Layer-by-layer specialization evolution
            ax_bottom = plt.subplot(gs_sub[1, 0])

            for data in specialization_data:
                model_name = data['Model']
                layer_specs = data['Layer Specializations']
                color = self.model_colors.get(model_name, '#333333')

                x_positions = range(len(layer_specs))
                ax_bottom.plot(x_positions, layer_specs, 'o-',
                             label=model_name, color=color, linewidth=2, markersize=6)

            ax_bottom.set_title('Layer-wise Specialization Evolution', fontsize=10, fontweight='bold')
            ax_bottom.set_xlabel('Layer Index')
            ax_bottom.set_ylabel('Specialization Score')
            ax_bottom.legend(fontsize=8)
            ax_bottom.grid(True, alpha=0.3)
            ax_bottom.set_ylim(0, 1)

            ax.axis('off')  # Hide parent axis
            ax.set_title('Layer Specialization Analysis', fontsize=12, fontweight='bold', pad=20)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for specialization analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Layer Specialization Analysis')
            ax.axis('off')