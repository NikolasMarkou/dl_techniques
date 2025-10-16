"""
Spectral Analysis Visualization Module (WeightWatcher Integration)

Creates visualizations for spectral analysis results, including power-law fits.
This module is responsible for generating both high-level summary dashboards
and detailed, per-layer diagnostic plots to interpret the spectral properties
of model weights.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
from typing import List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseVisualizer
from ..constants import (
    SPECTRAL_DEFAULT_BINS, SPECTRAL_DEFAULT_FIG_SIZE, SPECTRAL_DEFAULT_DPI,
    MetricNames, SPECTRAL_OVER_TRAINED_THRESH, SPECTRAL_UNDER_TRAINED_THRESH
)
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

class SpectralVisualizer(BaseVisualizer):
    """
    Creates visualizations for spectral analysis results.

    This class handles the generation of all plots related to the spectral
    analysis of model weights, including summary dashboards that compare
    multiple models and detailed diagnostic plots for individual layers.
    """

    def create_visualizations(self) -> None:
        """
        Main entry point for generating all spectral visualizations.

        This method orchestrates the creation of the summary dashboard and the
        individual, per-layer power-law fit plots. It acts as the primary
        interface called by the ModelAnalyzer.
        """
        # Abort if no spectral analysis data is available in the results object.
        if self.results.spectral_analysis is None or self.results.spectral_analysis.empty:
            logger.info("No spectral analysis data to visualize.")
            return

        # 1. Create a high-level summary dashboard for comparing models.
        self._create_summary_dashboard()

        # 2. Create detailed diagnostic plots for each analyzed layer, saved in a subdirectory.
        self._create_per_layer_plots()

    def _create_summary_dashboard(self) -> None:
        """
        Create an expanded 2x2 summary dashboard with per-layer evolution plots.

        This dashboard provides a comprehensive overview:
        - Top Row: Violin plots showing the distribution of metrics across all layers for each model.
        - Bottom Row: Scatter plots showing the evolution of metrics across the network's depth.
        """
        # Initialize a 2x2 matplotlib figure.
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Spectral Analysis Summary', fontsize=18, fontweight='bold')

        # --- Top-left: Alpha distributions per model using violin plots. ---
        self._plot_summary_distribution(axes[0, 0], MetricNames.ALPHA, 'Power-Law Exponent (α) Distribution')

        # --- Top-right: Concentration Score distributions per model using violin plots. ---
        self._plot_summary_distribution(axes[0, 1], MetricNames.CONCENTRATION_SCORE, 'Concentration Score Distribution')

        # --- Bottom-left: Alpha value for each layer, shown as a scatter plot. ---
        self._plot_spectral_evolution_across_layers(
            axes[1, 0],
            metric=MetricNames.ALPHA,
            title='Alpha (α) per Layer',
            y_label='Power-Law Exponent (α)',
            add_ref_lines=True  # Adds ideal range and boundaries.
        )

        # --- Bottom-right: Stable Rank for each layer, shown as a scatter plot. ---
        self._plot_spectral_evolution_across_layers(
            axes[1, 1],
            metric=MetricNames.STABLE_RANK,
            title='Stable Rank per Layer',
            y_label='Stable Rank (Effective Dimensionality)',
            log_scale=True  # Use a log scale for better visualization of rank.
        )

        # Add a single, shared legend for all models to the figure.
        models_with_data = self._get_models_with_data()
        if models_with_data:
            self._create_figure_legend(fig, title="Models", specific_models=models_with_data)

        # Adjust layout to prevent titles and labels from overlapping.
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        if self.config.save_plots:
            self._save_figure(fig, 'spectral_summary')
        plt.close(fig)

    def _get_alpha_color(self, alpha: float) -> str:
        """
        Get a diagnostic color based on the alpha value's interpretation.

        Args:
            alpha: The power-law exponent value.
        Returns:
            A string representing the color (red, green, or orange).
        """
        if alpha < SPECTRAL_OVER_TRAINED_THRESH:
            return 'darkred'
        elif alpha > SPECTRAL_UNDER_TRAINED_THRESH:
            return 'darkorange'
        else:
            return 'darkgreen'

    def _plot_summary_distribution(self, ax: plt.Axes, metric: str, title: str) -> None:
        """
        Plot per-layer metric distributions for each model using VIOLIN PLOTS.

        This now includes a y-axis starting at zero and diagnostic thresholds for the alpha plot.

        Args:
            ax: The matplotlib Axes object to plot on.
            metric: The name of the metric to plot from the spectral_analysis DataFrame.
            title: The title for the subplot.
        """
        df = self.results.spectral_analysis
        if metric not in df.columns or df[metric].isnull().all():
            ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
            return

        model_order = self._sort_models_consistently(df['model_name'].unique().tolist())

        data_to_plot = [df[df['model_name'] == name][metric].dropna().values for name in model_order]

        if not any(len(d) > 0 for d in data_to_plot):
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
            return

        parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=False, showextrema=False)

        for i, pc in enumerate(parts['bodies']):
            color = self._get_model_color(model_order[i])
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        parts['cmeans'].set_edgecolor('black')
        parts['cmeans'].set_linewidth(2)

        ax.set_title(title)
        ax.set_ylabel('Metric Value Distribution')
        ax.set_xlabel('Model')
        ax.set_xticks(np.arange(1, len(model_order) + 1))
        ax.set_xticklabels(model_order, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # --- NEW: Set y-axis to start at 0 ---
        ax.set_ylim(bottom=0)

        # --- NEW: Add diagnostic thresholds and ideal range ONLY for the Alpha plot ---
        if metric == MetricNames.ALPHA:
            ax.axhspan(SPECTRAL_OVER_TRAINED_THRESH, SPECTRAL_UNDER_TRAINED_THRESH, color='green', alpha=0.1, label='Ideal Range')
            ax.axhline(SPECTRAL_OVER_TRAINED_THRESH, color='red', linestyle=':', label='Over-trained boundary')
            ax.axhline(SPECTRAL_UNDER_TRAINED_THRESH, color='orange', linestyle=':', label='Under-trained boundary')
            # Add a local legend for these specific lines.
            ax.legend(fontsize='small', loc='best')

    def _plot_spectral_evolution_across_layers(self, ax: plt.Axes, metric: str, title: str, y_label: str, log_scale: bool = False, add_ref_lines: bool = False):
        """
        Plot per-layer metrics as a SCATTER PLOT with unique labels for duplicate names.

        Args:
            ax: The matplotlib Axes object to plot on.
            metric, title, y_label: Plotting parameters.
            log_scale: Whether to use a logarithmic y-axis.
            add_ref_lines: Whether to add reference lines for alpha.
        """
        df = self.results.spectral_analysis
        if metric not in df.columns or df[metric].isnull().all():
            ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
            return

        model_order = self._get_models_with_data()

        # Sort all layers globally by their discovered ID to maintain network order.
        all_layers_df = df[['layer_id', 'name']].drop_duplicates().sort_values('layer_id')
        original_layer_names = all_layers_df['name'].tolist()

        # [+] NEW: Generate unique labels for the x-axis to handle name collisions.
        unique_labels = []
        name_counts = {}
        for name in original_layer_names:
            if name in name_counts:
                name_counts[name] += 1
                unique_labels.append(f"{name}_{name_counts[name]}")
            else:
                name_counts[name] = 0
                unique_labels.append(name) # Keep the first one as is

        # Create a mapping from original name and layer_id to its unique x-position.
        layer_id_to_x_pos = {row['layer_id']: i for i, row in all_layers_df.reset_index().iterrows()}

        # Set up the x-axis with the new unique labels.
        ax.set_xticks(range(len(unique_labels)))
        ax.set_xticklabels(unique_labels, rotation=45, ha='right', fontsize='small')

        for model_name in model_order:
            model_df = df[df['model_name'] == model_name].sort_values('layer_id')
            if not model_df.empty:
                color = self._get_model_color(model_name)
                # Map each layer's ID to its correct x-axis position.
                x_pos = [layer_id_to_x_pos.get(layer_id) for layer_id in model_df['layer_id']]
                # Filter out any potential None values if a layer_id isn't in the map
                valid_x_pos = [p for p in x_pos if p is not None]
                valid_y_values = model_df[metric].values[:len(valid_x_pos)]

                ax.scatter(valid_x_pos, valid_y_values, color=color, label=model_name, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)

        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel('Layer Name (Network Depth)')
        ax.grid(True, linestyle='--', alpha=0.6)

        if log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))

        if add_ref_lines:
            ax.axhspan(SPECTRAL_OVER_TRAINED_THRESH, SPECTRAL_UNDER_TRAINED_THRESH, color='green', alpha=0.1, label='Ideal Range')
            ax.axhline(SPECTRAL_OVER_TRAINED_THRESH, color='red', linestyle=':', label='Over-trained boundary')
            ax.axhline(SPECTRAL_UNDER_TRAINED_THRESH, color='orange', linestyle=':', label='Under-trained boundary')
            ax.legend(fontsize='small', loc='best')

    def _get_models_with_data(self) -> List[str]:
        """Get a sorted list of models with valid spectral analysis data."""
        if self.results.spectral_analysis is not None and not self.results.spectral_analysis.empty:
            return sorted(self.results.spectral_analysis['model_name'].unique())
        return []

    def _create_per_layer_plots(self) -> None:
        """
        Iterate through results to generate a detailed power-law fit plot for each layer.
        """
        plot_dir = self.output_dir / "spectral_plots"
        if self.config.save_plots:
            plot_dir.mkdir(exist_ok=True)

        df = self.results.spectral_analysis
        esds = self.results.spectral_esds

        for index, row in df.iterrows():
            model_name, layer_id = row['model_name'], int(row['layer_id'])

            if not (model_name in esds and layer_id in esds[model_name]):
                continue

            evals = esds[model_name][layer_id]
            if evals is None or len(evals) == 0 or row.get(MetricNames.STATUS) != 'success':
                continue

            alpha, xmin, D, sigma = row.get(MetricNames.ALPHA), row.get(MetricNames.XMIN), row.get(MetricNames.D), row.get(MetricNames.SIGMA)
            layer_name = row.get('name', f"layer_{layer_id}")

            self._plot_powerlaw_fit(evals, alpha, xmin, D, sigma, model_name, layer_name, layer_id, plot_dir)

    def _plot_powerlaw_fit(self, evals, alpha, xmin, D, sigma, model_name, layer_name, layer_id, savedir):
        """
        Create and save an improved, interpretable power-law fit plot for a single layer.

        Args:
            evals, alpha, xmin, D, sigma: Data and parameters for the plot.
            model_name, layer_name, layer_id: Identifiers for the layer.
            savedir: The directory to save the plot in.
        """
        try:
            fig, ax = plt.subplots(figsize=SPECTRAL_DEFAULT_FIG_SIZE)

            hist, bin_edges = np.histogram(evals, bins=SPECTRAL_DEFAULT_BINS)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            valid_mask = (hist > 0) & (bin_centers > 0)
            if not np.any(valid_mask):
                plt.close(fig)
                return

            hist, bin_centers = hist[valid_mask], bin_centers[valid_mask]
            hist = hist / (np.sum(hist) * np.diff(bin_edges)[0])

            ax.loglog(bin_centers, hist, '.', markersize=8, label='Empirical Spectral Density (ESD)', alpha=0.7)

            if xmin > 0 and alpha > 0:
                x_range = np.logspace(np.log10(xmin), np.log10(np.max(evals)), 100)
                C = (alpha - 1) * (xmin ** (alpha - 1))
                y_fit = C * x_range ** (-alpha)

                ax.loglog(x_range, y_fit, 'r-', linewidth=2, label=f'Power-law fit: α={alpha:.3f}')
                ax.axvline(x=xmin, color='r', linestyle='--', label=f'xmin={xmin:.3e}')
                ax.axvspan(xmin, ax.get_xlim()[1], color='grey', alpha=0.1, label='Fitted Tail Region')

            title_color = self._get_alpha_color(alpha)
            title_text = f"Log-Log ESD for {model_name} - {layer_name}\nα={alpha:.3f} (D={D:.3f}, σ={sigma:.3f})"
            ax.set_title(title_text, color=title_color, fontweight='bold')

            ax.set_xlabel("Eigenvalue (λ)")
            ax.set_ylabel("Probability Density")
            ax.legend()
            ax.grid(True, which="both", ls="--", alpha=0.5)

            if self.config.save_plots:
                sane_layer_name = layer_name.replace('/', '_')
                filepath = savedir / f"{model_name}_layer_{layer_id}_{sane_layer_name}_powerlaw.png"
                fig.savefig(filepath, dpi=SPECTRAL_DEFAULT_DPI, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            logger.warning(f"Error creating power-law plot for layer {layer_id} of {model_name}: {e}")
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                plt.close(fig)

# ---------------------------------------------------------------------
