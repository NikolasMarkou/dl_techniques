"""
Spectral Analysis Visualization Module (WeightWatcher Integration)

Creates visualizations for spectral analysis results, including power-law fits.
This module is responsible for generating both high-level summary dashboards
and detailed, per-layer diagnostic plots to interpret the spectral properties
of model weights.
"""

import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

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
        individual, per-layer power-law fit plots.
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
        - Top Row: Violin plots showing the distribution of metrics across all layers.
        - Bottom Row: Scatter plots showing the evolution of metrics across depth.
        """
        # Initialize a 2x2 matplotlib figure.
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Spectral Analysis Summary', fontsize=18, fontweight='bold')

        # --- Top-left: Alpha distributions per model using violin plots ---
        self._plot_summary_distribution(
            axes[0, 0],
            MetricNames.ALPHA,
            'Power-Law Exponent (α) Distribution'
        )

        # --- Top-right: Concentration Score distributions per model ---
        self._plot_summary_distribution(
            axes[0, 1],
            MetricNames.CONCENTRATION_SCORE,
            'Concentration Score Distribution'
        )

        # --- Bottom-left: Alpha value for each layer (scatter) ---
        self._plot_spectral_evolution_across_layers(
            axes[1, 0],
            metric=MetricNames.ALPHA,
            title='Alpha (α) per Layer',
            y_label='Power-Law Exponent (α)',
            add_ref_lines=True  # Adds ideal range and boundaries.
        )

        # --- Bottom-right: Stable Rank for each layer (scatter) ---
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
            A hex color string (red, green, or orange).
        """
        if alpha < SPECTRAL_OVER_TRAINED_THRESH:
            return '#d62728'  # Red (Over-trained)
        elif alpha > SPECTRAL_UNDER_TRAINED_THRESH:
            return '#ff7f0e'  # Orange (Under-trained)
        else:
            return '#2ca02c'  # Green (Good)

    def _plot_summary_distribution(self, ax: plt.Axes, metric: str, title: str) -> None:
        """
        Plot per-layer metric distributions for each model using Violin Plots.

        Args:
            ax: The matplotlib Axes object to plot on.
            metric: The name of the metric to plot from the DataFrame.
            title: The title for the subplot.
        """
        df = self.results.spectral_analysis
        if metric not in df.columns or df[metric].isnull().all():
            self._plot_no_data(ax, title, metric)
            return

        model_order = self._sort_models_consistently(df['model_name'].unique().tolist())
        data_to_plot = [df[df['model_name'] == name][metric].dropna().values for name in model_order]

        if not any(len(d) > 0 for d in data_to_plot):
            self._plot_no_data(ax, title, metric)
            return

        parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=False, showextrema=False)

        for i, pc in enumerate(parts['bodies']):
            color = self._get_model_color(model_order[i])
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        if 'cmeans' in parts:
            parts['cmeans'].set_edgecolor('black')
            parts['cmeans'].set_linewidth(2)

        ax.set_title(title)
        ax.set_ylabel('Metric Value')
        ax.set_xticks(np.arange(1, len(model_order) + 1))
        ax.set_xticklabels(model_order, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(bottom=0)

        # Add diagnostic thresholds ONLY for the Alpha plot
        if metric == MetricNames.ALPHA:
            ax.axhspan(
                SPECTRAL_OVER_TRAINED_THRESH, SPECTRAL_UNDER_TRAINED_THRESH,
                color='green', alpha=0.1, label='Ideal Range'
            )
            ax.axhline(SPECTRAL_OVER_TRAINED_THRESH, color='red', linestyle=':', label='Over-trained')
            ax.axhline(SPECTRAL_UNDER_TRAINED_THRESH, color='orange', linestyle=':', label='Under-trained')

    def _plot_spectral_evolution_across_layers(
        self, ax: plt.Axes, metric: str, title: str, y_label: str,
        log_scale: bool = False, add_ref_lines: bool = False
    ) -> None:
        """
        Plot per-layer metrics as a Scatter Plot with unique labels.

        Args:
            ax: The matplotlib Axes object.
            metric: Metric column name.
            title: Plot title.
            y_label: Y-axis label.
            log_scale: Whether to use log scale on Y.
            add_ref_lines: Whether to add alpha diagnostic lines.
        """
        df = self.results.spectral_analysis
        if metric not in df.columns or df[metric].isnull().all():
            self._plot_no_data(ax, title, metric)
            return

        model_order = self._get_models_with_data()

        # Sort layers globally by ID to maintain network order
        all_layers_df = df[['layer_id', 'name']].drop_duplicates().sort_values('layer_id')
        original_layer_names = all_layers_df['name'].tolist()

        # Generate unique labels to handle duplicate layer names
        unique_labels = []
        name_counts = {}
        for name in original_layer_names:
            name_counts[name] = name_counts.get(name, 0) + 1
            label = f"{name}_{name_counts[name]}" if name_counts[name] > 1 else name
            unique_labels.append(label)

        # Map ID to x-position
        layer_id_to_x = {row['layer_id']: i for i, row in all_layers_df.reset_index().iterrows()}

        ax.set_xticks(range(len(unique_labels)))
        # Only show every Nth label if there are too many layers
        if len(unique_labels) > 20:
            step = len(unique_labels) // 20
            ax.set_xticks(range(0, len(unique_labels), step))
            ax.set_xticklabels([unique_labels[i] for i in range(0, len(unique_labels), step)],
                               rotation=45, ha='right', fontsize='small')
        else:
            ax.set_xticklabels(unique_labels, rotation=45, ha='right', fontsize='small')

        for model_name in model_order:
            model_df = df[df['model_name'] == model_name].sort_values('layer_id')
            if not model_df.empty:
                color = self._get_model_color(model_name)

                x_pos = [layer_id_to_x.get(lid) for lid in model_df['layer_id']]
                y_vals = model_df[metric].values

                # Filter out Nones just in case
                valid_indices = [i for i, x in enumerate(x_pos) if x is not None]

                if valid_indices:
                    x_final = [x_pos[i] for i in valid_indices]
                    y_final = y_vals[valid_indices]

                    ax.scatter(
                        x_final, y_final,
                        color=color, label=model_name, s=50,
                        alpha=0.8, edgecolors='black', linewidth=0.5
                    )

        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel('Layer (Network Depth)')
        ax.grid(True, linestyle='--', alpha=0.6)

        if log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))

        if add_ref_lines:
            ax.axhspan(SPECTRAL_OVER_TRAINED_THRESH, SPECTRAL_UNDER_TRAINED_THRESH, color='green', alpha=0.1)
            ax.axhline(SPECTRAL_OVER_TRAINED_THRESH, color='red', linestyle=':')
            ax.axhline(SPECTRAL_UNDER_TRAINED_THRESH, color='orange', linestyle=':')

    def _get_models_with_data(self) -> List[str]:
        """Get a sorted list of models with valid spectral analysis data."""
        if self.results.spectral_analysis is not None and not self.results.spectral_analysis.empty:
            return sorted(self.results.spectral_analysis['model_name'].unique())
        return []

    def _plot_no_data(self, ax: plt.Axes, title: str, metric: str) -> None:
        """Helper to display a 'No Data' message on a subplot."""
        ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')

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

            alpha = row.get(MetricNames.ALPHA)
            xmin = row.get(MetricNames.XMIN)
            D = row.get(MetricNames.D)
            sigma = row.get(MetricNames.SIGMA)
            layer_name = row.get('name', f"layer_{layer_id}")

            self._plot_powerlaw_fit(
                evals, alpha, xmin, D, sigma,
                model_name, layer_name, layer_id, plot_dir
            )

    def _plot_powerlaw_fit(
            self, evals: np.ndarray, alpha: float, xmin: float, D: float, sigma: float,
            model_name: str, layer_name: str, layer_id: int, savedir: str
    ) -> None:
        """
        Create and save a power-law fit plot using Logarithmic Binning.

        Updated to anchor the fit line visually to the data and label xmin.

        Args:
            evals: Array of eigenvalues.
            alpha, xmin, D, sigma: Fit parameters.
            model_name, layer_name, layer_id: Identifiers.
            savedir: Output directory.
        """
        try:
            fig, ax = plt.subplots(figsize=SPECTRAL_DEFAULT_FIG_SIZE)

            # Filter valid eigenvalues
            evals_clean = evals[evals > 1e-10]
            if len(evals_clean) == 0:
                plt.close(fig)
                return

            min_val, max_val = np.min(evals_clean), np.max(evals_clean)

            # --- 1. Empirical Data (Logarithmic Binning) ---
            # Create bins equally spaced in log-space
            bins = np.logspace(np.log10(min_val), np.log10(max_val), num=SPECTRAL_DEFAULT_BINS)

            # density=True normalizes by bin width so area sums to 1
            hist, bin_edges = np.histogram(evals_clean, bins=bins, density=True)
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean

            # Filter zero-count bins
            valid_mask = hist > 0

            # Plot the blue dots
            ax.loglog(
                bin_centers[valid_mask], hist[valid_mask],
                '.', markersize=10, label='Empirical Spectral Density',
                alpha=0.6, color='#1f77b4', markeredgecolor='none'
            )

            # --- 2. Theoretical Fit Line (Anchored) ---
            if xmin > 0 and alpha > 1:
                # Range for the red line
                x_range = np.logspace(np.log10(xmin), np.log10(max_val), 100)

                # Calculate theoretical PDF shape: x^(-alpha)
                y_shape = x_range ** (-alpha)

                # --- VISUAL CORRECTION: Anchoring ---
                # Instead of purely theoretical scaling, we anchor the line to the
                # histogram bin closest to xmin. This ensures the line touches the dots.

                # Find the bin center closest to xmin that has data
                valid_centers = bin_centers[valid_mask]
                valid_hist = hist[valid_mask]

                # Get indices of bins >= xmin
                tail_indices = np.where(valid_centers >= xmin)[0]

                if len(tail_indices) > 0:
                    # Use the first valid bin in the tail as the anchor
                    anchor_idx = tail_indices[0]
                    anchor_x = valid_centers[anchor_idx]
                    anchor_y = valid_hist[anchor_idx]

                    # Calculate scaling factor K so that y = K * x^(-alpha) passes through anchor
                    # K = y_anchor / (x_anchor^(-alpha)) = y_anchor * x_anchor^(alpha)
                    scaling_factor = anchor_y * (anchor_x ** alpha)
                    y_fit = scaling_factor * y_shape
                else:
                    # Fallback to theoretical scaling if alignment fails
                    N_tail = np.sum(evals_clean >= xmin)
                    tail_ratio = N_tail / len(evals_clean)
                    C = (alpha - 1) * (xmin ** (alpha - 1))
                    y_fit = tail_ratio * C * y_shape

                ax.loglog(x_range, y_fit, 'r-', linewidth=3, label=f'Fit: α={alpha:.3f}')

                # --- 3. Vertical Line & Label ---
                ax.axvline(x=xmin, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)

                # Add X-axis label for xmin
                # We place it at the bottom of the plot, just above the axis
                ymin_plot, ymax_plot = ax.get_ylim()

                # Add background box for readability
                ax.text(
                    xmin, ymin_plot, f' $x_{{min}}={xmin:.2f}$',
                    color='#444444', fontsize=10, fontweight='bold',
                    ha='left', va='bottom', rotation=90,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )

            # Styling
            title_color = self._get_alpha_color(alpha)
            ax.set_title(
                f"Log-Log ESD for {model_name} - {layer_name}\nα={alpha:.3f} (D={D:.3f}, σ={sigma:.3f})",
                color=title_color, fontweight='bold', pad=15
            )
            ax.set_xlabel("Eigenvalue (λ)", fontsize=11)
            ax.set_ylabel("Probability Density P(λ)", fontsize=11)
            ax.legend(loc='best', fontsize='small', framealpha=0.9)

            # Improve grid
            ax.grid(True, which="major", linestyle='-', alpha=0.3, color='gray')
            ax.grid(True, which="minor", linestyle=':', alpha=0.1, color='gray')

            if self.config.save_plots:
                sane_layer_name = layer_name.replace('/', '_').replace(':', '')
                filepath = savedir / f"{model_name}_layer_{layer_id}_{sane_layer_name}_powerlaw.png"
                fig.savefig(filepath, dpi=SPECTRAL_DEFAULT_DPI, bbox_inches='tight')

            plt.close(fig)

        except Exception as e:
            logger.warning(f"Error creating power-law plot for layer {layer_id} of {model_name}: {e}")
            if 'fig' in locals():
                plt.close(fig)

# ---------------------------------------------------------------------
