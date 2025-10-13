"""
Spectral Analysis Visualization Module (WeightWatcher Integration)

Creates visualizations for spectral analysis results, including power-law fits.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseVisualizer
from ..constants import (
    SPECTRAL_DEFAULT_BINS, SPECTRAL_DEFAULT_FIG_SIZE, SPECTRAL_DEFAULT_DPI,
    MetricNames
)
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

class SpectralVisualizer(BaseVisualizer):
    """Creates visualizations for spectral analysis results."""

    def create_visualizations(self) -> None:
        """
        Create summary plots and detailed per-layer power-law fit plots.
        """
        if self.results.spectral_analysis is None or self.results.spectral_analysis.empty:
            logger.info("No spectral analysis data to visualize.")
            return

        # 1. Create a summary dashboard
        self._create_summary_dashboard()

        # 2. Create detailed per-layer plots in a subdirectory
        self._create_per_layer_plots()

    def _create_summary_dashboard(self) -> None:
        """Create a summary dashboard comparing models on key spectral metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Spectral Analysis Summary', fontsize=16, fontweight='bold')

        # Plot 1: Mean Alpha per model
        self._plot_summary_metric(axes[0], MetricNames.ALPHA, 'Mean Power-Law Exponent (α)')

        # Plot 2: Mean Concentration Score per model
        self._plot_summary_metric(axes[1], MetricNames.CONCENTRATION_SCORE, 'Mean Concentration Score')

        # Add shared legend
        models_with_data = self._get_models_with_data()
        if models_with_data:
            self._create_figure_legend(fig, title="Models", specific_models=models_with_data)

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        if self.config.save_plots:
            self._save_figure(fig, 'spectral_summary')
        plt.close(fig)

    def _plot_summary_metric(self, ax: plt.Axes, metric: str, title: str) -> None:
        """Helper to plot a summary metric as a bar chart."""
        df = self.results.spectral_analysis
        if metric not in df.columns:
            ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
            return

        # Group by model and calculate mean
        summary_data = df.groupby('model_name')[metric].mean()
        model_order = self._sort_models_consistently(summary_data.index.tolist())

        colors = [self._get_model_color(name) for name in model_order]
        summary_data.loc[model_order].plot(kind='bar', ax=ax, color=colors)

        ax.set_title(title)
        ax.set_ylabel('Mean Value')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    def _get_models_with_data(self) -> List[str]:
        """Get models that have spectral analysis data."""
        if self.results.spectral_analysis is not None and not self.results.spectral_analysis.empty:
            return sorted(self.results.spectral_analysis['model_name'].unique())
        return []

    def _create_per_layer_plots(self) -> None:
        """Create and save power-law fit plots for each analyzed layer."""
        plot_dir = self.output_dir / "spectral_plots"
        if self.config.save_plots:
            plot_dir.mkdir(exist_ok=True)

        df = self.results.spectral_analysis
        esds = self.results.spectral_esds

        for index, row in df.iterrows():
            model_name = row['model_name']
            layer_id = int(row['layer_id'])

            if model_name not in esds or layer_id not in esds[model_name]:
                continue

            evals = esds[model_name][layer_id]
            if evals is None or len(evals) == 0:
                continue

            status = row.get(MetricNames.STATUS)
            if status != 'success':
                continue

            alpha = row.get(MetricNames.ALPHA)
            xmin = row.get(MetricNames.XMIN)
            D = row.get(MetricNames.D)
            sigma = row.get(MetricNames.SIGMA)
            layer_name = row.get('name', f"layer_{layer_id}")

            self._plot_powerlaw_fit(evals, alpha, xmin, D, sigma, model_name, layer_name, layer_id, plot_dir)

    def _plot_powerlaw_fit(self, evals, alpha, xmin, D, sigma, model_name, layer_name, layer_id, savedir):
        """Helper function to create and save a single power-law fit plot."""
        try:
            fig = plt.figure(figsize=SPECTRAL_DEFAULT_FIG_SIZE)
            ax = fig.add_subplot(111)

            hist, bin_edges = np.histogram(evals, bins=SPECTRAL_DEFAULT_BINS)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            valid_mask = (hist > 0) & (bin_centers > 0)
            if not np.any(valid_mask):
                plt.close(fig)
                return

            hist, bin_centers = hist[valid_mask], bin_centers[valid_mask]
            hist = hist / (np.sum(hist) * (bin_edges[1] - bin_edges[0]))

            ax.loglog(bin_centers, hist, 'o', markersize=4, label='Eigenvalue Distribution')

            if xmin > 0 and alpha > 0:
                x_range = np.logspace(np.log10(xmin), np.log10(np.max(evals)), 100)
                y_fit = (alpha - 1) * (xmin ** (alpha - 1)) * x_range ** (-alpha)
                ax.loglog(x_range, y_fit, 'r-', label=f'Power-law fit: α={alpha:.3f}')
                ax.axvline(x=xmin, color='r', linestyle='--', label=f'xmin={xmin:.3e}')

            ax.set_title(f"Log-Log ESD for {model_name} - {layer_name}\nα={alpha:.3f}, D={D:.3f}, σ={sigma:.3f}")
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