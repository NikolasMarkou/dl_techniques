"""
Spectral Analysis Visualization Module (WeightWatcher Integration)

Creates visualizations for spectral analysis results, including power-law fits.
This module is responsible for generating both high-level summary dashboards
and detailed, per-layer diagnostic plots to interpret the spectral properties
of model weights.
"""

import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseVisualizer
from ..constants import (
    SPECTRAL_DEFAULT_BINS, SPECTRAL_DEFAULT_FIG_SIZE, SPECTRAL_DEFAULT_DPI,
    MetricNames, SPECTRAL_OVER_TRAINED_THRESH, SPECTRAL_UNDER_TRAINED_THRESH,
    SPECTRAL_EPSILON
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
        if self.config.spectral_per_layer_diagnostics:
            self._create_per_layer_plots()

        # 3. Create correlation trap overlay plots (original vs randomized ESD).
        if self.config.spectral_randomize:
            self._create_trap_overlay_plots()

    def _create_summary_dashboard(self) -> None:
        """
        Create an expanded summary dashboard focusing on the Phases of Learning.

        Layout:
        - Top Row: Alpha Histograms with Phase backgrounds (Pink/Overfit, Yellow/Underfit).
        - Bottom Row: Evolution of metrics (Alpha, Stable Rank) across layers.
        """
        # Initialize a 2x2 matplotlib figure.
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Spectral Analysis Summary: Phases of Learning', fontsize=18, fontweight='bold')

        # --- Top-left: Alpha distributions (Histogram with Phases) ---
        self._plot_phase_histogram(
            axes[0, 0],
            MetricNames.ALPHA,
            'Power-Law Exponent (α) Distribution'
        )

        # --- Top-right: Concentration Score distributions ---
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
            y_label='Stable Rank',
            log_scale=True
        )

        # Add a single, shared legend for all models to the figure.
        models_with_data = self._get_models_with_data()
        if models_with_data:
            self._create_figure_legend(fig, title="Models", specific_models=models_with_data)

        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        if self.config.save_plots:
            self._save_figure(fig, 'spectral_summary')
        plt.close(fig)

    def _plot_phase_histogram(self, ax: plt.Axes, metric: str, title: str) -> None:
        """
        Plot histograms of metrics with colored background regions (Phases of Learning).

        Matches the style of Figure 2 (Falcon vs Llama).
        """
        df = self.results.spectral_analysis
        if metric not in df.columns or df[metric].isnull().all():
            self._plot_no_data(ax, title, metric)
            return

        model_order = self._sort_models_consistently(df['model_name'].unique().tolist())

        # Determine bin range
        valid_vals = df[metric].dropna().values
        if len(valid_vals) == 0:
            self._plot_no_data(ax, title, metric)
            return

        min_val, max_val = np.min(valid_vals), np.max(valid_vals)
        # Ensure we cover the critical regions 2.0 and 6.0
        plot_min = min(min_val, 1.0)
        plot_max = max(max_val, 8.0)
        bins = np.linspace(plot_min, plot_max, 50)

        # Plot histograms for each model
        for model_name in model_order:
            model_data = df[df['model_name'] == model_name][metric].dropna().values
            if len(model_data) > 0:
                color = self._get_model_color(model_name)
                mean_val = np.mean(model_data)
                ax.hist(
                    model_data, bins=bins, alpha=0.6, color=color,
                    label=f"{model_name}: <α>={mean_val:.2f}",
                    density=True, histtype='stepfilled', edgecolor='black'
                )

        # --- Background Phases (The "SETOL" look) ---
        ylim = ax.get_ylim()

        # Overfit / Glassy Phase (Pink)
        ax.axvspan(plot_min, SPECTRAL_OVER_TRAINED_THRESH, color='#ffebee', alpha=0.5, zorder=0)
        ax.text(SPECTRAL_OVER_TRAINED_THRESH - 0.2, ylim[1]*0.8, 'overfit',
                color='#c62828', fontweight='bold', ha='right')

        # Underfit / Random Phase (Yellow)
        ax.axvspan(SPECTRAL_UNDER_TRAINED_THRESH, plot_max, color='#fffde7', alpha=0.5, zorder=0)
        ax.text(SPECTRAL_UNDER_TRAINED_THRESH + 0.2, ylim[1]*0.8, 'underfit',
                color='#f57f17', fontweight='bold', ha='left')

        # Vertical Boundary Lines
        ax.axvline(SPECTRAL_OVER_TRAINED_THRESH, color='#e91e63', linestyle=':', linewidth=2)
        ax.axvline(SPECTRAL_UNDER_TRAINED_THRESH, color='#fbc02d', linestyle=':', linewidth=2)

        ax.set_title(title)
        ax.set_ylabel('Density')
        ax.set_xlabel('Alpha (α)')
        ax.set_xlim(plot_min, plot_max)
        ax.grid(False) # Turn off grid to emphasize background regions
        ax.legend(loc='upper right', fontsize='small')

    def _plot_summary_distribution(self, ax: plt.Axes, metric: str, title: str) -> None:
        """Plot per-layer metric distributions using standard histograms/KDE."""
        df = self.results.spectral_analysis
        if metric not in df.columns or df[metric].isnull().all():
            self._plot_no_data(ax, title, metric)
            return

        model_order = self._sort_models_consistently(df['model_name'].unique().tolist())

        for model_name in model_order:
            data = df[df['model_name'] == model_name][metric].dropna().values
            if len(data) > 0:
                color = self._get_model_color(model_name)
                ax.hist(data, bins=30, alpha=0.5, color=color, label=model_name, density=True)

        ax.set_title(title)
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize='small')

    def _plot_spectral_evolution_across_layers(
        self, ax: plt.Axes, metric: str, title: str, y_label: str,
        log_scale: bool = False, add_ref_lines: bool = False
    ) -> None:
        """Scatter plot of metrics vs network depth."""
        df = self.results.spectral_analysis
        if metric not in df.columns or df[metric].isnull().all():
            self._plot_no_data(ax, title, metric)
            return

        model_order = self._get_models_with_data()

        # Sort layers globally by ID
        all_layers_df = df[['layer_id', 'name']].drop_duplicates().sort_values('layer_id')

        # Generate X-ticks (Truncated)
        original_names = all_layers_df['name'].tolist()
        unique_labels = []
        name_counts = {}
        for name in original_names:
            name_counts[name] = name_counts.get(name, 0) + 1
            label = f"{name}_{name_counts[name]}" if name_counts[name] > 1 else name
            unique_labels.append(label)

        # Map ID to x-position
        layer_id_to_x = {row['layer_id']: i for i, row in all_layers_df.reset_index().iterrows()}

        # Ticks configuration
        ax.set_xticks(range(len(unique_labels)))
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

                # Remove Nones
                valid_mask = [x is not None for x in x_pos]
                ax.scatter(
                    np.array(x_pos)[valid_mask], y_vals[valid_mask],
                    color=color, label=model_name, s=50,
                    alpha=0.8, edgecolors='black', linewidth=0.5
                )
                # Connect dots
                ax.plot(np.array(x_pos)[valid_mask], y_vals[valid_mask], color=color, alpha=0.3)

        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel('Layer (Network Depth)')
        ax.grid(True, linestyle='--', alpha=0.6)

        if log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))

        if add_ref_lines:
            # Green Zone (Good)
            ax.axhspan(SPECTRAL_OVER_TRAINED_THRESH, SPECTRAL_UNDER_TRAINED_THRESH,
                      color='green', alpha=0.05)
            # Lines
            ax.axhline(SPECTRAL_OVER_TRAINED_THRESH, color='red', linestyle=':', alpha=0.5)
            ax.axhline(SPECTRAL_UNDER_TRAINED_THRESH, color='orange', linestyle=':', alpha=0.5)

    def _get_models_with_data(self) -> List[str]:
        """Get sorted list of models with data."""
        if self.results.spectral_analysis is not None and not self.results.spectral_analysis.empty:
            return sorted(self.results.spectral_analysis['model_name'].unique())
        return []

    def _plot_no_data(self, ax: plt.Axes, title: str, metric: str) -> None:
        """Helper to display 'No Data' message."""
        ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')

    def _create_per_layer_plots(self) -> None:
        """
        Iterate through results to generate detailed 2x2 diagnostic plots for each layer.
        Matches the style of Figure 1.
        """
        plot_dir = self.output_dir / "spectral_plots"
        if self.config.save_plots:
            plot_dir.mkdir(exist_ok=True)

        df = self.results.spectral_analysis
        esds = self.results.spectral_esds

        for index, row in df.iterrows():
            model_name = row['model_name']
            layer_id = int(row['layer_id'])

            if not (model_name in esds and layer_id in esds[model_name]):
                continue

            evals = esds[model_name][layer_id]
            if evals is None or len(evals) == 0 or row.get(MetricNames.STATUS) != 'success':
                continue

            # Extract metrics
            metrics = {
                'alpha': row.get(MetricNames.ALPHA),
                'xmin': row.get(MetricNames.XMIN),
                'xmax': row.get(MetricNames.LAMBDA_MAX),
                'D': row.get(MetricNames.D),
                'sigma': row.get(MetricNames.SIGMA),
                'layer_name': row.get('name', f"layer_{layer_id}")
            }

            self._plot_detailed_layer_diagnostics(
                evals, metrics, model_name, layer_id, plot_dir
            )

    def _plot_detailed_layer_diagnostics(
        self, evals: np.ndarray, metrics: dict,
        model_name: str, layer_id: int, savedir: str
    ) -> None:
        """
        Create a 4-panel diagnostic plot (Log-Log, Lin-Lin, Log-Lin, KS Distance).
        Matches Figure 1 of the SETOL paper.
        """
        try:
            evals_clean = evals[evals > SPECTRAL_EPSILON]
            if len(evals_clean) == 0: return

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            layer_name = metrics['layer_name']
            alpha = metrics['alpha']
            xmin = metrics['xmin']
            xmax = metrics['xmax']

            # Common data setup
            dynamic_bins = min(int(np.sqrt(len(evals_clean))), SPECTRAL_DEFAULT_BINS)

            # --- Panel (a): Log-Log ESD (The classic Power Law plot) ---
            ax_loglog = axes[0, 0]
            bins_log = np.logspace(np.log10(np.min(evals_clean)), np.log10(np.max(evals_clean)), dynamic_bins)
            hist, _ = np.histogram(evals_clean, bins=bins_log, density=True)
            bin_centers = np.sqrt(bins_log[:-1] * bins_log[1:])
            mask = hist > 0

            ax_loglog.loglog(bin_centers[mask], hist[mask], '.', markersize=8, label='ESD')

            # Fit line
            if xmin > 0 and alpha > 1:
                x_range = np.logspace(np.log10(xmin), np.log10(np.max(evals_clean)), 50)
                # PDF Scaling: Match tail mass fraction
                tail_frac = np.sum(evals_clean >= xmin) / len(evals_clean)
                y_fit = tail_frac * ((alpha - 1) / xmin) * ((x_range / xmin) ** (-alpha))
                ax_loglog.loglog(x_range, y_fit, 'r--', label=f'Fit $\\alpha={alpha:.2f}$')

            ax_loglog.axvline(xmin, color='r', linestyle='-', alpha=0.6, label='$x_{min}$')
            ax_loglog.axvline(xmax, color='orange', linestyle='-', alpha=0.6, label='$x_{max}$')

            ax_loglog.set_title(f"(a) Log-Log ESD for {model_name} {layer_name}")
            ax_loglog.set_ylabel('Density $P(\\lambda)$')
            ax_loglog.set_xlabel('Eigenvalue $\\lambda$')
            ax_loglog.legend(loc='best', fontsize='small')

            # --- Panel (b): Lin-Lin ESD (Standard Histogram) ---
            ax_lin = axes[0, 1]
            ax_lin.hist(evals_clean, bins=dynamic_bins, density=True, alpha=0.9, label='ESD')
            ax_lin.axvline(xmin, color='r', linestyle='-', alpha=0.6, label='$x_{min}$')
            ax_lin.axvline(xmax, color='orange', linestyle='-', alpha=0.6, label='$x_{max}$')

            ax_lin.set_title(f"(b) Lin-Lin ESD for {model_name} {layer_name}")
            ax_lin.set_ylabel('$P(\\lambda)$')
            ax_lin.set_xlabel('$\\lambda$')
            ax_lin.legend(loc='best', fontsize='small')

            # --- Panel (c): Log-Lin ESD (Histogram of Log Eigenvalues) ---
            # X axis is log(lambda), Y axis is linear density
            ax_loglin = axes[1, 0]
            log_evals = np.log10(evals_clean)
            ax_loglin.hist(log_evals, bins=dynamic_bins, density=True, alpha=0.9)

            ax_loglin.axvline(np.log10(xmin), color='r', linestyle='-', alpha=0.6, label='log $x_{min}$')
            if xmax > 0:
                ax_loglin.axvline(np.log10(xmax), color='orange', linestyle='-', alpha=0.6, label='log $x_{max}$')

            ax_loglin.set_title(f"(c) Log-Lin ESD for {model_name} {layer_name}")
            ax_loglin.set_xlabel('log$_{10} \\lambda$')
            ax_loglin.legend(loc='best', fontsize='small')

            # --- Panel (d): KS Distance vs xmin (Optimization Landscape) ---
            ax_ks = axes[1, 1]
            xmins, ks_dists = self._scan_ks_distances(evals_clean)

            if len(xmins) > 0:
                ax_ks.plot(xmins, ks_dists, '-', linewidth=1, label='$D_{KS}$')
                ax_ks.axvline(xmin, color='r', linestyle='-', alpha=0.6, label='Selected $x_{min}$')

                # Point to the minimum
                min_ks_idx = np.argmin(ks_dists)
                ax_ks.scatter([xmins[min_ks_idx]], [ks_dists[min_ks_idx]], color='r', s=30)

            ax_ks.set_title(f"(d) KS Distance ($D_{{KS}}$) vs $x_{{min}}$")
            ax_ks.set_xlabel('$x_{min}$ candidate')
            ax_ks.set_ylabel('$D_{KS}$')
            ax_ks.legend(loc='best', fontsize='small')

            plt.tight_layout()

            if self.config.save_plots:
                sane_layer_name = layer_name.replace('/', '_').replace(':', '')
                filepath = savedir / f"{model_name}_layer_{layer_id}_{sane_layer_name}_diagnostics.png"
                fig.savefig(filepath, dpi=SPECTRAL_DEFAULT_DPI, bbox_inches='tight')

            plt.close(fig)

        except Exception as e:
            logger.warning(f"Error creating diagnostic plots for {model_name} layer {layer_id}: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def _create_trap_overlay_plots(self) -> None:
        """
        Create ESD overlay plots comparing original vs randomized weight spectra
        with Marchenko-Pastur edge and Tracy-Widom threshold markers.

        Only layers with randomization data (spectral_randomize=True) are plotted.
        Layers with detected traps get prominent spike markers.
        """
        df = self.results.spectral_analysis
        esds = self.results.spectral_esds
        rand_esds = getattr(self.results, 'spectral_rand_esds', {})

        if not rand_esds:
            logger.info("No randomized ESD data available for trap overlay plots.")
            return

        plot_dir = self.output_dir / "trap_plots"
        if self.config.save_plots:
            plot_dir.mkdir(exist_ok=True)

        for index, row in df.iterrows():
            model_name = row['model_name']
            layer_id = int(row['layer_id'])

            # Need both original and randomized eigenvalues
            if model_name not in esds or layer_id not in esds[model_name]:
                continue
            if model_name not in rand_esds or layer_id not in rand_esds[model_name]:
                continue

            evals = esds[model_name][layer_id]
            rand_evals = rand_esds[model_name][layer_id]
            if evals is None or rand_evals is None:
                continue

            # Extract trap metrics from the DataFrame row
            has_trap = row.get(MetricNames.HAS_TRAP, False)
            mp_lambda_plus = row.get(MetricNames.MP_LAMBDA_PLUS, 0)
            trap_threshold = row.get(MetricNames.TRAP_THRESHOLD, 0)
            severity_label = row.get(MetricNames.TRAP_SEVERITY_LABEL, 'none')
            num_spikes = row.get(MetricNames.NUM_RAND_SPIKES, 0)
            layer_name = row.get('name', f'layer_{layer_id}')

            try:
                self._plot_trap_overlay(
                    evals, rand_evals, layer_name, model_name, layer_id,
                    has_trap, mp_lambda_plus, trap_threshold,
                    severity_label, num_spikes, plot_dir
                )
            except Exception as e:
                logger.warning(f"Error creating trap overlay for {model_name} layer {layer_id}: {e}")

    def _plot_trap_overlay(
        self, evals: np.ndarray, rand_evals: np.ndarray,
        layer_name: str, model_name: str, layer_id: int,
        has_trap: bool, mp_lambda_plus: float, trap_threshold: float,
        severity_label: str, num_spikes: int, savedir
    ) -> None:
        """
        Create a 2-panel plot: Linear ESD overlay (left) and Log-Log ESD overlay (right).

        Shows original vs randomized weight spectra with MP edge and TW threshold markers.
        Trap spikes are prominently annotated when detected.
        """
        evals_clean = evals[evals > SPECTRAL_EPSILON]
        rand_clean = rand_evals[rand_evals > SPECTRAL_EPSILON]
        if len(evals_clean) == 0 or len(rand_clean) == 0:
            return

        fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(16, 6))

        trap_status = f"TRAP ({severity_label})" if has_trap else "Clean"
        fig.suptitle(
            f'Correlation Trap Analysis: {model_name} / {layer_name} — [{trap_status}]',
            fontsize=14, fontweight='bold',
            color='#c62828' if has_trap else '#2e7d32'
        )

        # Shared bin range for both histograms
        all_evals = np.concatenate([evals_clean, rand_clean])
        bin_min, bin_max = np.min(all_evals), np.max(all_evals)

        # --- Panel 1: Linear Scale ESD ---
        bins_lin = np.linspace(bin_min, bin_max, 80)
        ax_lin.hist(evals_clean, bins=bins_lin, alpha=0.6, density=True,
                    color='#2e7d32', label='Original W', edgecolor='black', linewidth=0.3)
        ax_lin.hist(rand_clean, bins=bins_lin, alpha=0.5, density=True,
                    color='#c62828', label='Randomized W$^{rand}$', edgecolor='black', linewidth=0.3)

        # MP edge and threshold markers
        if mp_lambda_plus > 0:
            ax_lin.axvline(mp_lambda_plus, color='#1565c0', linestyle='--', linewidth=2,
                          label=f'$\\lambda_+$ (MP edge) = {mp_lambda_plus:.2f}')
        if trap_threshold > 0:
            ax_lin.axvline(trap_threshold, color='#ff6f00', linestyle='--', linewidth=2,
                          label=f'Threshold ($\\lambda_+$ + $\\Delta_{{TW}}$) = {trap_threshold:.2f}')

        # Mark trap spikes
        if has_trap and trap_threshold > 0:
            spike_evals = rand_clean[rand_clean > trap_threshold]
            if len(spike_evals) > 0:
                ylim = ax_lin.get_ylim()
                for spike in spike_evals[:3]:  # Mark up to 3 spikes
                    ax_lin.axvline(spike, color='#d50000', linestyle='-', linewidth=2.5, alpha=0.8)
                ax_lin.text(np.max(spike_evals), ylim[1] * 0.85, f'TRAP!\n{num_spikes} spike(s)',
                           color='#d50000', fontsize=11, fontweight='bold', ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcdd2', alpha=0.9))

        ax_lin.set_xlabel('Eigenvalue $\\lambda$')
        ax_lin.set_ylabel('Density')
        ax_lin.set_title('(a) Linear Scale ESD Overlay')
        ax_lin.legend(loc='upper right', fontsize='small')
        ax_lin.grid(True, alpha=0.3)

        # --- Panel 2: Log-Log Scale ESD ---
        bins_log = np.logspace(np.log10(max(bin_min, SPECTRAL_EPSILON)),
                               np.log10(bin_max), 60)

        hist_orig, _ = np.histogram(evals_clean, bins=bins_log, density=True)
        hist_rand, _ = np.histogram(rand_clean, bins=bins_log, density=True)
        bin_centers = np.sqrt(bins_log[:-1] * bins_log[1:])

        mask_orig = hist_orig > 0
        mask_rand = hist_rand > 0

        if np.any(mask_orig):
            ax_log.loglog(bin_centers[mask_orig], hist_orig[mask_orig], 'o',
                         color='#2e7d32', markersize=5, alpha=0.7, label='Original W')
        if np.any(mask_rand):
            ax_log.loglog(bin_centers[mask_rand], hist_rand[mask_rand], 's',
                         color='#c62828', markersize=5, alpha=0.7, label='Randomized W$^{rand}$')

        # MP edge and threshold on log scale
        if mp_lambda_plus > 0:
            ax_log.axvline(mp_lambda_plus, color='#1565c0', linestyle='--', linewidth=2,
                          label=f'$\\lambda_+$ = {mp_lambda_plus:.2f}')
        if trap_threshold > 0:
            ax_log.axvline(trap_threshold, color='#ff6f00', linestyle='--', linewidth=2,
                          label=f'Threshold = {trap_threshold:.2f}')

        ax_log.set_xlabel('Eigenvalue $\\lambda$ (log scale)')
        ax_log.set_ylabel('Density (log scale)')
        ax_log.set_title('(b) Log-Log Scale ESD Overlay')
        ax_log.legend(loc='upper right', fontsize='small')
        ax_log.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.config.save_plots:
            sane_name = layer_name.replace('/', '_').replace(':', '')
            filepath = savedir / f"{model_name}_layer_{layer_id}_{sane_name}_trap_overlay.png"
            fig.savefig(filepath, dpi=SPECTRAL_DEFAULT_DPI, bbox_inches='tight')

        plt.close(fig)

    def _scan_ks_distances(self, evals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate KS distance for a range of potential xmin values.

        Replicates the logic of fit_powerlaw (with precomputed suffix sums)
        and returns the full profile for plotting. No subsampling — the curve
        must match the actual fit result exactly.
        """
        try:
            data = np.sort(evals)
            N = len(data)
            if N < 10:
                return np.array([]), np.array([])

            scan_indices = np.arange(0, N - 5)
            xmins = data[scan_indices]
            ks_dists = np.ones(len(scan_indices), dtype=np.float64)

            log_data = np.log(data)
            # Precompute suffix sums for O(1) tail-sum lookups
            tail_sums = np.cumsum(log_data[::-1])[::-1]

            for j, i in enumerate(scan_indices):
                curr_xmin = data[i]
                if curr_xmin <= 0:
                    continue

                n_tail = N - i
                sum_log_tail = tail_sums[i]
                denom = sum_log_tail - n_tail * log_data[i]

                if denom <= SPECTRAL_EPSILON:
                    continue

                alpha = 1.0 + n_tail / denom
                if alpha <= 1.0:
                    continue

                cdf_emp = np.arange(n_tail) / n_tail
                cdf_theo = 1 - (data[i:] / curr_xmin) ** (-alpha + 1.0)
                ks_dists[j] = np.max(np.abs(cdf_emp - cdf_theo))

            return xmins, ks_dists

        except Exception:
            return np.array([]), np.array([])

# ---------------------------------------------------------------------
