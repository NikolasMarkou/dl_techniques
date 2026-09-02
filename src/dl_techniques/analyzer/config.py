"""
Configuration for Model Analyzer

Configuration classes and plotting setup utilities.
"""

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

# ---------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """Configuration for all analysis types."""

    # Analysis toggles
    analyze_weights: bool = True
    analyze_calibration: bool = True
    analyze_information_flow: bool = True
    analyze_training_dynamics: bool = True
    analyze_spectral: bool = True

    # Sampling parameters
    n_samples: int = 1000

    # Weight analysis options
    weight_layer_types: Optional[List[str]] = None
    analyze_biases: bool = False
    compute_weight_pca: bool = True

    # Calibration options
    calibration_bins: int = 10

    # Training analysis options
    smooth_training_curves: bool = True
    smoothing_window: int = 5

    # Spectral Analysis (WeightWatcher) options
    spectral_min_evals: int = 10
    spectral_max_evals: int = 15000
    spectral_glorot_fix: bool = False
    spectral_randomize: bool = False
    # Number of independent element-wise permutations averaged per layer when
    # `spectral_randomize` is on. A single draw made every correlation-trap verdict
    # a coin flip; >= 5 is required (plan-2026-09-01T225724-e79ad4bd D-017).
    spectral_n_randomizations: int = 5
    spectral_concentration_analysis: bool = True

    # NEW FLAG: Controls generation of individual layer plot files
    spectral_per_layer_diagnostics: bool = False
    spectral_bootstraps: int = 50

    # Visualization settings
    plot_style: str = 'publication'
    color_palette: str = 'deep'
    fig_width: int = 12
    fig_height: int = 8
    dpi: int = 300
    save_plots: bool = True
    save_format: str = 'png'

    # Advanced options
    verbose: bool = True
    # DECISION plan-2026-09-01T225724-e79ad4bd/D-031
    # Seed for EVERY stochastic site in the package: the `DataSampler` draw, the
    # spectral randomization permutations, the goodness-of-fit bootstrap and
    # `_power_iteration`. `None` keeps the historical unseeded behaviour.
    # See decisions.md D-031.
    random_state: Optional[int] = None

    # JSON serialization options
    json_include_per_sample_data: bool = False  # Set to False to exclude bulky per-sample arrays (e.g., confidence, entropy)
    # When True, `save_results` emits `spectral_esds` (and `spectral_rand_esds`
    # when randomization ran) into the artifact. Kept False by default: one entry
    # per analyzed layer, each the full eigenvalue spectrum.
    json_include_raw_esds: bool = False

    # Configurable visualization parameters (addressing hardcoded values)
    max_layers_heatmap: int = 12  # Maximum layers to show in weight health heatmap
    max_layers_info_flow: int = 8  # Maximum layers to show in information flow analysis
    pareto_analysis_threshold: int = 2  # Minimum models needed for Pareto analysis

    # Performance settings
    # DECISION plan-2026-09-01T225724-e79ad4bd/D-022
    # Budget for the activations `InformationFlowAnalyzer` holds in memory at once.
    # This defaults to a real number, NOT None: the analyzer retains every wrapped
    # layer's output simultaneously, which is an analytic 16.94 GB for ResNet50 at
    # B=200. `None` means explicitly unbounded and is honoured as such.
    # See decisions.md D-022.
    memory_limit_mb: Optional[int] = 2048

    # DECISION plan-2026-09-01T225724-e79ad4bd/D-029
    # `setup_plotting_style` mutates process-global matplotlib state and stashes the
    # pre-existing rcParams here. It is a DECLARED field, not an attribute conjured
    # by that method: an undeclared attribute is absent from `fields()`/`asdict()`
    # and forced `save_results` to filter it out BY STRING NAME. Do NOT un-declare
    # it, and do NOT serialize it - it is private and holds an unJSONable RcParams.
    # See decisions.md D-029.
    _original_rcParams: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def get_figure_size(self, scale: float = 1.0) -> Tuple[float, float]:
        """Get figure size with optional scaling."""
        return (self.fig_width * scale, self.fig_height * scale)

    def setup_plotting_style(self) -> None:
        """Set up matplotlib style based on configuration."""
        # Use non-interactive backend for file-based rendering
        matplotlib.use('Agg')

        # Save current rcParams to restore later if needed
        self._original_rcParams = plt.rcParams.copy()

        plt.style.use('default')

        # Apply style presets
        style_settings = {
            'publication': {
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 11,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 14,
                'lines.linewidth': 2,
                'lines.markersize': 6,
                'axes.linewidth': 1,
                'grid.alpha': 0.3,
            },
            'presentation': {
                'font.size': 14,
                'axes.titlesize': 18,
                'axes.labelsize': 16,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'legend.fontsize': 14,
                'figure.titlesize': 20,
                'lines.linewidth': 3,
                'lines.markersize': 10,
                'axes.linewidth': 2,
                'grid.alpha': 0.4,
            },
            'draft': {
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 11,
                'figure.titlesize': 16,
                'lines.linewidth': 2.5,
                'lines.markersize': 8,
                'axes.linewidth': 1.5,
                'grid.alpha': 0.3,
            }
        }

        if self.plot_style in style_settings:
            plt.rcParams.update(style_settings[self.plot_style])

        plt.rcParams.update({
            'figure.figsize': (self.fig_width, self.fig_height),
            'figure.dpi': 100,
            'savefig.dpi': self.dpi,
            'axes.grid': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.axisbelow': True,
            'figure.autolayout': False,
        })

        sns.set_theme(style='whitegrid', palette=self.color_palette)

    def restore_plotting_style(self) -> None:
        """Restore the rcParams captured by the last ``setup_plotting_style`` call.

        ``setup_plotting_style`` mutates process-global matplotlib state, which
        outlives the ``ModelAnalyzer`` that triggered it. This puts it back. The
        backend is deliberately NOT restored: ``Agg`` is a repo-wide headless
        requirement, not part of this configuration's styling.

        Does nothing if no style has been applied yet.
        """
        if self._original_rcParams is None:
            return
        plt.rcParams.update(self._original_rcParams)

# ---------------------------------------------------------------------