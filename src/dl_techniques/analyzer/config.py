"""
Configuration for Model Analyzer

Configuration classes and plotting setup utilities.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple

# ---------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """Configuration for all analysis types."""

    # Analysis toggles
    analyze_weights: bool = True
    analyze_calibration: bool = True
    analyze_information_flow: bool = True
    analyze_training_dynamics: bool = True

    # Sampling parameters
    n_samples: int = 1000
    n_samples_per_digit: int = 3
    sample_digits: Optional[List[int]] = None

    # Layer selection
    activation_layer_name: Optional[str] = None
    activation_layer_index: Optional[int] = None

    # Weight analysis options
    weight_layer_types: Optional[List[str]] = None
    analyze_biases: bool = False
    compute_weight_pca: bool = True

    # Calibration options
    calibration_bins: int = 10

    # Training analysis options
    smooth_training_curves: bool = True
    smoothing_window: int = 5

    # Visualization settings
    plot_style: str = 'publication'
    color_palette: str = 'deep'
    fig_width: int = 12
    fig_height: int = 8
    dpi: int = 300
    save_plots: bool = True
    save_format: str = 'png'

    # Advanced options
    show_statistical_tests: bool = True
    show_confidence_intervals: bool = True
    verbose: bool = True

    # Configurable visualization parameters (addressing hardcoded values)
    max_layers_heatmap: int = 12  # Maximum layers to show in weight health heatmap
    max_layers_info_flow: int = 8  # Maximum layers to show in information flow analysis
    pareto_analysis_threshold: int = 2  # Minimum models needed for Pareto analysis

    # Exception handling specificity
    catch_specific_exceptions: bool = True  # When True, catch more specific exceptions

    # Performance settings
    enable_parallel_analysis: bool = False  # For future parallel processing
    memory_limit_mb: Optional[int] = None  # For memory-constrained environments

    def get_figure_size(self, scale: float = 1.0) -> Tuple[float, float]:
        """Get figure size with optional scaling."""
        return (self.fig_width * scale, self.fig_height * scale)

    def setup_plotting_style(self) -> None:
        """Set up matplotlib style based on configuration."""
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

# ---------------------------------------------------------------------