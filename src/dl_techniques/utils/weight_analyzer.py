"""
Enhanced Weight Distribution Analysis for Neural Networks with Different Normalization Schemes.

This module provides comprehensive analysis and visualization tools for neural network weight
distributions across different layers and normalization approaches. It supports various
analysis methods including:
- Channel-wise weight distributions
- Layer-wise comparisons
- Multiple norm distributions (L1, L2, RMS)
- Weight direction and orthogonality analysis
- Statistical significance testing
- Hierarchical clustering of weight patterns
- Export capabilities for analysis results
"""
import json
import keras
import numpy as np
import pandas as pd
import seaborn as sns
from keras import Model
from scipy import stats
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

# ------------------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------------------

from .logger import logger

# ------------------------------------------------------------------------------


@dataclass
class WeightAnalyzerConfig:
    """Configuration for weight analysis parameters.

    Attributes:
        compute_l1_norm: Whether to compute L1 norms
        compute_l2_norm: Whether to compute L2 norms
        compute_rms_norm: Whether to compute RMS norms
        compute_statistics: Whether to compute basic statistics
        compute_histograms: Whether to compute histograms
        analyze_biases: Whether to analyze bias terms
        layer_types: List of layer types to analyze
        plot_style: Matplotlib style to use
        color_palette: Seaborn color palette to use
        fig_width: Width of output figures
        fig_height: Height of output figures
        dpi: DPI for saved figures
        save_plots: Whether to save plots
        save_stats: Whether to save statistics
        export_format: Format for exported files
    """
    # Norm analysis options
    compute_l1_norm: bool = True
    compute_l2_norm: bool = True
    compute_rms_norm: bool = True

    # Distribution analysis options
    compute_statistics: bool = True
    compute_histograms: bool = True

    # Layer analysis options
    analyze_biases: bool = False
    layer_types: Optional[List[str]] = None

    # Visualization options
    plot_style: str = 'default'  # Use matplotlib's default style
    color_palette: str = 'deep'  # Use seaborn's default palette
    fig_width: int = 12
    fig_height: int = 8
    dpi: int = 300

    # Export options
    save_plots: bool = True
    save_stats: bool = True
    export_format: str = 'png'

    def setup_plotting_style(self) -> None:
        """Set up matplotlib and seaborn plotting styles safely."""
        try:
            # Reset to matplotlib defaults first
            plt.style.use('default')

            # Set up seaborn defaults
            sns.set_theme(style='whitegrid')
            sns.set_palette(self.color_palette)

            # Apply any custom matplotlib style if specified
            if self.plot_style != 'default':
                try:
                    plt.style.use(self.plot_style)
                except Exception as e:
                    logger.warning(f"Could not apply style {self.plot_style}, falling back to default. Error: {e}")
        except Exception as e:
            logger.warning(f"Error setting up plotting style: {e}")
            # Ensure we have a workable style
            plt.style.use('default')

# ------------------------------------------------------------------------------


class LayerWeightStatistics:
    """Container for layer-specific weight statistics."""

    def __init__(self, layer_name: str, weights: np.ndarray) -> None:
        """
        Initialize layer statistics container.

        Args:
            layer_name: Name of the analyzed layer
            weights: Weight tensor from the layer

        Raises:
            ValueError: If weights array is empty or invalid
        """
        if not isinstance(weights, np.ndarray) or weights.size == 0:
            raise ValueError("Weights must be a non-empty numpy array")

        self.layer_name = layer_name
        self._compute_basic_stats(weights)
        self._compute_norm_stats(weights)
        self._compute_direction_stats(weights)

    def _compute_basic_stats(self, weights: np.ndarray) -> None:
        """Compute basic statistical measures of weights."""
        try:
            flat_weights = weights.flatten()
            self.basic_stats = {
                'mean': np.mean(flat_weights),
                'std': np.std(flat_weights),
                'median': np.median(flat_weights),
                'min': np.min(flat_weights),
                'max': np.max(flat_weights),
                'skewness': float(stats.skew(flat_weights)),
                'kurtosis': float(stats.kurtosis(flat_weights))
            }
        except Exception as e:
            logger.error(f"Error computing basic stats for {self.layer_name}: {e}")
            self.basic_stats = {}

    def _compute_norm_stats(self, weights: np.ndarray) -> None:
        """Compute various norm statistics of weights."""
        try:
            self.norm_stats = {
                'l1_norm': float(np.sum(np.abs(weights))),
                'l2_norm': float(np.sqrt(np.sum(weights ** 2))),
                'rms_norm': float(np.sqrt(np.mean(weights ** 2))),
                'max_norm': float(np.max(np.abs(weights)))
            }
        except Exception as e:
            logger.error(f"Error computing norm stats for {self.layer_name}: {e}")
            self.norm_stats = {}

    def _compute_direction_stats(self, weights: np.ndarray) -> None:
        """Compute directional statistics for weights."""
        try:
            if len(weights.shape) >= 2:
                w_flat = weights.reshape(weights.shape[0], -1)
                norms = np.sqrt(np.sum(w_flat ** 2, axis=1))

                # Compute cosine similarities safely
                cosine_sim = w_flat @ w_flat.T
                norm_outer = np.outer(norms, norms)
                mask = norm_outer > 1e-8
                cosine_sim[mask] /= norm_outer[mask]
                cosine_sim[~mask] = 0

                self.direction_stats = {
                    'cosine_similarities': cosine_sim,
                    'mean_orthogonality': float(np.mean(np.abs(cosine_sim - np.eye(len(cosine_sim))))),
                    'filter_norms': norms.tolist()
                }
            else:
                self.direction_stats = {}
        except Exception as e:
            logger.error(f"Error computing direction stats for {self.layer_name}: {e}")
            self.direction_stats = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to a dictionary format."""
        return {
            'layer_name': self.layer_name,
            'basic_stats': self.basic_stats,
            'norm_stats': self.norm_stats,
            'direction_stats': self.direction_stats
        }


class WeightAnalyzer:
    """Enhanced analyzer for neural network weight distributions."""

    def __init__(
            self,
            models: Dict[str, Model],
            config: Optional[WeightAnalyzerConfig] = None,
            output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize the weight analyzer.

        Args:
            models: Dictionary mapping model names to Keras models
            config: Configuration object for analysis parameters
            output_dir: Directory to save analysis outputs

        Raises:
            ValueError: If models dict is empty or invalid
        """
        if not models or not all(isinstance(m, Model) for m in models.values()):
            raise ValueError("Models must be a non-empty dict of Keras models")

        self.models = models
        self.config = config or WeightAnalyzerConfig()
        self.output_dir = Path(output_dir or "weight_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analysis results container
        self.layer_statistics: Dict[str, Dict[str, LayerWeightStatistics]] = {}

        # Perform initial analysis
        self._analyze_models()

    def _analyze_models(self) -> None:
        """Analyze weights of all models."""
        for model_name, model in self.models.items():
            self.layer_statistics[model_name] = {}

            for layer in model.layers:
                weights = layer.get_weights()

                if not weights or (
                        self.config.layer_types and
                        layer.__class__.__name__ not in self.config.layer_types
                ):
                    continue

                try:
                    for idx, w in enumerate(weights):
                        if len(w.shape) < 2 and not self.config.analyze_biases:
                            continue

                        stats = LayerWeightStatistics(
                            f"{layer.name}_weight_{idx}",
                            w
                        )
                        self.layer_statistics[model_name][stats.layer_name] = stats
                except Exception as e:
                    logger.error(f"Error analyzing layer {layer.name}: {e}")

    def plot_norm_distributions(
            self,
            norm_types: Optional[List[str]] = None,
            save: bool = True
    ) -> plt.Figure:
        """
        Plot norm distributions for all models.

        Args:
            norm_types: List of norm types to plot. Defaults to ['l2_norm', 'rms_norm']
            save: Whether to save the plot

        Returns:
            matplotlib.figure.Figure: Generated figure

        Raises:
            ValueError: If no valid norm data is available
        """
        norm_types = norm_types or ['l2_norm', 'rms_norm']

        # Validate norm types
        for norm_type in norm_types:
            if not any(
                    norm_type in stats.norm_stats
                    for layer_stats in self.layer_statistics.values()
                    for stats in layer_stats.values()
            ):
                raise ValueError(f"No valid data for norm type: {norm_type}")

        fig, axes = plt.subplots(
            len(norm_types), 1,
            figsize=(self.config.fig_width, self.config.fig_height * len(norm_types)),
            squeeze=False
        )
        axes = axes.ravel()

        for ax, norm_type in zip(axes, norm_types):
            for model_name, layer_stats in self.layer_statistics.items():
                norms = [
                    stats.norm_stats[norm_type]
                    for stats in layer_stats.values()
                    if norm_type in stats.norm_stats
                ]

                if norms:
                    sns.kdeplot(data=norms, label=model_name, ax=ax)

            ax.set_title(f'{norm_type.replace("_", " ").title()} Distribution')
            ax.set_xlabel('Norm Value')
            ax.legend()

        plt.tight_layout()

        if save and self.config.save_plots:
            fig.savefig(
                self.output_dir / f"norm_distributions.{self.config.export_format}",
                dpi=self.config.dpi
            )

        return fig

    def plot_layer_comparisons(
            self,
            metrics: Optional[List[str]] = None,
            save: bool = True
    ) -> plt.Figure:
        """
        Plot layer-wise comparisons across models.

        Args:
            metrics: List of metrics to plot. Defaults to ['mean', 'std', 'l2_norm', 'mean_orthogonality']
            save: Whether to save the plot

        Returns:
            matplotlib.figure.Figure: Generated figure

        Raises:
            ValueError: If no valid metric data is available
        """
        metrics = metrics or ['mean', 'std', 'l2_norm', 'mean_orthogonality']

        fig, axes = plt.subplots(
            len(metrics), 1,
            figsize=(self.config.fig_width, self.config.fig_height * len(metrics)),
            squeeze=False
        )
        axes = axes.ravel()

        for ax, metric in zip(axes, metrics):
            data = []
            model_names = []
            layer_names = []

            for model_name, layer_stats in self.layer_statistics.items():
                for layer_name, stats in layer_stats.items():
                    value = None

                    # Try to find the metric in different stat categories
                    if metric in stats.basic_stats:
                        value = stats.basic_stats[metric]
                    elif metric in stats.norm_stats:
                        value = stats.norm_stats[metric]
                    elif metric in stats.direction_stats and metric != 'cosine_similarities':
                        value = stats.direction_stats[metric]

                    if value is not None:
                        data.append(value)
                        model_names.append(model_name)
                        layer_names.append(layer_name)

            if data:
                df = pd.DataFrame({
                    'Model': model_names,
                    'Layer': layer_names,
                    'Value': data
                })

                sns.barplot(data=df, x='Layer', y='Value', hue='Model', ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()} by Layer')
                ax.tick_params(axis='x', rotation=45)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save and self.config.save_plots:
            fig.savefig(
                self.output_dir / f"layer_comparisons.{self.config.export_format}",
                dpi=self.config.dpi
            )

        return fig

    def plot_weight_distributions(
            self,
            plot_type: str = 'histogram',
            save: bool = True
    ) -> plt.Figure:
        """
        Plot weight distributions for all models.

        Args:
            plot_type: Type of plot ('histogram' or 'violin')
            save: Whether to save the plot

        Returns:
            matplotlib.figure.Figure: Generated figure

        Raises:
            ValueError: If plot_type is invalid
        """
        if plot_type not in ['histogram', 'violin']:
            raise ValueError("plot_type must be 'histogram' or 'violin'")

        fig = plt.figure(figsize=(self.config.fig_width, self.config.fig_height))

        for model_name, layer_stats in self.layer_statistics.items():
            weights = [
                stats.basic_stats['mean']
                for stats in layer_stats.values()
                if 'mean' in stats.basic_stats
            ]

            if plot_type == 'histogram':
                plt.hist(weights, bins=50, alpha=0.5, label=model_name)
            else:
                plt.violinplot(weights, points=100, vert=False)

        plt.title('Weight Distribution Comparison')
        plt.xlabel('Weight Value')
        plt.legend()

        if save and self.config.save_plots:
            fig.savefig(
                self.output_dir / f"weight_dist_{plot_type}.{self.config.export_format}",
                dpi=self.config.dpi
            )

        return fig

    def plot_layer_weight_histograms(
            self,
            max_layers_per_figure: int = 9,
            save: bool = True
    ) -> List[plt.Figure]:
        """
        Plot weight histograms for each layer, comparing models side by side.

        Args:
            max_layers_per_figure: Maximum number of layers to plot in a single figure
            save: Whether to save the plots

        Returns:
            List[matplotlib.figure.Figure]: List of generated figures

        Raises:
            ValueError: If no valid weight data is available
        """
        # Collect all unique layer names across models
        all_layer_names = set()
        for layer_stats in self.layer_statistics.values():
            all_layer_names.update(layer_stats.keys())

        if not all_layer_names:
            raise ValueError("No valid layer data available for plotting")

        # Sort layer names for consistent ordering
        layer_names = sorted(all_layer_names)
        n_layers = len(layer_names)

        # Calculate number of figures needed
        n_figures = (n_layers + max_layers_per_figure - 1) // max_layers_per_figure
        figures = []

        for fig_idx in range(n_figures):
            # Calculate grid dimensions for this figure
            start_idx = fig_idx * max_layers_per_figure
            end_idx = min(start_idx + max_layers_per_figure, n_layers)
            n_layers_this_fig = end_idx - start_idx

            n_rows = int(np.ceil(np.sqrt(n_layers_this_fig)))
            n_cols = int(np.ceil(n_layers_this_fig / n_rows))

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(self.config.fig_width, self.config.fig_height),
                squeeze=False
            )

            # Plot each layer in this figure
            for i, layer_name in enumerate(layer_names[start_idx:end_idx]):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]

                # Plot weights for each model that has this layer
                for model_name, layer_stats in self.layer_statistics.items():
                    if layer_name in layer_stats:
                        stats = layer_stats[layer_name]
                        if 'mean' in stats.basic_stats:
                            weights = [stats.basic_stats['mean']]
                            ax.hist(
                                weights,
                                bins='auto',
                                alpha=0.5,
                                label=model_name,
                                density=True  # Normalize to compare distributions
                            )

                ax.set_title(layer_name, fontsize=10)
                ax.tick_params(labelsize=8)
                if i == 0:  # Only show legend on first subplot
                    ax.legend(fontsize=8)

            # Remove empty subplots
            for i in range(n_layers_this_fig, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                fig.delaxes(axes[row, col])

            plt.tight_layout()

            if save and self.config.save_plots:
                fig.savefig(
                    self.output_dir / f"layer_histograms_{fig_idx + 1}.{self.config.export_format}",
                    dpi=self.config.dpi,
                    bbox_inches='tight'
                )

            figures.append(fig)

        return figures

    def save_analysis_results(self, filename: str = "analysis_results") -> None:
        """
        Save analysis results to JSON.

        Args:
            filename: Base filename for saving results

        Raises:
            IOError: If unable to save results
        """
        if not self.config.save_stats:
            return

        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_models': len(self.models),
                'config': self.config.__dict__
            },
            'model_statistics': {
                model_name: {
                    layer_name: stats.to_dict()
                    for layer_name, stats in layer_stats.items()
                }
                for model_name, layer_stats in self.layer_statistics.items()
            }
        }

        try:
            with open(self.output_dir / f"{filename}.json", 'w') as f:
                json.dump(results, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving analysis results: {e}")
            raise

    def compute_statistical_tests(
            self,
            metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform statistical tests comparing weight distributions.

        Args:
            metrics: List of metrics to test. Defaults to ['mean', 'std', 'l2_norm']

        Returns:
            Dictionary containing test results
        """
        metrics = metrics or ['mean', 'std', 'l2_norm']
        results = {}

        for metric in metrics:
            try:
                model_data = {}
                for model_name, layer_stats in self.layer_statistics.items():
                    metric_values = []
                    for stats in layer_stats.values():
                        if metric in stats.basic_stats:
                            metric_values.append(stats.basic_stats[metric])
                        elif metric in stats.norm_stats:
                            metric_values.append(stats.norm_stats[metric])

                    if metric_values:
                        model_data[model_name] = metric_values

                if len(model_data) >= 2:  # Need at least 2 groups for comparison
                    h_stat, p_value = stats.kruskal(*model_data.values())
                    results[f'{metric}_kruskal'] = {
                        'statistic': float(h_stat),
                        'p_value': float(p_value)
                    }
            except Exception as e:
                logger.error(f"Error computing statistics for {metric}: {e}")

        return results

