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


class LayerAnalysis:
    """Container for layer-specific analysis results."""

    def __init__(self, layer_name: str):
        """
        Initialize layer analysis container.

        Args:
            layer_name: Name of the layer being analyzed
        """
        self.layer_name = layer_name
        self.weight_stats: Dict[str, np.ndarray] = {}
        self.norm_stats: Dict[str, np.ndarray] = {}
        self.direction_stats: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Any] = {}

    def add_weight_stats(self, weights: np.ndarray) -> None:
        """Add basic weight statistics."""
        self.weight_stats.update({
            'mean': np.mean(weights),
            'std': np.std(weights),
            'median': np.median(weights),
            'skewness': stats.skew(weights.flatten()),
            'kurtosis': stats.kurtosis(weights.flatten())
        })

    def add_norm_stats(self, weights: np.ndarray) -> None:
        """Compute and store various norm statistics."""
        self.norm_stats.update({
            'l1_norm': np.sum(np.abs(weights)),
            'l2_norm': np.sqrt(np.sum(weights ** 2)),
            'rms_norm': np.sqrt(np.mean(weights ** 2)),
            'max_norm': np.max(np.abs(weights))
        })

    def add_direction_stats(self, weights: np.ndarray) -> None:
        """Compute directional statistics for weights."""
        if len(weights.shape) >= 2:
            w_flat = weights.reshape(weights.shape[0], -1)
            cosine_sim = w_flat @ w_flat.T
            norms = np.sqrt(np.sum(w_flat ** 2, axis=1))
            cosine_sim /= np.outer(norms, norms) + 1e-8

            self.direction_stats.update({
                'cosine_similarities': cosine_sim,
                'mean_orthogonality': np.mean(np.abs(cosine_sim - np.eye(len(cosine_sim)))),
                'filter_norms': norms
            })

# ------------------------------------------------------------------------------


class WeightAnalyzer:
    """Enhanced analyzer for neural network weight distributions."""

    def __init__(
            self,
            models: Dict[str, keras.Model],
            config: Optional[WeightAnalyzerConfig] = None,
            output_dir: Optional[Union[str, Path]] = "weight_analysis"
    ):
        """
        Initialize analyzer with multiple models for comparison.

        Args:
            models: Dictionary mapping model names to keras models
            config: Configuration object for analysis parameters
            output_dir: Directory to save analysis outputs
        """
        self.models = models
        self.config = config or WeightAnalyzerConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style safely
        self.config.setup_plotting_style()

        # Initialize analysis containers
        self.layer_analyses: Dict[str, Dict[str, LayerAnalysis]] = {
            name: {} for name in models.keys()
        }

        # Perform initial analysis
        self._analyze_all_models()
        logger.info("Initialized WeightAnalyzer with %d models", len(models))

    def _analyze_all_models(self) -> None:
        """Perform comprehensive analysis on all models."""
        for model_name, model in self.models.items():
            logger.info("Analyzing model: %s", model_name)
            self._analyze_model(model_name, model)

    def _analyze_model(self, model_name: str, model: keras.Model) -> None:
        """
        Analyze a single model's weights.

        Args:
            model_name: Name identifier for the model
            model: Keras model to analyze
        """
        for layer in model.layers:
            weights = layer.get_weights()
            if not weights or (
                    self.config.layer_types and
                    layer.__class__.__name__ not in self.config.layer_types
            ):
                continue

            layer_analysis = LayerAnalysis(layer.name)

            for w in weights:
                if len(w.shape) < 2 and not self.config.analyze_biases:
                    continue

                layer_analysis.add_weight_stats(w)
                layer_analysis.add_norm_stats(w)
                layer_analysis.add_direction_stats(w)

            self.layer_analyses[model_name][layer.name] = layer_analysis

    def plot_norm_distributions(
            self,
            norm_types: Optional[List[str]] = None,
            save_prefix: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot specified norm distributions for all models.

        Args:
            norm_types: List of norm types to plot ('l1', 'l2', 'rms')
            save_prefix: Prefix for saved plot files

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        norm_types = norm_types or ['l2', 'rms']
        fig, axes = plt.subplots(
            len(norm_types), 1,
            figsize=(self.config.fig_width, self.config.fig_height * len(norm_types))
        )
        if len(norm_types) == 1:
            axes = [axes]

        for ax, norm_type in zip(axes, norm_types):
            for model_name, layer_analyses in self.layer_analyses.items():
                norms = [
                    analysis.norm_stats[f'{norm_type}_norm']
                    for analysis in layer_analyses.values()
                ]
                sns.kdeplot(data=norms, label=model_name, ax=ax)

            ax.set_title(f'{norm_type.upper()} Norm Distribution')
            ax.set_xlabel(f'{norm_type.upper()} Norm')
            ax.legend()

        plt.tight_layout()
        if self.config.save_plots and save_prefix is not None and len(save_prefix) > 0:
            fig.savefig(
                self.output_dir / f"{save_prefix}_distributions.{self.config.export_format}",
                dpi=self.config.dpi
            )
        return fig

    def plot_weight_distributions(
            self,
            plot_type: str = 'histogram',
            save_prefix: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot weight value distributions for all models.

        Args:
            plot_type: Type of plot ('histogram' or 'violin')
            save_prefix: Prefix for saved plot files

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, axes = plt.subplots(
            len(self.models), 1,
            figsize=(self.config.fig_width, self.config.fig_height * len(self.models))
        )
        if len(self.models) == 1:
            axes = [axes]

        for ax, (model_name, layer_analyses) in zip(axes, self.layer_analyses.items()):
            all_weights = []
            labels = []

            for layer_name, analysis in layer_analyses.items():
                weights = analysis.weight_stats['mean']
                all_weights.extend([weights])
                labels.extend([layer_name])

            if plot_type == 'histogram':
                sns.histplot(data=all_weights, stat='density', kde=True, ax=ax)
            else:  # violin plot
                sns.violinplot(data=all_weights, ax=ax)

            ax.set_title(f'{model_name} Weight Distribution')
            ax.set_xlabel('Weight Value')

        plt.tight_layout()
        if self.config.save_plots and save_prefix is not None and len(save_prefix) > 0:
            fig.savefig(
                self.output_dir / f"{save_prefix}_{plot_type}.{self.config.export_format}",
                dpi=self.config.dpi
            )
        return fig

    def plot_layer_comparisons(self, save_prefix: Optional[str] = None) -> plt.Figure:
        """
        Plot layer-wise comparisons across models.

        Args:
            save_prefix: Prefix for saved plot files

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        metrics = ['mean', 'std', 'l2_norm', 'mean_orthogonality']
        fig, axes = plt.subplots(
            len(metrics), 1,
            figsize=(self.config.fig_width, self.config.fig_height * len(metrics))
        )

        for ax, metric in zip(axes, metrics):
            data = []
            model_names = []
            layer_names = []

            for model_name, layer_analyses in self.layer_analyses.items():
                for layer_name, analysis in layer_analyses.items():
                    if metric in analysis.weight_stats:
                        value = analysis.weight_stats[metric]
                    elif metric in analysis.norm_stats:
                        value = analysis.norm_stats[metric]
                    elif metric in analysis.direction_stats:
                        value = analysis.direction_stats[metric]
                    else:
                        continue

                    data.append(value)
                    model_names.append(model_name)
                    layer_names.append(layer_name)

            df = pd.DataFrame({
                'Model': model_names,
                'Layer': layer_names,
                'Value': data
            })

            sns.barplot(data=df, x='Layer', y='Value', hue='Model', ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()} by Layer')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if self.config.save_plots and save_prefix is not None and len(save_prefix) > 0:
            fig.savefig(
                self.output_dir / f"{save_prefix}.{self.config.export_format}",
                dpi=self.config.dpi
            )
        return fig

    def compute_statistical_tests(self) -> Dict[str, Dict[str, float]]:
        """Perform statistical tests comparing weight distributions."""
        results = {}

        for metric in ['mean', 'std', 'l2_norm']:
            model_data = {
                model_name: [
                    analysis.weight_stats.get(metric, analysis.norm_stats.get(metric))
                    for analysis in layer_analyses.values()
                ]
                for model_name, layer_analyses in self.layer_analyses.items()
            }

            # Perform Kruskal-Wallis H-test
            h_stat, p_value = stats.kruskal(*model_data.values())

            results[f'{metric}_kruskal'] = {
                'statistic': float(h_stat),
                'p_value': float(p_value)
            }

        return results

    def save_analysis_results(self, filename: str = "analysis_results") -> None:
        """
        Save analysis results to JSON file.

        Args:
            filename: Base filename for saving results
        """
        if not self.config.save_stats:
            return

        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_models': len(self.models),
                'config': self.config.__dict__
            },
            'statistical_tests': self.compute_statistical_tests(),
            'model_summaries': {}
        }

        for model_name, layer_analyses in self.layer_analyses.items():
            model_results = {}

            for layer_name, analysis in layer_analyses.items():
                layer_results = {
                    'weight_stats': {
                        k: float(v) for k, v in analysis.weight_stats.items()
                    },
                    'norm_stats': {
                        k: float(v) for k, v in analysis.norm_stats.items()
                    }
                }
                model_results[layer_name] = layer_results

            results['model_summaries'][model_name] = model_results

        with open(self.output_dir / f"{filename}.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Saved analysis results to %s", filename)


# ------------------------------------------------------------------------------


class WeightAnalysisReport:
    """Generate comprehensive PDF reports of weight analysis results."""

    def __init__(
            self,
            analyzer: WeightAnalyzer,
            output_file: str = "weight_analysis_report.pdf"
    ):
        """
        Initialize report generator.

        Args:
            analyzer: Completed WeightAnalyzer instance
            output_file: Path to save the PDF report
        """
        self.analyzer = analyzer
        self.output_file = output_file

    def generate_report(self) -> None:
        """Generate and save the analysis report."""
        try:
            import matplotlib.backends.backend_pdf as pdf_backend

            with pdf_backend.PdfPages(self.output_file) as pdf:
                # Title page
                fig = plt.figure(figsize=(11, 8.5))
                fig.text(
                    0.5, 0.5,
                    "Neural Network Weight Analysis Report",
                    ha='center', va='center', fontsize=24
                )
                fig.text(
                    0.5, 0.4,
                    f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    ha='center', va='center', fontsize=14
                )
                pdf.savefig(fig)
                plt.close(fig)

                # Plot norm distributions
                fig = self.analyzer.plot_norm_distributions()
                pdf.savefig(fig)
                plt.close(fig)

                # Plot weight distributions
                fig = self.analyzer.plot_weight_distributions()
                pdf.savefig(fig)
                plt.close(fig)

                # Plot layer comparisons
                fig = self.analyzer.plot_layer_comparisons()
                pdf.savefig(fig)
                plt.close(fig)

                # Statistical summary
                fig = plt.figure(figsize=(11, 8.5))
                test_results = self.analyzer.compute_statistical_tests()
                text = "Statistical Analysis Summary\n\n"

                for test_name, result in test_results.items():
                    text += f"{test_name}:\n"
                    text += f"  Statistic: {result['statistic']:.4f}\n"
                    text += f"  p-value: {result['p_value']:.4f}\n\n"

                fig.text(0.1, 0.9, text, fontsize=12, va='top')
                pdf.savefig(fig)
                plt.close(fig)

            logger.info("Generated analysis report: %s", self.output_file)

        except Exception as e:
            logger.error("Error generating report: %s", str(e))
            raise
