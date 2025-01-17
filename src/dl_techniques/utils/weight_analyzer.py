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
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .logger import logger

@dataclass
class AnalysisConfig:
    """Configuration for weight analysis parameters."""

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
    plot_style: str = 'seaborn'
    color_palette: str = 'husl'
    fig_width: int = 12
    fig_height: int = 8
    dpi: int = 300

    # Export options
    save_plots: bool = True
    save_stats: bool = True
    export_format: str = 'png'


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


class WeightAnalyzer:
    """Enhanced analyzer for neural network weight distributions."""

    def __init__(
            self,
            models: Dict[str, tf.keras.Model],
            config: Optional[AnalysisConfig] = None,
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
        self.config = config or AnalysisConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plot style
        plt.style.use(self.config.plot_style)
        sns.set_palette(self.config.color_palette)

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

    def _analyze_model(self, model_name: str, model: tf.keras.Model) -> None:
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
            save_prefix: str = "norms"
    ) -> None:
        """
        Plot specified norm distributions for all models.

        Args:
            norm_types: List of norm types to plot ('l1', 'l2', 'rms')
            save_prefix: Prefix for saved plot files
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
        if self.config.save_plots:
            plt.savefig(
                self.output_dir / f"{save_prefix}_distributions.{self.config.export_format}",
                dpi=self.config.dpi
            )
        plt.close()

    def plot_weight_distributions(
            self,
            plot_type: str = 'histogram',
            save_prefix: str = "weights"
    ) -> None:
        """
        Plot weight value distributions for all models.

        Args:
            plot_type: Type of plot ('histogram' or 'violin')
            save_prefix: Prefix for saved plot files
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
        if self.config.save_plots:
            plt.savefig(
                self.output_dir / f"{save_prefix}_{plot_type}.{self.config.export_format}",
                dpi=self.config.dpi
            )
        plt.close()

    def plot_layer_comparisons(self, save_prefix: str = "layer_comparison") -> None:
        """Plot layer-wise comparisons across models."""
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
        if self.config.save_plots:
            plt.savefig(
                self.output_dir / f"{save_prefix}.{self.config.export_format}",
                dpi=self.config.dpi
            )
        plt.close()

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


def analyze_models(
        models: Dict[str, tf.keras.Model],
        config: Optional[AnalysisConfig] = None,
        output_dir: str = "weight_analysis"
) -> WeightAnalyzer:
    """
    Perform comprehensive weight analysis on multiple models.

    Args:
        models: Dictionary mapping model names to keras models
        config: Configuration object for analysis parameters
        output_dir: Directory to save analysis outputs

    Returns:
        WeightAnalyzer instance with completed analysis
    """
    analyzer = WeightAnalyzer(models, config, output_dir)

    # Generate standard plots
    analyzer.plot_norm_distributions()
    analyzer.plot_weight_distributions()
    analyzer.plot_layer_comparisons()

    # Save analysis results
    analyzer.save_analysis_results()

    logger.info("Completed model analysis")
    return analyzer


if __name__ == "__main__":
    # Example usage with custom configuration
    config = AnalysisConfig(
        compute_l1_norm=True,
        compute_rms_norm=True,
        analyze_biases=True,
        layer_types=['Dense', 'Conv2D'],
        plot_style='seaborn-darkgrid',
        color_palette='deep',
        save_stats=True,
        export_format='pdf'
    )

    try:
        # Load example models
        baseline_model = tf.keras.models.load_model("baseline.keras")
        rms_model = tf.keras.models.load_model("rms_norm.keras")
        logit_model = tf.keras.models.load_model("logit_norm.keras")

        models = {
            "Baseline": baseline_model,
            "RMSNorm": rms_model,
            "LogitNorm": logit_model
        }

        # Perform analysis
        analyzer = analyze_models(
            models=models,
            config=config,
            output_dir="weight_analysis_results"
        )

        # Additional custom analysis examples

        # 1. Layer-specific analysis
        analyzer.plot_layer_comparisons(save_prefix="detailed_layer_comparison")

        # 2. Custom norm distribution analysis
        analyzer.plot_norm_distributions(
            norm_types=['l1', 'l2', 'rms', 'max'],
            save_prefix="comprehensive_norms"
        )

        # 3. Different weight distribution visualization
        analyzer.plot_weight_distributions(
            plot_type='violin',
            save_prefix="weight_violin_plots"
        )

        # 4. Get statistical test results
        test_results = analyzer.compute_statistical_tests()
        print("\nStatistical Test Results:")
        print("========================")
        for test_name, result in test_results.items():
            print(f"{test_name}:")
            print(f"  Statistic: {result['statistic']:.4f}")
            print(f"  p-value: {result['p_value']:.4f}")

    except FileNotFoundError as e:
        logger.error("Model file not found: %s", str(e))
    except Exception as e:
        logger.error("Error during analysis: %s", str(e))
        raise


def plot_weight_evolution(
        model_checkpoints: Dict[str, List[tf.keras.Model]],
        config: Optional[AnalysisConfig] = None,
        output_dir: str = "weight_evolution"
) -> None:
    """
    Analyze weight distribution evolution across training checkpoints.

    Args:
        model_checkpoints: Dictionary mapping model names to lists of checkpoints
        config: Configuration for analysis
        output_dir: Directory to save evolution analysis
    """
    if config is None:
        config = AnalysisConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track metrics across checkpoints
    metrics = {
        'l2_norms': {},
        'weight_means': {},
        'weight_stds': {},
        'orthogonality': {}
    }

    # Analyze each checkpoint
    for model_name, checkpoints in model_checkpoints.items():
        metrics['l2_norms'][model_name] = []
        metrics['weight_means'][model_name] = []
        metrics['weight_stds'][model_name] = []
        metrics['orthogonality'][model_name] = []

        for checkpoint in checkpoints:
            analyzer = WeightAnalyzer(
                models={model_name: checkpoint},
                config=config,
                output_dir=output_dir / "temp"
            )

            # Aggregate metrics across layers
            model_metrics = analyzer.layer_analyses[model_name]
            metrics['l2_norms'][model_name].append(
                np.mean([
                    layer.norm_stats['l2_norm']
                    for layer in model_metrics.values()
                ])
            )
            metrics['weight_means'][model_name].append(
                np.mean([
                    layer.weight_stats['mean']
                    for layer in model_metrics.values()
                ])
            )
            metrics['weight_stds'][model_name].append(
                np.mean([
                    layer.weight_stats['std']
                    for layer in model_metrics.values()
                ])
            )
            metrics['orthogonality'][model_name].append(
                np.mean([
                    layer.direction_stats.get('mean_orthogonality', 0)
                    for layer in model_metrics.values()
                ])
            )

    # Plot evolution of metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    checkpoint_indices = range(len(next(iter(model_checkpoints.values()))))

    for (metric_name, metric_values), ax in zip(metrics.items(), axes.flat):
        for model_name, values in metric_values.items():
            ax.plot(checkpoint_indices, values, label=model_name, marker='o')

        ax.set_title(f'{metric_name.replace("_", " ").title()} Evolution')
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"weight_evolution.{config.export_format}",
        dpi=config.dpi
    )
    plt.close()


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

            pdf = pdf_backend.PdfPages(self.output_file)

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
            plt.close()

            # Add all standard plots
            self.analyzer.plot_norm_distributions()
            pdf.savefig()

            self.analyzer.plot_weight_distributions()
            pdf.savefig()

            self.analyzer.plot_layer_comparisons()
            pdf.savefig()

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
            plt.close()

            pdf.close()
            logger.info("Generated analysis report: %s", self.output_file)

        except Exception as e:
            logger.error("Error generating report: %s", str(e))
            raise
