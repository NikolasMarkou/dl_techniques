"""
Model Analyzer Main Module
============================================================================

Main coordinator class that orchestrates all analysis and visualization components.
"""

import json
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Set

# Local imports
from .config import AnalysisConfig
from .data_types import DataInput, AnalysisResults
from .utils import find_pareto_front, normalize_metric

# Analyzers
from .analyzers.weight_analyzer import WeightAnalyzer
from .analyzers.calibration_analyzer import CalibrationAnalyzer
from .analyzers.information_flow_analyzer import InformationFlowAnalyzer
from .analyzers.training_dynamics_analyzer import TrainingDynamicsAnalyzer

# Visualizers
from .visualizers.weight_visualizer import WeightVisualizer
from .visualizers.calibration_visualizer import CalibrationVisualizer
from .visualizers.information_flow_visualizer import InformationFlowVisualizer
from .visualizers.training_dynamics_visualizer import TrainingDynamicsVisualizer
from .visualizers.summary_visualizer import SummaryVisualizer

from dl_techniques.utils.logger import logger


class ModelAnalyzer:
    """
    Model analyzer with training dynamics and improved visualizations.

    This is the main coordinator class that orchestrates all analysis and
    visualization components.
    """

    def __init__(
        self,
        models: Dict[str, keras.Model],
        config: Optional[AnalysisConfig] = None,
        output_dir: Optional[Union[str, Path]] = None,
        training_history: Optional[Dict[str, Dict[str, List[float]]]] = None
    ):
        """
        Initialize the analyzer.

        Args:
            models: Dictionary mapping model names to Keras models
            config: Analysis configuration
            output_dir: Output directory for plots and results
            training_history: Optional training history for each model
        """
        if not models:
            raise ValueError("At least one model must be provided")

        self.models = models
        self.config = config or AnalysisConfig()
        self.output_dir = Path(output_dir or f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        self.config.setup_plotting_style()

        # Initialize results container
        self.results = AnalysisResults(config=self.config)

        # Store training history if provided
        if training_history:
            self.results.training_history = training_history

        # Cache for model outputs
        self._prediction_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # Initialize model colors for consistent visualization
        self.model_colors: Dict[str, str] = {}
        self._setup_model_colors()

        # Initialize analyzers
        self._init_analyzers()

        logger.info(f"ModelAnalyzer initialized with {len(models)} models")

    def _setup_model_colors(self) -> None:
        """Set up consistent colors for models across all visualizations."""
        # Define consistent colors for models (avoiding yellow)
        model_names = sorted(self.models.keys())  # Sort for consistency
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(model_names)]
        self.model_colors = dict(zip(model_names, color_palette))

    def _init_analyzers(self) -> None:
        """Initialize all analyzer instances."""
        self.analyzers = {
            'weights': WeightAnalyzer(self.models, self.config),
            'calibration': CalibrationAnalyzer(self.models, self.config),
            'information_flow': InformationFlowAnalyzer(self.models, self.config),
            'training_dynamics': TrainingDynamicsAnalyzer(self.models, self.config)
        }

    def analyze(
        self,
        data: Optional[Union[DataInput, tuple, Any]] = None,
        analysis_types: Optional[Set[str]] = None
    ) -> AnalysisResults:
        """
        Run comprehensive or selected analyses on models.

        Args:
            data: Input data
            analysis_types: Set of analysis types to run

        Returns:
            AnalysisResults object containing all results
        """
        if analysis_types is None:
            analysis_types = {
                'weights' if self.config.analyze_weights else None,
                'calibration' if self.config.analyze_calibration else None,
                'information_flow' if self.config.analyze_information_flow else None,
                'training_dynamics' if self.config.analyze_training_dynamics else None,
            }
            analysis_types.discard(None)

        # Validate data requirement
        data_required = {'calibration', 'information_flow'}
        if analysis_types & data_required and data is None:
            raise ValueError(f"Data is required for: {analysis_types & data_required}")

        logger.info(f"Running analyses: {analysis_types}")

        # Convert data to standard format
        if data is not None:
            if isinstance(data, tuple):
                data = DataInput.from_tuple(data)
            elif not isinstance(data, DataInput):
                data = DataInput.from_object(data)

        # Cache predictions if needed
        if data is not None and analysis_types & data_required:
            self._cache_predictions(data)

        # Run selected analyses
        for analysis_type in analysis_types:
            if analysis_type in self.analyzers:
                analyzer = self.analyzers[analysis_type]

                # Check if analyzer requires data
                if analyzer.requires_data() and data is None:
                    logger.warning(f"Skipping {analysis_type} analysis - requires data")
                    continue

                # Run analysis
                analyzer.analyze(self.results, data, self._prediction_cache)

        # Create visualizations
        self._create_visualizations(analysis_types)

        # Create summary dashboard
        self.create_summary_dashboard()

        # Save results
        self.save_results()

        return self.results

    def _cache_predictions(self, data: DataInput) -> None:
        """Cache model predictions to avoid redundant computation."""
        x_data, y_data = data.x_data, data.y_data

        # Sample if needed
        if len(x_data) > self.config.n_samples:
            indices = np.random.choice(len(x_data), self.config.n_samples, replace=False)
            x_data = x_data[indices]
            y_data = y_data[indices]

        logger.info("Caching model predictions...")
        for model_name, model in tqdm(self.models.items(), desc="Computing predictions"):
            predictions = model.predict(x_data, verbose=0)
            self._prediction_cache[model_name] = {
                'predictions': predictions,
                'x_data': x_data,
                'y_data': y_data
            }

        # Also evaluate models with better metric handling
        for model_name, model in self.models.items():
            try:
                metrics = model.evaluate(x_data, y_data, verbose=0)
                metric_names = getattr(model, 'metrics_names', ['loss'])

                # Handle different metric name patterns
                metric_dict = {}
                for i, name in enumerate(metric_names):
                    if i < len(metrics):
                        value = metrics[i] if isinstance(metrics, (list, tuple)) else metrics
                        metric_dict[name] = value

                        # Add common aliases for accuracy
                        if name in ['compile_metrics', 'categorical_accuracy', 'sparse_categorical_accuracy']:
                            metric_dict['accuracy'] = value

                self.results.model_metrics[model_name] = metric_dict
            except Exception as e:
                logger.warning(f"Could not evaluate model {model_name}: {e}")
                self.results.model_metrics[model_name] = {'loss': 0.0, 'accuracy': 0.0}

    def _create_visualizations(self, analysis_types: Set[str]) -> None:
        """Create visualizations for completed analyses."""
        visualizers = {
            'weights': WeightVisualizer,
            'calibration': CalibrationVisualizer,
            'information_flow': InformationFlowVisualizer,
            'training_dynamics': TrainingDynamicsVisualizer
        }

        for analysis_type in analysis_types:
            if analysis_type in visualizers:
                visualizer_class = visualizers[analysis_type]
                visualizer = visualizer_class(
                    self.results, self.config, self.output_dir, self.model_colors
                )
                visualizer.create_visualizations()

    def create_summary_dashboard(self) -> None:
        """Create summary dashboard visualization."""
        visualizer = SummaryVisualizer(
            self.results, self.config, self.output_dir, self.model_colors
        )
        visualizer.create_visualizations()

    def save_results(self, filename: str = "analysis_results.json") -> None:
        """Save analysis results to JSON file."""
        results_dict = {
            'timestamp': self.results.analysis_timestamp,
            'config': self.config.__dict__,
            'model_metrics': self.results.model_metrics,
            'weight_stats': self.results.weight_stats,
            'calibration_metrics': self.results.calibration_metrics,
            'training_metrics': self._serialize_training_metrics() if self.results.training_metrics else None,
        }

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, dict):
                # Use non-serializable fields from results
                skip_fields = getattr(self.results, '_non_serializable_fields', set())
                return {k: convert_numpy(v) for k, v in obj.items()
                       if k not in skip_fields}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        results_dict = convert_numpy(results_dict)

        try:
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2)
            logger.info(f"Saved results to: {filepath}")
        except Exception as e:
            logger.error(f"Could not save results: {e}")

    def _serialize_training_metrics(self) -> Dict[str, Any]:
        """Serialize training metrics for JSON storage."""
        metrics = self.results.training_metrics
        return {
            'epochs_to_convergence': metrics.epochs_to_convergence,
            'training_stability_score': metrics.training_stability_score,
            'overfitting_index': metrics.overfitting_index,
            'peak_performance': metrics.peak_performance,
            'final_gap': metrics.final_gap,
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the analysis."""
        summary = {
            'n_models': len(self.models),
            'analyses_performed': [],
            'model_performance': {},
            'calibration_summary': {},
            'weight_summary': {},
            'training_summary': {}
        }

        # Check which analyses were performed
        if any(self.results.weight_stats.values()):
            summary['analyses_performed'].append('weight_analysis')
        if self.results.calibration_metrics:
            summary['analyses_performed'].append('confidence_calibration_analysis')
        if self.results.information_flow:
            summary['analyses_performed'].append('information_flow_analysis')
        if self.results.training_metrics:
            summary['analyses_performed'].append('training_dynamics_analysis')

        # Add summaries
        for model_name, metrics in self.results.model_metrics.items():
            summary['model_performance'][model_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'loss': metrics.get('loss', 0)
            }

        for model_name, metrics in self.results.calibration_metrics.items():
            summary['calibration_summary'][model_name] = {
                'ece': metrics.get('ece', 0),
                'brier_score': metrics.get('brier_score', 0),
                'mean_entropy': metrics.get('mean_entropy', 0)
            }

        for model_name, weight_stats in self.results.weight_stats.items():
            n_params = sum(np.prod(s['shape']) for s in weight_stats.values())
            summary['weight_summary'][model_name] = {
                'total_parameters': n_params,
                'n_weight_tensors': len(weight_stats)
            }

        # Add training summary if available
        if self.results.training_metrics:
            for model_name in self.models:
                summary['training_summary'][model_name] = {
                    'epochs_to_convergence': self.results.training_metrics.epochs_to_convergence.get(model_name, 0),
                    'overfitting_index': self.results.training_metrics.overfitting_index.get(model_name, 0),
                    'peak_accuracy': self.results.training_metrics.peak_performance.get(
                        model_name, {}).get('val_accuracy', 0)
                }

        return summary

    def create_pareto_analysis(self, save_plot: bool = True) -> Optional[plt.Figure]:
        """Create Pareto front analysis for hyperparameter sweep scenarios.

        This is particularly useful when analyzing many models (>10) to identify
        the Pareto-optimal ones that balance performance vs overfitting.

        Returns:
            Figure object if successful, None otherwise
        """
        import matplotlib.pyplot as plt

        if not self.results.training_metrics or not self.results.training_metrics.peak_performance:
            logger.warning("No training metrics available for Pareto analysis")
            return None

        # Prepare data
        models = []
        peak_accuracies = []
        overfitting_indices = []
        convergence_speeds = []

        for model_name in sorted(self.models.keys()):
            if model_name in self.results.training_metrics.peak_performance:
                models.append(model_name)
                peak_accuracies.append(
                    self.results.training_metrics.peak_performance[model_name].get('val_accuracy', 0)
                )
                overfitting_indices.append(
                    self.results.training_metrics.overfitting_index.get(model_name, 0)
                )
                convergence_speeds.append(
                    self.results.training_metrics.epochs_to_convergence.get(model_name, 0)
                )

        if len(models) < 2:
            logger.warning("Need at least 2 models for Pareto analysis")
            return None

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Peak Accuracy vs Overfitting Index
        scatter = ax1.scatter(overfitting_indices, peak_accuracies,
                             c=convergence_speeds, s=100, alpha=0.7,
                             cmap='viridis', edgecolors='black', linewidth=1)

        # Find Pareto front
        pareto_indices = find_pareto_front(
            np.array(overfitting_indices) * -1,  # Minimize overfitting (negate for maximization)
            np.array(peak_accuracies)  # Maximize accuracy
        )

        # Highlight Pareto optimal models
        pareto_x = [overfitting_indices[i] for i in pareto_indices]
        pareto_y = [peak_accuracies[i] for i in pareto_indices]
        ax1.plot(pareto_x, pareto_y, 'r--', alpha=0.5, linewidth=2)
        ax1.scatter(pareto_x, pareto_y, color='red', s=200, marker='*',
                   edgecolors='black', linewidth=2, zorder=5, label='Pareto Optimal')

        # Add labels for Pareto optimal models
        for idx in pareto_indices:
            ax1.annotate(models[idx], (overfitting_indices[idx], peak_accuracies[idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax1.set_xlabel('Overfitting Index (lower is better)')
        ax1.set_ylabel('Peak Validation Accuracy')
        ax1.set_title('Pareto Front: Accuracy vs Overfitting')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add colorbar
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Convergence Speed (epochs)', rotation=270, labelpad=20)

        # Plot 2: Model ranking heatmap
        metrics_matrix = np.array([
            normalize_metric(peak_accuracies, higher_better=True),
            normalize_metric(overfitting_indices, higher_better=False),
            normalize_metric(convergence_speeds, higher_better=False)
        ])

        im = ax2.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(['Peak Accuracy', 'Low Overfitting', 'Fast Convergence'])
        ax2.set_title('Normalized Performance Metrics')

        # Add text annotations
        for i in range(3):
            for j in range(len(models)):
                text = ax2.text(j, i, f'{metrics_matrix[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=8)

        # Add colorbar
        cbar2 = plt.colorbar(im, ax=ax2)
        cbar2.set_label('Normalized Score', rotation=270, labelpad=20)

        plt.suptitle('Pareto Analysis for Model Selection', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_plot and self.config.save_plots:
            self._save_figure(fig, 'pareto_analysis')

        return fig

    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure with configured settings."""
        try:
            filepath = self.output_dir / f"{name}.{self.config.save_format}"
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none', pad_inches=0.1)
            logger.info(f"Saved plot: {filepath}")
        except Exception as e:
            logger.error(f"Could not save figure {name}: {e}")