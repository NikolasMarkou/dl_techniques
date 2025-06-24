"""
Model Analyzer for Neural Networks
==============================================

A comprehensive, modular analyzer that combines activation, weight, calibration,
and probability distribution analysis into a single, easy-to-use interface.

Key Features:
- Unified configuration system
- Modular analysis components
- Consistent visualization style
- Minimal code duplication
- Clean, publication-ready plots
- Flexible layer selection
- Improved error handling

Example Usage:
    ```python
    from analyzer import ModelAnalyzer, AnalysisConfig

    # Configure analysis
    config = AnalysisConfig(
        analyze_activations=True,
        analyze_weights=True,
        analyze_calibration=True,
        plot_style='publication',
        activation_layer_name='conv2d_1'  # Optional: specify layer
    )

    # Create analyzer
    analyzer = ModelAnalyzer(models, config=config)

    # Run comprehensive analysis
    results = analyzer.analyze(test_data)

    # Or run specific analyses
    analyzer.analyze_weights()
    analyzer.analyze_calibration(test_data)
    ```
"""

import json
import keras
import scipy
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Union, Any, Tuple, Set, NamedTuple


# ------------------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------------------

from .logger import logger
from .calibration_metrics import (
    compute_ece,
    compute_brier_score,
    compute_reliability_data,
    compute_prediction_entropy_stats
)

# ------------------------------------------------------------------------------
# Data Type Definitions
# ------------------------------------------------------------------------------

class DataInput(NamedTuple):
    """Structured data input type."""
    x_data: np.ndarray
    y_data: np.ndarray

    @classmethod
    def from_tuple(cls, data: Tuple[np.ndarray, np.ndarray]) -> 'DataInput':
        """Create from tuple."""
        return cls(x_data=data[0], y_data=data[1])

    @classmethod
    def from_object(cls, data: Any) -> 'DataInput':
        """Create from object with x_test and y_test attributes."""
        return cls(x_data=data.x_test, y_data=data.y_test)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """configuration for all analysis types."""

    # Analysis toggles
    analyze_activations: bool = True
    analyze_weights: bool = True
    analyze_calibration: bool = True
    analyze_probability_distributions: bool = True
    analyze_information_flow: bool = True

    # Sampling parameters
    n_samples: int = 1000
    n_samples_per_digit: int = 3
    sample_digits: Optional[List[int]] = None

    # Layer selection
    activation_layer_name: Optional[str] = None  # Specify layer for activation analysis
    activation_layer_index: Optional[int] = None  # Alternative: specify by index

    # Weight analysis options
    weight_layer_types: Optional[List[str]] = None
    analyze_biases: bool = False
    compute_weight_correlations: bool = True
    compute_weight_pca: bool = True

    # Calibration options
    calibration_bins: int = 10

    # Visualization settings
    plot_style: str = 'publication'  # 'publication', 'presentation', 'draft'
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

    def get_figure_size(self, scale: float = 1.0) -> Tuple[float, float]:
        """Get figure size with optional scaling."""
        return (self.fig_width * scale, self.fig_height * scale)

    def setup_plotting_style(self) -> None:
        """Set up matplotlib style based on configuration."""
        # Reset to default first
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

        # Apply selected style
        if self.plot_style in style_settings:
            plt.rcParams.update(style_settings[self.plot_style])

        # Common settings
        plt.rcParams.update({
            'figure.figsize': (self.fig_width, self.fig_height),
            'figure.dpi': 100,
            'savefig.dpi': self.dpi,
            'axes.grid': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.axisbelow': True,
            'figure.autolayout': False,  # Disabled to avoid conflicts with manual layout
        })

        # Set seaborn theme
        sns.set_theme(style='whitegrid', palette=self.color_palette)

# ------------------------------------------------------------------------------
# Base Analysis Results
# ------------------------------------------------------------------------------

@dataclass
class AnalysisResults:
    """Container for all analysis results."""

    # Model performance
    model_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Activation analysis
    activation_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Weight analysis
    weight_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    weight_correlations: Optional[pd.DataFrame] = None
    weight_pca: Optional[Dict[str, Any]] = None

    # Calibration analysis
    calibration_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reliability_data: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    # Probability analysis
    confidence_metrics: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    # Information flow
    information_flow: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Optional[AnalysisConfig] = None

# ------------------------------------------------------------------------------
# Utility Functions for Safe Plotting
# ------------------------------------------------------------------------------

def safe_set_xticklabels(ax, labels, rotation=0, max_labels=10):
    """Safely set x-tick labels with proper handling."""
    try:
        if len(labels) > max_labels:
            # Reduce number of labels if too many
            step = len(labels) // max_labels
            indices = range(0, len(labels), step)
            ax.set_xticks([i for i in indices])
            ax.set_xticklabels([labels[i] for i in indices], rotation=rotation, ha='right' if rotation > 0 else 'center')
        else:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=rotation, ha='right' if rotation > 0 else 'center')
    except Exception as e:
        logger.warning(f"Could not set x-tick labels: {e}")

def safe_tight_layout(fig, **kwargs):
    """Safely apply tight_layout with error handling."""
    try:
        fig.tight_layout(**kwargs)
    except Exception as e:
        logger.warning(f"Could not apply tight_layout: {e}")
        # Try alternative layout adjustment
        try:
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
        except Exception:
            pass

# ------------------------------------------------------------------------------
# Model Analyzer
# ------------------------------------------------------------------------------

class ModelAnalyzer:
    """
    Model analyzer for comprehensive neural network model analysis.

    This class combines activation, weight, calibration, and probability
    distribution analysis into a single, configurable interface.

    Attributes:
        models: Dictionary of models to analyze
        config: Analysis configuration
        output_dir: Directory for saving outputs
        results: Container for all analysis results
    """

    def __init__(
        self,
        models: Dict[str, keras.Model],
        config: Optional[AnalysisConfig] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the analyzer.

        Args:
            models: Dictionary mapping model names to Keras models
            config: Analysis configuration (uses defaults if None)
            output_dir: Output directory for plots and results
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

        # Cache for model outputs
        self._prediction_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # Set up activation extraction models
        self._setup_activation_models()

        logger.info(f"ModelAnalyzer initialized with {len(models)} models")

    def _setup_activation_models(self) -> None:
        """Set up models for extracting intermediate activations."""
        self.activation_models = {}
        self.layer_extraction_models = {}

        for model_name, model in self.models.items():
            # Find best layer for visualization
            best_viz_layer = self._find_visualization_layer(model)

            if best_viz_layer is not None:
                try:
                    self.activation_models[model_name] = keras.Model(
                        inputs=model.input,
                        outputs=[best_viz_layer.output, model.output]
                    )
                except Exception as e:
                    logger.warning(f"Could not create activation model for {model_name}: {e}")

            # Set up multi-layer extraction for information flow
            extraction_layers = self._get_extraction_layers(model)
            if extraction_layers:
                try:
                    self.layer_extraction_models[model_name] = {
                        'model': keras.Model(inputs=model.input, outputs=[l['output'] for l in extraction_layers]),
                        'layer_info': extraction_layers
                    }
                except Exception as e:
                    logger.warning(f"Could not create layer extraction model for {model_name}: {e}")

    def _find_visualization_layer(self, model: keras.Model) -> Optional[keras.layers.Layer]:
        """Find the layer for activation visualization based on config."""
        # Check if specific layer is requested
        if self.config.activation_layer_name:
            try:
                return model.get_layer(self.config.activation_layer_name)
            except:
                logger.warning(f"Layer '{self.config.activation_layer_name}' not found, using default selection")

        if self.config.activation_layer_index is not None:
            try:
                return model.layers[self.config.activation_layer_index]
            except IndexError:
                logger.warning(f"Layer index {self.config.activation_layer_index} out of range, using default selection")

        # Default heuristic
        return self._find_best_visualization_layer(model)

    def _find_best_visualization_layer(self, model: keras.Model) -> Optional[keras.layers.Layer]:
        """Find the best layer for activation visualization using heuristics."""
        # Priority: Conv2D > Dense > any layer with meaningful output
        conv_layers = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)]
        if conv_layers:
            return conv_layers[-1]

        dense_layers = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
        if dense_layers:
            return dense_layers[len(dense_layers) // 2]

        # Fallback to any layer with output
        for layer in reversed(model.layers):
            if hasattr(layer, 'output') and layer.output is not None:
                return layer

        return None

    def _get_extraction_layers(self, model: keras.Model) -> List[Dict[str, Any]]:
        """Get layers suitable for information flow analysis."""
        extraction_layers = []

        for layer in model.layers:
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense,
                                keras.layers.BatchNormalization, keras.layers.LayerNormalization)):
                if hasattr(layer, 'output') and layer.output is not None:
                    extraction_layers.append({
                        'name': layer.name,
                        'type': layer.__class__.__name__,
                        'output': layer.output
                    })

        return extraction_layers

    # ------------------------------------------------------------------------------
    # Main Analysis Interface
    # ------------------------------------------------------------------------------

    def analyze(
        self,
        data: Optional[Union[DataInput, Tuple[np.ndarray, np.ndarray], Any]] = None,
        analysis_types: Optional[Set[str]] = None
    ) -> AnalysisResults:
        """
        Run comprehensive or selected analyses on models.

        Args:
            data: Input data - can be DataInput, tuple (x, y), or object with x_test/y_test
            analysis_types: Set of analysis types to run. If None, runs all enabled analyses.
                          Options: {'activations', 'weights', 'calibration', 'probability', 'information_flow'}

        Returns:
            AnalysisResults object containing all results
        """
        if analysis_types is None:
            analysis_types = {
                'activations' if self.config.analyze_activations else None,
                'weights' if self.config.analyze_weights else None,
                'calibration' if self.config.analyze_calibration else None,
                'probability' if self.config.analyze_probability_distributions else None,
                'information_flow' if self.config.analyze_information_flow else None,
            }
            analysis_types.discard(None)

        # Validate data requirement
        data_required = {'activations', 'calibration', 'probability', 'information_flow'}
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
        if 'weights' in analysis_types:
            self.analyze_weights()

        if 'activations' in analysis_types and data is not None:
            self.analyze_activations(data)

        if 'calibration' in analysis_types and data is not None:
            self.analyze_calibration(data)

        if 'probability' in analysis_types and data is not None:
            self.analyze_probability_distributions(data)

        if 'information_flow' in analysis_types and data is not None:
            self.analyze_information_flow(data)

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

        # Also evaluate models
        for model_name, model in self.models.items():
            metrics = model.evaluate(x_data, y_data, verbose=0)
            self.results.model_metrics[model_name] = dict(zip(model.metrics_names, metrics))

    # ------------------------------------------------------------------------------
    # Weight Analysis
    # ------------------------------------------------------------------------------

    def analyze_weights(self) -> None:
        """Analyze weight distributions across all models."""
        logger.info("Analyzing weight distributions...")

        for model_name, model in self.models.items():
            self.results.weight_stats[model_name] = {}

            for layer in model.layers:
                # Filter by layer type if specified
                if (self.config.weight_layer_types and
                    layer.__class__.__name__ not in self.config.weight_layer_types):
                    continue

                weights = layer.get_weights()
                if not weights:
                    continue

                for idx, w in enumerate(weights):
                    # Skip biases if not analyzing them
                    if len(w.shape) < 2 and not self.config.analyze_biases:
                        continue

                    weight_name = f"{layer.name}_w{idx}"
                    stats = self._compute_weight_statistics(w)
                    self.results.weight_stats[model_name][weight_name] = stats

        # Compute correlations if requested
        if self.config.compute_weight_correlations:
            self._compute_weight_correlations()

        # Compute PCA if requested
        if self.config.compute_weight_pca:
            self._compute_weight_pca()

        # Create visualizations
        self._plot_weight_analysis()

    def _compute_weight_statistics(self, weights: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive statistics for a weight tensor."""
        flat_weights = weights.flatten()

        stats = {
            'shape': weights.shape,
            'basic': {
                'mean': float(np.mean(flat_weights)),
                'std': float(np.std(flat_weights)),
                'median': float(np.median(flat_weights)),
                'min': float(np.min(flat_weights)),
                'max': float(np.max(flat_weights)),
                'skewness': float(scipy.stats.skew(flat_weights)),
                'kurtosis': float(scipy.stats.kurtosis(flat_weights)),
            },
            'norms': {
                'l1': float(np.sum(np.abs(weights))),
                'l2': float(np.sqrt(np.sum(weights ** 2))),
                'max': float(np.max(np.abs(weights))),
                'rms': float(np.sqrt(np.mean(weights ** 2))),
            },
            'distribution': {
                'zero_fraction': float(np.mean(np.abs(flat_weights) < 1e-6)),
                'positive_fraction': float(np.mean(flat_weights > 0)),
                'negative_fraction': float(np.mean(flat_weights < 0)),
            }
        }

        # Add spectral norm for 2D weights
        if len(weights.shape) == 2:
            try:
                stats['norms']['spectral'] = float(np.linalg.norm(weights, 2))
            except:
                stats['norms']['spectral'] = 0.0

        return stats

    def _compute_weight_correlations(self) -> None:
        """Compute correlations between model weight patterns."""
        # Aggregate features for each model
        model_features = {}

        for model_name, weight_stats in self.results.weight_stats.items():
            features = []
            for layer_stats in weight_stats.values():
                # Extract key metrics
                features.extend([
                    layer_stats['basic']['mean'],
                    layer_stats['basic']['std'],
                    layer_stats['norms']['l2'],
                    layer_stats['distribution']['zero_fraction']
                ])

            if features:
                model_features[model_name] = features

        # Create correlation matrix
        if len(model_features) >= 2:
            # Ensure all feature vectors have same length
            min_len = min(len(f) for f in model_features.values())
            aligned_features = {k: v[:min_len] for k, v in model_features.items()}

            feature_df = pd.DataFrame(aligned_features).T
            self.results.weight_correlations = feature_df.corr()

    def _compute_weight_pca(self) -> None:
        """Perform PCA analysis on weight patterns."""
        all_features = []
        labels = []

        for model_name, weight_stats in self.results.weight_stats.items():
            for layer_name, stats in weight_stats.items():
                features = [
                    stats['basic']['mean'],
                    stats['basic']['std'],
                    stats['basic']['skewness'],
                    stats['norms']['l2'],
                    stats['distribution']['zero_fraction']
                ]
                all_features.append(features)
                labels.append(f"{model_name}_{layer_name.split('_')[0]}")

        if len(all_features) >= 3:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(all_features)

            # Perform PCA
            pca = PCA(n_components=min(3, len(features_scaled[0])))
            pca_result = pca.fit_transform(features_scaled)

            self.results.weight_pca = {
                'components': pca_result,
                'explained_variance': pca.explained_variance_ratio_,
                'labels': labels
            }

    def _plot_weight_analysis(self) -> None:
        """Create weight analysis visualizations."""
        # FIXED: Use subplots_adjust instead of constrained_layout
        fig = plt.figure(figsize=self.config.get_figure_size(1.5))
        gs = plt.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.4)

        # 1. Norm distributions
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_weight_norm_distributions(ax1)

        # 2. Statistical comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_weight_statistics_comparison(ax2)

        # 3. Zero fraction analysis
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_weight_sparsity(ax3)

        # 4. Correlation matrix (if available)
        if self.results.weight_correlations is not None and not self.results.weight_correlations.empty:
            ax4 = fig.add_subplot(gs[2, 0])
            sns.heatmap(self.results.weight_correlations, annot=True, cmap='coolwarm',
                       center=0, ax=ax4, cbar_kws={'label': 'Correlation'})
            ax4.set_title('Model Weight Pattern Correlations')

        # 5. PCA plot (if available)
        if self.results.weight_pca:
            ax5 = fig.add_subplot(gs[2, 1])
            self._plot_weight_pca(ax5)

        plt.suptitle('Weight Distribution Analysis', fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'weight_analysis')
        plt.close(fig)

    def _plot_weight_norm_distributions(self, ax) -> None:
        """Plot weight norm distributions using seaborn."""
        norm_data = []
        for model_name, weight_stats in self.results.weight_stats.items():
            for layer_name, stats in weight_stats.items():
                norm_data.append({
                    'Model': model_name,
                    'Layer': layer_name.split('_')[0],
                    'L2 Norm': stats['norms']['l2'],
                    'RMS Norm': stats['norms']['rms']
                })

        if norm_data:
            df = pd.DataFrame(norm_data)

            # FIXED: Use seaborn violin plot with proper handling
            try:
                sns.violinplot(data=df, x='Model', y='L2 Norm', ax=ax,
                               inner='quartile', hue='Model', palette=self.config.color_palette, legend=False)

                # FIXED: Safe x-tick label rotation
                labels = [t.get_text() for t in ax.get_xticklabels()]
                if len(labels) > 3:
                    safe_set_xticklabels(ax, labels, rotation=45)

            except Exception as e:
                logger.warning(f"Could not create violin plot: {e}")
                # Fallback to simple bar plot
                model_means = df.groupby('Model')['L2 Norm'].mean()
                ax.bar(range(len(model_means)), model_means.values)
                safe_set_xticklabels(ax, model_means.index.tolist())

            ax.set_ylabel('L2 Norm')
            ax.set_title('Weight Norm Distributions by Model')
            ax.grid(True, alpha=0.3)

    def _plot_weight_statistics_comparison(self, ax) -> None:
        """Plot comparison of weight statistics."""
        stats_data = []
        for model_name, weight_stats in self.results.weight_stats.items():
            mean_vals = [s['basic']['mean'] for s in weight_stats.values()]
            std_vals = [s['basic']['std'] for s in weight_stats.values()]

            if mean_vals:
                stats_data.append({
                    'Model': model_name,
                    'Avg Mean': np.mean(mean_vals),
                    'Avg Std': np.mean(std_vals),
                    'Mean Std': np.std(mean_vals),
                    'Std Std': np.std(std_vals)
                })

        if stats_data:
            df = pd.DataFrame(stats_data)

            # Create grouped bar plot
            x = np.arange(len(df))
            width = 0.35

            ax.bar(x - width/2, df['Avg Mean'], width, label='Mean',
                  yerr=df['Mean Std'], capsize=5, alpha=0.8)
            ax.bar(x + width/2, df['Avg Std'], width, label='Std Dev',
                  yerr=df['Std Std'], capsize=5, alpha=0.8)

            ax.set_xlabel('Model')
            ax.set_ylabel('Value')
            ax.set_title('Weight Statistics Comparison')
            # FIXED: Safe x-tick handling
            safe_set_xticklabels(ax, df['Model'].tolist(), rotation=45 if len(df) > 3 else 0)
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_weight_sparsity(self, ax) -> None:
        """Plot weight sparsity analysis using seaborn."""
        sparsity_data = []
        for model_name, weight_stats in self.results.weight_stats.items():
            for layer_name, stats in weight_stats.items():
                sparsity_data.append({
                    'Model': model_name,
                    'Layer': layer_name.split('_')[0],
                    'Zero Fraction': stats['distribution']['zero_fraction']
                })

        if sparsity_data:
            df = pd.DataFrame(sparsity_data)

            # FIXED: Use seaborn box plot with proper handling
            try:
                sns.boxplot(data=df, x='Model', y='Zero Fraction', ax=ax,
                            hue='Model', palette=self.config.color_palette, legend=False)

                # FIXED: Safe x-tick label rotation
                labels = [t.get_text() for t in ax.get_xticklabels()]
                if len(labels) > 3:
                    safe_set_xticklabels(ax, labels, rotation=45)

            except Exception as e:
                logger.warning(f"Could not create box plot: {e}")
                # Fallback to simple bar plot
                model_means = df.groupby('Model')['Zero Fraction'].mean()
                ax.bar(range(len(model_means)), model_means.values)
                safe_set_xticklabels(ax, model_means.index.tolist())

            ax.set_ylabel('Zero Fraction')
            ax.set_title('Weight Sparsity by Model')
            ax.grid(True, alpha=0.3)

    def _plot_weight_pca(self, ax) -> None:
        """Plot PCA results for weights."""
        if not self.results.weight_pca:
            return

        components = self.results.weight_pca['components']
        labels = self.results.weight_pca['labels']
        explained_var = self.results.weight_pca['explained_variance']

        # Extract model names from labels
        model_names = [label.split('_')[0] for label in labels]
        unique_models = list(set(model_names))
        colors = plt.cm.Set3(np.arange(len(unique_models)))

        # Create scatter plot
        for i, model in enumerate(unique_models):
            mask = np.array([m == model for m in model_names])
            ax.scatter(components[mask, 0], components[mask, 1],
                      c=[colors[i]], label=model, alpha=0.7, s=60)

        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        ax.set_title('PCA of Weight Patterns')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------------------
    # Activation Analysis
    # ------------------------------------------------------------------------------

    def analyze_activations(self, data: DataInput) -> None:
        """Analyze activation distributions."""
        logger.info("Analyzing activation distributions...")

        for model_name in self.activation_models:
            self.results.activation_stats[model_name] = {}

            # Get activations for different layers
            activations = self._get_layer_activations(model_name, data.x_data)

            for layer_name, acts in activations.items():
                stats = self._compute_activation_statistics(acts)
                self.results.activation_stats[model_name][layer_name] = stats

        # Create visualizations
        self._plot_activation_analysis()

    def _get_layer_activations(self, model_name: str, x_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Get activations from multiple layers."""
        activations = {}

        # Use cached data if available
        if model_name in self._prediction_cache:
            x_data = self._prediction_cache[model_name]['x_data']
        else:
            # Sample if needed
            if len(x_data) > self.config.n_samples:
                indices = np.random.choice(len(x_data), self.config.n_samples, replace=False)
                x_data = x_data[indices]

        if model_name in self.activation_models:
            try:
                # Get activations (limit samples for memory efficiency)
                outputs = self.activation_models[model_name].predict(x_data[:100], verbose=0)

                if isinstance(outputs, list) and len(outputs) >= 2:
                    # First output is the visualization layer
                    activations['viz_layer'] = outputs[0]
                    # Store sample inputs for visualization
                    activations['sample_inputs'] = x_data[:10]  # Keep some samples
            except Exception as e:
                logger.warning(f"Could not get activations for {model_name}: {e}")

        return activations

    def _compute_activation_statistics(self, activations: np.ndarray) -> Dict[str, Any]:
        """Compute statistics for activation tensor."""
        flat_acts = activations.flatten()

        return {
            'shape': activations.shape,
            'mean': float(np.mean(flat_acts)),
            'std': float(np.std(flat_acts)),
            'sparsity': float(np.mean(np.abs(flat_acts) < 1e-5)),
            'positive_ratio': float(np.mean(flat_acts > 0)),
            'percentiles': {
                'p25': float(np.percentile(flat_acts, 25)),
                'p50': float(np.percentile(flat_acts, 50)),
                'p75': float(np.percentile(flat_acts, 75))
            },
            'raw_activations': activations  # Store for visualization
        }

    def _plot_activation_analysis(self) -> None:
        """Create activation analysis visualizations."""
        if not any(self.results.activation_stats.values()):
            return

        # FIXED: Use subplots_adjust instead of constrained_layout
        fig, axes = plt.subplots(2, 2, figsize=self.config.get_figure_size())
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # 1. Activation distributions
        ax1 = axes[0, 0]
        self._plot_activation_distributions(ax1)

        # 2. Sparsity comparison
        ax2 = axes[0, 1]
        self._plot_activation_sparsity(ax2)

        # 3. Sample activations (now implemented!)
        ax3 = axes[1, 0]
        self._plot_sample_activations(ax3)

        # 4. Statistics summary
        ax4 = axes[1, 1]
        self._plot_activation_summary(ax4)

        plt.suptitle('Activation Analysis', fontsize=16, fontweight='bold')

        if self.config.save_plots:
            self._save_figure(fig, 'activation_analysis')
        plt.close(fig)

    def _plot_activation_distributions(self, ax) -> None:
        """Plot activation value distributions."""
        for model_name, layer_stats in self.results.activation_stats.items():
            for layer_name, stats in layer_stats.items():
                if 'raw_activations' in stats:
                    # Use actual activation values
                    flat_acts = stats['raw_activations'].flatten()
                    # Sample for efficiency
                    if len(flat_acts) > 10000:
                        flat_acts = np.random.choice(flat_acts, 10000, replace=False)

                    ax.hist(flat_acts, bins=50, alpha=0.5, density=True,
                           label=f"{model_name}_{layer_name}", edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Density')
        ax.set_title('Activation Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_activation_sparsity(self, ax) -> None:
        """Plot activation sparsity comparison."""
        sparsity_data = []
        for model_name, layer_stats in self.results.activation_stats.items():
            for layer_name, stats in layer_stats.items():
                if layer_name != 'sample_inputs':  # Skip sample inputs
                    sparsity_data.append({
                        'Model': model_name,
                        'Layer': layer_name,
                        'Sparsity': stats['sparsity'],
                        'Positive Ratio': stats['positive_ratio']
                    })

        if sparsity_data:
            df = pd.DataFrame(sparsity_data)

            # Bar plot
            x = np.arange(len(df))
            width = 0.35

            ax.bar(x - width/2, df['Sparsity'], width, label='Sparsity', alpha=0.8)
            ax.bar(x + width/2, df['Positive Ratio'], width, label='Positive Ratio', alpha=0.8)

            ax.set_xlabel('Model/Layer')
            ax.set_ylabel('Ratio')
            ax.set_title('Activation Sparsity Analysis')
            # FIXED: Safe x-tick handling
            labels = [f"{row['Model'][:8]}\n{row['Layer'][:8]}" for _, row in df.iterrows()]
            safe_set_xticklabels(ax, labels, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_sample_activations(self, ax) -> None:
        """Plot sample activation maps - now implemented!"""
        # Find first model with Conv2D activations
        conv_activations = None
        model_name = None

        for m_name, layer_stats in self.results.activation_stats.items():
            for layer_name, stats in layer_stats.items():
                if 'raw_activations' in stats and len(stats['shape']) == 4:  # Conv layer
                    conv_activations = stats['raw_activations']
                    model_name = m_name
                    break
            if conv_activations is not None:
                break

        if conv_activations is not None:
            # Select first sample
            sample_acts = conv_activations[0]  # Shape: (height, width, channels)

            # Select up to 16 feature maps
            n_features = min(16, sample_acts.shape[-1])
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols

            # Create subgrid - FIXED: Avoid complex subplot management
            try:
                from matplotlib.gridspec import GridSpecFromSubplotSpec
                gs_sub = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=ax.get_gridspec()[1, 0],
                                                wspace=0.1, hspace=0.1)

                for i in range(n_features):
                    row = i // n_cols
                    col = i % n_cols
                    ax_sub = plt.subplot(gs_sub[row, col])

                    # Plot feature map
                    im = ax_sub.imshow(sample_acts[:, :, i], cmap='viridis', aspect='auto')
                    ax_sub.axis('off')
                    ax_sub.set_title(f'F{i}', fontsize=8)

                # Main axis adjustments
                ax.set_visible(False)
                ax.text(0.5, 1.05, f'Sample Activation Maps - {model_name}',
                       transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')
            except Exception as e:
                logger.warning(f"Could not create activation subplots: {e}")
                ax.text(0.5, 0.5, f'Activation visualization failed\n{str(e)[:50]}...',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Sample Activations')
                ax.axis('off')
        else:
            # Fallback for non-conv layers
            ax.text(0.5, 0.5, 'No convolutional activations available\nfor visualization',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sample Activations')
            ax.axis('off')

    def _plot_activation_summary(self, ax) -> None:
        """Plot activation statistics summary."""
        ax.axis('tight')
        ax.axis('off')

        # Create summary table
        summary_data = []
        for model_name, layer_stats in self.results.activation_stats.items():
            for layer_name, stats in layer_stats.items():
                if layer_name != 'sample_inputs':  # Skip sample inputs
                    summary_data.append([
                        f"{model_name[:10]}",
                        f"{layer_name[:10]}",
                        f"{stats['mean']:.3f}",
                        f"{stats['std']:.3f}",
                        f"{stats['sparsity']:.2%}"
                    ])

        if summary_data:
            table = ax.table(
                cellText=summary_data,
                colLabels=['Model', 'Layer', 'Mean', 'Std', 'Sparsity'],
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # Style header
            for i in range(5):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Activation Statistics Summary', pad=20)

    # ------------------------------------------------------------------------------
    # Calibration Analysis
    # ------------------------------------------------------------------------------

    def analyze_calibration(self, data: DataInput) -> None:
        """Analyze model calibration."""
        logger.info("Analyzing calibration...")

        # Get cached predictions
        for model_name in self.models:
            if model_name not in self._prediction_cache:
                continue

            cache = self._prediction_cache[model_name]
            y_pred_proba = cache['predictions']
            y_true = cache['y_data']

            # Convert to class indices if needed
            if len(y_true.shape) > 1:
                y_true_idx = np.argmax(y_true, axis=1)
            else:
                y_true_idx = y_true

            # Compute calibration metrics
            ece = compute_ece(y_true_idx, y_pred_proba, self.config.calibration_bins)
            reliability_data = compute_reliability_data(y_true_idx, y_pred_proba, self.config.calibration_bins)
            brier_score = compute_brier_score(cache['y_data'], y_pred_proba)
            entropy_stats = compute_prediction_entropy_stats(y_pred_proba)

            self.results.calibration_metrics[model_name] = {
                'ece': ece,
                'brier_score': brier_score,
                **entropy_stats
            }

            self.results.reliability_data[model_name] = reliability_data

        # Create visualizations
        self._plot_calibration_analysis()

    def _plot_calibration_analysis(self) -> None:
        """Create calibration analysis visualizations."""
        # FIXED: Use subplots_adjust instead of constrained_layout
        fig, axes = plt.subplots(2, 2, figsize=self.config.get_figure_size())
        fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Changed from 0.3 to 0.4

        # Temporarily reduce font sizes for this multi-subplot figure
        original_titlesize = plt.rcParams['axes.titlesize']
        original_labelsize = plt.rcParams['axes.labelsize']
        plt.rcParams.update({
            'axes.titlesize': original_titlesize * 0.85,
            'axes.labelsize': original_labelsize * 0.9
        })

        # 1. Reliability diagrams
        ax1 = axes[0, 0]
        self._plot_reliability_diagrams(ax1)

        # 2. Calibration metrics comparison
        ax2 = axes[0, 1]
        self._plot_calibration_metrics(ax2)

        # 3. Confidence distributions
        ax3 = axes[1, 0]
        self._plot_confidence_distributions(ax3)

        # 4. Entropy analysis
        ax4 = axes[1, 1]
        self._plot_entropy_analysis(ax4)

        plt.suptitle('Calibration Analysis', fontsize=16, fontweight='bold')

        if self.config.save_plots:
            self._save_figure(fig, 'calibration_analysis')
        plt.close(fig)

        # Restore original font sizes
        plt.rcParams.update({
            'axes.titlesize': original_titlesize,
            'axes.labelsize': original_labelsize
        })

    def _plot_reliability_diagrams(self, ax) -> None:
        """Plot reliability diagrams."""
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')

        colors = plt.cm.Set3(np.arange(len(self.models)))

        for i, (model_name, rel_data) in enumerate(self.results.reliability_data.items()):
            ax.plot(rel_data['bin_centers'], rel_data['bin_accuracies'],
                   'o-', color=colors[i], label=model_name, linewidth=2, markersize=8)

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagrams')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    def _plot_calibration_metrics(self, ax) -> None:
        """Plot calibration metrics comparison."""
        metrics_df = pd.DataFrame(self.results.calibration_metrics).T

        if not metrics_df.empty:
            # Select key metrics
            key_metrics = ['ece', 'brier_score', 'mean_entropy']
            available_metrics = [m for m in key_metrics if m in metrics_df.columns]

            if available_metrics:
                metrics_df[available_metrics].plot(kind='bar', ax=ax, alpha=0.8)
                ax.set_xlabel('Model')
                ax.set_ylabel('Value')
                ax.set_title('Calibration Metrics Comparison')
                ax.legend(title='Metric')
                ax.grid(True, alpha=0.3)

                # FIXED: Safe x-tick label rotation
                if len(metrics_df) > 3:
                    safe_set_xticklabels(ax, metrics_df.index.tolist(), rotation=45)

    def _plot_confidence_distributions(self, ax) -> None:
        """Plot confidence score distributions."""
        confidence_data = []

        for model_name, cache in self._prediction_cache.items():
            if model_name in self.results.calibration_metrics:
                max_probs = np.max(cache['predictions'], axis=1)
                for conf in max_probs:
                    confidence_data.append({
                        'Model': model_name,
                        'Confidence': conf
                    })

        if confidence_data:
            df = pd.DataFrame(confidence_data)

            # Use seaborn for cleaner visualization
            for model in df['Model'].unique():
                model_data = df[df['Model'] == model]['Confidence']
                ax.hist(model_data, bins=20, alpha=0.5, density=True,
                       label=model, edgecolor='black', linewidth=0.5)

            ax.set_xlabel('Confidence (Max Probability)')
            ax.set_ylabel('Density')
            ax.set_title('Confidence Score Distributions')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_entropy_analysis(self, ax) -> None:
        """Plot prediction entropy analysis."""
        entropy_data = []

        for model_name, metrics in self.results.calibration_metrics.items():
            if 'mean_entropy' in metrics:
                entropy_data.append({
                    'Model': model_name,
                    'Mean': metrics['mean_entropy'],
                    'Std': metrics.get('std_entropy', 0),
                    'Median': metrics.get('median_entropy', metrics['mean_entropy'])
                })

        if entropy_data:
            df = pd.DataFrame(entropy_data)

            x = np.arange(len(df))
            ax.bar(x, df['Mean'], yerr=df['Std'], capsize=5, alpha=0.8)

            # Add median markers
            ax.scatter(x, df['Median'], color='red', s=100, marker='_',
                      linewidths=3, label='Median')

            ax.set_xlabel('Model')
            ax.set_ylabel('Entropy')
            ax.set_title('Prediction Entropy Analysis')
            # FIXED: Safe x-tick handling
            safe_set_xticklabels(ax, df['Model'].tolist(), rotation=45 if len(df) > 3 else 0)
            ax.legend()
            ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------------------
    # Probability Distribution Analysis
    # ------------------------------------------------------------------------------

    def analyze_probability_distributions(self, data: DataInput) -> None:
        """Analyze probability distributions and confidence patterns."""
        logger.info("Analyzing probability distributions...")

        # Compute confidence metrics
        for model_name, cache in self._prediction_cache.items():
            predictions = cache['predictions']
            self.results.confidence_metrics[model_name] = self._compute_confidence_metrics(predictions)

        # Create visualizations
        self._plot_probability_analysis()

    def _compute_confidence_metrics(self, probabilities: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute various confidence metrics."""
        max_prob = np.max(probabilities, axis=1)

        # Entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)

        # Margin (difference between top 2)
        sorted_probs = np.sort(probabilities, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        # Gini coefficient
        gini = 1 - np.sum(sorted_probs**2, axis=1)

        return {
            'max_probability': max_prob,
            'entropy': entropy,
            'margin': margin,
            'gini_coefficient': gini
        }

    def _plot_probability_analysis(self) -> None:
        """Create probability distribution visualizations."""
        # FIXED: Use subplots_adjust instead of constrained_layout
        fig = plt.figure(figsize=self.config.get_figure_size(1.5))
        gs = plt.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.4)

        # 1. Confidence landscape
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_confidence_landscape(ax1)

        # 2. Model agreement
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_model_agreement(ax2)

        # 3. Uncertainty comparison
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_uncertainty_comparison(ax3)

        # 4. Per-class confidence
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_class_confidence(ax4)

        plt.suptitle('Probability Distribution Analysis', fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'probability_analysis')
        plt.close(fig)

    def _plot_confidence_landscape(self, ax) -> None:
        """Plot confidence vs entropy landscape."""
        colors = plt.cm.Set3(np.arange(len(self.models)))

        for i, (model_name, metrics) in enumerate(self.results.confidence_metrics.items()):
            scatter = ax.scatter(metrics['max_probability'], metrics['entropy'],
                               c=metrics['margin'], cmap='viridis',
                               alpha=0.6, s=20, label=model_name)

        ax.set_xlabel('Confidence (Max Probability)')
        ax.set_ylabel('Entropy')
        ax.set_title('Confidence vs Uncertainty Landscape')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Margin', rotation=270, labelpad=15)

        ax.grid(True, alpha=0.3)

    def _plot_model_agreement(self, ax) -> None:
        """Plot model agreement analysis."""
        if len(self.models) < 2:
            ax.text(0.5, 0.5, 'Model agreement requires\nat least 2 models',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Agreement')
            ax.axis('off')
            return

        # Compute agreement matrix
        model_names = list(self.models.keys())
        n_models = len(model_names)
        agreement_matrix = np.zeros((n_models, n_models))

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    pred1 = np.argmax(self._prediction_cache[model1]['predictions'], axis=1)
                    pred2 = np.argmax(self._prediction_cache[model2]['predictions'], axis=1)
                    agreement_matrix[i, j] = np.mean(pred1 == pred2)

        # Plot heatmap
        sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='viridis',
                   xticklabels=model_names, yticklabels=model_names,
                   ax=ax, cbar_kws={'label': 'Agreement'})
        ax.set_title('Model Agreement Matrix')

    def _plot_uncertainty_comparison(self, ax) -> None:
        """Compare uncertainty metrics across models."""
        uncertainty_data = []

        for model_name, metrics in self.results.confidence_metrics.items():
            uncertainty_data.append({
                'Model': model_name,
                'Mean Entropy': np.mean(metrics['entropy']),
                'Mean Margin': np.mean(metrics['margin']),
                'Std Entropy': np.std(metrics['entropy']),
                'Std Margin': np.std(metrics['margin'])
            })

        if uncertainty_data:
            df = pd.DataFrame(uncertainty_data)

            # Create grouped bar plot
            x = np.arange(len(df))
            width = 0.35

            ax.bar(x - width/2, df['Mean Entropy'], width,
                  yerr=df['Std Entropy'], label='Entropy',
                  capsize=5, alpha=0.8)
            ax.bar(x + width/2, df['Mean Margin'], width,
                  yerr=df['Std Margin'], label='Margin',
                  capsize=5, alpha=0.8)

            ax.set_xlabel('Model')
            ax.set_ylabel('Value')
            ax.set_title('Uncertainty Metrics Comparison')
            # FIXED: Safe x-tick handling
            safe_set_xticklabels(ax, df['Model'].tolist(), rotation=45 if len(df) > 3 else 0)
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_class_confidence(self, ax) -> None:
        """Plot per-class confidence analysis."""
        # Get true labels
        y_true = list(self._prediction_cache.values())[0]['y_data']
        if len(y_true.shape) > 1:
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            y_true_idx = y_true

        n_classes = len(np.unique(y_true_idx))

        # Compute per-class confidence
        class_confidence = {}
        for model_name, cache in self._prediction_cache.items():
            predictions = cache['predictions']
            max_probs = np.max(predictions, axis=1)

            class_conf = []
            for c in range(n_classes):
                mask = y_true_idx == c
                if np.any(mask):
                    class_conf.append(np.mean(max_probs[mask]))
                else:
                    class_conf.append(0)

            class_confidence[model_name] = class_conf

        # Plot
        x = np.arange(n_classes)
        width = 0.8 / len(self.models)

        for i, (model_name, conf) in enumerate(class_confidence.items()):
            ax.bar(x + i * width, conf, width, label=model_name, alpha=0.8)

        ax.set_xlabel('Class')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Per-Class Confidence Analysis')
        ax.set_xticks(x + width * (len(self.models) - 1) / 2)
        ax.set_xticklabels([str(i) for i in range(n_classes)])
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------------------
    # Information Flow Analysis
    # ------------------------------------------------------------------------------

    def analyze_information_flow(self, data: DataInput) -> None:
        """Analyze information flow through network layers."""
        logger.info("Analyzing information flow...")

        # Get sample data
        x_sample = data.x_data[:min(200, len(data.x_data))]  # Use smaller sample for efficiency

        for model_name, extraction_data in self.layer_extraction_models.items():
            if extraction_data is None:
                continue

            # Get multi-layer outputs
            layer_outputs = extraction_data['model'].predict(x_sample, verbose=0)
            layer_info = extraction_data['layer_info']

            # Analyze each layer
            layer_analysis = {}
            for i, (output, info) in enumerate(zip(layer_outputs, layer_info)):
                analysis = self._analyze_layer_information(output, info)
                layer_analysis[info['name']] = analysis

            self.results.information_flow[model_name] = layer_analysis

        # Create visualizations
        self._plot_information_flow()

    def _analyze_layer_information(self, output: np.ndarray, layer_info: Dict) -> Dict[str, Any]:
        """Analyze information content of a layer's output."""
        # Flatten spatial dimensions if needed
        if len(output.shape) == 4:  # Conv layer
            output_flat = np.mean(output, axis=(1, 2))  # Global average pooling
        else:
            output_flat = output

        # Compute statistics
        analysis = {
            'layer_type': layer_info['type'],
            'output_shape': output.shape,
            'mean_activation': float(np.mean(output_flat)),
            'std_activation': float(np.std(output_flat)),
            'sparsity': float(np.mean(np.abs(output_flat) < 1e-5)),
        }

        # Compute effective rank (measure of dimensionality)
        if output_flat.shape[1] > 1:
            try:
                _, s, _ = np.linalg.svd(output_flat, full_matrices=False)
                s_normalized = s / np.sum(s)
                effective_rank = np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-10)))
                analysis['effective_rank'] = float(effective_rank)
            except:
                analysis['effective_rank'] = 0.0

        return analysis

    def _plot_information_flow(self) -> None:
        """Create information flow visualizations."""
        if not self.results.information_flow:
            return

        # FIXED: Use subplots_adjust instead of constrained_layout
        fig, axes = plt.subplots(2, 2, figsize=self.config.get_figure_size())
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # 1. Activation evolution
        ax1 = axes[0, 0]
        self._plot_activation_evolution(ax1)

        # 2. Sparsity evolution
        ax2 = axes[0, 1]
        self._plot_sparsity_evolution(ax2)

        # 3. Effective rank evolution
        ax3 = axes[1, 0]
        self._plot_effective_rank_evolution(ax3)

        # 4. Layer statistics summary
        ax4 = axes[1, 1]
        self._plot_layer_summary(ax4)

        plt.suptitle('Information Flow Analysis', fontsize=16, fontweight='bold')

        if self.config.save_plots:
            self._save_figure(fig, 'information_flow')
        plt.close(fig)

    def _plot_activation_evolution(self, ax) -> None:
        """Plot how activation statistics evolve through layers."""
        for model_name, layer_analysis in self.results.information_flow.items():
            means = []
            stds = []
            layer_names = []

            for layer_name, analysis in layer_analysis.items():
                means.append(analysis['mean_activation'])
                stds.append(analysis['std_activation'])
                layer_names.append(layer_name.split('_')[0])

            x = np.arange(len(means))
            ax.plot(x, means, 'o-', label=f'{model_name} (mean)', linewidth=2)
            ax.fill_between(x, np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds), alpha=0.3)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Activation Statistics')
        ax.set_title('Activation Evolution Through Network')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_sparsity_evolution(self, ax) -> None:
        """Plot how sparsity evolves through layers."""
        for model_name, layer_analysis in self.results.information_flow.items():
            sparsities = []

            for layer_name, analysis in layer_analysis.items():
                sparsities.append(analysis['sparsity'])

            ax.plot(range(len(sparsities)), sparsities, 'o-',
                   label=model_name, linewidth=2, markersize=8)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Sparsity')
        ax.set_title('Sparsity Evolution Through Network')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_effective_rank_evolution(self, ax) -> None:
        """Plot effective rank evolution."""
        for model_name, layer_analysis in self.results.information_flow.items():
            ranks = []

            for layer_name, analysis in layer_analysis.items():
                if 'effective_rank' in analysis:
                    ranks.append(analysis['effective_rank'])

            if ranks:
                ax.plot(range(len(ranks)), ranks, 'o-',
                       label=model_name, linewidth=2, markersize=8)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Effective Rank')
        ax.set_title('Information Dimensionality Through Network')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_layer_summary(self, ax) -> None:
        """Create summary table of layer statistics."""
        ax.axis('tight')
        ax.axis('off')

        # Collect summary data
        summary_data = []
        for model_name, layer_analysis in self.results.information_flow.items():
            for i, (layer_name, analysis) in enumerate(layer_analysis.items()):
                if i < 5:  # Limit to first 5 layers
                    summary_data.append([
                        model_name[:10],
                        layer_name[:15],
                        analysis['layer_type'][:10],
                        f"{analysis['mean_activation']:.3f}",
                        f"{analysis['sparsity']:.2%}"
                    ])

        if summary_data:
            table = ax.table(
                cellText=summary_data,
                colLabels=['Model', 'Layer', 'Type', 'Mean Act', 'Sparsity'],
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # Style header
            for i in range(5):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Layer Information Summary', pad=20)

    # ------------------------------------------------------------------------------
    # Summary Dashboard
    # ------------------------------------------------------------------------------

    def create_summary_dashboard(self) -> None:
        """Create a comprehensive summary dashboard."""
        # FIXED: Use subplots_adjust instead of constrained_layout for better control
        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4)

        # 1. Model performance comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_model_performance(ax1)

        # 2. Key metrics summary
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_key_metrics_summary(ax2)

        # 3. Weight statistics
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_weight_summary(ax3)

        # 4. Calibration summary
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_calibration_summary(ax4)

        # 5. Confidence analysis (improved with seaborn)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_confidence_summary(ax5)

        # 6. Analysis overview
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_analysis_overview(ax6)

        plt.suptitle('Model Analysis Summary Dashboard', fontsize=18, fontweight='bold')

        # FIXED: Use subplots_adjust instead of constrained_layout
        fig.subplots_adjust(top=0.93, bottom=0.07, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'summary_dashboard')
        plt.close(fig)

    def _plot_model_performance(self, ax) -> None:
        """Plot model performance metrics."""
        if not self.results.model_metrics:
            ax.text(0.5, 0.5, 'No performance metrics available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Performance')
            ax.axis('off')
            return

        # Extract accuracy and loss
        models = []
        accuracies = []
        losses = []

        for model_name, metrics in self.results.model_metrics.items():
            models.append(model_name)
            accuracies.append(metrics.get('accuracy', 0))
            losses.append(metrics.get('loss', 0))

        x = np.arange(len(models))
        width = 0.35

        ax2 = ax.twinx()

        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='green')
        bars2 = ax2.bar(x + width/2, losses, width, label='Loss', alpha=0.8, color='red')

        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy', color='green')
        ax2.set_ylabel('Loss', color='red')
        ax.set_title('Model Performance Comparison')
        # FIXED: Safe x-tick handling
        safe_set_xticklabels(ax, models, rotation=45 if len(models) > 3 else 0)

        # Add value labels
        for bar, val in zip(bars1, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        for bar, val in zip(bars2, losses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax.grid(True, alpha=0.3, axis='y')

    def _plot_key_metrics_summary(self, ax) -> None:
        """Plot summary of key metrics across analyses."""
        ax.axis('tight')
        ax.axis('off')

        # Collect key metrics
        summary_data = []
        for model_name in self.models:
            row = [model_name[:15]]

            # Add accuracy
            if model_name in self.results.model_metrics:
                acc = self.results.model_metrics[model_name].get('accuracy', 0)
                row.append(f"{acc:.3f}")
            else:
                row.append("-")

            # Add ECE
            if model_name in self.results.calibration_metrics:
                ece = self.results.calibration_metrics[model_name].get('ece', 0)
                row.append(f"{ece:.3f}")
            else:
                row.append("-")

            # Add weight stats
            if model_name in self.results.weight_stats:
                n_params = sum(np.prod(s['shape']) for s in self.results.weight_stats[model_name].values())
                row.append(f"{n_params:,}")
            else:
                row.append("-")

            # Add mean entropy
            if model_name in self.results.calibration_metrics:
                entropy = self.results.calibration_metrics[model_name].get('mean_entropy', 0)
                row.append(f"{entropy:.3f}")
            else:
                row.append("-")

            summary_data.append(row)

        if summary_data:
            table = ax.table(
                cellText=summary_data,
                colLabels=['Model', 'Accuracy', 'ECE', '# Params', 'Entropy'],
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            # Style header
            for i in range(5):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Key Metrics Summary', pad=20)

    def _plot_weight_summary(self, ax) -> None:
        """Plot weight distribution summary."""
        if not any(self.results.weight_stats.values()):
            ax.text(0.5, 0.5, 'No weight analysis available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Distribution Summary')
            ax.axis('off')
            return

        # Compute overall statistics
        model_stats = []
        for model_name, weight_stats in self.results.weight_stats.items():
            all_means = [s['basic']['mean'] for s in weight_stats.values()]
            all_stds = [s['basic']['std'] for s in weight_stats.values()]
            all_zeros = [s['distribution']['zero_fraction'] for s in weight_stats.values()]

            if all_means:
                model_stats.append({
                    'Model': model_name,
                    'Mean±Std': f"{np.mean(all_means):.3f}±{np.mean(all_stds):.3f}",
                    'Sparsity': np.mean(all_zeros)
                })

        if model_stats:
            df = pd.DataFrame(model_stats)

            # Bar plot of sparsity
            ax.bar(df['Model'], df['Sparsity'], alpha=0.8)
            ax.set_ylabel('Average Sparsity')
            ax.set_title('Weight Sparsity Summary')
            # FIXED: Safe x-tick handling
            safe_set_xticklabels(ax, df['Model'].tolist(), rotation=45 if len(df) > 3 else 0)

            # Add text annotations
            for i, row in df.iterrows():
                ax.text(i, row['Sparsity'] + 0.01, row['Mean±Std'],
                       ha='center', va='bottom', fontsize=9)

            ax.grid(True, alpha=0.3, axis='y')

    def _plot_calibration_summary(self, ax) -> None:
        """Plot calibration summary."""
        if not self.results.calibration_metrics:
            ax.text(0.5, 0.5, 'No calibration analysis available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Calibration Summary')
            ax.axis('off')
            return

        # Extract ECE and Brier scores
        models = []
        eces = []
        briers = []

        for model_name, metrics in self.results.calibration_metrics.items():
            models.append(model_name)
            eces.append(metrics.get('ece', 0))
            briers.append(metrics.get('brier_score', 0))

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, eces, width, label='ECE', alpha=0.8)
        ax.bar(x + width/2, briers, width, label='Brier Score', alpha=0.8)

        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Calibration Summary')
        # FIXED: Safe x-tick handling
        safe_set_xticklabels(ax, models, rotation=45 if len(models) > 3 else 0)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_confidence_summary(self, ax) -> None:
        """Plot confidence analysis summary using seaborn."""
        if not self.results.confidence_metrics:
            ax.text(0.5, 0.5, 'No confidence analysis available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Confidence Analysis Summary')
            ax.axis('off')
            return

        # Build dataframe for seaborn
        confidence_data = []
        for model_name, metrics in self.results.confidence_metrics.items():
            for conf in metrics['max_probability']:
                confidence_data.append({
                    'Model': model_name,
                    'Confidence': conf
                })

        if confidence_data:
            df = pd.DataFrame(confidence_data)

            # FIXED: Use seaborn violin plot with safe handling
            try:
                sns.violinplot(data=df, x='Model', y='Confidence', ax=ax,
                               inner='quartile', hue='Model', palette=self.config.color_palette, legend=False)

                # FIXED: Safe x-tick label rotation
                labels = [t.get_text() for t in ax.get_xticklabels()]
                if len(labels) > 3:
                    safe_set_xticklabels(ax, labels, rotation=45)

            except Exception as e:
                logger.warning(f"Could not create confidence violin plot: {e}")
                # Fallback to simple histogram
                for model in df['Model'].unique():
                    model_data = df[df['Model'] == model]['Confidence']
                    ax.hist(model_data, bins=20, alpha=0.5, label=model)
                ax.legend()

            ax.set_ylabel('Confidence (Max Probability)')
            ax.set_title('Confidence Distribution Summary')
            ax.grid(True, alpha=0.3, axis='y')

    def _plot_analysis_overview(self, ax) -> None:
        """Plot overview of analyses performed."""
        ax.axis('off')

        # Create analysis summary text
        summary_text = "Analysis Overview\n" + "="*50 + "\n\n"

        # Models analyzed
        summary_text += f"Models Analyzed: {len(self.models)}\n"
        for model_name in self.models:
            summary_text += f"  • {model_name}\n"

        summary_text += f"\nAnalyses Performed:\n"

        # Check which analyses were performed
        if any(self.results.weight_stats.values()):
            n_weights = sum(len(stats) for stats in self.results.weight_stats.values())
            summary_text += f"  ✓ Weight Analysis ({n_weights} tensors)\n"

        if any(self.results.activation_stats.values()):
            n_acts = sum(len(stats) for stats in self.results.activation_stats.values())
            summary_text += f"  ✓ Activation Analysis ({n_acts} layers)\n"

        if self.results.calibration_metrics:
            summary_text += f"  ✓ Calibration Analysis\n"

        if self.results.confidence_metrics:
            summary_text += f"  ✓ Probability Distribution Analysis\n"

        if self.results.information_flow:
            summary_text += f"  ✓ Information Flow Analysis\n"

        # Add timestamp
        summary_text += f"\nAnalysis Timestamp: {self.results.analysis_timestamp}\n"

        # Add output directory
        summary_text += f"Output Directory: {self.output_dir}\n"

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace')

    # ------------------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------------------

    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure with configured settings and error handling."""
        try:
            filepath = self.output_dir / f"{name}.{self.config.save_format}"
            # FIXED: Save with explicit bbox_inches and pad_inches to avoid layout issues
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none', pad_inches=0.1)
            logger.info(f"Saved plot: {filepath}")
        except Exception as e:
            logger.error(f"Could not save figure {name}: {e}")
            # Try saving without bbox_inches as fallback
            try:
                filepath = self.output_dir / f"{name}_fallback.{self.config.save_format}"
                fig.savefig(filepath, dpi=self.config.dpi, facecolor='white', edgecolor='none')
                logger.info(f"Saved plot (fallback): {filepath}")
            except Exception as e2:
                logger.error(f"Fallback save also failed for {name}: {e2}")

    def save_results(self, filename: str = "analysis_results.json") -> None:
        """Save analysis results to JSON file with configurable filename."""
        results_dict = {
            'timestamp': self.results.analysis_timestamp,
            'config': self.config.__dict__,
            'model_metrics': self.results.model_metrics,
            'weight_stats': self.results.weight_stats,
            'activation_stats': self.results.activation_stats,
            'calibration_metrics': self.results.calibration_metrics,
        }

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            # FIXED: Handle numpy scalars properly
            if isinstance(obj, np.generic):
                return obj.item()  # Convert numpy scalar to python equivalent
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items() if k != 'raw_activations'}
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

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the analysis."""
        summary = {
            'n_models': len(self.models),
            'analyses_performed': [],
            'model_performance': {},
            'calibration_summary': {},
            'weight_summary': {}
        }

        # Check which analyses were performed
        if any(self.results.weight_stats.values()):
            summary['analyses_performed'].append('weight_analysis')
        if any(self.results.activation_stats.values()):
            summary['analyses_performed'].append('activation_analysis')
        if self.results.calibration_metrics:
            summary['analyses_performed'].append('calibration_analysis')
        if self.results.confidence_metrics:
            summary['analyses_performed'].append('probability_analysis')
        if self.results.information_flow:
            summary['analyses_performed'].append('information_flow')

        # Add model performance
        for model_name, metrics in self.results.model_metrics.items():
            summary['model_performance'][model_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'loss': metrics.get('loss', 0)
            }

        # Add calibration summary
        for model_name, metrics in self.results.calibration_metrics.items():
            summary['calibration_summary'][model_name] = {
                'ece': metrics.get('ece', 0),
                'brier_score': metrics.get('brier_score', 0)
            }

        # Add weight summary
        for model_name, weight_stats in self.results.weight_stats.items():
            n_params = sum(np.prod(s['shape']) for s in weight_stats.values())
            summary['weight_summary'][model_name] = {
                'total_parameters': n_params,
                'n_weight_tensors': len(weight_stats)
            }

        return summary