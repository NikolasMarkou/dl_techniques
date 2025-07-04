"""
Enhanced Model Analyzer for Neural Networks
============================================================================

A comprehensive, modular analyzer with training dynamics and refined visualizations

Key Enhancements:
- Added comprehensive training dynamics analysis
- Removed configuration redundancy
- Improved weight correlation methodology
- Added quantitative training metrics
- Enhanced summary dashboard with training insights
- Refined code organization and documentation

Example Usage:
    ```python
    from analyzer import ModelAnalyzer, AnalysisConfig

    # Configure analysis
    config = AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=True,
        analyze_training_dynamics=True,
        plot_style='publication'
    )

    # Create analyzer with training history
    analyzer = ModelAnalyzer(
        models=models,
        config=config,
        training_history=training_histories
    )

    # Run comprehensive analysis
    results = analyzer.analyze(test_data)
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
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpecFromSubplotSpec
from typing import Dict, List, Optional, Union, Any, Tuple, Set, NamedTuple

# ------------------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.calibration_metrics import (
    compute_ece,
    compute_brier_score,
    compute_reliability_data,
    compute_prediction_entropy_stats
)

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

# Health score thresholds and weights
WEIGHT_HEALTH_L2_NORMALIZER = 10.0  # L2 norm normalization factor for weight health calculation
WEIGHT_HEALTH_SPARSITY_THRESHOLD = 0.8  # Maximum acceptable sparsity before considering weights unhealthy
LAYER_SPECIALIZATION_MAX_RANK = 10.0  # Maximum effective rank for normalization in specialization analysis
ACTIVATION_MAGNITUDE_NORMALIZER = 5.0  # Activation magnitude normalization factor for health scoring

# Training analysis constants
CONVERGENCE_THRESHOLD = 0.95  # Fraction of peak performance to consider model "converged"
TRAINING_STABILITY_WINDOW = 10  # Number of recent epochs to analyze for stability (higher = smoother estimate)
OVERFITTING_ANALYSIS_FRACTION = 0.33  # Final fraction of training to analyze for overfitting metrics

# Metric name patterns for flexible history parsing
LOSS_PATTERNS = ['loss', 'total_loss', 'train_loss']
VAL_LOSS_PATTERNS = ['val_loss', 'validation_loss', 'valid_loss']
ACC_PATTERNS = ['accuracy', 'acc', 'categorical_accuracy', 'sparse_categorical_accuracy', 
                'binary_accuracy', 'top_k_categorical_accuracy']
VAL_ACC_PATTERNS = ['val_accuracy', 'val_acc', 'validation_accuracy', 'val_categorical_accuracy',
                    'val_sparse_categorical_accuracy', 'val_binary_accuracy']

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
    """Configuration for all analysis types."""

    # Analysis toggles (removed redundant flags)
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

    def get_figure_size(self, scale: float = 1.0) -> Tuple[float, float]:
        """Get figure size with optional scaling."""
        return (self.fig_width * scale, self.fig_height * scale)

    def setup_plotting_style(self) -> None:
        """Set up matplotlib style based on configuration."""
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

# ------------------------------------------------------------------------------
# Training Metrics Container
# ------------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Container for computed training metrics.
    
    Attributes:
        epochs_to_convergence: Number of epochs to reach 95% of peak performance
        training_stability_score: Standard deviation of recent validation losses (lower = more stable)
        overfitting_index: Average gap between validation and training loss in final third of training
                          Positive values indicate overfitting, negative indicate underfitting
        peak_performance: Best validation metrics achieved during training with epoch info
        final_gap: Difference between validation and training loss at end of training
        smoothed_curves: Smoothed versions of training curves for cleaner visualization
    """
    epochs_to_convergence: Dict[str, int] = field(default_factory=dict)
    training_stability_score: Dict[str, float] = field(default_factory=dict)
    overfitting_index: Dict[str, float] = field(default_factory=dict)
    peak_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    final_gap: Dict[str, float] = field(default_factory=dict)
    smoothed_curves: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

# ------------------------------------------------------------------------------
# Analysis Results Container
# ------------------------------------------------------------------------------

@dataclass
class AnalysisResults:
    """Container for all analysis results."""

    # Model performance
    model_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Activation analysis (now part of information flow)
    activation_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Weight analysis
    weight_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    weight_pca: Optional[Dict[str, Any]] = None

    # Calibration analysis
    calibration_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reliability_data: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    confidence_metrics: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    # Information flow
    information_flow: Dict[str, Any] = field(default_factory=dict)

    # Training history and dynamics
    training_history: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    training_metrics: Optional[TrainingMetrics] = None

    # Metadata
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Optional[AnalysisConfig] = None

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def safe_set_xticklabels(ax, labels, rotation=0, max_labels=10):
    """Safely set x-tick labels with proper handling."""
    try:
        if len(labels) > max_labels:
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
        try:
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
        except Exception:
            pass

def smooth_curve(values: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply smoothing to a curve using a moving average."""
    if len(values) < window_size:
        return values
    
    # Pad the array to handle edges
    padded = np.pad(values, (window_size//2, window_size//2), mode='edge')
    
    # Apply moving average
    smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    
    return smoothed

def find_metric_in_history(history: Dict[str, List[float]], patterns: List[str]) -> Optional[List[float]]:
    """Flexibly find a metric in training history by checking multiple possible names.
    
    Args:
        history: Training history dictionary
        patterns: List of possible metric names to check
        
    Returns:
        The metric values if found, None otherwise
    """
    for pattern in patterns:
        if pattern in history:
            return history[pattern]
    
    # Also check for partial matches (e.g., 'sparse_categorical_accuracy' matches 'accuracy' pattern)
    for key in history:
        for pattern in patterns:
            if pattern in key:
                return history[key]
    
    return None

# ------------------------------------------------------------------------------
# Enhanced Model Analyzer
# ------------------------------------------------------------------------------

class ModelAnalyzer:
    """
    Enhanced model analyzer with training dynamics and improved visualizations.
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

        # Set up activation extraction models
        self._setup_activation_models()

        logger.info(f"ModelAnalyzer initialized with {len(models)} models")

    def _setup_model_colors(self) -> None:
        """Set up consistent colors for models across all visualizations."""
        # Define consistent colors for models (avoiding yellow)
        model_names = sorted(self.models.keys())  # Sort for consistency
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(model_names)]
        self.model_colors = dict(zip(model_names, color_palette))

    def _setup_activation_models(self) -> None:
        """Set up models for extracting intermediate activations."""
        self.activation_models = {}
        self.layer_extraction_models = {}

        for model_name, model in self.models.items():
            try:
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
                        self.layer_extraction_models[model_name] = None
                else:
                    logger.warning(f"No suitable layers found for extraction in {model_name}")
                    self.layer_extraction_models[model_name] = None
            except Exception as e:
                logger.error(f"Failed to setup activation models for {model_name}: {e}")
                self.layer_extraction_models[model_name] = None

    def _get_extraction_layers(self, model: keras.Model) -> List[Dict[str, Any]]:
        """Get layers suitable for information flow analysis."""
        extraction_layers = []

        try:
            # Check if model has layers attribute
            if not hasattr(model, 'layers'):
                logger.warning(f"Model does not have 'layers' attribute")
                return extraction_layers

            for layer in model.layers:
                try:
                    # Check for common layer types that are suitable for analysis
                    if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense,
                                        keras.layers.BatchNormalization, keras.layers.LayerNormalization)):
                        if hasattr(layer, 'output') and layer.output is not None:
                            extraction_layers.append({
                                'name': layer.name,
                                'type': layer.__class__.__name__,
                                'output': layer.output
                            })
                except Exception as e:
                    logger.warning(f"Could not process layer {getattr(layer, 'name', 'unknown')}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Failed to extract layers from model: {e}")

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
        if 'weights' in analysis_types:
            self.analyze_weights()

        if 'calibration' in analysis_types and data is not None:
            self.analyze_confidence_and_calibration(data)

        if 'information_flow' in analysis_types and data is not None:
            self.analyze_information_flow(data)

        if 'training_dynamics' in analysis_types and self.results.training_history:
            self.analyze_training_dynamics()

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

    # ------------------------------------------------------------------------------
    # Enhanced Weight Analysis
    # ------------------------------------------------------------------------------

    def analyze_weights(self) -> None:
        """Analyze weight distributions with improved visualizations."""
        logger.info("Analyzing weight distributions...")

        for model_name, model in self.models.items():
            self.results.weight_stats[model_name] = {}

            for layer in model.layers:
                if (self.config.weight_layer_types and
                    layer.__class__.__name__ not in self.config.weight_layer_types):
                    continue

                weights = layer.get_weights()
                if not weights:
                    continue

                for idx, w in enumerate(weights):
                    if len(w.shape) < 2 and not self.config.analyze_biases:
                        continue

                    weight_name = f"{layer.name}_w{idx}"
                    stats = self._compute_weight_statistics(w)
                    self.results.weight_stats[model_name][weight_name] = stats

        # Compute PCA if requested (removed weight correlations)
        if self.config.compute_weight_pca:
            self._compute_weight_pca()

        # Create visualizations
        self._plot_enhanced_weight_analysis()

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

        if len(weights.shape) == 2:
            try:
                stats['norms']['spectral'] = float(np.linalg.norm(weights, 2))
            except:
                stats['norms']['spectral'] = 0.0

        return stats

    def _compute_weight_pca(self) -> None:
        """Perform PCA analysis specifically on final layer weights."""
        final_layer_features = []
        labels = []

        for model_name, model in self.models.items():
            # Find the final dense layer
            final_dense = None
            for layer in reversed(model.layers):
                if isinstance(layer, keras.layers.Dense):
                    final_dense = layer
                    break

            if final_dense and final_dense.get_weights():
                weights = final_dense.get_weights()[0]  # Get kernel weights
                # Flatten and take a subset if too large
                flat_weights = weights.flatten()
                if len(flat_weights) > 1000:
                    flat_weights = flat_weights[::len(flat_weights)//1000]

                # Check for finite values
                if np.all(np.isfinite(flat_weights)):
                    final_layer_features.append(flat_weights)
                    labels.append(model_name)
                else:
                    logger.warning(f"Skipping {model_name} in PCA due to non-finite weights")

        if len(final_layer_features) >= 2:
            # Ensure all features have same length
            min_len = min(len(f) for f in final_layer_features)
            aligned_features = [f[:min_len] for f in final_layer_features]

            try:
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(aligned_features)

                # Perform PCA
                pca = PCA(n_components=min(3, len(features_scaled)))
                pca_result = pca.fit_transform(features_scaled)

                self.results.weight_pca = {
                    'components': pca_result,
                    'explained_variance': pca.explained_variance_ratio_,
                    'labels': labels
                }
            except Exception as e:
                logger.warning(f"Could not perform PCA on final layer weights: {e}")

    def _plot_enhanced_weight_analysis(self) -> None:
        """Create unified Weight Learning Journey visualization."""
        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 1, figure=fig, hspace=0.4, height_ratios=[1, 1])

        # Top: Weight magnitude evolution through layers
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_weight_learning_evolution(ax1)

        # Bottom: Weight health heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_weight_health_heatmap(ax2)

        plt.suptitle('Weight Learning Journey', fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'weight_learning_journey')
        plt.close(fig)

    def _plot_weight_learning_evolution(self, ax) -> None:
        """Plot how weight magnitudes evolve through network layers."""
        evolution_data = {}

        for model_name, weight_stats in self.results.weight_stats.items():
            layer_indices = []
            l2_norms = []
            mean_abs_weights = []

            # Sort layers by name to get consistent ordering
            sorted_layers = sorted(weight_stats.items(), key=lambda x: x[0])

            for idx, (layer_name, stats) in enumerate(sorted_layers):
                layer_indices.append(idx)
                l2_norms.append(stats['norms']['l2'])
                mean_abs_weights.append(abs(stats['basic']['mean']))

            evolution_data[model_name] = {
                'indices': layer_indices,
                'l2_norms': l2_norms,
                'mean_abs_weights': mean_abs_weights
            }

        if evolution_data:
            # Plot L2 norm evolution
            for model_name, data in evolution_data.items():
                color = self.model_colors.get(model_name, '#333333')
                ax.plot(data['indices'], data['l2_norms'], 'o-',
                       label=f'{model_name}', color=color, linewidth=2, markersize=6)

            ax.set_xlabel('Layer Index')
            ax.set_ylabel('L2 Norm')
            ax.set_title('Weight Magnitude Evolution Through Network Depth', fontsize=12, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            # Add insight text
            ax.text(0.02, 0.98, 'Higher values indicate larger weight magnitudes\nSteep changes suggest learning transitions',
                   transform=ax.transAxes, ha='left', va='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))
        else:
            ax.text(0.5, 0.5, 'No weight evolution data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Magnitude Evolution')
            ax.axis('off')

    def _plot_weight_health_heatmap(self, ax) -> None:
        """Create a comprehensive weight health heatmap."""
        if not self.results.weight_stats:
            ax.text(0.5, 0.5, 'No weight statistics available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Health Heatmap')
            ax.axis('off')
            return

        # Prepare health metrics - each model uses its own layer sequence
        health_metrics = []
        models = sorted(self.results.weight_stats.keys())
        max_layers = 12  # Show first 12 layers for each model

        for model_name in models:
            weight_stats = self.results.weight_stats[model_name]
            model_health = []

            # Get this model's layers in order (first 12)
            model_layers = sorted(list(weight_stats.keys()))[:max_layers]

            for layer_idx in range(max_layers):
                if layer_idx < len(model_layers):
                    layer_name = model_layers[layer_idx]
                    stats = weight_stats[layer_name]

                    # Calculate health score (0-1, higher is better)
                    # Normalize L2 norm (smaller is often better, but not too small)
                    l2_norm = stats['norms']['l2']
                    norm_health = 1.0 / (1.0 + l2_norm / WEIGHT_HEALTH_L2_NORMALIZER)

                    # Sparsity (moderate sparsity is ok, too much is bad)
                    sparsity = stats['distribution']['zero_fraction']
                    sparsity_health = 1.0 - min(sparsity, WEIGHT_HEALTH_SPARSITY_THRESHOLD)

                    # Weight distribution (closer to normal is better)
                    weight_std = stats['basic']['std']
                    weight_mean = abs(stats['basic']['mean'])
                    dist_health = min(weight_std / (weight_mean + 1e-6), 1.0)  # Ratio of std to mean

                    # Combined health score
                    health_score = (norm_health + sparsity_health + dist_health) / 3.0
                    model_health.append(health_score)
                else:
                    # Model has fewer layers than max_layers
                    model_health.append(0.0)

            health_metrics.append(model_health)

        if health_metrics:
            # Create heatmap
            health_array = np.array(health_metrics)

            im = ax.imshow(health_array, cmap='RdYlGn', aspect='auto',
                          vmin=0, vmax=1, interpolation='nearest')

            # Set labels
            ax.set_title('Weight Health Across Layers', fontsize=12, fontweight='bold')
            ax.set_xlabel('Layer Position')
            ax.set_ylabel('Model')

            # Set ticks
            ax.set_xticks(range(max_layers))
            ax.set_xticklabels([f'L{i}' for i in range(max_layers)], fontsize=9)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models, fontsize=9)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Health Score', rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=9)

            # Add value annotations for better readability
            for i in range(len(models)):
                for j in range(max_layers):
                    value = health_array[i, j]
                    text_color = 'white' if value < 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=text_color, fontsize=8, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for health heatmap',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Health Heatmap')
            ax.axis('off')

    # ------------------------------------------------------------------------------
    # Merged Confidence and Calibration Analysis
    # ------------------------------------------------------------------------------

    def analyze_confidence_and_calibration(self, data: DataInput) -> None:
        """Analyze model confidence and calibration in a unified way."""
        logger.info("Analyzing confidence and calibration...")

        for model_name in self.models:
            if model_name not in self._prediction_cache:
                continue

            cache = self._prediction_cache[model_name]
            y_pred_proba = cache['predictions']
            y_true = cache['y_data']

            # Convert to class indices if needed - handle different data types
            y_true = np.asarray(y_true)  # Ensure numpy array
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                y_true_idx = np.argmax(y_true, axis=1)
            else:
                y_true_idx = y_true.flatten().astype(int)

            # Compute calibration metrics
            ece = compute_ece(y_true_idx, y_pred_proba, self.config.calibration_bins)
            reliability_data = compute_reliability_data(y_true_idx, y_pred_proba, self.config.calibration_bins)
            brier_score = compute_brier_score(cache['y_data'], y_pred_proba)
            entropy_stats = compute_prediction_entropy_stats(y_pred_proba)

            # Compute per-class ECE with validated bins
            n_classes = y_pred_proba.shape[1]
            per_class_ece = []
            per_class_bins = max(2, self.config.calibration_bins // 2)  # Ensure minimum 2 bins

            for c in range(n_classes):
                class_mask = y_true_idx == c
                if np.any(class_mask):
                    class_ece = compute_ece(y_true_idx[class_mask], y_pred_proba[class_mask],
                                          per_class_bins)
                    per_class_ece.append(class_ece)
                else:
                    per_class_ece.append(0.0)

            self.results.calibration_metrics[model_name] = {
                'ece': ece,
                'brier_score': brier_score,
                'per_class_ece': per_class_ece,
                **entropy_stats
            }

            self.results.reliability_data[model_name] = reliability_data

            # Compute confidence metrics
            self.results.confidence_metrics[model_name] = self._compute_confidence_metrics(y_pred_proba)

        # Create unified visualization
        self._plot_confidence_calibration_analysis()

    def _compute_confidence_metrics(self, probabilities: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute various confidence metrics."""
        max_prob = np.max(probabilities, axis=1)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)

        sorted_probs = np.sort(probabilities, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        gini = 1 - np.sum(sorted_probs**2, axis=1)

        return {
            'max_probability': max_prob,
            'entropy': entropy,
            'margin': margin,
            'gini_coefficient': gini
        }

    def _plot_confidence_calibration_analysis(self) -> None:
        """Create unified confidence and calibration visualizations."""
        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Reliability Diagram
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_enhanced_reliability_diagram(ax1)

        # 2. Confidence Distributions (Raincloud)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_confidence_raincloud(ax2)

        # 3. Per-Class ECE
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_per_class_ece(ax3)

        # 4. Uncertainty Landscape
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_uncertainty_landscape(ax4)

        plt.suptitle('Confidence and Calibration Analysis', fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'confidence_calibration_analysis')
        plt.close(fig)

    def _plot_enhanced_reliability_diagram(self, ax) -> None:
        """Plot enhanced reliability diagram with confidence intervals."""
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')

        for model_name, rel_data in self.results.reliability_data.items():
            color = self.model_colors.get(model_name, '#333333')

            # Plot main line
            ax.plot(rel_data['bin_centers'], rel_data['bin_accuracies'],
                   'o-', color=color, label=model_name, linewidth=2, markersize=8)

            # Add shaded confidence region if we have sample counts
            if 'bin_counts' in rel_data:
                # Simple confidence interval based on binomial proportion
                counts = rel_data['bin_counts']
                props = rel_data['bin_accuracies']
                se = np.sqrt(props * (1 - props) / (counts + 1))

                ax.fill_between(rel_data['bin_centers'],
                               props - 1.96*se, props + 1.96*se,
                               alpha=0.2, color=color)

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagrams with 95% CI')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    def _plot_confidence_raincloud(self, ax) -> None:
        """Plot confidence distributions as raincloud plot."""
        confidence_data = []

        for model_name, metrics in self.results.confidence_metrics.items():
            # Safety check for required keys
            if 'max_probability' not in metrics:
                logger.warning(f"Missing 'max_probability' key for model {model_name}")
                continue

            for conf in metrics['max_probability']:
                confidence_data.append({
                    'Model': model_name,
                    'Confidence': conf
                })

        if confidence_data:
            df = pd.DataFrame(confidence_data)

            # Sort models to match color order
            model_order = sorted(df['Model'].unique())

            # Create violin plot with quartiles using matplotlib
            parts = ax.violinplot([df[df['Model'] == m]['Confidence'].values
                                  for m in model_order],
                                 positions=range(len(model_order)),
                                 vert=False, showmeans=False, showmedians=True,
                                 showextrema=True)

            # Color the violin plots
            for i, model in enumerate(model_order):
                color = self.model_colors.get(model, '#333333')
                parts['bodies'][i].set_facecolor(color)
                parts['bodies'][i].set_alpha(0.7)

            # Color other violin parts
            for partname in ['cmeans', 'cmaxes', 'cmins', 'cbars', 'cmedians', 'cquantiles']:
                if partname in parts:
                    parts[partname].set_color('black')
                    parts[partname].set_alpha(0.8)

            # Add model labels
            ax.set_yticks(range(len(model_order)))
            ax.set_yticklabels(model_order)

            # Add summary statistics as text
            for i, model in enumerate(model_order):
                model_data = df[df['Model'] == model]['Confidence']
                mean_conf = model_data.mean()
                ax.text(mean_conf, i, f'{mean_conf:.3f}',
                       ha='center', va='bottom', fontsize=9)

            ax.set_xlabel('Confidence (Max Probability)')
            ax.set_ylabel('')  # Remove the y-axis label
            ax.set_title('Confidence Score Distributions')
            ax.grid(True, alpha=0.3, axis='x')

    def _plot_per_class_ece(self, ax) -> None:
        """Plot per-class Expected Calibration Error."""
        ece_data = []

        for model_name, metrics in self.results.calibration_metrics.items():
            if 'per_class_ece' in metrics:
                for class_idx, ece in enumerate(metrics['per_class_ece']):
                    ece_data.append({
                        'Model': model_name,
                        'Class': str(class_idx),
                        'ECE': ece
                    })

        if ece_data:
            df = pd.DataFrame(ece_data)

            # Sort models to match color order
            model_order = sorted(df['Model'].unique())
            n_models = len(model_order)
            n_classes = len(df['Class'].unique())

            x = np.arange(n_classes)
            width = 0.8 / n_models

            for i, model in enumerate(model_order):
                model_data = df[df['Model'] == model]
                color = self.model_colors.get(model, '#333333')
                ax.bar(x + i * width, model_data['ECE'], width,
                       label=model, alpha=0.8, color=color)

            ax.set_xlabel('Class')
            ax.set_ylabel('Expected Calibration Error')
            ax.set_title('Per-Class Calibration Error')
            ax.set_xticks(x + width * (n_models - 1) / 2)
            ax.set_xticklabels([str(i) for i in range(n_classes)])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

    def _plot_uncertainty_landscape(self, ax) -> None:
        """Plot uncertainty landscape with density contours for each model."""
        from scipy.stats import gaussian_kde

        # Sort models for consistent ordering
        model_order = sorted(self.results.confidence_metrics.keys())

        # Plot contours for each model
        for model_name in model_order:
            metrics = self.results.confidence_metrics[model_name]

            # Safety checks for required keys
            if 'max_probability' not in metrics or 'entropy' not in metrics:
                logger.warning(f"Missing confidence metrics keys for model {model_name}")
                continue

            confidence = metrics['max_probability']
            entropy = metrics['entropy']
            color = self.model_colors.get(model_name, '#333333')

            if len(confidence) < 10:  # Skip if too few points
                continue

            try:
                # Create 2D KDE
                xy = np.vstack([confidence, entropy])
                kde = gaussian_kde(xy)

                # Create grid for contour plot
                conf_min, conf_max = confidence.min(), confidence.max()
                ent_min, ent_max = entropy.min(), entropy.max()

                # Add some padding
                conf_range = conf_max - conf_min
                ent_range = ent_max - ent_min
                conf_min -= 0.05 * conf_range
                conf_max += 0.05 * conf_range
                ent_min -= 0.05 * ent_range
                ent_max += 0.05 * ent_range

                # Create meshgrid
                xx = np.linspace(conf_min, conf_max, 100)
                yy = np.linspace(ent_min, ent_max, 100)
                X, Y = np.meshgrid(xx, yy)

                # Evaluate KDE on grid
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = kde(positions).reshape(X.shape)

                # Plot contours
                contours = ax.contour(X, Y, Z, levels=5, colors=[color],
                                     alpha=0.8, linewidths=2)
                ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

                # Plot filled contours with transparency
                ax.contourf(X, Y, Z, levels=5, colors=[color], alpha=0.2)

                # Add a dummy line for the legend
                ax.plot([], [], color=color, linewidth=3, label=model_name)

            except Exception as e:
                logger.warning(f"Could not create density contours for {model_name}: {e}")
                # Fallback: plot a simple scatter with low alpha
                ax.scatter(confidence, entropy, color=color, alpha=0.3,
                          s=20, label=model_name)

        ax.set_xlabel('Confidence (Max Probability)')
        ax.set_ylabel('Entropy')
        ax.set_title('Uncertainty Landscape (Density Contours)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)

    # ------------------------------------------------------------------------------
    # Enhanced Information Flow Analysis
    # ------------------------------------------------------------------------------

    def analyze_information_flow(self, data: DataInput) -> None:
        """Analyze information flow through network, including activation patterns."""
        logger.info("Analyzing information flow and activations...")

        # Get sample data
        x_sample = data.x_data[:min(200, len(data.x_data))]

        for model_name, extraction_data in self.layer_extraction_models.items():
            if extraction_data is None:
                logger.warning(f"No extraction data available for model {model_name}")
                continue

            try:
                # Get multi-layer outputs
                layer_outputs = extraction_data['model'].predict(x_sample, verbose=0)
                layer_info = extraction_data['layer_info']

                # Analyze each layer
                layer_analysis = {}
                for i, (output, info) in enumerate(zip(layer_outputs, layer_info)):
                    analysis = self._analyze_layer_information(output, info)
                    layer_analysis[info['name']] = analysis

                self.results.information_flow[model_name] = layer_analysis

                # Store detailed activation stats for key layers
                self._analyze_key_layer_activations(model_name, layer_outputs, layer_info)
            except Exception as e:
                logger.error(f"Failed to analyze information flow for {model_name}: {e}")
                continue

        # Create visualizations
        self._plot_enhanced_information_flow()

    def _analyze_layer_information(self, output: np.ndarray, layer_info: Dict) -> Dict[str, Any]:
        """Enhanced analysis of layer information content."""
        # Flatten spatial dimensions if needed
        if len(output.shape) == 4:  # Conv layer
            output_flat = np.mean(output, axis=(1, 2))
            spatial_output = output
        else:
            output_flat = output
            spatial_output = None

        # Compute statistics
        analysis = {
            'layer_type': layer_info['type'],
            'output_shape': output.shape,
            'mean_activation': float(np.mean(output_flat)),
            'std_activation': float(np.std(output_flat)),
            'sparsity': float(np.mean(np.abs(output_flat) < 1e-5)),
            'positive_ratio': float(np.mean(output_flat > 0)),
        }

        # Add spatial statistics for conv layers
        if spatial_output is not None:
            analysis['spatial_mean'] = float(np.mean(spatial_output))
            analysis['spatial_std'] = float(np.std(spatial_output))
            analysis['channel_variance'] = float(np.var(np.mean(spatial_output, axis=(0, 1, 2))))

        # Compute effective rank
        if output_flat.shape[1] > 1:
            try:
                _, s, _ = np.linalg.svd(output_flat, full_matrices=False)
                s_normalized = s / np.sum(s)
                effective_rank = np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-10)))
                analysis['effective_rank'] = float(effective_rank)
            except:
                analysis['effective_rank'] = 0.0

        return analysis

    def _analyze_key_layer_activations(self, model_name: str, layer_outputs: List[np.ndarray],
                                     layer_info: List[Dict]) -> None:
        """Analyze activations for key layers in detail."""
        # Find key layers (last conv and middle dense)
        conv_indices = [i for i, info in enumerate(layer_info) if info['type'] == 'Conv2D']
        dense_indices = [i for i, info in enumerate(layer_info) if info['type'] == 'Dense']

        key_indices = []
        if conv_indices:
            key_indices.append(conv_indices[-1])  # Last conv layer
        if dense_indices and len(dense_indices) > 1:
            key_indices.append(dense_indices[len(dense_indices)//2])  # Middle dense layer

        self.results.activation_stats[model_name] = {}

        for idx in key_indices:
            if idx < len(layer_outputs):
                layer_name = layer_info[idx]['name']
                activations = layer_outputs[idx]

                flat_acts = activations.flatten()

                self.results.activation_stats[model_name][layer_name] = {
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
                    'sample_activations': activations[:10] if len(activations.shape) == 4 else None
                }

    def _plot_enhanced_information_flow(self) -> None:
        """Create enhanced information flow visualizations."""
        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Top row: Flow overview
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_activation_flow_overview(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_effective_rank_evolution(ax2)

        # Bottom row: Actionable insights
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_activation_health_dashboard(ax3)

        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_layer_specialization_analysis(ax4)

        plt.suptitle('Information Flow and Activation Analysis', fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'information_flow_analysis')
        plt.close(fig)

    def _plot_activation_flow_overview(self, ax) -> None:
        """Plot activation statistics evolution through layers."""
        for model_name in sorted(self.results.information_flow.keys()):
            layer_analysis = self.results.information_flow[model_name]
            means = []
            stds = []
            layer_positions = []

            for i, (layer_name, analysis) in enumerate(layer_analysis.items()):
                means.append(analysis['mean_activation'])
                stds.append(analysis['std_activation'])
                layer_positions.append(i)

            # Plot mean with std as shaded region
            means = np.array(means)
            stds = np.array(stds)

            color = self.model_colors.get(model_name, '#333333')
            line = ax.plot(layer_positions, means, 'o-', label=f'{model_name}',
                          linewidth=2, markersize=6, color=color)
            ax.fill_between(layer_positions, means - stds, means + stds,
                           alpha=0.2, color=color)

        ax.set_xlabel('Layer Depth')
        ax.set_ylabel('Activation Statistics')
        ax.set_title('Activation Mean ± Std Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_effective_rank_evolution(self, ax) -> None:
        """Plot effective rank evolution through network."""
        for model_name in sorted(self.results.information_flow.keys()):
            layer_analysis = self.results.information_flow[model_name]
            ranks = []
            positions = []

            for i, (layer_name, analysis) in enumerate(layer_analysis.items()):
                if 'effective_rank' in analysis and analysis['effective_rank'] > 0:
                    ranks.append(analysis['effective_rank'])
                    positions.append(i)

            if ranks:
                color = self.model_colors.get(model_name, '#333333')
                ax.plot(positions, ranks, 'o-', label=model_name,
                       linewidth=2, markersize=8, color=color)

        ax.set_xlabel('Layer Depth')
        ax.set_ylabel('Effective Rank')
        ax.set_title('Information Dimensionality Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_activation_health_dashboard(self, ax) -> None:
        """Create an activation health dashboard showing model health metrics."""
        if not self.results.information_flow:
            ax.text(0.5, 0.5, 'No activation data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Activation Health Dashboard')
            ax.axis('off')
            return

        # Prepare health metrics data
        health_data = []

        for model_name in sorted(self.results.information_flow.keys()):
            layer_analysis = self.results.information_flow[model_name]

            for layer_name, analysis in layer_analysis.items():
                # Calculate health metrics
                sparsity = analysis.get('sparsity', 0.0)
                positive_ratio = analysis.get('positive_ratio', 0.5)
                mean_activation = abs(analysis.get('mean_activation', 0.0))

                # Health indicators
                dead_neurons = sparsity  # High sparsity indicates dead neurons
                saturation = 1.0 - positive_ratio if positive_ratio > 0.9 else 0.0  # Very high positive ratio suggests saturation
                activation_magnitude = min(mean_activation, ACTIVATION_MAGNITUDE_NORMALIZER) / ACTIVATION_MAGNITUDE_NORMALIZER

                health_data.append({
                    'Model': model_name,
                    'Layer': layer_name.split('_')[0] if '_' in layer_name else layer_name[:8],
                    'Dead Neurons': dead_neurons,
                    'Saturation': saturation,
                    'Activation Level': activation_magnitude
                })

        if health_data:
            df = pd.DataFrame(health_data)

            # Create heatmap data
            models = sorted(df['Model'].unique())
            layers = df['Layer'].unique()[:8]  # Limit to first 8 layers for readability

            # Create separate heatmaps for each metric
            metrics = ['Dead Neurons', 'Saturation', 'Activation Level']

            # Create subplots within the main axis
            gs_sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(),
                                           wspace=0.4, hspace=0.1)

            for idx, metric in enumerate(metrics):
                ax_sub = plt.subplot(gs_sub[0, idx])

                # Create pivot table for heatmap
                df_filtered = df[df['Layer'].isin(layers)]
                heatmap_data = df_filtered.pivot_table(values=metric, index='Model', columns='Layer', fill_value=0)

                # Ensure we have the right models in the right order
                heatmap_data = heatmap_data.reindex(models, fill_value=0)

                # Choose colormap based on metric
                if metric == 'Dead Neurons':
                    cmap = 'Reds'  # Red = bad (more dead neurons)
                    vmax = 1.0
                elif metric == 'Saturation':
                    cmap = 'Oranges'  # Orange = bad (more saturation)
                    vmax = 1.0
                else:  # Activation Level
                    cmap = 'Greens'  # Green = good (healthy activation)
                    vmax = 1.0

                # Create heatmap
                im = ax_sub.imshow(heatmap_data.values, cmap=cmap, aspect='auto',
                                  vmin=0, vmax=vmax, interpolation='nearest')

                # Set labels
                ax_sub.set_title(metric, fontsize=10, fontweight='bold')
                ax_sub.set_xticks(range(len(heatmap_data.columns)))
                ax_sub.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=8)
                ax_sub.set_yticks(range(len(heatmap_data.index)))

                # Only show model names on the leftmost heatmap
                if idx == 0:
                    ax_sub.set_yticklabels(heatmap_data.index, fontsize=8)
                else:
                    ax_sub.set_yticklabels([])

                # Add colorbar
                from matplotlib import cm
                cbar = plt.colorbar(im, ax=ax_sub, shrink=0.6)
                cbar.ax.tick_params(labelsize=7)

            ax.axis('off')  # Hide the parent axis
            ax.set_title('Activation Health Dashboard', fontsize=12, fontweight='bold', pad=20)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for health analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Activation Health Dashboard')
            ax.axis('off')

    def _plot_layer_specialization_analysis(self, ax) -> None:
        """Analyze how specialized each model's layers have become."""
        if not self.results.information_flow:
            ax.text(0.5, 0.5, 'No activation data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Layer Specialization Analysis')
            ax.axis('off')
            return

        # Calculate specialization metrics for each model
        specialization_data = []

        for model_name in sorted(self.results.information_flow.keys()):
            layer_analysis = self.results.information_flow[model_name]

            total_specialization = 0
            layer_count = 0
            layer_specializations = []

            for layer_name, analysis in layer_analysis.items():
                # Specialization indicators:
                # 1. Low sparsity (neurons are active)
                # 2. Balanced positive ratio (not all saturated)
                # 3. Good effective rank (diverse representations)

                sparsity = analysis.get('sparsity', 1.0)
                positive_ratio = analysis.get('positive_ratio', 0.5)
                effective_rank = analysis.get('effective_rank', 1.0)

                # Calculate specialization score (0-1, higher is better)
                # Low sparsity is good (1 - sparsity)
                activation_health = 1.0 - sparsity

                # Balanced activation is good (closer to 0.5 is better)
                balance_score = 1.0 - abs(positive_ratio - 0.5) * 2

                # Higher effective rank is good (normalize by a reasonable max)
                rank_score = min(effective_rank / LAYER_SPECIALIZATION_MAX_RANK, 1.0) if effective_rank > 0 else 0.0

                # Combined specialization score
                layer_spec = (activation_health + balance_score + rank_score) / 3.0
                layer_specializations.append(layer_spec)

                total_specialization += layer_spec
                layer_count += 1

            if layer_count > 0:
                avg_specialization = total_specialization / layer_count
                specialization_data.append({
                    'Model': model_name,
                    'Average Specialization': avg_specialization,
                    'Layer Specializations': layer_specializations[:10]  # Limit to first 10 layers
                })

        if specialization_data:
            # Create two sub-visualizations
            gs_sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(),
                                           hspace=0.4, height_ratios=[1, 1.5])

            # Top: Overall specialization comparison
            ax_top = plt.subplot(gs_sub[0, 0])
            models = [d['Model'] for d in specialization_data]
            avg_specs = [d['Average Specialization'] for d in specialization_data]

            bars = ax_top.bar(range(len(models)), avg_specs, alpha=0.8)

            # Color bars with model colors
            for i, model in enumerate(models):
                color = self.model_colors.get(model, '#333333')
                bars[i].set_facecolor(color)

            ax_top.set_title('Overall Model Specialization', fontsize=10, fontweight='bold')
            ax_top.set_ylabel('Specialization Score')
            ax_top.set_xticks(range(len(models)))
            ax_top.set_xticklabels([])  # Remove model names from x-axis
            ax_top.grid(True, alpha=0.3, axis='y')
            ax_top.set_ylim(0, 1)

            # Add value labels on bars
            for i, v in enumerate(avg_specs):
                ax_top.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

            # Bottom: Layer-by-layer specialization evolution
            ax_bottom = plt.subplot(gs_sub[1, 0])

            for data in specialization_data:
                model_name = data['Model']
                layer_specs = data['Layer Specializations']
                color = self.model_colors.get(model_name, '#333333')

                x_positions = range(len(layer_specs))
                ax_bottom.plot(x_positions, layer_specs, 'o-',
                             label=model_name, color=color, linewidth=2, markersize=6)

            ax_bottom.set_title('Layer-wise Specialization Evolution', fontsize=10, fontweight='bold')
            ax_bottom.set_xlabel('Layer Index')
            ax_bottom.set_ylabel('Specialization Score')
            ax_bottom.legend(fontsize=8)
            ax_bottom.grid(True, alpha=0.3)
            ax_bottom.set_ylim(0, 1)

            ax.axis('off')  # Hide parent axis
            ax.set_title('Layer Specialization Analysis', fontsize=12, fontweight='bold', pad=20)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for specialization analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Layer Specialization Analysis')
            ax.axis('off')

    # ------------------------------------------------------------------------------
    # NEW: Training Dynamics Analysis
    # ------------------------------------------------------------------------------

    def analyze_training_dynamics(self) -> None:
        """Analyze training history to understand how models learned."""
        logger.info("Analyzing training dynamics...")

        # Initialize training metrics container
        self.results.training_metrics = TrainingMetrics()

        for model_name, history in self.results.training_history.items():
            if not history:
                logger.warning(f"No training history available for {model_name}")
                continue

            # Compute quantitative metrics
            self._compute_training_metrics(model_name, history)

            # Apply smoothing if requested
            if self.config.smooth_training_curves:
                self._smooth_training_curves(model_name, history)

        # Create visualizations
        self._plot_training_dynamics()

    def _compute_training_metrics(self, model_name: str, history: Dict[str, List[float]]) -> None:
        """Compute quantitative metrics from training history."""
        metrics = self.results.training_metrics

        # Extract metrics using flexible pattern matching
        train_loss = find_metric_in_history(history, LOSS_PATTERNS)
        val_loss = find_metric_in_history(history, VAL_LOSS_PATTERNS)
        train_acc = find_metric_in_history(history, ACC_PATTERNS)
        val_acc = find_metric_in_history(history, VAL_ACC_PATTERNS)

        # Epochs to convergence (95% of max validation accuracy)
        if val_acc:
            max_val_acc = max(val_acc)
            threshold = CONVERGENCE_THRESHOLD * max_val_acc
            epochs_to_conv = next((i for i, acc in enumerate(val_acc) if acc >= threshold), len(val_acc))
            metrics.epochs_to_convergence[model_name] = epochs_to_conv

        # Training stability score (lower is more stable)
        if val_loss and len(val_loss) > TRAINING_STABILITY_WINDOW:
            recent_losses = val_loss[-TRAINING_STABILITY_WINDOW:]
            stability_score = np.std(recent_losses)
            metrics.training_stability_score[model_name] = stability_score

        # Overfitting index
        if train_loss and val_loss:
            n_epochs = len(train_loss)
            final_third_start = int(n_epochs * (1 - OVERFITTING_ANALYSIS_FRACTION))
            
            train_final = np.mean(train_loss[final_third_start:])
            val_final = np.mean(val_loss[final_third_start:])
            overfitting_index = val_final - train_final
            
            metrics.overfitting_index[model_name] = overfitting_index
            metrics.final_gap[model_name] = val_loss[-1] - train_loss[-1]

        # Peak performance
        if val_acc:
            best_epoch = np.argmax(val_acc)
            metrics.peak_performance[model_name] = {
                'epoch': best_epoch,
                'val_accuracy': val_acc[best_epoch],
                'val_loss': val_loss[best_epoch] if val_loss and best_epoch < len(val_loss) else None
            }
        
        # Log warning if no metrics found
        if not any([train_loss, val_loss, train_acc, val_acc]):
            logger.warning(f"No recognized training metrics found for {model_name}. Available keys: {list(history.keys())}")

    def _smooth_training_curves(self, model_name: str, history: Dict[str, List[float]]) -> None:
        """Apply smoothing to training curves for cleaner visualization."""
        smoothed = {}
        
        for metric_name, values in history.items():
            if isinstance(values, list) and len(values) > self.config.smoothing_window:
                smoothed_values = smooth_curve(np.array(values), self.config.smoothing_window)
                smoothed[metric_name] = smoothed_values
            else:
                smoothed[metric_name] = values
                
        self.results.training_metrics.smoothed_curves[model_name] = smoothed

    def _plot_training_dynamics(self) -> None:
        """Create comprehensive training dynamics visualizations."""
        fig = plt.figure(figsize=(16, 12))
        gs = plt.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                         height_ratios=[1, 1, 0.8])

        # 1. Loss curves (train and validation)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_loss_curves(ax1)

        # 2. Accuracy curves (train and validation)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_accuracy_curves(ax2)

        # 3. Overfitting analysis
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_overfitting_analysis(ax3)

        # 4. Best epoch performance
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_best_epoch_performance(ax4)

        # 5. Training summary table
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_training_summary_table(ax5)

        plt.suptitle('Training Dynamics Analysis', fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.94, bottom=0.05, left=0.08, right=0.96)

        if self.config.save_plots:
            self._save_figure(fig, 'training_dynamics')
        plt.close(fig)

    def _plot_loss_curves(self, ax) -> None:
        """Plot training and validation loss curves."""
        for model_name in sorted(self.results.training_history.keys()):
            history = self.results.training_history[model_name]
            color = self.model_colors.get(model_name, '#333333')

            # Use smoothed curves if available
            if self.config.smooth_training_curves and model_name in self.results.training_metrics.smoothed_curves:
                curves = self.results.training_metrics.smoothed_curves[model_name]
            else:
                curves = history

            # Plot training loss using flexible pattern matching
            if curves == history:
                train_loss = find_metric_in_history(curves, LOSS_PATTERNS)
            else:
                train_loss = curves.get('loss', find_metric_in_history(curves, LOSS_PATTERNS))
                
            if train_loss:
                epochs = range(len(train_loss))
                ax.plot(epochs, train_loss, '-', color=color, 
                       linewidth=2, label=f'{model_name} (train)', alpha=0.8)

            # Plot validation loss using flexible pattern matching
            if curves == history:
                val_loss = find_metric_in_history(curves, VAL_LOSS_PATTERNS)
            else:
                val_loss = curves.get('val_loss', find_metric_in_history(curves, VAL_LOSS_PATTERNS))
                
            if val_loss:
                epochs = range(len(val_loss))
                ax.plot(epochs, val_loss, '--', color=color, 
                       linewidth=2, label=f'{model_name} (val)', alpha=0.8)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale often better for loss

    def _plot_accuracy_curves(self, ax) -> None:
        """Plot training and validation accuracy curves."""
        for model_name in sorted(self.results.training_history.keys()):
            history = self.results.training_history[model_name]
            color = self.model_colors.get(model_name, '#333333')

            # Use smoothed curves if available
            if self.config.smooth_training_curves and model_name in self.results.training_metrics.smoothed_curves:
                curves = self.results.training_metrics.smoothed_curves[model_name]
            else:
                curves = history

            # Plot training accuracy using flexible pattern matching
            if curves == history:
                train_acc = find_metric_in_history(curves, ACC_PATTERNS)
            else:
                train_acc = curves.get('accuracy', find_metric_in_history(curves, ACC_PATTERNS))
                
            if train_acc:
                epochs = range(len(train_acc))
                ax.plot(epochs, train_acc, '-', color=color, 
                       linewidth=2, label=f'{model_name} (train)', alpha=0.8)

            # Plot validation accuracy using flexible pattern matching
            if curves == history:
                val_acc = find_metric_in_history(curves, VAL_ACC_PATTERNS)
            else:
                val_acc = curves.get('val_accuracy', find_metric_in_history(curves, VAL_ACC_PATTERNS))
                
            if val_acc:
                epochs = range(len(val_acc))
                ax.plot(epochs, val_acc, '--', color=color, 
                       linewidth=2, label=f'{model_name} (val)', alpha=0.8)

                # Mark best epoch
                if model_name in self.results.training_metrics.peak_performance:
                    best_epoch = self.results.training_metrics.peak_performance[model_name]['epoch']
                    best_acc = self.results.training_metrics.peak_performance[model_name]['val_accuracy']
                    ax.scatter(best_epoch, best_acc, color=color, s=100, 
                             marker='*', edgecolor='black', linewidth=1, zorder=5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    def _plot_overfitting_analysis(self, ax) -> None:
        """Plot dedicated overfitting analysis."""
        overfitting_data = []

        for model_name in sorted(self.results.training_history.keys()):
            history = self.results.training_history[model_name]
            
            train_loss = find_metric_in_history(history, LOSS_PATTERNS)
            val_loss = find_metric_in_history(history, VAL_LOSS_PATTERNS)
            
            if train_loss and val_loss and len(train_loss) == len(val_loss):
                # Calculate gap over time
                gap = np.array(val_loss) - np.array(train_loss)
                
                color = self.model_colors.get(model_name, '#333333')
                epochs = range(len(gap))
                
                # Apply smoothing to gap if requested
                if self.config.smooth_training_curves:
                    gap_smooth = smooth_curve(gap, self.config.smoothing_window)
                    ax.plot(epochs, gap_smooth, '-', color=color, 
                           linewidth=2.5, label=model_name)
                else:
                    ax.plot(epochs, gap, '-', color=color, 
                           linewidth=2, label=model_name, alpha=0.8)
                
                # Add shaded region for positive gap (overfitting)
                ax.fill_between(epochs, 0, gap, where=(gap > 0), 
                               color=color, alpha=0.1)

        # Add reference line at 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss - Training Loss')
        ax.set_title('Overfitting Analysis (Gap Evolution)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        ax.text(0.02, 0.98, 'Above 0 = Overfitting\nBelow 0 = Underfitting',
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
               fontsize=9)

    def _plot_best_epoch_performance(self, ax) -> None:
        """Plot best epoch performance comparison."""
        peak_data = self.results.training_metrics.peak_performance
        
        if not peak_data:
            ax.text(0.5, 0.5, 'No peak performance data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Best Epoch Performance')
            ax.axis('off')
            return

        models = sorted(peak_data.keys())
        best_accs = [peak_data[m]['val_accuracy'] for m in models]
        best_epochs = [peak_data[m]['epoch'] for m in models]
        
        # Create grouped bar chart
        x = np.arange(len(models))
        width = 0.35
        
        # Accuracy bars
        bars1 = ax.bar(x - width/2, best_accs, width, label='Best Val Accuracy', alpha=0.8)
        
        # Normalize epochs for visualization
        max_epoch = max(best_epochs) if best_epochs else 1
        normalized_epochs = [e / max_epoch for e in best_epochs]
        bars2 = ax.bar(x + width/2, normalized_epochs, width, label='Epoch (normalized)', alpha=0.8)
        
        # Color bars by model
        for i, model in enumerate(models):
            color = self.model_colors.get(model, '#333333')
            bars1[i].set_facecolor(color)
            bars2[i].set_facecolor(color)
        
        # Add value labels
        for i, (acc, epoch) in enumerate(zip(best_accs, best_epochs)):
            ax.text(i - width/2, acc + 0.01, f'{acc:.3f}', 
                   ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, normalized_epochs[i] + 0.01, f'{epoch}', 
                   ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Value')
        ax.set_title('Peak Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)

    def _plot_training_summary_table(self, ax) -> None:
        """Create comprehensive training summary table."""
        # Prepare data
        table_data = []
        headers = ['Model', 'Final Acc', 'Best Acc', 'Best Epoch', 'Conv. Speed', 
                  'Stability', 'Overfit Index', 'Final Gap']
        
        for model_name in sorted(self.models.keys()):
            row = [model_name]
            
            # Final accuracy
            history = self.results.training_history.get(model_name, {})
            val_acc = find_metric_in_history(history, VAL_ACC_PATTERNS)
            final_acc = val_acc[-1] if val_acc else 0.0
            row.append(f'{final_acc:.3f}')
            
            # Best accuracy and epoch
            peak = self.results.training_metrics.peak_performance.get(model_name, {})
            best_acc = peak.get('val_accuracy', 0.0)
            best_epoch = peak.get('epoch', 0)
            row.append(f'{best_acc:.3f}')
            row.append(f'{best_epoch}')
            
            # Convergence speed
            conv_speed = self.results.training_metrics.epochs_to_convergence.get(model_name, 0)
            row.append(f'{conv_speed}')
            
            # Stability score
            stability = self.results.training_metrics.training_stability_score.get(model_name, 0.0)
            row.append(f'{stability:.3f}')
            
            # Overfitting index
            overfit = self.results.training_metrics.overfitting_index.get(model_name, 0.0)
            row.append(f'{overfit:.3f}')
            
            # Final gap
            final_gap = self.results.training_metrics.final_gap.get(model_name, 0.0)
            row.append(f'{final_gap:.3f}')
            
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
            table[(0, i)].set_height(0.08)
        
        # Color model rows
        for i, row_data in enumerate(table_data, 1):
            model_name = row_data[0]
            color = self.model_colors.get(model_name, '#F5F5F5')
            light_color = self._lighten_color(color, 0.8)
            
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor(light_color)
                cell.set_height(0.08)
                
                if j == 0:  # Model name
                    cell.set_text_props(weight='bold', fontsize=9)
                else:
                    cell.set_text_props(fontsize=9)
        
        ax.axis('off')
        ax.set_title('Training Metrics Summary', fontsize=12, fontweight='bold', pad=10)

    # ------------------------------------------------------------------------------
    # Enhanced Summary Dashboard
    # ------------------------------------------------------------------------------

    def create_summary_dashboard(self) -> None:
        """Create an enhanced summary dashboard with training insights."""
        fig = plt.figure(figsize=(16, 10))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                         height_ratios=[1, 1], width_ratios=[1.2, 1])

        # 1. Enhanced Performance Table (with training metrics)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_enhanced_performance_table(ax1)

        # 2. Model Similarity (unchanged)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_model_similarity(ax2)

        # 3. Confidence Distribution Profiles (unchanged)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_confidence_profile_summary(ax3)

        # 4. Calibration Performance Comparison (unchanged)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_calibration_performance_summary(ax4)

        plt.suptitle('Model Analysis Summary Dashboard', fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.96)

        if self.config.save_plots:
            self._save_figure(fig, 'enhanced_summary_dashboard')
        plt.close(fig)

    def _plot_enhanced_performance_table(self, ax) -> None:
        """Create enhanced performance table including training metrics."""
        # Prepare data for the table
        table_data = []
        
        # Adjust headers based on whether we have training data
        if self.results.training_metrics and self.results.training_metrics.peak_performance:
            headers = ['Model', 'Final Acc', 'Best Acc', 'Loss', 'ECE', 'Brier', 'Conv Speed', 'Overfit']
        else:
            headers = ['Model', 'Accuracy', 'Loss', 'ECE', 'Brier Score', 'Mean Entropy']

        for model_name in sorted(self.models.keys()):
            row_data = [model_name]

            # Get model metrics
            model_metrics = self.results.model_metrics.get(model_name, {})
            
            if self.results.training_metrics and self.results.training_metrics.peak_performance:
                # Include training insights
                # Final accuracy
                acc = (model_metrics.get('accuracy', 0.0) or
                      model_metrics.get('compile_metrics', 0.0) or
                      model_metrics.get('val_accuracy', 0.0) or 0.0)
                row_data.append(f'{acc:.3f}')
                
                # Best accuracy from training
                peak = self.results.training_metrics.peak_performance.get(model_name, {})
                best_acc = peak.get('val_accuracy', acc)
                row_data.append(f'{best_acc:.3f}')
                
                # Loss
                loss = model_metrics.get('loss', 0.0)
                row_data.append(f'{loss:.3f}')
                
                # ECE
                ece = self.results.calibration_metrics.get(model_name, {}).get('ece', 0.0)
                row_data.append(f'{ece:.3f}')
                
                # Brier Score
                brier = self.results.calibration_metrics.get(model_name, {}).get('brier_score', 0.0)
                row_data.append(f'{brier:.3f}')
                
                # Convergence speed
                conv_speed = self.results.training_metrics.epochs_to_convergence.get(model_name, 0)
                row_data.append(f'{conv_speed}')
                
                # Overfitting index
                overfit = self.results.training_metrics.overfitting_index.get(model_name, 0.0)
                row_data.append(f'{overfit:+.3f}')  # Show sign
                
            else:
                # Original table without training data
                # Accuracy
                acc = (model_metrics.get('accuracy', 0.0) or
                      model_metrics.get('compile_metrics', 0.0) or
                      model_metrics.get('val_accuracy', 0.0) or 0.0)
                row_data.append(f'{acc:.3f}')

                # Loss
                loss = model_metrics.get('loss', 0.0)
                row_data.append(f'{loss:.3f}')

                # ECE
                ece = self.results.calibration_metrics.get(model_name, {}).get('ece', 0.0)
                row_data.append(f'{ece:.3f}')

                # Brier Score
                brier = self.results.calibration_metrics.get(model_name, {}).get('brier_score', 0.0)
                row_data.append(f'{brier:.3f}')

                # Mean Entropy
                entropy = self.results.calibration_metrics.get(model_name, {}).get('mean_entropy', 0.0)
                row_data.append(f'{entropy:.3f}')

            table_data.append(row_data)

        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Color the header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
            table[(0, i)].set_height(0.08)

        # Color the model rows
        for i, row_data in enumerate(table_data, 1):
            model_name = row_data[0]
            color = self.model_colors.get(model_name, '#F5F5F5')

            # Apply light version of model color to the entire row
            light_color = self._lighten_color(color, 0.8)
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor(light_color)
                cell.set_height(0.08)

                # Ensure text fits properly
                if j == 0:  # Model name column
                    cell.set_text_props(weight='bold', fontsize=9)
                else:  # Metric columns
                    cell.set_text_props(fontsize=9)

        # Remove axis
        ax.axis('off')

    def _plot_calibration_performance_summary(self, ax) -> None:
        """Plot calibration performance comparison."""
        if not self.results.calibration_metrics:
            ax.text(0.5, 0.5, 'No calibration data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Calibration Performance')
            ax.axis('off')
            return

        # Prepare data for plotting
        models = sorted(self.results.calibration_metrics.keys())
        ece_values = []
        brier_values = []

        for model_name in models:
            metrics = self.results.calibration_metrics[model_name]
            ece_values.append(metrics.get('ece', 0.0))
            brier_values.append(metrics.get('brier_score', 0.0))

        # Create a scatter plot showing ECE vs Brier Score
        for i, model_name in enumerate(models):
            color = self.model_colors.get(model_name, '#333333')
            ax.scatter(ece_values[i], brier_values[i],
                      s=200, color=color, alpha=0.8,
                      edgecolors='black', linewidth=2,
                      label=model_name)

        # Add reference lines
        if ece_values and brier_values:
            # Perfect calibration line (diagonal)
            max_val = max(max(ece_values), max(brier_values))
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5,
                   label='Perfect Correlation')

            # Add quadrants for interpretation
            mean_ece = np.mean(ece_values)
            mean_brier = np.mean(brier_values)

            ax.axvline(mean_ece, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(mean_brier, color='gray', linestyle=':', alpha=0.5)

            # Add quadrant labels
            ax.text(0.02, 0.98, 'Well Calibrated\nLow Uncertainty',
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                   fontsize=8)

            ax.text(0.98, 0.02, 'Poorly Calibrated\nHigh Uncertainty',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                   fontsize=8)

        ax.set_xlabel('Expected Calibration Error (ECE)')
        ax.set_ylabel('Brier Score')
        ax.set_title('Calibration Performance Landscape')
        ax.grid(True, alpha=0.3)

        # Set axis limits to start from 0
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    def _plot_model_similarity(self, ax) -> None:
        """Plot model similarity based on weight PCA."""
        if not self.results.weight_pca:
            ax.text(0.5, 0.5, 'No weight PCA data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Similarity (Weight Space)')
            ax.axis('off')
            return

        components = self.results.weight_pca['components']
        labels = self.results.weight_pca['labels']
        explained_var = self.results.weight_pca['explained_variance']

        # Validate PCA components
        if len(components) == 0 or len(components[0]) < 2:
            ax.text(0.5, 0.5, 'Insufficient PCA components for visualization',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Similarity (Weight Space)')
            ax.axis('off')
            return

        # Create enhanced scatter plot using consistent colors
        for i, (label, comp) in enumerate(zip(labels, components)):
            color = self.model_colors.get(label, '#333333')
            ax.scatter(comp[0], comp[1], c=[color], label=label,
                      s=200, alpha=0.8, edgecolors='black', linewidth=2)

            # Add connecting lines to origin
            ax.plot([0, comp[0]], [0, comp[1]], '--', color=color, alpha=0.3)

        # Add origin
        ax.scatter(0, 0, c='black', s=50, marker='x')

        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})' if len(explained_var) > 1 else 'PC2')
        ax.set_title('Model Similarity (Weight Space)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    def _plot_confidence_profile_summary(self, ax) -> None:
        """Plot confidence distribution summary."""
        confidence_data = []

        for model_name, metrics in self.results.confidence_metrics.items():
            # Safety check for required keys
            if 'max_probability' not in metrics:
                logger.warning(f"Missing 'max_probability' key for model {model_name}")
                continue

            for conf in metrics['max_probability']:
                confidence_data.append({
                    'Model': model_name,
                    'Confidence': conf
                })

        if confidence_data:
            df = pd.DataFrame(confidence_data)
            model_order = sorted(df['Model'].unique())

            # Create enhanced violin plot
            parts = ax.violinplot([df[df['Model'] == m]['Confidence'].values
                                  for m in model_order],
                                 positions=range(len(model_order)),
                                 showmeans=True, showmedians=True)

            # Customize colors using consistent palette
            for i, model in enumerate(model_order):
                color = self.model_colors.get(model, '#333333')
                parts['bodies'][i].set_facecolor(color)
                parts['bodies'][i].set_alpha(0.6)

            ax.set_xticks(range(len(model_order)))
            ax.set_xticklabels([])  # Remove model names from x-axis
            ax.set_ylabel('Confidence (Max Probability)')
            ax.set_title('Confidence Distribution Profiles')
            ax.grid(True, alpha=0.3, axis='y')

    # ------------------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------------------

    def _lighten_color(self, color: str, factor: float) -> str:
        """Lighten a color by interpolating towards white."""
        import matplotlib.colors as mcolors

        # Convert color to RGB
        rgb = mcolors.to_rgb(color)

        # Interpolate towards white
        lightened = tuple(rgb[i] + (1 - rgb[i]) * factor for i in range(3))

        return lightened

    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure with configured settings."""
        try:
            filepath = self.output_dir / f"{name}.{self.config.save_format}"
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none', pad_inches=0.1)
            logger.info(f"Saved plot: {filepath}")
        except Exception as e:
            logger.error(f"Could not save figure {name}: {e}")

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
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()
                       if k not in ['raw_activations', 'sample_activations']}
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
        pareto_indices = self._find_pareto_front(
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
            self._normalize_metric(peak_accuracies, higher_better=True),
            self._normalize_metric(overfitting_indices, higher_better=False),
            self._normalize_metric(convergence_speeds, higher_better=False)
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
    
    def _find_pareto_front(self, costs1: np.ndarray, costs2: np.ndarray) -> List[int]:
        """Find indices of Pareto optimal points (maximizing both objectives)."""
        population_size = len(costs1)
        pareto_indices = []
        
        for i in range(population_size):
            is_pareto = True
            for j in range(population_size):
                if i != j:
                    # Check if j dominates i (better in both objectives)
                    if costs1[j] >= costs1[i] and costs2[j] >= costs2[i]:
                        if costs1[j] > costs1[i] or costs2[j] > costs2[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_indices.append(i)
                
        return sorted(pareto_indices)
    
    def _normalize_metric(self, values: List[float], higher_better: bool = True) -> np.ndarray:
        """Normalize metric values to 0-1 range."""
        arr = np.array(values)
        if len(arr) == 0:
            return arr
            
        min_val, max_val = arr.min(), arr.max()
        if max_val == min_val:
            return np.ones_like(arr) * 0.5
            
        normalized = (arr - min_val) / (max_val - min_val)
        
        if not higher_better:
            normalized = 1 - normalized
            
        return normalized