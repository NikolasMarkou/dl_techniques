"""
Model Analyzer for Neural Networks - Enhanced Version
=====================================================

A comprehensive, modular analyzer with improved visualizations that reduce
redundancy and provide clearer insights through better plot organization.

Key Improvements:
- Merged calibration and probability analysis
- Merged activation and information flow analysis
- Redesigned summary dashboard with comparative visualizations
- Enhanced weight analysis with modern plot types
- Cleaner, more focused visualization pages

Example Usage:
    ```python
    from analyzer import ModelAnalyzer, AnalysisConfig

    # Configure analysis
    config = AnalysisConfig(
        analyze_activations=True,
        analyze_weights=True,
        analyze_calibration=True,
        plot_style='publication'
    )

    # Create analyzer
    analyzer = ModelAnalyzer(models, config=config)

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
    """Configuration for all analysis types."""

    # Analysis toggles
    analyze_activations: bool = True  # Now part of information flow
    analyze_weights: bool = True
    analyze_calibration: bool = True  # Now includes probability distributions
    analyze_probability_distributions: bool = True  # Merged with calibration
    analyze_information_flow: bool = True  # Now includes activations

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
    compute_weight_correlations: bool = True
    compute_weight_pca: bool = True

    # Calibration options
    calibration_bins: int = 10

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
    weight_correlations: Optional[pd.DataFrame] = None
    weight_pca: Optional[Dict[str, Any]] = None

    # Calibration analysis (now includes probability metrics)
    calibration_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reliability_data: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    confidence_metrics: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    # Information flow (now includes activation analysis)
    information_flow: Dict[str, Any] = field(default_factory=dict)

    # Training history (if available)
    training_history: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

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

# ------------------------------------------------------------------------------
# Enhanced Model Analyzer
# ------------------------------------------------------------------------------

class ModelAnalyzer:
    """
    Enhanced model analyzer with improved visualizations and reduced redundancy.
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

        # Set up activation extraction models
        self._setup_activation_models()

        logger.info(f"ModelAnalyzer initialized with {len(models)} models")

    def _setup_activation_models(self) -> None:
        """Set up models for extracting intermediate activations."""
        self.activation_models = {}
        self.layer_extraction_models = {}

        for model_name, model in self.models.items():
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

        # Compute correlations if requested
        if self.config.compute_weight_correlations:
            self._compute_weight_correlations()

        # Compute PCA if requested
        if self.config.compute_weight_pca:
            self._compute_weight_pca()

        # Create visualizations
        self._plot_enhanced_weight_analysis()

        # Create separate correlation clustermap if we have correlations
        if self.results.weight_correlations is not None:
            self._plot_weight_correlation_clustermap()

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

    def _compute_weight_correlations(self) -> None:
        """Compute correlations between model weight patterns."""
        model_features = {}

        for model_name, weight_stats in self.results.weight_stats.items():
            features = []
            for layer_stats in weight_stats.values():
                # Extract features, checking for validity
                feat_values = [
                    layer_stats['basic']['mean'],
                    layer_stats['basic']['std'],
                    layer_stats['norms']['l2'],
                    layer_stats['distribution']['zero_fraction']
                ]

                # Check if all features are finite
                if all(np.isfinite(v) for v in feat_values):
                    features.extend(feat_values)
                else:
                    logger.warning(f"Skipping layer with non-finite values in {model_name}")

            if features:
                model_features[model_name] = features

        if len(model_features) >= 2:
            # Ensure all feature vectors have same length
            min_len = min(len(f) for f in model_features.values())
            if min_len == 0:
                logger.warning("No valid features found for correlation computation")
                return

            aligned_features = {k: v[:min_len] for k, v in model_features.items()}

            # Create DataFrame and compute correlations
            feature_df = pd.DataFrame(aligned_features).T

            # Check for constant columns which would produce NaN correlations
            constant_cols = feature_df.columns[feature_df.nunique() == 1]
            if len(constant_cols) > 0:
                logger.warning(f"Removing {len(constant_cols)} constant columns before correlation")
                feature_df = feature_df.drop(columns=constant_cols)

            # Only compute correlation if we have valid data
            if not feature_df.empty and feature_df.shape[1] > 0:
                try:
                    corr_matrix = feature_df.corr()

                    # Replace any remaining NaN values with 0
                    if corr_matrix.isnull().any().any():
                        logger.warning("Correlation matrix contains NaN values, filling with 0")
                        corr_matrix = corr_matrix.fillna(0)

                    self.results.weight_correlations = corr_matrix
                except Exception as e:
                    logger.error(f"Failed to compute correlations: {e}")
            else:
                logger.warning("Insufficient valid features for correlation computation")

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
        """Create enhanced weight analysis visualizations."""
        fig = plt.figure(figsize=(14, 8))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

        # 1. Weight Norm Raincloud Plot (takes full top row)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_weight_norm_raincloud(ax1)

        # 2. Overall Sparsity Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_weight_sparsity_comparison(ax2)

        # 3. PCA of Final Layer Weights
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_final_layer_pca(ax3)

        plt.suptitle('Weight Distribution Analysis', fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'weight_analysis')
        plt.close(fig)

    def _plot_weight_norm_raincloud(self, ax) -> None:
        """Plot weight norm distributions using violin plots (raincloud-style)."""
        norm_data = []
        for model_name, weight_stats in self.results.weight_stats.items():
            for layer_name, stats in weight_stats.items():
                norm_data.append({
                    'Model': model_name,
                    'Layer': layer_name.split('_')[0],
                    'L2 Norm': stats['norms']['l2'],
                })

        if norm_data:
            df = pd.DataFrame(norm_data)

            # Create violin plot with inner quartiles
            # Updated API: use hue instead of palette, density_norm instead of scale
            sns.violinplot(data=df, x='L2 Norm', y='Model', ax=ax,
                          inner='quartile', hue='Model', palette=self.config.color_palette,
                          orient='h', cut=0, density_norm='width', legend=False)

            # Add strip plot on top for individual points
            sns.stripplot(data=df, x='L2 Norm', y='Model', ax=ax,
                         size=3, alpha=0.5, color='black', orient='h')

            ax.set_xlabel('L2 Norm of Layer Weights')
            ax.set_title('Weight Norm Distributions by Model')
            ax.grid(True, alpha=0.3, axis='x')

    def _plot_weight_sparsity_comparison(self, ax) -> None:
        """Plot overall sparsity comparison."""
        sparsity_data = []
        for model_name, weight_stats in self.results.weight_stats.items():
            all_zeros = [s['distribution']['zero_fraction'] for s in weight_stats.values()]
            if all_zeros:
                sparsity_data.append({
                    'Model': model_name,
                    'Mean Sparsity': np.mean(all_zeros),
                    'Std Sparsity': np.std(all_zeros)
                })

        if sparsity_data:
            df = pd.DataFrame(sparsity_data)

            # Clean bar chart
            x = np.arange(len(df))
            ax.bar(x, df['Mean Sparsity'], yerr=df['Std Sparsity'],
                   capsize=5, alpha=0.8, color=sns.color_palette(self.config.color_palette))

            ax.set_xlabel('Model')
            ax.set_ylabel('Average Sparsity')
            ax.set_title('Weight Sparsity Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(df['Model'], rotation=45 if len(df) > 3 else 0)
            ax.grid(True, alpha=0.3, axis='y')

    def _plot_final_layer_pca(self, ax) -> None:
        """Plot PCA of final layer weights."""
        if not self.results.weight_pca:
            ax.text(0.5, 0.5, 'Insufficient data for PCA analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PCA of Final Layer Weights')
            ax.axis('off')
            return

        components = self.results.weight_pca['components']
        labels = self.results.weight_pca['labels']
        explained_var = self.results.weight_pca['explained_variance']

        # Create scatter plot
        colors = plt.cm.Set3(np.arange(len(labels)))

        for i, (label, comp) in enumerate(zip(labels, components)):
            ax.scatter(comp[0], comp[1], c=[colors[i]], label=label,
                      s=100, alpha=0.8, edgecolors='black', linewidth=1)

        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        ax.set_title('PCA of Final Layer Weights')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_weight_correlation_clustermap(self) -> None:
        """Create a separate clustered heatmap for weight correlations."""
        if self.results.weight_correlations is None or len(self.results.weight_correlations) < 2:
            return

        # Check for non-finite values
        corr_matrix = self.results.weight_correlations
        if not np.all(np.isfinite(corr_matrix.values)):
            logger.warning("Correlation matrix contains non-finite values, skipping clustermap")
            return

        # Check if correlation matrix has enough variation
        if corr_matrix.shape[0] < 2 or corr_matrix.shape[1] < 2:
            logger.warning("Correlation matrix too small for clustering")
            return

        try:
            # Create clustermap
            g = sns.clustermap(corr_matrix,
                              annot=True, fmt='.2f',
                              cmap='coolwarm', center=0,
                              figsize=(8, 8),
                              cbar_kws={'label': 'Correlation'})

            g.fig.suptitle('Clustered Model Weight Pattern Correlations', y=0.98)

            if self.config.save_plots:
                self._save_figure(g.fig, 'weight_correlation_clustermap')
            plt.close(g.fig)
        except Exception as e:
            logger.warning(f"Could not create correlation clustermap: {e}")

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

            # Compute per-class ECE
            n_classes = y_pred_proba.shape[1]
            per_class_ece = []
            for c in range(n_classes):
                class_mask = y_true_idx == c
                if np.any(class_mask):
                    class_ece = compute_ece(y_true_idx[class_mask], y_pred_proba[class_mask],
                                          self.config.calibration_bins // 2)
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

        colors = plt.cm.Set3(np.arange(len(self.models)))

        for i, (model_name, rel_data) in enumerate(self.results.reliability_data.items()):
            # Plot main line
            ax.plot(rel_data['bin_centers'], rel_data['bin_accuracies'],
                   'o-', color=colors[i], label=model_name, linewidth=2, markersize=8)

            # Add shaded confidence region if we have sample counts
            if 'bin_counts' in rel_data:
                # Simple confidence interval based on binomial proportion
                counts = rel_data['bin_counts']
                props = rel_data['bin_accuracies']
                se = np.sqrt(props * (1 - props) / (counts + 1))

                ax.fill_between(rel_data['bin_centers'],
                               props - 1.96*se, props + 1.96*se,
                               alpha=0.2, color=colors[i])

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
            for conf in metrics['max_probability']:
                confidence_data.append({
                    'Model': model_name,
                    'Confidence': conf
                })

        if confidence_data:
            df = pd.DataFrame(confidence_data)

            # Create violin plot with quartiles using updated API
            sns.violinplot(data=df, y='Model', x='Confidence', ax=ax,
                          inner='quartile', hue='Model', palette=self.config.color_palette,
                          orient='h', cut=0, density_norm='width', legend=False)

            # Add summary statistics as text
            for i, model in enumerate(df['Model'].unique()):
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

            # Create grouped bar plot
            n_models = len(df['Model'].unique())
            n_classes = len(df['Class'].unique())

            x = np.arange(n_classes)
            width = 0.8 / n_models

            for i, model in enumerate(df['Model'].unique()):
                model_data = df[df['Model'] == model]
                ax.bar(x + i * width, model_data['ECE'], width,
                       label=model, alpha=0.8)

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

        # Set up colors for each model
        colors = plt.cm.Set3(np.arange(len(self.models)))

        # Plot contours for each model
        for i, (model_name, metrics) in enumerate(self.results.confidence_metrics.items()):
            confidence = metrics['max_probability']
            entropy = metrics['entropy']

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
                contours = ax.contour(X, Y, Z, levels=5, colors=[colors[i]],
                                     alpha=0.8, linewidths=2)
                ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

                # Plot filled contours with transparency
                ax.contourf(X, Y, Z, levels=5, colors=[colors[i]], alpha=0.2)

                # Add a dummy line for the legend
                ax.plot([], [], color=colors[i], linewidth=3, label=model_name)

            except Exception as e:
                logger.warning(f"Could not create density contours for {model_name}: {e}")
                # Fallback: plot a simple scatter with low alpha
                ax.scatter(confidence, entropy, color=colors[i], alpha=0.3,
                          s=20, label=model_name)

        ax.set_xlabel('Confidence (Max Probability)')
        ax.set_ylabel('Entropy')
        ax.set_title('Uncertainty Landscape (Density Contours)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)

    # ------------------------------------------------------------------------------
    # Enhanced Information Flow Analysis (includes Activations)
    # ------------------------------------------------------------------------------

    def analyze_information_flow(self, data: DataInput) -> None:
        """Analyze information flow through network, including activation patterns."""
        logger.info("Analyzing information flow and activations...")

        # Get sample data
        x_sample = data.x_data[:min(200, len(data.x_data))]

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

            # Store detailed activation stats for key layers
            self._analyze_key_layer_activations(model_name, layer_outputs, layer_info)

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

        # Bottom row: Deep dive into key layer
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_key_layer_comparison(ax3)

        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_sample_feature_maps_comparison(ax4)

        plt.suptitle('Information Flow and Activation Analysis', fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'information_flow_analysis')
        plt.close(fig)

    def _plot_activation_flow_overview(self, ax) -> None:
        """Plot activation statistics evolution through layers."""
        for model_name, layer_analysis in self.results.information_flow.items():
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

            line = ax.plot(layer_positions, means, 'o-', label=f'{model_name}',
                          linewidth=2, markersize=6)
            ax.fill_between(layer_positions, means - stds, means + stds,
                           alpha=0.2, color=line[0].get_color())

        ax.set_xlabel('Layer Depth')
        ax.set_ylabel('Activation Statistics')
        ax.set_title('Activation Mean ± Std Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_effective_rank_evolution(self, ax) -> None:
        """Plot effective rank evolution through network."""
        for model_name, layer_analysis in self.results.information_flow.items():
            ranks = []
            positions = []

            for i, (layer_name, analysis) in enumerate(layer_analysis.items()):
                if 'effective_rank' in analysis and analysis['effective_rank'] > 0:
                    ranks.append(analysis['effective_rank'])
                    positions.append(i)

            if ranks:
                ax.plot(positions, ranks, 'o-', label=model_name,
                       linewidth=2, markersize=8)

        ax.set_xlabel('Layer Depth')
        ax.set_ylabel('Effective Rank')
        ax.set_title('Information Dimensionality Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_key_layer_comparison(self, ax) -> None:
        """Compare activation distributions at a key layer across models."""
        # Find a common layer type across models
        activation_data = []

        for model_name, layer_stats in self.results.activation_stats.items():
            for layer_name, stats in layer_stats.items():
                activation_data.append({
                    'Model': model_name,
                    'Layer': layer_name.split('_')[0],
                    'Mean': stats['mean'],
                    'Std': stats['std'],
                    'Sparsity': stats['sparsity']
                })

        if activation_data:
            df = pd.DataFrame(activation_data)

            # Create violin plot for activation statistics
            metric_df = df.melt(id_vars=['Model', 'Layer'],
                               value_vars=['Mean', 'Std', 'Sparsity'],
                               var_name='Metric', value_name='Value')

            sns.violinplot(data=metric_df, x='Metric', y='Value', hue='Model',
                          ax=ax, palette=self.config.color_palette)

            ax.set_title('Key Layer Activation Statistics Comparison')
            ax.grid(True, alpha=0.3, axis='y')

    def _plot_sample_feature_maps_comparison(self, ax) -> None:
        """Compare sample feature maps across models."""
        # Find models with conv activations
        conv_samples = {}

        for model_name, layer_stats in self.results.activation_stats.items():
            for layer_name, stats in layer_stats.items():
                if stats.get('sample_activations') is not None:
                    conv_samples[model_name] = stats['sample_activations'][0]
                    break

        if len(conv_samples) >= 2:
            n_models = len(conv_samples)
            n_features = min(4, conv_samples[list(conv_samples.keys())[0]].shape[-1])

            # Create grid
            from matplotlib.gridspec import GridSpecFromSubplotSpec
            gs_sub = GridSpecFromSubplotSpec(n_models, n_features,
                                           subplot_spec=ax.get_gridspec()[1, 1],
                                           wspace=0.05, hspace=0.1)

            for i, (model_name, activations) in enumerate(conv_samples.items()):
                for j in range(n_features):
                    ax_sub = plt.subplot(gs_sub[i, j])
                    ax_sub.imshow(activations[:, :, j], cmap='viridis', aspect='auto')
                    ax_sub.axis('off')

                    if j == 0:
                        ax_sub.set_ylabel(model_name[:10], rotation=0,
                                         ha='right', va='center', fontsize=8)
                    if i == 0:
                        ax_sub.set_title(f'F{j}', fontsize=8)

            ax.set_visible(False)
            ax.text(0.5, 1.05, 'Sample Feature Map Comparison',
                   transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient convolutional activations\nfor comparison',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Map Comparison')
            ax.axis('off')

    # ------------------------------------------------------------------------------
    # Enhanced Summary Dashboard
    # ------------------------------------------------------------------------------

    def create_summary_dashboard(self) -> None:
        """Create a focused, visual summary dashboard."""
        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Overall Performance Radar Chart
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        self._plot_performance_radar(ax1)

        # 2. Model Similarity (Weight PCA)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_model_similarity(ax2)

        # 3. Confidence Profile
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_confidence_profile_summary(ax3)

        # 4. Training Dynamics (if available)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_training_dynamics(ax4)

        plt.suptitle('Model Analysis Summary Dashboard', fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.93, bottom=0.07, left=0.1, right=0.95)

        if self.config.save_plots:
            self._save_figure(fig, 'summary_dashboard')
        plt.close(fig)

    def _plot_performance_radar(self, ax) -> None:
        """Create radar chart for multi-metric comparison."""
        categories = ['Accuracy', 'ECE\n(inverted)', 'Brier\n(inverted)',
                     'Entropy\n(normalized)', 'Loss\n(inverted)']

        # Prepare data
        radar_data = {}
        for model_name in self.models:
            values = []

            # Accuracy
            acc = self.results.model_metrics.get(model_name, {}).get('accuracy', 0.5)
            values.append(acc)

            # ECE (inverted so higher is better)
            ece = self.results.calibration_metrics.get(model_name, {}).get('ece', 0.1)
            values.append(1 - min(ece, 1))

            # Brier Score (inverted)
            brier = self.results.calibration_metrics.get(model_name, {}).get('brier_score', 0.5)
            values.append(1 - min(brier, 1))

            # Entropy (normalized)
            entropy = self.results.calibration_metrics.get(model_name, {}).get('mean_entropy', 1.0)
            values.append(entropy / 2.5)  # Normalize to [0, 1]

            # Loss (inverted)
            loss = self.results.model_metrics.get(model_name, {}).get('loss', 1.0)
            values.append(1 / (1 + loss))  # Transform to [0, 1]

            radar_data[model_name] = values

        # Number of variables
        num_vars = len(categories)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Complete the circle
        angles += angles[:1]

        # Plot data
        colors = plt.cm.Set3(np.arange(len(self.models)))

        for i, (model_name, values) in enumerate(radar_data.items()):
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

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

        # Create enhanced scatter plot
        colors = plt.cm.Set3(np.arange(len(labels)))

        for i, (label, comp) in enumerate(zip(labels, components)):
            ax.scatter(comp[0], comp[1], c=[colors[i]], label=label,
                      s=200, alpha=0.8, edgecolors='black', linewidth=2)

            # Add connecting lines to origin
            ax.plot([0, comp[0]], [0, comp[1]], '--', color=colors[i], alpha=0.3)

        # Add origin
        ax.scatter(0, 0, c='black', s=50, marker='x')

        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        ax.set_title('Model Similarity (Weight Space)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

    def _plot_confidence_profile_summary(self, ax) -> None:
        """Plot confidence distribution summary."""
        confidence_data = []

        for model_name, metrics in self.results.confidence_metrics.items():
            for conf in metrics['max_probability']:
                confidence_data.append({
                    'Model': model_name,
                    'Confidence': conf
                })

        if confidence_data:
            df = pd.DataFrame(confidence_data)

            # Create enhanced violin plot
            parts = ax.violinplot([df[df['Model'] == m]['Confidence'].values
                                  for m in df['Model'].unique()],
                                 positions=range(len(df['Model'].unique())),
                                 showmeans=True, showmedians=True)

            # Customize colors
            colors = plt.cm.Set3(np.arange(len(df['Model'].unique())))
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)

            ax.set_xticks(range(len(df['Model'].unique())))
            ax.set_xticklabels(df['Model'].unique(), rotation=45 if len(df['Model'].unique()) > 3 else 0)
            ax.set_ylabel('Confidence (Max Probability)')
            ax.set_title('Confidence Distribution Profiles')
            ax.grid(True, alpha=0.3, axis='y')

    def _plot_training_dynamics(self, ax) -> None:
        """Plot training dynamics if history is available."""
        if not self.results.training_history:
            ax.text(0.5, 0.5, 'No training history available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Dynamics')
            ax.axis('off')
            return

        # Plot validation accuracy curves
        for model_name, history in self.results.training_history.items():
            if 'val_accuracy' in history:
                epochs = range(1, len(history['val_accuracy']) + 1)
                ax.plot(epochs, history['val_accuracy'], '-', label=model_name,
                       linewidth=2, marker='o', markersize=3)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Training Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------------------

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
        if self.results.calibration_metrics:
            summary['analyses_performed'].append('confidence_calibration_analysis')
        if self.results.information_flow:
            summary['analyses_performed'].append('information_flow_analysis')

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

        return summary