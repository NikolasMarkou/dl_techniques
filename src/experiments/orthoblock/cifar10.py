"""
OrthoBlock Effectiveness Study on CIFAR-10

This script conducts a comprehensive evaluation of OrthoBlock layers against
traditional dense layers on the CIFAR-10 image classification task. The primary
goal is to investigate whether enforcing orthogonality constraints in the final
dense layers of a CNN can improve training stability, generalization, and model
calibration.

Hypothesis:
-----------
The hypothesis is that OrthoBlock's orthonormal regularization (W^T * W ≈ I)
creates more stable and disentangled feature representations. This stability is
expected to lead to less overfitting, better calibrated predictions, and more
consistent performance across multiple training runs compared to standard dense
layers, even those with L2 regularization.

Experimental Design:
--------------------
- **Dataset**: CIFAR-10 (32×32 RGB images, 10 classes), using standardized
  dataset builder with proper augmentation.

- **Model Architecture**: A consistent ResNet-inspired CNN is used across all
  experimental variants to ensure a fair comparison. The architecture consists
  of four convolutional blocks with residual connections, followed by global
  pooling and a final dense classification head.

- **Experimental Variants**: The core of the experiment involves swapping out the
  final dense layer with different implementations:
    1. **OrthoBlock Variants**: Models using OrthoBlock with varying strengths of
       orthogonal regularization and scaling initializations.
    2. **Baseline Models**: Models using standard Keras Dense layers, including
       an unregularized baseline and a version with L2 weight decay.

- **Multi-Run Analysis**: Each model variant is trained multiple times (controlled
  by `n_runs`) to ensure statistical significance.

This version integrates the visualization framework and uses standardized dataset loaders.
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import gc
import json
import keras
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable, Optional

from dl_techniques.utils.logger import logger
from dl_techniques.layers.orthoblock import OrthoBlock
from dl_techniques.layers.orthoblock_v2 import OrthoBlockV2

from dl_techniques.datasets.vision.common import (
    create_dataset_builder,
)

from dl_techniques.visualization import (
    VisualizationManager,
    VisualizationPlugin,
    TrainingHistory,
    ModelComparison,
    ClassificationResults,
    MultiModelClassification,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingCurvesVisualization,
    ModelComparisonBarChart,
    PerformanceRadarChart,
    ConfusionMatrixVisualization,
)

from dl_techniques.analyzer import (
    ModelAnalyzer,
    AnalysisConfig,
)


# ==============================================================================
# CUSTOM DATA STRUCTURES
# ==============================================================================

@dataclass
class OrthogonalityData:
    """
    Data structure for orthogonality tracking visualization.

    Attributes:
        model_names: Names of OrthoBlock models
        orthogonality_history: Dict mapping model names to orthogonality metrics over epochs
        scale_history: Dict mapping model names to scale parameter statistics over epochs
    """
    model_names: List[str]
    orthogonality_history: Dict[str, List[Dict[str, float]]]
    scale_history: Dict[str, List[Dict[str, Dict[str, float]]]]


@dataclass
class StatisticalComparisonData:
    """
    Data structure for statistical comparison across multiple runs.

    Attributes:
        model_names: Names of all models
        statistics: Dict mapping model names to their statistical metrics
    """
    model_names: List[str]
    statistics: Dict[str, Dict[str, Dict[str, float]]]


# ==============================================================================
# CUSTOM VISUALIZATION PLUGINS
# ==============================================================================

class OrthogonalityAnalysisVisualization(VisualizationPlugin):
    """Visualization plugin for OrthoBlock orthogonality analysis."""

    @property
    def name(self) -> str:
        return "orthogonality_analysis"

    @property
    def description(self) -> str:
        return "Visualizes orthogonality metrics and scale parameters during training"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, OrthogonalityData)

    def create_visualization(
        self,
        data: OrthogonalityData,
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create orthogonality analysis dashboard."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(20, 10), dpi=self.config.dpi)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Orthogonality measure over training
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_orthogonality(data, ax1)

        # 2. Scale parameter mean over training
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_scale_means(data, ax2)

        # 3. Scale parameter std over training
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_scale_stds(data, ax3)

        # 4. Final scale statistics
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_final_scales(data, ax4)

        plt.suptitle(
            'OrthoBlock Orthogonality Analysis',
            fontsize=self.config.title_fontsize + 2,
            fontweight='bold'
        )

        return fig

    def _plot_orthogonality(self, data: OrthogonalityData, ax: Any) -> None:
        """Plot orthogonality measure over training."""
        colors = plt.cm.tab10(np.linspace(0, 1, len(data.model_names)))

        for idx, name in enumerate(data.model_names):
            if name in data.orthogonality_history and data.orthogonality_history[name]:
                epochs = range(len(data.orthogonality_history[name]))
                ortho_values = [m.get('layer_0', 0) for m in data.orthogonality_history[name]]
                ax.plot(epochs, ortho_values, label=name, color=colors[idx],
                       linewidth=2, marker='o', markersize=3, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=self.config.label_fontsize)
        ax.set_ylabel('||W^T W - I||_F', fontsize=self.config.label_fontsize)
        ax.set_title('Orthogonality Measure Over Training', fontsize=self.config.title_fontsize)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)

    def _plot_scale_means(self, data: OrthogonalityData, ax: Any) -> None:
        """Plot scale parameter means over training."""
        colors = plt.cm.tab10(np.linspace(0, 1, len(data.model_names)))

        for idx, name in enumerate(data.model_names):
            if name in data.scale_history and data.scale_history[name]:
                epochs = range(len(data.scale_history[name]))
                means = [m.get('layer_0', {}).get('mean', 0) for m in data.scale_history[name]]
                if any(means):
                    ax.plot(epochs, means, label=name, color=colors[idx],
                           linewidth=2, marker='o', markersize=3, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Mean Scale Value', fontsize=self.config.label_fontsize)
        ax.set_title('Scale Parameter Mean Over Training', fontsize=self.config.title_fontsize)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)

    def _plot_scale_stds(self, data: OrthogonalityData, ax: Any) -> None:
        """Plot scale parameter standard deviations over training."""
        colors = plt.cm.tab10(np.linspace(0, 1, len(data.model_names)))

        for idx, name in enumerate(data.model_names):
            if name in data.scale_history and data.scale_history[name]:
                epochs = range(len(data.scale_history[name]))
                stds = [m.get('layer_0', {}).get('std', 0) for m in data.scale_history[name]]
                if any(stds):
                    ax.plot(epochs, stds, label=name, color=colors[idx],
                           linewidth=2, marker='o', markersize=3, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Scale Std', fontsize=self.config.label_fontsize)
        ax.set_title('Scale Parameter Std Over Training', fontsize=self.config.title_fontsize)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)

    def _plot_final_scales(self, data: OrthogonalityData, ax: Any) -> None:
        """Plot final scale parameter statistics."""
        stats_keys = ['mean', 'min', 'max']
        width = 0.15
        x = np.arange(len(stats_keys))

        for i, name in enumerate(data.model_names):
            if name in data.scale_history and data.scale_history[name]:
                final_scales = data.scale_history[name][-1].get('layer_0', {})
                if final_scales:
                    values = [final_scales.get(k, 0) for k in stats_keys]
                    ax.bar(x + i * width, values, width, label=name, alpha=0.7)

        ax.set_xlabel('Statistic', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Value', fontsize=self.config.label_fontsize)
        ax.set_title('Final Scale Parameter Statistics', fontsize=self.config.title_fontsize)
        ax.set_xticks(x + width * (len(data.model_names) - 1) / 2)
        ax.set_xticklabels(stats_keys)
        ax.legend(loc='best')
        ax.grid(alpha=0.3, axis='y')


class StatisticalComparisonVisualization(VisualizationPlugin):
    """Visualization plugin for statistical comparison across multiple runs."""

    @property
    def name(self) -> str:
        return "statistical_comparison"

    @property
    def description(self) -> str:
        return "Visualizes statistical comparisons with mean ± std across runs"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, StatisticalComparisonData)

    def create_visualization(
        self,
        data: StatisticalComparisonData,
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create statistical comparison plots."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=self.config.dpi)

        # 1. Accuracy comparison
        self._plot_metric_comparison(
            data, axes[0], 'accuracy',
            'Test Accuracy Comparison (Mean ± Std)',
            'Accuracy'
        )

        # 2. Loss comparison
        self._plot_metric_comparison(
            data, axes[1], 'loss',
            'Test Loss Comparison (Mean ± Std)',
            'Loss'
        )

        plt.suptitle(
            'Statistical Comparison Across Runs',
            fontsize=self.config.title_fontsize + 2,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()

        return fig

    def _plot_metric_comparison(
        self,
        data: StatisticalComparisonData,
        ax: Any,
        metric: str,
        title: str,
        ylabel: str
    ) -> None:
        """Plot comparison for a specific metric."""
        models = data.model_names
        means = [data.statistics[model][metric]['mean'] for model in models]
        stds = [data.statistics[model][metric]['std'] for model in models]

        # Color OrthoBlock models differently
        colors = [self.config.color_scheme.primary if 'OrthoBlock' in model
                 else self.config.color_scheme.secondary for model in models]

        bars = ax.bar(range(len(models)), means, yerr=stds, capsize=5,
                     color=colors, alpha=0.7, edgecolor='black')

        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(alpha=0.3, axis='y')

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + std + (max(means) * 0.01),
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom',
                fontsize=8
            )


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class TrainingConfig:
    """
    Minimal training configuration for dataset builder compatibility.

    Attributes:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        batch_size: Training batch size
        epochs: Number of training epochs
    """
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    num_classes: int = 10
    batch_size: int = 128
    epochs: int = 100


@dataclass
class OrthoBlockExperimentConfig:
    """
    Configuration for the OrthoBlock effectiveness experiment.

    This class encapsulates all configurable parameters for systematically
    evaluating OrthoBlock against traditional dense layers and alternatives.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dense_units: int = 256
    dropout_rates: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5])
    final_dropout: float = 0.5
    kernel_size: Tuple[int, int] = (3, 3)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_residual: bool = True
    activation: str = 'relu'
    output_activation: str = 'softmax'

    # --- Training Parameters ---
    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 20
    monitor_metric: str = 'val_accuracy'

    # --- Dense Layer Variants ---
    dense_variants: Dict[str, Callable] = field(default_factory=lambda: {
        'OrthoBlock_V2_Strong': lambda config: ('orthoblock_v2', {
            'ortho_reg_factor': 1.0,
        }),
        'OrthoBlock_V2_Medium': lambda config: ('orthoblock_v2', {
            'ortho_reg_factor': 0.5,
        }),
        'OrthoBlock_V2_LowScale': lambda config: ('orthoblock_v2', {
            'ortho_reg_factor': 0.1,
        }),
        'OrthoBlock_Strong': lambda config: ('orthoblock', {
            'ortho_reg_factor': 1.0,
            'scale_initial_value': 0.5
        }),
        'OrthoBlock_Medium': lambda config: ('orthoblock', {
            'ortho_reg_factor': 0.1,
            'scale_initial_value': 0.5
        }),
        'OrthoBlock_Weak': lambda config: ('orthoblock', {
            'ortho_reg_factor': 0.01,
            'scale_initial_value': 0.5
        }),
        'OrthoBlock_HighScale': lambda config: ('orthoblock', {
            'ortho_reg_factor': 0.1,
            'scale_initial_value': 0.8
        }),
        'OrthoBlock_LowScale': lambda config: ('orthoblock', {
            'ortho_reg_factor': 0.1,
            'scale_initial_value': 0.2
        }),
        'Dense_Standard': lambda config: ('dense', {}),
        'Dense_L2': lambda config: ('dense', {
            'kernel_regularizer': keras.regularizers.L2(0.01)
        })
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "orthoblock_effectiveness_study"
    random_seed: int = 42
    n_runs: int = 1  # Multiple runs for statistical significance

    # --- Analysis Configuration ---
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=True,
        calibration_bins=15,
        save_plots=True,
        plot_style='publication',
    ))


# ==============================================================================
# DENSE LAYER FACTORY
# ==============================================================================

def create_dense_layer(
    layer_type: str,
    layer_params: Dict[str, Any],
    units: int,
    activation: str = 'relu',
    name: Optional[str] = None
) -> keras.layers.Layer:
    """
    Factory function to create different dense layer types.

    Args:
        layer_type: Type of dense layer ('orthoblock', 'dense')
        layer_params: Parameters specific to the layer type
        units: Number of units in the layer
        activation: Activation function
        name: Optional layer name

    Returns:
        Configured dense layer
    """
    if layer_type == 'orthoblock':
        return OrthoBlock(
            units=units,
            activation=activation,
            ortho_reg_factor=layer_params.get('ortho_reg_factor', 0.1),
            scale_initial_value=layer_params.get('scale_initial_value', 0.5),
            use_bias=layer_params.get('use_bias', True),
            kernel_initializer=layer_params.get('kernel_initializer', 'glorot_uniform'),
            name=name
        )
    elif layer_type == 'orthoblock_v2':
        return OrthoBlockV2(
            units=units,
            activation=activation,
            ortho_reg_factor=layer_params.get('ortho_reg_factor', 0.1),
            use_bias=layer_params.get('use_bias', True),
            kernel_initializer=layer_params.get('kernel_initializer', 'glorot_uniform'),
            name=name
        )
    elif layer_type == 'dense':
        return keras.layers.Dense(
            units=units,
            activation=activation,
            use_bias=layer_params.get('use_bias', True),
            kernel_initializer=layer_params.get('kernel_initializer', 'glorot_uniform'),
            kernel_regularizer=layer_params.get('kernel_regularizer', None),
            name=name
        )
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


# ==============================================================================
# MODEL ARCHITECTURE BUILDING
# ==============================================================================

def build_conv_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: OrthoBlockExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a convolutional block with residual connections.

    Args:
        inputs: Input tensor
        filters: Number of filters
        config: Experiment configuration
        block_index: Index of the current block

    Returns:
        Output tensor after convolutional block
    """
    shortcut = inputs

    # First convolution
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_1'
    )(inputs)

    x = keras.layers.BatchNormalization(name=f'bn{block_index}_1')(x)
    x = keras.layers.Activation(config.activation)(x)

    # Second convolution
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_2'
    )(x)

    x = keras.layers.BatchNormalization(name=f'bn{block_index}_2')(x)

    # Adjust shortcut if needed
    if config.use_residual and shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters, (1, 1), padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'conv{block_index}_shortcut'
        )(shortcut)

        shortcut = keras.layers.BatchNormalization(name=f'bn{block_index}_shortcut')(shortcut)

    # Add and activate
    if config.use_residual:
        x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation(config.activation)(x)

    # Max pooling (except last block)
    if block_index < len(config.conv_filters) - 1:
        x = keras.layers.MaxPooling2D((2, 2))(x)

    # Dropout
    dropout_rate = (config.dropout_rates[block_index]
                   if block_index < len(config.dropout_rates) else 0.0)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def build_model(
    config: OrthoBlockExperimentConfig,
    layer_type: str,
    layer_params: Dict[str, Any],
    name: str
) -> keras.Model:
    """
    Build a complete CNN model with specified dense layer type.

    Args:
        config: Experiment configuration
        layer_type: Type of dense layer
        layer_params: Parameters for dense layer
        name: Model name

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)

    # Global average pooling
    x = keras.layers.GlobalMaxPooling2D()(x)

    # Create the specified dense layer
    x = create_dense_layer(
        layer_type=layer_type,
        layer_params=layer_params,
        units=config.dense_units,
        activation=config.activation,
        name=f'{name}_dense'
    )(x)

    # Additional dropout for non-OrthoBlock layers
    if layer_type != 'orthoblock':
        dropout_rate = layer_params.get('dropout_rate', config.final_dropout)
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = keras.layers.Dense(
        config.num_classes,
        activation=config.output_activation,
        kernel_initializer=config.kernel_initializer,
        name='predictions'
    )(x)

    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{name}_model')

    optimizer = keras.optimizers.AdamW(learning_rate=config.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # Use sparse for integer labels
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )

    return model


# ==============================================================================
# ORTHOGONALITY TRACKING CALLBACK
# ==============================================================================

class OrthogonalityTracker(keras.callbacks.Callback):
    """Callback to track orthogonality metrics during training."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.ortho_layers = []
        self.orthogonality_history = []
        self.scale_history = []

    def on_train_begin(self, logs=None):
        # Find all OrthoBlock layers
        for layer in self.model.layers:
            if isinstance(layer, OrthoBlock):
                self.ortho_layers.append(layer)

        logger.info(f"Found {len(self.ortho_layers)} OrthoBlock layers in {self.model_name}")

    def calculate_orthogonality(self, layer: OrthoBlock) -> float:
        """Calculate orthogonality measure ||W^T W - I||_F for a layer."""
        if not hasattr(layer, 'dense') or not hasattr(layer.dense, 'kernel'):
            return 0.0

        weights = layer.dense.kernel.numpy()
        wt_w = np.matmul(weights.T, weights)
        identity = np.eye(wt_w.shape[0], wt_w.shape[1])
        diff = wt_w - identity
        return float(np.sqrt(np.sum(np.square(diff))))

    def get_scale_statistics(self, layer: OrthoBlock) -> Dict[str, float]:
        """Get statistics of constrained scale parameters."""
        if not hasattr(layer, 'constrained_scale') or not hasattr(layer.constrained_scale, 'multiplier'):
            return {}

        scales = layer.constrained_scale.multiplier.numpy()
        return {
            'mean': float(np.mean(scales)),
            'std': float(np.std(scales)),
            'min': float(np.min(scales)),
            'max': float(np.max(scales))
        }

    def on_epoch_end(self, epoch, logs=None):
        # Track orthogonality metrics for each OrthoBlock
        epoch_ortho = {}
        epoch_scales = {}

        for i, layer in enumerate(self.ortho_layers):
            ortho_value = self.calculate_orthogonality(layer)
            epoch_ortho[f'layer_{i}'] = ortho_value

            scale_stats = self.get_scale_statistics(layer)
            if scale_stats:
                epoch_scales[f'layer_{i}'] = scale_stats

        self.orthogonality_history.append(epoch_ortho)
        self.scale_history.append(epoch_scales)


# ==============================================================================
# STATISTICAL ANALYSIS UTILITIES
# ==============================================================================

def calculate_run_statistics(
    results_per_run: Dict[str, List[Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate statistics across multiple runs for each model.

    Args:
        results_per_run: Dictionary mapping model names to lists of results across runs

    Returns:
        Dictionary with mean, std, min, max for each model and metric
    """
    statistics = {}

    for model_name, run_results in results_per_run.items():
        if not run_results:
            continue

        statistics[model_name] = {}
        metrics = run_results[0].keys()

        for metric in metrics:
            values = [result[metric] for result in run_results if metric in result]
            if values:
                statistics[model_name][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }

    return statistics


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def train_single_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: OrthoBlockExperimentConfig,
    steps_per_epoch: int,
    val_steps: int,
    model_name: str,
    output_dir: Path,
    ortho_tracker: Optional[OrthogonalityTracker] = None
) -> Dict[str, List[float]]:
    """
    Train a single model and return its history.

    Args:
        model: Keras model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        config: Experiment configuration
        steps_per_epoch: Number of steps per epoch
        val_steps: Number of validation steps
        model_name: Name of the model
        output_dir: Directory to save checkpoints
        ortho_tracker: Optional orthogonality tracker callback

    Returns:
        Training history dictionary
    """
    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f'{model_name}_best.keras'),
            monitor=config.monitor_metric,
            save_best_only=True,
            verbose=0
        )
    ]

    # Add orthogonality tracker if provided
    if ortho_tracker:
        callbacks.append(ortho_tracker)

    # Train the model
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )

    return history.history


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: OrthoBlockExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete OrthoBlock effectiveness experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary containing all experimental results
    """
    keras.utils.set_random_seed(config.random_seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualization manager
    viz_config = PlotConfig(
        style=PlotStyle.SCIENTIFIC,
        color_scheme=ColorScheme(
            primary='#FF8C00',  # Orange for OrthoBlock
            secondary='#4169E1',  # Blue for Dense
        ),
        title_fontsize=14,
        save_format='png'
    )

    viz_manager = VisualizationManager(
        experiment_name=config.experiment_name,
        output_dir=experiment_dir / "visualizations",
        config=viz_config
    )

    # Register visualization plugins
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart)
    viz_manager.register_template("performance_radar", PerformanceRadarChart)
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
    viz_manager.register_template("orthogonality_analysis", OrthogonalityAnalysisVisualization)
    viz_manager.register_template("statistical_comparison", StatisticalComparisonVisualization)

    logger.info("🚀 Starting OrthoBlock Effectiveness Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("📊 Loading CIFAR-10 dataset using standardized builder...")

    # Create training config for dataset builder
    train_config = TrainingConfig(
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        batch_size=config.batch_size,
        epochs=config.epochs
    )

    # Create dataset builder
    dataset_builder = create_dataset_builder('cifar10', train_config)

    # Build datasets
    train_ds, val_ds, steps_per_epoch, val_steps = dataset_builder.build()

    # Get test data for evaluation
    test_data = dataset_builder.get_test_data()
    class_names = dataset_builder.get_class_names()

    logger.info("✅ Dataset loaded successfully")
    logger.info(f"Steps per epoch: {steps_per_epoch}, Validation steps: {val_steps}")

    # ===== MULTI-RUN TRAINING =====
    logger.info(f"🏋️ Running {config.n_runs} repetitions for statistical significance...")

    all_trained_models = {}
    all_histories = {}
    all_orthogonality_trackers = {}
    results_per_run = {variant_name: [] for variant_name in config.dense_variants.keys()}

    for run_idx in range(config.n_runs):
        logger.info(f"--- Starting run {run_idx + 1}/{config.n_runs} ---")
        run_seed = config.random_seed + run_idx * 1000
        keras.utils.set_random_seed(run_seed)

        current_run_models = {}
        current_run_histories = {}
        current_run_trackers = {}

        for variant_name, variant_factory in config.dense_variants.items():
            logger.info(f"Training {variant_name} (Run {run_idx + 1})...")

            layer_type, layer_params = variant_factory(config)
            model = build_model(config, layer_type, layer_params, f"{variant_name}_run{run_idx}")

            if run_idx == 0:
                model.summary(print_fn=logger.info)

            # Create orthogonality tracker for OrthoBlock models
            ortho_tracker = None
            if layer_type == 'orthoblock':
                ortho_tracker = OrthogonalityTracker(f"{variant_name}_run{run_idx}")

            # Train the model
            history = train_single_model(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                config=config,
                steps_per_epoch=steps_per_epoch,
                val_steps=val_steps,
                model_name=f"{variant_name}_run{run_idx}",
                output_dir=experiment_dir / "checkpoints" / f"run_{run_idx}",
                ortho_tracker=ortho_tracker
            )

            # Evaluate model
            try:
                predictions = model.predict(test_data.x_data, verbose=0)
                y_pred_classes = np.argmax(predictions, axis=1)
                y_true = test_data.y_data.astype(int)

                accuracy = np.mean(y_pred_classes == y_true)
                loss = -np.mean(np.log(predictions[np.arange(len(y_true)), y_true] + 1e-7))

                results_per_run[variant_name].append({
                    'accuracy': accuracy,
                    'loss': loss
                })

                logger.info(f"{variant_name} (Run {run_idx + 1}): Accuracy={accuracy:.4f}, Loss={loss:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {variant_name} (Run {run_idx + 1}): {e}", exc_info=True)

            current_run_models[variant_name] = model
            current_run_histories[variant_name] = history
            if ortho_tracker:
                current_run_trackers[variant_name] = ortho_tracker

        # Save last run's models for analysis
        if run_idx == config.n_runs - 1:
            all_trained_models = current_run_models
            all_histories = current_run_histories
            all_orthogonality_trackers = current_run_trackers

        # Clean up
        del current_run_models, current_run_histories, current_run_trackers
        gc.collect()

    # ===== STATISTICAL ANALYSIS =====
    logger.info("📊 Calculating statistics across runs...")
    run_statistics = calculate_run_statistics(results_per_run)

    # ===== VISUALIZATION GENERATION =====
    logger.info("🖼️ Generating visualizations using framework...")

    # 1. Statistical comparison
    stat_comparison_data = StatisticalComparisonData(
        model_names=list(run_statistics.keys()),
        statistics=run_statistics
    )

    viz_manager.visualize(
        data=stat_comparison_data,
        plugin_name="statistical_comparison",
        show=False
    )

    # 2. Training curves (from last run)
    training_histories = {
        name: TrainingHistory(
            epochs=list(range(len(hist['loss']))),
            train_loss=hist['loss'],
            val_loss=hist['val_loss'],
            train_metrics={'accuracy': hist['accuracy']},
            val_metrics={'accuracy': hist['val_accuracy']}
        )
        for name, hist in all_histories.items()
    }

    viz_manager.visualize(
        data=training_histories,
        plugin_name="training_curves",
        show=False
    )

    # 3. Orthogonality analysis
    if all_orthogonality_trackers:
        ortho_data = OrthogonalityData(
            model_names=list(all_orthogonality_trackers.keys()),
            orthogonality_history={
                name: tracker.orthogonality_history
                for name, tracker in all_orthogonality_trackers.items()
            },
            scale_history={
                name: tracker.scale_history
                for name, tracker in all_orthogonality_trackers.items()
            }
        )

        viz_manager.visualize(
            data=ortho_data,
            plugin_name="orthogonality_analysis",
            show=False
        )

    # 4. Model comparison
    comparison_data = ModelComparison(
        model_names=list(run_statistics.keys()),
        metrics={
            name: {
                'accuracy': stats['accuracy']['mean'],
                'top_5_accuracy': stats.get('top_5_accuracy', {}).get('mean', 0.0)
            }
            for name, stats in run_statistics.items()
        }
    )

    viz_manager.visualize(
        data=comparison_data,
        plugin_name="model_comparison_bars",
        sort_by='accuracy',
        show=False
    )

    viz_manager.visualize(
        data=comparison_data,
        plugin_name="performance_radar",
        normalize=True,
        show=False
    )

    # 5. Confusion matrices
    try:
        raw_predictions = {
            name: model.predict(test_data.x_data, verbose=0)
            for name, model in all_trained_models.items()
        }
        class_predictions = {
            name: np.argmax(preds, axis=1)
            for name, preds in raw_predictions.items()
        }
        y_true_labels = test_data.y_data.astype(int)

        model_results = {
            name: ClassificationResults(
                y_true=y_true_labels,
                y_pred=preds,
                y_prob=raw_predictions[name],
                class_names=class_names,
                model_name=name
            )
            for name, preds in class_predictions.items()
        }

        multi_model_data = MultiModelClassification(
            y_true=y_true_labels,
            model_results=model_results,
            class_names=class_names
        )

        viz_manager.visualize(
            data=multi_model_data,
            plugin_name="confusion_matrix",
            normalize='true',
            show=False
        )
    except Exception as e:
        logger.warning(f"Could not generate confusion matrices: {e}")

    # ===== MODEL ANALYSIS =====
    logger.info("📊 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None

    try:
        analyzer = ModelAnalyzer(
            models=all_trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )

        model_analysis_results = analyzer.analyze(data=test_data)
        logger.info("✅ Model analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Model analysis failed: {e}", exc_info=True)

    # ===== RESULTS COMPILATION =====
    results = {
        'run_statistics': run_statistics,
        'model_analysis': model_analysis_results,
        'histories': all_histories,
        'orthogonality_trackers': all_orthogonality_trackers,
        'trained_models': all_trained_models,
        'config': config,
        'experiment_dir': experiment_dir
    }

    save_experiment_results(results, experiment_dir)
    print_experiment_summary(results)

    return results


# ==============================================================================
# RESULTS SAVING AND REPORTING
# ==============================================================================

def save_experiment_results(results: Dict[str, Any], experiment_dir: Path) -> None:
    """
    Save experiment results in multiple formats.

    Args:
        results: Experiment results dictionary
        experiment_dir: Directory to save results
    """
    try:
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_numpy(i) for i in obj]
            return obj

        # Save configuration
        config_dict = {
            'experiment_name': results['config'].experiment_name,
            'dense_variants': list(results['config'].dense_variants.keys()),
            'epochs': results['config'].epochs,
            'batch_size': results['config'].batch_size,
            'learning_rate': results['config'].learning_rate,
            'n_runs': results['config'].n_runs,
            'random_seed': results['config'].random_seed,
            'architecture': {
                'conv_filters': results['config'].conv_filters,
                'dense_units': results['config'].dense_units,
                'use_residual': results['config'].use_residual
            }
        }
        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save statistical results
        stats_converted = convert_numpy(results['run_statistics'])
        with open(experiment_dir / "statistical_results.json", 'w') as f:
            json.dump(stats_converted, f, indent=2)

        # Save orthogonality data
        if results['orthogonality_trackers']:
            ortho_data = {
                name: {
                    'orthogonality_history': convert_numpy(t.orthogonality_history),
                    'scale_history': convert_numpy(t.scale_history)
                }
                for name, t in results['orthogonality_trackers'].items()
            }
            with open(experiment_dir / "orthogonality_data.json", 'w') as f:
                json.dump(ortho_data, f, indent=2)

        # Save models
        models_dir = experiment_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for name, model in results['trained_models'].items():
            model.save(models_dir / f"{name}.keras")

        logger.info("💾 Experiment results saved successfully")

    except Exception as e:
        logger.error(f"Failed to save experiment results: {e}", exc_info=True)


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print comprehensive experiment summary.

    Args:
        results: Experiment results dictionary
    """
    logger.info("=" * 80)
    logger.info("📋 ORTHOBLOCK EFFECTIVENESS EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # Statistical results
    if 'run_statistics' in results and results['run_statistics']:
        logger.info("\n🎯 STATISTICAL RESULTS (Mean ± Std across runs):")
        logger.info(f"{'Model':<24} {'Accuracy':<20} {'Loss':<20} {'Runs':<8}")
        logger.info("-" * 75)

        # Sort by accuracy
        sorted_stats = sorted(
            results['run_statistics'].items(),
            key=lambda x: x[1]['accuracy']['mean'],
            reverse=True
        )

        for name, stats in sorted_stats:
            acc_m = stats['accuracy']['mean']
            acc_s = stats['accuracy']['std']
            loss_m = stats['loss']['mean']
            loss_s = stats['loss']['std']
            n_runs = stats['accuracy']['count']
            logger.info(
                f"{name:<24} {acc_m:.4f} ± {acc_s:.4f}    "
                f"{loss_m:.4f} ± {loss_s:.4f}    {n_runs:<8}"
            )

    # Calibration analysis
    analysis = results.get('model_analysis')
    if analysis and analysis.calibration_metrics:
        logger.info("\n🎯 CALIBRATION ANALYSIS (from last run):")
        logger.info(f"{'Model':<24} {'ECE':<12} {'Brier':<12} {'Mean Entropy':<15}")
        logger.info("-" * 67)
        for name, cal_metrics in analysis.calibration_metrics.items():
            logger.info(
                f"{name:<24} {cal_metrics.get('ece', 0.0):<12.4f} "
                f"{cal_metrics.get('brier_score', 0.0):<12.4f} "
                f"{cal_metrics.get('mean_entropy', 0.0):<15.4f}"
            )

    # Key insights
    logger.info("\n💡 KEY INSIGHTS:")
    if 'run_statistics' in results:
        stats = results['run_statistics']
        best_acc_model = max(stats, key=lambda m: stats[m]['accuracy']['mean'])
        logger.info(
            f"   🏆 Best Accuracy: {best_acc_model} "
            f"({stats[best_acc_model]['accuracy']['mean']:.4f})"
        )

        most_stable_model = min(stats, key=lambda m: stats[m]['accuracy']['std'])
        logger.info(
            f"   📊 Most Stable: {most_stable_model} "
            f"(Acc Std: {stats[most_stable_model]['accuracy']['std']:.4f})"
        )

        if results.get('orthogonality_trackers'):
            logger.info("\n   📐 Orthogonality Dynamics (Initial → Final ||W^T W - I||_F):")
            for name, tracker in results['orthogonality_trackers'].items():
                if tracker.orthogonality_history:
                    initial = tracker.orthogonality_history[0].get('layer_0', 0)
                    final = tracker.orthogonality_history[-1].get('layer_0', 0)
                    logger.info(f"      {name:<22}: {initial:.3f} → {final:.3f}")

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Main execution function for the OrthoBlock effectiveness experiment."""
    logger.info("🚀 OrthoBlock Effectiveness Experiment")
    logger.info("=" * 80)

    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.info(f"GPU memory growth configuration failed: {e}")

    # Initialize configuration
    config = OrthoBlockExperimentConfig()

    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Dense layer variants: {list(config.dense_variants.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"   Number of runs: {config.n_runs}")
    logger.info(f"   Architecture: {len(config.conv_filters)} conv blocks, {config.dense_units} dense units")
    logger.info("")

    try:
        results = run_experiment(config)
        logger.info("✅ OrthoBlock effectiveness experiment completed successfully!")

    except Exception as e:
        logger.error(f"❌ Experiment failed with an unhandled exception: {e}", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()