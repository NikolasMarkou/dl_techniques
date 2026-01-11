"""
CIFAR-10 Stacked RigidSimplex Depth Comparison: Evaluating Deep Geometric Projections
======================================================================================

This experiment investigates how RigidSimplexLayer performs when stacked to create
deeper projection networks. By comparing 2-5 stacked layers against equivalent
Dense configurations, we evaluate whether geometric constraints based on Equiangular
Tight Frames (ETF) maintain their benefits in deeper architectures.

Research Questions
------------------

1. **Depth Scaling**: How does RigidSimplexLayer performance scale with depth?
2. **Gradient Flow**: Do stacked Simplex layers maintain gradient flow without normalization?
3. **Orthogonality Propagation**: How does orthogonality error accumulate across layers?
4. **Parameter Efficiency**: How does the parameter/accuracy trade-off change with depth?

Experimental Design
-------------------

**Dataset**: CIFAR-10 (10 classes, 32x32 RGB images)

**Model Architecture**: CNN backbone followed by stacked projection layers:
- Convolutional feature extractor (shared across all variants)
- Global average pooling
- N stacked projection layers (Dense or RigidSimplex, NO normalization between)
- Classification head

**Configurations Evaluated**:
- Dense_2L, Dense_3L, Dense_4L, Dense_5L: Stacked Dense layers
- Simplex_2L, Simplex_3L, Simplex_4L, Simplex_5L: Stacked RigidSimplex layers

**Key Difference from Previous Experiment**:
No BatchNormalization between projection layers - testing raw layer behavior.
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import gc
import keras
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

from keras import ops
from keras import layers
from keras import regularizers

from dl_techniques.utils.logger import logger

from dl_techniques.datasets.vision.common import (
    create_dataset_builder
)

from dl_techniques.visualization import (
    VisualizationManager,
    VisualizationPlugin,
    TrainingHistory,
    ModelComparison,
    ClassificationResults,
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

# Local import for RigidSimplexLayer
from dl_techniques.layers.rigid_simplex_layer import RigidSimplexLayer


# ==============================================================================
# CUSTOM DATA STRUCTURES
# ==============================================================================

@dataclass
class DepthComparisonData:
    """
    Data structure for depth comparison visualization.

    Attributes:
        config_names: Names of configurations compared
        metrics: Dictionary mapping config names to their metrics
        histories: Dictionary mapping config names to training histories
        depth_info: Dictionary mapping config names to their depth
    """
    config_names: List[str]
    metrics: Dict[str, Dict[str, float]]
    histories: Dict[str, Dict[str, List[float]]]
    depth_info: Dict[str, int]


@dataclass
class StackedSimplexAnalysis:
    """
    Data structure for analyzing stacked Simplex layers.

    Attributes:
        config_name: Name of the configuration
        num_layers: Number of stacked layers
        rotation_matrices: List of rotation matrices per layer
        scale_values: List of scale values per layer
        orthogonality_errors: List of orthogonality errors per layer
        cumulative_transform: Product of all rotation matrices
    """
    config_name: str
    num_layers: int
    rotation_matrices: List[np.ndarray]
    scale_values: List[float]
    orthogonality_errors: List[float]
    cumulative_transform: np.ndarray


# ==============================================================================
# CUSTOM VISUALIZATION PLUGINS
# ==============================================================================

class DepthComparisonDashboard(VisualizationPlugin):
    """Dashboard comparing performance across different depths."""

    @property
    def name(self) -> str:
        return "depth_comparison_dashboard"

    @property
    def description(self) -> str:
        return "Dashboard comparing Dense vs Simplex at different depths"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, DepthComparisonData)

    def create_visualization(
        self,
        data: DepthComparisonData,
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create depth comparison dashboard."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(20, 16), dpi=self.config.dpi)
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. Training accuracy curves
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_training_curves(data, ax1, 'accuracy', 'Training Accuracy')

        # 2. Validation accuracy curves
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_training_curves(data, ax2, 'val_accuracy', 'Validation Accuracy')

        # 3. Training loss curves
        ax3 = fig.add_subplot(gs[2, :2])
        self._plot_training_curves(data, ax3, 'loss', 'Training Loss')

        # 4. Validation loss curves
        ax4 = fig.add_subplot(gs[3, :2])
        self._plot_training_curves(data, ax4, 'val_loss', 'Validation Loss')

        # 5. Accuracy vs Depth comparison
        ax5 = fig.add_subplot(gs[0, 2])
        self._plot_accuracy_vs_depth(data, ax5)

        # 6. Loss vs Depth comparison
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_loss_vs_depth(data, ax6)

        # 7. Parameter count vs Depth
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_params_vs_depth(data, ax7)

        # 8. Summary table
        ax8 = fig.add_subplot(gs[3, 2])
        self._plot_summary_table(data, ax8)

        plt.suptitle(
            'Stacked Layer Depth Comparison: Dense vs RigidSimplex',
            fontsize=self.config.title_fontsize + 4,
            fontweight='bold',
            y=0.98
        )

        return fig

    def _plot_training_curves(
        self,
        data: DepthComparisonData,
        ax: Any,
        metric: str,
        title: str
    ) -> None:
        """Plot training curves with depth-based styling."""
        import matplotlib.pyplot as plt

        # Color maps for Dense (blues) and Simplex (oranges)
        dense_cmap = plt.cm.Blues
        simplex_cmap = plt.cm.Oranges

        for name in data.config_names:
            if name in data.histories and metric in data.histories[name]:
                epochs = range(len(data.histories[name][metric]))
                depth = data.depth_info.get(name, 2)

                # Normalize depth to color intensity (2-5 -> 0.4-0.9)
                intensity = 0.4 + (depth - 2) * 0.15

                if 'Dense' in name:
                    color = dense_cmap(intensity)
                    linestyle = '-'
                else:
                    color = simplex_cmap(intensity)
                    linestyle = '--'

                ax.plot(
                    epochs,
                    data.histories[name][metric],
                    label=f"{name}",
                    color=color,
                    linewidth=2,
                    linestyle=linestyle,
                    alpha=0.85
                )

        ax.set_xlabel('Epoch', fontsize=self.config.label_fontsize)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    def _plot_accuracy_vs_depth(self, data: DepthComparisonData, ax: Any) -> None:
        """Plot accuracy vs depth for both layer types."""
        import matplotlib.pyplot as plt

        dense_depths = []
        dense_accs = []
        simplex_depths = []
        simplex_accs = []

        for name in data.config_names:
            depth = data.depth_info.get(name, 2)
            acc = data.metrics[name].get('accuracy', 0.0)

            if 'Dense' in name:
                dense_depths.append(depth)
                dense_accs.append(acc)
            else:
                simplex_depths.append(depth)
                simplex_accs.append(acc)

        ax.plot(dense_depths, dense_accs, 'o-', color='#2E86AB',
                linewidth=2, markersize=10, label='Dense')
        ax.plot(simplex_depths, simplex_accs, 's--', color='#F18F01',
                linewidth=2, markersize=10, label='Simplex')

        ax.set_xlabel('Number of Stacked Layers', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Test Accuracy', fontsize=self.config.label_fontsize)
        ax.set_title('Accuracy vs Depth', fontsize=self.config.title_fontsize)
        ax.set_xticks([2, 3, 4, 5])
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    def _plot_loss_vs_depth(self, data: DepthComparisonData, ax: Any) -> None:
        """Plot loss vs depth for both layer types."""
        import matplotlib.pyplot as plt

        dense_depths = []
        dense_loss = []
        simplex_depths = []
        simplex_loss = []

        for name in data.config_names:
            depth = data.depth_info.get(name, 2)
            loss = data.metrics[name].get('loss', 0.0)

            if 'Dense' in name:
                dense_depths.append(depth)
                dense_loss.append(loss)
            else:
                simplex_depths.append(depth)
                simplex_loss.append(loss)

        ax.plot(dense_depths, dense_loss, 'o-', color='#2E86AB',
                linewidth=2, markersize=10, label='Dense')
        ax.plot(simplex_depths, simplex_loss, 's--', color='#F18F01',
                linewidth=2, markersize=10, label='Simplex')

        ax.set_xlabel('Number of Stacked Layers', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Test Loss', fontsize=self.config.label_fontsize)
        ax.set_title('Loss vs Depth', fontsize=self.config.title_fontsize)
        ax.set_xticks([2, 3, 4, 5])
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    def _plot_params_vs_depth(self, data: DepthComparisonData, ax: Any) -> None:
        """Plot parameter count vs depth."""
        import matplotlib.pyplot as plt

        dense_depths = []
        dense_params = []
        simplex_depths = []
        simplex_params = []

        for name in data.config_names:
            depth = data.depth_info.get(name, 2)
            params = data.metrics[name].get('trainable_params', 0)

            if 'Dense' in name:
                dense_depths.append(depth)
                dense_params.append(params / 1e6)  # Convert to millions
            else:
                simplex_depths.append(depth)
                simplex_params.append(params / 1e6)

        ax.plot(dense_depths, dense_params, 'o-', color='#2E86AB',
                linewidth=2, markersize=10, label='Dense')
        ax.plot(simplex_depths, simplex_params, 's--', color='#F18F01',
                linewidth=2, markersize=10, label='Simplex')

        ax.set_xlabel('Number of Stacked Layers', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Parameters (Millions)', fontsize=self.config.label_fontsize)
        ax.set_title('Parameters vs Depth', fontsize=self.config.title_fontsize)
        ax.set_xticks([2, 3, 4, 5])
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    def _plot_summary_table(self, data: DepthComparisonData, ax: Any) -> None:
        """Plot summary table."""
        ax.axis('off')

        table_data = []
        for name in sorted(data.config_names, key=lambda x: (x.split('_')[0], data.depth_info.get(x, 0))):
            metrics = data.metrics[name]
            row = [
                name,
                str(data.depth_info.get(name, '?')),
                f"{metrics.get('accuracy', 0.0):.4f}",
                f"{metrics.get('loss', 0.0):.4f}",
                f"{metrics.get('trainable_params', 0):,}"
            ]
            table_data.append(row)

        table = ax.table(
            cellText=table_data,
            colLabels=['Config', 'Depth', 'Acc', 'Loss', 'Params'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.6)

        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Performance Summary', fontsize=self.config.title_fontsize, pad=20)


class StackedSimplexAnalysisDashboard(VisualizationPlugin):
    """Dashboard for analyzing stacked Simplex layer internals."""

    @property
    def name(self) -> str:
        return "stacked_simplex_analysis"

    @property
    def description(self) -> str:
        return "Analysis of stacked RigidSimplexLayer parameters"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, list) and all(isinstance(d, StackedSimplexAnalysis) for d in data)

    def create_visualization(
        self,
        data: List[StackedSimplexAnalysis],
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create stacked simplex analysis dashboard."""
        import matplotlib.pyplot as plt

        n_configs = len(data)
        max_layers = max(d.num_layers for d in data)

        fig = plt.figure(figsize=(5 * n_configs, 4 * 4), dpi=self.config.dpi)
        gs = fig.add_gridspec(4, n_configs, hspace=0.4, wspace=0.3)

        for col, analysis in enumerate(data):
            # Row 1: Scale values per layer
            ax1 = fig.add_subplot(gs[0, col])
            self._plot_scale_values(analysis, ax1)

            # Row 2: Orthogonality errors per layer
            ax2 = fig.add_subplot(gs[1, col])
            self._plot_ortho_errors(analysis, ax2)

            # Row 3: Cumulative transform analysis
            ax3 = fig.add_subplot(gs[2, col])
            self._plot_cumulative_transform(analysis, ax3)

            # Row 4: Layer-wise rotation matrix norms
            ax4 = fig.add_subplot(gs[3, col])
            self._plot_rotation_norms(analysis, ax4)

        plt.suptitle(
            'Stacked RigidSimplexLayer Internal Analysis',
            fontsize=self.config.title_fontsize + 2,
            fontweight='bold'
        )

        return fig

    def _plot_scale_values(self, data: StackedSimplexAnalysis, ax: Any) -> None:
        """Plot scale values per layer."""
        layers = range(1, data.num_layers + 1)
        ax.bar(layers, data.scale_values, color='#F18F01', alpha=0.8, edgecolor='black')

        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Unit scale')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Scale Value')
        ax.set_title(f'{data.config_name}\nLearned Scale per Layer')
        ax.set_xticks(layers)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

        # Annotate values
        for i, v in enumerate(data.scale_values):
            ax.text(i + 1, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)

    def _plot_ortho_errors(self, data: StackedSimplexAnalysis, ax: Any) -> None:
        """Plot orthogonality errors per layer."""
        layers = range(1, data.num_layers + 1)
        ax.bar(layers, data.orthogonality_errors, color='#A23B72', alpha=0.8, edgecolor='black')

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('‖R^T R - I‖_F')
        ax.set_title(f'Orthogonality Error per Layer\nTotal: {sum(data.orthogonality_errors):.2f}')
        ax.set_xticks(layers)
        ax.grid(alpha=0.3, axis='y')

    def _plot_cumulative_transform(self, data: StackedSimplexAnalysis, ax: Any) -> None:
        """Plot cumulative transform properties."""
        import matplotlib.pyplot as plt

        # Analyze cumulative transform
        C = data.cumulative_transform
        CtC = C.T @ C

        # Show subset if large
        if CtC.shape[0] > 32:
            CtC = CtC[:32, :32]

        im = ax.imshow(CtC, cmap='RdBu', aspect='auto')
        ax.set_title(f'Cumulative Transform\n(R1 @ R2 @ ... @ Rn)^T @ (R1 @ R2 @ ... @ Rn)')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_rotation_norms(self, data: StackedSimplexAnalysis, ax: Any) -> None:
        """Plot rotation matrix spectral properties per layer."""
        layers = range(1, data.num_layers + 1)

        # Compute spectral norms and condition numbers
        spectral_norms = []
        condition_numbers = []

        for R in data.rotation_matrices:
            s = np.linalg.svd(R, compute_uv=False)
            spectral_norms.append(s[0])  # Largest singular value
            condition_numbers.append(s[0] / (s[-1] + 1e-8))

        x = np.arange(len(layers))
        width = 0.35

        bars1 = ax.bar(x - width/2, spectral_norms, width, label='Spectral Norm',
                       color='#2E86AB', alpha=0.8)

        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, condition_numbers, width, label='Condition #',
                        color='#F18F01', alpha=0.8)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Spectral Norm', color='#2E86AB')
        ax2.set_ylabel('Condition Number', color='#F18F01')
        ax.set_title('Rotation Matrix Spectral Properties')
        ax.set_xticks(x)
        ax.set_xticklabels(layers)

        # Combined legend
        ax.legend([bars1, bars2], ['Spectral Norm', 'Condition #'],
                  loc='upper right', fontsize=8)
        ax.grid(alpha=0.3, axis='y')


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class TrainingConfig:
    """Minimal training configuration for dataset builder compatibility."""
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    num_classes: int = 10
    batch_size: int = 64
    epochs: int = 100


@dataclass
class StackedLayerConfig:
    """
    Configuration for a stacked layer variant.

    Attributes:
        name: Display name for the configuration
        layer_type: Either 'dense' or 'simplex'
        num_layers: Number of stacked layers (2-5)
        units_per_layer: Units in each layer
        simplex_kwargs: Additional kwargs for RigidSimplexLayer
    """
    name: str
    layer_type: str  # 'dense' or 'simplex'
    num_layers: int
    units_per_layer: int = 128
    simplex_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'scale_min': 0.5,
        'scale_max': 2.0,
        'orthogonality_penalty': 1e-4
    })


@dataclass
class ExperimentConfig:
    """
    Configuration for the stacked RigidSimplex depth comparison experiment.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)

    # --- Backbone Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_batch_norm: bool = True  # Only in backbone
    backbone_dropout: float = 0.25

    # --- Stacked Layer Configuration ---
    units_per_layer: int = 128

    # --- Training Parameters ---
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 50
    monitor_metric: str = 'val_accuracy'

    # --- Stacked Configurations to Evaluate ---
    stacked_configs: List[StackedLayerConfig] = field(default_factory=lambda: [
        # Dense baselines at different depths
        StackedLayerConfig(name='Dense_2L', layer_type='dense', num_layers=2),
        StackedLayerConfig(name='Dense_3L', layer_type='dense', num_layers=3),
        StackedLayerConfig(name='Dense_4L', layer_type='dense', num_layers=4),
        StackedLayerConfig(name='Dense_5L', layer_type='dense', num_layers=5),
        # RigidSimplex at different depths
        StackedLayerConfig(name='Simplex_2L', layer_type='simplex', num_layers=2),
        StackedLayerConfig(name='Simplex_3L', layer_type='simplex', num_layers=3),
        StackedLayerConfig(name='Simplex_4L', layer_type='simplex', num_layers=4),
        StackedLayerConfig(name='Simplex_5L', layer_type='simplex', num_layers=5),
    ])

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "cifar10_stacked_simplex_depth"
    random_seed: int = 42

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
# MODEL ARCHITECTURE BUILDING UTILITIES
# ==============================================================================

def build_backbone(
    inputs: keras.KerasTensor,
    config: ExperimentConfig
) -> keras.KerasTensor:
    """
    Build the shared convolutional backbone.

    Args:
        inputs: Input tensor
        config: Experiment configuration

    Returns:
        Feature tensor after backbone processing
    """
    x = inputs

    for i, filters in enumerate(config.conv_filters):
        # First conv in block
        x = layers.Conv2D(
            filters=filters,
            kernel_size=config.kernel_size,
            padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=regularizers.L2(config.weight_decay),
            name=f'conv_block_{i}_conv1'
        )(x)

        if config.use_batch_norm:
            x = layers.BatchNormalization(name=f'conv_block_{i}_bn1')(x)

        x = layers.Activation('relu', name=f'conv_block_{i}_relu1')(x)

        # Second conv in block
        x = layers.Conv2D(
            filters=filters,
            kernel_size=config.kernel_size,
            padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=regularizers.L2(config.weight_decay),
            name=f'conv_block_{i}_conv2'
        )(x)

        if config.use_batch_norm:
            x = layers.BatchNormalization(name=f'conv_block_{i}_bn2')(x)

        x = layers.Activation('relu', name=f'conv_block_{i}_relu2')(x)

        # Pooling (except last block)
        if i < len(config.conv_filters) - 1:
            x = layers.MaxPooling2D(
                pool_size=config.pool_size,
                name=f'conv_block_{i}_pool'
            )(x)

        # Dropout
        if config.backbone_dropout > 0:
            x = layers.Dropout(
                config.backbone_dropout,
                name=f'conv_block_{i}_dropout'
            )(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

    return x


def build_stacked_projection(
    inputs: keras.KerasTensor,
    stacked_config: StackedLayerConfig,
    config: ExperimentConfig
) -> keras.KerasTensor:
    """
    Build stacked projection layers (Dense or RigidSimplex).

    NO BATCH NORMALIZATION between layers - testing raw layer behavior.

    Args:
        inputs: Input tensor from backbone
        stacked_config: Configuration for the stacked layers
        config: Experiment configuration

    Returns:
        Projected feature tensor
    """
    x = inputs

    for layer_idx in range(stacked_config.num_layers):
        if stacked_config.layer_type == 'dense':
            x = layers.Dense(
                units=stacked_config.units_per_layer,
                kernel_initializer=config.kernel_initializer,
                kernel_regularizer=regularizers.L2(config.weight_decay),
                name=f'proj_dense_{layer_idx}'
            )(x)

        elif stacked_config.layer_type == 'simplex':
            x = RigidSimplexLayer(
                units=stacked_config.units_per_layer,
                name=f'proj_simplex_{layer_idx}',
                **stacked_config.simplex_kwargs
            )(x)

        else:
            raise ValueError(f"Unknown layer type: {stacked_config.layer_type}")

        # Activation after each layer (NO BatchNorm)
        x = layers.Activation('relu', name=f'proj_relu_{layer_idx}')(x)

    return x


def build_model(
    config: ExperimentConfig,
    stacked_config: StackedLayerConfig
) -> keras.Model:
    """
    Build a complete CNN model with stacked projection layers.

    Args:
        config: Experiment configuration
        stacked_config: Configuration for the stacked projection layers

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = keras.Input(shape=config.input_shape, name='input')

    # Shared backbone
    features = build_backbone(inputs, config)

    # Stacked projection layers (NO normalization between)
    projected = build_stacked_projection(features, stacked_config, config)

    # Dropout before classification
    projected = layers.Dropout(0.5, name='final_dropout')(projected)

    # Classification head
    logits = layers.Dense(
        units=config.num_classes,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=regularizers.L2(config.weight_decay),
        name='logits'
    )(projected)

    # Softmax output
    predictions = layers.Activation('softmax', name='predictions')(logits)

    # Create model
    model = keras.Model(
        inputs=inputs,
        outputs=predictions,
        name=f'{stacked_config.name}_model'
    )

    # Compile
    optimizer = keras.optimizers.AdamW(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )

    return model


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def train_single_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: ExperimentConfig,
    steps_per_epoch: int,
    val_steps: int,
    config_name: str,
    output_dir: Path
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
        config_name: Name of configuration
        output_dir: Directory to save checkpoints

    Returns:
        Training history dictionary
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            filepath=str(checkpoint_dir / f'{config_name}_best.keras'),
            monitor=config.monitor_metric,
            save_best_only=True,
            verbose=0
        )
    ]

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


def extract_stacked_simplex_analysis(
    model: keras.Model,
    config_name: str
) -> Optional[StackedSimplexAnalysis]:
    """
    Extract analysis data from stacked RigidSimplexLayers in model.

    Args:
        model: Trained Keras model
        config_name: Name of the configuration

    Returns:
        StackedSimplexAnalysis if model contains RigidSimplexLayers, else None
    """
    # Find all RigidSimplexLayers
    simplex_layers = [
        layer for layer in model.layers
        if isinstance(layer, RigidSimplexLayer)
    ]

    if not simplex_layers:
        return None

    rotation_matrices = []
    scale_values = []
    orthogonality_errors = []

    for layer in simplex_layers:
        R = ops.convert_to_numpy(layer.rotation_kernel)
        scale = float(ops.convert_to_numpy(layer.global_scale)[0])

        rotation_matrices.append(R)
        scale_values.append(scale)

        # Compute orthogonality error
        RtR = R.T @ R
        I = np.eye(R.shape[0])
        ortho_error = float(np.linalg.norm(RtR - I, 'fro'))
        orthogonality_errors.append(ortho_error)

    # Compute cumulative transform (product of square rotation matrices only)
    # First layer may have different input/output dims, so we only multiply
    # layers with matching dimensions (all layers after the first typically
    # have the same square rotation matrix shape)

    # Find layers with same-sized rotation matrices (square matrices of units_per_layer)
    square_rotations = [R for R in rotation_matrices if R.shape[0] == R.shape[1]]

    if len(square_rotations) >= 2:
        # Group by size and compute cumulative for matching sizes
        target_size = square_rotations[-1].shape[0]  # Use last layer's size
        matching_rotations = [R for R in square_rotations if R.shape[0] == target_size]

        if matching_rotations:
            cumulative = matching_rotations[0].copy()
            for R in matching_rotations[1:]:
                cumulative = cumulative @ R
        else:
            cumulative = square_rotations[-1].copy()
    elif square_rotations:
        cumulative = square_rotations[0].copy()
    else:
        # Fallback: use last rotation matrix
        cumulative = rotation_matrices[-1].copy()

    return StackedSimplexAnalysis(
        config_name=config_name,
        num_layers=len(simplex_layers),
        rotation_matrices=rotation_matrices,
        scale_values=scale_values,
        orthogonality_errors=orthogonality_errors,
        cumulative_transform=cumulative
    )


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete stacked RigidSimplex depth comparison experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary containing all experimental results
    """
    # Set random seed
    keras.utils.set_random_seed(config.random_seed)

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualization manager
    viz_config = PlotConfig(
        style=PlotStyle.SCIENTIFIC,
        color_scheme=ColorScheme(
            primary='#2E86AB',
            secondary='#F18F01',
            accent='#A23B72'
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
    viz_manager.register_template("depth_comparison_dashboard", DepthComparisonDashboard)
    viz_manager.register_template("stacked_simplex_analysis", StackedSimplexAnalysisDashboard)

    # Log experiment start
    logger.info("=" * 80)
    logger.info("🚀 Starting Stacked RigidSimplex Depth Comparison Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("📊 Loading CIFAR-10 dataset...")

    train_config = TrainingConfig(
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        batch_size=config.batch_size,
        epochs=config.epochs
    )

    dataset_builder = create_dataset_builder('cifar10', train_config)
    train_ds, val_ds, steps_per_epoch, val_steps = dataset_builder.build()

    test_data = dataset_builder.get_test_data()
    class_names = dataset_builder.get_class_names()

    logger.info("✅ Dataset loaded successfully")

    # ===== MODEL TRAINING PHASE =====
    logger.info("🏋️ Starting model training phase...")

    trained_models = {}
    all_histories = {}
    stacked_analyses = []

    for stacked_config in config.stacked_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"--- Training: {stacked_config.name} ---")
        logger.info(f"    Type: {stacked_config.layer_type}")
        logger.info(f"    Depth: {stacked_config.num_layers} layers")
        logger.info(f"    Units per layer: {stacked_config.units_per_layer}")

        # Build model
        model = build_model(config, stacked_config)

        # Log model summary
        model.summary(print_fn=lambda x: logger.info(x))
        logger.info(f"Total parameters: {model.count_params():,}")

        # Train
        history = train_single_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            config=config,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
            config_name=stacked_config.name,
            output_dir=experiment_dir
        )

        # Store results
        trained_models[stacked_config.name] = model
        all_histories[stacked_config.name] = history

        # Extract stacked Simplex analysis
        analysis = extract_stacked_simplex_analysis(model, stacked_config.name)
        if analysis is not None:
            stacked_analyses.append(analysis)
            logger.info(f"   Simplex scales: {[f'{s:.3f}' for s in analysis.scale_values]}")
            logger.info(f"   Ortho errors: {[f'{e:.2f}' for e in analysis.orthogonality_errors]}")

        logger.info(f"✅ {stacked_config.name} training completed!")

    # ===== MEMORY MANAGEMENT =====
    logger.info("\n🗑️ Triggering garbage collection...")
    gc.collect()

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("\n📈 Evaluating final model performance on test set...")

    performance_results = {}
    all_predictions = {}
    depth_info = {}

    for stacked_config in config.stacked_configs:
        name = stacked_config.name
        model = trained_models[name]

        logger.info(f"Evaluating model {name}...")

        # Get predictions
        predictions = model.predict(test_data.x_data, verbose=0)
        y_pred_classes = np.argmax(predictions, axis=1)

        # Calculate metrics
        y_true = test_data.y_data.astype(int)
        accuracy = np.mean(y_pred_classes == y_true)

        # Top-5 accuracy
        top_5_predictions = np.argsort(predictions, axis=1)[:, -5:]
        top5_acc = np.mean([
            y_true_val in top5_pred
            for y_true_val, top5_pred in zip(y_true, top_5_predictions)
        ])

        # Loss
        loss = -np.mean(
            np.log(predictions[np.arange(len(y_true)), y_true] + 1e-7)
        )

        # Parameter count
        trainable_params = sum(
            np.prod(w.shape) for w in model.trainable_weights
        )

        performance_results[name] = {
            'accuracy': accuracy,
            'top_5_accuracy': top5_acc,
            'loss': loss,
            'trainable_params': int(trainable_params)
        }

        all_predictions[name] = y_pred_classes
        depth_info[name] = stacked_config.num_layers

        logger.info(f"   {name} - Accuracy: {accuracy:.4f}, Top-5: {top5_acc:.4f}")

    # ===== VISUALIZATION GENERATION =====
    logger.info("\n🖼️ Generating visualizations...")

    # 1. Training curves comparison
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

    # 2. Model comparison
    comparison_data = ModelComparison(
        model_names=list(performance_results.keys()),
        metrics={
            name: {
                'accuracy': metrics['accuracy'],
                'top_5_accuracy': metrics['top_5_accuracy']
            }
            for name, metrics in performance_results.items()
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

    # 3. Confusion matrices
    for name, y_pred in all_predictions.items():
        classification_results = ClassificationResults(
            y_true=test_data.y_data.astype(int),
            y_pred=y_pred,
            class_names=class_names,
            model_name=name
        )

        viz_manager.visualize(
            data=classification_results,
            plugin_name="confusion_matrix",
            normalize='true',
            show=False
        )

    # 4. Depth comparison dashboard
    depth_comparison = DepthComparisonData(
        config_names=[sc.name for sc in config.stacked_configs],
        metrics=performance_results,
        histories=all_histories,
        depth_info=depth_info
    )

    viz_manager.visualize(
        data=depth_comparison,
        plugin_name="depth_comparison_dashboard",
        show=False
    )

    # 5. Stacked Simplex analysis dashboard
    if stacked_analyses:
        viz_manager.visualize(
            data=stacked_analyses,
            plugin_name="stacked_simplex_analysis",
            show=False
        )

    # ===== MODEL ANALYSIS =====
    logger.info("\n📊 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None

    try:
        analyzer = ModelAnalyzer(
            models=trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )

        model_analysis_results = analyzer.analyze(data=test_data)
        logger.info("✅ Model analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Model analysis failed: {e}", exc_info=True)

    # ===== RESULTS COMPILATION =====
    results_payload = {
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'histories': all_histories,
        'trained_models': trained_models,
        'predictions': all_predictions,
        'stacked_analyses': stacked_analyses,
        'depth_info': depth_info,
        'test_data': test_data,
        'class_names': class_names
    }

    # Print summary
    print_experiment_summary(results_payload)

    return results_payload


# ==============================================================================
# RESULTS REPORTING
# ==============================================================================

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print a comprehensive summary of experimental results.

    Args:
        results: Dictionary containing all experimental results
    """
    logger.info("\n" + "=" * 80)
    logger.info("📋 EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # ===== PERFORMANCE METRICS SECTION =====
    if 'performance_analysis' in results and results['performance_analysis']:
        logger.info("\n🎯 PERFORMANCE METRICS (on Test Set):")
        logger.info(f"{'Model':<15} {'Depth':<7} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12} {'Params':<15}")
        logger.info("-" * 80)

        # Sort by layer type, then depth
        sorted_results = sorted(
            results['performance_analysis'].items(),
            key=lambda x: (0 if 'Dense' in x[0] else 1, results['depth_info'].get(x[0], 0))
        )

        for model_name, metrics in sorted_results:
            depth = results['depth_info'].get(model_name, '?')
            accuracy = metrics.get('accuracy', 0.0)
            top5_acc = metrics.get('top_5_accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            params = metrics.get('trainable_params', 0)
            logger.info(
                f"{model_name:<15} {depth:<7} {accuracy:<12.4f} {top5_acc:<12.4f} "
                f"{loss:<12.4f} {params:<15,}"
            )

    # ===== DEPTH SCALING ANALYSIS =====
    if 'performance_analysis' in results and 'depth_info' in results:
        logger.info("\n📈 DEPTH SCALING ANALYSIS:")

        perf = results['performance_analysis']
        depth_info = results['depth_info']

        # Separate by type
        dense_results = {k: v for k, v in perf.items() if 'Dense' in k}
        simplex_results = {k: v for k, v in perf.items() if 'Simplex' in k}

        if dense_results:
            logger.info("\n   Dense Layers:")
            for name in sorted(dense_results.keys(), key=lambda x: depth_info.get(x, 0)):
                acc = dense_results[name]['accuracy']
                depth = depth_info.get(name, '?')
                logger.info(f"      {depth} layers: {acc:.4f}")

        if simplex_results:
            logger.info("\n   Simplex Layers:")
            for name in sorted(simplex_results.keys(), key=lambda x: depth_info.get(x, 0)):
                acc = simplex_results[name]['accuracy']
                depth = depth_info.get(name, '?')
                logger.info(f"      {depth} layers: {acc:.4f}")

    # ===== STACKED SIMPLEX ANALYSIS =====
    if 'stacked_analyses' in results and results['stacked_analyses']:
        logger.info("\n🔬 STACKED SIMPLEX LAYER ANALYSIS:")

        for analysis in sorted(results['stacked_analyses'], key=lambda x: x.num_layers):
            logger.info(f"\n   {analysis.config_name} ({analysis.num_layers} layers):")
            logger.info(f"      Scales: {[f'{s:.3f}' for s in analysis.scale_values]}")
            logger.info(f"      Ortho Errors: {[f'{e:.2f}' for e in analysis.orthogonality_errors]}")
            logger.info(f"      Total Ortho Error: {sum(analysis.orthogonality_errors):.2f}")

            # Cumulative transform analysis
            C = analysis.cumulative_transform
            CtC = C.T @ C
            cumulative_ortho = float(np.linalg.norm(CtC - np.eye(C.shape[0]), 'fro'))
            logger.info(f"      Cumulative Ortho Error: {cumulative_ortho:.2f}")

    # ===== BEST MODELS PER DEPTH =====
    if 'performance_analysis' in results and 'depth_info' in results:
        logger.info("\n🏆 BEST MODEL AT EACH DEPTH:")

        perf = results['performance_analysis']
        depth_info = results['depth_info']

        for depth in [2, 3, 4, 5]:
            depth_models = {k: v for k, v in perf.items() if depth_info.get(k) == depth}
            if depth_models:
                best = max(depth_models.items(), key=lambda x: x[1]['accuracy'])
                logger.info(f"   Depth {depth}: {best[0]} (Acc: {best[1]['accuracy']:.4f})")

    # ===== OVERALL BEST =====
    if 'performance_analysis' in results:
        best_model = max(
            results['performance_analysis'].items(),
            key=lambda x: x[1]['accuracy']
        )
        logger.info(f"\n🏆 OVERALL BEST: {best_model[0]}")
        logger.info(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        logger.info(f"   Top-5 Accuracy: {best_model[1]['top_5_accuracy']:.4f}")
        logger.info(f"   Depth: {results['depth_info'].get(best_model[0], '?')} layers")

    logger.info("\n" + "=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for the stacked RigidSimplex depth experiment.
    """
    logger.info("🚀 Stacked RigidSimplex Depth Comparison Experiment")
    logger.info("=" * 80)

    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ GPU configured: {len(gpus)} device(s) available")
        except RuntimeError as e:
            logger.warning(f"GPU configuration warning: {e}")

    # Initialize configuration
    config = ExperimentConfig()

    # Log configuration
    logger.info("\n⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Configurations: {[sc.name for sc in config.stacked_configs]}")
    logger.info(f"   Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"   Units per layer: {config.units_per_layer}")
    logger.info(f"   NOTE: No BatchNormalization between projection layers")
    logger.info("")

    try:
        results = run_experiment(config)
        logger.info("\n✅ Experiment completed successfully!")

    except Exception as e:
        logger.error(f"\n❌ Experiment failed with error: {e}", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()