"""
CIFAR-10 Rigid Simplex Layer Comparison: Evaluating Geometric Constraints
=========================================================================

This experiment conducts a comprehensive comparison of the RigidSimplexLayer
against standard Dense layers for image classification on CIFAR-10, systematically
evaluating the effectiveness of geometric constraints in neural networks.

The study addresses fundamental questions in deep learning: how do rigid geometric
constraints (Equiangular Tight Frames) affect model performance, training dynamics,
weight distributions, and feature representations? By comparing template matching
approaches (RigidSimplexLayer) with unconstrained learning (Dense), this experiment
provides insights into the trade-offs between parameter efficiency, training
stability, and model expressiveness.

Theoretical Background
----------------------

The RigidSimplexLayer implements a constrained projection based on Equiangular
Tight Frames (ETF). Key properties:

1. **Fixed Geometry**: The Simplex structure is frozen during training
2. **Learnable Rotation**: Only the input rotation matrix is trainable
3. **Bounded Scaling**: Global scale is constrained to a specified range
4. **Template Matching**: Forces the network to align inputs with fixed patterns

This contrasts with standard Dense layers which learn arbitrary linear transformations.

Experimental Design
-------------------

**Dataset**: CIFAR-10 (10 classes, 32x32 RGB images)
- 50,000 training images
- 10,000 test images
- Standard preprocessing with normalization and augmentation

**Model Architecture**: CNN backbone with interchangeable projection layers:
- Convolutional feature extractor (shared across all variants)
- Global average pooling
- Projection layer (Dense vs RigidSimplex variants)
- Classification head

**Layer Configurations Evaluated**:
1. **Dense_Baseline**: Standard Dense layer (unconstrained)
2. **Dense_L2**: Dense layer with L2 regularization
3. **Simplex_Default**: RigidSimplexLayer with default settings
4. **Simplex_Tight**: RigidSimplexLayer with tight scale bounds [0.8, 1.2]
5. **Simplex_Wide**: RigidSimplexLayer with wide scale bounds [0.1, 10.0]
6. **Simplex_HighOrtho**: RigidSimplexLayer with high orthogonality penalty
7. **Simplex_NoOrtho**: RigidSimplexLayer with zero orthogonality penalty
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
    MultiModelClassification,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingCurvesVisualization,
    ModelComparisonBarChart,
    PerformanceRadarChart,
    ConfusionMatrixVisualization,
    ROCPRCurves,
    ConvergenceAnalysis,
    OverfittingAnalysis,
    ErrorAnalysisDashboard,
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
class LayerComparisonData:
    """
    Data structure for layer comparison visualization.

    Attributes:
        layer_names: Names of layer configurations compared
        metrics: Dictionary mapping layer names to their metrics
        histories: Dictionary mapping layer names to training histories
        weight_stats: Optional dictionary of weight statistics per layer
    """
    layer_names: List[str]
    metrics: Dict[str, Dict[str, float]]
    histories: Dict[str, Dict[str, List[float]]]
    weight_stats: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class SimplexAnalysisData:
    """
    Data structure for Simplex-specific analysis visualization.

    Attributes:
        layer_name: Name of the layer configuration
        rotation_matrix: The learned rotation matrix
        scale_value: The learned scale value
        simplex_matrix: The fixed simplex matrix
        orthogonality_error: Deviation from orthogonality (||R^T R - I||)
    """
    layer_name: str
    rotation_matrix: np.ndarray
    scale_value: float
    simplex_matrix: np.ndarray
    orthogonality_error: float


# ==============================================================================
# CUSTOM VISUALIZATION PLUGINS
# ==============================================================================

class LayerComparisonDashboard(VisualizationPlugin):
    """Comprehensive dashboard for layer comparison."""

    @property
    def name(self) -> str:
        return "layer_comparison_dashboard"

    @property
    def description(self) -> str:
        return "Comprehensive dashboard comparing Dense vs Simplex layers"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, LayerComparisonData)

    def create_visualization(
        self,
        data: LayerComparisonData,
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create comprehensive comparison dashboard."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(24, 16), dpi=self.config.dpi)
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. Training accuracy comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_training_curves(data, ax1, 'accuracy', 'Training Accuracy')

        # 2. Validation accuracy comparison
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_training_curves(data, ax2, 'val_accuracy', 'Validation Accuracy')

        # 3. Training loss comparison
        ax3 = fig.add_subplot(gs[2, :2])
        self._plot_training_curves(data, ax3, 'loss', 'Training Loss')

        # 4. Validation loss comparison
        ax4 = fig.add_subplot(gs[3, :2])
        self._plot_training_curves(data, ax4, 'val_loss', 'Validation Loss')

        # 5. Final accuracy comparison (bar chart)
        ax5 = fig.add_subplot(gs[0, 2])
        self._plot_final_metrics(data, ax5, 'accuracy', 'Final Accuracy')

        # 6. Top-5 accuracy comparison
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_final_metrics(data, ax6, 'top_5_accuracy', 'Top-5 Accuracy')

        # 7. Parameter efficiency comparison
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_parameter_efficiency(data, ax7)

        # 8. Summary table
        ax8 = fig.add_subplot(gs[3, 2])
        self._plot_summary_table(data, ax8)

        plt.suptitle(
            'Dense vs RigidSimplex Layer Comparison Dashboard',
            fontsize=self.config.title_fontsize + 4,
            fontweight='bold',
            y=0.98
        )

        return fig

    def _plot_training_curves(
        self,
        data: LayerComparisonData,
        ax: Any,
        metric: str,
        title: str
    ) -> None:
        """Plot training curves for a specific metric."""
        import matplotlib.pyplot as plt

        # Use distinct colors for Dense vs Simplex
        dense_colors = plt.cm.Blues(np.linspace(0.4, 0.8, 3))
        simplex_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, 5))

        color_idx_dense = 0
        color_idx_simplex = 0

        for name in data.layer_names:
            if name in data.histories and metric in data.histories[name]:
                epochs = range(len(data.histories[name][metric]))

                if 'Dense' in name:
                    color = dense_colors[color_idx_dense % len(dense_colors)]
                    color_idx_dense += 1
                    linestyle = '-'
                else:
                    color = simplex_colors[color_idx_simplex % len(simplex_colors)]
                    color_idx_simplex += 1
                    linestyle = '--'

                ax.plot(
                    epochs,
                    data.histories[name][metric],
                    label=name,
                    color=color,
                    linewidth=2,
                    linestyle=linestyle,
                    alpha=0.85
                )

        ax.set_xlabel('Epoch', fontsize=self.config.label_fontsize)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(alpha=0.3)

    def _plot_final_metrics(
        self,
        data: LayerComparisonData,
        ax: Any,
        metric: str,
        title: str
    ) -> None:
        """Plot bar chart of final metrics."""
        import matplotlib.pyplot as plt

        values = [
            data.metrics[name].get(metric, 0.0)
            for name in data.layer_names
        ]

        # Color by layer type
        colors = [
            '#2E86AB' if 'Dense' in name else '#F18F01'
            for name in data.layer_names
        ]

        bars = ax.bar(range(len(data.layer_names)), values, color=colors, alpha=0.8)

        ax.set_xticks(range(len(data.layer_names)))
        ax.set_xticklabels(data.layer_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.grid(alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom',
                fontsize=8
            )

    def _plot_parameter_efficiency(
        self,
        data: LayerComparisonData,
        ax: Any
    ) -> None:
        """Plot parameter efficiency (accuracy per parameter)."""
        import matplotlib.pyplot as plt

        if 'trainable_params' not in data.metrics.get(data.layer_names[0], {}):
            ax.text(0.5, 0.5, 'Parameter data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Parameter Efficiency', fontsize=self.config.title_fontsize)
            return

        accuracies = [data.metrics[name].get('accuracy', 0.0) for name in data.layer_names]
        params = [data.metrics[name].get('trainable_params', 1) for name in data.layer_names]

        # Efficiency = accuracy / (params / 1000)
        efficiency = [acc / (p / 1000) if p > 0 else 0 for acc, p in zip(accuracies, params)]

        colors = ['#2E86AB' if 'Dense' in name else '#F18F01' for name in data.layer_names]
        bars = ax.bar(range(len(data.layer_names)), efficiency, color=colors, alpha=0.8)

        ax.set_xticks(range(len(data.layer_names)))
        ax.set_xticklabels(data.layer_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy per 1K Params', fontsize=self.config.label_fontsize)
        ax.set_title('Parameter Efficiency', fontsize=self.config.title_fontsize)
        ax.grid(alpha=0.3, axis='y')

    def _plot_summary_table(
        self,
        data: LayerComparisonData,
        ax: Any
    ) -> None:
        """Plot summary table of all metrics."""
        ax.axis('off')

        # Prepare table data
        table_data = []
        for name in data.layer_names:
            metrics = data.metrics[name]
            row = [
                name[:15] + '...' if len(name) > 15 else name,
                f"{metrics.get('accuracy', 0.0):.4f}",
                f"{metrics.get('top_5_accuracy', 0.0):.4f}",
                f"{metrics.get('loss', 0.0):.4f}",
                f"{metrics.get('trainable_params', 0):,}"
            ]
            table_data.append(row)

        table = ax.table(
            cellText=table_data,
            colLabels=['Layer', 'Acc', 'Top-5', 'Loss', 'Params'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Performance Summary', fontsize=self.config.title_fontsize, pad=20)


class SimplexAnalysisDashboard(VisualizationPlugin):
    """Dashboard for analyzing Simplex layer internals."""

    @property
    def name(self) -> str:
        return "simplex_analysis_dashboard"

    @property
    def description(self) -> str:
        return "Dashboard analyzing RigidSimplexLayer learned parameters"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, list) and all(isinstance(d, SimplexAnalysisData) for d in data)

    def create_visualization(
        self,
        data: List[SimplexAnalysisData],
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create Simplex analysis dashboard."""
        import matplotlib.pyplot as plt

        n_models = len(data)
        fig = plt.figure(figsize=(6 * n_models, 12), dpi=self.config.dpi)
        gs = fig.add_gridspec(3, n_models, hspace=0.3, wspace=0.3)

        for idx, simplex_data in enumerate(data):
            # 1. Rotation matrix heatmap
            ax1 = fig.add_subplot(gs[0, idx])
            self._plot_rotation_matrix(simplex_data, ax1)

            # 2. Simplex matrix structure
            ax2 = fig.add_subplot(gs[1, idx])
            self._plot_simplex_structure(simplex_data, ax2)

            # 3. Orthogonality analysis
            ax3 = fig.add_subplot(gs[2, idx])
            self._plot_orthogonality_analysis(simplex_data, ax3)

        plt.suptitle(
            'RigidSimplexLayer Internal Analysis',
            fontsize=self.config.title_fontsize + 2,
            fontweight='bold'
        )

        return fig

    def _plot_rotation_matrix(self, data: SimplexAnalysisData, ax: Any) -> None:
        """Plot rotation matrix as heatmap."""
        import matplotlib.pyplot as plt

        # Show center portion if matrix is large
        R = data.rotation_matrix
        if R.shape[0] > 32:
            center = R.shape[0] // 2
            R = R[center-16:center+16, center-16:center+16]

        im = ax.imshow(R, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_title(f'{data.layer_name}\nRotation Matrix (scale={data.scale_value:.3f})',
                    fontsize=self.config.title_fontsize - 2)
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_simplex_structure(self, data: SimplexAnalysisData, ax: Any) -> None:
        """Plot simplex matrix Gram structure."""
        import matplotlib.pyplot as plt

        S = data.simplex_matrix
        gram = S.T @ S

        # Show subset if large
        if gram.shape[0] > 32:
            gram = gram[:32, :32]

        im = ax.imshow(gram, cmap='viridis', aspect='auto')
        ax.set_title(f'Simplex Gram Matrix (S^T S)\nShape: {data.simplex_matrix.shape}',
                    fontsize=self.config.title_fontsize - 2)
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_orthogonality_analysis(self, data: SimplexAnalysisData, ax: Any) -> None:
        """Plot orthogonality deviation analysis."""
        import matplotlib.pyplot as plt

        R = data.rotation_matrix
        RtR = R.T @ R
        I = np.eye(R.shape[0])
        deviation = RtR - I

        # Histogram of off-diagonal elements
        off_diag = deviation[~np.eye(deviation.shape[0], dtype=bool)]

        ax.hist(off_diag, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Ideal (0)')
        ax.set_title(f'Orthogonality Deviation\n||R^T R - I||_F = {data.orthogonality_error:.6f}',
                    fontsize=self.config.title_fontsize - 2)
        ax.set_xlabel('Deviation from Identity')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(alpha=0.3)


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
    batch_size: int = 64
    epochs: int = 100


@dataclass
class LayerConfig:
    """
    Configuration for a single layer variant.

    Attributes:
        name: Display name for the layer configuration
        layer_type: Either 'dense' or 'simplex'
        units: Output dimensionality
        kwargs: Additional keyword arguments for the layer
    """
    name: str
    layer_type: str  # 'dense' or 'simplex'
    units: int
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """
    Configuration for the CIFAR-10 RigidSimplex comparison experiment.

    This class encapsulates all configurable parameters for the experiment,
    including dataset configuration, model architecture parameters, training
    settings, layer configurations, and analysis configuration.
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
    use_batch_norm: bool = True
    backbone_dropout: float = 0.25

    # --- Projection Layer Configuration ---
    projection_units: int = 128  # Dimension before classification

    # --- Training Parameters ---
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 50
    monitor_metric: str = 'val_accuracy'

    # --- Layer Configurations to Evaluate ---
    layer_configs: List[LayerConfig] = field(default_factory=lambda: [
        # Dense baselines
        LayerConfig(
            name='Dense_Baseline',
            layer_type='dense',
            units=128,
            kwargs={}
        ),
        LayerConfig(
            name='Dense_L2',
            layer_type='dense',
            units=128,
            kwargs={'kernel_regularizer': 'l2'}
        ),
        # RigidSimplex variants
        LayerConfig(
            name='Simplex_Default',
            layer_type='simplex',
            units=128,
            kwargs={
                'scale_min': 0.5,
                'scale_max': 2.0,
                'orthogonality_penalty': 1e-4
            }
        ),
        LayerConfig(
            name='Simplex_Tight',
            layer_type='simplex',
            units=128,
            kwargs={
                'scale_min': 0.8,
                'scale_max': 1.2,
                'orthogonality_penalty': 1e-4
            }
        ),
        LayerConfig(
            name='Simplex_Wide',
            layer_type='simplex',
            units=128,
            kwargs={
                'scale_min': 0.1,
                'scale_max': 10.0,
                'orthogonality_penalty': 1e-4
            }
        ),
        LayerConfig(
            name='Simplex_HighOrtho',
            layer_type='simplex',
            units=128,
            kwargs={
                'scale_min': 0.5,
                'scale_max': 2.0,
                'orthogonality_penalty': 1e-2
            }
        ),
        LayerConfig(
            name='Simplex_NoOrtho',
            layer_type='simplex',
            units=128,
            kwargs={
                'scale_min': 0.5,
                'scale_max': 2.0,
                'orthogonality_penalty': 0.0
            }
        ),
    ])

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "cifar10_simplex_comparison"
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
        # Convolutional block
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


def build_projection_layer(
    inputs: keras.KerasTensor,
    layer_config: LayerConfig,
    config: ExperimentConfig
) -> keras.KerasTensor:
    """
    Build the projection layer (Dense or RigidSimplex).

    Args:
        inputs: Input tensor from backbone
        layer_config: Configuration for this specific layer
        config: Experiment configuration

    Returns:
        Projected feature tensor
    """
    if layer_config.layer_type == 'dense':
        # Build Dense layer
        kwargs = layer_config.kwargs.copy()

        # Handle regularizer string
        if kwargs.get('kernel_regularizer') == 'l2':
            kwargs['kernel_regularizer'] = regularizers.L2(config.weight_decay)

        x = layers.Dense(
            units=layer_config.units,
            kernel_initializer=config.kernel_initializer,
            name='projection_dense',
            **kwargs
        )(inputs)

    elif layer_config.layer_type == 'simplex':
        # Build RigidSimplexLayer
        x = RigidSimplexLayer(
            units=layer_config.units,
            name='projection_simplex',
            **layer_config.kwargs
        )(inputs)

    else:
        raise ValueError(f"Unknown layer type: {layer_config.layer_type}")

    return x


def build_model(
    config: ExperimentConfig,
    layer_config: LayerConfig
) -> keras.Model:
    """
    Build a complete CNN model with specified projection layer.

    Args:
        config: Experiment configuration
        layer_config: Configuration for the projection layer

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = keras.Input(shape=config.input_shape, name='input')

    # Shared backbone
    features = build_backbone(inputs, config)

    # Projection layer (Dense or RigidSimplex)
    projected = build_projection_layer(features, layer_config, config)

    # Batch norm and activation after projection
    if config.use_batch_norm:
        projected = layers.BatchNormalization(name='projection_bn')(projected)
    projected = layers.Activation('relu', name='projection_relu')(projected)

    # Dropout before classification
    projected = layers.Dropout(0.5, name='projection_dropout')(projected)

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
        name=f'{layer_config.name}_model'
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
    layer_name: str,
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
        layer_name: Name of layer configuration
        output_dir: Directory to save checkpoints

    Returns:
        Training history dictionary
    """
    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            filepath=str(checkpoint_dir / f'{layer_name}_best.keras'),
            monitor=config.monitor_metric,
            save_best_only=True,
            verbose=0
        )
    ]

    # Train
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


def extract_simplex_analysis(
    model: keras.Model,
    layer_name: str
) -> Optional[SimplexAnalysisData]:
    """
    Extract analysis data from RigidSimplexLayer in model.

    Args:
        model: Trained Keras model
        layer_name: Name of the layer configuration

    Returns:
        SimplexAnalysisData if model contains RigidSimplexLayer, else None
    """
    # Find RigidSimplexLayer
    simplex_layer = None
    for layer in model.layers:
        if isinstance(layer, RigidSimplexLayer):
            simplex_layer = layer
            break

    if simplex_layer is None:
        return None

    # Extract weights
    rotation_matrix = ops.convert_to_numpy(simplex_layer.rotation_kernel)
    scale_value = float(ops.convert_to_numpy(simplex_layer.global_scale)[0])
    simplex_matrix = ops.convert_to_numpy(simplex_layer.static_simplex)

    # Compute orthogonality error
    RtR = rotation_matrix.T @ rotation_matrix
    I = np.eye(rotation_matrix.shape[0])
    orthogonality_error = float(np.linalg.norm(RtR - I, 'fro'))

    return SimplexAnalysisData(
        layer_name=layer_name,
        rotation_matrix=rotation_matrix,
        scale_value=scale_value,
        simplex_matrix=simplex_matrix,
        orthogonality_error=orthogonality_error
    )


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete CIFAR-10 RigidSimplex comparison experiment.

    This function orchestrates the entire experimental pipeline, including:
    1. Dataset loading and preprocessing
    2. Model training for each layer configuration
    3. Model analysis and evaluation
    4. Visualization generation
    5. Results compilation and reporting

    Args:
        config: Experiment configuration specifying all parameters

    Returns:
        Dictionary containing all experimental results and analysis
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
            secondary='#F18F01'
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
    viz_manager.register_template("layer_comparison_dashboard", LayerComparisonDashboard)
    viz_manager.register_template("simplex_analysis_dashboard", SimplexAnalysisDashboard)
    
    # Newly registered plugins for comprehensive analysis
    viz_manager.register_template("roc_pr_curves", ROCPRCurves)
    viz_manager.register_template("convergence_analysis", ConvergenceAnalysis)
    viz_manager.register_template("overfitting_analysis", OverfittingAnalysis)
    viz_manager.register_template("error_analysis", ErrorAnalysisDashboard)

    # Log experiment start
    logger.info("=" * 80)
    logger.info("Starting CIFAR-10 RigidSimplex Layer Comparison Experiment")
    logger.info(f"Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("Loading CIFAR-10 dataset...")

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

    logger.info("Dataset loaded successfully")
    logger.info(f"   Steps per epoch: {steps_per_epoch}, Validation steps: {val_steps}")
    logger.info(f"   Test data shape: {test_data.x_data.shape}")

    # ===== MODEL TRAINING PHASE =====
    logger.info("Starting model training phase...")

    trained_models = {}
    all_histories = {}
    simplex_analyses = []

    for layer_config in config.layer_configs:
        logger.info(f"{'='*60}")
        logger.info(f"--- Training model: {layer_config.name} ---")
        logger.info(f"    Type: {layer_config.layer_type}, Units: {layer_config.units}")
        logger.info(f"    Kwargs: {layer_config.kwargs}")

        # Build model
        model = build_model(config, layer_config)

        # Log model summary
        model.summary(print_fn=lambda x: logger.info(x))
        logger.info(f"Total parameters: {model.count_params():,}")

        # Count trainable params in projection layer
        projection_params = 0
        for layer in model.layers:
            if 'projection' in layer.name:
                projection_params += layer.count_params()
        logger.info(f"Projection layer parameters: {projection_params:,}")

        # Train
        history = train_single_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            config=config,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
            layer_name=layer_config.name,
            output_dir=experiment_dir
        )

        # Store results
        trained_models[layer_config.name] = model
        all_histories[layer_config.name] = history

        # Extract Simplex analysis if applicable
        simplex_analysis = extract_simplex_analysis(model, layer_config.name)
        if simplex_analysis is not None:
            simplex_analyses.append(simplex_analysis)
            logger.info(f"   Simplex orthogonality error: {simplex_analysis.orthogonality_error:.6f}")
            logger.info(f"   Simplex learned scale: {simplex_analysis.scale_value:.4f}")

        logger.info(f"{layer_config.name} training completed!")

    # ===== MEMORY MANAGEMENT =====
    logger.info("Triggering garbage collection...")
    gc.collect()

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("Evaluating final model performance on test set...")

    performance_results = {}
    all_predictions = {}
    all_probabilities = {} # New: store probability outputs for advanced metrics

    for name, model in trained_models.items():
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

        # Count trainable params
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
        all_probabilities[name] = predictions # Store probabilities

        logger.info(f"   {name} - Accuracy: {accuracy:.4f}, Top-5: {top5_acc:.4f}")

    # ===== VISUALIZATION GENERATION =====
    logger.info("Generating visualizations...")

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
    
    # 1.1 Training Dynamics Analysis (New)
    # Check convergence properties
    viz_manager.visualize(
        data=training_histories,
        plugin_name="convergence_analysis",
        show=False
    )
    # Check overfitting properties
    viz_manager.visualize(
        data=training_histories,
        plugin_name="overfitting_analysis",
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

    # 3. Classification Results Collection and Individual Analysis
    all_classification_results = {}
    
    for name, y_pred in all_predictions.items():
        classification_results = ClassificationResults(
            y_true=test_data.y_data.astype(int),
            y_pred=y_pred,
            y_prob=all_probabilities[name], # Pass probabilities for advanced metrics
            class_names=class_names,
            model_name=name
        )
        all_classification_results[name] = classification_results

        # Individual Error Analysis (requires y_prob)
        viz_manager.visualize(
            data=classification_results,
            plugin_name="error_analysis",
            show=False,
            # Pass raw x_data if the plugin supports showing examples
            x_data=test_data.x_data 
        )

    # 4. Multi-Model Classification Analysis (New)
    # Combined analysis for Confusion Matrices and ROC/PR Curves
    multi_model_data = MultiModelClassification(
        results=all_classification_results,
        dataset_name="CIFAR-10 Test"
    )

    # Combined Confusion Matrices
    viz_manager.visualize(
        data=multi_model_data,
        plugin_name="confusion_matrix",
        normalize='true',
        show=False
    )

    # Combined ROC/PR Curves
    viz_manager.visualize(
        data=multi_model_data,
        plugin_name="roc_pr_curves",
        plot_type='both',
        show=False
    )

    # 5. Layer comparison dashboard
    layer_comparison = LayerComparisonData(
        layer_names=list(config.layer_configs[i].name for i in range(len(config.layer_configs))),
        metrics=performance_results,
        histories=all_histories
    )

    viz_manager.visualize(
        data=layer_comparison,
        plugin_name="layer_comparison_dashboard",
        show=False
    )

    # 6. Simplex analysis dashboard (if we have simplex models)
    if simplex_analyses:
        viz_manager.visualize(
            data=simplex_analyses,
            plugin_name="simplex_analysis_dashboard",
            show=False
        )

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None

    try:
        analyzer = ModelAnalyzer(
            models=trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )

        model_analysis_results = analyzer.analyze(data=test_data)
        logger.info("Model analysis completed successfully!")

    except Exception as e:
        logger.error(f"Model analysis failed: {e}", exc_info=True)

    # ===== RESULTS COMPILATION =====
    results_payload = {
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'histories': all_histories,
        'trained_models': trained_models,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'simplex_analyses': simplex_analyses,
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
    logger.info("=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # ===== PERFORMANCE METRICS SECTION =====
    if 'performance_analysis' in results and results['performance_analysis']:
        logger.info("PERFORMANCE METRICS (on Test Set):")
        logger.info(f"{'Model':<20} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12} {'Params':<15}")
        logger.info("-" * 75)

        # Sort by accuracy
        sorted_results = sorted(
            results['performance_analysis'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        for model_name, metrics in sorted_results:
            accuracy = metrics.get('accuracy', 0.0)
            top5_acc = metrics.get('top_5_accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            params = metrics.get('trainable_params', 0)
            logger.info(
                f"{model_name:<20} {accuracy:<12.4f} {top5_acc:<12.4f} "
                f"{loss:<12.4f} {params:<15,}"
            )

    # ===== SIMPLEX ANALYSIS SECTION =====
    if 'simplex_analyses' in results and results['simplex_analyses']:
        logger.info("SIMPLEX LAYER ANALYSIS:")
        logger.info(f"{'Model':<20} {'Scale':<12} {'Ortho Error':<15}")
        logger.info("-" * 50)

        for analysis in results['simplex_analyses']:
            logger.info(
                f"{analysis.layer_name:<20} {analysis.scale_value:<12.4f} "
                f"{analysis.orthogonality_error:<15.6f}"
            )

    # ===== DENSE VS SIMPLEX COMPARISON =====
    if 'performance_analysis' in results:
        perf = results['performance_analysis']

        dense_models = {k: v for k, v in perf.items() if 'Dense' in k}
        simplex_models = {k: v for k, v in perf.items() if 'Simplex' in k}

        if dense_models and simplex_models:
            best_dense = max(dense_models.items(), key=lambda x: x[1]['accuracy'])
            best_simplex = max(simplex_models.items(), key=lambda x: x[1]['accuracy'])

            logger.info("DENSE vs SIMPLEX COMPARISON:")
            logger.info(f"   Best Dense:   {best_dense[0]} (Acc: {best_dense[1]['accuracy']:.4f})")
            logger.info(f"   Best Simplex: {best_simplex[0]} (Acc: {best_simplex[1]['accuracy']:.4f})")

            diff = best_simplex[1]['accuracy'] - best_dense[1]['accuracy']
            if diff > 0:
                logger.info(f"   Simplex outperforms Dense by {diff:.4f}")
            elif diff < 0:
                logger.info(f"   Dense outperforms Simplex by {-diff:.4f}")
            else:
                logger.info(f"   Equal performance")

            # Parameter efficiency
            dense_params = best_dense[1].get('trainable_params', 1)
            simplex_params = best_simplex[1].get('trainable_params', 1)
            param_ratio = simplex_params / dense_params if dense_params > 0 else 0

            logger.info(f"   Parameter comparison:")
            logger.info(f"   Dense params:   {dense_params:,}")
            logger.info(f"   Simplex params: {simplex_params:,}")
            logger.info(f"   Ratio: {param_ratio:.2f}x")

    # ===== BEST MODEL IDENTIFICATION =====
    if 'performance_analysis' in results:
        best_model = max(
            results['performance_analysis'].items(),
            key=lambda x: x[1]['accuracy']
        )
        logger.info(f"BEST MODEL: {best_model[0]}")
        logger.info(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        logger.info(f"   Top-5 Accuracy: {best_model[1]['top_5_accuracy']:.4f}")

    logger.info("" + "=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for the RigidSimplex comparison experiment.
    """
    logger.info("CIFAR-10 RigidSimplex Layer Comparison Experiment")
    logger.info("=" * 80)

    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU configured: {len(gpus)} device(s) available")
        except RuntimeError as e:
            logger.warning(f"GPU configuration warning: {e}")

    # Initialize configuration
    config = ExperimentConfig()

    # Log configuration
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"   Layer Configurations: {[lc.name for lc in config.layer_configs]}")
    logger.info(f"   Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"   Backbone: {len(config.conv_filters)} conv blocks {config.conv_filters}")
    logger.info(f"   Projection units: {config.projection_units}")
    logger.info("")

    try:
        # Run experiment
        results = run_experiment(config)
        logger.info("Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
