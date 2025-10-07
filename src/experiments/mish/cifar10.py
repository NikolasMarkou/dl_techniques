"""
CIFAR-10 Activation Function Comparison: Evaluating Neural Network Activations
==============================================================================

This experiment conducts a comprehensive comparison of different activation functions
for image classification on CIFAR-10, systematically evaluating their effectiveness
in convolutional neural networks.

The study addresses fundamental questions in deep learning: how do different
activation functions affect model performance, training dynamics, weight distributions,
and feature representations? By comparing traditional activations (ReLU, Tanh) with
modern alternatives (GELU, Mish, SaturatedMish), this experiment provides insights
into the trade-offs between computational efficiency, training stability, and
model expressiveness.

Experimental Design
-------------------

**Dataset**: CIFAR-10 (10 classes, 32×32 RGB images)
- 50,000 training images
- 10,000 test images
- Standard preprocessing with normalization and augmentation

**Model Architecture**: ResNet-inspired CNN with the following components:
- Initial convolutional layer (32 filters)
- 4 convolutional blocks with optional residual connections
- Progressive filter scaling: [32, 64, 128, 256]
- Batch normalization and dropout regularization
- Global average pooling
- Dense classification layer with L2 regularization
- Softmax output layer for probability predictions

**Activation Functions Evaluated**:
1. **ReLU**: The baseline rectified linear unit - simple and efficient
2. **Tanh**: Hyperbolic tangent - smooth, bounded activation
3. **GELU**: Gaussian Error Linear Unit - smooth approximation of ReLU
4. **Mish**: Self-regularized non-monotonic activation function
5. **SaturatedMish**: Mish variants with saturation
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
from typing import Dict, Any, List, Tuple, Callable, Optional

from dl_techniques.utils.logger import logger
from dl_techniques.layers.activations.mish import mish, saturated_mish

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

# ==============================================================================
# SETUP SATURATED MISH VARIANTS
# ==============================================================================

def saturated_mish_1(x):
    """Saturated Mish with alpha=1.0."""
    return saturated_mish(x, alpha=1.0)


def saturated_mish_2(x):
    """Saturated Mish with alpha=2.0."""
    return saturated_mish(x, alpha=2.0)


# ==============================================================================
# CUSTOM DATA STRUCTURES
# ==============================================================================

@dataclass
class ActivationComparisonData:
    """
    Data structure for activation function comparison visualization.

    Attributes:
        activation_names: Names of activation functions compared
        metrics: Dictionary mapping activation names to their metrics
        histories: Dictionary mapping activation names to training histories
    """
    activation_names: List[str]
    metrics: Dict[str, Dict[str, float]]
    histories: Dict[str, Dict[str, List[float]]]


# ==============================================================================
# CUSTOM VISUALIZATION PLUGINS
# ==============================================================================

class ActivationComparisonDashboard(VisualizationPlugin):
    """Comprehensive dashboard for activation function comparison."""

    @property
    def name(self) -> str:
        return "activation_comparison_dashboard"

    @property
    def description(self) -> str:
        return "Comprehensive dashboard comparing activation functions"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, ActivationComparisonData)

    def create_visualization(
        self,
        data: ActivationComparisonData,
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create comprehensive comparison dashboard."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(20, 12), dpi=self.config.dpi)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Training accuracy comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_training_curves(data, ax1, 'accuracy', 'Training Accuracy')

        # 2. Validation accuracy comparison
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_training_curves(data, ax2, 'val_accuracy', 'Validation Accuracy')

        # 3. Loss comparison
        ax3 = fig.add_subplot(gs[2, :2])
        self._plot_training_curves(data, ax3, 'loss', 'Training Loss')

        # 4. Final metrics comparison
        ax4 = fig.add_subplot(gs[0, 2])
        self._plot_final_metrics(data, ax4, 'accuracy', 'Final Accuracy')

        # 5. Top-5 accuracy comparison
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_final_metrics(data, ax5, 'top_5_accuracy', 'Top-5 Accuracy')

        # 6. Summary table
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_summary_table(data, ax6)

        plt.suptitle(
            'Activation Function Comparison Dashboard',
            fontsize=self.config.title_fontsize + 2,
            fontweight='bold'
        )

        return fig

    def _plot_training_curves(
        self,
        data: ActivationComparisonData,
        ax: Any,
        metric: str,
        title: str
    ) -> None:
        """Plot training curves for a specific metric."""
        colors = plt.cm.tab10(np.linspace(0, 1, len(data.activation_names)))

        for idx, name in enumerate(data.activation_names):
            if name in data.histories and metric in data.histories[name]:
                epochs = range(len(data.histories[name][metric]))
                ax.plot(
                    epochs,
                    data.histories[name][metric],
                    label=name,
                    color=colors[idx],
                    linewidth=2,
                    alpha=0.8
                )

        ax.set_xlabel('Epoch', fontsize=self.config.label_fontsize)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

    def _plot_final_metrics(
        self,
        data: ActivationComparisonData,
        ax: Any,
        metric: str,
        title: str
    ) -> None:
        """Plot bar chart of final metrics."""
        values = [
            data.metrics[name].get(metric, 0.0)
            for name in data.activation_names
        ]

        colors = plt.cm.tab10(np.linspace(0, 1, len(data.activation_names)))
        bars = ax.bar(range(len(data.activation_names)), values, color=colors, alpha=0.7)

        ax.set_xticks(range(len(data.activation_names)))
        ax.set_xticklabels(data.activation_names, rotation=45, ha='right')
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
                fontsize=9
            )

    def _plot_summary_table(
        self,
        data: ActivationComparisonData,
        ax: Any
    ) -> None:
        """Plot summary table of all metrics."""
        ax.axis('off')

        # Prepare table data
        table_data = []
        for name in data.activation_names:
            row = [
                name,
                f"{data.metrics[name].get('accuracy', 0.0):.4f}",
                f"{data.metrics[name].get('top_5_accuracy', 0.0):.4f}",
                f"{data.metrics[name].get('loss', 0.0):.4f}"
            ]
            table_data.append(row)

        table = ax.table(
            cellText=table_data,
            colLabels=['Activation', 'Accuracy', 'Top-5 Acc', 'Loss'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Performance Summary', fontsize=self.config.title_fontsize, pad=20)


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
class ExperimentConfig:
    """
    Configuration for the CIFAR-10 activation function comparison experiment.

    This class encapsulates all configurable parameters for the experiment,
    including dataset configuration, model architecture parameters, training
    settings, activation function definitions, and analysis configuration.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dense_units: List[int] = field(default_factory=lambda: [64])
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.4, 0.5])
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_batch_norm: bool = True
    use_residual: bool = True

    # --- Training Parameters ---
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 50
    monitor_metric: str = 'val_accuracy'

    # --- Activation Functions to Evaluate ---
    activations: Dict[str, Callable] = field(default_factory=lambda: {
        'Mish': lambda: mish,
        'Sat_Mish_1': lambda: saturated_mish_1,
        'Sat_Mish_2': lambda: saturated_mish_2,
        'ReLU': lambda: keras.activations.relu,
        'Tanh': lambda: keras.activations.tanh,
        'GELU': lambda: keras.activations.gelu,
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "cifar10_activation_comparison"
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

def build_residual_block(
    inputs: keras.layers.Layer,
    filters: int,
    activation_fn: Callable,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a residual block with skip connections.

    Args:
        inputs: Input tensor to the residual block
        filters: Number of filters in the convolutional layers
        activation_fn: Activation function to use
        config: Experiment configuration containing architecture parameters
        block_index: Index of the current block (for naming layers)

    Returns:
        Output tensor after applying the residual block
    """
    shortcut = inputs

    # First convolutional layer
    x = keras.layers.Conv2D(
        filters, config.kernel_size, padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_1'
    )(inputs)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation_fn)(x)

    # Second convolutional layer
    x = keras.layers.Conv2D(
        filters, config.kernel_size, padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_2'
    )(x)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)

    # Adjust skip connection if dimensions don't match
    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay)
        )(shortcut)

        if config.use_batch_norm:
            shortcut = keras.layers.BatchNormalization()(shortcut)

    # Add skip connection and apply final activation
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation(activation_fn)(x)

    return x


def build_conv_block(
    inputs: keras.layers.Layer,
    filters: int,
    activation_fn: Callable,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a convolutional block with optional residual connections.

    Args:
        inputs: Input tensor to the convolutional block
        filters: Number of filters in the convolutional layers
        activation_fn: Activation function to use
        config: Experiment configuration containing architecture parameters
        block_index: Index of the current block (for naming and logic)

    Returns:
        Output tensor after applying the convolutional block
    """
    # Use residual connections for blocks after the first one (if enabled)
    if config.use_residual and block_index > 0:
        x = build_residual_block(inputs, filters, activation_fn, config, block_index)
    else:
        # Standard convolutional block
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=config.kernel_size,
            padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'conv{block_index}'
        )(inputs)

        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation_fn)(x)

    # Apply max pooling (except for the last convolutional block)
    if block_index < len(config.conv_filters) - 1:
        x = keras.layers.MaxPooling2D(config.pool_size)(x)

    # Apply dropout if specified for this layer
    dropout_rate = (config.dropout_rates[block_index]
                   if block_index < len(config.dropout_rates) else 0.0)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def build_model(
    config: ExperimentConfig,
    activation_fn: Callable,
    name: str
) -> keras.Model:
    """
    Build a complete CNN model for CIFAR-10 classification with specified activation.

    Args:
        config: Experiment configuration containing model architecture parameters
        activation_fn: Activation function to use throughout the network
        name: Name prefix for the model and its layers

    Returns:
        Compiled Keras model ready for training with softmax probability outputs
    """
    # Define input layer
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Initial convolutional layer
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0],
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='stem'
    )(x)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation_fn)(x)

    # Stack of convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, activation_fn, config, i)

    # Global average pooling to reduce spatial dimensions
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Dense classification layers
    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(
            units=units,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'dense_{j}'
        )(x)

        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation_fn)(x)

        # Apply dropout if specified for dense layers
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates):
            dropout_rate = config.dropout_rates[dense_dropout_idx]
            if dropout_rate > 0:
                x = keras.layers.Dropout(dropout_rate)(x)

    # Pre-softmax logits layer
    logits = keras.layers.Dense(
        units=config.num_classes,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='logits'
    )(x)

    # Final softmax layer for probability output
    predictions = keras.layers.Activation('softmax', name='predictions')(logits)

    # Create and compile the model
    model = keras.Model(inputs=inputs, outputs=predictions, name=f'{name}_model')

    # Compile with comprehensive metrics
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
# TRAINING UTILITIES
# ==============================================================================

def train_single_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: ExperimentConfig,
    steps_per_epoch: int,
    val_steps: int,
    activation_name: str,
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
        activation_name: Name of activation function
        output_dir: Directory to save checkpoints

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
            filepath=str(output_dir / f'{activation_name}_best.keras'),
            monitor=config.monitor_metric,
            save_best_only=True,
            verbose=0
        )
    ]

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

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete CIFAR-10 activation function comparison experiment.

    This function orchestrates the entire experimental pipeline, including:
    1. Dataset loading and preprocessing using standardized builder
    2. Model training for each activation function
    3. Model analysis and evaluation
    4. Visualization generation using framework
    5. Results compilation and reporting

    Args:
        config: Experiment configuration specifying all parameters

    Returns:
        Dictionary containing all experimental results and analysis
    """
    # Set random seed for reproducibility
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
            secondary='#A23B72',
            accent='#F18F01'
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
    viz_manager.register_template("activation_dashboard", ActivationComparisonDashboard)

    # Log experiment start
    logger.info("🚀 Starting CIFAR-10 Activation Function Comparison Experiment")
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
    logger.info(f"Test data shape: {test_data.x_data.shape}")
    logger.info(f"Class names: {class_names}")

    # ===== MODEL TRAINING PHASE =====
    logger.info("🏋️ Starting model training phase...")

    trained_models = {}
    all_histories = {}

    for activation_name, activation_fn_factory in config.activations.items():
        logger.info(f"--- Training model with {activation_name} activation ---")

        # Build model for this activation function
        model = build_model(config, activation_fn_factory(), activation_name)

        # Log model architecture
        model.summary(print_fn=logger.info)
        logger.info(f"Model {activation_name} parameters: {model.count_params():,}")

        # Train the model
        history = train_single_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            config=config,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
            activation_name=activation_name,
            output_dir=experiment_dir / "checkpoints"
        )

        # Store results
        trained_models[activation_name] = model
        all_histories[activation_name] = history
        logger.info(f"✅ {activation_name} training completed!")

    # ===== MEMORY MANAGEMENT =====
    logger.info("🗑️ Triggering garbage collection...")
    gc.collect()

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("📈 Evaluating final model performance on test set...")

    performance_results = {}
    all_predictions = {}

    for name, model in trained_models.items():
        logger.info(f"Evaluating model {name}...")

        # Get predictions
        predictions = model.predict(test_data.x_data, verbose=0)
        y_pred_classes = np.argmax(predictions, axis=1)

        # Calculate metrics manually for consistency
        y_true = test_data.y_data.astype(int)
        accuracy = np.mean(y_pred_classes == y_true)

        # Calculate top-5 accuracy
        top_5_predictions = np.argsort(predictions, axis=1)[:, -5:]
        top5_acc = np.mean([
            y_true_val in top5_pred
            for y_true_val, top5_pred in zip(y_true, top_5_predictions)
        ])

        # Calculate loss
        loss = -np.mean(
            np.log(predictions[np.arange(len(y_true)), y_true] + 1e-7)
        )

        performance_results[name] = {
            'accuracy': accuracy,
            'top_5_accuracy': top5_acc,
            'loss': loss
        }

        all_predictions[name] = y_pred_classes

        logger.info(f"Model {name} - Accuracy: {accuracy:.4f}, Top-5: {top5_acc:.4f}, Loss: {loss:.4f}")

    # ===== VISUALIZATION GENERATION =====
    logger.info("🖼️ Generating visualizations using framework...")

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

    # 3. Confusion matrices for each model
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

    # 4. Comprehensive activation comparison dashboard
    activation_comparison = ActivationComparisonData(
        activation_names=list(config.activations.keys()),
        metrics=performance_results,
        histories=all_histories
    )

    viz_manager.visualize(
        data=activation_comparison,
        plugin_name="activation_dashboard",
        show=False
    )

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("📊 Performing comprehensive analysis with ModelAnalyzer...")
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
        'test_data': test_data,
        'class_names': class_names
    }

    # Print comprehensive summary
    print_experiment_summary(results_payload)

    return results_payload


# ==============================================================================
# RESULTS REPORTING
# ==============================================================================

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print a comprehensive summary of experimental results.

    Args:
        results: Dictionary containing all experimental results and analysis
    """
    logger.info("=" * 80)
    logger.info("📋 EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # ===== PERFORMANCE METRICS SECTION =====
    if 'performance_analysis' in results and results['performance_analysis']:
        logger.info("🎯 PERFORMANCE METRICS (on Test Set):")
        logger.info(f"{'Model':<20} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12}")
        logger.info("-" * 60)

        # Sort by accuracy for better readability
        sorted_results = sorted(
            results['performance_analysis'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        for model_name, metrics in sorted_results:
            accuracy = metrics.get('accuracy', 0.0)
            top5_acc = metrics.get('top_5_accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            logger.info(f"{model_name:<20} {accuracy:<12.4f} {top5_acc:<12.4f} {loss:<12.4f}")

    # ===== CALIBRATION METRICS SECTION =====
    model_analysis = results.get('model_analysis')
    if model_analysis and model_analysis.calibration_metrics:
        logger.info("\n🎯 CALIBRATION METRICS:")
        logger.info(f"{'Model':<20} {'ECE':<12} {'Brier Score':<15} {'Mean Entropy':<12}")
        logger.info("-" * 65)

        for model_name, cal_metrics in model_analysis.calibration_metrics.items():
            logger.info(
                f"{model_name:<20} {cal_metrics['ece']:<12.4f} "
                f"{cal_metrics['brier_score']:<15.4f} {cal_metrics['mean_entropy']:<12.4f}"
            )

    # ===== BEST MODEL IDENTIFICATION =====
    if 'performance_analysis' in results:
        best_model = max(
            results['performance_analysis'].items(),
            key=lambda x: x[1]['accuracy']
        )
        logger.info(f"\n🏆 BEST MODEL: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for running the CIFAR-10 activation comparison experiment.
    """
    logger.info("🚀 CIFAR-10 Activation Function Comparison Experiment")
    logger.info("=" * 80)

    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.info(e)

    # Initialize experiment configuration
    config = ExperimentConfig()

    # Log key configuration parameters
    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Activation Functions: {list(config.activations.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"   Model Architecture: {len(config.conv_filters)} conv blocks, "
                f"{len(config.dense_units)} dense layers")
    logger.info(f"   Residual Connections: {config.use_residual}")
    logger.info("")

    try:
        # Run the complete experiment
        results = run_experiment(config)
        logger.info("✅ Experiment completed successfully!")

    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()