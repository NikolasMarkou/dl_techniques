"""
OrthoBlock Effectiveness Study on CIFAR-10
==========================================

This experiment conducts a comprehensive evaluation of OrthoBlock layers compared to
traditional dense layers and other normalization techniques on CIFAR-10 image
classification. The study addresses fundamental questions about orthonormal regularization:
Does enforcing orthogonality constraints (W^T * W ≈ I) combined with a band rms variant normalization
and constrained scaling improve training dynamics, model calibration, and generalization
while maintaining competitive performance?

The hypothesis is that OrthoBlock's orthonormal constraints create more stable feature
representations that are less prone to overfitting and provide better calibration, at the
potential cost of some expressiveness.

Experimental Design
-------------------

**Dataset**: CIFAR-10 (32×32 RGB images, 10 classes)
- 50,000 training images, 10,000 test images
- Standard preprocessing with normalization
- Balanced class distribution for fair calibration analysis

**Model Architecture**: ResNet-inspired CNN with different dense layer types
- 4 convolutional blocks with progressive filter scaling [32, 64, 128, 256]
- Residual connections for deep network training
- Global average pooling followed by configurable dense layer
- Identical architecture across all variants (only dense layer changes)

**OrthoBlock Variants Evaluated**:

1. **OrthoBlock_Strong**: High orthogonal regularization (λ=1.0)
2. **OrthoBlock_Medium**: Medium orthogonal regularization (λ=0.1)
3. **OrthoBlock_Weak**: Low orthogonal regularization (λ=0.01)
4. **OrthoBlock_Adaptive**: Scheduled orthogonal regularization
5. **Dense_Standard**: Standard dense layer (baseline)
6. **Dense_L2**: Dense layer with L2 regularization
7. **Dense_Dropout**: Dense layer with higher dropout
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
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.layers.orthoblock import OrthoBlock
from dl_techniques.utils.datasets import load_and_preprocess_cifar10
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

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
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 20
    monitor_metric: str = 'val_accuracy'

    # --- OrthoBlock Specific Parameters ---
    ortho_reg_factors: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0])
    scale_initial_values: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    ortho_scheduler_config: Dict[str, float] = field(default_factory=lambda: {
        'max_factor': 0.1,
        'min_factor': 0.001
    })

    # --- Dense Layer Variants ---
    dense_variants: Dict[str, Callable] = field(default_factory=lambda: {
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
        analyze_training_dynamics=True,
        calibration_bins=15,
        save_plots=True,
        plot_style='publication',
        show_statistical_tests=True,
        show_confidence_intervals=True,
        verbose=True
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
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
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

def calculate_run_statistics(results_per_run: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
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

        # Initialize statistics for this model
        statistics[model_name] = {}

        # Get all metrics from first run
        metrics = run_results[0].keys()

        for metric in metrics:
            values = [result[metric] for result in run_results if metric in result]

            if values:
                statistics[model_name][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

    return statistics

# ==============================================================================
# CUSTOM TRAINING WITH ORTHOGONALITY TRACKING
# ==============================================================================

def train_model_with_tracker(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
    ortho_tracker: OrthogonalityTracker
) -> keras.callbacks.History:
    """
    Train model with orthogonality tracking using manual callbacks.

    Args:
        model: Keras model to train
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        config: Training configuration
        ortho_tracker: Orthogonality tracker callback

    Returns:
        Training history
    """
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=config.output_dir / f"{config.model_name}_best.keras",
            monitor=config.monitor_metric,
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=config.monitor_metric,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ortho_tracker  # Add the orthogonality tracker
    ]

    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(config.output_dir / f"{config.model_name}_final.keras")

    return history

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
    # Set random seed
    keras.utils.set_random_seed(config.random_seed)

    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting OrthoBlock Effectiveness Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("📊 Loading CIFAR-10 dataset...")
    cifar10_data = load_and_preprocess_cifar10()
    logger.info("✅ Dataset loaded successfully")

    # ===== MULTIPLE RUNS FOR STATISTICAL SIGNIFICANCE =====
    logger.info(f"🔄 Running {config.n_runs} repetitions for statistical significance...")

    all_trained_models = {}
    all_histories = {}
    all_orthogonality_trackers = {}
    results_per_run = {variant_name: [] for variant_name in config.dense_variants.keys()}

    for run_idx in range(config.n_runs):
        logger.info(f"🏃 Starting run {run_idx + 1}/{config.n_runs}")
        run_seed = config.random_seed + run_idx * 1000
        keras.utils.set_random_seed(run_seed)

        current_run_models = {}
        current_run_histories = {}
        current_run_trackers = {}

        for variant_name, variant_factory in config.dense_variants.items():
            logger.info(f"--- Training {variant_name} (Run {run_idx + 1}) ---")
            layer_type, layer_params = variant_factory(config)
            model = build_model(config, layer_type, layer_params, f"{variant_name}_run{run_idx}")

            if run_idx == 0: model.summary(print_fn=logger.info)

            ortho_tracker = None
            if layer_type == 'orthoblock':
                ortho_tracker = OrthogonalityTracker(f"{variant_name}_run{run_idx}")

            training_config = TrainingConfig(
                epochs=config.epochs, batch_size=config.batch_size,
                early_stopping_patience=config.early_stopping_patience,
                monitor_metric=config.monitor_metric,
                model_name=f"{variant_name}_run{run_idx}",
                output_dir=experiment_dir / "training_plots" / f"run_{run_idx}" / variant_name
            )

            if ortho_tracker:
                history = train_model_with_tracker(
                    model, cifar10_data.x_train, cifar10_data.y_train,
                    cifar10_data.x_test, cifar10_data.y_test, training_config, ortho_tracker)
            else:
                history = train_model(
                    model, cifar10_data.x_train, cifar10_data.y_train,
                    cifar10_data.x_test, cifar10_data.y_test, training_config)

            try:
                logger.info(f"📊 Evaluating {variant_name} (Run {run_idx + 1})...")

                # ==============================================================================
                # BIG COMMENT: THIS IS THE DEFINITIVE FIX.
                # We are completely bypassing the unreliable `model.evaluate()` for accuracy.
                # Instead, we get raw predictions and compute accuracy manually. This is
                # foolproof and removes any dependency on Keras's inconsistent metric naming.
                # ==============================================================================

                # 1. Get model predictions
                predictions = model.predict(cifar10_data.x_test, verbose=0)

                # 2. Get predicted classes
                y_pred_classes = np.argmax(predictions, axis=1)

                # 3. Get true classes (handle both one-hot and integer labels)
                if cifar10_data.y_test.ndim > 1 and cifar10_data.y_test.shape[1] > 1:
                    y_true_classes = np.argmax(cifar10_data.y_test, axis=1)
                else:
                    y_true_classes = cifar10_data.y_test

                # 4. Calculate accuracy manually
                manual_accuracy = np.mean(y_pred_classes == y_true_classes)

                # We can still use evaluate to reliably get the loss value.
                eval_results = model.evaluate(cifar10_data.x_test, cifar10_data.y_test, verbose=0)
                metrics_dict = dict(zip(model.metrics_names, eval_results))
                loss_value = metrics_dict.get('loss', 0.0)

                # Store the manually calculated, correct accuracy.
                results_per_run[variant_name].append({
                    'accuracy': manual_accuracy,
                    'loss': loss_value
                    # Add other manual metrics here if needed
                })

                logger.info(f"✅ {variant_name} (Run {run_idx + 1}): Accuracy={manual_accuracy:.4f}")

            except Exception as e:
                logger.error(f"❌ Error evaluating {variant_name} (Run {run_idx + 1}): {e}", exc_info=True)

            current_run_models[variant_name] = model
            current_run_histories[variant_name] = history.history
            if ortho_tracker:
                current_run_trackers[variant_name] = ortho_tracker

        if run_idx == config.n_runs - 1:
            all_trained_models = current_run_models
            all_histories = current_run_histories
            all_orthogonality_trackers = current_run_trackers

        del current_run_models, current_run_histories, current_run_trackers
        gc.collect()

    logger.info("📈 Calculating statistics across runs...")
    run_statistics = calculate_run_statistics(results_per_run)

    logger.info("🔬 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None
    try:
        analyzer = ModelAnalyzer(
            models=all_trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis",
            training_history=all_histories
        )
        model_analysis_results = analyzer.analyze(data=DataInput.from_object(cifar10_data))
        logger.info("✅ Model analysis completed successfully!")
    except Exception as e:
        logger.error(f"❌ Model analysis failed: {e}", exc_info=True)

    logger.info("📊 Generating custom and multi-run visualizations...")
    create_statistical_comparison_plot(run_statistics, experiment_dir / "visualizations")
    create_orthogonality_analysis_plots(all_orthogonality_trackers, experiment_dir / "visualizations")

    try:
        vis_manager = VisualizationManager(output_dir=experiment_dir / "visualizations", config=VisualizationConfig(),
                                           timestamp_dirs=False)
        raw_predictions = {name: model.predict(cifar10_data.x_test, verbose=0) for name, model in
                           all_trained_models.items()}
        class_predictions = {name: np.argmax(preds, axis=1) for name, preds in raw_predictions.items()}
        y_true_classes = (
            np.argmax(cifar10_data.y_test, axis=1) if len(cifar10_data.y_test.shape) > 1 else cifar10_data.y_test)
        vis_manager.plot_confusion_matrices_comparison(
            y_true=y_true_classes, model_predictions=class_predictions, name='orthoblock_confusion_matrices',
            subdir='model_comparison', normalize=True, class_names=[str(i) for i in range(10)])
    except Exception as e:
        logger.warning(f"Could not generate custom confusion matrix plot: {e}")

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
# VISUALIZATION UTILITIES
# ==============================================================================

def create_statistical_comparison_plot(statistics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    """
    Create statistical comparison plots showing mean ± std across runs.

    Args:
        statistics: Statistical results from multiple runs
        output_dir: Directory to save plots
    """
    try:


        # ==============================================================================
        # FIXED: The function was receiving a path to a directory that might not exist
        # and was not creating it, leading to a FileNotFoundError.
        # This line ensures the directory exists before trying to save the plot.
        # ==============================================================================
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup plot style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Extract data
        models = list(statistics.keys())
        accuracies = [statistics[model]['accuracy']['mean'] for model in models]
        accuracy_stds = [statistics[model]['accuracy']['std'] for model in models]

        # Color coding for different model types
        colors = []
        for model in models:
            if 'OrthoBlock' in model:
                colors.append('orange')
            else:
                colors.append('skyblue')

        # Plot accuracy comparison
        ax1 = axes[0]
        bars = ax1.bar(models, accuracies, yerr=accuracy_stds, capsize=5, color=colors)
        ax1.set_title('Test Accuracy Comparison (Mean ± Std)')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, acc, std in zip(bars, accuracies, accuracy_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                    f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot loss comparison
        ax2 = axes[1]
        losses = [statistics[model]['loss']['mean'] for model in models]
        loss_stds = [statistics[model]['loss']['std'] for model in models]
        bars2 = ax2.bar(models, losses, yerr=loss_stds, capsize=5, color=colors)
        ax2.set_title('Test Loss Comparison (Mean ± Std)')
        ax2.set_ylabel('Loss')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, loss, std in zip(bars2, losses, loss_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{loss:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✅ Statistical comparison plot saved")

    except Exception as e:
        logger.error(f"❌ Failed to create statistical comparison plot: {e}")

def create_orthogonality_analysis_plots(trackers: Dict[str, OrthogonalityTracker], output_dir: Path) -> None:
    """
    Create orthogonality analysis plots for OrthoBlock models.

    Args:
        trackers: Dictionary of orthogonality trackers
        output_dir: Directory to save plots
    """
    try:
        # ==============================================================================
        # FIXED: Similar to the other plotting function, this one also needs to ensure
        # its output directory exists before proceeding.
        # ==============================================================================
        output_dir.mkdir(parents=True, exist_ok=True)

        if not trackers:
            logger.info("No orthogonality trackers found, skipping orthogonality plots")
            return

        # Orthogonality evolution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Orthogonality measure over training
        ax1 = axes[0, 0]
        for model_name, tracker in trackers.items():
            if tracker.orthogonality_history:
                epochs = range(len(tracker.orthogonality_history))
                ortho_values = [metrics.get('layer_0', 0) for metrics in tracker.orthogonality_history]
                ax1.plot(epochs, ortho_values, label=model_name, marker='o', markersize=2)

        ax1.set_title('Orthogonality Measure Over Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('||W^T W - I||_F')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Scale parameter evolution (mean)
        ax2 = axes[0, 1]
        for model_name, tracker in trackers.items():
            if tracker.scale_history:
                epochs = range(len(tracker.scale_history))
                scale_means = [metrics.get('layer_0', {}).get('mean', 0) for metrics in tracker.scale_history]
                if any(scale_means):
                    ax2.plot(epochs, scale_means, label=model_name, marker='o', markersize=2)

        ax2.set_title('Scale Parameter Mean Over Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Scale Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Scale parameter standard deviation
        ax3 = axes[1, 0]
        for model_name, tracker in trackers.items():
            if tracker.scale_history:
                epochs = range(len(tracker.scale_history))
                scale_stds = [metrics.get('layer_0', {}).get('std', 0) for metrics in tracker.scale_history]
                if any(scale_stds):
                    ax3.plot(epochs, scale_stds, label=model_name, marker='o', markersize=2)

        ax3.set_title('Scale Parameter Std Over Training')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Scale Std')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Final scale distributions
        ax4 = axes[1, 1]

        # ==============================================================================
        # FIXED: The variable 'stats' was defined inside a conditional block in the loop,
        # causing an UnboundLocalError if the condition was false for the last item.
        # It is now defined *before* the loop to guarantee it always exists.
        # ==============================================================================
        stats = ['mean', 'min', 'max']

        for i, (model_name, tracker) in enumerate(trackers.items()):
            if tracker.scale_history and tracker.scale_history[-1]:
                final_scales = tracker.scale_history[-1].get('layer_0', {})
                if final_scales:
                    values = [final_scales.get(stat, 0) for stat in stats]
                    x_pos = np.arange(len(stats)) + i * 0.2
                    ax4.bar(x_pos, values, width=0.15, label=model_name, alpha=0.7)

        ax4.set_title('Final Scale Parameter Statistics')
        ax4.set_xlabel('Statistic')
        ax4.set_ylabel('Value')
        ax4.set_xticks(np.arange(len(stats)))
        ax4.set_xticklabels(stats)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'orthogonality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✅ Orthogonality analysis plots saved")

    except Exception as e:
        logger.error(f"❌ Failed to create orthogonality analysis plots: {e}")

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
        def convert_numpy_to_python(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_python(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_to_python(item) for item in obj)
            else:
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

        # Convert and save statistical results
        statistical_results_converted = convert_numpy_to_python(results['run_statistics'])
        with open(experiment_dir / "statistical_results.json", 'w') as f:
            json.dump(statistical_results_converted, f, indent=2)

        # Save orthogonality data
        if results['orthogonality_trackers']:
            ortho_data = {}
            for model_name, tracker in results['orthogonality_trackers'].items():
                ortho_data[model_name] = {
                    'orthogonality_history': convert_numpy_to_python(tracker.orthogonality_history),
                    'scale_history': convert_numpy_to_python(tracker.scale_history)
                }

            with open(experiment_dir / "orthogonality_data.json", 'w') as f:
                json.dump(ortho_data, f, indent=2)

        # Save models
        models_dir = experiment_dir / "models"
        models_dir.mkdir(exist_ok=True)

        for name, model in results['trained_models'].items():
            model_path = models_dir / f"{name}.keras"
            model.save(model_path)

        logger.info("✅ Experiment results saved successfully")

    except Exception as e:
        logger.error(f"❌ Failed to save experiment results: {e}", exc_info=True)


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print comprehensive experiment summary.

    Args:
        results: Experiment results dictionary
    """
    logger.info("=" * 80)
    logger.info("📋 ORTHOBLOCK EFFECTIVENESS EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # ===== STATISTICAL RESULTS =====
    if 'run_statistics' in results and results['run_statistics']:
        logger.info("📊 STATISTICAL RESULTS (Mean ± Std across runs):")
        logger.info(f"{'Model':<22} {'Accuracy':<18} {'Loss':<18} {'Runs':<8}")
        logger.info("-" * 70)

        for model_name, stats in results['run_statistics'].items():
            acc_mean = stats['accuracy']['mean']
            acc_std = stats['accuracy']['std']
            loss_mean = stats['loss']['mean']
            loss_std = stats['loss']['std']
            n_runs = stats['accuracy']['count']

            logger.info(f"{model_name:<22} {acc_mean:.4f} ± {acc_std:.4f}    "
                        f"{loss_mean:.4f} ± {loss_std:.4f}    {n_runs:<8}")

    # ===== CALIBRATION & CONFIDENCE RESULTS =====
    analysis_res = results.get('model_analysis')
    if analysis_res and analysis_res.calibration_metrics:
        logger.info("\n🎯 CALIBRATION & CONFIDENCE ANALYSIS (from last run):")
        logger.info(f"{'Model':<22} {'ECE':<12} {'Brier':<12} {'Mean Entropy':<15}")
        logger.info("-" * 65)

        for model_name, cal_metrics in analysis_res.calibration_metrics.items():
            ece = cal_metrics.get('ece', 0.0)
            brier = cal_metrics.get('brier_score', 0.0)

            # --- MODIFIED: Get entropy from the dedicated confidence_metrics dictionary ---
            entropy = 0.0
            if analysis_res.confidence_metrics and model_name in analysis_res.confidence_metrics:
                confidence_data = analysis_res.confidence_metrics[model_name]
                entropy = confidence_data.get('mean_entropy', 0.0)

            logger.info(f"{model_name:<22} {ece:<12.4f} {brier:<12.4f} {entropy:<15.4f}")

    # ===== KEY INSIGHTS =====
    logger.info("\n🔍 KEY INSIGHTS:")

    if 'run_statistics' in results and results['run_statistics']:
        best_model = max(results['run_statistics'].items(),
                         key=lambda x: x[1]['accuracy']['mean'])
        logger.info(f"   🏆 Best Accuracy: {best_model[0]} ({best_model[1]['accuracy']['mean']:.4f})")

        most_stable = min(results['run_statistics'].items(),
                          key=lambda x: x[1]['accuracy']['std'])
        logger.info(f"   🎯 Most Stable: {most_stable[0]} (Accuracy Std: {most_stable[1]['accuracy']['std']:.4f})")

        ortho_models = {k: v for k, v in results['run_statistics'].items() if k.startswith('OrthoBlock')}
        dense_models = {k: v for k, v in results['run_statistics'].items() if k.startswith('Dense')}

        if ortho_models:
            logger.info("   🔗 OrthoBlock Analysis:")
            for model_name, stats in ortho_models.items():
                logger.info(f"      {model_name:<20}: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")

        if dense_models:
            logger.info("   📊 Dense Layer Analysis:")
            for model_name, stats in dense_models.items():
                logger.info(f"      {model_name:<20}: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")

        if results.get('orthogonality_trackers'):
            logger.info("   📐 Orthogonality Insights (Initial → Final ||W^T W - I||_F):")
            for model_name, tracker in results['orthogonality_trackers'].items():
                if tracker.orthogonality_history:
                    final_ortho = tracker.orthogonality_history[-1].get('layer_0', 0)
                    initial_ortho = tracker.orthogonality_history[0].get('layer_0', 0)
                    logger.info(f"      {model_name:<20}: {initial_ortho:.3f} → {final_ortho:.3f}")

    logger.info("=" * 80)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for the OrthoBlock effectiveness experiment.
    """
    logger.info("🚀 OrthoBlock Effectiveness Experiment")
    logger.info("=" * 80)

    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.info(f"GPU configuration: {e}")

    # Initialize configuration
    config = OrthoBlockExperimentConfig()

    # Log configuration
    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Dense layer variants: {list(config.dense_variants.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"   Number of runs: {config.n_runs}")
    logger.info(f"   Architecture: {len(config.conv_filters)} conv blocks, {config.dense_units} dense units")
    logger.info(f"   Orthogonal regularization factors: {config.ortho_reg_factors}")
    logger.info("")

    try:
        # Run experiment
        _ = run_experiment(config)
        logger.info("✅ OrthoBlock effectiveness experiment completed successfully!")

    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}", exc_info=True)
        raise

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()