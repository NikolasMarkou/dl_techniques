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
- **Dataset**: CIFAR-10 (32×32 RGB images, 10 classes), preprocessed with
  standard normalization.

- **Model Architecture**: A consistent ResNet-inspired CNN is used across all
  experimental variants to ensure a fair comparison. The architecture consists
  of four convolutional blocks with residual connections, followed by global
  pooling and a final dense classification head.

- **Experimental Variants**: The core of the experiment involves swapping out the
  final dense layer with different implementations:
    1.  **OrthoBlock Variants**: Models using OrthoBlock with varying strengths of
        orthogonal regularization and scaling initializations.
    2.  **Baseline Models**: Models using standard Keras Dense layers, including
        an unregularized baseline and a version with L2 weight decay.

- **Multi-Run Analysis**: To ensure statistical significance, each model variant
  is trained multiple times (controlled by `n_runs`). The final performance is
  reported as the mean and standard deviation across these runs.

Workflow:
---------
1.  **Configuration**: All experimental parameters are defined in the
    `OrthoBlockExperimentConfig` dataclass.
2.  **Data Loading**: The CIFAR-10 dataset is loaded and preprocessed.
3.  **Iterative Training**: The script iterates through each defined model
    variant and for each `n_run`:
    a. Builds a new instance of the CNN model.
    b. For OrthoBlock models, an `OrthogonalityTracker` callback is attached to
       monitor the ||W^T * W - I|| Frobenius norm during training.
    c. The model is trained with early stopping.
4.  **Analysis & Visualization**:
    a. Final performance metrics (accuracy, loss) are aggregated across all runs,
       and statistical summaries (mean, std) are computed.
    b. The models from the final run are passed to the `ModelAnalyzer` for a
       deep-dive into calibration (ECE, Brier score), weights, and activations.
    c. The new `VisualizationManager` is used to generate comparative confusion
       matrices.
    d. Custom plots are generated to visualize the statistical performance
       comparison and the orthogonality dynamics during training.
5.  **Reporting**: All results, plots, model weights, and configuration files are
    saved to a unique timestamped directory. A comprehensive summary is printed
    to the console.

Usage:
------
To run the full experiment with the default configuration, execute the script from
the command line:
    $ python cifar10.py
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
from dl_techniques.visualization import (
    VisualizationManager,
    ClassificationResults,
    MultiModelClassification,
    ConfusionMatrixVisualization
)

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
        layer_type: Type of dense layer ('orthoblock', 'dense').
        layer_params: Parameters specific to the layer type.
        units: Number of units in the layer.
        activation: Activation function.
        name: Optional layer name.

    Returns:
        Configured dense layer.
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
        inputs: Input tensor.
        filters: Number of filters.
        config: Experiment configuration.
        block_index: Index of the current block.

    Returns:
        Output tensor after convolutional block.
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
        config: Experiment configuration.
        layer_type: Type of dense layer.
        layer_params: Parameters for dense layer.
        name: Model name.

    Returns:
        Compiled Keras model.
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
        results_per_run: Dictionary mapping model names to lists of results across runs.

    Returns:
        Dictionary with mean, std, min, max for each model and metric.
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
        model: Keras model to train.
        x_train: Training data.
        y_train: Training labels.
        x_val: Validation data.
        y_val: Validation labels.
        config: Training configuration.
        ortho_tracker: Orthogonality tracker callback.

    Returns:
        Training history.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

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
        ortho_tracker
    ]

    history = model.fit(
        x_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    model.save(config.output_dir / f"{config.model_name}_final.keras")
    return history

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: OrthoBlockExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete OrthoBlock effectiveness experiment.

    Args:
        config: Experiment configuration.

    Returns:
        Dictionary containing all experimental results.
    """
    keras.utils.set_random_seed(config.random_seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    viz_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        experiment_name=config.experiment_name
    )
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)

    logger.info("Starting OrthoBlock Effectiveness Experiment")
    logger.info(f"Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    logger.info("Loading CIFAR-10 dataset...")
    cifar10_data = load_and_preprocess_cifar10()
    logger.info("Dataset loaded successfully.")

    logger.info(f"Running {config.n_runs} repetitions for statistical significance...")
    all_trained_models = {}
    all_histories = {}
    all_orthogonality_trackers = {}
    results_per_run = {variant_name: [] for variant_name in config.dense_variants.keys()}

    for run_idx in range(config.n_runs):
        logger.info(f"Starting run {run_idx + 1}/{config.n_runs}")
        run_seed = config.random_seed + run_idx * 1000
        keras.utils.set_random_seed(run_seed)

        current_run_models, current_run_histories, current_run_trackers = {}, {}, {}

        for variant_name, variant_factory in config.dense_variants.items():
            logger.info(f"--- Training {variant_name} (Run {run_idx + 1}) ---")
            layer_type, layer_params = variant_factory(config)
            model = build_model(config, layer_type, layer_params, f"{variant_name}_run{run_idx}")

            if run_idx == 0:
                model.summary(print_fn=logger.info)

            ortho_tracker = OrthogonalityTracker(f"{variant_name}_run{run_idx}") if layer_type == 'orthoblock' else None

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
                    cifar10_data.x_test, cifar10_data.y_test, training_config, ortho_tracker
                )
            else:
                history = train_model(
                    model, cifar10_data.x_train, cifar10_data.y_train,
                    cifar10_data.x_test, cifar10_data.y_test, training_config
                )

            try:
                logger.info(f"Evaluating {variant_name} (Run {run_idx + 1})...")
                predictions = model.predict(cifar10_data.x_test, verbose=0)
                y_pred_classes = np.argmax(predictions, axis=1)

                if cifar10_data.y_test.ndim > 1 and cifar10_data.y_test.shape[1] > 1:
                    y_true_classes = np.argmax(cifar10_data.y_test, axis=1)
                else:
                    y_true_classes = cifar10_data.y_test

                manual_accuracy = np.mean(y_pred_classes == y_true_classes)
                eval_results = model.evaluate(cifar10_data.x_test, cifar10_data.y_test, verbose=0)
                metrics_dict = dict(zip(model.metrics_names, eval_results))
                loss_value = metrics_dict.get('loss', 0.0)

                results_per_run[variant_name].append({'accuracy': manual_accuracy, 'loss': loss_value})
                logger.info(f"{variant_name} (Run {run_idx + 1}): Accuracy={manual_accuracy:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {variant_name} (Run {run_idx + 1}): {e}", exc_info=True)

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

    logger.info("Calculating statistics across runs...")
    run_statistics = calculate_run_statistics(results_per_run)

    logger.info("Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None
    try:
        analyzer = ModelAnalyzer(
            models=all_trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis",
            training_history=all_histories
        )
        model_analysis_results = analyzer.analyze(data=DataInput.from_object(cifar10_data))
        logger.info("Model analysis completed successfully.")
    except Exception as e:
        logger.error(f"Model analysis failed: {e}", exc_info=True)

    logger.info("Generating custom and multi-run visualizations...")
    create_statistical_comparison_plot(run_statistics, experiment_dir / "visualizations")
    create_orthogonality_analysis_plots(all_orthogonality_trackers, experiment_dir / "visualizations")

    try:
        raw_predictions = {name: model.predict(cifar10_data.x_test, verbose=0)
                           for name, model in all_trained_models.items()}
        class_predictions = {name: np.argmax(preds, axis=1)
                             for name, preds in raw_predictions.items()}
        y_true_labels = (np.argmax(cifar10_data.y_test, axis=1)
                         if cifar10_data.y_test.ndim > 1 else cifar10_data.y_test)
        class_names = [str(i) for i in range(config.num_classes)]

        model_results = {
            name: ClassificationResults(
                y_true=y_true_labels, y_pred=preds, y_prob=raw_predictions[name],
                class_names=class_names, model_name=name
            ) for name, preds in class_predictions.items()
        }
        multi_model_data = MultiModelClassification(
            y_true=y_true_labels, model_results=model_results, class_names=class_names
        )
        viz_manager.visualize(
            data=multi_model_data,
            plugin_name="confusion_matrix",
            normalize='true',
            title='Dense Layer Variant Confusion Matrix Comparison'
        )
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix plot via visualization framework: {e}")

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
        statistics: Statistical results from multiple runs.
        output_dir: Directory to save plots.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        models = list(statistics.keys())
        accuracies = [statistics[model]['accuracy']['mean'] for model in models]
        accuracy_stds = [statistics[model]['accuracy']['std'] for model in models]
        colors = ['orange' if 'OrthoBlock' in model else 'skyblue' for model in models]

        ax1 = axes[0]
        bars = ax1.bar(models, accuracies, yerr=accuracy_stds, capsize=5, color=colors)
        ax1.set_title('Test Accuracy Comparison (Mean ± Std)')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)

        for bar, acc, std in zip(bars, accuracies, accuracy_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + std + 0.005,
                     f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)

        ax2 = axes[1]
        losses = [statistics[model]['loss']['mean'] for model in models]
        loss_stds = [statistics[model]['loss']['std'] for model in models]
        bars2 = ax2.bar(models, losses, yerr=loss_stds, capsize=5, color=colors)
        ax2.set_title('Test Loss Comparison (Mean ± Std)')
        ax2.set_ylabel('Loss')
        ax2.tick_params(axis='x', rotation=45)

        for bar, loss, std in zip(bars2, losses, loss_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + std + 0.01,
                     f'{loss:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Statistical comparison plot saved.")
    except Exception as e:
        logger.error(f"Failed to create statistical comparison plot: {e}")

def create_orthogonality_analysis_plots(trackers: Dict[str, OrthogonalityTracker], output_dir: Path) -> None:
    """
    Create orthogonality analysis plots for OrthoBlock models.

    Args:
        trackers: Dictionary of orthogonality trackers.
        output_dir: Directory to save plots.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not trackers:
            logger.info("No orthogonality trackers found, skipping plots.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()

        for name, tracker in trackers.items():
            if tracker.orthogonality_history:
                epochs = range(len(tracker.orthogonality_history))
                ortho = [m.get('layer_0', 0) for m in tracker.orthogonality_history]
                ax1.plot(epochs, ortho, label=name, marker='o', markersize=2)
            if tracker.scale_history:
                epochs = range(len(tracker.scale_history))
                means = [m.get('layer_0', {}).get('mean', 0) for m in tracker.scale_history]
                stds = [m.get('layer_0', {}).get('std', 0) for m in tracker.scale_history]
                if any(means):
                    ax2.plot(epochs, means, label=name, marker='o', markersize=2)
                if any(stds):
                    ax3.plot(epochs, stds, label=name, marker='o', markersize=2)

        ax1.set(title='Orthogonality Measure Over Training', xlabel='Epoch', ylabel='||W^T W - I||_F')
        ax2.set(title='Scale Parameter Mean Over Training', xlabel='Epoch', ylabel='Mean Scale Value')
        ax3.set(title='Scale Parameter Std Over Training', xlabel='Epoch', ylabel='Scale Std')
        for ax in [ax1, ax2, ax3]:
            ax.legend()
            ax.grid(True, alpha=0.3)

        stats_keys = ['mean', 'min', 'max']
        for i, (name, tracker) in enumerate(trackers.items()):
            if tracker.scale_history and tracker.scale_history[-1]:
                final_scales = tracker.scale_history[-1].get('layer_0', {})
                if final_scales:
                    values = [final_scales.get(k, 0) for k in stats_keys]
                    x_pos = np.arange(len(stats_keys)) + i * 0.2
                    ax4.bar(x_pos, values, width=0.15, label=name, alpha=0.7)

        ax4.set(title='Final Scale Parameter Statistics', xlabel='Statistic', ylabel='Value')
        ax4.set_xticks(np.arange(len(stats_keys)) + 0.2)
        ax4.set_xticklabels(stats_keys)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'orthogonality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Orthogonality analysis plots saved.")
    except Exception as e:
        logger.error(f"Failed to create orthogonality analysis plots: {e}")

# ==============================================================================
# RESULTS SAVING AND REPORTING
# ==============================================================================

def save_experiment_results(results: Dict[str, Any], experiment_dir: Path) -> None:
    """
    Save experiment results in multiple formats.

    Args:
        results: Experiment results dictionary.
        experiment_dir: Directory to save results.
    """
    try:
        def convert_numpy(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [convert_numpy(i) for i in obj]
            return obj

        config_dict = {
            'experiment_name': results['config'].experiment_name,
            'dense_variants': list(results['config'].dense_variants.keys()),
            'epochs': results['config'].epochs, 'batch_size': results['config'].batch_size,
            'learning_rate': results['config'].learning_rate, 'n_runs': results['config'].n_runs,
            'random_seed': results['config'].random_seed,
            'architecture': {
                'conv_filters': results['config'].conv_filters,
                'dense_units': results['config'].dense_units,
                'use_residual': results['config'].use_residual
            }
        }
        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        stats_converted = convert_numpy(results['run_statistics'])
        with open(experiment_dir / "statistical_results.json", 'w') as f:
            json.dump(stats_converted, f, indent=2)

        if results['orthogonality_trackers']:
            ortho_data = {name: {'orthogonality_history': convert_numpy(t.orthogonality_history),
                                 'scale_history': convert_numpy(t.scale_history)}
                          for name, t in results['orthogonality_trackers'].items()}
            with open(experiment_dir / "orthogonality_data.json", 'w') as f:
                json.dump(ortho_data, f, indent=2)

        models_dir = experiment_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for name, model in results['trained_models'].items():
            model.save(models_dir / f"{name}.keras")

        logger.info("Experiment results saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save experiment results: {e}", exc_info=True)

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print comprehensive experiment summary.

    Args:
        results: Experiment results dictionary.
    """
    logger.info("=" * 80)
    logger.info("ORTHOBLOCK EFFECTIVENESS EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    if 'run_statistics' in results and results['run_statistics']:
        logger.info("STATISTICAL RESULTS (Mean ± Std across runs):")
        logger.info(f"{'Model':<22} {'Accuracy':<18} {'Loss':<18} {'Runs':<8}")
        logger.info("-" * 70)
        for name, stats in results['run_statistics'].items():
            acc_m, acc_s = stats['accuracy']['mean'], stats['accuracy']['std']
            loss_m, loss_s = stats['loss']['mean'], stats['loss']['std']
            n_runs = stats['accuracy']['count']
            logger.info(f"{name:<22} {acc_m:.4f} ± {acc_s:.4f}    "
                        f"{loss_m:.4f} ± {loss_s:.4f}    {n_runs:<8}")

    analysis = results.get('model_analysis')
    if analysis and analysis.calibration_metrics:
        logger.info("\nCALIBRATION & CONFIDENCE ANALYSIS (from last run):")
        logger.info(f"{'Model':<22} {'ECE':<12} {'Brier':<12} {'Mean Entropy':<15}")
        logger.info("-" * 65)
        for name, cal_metrics in analysis.calibration_metrics.items():
            entropy = analysis.confidence_metrics.get(name, {}).get('mean_entropy', 0.0)
            logger.info(f"{name:<22} {cal_metrics.get('ece', 0.0):<12.4f} "
                        f"{cal_metrics.get('brier_score', 0.0):<12.4f} {entropy:<15.4f}")

    logger.info("\nKEY INSIGHTS:")
    if 'run_statistics' in results and results['run_statistics']:
        stats = results['run_statistics']
        best_acc_model = max(stats, key=lambda m: stats[m]['accuracy']['mean'])
        logger.info(f"   Best Accuracy: {best_acc_model} ({stats[best_acc_model]['accuracy']['mean']:.4f})")
        most_stable_model = min(stats, key=lambda m: stats[m]['accuracy']['std'])
        logger.info(f"   Most Stable: {most_stable_model} (Acc Std: {stats[most_stable_model]['accuracy']['std']:.4f})")

        if results.get('orthogonality_trackers'):
            logger.info("   Orthogonality Insights (Initial → Final ||W^T W - I||_F):")
            for name, tracker in results['orthogonality_trackers'].items():
                if tracker.orthogonality_history:
                    final = tracker.orthogonality_history[-1].get('layer_0', 0)
                    initial = tracker.orthogonality_history[0].get('layer_0', 0)
                    logger.info(f"      {name:<20}: {initial:.3f} → {final:.3f}")

    logger.info("=" * 80)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Main execution function for the OrthoBlock effectiveness experiment."""
    logger.info("OrthoBlock Effectiveness Experiment")
    logger.info("=" * 80)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.info(f"GPU memory growth configuration failed: {e}")

    config = OrthoBlockExperimentConfig()
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"   Dense layer variants: {list(config.dense_variants.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"   Number of runs: {config.n_runs}")
    logger.info(f"   Architecture: {len(config.conv_filters)} conv blocks, {config.dense_units} dense units")
    logger.info(f"   Orthogonal regularization factors: {config.ortho_reg_factors}\n")

    try:
        _ = run_experiment(config)
        logger.info("OrthoBlock effectiveness experiment completed successfully.")
    except Exception as e:
        logger.error(f"Experiment failed with an unhandled exception: {e}", exc_info=True)
        raise

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()