"""
CIFAR-10 Calibration Effectiveness Experiment: GoodhartAwareLoss vs Label Smoothing
=================================================================================

This module implements a comprehensive experiment to evaluate and compare the calibration
effectiveness of different loss functions applied to Convolutional Neural Networks (CNNs)
for CIFAR-10 classification. The experiment systematically compares loss function variants
with comprehensive analysis tools following the project's standard analysis framework.

EXPERIMENT OVERVIEW
------------------
The experiment trains multiple CNN models with identical architectures but different
loss functions applied during training:

1. **Standard Cross-Entropy**: Baseline cross-entropy loss
2. **Label Smoothing**: Cross-entropy with label smoothing (α=0.1)
3. **GoodhartAwareLoss**: Information-theoretic loss combining temperature scaling,
   entropy regularization, and mutual information constraints

RESEARCH HYPOTHESIS
------------------
**Core Claims Being Tested:**
- GoodhartAwareLoss should produce better-calibrated models (lower ECE)
- GoodhartAwareLoss should maintain competitive accuracy on clean, in-distribution data
- GoodhartAwareLoss should produce less overconfident predictions (higher entropy)
- GoodhartAwareLoss should show better reliability diagram alignment with perfect calibration

METHODOLOGY
-----------
Each model follows identical training protocols using the project's standard training pipeline:
- Architecture: ResNet-like CNN optimized for CIFAR-10
- Dataset: CIFAR-10 (50k train, 10k test, 10 classes)
- Training: Same optimizer, learning rate schedule, epochs, and data augmentation
- Evaluation: Comprehensive calibration and performance analysis using project tools

MODEL ARCHITECTURE
------------------
- Input: 32x32x3 RGB CIFAR-10 images
- Conv Blocks: Conv2D → BatchNorm → ReLU → Optional Dropout
- Residual connections for improved gradient flow
- Global Average Pooling before final classification layer
- Output: Dense(10) → Loss-specific processing → Softmax
- Regularization: L2 weight decay, dropout, and data augmentation

CALIBRATION METRICS
------------------
**Expected Calibration Error (ECE):**
- Primary metric for measuring calibration quality
- Lower values indicate better calibration (perfect = 0.0)

**Reliability Diagram:**
- Visual representation of calibration quality
- Perfect calibration follows y=x diagonal

**Brier Score:**
- Proper scoring rule measuring prediction quality
- Lower values indicate better probabilistic predictions

**Average Prediction Entropy:**
- Higher entropy indicates appropriate uncertainty
- Lower entropy may indicate overconfidence

ANALYSIS OUTPUTS
---------------
**Training Analysis:**
- Training/validation accuracy and loss curves for all models
- Early stopping behavior and convergence patterns
- Comparative performance metrics using ModelAnalyzer

**Weight Distribution Analysis:**
- L1, L2, and RMS norm distributions across layers using WeightAnalyzer
- Layer-wise weight statistics comparison
- Weight distribution heatmaps and histograms

**Calibration Analysis:**
- Expected Calibration Error comparison
- Reliability diagrams for all models
- Confidence distribution analysis
- Prediction entropy analysis

**Model Performance Analysis:**
- Confusion matrices for each loss function
- Classification accuracy, precision, recall, F1-scores using ModelAnalyzer
- Model comparison visualizations using VisualizationManager

CONFIGURATION
------------
All experiment parameters are centralized in the ExperimentConfig class:
- Model architecture parameters (filters, layers, regularization)
- Training hyperparameters (epochs, batch size, learning rate)
- Loss function parameters (GAL weights, label smoothing alpha)
- Calibration analysis settings (number of bins, confidence thresholds)
- Analysis options (which metrics to compute, plot formats)

USAGE
-----
To run with default settings:
    python cifar10_calibration_experiment.py

To customize hyperparameters:
    config = ExperimentConfig()
    config.gal_temperature = 2.5
    config.gal_entropy_weight = 0.15
    config.epochs = 200
    results = run_calibration_experiment(config)
"""

# ------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable

import keras
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.goodhart_loss import GoodhartAwareLoss
from dl_techniques.utils.model_analyzer import ModelAnalyzer
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.weight_analyzer import WeightAnalyzerConfig, WeightAnalyzer
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig
from dl_techniques.utils.datasets import load_and_preprocess_cifar10
from dl_techniques.utils.calibration_metrics import (
    compute_ece,
    compute_brier_score,
    compute_reliability_data,
    compute_prediction_entropy_stats
)

# ------------------------------------------------------------------------------
# 2. Single Configuration Class
# ------------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Unified configuration for the CIFAR-10 calibration experiment.

    Contains all parameters for model architecture, training, loss functions,
    calibration analysis, weight analysis, and visualization in a single
    consolidated configuration class.
    """
    # Dataset Configuration
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)
    validation_split: float = 0.1

    # Model Architecture Parameters
    conv_filters: List[int] = (32, 64, 128, 256)
    dense_units: List[int] = (128,)
    dropout_rates: List[float] = (0.25, 0.25, 0.3, 0.4, 0.5)  # conv layers + dense
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_batch_norm: bool = True
    use_residual: bool = True

    # Training Parameters
    epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5
    monitor_metric: str = 'val_accuracy'

    # Loss Functions to Test
    loss_functions: Dict[str, Callable] = field(default_factory=lambda: {
        'goodhart_aware': lambda: GoodhartAwareLoss(
            temperature=2.0,
            entropy_weight=0.1,
            mi_weight=0.01
        ),
        'crossentropy': lambda: keras.losses.CategoricalCrossentropy(),
        'label_smoothing': lambda: keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    })

    # Loss Function Specific Parameters
    label_smoothing_alpha: float = 0.1
    gal_temperature: float = 2.0
    gal_entropy_weight: float = 0.1
    gal_mi_weight: float = 0.01

    # Calibration Analysis Parameters
    calibration_bins: int = 15
    confidence_threshold: float = 0.5
    bootstrap_samples: int = 1000
    analyze_calibration: bool = True

    # Weight Analysis Parameters
    compute_l1_norm: bool = True
    compute_l2_norm: bool = True
    compute_rms_norm: bool = True
    analyze_biases: bool = True

    # Visualization Parameters
    save_plots: bool = True
    plot_format: str = 'png'
    plot_style: str = 'default'

    # Data Augmentation
    use_data_augmentation: bool = True
    rotation_range: float = 15.0
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    horizontal_flip: bool = True

    # Experiment Parameters
    output_dir: Path = Path("results")
    experiment_name: str = "cifar10_calibration_experiment"
    random_seed: int = 42

# ------------------------------------------------------------------------------
# 3. Model Building Utilities
# ------------------------------------------------------------------------------

def build_residual_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """Build a residual block for improved gradient flow.

    Args:
        inputs: Input tensor
        filters: Number of filters
        config: Experiment configuration
        block_index: Index of the block for naming and dropout rate

    Returns:
        Output tensor after residual block
    """
    shortcut = inputs

    # First conv layer
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_1'
    )(inputs)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Second conv layer
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_2'
    )(x)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)

    # Adjust shortcut if needed
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

    # Add shortcut and activation
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)

    return x


def build_conv_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """Build a convolutional block with specified configuration.

    Args:
        inputs: Input tensor
        filters: Number of filters for conv layer
        config: Experiment configuration
        block_index: Index of the block for naming and dropout rate

    Returns:
        Output tensor after applying conv block
    """
    if config.use_residual and block_index > 0:
        x = build_residual_block(inputs, filters, config, block_index)
    else:
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
        x = keras.layers.Activation('relu')(x)

    # Pooling and dropout
    if block_index < len(config.conv_filters) - 1:  # Don't pool after last conv block
        x = keras.layers.MaxPooling2D(config.pool_size)(x)

    dropout_rate = config.dropout_rates[block_index] if block_index < len(config.dropout_rates) else 0.0
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def build_model(config: ExperimentConfig, loss_fn: Callable, name: str) -> keras.Model:
    """Build CNN model optimized for CIFAR-10 with specified loss function.

    Args:
        config: Experiment configuration
        loss_fn: Loss function to use for training
        name: Name identifier for the model

    Returns:
        Compiled Keras model
    """
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Initial conv layer
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0],
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='initial_conv'
    )(x)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)

    # Global average pooling instead of flatten
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Dense layers
    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(
            units=units,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'dense_{j}'
        )(x)

        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        # Apply dropout to dense layers
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates):
            dropout_rate = config.dropout_rates[dense_dropout_idx]
            if dropout_rate > 0:
                x = keras.layers.Dropout(dropout_rate)(x)

    # Output layer (logits)
    logits = keras.layers.Dense(
        config.num_classes,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='logits'
    )(x)

    # Apply softmax for predictions
    outputs = keras.layers.Activation('softmax', name='predictions')(logits)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{name}_model')

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    return model

# ------------------------------------------------------------------------------
# 4. Calibration Analysis Utilities
# ------------------------------------------------------------------------------

class CalibrationAnalyzer:
    """Analyzer for model calibration effectiveness and reliability."""

    def __init__(self, models: Dict[str, keras.Model], config: ExperimentConfig):
        """Initialize the calibration analyzer.

        Args:
            models: Dictionary of trained models
            config: Experiment configuration
        """
        self.models = models
        self.config = config
        self.calibration_metrics: Dict[str, Dict[str, float]] = {}
        self.reliability_data: Dict[str, Dict[str, np.ndarray]] = {}

    def analyze_calibration(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Analyze calibration for all models.

        Args:
            x_test: Test input data
            y_test: Test target data (one-hot encoded)

        Returns:
            Dictionary containing calibration metrics for each model
        """
        logger.info("Analyzing calibration effectiveness...")

        y_true_classes = np.argmax(y_test, axis=1)

        for name, model in self.models.items():
            logger.info(f"Computing calibration metrics for {name} model...")

            # Get predictions
            y_pred_proba = model.predict(x_test, verbose=0)

            # Compute calibration metrics
            ece = compute_ece(y_true_classes, y_pred_proba, self.config.calibration_bins)
            reliability_data = compute_reliability_data(
                y_true_classes,
                y_pred_proba,
                self.config.calibration_bins
            )
            brier_score = compute_brier_score(y_test, y_pred_proba)
            entropy_stats = compute_prediction_entropy_stats(y_pred_proba)

            self.calibration_metrics[name] = {
                'ece': ece,
                'brier_score': brier_score,
                'mean_entropy': entropy_stats['mean_entropy'],
                'std_entropy': entropy_stats['std_entropy'],
                'median_entropy': entropy_stats['median_entropy']
            }

            self.reliability_data[name] = reliability_data

            logger.info(f"Calibration metrics for {name}:")
            logger.info(f"  ECE: {ece:.4f}")
            logger.info(f"  Brier Score: {brier_score:.4f}")
            logger.info(f"  Mean Entropy: {entropy_stats['mean_entropy']:.4f}")

        return self.calibration_metrics

    def plot_reliability_diagrams(self, output_dir: Path) -> None:
        """Plot reliability diagrams for all models.

        Args:
            output_dir: Directory to save plots
        """
        if not self.reliability_data:
            logger.warning("No reliability data available for plotting")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Individual reliability diagrams
        for model_name, reliability_data in self.reliability_data.items():
            plt.figure(figsize=(8, 8))

            # Plot perfect calibration line
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')

            # Plot model calibration
            bin_centers = reliability_data['bin_centers']
            bin_accuracies = reliability_data['bin_accuracies']
            bin_counts = reliability_data['bin_counts']

            # Use bin counts for marker sizes
            sizes = 100 * bin_counts / np.max(bin_counts + 1)

            scatter = plt.scatter(
                bin_centers,
                bin_accuracies,
                s=sizes,
                alpha=0.7,
                label=f'{model_name.replace("_", " ").title()} Calibration',
                color='red'
            )

            plt.xlabel('Confidence', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title(f'Reliability Diagram - {model_name.replace("_", " ").title()}', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])

            # Add colorbar for bin counts
            cbar = plt.colorbar(scatter)
            cbar.set_label('Number of Samples', rotation=270, labelpad=15)

            plt.tight_layout()
            plt.savefig(output_dir / f'reliability_{model_name}.{self.config.plot_format}',
                       dpi=300, bbox_inches='tight')
            plt.close()

        # Comparison reliability diagram
        plt.figure(figsize=(10, 8))

        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Perfect Calibration')

        colors = ['blue', 'red', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']

        for i, (model_name, reliability_data) in enumerate(self.reliability_data.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            plt.plot(
                reliability_data['bin_centers'],
                reliability_data['bin_accuracies'],
                marker=marker,
                markersize=8,
                linewidth=2,
                label=model_name.replace('_', ' ').title(),
                color=color,
                alpha=0.8
            )

        plt.xlabel('Confidence', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Calibration Comparison: Loss Functions', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / f'calibration_comparison.{self.config.plot_format}',
                   dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved reliability diagrams to {output_dir}")

    def plot_calibration_metrics_comparison(self, output_dir: Path) -> None:
        """Plot comparison of calibration metrics across models.

        Args:
            output_dir: Directory to save plots
        """
        if not self.calibration_metrics:
            logger.warning("No calibration metrics available for plotting")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_to_plot = ['ece', 'brier_score', 'mean_entropy']
        metric_labels = ['Expected Calibration Error', 'Brier Score', 'Mean Prediction Entropy']

        for metric, label in zip(metrics_to_plot, metric_labels):
            plt.figure(figsize=(10, 6))

            models = list(self.calibration_metrics.keys())
            values = [self.calibration_metrics[model][metric] for model in models]

            bars = plt.bar([m.replace('_', ' ').title() for m in models], values)
            plt.title(f'{label} Comparison')
            plt.ylabel(label)
            plt.xticks(rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height(),
                    f'{value:.4f}',
                    ha='center',
                    va='bottom'
                )

            plt.tight_layout()
            plt.savefig(output_dir / f'{metric}_comparison.{self.config.plot_format}',
                       dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Saved calibration metrics plots to {output_dir}")

    def save_calibration_analysis(self, output_dir: Path) -> None:
        """Save calibration analysis results to file.

        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save calibration statistics
        with open(output_dir / 'calibration_statistics.txt', 'w') as f:
            f.write("Calibration Analysis Results\n")
            f.write("=" * 50 + "\n\n")

            for model_name, metrics in self.calibration_metrics.items():
                f.write(f"{model_name.upper().replace('_', ' ')} MODEL:\n")
                f.write("-" * 30 + "\n")
                for metric, value in metrics.items():
                    f.write(f"{metric.replace('_', ' ').title()}: {value:.6f}\n")
                f.write("\n")

        logger.info(f"Saved calibration analysis to {output_dir}")

# ------------------------------------------------------------------------------
# 5. Experiment Runner
# ------------------------------------------------------------------------------

def run_calibration_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete calibration experiment with comprehensive analysis.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary containing all experiment results
    """
    # Set random seed for reproducibility
    keras.utils.set_random_seed(config.random_seed)

    # Setup directories and managers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    vis_config = VisualizationConfig()
    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=vis_config,
        timestamp_dirs=False
    )

    logger.info("🚀 Starting CIFAR-10 Calibration Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # Load and preprocess data
    logger.info("📊 Loading CIFAR-10 dataset...")
    cifar10_data = load_and_preprocess_cifar10()

    logger.info(f"✅ Dataset loaded:")
    logger.info(f"   Training samples: {cifar10_data.x_train.shape[0]}")
    logger.info(f"   Test samples: {cifar10_data.x_test.shape[0]}")
    logger.info(f"   Input shape: {cifar10_data.x_train.shape[1:]}")
    logger.info(f"   Number of classes: {config.num_classes}")

    # Train models
    models = {}
    all_histories = {
        'accuracy': {},
        'loss': {},
        'val_accuracy': {},
        'val_loss': {}
    }

    # Training phase
    for loss_name, loss_fn_factory in config.loss_functions.items():
        logger.info(f"\n🏗️  Building and training {loss_name} model...")
        logger.info("-" * 50)

        # Build model
        loss_fn = loss_fn_factory()
        model = build_model(config, loss_fn, loss_name)

        logger.info(f"✅ Model built with {model.count_params():,} parameters")

        # Configure training for this model
        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric,
            model_name=loss_name,
            output_dir=experiment_dir / loss_name
        )

        # Train the model
        logger.info(f"🏃 Training {loss_name} model...")
        history = train_model(
            model,
            cifar10_data.x_train,
            cifar10_data.y_train,
            cifar10_data.x_test,
            cifar10_data.y_test,
            training_config
        )

        # Store the model and history
        models[loss_name] = model
        for metric in ['accuracy', 'loss']:
            all_histories[metric][loss_name] = history.history[metric]
            all_histories[f'val_{metric}'][loss_name] = history.history[f'val_{metric}']

        logger.info(f"✅ {loss_name} training completed!")
        logger.info(f"   Best val accuracy: {max(history.history['val_accuracy']):.4f}")
        logger.info(f"   Final val loss: {min(history.history['val_loss']):.4f}")

    # Weight Analysis phase
    logger.info("\n📊 Performing weight distribution analysis...")

    try:
        analysis_config = WeightAnalyzerConfig(
            compute_l1_norm=config.compute_l1_norm,
            compute_l2_norm=config.compute_l2_norm,
            compute_rms_norm=config.compute_rms_norm,
            analyze_biases=config.analyze_biases,
            save_plots=config.save_plots,
            export_format=config.plot_format
        )

        weight_analyzer = WeightAnalyzer(
            models=models,
            config=analysis_config,
            output_dir=experiment_dir / "weight_analysis"
        )

        if weight_analyzer.has_valid_analysis():
            weight_analyzer.plot_comprehensive_dashboard()
            weight_analyzer.plot_norm_distributions()
            weight_analyzer.plot_layer_comparisons(['mean', 'std', 'l2_norm'])
            weight_analyzer.save_analysis_results()
            logger.info("✅ Weight analysis completed successfully!")
        else:
            logger.warning("❌ No valid weight data found for analysis")

    except Exception as e:
        logger.error(f"Weight analysis failed: {e}")
        logger.info("Continuing with experiment without weight analysis...")

    # Calibration Analysis phase
    if config.analyze_calibration:
        logger.info("\n🎯 Performing calibration effectiveness analysis...")

        try:
            calibration_analyzer = CalibrationAnalyzer(models, config)
            calibration_metrics = calibration_analyzer.analyze_calibration(
                cifar10_data.x_test,
                cifar10_data.y_test
            )

            # Generate calibration visualizations
            calibration_dir = experiment_dir / "calibration_analysis"
            calibration_analyzer.plot_reliability_diagrams(calibration_dir)
            calibration_analyzer.plot_calibration_metrics_comparison(calibration_dir)
            calibration_analyzer.save_calibration_analysis(calibration_dir)

            logger.info("✅ Calibration analysis completed successfully!")

        except Exception as e:
            logger.error(f"Calibration analysis failed: {e}")
            calibration_metrics = {}

    # Generate training history visualizations
    for metric in ['accuracy', 'loss']:
        combined_histories = {
            name: {
                metric: all_histories[metric][name],
                f'val_{metric}': all_histories[f'val_{metric}'][name]
            }
            for name in models.keys()
        }

        vis_manager.plot_history(
            combined_histories,
            [metric],
            f'training_{metric}_comparison',
            title=f'Loss Functions {metric.capitalize()} Comparison'
        )

    # Generate predictions for confusion matrices
    model_predictions = {}
    for name, model in models.items():
        predictions = model.predict(cifar10_data.x_test, verbose=0)
        model_predictions[name] = predictions

    # Plot confusion matrices comparison
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    vis_manager.plot_confusion_matrices_comparison(
        y_true=np.argmax(cifar10_data.y_test, axis=1),
        model_predictions=model_predictions,
        name='loss_function_confusion_matrices',
        subdir='model_comparison',
        normalize=True,
        class_names=class_names
    )

    # Analyze model performance
    logger.info("\n📈 Analyzing model performance...")
    analyzer = ModelAnalyzer(models, vis_manager)
    performance_results = analyzer.analyze_models(cifar10_data)

    # Combine all results
    results = {
        'models': models,
        'histories': all_histories,
        'performance_analysis': performance_results,
        'experiment_config': config
    }

    if config.analyze_calibration:
        results['calibration_analysis'] = calibration_metrics

    # Print comprehensive results summary
    print_experiment_summary(results)

    return results


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print a comprehensive summary of experiment results.

    Args:
        results: Dictionary containing all experiment results
    """
    logger.info("\n" + "=" * 80)
    logger.info("📋 EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # Performance comparison
    logger.info("\n🎯 PERFORMANCE METRICS:")
    logger.info(f"{'Model':<20} {'Accuracy':<12} {'Top-5 Acc':<12}")
    logger.info("-" * 50)

    for model_name, metrics in results['performance_analysis'].items():
        if isinstance(metrics, dict):
            accuracy = metrics.get('accuracy', 0.0)
            top5_acc = metrics.get('top5_accuracy', 0.0)
            logger.info(f"{model_name:<20} {accuracy:<12.4f} {top5_acc:<12.4f}")

    # Calibration comparison
    if 'calibration_analysis' in results:
        logger.info("\n🎯 CALIBRATION METRICS:")
        logger.info(f"{'Model':<20} {'ECE':<12} {'Brier Score':<15} {'Mean Entropy':<12}")
        logger.info("-" * 65)

        for model_name, cal_metrics in results['calibration_analysis'].items():
            logger.info(
                f"{model_name:<20} "
                f"{cal_metrics['ece']:<12.4f} "
                f"{cal_metrics['brier_score']:<15.4f} "
                f"{cal_metrics['mean_entropy']:<12.4f}"
            )

    # Final training metrics
    logger.info("\n🏁 FINAL TRAINING METRICS:")
    logger.info(f"{'Model':<20} {'Val Accuracy':<15} {'Val Loss':<12}")
    logger.info("-" * 50)

    for model_name in results['models'].keys():
        final_val_acc = results['histories']['val_accuracy'][model_name][-1]
        final_val_loss = results['histories']['val_loss'][model_name][-1]
        logger.info(f"{model_name:<20} {final_val_acc:<15.4f} {final_val_loss:<12.4f}")

    logger.info("=" * 80)

# ------------------------------------------------------------------------------
# 6. Main Execution
# ------------------------------------------------------------------------------

def main() -> None:
    """Main execution function for running the calibration experiment."""
    logger.info("🚀 CIFAR-10 Calibration Experiment: Loss Function Comparison")
    logger.info("=" * 80)

    # Create configuration
    config = ExperimentConfig()

    # Display configuration
    logger.info("⚙️  EXPERIMENT CONFIGURATION:")
    logger.info(f"   Dataset: {config.dataset_name.upper()}")
    logger.info(f"   Epochs: {config.epochs}")
    logger.info(f"   Batch Size: {config.batch_size}")
    logger.info(f"   Learning Rate: {config.learning_rate}")
    logger.info(f"   Loss Functions: {list(config.loss_functions.keys())}")
    logger.info(f"   Label Smoothing α: {config.label_smoothing_alpha}")
    logger.info(f"   GAL Temperature: {config.gal_temperature}")
    logger.info(f"   GAL Entropy Weight: {config.gal_entropy_weight}")
    logger.info(f"   GAL MI Weight: {config.gal_mi_weight}")
    logger.info(f"   Calibration Bins: {config.calibration_bins}")
    logger.info("")

    # Run experiment
    try:
        results = run_calibration_experiment(config)
        logger.info("\n✅ Experiment completed successfully!")

        # Save results summary
        summary_path = results['experiment_config'].output_dir / "experiment_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("CIFAR-10 Calibration Experiment Results\n")
            f.write("=" * 50 + "\n\n")
            f.write("Configuration:\n")
            for key, value in results['experiment_config'].__dict__.items():
                if not callable(value):
                    f.write(f"  {key}: {value}\n")
            f.write(f"\nResults saved to: {summary_path.parent}\n")

        return results

    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}")
        raise


if __name__ == "__main__":
    results = main()