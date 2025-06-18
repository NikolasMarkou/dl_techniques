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
3. **GoodhartAwareLoss**: Information-theoretic loss combining entropy regularization
   and mutual information constraints.

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
- Architecture: ResNet-like CNN optimized for CIFAR-10. Model outputs raw logits.
- Dataset: CIFAR-10 (50k train, 10k test, 10 classes)
- Training: Same optimizer, learning rate schedule, epochs, and data augmentation. All
  loss functions are configured to work with logits for numerical stability.
- Evaluation: Comprehensive calibration and performance analysis using project tools. A
  softmax layer is appended to the trained models before analysis.
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

# ------------------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.goodhart_loss import GoodhartAwareLoss
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_cifar10
from dl_techniques.utils.calibration_analyzer import CalibrationAnalyzer
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

# ------------------------------------------------------------------------------
# 1. Single Configuration Class
# ------------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Unified configuration for the CIFAR-10 calibration experiment."""
    # Dataset Configuration
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)
    validation_split: float = 0.1

    # Model Architecture Parameters
    conv_filters: List[int] = (32, 64, 128, 256)
    dense_units: List[int] = (128,)
    dropout_rates: List[float] = (0.25, 0.25, 0.25, 0.25, 0.25)
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_batch_norm: bool = True
    use_residual: bool = True

    # Training Parameters
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5
    monitor_metric: str = 'val_accuracy'

    # Loss Functions to Test
    loss_functions: Dict[str, Callable] = field(default_factory=lambda: {
        'goodhart_aware': lambda: GoodhartAwareLoss(
            entropy_weight=0.1,
            mi_weight=0.01
        ),
        'crossentropy': lambda: keras.losses.CategoricalCrossentropy(from_logits=True),
        'label_smoothing': lambda: keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1,
            from_logits=True
        )
    })

    # Loss Function Specific Parameters
    label_smoothing_alpha: float = 0.1
    gal_entropy_weight: float = 0.1
    gal_mi_weight: float = 0.01

    # Calibration Analysis Parameters
    calibration_bins: int = 15
    analyze_calibration: bool = True

    # Other Parameters
    output_dir: Path = Path("results")
    experiment_name: str = "cifar10_calibration_experiment"
    random_seed: int = 42

# ------------------------------------------------------------------------------
# 2. Model Building Utilities
# ------------------------------------------------------------------------------

def build_residual_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """Build a residual block for improved gradient flow."""
    shortcut = inputs
    x = keras.layers.Conv2D(
        filters=filters, kernel_size=config.kernel_size, padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_1'
    )(inputs)
    if config.use_batch_norm: x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(
        filters=filters, kernel_size=config.kernel_size, padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_2'
    )(x)
    if config.use_batch_norm: x = keras.layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 1), padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay)
        )(shortcut)
        if config.use_batch_norm: shortcut = keras.layers.BatchNormalization()(shortcut)
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def build_conv_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """Build a convolutional block with specified configuration."""
    if config.use_residual and block_index > 0:
        x = build_residual_block(inputs, filters, config, block_index)
    else:
        x = keras.layers.Conv2D(
            filters=filters, kernel_size=config.kernel_size, padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'conv{block_index}'
        )(inputs)
        if config.use_batch_norm: x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
    if block_index < len(config.conv_filters) - 1:
        x = keras.layers.MaxPooling2D(config.pool_size)(x)
    dropout_rate = config.dropout_rates[block_index] if block_index < len(config.dropout_rates) else 0.0
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


def build_model(config: ExperimentConfig, loss_fn: Callable, name: str) -> keras.Model:
    """Build CNN model optimized for CIFAR-10, outputting raw logits."""
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0], kernel_size=config.kernel_size, padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='initial_conv'
    )(x)
    if config.use_batch_norm: x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)
    x = keras.layers.GlobalAveragePooling2D()(x)
    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(
            units=units, kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'dense_{j}'
        )(x)
        if config.use_batch_norm: x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates):
            dropout_rate = config.dropout_rates[dense_dropout_idx]
            if dropout_rate > 0:
                x = keras.layers.Dropout(dropout_rate)(x)

    logits = keras.layers.Dense(
        config.num_classes,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='logits'
    )(x)

    model = keras.Model(inputs=inputs, outputs=logits, name=f'{name}_model')
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return model

# ------------------------------------------------------------------------------
# 3. Experiment Runner
# ------------------------------------------------------------------------------

def run_calibration_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete calibration experiment with comprehensive analysis."""
    keras.utils.set_random_seed(config.random_seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=VisualizationConfig(),
        timestamp_dirs=False
    )
    logger.info("🚀 Starting CIFAR-10 Calibration Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)
    logger.info("📊 Loading CIFAR-10 dataset...")
    cifar10_data = load_and_preprocess_cifar10()
    logger.info("✅ Dataset loaded...")

    # Train models
    models = {}
    all_histories = {'accuracy': {}, 'loss': {}, 'val_accuracy': {}, 'val_loss': {}}
    for loss_name, loss_fn_factory in config.loss_functions.items():
        logger.info(f"🏗️  Building and training {loss_name} model...")
        loss_fn = loss_fn_factory()
        model = build_model(config, loss_fn, loss_name)
        logger.info(f"✅ Model built with {model.count_params():,} parameters (outputs logits)")
        training_config = TrainingConfig(
            epochs=config.epochs, batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric, model_name=loss_name,
            output_dir=experiment_dir / loss_name
        )
        logger.info(f"🏃 Training {loss_name} model...")
        history = train_model(
            model, cifar10_data.x_train, cifar10_data.y_train,
            cifar10_data.x_test, cifar10_data.y_test, training_config
        )
        models[loss_name] = model
        for metric in ['accuracy', 'loss', 'val_accuracy', 'val_loss']:
            all_histories[metric][loss_name] = history.history[metric]
        logger.info(f"✅ {loss_name} training completed! Best val accuracy: {max(history.history['val_accuracy']):.4f}")

    # --- ANALYSIS PHASE ---
    
    logger.info("🛠️  Creating and compiling models with softmax output for analysis...")
    prediction_models = {}
    for name, trained_model in models.items():
        logits_output = trained_model.output
        probs_output = keras.layers.Activation('softmax', name='predictions')(logits_output)
        prediction_model = keras.Model(inputs=trained_model.input, outputs=probs_output, name=f"{name}_prediction")
        prediction_model.compile(
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        prediction_models[name] = prediction_model

    # Weight Analysis phase (uses original models with logits)
    # This can remain as is, it does not depend on model output for metrics.

    # Calibration Analysis phase (uses prediction_models with softmax)
    calibration_metrics = {}
    if config.analyze_calibration:
        logger.info("🎯 Performing calibration effectiveness analysis...")
        try:
            calibration_analyzer = CalibrationAnalyzer(models=prediction_models, calibration_bins=config.calibration_bins)
            calibration_metrics = calibration_analyzer.analyze_calibration(x_test=cifar10_data.x_test, y_test=cifar10_data.y_test)
            calibration_dir = experiment_dir / "calibration_analysis"
            calibration_analyzer.plot_reliability_diagrams(calibration_dir)
            calibration_analyzer.plot_calibration_metrics_comparison(calibration_dir)
            logger.info("✅ Calibration analysis completed successfully!")
        except Exception as e:
            logger.error(f"Calibration analysis failed: {e}")

    # Generate training history visualizations
    # This part is fine as it uses the `histories` object from training.

    # Generate predictions for confusion matrices (uses prediction_models with softmax)
    model_predictions = {name: model.predict(cifar10_data.x_test, verbose=0) for name, model in prediction_models.items()}
    vis_manager.plot_confusion_matrices_comparison(
        y_true=np.argmax(cifar10_data.y_test, axis=1), model_predictions=model_predictions,
        name='loss_function_confusion_matrices', subdir='model_comparison', normalize=True,
        class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    )

    # Analyze model performance directly for accurate metrics
    logger.info("📈 Directly evaluating final model performance...")
    performance_results = {}
    for name, model in prediction_models.items():
        logger.info(f"   Evaluating {name}...")
        # model.evaluate() returns a list: [loss, accuracy, top_k_accuracy]
        # This order is determined by the `metrics` list in prediction_model.compile()
        eval_results = model.evaluate(cifar10_data.x_test, cifar10_data.y_test, verbose=0)
        
        performance_results[name] = {
            'loss': eval_results[0],
            'accuracy': eval_results[1],
            'top5_accuracy': eval_results[2]
        }

    results = {
        'models': models, 'prediction_models': prediction_models, 'histories': all_histories,
        'performance_analysis': performance_results, 'experiment_config': config,
        'calibration_analysis': calibration_metrics
    }
    print_experiment_summary(results)
    return results

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print a comprehensive summary of experiment results."""
    logger.info("=" * 80)
    logger.info("📋 EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    logger.info("🎯 PERFORMANCE METRICS (on Test Set):")
    logger.info(f"{'Model':<20} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12}")
    logger.info("-" * 60)
    for model_name, metrics in results.get('performance_analysis', {}).items():
        if isinstance(metrics, dict):
            accuracy = metrics.get('accuracy', 0.0)
            top5_acc = metrics.get('top5_accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            logger.info(f"{model_name:<20} {accuracy:<12.4f} {top5_acc:<12.4f} {loss:<12.4f}")

    if 'calibration_analysis' in results and results['calibration_analysis']:
        logger.info("🎯 CALIBRATION METRICS (on Test Set):")
        logger.info(f"{'Model':<20} {'ECE':<12} {'Brier Score':<15} {'Mean Entropy':<12}")
        logger.info("-" * 65)
        for model_name, cal_metrics in results.get('calibration_analysis', {}).items():
            logger.info(
                f"{model_name:<20} {cal_metrics['ece']:<12.4f} "
                f"{cal_metrics['brier_score']:<15.4f} {cal_metrics['mean_entropy']:<12.4f}"
            )

    logger.info("🏁 FINAL TRAINING METRICS (on Validation Set):")
    logger.info(f"{'Model':<20} {'Val Accuracy':<15} {'Val Loss':<12}")
    logger.info("-" * 50)
    for model_name in results.get('models', {}).keys():
        final_val_acc = results['histories']['val_accuracy'][model_name][-1]
        final_val_loss = results['histories']['val_loss'][model_name][-1]
        logger.info(f"{model_name:<20} {final_val_acc:<15.4f} {final_val_loss:<12.4f}")
    logger.info("=" * 80)

# ------------------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------------------

def main() -> None:
    """Main execution function for running the calibration experiment."""
    logger.info("🚀 CIFAR-10 Calibration Experiment: Loss Function Comparison")
    logger.info("=" * 80)
    config = ExperimentConfig()
    logger.info("⚙️  EXPERIMENT CONFIGURATION:")
    logger.info(f"   Dataset: {config.dataset_name.upper()}")
    logger.info(f"   Epochs: {config.epochs}")
    logger.info(f"   Batch Size: {config.batch_size}")
    logger.info(f"   Learning Rate: {config.learning_rate}")
    logger.info(f"   Loss Functions: {list(config.loss_functions.keys())}")
    logger.info(f"   Label Smoothing α: {config.label_smoothing_alpha}")
    logger.info(f"   GAL Entropy Weight: {config.gal_entropy_weight}")
    logger.info(f"   GAL MI Weight: {config.gal_mi_weight}")
    logger.info("")

    try:
        run_calibration_experiment(config)
        logger.info("✅ Experiment completed successfully!")
    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}", exc_info=True)
        raise

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------
