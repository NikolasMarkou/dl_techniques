"""
Colored CIFAR10 Spurious Correlation Experiment: GoodhartAwareLoss vs. Baselines
================================================================================

This module implements a comprehensive experiment to evaluate and compare the robustness
of different loss functions against spurious correlations. The experiment uses a synthetic
Colored CIFAR10 dataset to test whether models can avoid "gaming the metric" by exploiting
dataset artifacts, a key manifestation of Goodhart's Law in machine learning.

The experiment systematically compares loss function variants using the project's
standard analysis framework, focusing on metrics that quantify robustness to
distributional shifts.

EXPERIMENT OVERVIEW
------------------
The experiment trains multiple CNN models with identical architectures but different
loss functions applied during training:

1. **Standard Cross-Entropy**: A baseline model expected to overfit to spurious features.
2. **Label Smoothing**: A common regularization technique tested for its robustness benefits.
3. **GoodhartAwareLoss**: An information-theoretic loss designed to prevent metric gaming
   by regularizing the model's internal information flow and output uncertainty.

SPURIOUS CORRELATION DESIGN
---------------------------
A synthetic Colored CIFAR10 dataset is generated with a controlled distribution shift
between the training and test sets:

- **Training Set (High Correlation)**: A strong, spurious correlation (e.g., 95%) is
  introduced between the digit's class and its color. For example, the digit '7' is
  colored red 95% of the time. A model can achieve high training accuracy by simply
  learning this color-to-class mapping.

- **Test Set (Zero Correlation)**: The spurious correlation is completely removed.
  Colors are assigned randomly, providing no information about the digit's class.
  A model's performance on this set reveals whether it learned the true underlying
  feature (digit shape) or relied on the spurious shortcut (color).

RESEARCH HYPOTHESIS
------------------
**Core Claims Being Tested:**
- GoodhartAwareLoss (GAL) should maintain higher test accuracy on the uncorrelated
  test set, demonstrating superior robustness.
- GAL should exhibit a smaller "generalization gap" (the drop in accuracy from the
  training set to the test set), indicating less overfitting to spurious patterns.
- GAL should force the model to learn the true, causal features (digit morphology)
  instead of the easy, spurious ones (color).

METHODOLOGY
-----------
Each model follows an identical training and evaluation protocol to ensure a fair comparison:

- **Architecture**: A standard CNN optimized for 28x28 images. All models share the
  exact same architecture and weight initialization. The model outputs raw logits for
  numerical stability.
- **Dataset**: The generated Colored CIFAR10 dataset with a 95% train / 0% test correlation.
- **Training**: All models are trained using the project's standard `train_model` utility,
  ensuring identical optimizers, learning rate schedules, and epochs.
- **Evaluation**: Robustness is quantified by measuring the test set accuracy and the
  generalization gap. A modular `RobustnessAnalyzer` handles the evaluation.

ROBUSTNESS METRICS
------------------
**Test Set Accuracy (Primary Metric):**
- The model's accuracy on the "fair" test set where the color correlation is broken.
- This is the single most important measure of a model's robustness. Higher is better.

**Generalization Gap:**
- Defined as: `Train Accuracy - Test Accuracy`.
- Measures the degree to which a model has overfitted to the spurious correlations
  present only in the training data. A smaller gap indicates more robust learning.

ANALYSIS OUTPUTS
---------------
The experiment produces a set of comparative analyses and visualizations:

**Robustness Analysis:**
- A bar chart comparing the Test Accuracy and Generalization Gap across all tested
  loss functions.
- A clear, data-driven summary verdict on the research hypothesis.

**Training Analysis:**
- Training and validation accuracy/loss curves for all models, managed by the
  `train_model` utility.
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

# --- Core project utilities (preserved from original) ---
from dl_techniques.utils.logger import logger
from dl_techniques.losses.goodhart_loss import GoodhartAwareLoss
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_cifar10
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

# --- REPLACEMENT: ModelAnalyzer replaces all older, separate analyzers ---
from dl_techniques.utils.analyzer import (
    ModelAnalyzer,
    AnalysisConfig,
    AnalysisResults,
    DataInput
)


# ------------------------------------------------------------------------------
# 2. Configuration Class (Unchanged)
# ------------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for the CIFAR-10 loss comparison experiment."""
    # Dataset Configuration
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)

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
    epochs: int = 0
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    monitor_metric: str = 'val_accuracy'

    # Loss Functions to Test
    loss_functions: Dict[str, Callable] = field(default_factory=lambda: {
        'CrossEntropy': lambda: keras.losses.CategoricalCrossentropy(from_logits=True),
        'LabelSmoothing': lambda: keras.losses.CategoricalCrossentropy(label_smoothing=0.1, from_logits=True),
        'FocalLoss': lambda: keras.losses.CategoricalFocalCrossentropy(gamma=2.0, from_logits=True),
        'GoodhartAware': lambda: GoodhartAwareLoss(entropy_weight=0.1, mi_weight=0.01, from_logits=True),
    })

    # Analysis Parameters
    calibration_bins: int = 15

    # Other Parameters
    output_dir: Path = Path("results")
    experiment_name: str = "cifar10_loss_comparison_analyzer"
    random_seed: int = 42


# ------------------------------------------------------------------------------
# 3. Model Building Utilities (Unchanged)
# ------------------------------------------------------------------------------

def build_residual_block(inputs: keras.layers.Layer, filters: int, config: ExperimentConfig,
                         block_index: int) -> keras.layers.Layer:
    shortcut = inputs
    x = keras.layers.Conv2D(filters, config.kernel_size, padding='same', kernel_initializer=config.kernel_initializer,
                            kernel_regularizer=keras.regularizers.L2(config.weight_decay), name=f'conv{block_index}_1')(
        inputs)
    if config.use_batch_norm: x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters, config.kernel_size, padding='same', kernel_initializer=config.kernel_initializer,
                            kernel_regularizer=keras.regularizers.L2(config.weight_decay), name=f'conv{block_index}_2')(
        x)
    if config.use_batch_norm: x = keras.layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer=config.kernel_initializer,
                                       kernel_regularizer=keras.regularizers.L2(config.weight_decay))(shortcut)
        if config.use_batch_norm: shortcut = keras.layers.BatchNormalization()(shortcut)
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def build_conv_block(inputs: keras.layers.Layer, filters: int, config: ExperimentConfig,
                     block_index: int) -> keras.layers.Layer:
    if config.use_residual and block_index > 0:
        x = build_residual_block(inputs, filters, config, block_index)
    else:
        x = keras.layers.Conv2D(filters, config.kernel_size, padding='same',
                                kernel_initializer=config.kernel_initializer,
                                kernel_regularizer=keras.regularizers.L2(config.weight_decay),
                                name=f'conv{block_index}')(inputs)
        if config.use_batch_norm: x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
    if block_index < len(config.conv_filters) - 1: x = keras.layers.MaxPooling2D(config.pool_size)(x)
    dropout_rate = config.dropout_rates[block_index] if block_index < len(config.dropout_rates) else 0.0
    if dropout_rate > 0: x = keras.layers.Dropout(dropout_rate)(x)
    return x


def build_model(config: ExperimentConfig, loss_fn: Callable, name: str) -> keras.Model:
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs
    x = keras.layers.Conv2D(config.conv_filters[0], config.kernel_size, padding='same',
                            kernel_initializer=config.kernel_initializer,
                            kernel_regularizer=keras.regularizers.L2(config.weight_decay), name='initial_conv')(x)
    if config.use_batch_norm: x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    for i, filters in enumerate(config.conv_filters): x = build_conv_block(x, filters, config, i)
    x = keras.layers.GlobalAveragePooling2D()(x)
    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(units, kernel_initializer=config.kernel_initializer,
                               kernel_regularizer=keras.regularizers.L2(config.weight_decay), name=f'dense_{j}')(x)
        if config.use_batch_norm: x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates):
            dropout_rate = config.dropout_rates[dense_dropout_idx]
            if dropout_rate > 0: x = keras.layers.Dropout(dropout_rate)(x)
    logits = keras.layers.Dense(config.num_classes, kernel_initializer=config.kernel_initializer,
                                kernel_regularizer=keras.regularizers.L2(config.weight_decay), name='logits')(x)
    model = keras.Model(inputs=inputs, outputs=logits, name=f'{name}_model')
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model


# ------------------------------------------------------------------------------
# 4. Experiment Runner
# ------------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete experiment."""
    keras.utils.set_random_seed(config.random_seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=VisualizationConfig(),
        timestamp_dirs=False
    )

    logger.info("🚀 Starting CIFAR-10 Loss Comparison Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)
    logger.info("📊 Loading CIFAR-10 dataset...")
    cifar10_data = load_and_preprocess_cifar10()
    logger.info("✅ Dataset loaded...")

    # --- TRAINING PHASE ---
    models = {}
    all_histories = {}
    for loss_name, loss_fn_factory in config.loss_functions.items():
        logger.info(f"--- Training model: {loss_name} ---")
        model = build_model(config, loss_fn_factory(), loss_name)
        training_config = TrainingConfig(
            epochs=config.epochs, batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric, model_name=loss_name,
            output_dir=experiment_dir / "training_plots" / loss_name
        )
        history = train_model(
            model, cifar10_data.x_train, cifar10_data.y_train,
            cifar10_data.x_test, cifar10_data.y_test, training_config
        )
        models[loss_name] = model
        all_histories[loss_name] = history.history  # Correct format for vis_manager
        logger.info(f"✅ {loss_name} training completed!")

    # --- ANALYSIS PHASE ---

    logger.info("🛠️  Creating models with softmax output for analysis...")
    analysis_models = {}
    for name, trained_model in models.items():
        logits_output = trained_model.output
        probs_output = keras.layers.Activation('softmax', name='predictions')(logits_output)
        analysis_model = keras.Model(inputs=trained_model.input, outputs=probs_output, name=f"{name}_prediction")
        analysis_model.compile(loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
        analysis_models[name] = analysis_model

    # --- NEW: Free up memory ---
    logger.info("🗑️  Releasing original training models from memory...")
    del models
    import gc
    gc.collect()  # Ask the garbage collector to run
    # --- END NEW ---

    # === MODEL ANALYZER BLOCK ===
    logger.info("📊 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None
    try:
        analyzer_config = AnalysisConfig(
            analyze_weights=True, analyze_calibration=True, analyze_probability_distributions=True,
            analyze_activations=False, analyze_information_flow=False,
            calibration_bins=config.calibration_bins, save_plots=True, plot_style='publication',
        )
        analyzer = ModelAnalyzer(
            models=analysis_models, config=analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )
        model_analysis_results = analyzer.analyze(data=DataInput.from_object(cifar10_data))
        logger.info("✅ Model analysis completed successfully!")
    except Exception as e:
        logger.error(f"Model analysis failed: {e}", exc_info=True)

    # --- Other Visualizations (Correct usage of VisualizationManager) ---
    logger.info("🖼️  Generating training history and confusion matrix plots...")

    vis_manager.plot_history(
        histories=all_histories,
        metrics=['accuracy', 'loss'],
        name='training_comparison',
        subdir='training_plots',
        title='Loss Functions Training & Validation Comparison'
    )

    raw_predictions = {name: model.predict(cifar10_data.x_test, verbose=0) for name, model in analysis_models.items()}
    class_predictions = {name: np.argmax(preds, axis=1) for name, preds in raw_predictions.items()}
    vis_manager.plot_confusion_matrices_comparison(
        y_true=cifar10_data.y_test,
        model_predictions=class_predictions,
        name='loss_function_confusion_matrices',
        subdir='model_comparison',
        normalize=True,
        class_names=[str(i) for i in range(10)]
    )

    # --- Final Performance Evaluation ---
    logger.info("📈 Evaluating final model performance on full test set...")
    performance_results = {}
    for name, model in analysis_models.items():
        eval_results = model.evaluate(cifar10_data.x_test, cifar10_data.y_test, verbose=0)
        performance_results[name] = dict(zip(model.metrics_names, eval_results))

    # --- SUMMARY ---
    results_payload = {
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'histories': all_histories
    }
    print_experiment_summary(results_payload)
    return results_payload


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print a comprehensive summary of experiment results."""
    logger.info("\n" + "=" * 80)
    logger.info("📋 EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    if 'performance_analysis' in results and results['performance_analysis']:
        logger.info("🎯 PERFORMANCE METRICS (on Full Test Set):")
        logger.info(f"{'Model':<20} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12}")
        logger.info("-" * 60)
        for model_name, metrics in results['performance_analysis'].items():
            accuracy = metrics.get('accuracy', 0.0)
            top5_acc = metrics.get('top_k_categorical_accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            logger.info(f"{model_name:<20} {accuracy:<12.4f} {top5_acc:<12.4f} {loss:<12.4f}")

    model_analysis = results.get('model_analysis')
    if model_analysis and model_analysis.calibration_metrics:
        logger.info("\n🎯 CALIBRATION METRICS (from Model Analyzer):")
        logger.info(f"{'Model':<20} {'ECE':<12} {'Brier Score':<15} {'Mean Entropy':<12}")
        logger.info("-" * 65)
        for model_name, cal_metrics in model_analysis.calibration_metrics.items():
            logger.info(
                f"{model_name:<20} {cal_metrics['ece']:<12.4f} "
                f"{cal_metrics['brier_score']:<15.4f} {cal_metrics['mean_entropy']:<12.4f}"
            )

    if 'histories' in results and results['histories']:
        logger.info("\n🏁 FINAL TRAINING METRICS (on Validation Set):")
        logger.info(f"{'Model':<20} {'Val Accuracy':<15} {'Val Loss':<12}")
        logger.info("-" * 50)
        for model_name, history_dict in results['histories'].items():
            # Safely get the last value of the validation metrics
            final_val_acc = history_dict.get('val_accuracy', [0.0])[-1]
            final_val_loss = history_dict.get('val_loss', [0.0])[-1]
            logger.info(f"{model_name:<20} {final_val_acc:<15.4f} {final_val_loss:<12.4f}")

    logger.info("=" * 80)


# ------------------------------------------------------------------------------
# 5. Main Execution
# ------------------------------------------------------------------------------

def main() -> None:
    """Main execution function for running the experiment."""
    logger.info("🚀 CIFAR-10 Loss Function Comparison")
    logger.info("=" * 80)
    config = ExperimentConfig()
    logger.info("⚙️  EXPERIMENT CONFIGURATION:")
    logger.info(f"   Loss Functions: {list(config.loss_functions.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch Size: {config.batch_size}\n")

    try:
        run_experiment(config)
        logger.info("✅ Experiment completed successfully!")
    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()