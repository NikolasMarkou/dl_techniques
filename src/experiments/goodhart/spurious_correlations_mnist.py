"""
Colored MNIST Spurious Correlation Experiment: GoodhartAwareLoss vs. Baselines
==============================================================================

This experiment evaluates the robustness of different loss functions against
spurious correlations using a synthetic Colored MNIST dataset. It tests whether
models can avoid "gaming the metric" by exploiting dataset artifacts, a key
manifestation of Goodhart's Law in machine learning.

The study systematically compares standard cross-entropy, label smoothing,
focal loss, and the information-theoretic GoodhartAwareLoss to determine which
formulations produce models that generalize based on true features (digit shape)
rather than spurious shortcuts (color).

Experimental Design
-------------------

**Dataset**: Colored MNIST (10 classes, 28×28 RGB images)
- A synthetic dataset with a controlled distribution shift.
- **Training Set**: High (95%) spurious correlation between digit class and color.
- **Test Set**: Zero correlation; color is random and uninformative.
- **Validation Set**: Intermediate (50%) correlation for monitoring.

**Model Architecture**: ResNet-inspired CNN with the following components:
- Initial convolutional layer (32 filters)
- 3 convolutional blocks with residual connections
- Progressive filter scaling: [32, 64, 128]
- Batch normalization and dropout regularization
- Global average pooling
- Dense classification layers with L2 regularization

**Loss Functions Evaluated**:

1.  **Standard Cross-Entropy**: The baseline, expected to overfit to spurious features.
2.  **Label Smoothing**: Regularization to reduce overconfidence.
3.  **Focal Loss**: Down-weights easy examples to focus on hard ones.
4.  **Goodhart-Aware Loss**: Combines cross-entropy with entropy and mutual
    information regularization to improve robustness.

Comprehensive Analysis Pipeline
------------------------------

The experiment employs a multi-faceted analysis approach using the `ModelAnalyzer`
and experiment-specific metrics:

**Training Analysis**:
- Training and validation curves for all loss functions.
- Convergence behavior and early stopping.

**Model Performance Evaluation**:
- **Test Set Accuracy**: The primary metric for robustness on the uncorrelated test set.
- Top-k accuracy and loss values.

**Calibration and Distribution Analysis** (via `ModelAnalyzer`):
- Expected Calibration Error (ECE) and Brier score.
- Reliability diagrams and confidence histograms.
- Entropy and probability distribution analysis of model outputs.

**Weight and Activation Analysis** (via `ModelAnalyzer`):
- Layer-wise weight distribution statistics.
- Analysis of information flow and feature representations.

**Spurious Correlation Analysis** (Custom Metrics):
- **Generalization Gap**: `Train Accuracy - Test Accuracy`, measuring overfitting
  to spurious correlations.
- **Color Dependency Score**: Quantifies how much predictions change when image
  colors are shuffled, directly measuring reliance on the spurious feature.

Expected Outcomes and Insights
------------------------------

This experiment is designed to reveal:

1.  **Robustness to Spurious Correlations**: Which loss functions produce models
    that maintain high accuracy when the spurious correlation is removed.
2.  **Generalization vs. Shortcut Learning**: The extent to which models learn
    the intended features (digit shape) versus unintended shortcuts (color).
3.  **Information-Theoretic Benefits**: Whether the Goodhart-Aware Loss provides
    measurable advantages in preventing overfitting to spurious signals.
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable


import keras
import numpy as np

# --- Core project utilities ---
from dl_techniques.utils.logger import logger
from dl_techniques.losses.goodhart_loss import GoodhartAwareLoss
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

# --- ModelAnalyzer for comprehensive analysis ---
from dl_techniques.utils.analyzer import (
    ModelAnalyzer,
    AnalysisConfig,
    AnalysisResults,
    DataInput
)


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """Unified configuration for the Colored MNIST spurious correlation experiment."""

    # --- Dataset Configuration ---
    dataset_name: str = "colored_mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (28, 28, 3)

    # --- Spurious Correlation Parameters ---
    train_correlation_strength: float = 0.95
    test_correlation_strength: float = 0.0
    validation_correlation_strength: float = 0.5
    validation_split: float = 0.1

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    dense_units: List[int] = field(default_factory=lambda: [128])
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.5])
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_batch_norm: bool = True
    use_residual: bool = True

    # --- Training Parameters ---
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 8
    monitor_metric: str = 'val_accuracy'

    # --- Loss Functions to Evaluate ---
    loss_functions: Dict[str, Callable] = field(default_factory=lambda: {
        'CrossEntropy': lambda: keras.losses.CategoricalCrossentropy(from_logits=True),
        'LabelSmoothing': lambda: keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1, from_logits=True
        ),
        'FocalLoss': lambda: keras.losses.CategoricalFocalCrossentropy(
            gamma=2.0, from_logits=True
        ),
        'GoodhartAware': lambda: GoodhartAwareLoss(
            entropy_weight=0.15, mi_weight=0.02, from_logits=True
        ),
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "colored_mnist_spurious_correlation"
    random_seed: int = 42

    # --- Analysis Configuration ---
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_probability_distributions=True,
        analyze_activations=False,  # Can be slow, disable by default
        analyze_information_flow=False,
        calibration_bins=15,
        save_plots=True,
        plot_style='publication',
    ))


# ==============================================================================
# DATASET GENERATION
# ==============================================================================

@dataclass
class ColoredMNISTData:
    """Container for the Colored MNIST dataset splits."""
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

def colorize_mnist(
    images: np.ndarray, labels: np.ndarray, correlation: float, num_classes: int
) -> np.ndarray:
    """Colorizes MNIST images with a specified label-color correlation."""
    color_palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
    ]
    normalized_images = np.expand_dims(images.astype(np.float32) / 255.0, axis=-1)
    colored_images = np.zeros((*images.shape, 3), dtype=np.float32)

    for i, label in enumerate(labels):
        if np.random.rand() < correlation:
            color_idx = int(label)
        else:
            color_idx = np.random.randint(num_classes)
        color = np.array(color_palette[color_idx], dtype=np.float32) / 255.0
        colored_images[i] = normalized_images[i] * color
    return colored_images

def create_colored_mnist_dataset(config: ExperimentConfig) -> ColoredMNISTData:
    """Generates the full Colored MNIST dataset with specified correlations."""
    logger.info("🎨 Generating Colored MNIST dataset...")
    np.random.seed(config.random_seed)
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = keras.datasets.mnist.load_data()

    # Create validation split
    val_size = int(len(x_train_orig) * config.validation_split)
    indices = np.random.permutation(len(x_train_orig))
    x_train_split, y_train_split = x_train_orig[indices[val_size:]], y_train_orig[indices[val_size:]]
    x_val_split, y_val_split = x_train_orig[indices[:val_size]], y_train_orig[indices[:val_size]]

    # Colorize each split
    x_train = colorize_mnist(x_train_split, y_train_split, config.train_correlation_strength, config.num_classes)
    x_val = colorize_mnist(x_val_split, y_val_split, config.validation_correlation_strength, config.num_classes)
    x_test = colorize_mnist(x_test_orig, y_test_orig, config.test_correlation_strength, config.num_classes)

    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train_split, config.num_classes)
    y_val = keras.utils.to_categorical(y_val_split, config.num_classes)
    y_test = keras.utils.to_categorical(y_test_orig, config.num_classes)

    logger.info("✅ Dataset generated:")
    logger.info(f"   Train: {len(x_train)} samples, {config.train_correlation_strength:.0%} correlation")
    logger.info(f"   Val:   {len(x_val)} samples, {config.validation_correlation_strength:.0%} correlation")
    logger.info(f"   Test:  {len(x_test)} samples, {config.test_correlation_strength:.0%} correlation")

    return ColoredMNISTData(x_train, y_train, x_val, y_val, x_test, y_test)


# ==============================================================================
# MODEL ARCHITECTURE (Adapted from Experiment 1)
# ==============================================================================

def build_residual_block(
    inputs: keras.layers.Layer, filters: int, config: ExperimentConfig, block_index: int
) -> keras.layers.Layer:
    """Builds a residual block with skip connections."""
    shortcut = inputs
    x = keras.layers.Conv2D(filters, config.kernel_size, padding='same',
                            kernel_initializer=config.kernel_initializer,
                            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
                            name=f'conv{block_index}_1')(inputs)
    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters, config.kernel_size, padding='same',
                            kernel_initializer=config.kernel_initializer,
                            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
                            name=f'conv{block_index}_2')(x)
    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(filters, (1, 1), padding='same',
                                       kernel_initializer=config.kernel_initializer,
                                       kernel_regularizer=keras.regularizers.L2(config.weight_decay))(shortcut)
        if config.use_batch_norm:
            shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.Add()([x, shortcut])
    return keras.layers.Activation('relu')(x)

def build_conv_block(
    inputs: keras.layers.Layer, filters: int, config: ExperimentConfig, block_index: int
) -> keras.layers.Layer:
    """Builds a convolutional block, optionally with residual connections."""
    if config.use_residual and block_index > 0:
        x = build_residual_block(inputs, filters, config, block_index)
    else:
        x = keras.layers.Conv2D(filters, config.kernel_size, padding='same',
                                kernel_initializer=config.kernel_initializer,
                                kernel_regularizer=keras.regularizers.L2(config.weight_decay),
                                name=f'conv{block_index}')(inputs)
        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

    if block_index < len(config.conv_filters) - 1:
        x = keras.layers.MaxPooling2D(config.pool_size)(x)

    dropout_rate = config.dropout_rates[block_index] if block_index < len(config.dropout_rates) else 0.0
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)
    return x

def build_model(config: ExperimentConfig, loss_fn: Callable, name: str) -> keras.Model:
    """Builds a complete CNN model for Colored MNIST classification."""
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = keras.layers.Conv2D(config.conv_filters[0], config.kernel_size, padding='same',
                            kernel_initializer=config.kernel_initializer,
                            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
                            name='initial_conv')(inputs)
    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)

    x = keras.layers.GlobalAveragePooling2D()(x)

    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(units, kernel_initializer=config.kernel_initializer,
                               kernel_regularizer=keras.regularizers.L2(config.weight_decay),
                               name=f'dense_{j}')(x)
        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates) and config.dropout_rates[dense_dropout_idx] > 0:
            x = keras.layers.Dropout(config.dropout_rates[dense_dropout_idx])(x)

    logits = keras.layers.Dense(config.num_classes, kernel_initializer=config.kernel_initializer,
                                kernel_regularizer=keras.regularizers.L2(config.weight_decay),
                                name='logits')(x)
    model = keras.Model(inputs=inputs, outputs=logits, name=f'{name}_model')
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model


# ==============================================================================
# EXPERIMENT-SPECIFIC ANALYSIS
# ==============================================================================

def compute_color_dependency_score(model: keras.Model, x_test: np.ndarray, n_shuffles: int = 3) -> float:
    """Computes how much a model's predictions change when colors are shuffled."""
    original_preds = np.argmax(model.predict(x_test, verbose=0), axis=1)
    changes = []
    for _ in range(n_shuffles):
        shuffled_images = x_test.copy()
        for i in range(len(shuffled_images)):
            shuffled_images[i] = shuffled_images[i][:, :, np.random.permutation(3)]
        shuffled_preds = np.argmax(model.predict(shuffled_images, verbose=0), axis=1)
        changes.append(np.mean(original_preds != shuffled_preds))
    return float(np.mean(changes))

def analyze_robustness(
    models: Dict[str, keras.Model],
    data: ColoredMNISTData
) -> Dict[str, Dict[str, float]]:
    """Computes experiment-specific robustness metrics."""
    logger.info("🛡️ Computing spurious correlation robustness metrics...")
    robustness_results = {}
    for name, model in models.items():
        logger.info(f"   Analyzing robustness for {name}...")
        train_loss, train_acc = model.evaluate(data.x_train, data.y_train, verbose=0, batch_size=512)
        test_loss, test_acc = model.evaluate(data.x_test, data.y_test, verbose=0, batch_size=512)
        color_dependency = compute_color_dependency_score(model, data.x_test)
        robustness_results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'generalization_gap': train_acc - test_acc,
            'color_dependency': color_dependency,
        }
    logger.info("✅ Robustness analysis completed.")
    return robustness_results

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Runs the complete Colored MNIST spurious correlation experiment."""
    keras.utils.set_random_seed(config.random_seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    vis_manager = VisualizationManager(output_dir=experiment_dir / "visualizations", timestamp_dirs=False)

    logger.info("🚀 Starting Colored MNIST Spurious Correlation Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    dataset = create_colored_mnist_dataset(config)

    # ===== MODEL TRAINING PHASE =====
    logger.info("🏋️ Starting model training phase...")
    models, all_histories = {}, {}
    for loss_name, loss_fn_factory in config.loss_functions.items():
        logger.info(f"--- Training model with {loss_name} loss ---")
        model = build_model(config, loss_fn_factory(), loss_name)
        training_config = TrainingConfig(
            epochs=config.epochs, batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric, model_name=loss_name,
            output_dir=experiment_dir / "training_plots" / loss_name
        )
        history = train_model(model, dataset.x_train, dataset.y_train,
                              dataset.x_val, dataset.y_val, training_config)
        models[loss_name] = model
        all_histories[loss_name] = history.history

    # ===== MODEL ANALYSIS PREPARATION =====
    logger.info("🛠️ Creating analysis models with softmax outputs...")
    analysis_models = {}
    for name, trained_model in models.items():
        logits_output = trained_model.output
        probs_output = keras.layers.Activation('softmax', name='predictions')(logits_output)
        analysis_model = keras.Model(inputs=trained_model.input, outputs=probs_output, name=f"{name}_prediction")
        analysis_model.set_weights(trained_model.get_weights())
        analysis_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        analysis_models[name] = analysis_model

    gc.collect()

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("📊 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None
    try:
        analyzer = ModelAnalyzer(models=analysis_models, config=config.analyzer_config,
                                 output_dir=experiment_dir / "model_analysis")
        model_analysis_results = analyzer.analyze(data=DataInput.from_object(dataset))
        logger.info("✅ Model analysis completed successfully!")
    except Exception as e:
        logger.error(f"❌ Model analysis failed: {e}", exc_info=True)

    # ===== SPURIOUS CORRELATION ANALYSIS =====
    robustness_results = analyze_robustness(analysis_models, dataset)

    # ===== VISUALIZATION GENERATION =====
    logger.info("🖼️ Generating training history and confusion matrix plots...")
    vis_manager.plot_history(all_histories, ['accuracy', 'loss'], 'training_comparison',
                             title='Loss Functions Training & Validation Comparison')

    class_predictions = {name: np.argmax(model.predict(dataset.x_test, verbose=0), axis=1)
                         for name, model in analysis_models.items()}
    vis_manager.plot_confusion_matrices_comparison(
        y_true=np.argmax(dataset.y_test, axis=1),
        model_predictions=class_predictions,
        name='loss_function_confusion_matrices',
        subdir='model_comparison', normalize=True,
        class_names=[str(i) for i in range(config.num_classes)]
    )

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("📈 Evaluating final model performance on test set...")
    performance_results = {name: dict(zip(model.metrics_names,
                                          model.evaluate(dataset.x_test, dataset.y_test, verbose=0)))
                           for name, model in analysis_models.items()}

    # ===== RESULTS COMPILATION =====
    results_payload = {
        'config': config,
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'robustness_analysis': robustness_results,
        'histories': all_histories
    }
    print_experiment_summary(results_payload)
    return results_payload


# ==============================================================================
# RESULTS REPORTING
# ==============================================================================

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Prints a comprehensive summary of the spurious correlation experiment results."""
    logger.info("=" * 80)
    logger.info("📋 SPURIOUS CORRELATION EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # --- Performance Metrics ---
    if results.get('performance_analysis'):
        logger.info("🎯 PERFORMANCE METRICS (on Test Set with 0% Color Correlation):")
        logger.info(f"{'Model':<20} {'Accuracy':<12} {'Loss':<12}")
        logger.info("-" * 45)
        for name, metrics in results['performance_analysis'].items():
            logger.info(f"{name:<20} {metrics.get('accuracy', 0.0):<12.4f} {metrics.get('loss', 0.0):<12.4f}")

    # --- Robustness Metrics ---
    if results.get('robustness_analysis'):
        logger.info("\n🛡️ ROBUSTNESS METRICS:")
        logger.info(f"{'Model':<20} {'Test Acc':<12} {'Gen Gap':<12} {'Color Dep':<12}")
        logger.info("-" * 58)
        for name, metrics in results['robustness_analysis'].items():
            logger.info(f"{name:<20} {metrics.get('test_accuracy', 0.0):<12.4f} "
                        f"{metrics.get('generalization_gap', 0.0):<12.4f} "
                        f"{metrics.get('color_dependency', 0.0):<12.4f}")

    # --- Calibration Metrics ---
    analysis = results.get('model_analysis')
    if analysis and analysis.calibration_metrics:
        logger.info("\n🎯 CALIBRATION METRICS (from Model Analyzer):")
        logger.info(f"{'Model':<20} {'ECE':<12} {'Brier Score':<15}")
        logger.info("-" * 49)
        for name, metrics in analysis.calibration_metrics.items():
            logger.info(f"{name:<20} {metrics.get('ece', 0.0):<12.4f} {metrics.get('brier_score', 0.0):<15.4f}")

    # --- Final Verdict ---
    if results.get('robustness_analysis'):
        logger.info("\n" + "=" * 80)
        logger.info("🔍 EXPERIMENTAL VERDICT:")
        rob_res = results['robustness_analysis']
        best_model = max(rob_res, key=lambda k: rob_res[k]['test_accuracy'])
        most_robust = min(rob_res, key=lambda k: rob_res[k]['color_dependency'])
        logger.info(f"🏆 Highest Test Accuracy:      {best_model} ({rob_res[best_model]['test_accuracy']:.4f})")
        logger.info(f"🛡️ Lowest Color Dependency:    {most_robust} ({rob_res[most_robust]['color_dependency']:.4f})")

        if 'GoodhartAware' in rob_res and 'CrossEntropy' in rob_res:
            gal_acc = rob_res['GoodhartAware']['test_accuracy']
            ce_acc = rob_res['CrossEntropy']['test_accuracy']
            acc_diff = gal_acc - ce_acc
            logger.info(f"   GoodhartAware vs. CrossEntropy Accuracy: {acc_diff:+.4f}")
            if acc_diff > 0.02:
                logger.info("   VERDICT: 🎉 Strong support for Goodhart-Aware loss effectiveness.")
            elif acc_diff > 0:
                logger.info("   VERDICT: ✅ Positive result for Goodhart-Aware loss.")
            else:
                logger.info("   VERDICT: ➖ Neutral or negative result for this configuration.")

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Main execution function for running the experiment."""
    logger.info("🚀 Colored MNIST Spurious Correlation Experiment")
    logger.info("=" * 80)
    config = ExperimentConfig()

    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Train/Val/Test Correlation: {config.train_correlation_strength:.0%} / "
                f"{config.validation_correlation_strength:.0%} / {config.test_correlation_strength:.0%}")
    logger.info(f"   Loss Functions: {list(config.loss_functions.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info("")

    try:
        run_experiment(config)
        logger.info("✅ Experiment completed successfully!")
    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()