"""
CIFAR-10 Loss Function Comparison: Evaluating Goodhart-Aware Training
=====================================================================

This experiment conducts a comprehensive comparison of different loss functions
for image classification on CIFAR-10, with particular emphasis on evaluating
the effectiveness of the GoodhartAwareLoss against traditional approaches.

The study addresses a fundamental question in deep learning: how do different
loss formulations affect model robustness, calibration, and generalization?
By comparing standard cross-entropy with more sophisticated loss functions,
this experiment provides insights into the trade-offs between accuracy,
confidence calibration, and resistance to overfitting.

Experimental Design
-------------------

**Dataset**: CIFAR-10 (10 classes, 32×32 RGB images)
- 50,000 training images
- 10,000 test images
- Standard preprocessing with normalization

**Model Architecture**: ResNet-inspired CNN with the following components:
- Initial convolutional layer (32 filters)
- 4 convolutional blocks with residual connections
- Progressive filter scaling: [32, 64, 128, 256]
- Batch normalization and dropout regularization
- Global average pooling
- Dense classification layers with L2 regularization
- Configurable architecture parameters for systematic studies

**Loss Functions Evaluated**:

1. **Standard Cross-Entropy**: The baseline approach for multi-class classification
2. **Label Smoothing**: Cross-entropy with soft targets (α=0.1) to reduce overconfidence
3. **Focal Loss**: Addresses class imbalance by down-weighting easy examples (γ=2.0)
4. **Goodhart-Aware Loss**: Information-theoretic approach combining:
   - Cross-entropy for task accuracy
   - Entropy regularization to maintain prediction uncertainty
   - Mutual information regularization to compress irrelevant features

Comprehensive Analysis Pipeline
------------------------------

The experiment employs a multi-faceted analysis approach:

**Training Analysis**:
- Training and validation curves for all loss functions
- Convergence behavior and stability metrics
- Early stopping based on validation accuracy

**Model Performance Evaluation**:
- Test set accuracy and top-k accuracy
- Loss values and convergence characteristics
- Statistical significance testing across runs

**Calibration Analysis** (via ModelAnalyzer):
- Expected Calibration Error (ECE) with configurable binning
- Brier score for probabilistic prediction quality
- Reliability diagrams and calibration plots
- Confidence histogram analysis

**Weight and Activation Analysis**:
- Layer-wise weight distribution statistics
- Activation pattern analysis across the network
- Information flow characteristics
- Feature representation quality

**Probability Distribution Analysis**:
- Output probability distribution characteristics
- Entropy analysis of predictions
- Uncertainty quantification metrics

**Visual Analysis**:
- Training history comparison plots
- Confusion matrices for each loss function
- Calibration and reliability diagrams
- Weight distribution visualizations

Configuration and Customization
-------------------------------

The experiment is highly configurable through the ``ExperimentConfig`` class:

**Architecture Parameters**:
- ``conv_filters``: Filter counts for convolutional layers
- ``dense_units``: Hidden unit counts for dense layers
- ``dropout_rates``: Dropout probabilities per layer
- ``weight_decay``: L2 regularization strength

**Training Parameters**:
- ``epochs``: Number of training epochs
- ``batch_size``: Training batch size
- ``learning_rate``: Adam optimizer learning rate
- ``early_stopping_patience``: Patience for early stopping

**Loss Function Parameters**:
- Easily extensible loss function dictionary
- Configurable hyperparameters for each loss
- Support for custom loss implementations

**Analysis Parameters**:
- ``calibration_bins``: Number of bins for calibration analysis
- Output directory structure and naming
- Visualization and plotting options

Expected Outcomes and Insights
------------------------------

This experiment is designed to reveal:

1. **Accuracy vs. Calibration Trade-offs**: How different loss functions balance
   task performance with prediction reliability

2. **Robustness Characteristics**: Which approaches produce more robust models
   that generalize better to unseen data

3. **Information-Theoretic Benefits**: Whether the Goodhart-Aware Loss's
   information bottleneck principle provides measurable advantages

4. **Training Dynamics**: How different loss formulations affect convergence
   speed, stability, and final performance

Usage Example
-------------

Basic usage with default configuration:

    ```python
    from pathlib import Path

    # Run with default settings
    config = ExperimentConfig()
    results = run_experiment(config)

    # Access results
    performance = results['performance_analysis']
    calibration = results['model_analysis'].calibration_metrics
    ```

Advanced usage with custom configuration:

    ```python
    # Custom configuration
    config = ExperimentConfig(
        epochs=50,
        batch_size=128,
        learning_rate=0.0001,
        output_dir=Path("custom_results"),
        calibration_bins=20,
        # Add custom loss functions
        loss_functions={
            'CrossEntropy': lambda: keras.losses.CategoricalCrossentropy(from_logits=True),
            'CustomGoodhart': lambda: GoodhartAwareLoss(
                entropy_weight=0.05,
                mi_weight=0.005,
                from_logits=True
            )
        }
    )

    results = run_experiment(config)
    ```

Output Structure
----------------

The experiment generates comprehensive outputs in the specified directory:

    ```
    results/
    ├── cifar10_loss_comparison_analyzer_YYYYMMDD_HHMMSS/
    │   ├── model_analysis/              # ModelAnalyzer outputs
    │   │   ├── calibration_analysis/    # ECE, Brier scores, reliability diagrams
    │   │   ├── weight_analysis/         # Weight distribution analysis
    │   │   ├── activation_analysis/     # Layer activation patterns
    │   │   └── probability_analysis/    # Output distribution analysis
    │   ├── training_plots/              # Individual model training curves
    │   │   ├── CrossEntropy/
    │   │   ├── LabelSmoothing/
    │   │   ├── FocalLoss/
    │   │   └── GoodhartAware/
    │   └── visualizations/              # Comparative analysis plots
    │       ├── training_comparison.png  # Side-by-side training curves
    │       └── loss_function_confusion_matrices.png
    ```

Theoretical Foundation
----------------------

This experiment is grounded in several key theoretical frameworks:

**Information Theory**: The Goodhart-Aware Loss leverages information-theoretic
principles to balance task performance with model robustness, drawing from:
- Information Bottleneck Principle (Tishby et al.)
- Entropy regularization for calibration (Pereyra et al.)
- Mutual information constraints for generalization

**Statistical Learning Theory**: The comparison addresses fundamental questions
about the bias-variance trade-off and how different loss formulations affect
generalization bounds.

**Calibration Theory**: The analysis framework evaluates how well predicted
probabilities reflect true confidence, crucial for reliable decision-making
in real-world applications.
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
        # Create new model with softmax output
        logits_output = trained_model.output
        probs_output = keras.layers.Activation('softmax', name='predictions')(logits_output)
        analysis_model = keras.Model(inputs=trained_model.input, outputs=probs_output, name=f"{name}_prediction")

        # FIXED: Transfer weights from trained model to analysis model
        analysis_model.set_weights(trained_model.get_weights())

        # Compile with appropriate metrics
        analysis_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        analysis_models[name] = analysis_model

    # --- NEW: Free up memory (but keep models for performance evaluation) ---
    logger.info("🗑️  Releasing original training models from memory...")
    # Don't delete models yet - we need them for verification
    import gc
    gc.collect()  # Ask the garbage collector to run
    # --- END NEW ---

    # === MODEL ANALYZER BLOCK ===
    logger.info("📊 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None
    try:
        analyzer_config = AnalysisConfig(
            analyze_weights=True,
            analyze_calibration=True,
            analyze_probability_distributions=True,
            analyze_activations=True,
            analyze_information_flow=True,
            calibration_bins=config.calibration_bins,
            save_plots=True,
            plot_style='publication',
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

    # Debug: Check data format
    logger.info(f"Test data shape: {cifar10_data.x_test.shape}, {cifar10_data.y_test.shape}")
    logger.info(f"Test labels sample: {cifar10_data.y_test[:5]}")

    performance_results = {}
    for name, model in analysis_models.items():
        eval_results = model.evaluate(cifar10_data.x_test, cifar10_data.y_test, verbose=0)

        # FIXED: Handle different metric naming conventions
        metrics_dict = dict(zip(model.metrics_names, eval_results))

        # Debug logging to see what metrics are actually returned
        logger.info(f"Raw metrics for {name}: {metrics_dict}")

        # Map common metric name variations to standard names
        standardized_metrics = {}
        for metric_name, value in metrics_dict.items():
            if metric_name == 'accuracy':
                standardized_metrics['accuracy'] = value
            elif 'top' in metric_name.lower():
                standardized_metrics['top_k_categorical_accuracy'] = value
            elif metric_name == 'compile_metrics':  # Handle edge case
                standardized_metrics['accuracy'] = value
            else:
                standardized_metrics[metric_name] = value

        performance_results[name] = standardized_metrics

        # Ensure top_k_categorical_accuracy exists even if not found
        if 'top_k_categorical_accuracy' not in standardized_metrics:
            standardized_metrics['top_k_categorical_accuracy'] = 0.0
            logger.warning(f"Top-K accuracy not found for {name}, setting to 0.0")

        # FIXED: Add debug logging to verify weights were transferred correctly
        logger.info(f"Model {name} final metrics: {standardized_metrics}")

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