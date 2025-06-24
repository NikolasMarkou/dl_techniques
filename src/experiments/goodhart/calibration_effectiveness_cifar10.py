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

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable
import gc

import keras
import numpy as np

# --- Core project utilities ---
from dl_techniques.utils.logger import logger
from dl_techniques.losses.goodhart_loss import GoodhartAwareLoss
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_cifar10
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
    """
    Configuration for the CIFAR-10 loss comparison experiment.

    This class encapsulates all configurable parameters for the experiment,
    including dataset configuration, model architecture parameters, training
    settings, loss function definitions, and analysis configuration.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = (32, 64, 128, 256)  # Filter counts for each conv block
    dense_units: List[int] = (128,)  # Hidden units in dense layers
    dropout_rates: List[float] = (0.25, 0.25, 0.25, 0.25, 0.25)  # Dropout per layer
    kernel_size: Tuple[int, int] = (3, 3)  # Convolution kernel size
    pool_size: Tuple[int, int] = (2, 2)  # Max pooling window size
    weight_decay: float = 1e-4  # L2 regularization strength
    kernel_initializer: str = 'he_normal'  # Weight initialization scheme
    use_batch_norm: bool = True  # Enable batch normalization
    use_residual: bool = True  # Enable residual connections

    # --- Training Parameters ---
    epochs: int = 50  # Number of training epochs
    batch_size: int = 64  # Training batch size
    learning_rate: float = 0.001  # Adam optimizer learning rate
    early_stopping_patience: int = 15  # Patience for early stopping
    monitor_metric: str = 'val_accuracy'  # Metric to monitor for early stopping

    # --- Loss Functions to Evaluate ---
    loss_functions: Dict[str, Callable] = field(default_factory=lambda: {
        'CrossEntropy': lambda: keras.losses.CategoricalCrossentropy(
            from_logits=True),
        'LabelSmoothing': lambda: keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1, from_logits=True
        ),
        'FocalLoss': lambda: keras.losses.CategoricalFocalCrossentropy(
            gamma=2.0, from_logits=True
        ),
        'GoodhartAware': lambda: GoodhartAwareLoss(
            entropy_weight=0.1, mi_weight=0.01, from_logits=True
        ),
        'GoodhartAware_ls': lambda: GoodhartAwareLoss(
            entropy_weight=0.1, mi_weight=0.01, from_logits=True, label_smoothing=0.1
        ),
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")  # Output directory for results
    experiment_name: str = "cifar10_loss_comparison_analyzer"  # Experiment name
    random_seed: int = 42  # Random seed for reproducibility

    # --- Analysis Configuration ---
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,  # Analyze weight distributions
        analyze_calibration=True,  # Analyze model calibration
        analyze_activations=True,  # Analyze activation patterns
        analyze_information_flow=True,  # Analyze information flow
        calibration_bins=15,  # Number of bins for calibration analysis
        save_plots=True,  # Save analysis plots
        plot_style='publication',  # Publication-ready plot style
    ))


# ==============================================================================
# MODEL ARCHITECTURE BUILDING UTILITIES
# ==============================================================================

def build_residual_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a residual block with skip connections.

    This function creates a residual block consisting of two convolutional layers
    with batch normalization and ReLU activation, plus a skip connection that
    bypasses the block. If the input and output dimensions don't match, a 1x1
    convolution is used to adjust the skip connection.

    Args:
        inputs: Input tensor to the residual block
        filters: Number of filters in the convolutional layers
        config: Experiment configuration containing architecture parameters
        block_index: Index of the current block (for naming layers)

    Returns:
        Output tensor after applying the residual block
    """
    # Store the original input for the skip connection
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
    x = keras.layers.Activation('relu')(x)

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
    x = keras.layers.Activation('relu')(x)

    return x


def build_conv_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a convolutional block with optional residual connections.

    This function creates either a standard convolutional block or a residual
    block based on the configuration. It includes optional max pooling and
    dropout regularization.

    Args:
        inputs: Input tensor to the convolutional block
        filters: Number of filters in the convolutional layers
        config: Experiment configuration containing architecture parameters
        block_index: Index of the current block (for naming and logic)

    Returns:
        Output tensor after applying the convolutional block
    """
    # Use residual connections for blocks after the first one (if enabled)
    if config.use_residual and block_index > 0:
        x = build_residual_block(inputs, filters, config, block_index)
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
        x = keras.layers.Activation('relu')(x)

    # Apply max pooling (except for the last convolutional block)
    if block_index < len(config.conv_filters) - 1:
        x = keras.layers.MaxPooling2D(config.pool_size)(x)

    # Apply dropout if specified for this layer
    dropout_rate = (config.dropout_rates[block_index]
                   if block_index < len(config.dropout_rates) else 0.0)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def build_model(config: ExperimentConfig, loss_fn: Callable, name: str) -> keras.Model:
    """
    Build a complete CNN model for CIFAR-10 classification.

    This function constructs a ResNet-inspired CNN with configurable architecture
    parameters. The model includes convolutional blocks, global average pooling,
    and dense classification layers.

    Args:
        config: Experiment configuration containing model architecture parameters
        loss_fn: Loss function to use for training
        name: Name prefix for the model and its layers

    Returns:
        Compiled Keras model ready for training
    """
    # Define input layer
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Initial convolutional layer
    x = keras.layers.Conv2D(
        config.conv_filters[0],
        config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='initial_conv'
    )(x)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Stack of convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)

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
        x = keras.layers.Activation('relu')(x)

        # Apply dropout if specified for dense layers
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates):
            dropout_rate = config.dropout_rates[dense_dropout_idx]
            if dropout_rate > 0:
                x = keras.layers.Dropout(dropout_rate)(x)

    # Final classification layer (logits output)
    logits = keras.layers.Dense(
        units=config.num_classes,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='logits'
    )(x)

    # Create and compile the model
    model = keras.Model(inputs=inputs, outputs=logits, name=f'{name}_model')
    optimizer = keras.optimizers.AdamW(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    return model


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete CIFAR-10 loss function comparison experiment.

    This function orchestrates the entire experimental pipeline, including:
    1. Dataset loading and preprocessing
    2. Model training for each loss function
    3. Model analysis and evaluation
    4. Visualization generation
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
    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=VisualizationConfig(),
        timestamp_dirs=False
    )

    # Log experiment start
    logger.info("🚀 Starting CIFAR-10 Loss Comparison Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("📊 Loading CIFAR-10 dataset...")
    cifar10_data = load_and_preprocess_cifar10()
    logger.info("✅ Dataset loaded successfully")

    # ===== MODEL TRAINING PHASE =====
    logger.info("🏋️ Starting model training phase...")
    models = {}  # Store trained models
    all_histories = {}  # Store training histories

    for loss_name, loss_fn_factory in config.loss_functions.items():
        logger.info(f"--- Training model with {loss_name} loss ---")

        # Build model for this loss function
        model = build_model(config, loss_fn_factory(), loss_name)

        # Configure training parameters
        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric,
            model_name=loss_name,
            output_dir=experiment_dir / "training_plots" / loss_name
        )

        # Train the model
        history = train_model(
            model, cifar10_data.x_train, cifar10_data.y_train,
            cifar10_data.x_test, cifar10_data.y_test, training_config
        )

        # Store results
        models[loss_name] = model
        all_histories[loss_name] = history.history
        logger.info(f"✅ {loss_name} training completed!")

    # ===== MODEL ANALYSIS PREPARATION =====
    logger.info("🛠️ Creating analysis models with softmax outputs...")
    analysis_models = {}

    for name, trained_model in models.items():
        logger.info(f"Preparing analysis model for {name}...")
        logger.info(f"Original model metrics: {trained_model.metrics_names}")

        # Create new model with softmax output for probability analysis
        logits_output = trained_model.output
        probs_output = keras.layers.Activation('softmax', name='predictions')(logits_output)
        analysis_model = keras.Model(
            inputs=trained_model.input,
            outputs=probs_output,
            name=f"{name}_prediction"
        )

        # Transfer weights from trained model to analysis model
        logger.info(f"Transferring weights for {name}...")
        analysis_model.set_weights(trained_model.get_weights())
        logger.info(f"Weight transfer completed for {name}")

        # Test model before compilation to verify weights are correct
        logger.info(f"Testing model {name} before compilation...")
        test_sample = cifar10_data.x_test[:1]  # Single sample for testing
        test_pred_before = analysis_model.predict(test_sample, verbose=0)
        logger.info(f"Model {name} prediction shape: {test_pred_before.shape}")
        logger.info(f"Model {name} prediction sample: {test_pred_before[0][:3]}")

        # Compile the analysis model with appropriate metrics
        try:
            analysis_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=[
                    keras.metrics.CategoricalAccuracy(name='accuracy'),
                    keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
                ]
            )
            logger.info(f"Model {name} compiled successfully with explicit metrics")
        except Exception as e:
            logger.warning(f"Explicit metric compilation failed for {name}: {e}")
            # Fallback to simple compilation
            analysis_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info(f"Model {name} compiled with fallback metrics")

        # Verify compilation worked correctly
        logger.info(f"Model {name} metric names after compilation: {analysis_model.metrics_names}")

        # Test after compilation to ensure consistency
        test_pred_after = analysis_model.predict(test_sample, verbose=0)
        logger.info(f"Model {name} prediction after compilation: {test_pred_after[0][:3]}")

        analysis_models[name] = analysis_model

    # ===== MEMORY MANAGEMENT =====
    logger.info("🗑️ Releasing original training models from memory...")
    # Keep models for now but trigger garbage collection
    gc.collect()

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("📊 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None

    try:
        # Initialize the model analyzer
        analyzer = ModelAnalyzer(
            models=analysis_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )

        # Run comprehensive analysis
        model_analysis_results = analyzer.analyze(data=DataInput.from_object(cifar10_data))
        logger.info("✅ Model analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Model analysis failed: {e}", exc_info=True)

    # ===== VISUALIZATION GENERATION =====
    logger.info("🖼️ Generating training history and confusion matrix plots...")

    # Plot training history comparison
    vis_manager.plot_history(
        histories=all_histories,
        metrics=['accuracy', 'loss'],
        name='training_comparison',
        subdir='training_plots',
        title='Loss Functions Training & Validation Comparison'
    )

    # Generate confusion matrices for model comparison
    raw_predictions = {
        name: model.predict(cifar10_data.x_test, verbose=0)
        for name, model in analysis_models.items()
    }
    class_predictions = {
        name: np.argmax(preds, axis=1)
        for name, preds in raw_predictions.items()
    }

    vis_manager.plot_confusion_matrices_comparison(
        y_true=cifar10_data.y_test,
        model_predictions=class_predictions,
        name='loss_function_confusion_matrices',
        subdir='model_comparison',
        normalize=True,
        class_names=[str(i) for i in range(10)]
    )

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("📈 Evaluating final model performance on test set...")

    # Debug information about test data format
    logger.info(f"Test data shape: {cifar10_data.x_test.shape}, {cifar10_data.y_test.shape}")
    logger.info(f"Test labels sample: {cifar10_data.y_test[:5]}")

    performance_results = {}

    for name, model in analysis_models.items():
        logger.info(f"Evaluating model {name}...")

        # Get model evaluation metrics
        eval_results = model.evaluate(cifar10_data.x_test, cifar10_data.y_test, verbose=0)
        metrics_dict = dict(zip(model.metrics_names, eval_results))

        # Debug logging for metric inspection
        logger.info(f"Raw metrics for {name}: {metrics_dict}")

        # Calculate manual top-5 accuracy as backup
        logger.info(f"Computing manual top-5 accuracy for {name}...")
        predictions = model.predict(cifar10_data.x_test, verbose=0)
        y_true_indices = np.argmax(cifar10_data.y_test, axis=1)

        # Manual top-5 accuracy calculation
        top_5_predictions = np.argsort(predictions, axis=1)[:, -5:]  # Get top 5 predictions
        manual_top5_acc = np.mean([
            y_true in top5_pred
            for y_true, top5_pred in zip(y_true_indices, top_5_predictions)
        ])
        logger.info(f"Manual top-5 accuracy for {name}: {manual_top5_acc:.4f}")

        # Manual top-1 accuracy for verification
        manual_top1_acc = np.mean(np.argmax(predictions, axis=1) == y_true_indices)
        logger.info(f"Manual top-1 accuracy for {name}: {manual_top1_acc:.4f}")

        # Standardize metric names for consistent reporting
        standardized_metrics = {}
        for metric_name, value in metrics_dict.items():
            if metric_name in ['accuracy', 'compile_metrics']:
                standardized_metrics['accuracy'] = value
            elif 'top' in metric_name.lower() and '5' in metric_name:
                standardized_metrics['top_k_categorical_accuracy'] = value
            elif metric_name == 'loss':
                standardized_metrics['loss'] = value
            else:
                # Keep any other metrics as-is
                standardized_metrics[metric_name] = value

        performance_results[name] = standardized_metrics

        # Use manual calculation if built-in top-5 accuracy is not available
        if ('top_k_categorical_accuracy' not in standardized_metrics or
            standardized_metrics['top_k_categorical_accuracy'] == 0.0):
            standardized_metrics['top_k_categorical_accuracy'] = manual_top5_acc
            logger.info(f"Using manual top-5 accuracy for {name}: {manual_top5_acc:.4f}")

        # Warn about potentially problematic accuracy values
        if standardized_metrics['accuracy'] < 0.2:
            logger.warning(f"Low accuracy detected for {name}: {standardized_metrics['accuracy']:.4f}")
            logger.warning(f"Manual accuracy: {manual_top1_acc:.4f}")
            logger.warning("This may indicate weight transfer issues or untrained models")

        # Log final metrics for this model
        logger.info(f"Model {name} final metrics: {standardized_metrics}")

    # ===== RESULTS COMPILATION =====
    results_payload = {
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'histories': all_histories
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

    This function generates a detailed report of all experimental outcomes,
    including performance metrics, calibration analysis, and training progress.
    The summary is formatted for clear readability and easy interpretation.

    Args:
        results: Dictionary containing all experimental results and analysis
    """
    logger.info("=" * 80)
    logger.info("📋 EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # ===== PERFORMANCE METRICS SECTION =====
    if 'performance_analysis' in results and results['performance_analysis']:
        logger.info("🎯 PERFORMANCE METRICS (on Full Test Set):")
        logger.info(f"{'Model':<20} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12}")
        logger.info("-" * 60)

        for model_name, metrics in results['performance_analysis'].items():
            accuracy = metrics.get('accuracy', 0.0)
            top5_acc = metrics.get('top_k_categorical_accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            logger.info(f"{model_name:<20} {accuracy:<12.4f} {top5_acc:<12.4f} {loss:<12.4f}")

    # ===== CALIBRATION METRICS SECTION =====
    model_analysis = results.get('model_analysis')
    if model_analysis and model_analysis.calibration_metrics:
        logger.info("🎯 CALIBRATION METRICS (from Model Analyzer):")
        logger.info(f"{'Model':<20} {'ECE':<12} {'Brier Score':<15} {'Mean Entropy':<12}")
        logger.info("-" * 65)

        for model_name, cal_metrics in model_analysis.calibration_metrics.items():
            logger.info(
                f"{model_name:<20} {cal_metrics['ece']:<12.4f} "
                f"{cal_metrics['brier_score']:<15.4f} {cal_metrics['mean_entropy']:<12.4f}"
            )

    # ===== TRAINING METRICS SECTION =====
    if 'histories' in results and results['histories']:
        # Check if any model actually has training history data
        has_training_data = False
        for model_name, history_dict in results['histories'].items():
            if (history_dict.get('val_accuracy') and len(history_dict['val_accuracy']) > 0 and
                history_dict.get('val_loss') and len(history_dict['val_loss']) > 0):
                has_training_data = True
                break

        if has_training_data:
            logger.info("🏁 FINAL TRAINING METRICS (on Validation Set):")
            logger.info(f"{'Model':<20} {'Val Accuracy':<15} {'Val Loss':<12}")
            logger.info("-" * 50)

            for model_name, history_dict in results['histories'].items():
                # Check if this specific model has actual training data
                if (history_dict.get('val_accuracy') and len(history_dict['val_accuracy']) > 0 and
                    history_dict.get('val_loss') and len(history_dict['val_loss']) > 0):
                    final_val_acc = history_dict['val_accuracy'][-1]
                    final_val_loss = history_dict['val_loss'][-1]
                    logger.info(f"{model_name:<20} {final_val_acc:<15.4f} {final_val_loss:<12.4f}")
                else:
                    logger.info(f"{model_name:<20} {'Not trained':<15} {'Not trained':<12}")
        else:
            logger.info("🏁 TRAINING STATUS:")
            logger.info("⚠️  Models were not trained (epochs=0) - no training metrics available")

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for running the CIFAR-10 loss comparison experiment.

    This function serves as the entry point for the experiment, handling
    configuration setup, experiment execution, and error handling.
    """
    logger.info("🚀 CIFAR-10 Loss Function Comparison")
    logger.info("=" * 80)

    # Initialize experiment configuration
    config = ExperimentConfig()

    # Log key configuration parameters
    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Loss Functions: {list(config.loss_functions.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"   Model Architecture: {len(config.conv_filters)} conv blocks, "
                f"{len(config.dense_units)} dense layers")
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