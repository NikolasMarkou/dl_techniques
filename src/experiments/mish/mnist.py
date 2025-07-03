"""
MNIST Activation Function Comparison: Evaluating Neural Network Activations
==========================================================================

This experiment conducts a comprehensive comparison of different activation functions
for image classification on MNIST, systematically evaluating their effectiveness
in convolutional neural networks.

The study addresses fundamental questions in deep learning: how do different
activation functions affect model performance, training dynamics, weight distributions,
and feature representations? By comparing traditional activations (ReLU, Tanh) with
modern alternatives (GELU, Mish, SaturatedMish), this experiment provides insights
into the trade-offs between computational efficiency, training stability, and
model expressiveness.

Experimental Design
-------------------

**Dataset**: MNIST (10 classes, 28×28 grayscale images)
- 60,000 training images
- 10,000 test images
- Standard preprocessing with normalization

**Model Architecture**: ResNet-inspired CNN with the following components:
- Initial convolutional layer (32 filters)
- 3 convolutional blocks with optional residual connections
- Progressive filter scaling: [32, 64, 128]
- Batch normalization and dropout regularization
- Global average pooling
- Dense classification layer with L2 regularization
- Softmax output layer for probability predictions
- Configurable architecture parameters for systematic studies

**Activation Functions Evaluated**:

1. **ReLU**: The baseline rectified linear unit - simple and efficient
2. **Tanh**: Hyperbolic tangent - smooth, bounded activation
3. **GELU**: Gaussian Error Linear Unit - smooth approximation of ReLU
4. **Mish**: Self-regularized non-monotonic activation function
5. **SaturatedMish (α=1.0)**: Mish with saturation control parameter
6. **SaturatedMish (α=2.0)**: Mish with stronger saturation control

Comprehensive Analysis Pipeline
------------------------------

The experiment employs a multi-faceted analysis approach:

**Training Analysis**:
- Training and validation curves for all activation functions
- Convergence behavior and stability metrics
- Early stopping based on validation accuracy
- Learning dynamics comparison

**Model Performance Evaluation**:
- Test set accuracy and top-k accuracy
- Loss values and convergence characteristics
- Statistical significance testing across runs
- Per-class performance analysis

**Calibration Analysis** (via ModelAnalyzer):
- Expected Calibration Error (ECE) with configurable binning
- Brier score for probabilistic prediction quality
- Reliability diagrams and calibration plots
- Confidence histogram analysis

**Weight and Activation Analysis**:
- Layer-wise weight distribution statistics
- Weight norm analysis (L1, L2, spectral norms)
- Activation pattern analysis across the network
- Dead neuron detection and sparsity metrics
- Gradient flow characteristics

**Information Flow Analysis**:
- Feature representation quality metrics
- Layer-wise information bottleneck analysis
- Activation entropy and mutual information
- Receptive field analysis

**Visual Analysis**:
- Training history comparison plots
- Confusion matrices for each activation function
- Weight distribution visualizations
- Activation heatmaps and statistics
- Gradient flow diagrams

Configuration and Customization
-------------------------------

The experiment is highly configurable through the ``ExperimentConfig`` class:

**Architecture Parameters**:
- ``conv_filters``: Filter counts for convolutional layers
- ``dense_units``: Hidden unit counts for dense layers
- ``dropout_rates``: Dropout probabilities per layer
- ``weight_decay``: L2 regularization strength
- ``use_residual``: Enable residual connections

**Training Parameters**:
- ``epochs``: Number of training epochs
- ``batch_size``: Training batch size
- ``learning_rate``: Adam optimizer learning rate
- ``early_stopping_patience``: Patience for early stopping

**Activation Function Parameters**:
- Easily extensible activation function dictionary
- Support for custom activation implementations
- Configurable parameters for parameterized activations

**Analysis Parameters**:
- ``calibration_bins``: Number of bins for calibration analysis
- Output directory structure and naming
- Visualization and plotting options

Expected Outcomes and Insights
------------------------------

This experiment is designed to reveal:

1. **Activation Function Trade-offs**: How different activations balance
   expressiveness with training stability and computational efficiency

2. **Training Dynamics**: Which activation functions lead to faster convergence,
   better gradient flow, and more stable training

3. **Weight Distribution Effects**: How activation choice influences weight
   distributions, sparsity, and effective network capacity

4. **Feature Quality**: The impact of activation functions on learned feature
   representations and their discriminative power

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
        epochs=30,
        batch_size=256,
        learning_rate=0.0005,
        output_dir=Path("custom_results"),
        calibration_bins=20,
        use_residual=True,
        # Add custom activation functions
        activations={
            'relu': keras.activations.relu,
            'custom_mish': lambda x: mish(x, beta=0.5)
        }
    )

    results = run_experiment(config)
    ```

Theoretical Foundation
----------------------

This experiment is grounded in several key theoretical frameworks:

**Activation Function Theory**: The comparison addresses fundamental properties
of activation functions including:
- Smoothness and differentiability (important for gradient flow)
- Bounded vs unbounded outputs (affecting gradient stability)
- Monotonicity and self-gating properties (Mish, GELU)
- Computational complexity and hardware efficiency

**Deep Learning Optimization**: The analysis framework evaluates how activation
choices affect optimization landscapes, including:
- Gradient flow and vanishing/exploding gradient problems
- Loss surface smoothness and convexity
- Critical points and saddle point escape
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import gc
import keras
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable

from dl_techniques.utils.logger import logger
from dl_techniques.layers.activations.mish import mish
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_mnist
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

from dl_techniques.utils.analyzer import (
    ModelAnalyzer,
    AnalysisConfig,
    DataInput
)

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration for the MNIST activation function comparison experiment.

    This class encapsulates all configurable parameters for the experiment,
    including dataset configuration, model architecture parameters, training
    settings, activation function definitions, and analysis configuration.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (28, 28, 1)

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])  # Filter counts for each conv block
    dense_units: List[int] = field(default_factory=lambda: [64])  # Hidden units in dense layers
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.4])  # Dropout per layer
    kernel_size: Tuple[int, int] = (3, 3)  # Convolution kernel size
    pool_size: Tuple[int, int] = (2, 2)  # Max pooling window size
    weight_decay: float = 1e-4  # L2 regularization strength
    kernel_initializer: str = 'he_normal'  # Weight initialization scheme
    use_batch_norm: bool = True  # Enable batch normalization
    use_residual: bool = True  # Enable residual connections

    # --- Training Parameters ---
    epochs: int = 100  # Number of training epochs
    batch_size: int = 128  # Training batch size
    learning_rate: float = 0.001  # Adam optimizer learning rate
    early_stopping_patience: int = 10  # Patience for early stopping
    monitor_metric: str = 'val_accuracy'  # Metric to monitor for early stopping

    # --- Activation Functions to Evaluate ---
    activations: Dict[str, Callable] = field(default_factory=lambda: {
        'Mish': lambda: mish,
        'ReLU': lambda: keras.activations.relu,
        'Tanh': lambda: keras.activations.tanh,
        'GELU': lambda: keras.activations.gelu,
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")  # Output directory for results
    experiment_name: str = "mnist_activation_comparison"  # Experiment name
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
    activation_fn: Callable,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a residual block with skip connections.

    This function creates a residual block consisting of two convolutional layers
    with batch normalization and the specified activation function, plus a skip
    connection that bypasses the block. If the input and output dimensions don't
    match, a 1x1 convolution is used to adjust the skip connection.

    Args:
        inputs: Input tensor to the residual block
        filters: Number of filters in the convolutional layers
        activation_fn: Activation function to use
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

    This function creates either a standard convolutional block or a residual
    block based on the configuration. It includes optional max pooling and
    dropout regularization.

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


def build_model(config: ExperimentConfig, activation_fn: Callable, name: str) -> keras.Model:
    """
    Build a complete CNN model for MNIST classification with specified activation.

    This function constructs a ResNet-inspired CNN with configurable architecture
    parameters. The model includes convolutional blocks, global average pooling,
    dense classification layers, and a final softmax layer for probability output.
    The specified activation function is used throughout the network.

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
        kernel_size=(5, 5),  # Larger kernel for initial feature extraction
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
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )

    return model


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete MNIST activation function comparison experiment.

    This function orchestrates the entire experimental pipeline, including:
    1. Dataset loading and preprocessing
    2. Model training for each activation function
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
    logger.info("🚀 Starting MNIST Activation Function Comparison Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("📊 Loading MNIST dataset...")
    mnist_data = load_and_preprocess_mnist()
    logger.info("✅ Dataset loaded successfully")

    # ===== MODEL TRAINING PHASE =====
    logger.info("🏋️ Starting model training phase...")
    trained_models = {}  # Store trained models
    all_histories = {}  # Store training histories

    for activation_name, activation_fn_factory in config.activations.items():
        logger.info(f"--- Training model with {activation_name} activation ---")

        # Build model for this activation function
        model = build_model(config, activation_fn_factory(), activation_name)

        # Log model architecture info
        logger.info(f"Model {activation_name} parameters: {model.count_params():,}")
        logger.info(f"Model {activation_name} metrics: {model.metrics_names}")

        # Configure training parameters
        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric,
            model_name=activation_name,
            output_dir=experiment_dir / "training_plots" / activation_name
        )

        # Train the model
        history = train_model(
            model, mnist_data.x_train, mnist_data.y_train,
            mnist_data.x_test, mnist_data.y_test, training_config
        )

        # Store results
        trained_models[activation_name] = model
        all_histories[activation_name] = history.history
        logger.info(f"✅ {activation_name} training completed!")

    # ===== MEMORY MANAGEMENT =====
    logger.info("🗑️ Triggering garbage collection...")
    gc.collect()

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("📊 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None

    try:
        # Initialize the model analyzer with trained models
        analyzer = ModelAnalyzer(
            models=trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )

        # Run comprehensive analysis
        model_analysis_results = analyzer.analyze(data=DataInput.from_object(mnist_data))
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
        title='Activation Functions Training & Validation Comparison'
    )

    # Generate confusion matrices for model comparison
    raw_predictions = {
        name: model.predict(mnist_data.x_test, verbose=0)
        for name, model in trained_models.items()
    }
    class_predictions = {
        name: np.argmax(preds, axis=1)
        for name, preds in raw_predictions.items()
    }

    vis_manager.plot_confusion_matrices_comparison(
        y_true=mnist_data.y_test,
        model_predictions=class_predictions,
        name='activation_function_confusion_matrices',
        subdir='model_comparison',
        normalize=True,
        class_names=[str(i) for i in range(10)]
    )

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("📈 Evaluating final model performance on test set...")

    performance_results = {}

    for name, model in trained_models.items():
        logger.info(f"Evaluating model {name}...")

        # Get model evaluation metrics
        eval_results = model.evaluate(mnist_data.x_test, mnist_data.y_test, verbose=0)
        metrics_dict = dict(zip(model.metrics_names, eval_results))

        # Store standardized metrics
        performance_results[name] = {
            'accuracy': metrics_dict.get('accuracy', 0.0),
            'top_5_accuracy': metrics_dict.get('top_5_accuracy', 0.0),
            'loss': metrics_dict.get('loss', 0.0)
        }

        # Log final metrics for this model
        logger.info(f"Model {name} final metrics: {performance_results[name]}")

    # ===== RESULTS COMPILATION =====
    results_payload = {
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'histories': all_histories,
        'trained_models': trained_models  # Include trained models in results
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
            top5_acc = metrics.get('top_5_accuracy', 0.0)
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
    Main execution function for running the MNIST activation comparison experiment.

    This function serves as the entry point for the experiment, handling
    configuration setup, experiment execution, and error handling.
    """
    logger.info("🚀 MNIST Activation Function Comparison Experiment")
    logger.info("=" * 80)

    # Initialize experiment configuration
    config = ExperimentConfig()

    # Log key configuration parameters
    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Activation Functions: {list(config.activations.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"   Model Architecture: {len(config.conv_filters)} conv blocks, "
                f"{len(config.dense_units)} dense layers")
    logger.info(f"   Residual Connections: {config.use_residual}")
    logger.info(f"   Output: Softmax probabilities")
    logger.info("")

    try:
        # Run the complete experiment
        _ = run_experiment(config)
        logger.info("✅ Experiment completed successfully!")

    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()