# DL-Techniques Experiment Guidelines

## Table of Contents
1. [Overview](#overview)
2. [Experiment Structure](#experiment-structure)
3. [Documentation Standards](#documentation-standards)
4. [Configuration Pattern](#configuration-pattern)
5. [Architecture Building](#architecture-building)
6. [Training Integration](#training-integration)
7. [Analysis Integration](#analysis-integration)
8. [Results Management](#results-management)
9. [Code Quality Standards](#code-quality-standards)
10. [Example Template](#example-template)
11. [Best Practices](#best-practices)

## Overview

This guide establishes standards for creating high-quality, reproducible experiments in the DL-Techniques library. All experiments should follow these patterns to ensure consistency, maintainability, and scientific rigor.

### Key Principles

- **Reproducibility**: All experiments must be fully reproducible with proper random seed management
- **Modularity**: Code should be organized into reusable, testable components
- **Documentation**: Comprehensive documentation explaining the scientific rationale
- **Integration**: Leverage existing library utilities (ModelAnalyzer, TrainingConfig, etc.)
- **Configurability**: All parameters should be configurable through dataclass configs
- **Analysis**: Include comprehensive analysis using the ModelAnalyzer
- **Visualization**: Generate publication-ready visualizations and reports

## Experiment Structure

### Required File Organization

```python
"""
Experiment Title: Brief Description of the Scientific Question
====================================================================

Comprehensive docstring explaining:
- Scientific motivation and hypothesis
- Experimental design and methodology
- Expected outcomes and insights
- Usage examples and configuration options
- Theoretical foundation

The docstring should be substantial (200-500 lines) and serve as both
documentation and a mini research paper explaining the experiment.
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

# Standard library imports
import gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable

# Keras and core ML libraries
import keras
import numpy as np

# DL-Techniques imports (organized by category)
from dl_techniques.utils.logger import logger
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_[dataset]
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

# Experiment-specific imports
from dl_techniques.layers.[category].[specific_layer] import SpecificLayer
from dl_techniques.losses.[loss_name] import SpecificLoss

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the [Experiment Name] experiment."""
    # Configuration details...

# ==============================================================================
# MODEL ARCHITECTURE BUILDING UTILITIES
# ==============================================================================

def build_component(args) -> keras.layers.Layer:
    """Build reusable architectural components."""
    pass

def build_model(config, variant_params, name) -> keras.Model:
    """Build complete model with specified configuration."""
    pass

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete experiment pipeline."""
    pass

# ==============================================================================
# RESULTS REPORTING
# ==============================================================================

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive experiment summary."""
    pass

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Main execution function."""
    pass

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
```

## Documentation Standards

### Experiment Header Documentation

Every experiment must start with a comprehensive docstring that includes:

```python
"""
Experiment Title: Clear Description of What You're Investigating
==================================================================

Scientific motivation paragraph explaining why this experiment matters,
what fundamental questions it addresses, and how it contributes to
understanding of deep learning principles.

Experimental Design
-------------------

**Dataset**: Clear description of dataset used
- Number of samples, classes, dimensions
- Preprocessing steps applied
- Train/test split methodology

**Model Architecture**: Description of base architecture
- Key architectural components
- Parameter counts and design choices
- Rationale for architectural decisions

**Experimental Variables**: What you're comparing
1. **Variable 1**: Description and rationale
2. **Variable 2**: Description and rationale
3. **Control Variables**: What stays constant

Comprehensive Analysis Pipeline
------------------------------

**Training Analysis**:
- Training dynamics and convergence behavior
- Loss curve analysis and early stopping criteria
- Learning rate schedules and optimization details

**Model Performance Evaluation**:
- Primary metrics (accuracy, loss)
- Secondary metrics (top-k, per-class performance)
- Statistical significance testing approach

**Deep Analysis** (via ModelAnalyzer):
- Calibration analysis (ECE, Brier score, reliability)
- Weight distribution analysis
- Information flow characteristics
- Activation pattern analysis

**Visualization Pipeline**:
- Training history comparisons
- Confusion matrices and error analysis
- Calibration plots and reliability diagrams
- Weight and activation visualizations

Configuration and Customization
-------------------------------

Detailed explanation of configuration options and how to customize
the experiment for different research questions.

Expected Outcomes and Insights
------------------------------

Clear statements of what you expect to discover and how results
will be interpreted. Include potential implications for:
1. Model design decisions
2. Training methodologies
3. Theoretical understanding

Usage Example
-------------

    ```python
    # Basic usage
    config = ExperimentConfig()
    results = run_experiment(config)

    # Custom configuration
    config = ExperimentConfig(
        epochs=100,
        batch_size=64,
        # ... other custom parameters
    )
    results = run_experiment(config)
    ```

Theoretical Foundation
----------------------

Optional section for experiments with strong theoretical basis,
explaining the mathematical or conceptual framework underlying
the experimental design.
"""
```

### Function Documentation

All functions must have comprehensive docstrings:

```python
def build_residual_block(
    inputs: keras.layers.Layer,
    filters: int,
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
        config: Experiment configuration containing architecture parameters
        block_index: Index of the current block (for naming layers)

    Returns:
        Output tensor after applying the residual block

    Raises:
        ValueError: If configuration parameters are invalid

    Example:
        >>> x = keras.layers.Input(shape=(32, 32, 3))
        >>> block = build_residual_block(x, 64, config, 0)
        >>> print(block.shape)
        (None, 32, 32, 64)
    """
```

## Configuration Pattern

### Required Configuration Structure

All experiments must use a dataclass-based configuration:

```python
@dataclass
class ExperimentConfig:
    """
    Configuration for the [Experiment Name] experiment.

    This class encapsulates all configurable parameters for the experiment,
    including dataset configuration, model architecture parameters, training
    settings, experimental variables, and analysis configuration.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (28, 28, 1)

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    dense_units: List[int] = field(default_factory=lambda: [64])
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.5])
    kernel_size: Tuple[int, int] = (3, 3)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_batch_norm: bool = True
    use_residual: bool = True

    # --- Training Parameters ---
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    monitor_metric: str = 'val_accuracy'

    # --- Experimental Variables (The core of what you're testing) ---
    experimental_variants: Dict[str, Callable] = field(default_factory=lambda: {
        'Variant_A': lambda: create_variant_a(),
        'Variant_B': lambda: create_variant_b(),
        'Variant_C': lambda: create_variant_c(),
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "my_experiment"
    random_seed: int = 42

    # --- Analysis Configuration ---
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=True,
        analyze_training_dynamics=True,
        calibration_bins=15,
        save_plots=True,
        plot_style='publication',
    ))
```

### Configuration Best Practices

1. **Group Related Parameters**: Organize parameters into logical sections
2. **Provide Sensible Defaults**: All parameters should have reasonable defaults
3. **Use Type Hints**: Always specify types for all parameters
4. **Factory Functions**: Use `field(default_factory=lambda: ...)` for mutable defaults
5. **Documentation**: Document each parameter's purpose and valid ranges

## Architecture Building

### Modular Architecture Components

Break down model building into reusable components:

```python
def build_conv_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """Build a convolutional block with optional residual connections."""

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

    # Apply dropout if specified
    dropout_rate = (config.dropout_rates[block_index]
                   if block_index < len(config.dropout_rates) else 0.0)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x

def build_model(
    config: ExperimentConfig,
    variant_specific_param: Any,
    name: str
) -> keras.Model:
    """
    Build a complete model for the experiment.

    Args:
        config: Experiment configuration
        variant_specific_param: Parameter specific to this experimental variant
        name: Model name for identification

    Returns:
        Compiled Keras model ready for training
    """
    # Define input layer
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Build architecture systematically
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)

        # Add pooling except for last layer
        if i < len(config.conv_filters) - 1:
            x = keras.layers.MaxPooling2D((2, 2))(x)

    # Global pooling and dense layers
    x = keras.layers.GlobalAveragePooling2D()(x)

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

    # Apply variant-specific modifications here
    x = apply_variant_modifications(x, variant_specific_param)

    # Output layer
    outputs = keras.layers.Dense(
        units=config.num_classes,
        activation='softmax',
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
```

## Training Integration

### Use Standard Training Utilities

Always leverage the existing training infrastructure:

```python
def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete experimental pipeline."""

    # Set reproducibility
    keras.utils.set_random_seed(config.random_seed)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"📊 Loading {config.dataset_name} dataset...")
    dataset = load_dataset_function()
    logger.info("✅ Dataset loaded successfully")

    # Training phase
    logger.info("🏋️ Starting model training phase...")
    trained_models = {}
    all_histories = {}

    for variant_name, variant_factory in config.experimental_variants.items():
        logger.info(f"--- Training model with {variant_name} ---")

        # Build model for this variant
        variant_param = variant_factory()
        model = build_model(config, variant_param, variant_name)

        # Log model info
        model.summary(print_fn=logger.info)
        logger.info(f"Model {variant_name} parameters: {model.count_params():,}")

        # Configure training
        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric,
            model_name=variant_name,
            output_dir=experiment_dir / "training_plots" / variant_name
        )

        # Train the model
        history = train_model(
            model, dataset.x_train, dataset.y_train,
            dataset.x_test, dataset.y_test, training_config
        )

        # Store results
        trained_models[variant_name] = model
        all_histories[variant_name] = history.history
        logger.info(f"✅ {variant_name} training completed!")

    return {
        'trained_models': trained_models,
        'histories': all_histories,
        'dataset': dataset,
        'experiment_dir': experiment_dir
    }
```

## Analysis Integration

### Mandatory ModelAnalyzer Integration

Every experiment must include comprehensive analysis:

```python
def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    # ... training code above ...

    # ===== MEMORY MANAGEMENT =====
    logger.info("🗑️ Triggering garbage collection...")
    gc.collect()

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("📊 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None

    try:
        # Initialize the model analyzer
        analyzer = ModelAnalyzer(
            models=trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis",
            training_history=all_histories  # Include training history
        )

        # Run comprehensive analysis
        model_analysis_results = analyzer.analyze(data=DataInput.from_object(dataset))
        logger.info("✅ Model analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Model analysis failed: {e}", exc_info=True)

    # ===== VISUALIZATION GENERATION =====
    logger.info("🖼️ Generating visualizations...")

    # Initialize visualization manager
    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=VisualizationConfig(),
        timestamp_dirs=False
    )

    # Plot training history comparison
    vis_manager.plot_history(
        histories=all_histories,
        metrics=['accuracy', 'loss'],
        name='training_comparison',
        subdir='training_plots',
        title=f'{config.experiment_name} Training Comparison'
    )

    # Generate confusion matrices
    raw_predictions = {
        name: model.predict(dataset.x_test, verbose=0)
        for name, model in trained_models.items()
    }
    class_predictions = {
        name: np.argmax(preds, axis=1)
        for name, preds in raw_predictions.items()
    }

    vis_manager.plot_confusion_matrices_comparison(
        y_true=np.argmax(dataset.y_test, axis=1),
        model_predictions=class_predictions,
        name='confusion_matrices_comparison',
        subdir='model_comparison',
        normalize=True,
        class_names=[str(i) for i in range(config.num_classes)]
    )

    # ===== PERFORMANCE EVALUATION =====
    logger.info("📈 Evaluating final model performance...")
    performance_results = evaluate_models(trained_models, dataset)

    return {
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'histories': all_histories,
        'trained_models': trained_models,
        'config': config
    }
```

### Performance Evaluation Pattern

```python
def evaluate_models(models: Dict[str, keras.Model], dataset) -> Dict[str, Dict[str, float]]:
    """Evaluate all models and return standardized metrics."""
    performance_results = {}

    for name, model in models.items():
        logger.info(f"Evaluating model {name}...")

        # Get model evaluation metrics
        eval_results = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
        metrics_dict = dict(zip(model.metrics_names, eval_results))

        # Manual accuracy verification if needed
        predictions = model.predict(dataset.x_test, verbose=0)
        y_true_indices = np.argmax(dataset.y_test, axis=1)
        manual_accuracy = np.mean(np.argmax(predictions, axis=1) == y_true_indices)

        # Store standardized metrics
        performance_results[name] = {
            'accuracy': metrics_dict.get('accuracy', manual_accuracy),
            'top_5_accuracy': metrics_dict.get('top_5_accuracy', manual_accuracy),
            'loss': metrics_dict.get('loss', 0.0)
        }

        logger.info(f"Model {name} metrics: {performance_results[name]}")

    return performance_results
```

## Results Management

### Comprehensive Summary Reporting

```python
def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print a comprehensive summary of experimental results.

    This function generates a detailed report of all experimental outcomes,
    including performance metrics, calibration analysis, and training progress.
    """
    logger.info("=" * 80)
    logger.info("📋 EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # ===== PERFORMANCE METRICS SECTION =====
    if 'performance_analysis' in results:
        logger.info("🎯 PERFORMANCE METRICS (on Test Set):")
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
        logger.info("🎯 CALIBRATION METRICS:")
        logger.info(f"{'Model':<20} {'ECE':<12} {'Brier Score':<15} {'Mean Entropy':<12}")
        logger.info("-" * 65)

        for model_name, cal_metrics in model_analysis.calibration_metrics.items():
            logger.info(
                f"{model_name:<20} {cal_metrics['ece']:<12.4f} "
                f"{cal_metrics['brier_score']:<15.4f} {cal_metrics['mean_entropy']:<12.4f}"
            )

    # ===== TRAINING METRICS SECTION =====
    if model_analysis and model_analysis.training_metrics:
        logger.info("🏁 TRAINING DYNAMICS:")
        logger.info(f"{'Model':<20} {'Convergence':<12} {'Stability':<12} {'Overfitting':<12}")
        logger.info("-" * 60)

        for model_name in results.get('trained_models', {}):
            conv_epochs = model_analysis.training_metrics.epochs_to_convergence.get(model_name, 0)
            stability = model_analysis.training_metrics.training_stability_score.get(model_name, 0.0)
            overfit = model_analysis.training_metrics.overfitting_index.get(model_name, 0.0)

            logger.info(f"{model_name:<20} {conv_epochs:<12} {stability:<12.4f} {overfit:<12.4f}")

    logger.info("=" * 80)
```

### Results Persistence

```python
def save_experiment_results(results: Dict[str, Any], experiment_dir: Path) -> None:
    """Save experiment results in multiple formats."""

    # Save configuration
    config_path = experiment_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        # Convert config to JSON-serializable format
        config_dict = asdict(results['config'])
        json.dump(config_dict, f, indent=2, default=str)

    # Save performance summary
    performance_path = experiment_dir / "performance_summary.json"
    with open(performance_path, 'w') as f:
        json.dump(results['performance_analysis'], f, indent=2)

    # Save models (optional - can be large)
    models_dir = experiment_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for name, model in results['trained_models'].items():
        model_path = models_dir / f"{name}.keras"
        model.save(model_path)
        logger.info(f"Saved model {name} to {model_path}")
```

## Code Quality Standards

### Logging Standards

```python
# Use the project logger throughout
from dl_techniques.utils.logger import logger

# Log experiment phases clearly
logger.info("🚀 Starting [Experiment Name]")
logger.info("📊 Loading dataset...")
logger.info("🏋️ Training models...")
logger.info("📈 Analyzing results...")
logger.info("✅ Experiment completed!")

# Log important details
logger.info(f"Model {name} parameters: {model.count_params():,}")
logger.info(f"Training completed in {epochs} epochs")

# Log warnings and errors appropriately
logger.warning(f"Low accuracy detected: {accuracy:.4f}")
logger.error(f"Training failed: {e}", exc_info=True)
```

### Error Handling

```python
# Wrap major sections in try-catch blocks
try:
    # Model training
    history = train_model(...)
    trained_models[variant_name] = model
    all_histories[variant_name] = history.history
    logger.info(f"✅ {variant_name} training completed!")

except Exception as e:
    logger.error(f"❌ Training failed for {variant_name}: {e}", exc_info=True)
    # Continue with other variants or handle gracefully
    continue

# Analysis should be robust to failures
try:
    model_analysis_results = analyzer.analyze(data=DataInput.from_object(dataset))
    logger.info("✅ Model analysis completed!")
except Exception as e:
    logger.error(f"❌ Model analysis failed: {e}", exc_info=True)
    model_analysis_results = None  # Allow experiment to continue
```

### Memory Management

```python
# Explicit garbage collection
import gc

# After training phase
logger.info("🗑️ Triggering garbage collection...")
gc.collect()

# Clear large objects when no longer needed
if not save_models:
    del trained_models
    gc.collect()
```

### Reproducibility

```python
# Always set random seeds
keras.utils.set_random_seed(config.random_seed)

# Document the environment
logger.info(f"Keras version: {keras.__version__}")
logger.info(f"Random seed: {config.random_seed}")
logger.info(f"Configuration: {config}")
```

## Example Template

Here's a minimal template following all the guidelines:

```python
"""
Experiment Title: Template for DL-Techniques Experiments
=======================================================

Brief description of what this experiment investigates and why it matters.

Experimental Design
-------------------

**Dataset**: Description of dataset
**Model Architecture**: Description of base architecture
**Experimental Variables**: What you're comparing

Usage Example
-------------

    ```python
    config = ExperimentConfig()
    results = run_experiment(config)
    ```
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import keras
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable

from dl_techniques.utils.logger import logger
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_mnist
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the template experiment."""

    # Dataset configuration
    dataset_name: str = "mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (28, 28, 1)

    # Architecture parameters
    conv_filters: List[int] = field(default_factory=lambda: [32, 64])
    dense_units: List[int] = field(default_factory=lambda: [64])
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.5])

    # Training parameters
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001

    # Experimental variants
    experimental_variants: Dict[str, Callable] = field(default_factory=lambda: {
        'Baseline': lambda: 'relu',
        'Alternative': lambda: 'gelu',
    })

    # Experiment configuration
    output_dir: Path = Path("results")
    experiment_name: str = "template_experiment"
    random_seed: int = 42

    # Analysis configuration
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        save_plots=True,
    ))

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

def build_model(config: ExperimentConfig, activation: str, name: str) -> keras.Model:
    """Build model with specified activation."""

    inputs = keras.layers.Input(shape=config.input_shape)
    x = inputs

    # Convolutional layers
    for i, filters in enumerate(config.conv_filters):
        x = keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = keras.layers.Activation(activation)(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        if i < len(config.dropout_rates):
            x = keras.layers.Dropout(config.dropout_rates[i])(x)

    # Dense layers
    x = keras.layers.GlobalAveragePooling2D()(x)
    for units in config.dense_units:
        x = keras.layers.Dense(units)(x)
        x = keras.layers.Activation(activation)(x)

    # Output
    outputs = keras.layers.Dense(config.num_classes, activation='softmax')(x)

    # Compile
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{name}_model')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete experiment."""

    # Setup
    keras.utils.set_random_seed(config.random_seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting Template Experiment")

    # Load data
    dataset = load_and_preprocess_mnist()

    # Train models
    trained_models = {}
    all_histories = {}

    for variant_name, variant_factory in config.experimental_variants.items():
        logger.info(f"Training {variant_name}...")

        activation = variant_factory()
        model = build_model(config, activation, variant_name)

        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            model_name=variant_name,
            output_dir=experiment_dir / "training" / variant_name
        )

        history = train_model(
            model, dataset.x_train, dataset.y_train,
            dataset.x_test, dataset.y_test, training_config
        )

        trained_models[variant_name] = model
        all_histories[variant_name] = history.history

    # Analysis
    try:
        analyzer = ModelAnalyzer(
            models=trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "analysis",
            training_history=all_histories
        )
        model_analysis = analyzer.analyze(data=DataInput.from_object(dataset))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        model_analysis = None

    # Results
    results = {
        'trained_models': trained_models,
        'histories': all_histories,
        'model_analysis': model_analysis,
        'config': config
    }

    print_experiment_summary(results)
    return results

# ==============================================================================
# RESULTS REPORTING
# ==============================================================================

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print experiment summary."""
    logger.info("=" * 50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 50)

    # Add your summary logic here
    for name in results['trained_models']:
        logger.info(f"Model {name}: Trained successfully")

    logger.info("=" * 50)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Main execution function."""
    config = ExperimentConfig()
    run_experiment(config)

if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Scientific Rigor
- **Clear Hypothesis**: State what you expect to find and why
- **Control Variables**: Keep everything constant except what you're testing
- **Statistical Significance**: Use appropriate sample sizes and statistical tests
- **Reproducibility**: Document everything needed to reproduce results

### 2. Code Organization
- **Modular Design**: Break complex functions into smaller, testable components
- **Type Hints**: Use type hints throughout for better code documentation
- **Error Handling**: Gracefully handle failures without crashing the entire experiment
- **Memory Management**: Explicitly manage memory for large experiments

### 3. Documentation
- **Comprehensive Docstrings**: Explain the scientific rationale, not just the code
- **Usage Examples**: Include clear examples of how to run and configure
- **Parameter Explanation**: Document what each configuration parameter does
- **Results Interpretation**: Explain how to interpret the outputs

### 4. Integration
- **Use Library Tools**: Leverage ModelAnalyzer, TrainingConfig, and VisualizationManager
- **Standard Patterns**: Follow established patterns for consistency
- **Extensibility**: Make it easy to add new variants or configurations
- **Reusability**: Write components that can be reused in other experiments

### 5. Output Management
- **Structured Directories**: Organize outputs logically
- **Multiple Formats**: Save results in both human-readable and machine-readable formats
- **Versioning**: Include timestamps and configuration snapshots
- **Visualization**: Generate publication-ready plots and summaries

This guide ensures that all experiments in the DL-Techniques library maintain high standards of scientific rigor, code quality, and reproducibility while being easy to understand, extend, and reuse.