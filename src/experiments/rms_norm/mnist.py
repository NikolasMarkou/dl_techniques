"""
MNIST CNN Normalization Comparison Experiment
===========================================

This module implements a comprehensive experiment to evaluate and compare the effectiveness
of different normalization techniques applied to Convolutional Neural Networks (CNNs) for
MNIST digit classification. The experiment systematically compares three model variants:
baseline (no normalization), RMS normalization, and Logit normalization.

EXPERIMENT OVERVIEW
------------------
The experiment trains three CNN models with identical architectures but different
normalization strategies applied to the final layer before softmax activation:

1. **Baseline Model**: Standard CNN without additional normalization
2. **RMS Normalization**: Applies Root Mean Square normalization to stabilize outputs
3. **Logit Normalization**: Applies temperature-scaled logit normalization for calibrated predictions*
4. **Band RMS Normalization**: Root Mean Square Normalization with Bounded Spherical Shell Constraints*

METHODOLOGY
-----------
Each model follows the same training protocol:
- Architecture: 4 convolutional blocks (16, 32, 64, 128 filters) with batch normalization
- Dense layer: Single hidden layer with 32 units
- Regularization: Dropout layers and L2 weight decay
- Training: Adam optimizer with early stopping based on validation accuracy
- Evaluation: Comprehensive performance and weight distribution analysis

MODEL ARCHITECTURE
------------------
- Input: 28x28x1 grayscale MNIST images
- Conv Blocks: Conv2D → BatchNorm → ReLU → MaxPool2D → Dropout
- Dense Block: Dense → BatchNorm → ReLU → Dropout
- Output: Dense(10) → [Optional Normalization] → Softmax
- Regularization: L2 weight decay (0.01) and dropout (0.25-0.5)

ANALYSIS OUTPUTS
---------------
The experiment generates comprehensive analysis across multiple dimensions:

**Training Analysis:**
- Training/validation accuracy and loss curves for all models
- Early stopping behavior and convergence patterns
- Comparative performance metrics

**Weight Distribution Analysis:**
- L1, L2, and RMS norm distributions across layers
- Layer-wise weight statistics comparison
- Weight distribution heatmaps and histograms
- Bias term analysis for all normalization variants

**Model Performance Analysis:**
- Confusion matrices for each model variant
- Classification accuracy, precision, recall, F1-scores
- Model comparison visualizations
- Performance metric summaries

**Visualization Outputs:**
- Training history plots (accuracy/loss over epochs)
- Weight norm distribution comparisons
- Layer-wise statistical analysis plots
- Confusion matrix heatmaps
- Weight histogram distributions

CONFIGURATION
------------
All experiment parameters are centralized in the Config class:
- Model architecture parameters (filters, units, dropout rates)
- Training hyperparameters (epochs, batch size, learning rate)
- Normalization settings (temperature for LogitNorm)
- Analysis options (which metrics to compute, plot formats)
- Output directory and experiment naming

USAGE
-----
To run with default settings:
    python mnist_normalization_experiment.py

To customize the experiment, modify the Config class parameters:
    config = Config()
    config.epochs = 10
    config.batch_size = 256
    config.conv_filters = [32, 64, 128, 256]
    results = run_experiment(config)

DEPENDENCIES
-----------
- TensorFlow/Keras for deep learning models
- NumPy for numerical computations
- Custom dl_techniques package for:
  - RMSNorm and LogitNorm layers
  - Training utilities and model analysis
  - Visualization and weight analysis tools
  - MNIST data preprocessing utilities

OUTPUT STRUCTURE
---------------
results/
├── mnist_normalization_with_weight_analysis_TIMESTAMP/
│   ├── baseline/          # Baseline model checkpoints and logs
│   ├── rms/              # RMS normalized model outputs
│   ├── logit/            # Logit normalized model outputs
│   ├── visualizations/   # Training plots and comparisons
│   └── weight_analysis/  # Weight distribution analysis plots

RESEARCH APPLICATIONS
--------------------
This experiment framework is designed for:
- Comparing normalization technique effectiveness
- Analyzing training stability and convergence
- Understanding weight distribution patterns
- Evaluating model calibration and confidence
- Benchmarking custom normalization methods

The modular design allows easy extension to additional normalization techniques,
different datasets, or alternative model architectures while maintaining
comprehensive analysis capabilities.

Organization:
1. Imports and type definitions
2. Single configuration class
3. Model building utilities
4. Experiment runner
5. Main execution
"""

# ------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------

import keras
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal, Tuple, Union

from dl_techniques.utils.logger import logger
from dl_techniques.utils.model_analyzer import ModelAnalyzer
from dl_techniques.layers.band_rms import BandRMS
from dl_techniques.layers.rms_logit_norm import RMSNorm, LogitNorm
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_mnist
from dl_techniques.utils.weight_analyzer import WeightAnalyzerConfig, WeightAnalyzer
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

# ------------------------------------------------------------------------------
# 2. Single Configuration Class
# ------------------------------------------------------------------------------

@dataclass
class Config:
    """Unified configuration for the MNIST normalization experiment.

    Contains all parameters for model architecture, training, visualization,
    and weight analysis in a single consolidated configuration class.
    """
    # Model Architecture Parameters
    input_shape: Tuple[int, ...] = (28, 28, 1)
    num_classes: int = 10
    conv_filters: List[int] = (16, 32, 64, 128)
    dense_units: List[int] = (32,)
    dropout_rates: List[float] = (0.25, 0.25, 0.25, 0.25, 0.5)
    kernel_size: Union[int, Tuple[int, int]] = (3, 3)
    pool_size: Union[int, Tuple[int, int]] = (2, 2)
    weight_decay: float = 0.01
    gaussian_noise_std: float = 0.1
    temperature: float = 0.04

    # Training Parameters
    epochs: int = 3
    batch_size: int = 128
    early_stopping_patience: int = 5
    monitor_metric: str = 'val_accuracy'
    learning_rate: float = 0.001

    # Weight Analysis Parameters
    compute_l1_norm: bool = True
    compute_l2_norm: bool = True
    compute_rms_norm: bool = True
    analyze_biases: bool = True
    save_plots: bool = True
    plot_format: str = 'png'
    plot_style: str = 'default'

    # Experiment Parameters
    output_dir: Path = Path("results")
    experiment_name: str = "mnist_normalization_with_weight_analysis"

    # Model variants to test
    normalization_types: List[Optional[Literal['rms', 'logit']]] = (None, 'rms', 'logit', 'band_rms')

# ------------------------------------------------------------------------------
# 3. Model Building Utilities
# ------------------------------------------------------------------------------

def build_conv_block(
        inputs: keras.layers.Layer,
        filters: int,
        config: Config,
        block_index: int
) -> keras.layers.Layer:
    """Build a convolutional block with normalization and activation.

    Args:
        inputs: Input tensor
        filters: Number of filters for conv layer
        config: Unified configuration
        block_index: Index of the block for naming and dropout rate

    Returns:
        Output tensor after applying conv block
    """
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index + 1}'
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D(pool_size=config.pool_size)(x)
    dropout_rate = config.dropout_rates[block_index]
    if dropout_rate is not None and dropout_rate > 0.0:
        x = keras.layers.Dropout(config.dropout_rates[block_index])(x)
    return x


def build_dense_block(
        inputs: keras.layers.Layer,
        units: int,
        config: Config,
        dropout_rate: float
) -> keras.layers.Layer:
    """Build a dense block with normalization and activation.

    Args:
        inputs: Input tensor
        units: Number of dense units
        config: Unified configuration
        dropout_rate: Dropout rate to apply

    Returns:
        Output tensor after applying dense block
    """
    x = keras.layers.Dense(
        units=units,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.L2(config.weight_decay)
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return keras.layers.Dropout(dropout_rate)(x)


def build_model(config: Config, norm_type: Optional[Literal['rms', 'logit', 'band_rms']] = None) -> keras.Model:
    """Build complete CNN model with specified configuration.

    Args:
        config: Unified configuration
        norm_type: Type of normalization to apply ('rms', 'logit', 'band_rms', or None)

    Returns:
        Compiled Keras model
    """
    inputs = keras.layers.Input(shape=config.input_shape)
    x = inputs

    # Convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)

    # Dense layers
    x = keras.layers.Flatten()(x)
    for units in config.dense_units:
        x = build_dense_block(x, units, config, config.dropout_rates[-1])

    # Output layer
    x = keras.layers.Dense(
        config.num_classes,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.L2(config.weight_decay)
    )(x)

    # Add normalization if specified
    if norm_type == 'rms':
        x = RMSNorm()(x)
    elif norm_type == 'logit':
        x = LogitNorm(temperature=config.temperature)(x)
    elif norm_type == 'band_rms':
        x = BandRMS()(x)

    outputs = keras.layers.Activation('softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ------------------------------------------------------------------------------
# 4. Experiment Runner
# ------------------------------------------------------------------------------

def run_experiment(config: Config) -> Dict[str, Any]:
    """Run the complete normalization comparison experiment with weight analysis.

    Args:
        config: Unified experiment configuration

    Returns:
        Dictionary containing experiment results including weight analysis
    """
    # Setup directories and managers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    vis_config = VisualizationConfig()
    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=vis_config
    )

    # Load and preprocess data
    mnist_data = load_and_preprocess_mnist()

    # Train models
    models = {}
    all_histories = {
        'accuracy': {},
        'loss': {},
        'val_accuracy': {},
        'val_loss': {}
    }

    # Create model names based on normalization types
    model_names = {
        None: 'baseline',
        'rms': 'rms',
        'logit': 'logit',
        'band_rms': 'band_rms'
    }

    # Training phase
    for norm_type in config.normalization_types:
        name = model_names[norm_type]
        logger.info(f"Training {name} model...")
        model = build_model(config, norm_type)
        model.summary(print_fn=logger.info)

        # Configure training for this model
        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric,
            model_name=name,
            output_dir=experiment_dir / name
        )

        # Train the model
        history = train_model(
            model,
            mnist_data.x_train,
            mnist_data.y_train,
            mnist_data.x_test,
            mnist_data.y_test,
            training_config
        )

        # Store the model
        models[name] = model

        # Structure the history data
        for metric in ['accuracy', 'loss']:
            all_histories[metric][name] = history.history[metric]
            all_histories[f'val_{metric}'][name] = history.history[f'val_{metric}']

    # Weight Analysis phase
    logger.info("Performing weight distribution analysis...")

    # Configure weight analyzer
    analysis_config = WeightAnalyzerConfig(
        compute_l1_norm=config.compute_l1_norm,
        compute_l2_norm=config.compute_l2_norm,
        compute_rms_norm=config.compute_rms_norm,
        analyze_biases=config.analyze_biases,
        save_plots=config.save_plots,
        export_format=config.plot_format,
        plot_style='seaborn-darkgrid',
        color_palette='deep'
    )

    # Initialize and run weight analyzer
    weight_analyzer = WeightAnalyzer(
        models=models,
        config=analysis_config,
        output_dir=experiment_dir / "weight_analysis"
    )

    # Generate standard analysis plots
    weight_analyzer.plot_norm_distributions()
    weight_analyzer.plot_layer_comparisons(['mean', 'std', 'l2_norm'])
    weight_analyzer.plot_weight_distributions_heatmap(n_bins=50)
    weight_analyzer.plot_layer_weight_histograms()

    # Generate visualizations for training history
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
            f'training_{metric}',
            title=f'Model {metric.capitalize()} Comparison'
        )

    # Generate predictions for confusion matrices
    model_predictions = {}
    for name, model in models.items():
        predictions = model.predict(mnist_data.x_test)
        model_predictions[name] = predictions

    # Plot confusion matrices comparison
    vis_manager.plot_confusion_matrices_comparison(
        y_true=np.argmax(mnist_data.y_test, axis=1),
        model_predictions=model_predictions,
        name='confusion_matrices_comparison',
        subdir='model_comparison',
        normalize=True,
        class_names=[str(i) for i in range(10)]
    )

    # Analyze model performance
    analyzer = ModelAnalyzer(models, vis_manager)
    performance_results = analyzer.analyze_models(mnist_data)

    # Combine all results
    return {
        'models': models,
        'histories': all_histories,
        'performance_analysis': performance_results
    }


# ------------------------------------------------------------------------------
# 5. Main Execution
# ------------------------------------------------------------------------------

def main():
    """Main execution function for running the experiment."""
    # Create unified configuration
    config = Config()

    # Run experiment and print results
    results = run_experiment(config)

    logger.info("Experiment Results:")

    # Print performance metrics
    logger.info("Performance Metrics:")
    for model_name, metrics in results['performance_analysis'].items():
        logger.info(f"{model_name} Model:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()