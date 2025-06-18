"""
MNIST CNN Activation Function Comparison Experiment
================================================

This module implements a comprehensive experiment to evaluate and compare the effectiveness
of different activation functions applied to Convolutional Neural Networks (CNNs) for
MNIST digit classification. The experiment systematically compares multiple activation
function variants with comprehensive analysis tools.

EXPERIMENT OVERVIEW
------------------
The experiment trains multiple CNN models with identical architectures but different
activation functions applied throughout the network:

1. **ReLU**: Standard rectified linear unit activation
2. **Tanh**: Hyperbolic tangent activation
3. **GELU**: Gaussian Error Linear Unit activation
4. **Mish**: Mish activation function
5. **SaturatedMish (α=1.0)**: Mish with saturation parameter α=1.0
6. **SaturatedMish (α=2.0)**: Mish with saturation parameter α=2.0

METHODOLOGY
-----------
Each model follows the same training protocol:
- Architecture: 3 convolutional blocks (16, 32, 64 filters) with batch normalization
- Dense layer: Single hidden layer with 32 units
- Regularization: Dropout layers and L2 weight decay
- Training: Adam optimizer with early stopping based on validation accuracy
- Evaluation: Comprehensive performance and weight distribution analysis

MODEL ARCHITECTURE
------------------
- Input: 28x28x1 grayscale MNIST images
- Conv Blocks: Conv2D → BatchNorm → [ACTIVATION] → Dropout
- Dense Block: Dense → [ACTIVATION] → Dropout
- Output: Dense(10) → Softmax
- Regularization: L2 weight decay and dropout

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
- Bias term analysis for all activation variants

**Activation Analysis:**
- Activation pattern statistics (mean, std, sparsity, kurtosis, skewness)
- Activation distribution comparisons
- Layer-wise activation behavior analysis

**Model Performance Analysis:**
- Confusion matrices for each activation function
- Classification accuracy, precision, recall, F1-scores
- Model comparison visualizations
- Performance metric summaries

**Visualization Outputs:**
- Training history plots (accuracy/loss over epochs)
- Weight norm distribution comparisons
- Activation statistics comparison plots
- Confusion matrix heatmaps
- Weight histogram distributions

CONFIGURATION
------------
All experiment parameters are centralized in the ExperimentConfig class:
- Model architecture parameters (filters, units, dropout rates)
- Training hyperparameters (epochs, batch size, learning rate)
- Activation function configurations
- Analysis options (which metrics to compute, plot formats)
- Output directory and experiment naming

USAGE
-----
To run with default settings:
    python mnist_activation_analysis_experiment.py

To customize the experiment, modify the ExperimentConfig class parameters:
    config = ExperimentConfig()
    config.epochs = 20
    config.batch_size = 256
    config.conv_filters = [32, 64, 128]
    results = run_experiment(config)

DEPENDENCIES
-----------
- TensorFlow/Keras for deep learning models
- NumPy for numerical computations
- Custom dl_techniques package for:
  - Mish and SaturatedMish activation layers
  - Training utilities and model analysis
  - Visualization and weight analysis tools
  - MNIST data preprocessing utilities

OUTPUT STRUCTURE
---------------
results/
├── mnist_activation_analysis_TIMESTAMP/
│   ├── relu/             # ReLU model outputs
│   ├── tanh/             # Tanh model outputs
│   ├── gelu/             # GELU model outputs
│   ├── mish/             # Mish model outputs
│   ├── saturated_mish_1/ # SaturatedMish α=1.0 outputs
│   ├── saturated_mish_2/ # SaturatedMish α=2.0 outputs
│   ├── visualizations/   # Training plots and comparisons
│   ├── weight_analysis/  # Weight distribution analysis plots
│   └── activation_analysis/ # Activation pattern analysis

RESEARCH APPLICATIONS
--------------------
This experiment framework is designed for:
- Comparing activation function effectiveness
- Analyzing training stability and convergence
- Understanding weight distribution patterns
- Evaluating activation behavior and sparsity
- Benchmarking custom activation functions

The modular design allows easy extension to additional activation functions,
different datasets, or alternative model architectures while maintaining
comprehensive analysis capabilities.

Organization:
1. Imports and type definitions
2. Single configuration class
3. Model building utilities
4. Activation analysis utilities
5. Experiment runner
6. Main execution
"""

# ------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------

from pathlib import Path
from functools import partial
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Tuple

import keras
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from keras.api.activations import gelu, relu, tanh

# ------------------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.mish import mish, saturated_mish
from dl_techniques.utils.model_analyzer import ModelAnalyzer
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.weight_analyzer import WeightAnalyzerConfig, WeightAnalyzer
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig
from dl_techniques.utils.datasets import load_and_preprocess_mnist

# ------------------------------------------------------------------------------
# 2. Single Configuration Class
# ------------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Unified configuration for the MNIST activation function experiment.

    Contains all parameters for model architecture, training, visualization,
    weight analysis, and activation analysis in a single consolidated configuration class.
    """
    # Model Architecture Parameters
    input_shape: Tuple[int, ...] = (28, 28, 1)
    num_classes: int = 10
    conv_filters: List[int] = (16, 32, 64)
    dense_units: List[int] = (32,)
    dropout_rates: List[float] = (0.25, 0.25, 0.25, 0.25)
    kernel_size: Tuple[int, int] = (5, 5)
    pool_size: Tuple[int, int] = (2, 2)
    weight_decay: float = 0.0001
    kernel_initializer: str = 'he_normal'

    # Training Parameters
    epochs: int = 10
    batch_size: int = 128
    early_stopping_patience: int = 10
    monitor_metric: str = 'val_accuracy'
    learning_rate: float = 0.001
    validation_split: float = 0.1

    # Activation Functions to Test
    activations: Dict[str, Callable] = field(default_factory=lambda: {
        'relu': relu,
        'tanh': tanh,
        'gelu': gelu,
        'mish': mish,
        'saturated_mish_1': partial(saturated_mish, alpha=1.0),
        'saturated_mish_2': partial(saturated_mish, alpha=2.0)
    })

    # Weight Analysis Parameters
    compute_l1_norm: bool = True
    compute_l2_norm: bool = True
    compute_rms_norm: bool = True
    analyze_biases: bool = True
    save_plots: bool = True
    plot_format: str = 'png'
    plot_style: str = 'default'

    # Activation Analysis Parameters
    analyze_activations: bool = True
    activation_sample_size: int = 1000
    activation_metrics: List[str] = field(default_factory=lambda: [
        'mean', 'std', 'sparsity', 'kurtosis', 'skewness'
    ])

    # Experiment Parameters
    output_dir: Path = Path("results")
    experiment_name: str = "mnist_activation_analysis"

# ------------------------------------------------------------------------------
# 3. Model Building Utilities
# ------------------------------------------------------------------------------

def build_conv_block(
        inputs: keras.layers.Layer,
        filters: int,
        config: ExperimentConfig,
        activation: Callable,
        block_index: int
) -> keras.layers.Layer:
    """Build a convolutional block with specified activation function.

    Args:
        inputs: Input tensor
        filters: Number of filters for conv layer
        config: Unified configuration
        activation: Activation function to use
        block_index: Index of the block for naming and dropout rate

    Returns:
        Output tensor after applying conv block
    """
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.kernel_size,
        padding='same',
        strides=config.pool_size,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index + 1}'
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = activation(x)
    dropout_rate = config.dropout_rates[block_index]
    if dropout_rate > 0.0:
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


def build_dense_block(
        inputs: keras.layers.Layer,
        units: int,
        config: ExperimentConfig,
        activation: Callable,
        dropout_rate: float
) -> keras.layers.Layer:
    """Build a dense block with specified activation function.

    Args:
        inputs: Input tensor
        units: Number of dense units
        config: Unified configuration
        activation: Activation function to use
        dropout_rate: Dropout rate to apply

    Returns:
        Output tensor after applying dense block
    """
    x = keras.layers.Dense(
        units=units,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay)
    )(inputs)
    x = activation(x)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


def build_model(config: ExperimentConfig, activation: Callable, name: str) -> keras.Model:
    """Build complete CNN model with specified activation function.

    Args:
        config: Unified configuration
        activation: Activation function to use throughout the model
        name: Name identifier for the model

    Returns:
        Compiled Keras model
    """
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, activation, i)

    # Dense layers
    x = keras.layers.Flatten()(x)
    for units in config.dense_units:
        x = build_dense_block(x, units, config, activation, config.dropout_rates[-1])

    # Output layer
    outputs = keras.layers.Dense(
        config.num_classes,
        activation='softmax',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'{name}_output'
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{name}_model')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ------------------------------------------------------------------------------
# 4. Activation Analysis Utilities
# ------------------------------------------------------------------------------

class ActivationAnalyzer:
    """Analyzer for activation function behavior and statistics."""

    def __init__(self, models: Dict[str, keras.Model], config: ExperimentConfig):
        """Initialize the activation analyzer.

        Args:
            models: Dictionary of trained models
            config: Experiment configuration
        """
        self.models = models
        self.config = config
        self.activation_stats: Dict[str, Dict[str, float]] = {}

    def analyze_activations(self, test_data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Analyze activation patterns for all models.

        Args:
            test_data: Test data for activation analysis

        Returns:
            Dictionary containing activation statistics for each model
        """
        logger.info("Analyzing activation patterns...")

        for name, model in self.models.items():
            logger.info(f"Analyzing activations for {name} model...")

            # Find the last convolutional layer for activation analysis
            conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
            if not conv_layers:
                logger.warning(f"No convolutional layers found in {name} model")
                continue

            last_conv_layer = conv_layers[-1]

            # Create intermediate model to extract activations
            activation_model = keras.Model(
                inputs=model.input,
                outputs=last_conv_layer.output
            )

            # Get activations
            sample_data = test_data[:self.config.activation_sample_size]
            activations = activation_model.predict(sample_data, verbose=0)

            # Compute statistics
            self.activation_stats[name] = {
                'mean': float(np.mean(activations)),
                'std': float(np.std(activations)),
                'sparsity': float(np.mean(np.abs(activations) < 1e-5)),
                'kurtosis': float(stats.kurtosis(activations.flatten())),
                'skewness': float(stats.skew(activations.flatten())),
                'max': float(np.max(activations)),
                'min': float(np.min(activations)),
                'non_zero_fraction': float(np.mean(activations > 0))
            }

            logger.info(f"Completed activation analysis for {name}")

        return self.activation_stats

    def plot_activation_statistics(self, output_dir: Path) -> None:
        """Plot activation statistics comparison.

        Args:
            output_dir: Directory to save plots
        """
        if not self.activation_stats:
            logger.warning("No activation statistics to plot")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot each metric
        for metric in self.config.activation_metrics:
            if metric not in ['mean', 'std', 'sparsity', 'kurtosis', 'skewness']:
                continue

            plt.figure(figsize=(10, 6))

            models = list(self.activation_stats.keys())
            values = [self.activation_stats[model][metric] for model in models]

            bars = plt.bar(models, values)
            plt.title(f'Activation {metric.capitalize()} Comparison')
            plt.ylabel(metric.capitalize())
            plt.xticks(rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{value:.4f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(output_dir / f'activation_{metric}_comparison.{self.config.plot_format}')
            plt.close()

        logger.info(f"Saved activation statistics plots to {output_dir}")

    def save_activation_analysis(self, output_dir: Path) -> None:
        """Save activation analysis results to file.

        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save statistics as text file
        with open(output_dir / 'activation_statistics.txt', 'w') as f:
            f.write("Activation Function Statistics\n")
            f.write("=" * 50 + "\n\n")

            for model_name, stats in self.activation_stats.items():
                f.write(f"{model_name.upper()} ACTIVATION:\n")
                f.write("-" * 30 + "\n")
                for metric, value in stats.items():
                    f.write(f"{metric}: {value:.6f}\n")
                f.write("\n")

        logger.info(f"Saved activation analysis to {output_dir}")

# ------------------------------------------------------------------------------
# 5. Experiment Runner
# ------------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete activation function comparison experiment with comprehensive analysis.

    Args:
        config: Unified experiment configuration

    Returns:
        Dictionary containing experiment results including all analyses
    """
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

    # Training phase
    for name, activation in config.activations.items():
        logger.info(f"Training {name} model...")
        model = build_model(config, activation, name)
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

        # Store the model and history
        models[name] = model
        for metric in ['accuracy', 'loss']:
            all_histories[metric][name] = history.history[metric]
            all_histories[f'val_{metric}'][name] = history.history[f'val_{metric}']

    # Weight Analysis phase
    logger.info("Performing weight distribution analysis...")

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

    # Activation Analysis phase
    if config.analyze_activations:
        logger.info("Performing activation pattern analysis...")

        try:
            activation_analyzer = ActivationAnalyzer(models, config)
            activation_stats = activation_analyzer.analyze_activations(mnist_data.x_test)

            # Plot and save activation analysis
            activation_dir = experiment_dir / "activation_analysis"
            activation_analyzer.plot_activation_statistics(activation_dir)
            activation_analyzer.save_activation_analysis(activation_dir)

            logger.info("✅ Activation analysis completed successfully!")

        except Exception as e:
            logger.error(f"Activation analysis failed: {e}")
            activation_stats = {}

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
            title=f'Activation Functions {metric.capitalize()} Comparison'
        )

    # Generate predictions for confusion matrices
    model_predictions = {}
    for name, model in models.items():
        predictions = model.predict(mnist_data.x_test, verbose=0)
        model_predictions[name] = predictions

    # Plot confusion matrices comparison
    vis_manager.plot_confusion_matrices_comparison(
        y_true=np.argmax(mnist_data.y_test, axis=1),
        model_predictions=model_predictions,
        name='activation_confusion_matrices',
        subdir='model_comparison',
        normalize=True,
        class_names=[str(i) for i in range(10)]
    )

    # Analyze model performance
    analyzer = ModelAnalyzer(models, vis_manager)
    performance_results = analyzer.analyze_models(mnist_data)

    # Combine all results
    results = {
        'models': models,
        'histories': all_histories,
        'performance_analysis': performance_results,
        'experiment_config': config
    }

    if config.analyze_activations:
        results['activation_analysis'] = activation_stats

    return results


# ------------------------------------------------------------------------------
# 6. Main Execution
# ------------------------------------------------------------------------------

def main() -> None:
    """Main execution function for running the activation function experiment."""
    logger.info("Starting MNIST activation function comparison experiment with comprehensive analysis")

    # Create unified configuration
    config = ExperimentConfig()

    # Run experiment
    results = run_experiment(config)

    # Print comprehensive results
    logger.info("="*80)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("="*80)

    # Print activation function performance comparison
    logger.info("Model Performance Comparison:")
    logger.info("-" * 50)
    for model_name, metrics in results['performance_analysis'].items():
        logger.info(f"{model_name.upper()} ACTIVATION:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")

    # Print activation analysis if available
    if 'activation_analysis' in results:
        logger.info("Activation Pattern Analysis:")
        logger.info("-" * 50)
        for model_name, stats in results['activation_analysis'].items():
            logger.info(f"{model_name.upper()} ACTIVATIONS:")
            for metric, value in stats.items():
                logger.info(f"  {metric}: {value:.4f}")

    # Print final training metrics
    logger.info("Final Training Metrics:")
    logger.info("-" * 50)
    for model_name in results['models'].keys():
        final_val_acc = results['histories']['val_accuracy'][model_name][-1]
        final_val_loss = results['histories']['val_loss'][model_name][-1]
        logger.info(f"{model_name}: Val Acc = {final_val_acc:.4f}, Val Loss = {final_val_loss:.4f}")

    logger.info("="*80)
    logger.info("Experiment completed successfully! Check results directory for detailed analysis.")
    logger.info("="*80)


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()