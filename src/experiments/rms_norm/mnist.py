"""
MNIST CNN Normalization Comparison
================================

This module implements a configurable experiment for comparing different
normalization techniques in CNN models for MNIST digit classification.

Features:
- Multiple normalization types (RMS, Logit)
- Configurable model architectures
- Comprehensive visualization
- Detailed analysis tools

Organization:
1. Imports and type definitions
2. Configuration classes
3. Model building utilities
4. Analysis utilities
5. Experiment runner
6. Main execution
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
from dl_techniques.layers.rms_logit_norm import RMSNorm, LogitNorm
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_mnist
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig
from dl_techniques.utils.weight_analyzer import WeightAnalyzerConfig, WeightAnalyzer, WeightAnalysisReport


# ------------------------------------------------------------------------------
# 2. Configuration Classes
# ------------------------------------------------------------------------------

@dataclass
class WeightAnalysisConfig:
    """Configuration for weight distribution analysis.

    Attributes:
        compute_l1_norm: Whether to compute L1 norms
        compute_l2_norm: Whether to compute L2 norms
        compute_rms_norm: Whether to compute RMS norms
        analyze_biases: Whether to analyze bias terms
        save_plots: Whether to save analysis plots
        plot_format: Format for saving plots
        plot_style: Style for matplotlib plots
    """
    compute_l1_norm: bool = True
    compute_l2_norm: bool = True
    compute_rms_norm: bool = True
    analyze_biases: bool = True
    save_plots: bool = True
    plot_format: str = 'png'
    plot_style: str = 'default'  # Using default style for better compatibility


@dataclass
class ModelConfig:
    """Configuration for model architecture and training.

    Attributes:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of classification categories
        conv_filters: Number of filters in each convolutional layer
        dense_units: Number of units in each dense layer
        dropout_rates: Dropout rate for each layer (conv and dense)
        kernel_size: Size of convolutional kernels
        pool_size: Size of max pooling windows
        weight_decay: L2 regularization factor
        norm_type: Type of normalization to apply ('rms', 'logit', or None)
        gaussian_noise_std: Standard deviation for Gaussian noise
        temperature: Temperature parameter for LogitNorm
    """
    input_shape: Tuple[int, ...] = (28, 28, 1)
    num_classes: int = 10
    conv_filters: List[int] = (16, 32, 64, 128)
    dense_units: List[int] = (32,)
    dropout_rates: List[float] = (0.25, 0.25, 0.25, 0.25, 0.5)
    kernel_size: Union[int, Tuple[int, int]] = (3, 3)
    pool_size: Union[int, Tuple[int, int]] = (2, 2)
    weight_decay: float = 0.01
    norm_type: Optional[Literal[None, 'rms', 'logit']] = None
    gaussian_noise_std: float = 0.1
    temperature: float = 0.04


@dataclass
class ExperimentConfig:
    """Configuration for the overall experiment.

    Attributes:
        model_configs: Dictionary mapping model names to their configurations
        training: Configuration for model training parameters
        visualization: Configuration for visualization settings
        weight_analysis: Configuration for weight analysis
        output_dir: Directory for saving experiment outputs
        experiment_name: Name of the experiment for logging and saving
    """
    model_configs: Dict[str, ModelConfig]
    training: TrainingConfig
    visualization: VisualizationConfig
    weight_analysis: WeightAnalysisConfig
    output_dir: Path
    experiment_name: str


# ------------------------------------------------------------------------------
# 3. Model Building Utilities
# ------------------------------------------------------------------------------
def build_conv_block(
        inputs: keras.layers.Layer,
        filters: int,
        config: ModelConfig,
        block_index: int
) -> keras.layers.Layer:
    """Build a convolutional block with normalization and activation.

    Args:
        inputs: Input tensor
        filters: Number of filters for conv layer
        config: Model configuration
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
        config: ModelConfig,
        dropout_rate: float
) -> keras.layers.Layer:
    """Build a dense block with normalization and activation.

    Args:
        inputs: Input tensor
        units: Number of dense units
        config: Model configuration
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


def build_model(config: ModelConfig) -> keras.Model:
    """Build complete CNN model with specified configuration.

    Args:
        config: Model configuration

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
    if config.norm_type == 'rms':
        x = RMSNorm()(x)
    elif config.norm_type == 'logit':
        x = LogitNorm(temperature=config.temperature)(x)

    outputs = keras.layers.Activation('softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ------------------------------------------------------------------------------
# 4. Experiment Runner
# ------------------------------------------------------------------------------
def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete normalization comparison experiment with weight analysis.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary containing experiment results including weight analysis
    """
    # Setup directories and managers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=config.visualization
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
    for name, model_config in config.model_configs.items():
        logger.info(f"\nTraining {name} model...")
        model = build_model(model_config)
        model.summary(print_fn=logger.info)

        # Configure training for this model
        model_training_config = config.training
        model_training_config.model_name = name
        model_training_config.output_dir = experiment_dir / name

        # Train the model
        history = train_model(
            model,
            mnist_data.x_train,
            mnist_data.y_train,
            mnist_data.x_test,
            mnist_data.y_test,
            model_training_config
        )

        # Store the model
        models[name] = model

        # Save model in .keras format
        model.save(experiment_dir / name / f"{name}_model.keras")

        # Structure the history data
        for metric in ['accuracy', 'loss']:
            all_histories[metric][name] = history.history[metric]
            all_histories[f'val_{metric}'][name] = history.history[f'val_{metric}']

    # Weight Analysis phase
    logger.info("\nPerforming weight distribution analysis...")

    # Configure weight analyzer
    analysis_config = WeightAnalyzerConfig(
        compute_l1_norm=config.weight_analysis.compute_l1_norm,
        compute_l2_norm=config.weight_analysis.compute_l2_norm,
        compute_rms_norm=config.weight_analysis.compute_rms_norm,
        analyze_biases=config.weight_analysis.analyze_biases,
        save_plots=config.weight_analysis.save_plots,
        export_format=config.weight_analysis.plot_format,
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
    weight_analyzer.plot_norm_distributions(save_prefix="norm_distributions")
    weight_analyzer.plot_weight_distributions(save_prefix="weight_distributions")
    weight_analyzer.plot_layer_comparisons(save_prefix="layer_comparisons")

    # Compute statistical tests
    statistical_results = weight_analyzer.compute_statistical_tests()

    # Generate comprehensive analysis report
    report_generator = WeightAnalysisReport(
        analyzer=weight_analyzer,
        output_file=str(experiment_dir / "weight_analysis" / "analysis_report.pdf")
    )
    report_generator.generate_report()

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
        'performance_analysis': performance_results,
        'weight_analysis': {
            'statistical_tests': statistical_results,
            'analyzer': weight_analyzer
        }
    }


# ------------------------------------------------------------------------------
# 5. Main Execution
# ------------------------------------------------------------------------------

def main():
    """Main execution function for running the experiment."""
    # Define model configurations
    base_model_config = ModelConfig()
    model_configs = {
        'baseline': base_model_config,
        'rms': ModelConfig(norm_type='rms'),
        'logit': ModelConfig(norm_type='logit')
    }

    # Define training configuration
    training_config = TrainingConfig(
        epochs=2,
        batch_size=128,
        early_stopping_patience=5,
        monitor_metric='val_accuracy'
    )

    # Define weight analysis configuration
    weight_analysis_config = WeightAnalysisConfig(
        compute_l1_norm=True,
        compute_l2_norm=True,
        compute_rms_norm=True,
        analyze_biases=True,
        save_plots=True,
        plot_format='png',
        plot_style='default'  # Using default matplotlib style
    )

    # Create experiment configuration
    experiment_config = ExperimentConfig(
        model_configs=model_configs,
        training=training_config,
        visualization=VisualizationConfig(),
        weight_analysis=weight_analysis_config,
        output_dir=Path("experiments"),
        experiment_name="mnist_normalization_with_weight_analysis"
    )

    # Run experiment and print results
    results = run_experiment(experiment_config)

    logger.info("\nExperiment Results:")

    # Print performance metrics
    logger.info("\nPerformance Metrics:")
    for model_name, metrics in results['performance_analysis'].items():
        logger.info(f"\n{model_name} Model:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    # Print weight analysis statistical results
    logger.info("\nWeight Analysis Statistical Tests:")
    for test_name, result in results['weight_analysis']['statistical_tests'].items():
        logger.info(f"\n{test_name}:")
        logger.info(f"  Statistic: {result['statistic']:.4f}")
        logger.info(f"  p-value: {result['p_value']:.4f}")

# ------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
