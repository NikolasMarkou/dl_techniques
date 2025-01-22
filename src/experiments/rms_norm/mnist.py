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

#------------------------------------------------------------------------------
# 1. Imports and Dependencies
#------------------------------------------------------------------------------

import keras
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from matplotlib import pyplot as plt
from typing import Dict, Any, Optional, List, Literal, Tuple, Union

from dl_techniques.utils.logger import logger
from dl_techniques.layers.rms_logit_norm import RMSNorm, LogitNorm
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_mnist, MNISTData
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig


#------------------------------------------------------------------------------
# 2. Configuration Classes
#------------------------------------------------------------------------------
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
        output_dir: Directory for saving experiment outputs
        experiment_name: Name of the experiment for logging and saving
    """
    model_configs: Dict[str, ModelConfig]
    training: TrainingConfig
    visualization: VisualizationConfig
    output_dir: Path
    experiment_name: str


#------------------------------------------------------------------------------
# 3. Model Building Utilities
#------------------------------------------------------------------------------
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
        name=f'conv{block_index+1}'
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


#------------------------------------------------------------------------------
# 4. Analysis Utilities
#------------------------------------------------------------------------------
class ModelAnalyzer:
    """Analyzer for model comparison with visualization support."""

    def __init__(
        self,
        models: Dict[str, keras.Model],
        vis_manager: VisualizationManager
    ):
        """Initialize analyzer with models and visualization manager.

        Args:
            models: Dictionary of models to analyze
            vis_manager: Visualization manager instance
        """
        self.models = models
        self.vis_manager = vis_manager

    def generate_predictions(
        self,
        image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate predictions from all models for a single image.

        Args:
            image: Input image

        Returns:
            Dictionary mapping model names to their predictions
        """
        return {
            name: model.predict(np.expand_dims(image, 0))
            for name, model in self.models.items()
        }

    def analyze_models(
        self,
        data: MNISTData,
        sample_digits: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive model analysis.

        Args:
            data: MNIST dataset splits
            sample_digits: Optional list of digits to analyze

        Returns:
            Dictionary containing analysis results
        """
        results = {}

        # Model evaluation
        for name, model in self.models.items():
            evaluation = model.evaluate(data.x_test, data.y_test)
            results[name] = dict(zip(model.metrics_names, evaluation))

        # Sample digit analysis
        if sample_digits is None:
            sample_digits = list(range(10))

        for digit in sample_digits:
            digit_indices = np.where(np.argmax(data.y_test, axis=1) == digit)[0]
            if len(digit_indices) > 0:
                digit_idx = digit_indices[0]
                digit_image = data.x_test[digit_idx]
                predictions = self.generate_predictions(digit_image)

                self.vis_manager.compare_images(
                    [digit_image] * len(self.models),
                    [f"{name} (conf: {pred.max():.2f})"
                     for name, pred in predictions.items()],
                    f"digit_{digit}_comparison"
                )

        return results


#------------------------------------------------------------------------------
# 5. Experiment Runner
#------------------------------------------------------------------------------
def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete normalization comparison experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary containing experiment results
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

        # Properly structure the history data
        for metric in ['accuracy', 'loss']:
            all_histories[metric][name] = history.history[metric]
            all_histories[f'val_{metric}'][name] = history.history[f'val_{metric}']

    # Generate visualizations with properly structured data
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

    # Analyze results
    analyzer = ModelAnalyzer(models, vis_manager)
    results = analyzer.analyze_models(mnist_data)

    return {
        'models': models,
        'histories': all_histories,
        'analysis': results
    }


def plot_history(
        self,
        histories: Dict[str, Dict[str, List[float]]],
        metrics: List[str],
        name: str,
        title: Optional[str] = None,
        subdir: Optional[str] = None
) -> Path:
    """Plot training history metrics.

    Args:
        histories: Dictionary mapping model names to their metric histories
        metrics: List of metrics to plot
        name: Base name for saving
        title: Optional overall title for the plot
        subdir: Optional subdirectory for saving

    Returns:
        Path where figure was saved
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(self.config.fig_size[0] * n_metrics, self.config.fig_size[1])
    )

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for model_name, history in histories.items():
            if metric in history:
                ax.plot(
                    history[metric],
                    label=f'{model_name} (Training)',
                    linestyle='-'
                )
            if f'val_{metric}' in history:
                ax.plot(
                    history[f'val_{metric}'],
                    label=f'{model_name} (Validation)',
                    linestyle='--'
                )

        ax.set_title(f'{metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)

    if title:
        fig.suptitle(title, fontsize=self.config.title_fontsize * 1.2)

    return self.save_figure(fig, name, subdir)


#------------------------------------------------------------------------------
# 6. Main Execution
#------------------------------------------------------------------------------
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

    # Create experiment configuration
    experiment_config = ExperimentConfig(
        model_configs=model_configs,
        training=training_config,
        visualization=VisualizationConfig(),
        output_dir=Path("experiments"),
        experiment_name="mnist_normalization"
    )

    # Run experiment and print results
    results = run_experiment(experiment_config)

    logger.info("Experiment Results:")
    for model_name, metrics in results['analysis'].items():
        logger.info(f"{model_name} Model:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()