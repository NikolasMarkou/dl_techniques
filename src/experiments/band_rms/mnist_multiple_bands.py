import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Tuple, Dict, Any, Optional, List, Union
from dataclasses import dataclass

import keras
import tensorflow as tf
from keras import Model
from keras.api.optimizers import Adam
from keras.api.metrics import SparseCategoricalAccuracy
from keras.api.losses import SparseCategoricalCrossentropy

from dl_techniques.layers.band_rms_norm import BandRMSNorm


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for band width comparison experiment."""
    # Model architecture
    INPUT_SHAPE: Tuple[int, int, int] = (28, 28, 1)
    NUM_CLASSES: int = 10

    # CNN parameters
    CONV1_FILTERS: int = 32
    CONV2_FILTERS: int = 64
    CONV3_FILTERS: int = 128
    KERNEL_SIZE: Tuple[int, int] = (3, 3)
    ACTIVATION: str = "gelu"

    # Normalization parameters
    BAND_WIDTHS: Tuple[float, ...] = (0.01, 0.025, 0.05, 0.075, 0.1)
    EPSILON: float = 1e-6
    BAND_REGULARIZER: float = 1e-5

    # Training parameters
    BATCH_SIZE: int = 128
    EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.2
    INITIAL_LEARNING_RATE: float = 0.001
    MIN_LEARNING_RATE: float = 1e-6
    LR_REDUCTION_FACTOR: float = 0.2

    # Early stopping
    PATIENCE: int = 7
    LR_PATIENCE: int = 3

    # File paths
    MODEL_SAVE_PATH: str = "mnist_band_comparison_{band_width}.keras"


class BandAnalyzer:
    """Analyzes learned band parameters from BandRMSNorm layers.

    Args:
        model: Trained model containing BandRMSNorm layers
    """

    def __init__(self, model: Model):
        self.model = model
        self._band_layers = self._find_band_layers()

    def _find_band_layers(self) -> List[BandRMSNorm]:
        """Find all BandRMSNorm layers in the model."""
        return [layer for layer in self.model.layers
                if isinstance(layer, BandRMSNorm)]

    def get_learned_bands(self) -> Dict[str, np.ndarray]:
        """Extract learned band parameters from each BandRMSNorm layer.

        Returns:
            Dictionary mapping layer names to their learned band parameters
        """
        learned_bands = {}
        for layer in self._band_layers:
            # Get band_param weights and convert to numpy
            band_param = layer.band_param.numpy()

            # Convert band_param to actual scale factor using hard sigmoid
            scale = (1.0 - layer.max_band_width) + (
                    layer.max_band_width *
                    keras.activations.hard_sigmoid(tf.convert_to_tensor(band_param))
            ).numpy()

            learned_bands[layer.name] = scale
        return learned_bands

    def plot_band_distribution(self) -> None:
        """Plot distribution of learned band parameters."""
        learned_bands = self.get_learned_bands()

        fig, axes = plt.subplots(len(learned_bands), 1,
                                 figsize=(10, 4 * len(learned_bands)))
        if len(learned_bands) == 1:
            axes = [axes]

        for (layer_name, band_values), ax in zip(learned_bands.items(), axes):
            band_values_flat = band_values.flatten()
            sns.histplot(band_values_flat, ax=ax)
            ax.set_title(f'Band Distribution - {layer_name}')
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Count')

            # Add statistics
            ax.axvline(np.mean(band_values_flat), color='r',
                       linestyle='--', label='Mean')
            ax.axvline(np.median(band_values_flat), color='g',
                       linestyle='--', label='Median')
            ax.legend()

        plt.tight_layout()
        plt.show()


def create_band_model(
        band_width: float,
        input_shape: Tuple[int, int, int] = ExperimentConfig.INPUT_SHAPE,
        num_classes: int = ExperimentConfig.NUM_CLASSES,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None
) -> Model:
    """Create model with specified band width for BandRMSNorm layers.

    Args:
        band_width: Maximum band width for BandRMSNorm layers
        input_shape: Input image shape
        num_classes: Number of output classes
        kernel_regularizer: Regularization for kernels

    Returns:
        Configured model
    """
    if kernel_regularizer is None:
        kernel_regularizer = keras.regularizers.L2(1e-4)

    model = keras.Sequential([
        keras.layers.Conv2D(
            ExperimentConfig.CONV1_FILTERS,
            ExperimentConfig.KERNEL_SIZE,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer,
            input_shape=input_shape
        ),
        BandRMSNorm(
            max_band_width=band_width,
            epsilon=ExperimentConfig.EPSILON,
            band_regularizer=keras.regularizers.L2(ExperimentConfig.BAND_REGULARIZER),
            name=f'band_norm_1'
        ),
        keras.layers.Activation(ExperimentConfig.ACTIVATION),
        keras.layers.MaxPooling2D(),
        keras.layers.Dropout(0.1),

        keras.layers.Conv2D(
            ExperimentConfig.CONV2_FILTERS,
            ExperimentConfig.KERNEL_SIZE,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        ),
        BandRMSNorm(
            max_band_width=band_width,
            epsilon=ExperimentConfig.EPSILON,
            band_regularizer=keras.regularizers.L2(ExperimentConfig.BAND_REGULARIZER),
            name=f'band_norm_2'
        ),
        keras.layers.Activation(ExperimentConfig.ACTIVATION),
        keras.layers.MaxPooling2D(),
        keras.layers.Dropout(0.2),

        keras.layers.Flatten(),
        keras.layers.Dense(
            256,
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        ),
        BandRMSNorm(
            max_band_width=band_width,
            epsilon=ExperimentConfig.EPSILON,
            band_regularizer=keras.regularizers.L2(ExperimentConfig.BAND_REGULARIZER),
            name=f'band_norm_3'
        ),
        keras.layers.Activation(ExperimentConfig.ACTIVATION),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(
            num_classes,
            activation='softmax',
            kernel_initializer='he_normal'
        )
    ])

    return model


def create_callbacks(model_path: str) -> List[keras.callbacks.Callback]:
    """Create training callbacks."""
    return [
        keras.callbacks.ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor='val_sparse_categorical_accuracy'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            patience=ExperimentConfig.PATIENCE,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_sparse_categorical_accuracy',
            factor=ExperimentConfig.LR_REDUCTION_FACTOR,
            patience=ExperimentConfig.LR_PATIENCE,
            min_lr=ExperimentConfig.MIN_LEARNING_RATE
        )
    ]


def plot_comparison_results(
        histories: Dict[float, keras.callbacks.History],
        band_analyses: Dict[float, Dict[str, np.ndarray]]
) -> None:
    """Plot training histories and band distributions for different widths.

    Args:
        histories: Dictionary mapping band width to training history
        band_analyses: Dictionary mapping band width to learned band parameters
    """
    # Plot training histories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for band_width, history in histories.items():
        ax1.plot(history.history['sparse_categorical_accuracy'],
                 label=f'band={band_width:.2f}')
        ax2.plot(history.history['val_sparse_categorical_accuracy'],
                 label=f'band={band_width:.2f}')

    ax1.set_title('Training Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Plot final band distributions
    num_bands = len(band_analyses)
    fig, axes = plt.subplots(num_bands, 3, figsize=(15, 5 * num_bands))

    for idx, (band_width, layer_bands) in enumerate(band_analyses.items()):
        for layer_idx, (layer_name, band_values) in enumerate(layer_bands.items()):
            ax = axes[idx, layer_idx]
            band_values_flat = band_values.flatten()

            sns.histplot(band_values_flat, ax=ax)
            ax.set_title(f'Band={band_width:.2f}, {layer_name}')
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Count')

            # Add statistics
            mean_val = np.mean(band_values_flat)
            median_val = np.median(band_values_flat)
            ax.axvline(mean_val, color='r', linestyle='--',
                       label=f'Mean={mean_val:.3f}')
            ax.axvline(median_val, color='g', linestyle='--',
                       label=f'Median={median_val:.3f}')
            ax.legend()

    plt.tight_layout()
    plt.show()


def run_band_width_experiment() -> Tuple[
    Dict[float, Model],
    Dict[float, keras.callbacks.History],
    Dict[float, Dict[str, np.ndarray]]
]:
    """Run experiment comparing different band widths.

    Returns:
        Tuple of (models dictionary, histories dictionary, band analyses dictionary)
    """
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Train models with different band widths
    models = {}
    histories = {}
    band_analyses = {}

    for band_width in ExperimentConfig.BAND_WIDTHS:
        print(f"\nTraining model with band width {band_width}...")

        model = create_band_model(band_width=band_width)

        model.compile(
            optimizer=Adam(learning_rate=ExperimentConfig.INITIAL_LEARNING_RATE),
            loss=SparseCategoricalCrossentropy(from_logits=False),
            metrics=[SparseCategoricalAccuracy()]
        )

        history = model.fit(
            x_train,
            y_train,
            batch_size=ExperimentConfig.BATCH_SIZE,
            epochs=ExperimentConfig.EPOCHS,
            validation_split=ExperimentConfig.VALIDATION_SPLIT,
            callbacks=create_callbacks(
                ExperimentConfig.MODEL_SAVE_PATH.format(band_width=band_width)
            )
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f"Band width {band_width} test accuracy: {test_acc:.4f}")

        # Analyze learned bands
        analyzer = BandAnalyzer(model)
        learned_bands = analyzer.get_learned_bands()

        models[band_width] = model
        histories[band_width] = history
        band_analyses[band_width] = learned_bands

    # Plot comparison results
    plot_comparison_results(histories, band_analyses)

    return models, histories, band_analyses


def analyze_band_results(
        band_analyses: Dict[float, Dict[str, np.ndarray]],
        histories: Dict[float, keras.callbacks.History]
) -> str:
    """Analyze and format results from band width experiments.

    Args:
        band_analyses: Dictionary mapping band widths to learned band parameters
        histories: Dictionary mapping band widths to training histories

    Returns:
        Formatted string containing analysis results
    """
    results = []
    results.append("=" * 80)
    results.append("BAND RMS NORMALIZATION EXPERIMENT RESULTS")
    results.append("=" * 80)

    # Analyze each band width configuration
    for band_width, layer_bands in band_analyses.items():
        results.append(f"\nBAND WIDTH: {band_width:.3f}")
        results.append("-" * 40)

        # Get final training metrics
        history = histories[band_width]
        final_train_acc = history.history['sparse_categorical_accuracy'][-1]
        final_val_acc = history.history['val_sparse_categorical_accuracy'][-1]

        results.append(f"Final Training Accuracy: {final_train_acc:.4f}")
        results.append(f"Final Validation Accuracy: {final_val_acc:.4f}")

        # Analyze learned bands for each layer
        results.append("\nLearned Band Parameters:")
        for layer_name, band_values in layer_bands.items():
            band_values_flat = band_values.flatten()

            # Calculate statistics
            mean_val = np.mean(band_values_flat)
            median_val = np.median(band_values_flat)
            std_val = np.std(band_values_flat)
            min_val = np.min(band_values_flat)
            max_val = np.max(band_values_flat)

            # Calculate percentiles
            p25 = np.percentile(band_values_flat, 25)
            p75 = np.percentile(band_values_flat, 75)

            results.append(f"\n{layer_name}:")
            results.append(f"  Mean Scale Factor: {mean_val:.4f}")
            results.append(f"  Median Scale Factor: {median_val:.4f}")
            results.append(f"  Standard Deviation: {std_val:.4f}")
            results.append(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
            results.append(f"  25th-75th Percentile: [{p25:.4f}, {p75:.4f}]")

            # Analyze scale factor distribution
            below_mean = np.sum(band_values_flat < mean_val) / len(band_values_flat)
            results.append(f"  Proportion below mean: {below_mean:.2%}")

            # Calculate effective band utilization
            total_range = max_val - min_val
            utilized_range = total_range / band_width
            results.append(f"  Band utilization: {utilized_range:.2%}")

    # Compare performance across band widths
    results.append("\n" + "=" * 80)
    results.append("COMPARATIVE ANALYSIS")
    results.append("=" * 80)

    # Find best performing configuration
    best_acc = 0
    best_band = 0
    for band_width, history in histories.items():
        acc = history.history['val_sparse_categorical_accuracy'][-1]
        if acc > best_acc:
            best_acc = acc
            best_band = band_width

    results.append(f"\nBest performing band width: {best_band:.3f}")
    results.append(f"Best validation accuracy: {best_acc:.4f}")

    # Compare convergence speeds
    convergence_epochs = {}
    target_acc = 0.95  # Threshold for "good" accuracy

    for band_width, history in histories.items():
        val_acc = history.history['val_sparse_categorical_accuracy']
        epochs_to_converge = next(
            (i for i, acc in enumerate(val_acc) if acc >= target_acc),
            len(val_acc)
        )
        convergence_epochs[band_width] = epochs_to_converge

    results.append(f"\nEpochs to reach {target_acc:.0%} validation accuracy:")
    for band_width, epochs in convergence_epochs.items():
        results.append(f"  Band width {band_width:.3f}: {epochs} epochs")

    # Analyze stability
    stability_metrics = {}
    for band_width, history in histories.items():
        val_acc = history.history['val_sparse_categorical_accuracy']
        stability = np.std(val_acc[-5:])  # Standard deviation of last 5 epochs
        stability_metrics[band_width] = stability

    results.append("\nTraining stability (lower is better):")
    for band_width, stability in stability_metrics.items():
        results.append(f"  Band width {band_width:.3f}: {stability:.6f}")

    return "\n".join(results)


if __name__ == "__main__":
    models, histories, band_analyses = run_band_width_experiment()

    # Generate and print analysis
    analysis_text = analyze_band_results(band_analyses, histories)
    print(analysis_text)

    # Optionally save analysis to file
    with open("band_rms_analysis.txt", "w") as f:
        f.write(analysis_text)