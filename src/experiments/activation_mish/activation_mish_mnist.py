"""
MNIST Activation Function Comparison Experiment
============================================

This script implements a comparison of different activation functions:
- GELU (baseline)
- Mish
- ScaledMish

The experiment uses a 3-level CNN architecture on the MNIST dataset with L2 regularization.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import os
import numpy as np
import tensorflow as tf
import keras
from keras.api.datasets import mnist
from keras.api.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Input, BatchNormalization, Dropout
)
from keras.api.activations import gelu
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass

from dl_techniques.layers.mish import Mish,ScaledMish


# ---------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 0.001
    l2_regularization: float = 0.01
    validation_split: float = 0.1
    kernel_initializer: str = 'he_normal'


# ---------------------------------------------------------------------


class ActivationExperiment:
    """Class to manage the activation function comparison experiment."""

    def __init__(self, config: ExperimentConfig):
        """Initialize the experiment.

        Args:
            config: Configuration parameters for the experiment
        """
        self.config = config
        self._prepare_data()
        self.activations = self._setup_activations()
        self.histories: Dict[str, Any] = {}
        self.models: Dict[str, keras.Model] = {}

    def _prepare_data(self) -> None:
        """Prepare MNIST dataset for training."""
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Reshape and normalize
        self.x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        self.x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        # Convert labels to categorical
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)

    def _setup_activations(self) -> Dict[str, Callable]:
        """Setup different activation functions for comparison."""
        return {
            'gelu': gelu,
            'mish': Mish(),
            'scaled_mish': ScaledMish(alpha=2.0)
        }

    def create_model(self, activation: Callable) -> keras.Model:
        """Create a 3-level CNN model with the specified activation.

        Args:
            activation: Activation function to use in convolutional layers

        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=(28, 28, 1))

        # First convolutional block
        x = Conv2D(
            16, (3, 3),
            padding='same',
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(self.config.l2_regularization)
        )(inputs)
        x = activation(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Second convolutional block
        x = Conv2D(
            32, (3, 3),
            padding='same',
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(self.config.l2_regularization)
        )(x)
        x = activation(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Third convolutional block
        x = Conv2D(
            64, (3, 3),
            padding='same',
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(self.config.l2_regularization)
        )(x)
        x = activation(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Dense layers
        x = Flatten()(x)
        x = Dense(
            32,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(self.config.l2_regularization)
        )(x)
        x = activation(x)
        x = Dropout(0.25)(x)
        outputs = Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_all_models(self) -> None:
        """Train all models with different activation functions."""
        for name, activation in self.activations.items():
            print(f"\nTraining model with {name} activation...")
            model = self.create_model(activation)

            history = model.fit(
                self.x_train, self.y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_split=self.config.validation_split,
                verbose=1
            )

            self.models[name] = model
            self.histories[name] = history.history

    def analyze_activations(self) -> Dict[str, Dict[str, float]]:
        """Analyze activation patterns and compute statistics.

        Returns:
            Dictionary containing activation statistics for each model
        """
        stats_dict = {}

        for name, model in self.models.items():
            # Get activations from the last convolutional layer
            activation_model = keras.Model(
                inputs=model.input,
                outputs=model.layers[-4].output  # Before flatten layer
            )
            activations = activation_model.predict(self.x_test[:1000])

            stats_dict[name] = {
                'mean': float(np.mean(activations)),
                'std': float(np.std(activations)),
                'sparsity': float(np.mean(np.abs(activations) < 1e-5)),
                'kurtosis': float(stats.kurtosis(activations.flatten())),
                'skewness': float(stats.skew(activations.flatten()))
            }

        return stats_dict

    def plot_results(self, save_dir: str = 'results') -> None:
        """Generate and save visualization plots."""
        os.makedirs(save_dir, exist_ok=True)

        # Plot training histories
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for name, history in self.histories.items():
            ax1.plot(history['accuracy'], label=f'{name}')
            ax2.plot(history['val_accuracy'], label=f'{name}')

        ax1.set_title('Training Accuracy')
        ax2.set_title('Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax2.set_xlabel('Epoch')
        ax1.legend()
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        plt.close()

        # Plot activation statistics
        stats = self.analyze_activations()
        metrics = ['mean', 'std', 'sparsity', 'kurtosis', 'skewness']

        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))

        for idx, metric in enumerate(metrics):
            values = [stats[model][metric] for model in stats]
            axes[idx].bar(list(stats.keys()), values)
            axes[idx].set_title(f'{metric.capitalize()} Comparison')
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'activation_metrics.png'))
        plt.close()


# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete experiment."""
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Initialize and run experiment
    config = ExperimentConfig()
    experiment = ActivationExperiment(config)
    experiment.train_all_models()
    experiment.plot_results()

    # Print final statistics
    stats = experiment.analyze_activations()
    print("\nFinal Activation Statistics:")
    for model_name, model_stats in stats.items():
        print("=" * 50)
        print(f"{model_name}:")
        for metric, value in model_stats.items():
            print(f"{metric}: {value:.4f}")


# ---------------------------------------------------------------------


if __name__ == "__main__":
    main()