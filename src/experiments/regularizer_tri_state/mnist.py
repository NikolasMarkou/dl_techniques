"""
MNIST Regularizer Comparison Experiment
=====================================

This script implements a comprehensive comparison of different regularization techniques:
- Baseline (no regularization)
- l2
- Tri-State Preference Regularizer

The experiment uses a 3-level CNN architecture on the MNIST dataset.
"""

import os
import keras
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Any
from keras.api.datasets import mnist
from keras.api.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Input, BatchNormalization, Dropout
)
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from dataclasses import dataclass

from dl_techniques.utils.logger import logger
from dl_techniques.regularizers.tri_state_preference import TriStatePreferenceRegularizer


# Set random seeds for reproducibility
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 0.001
    regularizer_scale: float = 0.01
    regularizer_multiplier: float = 1.0
    validation_split: float = 0.1


class RegularizerExperiment:
    """Class to manage the regularizer comparison experiment."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the experiment.

        Args:
            config: Configuration parameters for the experiment
        """
        self.config = config
        self._prepare_data()
        self.regularizers = self._setup_regularizers()
        self.histories: Dict[str, Any] = {}
        self.models: Dict[str, keras.Model] = {}

    def _prepare_data(self) -> None:
        """Prepare MNIST dataset for training."""
        # Load and preprocess MNIST data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Reshape and normalize
        self.x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        self.x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        # Convert labels to categorical
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)

    def _setup_regularizers(self) -> Dict[str, Optional[keras.regularizers.Regularizer]]:
        """Setup different regularizers for comparison."""
        return {
            'baseline': None,
            'l2': keras.regularizers.L2(0.01),
            'tri_state_4.0': TriStatePreferenceRegularizer(multiplier=self.config.regularizer_multiplier, scale=4.0),
            'tri_state_8.0': TriStatePreferenceRegularizer(multiplier=self.config.regularizer_multiplier, scale=8.0),
        }

    def create_model(self,
                     regularizer: Optional[keras.regularizers.Regularizer]) -> keras.Model:
        """
        Create a 3-level CNN model with the specified regularizer.

        Args:
            regularizer: Regularizer to apply to convolutional layers

        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=(28, 28, 1))

        # First convolutional block
        x = Conv2D(16, (3, 3), activation='linear', padding='same',
                   kernel_regularizer=regularizer)(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Second convolutional block
        x = Conv2D(32, (3, 3), activation='linear', padding='same',
                   kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Third convolutional block
        x = Conv2D(64, (3, 3), activation='linear', padding='same',
                   kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Dense layers
        x = Flatten()(x)
        x = Dense(32, activation='linear')(x)
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
        """Train all models with different regularizers."""
        for name, regularizer in self.regularizers.items():
            print(f"\nTraining {name} model...")
            model = self.create_model(regularizer)

            # Train model
            history = model.fit(
                self.x_train, self.y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_split=self.config.validation_split,
                verbose=1
            )

            # Store model and history
            self.models[name] = model
            self.histories[name] = history.history

    def analyze_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze weight distributions and compute statistics.

        Returns:
            Dictionary containing weight statistics for each model
        """
        stats_dict = {}

        for name, model in self.models.items():
            # Collect all weights from convolutional layers
            conv_weights = []
            for layer in model.layers:
                if isinstance(layer, Conv2D):
                    weights = layer.get_weights()[0]
                    conv_weights.extend(weights.flatten())

            conv_weights = np.array(conv_weights)

            # Calculate statistics
            stats_dict[name] = {
                'mean': float(np.mean(conv_weights)),
                'std': float(np.std(conv_weights)),
                'sparsity': float(np.mean(np.abs(conv_weights) < 1e-5)),
                'kurtosis': float(stats.kurtosis(conv_weights)),
                'skewness': float(stats.skew(conv_weights))
            }

        return stats_dict

    def plot_results(self, save_dir: str = 'results') -> None:
        """
        Generate and save visualization plots.

        Args:
            save_dir: Directory to save the plots
        """
        os.makedirs(save_dir, exist_ok=True)

        # Plot training histories
        self._plot_histories(save_dir)

        # Plot weight distributions
        self._plot_weight_distributions(save_dir)

        # Plot layer-wise weight distributions
        self._plot_layer_weight_distributions(save_dir)

        # Plot evaluation metrics
        self._plot_evaluation_metrics(save_dir)

    def _plot_histories(self, save_dir: str) -> None:
        """Plot training histories."""
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

    def _plot_weight_distributions(self, save_dir: str) -> None:
        """Plot weight distributions within [-1.25, 1.25] range."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()

        range_min, range_max = -1.25, 1.25

        for idx, (name, model) in enumerate(self.models.items()):
            weights = []
            for layer in model.layers:
                if isinstance(layer, Conv2D):
                    weights.extend(layer.get_weights()[0].flatten())

            weights = np.array(weights)

            # Calculate percentage of weights outside range
            outside_range = np.sum(
                (weights < range_min) | (weights > range_max)
            ) / len(weights) * 100

            sns.histplot(
                weights,
                bins=50,
                ax=axes[idx],
                stat='density',
                element='step',
                fill=True,
                alpha=0.3
            )

            # Add vertical lines at -1, 0, 1
            for x in [-1, 0, 1]:
                axes[idx].axvline(x=x, color='red', linestyle='--', alpha=0.5)

            axes[idx].set_xlim(range_min, range_max)
            axes[idx].set_title(f'{name} Weight Distribution\n{outside_range:.1f}% weights outside [-1.25, 1.25]')
            axes[idx].set_xlabel('Weight Value')
            axes[idx].set_ylabel('Density')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weight_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_evaluation_metrics(self, save_dir: str) -> None:
        """Plot evaluation metrics comparison."""
        stats_dict = self.analyze_weights()

        metrics = ['mean', 'std', 'sparsity', 'kurtosis', 'skewness']
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))

        for idx, metric in enumerate(metrics):
            values = [stats_dict[model][metric] for model in stats_dict]
            axes[idx].bar(list(stats_dict.keys()), values)
            axes[idx].set_title(f'{metric.capitalize()} Comparison')
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'evaluation_metrics.png'))
        plt.close()

    def analyze_weights_by_layer(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Analyze weight distributions layer by layer.

        Returns:
            Dictionary containing weight statistics for each layer in each model
        """
        stats_dict = {}

        for name, model in self.models.items():
            layer_stats = []
            for layer in model.layers:
                if isinstance(layer, Conv2D):
                    weights = layer.get_weights()[0].flatten()
                    layer_stats.append({
                        'mean': float(np.mean(weights)),
                        'std': float(np.std(weights)),
                        'sparsity': float(np.mean(np.abs(weights) < 1e-5)),
                        'in_range': float(np.mean(
                            (weights >= -1.25) & (weights <= 1.25)
                        )) * 100
                    })
            stats_dict[name] = layer_stats

        return stats_dict

    def _plot_layer_weight_distributions(self, save_dir: str) -> None:
        """Plot weight distributions by layer for each model."""
        for name, model in self.models.items():
            conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]
            n_layers = len(conv_layers)

            fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
            if n_layers == 1:
                axes = [axes]

            range_min, range_max = -1.25, 1.25

            for idx, layer in enumerate(conv_layers):
                weights = layer.get_weights()[0].flatten()

                # Calculate percentage of weights outside range
                outside_range = np.sum(
                    (weights < range_min) | (weights > range_max)
                ) / len(weights) * 100

                sns.histplot(
                    weights,
                    bins=50,
                    ax=axes[idx],
                    stat='density',
                    element='step',
                    fill=True,
                    alpha=0.3
                )

                # Add vertical lines at -1, 0, 1
                for x in [-1, 0, 1]:
                    axes[idx].axvline(x=x, color='red', linestyle='--', alpha=0.5)

                axes[idx].set_xlim(range_min, range_max)
                axes[idx].set_title(f'Layer {idx + 1}\n{outside_range:.1f}% outside range')
                axes[idx].set_xlabel('Weight Value')
                axes[idx].set_ylabel('Density')
                axes[idx].grid(True, alpha=0.3)

            plt.suptitle(f'{name} Weight Distributions by Layer', y=1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'weight_dist_by_layer_{name}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Run the complete experiment."""
    # Initialize experiment
    config = ExperimentConfig()
    experiment = RegularizerExperiment(config)

    # Train models
    experiment.train_all_models()

    # Analyze and plot results
    experiment.plot_results()

    # Print final statistics
    stats = experiment.analyze_weights()
    logger.info("Final Weight Statistics:")
    for model_name, model_stats in stats.items():
        logger.info("=======================================================")
        logger.info(f"{model_name}:")
        for metric, value in model_stats.items():
            logger.info(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()