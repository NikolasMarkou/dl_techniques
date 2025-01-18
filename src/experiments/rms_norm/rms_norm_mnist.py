"""
MNIST CNN Normalization Comparison
================================

This module implements three CNN variants for MNIST digit classification:
1. Baseline CNN (no additional normalization)
2. CNN with RMSNorm (root mean square normalization)
3. CNN with LogitNorm (logit normalization with temperature)

It provides tools for:
- Model training and evaluation
- Activation visualization and comparison
- Confidence distribution analysis
- OOD detection metrics

Key Components
-------------
1. Models:
    - build_model(): Creates CNN with configurable normalization
    - Supports RMSNorm, LogitNorm, or no normalization

2. Visualization Tools:
    - ActivationVisualizer: Compare layer activations
    - ConfidenceAnalyzer: Analyze prediction confidence
    - Supports comparative analysis between variants

3. Analysis Tools:
    - Confidence distribution metrics
    - Activation heatmaps
    - Model calibration measures
    - OOD detection scores

Features
--------
- Comprehensive model comparison
- Detailed activation analysis
- Confidence calibration metrics
- Training with various normalizations
- Visualization of normalization effects

Usage
-----
Run main() to:
1. Train all model variants
2. Generate activation visualizations
3. Compare confidence distributions
4. Analyze normalization effects
5. Generate comparative metrics

The script automatically saves:
- Model checkpoints
- Activation visualizations
- Confidence distribution plots
- Performance metrics
"""

import keras
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from keras import layers, Model
from typing import Tuple, Dict, Any, Optional, List, Literal


from dl_techniques.layers.logit_norm import RMSNorm, LogitNorm


def load_and_preprocess_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def build_model(
        norm_type: Optional[Literal[None, 'rms', 'logit']] = None,
        gaussian_noise_std: float = 0.1,
        temperature: float = 0.04
) -> keras.Model:
    """Build CNN model with specified normalization.

    Args:
        norm_type: Type of normalization to use:
            None: No additional normalization
            'rms': Root Mean Square normalization
            'logit': Logit normalization with temperature
        gaussian_noise_std: if > 0 add gaussian noise between layers
        temperature: Temperature parameter for LogitNorm (ignored if not using LogitNorm)

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(28, 28, 1))

    # First convolutional block
    x = layers.Conv2D(16, kernel_size=(3, 3), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization()(x)
    if gaussian_noise_std > 0:
        x = layers.GaussianNoise(gaussian_noise_std)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Second convolutional block
    x = layers.Conv2D(32, kernel_size=(3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization()(x)
    if gaussian_noise_std > 0:
        x = layers.GaussianNoise(gaussian_noise_std)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Third convolutional block
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization()(x)
    if gaussian_noise_std > 0:
        x = layers.GaussianNoise(gaussian_noise_std)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GaussianNoise(0.1)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    if gaussian_noise_std > 0:
        x = layers.GaussianNoise(gaussian_noise_std)(x)
    x = layers.Dense(10)(x)

    # Add normalization if specified
    if norm_type == 'rms':
        x = RMSNorm(constant=1.0)(x)
    elif norm_type == 'logit':
        x = LogitNorm(temperature=temperature)(x)

    outputs = layers.Activation('softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(
        model: keras.Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        epochs: int = 10,
        batch_size: int = 128
) -> keras.callbacks.History:
    """Train model with callbacks and validation.

    Args:
        model: Keras model to train
        x_train: Training data
        y_train: Training labels
        x_test: Test data
        y_test: Test labels
        model_name: Name for saving model
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Training history
    """
    output_dir = Path(f"outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            output_dir / f'{model_name}.keras',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    return history


class ModelAnalyzer:
    """Class for analyzing and comparing different model variants."""

    def __init__(
            self,
            baseline_model: keras.Model,
            rms_model: keras.Model,
            logit_model: keras.Model
    ):
        """Initialize with all model variants."""
        self.models = {
            'baseline': baseline_model,
            'rms': rms_model,
            'logit': logit_model
        }

        self.activation_models = {
            name: self._create_activation_model(model)
            for name, model in self.models.items()
        }

    def _create_activation_model(
            self,
            model: keras.Model
    ) -> Dict[str, keras.Model]:
        """Create models to extract activations from conv layers."""
        activation_models = {}

        for layer in model.layers:
            if isinstance(layer, layers.Conv2D):
                activation_models[layer.name] = keras.Model(
                    inputs=model.input,
                    outputs=layer.output
                )

        return activation_models

    def get_activations(
            self,
            image: np.ndarray,
            model_type: str
    ) -> Dict[str, np.ndarray]:
        """Get activation maps for all conv layers of specified model."""
        if model_type not in self.models:
            raise ValueError(f"model_type must be one of {list(self.models.keys())}")

        activations = self.activation_models[model_type]

        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)

        return {
            layer_name: model.predict(image)
            for layer_name, model in activations.items()
        }

    def plot_activation_comparison(
            self,
            image: np.ndarray,
            digit: int,
            save_path: Optional[str] = None
    ) -> None:
        """Plot and compare activation heatmaps for all models."""
        # Get activations for all models
        model_activations = {
            name: self.get_activations(image, name)
            for name in self.models
        }

        n_layers = len(next(iter(model_activations.values())))
        n_models = len(self.models)

        fig, axes = plt.subplots(n_models, n_layers + 1, figsize=(15, 3 * n_models))

        # Plot original image in first column
        for i in range(n_models):
            axes[i, 0].imshow(np.squeeze(image), cmap='gray')
            axes[i, 0].set_title('Original')

        # Plot activations for each model and layer
        for i, (model_name, activations) in enumerate(model_activations.items()):
            for j, (layer_name, activation) in enumerate(activations.items(), 1):
                act = activation[0]
                act = np.mean(act, axis=-1)  # Average over channels
                axes[i, j].imshow(act, cmap='viridis')
                axes[i, j].set_title(f'{model_name}\n{layer_name}')

        plt.suptitle(f'Activation Heatmaps for Digit {digit}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def analyze_confidence(
            self,
            x_test: np.ndarray,
            y_test: np.ndarray,
            save_path: Optional[str] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Analyze confidence distributions for all models."""
        model_predictions = {
            name: model.predict(x_test)
            for name, model in self.models.items()
        }

        true_labels = np.argmax(y_test, axis=1)
        stats = {}

        for name, predictions in model_predictions.items():
            stats[name] = self._compute_confidence_stats(predictions, true_labels)

        if save_path:
            self._plot_confidence_comparison(stats, save_path)

        return stats

    def _compute_confidence_stats(
            self,
            predictions: np.ndarray,
            true_labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute confidence statistics for one model."""
        stats = {
            'confidence_matrix': np.zeros((10, 10)),
            'mean_confidence': np.zeros(10),
            'max_confidence': np.zeros(10),
            'std_confidence': np.zeros(10)
        }

        for digit in range(10):
            digit_mask = (true_labels == digit)
            digit_preds = predictions[digit_mask]

            stats['confidence_matrix'][digit] = np.mean(digit_preds, axis=0)
            correct_conf = digit_preds[:, digit]

            stats['mean_confidence'][digit] = np.mean(correct_conf)
            stats['max_confidence'][digit] = np.max(correct_conf)
            stats['std_confidence'][digit] = np.std(correct_conf)

        return stats

    def _plot_confidence_comparison(
            self,
            stats: Dict[str, Dict[str, np.ndarray]],
            save_path: str
    ) -> None:
        """Plot comprehensive confidence comparison between models."""
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(len(self.models), 3)

        # Plot confidence heatmaps
        for i, (model_name, model_stats) in enumerate(stats.items()):
            ax = fig.add_subplot(gs[i, 0])
            sns.heatmap(
                model_stats['confidence_matrix'],
                ax=ax,
                cmap='YlOrRd',
                vmin=0,
                vmax=1,
                annot=True,
                fmt='.2f'
            )
            ax.set_title(f'{model_name} Confidence Distribution')
            ax.set_xlabel('Predicted Digit')
            ax.set_ylabel('True Digit')

        # Plot mean confidence comparison
        ax = fig.add_subplot(gs[:, 1])
        x = np.arange(10)
        width = 1 / (len(self.models) + 1)

        for i, (name, model_stats) in enumerate(stats.items()):
            ax.bar(
                x + i * width - width * len(self.models) / 2,
                model_stats['mean_confidence'],
                width,
                label=name
            )

        ax.set_title('Mean Confidence per Digit')
        ax.set_xlabel('Digit')
        ax.set_ylabel('Mean Confidence')
        ax.legend()
        ax.set_xticks(x)

        # Plot confidence distributions
        ax = fig.add_subplot(gs[:, 2])
        box_data = [
            [stats[name]['confidence_matrix'][i, i] for i in range(10)]
            for name in self.models
        ]
        ax.boxplot(box_data, labels=list(self.models.keys()))
        ax.set_title('Distribution of Confidence Scores')
        ax.set_ylabel('Confidence')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


def main() -> None:
    """Main training and analysis pipeline."""
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist()

    # Build all model variants
    models = {
        'baseline': build_model(norm_type=None),
        'rms': build_model(norm_type='rms'),
        'logit': build_model(norm_type='logit')
    }

    # Train all models
    histories = {}
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        histories[name] = train_model(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            f'{name}_model'
        )

    # Create analyzer
    analyzer = ModelAnalyzer(
        models['baseline'],
        models['rms'],
        models['logit']
    )

    # Create output directories
    output_dir = Path("analysis_outputs")
    output_dir.mkdir(exist_ok=True)

    # Analyze models
    stats = analyzer.analyze_confidence(
        x_test,
        y_test,
        save_path=str(output_dir / 'confidence_comparison.png')
    )

    # Print summary statistics
    print("\nConfidence Summary Statistics:")
    for model_name, model_stats in stats.items():
        print(f"\n{model_name.title()} Model:")
        print(f"Average confidence: {np.mean(model_stats['mean_confidence']):.3f}")
        print(f"Max confidence: {np.max(model_stats['max_confidence']):.3f}")
        print(f"Average std deviation: {np.mean(model_stats['std_confidence']):.3f}")

    # Visualize activations for sample digits
    for digit in range(10):
        # Find first instance of digit in test set
        digit_idx = np.where(np.argmax(y_test, axis=1) == digit)[0][0]
        digit_image = x_test[digit_idx]

        # Plot and save activation maps
        analyzer.plot_activation_comparison(
            digit_image,
            digit,
            save_path=str(output_dir / f'digit_{digit}_activations.png')
        )

    # Plot training histories
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{name} (train)')
        plt.plot(history.history['val_accuracy'], label=f'{name} (val)')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} (train)')
        plt.plot(history.history['val_loss'], label=f'{name} (val)')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(output_dir / 'training_history.png'))
    plt.show()


if __name__ == "__main__":
    main()