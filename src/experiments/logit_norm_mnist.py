"""
MNIST CNN with LogitNorm and Baseline Comparison
==============================================

This module implements two CNNs for MNIST digit classification:
1. Baseline CNN
2. CNN with LogitNorm normalization

It also provides tools for visualizing and comparing activation heatmaps.
"""

import keras
import tensorflow as tf
from keras import layers, Model
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


from dl_techniques.layers.logit_norm import LogitNorm

def load_and_preprocess_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def build_model(use_logitnorm: bool = True) -> keras.Model:
    """Build CNN model with optional LogitNorm.

    Args:
        use_logitnorm: Whether to include LogitNorm layer

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(28, 28, 1))

    # First convolutional block
    x = layers.Conv2D(16, kernel_size=(3, 3), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Second convolutional block
    x = layers.Conv2D(32, kernel_size=(3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Third convolutional block
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(10)(x)

    # Add LogitNorm if specified
    if use_logitnorm:
        x = LogitNorm(constant=1.0)(x)

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
        epochs: int = 3,
        batch_size: int = 128
) -> keras.callbacks.History:
    """Train the model with callbacks and validation."""

    # Create output directory
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


class ActivationVisualizer:
    """Class for visualizing model activations and comparing models."""

    def __init__(
            self,
            baseline_model: keras.Model,
            logitnorm_model: keras.Model
    ) -> None:
        """Initialize with both models."""
        self.baseline_model = baseline_model
        self.logitnorm_model = logitnorm_model

        # Create activation models for each convolutional layer
        self.baseline_activations = self._create_activation_model(baseline_model)
        self.logitnorm_activations = self._create_activation_model(logitnorm_model)

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

    def get_activation_maps(
            self,
            image: np.ndarray,
            model_type: str = 'baseline'
    ) -> Dict[str, np.ndarray]:
        """Get activation maps for all conv layers."""
        if model_type not in ['baseline', 'logitnorm']:
            raise ValueError("model_type must be 'baseline' or 'logitnorm'")

        activations = (self.baseline_activations if model_type == 'baseline'
                       else self.logitnorm_activations)

        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)

        return {
            layer_name: model.predict(image)
            for layer_name, model in activations.items()
        }

    def plot_activation_heatmaps(
            self,
            image: np.ndarray,
            digit: int,
            save_path: Optional[str] = None
    ) -> None:
        """Plot and compare activation heatmaps for both models."""
        # Get activations for both models
        baseline_acts = self.get_activation_maps(image, 'baseline')
        logitnorm_acts = self.get_activation_maps(image, 'logitnorm')

        n_layers = len(baseline_acts)
        fig, axes = plt.subplots(2, n_layers + 1, figsize=(15, 6))

        # Plot original image
        axes[0, 0].imshow(np.squeeze(image), cmap='gray')
        axes[0, 0].set_title('Original')
        axes[1, 0].imshow(np.squeeze(image), cmap='gray')
        axes[1, 0].set_title('Original')

        # Plot activations for each layer
        for idx, layer_name in enumerate(baseline_acts.keys(), 1):
            # Baseline model activations
            baseline_act = baseline_acts[layer_name][0]
            baseline_act = np.mean(baseline_act, axis=-1)  # Average over channels
            axes[0, idx].imshow(baseline_act, cmap='viridis')
            axes[0, idx].set_title(f'Baseline\n{layer_name}')

            # LogitNorm model activations
            logitnorm_act = logitnorm_acts[layer_name][0]
            logitnorm_act = np.mean(logitnorm_act, axis=-1)  # Average over channels
            axes[1, idx].imshow(logitnorm_act, cmap='viridis')
            axes[1, idx].set_title(f'LogitNorm\n{layer_name}')

        plt.suptitle(f'Activation Heatmaps for Digit {digit}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()


def main() -> None:
    """Main training and visualization pipeline."""
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist()

    # Build and train baseline model
    print("\nTraining baseline model...")
    baseline_model = build_model(use_logitnorm=False)
    baseline_history = train_model(
        baseline_model,
        x_train,
        y_train,
        x_test,
        y_test,
        'baseline_model'
    )

    # Build and train LogitNorm model
    print("\nTraining LogitNorm model...")
    logitnorm_model = build_model(use_logitnorm=True)
    logitnorm_history = train_model(
        logitnorm_model,
        x_train,
        y_train,
        x_test,
        y_test,
        'logitnorm_model'
    )

    # Create visualizer
    visualizer = ActivationVisualizer(baseline_model, logitnorm_model)

    # Visualize activations for each digit
    output_dir = Path("activation_maps")
    output_dir.mkdir(exist_ok=True)

    for digit in range(10):
        # Find first instance of digit in test set
        digit_idx = np.where(np.argmax(y_test, axis=1) == digit)[0][0]
        digit_image = x_test[digit_idx]

        # Plot and save activation maps
        visualizer.plot_activation_heatmaps(
            digit_image,
            digit,
            save_path=str(output_dir / f'digit_{digit}_activations.png')
        )


if __name__ == "__main__":
    main()
