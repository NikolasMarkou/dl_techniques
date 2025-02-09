import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Tuple, Dict, Any, Optional, List
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
    """Configuration for normalization comparison experiment."""
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
    BAND_WIDTH: float = 0.2
    EPSILON: float = 1e-6
    BAND_REGULARIZER: float = 1e-5

    # Training parameters
    BATCH_SIZE: int = 128
    EPOCHS: int = 10
    VALIDATION_SPLIT: float = 0.2
    INITIAL_LEARNING_RATE: float = 0.001
    MIN_LEARNING_RATE: float = 1e-6
    LR_REDUCTION_FACTOR: float = 0.2

    # Early stopping
    PATIENCE: int = 7
    LR_PATIENCE: int = 3

    # Model variants
    VARIANTS: Tuple[str] = ("band_rms", "batch_norm", "layer_norm")

    # File paths
    MODEL_SAVE_PATH: str = "mnist_norm_comparison_{variant}.keras"


class NormalizationComparisonModel(Model):
    """MNIST classification model with configurable normalization layers.

    Args:
        norm_type: Type of normalization to use ('band_rms', 'batch_norm', 'layer_norm')
        input_shape: Input image shape
        num_classes: Number of output classes
        kernel_regularizer: Regularization for kernels
    """

    def __init__(
            self,
            norm_type: str,
            input_shape: Tuple[int, int, int] = ExperimentConfig.INPUT_SHAPE,
            num_classes: int = ExperimentConfig.NUM_CLASSES,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if kernel_regularizer is None:
            kernel_regularizer = keras.regularizers.L2(1e-4)

        self.norm_type = norm_type
        self._create_layers(input_shape, num_classes, kernel_regularizer)

    def _create_normalization(self) -> keras.layers.Layer:
        """Create normalization layer based on specified type."""
        if self.norm_type == "band_rms":
            return BandRMSNorm(
                max_band_width=ExperimentConfig.BAND_WIDTH,
                epsilon=ExperimentConfig.EPSILON,
                band_regularizer=keras.regularizers.L2(ExperimentConfig.BAND_REGULARIZER)
            )
        elif self.norm_type == "batch_norm":
            return keras.layers.BatchNormalization()
        elif self.norm_type == "layer_norm":
            return keras.layers.LayerNormalization(epsilon=ExperimentConfig.EPSILON)
        else:
            raise ValueError(f"Unknown normalization type: {self.norm_type}")

    def _create_layers(
            self,
            input_shape: Tuple[int, int, int],
            num_classes: int,
            kernel_regularizer: keras.regularizers.Regularizer
    ) -> None:
        """Create model layers."""
        # First conv block
        self.conv1 = keras.layers.Conv2D(
            ExperimentConfig.CONV1_FILTERS,
            ExperimentConfig.KERNEL_SIZE,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )
        self.norm1 = self._create_normalization()
        self.act1 = keras.layers.Activation(ExperimentConfig.ACTIVATION)
        self.pool1 = keras.layers.MaxPooling2D()
        self.drop1 = keras.layers.Dropout(0.1)

        # Second conv block
        self.conv2 = keras.layers.Conv2D(
            ExperimentConfig.CONV2_FILTERS,
            ExperimentConfig.KERNEL_SIZE,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )
        self.norm2 = self._create_normalization()
        self.act2 = keras.layers.Activation(ExperimentConfig.ACTIVATION)
        self.pool2 = keras.layers.MaxPooling2D()
        self.drop2 = keras.layers.Dropout(0.2)

        # Third conv block
        self.conv3 = keras.layers.Conv2D(
            ExperimentConfig.CONV3_FILTERS,
            ExperimentConfig.KERNEL_SIZE,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )
        self.norm3 = self._create_normalization()
        self.act3 = keras.layers.Activation(ExperimentConfig.ACTIVATION)
        self.pool3 = keras.layers.MaxPooling2D()
        self.drop3 = keras.layers.Dropout(0.3)

        # Output layers
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(
            256,
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )
        self.norm4 = self._create_normalization()
        self.act4 = keras.layers.Activation(ExperimentConfig.ACTIVATION)
        self.drop4 = keras.layers.Dropout(0.4)

        self.output_layer = keras.layers.Dense(
            num_classes,
            activation='softmax',
            kernel_initializer='he_normal'
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the model."""
        # First conv block
        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        # Third conv block
        x = self.conv3(x)
        x = self.norm3(x, training=training)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.drop3(x, training=training)

        # Output layers
        x = self.flatten(x)
        x = self.dense(x)
        x = self.norm4(x, training=training)
        x = self.act4(x)
        x = self.drop4(x, training=training)
        x = self.output_layer(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        config = super().get_config()
        config.update({
            'norm_type': self.norm_type
        })
        return config


def create_callbacks(model_path: str) -> List[keras.callbacks.Callback]:
    """Create training callbacks.

    Args:
        model_path: Path to save the model

    Returns:
        List of callbacks
    """
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


def plot_training_history(histories: Dict[str, keras.callbacks.History]) -> None:
    """Plot training histories for different normalization approaches.

    Args:
        histories: Dictionary mapping normalization type to training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for norm_type, history in histories.items():
        ax1.plot(history.history['sparse_categorical_accuracy'], label=f'{norm_type}')
        ax2.plot(history.history['val_sparse_categorical_accuracy'], label=f'{norm_type}')

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


def plot_confusion_matrices(models: Dict[str, Model], x_test: np.ndarray, y_test: np.ndarray) -> None:
    """Plot confusion matrices for different models.

    Args:
        models: Dictionary mapping normalization type to trained model
        x_test: Test images
        y_test: True labels
    """
    fig, axes = plt.subplots(1, len(models), figsize=(20, 6))

    for idx, (norm_type, model) in enumerate(models.items()):
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_test, y_pred_classes)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'Confusion Matrix - {norm_type}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()


def run_normalization_experiment() -> Tuple[Dict[str, Model], Dict[str, keras.callbacks.History]]:
    """Run experiment comparing different normalization approaches.

    Returns:
        Tuple of (models dictionary, histories dictionary)
    """
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Train models with different normalizations
    models = {}
    histories = {}

    for norm_type in ExperimentConfig.VARIANTS:
        print(f"\nTraining model with {norm_type} normalization...")

        model = NormalizationComparisonModel(norm_type=norm_type)

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
                ExperimentConfig.MODEL_SAVE_PATH.format(variant=norm_type)
            )
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f"{norm_type} Test accuracy: {test_acc:.4f}")

        models[norm_type] = model
        histories[norm_type] = history

    # Plot results
    plot_training_history(histories)
    plot_confusion_matrices(models, x_test, y_test)

    return models, histories


if __name__ == "__main__":
    run_normalization_experiment()