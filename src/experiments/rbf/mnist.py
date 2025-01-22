import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Tuple, Dict, Any
from dataclasses import dataclass

import keras
import tensorflow as tf
from keras import Model, Input
from keras.api.optimizers import Adam
from keras.api.metrics import SparseCategoricalAccuracy
from keras.api.losses import SparseCategoricalCrossentropy

from dl_techniques.layers.rbf import RBFLayer


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration constants for the CNN-RBF MNIST experiment."""
    # Model architecture
    INPUT_SHAPE: Tuple[int, int, int] = (28, 28, 1)
    NUM_CLASSES: int = 10

    # CNN parameters
    CONV1_FILTERS: int = 16
    CONV2_FILTERS: int = 32
    CONV3_FILTERS: int = 64
    KERNEL_SIZE: Tuple[int, int] = (3, 3)
    KERNEL_SIZE_STEM: Tuple[int, int] = (5, 5)
    ACTIVATION: str = "relu"

    # RBF-Dense block parameters
    RBF_UNITS: int = 32
    KERNEL_REGULARIZER: keras.regularizers.Regularizer = keras.regularizers.L2(1e-4)

    # Training parameters
    BATCH_SIZE: int = 128
    EPOCHS: int = 100
    VALIDATION_SPLIT: float = 0.1
    INITIAL_LEARNING_RATE: float = 0.001
    MIN_LEARNING_RATE: float = 1e-5
    LR_REDUCTION_FACTOR: float = 0.5

    # Early stopping
    PATIENCE: int = 5
    LR_PATIENCE: int = 3

    # File paths
    MODEL_SAVE_PATH: str = 'mnist_cnn_rbf_model.keras'


class MNISTCNNRBFModel(Model):
    """MNIST classification model using CNN layers followed by RBF-Dense block.

    Architecture:
    Conv1 → LayerNorm → GELU → MaxPool →
    Conv2 → LayerNorm → GELU → MaxPool →
    Conv3 → LayerNorm → GELU → MaxPool →
    Flatten → RBF Block → Output

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        rbf_units: Number of RBF centers
        kernel_regularizer: Regularization for kernels
    """

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = ExperimentConfig.INPUT_SHAPE,
            num_classes: int = ExperimentConfig.NUM_CLASSES,
            rbf_units: int = ExperimentConfig.RBF_UNITS,
            kernel_regularizer: keras.regularizers.Regularizer = ExperimentConfig.KERNEL_REGULARIZER,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Save configuration
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.rbf_units = rbf_units
        self.kernel_regularizer = kernel_regularizer

        # First conv block
        self.conv1 = keras.layers.Conv2D(
            ExperimentConfig.CONV1_FILTERS,
            ExperimentConfig.KERNEL_SIZE_STEM,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )
        self.norm1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.Activation(ExperimentConfig.ACTIVATION)
        self.pool1 = keras.layers.MaxPooling2D()

        # Second conv block
        self.conv2 = keras.layers.Conv2D(
            ExperimentConfig.CONV2_FILTERS,
            ExperimentConfig.KERNEL_SIZE,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )
        self.norm2 = keras.layers.BatchNormalization()
        self.act2 = keras.layers.Activation(ExperimentConfig.ACTIVATION)
        self.pool2 = keras.layers.MaxPooling2D()

        # Third conv block
        self.conv3 = keras.layers.Conv2D(
            ExperimentConfig.CONV3_FILTERS,
            ExperimentConfig.KERNEL_SIZE,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )
        self.norm3 = keras.layers.BatchNormalization()
        self.act3 = keras.layers.Activation(ExperimentConfig.ACTIVATION)
        self.pool3 = keras.layers.MaxPooling2D()

        # Flatten and RBF block
        self.flatten = keras.layers.Flatten()

        self.rbf = RBFLayer(
            units=rbf_units,
        )

        self.output_layer = keras.layers.Dense(
            units=num_classes,
            activation='softmax',
            kernel_initializer="he_normal",
            kernel_regularizer=None
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)
            training: Whether in training mode

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First conv block
        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)

        # Third conv block
        x = self.conv3(x)
        x = self.norm3(x, training=training)
        x = self.act3(x)
        x = self.pool3(x)

        # RBF block
        x = self.flatten(x)
        x = self.rbf(x, training=training)

        # output
        x = self.output_layer(x, training=training)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'rbf_units': self.rbf_units,
            'kernel_regularizer': self.kernel_regularizer
        })
        return config


def plot_confusion_matrix(model: Model, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """Plot confusion matrix for model predictions.

    Args:
        model: Trained model
        x_test: Test images
        y_test: True labels
    """
    # Get predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def train_mnist_rbf() -> Tuple[Model, tf.keras.callbacks.History]:
    """Train the RBF-Dense model on MNIST dataset.

    Returns:
        Tuple of (trained model, training history)
    """
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Create and compile model
    model = MNISTCNNRBFModel()

    model.compile(
        optimizer=Adam(learning_rate=ExperimentConfig.INITIAL_LEARNING_RATE),
        loss=SparseCategoricalCrossentropy(from_logits=False),
        metrics=[SparseCategoricalAccuracy()]
    )

    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ExperimentConfig.MODEL_SAVE_PATH,
            save_best_only=True,
            monitor='val_sparse_categorical_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            patience=ExperimentConfig.PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_sparse_categorical_accuracy',
            factor=ExperimentConfig.LR_REDUCTION_FACTOR,
            patience=ExperimentConfig.LR_PATIENCE,
            min_lr=ExperimentConfig.MIN_LEARNING_RATE
        )
    ]

    # Train model
    history = model.fit(
        x_train,
        y_train,
        batch_size=ExperimentConfig.BATCH_SIZE,
        epochs=ExperimentConfig.EPOCHS,
        validation_split=ExperimentConfig.VALIDATION_SPLIT,
        callbacks=callbacks
    )

    # Evaluate and visualize
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(model, x_test, y_test)

    return model, history


if __name__ == "__main__":
    train_mnist_rbf()
