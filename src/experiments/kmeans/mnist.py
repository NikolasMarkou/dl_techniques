import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Tuple
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import tensorflow as tf
from keras import Model, Input
from keras.api.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization,
    Dropout, Layer, ReLU
)
from keras.api.optimizers import Adam
from keras.api.regularizers import L2
from keras.api.initializers import HeNormal
from keras.api.metrics import SparseCategoricalAccuracy
from keras.api.losses import SparseCategoricalCrossentropy

from dl_techniques.layers.differentiable_kmeans import DifferentiableKMeansLayer


@dataclass(frozen=True)
class ModelConfig:
    """Configuration constants for the MNIST CNN model."""
    # Model architecture
    INPUT_SHAPE: Tuple[int, int, int] = (28, 28, 1)
    NUM_CLASSES: int = 10

    # Layer sizes
    CONV1_FILTERS: int = 16
    CONV2_FILTERS: int = 32
    CONV3_FILTERS: int = 64
    DENSE_UNITS: int = 32
    KERNEL_SIZE: Tuple[int, int] = (3, 3)

    # Regularization
    KERNEL_REGULARIZER: float = 1e-4
    DROPOUT_RATE: float = 0.3
    USE_BATCH_NORM: bool = True

    # Training parameters
    BATCH_SIZE: int = 128
    EPOCHS: int = 5
    VALIDATION_SPLIT: float = 0.1
    INITIAL_LEARNING_RATE: float = 0.001
    MIN_LEARNING_RATE: float = 1e-5
    LR_REDUCTION_FACTOR: float = 0.5

    # Early stopping
    PATIENCE: int = 5
    LR_PATIENCE: int = 3

    # File paths
    MODEL_SAVE_PATH: str = 'mnist_model.keras'


class MNISTConvNet(Model):
    """CNN model for MNIST digit classification.

    Architecture follows the pattern:
    Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        kernel_regularizer: L2 regularization factor for conv/dense layers
        dropout_rate: Dropout rate between layers
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = ModelConfig.INPUT_SHAPE,
            num_classes: int = ModelConfig.NUM_CLASSES,
            kernel_regularizer: float = ModelConfig.KERNEL_REGULARIZER,
            dropout_rate: float = ModelConfig.DROPOUT_RATE,
            use_batch_norm: bool = ModelConfig.USE_BATCH_NORM,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Save configuration
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Initialize layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize model layers with proper regularization and initialization."""
        # Common parameters for Conv2D layers
        conv_params = {
            'kernel_initializer': HeNormal(),
            'kernel_regularizer': L2(self.kernel_regularizer),
            'use_bias': not self.use_batch_norm  # No bias when using BatchNorm
        }

        # First conv block
        self.conv1 = Conv2D(ModelConfig.CONV1_FILTERS, ModelConfig.KERNEL_SIZE, padding='same', **conv_params)
        self.bn1 = BatchNormalization() if self.use_batch_norm else None
        self.act1 = ReLU()
        self.pool1 = MaxPooling2D()
        self.drop1 = Dropout(self.dropout_rate)

        # Second conv block
        self.conv2 = Conv2D(ModelConfig.CONV2_FILTERS, ModelConfig.KERNEL_SIZE, padding='same', **conv_params)
        self.bn2 = BatchNormalization() if self.use_batch_norm else None
        self.act2 = ReLU()
        self.pool2 = MaxPooling2D()
        self.drop2 = Dropout(self.dropout_rate)

        # Third conv block
        self.conv3 = Conv2D(ModelConfig.CONV3_FILTERS, ModelConfig.KERNEL_SIZE, padding='same', **conv_params)
        self.bn3 = BatchNormalization() if self.use_batch_norm else None
        self.act3 = ReLU()
        self.pool3 = MaxPooling2D()
        self.drop3 = Dropout(self.dropout_rate)

        # Flatten and dense layers
        self.flatten = Flatten()
        self.dense1 = Dense(
            ModelConfig.DENSE_UNITS,
            kernel_initializer=HeNormal(),
            kernel_regularizer=L2(self.kernel_regularizer)
        )
        self.bn_dense = BatchNormalization() if self.use_batch_norm else None
        self.act_dense = ReLU()
        self.drop_dense = Dropout(self.dropout_rate)

        # Output layer
        self.output_layer = Dense(
            self.num_classes,
            activation='softmax',
            kernel_initializer=HeNormal(),
            kernel_regularizer=L2(self.kernel_regularizer)
        )

        self.kmeans = DifferentiableKMeansLayer(n_clusters=32)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)
            training: Whether the model is in training mode

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First conv block
        x = self.conv1(inputs)
        if self.bn1:
            x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        # Second conv block
        x = self.conv2(x)
        if self.bn2:
            x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        # Third conv block
        x = self.conv3(x)
        if self.bn3:
            x = self.bn3(x, training=training)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.drop3(x, training=training)

        # Dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        if self.bn_dense:
            x = self.bn_dense(x, training=training)
        x = self.kmeans(x)

        # Output layer
        return self.output_layer(x)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'kernel_regularizer': self.kernel_regularizer,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        })
        return config


def plot_confusion_matrix(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> None:
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


def visualize_layer_activations(
        model: tf.keras.Model,
        x_test: np.ndarray,
        y_test: np.ndarray,
        layer_names: List[str] = ['conv1', 'conv2', 'conv3']
) -> None:
    """Visualize activations for each class in a single plot.

    Args:
        model: Trained model
        x_test: Test images
        y_test: True labels
        layer_names: Names of layers to visualize
    """

    # Function to get layer outputs
    def get_layer_outputs(example):
        outputs = {}
        x = example

        # First conv block
        x = model.conv1(x)
        outputs['conv1'] = x
        if model.bn1:
            x = model.bn1(x)
        x = model.act1(x)
        x = model.pool1(x)
        x = model.drop1(x)

        # Second conv block
        x = model.conv2(x)
        outputs['conv2'] = x
        if model.bn2:
            x = model.bn2(x)
        x = model.act2(x)
        x = model.pool2(x)
        x = model.drop2(x)

        # Third conv block
        x = model.conv3(x)
        outputs['conv3'] = x
        if model.bn3:
            x = model.bn3(x)
        x = model.act3(x)
        x = model.pool3(x)
        x = model.drop3(x)

        return outputs

    # Get one example from each class
    examples = {}
    for digit in range(10):
        idx = np.where(y_test == digit)[0][0]
        examples[digit] = x_test[idx:idx + 1]

    # Create a single large figure
    n_features = 4  # Number of feature maps to show per layer
    n_layers = len(layer_names)

    # Calculate figure size based on content
    plt.figure(figsize=(20, 25))
    plt.suptitle('Layer Activations for All Digits', fontsize=16, y=0.95)

    # For each digit
    for digit, example in examples.items():
        # Get activations for this example
        layer_outputs = get_layer_outputs(example)

        # Plot original image first
        plt.subplot(10, 1 + n_layers * n_features, digit * (1 + n_layers * n_features) + 1)
        plt.imshow(example[0, ..., 0], cmap='gray')
        plt.axis('off')
        if digit == 0:
            plt.title('Input', pad=10)
        plt.text(-0.1, 0.5, f'Digit {digit}', transform=plt.gca().transAxes,
                 verticalalignment='center', horizontalalignment='right')

        # For each layer
        for layer_idx, layer_name in enumerate(layer_names):
            activations = layer_outputs[layer_name]

            # Plot feature maps
            for feature_idx in range(min(n_features, activations.shape[-1])):
                plt.subplot(10, 1 + n_layers * n_features,
                            digit * (1 + n_layers * n_features) + 2 + layer_idx * n_features + feature_idx)
                plt.imshow(activations[0, ..., feature_idx], cmap='viridis')
                plt.axis('off')

                # Add layer name as title only for the first row
                if digit == 0 and feature_idx == 0:
                    plt.title(f'{layer_name}', pad=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust to prevent title overlap
    plt.show()

def train_and_visualize_mnist() -> None:
    """Train the CNN model on MNIST dataset and visualize results."""
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Create and train model
    model = MNISTConvNet()
    model.compile(
        optimizer=Adam(learning_rate=ModelConfig.INITIAL_LEARNING_RATE),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()]
    )

    # Train model
    model.fit(
        x_train,
        y_train,
        batch_size=ModelConfig.BATCH_SIZE,
        epochs=ModelConfig.EPOCHS,
        validation_split=ModelConfig.VALIDATION_SPLIT,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                ModelConfig.MODEL_SAVE_PATH,
                save_best_only=True,
                monitor='val_sparse_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=ModelConfig.PATIENCE,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_sparse_categorical_accuracy',
                factor=ModelConfig.LR_REDUCTION_FACTOR,
                patience=ModelConfig.LR_PATIENCE,
                min_lr=ModelConfig.MIN_LEARNING_RATE
            )
        ]
    )

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(model, x_test, y_test)

    # Visualize activations
    print("\nGenerating activation visualizations...")
    visualize_layer_activations(model, x_test, y_test)


if __name__ == "__main__":
    train_and_visualize_mnist()
