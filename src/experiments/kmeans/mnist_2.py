import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from typing import List, Tuple
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import tensorflow as tf
from keras import Model, Input
from keras.api.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization,
    Dropout, ReLU
)
from keras.api.optimizers import Adam
from keras.api.regularizers import L2
from keras.api.initializers import HeNormal
from keras.api.metrics import SparseCategoricalAccuracy
from keras.api.losses import SparseCategoricalCrossentropy

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from dl_techniques.layers.kmeans import KMeansLayer


# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration constants for the experiment."""
    # Model architecture
    INPUT_SHAPE: Tuple[int, int, int] = (28, 28, 1)
    NUM_CLASSES: int = 10

    # Layer sizes
    CONV1_FILTERS: int = 16
    CONV2_FILTERS: int = 32
    CONV3_FILTERS: int = 64
    DENSE_UNITS: int = 32
    KERNEL_SIZE: Tuple[int, int] = (3, 3)

    # KMeans
    N_CLUSTERS: int = 32
    TEMPERATURE: float = 0.1
    MOMENTUM: float = 0.9

    # Regularization
    KERNEL_REGULARIZER: float = 1e-4
    DROPOUT_RATE: float = 0.3
    USE_BATCH_NORM: bool = True

    # Training parameters
    BATCH_SIZE: int = 128
    EPOCHS: int = 3
    VALIDATION_SPLIT: float = 0.1
    INITIAL_LEARNING_RATE: float = 0.001
    MIN_LEARNING_RATE: float = 1e-5
    LR_REDUCTION_FACTOR: float = 0.5

    # Early stopping
    PATIENCE: int = 5
    LR_PATIENCE: int = 3

    # File paths
    MODEL_SAVE_PATH: str = 'mnist_model.keras'


# ---------------------------------------------------------------------


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
            input_shape: Tuple[int, int, int] = ExperimentConfig.INPUT_SHAPE,
            num_classes: int = ExperimentConfig.NUM_CLASSES,
            kernel_regularizer: float = ExperimentConfig.KERNEL_REGULARIZER,
            dropout_rate: float = ExperimentConfig.DROPOUT_RATE,
            use_batch_norm: bool = ExperimentConfig.USE_BATCH_NORM,
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
        self.build_model()

    def _init_layers(self) -> None:
        """Initialize model layers with proper regularization and initialization."""
        # Common parameters for Conv2D layers
        conv_params = {
            'kernel_initializer': HeNormal(),
            'kernel_regularizer': L2(self.kernel_regularizer),
            'use_bias': not self.use_batch_norm  # No bias when using BatchNorm
        }

        # First conv block
        self.conv1 = Conv2D(ExperimentConfig.CONV1_FILTERS, (5, 5), padding='same', **conv_params)
        self.bn1 = BatchNormalization() if self.use_batch_norm else None
        self.act1 = ReLU()
        self.pool1 = MaxPooling2D()
        self.drop1 = Dropout(self.dropout_rate)

        # Second conv block
        self.conv2 = Conv2D(ExperimentConfig.CONV2_FILTERS, ExperimentConfig.KERNEL_SIZE, padding='same', **conv_params)
        self.bn2 = BatchNormalization() if self.use_batch_norm else None
        self.act2 = ReLU()
        self.pool2 = MaxPooling2D()
        self.drop2 = Dropout(self.dropout_rate)

        # Third conv block
        self.conv3 = Conv2D(ExperimentConfig.CONV3_FILTERS, ExperimentConfig.KERNEL_SIZE, padding='same', **conv_params)
        self.bn3 = BatchNormalization() if self.use_batch_norm else None
        self.act3 = ReLU()
        self.pool3 = MaxPooling2D()
        self.drop3 = Dropout(self.dropout_rate)

        self.kmeans_flat_features = \
            KMeansLayer(
                n_clusters=ExperimentConfig.N_CLUSTERS,
                temperature=ExperimentConfig.TEMPERATURE,
                momentum=ExperimentConfig.MOMENTUM,
                name="kmeans_flat_features"
            )

        # Flatten and dense layers
        self.flatten = Flatten()
        self.dense1 = Dense(
            ExperimentConfig.DENSE_UNITS,
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

    def build_model(self) -> None:
        """Initialize model layers with proper input shape.

        This method explicitly creates the input layer and connects
        all layers to ensure proper model building.
        """
        # Create explicit input layer
        inputs = Input(shape=self.input_shape)

        # First conv block
        x = self.conv1(inputs)
        if self.bn1:
            x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Second conv block
        x = self.conv2(x)
        if self.bn2:
            x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Third conv block
        x = self.conv3(x)
        if self.bn3:
            x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # KMeans layer
        x = self.kmeans_flat_features(x)

        # Dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        if self.bn_dense:
            x = self.bn_dense(x)
        x = self.act_dense(x)
        x = self.drop_dense(x)

        # Output layer
        outputs = self.output_layer(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='MNISTConvNet')

        # Copy the weights if they exist
        if self.built:
            model.set_weights(self.get_weights())

        self.built = True

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
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
        x = self.kmeans_flat_features(x, training=training)

        # Dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        if self.bn_dense:
            x = self.bn_dense(x, training=training)

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


# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------

def visualize_centroids(
        model: Model,
        x_data: np.ndarray,
        y_data: np.ndarray,
        method: str = 'tsne',
        perplexity: float = 30.0,
        n_iter: int = 1000,
        figsize: Tuple[int, int] = (12, 8)
) -> None:
    """Visualize kmeans centroids and their relationships with data points.

    Args:
        model: Trained model containing DifferentiableKMeansLayer
        x_data: Input data
        y_data: Labels for coloring points
        method: Dimensionality reduction method ('tsne' or 'pca')
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
        figsize: Figure size for the plot
    """
    # Build model if not already built
    if not model.built:
        model.build((None,) + model.input_shape[1:])

    # Get pre-kmeans features using forward pass
    x = x_data

    # First conv block
    x = model.conv1(x)
    if model.bn1:
        x = model.bn1(x, training=False)
    x = model.act1(x)
    x = model.pool1(x)
    x = model.drop1(x, training=False)

    # Second conv block
    x = model.conv2(x)
    if model.bn2:
        x = model.bn2(x, training=False)
    x = model.act2(x)
    x = model.pool2(x)
    x = model.drop2(x, training=False)

    # Third conv block
    x = model.conv3(x)
    if model.bn3:
        x = model.bn3(x, training=False)
    x = model.act3(x)
    x = model.pool3(x)
    x = model.drop3(x, training=False)

    # Get pre-kmeans features and convert to numpy
    pre_kmeans_features = x

    # Get kmeans layer and its centroids
    kmeans_layer = model.kmeans_flat_features

    # Get cluster assignments using original features
    assignments = kmeans_layer(pre_kmeans_features, training=False)
    cluster_labels = tf.argmax(assignments, axis=1).numpy()

    # Flatten the features using TensorFlow operations to match centroid dimensions
    batch_size = tf.shape(pre_kmeans_features)[0]
    feature_size_per_channel = tf.reduce_prod(tf.shape(pre_kmeans_features)[1:])
    n_channels = pre_kmeans_features.shape[-1]

    # Reshape features to (batch_size, height * width, channels)
    flattened_features = tf.reshape(pre_kmeans_features,
                                    [batch_size, -1, n_channels])

    # Average across spatial dimensions to get (batch_size, channels)
    averaged_features = tf.reduce_mean(flattened_features, axis=1)
    averaged_features = averaged_features.numpy()

    # Get centroids and ensure they match the feature dimensions
    centroids = kmeans_layer.centroids.numpy()

    # Print shapes for debugging
    logger.info(f"Averaged features shape: {averaged_features.shape}")
    logger.info(f"Centroids shape: {centroids.shape}")

    # Combine averaged features and centroids for joint embedding
    combined_data = np.vstack([averaged_features, centroids])

    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    else:  # PCA
        reducer = PCA(n_components=2)

    # Perform dimensionality reduction
    embedded_data = reducer.fit_transform(combined_data)

    # Split embedded data back into features and centroids
    embedded_features = embedded_data[:-len(centroids)]
    embedded_centroids = embedded_data[-len(centroids):]

    # Create visualization
    plt.figure(figsize=figsize)

    # Plot data points
    scatter = plt.scatter(
        embedded_features[:, 0],
        embedded_features[:, 1],
        c=y_data,
        cmap='tab10',
        alpha=0.5,
        s=50
    )

    # Plot centroids
    plt.scatter(
        embedded_centroids[:, 0],
        embedded_centroids[:, 1],
        c='red',
        marker='*',
        s=200,
        label='Centroids'
    )

    # Add labels and legend
    plt.title(f'Centroids and Data Points ({method.upper()} projection)')
    plt.colorbar(scatter, label='Digit Class')
    plt.legend()

    # Show centroid connectivity
    for i, centroid in enumerate(embedded_centroids):
        # Draw lines to closest centroids
        distances = np.sqrt(np.sum((embedded_centroids - centroid) ** 2, axis=1))
        closest_idx = np.argsort(distances)[1:4]  # Get 3 closest centroids
        for idx in closest_idx:
            plt.plot(
                [centroid[0], embedded_centroids[idx][0]],
                [centroid[1], embedded_centroids[idx][1]],
                'gray',
                alpha=0.2
            )

    plt.show()


def visualize_centroid_evolution(
        model: Model,
        x_data: np.ndarray,
        training_steps: int = 100,
        batch_size: int = 128,
        method: str = 'pca'  # Default to PCA for consistency across steps
) -> None:
    """Visualize how centroids evolve during training with meaningful metrics.

    Args:
        model: Model containing DifferentiableKMeansLayer
        x_data: Input data for training
        training_steps: Number of training steps to track
        batch_size: Batch size for training
        method: Dimensionality reduction method ('tsne' or 'pca')
    """
    # Get kmeans layer
    kmeans_layer = model.kmeans_flat_features

    # Initialize storage for tracking
    centroid_history = []
    distance_metrics = []
    stability_metrics = []

    # Store initial state
    initial_centroids = kmeans_layer.centroids.numpy().copy()
    centroid_history.append(initial_centroids)

    # Setup the reducer once
    reducer = PCA(n_components=2) if method.lower() == 'pca' else TSNE(n_components=2)

    # Training and tracking loop
    for step in range(training_steps):
        # Get random batch
        indices = np.random.choice(len(x_data), batch_size)
        batch_x = x_data[indices]

        # Forward pass to update centroids
        x = batch_x

        # Process through the network up to kmeans
        x = model.conv1(x)
        if model.bn1:
            x = model.bn1(x, training=True)
        x = model.act1(x)
        x = model.pool1(x)
        x = model.drop1(x, training=True)

        x = model.conv2(x)
        if model.bn2:
            x = model.bn2(x, training=True)
        x = model.act2(x)
        x = model.pool2(x)
        x = model.drop2(x, training=True)

        x = model.conv3(x)
        if model.bn3:
            x = model.bn3(x, training=True)
        x = model.act3(x)
        x = model.pool3(x)
        x = model.drop3(x, training=True)

        # Update centroids
        _ = kmeans_layer(x, training=True)

        # Store current state
        current_centroids = kmeans_layer.centroids.numpy().copy()
        centroid_history.append(current_centroids)

        # Compute metrics
        if step > 0:
            # Average movement from previous step
            movement = np.mean(np.linalg.norm(
                current_centroids - centroid_history[-2], axis=1
            ))
            stability_metrics.append(movement)

            # Average pairwise distances between centroids
            pairwise_distances = []
            for i in range(len(current_centroids)):
                for j in range(i + 1, len(current_centroids)):
                    dist = np.linalg.norm(current_centroids[i] - current_centroids[j])
                    pairwise_distances.append(dist)
            distance_metrics.append(np.mean(pairwise_distances))

    # Reshape centroids for visualization
    reshaped_centroids = [c.reshape(c.shape[0], -1) for c in centroid_history]

    # Create visualization with three subplots
    fig = plt.figure(figsize=(15, 12))

    # 1. Show centroid trajectories
    ax1 = fig.add_subplot(221)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_centroids)))

    # Project all centroids at once for consistency
    all_centroids = np.vstack(reshaped_centroids)
    projected_centroids = reducer.fit_transform(all_centroids)

    # Split back into time steps
    n_centroids = len(initial_centroids)
    projected_history = [
        projected_centroids[i:i + n_centroids]
        for i in range(0, len(projected_centroids), n_centroids)
    ]

    # Plot trajectories
    for i in range(n_centroids):
        trajectory = np.array([step[i] for step in projected_history])
        ax1.plot(trajectory[:, 0], trajectory[:, 1], '-', color=colors[i], alpha=0.5)
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c=[colors[i]], marker='*')

    ax1.set_title('Centroid Trajectories')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')

    # 2. Plot stability metrics
    ax2 = fig.add_subplot(222)
    ax2.plot(stability_metrics, '-o')
    ax2.set_title('Centroid Stability Over Time')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Average Movement')

    # 3. Plot distance metrics
    ax3 = fig.add_subplot(223)
    ax3.plot(distance_metrics, '-o')
    ax3.set_title('Average Inter-Centroid Distance')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Average Distance')

    # 4. Plot final arrangement with connections
    ax4 = fig.add_subplot(224)
    final_projected = projected_history[-1]

    # Plot final centroids
    scatter = ax4.scatter(
        final_projected[:, 0],
        final_projected[:, 1],
        c=range(n_centroids),
        cmap='rainbow',
        s=100
    )

    # Draw connections between close centroids
    for i, centroid in enumerate(final_projected):
        distances = np.linalg.norm(final_projected - centroid, axis=1)
        closest_idx = np.argsort(distances)[1:4]  # Get 3 closest
        for idx in closest_idx:
            ax4.plot(
                [centroid[0], final_projected[idx][0]],
                [centroid[1], final_projected[idx][1]],
                'gray',
                alpha=0.2
            )

    ax4.set_title('Final Centroid Arrangement')
    plt.colorbar(scatter, ax=ax4, label='Centroid Index')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------


def train_and_visualize_mnist() -> None:
    """Train the CNN model on MNIST dataset and visualize results."""
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Create model instance with configuration
    model = MNISTConvNet(
        input_shape=ExperimentConfig.INPUT_SHAPE,
        num_classes=ExperimentConfig.NUM_CLASSES,
        kernel_regularizer=ExperimentConfig.KERNEL_REGULARIZER,
        dropout_rate=ExperimentConfig.DROPOUT_RATE,
        use_batch_norm=ExperimentConfig.USE_BATCH_NORM
    )

    # Build the model with proper input shape
    # model.build((None,) + ExperimentConfig.INPUT_SHAPE)

    # Print model summary to verify architecture
    model.summary()

    # Compile model with optimizer, loss, and metrics
    model.compile(
        optimizer=Adam(
            learning_rate=ExperimentConfig.INITIAL_LEARNING_RATE
        ),
        loss=SparseCategoricalCrossentropy(),
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

    # Rest of the visualization code remains the same...
    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Plot confusion matrix
    logger.info("Generating confusion matrix...")
    plot_confusion_matrix(model, x_test, y_test)

    # Visualize activations
    # logger.info("Generating activation visualizations...")
    # visualize_layer_activations(model, x_test, y_test)

    # Visualize centroids
    logger.info("Visualizing centroids...")
    visualize_centroids(model, x_test, y_test)

    logger.info("Visualizing centroid evolution...")
    visualize_centroid_evolution(model, x_test)

    return model, history


# ---------------------------------------------------------------------

if __name__ == "__main__":
    train_and_visualize_mnist()
