import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from tensorflow.python.keras import layers
from typing import Optional, Union, Literal, List, Any, Tuple, Dict

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.python import keras
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.differentiable_kmeans import DifferentiableKMeansLayer

# ---------------------------------------------------------------------


# Type aliases
TensorLike = Union[tf.Tensor, np.ndarray]
MetricsDict = Dict[str, float]


# ---------------------------------------------------------------------

@dataclass
class ClusteringMetrics:
    """Container for clustering quality metrics."""
    silhouette: float
    inertia: float
    cluster_sizes: np.ndarray
    cluster_distribution: np.ndarray
    mean_distance: float


# ---------------------------------------------------------------------

class ClusteringLoss(keras.losses.Loss):
    """Custom loss function for clustering quality.

    Combines intra-cluster distance with cluster distribution penalty to encourage
    balanced cluster assignments.

    Args:
        distance_weight: Weight for intra-cluster distance term
        distribution_weight: Weight for cluster distribution term
        name: Name of the loss function

    Raises:
        ValueError: If weights are invalid (negative, zero, inf, or nan)
    """

    def __init__(
            self,
            distance_weight: float = 1.0,
            distribution_weight: float = 0.5,
            name: str = 'clustering_loss'
    ) -> None:
        super().__init__(name=name)

        # Validate weights
        self._validate_weight(distance_weight, 'distance_weight')
        self._validate_weight(distribution_weight, 'distribution_weight')

        self.distance_weight = distance_weight
        self.distribution_weight = distribution_weight

    def _validate_weight(self, weight: float, name: str) -> None:
        """Validate a weight parameter.

        Args:
            weight: Weight value to validate
            name: Name of the weight parameter for error messages

        Raises:
            ValueError: If weight is invalid
        """
        if not isinstance(weight, (int, float)):
            raise ValueError(f"{name} must be a number, got {type(weight)}")
        if weight <= 0:
            raise ValueError(f"{name} must be positive, got {weight}")
        if np.isinf(weight):
            raise ValueError(f"{name} cannot be infinite")
        if np.isnan(weight):
            raise ValueError(f"{name} cannot be NaN")

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the clustering loss.

        Args:
            y_true: True cluster assignments or target values
            y_pred: Predicted cluster assignments

        Returns:
            Combined clustering loss value
        """
        # Compute intra-cluster distances (mean squared error)
        cluster_distances = tf.reduce_mean(
            tf.square(y_pred - y_true),
            axis=-1
        )

        # Compute cluster distribution penalty
        cluster_distribution = tf.reduce_mean(y_pred, axis=0)
        uniform_distribution = tf.ones_like(cluster_distribution) / tf.cast(
            tf.shape(y_pred)[-1],
            tf.float32
        )
        distribution_penalty = tf.reduce_mean(
            tf.square(cluster_distribution - uniform_distribution)
        )

        # Scale the cluster distances to be in a similar range as distribution penalty
        scaled_distances = tf.reduce_mean(cluster_distances)

        # Combine losses
        total_loss = (
                self.distance_weight * scaled_distances +
                self.distribution_weight * distribution_penalty
        )

        return total_loss


# ---------------------------------------------------------------------

class ClusteringMetricsCallback(keras.callbacks.Callback):
    """Callback to monitor clustering quality metrics during training.

    Args:
        validation_data: Data to use for computing metrics
        visualization_freq: How often to create visualizations (in epochs)
        log_dir: Directory to save visualizations
    """

    def __init__(
            self,
            validation_data: Optional[TensorLike] = None,
            visualization_freq: int = 5,
            log_dir: str = './clustering_logs'
    ) -> None:
        super().__init__()
        self.validation_data = validation_data
        self.visualization_freq = visualization_freq
        self.log_dir = log_dir
        self.metrics_history: Dict[str, List[float]] = {
            'silhouette_score': [],
            'cluster_distribution': [],
            'inertia': [],
            'mean_distance': []
        }

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Compute metrics at the end of each epoch."""
        if self.validation_data is not None:
            # Get cluster assignments
            assignments = self.model.predict(self.validation_data)

            # Compute metrics
            metrics = compute_clustering_metrics(
                self.validation_data,
                assignments
            )

            # Update history
            self.metrics_history['silhouette_score'].append(metrics.silhouette)
            self.metrics_history['inertia'].append(metrics.inertia)
            self.metrics_history['mean_distance'].append(metrics.mean_distance)

            # Add to logs
            if logs is not None:
                logs.update({
                    'val_silhouette': metrics.silhouette,
                    'val_inertia': metrics.inertia,
                    'val_mean_distance': metrics.mean_distance
                })

            # Create visualizations periodically
            if epoch % self.visualization_freq == 0:
                self._create_visualizations(epoch, metrics)

    def _create_visualizations(
            self,
            epoch: int,
            metrics: ClusteringMetrics
    ) -> None:
        """Create and save visualization plots."""
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Plot cluster sizes
        axes[0, 0].bar(
            range(len(metrics.cluster_sizes)),
            metrics.cluster_sizes
        )
        axes[0, 0].set_title('Cluster Sizes')

        # Plot metrics history
        for metric_name, values in self.metrics_history.items():
            if len(values) > 1:  # Need at least 2 points to plot
                axes[0, 1].plot(values, label=metric_name)
        axes[0, 1].set_title('Metrics History')
        axes[0, 1].legend()

        # Plot cluster distribution
        axes[1, 0].bar(
            range(len(metrics.cluster_distribution)),
            metrics.cluster_distribution
        )
        axes[1, 0].set_title('Cluster Distribution')

        # Save plot
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/clustering_viz_epoch_{epoch}.png')
        plt.close()


# ---------------------------------------------------------------------


def compute_clustering_metrics(
        data: TensorLike,
        assignments: TensorLike
) -> ClusteringMetrics:
    """Compute various clustering quality metrics.

    Args:
        data: Input data points
        assignments: Cluster assignments (soft or hard)

    Returns:
        ClusteringMetrics object containing computed metrics
    """
    # Convert to numpy if needed
    if isinstance(data, tf.Tensor):
        data = data.numpy()
    if isinstance(assignments, tf.Tensor):
        assignments = assignments.numpy()

    # Get hard assignments for some metrics
    hard_assignments = np.argmax(assignments, axis=1)

    # Compute silhouette score
    try:
        silhouette = silhouette_score(data, hard_assignments)
    except ValueError:
        silhouette = -1  # Invalid clustering

    # Compute inertia (within-cluster sum of squares)
    inertia = 0
    for i in range(assignments.shape[1]):
        mask = hard_assignments == i
        if np.any(mask):
            centroid = np.mean(data[mask], axis=0)
            inertia += np.sum(np.square(data[mask] - centroid))

    # Compute cluster sizes and distribution
    cluster_sizes = np.bincount(
        hard_assignments,
        minlength=assignments.shape[1]
    )
    cluster_distribution = cluster_sizes / len(data)

    # Compute mean distance to assigned centroid
    mean_distance = np.mean([
        np.linalg.norm(data[i] - data[hard_assignments == hard_assignments[i]].mean(axis=0))
        for i in range(len(data))
    ])

    return ClusteringMetrics(
        silhouette=silhouette,
        inertia=inertia,
        cluster_sizes=cluster_sizes,
        cluster_distribution=cluster_distribution,
        mean_distance=mean_distance
    )


# ---------------------------------------------------------------------


def generate_test_data(
        n_samples: int,
        n_features: int,
        n_clusters: int,
        noise_std: float = 0.1,
        random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic clustering data.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        n_clusters: Number of true clusters
        noise_std: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (data, true_labels)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate cluster centers
    centers = np.random.randn(n_clusters, n_features)

    # Generate samples around centers
    samples_per_cluster = n_samples // n_clusters
    data = []
    labels = []

    for i, center in enumerate(centers):
        cluster_samples = center + np.random.randn(samples_per_cluster, n_features) * noise_std
        data.append(cluster_samples)
        labels.append(np.full(samples_per_cluster, i))

    return (
        np.vstack(data).astype(np.float32),
        np.hstack(labels).astype(np.int32)
    )


# ---------------------------------------------------------------------

def visualize_clusters_2d(
        data: TensorLike,
        assignments: TensorLike,
        title: str = 'Cluster Visualization',
        save_path: Optional[str] = None
) -> None:
    """Visualize clustering results in 2D using t-SNE.

    Args:
        data: Input data points
        assignments: Cluster assignments (soft or hard)
        title: Plot title
        save_path: Optional path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(data, tf.Tensor):
        data = data.numpy()
    if isinstance(assignments, tf.Tensor):
        assignments = assignments.numpy()

    # Get hard assignments
    labels = np.argmax(assignments, axis=1)

    # Reduce dimensionality for visualization
    if data.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        data_2d = tsne.fit_transform(data)
    else:
        data_2d = data

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        data_2d[:, 0],
        data_2d[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6
    )
    plt.colorbar(scatter)
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------


def create_clustering_pipeline(
        input_shape: Tuple[int, ...],
        n_clusters: int,
        temperature: float = 0.1,
        learning_rate: float = 0.001,
        distance_weight: float = 1.0,
        distribution_weight: float = 0.5
) -> keras.Model:
    """Create a complete clustering pipeline.

    Args:
        input_shape: Shape of input data
        n_clusters: Number of clusters
        temperature: Softmax temperature
        learning_rate: Learning rate for optimizer
        distance_weight: Weight for distance term in loss
        distribution_weight: Weight for distribution term in loss

    Returns:
        Compiled Keras model

    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Input validation
        if not all(dim > 0 for dim in input_shape):
            raise ValueError(f"Invalid input shape: {input_shape}")
        if n_clusters < 1:
            raise ValueError(f"Invalid number of clusters: {n_clusters}")
        if temperature <= 0:
            raise ValueError(f"Invalid temperature: {temperature}")

        # Create model
        inputs = keras.Input(shape=input_shape)

        # Preprocessing
        x = keras.layers.LayerNormalization()(inputs)
        x = keras.layers.GaussianNoise(0.01)(x)

        # Clustering
        kmeans = DifferentiableKMeansLayer(
            n_clusters=n_clusters,
            temperature=temperature
        )
        outputs = kmeans(x)

        # Create and compile model
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=ClusteringLoss(
                distance_weight=distance_weight,
                distribution_weight=distribution_weight
            )
        )

        return model

    except Exception as e:
        raise RuntimeError(f"Error creating clustering pipeline: {str(e)}") from e

# ---------------------------------------------------------------------