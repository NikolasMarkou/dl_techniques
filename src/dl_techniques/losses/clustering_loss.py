"""
A dual-objective loss function for deep clustering.

This loss is designed for deep learning models that perform unsupervised
clustering by learning an embedding space. It guides the model to produce
both compact and well-balanced clusters by combining two distinct objectives:
a primary clustering assignment loss and a regularization term for cluster
distribution.

Conceptual Overview:
    The fundamental goal is to simultaneously optimize the quality of the
    learned feature representations and the quality of the cluster
    assignments. This is achieved by penalizing two undesirable outcomes:
    1.  Poor cluster assignments, where the model's predicted cluster
        probabilities for a sample diverge from a target distribution. This
        target is often derived in a self-supervised manner from a more
        confident version of the model's own predictions.
    2.  Trivial or degenerate solutions, such as the model assigning all data
        points to a single cluster ("cluster collapse").

Architectural Design:
    The total loss is a weighted sum of two components:
    1.  Intra-cluster Distance Loss: This term measures the discrepancy
        between the model's predicted soft cluster assignments and a target
        assignment distribution. In many deep clustering frameworks (like DEC),
        this target is a "sharpened" version of the predictions, which helps
        to improve cluster purity and separation over time. The implementation
        here uses Mean Squared Error (MSE) as the distance metric.
    2.  Cluster Distribution Penalty: This term acts as a regularizer. It
        calculates the mean frequency of assignment for each cluster across an
        entire batch and penalizes any deviation from a uniform distribution.
        This encourages the model to utilize all available clusters equally,
        preventing it from ignoring some clusters entirely.

Mathematical Formulation:
    Let `P` be the `(N, K)` matrix of predicted soft assignments for a batch
    of `N` samples and `K` clusters, and let `Q` be the target distribution.

    1.  The distance loss (`L_dist`) is the mean squared error between the
        predicted and target distributions:

        L_dist = MSE(P, Q) = 1/N * Σᵢ ||pᵢ - qᵢ||²

    2.  The distribution penalty (`L_distrib`) is the MSE between the actual
        cluster assignment frequencies and a uniform distribution. Let `f_j`
        be the frequency of cluster `j`, computed as `f_j = 1/N * Σᵢ pᵢⱼ`.

        L_distrib = MSE(f, u) = 1/K * Σⱼ (fⱼ - 1/K)²

    The final loss is a weighted combination of these two terms:

        L_total = w_dist * L_dist + w_distrib * L_distrib

References:
    The architectural pattern of combining a primary clustering objective with
    a distribution-balancing regularizer is a common technique in deep
    unsupervised clustering. It draws inspiration from foundational works
    such as:
    -   Xie, J., Girshick, R., & Farhadi, A. (2016). "Unsupervised Deep
        Embedding for Clustering Analysis." ICML.
    -   Guo, X., et al. (2017). "Improved Deep Embedded Clustering with Local
        Structure Preservation." IJCAI.
"""

import os
import keras
import numpy as np
from keras import ops
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.metrics import silhouette_score
from typing import Optional, Union, List, Any, Dict

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# Type aliases
TensorLike = Union[keras.KerasTensor, np.ndarray]
MetricsDict = Dict[str, float]

# ---------------------------------------------------------------------

@dataclass
class ClusteringMetrics:
    """Container for clustering quality metrics.

    Attributes:
        silhouette (float): Silhouette score for clustering quality (-1 to 1, higher is better).
        inertia (float): Within-cluster sum of squares (lower is better).
        cluster_sizes (np.ndarray): Number of samples in each cluster.
        cluster_distribution (np.ndarray): Proportion of samples in each cluster.
        mean_distance (float): Average distance to cluster centroid.
    """
    silhouette: float
    inertia: float
    cluster_sizes: np.ndarray
    cluster_distribution: np.ndarray
    mean_distance: float


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ClusteringLoss(keras.losses.Loss):
    """Custom loss function for clustering quality.

    Combines intra-cluster distance with cluster distribution penalty to encourage
    balanced cluster assignments. The loss function optimizes both the quality of
    clusters (compactness) and their balance (avoiding empty clusters).

    Args:
        distance_weight (float): Weight for intra-cluster distance term. Must be positive.
            Higher values emphasize cluster compactness.
        distribution_weight (float): Weight for cluster distribution term. Must be positive.
            Higher values emphasize balanced cluster sizes.
        name (str): Name of the loss function.

    Raises:
        ValueError: If weights are invalid (negative, zero, inf, or nan).

    Example:
        >>> loss = ClusteringLoss(distance_weight=1.0, distribution_weight=0.5)
        >>> model.compile(optimizer='adam', loss=loss)
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

        logger.info(f"Initialized {name} with distance_weight={distance_weight}, "
                    f"distribution_weight={distribution_weight}")

    def _validate_weight(self, weight: float, name: str) -> None:
        """Validate a weight parameter.

        Args:
            weight (float): Weight value to validate.
            name (str): Name of the weight parameter for error messages.

        Raises:
            ValueError: If weight is invalid.
        """
        if not isinstance(weight, (int, float)):
            raise ValueError(f"{name} must be a number, got {type(weight)}")
        if weight <= 0:
            raise ValueError(f"{name} must be positive, got {weight}")
        if np.isinf(weight):
            raise ValueError(f"{name} cannot be infinite")
        if np.isnan(weight):
            raise ValueError(f"{name} cannot be NaN")

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the clustering loss.

        Args:
            y_true (keras.KerasTensor): True cluster assignments or target values.
            y_pred (keras.KerasTensor): Predicted cluster assignments (soft assignments).

        Returns:
            keras.KerasTensor: Combined clustering loss value.
        """
        # Compute intra-cluster distances (mean squared error)
        cluster_distances = ops.mean(
            ops.square(y_pred - y_true),
            axis=-1
        )

        # Compute cluster distribution penalty
        cluster_distribution = ops.mean(y_pred, axis=0)
        uniform_distribution = ops.ones_like(cluster_distribution) / ops.cast(
            ops.shape(y_pred)[-1],
            dtype="float32"
        )
        distribution_penalty = ops.mean(
            ops.square(cluster_distribution - uniform_distribution)
        )

        # Scale the cluster distances to be in a similar range as distribution penalty
        scaled_distances = ops.mean(cluster_distances)

        # Combine losses
        total_loss = (
                self.distance_weight * scaled_distances +
                self.distribution_weight * distribution_penalty
        )

        return total_loss

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the loss function.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'distance_weight': self.distance_weight,
            'distribution_weight': self.distribution_weight
        })
        return config


# ---------------------------------------------------------------------

class ClusteringMetricsCallback(keras.callbacks.Callback):
    """Callback to monitor clustering quality metrics during training.

    This callback computes and logs various clustering metrics during training,
    including silhouette score, inertia, and cluster distribution. It can also
    create visualizations periodically.

    Args:
        validation_data (Optional[TensorLike]): Data to use for computing metrics.
        visualization_freq (int): How often to create visualizations (in epochs).
        log_dir (str): Directory to save visualizations.

    Example:
        >>> callback = ClusteringMetricsCallback(
        ...     validation_data=val_data,
        ...     visualization_freq=10,
        ...     log_dir='./logs'
        ... )
        >>> model.fit(x_train, y_train, callbacks=[callback])
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

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"ClusteringMetricsCallback initialized with log_dir: {log_dir}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Compute metrics at the end of each epoch.

        Args:
            epoch (int): Current epoch number.
            logs (Optional[Dict]): Dictionary of logs from training.
        """
        if self.validation_data is not None:
            try:
                # Get cluster assignments
                assignments = self.model.predict(self.validation_data, verbose=0)

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

                # Log metrics
                logger.info(f"Epoch {epoch}: Silhouette={metrics.silhouette:.4f}, "
                            f"Inertia={metrics.inertia:.4f}, Mean Distance={metrics.mean_distance:.4f}")

                # Create visualizations periodically
                if epoch % self.visualization_freq == 0:
                    self._create_visualizations(epoch, metrics)

            except Exception as e:
                logger.error(f"Error computing clustering metrics at epoch {epoch}: {str(e)}")

    def _create_visualizations(
            self,
            epoch: int,
            metrics: ClusteringMetrics
    ) -> None:
        """Create and save visualization plots.

        Args:
            epoch (int): Current epoch number.
            metrics (ClusteringMetrics): Computed clustering metrics.
        """
        try:
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))

            # Plot cluster sizes
            axes[0, 0].bar(
                range(len(metrics.cluster_sizes)),
                metrics.cluster_sizes
            )
            axes[0, 0].set_title('Cluster Sizes')
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('Number of Samples')

            # Plot metrics history
            for metric_name, values in self.metrics_history.items():
                if len(values) > 1:  # Need at least 2 points to plot
                    axes[0, 1].plot(values, label=metric_name)
            axes[0, 1].set_title('Metrics History')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Metric Value')
            axes[0, 1].legend()

            # Plot cluster distribution
            axes[1, 0].bar(
                range(len(metrics.cluster_distribution)),
                metrics.cluster_distribution
            )
            axes[1, 0].set_title('Cluster Distribution')
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Proportion')

            # Plot silhouette score over time
            if len(self.metrics_history['silhouette_score']) > 1:
                axes[1, 1].plot(self.metrics_history['silhouette_score'], 'b-', linewidth=2)
                axes[1, 1].set_title('Silhouette Score Over Time')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Silhouette Score')
                axes[1, 1].grid(True, alpha=0.3)

            # Save plot
            plt.tight_layout()
            save_path = os.path.join(self.log_dir, f'clustering_viz_epoch_{epoch}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved clustering visualization to {save_path}")

        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")


# ---------------------------------------------------------------------

def compute_clustering_metrics(
        data: TensorLike,
        assignments: TensorLike
) -> ClusteringMetrics:
    """Compute various clustering quality metrics.

    Calculates comprehensive clustering metrics including silhouette score,
    inertia, cluster sizes, and mean distances to evaluate clustering quality.

    Args:
        data (TensorLike): Input data points of shape (n_samples, n_features).
        assignments (TensorLike): Cluster assignments of shape (n_samples, n_clusters).
            Can be soft (probabilities) or hard (one-hot) assignments.

    Returns:
        ClusteringMetrics: Object containing computed metrics.

    Raises:
        ValueError: If data or assignments have invalid shapes.

    Example:
        >>> data = np.random.randn(100, 10)
        >>> assignments = np.random.rand(100, 3)  # 3 clusters
        >>> metrics = compute_clustering_metrics(data, assignments)
        >>> print(f"Silhouette score: {metrics.silhouette}")
    """
    # Convert to numpy if needed
    if hasattr(data, 'numpy'):
        data = data.numpy()
    if hasattr(assignments, 'numpy'):
        assignments = assignments.numpy()

    # Validate inputs
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")
    if assignments.ndim != 2:
        raise ValueError(f"Assignments must be 2D, got shape {assignments.shape}")
    if data.shape[0] != assignments.shape[0]:
        raise ValueError(f"Data and assignments must have same number of samples: "
                         f"{data.shape[0]} vs {assignments.shape[0]}")

    # Get hard assignments for some metrics
    hard_assignments = np.argmax(assignments, axis=1)

    # Compute silhouette score
    try:
        unique_labels = np.unique(hard_assignments)
        if len(unique_labels) > 1 and len(unique_labels) < len(data):
            silhouette = silhouette_score(data, hard_assignments)
        else:
            silhouette = -1.0  # Invalid clustering (all same cluster or each point its own cluster)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Could not compute silhouette score: {str(e)}")
        silhouette = -1.0  # Invalid clustering

    # Compute inertia (within-cluster sum of squares)
    inertia = 0.0
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
    mean_distances = []
    for i in range(assignments.shape[1]):
        mask = hard_assignments == i
        if np.any(mask):
            centroid = np.mean(data[mask], axis=0)
            distances = np.linalg.norm(data[mask] - centroid, axis=1)
            mean_distances.extend(distances)

    mean_distance = np.mean(mean_distances) if mean_distances else 0.0

    return ClusteringMetrics(
        silhouette=float(silhouette),
        inertia=float(inertia),
        cluster_sizes=cluster_sizes,
        cluster_distribution=cluster_distribution,
        mean_distance=float(mean_distance)
    )

# ---------------------------------------------------------------------
